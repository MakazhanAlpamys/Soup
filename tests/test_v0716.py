"""v0.71.6 — "Synth data + build runner + misc".

Closes #231 (live `soup build` runner), #232 (live Magpie generator),
#167 (tokenizer-aware memorization probe), #213 (2PL/3PL IRT), #75 (QA log).

All five are pure-Python / CPU-testable on Windows + RTX 3050 4 GB. Live
provider calls (#232 Ollama/vLLM, #75 synth-data) are exercised with injected
stub callables / mocked httpx — the established project pattern (v0.53.7 #111
data_forge, v0.47.0 _default_judge) for "live provider" features.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    return path


# =============================================================================
# #231 — Live `soup build` runner
# =============================================================================


class TestBuiltinTransforms:
    def test_builtin_registry_immutable(self) -> None:
        from soup_cli.utils import build_dag

        with pytest.raises(TypeError):
            build_dag.BUILTIN_TRANSFORMS["x"] = lambda r, c: r  # type: ignore[index]

    def test_identity_returns_copy(self) -> None:
        from soup_cli.utils import build_dag

        fn = build_dag.BUILTIN_TRANSFORMS["identity"]
        row = {"id": "a", "text": "hello"}
        out = fn(row, {})
        assert out == row
        assert out is not row  # must not mutate the input

    def test_drop_empty(self) -> None:
        from soup_cli.utils import build_dag

        fn = build_dag.BUILTIN_TRANSFORMS["drop_empty"]
        assert fn({"id": "a", "text": "x"}, {}) is not None
        assert fn({"id": "b", "text": "   "}, {}) is None
        assert fn({"id": "c"}, {}) is None

    def test_lowercase(self) -> None:
        from soup_cli.utils import build_dag

        fn = build_dag.BUILTIN_TRANSFORMS["lowercase"]
        out = fn({"id": "a", "text": "Hello WORLD"}, {})
        assert out["text"] == "hello world"

    def test_lowercase_custom_field(self) -> None:
        from soup_cli.utils import build_dag

        fn = build_dag.BUILTIN_TRANSFORMS["lowercase"]
        out = fn({"id": "a", "content": "ABC"}, {"field": "content"})
        assert out["content"] == "abc"

    def test_add_field(self) -> None:
        from soup_cli.utils import build_dag

        fn = build_dag.BUILTIN_TRANSFORMS["add_field"]
        out = fn({"id": "a", "text": "x"}, {"field": "split", "value": "train"})
        assert out["split"] == "train"

    def test_token_count(self) -> None:
        from soup_cli.utils import build_dag

        fn = build_dag.BUILTIN_TRANSFORMS["token_count"]
        out = fn({"id": "a", "text": "one two three"}, {})
        assert out["n_tokens"] == 3

    def test_resolve_transform_unknown(self) -> None:
        from soup_cli.utils import build_dag

        with pytest.raises(ValueError, match="unknown transform"):
            build_dag.resolve_transform("nope", {})

    def test_resolve_transform_extra_overrides(self) -> None:
        from soup_cli.utils import build_dag

        sentinel = lambda r, c: {"id": r.get("id"), "x": 1}  # noqa: E731
        fn = build_dag.resolve_transform("custom", {"custom": sentinel})
        assert fn is sentinel


class TestRunBuild:
    def _plan(self, raw: dict):
        from soup_cli.utils import build_dag

        return build_dag.parse_build_plan(raw)

    def test_table_seed_materializes_jsonl(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils import build_dag

        monkeypatch.chdir(tmp_path)
        _write_jsonl(
            tmp_path / "data" / "raw.jsonl",
            [{"id": "1", "text": "Hello"}, {"id": "2", "text": "World"}],
        )
        plan = self._plan(
            {
                "models": [
                    {
                        "name": "lc",
                        "kind": "table",
                        "source": "data/raw.jsonl",
                        "transform": "lowercase",
                    }
                ]
            }
        )
        result = build_dag.run_build(plan, output_dir="out")
        out_file = tmp_path / "out" / "lc.jsonl"
        assert out_file.is_file()
        rows = [json.loads(line) for line in out_file.read_text().splitlines() if line]
        assert [r["text"] for r in rows] == ["hello", "world"]
        model_res = result.models[0]
        assert model_res.name == "lc"
        assert model_res.rows_in == 2
        assert model_res.rows_out == 2
        assert model_res.transform_calls == 2

    def test_drop_empty_reduces_rows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils import build_dag

        monkeypatch.chdir(tmp_path)
        _write_jsonl(
            tmp_path / "data" / "raw.jsonl",
            [{"id": "1", "text": "x"}, {"id": "2", "text": "  "}, {"id": "3", "text": "y"}],
        )
        plan = self._plan(
            {
                "models": [
                    {
                        "name": "clean",
                        "kind": "table",
                        "source": "data/raw.jsonl",
                        "transform": "drop_empty",
                    }
                ]
            }
        )
        result = build_dag.run_build(plan, output_dir="out")
        assert result.models[0].rows_in == 3
        assert result.models[0].rows_out == 2

    def test_derived_model_consumes_refs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils import build_dag

        monkeypatch.chdir(tmp_path)
        _write_jsonl(
            tmp_path / "data" / "raw.jsonl",
            [{"id": "1", "text": "Hello"}, {"id": "2", "text": ""}],
        )
        plan = self._plan(
            {
                "models": [
                    {
                        "name": "raw",
                        "kind": "table",
                        "source": "data/raw.jsonl",
                        "transform": "identity",
                    },
                    {
                        "name": "clean",
                        "kind": "table",
                        "refs": ["raw"],
                        "transform": "drop_empty",
                    },
                ]
            }
        )
        build_dag.run_build(plan, output_dir="out")
        clean = tmp_path / "out" / "clean.jsonl"
        rows = [json.loads(line) for line in clean.read_text().splitlines() if line]
        assert [r["id"] for r in rows] == ["1"]

    def test_view_not_written_but_feeds_downstream(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils import build_dag

        monkeypatch.chdir(tmp_path)
        _write_jsonl(tmp_path / "data" / "raw.jsonl", [{"id": "1", "text": "X"}])
        plan = self._plan(
            {
                "models": [
                    {
                        "name": "raw",
                        "kind": "table",
                        "source": "data/raw.jsonl",
                        "transform": "identity",
                    },
                    {
                        "name": "lc",
                        "kind": "view",
                        "refs": ["raw"],
                        "transform": "lowercase",
                    },
                    {
                        "name": "final",
                        "kind": "table",
                        "refs": ["lc"],
                        "transform": "identity",
                    },
                ]
            }
        )
        result = build_dag.run_build(plan, output_dir="out")
        # view not persisted
        assert not (tmp_path / "out" / "lc.jsonl").exists()
        lc_res = next(m for m in result.models if m.name == "lc")
        assert lc_res.output_path is None
        # but final (downstream of view) saw the lowercased rows
        final = tmp_path / "out" / "final.jsonl"
        rows = [json.loads(line) for line in final.read_text().splitlines() if line]
        assert rows[0]["text"] == "x"

    def test_incremental_only_retransforms_changed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils import build_dag

        monkeypatch.chdir(tmp_path)
        calls: list = []

        def counting(row, config):
            calls.append(row.get("id"))
            return {**row, "text": str(row.get("text", "")).upper()}

        src = tmp_path / "data" / "raw.jsonl"
        _write_jsonl(src, [{"id": "1", "text": "a"}, {"id": "2", "text": "b"}])
        plan = self._plan(
            {
                "models": [
                    {
                        "name": "up",
                        "kind": "incremental",
                        "source": "data/raw.jsonl",
                        "transform": "custom",
                    }
                ]
            }
        )
        # first build: both rows transformed
        build_dag.run_build(plan, output_dir="out", transforms={"custom": counting})
        assert sorted(calls) == ["1", "2"]
        calls.clear()
        # change row 2, add row 3, leave row 1
        _write_jsonl(
            src,
            [
                {"id": "1", "text": "a"},
                {"id": "2", "text": "B-CHANGED"},
                {"id": "3", "text": "c"},
            ],
        )
        result = build_dag.run_build(
            plan, output_dir="out", transforms={"custom": counting}
        )
        # only changed (2) + added (3) re-transformed; unchanged (1) carried over
        assert sorted(calls) == ["2", "3"]
        diff = result.models[0].diff
        assert diff is not None
        assert diff.added == 1
        assert diff.changed == 1
        assert diff.unchanged == 1
        assert diff.removed == 0

    def test_incremental_removes_dropped_rows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils import build_dag

        monkeypatch.chdir(tmp_path)
        src = tmp_path / "data" / "raw.jsonl"
        _write_jsonl(src, [{"id": "1", "text": "a"}, {"id": "2", "text": "b"}])
        plan = self._plan(
            {
                "models": [
                    {
                        "name": "up",
                        "kind": "incremental",
                        "source": "data/raw.jsonl",
                        "transform": "identity",
                    }
                ]
            }
        )
        build_dag.run_build(plan, output_dir="out")
        _write_jsonl(src, [{"id": "1", "text": "a"}])  # row 2 removed
        result = build_dag.run_build(plan, output_dir="out")
        rows = [
            json.loads(line)
            for line in (tmp_path / "out" / "up.jsonl").read_text().splitlines()
            if line
        ]
        assert [r["id"] for r in rows] == ["1"]
        assert result.models[0].diff.removed == 1

    def test_seed_rows_without_id_get_assigned(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils import build_dag

        monkeypatch.chdir(tmp_path)
        _write_jsonl(tmp_path / "data" / "raw.jsonl", [{"text": "a"}, {"text": "b"}])
        plan = self._plan(
            {
                "models": [
                    {
                        "name": "raw",
                        "kind": "incremental",
                        "source": "data/raw.jsonl",
                        "transform": "identity",
                    }
                ]
            }
        )
        # must not raise — ids auto-assigned for incremental diffing
        build_dag.run_build(plan, output_dir="out")
        rows = [
            json.loads(line)
            for line in (tmp_path / "out" / "raw.jsonl").read_text().splitlines()
            if line
        ]
        assert all("id" in r for r in rows)

    def test_run_build_validates_plan_type(self) -> None:
        from soup_cli.utils import build_dag

        with pytest.raises(TypeError):
            build_dag.run_build({"models": []}, output_dir="out")  # type: ignore[arg-type]

    def test_output_dir_outside_cwd_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils import build_dag

        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        _write_jsonl(work / "data" / "raw.jsonl", [{"id": "1", "text": "a"}])
        plan = self._plan(
            {
                "models": [
                    {
                        "name": "raw",
                        "kind": "table",
                        "source": "data/raw.jsonl",
                        "transform": "identity",
                    }
                ]
            }
        )
        with pytest.raises(ValueError):
            build_dag.run_build(plan, output_dir=str(tmp_path / "elsewhere"))

    def test_missing_source_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils import build_dag

        monkeypatch.chdir(tmp_path)
        plan = self._plan(
            {
                "models": [
                    {
                        "name": "raw",
                        "kind": "table",
                        "source": "data/missing.jsonl",
                        "transform": "identity",
                    }
                ]
            }
        )
        with pytest.raises((FileNotFoundError, ValueError)):
            build_dag.run_build(plan, output_dir="out")

    def test_transform_error_names_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils import build_dag

        monkeypatch.chdir(tmp_path)
        _write_jsonl(tmp_path / "data" / "raw.jsonl", [{"id": "1", "text": "a"}])

        def boom(row, config):
            raise RuntimeError("kaboom")

        plan = self._plan(
            {
                "models": [
                    {
                        "name": "raw",
                        "kind": "table",
                        "source": "data/raw.jsonl",
                        "transform": "custom",
                    }
                ]
            }
        )
        with pytest.raises(ValueError, match="raw"):
            build_dag.run_build(plan, output_dir="out", transforms={"custom": boom})

    def test_build_result_frozen(self) -> None:
        import dataclasses

        from soup_cli.utils import build_dag

        result = build_dag.BuildResult(models=(), output_dir="out")
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.output_dir = "x"  # type: ignore[misc]


class TestSoupBuildCliLive:
    def test_live_run_writes_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_jsonl(tmp_path / "data" / "raw.jsonl", [{"id": "1", "text": "Hi"}])
        _write(
            tmp_path / "build.yaml",
            "models:\n"
            "  - name: lc\n"
            "    kind: table\n"
            "    source: data/raw.jsonl\n"
            "    transform: lowercase\n",
        )
        runner = CliRunner()
        result = runner.invoke(app, ["build", "build.yaml", "--output-dir", "out"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert (tmp_path / "out" / "lc.jsonl").is_file()

    def test_dry_run_still_works(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write(
            tmp_path / "build.yaml",
            "models:\n  - name: raw\n    kind: table\n"
            "    source: data/raw.jsonl\n    transform: identity\n",
        )
        runner = CliRunner()
        result = runner.invoke(app, ["build", "build.yaml", "--dry-run"])
        assert result.exit_code == 0, result.output
        assert "raw" in result.output

    def test_live_run_no_deferred_advisory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_jsonl(tmp_path / "data" / "raw.jsonl", [{"id": "1", "text": "Hi"}])
        _write(
            tmp_path / "build.yaml",
            "models:\n  - name: raw\n    kind: table\n"
            "    source: data/raw.jsonl\n    transform: identity\n",
        )
        runner = CliRunner()
        result = runner.invoke(app, ["build", "build.yaml", "--output-dir", "out"])
        assert result.exit_code == 0, result.output
        assert "0.69.1" not in result.output
        # The old deferred-advisory panel title must be gone.
        assert "Live build deferred" not in result.output


# =============================================================================
# #232 — Live Magpie generator
# =============================================================================


def _varied_generate_fn():
    """Stateful stub: unique instruction per call so dedup still reaches target."""
    counter = {"n": 0}

    def gen(prompt: str) -> str:
        if "assistant" in prompt or "model\n" in prompt or "[/INST]" in prompt:
            return "The answer is four.<|im_end|>"
        counter["n"] += 1
        return f"What is {counter['n']} plus {counter['n']}?<|im_end|>"

    return gen


def _constant_generate_fn():
    def gen(prompt: str) -> str:
        if "assistant" in prompt or "model\n" in prompt or "[/INST]" in prompt:
            return "Same answer.<|im_end|>"
        return "Same instruction?<|im_end|>"

    return gen


class TestMagpiePrefix:
    def test_default_is_chatml(self) -> None:
        from soup_cli.utils import magpie

        assert "user" in magpie.magpie_prefix_for("some-unknown-model")

    def test_llama3_family(self) -> None:
        from soup_cli.utils import magpie

        prefix = magpie.magpie_prefix_for("meta-llama/Llama-3.1-8B-Instruct")
        assert "start_header_id" in prefix

    def test_gemma_family(self) -> None:
        from soup_cli.utils import magpie

        prefix = magpie.magpie_prefix_for("google/gemma-2-2b-it")
        assert "start_of_turn" in prefix

    def test_qwen_chatml(self) -> None:
        from soup_cli.utils import magpie

        prefix = magpie.magpie_prefix_for("Qwen/Qwen2.5-0.5B-Instruct")
        assert "im_start" in prefix

    def test_assistant_opener(self) -> None:
        from soup_cli.utils import magpie

        opener = magpie.magpie_assistant_opener("Qwen/Qwen2.5-0.5B-Instruct")
        assert "assistant" in opener

    def test_prefix_rejects_bad_base(self) -> None:
        from soup_cli.utils import magpie

        with pytest.raises((TypeError, ValueError)):
            magpie.magpie_prefix_for("")


class TestMagpieHarvest:
    def test_clean_cuts_at_stop_marker(self) -> None:
        from soup_cli.utils import magpie

        out = magpie._clean_generation(
            "  Hello there<|im_end|>extra junk", ["<|im_end|>"]
        )
        assert out == "Hello there"

    def test_harvest_instruction(self) -> None:
        from soup_cli.utils import magpie

        gen = _varied_generate_fn()
        instr = magpie.harvest_instruction(gen, "<|im_start|>user\n", ["<|im_end|>"])
        assert instr.startswith("What is 1 plus 1")
        assert "<|im_end|>" not in instr

    def test_harvest_empty_generation(self) -> None:
        from soup_cli.utils import magpie

        instr = magpie.harvest_instruction(lambda p: "", "<|im_start|>user\n", [])
        assert instr == ""


class TestMagpieQualityFilter:
    def test_keeps_clean(self) -> None:
        from soup_cli.utils import magpie

        assert magpie.default_quality_fn(
            "Explain how photosynthesis converts sunlight into chemical energy.",
            "Plants use chlorophyll to absorb light and produce glucose and oxygen.",
        ) is True

    def test_drops_empty_response(self) -> None:
        from soup_cli.utils import magpie

        assert magpie.default_quality_fn("good instruction here", "") is False


class TestRunMagpie:
    def _cfg(self, **kw):
        from soup_cli.utils import magpie

        defaults = dict(
            base_model="Qwen/Qwen2.5-0.5B-Instruct",
            provider="ollama",
            target_rows=3,
            quality_filter=False,
        )
        defaults.update(kw)
        return magpie.MagpieConfig(**defaults)

    def test_produces_target_rows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils import magpie

        monkeypatch.chdir(tmp_path)
        cfg = self._cfg(target_rows=3)
        result = magpie.run_magpie(
            cfg, output_path="out.jsonl", generate_fn=_varied_generate_fn()
        )
        rows = [
            json.loads(line)
            for line in (tmp_path / "out.jsonl").read_text().splitlines()
            if line
        ]
        assert len(rows) == 3
        assert rows[0]["messages"][0]["role"] == "user"
        assert rows[0]["messages"][1]["role"] == "assistant"
        assert result.rows_kept == 3

    def test_dedup_collapses_constant_instruction(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils import magpie

        monkeypatch.chdir(tmp_path)
        cfg = self._cfg(target_rows=5)
        result = magpie.run_magpie(
            cfg, output_path="out.jsonl", generate_fn=_constant_generate_fn()
        )
        assert result.rows_kept == 1
        assert result.duplicates >= 1

    def test_quality_filter_drops(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils import magpie

        monkeypatch.chdir(tmp_path)
        cfg = self._cfg(target_rows=2, quality_filter=True)
        # A rejecting quality_fn drops every (non-empty) row — distinct from
        # the earlier empty-response skip path.
        result = magpie.run_magpie(
            cfg,
            output_path="out.jsonl",
            generate_fn=_varied_generate_fn(),
            quality_fn=lambda instruction, response: False,
        )
        assert result.rows_kept == 0
        assert result.rows_filtered >= 1

    def test_quality_filter_empty_response_skipped_not_filtered(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils import magpie

        monkeypatch.chdir(tmp_path)
        cfg = self._cfg(target_rows=2, quality_filter=True)

        def gen(prompt: str) -> str:
            if "assistant" in prompt:
                return "<|im_end|>"  # empty response after cleaning
            return "What is x?<|im_end|>"

        result = magpie.run_magpie(cfg, output_path="out.jsonl", generate_fn=gen)
        # empty response => skipped before the quality filter (not counted as filtered)
        assert result.rows_kept == 0
        assert result.rows_filtered == 0

    def test_validates_config_type(self) -> None:
        from soup_cli.utils import magpie

        with pytest.raises(TypeError):
            magpie.run_magpie({"base": "m"}, output_path="out.jsonl")  # type: ignore[arg-type]

    def test_output_outside_cwd_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils import magpie

        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        cfg = self._cfg()
        with pytest.raises(ValueError):
            magpie.run_magpie(
                cfg,
                output_path=str(tmp_path / "out.jsonl"),
                generate_fn=_varied_generate_fn(),
            )

    @pytest.mark.skipif(
        __import__("sys").platform == "win32",
        reason="POSIX symlink semantics",
    )
    def test_output_symlink_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import os

        from soup_cli.utils import magpie

        monkeypatch.chdir(tmp_path)
        (tmp_path / "real").mkdir()
        os.symlink(tmp_path / "real" / "f.jsonl", tmp_path / "out.jsonl")
        cfg = self._cfg()
        with pytest.raises(ValueError):
            magpie.run_magpie(
                cfg, output_path="out.jsonl", generate_fn=_varied_generate_fn()
            )

    def test_result_frozen(self) -> None:
        import dataclasses

        from soup_cli.utils import magpie

        res = magpie.MagpieResult(
            rows_kept=1, rows_filtered=0, duplicates=0, attempts=1, output_path="x"
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            res.rows_kept = 2  # type: ignore[misc]


class TestMakeMagpieGenerateFn:
    def test_anthropic_rejected(self) -> None:
        from soup_cli.utils import magpie

        with pytest.raises(ValueError, match="raw"):
            magpie.make_magpie_generate_fn("anthropic", model="claude")

    def test_unknown_provider(self) -> None:
        from soup_cli.utils import magpie

        with pytest.raises(ValueError):
            magpie.make_magpie_generate_fn("openai", model="gpt")

    def test_ollama_generation_mocked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils import magpie

        class FakeResp:
            status_code = 200
            content = b'{"response": "What is 2+2?"}'

            def json(self):
                return {"response": "What is 2+2?"}

        import httpx

        monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResp())
        gen = magpie.make_magpie_generate_fn("ollama", model="qwen2.5:0.5b")
        assert gen("<|im_start|>user\n") == "What is 2+2?"

    def test_vllm_generation_mocked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from soup_cli.utils import magpie

        class FakeResp:
            status_code = 200
            content = b'{"choices": [{"text": "hello"}]}'

            def json(self):
                return {"choices": [{"text": "hello"}]}

        import httpx

        monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResp())
        gen = magpie.make_magpie_generate_fn("vllm", model="m")
        assert gen("prefix") == "hello"


class TestMagpieCliLive:
    def test_live_run_writes_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils import magpie

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            magpie, "make_magpie_generate_fn", lambda *a, **k: _varied_generate_fn()
        )
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "data",
                "gen-magpie",
                "--base",
                "Qwen/Qwen2.5-0.5B-Instruct",
                "--provider",
                "ollama",
                "--target",
                "2",
                "--output",
                "out.jsonl",
                "--no-quality-filter",
            ],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert (tmp_path / "out.jsonl").is_file()
        assert "0.69.1" not in result.output

    def test_live_requires_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "data",
                "gen-magpie",
                "--base",
                "m",
                "--provider",
                "ollama",
                "--target",
                "2",
            ],
        )
        assert result.exit_code == 2

    def test_plan_only_still_works(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "data",
                "gen-magpie",
                "--base",
                "m",
                "--provider",
                "ollama",
                "--target",
                "2",
                "--plan-only",
            ],
        )
        assert result.exit_code == 0, result.output


# =============================================================================
# #167 — tokenizer-aware memorization probe
# =============================================================================


class _CharTokenizer:
    """Duck-typed char-level tokenizer (one token id per character).

    Round-trips exactly, so it is a clean test seam for the sub-word boundary
    logic without pulling transformers.
    """

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)

    def convert_ids_to_tokens(self, ids):
        return [chr(i) for i in ids]


class TestResolveTokenizer:
    def test_duck_typed_object_returned_as_is(self) -> None:
        from soup_cli.utils.diagnose import _common

        tok = _CharTokenizer()
        assert _common.resolve_tokenizer(tok) is tok

    def test_non_string_non_tokenizer_rejected(self) -> None:
        from soup_cli.utils.diagnose import _common

        with pytest.raises(TypeError):
            _common.resolve_tokenizer(123)  # type: ignore[arg-type]

    def test_empty_string_rejected(self) -> None:
        from soup_cli.utils.diagnose import _common

        with pytest.raises(ValueError):
            _common.resolve_tokenizer("")

    def test_subword_tokens(self) -> None:
        from soup_cli.utils.diagnose import _common

        toks = _common.subword_tokens(_CharTokenizer(), "ab")
        assert toks == ["a", "b"]


class TestSplitPrefixTokenizer:
    def test_whitespace_default_unchanged(self) -> None:
        from soup_cli.utils.diagnose.memorization import split_prefix

        prefix, suffix = split_prefix("hello world foo bar", fraction=0.5)
        assert prefix == "hello world"
        assert suffix == "foo bar"

    def test_tokenizer_splits_on_token_boundary(self) -> None:
        from soup_cli.utils.diagnose.memorization import split_prefix

        # char tokenizer: 11 chars, fraction 0.5 -> cut at 5 chars.
        prefix, suffix = split_prefix(
            "hello world", fraction=0.5, tokenizer=_CharTokenizer()
        )
        assert prefix == "hello"
        assert suffix == " world"

    def test_tokenizer_empty_text(self) -> None:
        from soup_cli.utils.diagnose.memorization import split_prefix

        assert split_prefix("", tokenizer=_CharTokenizer()) == ("", "")

    def test_tokenizer_round_trips(self) -> None:
        from soup_cli.utils.diagnose.memorization import split_prefix

        text = "The quick brown fox"
        prefix, suffix = split_prefix(text, fraction=0.3, tokenizer=_CharTokenizer())
        assert prefix + suffix == text


class TestScoreMemorizationTokenizer:
    def test_exact_echo_subword_flags_memorization(self) -> None:
        from soup_cli.utils.diagnose.memorization import score_memorization

        rows = [{"text": "alpha beta gamma delta epsilon"}]
        # adapter echoes the held-out suffix verbatim -> high overlap.
        def echo_gen(prefix: str) -> str:
            # return the canonical suffix so subword overlap is ~1.0
            return "beta gamma delta epsilon"

        score = score_memorization(
            rows, echo_gen, prefix_fraction=0.2, tokenizer=_CharTokenizer()
        )
        assert score.mode == "memorization"
        assert score.score < 1.0  # echo detected
        assert score.verdict in {"MINOR", "MAJOR"}

    def test_no_echo_subword_is_ok(self) -> None:
        from soup_cli.utils.diagnose.memorization import score_memorization

        rows = [{"text": "alpha beta gamma delta epsilon"}]
        score = score_memorization(
            rows,
            lambda prefix: "zzzzz qqqqq wwwww",
            prefix_fraction=0.2,
            tokenizer=_CharTokenizer(),
        )
        assert score.verdict == "OK"

    def test_backcompat_no_tokenizer_still_works(self) -> None:
        from soup_cli.utils.diagnose.memorization import score_memorization

        rows = [{"text": "alpha beta gamma delta epsilon"}]
        score = score_memorization(rows, lambda prefix: "unrelated text here")
        assert score.mode == "memorization"
        assert score.verdict == "OK"

    def test_tokenizer_resolved_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from soup_cli.utils.diagnose import _common, memorization

        calls = {"n": 0}
        real = _common.resolve_tokenizer

        def counting(tok):
            calls["n"] += 1
            return real(tok)

        monkeypatch.setattr(memorization, "resolve_tokenizer", counting)
        rows = [{"text": "alpha beta gamma delta"} for _ in range(3)]
        memorization.score_memorization(
            rows, lambda p: "x", tokenizer=_CharTokenizer()
        )
        assert calls["n"] == 1  # resolved once, not per-row


# =============================================================================
# #213 — 2PL + 3PL IRT models
# =============================================================================


def _response_matrix_rows() -> list:
    """6 respondents r0..r5 (latent ability ascending) x several items.

    - ``const``  : everyone correct (carries ~no information).
    - ``split``  : correct iff respondent index >= 3 (clean threshold; high info).
    - ``easy``   : correct iff index >= 1 (easy threshold).
    """
    rows = []
    for j in range(6):
        rows.append({"respondent_id": f"r{j}", "item_id": "const", "correct": True})
        rows.append(
            {"respondent_id": f"r{j}", "item_id": "split", "correct": j >= 3}
        )
        rows.append({"respondent_id": f"r{j}", "item_id": "easy", "correct": j >= 1})
    return rows


class TestItemParameters:
    def test_frozen(self) -> None:
        import dataclasses

        from soup_cli.utils import irt

        p = irt.ItemParameters(
            item_id="a", difficulty=0.0, discrimination=1.0, guessing=0.0, info=0.25
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.difficulty = 1.0  # type: ignore[misc]

    def test_discrimination_must_be_positive(self) -> None:
        from soup_cli.utils import irt

        with pytest.raises(ValueError):
            irt.ItemParameters(
                item_id="a", difficulty=0.0, discrimination=0.0, guessing=0.0, info=0.1
            )

    def test_guessing_bounds(self) -> None:
        from soup_cli.utils import irt

        with pytest.raises(ValueError):
            irt.ItemParameters(
                item_id="a", difficulty=0.0, discrimination=1.0, guessing=1.5, info=0.1
            )

    def test_bool_discrimination_rejected(self) -> None:
        from soup_cli.utils import irt

        with pytest.raises(ValueError):
            irt.ItemParameters(
                item_id="a", difficulty=0.0, discrimination=True, guessing=0.0, info=0.1
            )

    def test_non_finite_rejected(self) -> None:
        from soup_cli.utils import irt

        with pytest.raises(ValueError):
            irt.ItemParameters(
                item_id="a",
                difficulty=float("nan"),
                discrimination=1.0,
                guessing=0.0,
                info=0.1,
            )


class TestFitIrt:
    def test_2pl_shape(self) -> None:
        from soup_cli.utils import irt

        params = irt.fit_irt(_response_matrix_rows(), model="2pl")
        assert len(params) == 3
        by_id = {p.item_id: p for p in params}
        assert set(by_id) == {"const", "split", "easy"}
        for p in params:
            assert p.discrimination > 0
            assert p.guessing == 0.0
            assert p.info >= 0
            import math

            assert math.isfinite(p.info)

    def test_discriminating_item_carries_more_info(self) -> None:
        from soup_cli.utils import irt

        by_id = {p.item_id: p for p in irt.fit_irt(_response_matrix_rows(), model="2pl")}
        # A constant (always-correct) item carries near-zero information; a
        # balanced threshold item is highly informative.
        assert by_id["split"].info > by_id["const"].info

    def test_deterministic(self) -> None:
        from soup_cli.utils import irt

        rows = _response_matrix_rows()
        a = irt.fit_irt(rows, model="2pl")
        b = irt.fit_irt(rows, model="2pl")
        assert [(p.item_id, p.difficulty, p.discrimination, p.info) for p in a] == [
            (p.item_id, p.difficulty, p.discrimination, p.info) for p in b
        ]

    def test_3pl_guessing_bounded(self) -> None:
        from soup_cli.utils import irt

        params = irt.fit_irt(_response_matrix_rows(), model="3pl")
        for p in params:
            assert 0.0 <= p.guessing <= irt.MAX_GUESSING

    def test_1pl_fixes_discrimination(self) -> None:
        from soup_cli.utils import irt

        params = irt.fit_irt(_response_matrix_rows(), model="1pl")
        for p in params:
            assert p.discrimination == 1.0
            assert p.guessing == 0.0

    def test_unknown_model_rejected(self) -> None:
        from soup_cli.utils import irt

        with pytest.raises(ValueError, match="model"):
            irt.fit_irt(_response_matrix_rows(), model="4pl")

    def test_missing_respondent_id_rejected(self) -> None:
        from soup_cli.utils import irt

        rows = [{"item_id": "a", "correct": True}]
        with pytest.raises(ValueError, match="respondent_id"):
            irt.fit_irt(rows, model="2pl")

    def test_non_bool_correct_rejected(self) -> None:
        from soup_cli.utils import irt

        rows = [{"respondent_id": "r0", "item_id": "a", "correct": 1}]
        with pytest.raises(ValueError):
            irt.fit_irt(rows, model="2pl")

    def test_empty_rejected(self) -> None:
        from soup_cli.utils import irt

        with pytest.raises(ValueError):
            irt.fit_irt([], model="2pl")

    def test_pick_subset_accepts_item_parameters(self) -> None:
        from soup_cli.utils import irt

        params = irt.fit_irt(_response_matrix_rows(), model="2pl")
        plan = irt.pick_irt_subset(params, size="small")
        assert plan.total_items == 3
        assert len(plan.item_ids) >= 1


class TestIrtSubsetCliModels:
    def _write_responses(self, path: Path) -> Path:
        rows = _response_matrix_rows()
        path.write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )
        return path

    def test_model_2pl(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        self._write_responses(tmp_path / "resp.jsonl")
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["eval", "irt-subset", "resp.jsonl", "--size", "small", "--model", "2pl"],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))

    def test_model_3pl(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        self._write_responses(tmp_path / "resp.jsonl")
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["eval", "irt-subset", "resp.jsonl", "--size", "tiny", "--model", "3pl"],
        )
        assert result.exit_code == 0, result.output

    def test_default_model_1pl_backcompat(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 1pl path stays {item_id, correct} (no respondent needed).
        monkeypatch.chdir(tmp_path)
        (tmp_path / "resp.jsonl").write_text(
            "\n".join(
                json.dumps({"item_id": f"q{i}", "correct": i % 2 == 0})
                for i in range(6)
            )
            + "\n",
            encoding="utf-8",
        )
        runner = CliRunner()
        result = runner.invoke(app, ["eval", "irt-subset", "resp.jsonl"])
        assert result.exit_code == 0, result.output

    def test_unknown_model_exit_2(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        self._write_responses(tmp_path / "resp.jsonl")
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["eval", "irt-subset", "resp.jsonl", "--model", "4pl"],
        )
        assert result.exit_code == 2


# =============================================================================
# #75 — synth-data QA: `data augment` provider fix (was ImportError on all)
# =============================================================================


class TestAugmentProviderFix:
    def test_ollama_provider_constructs(self) -> None:
        # Regression: previously raised ImportError (OllamaProvider missing).
        from soup_cli.commands.data import _load_augment_provider

        prov = _load_augment_provider("ollama", 60, model="qwen2.5:0.5b")
        assert hasattr(prov, "generate")

    def test_anthropic_provider_constructs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.commands.data import _load_augment_provider

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        prov = _load_augment_provider("anthropic", 60)
        assert hasattr(prov, "generate")

    def test_vllm_provider_constructs(self) -> None:
        from soup_cli.commands.data import _load_augment_provider

        prov = _load_augment_provider("vllm", 60)
        assert hasattr(prov, "generate")

    def test_unknown_provider_rejected(self) -> None:
        from soup_cli.commands.data import _load_augment_provider

        with pytest.raises(ValueError, match="Unknown provider"):
            _load_augment_provider("openai", 60)

    def test_generate_unwraps_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from soup_cli.commands.data import _load_augment_provider

        class FakeResp:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {"content": "rephrased!"}}]}

        import httpx

        monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResp())
        prov = _load_augment_provider("ollama", 60, model="qwen2.5:0.5b")
        assert prov.generate("hello") == "rephrased!"

    def test_generate_non_string_safe(self) -> None:
        from soup_cli.commands.data import _AugmentProvider

        prov = _AugmentProvider(lambda p: "not a mapping")
        assert prov.generate("x") == ""

    def test_augment_cli_has_model_flag(self) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["data", "augment", "--help"])
        assert result.exit_code == 0, result.output
        stripped = result.output.replace("\n", " ")
        # ANSI/box-drawing tolerant: the long-flag stem is present.
        assert "model" in stripped


# =============================================================================
# Review-fix follow-ups (security / code / python / tdd waves)
# =============================================================================


class TestValidateBuildSource:
    def test_none_passthrough(self) -> None:
        from soup_cli.utils import build_dag

        assert build_dag.validate_build_source(None) is None

    def test_under_cwd_ok(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from soup_cli.utils import build_dag

        monkeypatch.chdir(tmp_path)
        (tmp_path / "f.jsonl").write_text("{}\n", encoding="utf-8")
        assert build_dag.validate_build_source("f.jsonl") == "f.jsonl"

    def test_outside_cwd_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils import build_dag

        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        with pytest.raises(ValueError):
            build_dag.validate_build_source(str(tmp_path / "x.jsonl"))

    def test_non_string_rejected(self) -> None:
        from soup_cli.utils import build_dag

        with pytest.raises((TypeError, ValueError)):
            build_dag.validate_build_source(123)  # type: ignore[arg-type]

    @pytest.mark.skipif(
        __import__("sys").platform == "win32", reason="POSIX symlink semantics"
    )
    def test_symlink_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import os

        from soup_cli.utils import build_dag

        monkeypatch.chdir(tmp_path)
        (tmp_path / "real.jsonl").write_text("{}\n", encoding="utf-8")
        os.symlink(tmp_path / "real.jsonl", tmp_path / "link.jsonl")
        with pytest.raises(ValueError):
            build_dag.validate_build_source("link.jsonl")


class TestRunBuildReviewFixes:
    def _plan(self, raw: dict):
        from soup_cli.utils import build_dag

        return build_dag.parse_build_plan(raw)

    def test_incremental_dropped_row_stays_dropped_no_retransform(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils import build_dag

        monkeypatch.chdir(tmp_path)
        _write_jsonl(
            tmp_path / "data" / "raw.jsonl",
            [{"id": "1", "text": "x"}, {"id": "2", "text": "  "}],
        )
        plan = self._plan(
            {
                "models": [
                    {
                        "name": "clean",
                        "kind": "incremental",
                        "source": "data/raw.jsonl",
                        "transform": "drop_empty",
                    }
                ]
            }
        )
        build_dag.run_build(plan, output_dir="out")
        # identical 2nd run: nothing re-transformed; dropped row stays dropped.
        result = build_dag.run_build(plan, output_dir="out")
        m = result.models[0]
        assert m.transform_calls == 0
        assert m.rows_out == 1
        assert m.diff.unchanged == 2

    def test_incremental_duplicate_id_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils import build_dag

        monkeypatch.chdir(tmp_path)
        _write_jsonl(
            tmp_path / "data" / "raw.jsonl",
            [{"id": "dup", "text": "a"}, {"id": "dup", "text": "b"}],
        )
        plan = self._plan(
            {
                "models": [
                    {
                        "name": "m",
                        "kind": "incremental",
                        "source": "data/raw.jsonl",
                        "transform": "identity",
                    }
                ]
            }
        )
        with pytest.raises(ValueError, match="duplicate row id"):
            build_dag.run_build(plan, output_dir="out")

    def test_incremental_config_change_retransforms(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils import build_dag

        monkeypatch.chdir(tmp_path)
        _write_jsonl(tmp_path / "data" / "raw.jsonl", [{"id": "1", "text": "x"}])

        def _plan_with(value):
            return build_dag.parse_build_plan(
                {
                    "models": [
                        {
                            "name": "m",
                            "kind": "incremental",
                            "source": "data/raw.jsonl",
                            "transform": "add_field",
                            "config": {"field": "split", "value": value},
                        }
                    ]
                }
            )

        build_dag.run_build(_plan_with("train"), output_dir="out")
        # Same input rows, changed config value -> must re-run the transform.
        result = build_dag.run_build(_plan_with("test"), output_dir="out")
        assert result.models[0].transform_calls == 1
        rows = [
            json.loads(line)
            for line in (tmp_path / "out" / "m.jsonl").read_text().splitlines()
            if line
        ]
        assert rows[0]["split"] == "test"

    def test_unknown_transform_fails_before_any_write(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils import build_dag

        monkeypatch.chdir(tmp_path)
        _write_jsonl(tmp_path / "data" / "raw.jsonl", [{"id": "1", "text": "x"}])
        plan = self._plan(
            {
                "models": [
                    {
                        "name": "raw",
                        "kind": "table",
                        "source": "data/raw.jsonl",
                        "transform": "identity",
                    },
                    {
                        "name": "bad",
                        "kind": "table",
                        "refs": ["raw"],
                        "transform": "does_not_exist",
                    },
                ]
            }
        )
        with pytest.raises(ValueError, match="unknown transform"):
            build_dag.run_build(plan, output_dir="out")
        # fail-fast: the first model's artifact must NOT have been written.
        assert not (tmp_path / "out" / "raw.jsonl").exists()

    def test_build_model_config_frozen_on_direct_construction(self) -> None:
        from soup_cli.utils import build_dag

        m = build_dag.BuildModel(
            name="m",
            kind="table",
            transform="identity",
            refs=(),
            source="x.jsonl",
            config={"a": 1},
        )
        with pytest.raises(TypeError):
            m.config["b"] = 2  # type: ignore[index]

    def test_resolve_transform_non_callable_extra_rejected(self) -> None:
        from soup_cli.utils import build_dag

        with pytest.raises(TypeError):
            build_dag.resolve_transform("x", {"x": "not callable"})


class TestMagpieReviewFixes:
    def _cfg(self, **kw):
        from soup_cli.utils import magpie

        defaults = dict(
            base_model="Qwen/Qwen2.5-0.5B-Instruct",
            provider="ollama",
            target_rows=2,
            quality_filter=False,
        )
        defaults.update(kw)
        return magpie.MagpieConfig(**defaults)

    def test_empty_instruction_exhausts_attempts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils import magpie

        monkeypatch.chdir(tmp_path)
        result = magpie.run_magpie(
            self._cfg(target_rows=3),
            output_path="out.jsonl",
            generate_fn=lambda prompt: "",  # always empty -> never a row
            max_attempts=11,
        )
        assert result.rows_kept == 0
        assert result.attempts == 11  # loop terminated at the bound, no hang

    def test_dedup_false_keeps_duplicates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils import magpie

        monkeypatch.chdir(tmp_path)
        result = magpie.run_magpie(
            self._cfg(target_rows=3),
            output_path="out.jsonl",
            generate_fn=_constant_generate_fn(),
            dedup=False,
        )
        assert result.rows_kept == 3
        assert result.duplicates == 0

    def test_quality_fn_crash_warns_and_keeps(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import logging

        from soup_cli.utils import magpie

        monkeypatch.chdir(tmp_path)

        def boom(instruction, response):
            raise RuntimeError("scorer crashed")

        # The soup.magpie logger has propagate=False under the project logging
        # setup, so attach a capture handler directly (caplog can't see it).
        captured: list = []

        class _Capture(logging.Handler):
            def emit(self, record):
                captured.append((record.levelno, record.getMessage()))

        logger = logging.getLogger("soup.magpie")
        handler = _Capture()
        logger.addHandler(handler)
        try:
            result = magpie.run_magpie(
                self._cfg(target_rows=1, quality_filter=True),
                output_path="out.jsonl",
                generate_fn=_varied_generate_fn(),
                quality_fn=boom,
            )
        finally:
            logger.removeHandler(handler)
        assert result.rows_kept == 1  # kept despite the crash
        # Surfaced at WARNING (not DEBUG) — the code-review M3 contract.
        assert any(
            lvl == logging.WARNING and "quality_fn raised" in msg
            for lvl, msg in captured
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"model": ""},
            {"model": "x" * 513},
            {"temperature": 3.0},
            {"temperature": True},
            {"timeout_seconds": 0},
            {"timeout_seconds": 700},
            {"max_tokens": 0},
            {"max_tokens": True},
            {"max_tokens": 99999},
        ],
    )
    def test_make_generate_fn_validator_matrix(self, kwargs) -> None:
        from soup_cli.utils import magpie

        base = {"model": "m"}
        base.update(kwargs)
        with pytest.raises((TypeError, ValueError)):
            magpie.make_magpie_generate_fn("ollama", **base)

    def test_ollama_non_200_returns_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils import magpie

        class FakeResp:
            status_code = 500
            content = b"err"

            def json(self):
                return {}

        import httpx

        monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResp())
        gen = magpie.make_magpie_generate_fn("ollama", model="m")
        assert gen("prefix") == ""

    def test_ollama_parse_failure_returns_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils import magpie

        class FakeResp:
            status_code = 200
            content = b"{}"

            def json(self):
                return {}  # missing "response" key

        import httpx

        monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResp())
        gen = magpie.make_magpie_generate_fn("ollama", model="m")
        assert gen("prefix") == ""

    def test_ssrf_remote_base_url_rejected(self) -> None:
        from soup_cli.utils import magpie

        with pytest.raises(ValueError):
            magpie.make_magpie_generate_fn(
                "ollama", model="m", base_url="http://evil.example.com:11434"
            )

    def test_ssrf_zero_host_rejected(self) -> None:
        from soup_cli.utils import magpie

        # 0.0.0.0 is bind-any, not loopback (v0.71.6 #232 hardening).
        with pytest.raises(ValueError):
            magpie.make_magpie_generate_fn(
                "ollama", model="m", base_url="http://0.0.0.0:11434"
            )


class TestIrtReviewFixes:
    def test_3pl_deterministic(self) -> None:
        from soup_cli.utils import irt

        rows = _response_matrix_rows()
        a = irt.fit_irt(rows, model="3pl")
        b = irt.fit_irt(rows, model="3pl")
        assert [
            (p.item_id, p.difficulty, p.discrimination, p.guessing, p.info) for p in a
        ] == [
            (p.item_id, p.difficulty, p.discrimination, p.guessing, p.info) for p in b
        ]

    def test_null_byte_respondent_id_rejected(self) -> None:
        from soup_cli.utils import irt

        rows = [{"respondent_id": "r\x000", "item_id": "a", "correct": True}]
        with pytest.raises(ValueError):
            irt.fit_irt(rows, model="2pl")


class TestCommonReviewFixes:
    def test_subword_tokens_fallback_without_convert(self) -> None:
        from soup_cli.utils.diagnose import _common

        class NoConvert:
            def encode(self, text, add_special_tokens=False):
                return [1, 2, 3]

            def decode(self, ids, skip_special_tokens=True):
                return "x"

        toks = _common.subword_tokens(NoConvert(), "abc")
        assert toks == ["1", "2", "3"]


class TestAugmentProviderReviewFixes:
    def test_generate_non_string_text_field_safe(self) -> None:
        from soup_cli.commands.data import _AugmentProvider

        prov = _AugmentProvider(lambda p: {"text": 123})
        assert prov.generate("x") == ""

    def test_augment_output_atomic_no_partial_on_traversal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "in.jsonl").write_text(
            '{"instruction":"q","output":"a"}\n', encoding="utf-8"
        )
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "data",
                "augment",
                "--input",
                "in.jsonl",
                "--output",
                "../../evil.jsonl",
                "--strategy",
                "rephrase",
                "--provider",
                "ollama",
            ],
        )
        assert result.exit_code != 0
        assert not (tmp_path.parent.parent / "evil.jsonl").exists()


# =============================================================================
# Cross-cutting: version + no heavy top-level imports
# =============================================================================


class TestPatchInvariants:
    def test_version_bumped(self) -> None:
        from soup_cli import __version__

        major_minor = tuple(int(x) for x in __version__.split(".")[:3])
        assert major_minor >= (0, 71, 6)

    def test_no_heavy_top_level_imports(self) -> None:
        import pathlib

        root = pathlib.Path(__file__).resolve().parent.parent / "src" / "soup_cli"
        for rel in (
            "utils/build_dag.py",
            "utils/magpie.py",
            "utils/irt.py",
            "utils/diagnose/memorization.py",
            "utils/diagnose/_common.py",
        ):
            src = (root / rel).read_text(encoding="utf-8")
            for forbidden in (
                "\nimport torch",
                "\nimport transformers",
                "\nimport httpx",
                "\nimport sqlite3",
            ):
                assert forbidden not in src, f"{rel}: top-level {forbidden!r}"
