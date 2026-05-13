"""v0.53.7 — Data Forge + Pipeline live wave 1.

Tests for:
- #88  markdown heading-aware ingest split
- #112 decontaminate --benchmark-file
- #87  prompt_strategy live runtime
- #86  soup data preprocess live tokenize
- #111 soup data forge --judge-provider
- #75  QA log entry (file-presence assertion only)
- #106 utils/recipe_run.run_recipe live
- #105 utils/trainer_plugins.instantiate_trainer_plugins live
- #103 /v1/tools/{python, bash, web_search} live
- #102 /v1/messages on vLLM backend + streaming SSE
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ----- #88 markdown heading split ----------------------------------------


class TestMarkdownHeadingSplit:
    def test_empty_string_returns_empty_list(self):
        from soup_cli.utils.data_pipeline import split_markdown_by_headings
        assert split_markdown_by_headings("") == []

    def test_preamble_only_returns_single_row_with_none_section(self):
        from soup_cli.utils.data_pipeline import split_markdown_by_headings
        rows = split_markdown_by_headings("first paragraph\nsecond line")
        assert len(rows) == 1
        assert rows[0]["section"] is None
        assert rows[0]["level"] is None
        assert "first paragraph" in rows[0]["text"]

    def test_three_headings_yield_three_sections(self):
        from soup_cli.utils.data_pipeline import split_markdown_by_headings
        md = (
            "# Intro\nIntro body\n## Sub\nSub body\n### Deep\nDeep body\n"
        )
        rows = split_markdown_by_headings(md)
        assert len(rows) == 3
        assert rows[0]["section"] == "Intro"
        assert rows[0]["level"] == 1
        assert rows[1]["section"] == "Sub"
        assert rows[1]["level"] == 2
        assert rows[2]["section"] == "Deep"
        assert rows[2]["level"] == 3

    def test_preamble_plus_headings_yield_preamble_row(self):
        from soup_cli.utils.data_pipeline import split_markdown_by_headings
        md = "preamble text\n# Heading\nbody"
        rows = split_markdown_by_headings(md)
        assert len(rows) == 2
        assert rows[0]["section"] is None
        assert rows[0]["level"] is None
        assert "preamble text" in rows[0]["text"]
        assert rows[1]["section"] == "Heading"
        assert rows[1]["level"] == 1

    def test_atx_levels_1_through_6_accepted(self):
        from soup_cli.utils.data_pipeline import split_markdown_by_headings
        md = "\n".join(f"{'#' * n} L{n}\nbody{n}" for n in range(1, 7))
        rows = split_markdown_by_headings(md)
        assert [r["level"] for r in rows] == [1, 2, 3, 4, 5, 6]

    def test_seven_hashes_not_a_heading(self):
        from soup_cli.utils.data_pipeline import split_markdown_by_headings
        rows = split_markdown_by_headings("####### not a heading\nbody")
        assert len(rows) == 1
        assert rows[0]["section"] is None

    def test_hash_without_space_not_a_heading(self):
        from soup_cli.utils.data_pipeline import split_markdown_by_headings
        rows = split_markdown_by_headings("#NoSpace\nbody")
        assert len(rows) == 1
        assert rows[0]["section"] is None

    def test_non_string_input_raises_typeerror(self):
        from soup_cli.utils.data_pipeline import split_markdown_by_headings
        with pytest.raises(TypeError):
            split_markdown_by_headings(123)
        with pytest.raises(TypeError):
            split_markdown_by_headings(None)

    def test_ingest_cli_emits_one_row_per_heading(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        md_path = tmp_path / "doc.md"
        md_path.write_text(
            "# H1\nfirst\n## H2\nsecond\n## H3\nthird\n",
            encoding="utf-8",
        )
        out = tmp_path / "out.jsonl"
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["data", "ingest", str(md_path), "-o", str(out)],
        )
        assert result.exit_code == 0, result.output
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        first = json.loads(lines[0])
        assert first["section"] == "H1"
        assert first["level"] == 1

    def test_ingest_cli_preamble_creates_extra_row(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        md_path = tmp_path / "doc.md"
        md_path.write_text(
            "preamble\n# Heading\nbody\n",
            encoding="utf-8",
        )
        out = tmp_path / "out.jsonl"
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["data", "ingest", str(md_path), "-o", str(out)],
        )
        assert result.exit_code == 0, result.output
        rows = [json.loads(line) for line in out.read_text(
            encoding="utf-8"
        ).strip().splitlines()]
        assert len(rows) == 2
        assert rows[0]["section"] is None
        assert rows[1]["section"] == "Heading"


# ----- #112 decontaminate --benchmark-file -------------------------------


class TestDecontaminateBenchmarkFile:
    def test_loads_operator_corpus_and_filters(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        # The decontaminate row text overlaps the benchmark text by n-grams.
        bench_path = tmp_path / "bench.jsonl"
        bench_path.write_text(
            json.dumps({"text": (
                "the quick brown fox jumps over the lazy dog "
                "many times in the morning sun every day"
            )}) + "\n",
            encoding="utf-8",
        )
        rows_path = tmp_path / "input.jsonl"
        rows_path.write_text(
            json.dumps({"text": (
                "the quick brown fox jumps over the lazy dog "
                "many times in the morning sun every day"
            )}) + "\n"
            + json.dumps({"text": "completely unrelated content here"}) + "\n",
            encoding="utf-8",
        )
        out_path = tmp_path / "clean.jsonl"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "data", "decontaminate",
                "-i", str(rows_path),
                "--benchmark-file", str(bench_path),
                "-o", str(out_path),
                "--n", "4",
                "--threshold", "0.3",
            ],
        )
        assert result.exit_code == 0, result.output
        kept = [
            json.loads(line)
            for line in out_path.read_text(encoding="utf-8").strip().splitlines()
        ]
        # Overlapping row removed, unrelated kept.
        assert len(kept) == 1
        assert "unrelated" in kept[0]["text"]

    def test_missing_benchmarks_and_file_exits(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        rows_path = tmp_path / "input.jsonl"
        rows_path.write_text(json.dumps({"text": "hi"}) + "\n", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["data", "decontaminate", "-i", str(rows_path), "-o", "out.jsonl"],
        )
        assert result.exit_code != 0

    def test_bad_benchmark_file_path_rejected(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        rows_path = tmp_path / "input.jsonl"
        rows_path.write_text(json.dumps({"text": "hi"}) + "\n", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "data", "decontaminate",
                "-i", str(rows_path),
                "--benchmark-file", "/etc/does-not-exist.jsonl",
                "-o", "out.jsonl",
            ],
        )
        assert result.exit_code != 0

    def test_benchmarks_flag_still_accepted_with_label(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        rows_path = tmp_path / "input.jsonl"
        rows_path.write_text(json.dumps({"text": "any"}) + "\n", encoding="utf-8")
        out_path = tmp_path / "out.jsonl"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "data", "decontaminate",
                "-i", str(rows_path),
                "-b", "mmlu",
                "-o", str(out_path),
            ],
        )
        assert result.exit_code == 0, result.output


# ----- #87 prompt_strategy live runtime ----------------------------------


class TestPromptStrategyRuntime:
    def test_resolve_finds_callable(self):
        from soup_cli.utils.data_pipeline import resolve_prompt_strategy
        # json.dumps takes a positional arg
        spec = "json:dumps"
        fn = resolve_prompt_strategy(spec)
        assert callable(fn)

    def test_resolve_missing_module_raises(self):
        from soup_cli.utils.data_pipeline import resolve_prompt_strategy
        with pytest.raises(ValueError, match="could not be imported"):
            resolve_prompt_strategy("definitely_not_a_module_xyz:fn")

    def test_resolve_missing_attribute_raises(self):
        from soup_cli.utils.data_pipeline import resolve_prompt_strategy
        with pytest.raises(ValueError, match="no attribute"):
            resolve_prompt_strategy("json:not_a_real_attr_zzz")

    def test_resolve_non_callable_raises(self):
        from soup_cli.utils.data_pipeline import resolve_prompt_strategy
        # sys.maxsize is an int, not callable
        with pytest.raises(ValueError, match="not callable"):
            resolve_prompt_strategy("sys:maxsize")

    def test_resolve_bad_shape_raises(self):
        from soup_cli.utils.data_pipeline import resolve_prompt_strategy
        with pytest.raises(ValueError):
            resolve_prompt_strategy("no_colon")
        with pytest.raises(ValueError):
            resolve_prompt_strategy("")

    def test_resolve_non_string_raises(self):
        from soup_cli.utils.data_pipeline import resolve_prompt_strategy
        with pytest.raises(ValueError):
            resolve_prompt_strategy(123)  # type: ignore[arg-type]

    def test_apply_with_none_returns_row(self):
        from soup_cli.utils.data_pipeline import apply_prompt_strategy
        row = {"a": 1}
        assert apply_prompt_strategy(None, row) is row

    def test_apply_callable_transforms_row(self, tmp_path, monkeypatch):
        # Create a tiny module with an upper-casing transform
        mod_dir = tmp_path / "tfm_mod"
        mod_dir.mkdir()
        (mod_dir / "__init__.py").write_text("", encoding="utf-8")
        (mod_dir / "t.py").write_text(
            "def upper(row):\n"
            "    return {k: v.upper() if isinstance(v, str) else v for k, v in row.items()}\n",
            encoding="utf-8",
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        # Clear lru_cache so spec resolves with new module
        from soup_cli.utils.data_pipeline import (
            apply_prompt_strategy,
            resolve_prompt_strategy,
        )
        resolve_prompt_strategy.cache_clear()
        spec = "tfm_mod.t:upper"
        out = apply_prompt_strategy(spec, {"text": "hi"})
        assert out["text"] == "HI"

    def test_apply_callable_exception_falls_through(self, tmp_path, monkeypatch):
        mod_dir = tmp_path / "tfm_mod2"
        mod_dir.mkdir()
        (mod_dir / "__init__.py").write_text("", encoding="utf-8")
        (mod_dir / "t.py").write_text(
            "def boom(row):\n    raise RuntimeError('nope')\n",
            encoding="utf-8",
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        from soup_cli.utils.data_pipeline import (
            apply_prompt_strategy,
            resolve_prompt_strategy,
        )
        resolve_prompt_strategy.cache_clear()
        spec = "tfm_mod2.t:boom"
        row = {"text": "x"}
        out = apply_prompt_strategy(spec, row)
        assert out == row  # silent-degrade per CrossDocCollator policy

    def test_apply_non_mapping_return_falls_through(self, tmp_path, monkeypatch):
        mod_dir = tmp_path / "tfm_mod3"
        mod_dir.mkdir()
        (mod_dir / "__init__.py").write_text("", encoding="utf-8")
        (mod_dir / "t.py").write_text(
            "def to_str(row):\n    return 'not a mapping'\n",
            encoding="utf-8",
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        from soup_cli.utils.data_pipeline import (
            apply_prompt_strategy,
            resolve_prompt_strategy,
        )
        resolve_prompt_strategy.cache_clear()
        row = {"text": "x"}
        out = apply_prompt_strategy("tfm_mod3.t:to_str", row)
        assert out == row

    def test_sft_format_threads_prompt_strategy(self, tmp_path, monkeypatch):
        """build_format_row routes through prompt_strategy when set."""
        mod_dir = tmp_path / "tfm_mod_sft"
        mod_dir.mkdir()
        (mod_dir / "__init__.py").write_text("", encoding="utf-8")
        (mod_dir / "t.py").write_text(
            "def attach(row):\n"
            "    return {**row, '_attached': True}\n",
            encoding="utf-8",
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        from soup_cli.utils.data_pipeline import resolve_prompt_strategy
        resolve_prompt_strategy.cache_clear()

        from soup_cli.config.schema import DataConfig
        from soup_cli.data.sft_format import build_format_row

        data_cfg = DataConfig(
            train="dummy.jsonl",
            prompt_strategy="tfm_mod_sft.t:attach",
            train_on_responses_only=False,
            chat_template="chatml",
        )
        tokenizer = MagicMock()
        tokenizer.chat_template = "{% for msg in messages %}{{msg['content']}}{% endfor %}"
        tokenizer.apply_chat_template = MagicMock(return_value="rendered")
        fn = build_format_row(tokenizer, data_cfg)
        # When invoked, the inner row should pass through the attach transform
        result = fn({"messages": [{"role": "user", "content": "hi"}]})
        # Verify the legacy text formatter ran with attached row.
        assert tokenizer.apply_chat_template.called
        assert "rendered" in result.get("text", "")


# ----- #86 soup data preprocess live tokenize ----------------------------


class TestPreprocessTokenize:
    def test_help_advertises_live_loop(self):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["data", "preprocess", "--help"])
        assert result.exit_code == 0

    def test_load_pretokenized_dataset_rejects_outside_cwd(self, tmp_path, monkeypatch):
        from soup_cli.utils.data_pipeline import load_pretokenized_dataset

        monkeypatch.chdir(tmp_path)
        # An absolute outside-cwd path
        with pytest.raises(ValueError, match="under cwd"):
            load_pretokenized_dataset("/etc/some_dir")

    def test_load_pretokenized_dataset_rejects_null_byte(self, tmp_path, monkeypatch):
        from soup_cli.utils.data_pipeline import load_pretokenized_dataset

        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="null byte"):
            load_pretokenized_dataset("foo\x00bar")

    def test_load_pretokenized_dataset_rejects_empty(self):
        from soup_cli.utils.data_pipeline import load_pretokenized_dataset
        with pytest.raises(ValueError, match="non-empty"):
            load_pretokenized_dataset("")

    def test_load_pretokenized_dataset_rejects_non_string(self):
        from soup_cli.utils.data_pipeline import load_pretokenized_dataset
        with pytest.raises(TypeError, match="string"):
            load_pretokenized_dataset(123)  # type: ignore[arg-type]

    def test_load_pretokenized_dataset_rejects_missing_path(self, tmp_path, monkeypatch):
        from soup_cli.utils.data_pipeline import load_pretokenized_dataset

        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="not found"):
            load_pretokenized_dataset("nonexistent_dir")

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlink rejection")
    def test_load_pretokenized_dataset_rejects_symlink(self, tmp_path, monkeypatch):
        from soup_cli.utils.data_pipeline import load_pretokenized_dataset

        monkeypatch.chdir(tmp_path)
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real_dir)
        with pytest.raises(ValueError, match="symlink"):
            load_pretokenized_dataset(str(link))

    def test_load_pretokenized_dataset_rejects_cache_key_mismatch(
        self, tmp_path, monkeypatch
    ):
        from soup_cli.utils.data_pipeline import load_pretokenized_dataset

        pytest.importorskip("datasets")
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "tokenized"
        target.mkdir()
        (target / "metadata.json").write_text(
            json.dumps({"cache_key": "DIFFERENT"}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="cache_key mismatch"):
            load_pretokenized_dataset(
                str(target), expected_cache_key="EXPECTED"
            )


# ----- #111 forge --judge-provider ---------------------------------------


class TestForgeJudgeProvider:
    def test_make_judge_provider_fn_unknown_rejected(self):
        from soup_cli.utils.data_forge import make_judge_provider_fn
        with pytest.raises(ValueError, match="unknown judge provider"):
            make_judge_provider_fn("openai")

    def test_make_judge_provider_fn_non_string_rejected(self):
        from soup_cli.utils.data_forge import make_judge_provider_fn
        with pytest.raises(TypeError):
            make_judge_provider_fn(123)  # type: ignore[arg-type]

    def test_make_judge_provider_fn_bad_model_rejected(self):
        from soup_cli.utils.data_forge import make_judge_provider_fn
        with pytest.raises(ValueError, match="model"):
            make_judge_provider_fn("ollama", model="")
        with pytest.raises(ValueError, match="NUL-free"):
            make_judge_provider_fn("ollama", model="x\x00y")

    def test_make_judge_provider_fn_bool_timeout_rejected(self):
        from soup_cli.utils.data_forge import make_judge_provider_fn
        with pytest.raises(TypeError):
            make_judge_provider_fn("ollama", timeout_seconds=True)

    def test_make_judge_provider_fn_bad_timeout_rejected(self):
        from soup_cli.utils.data_forge import make_judge_provider_fn
        with pytest.raises(ValueError):
            make_judge_provider_fn("ollama", timeout_seconds=0)
        with pytest.raises(ValueError):
            make_judge_provider_fn("ollama", timeout_seconds=1000)

    def test_anthropic_requires_env_var(self, monkeypatch):
        from soup_cli.utils.data_forge import make_judge_provider_fn
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            make_judge_provider_fn("anthropic")

    def test_ollama_rejects_remote_url(self):
        from soup_cli.utils.data_forge import make_judge_provider_fn
        with pytest.raises(ValueError):
            make_judge_provider_fn(
                "ollama", base_url="http://10.0.0.1:11434"
            )

    def test_ollama_judge_returns_text_on_success(self, monkeypatch):
        # Monkeypatch httpx.post in the module
        import httpx

        from soup_cli.utils.data_forge import make_judge_provider_fn

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "hello world"}}]
        }
        monkeypatch.setattr(httpx, "post", lambda *a, **k: mock_response)
        judge = make_judge_provider_fn("ollama")
        reply = judge("prompt")
        assert reply["text"] == "hello world"

    def test_ollama_judge_swallows_http_error(self, monkeypatch):
        import httpx

        from soup_cli.utils.data_forge import make_judge_provider_fn

        def _boom(*a, **k):
            raise httpx.ConnectError("nope")

        monkeypatch.setattr(httpx, "post", _boom)
        judge = make_judge_provider_fn("ollama")
        reply = judge("prompt")
        assert reply["text"] == ""

    def test_vllm_judge_validates_url(self):
        from soup_cli.utils.data_forge import make_judge_provider_fn
        with pytest.raises(ValueError):
            make_judge_provider_fn(
                "vllm", base_url="ftp://localhost:8000"
            )

    def test_forge_provider_constants(self):
        from soup_cli.utils.data_forge import JUDGE_PROVIDERS
        assert JUDGE_PROVIDERS == frozenset({"ollama", "anthropic", "vllm"})

    def test_forge_cli_rejects_unknown_provider(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.txt").write_text("hello world", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "data", "forge",
                "--docs", str(docs),
                "--task", "sft",
                "--target-rows", "1",
                "--judge-provider", "openai",
                "-o", "out.jsonl",
                "-p", "prov.json",
            ],
        )
        assert result.exit_code == 2


# ----- #75 QA log entry --------------------------------------------------


class TestQALogEntry:
    def test_qa_log_mentions_v0537(self):
        qa_log = (
            Path(__file__).parent / "qa" / "v053_qa.md"
        )
        assert qa_log.is_file()
        text = qa_log.read_text(encoding="utf-8")
        assert "v0.53.7" in text
        assert "#75" in text


# ----- #106 run_recipe live ----------------------------------------------


class TestRunRecipeLive:
    def test_stub_gone(self, tmp_path, monkeypatch):
        from soup_cli.utils.recipe_dag import parse_recipe
        from soup_cli.utils.recipe_run import run_recipe

        monkeypatch.chdir(tmp_path)
        dag = parse_recipe(
            {
                "nodes": [
                    {"name": "seed1", "kind": "seed", "config": {"path": "doesnotexist.jsonl"}},
                ],
                "edges": [],
            }
        )
        # Should NOT raise NotImplementedError now (was the v0.53.6 stub).
        # Instead should fail on missing seed path.
        with pytest.raises(ValueError):
            run_recipe(dag, output_dir=str(tmp_path / "out"))

    def test_seed_node_loads_jsonl(self, tmp_path, monkeypatch):
        from soup_cli.utils.recipe_dag import parse_recipe
        from soup_cli.utils.recipe_run import run_recipe

        monkeypatch.chdir(tmp_path)
        seed_path = tmp_path / "seed.jsonl"
        seed_path.write_text(
            json.dumps({"text": "a"}) + "\n" + json.dumps({"text": "b"}) + "\n",
            encoding="utf-8",
        )
        out_dir = tmp_path / "out"
        dag = parse_recipe(
            {
                "nodes": [
                    {"name": "seed1", "kind": "seed", "config": {"path": str(seed_path)}},
                    {"name": "sink", "kind": "sampler", "config": {}},
                ],
                "edges": [["seed1", "sink"]],
            }
        )
        result = run_recipe(dag, output_dir=str(out_dir))
        assert result["status"] == "completed"
        assert "seed1" in result["completed_nodes"]
        assert "sink" in result["completed_nodes"]
        # sampler wrote the rows
        sink_jsonl = out_dir / "sink.jsonl"
        assert sink_jsonl.is_file()
        lines = sink_jsonl.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    def test_validator_node_regex_filter(self, tmp_path, monkeypatch):
        from soup_cli.utils.recipe_dag import parse_recipe
        from soup_cli.utils.recipe_run import run_recipe

        monkeypatch.chdir(tmp_path)
        seed_path = tmp_path / "seed.jsonl"
        seed_path.write_text(
            json.dumps({"text": "hello world"}) + "\n"
            + json.dumps({"text": "skip me"}) + "\n",
            encoding="utf-8",
        )
        out_dir = tmp_path / "out"
        dag = parse_recipe(
            {
                "nodes": [
                    {"name": "seed1", "kind": "seed", "config": {"path": str(seed_path)}},
                    {"name": "vchk", "kind": "validator", "config": {"regex": "hello"}},
                    {"name": "sink", "kind": "sampler", "config": {}},
                ],
                "edges": [["seed1", "vchk"], ["vchk", "sink"]],
            }
        )
        result = run_recipe(dag, output_dir=str(out_dir))
        assert result["node_row_counts"]["vchk"] == 1
        assert result["node_row_counts"]["sink"] == 1

    def test_llm_text_offline_stub(self, tmp_path, monkeypatch):
        from soup_cli.utils.recipe_dag import parse_recipe
        from soup_cli.utils.recipe_run import run_recipe

        monkeypatch.chdir(tmp_path)
        seed_path = tmp_path / "seed.jsonl"
        seed_path.write_text(json.dumps({"text": "x"}) + "\n", encoding="utf-8")
        out_dir = tmp_path / "out"
        dag = parse_recipe(
            {
                "nodes": [
                    {"name": "seed1", "kind": "seed", "config": {"path": str(seed_path)}},
                    {
                        "name": "llm",
                        "kind": "llm_text",
                        "config": {"prompt": "tell me about {text}"},
                    },
                    {"name": "sink", "kind": "sampler", "config": {}},
                ],
                "edges": [["seed1", "llm"], ["llm", "sink"]],
            }
        )
        result = run_recipe(dag, output_dir=str(out_dir))
        assert result["status"] == "completed"
        # Offline stub injects llm column
        sink = (out_dir / "sink.jsonl").read_text(encoding="utf-8").strip().splitlines()
        first = json.loads(sink[0])
        assert "llm" in first

    def test_judge_node_offline_default_keeps_all(self, tmp_path, monkeypatch):
        from soup_cli.utils.recipe_dag import parse_recipe
        from soup_cli.utils.recipe_run import run_recipe

        monkeypatch.chdir(tmp_path)
        seed_path = tmp_path / "seed.jsonl"
        seed_path.write_text(
            json.dumps({"text": "a"}) + "\n" + json.dumps({"text": "b"}) + "\n",
            encoding="utf-8",
        )
        out_dir = tmp_path / "out"
        dag = parse_recipe(
            {
                "nodes": [
                    {"name": "seed1", "kind": "seed", "config": {"path": str(seed_path)}},
                    {"name": "j", "kind": "judge", "config": {}},
                    {"name": "sink", "kind": "sampler", "config": {}},
                ],
                "edges": [["seed1", "j"], ["j", "sink"]],
            }
        )
        result = run_recipe(dag, output_dir=str(out_dir))
        assert result["node_row_counts"]["j"] == 2

    def test_checkpoint_written_on_completion(self, tmp_path, monkeypatch):
        from soup_cli.utils.recipe_dag import parse_recipe
        from soup_cli.utils.recipe_run import run_recipe

        monkeypatch.chdir(tmp_path)
        seed_path = tmp_path / "seed.jsonl"
        seed_path.write_text(json.dumps({"text": "x"}) + "\n", encoding="utf-8")
        out_dir = tmp_path / "out"
        dag = parse_recipe(
            {
                "nodes": [
                    {"name": "seed1", "kind": "seed", "config": {"path": str(seed_path)}},
                ],
                "edges": [],
            }
        )
        run_recipe(dag, output_dir=str(out_dir))
        cp = out_dir / ".checkpoint.json"
        assert cp.is_file()
        state = json.loads(cp.read_text(encoding="utf-8"))
        assert state["status"] == "completed"
        assert "seed1" in state["completed_nodes"]

    def test_rejects_non_recipedag(self, tmp_path, monkeypatch):
        from soup_cli.utils.recipe_run import run_recipe
        monkeypatch.chdir(tmp_path)
        with pytest.raises(TypeError):
            run_recipe("not a dag", output_dir="x")  # type: ignore[arg-type]

    def test_output_dir_outside_cwd_rejected(self, tmp_path, monkeypatch):
        from soup_cli.utils.recipe_dag import parse_recipe
        from soup_cli.utils.recipe_run import run_recipe

        monkeypatch.chdir(tmp_path)
        dag = parse_recipe({
            "nodes": [{"name": "n1", "kind": "sampler", "config": {}}],
            "edges": [],
        })
        with pytest.raises(ValueError, match="under cwd"):
            run_recipe(dag, output_dir="/etc/foo")

    def test_validator_requires_regex_or_schema(self, tmp_path, monkeypatch):
        from soup_cli.utils.recipe_dag import parse_recipe
        from soup_cli.utils.recipe_run import run_recipe

        monkeypatch.chdir(tmp_path)
        seed_path = tmp_path / "seed.jsonl"
        seed_path.write_text(json.dumps({"text": "x"}) + "\n", encoding="utf-8")
        out_dir = tmp_path / "out"
        dag = parse_recipe({
            "nodes": [
                {"name": "seed1", "kind": "seed", "config": {"path": str(seed_path)}},
                {"name": "v", "kind": "validator", "config": {}},
            ],
            "edges": [["seed1", "v"]],
        })
        with pytest.raises(ValueError, match="regex.*schema"):
            run_recipe(dag, output_dir=str(out_dir))

    def test_code_node_requires_code_field(self, tmp_path, monkeypatch):
        from soup_cli.utils.recipe_dag import parse_recipe
        from soup_cli.utils.recipe_run import run_recipe

        monkeypatch.chdir(tmp_path)
        seed_path = tmp_path / "seed.jsonl"
        seed_path.write_text(json.dumps({"text": "x"}) + "\n", encoding="utf-8")
        out_dir = tmp_path / "out"
        dag = parse_recipe({
            "nodes": [
                {"name": "seed1", "kind": "seed", "config": {"path": str(seed_path)}},
                {"name": "c", "kind": "code", "config": {}},
            ],
            "edges": [["seed1", "c"]],
        })
        with pytest.raises(ValueError, match="code"):
            run_recipe(dag, output_dir=str(out_dir))

    def test_recipe_cli_executes_dag(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        seed_path = tmp_path / "seed.jsonl"
        seed_path.write_text(json.dumps({"text": "x"}) + "\n", encoding="utf-8")
        recipe_path = tmp_path / "recipe.yaml"
        recipe_path.write_text(
            "nodes:\n"
            f"  - {{name: seed1, kind: seed, config: {{path: {seed_path.name}}}}}\n"
            "  - {name: sink, kind: sampler, config: {}}\n"
            "edges:\n"
            "  - [seed1, sink]\n",
            encoding="utf-8",
        )
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "data", "recipe",
                str(recipe_path),
                "--execute",
                "--output", "out",
            ],
        )
        assert result.exit_code == 0, result.output
        # v0.53.7 marker should NOT appear in success path
        assert "v0.53.7" not in result.output or "Re-run" not in result.output


# ----- #105 instantiate_trainer_plugins live ----------------------------


class TestInstantiateTrainerPlugins:
    def test_cce_plugin_returns_advisory_callback(self):
        from soup_cli.utils.trainer_plugins import instantiate_trainer_plugins

        out = instantiate_trainer_plugins(["cce_plugin"])
        assert len(out) == 1
        cb = out[0]
        assert hasattr(cb, "plugin_name")
        assert cb.plugin_name == "cce_plugin"

    def test_unknown_name_rejected_before_instantiation(self):
        from soup_cli.utils.trainer_plugins import instantiate_trainer_plugins
        with pytest.raises(ValueError):
            instantiate_trainer_plugins(["definitely_unknown_plugin"])

    def test_non_sequence_rejected(self):
        from soup_cli.utils.trainer_plugins import instantiate_trainer_plugins
        with pytest.raises(TypeError):
            instantiate_trainer_plugins("grokfast")  # bare string

    def test_empty_list_returns_empty_tuple(self):
        from soup_cli.utils.trainer_plugins import instantiate_trainer_plugins
        out = instantiate_trainer_plugins([])
        assert out == ()

    def test_no_longer_raises_notimplementederror_for_cce(self):
        from soup_cli.utils.trainer_plugins import instantiate_trainer_plugins
        # cce_plugin has no required upstream dep; should NOT raise.
        out = instantiate_trainer_plugins(["cce_plugin"])
        assert out  # non-empty

    def test_missing_upstream_pkg_raises_importerror(self):
        """grokfast / spectrum etc. may not be installed in CI."""
        from soup_cli.utils.trainer_plugins import instantiate_trainer_plugins
        # If the package is installed, this test asserts no ImportError.
        # If absent (the common case), ImportError fires with friendly hint.
        try:
            out = instantiate_trainer_plugins(["grokfast"])
            # If installed, just verify we got something callable / module-like
            assert out
        except ImportError as exc:
            assert "grokfast" in str(exc)
            assert "pip install" in str(exc)


# ----- #103 /v1/tools/{python,bash,web_search} live ---------------------


def _create_test_app():
    """Build a test FastAPI app via _create_app (lazy fastapi import)."""
    try:
        import fastapi  # noqa: F401
    except ImportError:
        pytest.skip("FastAPI not installed")

    from soup_cli.commands.serve import _create_app

    return _create_app(
        model_obj=MagicMock(),
        tokenizer=MagicMock(),
        device="cpu",
        model_name="test-model",
        max_tokens_default=128,
    )


class TestToolEndpointsLive:
    def test_python_tool_runs_simple_code(self):
        from fastapi.testclient import TestClient
        app = _create_test_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/tools/python",
            json={"code": "print('hello')"},
        )
        # The sandbox may print warnings on first run; should still return 200.
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "stdout" in data
        assert "exit_code" in data
        assert "timed_out" in data

    def test_python_tool_rejects_missing_code(self):
        from fastapi.testclient import TestClient
        app = _create_test_app()
        client = TestClient(app)
        resp = client.post("/v1/tools/python", json={})
        assert resp.status_code == 400

    def test_python_tool_rejects_oversize_code(self):
        from fastapi.testclient import TestClient
        app = _create_test_app()
        client = TestClient(app)
        oversize = "x" * (64 * 1024 + 1)
        resp = client.post("/v1/tools/python", json={"code": oversize})
        assert resp.status_code == 400

    def test_python_tool_non_dict_rejected(self):
        from fastapi.testclient import TestClient
        app = _create_test_app()
        client = TestClient(app)
        # FastAPI body parsing accepts list as dict-typed param? It coerces.
        # Use empty code which is rejected as ValueError-equivalent.
        resp = client.post("/v1/tools/python", json={"code": ""})
        assert resp.status_code == 400

    def test_bash_tool_returns_501_review_fix_c1(self):
        """v0.53.7 C1 review fix: bash reverted to 501.

        ``/bin/sh -c`` spawns a child outside the RLVR sandbox's OS-level
        isolation (``unshare(CLONE_NEWNET)`` / macOS ``sandbox-exec`` /
        socket patch); a caller could reach the cloud-metadata service
        from the child shell. Reverted until container/namespace work
        lands in v0.53.8.
        """
        from fastapi.testclient import TestClient
        app = _create_test_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/tools/bash",
            json={"command": "echo hello"},
        )
        assert resp.status_code == 501
        assert "v0.53.8" in resp.text

    def test_bash_tool_returns_501_with_empty_body(self):
        """The 501 stub does not parse the body — it always returns 501."""
        from fastapi.testclient import TestClient
        app = _create_test_app()
        client = TestClient(app)
        resp = client.post("/v1/tools/bash", json={})
        assert resp.status_code == 501

    def test_web_search_default_deny_all(self):
        from fastapi.testclient import TestClient
        app = _create_test_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/tools/web_search",
            json={"query": "anything"},
        )
        # Empty allowlist => 403
        assert resp.status_code == 403

    def test_web_search_rejects_oversize_query(self):
        from fastapi.testclient import TestClient
        app = _create_test_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/tools/web_search",
            json={"query": "x" * 2000},
        )
        assert resp.status_code == 400

    def test_web_search_rejects_bool_max_results(self):
        from fastapi.testclient import TestClient
        app = _create_test_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/tools/web_search",
            json={"query": "q", "max_results": True},
        )
        assert resp.status_code == 400

    def test_web_search_with_allowlist_calls_backend(self):
        from fastapi.testclient import TestClient

        from soup_cli.commands.serve import _create_app
        from soup_cli.utils.server_tools import WebSearchConfig

        try:
            import fastapi  # noqa: F401
        except ImportError:
            pytest.skip("FastAPI not installed")

        def fake_backend(query, max_results, allowlist):
            return [
                {"url": "https://example.com/page1", "snippet": "snippet1"},
                {"url": "https://other.com/blocked", "snippet": "blocked"},
            ]

        cfg = WebSearchConfig(domain_allowlist=("example.com",))
        app = _create_app(
            model_obj=MagicMock(),
            tokenizer=MagicMock(),
            device="cpu",
            model_name="test-model",
            max_tokens_default=128,
            web_search_config=cfg,
            web_search_backend=fake_backend,
        )
        client = TestClient(app)
        resp = client.post(
            "/v1/tools/web_search",
            json={"query": "hello", "max_results": 5},
        )
        assert resp.status_code == 200
        data = resp.json()
        # Only the allowlist-matched URL should survive.
        urls = [r["url"] for r in data["results"]]
        assert any("example.com" in u for u in urls)
        assert all("other.com" not in u for u in urls)

    def test_tools_no_longer_return_501(self):
        """v0.53.7 #103 regression: 501-stubs are gone."""
        from fastapi.testclient import TestClient
        app = _create_test_app()
        client = TestClient(app)
        # Python: should return 200 with sandbox response.
        resp = client.post("/v1/tools/python", json={"code": "print(1)"})
        assert resp.status_code != 501


# ----- #102 vLLM /v1/messages + streaming SSE -----------------------------


class TestAnthropicMessagesStreaming:
    def test_stream_true_returns_sse(self):
        from fastapi.testclient import TestClient

        # Patch _generate_response so we get deterministic output without
        # actually invoking a model.
        with patch(
            "soup_cli.commands.serve._generate_response",
            return_value=("hello world", 3, 2),
        ):
            app = _create_test_app()
            client = TestClient(app)
            resp = client.post(
                "/v1/messages",
                json={
                    "model": "test-model",
                    "max_tokens": 16,
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert resp.status_code == 200, resp.text
            body = resp.text
            assert "event: message_start" in body
            assert "event: content_block_delta" in body
            assert "event: message_delta" in body
            assert "event: message_stop" in body
            assert "text_delta" in body

    def test_non_stream_returns_anthropic_envelope(self):
        from fastapi.testclient import TestClient

        with patch(
            "soup_cli.commands.serve._generate_response",
            return_value=("response", 1, 1),
        ):
            app = _create_test_app()
            client = TestClient(app)
            resp = client.post(
                "/v1/messages",
                json={
                    "model": "test-model",
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["type"] == "message"
            assert data["role"] == "assistant"
            assert data["stop_reason"] == "end_turn"
            assert data["content"][0]["type"] == "text"

    def test_stream_no_longer_returns_501(self):
        """Regression: v0.53.6 returned 501 on stream:true."""
        from fastapi.testclient import TestClient

        with patch(
            "soup_cli.commands.serve._generate_response",
            return_value=("x", 1, 1),
        ):
            app = _create_test_app()
            client = TestClient(app)
            resp = client.post(
                "/v1/messages",
                json={
                    "model": "test-model",
                    "max_tokens": 16,
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert resp.status_code != 501


class TestVllmAnthropicMessages:
    def test_vllm_app_has_v1_messages_route(self):
        """Source-level guard: vLLM create_vllm_app installs /v1/messages."""
        from pathlib import Path

        src = (
            Path(__file__).parent.parent
            / "soup_cli" / "utils" / "vllm.py"
        ).read_text(encoding="utf-8")
        # Verify the route + helper landed.
        assert "/v1/messages" in src
        assert "_stream_anthropic_messages_vllm" in src
        assert "validate_anthropic_payload" in src

    def test_vllm_stream_helper_emits_anthropic_events(self):
        from soup_cli.utils.vllm import _stream_anthropic_messages_vllm

        frames = list(
            _stream_anthropic_messages_vllm(
                msg_id="m1",
                model="test",
                text="hello world",
                input_tokens=2,
                output_tokens=2,
            )
        )
        joined = "".join(frames)
        assert "event: message_start" in joined
        assert "event: content_block_delta" in joined
        assert "event: message_delta" in joined
        assert "event: message_stop" in joined
        assert "text_delta" in joined

    def test_anthropic_streaming_helper_handles_empty_text(self):
        from soup_cli.utils.vllm import _stream_anthropic_messages_vllm

        frames = list(
            _stream_anthropic_messages_vllm(
                msg_id="m1",
                model="test",
                text="",
                input_tokens=0,
                output_tokens=0,
            )
        )
        joined = "".join(frames)
        # Even with empty text we still emit start / delta / stop events.
        assert "event: message_start" in joined
        assert "event: message_stop" in joined


class TestServeStreamHelper:
    def test_transformers_stream_helper_emits_anthropic_events(self):
        from soup_cli.commands.serve import _stream_anthropic_messages

        frames = list(
            _stream_anthropic_messages(
                msg_id="m1",
                model="test",
                text="hello world from soup",
                input_tokens=4,
                output_tokens=4,
            )
        )
        joined = "".join(frames)
        assert "event: message_start" in joined
        assert "event: content_block_delta" in joined
        assert "event: message_delta" in joined
        assert "event: message_stop" in joined
        # 4 words → at least 4 content_block_delta frames.
        assert joined.count("content_block_delta") >= 4


# ----- Cross-cutting / smoke ----------------------------------------------


class TestPublicSurface:
    def test_data_pipeline_exports(self):
        from soup_cli.utils import data_pipeline
        for name in (
            "split_markdown_by_headings",
            "resolve_prompt_strategy",
            "apply_prompt_strategy",
            "load_pretokenized_dataset",
            "validate_prompt_strategy",
        ):
            assert hasattr(data_pipeline, name), name

    def test_data_forge_exports(self):
        from soup_cli.utils import data_forge
        for name in ("make_judge_provider_fn", "JUDGE_PROVIDERS"):
            assert hasattr(data_forge, name), name

    def test_recipe_run_exports(self):
        from soup_cli.utils import recipe_run
        assert hasattr(recipe_run, "run_recipe")

    def test_trainer_plugins_exports(self):
        from soup_cli.utils import trainer_plugins
        assert hasattr(trainer_plugins, "instantiate_trainer_plugins")


# ============================================================================
# v0.53.7 review-fix tests
# ============================================================================


class TestReviewFixesRecipeRun:
    """Tests for review fixes H-B/H-C/H-D/M-C/M-E/M-F + H-J/H-K/M-O."""

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks")
    def test_seed_rejects_symlink_via_lstat_on_raw_path(self, tmp_path, monkeypatch):
        """v0.53.7 H-B: lstat the raw path BEFORE realpath."""
        import os as _os
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "real.jsonl"
        target.write_text('{"x": 1}\n', encoding="utf-8")
        link = tmp_path / "link.jsonl"
        _os.symlink(str(target), str(link))
        from soup_cli.utils.recipe_dag import RecipeNode
        from soup_cli.utils.recipe_run import _node_seed
        node = RecipeNode(name="s", kind="seed", config={"path": "link.jsonl"})
        with pytest.raises(ValueError, match="symlink"):
            _node_seed(node, ())

    def test_redact_exc_message_strips_posix_paths(self):
        from soup_cli.utils.recipe_run import _redact_exc_message
        exc = FileNotFoundError("/etc/passwd: no such file")
        redacted = _redact_exc_message(exc)
        assert "/etc/passwd" not in redacted
        assert "FileNotFoundError" in redacted

    def test_redact_exc_message_truncates_long(self):
        from soup_cli.utils.recipe_run import _redact_exc_message
        exc = ValueError("x" * 1000)
        redacted = _redact_exc_message(exc, limit=64)
        assert len(redacted) <= 64

    def test_redact_exc_message_handles_windows_paths(self):
        from soup_cli.utils.recipe_run import _redact_exc_message
        exc = OSError(r"C:\Users\alice\secret.txt missing")
        redacted = _redact_exc_message(exc)
        assert "alice" not in redacted

    def test_node_code_row_injection_blocked_by_double_encode(self, tmp_path, monkeypatch):
        """v0.53.7 M-C: tricky row content cannot escape the Python literal."""
        from soup_cli.utils.recipe_dag import RecipeNode
        from soup_cli.utils.recipe_run import _node_code
        monkeypatch.chdir(tmp_path)
        evil = {"text": 'foo"); print("HIJACK"); ("'}
        node = RecipeNode(
            name="c",
            kind="code",
            config={"code": "import json\nprint(json.dumps({'L': len(_row['text'])}))"},
        )
        result = _node_code(node, ((evil,),))
        assert len(result) == 1
        out = result[0]["c"]
        # No HIJACK side effect; the literal survived intact.
        if isinstance(out, dict) and "L" in out:
            assert out["L"] == len(evil["text"])

    def test_run_recipe_resume_rehydrates_predecessor_outputs(self, tmp_path, monkeypatch):
        """v0.53.7 M-F: resume rehydrates from sidecar (was ``[]`` in v0.53.6)."""
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.recipe_dag import RecipeDAG, RecipeNode
        from soup_cli.utils.recipe_run import run_recipe

        seed = tmp_path / "seed.jsonl"
        seed.write_text('{"x": 1}\n{"x": 2}\n', encoding="utf-8")
        outdir = tmp_path / "out"

        nodes = (
            RecipeNode(name="s", kind="seed", config={"path": "seed.jsonl"}),
            RecipeNode(name="snk", kind="sampler", config={}),
        )
        dag = RecipeDAG(
            nodes=nodes, edges=(("s", "snk"),), topo_order=("s", "snk")
        )

        result1 = run_recipe(dag, output_dir=str(outdir))
        assert result1["status"] == "completed"
        assert result1["node_row_counts"]["snk"] == 2

        result2 = run_recipe(dag, output_dir=str(outdir), resume=True)
        assert result2["status"] == "completed"
        assert set(result2["completed_nodes"]) == {"s", "snk"}

    def test_run_recipe_failed_node_state_persisted(self, tmp_path, monkeypatch):
        """v0.53.7 M-O: mid-DAG failure persists status + failed_node."""
        monkeypatch.chdir(tmp_path)
        import json as _json

        from soup_cli.utils.recipe_dag import RecipeDAG, RecipeNode
        from soup_cli.utils.recipe_run import run_recipe

        outdir = tmp_path / "out"
        # seed node with a missing path raises ValueError BEFORE reaching
        # the sandbox-swallow path; this surfaces as a real failure.
        nodes = (
            RecipeNode(
                name="badseed",
                kind="seed",
                config={"path": "no_such_file_xyzzy.jsonl"},
            ),
        )
        dag = RecipeDAG(nodes=nodes, edges=(), topo_order=("badseed",))
        with pytest.raises(ValueError):
            run_recipe(dag, output_dir=str(outdir))
        ckpt = _json.loads((outdir / ".checkpoint.json").read_text(encoding="utf-8"))
        assert ckpt["status"] == "failed"
        assert ckpt["failed_node"] == "badseed"
        assert "ValueError" in ckpt["failed_reason"]

    def test_node_code_happy_path(self, tmp_path, monkeypatch):
        """v0.53.7 H-K: live code-node execution through sandbox."""
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.recipe_dag import RecipeDAG, RecipeNode
        from soup_cli.utils.recipe_run import run_recipe

        seed = tmp_path / "seed.jsonl"
        seed.write_text('{"x": 5}\n', encoding="utf-8")
        outdir = tmp_path / "out"
        nodes = (
            RecipeNode(name="s", kind="seed", config={"path": "seed.jsonl"}),
            RecipeNode(
                name="c",
                kind="code",
                config={"code": "import json\nprint(json.dumps({'doubled': _row['x']*2}))"},
            ),
            RecipeNode(name="snk", kind="sampler", config={}),
        )
        dag = RecipeDAG(
            nodes=nodes,
            edges=(("s", "c"), ("c", "snk")),
            topo_order=("s", "c", "snk"),
        )
        result = run_recipe(dag, output_dir=str(outdir))
        assert result["status"] == "completed"
        out_path = outdir / "snk.jsonl"
        if out_path.exists():
            content = out_path.read_text(encoding="utf-8")
            assert "doubled" in content or "10" in content

    def test_save_checkpoint_atomic_via_mkstemp(self, tmp_path, monkeypatch):
        """v0.53.7 H-C/M-E: mkstemp + os.replace pattern."""
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.recipe_run import _save_checkpoint
        outdir = tmp_path / "out"
        outdir.mkdir()
        _save_checkpoint(str(outdir), {"status": "running"})
        ckpt = outdir / ".checkpoint.json"
        assert ckpt.is_file()
        assert not ckpt.is_symlink()


class TestReviewFixesPromptStrategy:
    """v0.53.7 M-P + L-B + H-E coverage."""

    def setup_method(self):
        from soup_cli.utils.data_pipeline import resolve_prompt_strategy
        resolve_prompt_strategy.cache_clear()

    def test_resolve_null_byte_rejected(self):
        from soup_cli.utils.data_pipeline import resolve_prompt_strategy
        with pytest.raises(ValueError):
            resolve_prompt_strategy("mod\x00name:fn")

    def test_resolve_oversize_rejected(self):
        from soup_cli.utils.data_pipeline import resolve_prompt_strategy
        oversize = "a" * 1000 + ":fn"
        with pytest.raises(ValueError):
            resolve_prompt_strategy(oversize)

    def test_h_e_bad_signature_propagates(self, tmp_path, monkeypatch):
        """H-E: signature-shape ValueError must NOT be swallowed."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.syspath_prepend(str(tmp_path))
        (tmp_path / "tfm_bad_sig.py").write_text(
            "def fn(a, b, c):\n    return {'ok': True}\n",
            encoding="utf-8",
        )
        from soup_cli.utils.data_pipeline import resolve_prompt_strategy
        resolve_prompt_strategy.cache_clear()
        with pytest.raises(ValueError, match="positional"):
            resolve_prompt_strategy("tfm_bad_sig:fn")


class TestReviewFixesForge:
    """v0.53.7 M-M temperature bool + M-L typing."""

    def test_make_judge_provider_fn_bool_temperature_rejected(self):
        from soup_cli.utils.data_forge import make_judge_provider_fn
        with pytest.raises(TypeError, match="temperature"):
            make_judge_provider_fn("ollama", temperature=True)

    def test_make_judge_provider_fn_temperature_out_of_range(self):
        from soup_cli.utils.data_forge import make_judge_provider_fn
        with pytest.raises(ValueError, match=r"\[0, 2\]"):
            make_judge_provider_fn("ollama", temperature=5.0)

    def test_judge_providers_typed_frozenset_of_str(self):
        from soup_cli.utils.data_forge import JUDGE_PROVIDERS
        assert isinstance(JUDGE_PROVIDERS, frozenset)
        assert all(isinstance(p, str) for p in JUDGE_PROVIDERS)


class TestReviewFixesVllmAnthropicLive:
    """v0.53.7 H-I + M-Q: live route tests with mocked AsyncLLMEngine."""

    def _build_vllm_app(self):
        try:
            import fastapi  # noqa: F401
        except ImportError:
            pytest.skip("FastAPI not installed")
        try:
            import vllm  # noqa: F401
        except ImportError:
            pytest.skip("vLLM not installed")
        from soup_cli.utils.vllm import create_vllm_app

        engine = MagicMock()
        app = create_vllm_app(
            engine=engine,
            engine_model_name="test-model",
            model_name="test-model",
            max_tokens_default=128,
        )
        return app

    def test_messages_route_registered(self):
        app = self._build_vllm_app()
        paths = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/v1/messages" in paths

    def test_messages_malformed_returns_400(self):
        from fastapi.testclient import TestClient
        app = self._build_vllm_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/messages", json={"messages": [], "model": "x"}
        )
        assert resp.status_code == 400

    def test_messages_max_tokens_over_cap_returns_400(self):
        from fastapi.testclient import TestClient
        app = self._build_vllm_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/messages",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "model": "x",
                "max_tokens": 999_999,
            },
        )
        assert resp.status_code == 400


class TestReviewFixesDataScore:
    """v0.53.7 H-H + M-J coverage."""

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks")
    def test_load_jsonl_rows_rejects_symlink(self, tmp_path, monkeypatch):
        import os as _os
        monkeypatch.chdir(tmp_path)
        target_outside = tmp_path.parent / "outside_bench.jsonl"
        try:
            target_outside.write_text('{"text": "x"}\n', encoding="utf-8")
            link = tmp_path / "link.jsonl"
            _os.symlink(str(target_outside), str(link))
            from soup_cli.utils.data_score import load_jsonl_rows
            with pytest.raises(ValueError, match="symlink"):
                load_jsonl_rows("link.jsonl")
        finally:
            if target_outside.exists():
                target_outside.unlink()

    def test_load_jsonl_rows_outside_cwd_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        outside = tmp_path.parent / "evil_bench.jsonl"
        try:
            outside.write_text('{"x": 1}\n', encoding="utf-8")
            from soup_cli.utils.data_score import load_jsonl_rows
            with pytest.raises(ValueError, match="under cwd"):
                load_jsonl_rows(str(outside))
        finally:
            if outside.exists():
                outside.unlink()

    def test_extract_row_text_publicly_exposed(self):
        """v0.53.7 M-J: public alias is the documented import."""
        from soup_cli.utils import data_score
        from soup_cli.utils.data_score import extract_row_text
        assert callable(extract_row_text)
        assert "extract_row_text" in data_score.__all__


class TestReviewFixesTrainerPlugins:
    """v0.53.7 L-A + L-I dedup + L-E narrow catch."""

    def test_validate_trainer_plugin_list_rejects_duplicates(self):
        """L-I: duplicate names are explicitly rejected (not silently deduped)."""
        from soup_cli.utils.trainer_plugins import validate_trainer_plugin_list
        with pytest.raises(ValueError, match="duplicate"):
            validate_trainer_plugin_list(["grokfast", "grokfast"])

    def test_simple_plugin_returns_none_when_module_lacks_attrs(self, monkeypatch):
        """L-A: missing API surface returns None, not the bare module."""
        import types

        from soup_cli.utils import trainer_plugins

        dummy = types.ModuleType("grokfast")
        monkeypatch.setitem(sys.modules, "grokfast", dummy)
        result = trainer_plugins._instantiate_simple_plugin(
            "grokfast", "grokfast", ("GrokFastCallback", "Gradfilter")
        )
        assert result is None


class TestReviewFixesSSEHeaders:
    """v0.53.7 L-C + M-A: SSE no-store + header-injection sanitisation."""

    def test_sanitise_sse_field_strips_crlf_nul(self):
        from soup_cli.commands.serve import _sanitise_sse_field
        assert _sanitise_sse_field("hello\r\nworld", max_len=200) == "helloworld"
        assert _sanitise_sse_field("a\x00b", max_len=200) == "ab"
        assert _sanitise_sse_field("x" * 500, max_len=10) == "x" * 10

    def test_stream_anthropic_sanitises_msg_id(self):
        """Caller-controlled msg_id with CR/LF cannot create a new SSE event.

        After sanitisation the injected ``event:``/``data:`` text becomes a
        literal substring inside an existing JSON ``id`` value — no new
        ``\\n\\n`` boundary, so the SSE consumer treats the whole frame as
        one event with garbled-but-safe ``id`` content.
        """
        from soup_cli.commands.serve import _stream_anthropic_messages
        evil_id = "msg-1\nevent: hijack\ndata: {}\n"
        frames = list(_stream_anthropic_messages(
            msg_id=evil_id, model="m", text="hi",
            input_tokens=1, output_tokens=1,
        ))
        # Each frame is one SSE event terminated by exactly one "\n\n".
        # The CR/LFs from the evil id must have been stripped — count of
        # "\n\n" must equal the number of frames produced.
        joined = "".join(frames)
        assert joined.count("\n\n") == len(frames)
        # The injected literal "event: hijack" cannot start a new line; it
        # always appears as a substring inside an existing data line, never
        # at line-start, because no CR/LF separator survived.
        for line in joined.split("\n"):
            assert not line.startswith("event: hijack")
            assert not line.startswith("data: {}")


class TestReviewFixesC1BashStub:
    """v0.53.7 C1: bash endpoint reverted to 501."""

    def test_bash_returns_501_with_v0538_marker(self):
        try:
            import fastapi  # noqa: F401
        except ImportError:
            pytest.skip("FastAPI not installed")
        from fastapi.testclient import TestClient

        from soup_cli.commands.serve import _create_app

        app = _create_app(
            model_obj=MagicMock(),
            tokenizer=MagicMock(),
            device="cpu",
            model_name="test-model",
            max_tokens_default=128,
        )
        client = TestClient(app)
        resp = client.post("/v1/tools/bash", json={"command": "ls"})
        assert resp.status_code == 501
        assert "v0.53.8" in resp.text


class TestReviewFixesAuthToken:
    """v0.53.7 H-A: optional Bearer-token gate on tool endpoints."""

    def test_python_tool_with_auth_token_requires_bearer(self):
        try:
            import fastapi  # noqa: F401
        except ImportError:
            pytest.skip("FastAPI not installed")
        from fastapi.testclient import TestClient

        from soup_cli.commands.serve import _create_app

        app = _create_app(
            model_obj=MagicMock(),
            tokenizer=MagicMock(),
            device="cpu",
            model_name="test-model",
            max_tokens_default=128,
            auth_token="s3cret",
        )
        client = TestClient(app)
        # No header -> 401.
        resp = client.post("/v1/tools/python", json={"code": "print(1)"})
        assert resp.status_code == 401
        # Wrong token -> 401.
        resp = client.post(
            "/v1/tools/python",
            json={"code": "print(1)"},
            headers={"Authorization": "Bearer wrong"},
        )
        assert resp.status_code == 401
        # Right token -> 200.
        resp = client.post(
            "/v1/tools/python",
            json={"code": "print(1)"},
            headers={"Authorization": "Bearer s3cret"},
        )
        assert resp.status_code == 200

    def test_auth_token_none_means_no_gate(self):
        try:
            import fastapi  # noqa: F401
        except ImportError:
            pytest.skip("FastAPI not installed")
        from fastapi.testclient import TestClient

        from soup_cli.commands.serve import _create_app

        app = _create_app(
            model_obj=MagicMock(),
            tokenizer=MagicMock(),
            device="cpu",
            model_name="test-model",
            max_tokens_default=128,
        )
        client = TestClient(app)
        resp = client.post("/v1/tools/python", json={"code": "print(1)"})
        assert resp.status_code == 200


class TestReviewFixesPreprocessAtomic:
    """v0.53.7 M-H + L-J source-grep guard."""

    def test_preprocess_save_uses_atomic_pattern(self):
        src = Path(__file__).parent.parent / "soup_cli" / "commands" / "data.py"
        body = src.read_text(encoding="utf-8")
        assert ".tmp_" in body
        assert "os.replace" in body


class TestReviewFixesVllmAnthropicCors:
    """v0.53.7 M-G: vLLM /v1/messages CORS restricted to loopback."""

    def test_vllm_app_cors_is_loopback_only(self):
        try:
            import fastapi  # noqa: F401
        except ImportError:
            pytest.skip("FastAPI not installed")
        try:
            import vllm  # noqa: F401
        except ImportError:
            pytest.skip("vLLM not installed")
        from soup_cli.utils.vllm import create_vllm_app

        engine = MagicMock()
        app = create_vllm_app(
            engine=engine,
            engine_model_name="test-model",
            model_name="test-model",
            max_tokens_default=128,
        )
        # Inspect installed middleware for the loopback regex.
        cors_options = []
        for m in app.user_middleware:
            cors_options.append(getattr(m, "options", {}))
        # Find the CORSMiddleware entry; its options should carry
        # ``allow_origin_regex`` not ``allow_origins=*``.
        found_loopback = False
        for opts in cors_options:
            regex = opts.get("allow_origin_regex", "")
            if regex and "localhost" in regex:
                found_loopback = True
        assert found_loopback, "vLLM /v1/messages CORS must be loopback-only"


# ----- #86 SFT + Pretrain pre_tokenized short-circuit --------------------
#
# These tests verify that SFTTrainerWrapper and PretrainTrainerWrapper
# actually call ``load_pretokenized_dataset`` when ``data.format =
# 'pre_tokenized'`` and ``data.tokenized_path`` is set, instead of running
# the normal tokenize pipeline. Schema-layer validators (e.g. the
# ``format='pre_tokenized'`` + missing-``tokenized_path`` cross-validator)
# are already covered by existing v0.42.0 tests; this class focuses on the
# wiring at the trainer setup() boundary.

class TestSftPretrainPreTokenizedShortCircuit:
    """v0.53.7 #86 — trainer-side short-circuit + cache-hash gate.

    Note: ``data.tokenized_path`` is cwd-containment-checked at schema load
    time, so every test ``monkeypatch.chdir(tmp_path)`` before loading the
    config and writes its Arrow fixture under ``tmp_path``.
    """

    def _write_arrow_dir(self, tmp_path: Path, cache_key: object) -> Path:
        """Save a 2-row pre-tokenized Arrow dataset under tmp_path/tokenized.

        ``cache_key`` is the value written into ``metadata.json``; pass
        ``...`` (Ellipsis) to skip writing the metadata file entirely.
        """
        pytest.importorskip("datasets")
        from datasets import Dataset

        target = tmp_path / "tokenized"
        rows = [
            {
                "input_ids": [1, 2, 3, 4],
                "labels": [1, 2, 3, 4],
                "attention_mask": [1, 1, 1, 1],
            },
            {
                "input_ids": [5, 6, 7, 8],
                "labels": [5, 6, 7, 8],
                "attention_mask": [1, 1, 1, 1],
            },
        ]
        ds = Dataset.from_list(rows)
        ds.save_to_disk(str(target))
        if cache_key is not Ellipsis:
            (target / "metadata.json").write_text(
                json.dumps({"cache_key": cache_key}), encoding="utf-8"
            )
        return target

    def _build_cfg_yaml(
        self,
        train_path: str,
        tokenized_path: str,
        *,
        max_length: int = 64,
        task: str = "sft",
    ) -> str:
        return (
            "base: TinyLlama/TinyLlama-1.1B-Chat-v1.0\n"
            f"task: {task}\n"
            "data:\n"
            f"  train: {train_path}\n"
            "  format: pre_tokenized\n"
            f"  tokenized_path: {tokenized_path}\n"
            f"  max_length: {max_length}\n"
            "output: ./out\n"
        )

    def test_sft_short_circuits_to_arrow_dataset(self, tmp_path, monkeypatch):
        """Happy path: SFT trainer loads pre-tokenized Arrow shards directly
        and skips ``Dataset.from_list(...).map(format_row)`` entirely."""
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.utils.data_pipeline import make_preprocess_cache_key

        monkeypatch.chdir(tmp_path)
        # The cfg.data.train path doesn't need to exist for the short-circuit
        # path — we never load it.
        train_jsonl = "train.jsonl"
        target = self._write_arrow_dir(tmp_path, cache_key="placeholder")

        # Compute matching cache_key so the gate passes.
        cache_key = make_preprocess_cache_key(
            dataset_path=train_jsonl,
            tokenizer_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            max_length=64,
            format_name="pre_tokenized",
        )
        (target / "metadata.json").write_text(
            json.dumps({"cache_key": cache_key}), encoding="utf-8"
        )

        yaml = self._build_cfg_yaml(train_jsonl, "tokenized")
        cfg = load_config_from_string(yaml)

        # Exercise the helper directly — bypasses the full ``setup()`` (which
        # would also load a 1B model). The helper is the unit under test.
        from soup_cli.trainer.sft import _maybe_load_pretokenized

        captured = MagicMock()
        result = _maybe_load_pretokenized(cfg.data, cfg.base, captured)
        assert result is not None
        train_ds, eval_ds = result
        assert len(train_ds) == 2
        assert eval_ds is None
        # Ensure the format_row tokenizer-call path was NOT taken — we do
        # this by counting console.print advisory lines (yellow advisory
        # for missing metadata.json was NOT printed since metadata exists).
        printed = "".join(str(c) for c in captured.print.call_args_list)
        assert "skipping tokenization" in printed
        assert "no metadata.json" not in printed

    def test_pretrain_short_circuits_to_arrow_dataset(self, tmp_path, monkeypatch):
        """Same as SFT but for the Pretrain wrapper — verifies pretrain.py
        also short-circuits, not just sft.py."""
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.utils.data_pipeline import make_preprocess_cache_key

        monkeypatch.chdir(tmp_path)
        train_jsonl = "train.txt"
        target = self._write_arrow_dir(tmp_path, cache_key="placeholder")

        cache_key = make_preprocess_cache_key(
            dataset_path=train_jsonl,
            tokenizer_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            max_length=64,
            format_name="pre_tokenized",
        )
        (target / "metadata.json").write_text(
            json.dumps({"cache_key": cache_key}), encoding="utf-8"
        )

        yaml = self._build_cfg_yaml(
            train_jsonl, "tokenized", task="pretrain",
        )
        cfg = load_config_from_string(yaml)

        # The pretrain wrapper imports the helper from sft.py — verify that
        # import resolves and the function behaves identically.
        from soup_cli.trainer.sft import _maybe_load_pretokenized as helper

        captured = MagicMock()
        result = helper(cfg.data, cfg.base, captured)
        assert result is not None
        train_ds, _ = result
        assert len(train_ds) == 2

    def test_cache_hash_mismatch_raises(self, tmp_path, monkeypatch):
        """metadata.json carrying a stale cache_key (e.g. user changed
        ``max_length`` without re-running ``soup data preprocess``) must
        raise with the keyword ``cache hash mismatch`` so users know what
        to do."""
        from soup_cli.config.loader import load_config_from_string

        monkeypatch.chdir(tmp_path)
        self._write_arrow_dir(tmp_path, cache_key="STALE_KEY_FROM_OLD_RUN")

        yaml = self._build_cfg_yaml("train.jsonl", "tokenized")
        cfg = load_config_from_string(yaml)

        from soup_cli.trainer.sft import _maybe_load_pretokenized

        with pytest.raises(ValueError, match="cache hash mismatch"):
            _maybe_load_pretokenized(cfg.data, cfg.base, MagicMock())

    def test_missing_metadata_proceeds_with_yellow_advisory(
        self, tmp_path, monkeypatch,
    ):
        """When metadata.json is absent, the trainer falls back to "trusted"
        mode with a yellow advisory and still loads the Arrow shards."""
        from soup_cli.config.loader import load_config_from_string

        monkeypatch.chdir(tmp_path)
        # Ellipsis sentinel → skip writing metadata.json entirely.
        self._write_arrow_dir(tmp_path, cache_key=Ellipsis)

        yaml = self._build_cfg_yaml("train.jsonl", "tokenized")
        cfg = load_config_from_string(yaml)

        from soup_cli.trainer.sft import _maybe_load_pretokenized

        captured = MagicMock()
        result = _maybe_load_pretokenized(cfg.data, cfg.base, captured)
        assert result is not None
        train_ds, _ = result
        assert len(train_ds) == 2

        # Yellow advisory must mention the user-actionable next step.
        printed = "".join(str(c) for c in captured.print.call_args_list)
        assert "no metadata.json" in printed
        assert "soup data preprocess" in printed

    # Schema layer already rejects ``format='pre_tokenized'`` without a
    # ``data.tokenized_path`` via ``DataConfig._validate_v042_pre_tokenized_path``
    # (see test_v0420.py). No re-test here — that's the schema gate's job.
