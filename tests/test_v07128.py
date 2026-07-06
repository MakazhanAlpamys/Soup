"""v0.71.28 — `soup mcp serve` MCP server.

Covers the pure tool registry (handlers + guards), the SDK server wiring
(via the in-memory transport), and the `soup mcp` Typer command.
"""

from __future__ import annotations

import json
import os

import pytest

from soup_cli.mcp_server import registry as reg

# ---------------------------------------------------------------------------
# _sanitize — recursive C0/ESC strip on handler output
# ---------------------------------------------------------------------------


class TestSanitize:
    def test_strips_c0_and_esc_keeps_tab_newline_cr(self):
        raw = "a\x1b[31mred\x1b[0m\tb\nc\rd\x07\x7f"
        cleaned = reg._sanitize(raw)
        assert "\x1b" not in cleaned
        assert "\x07" not in cleaned
        assert "\x7f" not in cleaned
        assert "\t" in cleaned and "\n" in cleaned and "\r" in cleaned
        assert "red" in cleaned

    def test_recurses_dict_and_list(self):
        obj = {"k": ["x\x1by", {"n": "z\x00w"}], "keep": 5, "b": True, "none": None}
        out = reg._sanitize(obj)
        assert out["k"][0] == "xy"
        assert out["k"][1]["n"] == "zw"
        assert out["keep"] == 5
        assert out["b"] is True
        assert out["none"] is None

    def test_leaves_non_str_scalars_untouched(self):
        assert reg._sanitize(3.14) == 3.14
        assert reg._sanitize(42) == 42
        assert reg._sanitize(False) is False


# ---------------------------------------------------------------------------
# _read_json_under_cwd — cwd-contained, symlink-rejected, size-capped loader
# ---------------------------------------------------------------------------


class TestReadJsonUnderCwd:
    def test_reads_dict_under_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        p = tmp_path / "ev.json"
        p.write_text(json.dumps({"a": 1}), encoding="utf-8")
        assert reg._read_json_under_cwd("ev.json", "evidence") == {"a": 1}

    def test_rejects_outside_cwd(self, tmp_path, monkeypatch):
        work = tmp_path / "work"
        work.mkdir()
        (tmp_path / "evil.json").write_text("{}", encoding="utf-8")
        monkeypatch.chdir(work)
        with pytest.raises(reg.McpToolError):
            reg._read_json_under_cwd("../evil.json", "evidence")

    def test_missing_file_raises_tool_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(reg.McpToolError):
            reg._read_json_under_cwd("nope.json", "evidence")

    def test_non_dict_json_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        p = tmp_path / "arr.json"
        p.write_text("[1,2,3]", encoding="utf-8")
        with pytest.raises(reg.McpToolError):
            reg._read_json_under_cwd("arr.json", "evidence")

    def test_oversize_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        p = tmp_path / "big.json"
        p.write_text("{}", encoding="utf-8")
        with pytest.raises(reg.McpToolError):
            reg._read_json_under_cwd("big.json", "evidence", max_bytes=1)

    def test_error_message_has_no_raw_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(reg.McpToolError) as exc:
            reg._read_json_under_cwd("secret-name.json", "evidence")
        assert "secret-name.json" not in str(exc.value)


# ---------------------------------------------------------------------------
# ToolSpec / McpToolError basics
# ---------------------------------------------------------------------------


class TestToolSpec:
    def test_toolspec_is_frozen(self):
        spec = reg.ToolSpec(
            name="x",
            title="X",
            description="does x",
            input_schema={"type": "object"},
            handler=lambda args: {},
            mutating=False,
        )
        with pytest.raises((AttributeError, TypeError)):
            spec.name = "y"  # type: ignore[misc]

    def test_tool_error_is_exception(self):
        assert issubclass(reg.McpToolError, Exception)


# ---------------------------------------------------------------------------
# Read-only handlers (Part A)
# ---------------------------------------------------------------------------


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


_ADVISE_ROWS = [
    {"instruction": "Summarize this article", "output": "A short summary."},
    {"instruction": "Translate to French", "output": "Bonjour le monde."},
    {"instruction": "Write a poem about spring", "output": "Petals fall softly."},
    {"instruction": "Explain gravity", "output": "Mass attracts mass."},
    {"instruction": "List three fruits", "output": "Apple, pear, plum."},
]


class TestAdviseHandler:
    def test_returns_verdict_dict(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        p = tmp_path / "d.jsonl"
        _write_jsonl(p, _ADVISE_ROWS)
        out = reg.tool_advise({"data": "d.jsonl", "goal": "improve summaries"})
        assert "choice" in out and "task_category" in out
        assert 0.0 <= out["confidence"] <= 1.0
        assert "estimated_roi" in out

    def test_bad_path_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(reg.McpToolError):
            reg.tool_advise({"data": "missing.jsonl"})

    def test_missing_data_arg_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(reg.McpToolError):
            reg.tool_advise({})


class TestDataInspectValidateHandlers:
    def test_inspect_returns_stats(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        p = tmp_path / "d.jsonl"
        _write_jsonl(p, _ADVISE_ROWS)
        out = reg.tool_data_inspect({"data": "d.jsonl"})
        assert out["total"] == 5
        assert "columns" in out

    def test_validate_returns_issues_and_valid_rows(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        p = tmp_path / "d.jsonl"
        _write_jsonl(p, _ADVISE_ROWS)
        out = reg.tool_data_validate({"data": "d.jsonl"})
        assert "issues" in out and "valid_rows" in out

    def test_inspect_bad_path_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(reg.McpToolError):
            reg.tool_data_inspect({"data": "nope.jsonl"})

    def test_oversize_data_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(reg, "_MAX_DATA_BYTES", 1)
        p = tmp_path / "d.jsonl"
        _write_jsonl(p, _ADVISE_ROWS)
        with pytest.raises(reg.McpToolError) as exc:
            reg.tool_data_inspect({"data": "d.jsonl"})
        assert "exceeds" in str(exc.value)


class TestArgHelperBounds:
    def test_require_str_rejects_overlong(self):
        with pytest.raises(reg.McpToolError):
            reg.tool_recipes_show({"name": "x" * (reg._MAX_STR_LEN + 1)})

    def test_opt_str_rejects_overlong(self):
        with pytest.raises(reg.McpToolError):
            reg.tool_recipes_search({"query": "x" * (reg._MAX_STR_LEN + 1)})

    def test_opt_str_rejects_non_string(self):
        with pytest.raises(reg.McpToolError):
            reg.tool_recipes_search({"query": 123})


class TestDataScoreHandler:
    def test_returns_scorecard(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        p = tmp_path / "d.jsonl"
        _write_jsonl(p, _ADVISE_ROWS)
        out = reg.tool_data_score({"data": "d.jsonl"})
        assert out["total"] == 5
        assert "pii_flagged" in out and "educational_mean" in out
        assert isinstance(out["languages"], dict)


class TestDataDoctorHandler:
    def test_returns_report_dict(self, tmp_path, monkeypatch):
        from tests.test_v07127 import _FakeTokenizer

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "soup_cli.utils.data_doctor.resolve_tokenizer",
            lambda model, **kw: _FakeTokenizer(),
        )
        p = tmp_path / "chat.jsonl"
        _write_jsonl(
            p,
            [
                {
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "hello"},
                    ]
                }
                for _ in range(4)
            ],
        )
        out = reg.tool_data_doctor(
            {"data": "chat.jsonl", "model": "fake/model", "format": "chatml"}
        )
        assert out["overall"] in ("OK", "MINOR", "MAJOR")
        assert "checks" in out and isinstance(out["checks"], list)

    def test_missing_transformers_friendly_error(self, tmp_path, monkeypatch):
        import sys

        monkeypatch.chdir(tmp_path)
        # Simulate the REAL absence path: `import transformers` fails. (The
        # handler probes the dependency directly rather than relying on
        # resolve_tokenizer's wrapped exception type.)
        monkeypatch.setitem(sys.modules, "transformers", None)
        p = tmp_path / "chat.jsonl"
        _write_jsonl(p, [{"messages": [{"role": "user", "content": "hi"}]}])
        with pytest.raises(reg.McpToolError) as exc:
            reg.tool_data_doctor({"data": "chat.jsonl", "model": "x", "format": "chatml"})
        assert "soup-cli[train]" in str(exc.value) or "install" in str(exc.value).lower()


class TestRecipesHandlers:
    def test_search_returns_results(self):
        from soup_cli.recipes.catalog import RECIPES

        out = reg.tool_recipes_search({"query": "qwen"})
        assert out["count"] >= 1
        assert all("name" in r and "model" in r for r in out["results"])
        # each name resolves to a real catalog entry (not the "?" id() fallback)
        assert all(r["name"] in RECIPES for r in out["results"])
        # search results stay compact — no full yaml body
        assert all("yaml_str" not in r for r in out["results"])

    def test_show_returns_full_recipe(self):
        # pick a known recipe from the search
        name = reg.tool_recipes_search({"query": "qwen"})["results"][0]["name"]
        out = reg.tool_recipes_show({"name": name})
        assert out["name"] == name
        assert "yaml_str" in out and out["yaml_str"]

    def test_show_unknown_raises(self):
        with pytest.raises(reg.McpToolError):
            reg.tool_recipes_show({"name": "definitely-not-a-recipe-xyz"})


class TestRunsHandlers:
    def test_list_returns_runs_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "exp.db"))
        out = reg.tool_runs_list({})
        assert "runs" in out and isinstance(out["runs"], list)
        assert "count" in out

    def test_show_unknown_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "exp.db"))
        with pytest.raises(reg.McpToolError):
            reg.tool_runs_show({"run_id": "nope"})

    def test_limit_out_of_range_rejected(self, tmp_path, monkeypatch):
        # _opt_int rejects (not clamps) out-of-range values, before DB access.
        monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "exp.db"))
        with pytest.raises(reg.McpToolError):
            reg.tool_runs_list({"limit": 999999})
        with pytest.raises(reg.McpToolError):
            reg.tool_runs_list({"limit": 0})
        with pytest.raises(reg.McpToolError):
            reg.tool_runs_list({"limit": "ten"})


class TestRegistryHandlers:
    def test_list_returns_entries_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "reg.db"))
        out = reg.tool_registry_list({})
        assert "entries" in out and isinstance(out["entries"], list)
        assert "count" in out

    def test_show_unknown_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "reg.db"))
        with pytest.raises(reg.McpToolError):
            reg.tool_registry_show({"ref": "nonexistent-id"})


# ---------------------------------------------------------------------------
# build_registry — the tool table
# ---------------------------------------------------------------------------

_EXPECTED_READONLY = {
    "advise",
    "data_inspect",
    "data_validate",
    "data_score",
    "data_doctor",
    "recipes_search",
    "recipes_show",
    "runs_list",
    "runs_show",
    "registry_list",
    "registry_show",
    "profile",
    "diagnose_evidence",
    "ship_evidence",
}


class TestBuildRegistry:
    def test_readonly_tools_present(self):
        names = {s.name for s in reg.build_registry(allow_mutating=False)}
        assert _EXPECTED_READONLY <= names

    def test_names_unique(self):
        specs = reg.build_registry(allow_mutating=True)
        names = [s.name for s in specs]
        assert len(names) == len(set(names))

    def test_every_schema_is_valid_json_schema(self):
        import jsonschema

        for spec in reg.build_registry(allow_mutating=True):
            jsonschema.Draft202012Validator.check_schema(spec.input_schema)
            assert spec.input_schema.get("type") == "object"

    def test_every_spec_well_formed(self):
        for spec in reg.build_registry(allow_mutating=True):
            assert spec.name and spec.description
            assert callable(spec.handler)
            assert isinstance(spec.mutating, bool)


# ---------------------------------------------------------------------------
# Flagged read-only handlers (Part B): profile / diagnose / ship evidence
# ---------------------------------------------------------------------------

_MIN_CONFIG = "base: Qwen/Qwen2.5-0.5B\ntask: sft\ndata:\n  train: data.jsonl\n"


class TestProfileHandler:
    def test_returns_estimate(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "soup.yaml").write_text(_MIN_CONFIG, encoding="utf-8")
        out = reg.tool_profile({"config": "soup.yaml"})
        assert "total_memory_gb" in out
        assert "recommended_batch_size" in out
        assert "compatible_gpus" in out

    def test_unknown_gpu_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "soup.yaml").write_text(_MIN_CONFIG, encoding="utf-8")
        with pytest.raises(reg.McpToolError):
            reg.tool_profile({"config": "soup.yaml", "gpu": "nonesuch-gpu"})

    def test_missing_config_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(reg.McpToolError):
            reg.tool_profile({"config": "missing.yaml"})


class TestDiagnoseEvidenceHandler:
    def test_returns_report(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ev = {"scores": {"forgetting": {"score": 0.95}, "refusal": {"score": 0.99}}}
        (tmp_path / "ev.json").write_text(json.dumps(ev), encoding="utf-8")
        out = reg.tool_diagnose_evidence({"run_id": "r1", "evidence": "ev.json"})
        assert out["run_id"] == "r1"
        assert out["overall"] in ("OK", "MINOR", "MAJOR")
        assert "scores" in out and "forgetting" in out["scores"]

    def test_non_numeric_score_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ev = {"scores": {"forgetting": {"score": "high"}}}
        (tmp_path / "ev.json").write_text(json.dumps(ev), encoding="utf-8")
        with pytest.raises(reg.McpToolError):
            reg.tool_diagnose_evidence({"run_id": "r1", "evidence": "ev.json"})

    def test_out_of_range_score_raises_specific(self, tmp_path, monkeypatch):
        # score outside [0,1] -> classify_score raises ValueError; the handler
        # must surface a specific McpToolError, not a generic internal error.
        monkeypatch.chdir(tmp_path)
        ev = {"scores": {"forgetting": {"score": 1.5}}}
        (tmp_path / "ev.json").write_text(json.dumps(ev), encoding="utf-8")
        with pytest.raises(reg.McpToolError) as exc:
            reg.tool_diagnose_evidence({"run_id": "r1", "evidence": "ev.json"})
        assert "forgetting" in str(exc.value)

    def test_missing_evidence_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(reg.McpToolError):
            reg.tool_diagnose_evidence({"run_id": "r1", "evidence": "nope.json"})


class TestShipEvidenceHandler:
    def test_ship_verdict(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ev = {
            "task": {"mode": "metric", "base": 0.5, "tuned": 0.8},
            "benchmarks": {"mmlu": {"base": 0.70, "tuned": 0.72}},
        }
        (tmp_path / "ev.json").write_text(json.dumps(ev), encoding="utf-8")
        out = reg.tool_ship_evidence({"evidence": "ev.json"})
        assert out["decision"] == "SHIP"
        assert "task_win" in out and "benchmark_deltas" in out

    def test_dont_ship_on_regression(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ev = {
            "task": {"mode": "metric", "base": 0.5, "tuned": 0.8},
            "benchmarks": {"mmlu": {"base": 0.70, "tuned": 0.50}},
        }
        (tmp_path / "ev.json").write_text(json.dumps(ev), encoding="utf-8")
        out = reg.tool_ship_evidence({"evidence": "ev.json"})
        assert out["decision"] == "DON'T SHIP"

    def test_bad_mode_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # "pairwise" became a supported mode in v0.71.31; use a truly invalid one.
        ev = {"task": {"mode": "not_a_real_mode", "base": 0.5, "tuned": 0.8}, "benchmarks": {}}
        (tmp_path / "ev.json").write_text(json.dumps(ev), encoding="utf-8")
        with pytest.raises(reg.McpToolError):
            reg.tool_ship_evidence({"evidence": "ev.json"})

    def test_forgetting_threshold_arg(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # 8-point drop: regresses at default 0.05, OK at 0.10
        ev = {
            "task": {"mode": "metric", "base": 0.5, "tuned": 0.8},
            "benchmarks": {"mmlu": {"base": 0.70, "tuned": 0.62}},
        }
        (tmp_path / "ev.json").write_text(json.dumps(ev), encoding="utf-8")
        strict = reg.tool_ship_evidence({"evidence": "ev.json"})
        loose = reg.tool_ship_evidence({"evidence": "ev.json", "forgetting_threshold": 0.10})
        assert strict["decision"] == "DON'T SHIP"
        assert loose["decision"] == "SHIP"


# ---------------------------------------------------------------------------
# Mutating tools (Part C): plan-only train_start / export + --allow-mutating gate
# ---------------------------------------------------------------------------


def _spec(name, *, allow_mutating):
    return {s.name: s for s in reg.build_registry(allow_mutating=allow_mutating)}[name]


class TestMutatingTools:
    def test_present_and_marked_in_both_registries(self):
        for allow in (False, True):
            specs = {s.name: s for s in reg.build_registry(allow_mutating=allow)}
            assert "train_start" in specs and "export" in specs
            assert specs["train_start"].mutating is True
            assert specs["export"].mutating is True

    def test_train_start_refused_without_allow(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "soup.yaml").write_text(_MIN_CONFIG, encoding="utf-8")
        with pytest.raises(reg.McpToolError) as exc:
            _spec("train_start", allow_mutating=False).handler({"config": "soup.yaml"})
        assert "allow-mutating" in str(exc.value)

    def test_export_refused_without_allow(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(reg.McpToolError) as exc:
            _spec("export", allow_mutating=False).handler({"model": "m", "format": "gguf"})
        assert "allow-mutating" in str(exc.value)

    def test_train_start_plan_only_with_allow(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "soup.yaml").write_text(_MIN_CONFIG, encoding="utf-8")
        out = _spec("train_start", allow_mutating=True).handler({"config": "soup.yaml"})
        assert out["config_valid"] is True
        assert out["would_run"].startswith("soup train")
        assert "plan-only" in out["note"]

    def test_train_start_invalid_config_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "bad.yaml").write_text("task: sft\n", encoding="utf-8")  # missing base
        with pytest.raises(reg.McpToolError):
            _spec("train_start", allow_mutating=True).handler({"config": "bad.yaml"})

    def test_export_plan_only_with_allow(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        out = _spec("export", allow_mutating=True).handler(
            {"model": "out/adapter", "format": "gguf"}
        )
        assert out["would_run"].startswith("soup export")
        assert out["format"] == "gguf"
        assert "plan-only" in out["note"]

    def test_export_bad_format_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(reg.McpToolError):
            _spec("export", allow_mutating=True).handler(
                {"model": "out/adapter", "format": "nonsense-format"}
            )

    def test_registry_count_is_16_with_mutating(self):
        assert len(reg.build_registry(allow_mutating=True)) == 16
        assert len(reg.build_registry(allow_mutating=False)) == 16


# ---------------------------------------------------------------------------
# Server wiring (Part D) — via the SDK's in-memory transport
# ---------------------------------------------------------------------------


def _run(coro):
    import asyncio

    return asyncio.run(coro)


def _roundtrip(server, tool_name, args):
    from mcp.shared.memory import create_connected_server_and_client_session

    async def _go():
        async with create_connected_server_and_client_session(server) as client:
            await client.initialize()
            return await client.call_tool(tool_name, args)

    return _run(_go())


class TestServerRoundTrip:
    @pytest.fixture(autouse=True)
    def _need_mcp(self):
        # The SDK + its in-memory transport are only present with the [mcp]
        # extra. Installing `mcp` forces anyio>=4.5 at resolve time, so this one
        # guard covers both. Skips cleanly on a partial install.
        pytest.importorskip("mcp")

    def test_list_tools_returns_all_16(self):
        from mcp.shared.memory import create_connected_server_and_client_session

        from soup_cli.mcp_server.registry import build_registry
        from soup_cli.mcp_server.server import build_server

        server = build_server(build_registry(allow_mutating=True))

        async def _go():
            async with create_connected_server_and_client_session(server) as client:
                await client.initialize()
                return await client.list_tools()

        result = _run(_go())
        names = {t.name for t in result.tools}
        assert len(names) == 16
        assert "recipes_search" in names and "train_start" in names
        # every advertised tool carries an inputSchema object
        assert all(t.inputSchema.get("type") == "object" for t in result.tools)

    def test_call_recipes_search_returns_json(self):
        from soup_cli.mcp_server.registry import build_registry
        from soup_cli.mcp_server.server import build_server

        server = build_server(build_registry(allow_mutating=False))
        res = _roundtrip(server, "recipes_search", {"query": "qwen"})
        assert res.isError is False
        payload = json.loads(res.content[0].text)
        assert payload["count"] >= 1

    def test_unknown_tool_is_error(self):
        from soup_cli.mcp_server.registry import build_registry
        from soup_cli.mcp_server.server import build_server

        server = build_server(build_registry(allow_mutating=False))
        res = _roundtrip(server, "no_such_tool", {})
        assert res.isError is True

    def test_mutating_refused_without_allow(self, tmp_path, monkeypatch):
        from soup_cli.mcp_server.registry import build_registry
        from soup_cli.mcp_server.server import build_server

        monkeypatch.chdir(tmp_path)
        (tmp_path / "soup.yaml").write_text(_MIN_CONFIG, encoding="utf-8")
        server = build_server(build_registry(allow_mutating=False))
        res = _roundtrip(server, "train_start", {"config": "soup.yaml"})
        assert res.isError is True

    def test_bad_arg_is_error_not_crash(self, tmp_path, monkeypatch):
        from soup_cli.mcp_server.registry import build_registry
        from soup_cli.mcp_server.server import build_server

        monkeypatch.chdir(tmp_path)
        server = build_server(build_registry(allow_mutating=False))
        res = _roundtrip(server, "data_inspect", {"data": "does-not-exist.jsonl"})
        assert res.isError is True

    def test_output_is_sanitized(self):
        from soup_cli.mcp_server.registry import ToolSpec
        from soup_cli.mcp_server.server import build_server

        spec = ToolSpec(
            name="echo",
            title="Echo",
            description="echo",
            input_schema={"type": "object", "properties": {}},
            handler=lambda a: {"v": "a\x1bb\x07c"},
            mutating=False,
        )
        server = build_server([spec])
        res = _roundtrip(server, "echo", {})
        payload = json.loads(res.content[0].text)
        assert payload["v"] == "abc"  # control bytes stripped by _sanitize

    def test_error_message_is_sanitized(self):
        from soup_cli.mcp_server.registry import McpToolError, ToolSpec
        from soup_cli.mcp_server.server import build_server

        def _boom(args):
            raise McpToolError("bad\x1b[31mvalue\x07")

        spec = ToolSpec(
            name="boom",
            title="Boom",
            description="boom",
            input_schema={"type": "object", "properties": {}},
            handler=_boom,
            mutating=False,
        )
        server = build_server([spec])
        res = _roundtrip(server, "boom", {})
        assert res.isError is True
        text = res.content[0].text
        assert "\x1b" not in text and "\x07" not in text

    def test_handler_stdout_is_redirected_off_the_jsonrpc_channel(self, capsys):
        from soup_cli.mcp_server.registry import ToolSpec
        from soup_cli.mcp_server.server import build_server

        def _noisy(args):
            print("LEAK-TO-STDOUT")
            return {"ok": True}

        spec = ToolSpec(
            name="noisy",
            title="Noisy",
            description="noisy",
            input_schema={"type": "object", "properties": {}},
            handler=_noisy,
            mutating=False,
        )
        server = build_server([spec])
        res = _roundtrip(server, "noisy", {})
        assert res.isError is False
        # the handler's stdout print must NOT reach the process stdout channel
        assert "LEAK-TO-STDOUT" not in capsys.readouterr().out

    def test_non_serializable_result_is_error_not_crash(self):
        from soup_cli.mcp_server.registry import ToolSpec
        from soup_cli.mcp_server.server import build_server

        spec = ToolSpec(
            name="bad",
            title="Bad",
            description="bad",
            input_schema={"type": "object", "properties": {}},
            handler=lambda args: {"bad": {1, 2, 3}},  # a set is not JSON-serializable
            mutating=False,
        )
        server = build_server([spec])
        res = _roundtrip(server, "bad", {})
        assert res.isError is True
        assert "internal error" in res.content[0].text


# ---------------------------------------------------------------------------
# CLI wiring (Part D)
# ---------------------------------------------------------------------------


def _strip_ansi(text):
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", text)


class TestMcpCli:
    def test_registered_in_main_app(self):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        r = CliRunner().invoke(app, ["mcp", "--help"], env={"COLUMNS": "200"})
        assert r.exit_code == 0, (r.output, repr(r.exception))
        assert "serve" in _strip_ansi(r.output)

    def test_serve_help(self):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        r = CliRunner().invoke(app, ["mcp", "serve", "--help"], env={"COLUMNS": "200"})
        assert r.exit_code == 0, (r.output, repr(r.exception))
        assert "mutating" in _strip_ansi(r.output).lower()

    def test_missing_sdk_exits_friendly(self, monkeypatch):
        import sys

        from typer.testing import CliRunner

        from soup_cli.cli import app

        # Simulate the `mcp` SDK being absent: importing the server module fails.
        monkeypatch.setitem(sys.modules, "soup_cli.mcp_server.server", None)
        r = CliRunner().invoke(app, ["mcp", "serve"])
        assert r.exit_code == 1
        # the hint must name the exact extra (Rich must not eat the '[mcp]')
        assert "soup-cli[mcp]" in r.output


class TestRegistryNoSdkImport:
    def test_registry_source_has_no_mcp_import(self):
        import inspect

        import soup_cli.mcp_server.registry as registry_mod

        src = inspect.getsource(registry_mod)
        assert "import mcp" not in src
        assert "from mcp" not in src


# ---------------------------------------------------------------------------
# Additional coverage (tdd-review gaps)
# ---------------------------------------------------------------------------


class TestReadGuardsExtra:
    def test_malformed_json_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "bad.json").write_text("{not valid json", encoding="utf-8")
        with pytest.raises(reg.McpToolError):
            reg._read_json_under_cwd("bad.json", "evidence")

    @pytest.mark.skipif(os.name == "nt", reason="POSIX symlink semantics")
    def test_read_json_rejects_symlink(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "real.json").write_text('{"a": 1}', encoding="utf-8")
        os.symlink(tmp_path / "real.json", tmp_path / "link.json")
        with pytest.raises(reg.McpToolError):
            reg._read_json_under_cwd("link.json", "evidence")

    def test_sanitize_handles_tuple(self):
        out = reg._sanitize(("a\x1bb", "c"))
        assert out == ["ab", "c"]


class TestOptIntBoolGuard:
    def test_bool_limit_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "exp.db"))
        with pytest.raises(reg.McpToolError):
            reg.tool_runs_list({"limit": True})


class TestRunsRegistryHappyPaths:
    def test_runs_show_happy(self, tmp_path, monkeypatch):
        from soup_cli.experiment.tracker import ExperimentTracker

        monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "exp.db"))
        run_id = ExperimentTracker().start_run(
            {"base": "m", "task": "sft"}, "cpu", "CPU", {}
        )
        out = reg.tool_runs_show({"run_id": run_id})
        assert out["run_id"] == run_id

    def test_registry_show_happy(self, tmp_path, monkeypatch):
        from soup_cli.registry.store import RegistryStore

        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "reg.db"))
        with RegistryStore() as store:
            entry_id = store.push(
                name="mymodel", tag="v1", base_model="b", task="sft",
                run_id=None, config={"base": "b"},
            )
        out = reg.tool_registry_show({"ref": entry_id})
        assert out["id"] == entry_id

    def test_registry_list_filter_narrows(self, tmp_path, monkeypatch):
        from soup_cli.registry.store import RegistryStore

        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "reg.db"))
        with RegistryStore() as store:
            store.push(name="alpha", tag="v1", base_model="b", task="sft",
                       run_id=None, config={"base": "b"})
            store.push(name="beta", tag="v1", base_model="b", task="dpo",
                       run_id=None, config={"base": "b"})
        assert reg.tool_registry_list({})["count"] == 2
        narrowed = reg.tool_registry_list({"name": "alpha"})
        assert narrowed["count"] == 1
        assert narrowed["entries"][0]["name"] == "alpha"

    def test_registry_show_ambiguous_ref(self, tmp_path, monkeypatch):
        from soup_cli.registry.store import AmbiguousRefError, RegistryStore

        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "reg.db"))

        def _raise(self, ref):
            raise AmbiguousRefError("matches many")

        monkeypatch.setattr(RegistryStore, "resolve", _raise)
        with pytest.raises(reg.McpToolError) as exc:
            reg.tool_registry_show({"ref": "ab"})
        assert "ambiguous" in str(exc.value).lower()

    def test_show_missing_required_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "exp.db"))
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "reg.db"))
        with pytest.raises(reg.McpToolError):
            reg.tool_runs_show({})
        with pytest.raises(reg.McpToolError):
            reg.tool_registry_show({})


class TestLoadDataTranslation:
    def test_loader_csv_error_translated(self, tmp_path, monkeypatch):
        import csv

        monkeypatch.chdir(tmp_path)
        (tmp_path / "d.jsonl").write_text('{"a": 1}', encoding="utf-8")

        def _raise(path):
            raise csv.Error("bad csv")

        monkeypatch.setattr("soup_cli.data.loader.load_raw_data", _raise)
        with pytest.raises(reg.McpToolError) as exc:
            reg.tool_data_inspect({"data": "d.jsonl"})
        assert "cannot load data" in str(exc.value)


class TestDataDoctorBranches:
    def test_auto_detect_failure(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_jsonl(tmp_path / "d.jsonl", [{"random_field": 1}, {"random_field": 2}])
        with pytest.raises(reg.McpToolError) as exc:
            reg.tool_data_doctor({"data": "d.jsonl", "model": "x"})
        assert "auto-detect" in str(exc.value)


class TestDiagnoseEvidenceBranches:
    def test_scores_not_object(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "ev.json").write_text(json.dumps({"scores": [1, 2]}), encoding="utf-8")
        with pytest.raises(reg.McpToolError):
            reg.tool_diagnose_evidence({"run_id": "r", "evidence": "ev.json"})

    def test_score_entry_not_object(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "ev.json").write_text(
            json.dumps({"scores": {"forgetting": "high"}}), encoding="utf-8"
        )
        with pytest.raises(reg.McpToolError):
            reg.tool_diagnose_evidence({"run_id": "r", "evidence": "ev.json"})


class TestShipEvidenceMalformed:
    @pytest.mark.parametrize(
        "payload",
        [
            {"benchmarks": {}},  # task missing
            {"task": [1, 2], "benchmarks": {}},  # task not object
            {"task": {"mode": "metric", "base": 0.5}, "benchmarks": {}},  # tuned missing
            # non-numeric base
            {"task": {"mode": "metric", "base": "x", "tuned": 0.8}, "benchmarks": {}},
            # benchmarks not an object
            {"task": {"mode": "metric", "base": 0.5, "tuned": 0.8}, "benchmarks": [1]},
            {
                "task": {"mode": "metric", "base": 0.5, "tuned": 0.8},
                "benchmarks": {"m": {"base": 0.7}},  # entry missing tuned
            },
        ],
    )
    def test_malformed_ship_evidence_rejected(self, tmp_path, monkeypatch, payload):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "ev.json").write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(reg.McpToolError):
            reg.tool_ship_evidence({"evidence": "ev.json"})

    def test_bad_threshold_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ev = {"task": {"mode": "metric", "base": 0.5, "tuned": 0.8}, "benchmarks": {}}
        (tmp_path / "ev.json").write_text(json.dumps(ev), encoding="utf-8")
        with pytest.raises(reg.McpToolError):
            reg.tool_ship_evidence({"evidence": "ev.json", "forgetting_threshold": "high"})
        with pytest.raises(reg.McpToolError):
            reg.tool_ship_evidence({"evidence": "ev.json", "forgetting_threshold": 5.0})


class TestExportOutputArg:
    def test_export_includes_output(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        out = _spec("export", allow_mutating=True).handler(
            {"model": "out/adapter", "format": "gguf", "output": "dist/model.gguf"}
        )
        assert "--output" in out["would_run"]


class TestServePlumbing:
    def test_allow_mutating_flag_reaches_runner(self, monkeypatch):
        pytest.importorskip("mcp")
        from typer.testing import CliRunner

        import soup_cli.mcp_server.server as srv
        from soup_cli.cli import app

        calls = []
        monkeypatch.setattr(
            srv, "run_stdio_server", lambda *, allow_mutating: calls.append(allow_mutating)
        )
        r1 = CliRunner().invoke(app, ["mcp", "serve"])
        r2 = CliRunner().invoke(app, ["mcp", "serve", "--allow-mutating"])
        assert r1.exit_code == 0, (r1.output, repr(r1.exception))
        assert r2.exit_code == 0, (r2.output, repr(r2.exception))
        assert calls == [False, True]
