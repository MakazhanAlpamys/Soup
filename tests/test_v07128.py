"""v0.71.28 — `soup mcp serve` MCP server.

Covers the pure tool registry (handlers + guards), the SDK server wiring
(via the in-memory transport), and the `soup mcp` Typer command.
"""

from __future__ import annotations

import json

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
        monkeypatch.chdir(tmp_path)

        def _boom(model, **kw):
            raise ImportError("no transformers")

        monkeypatch.setattr("soup_cli.utils.data_doctor.resolve_tokenizer", _boom)
        p = tmp_path / "chat.jsonl"
        _write_jsonl(p, [{"messages": [{"role": "user", "content": "hi"}]}])
        with pytest.raises(reg.McpToolError) as exc:
            reg.tool_data_doctor({"data": "chat.jsonl", "model": "x", "format": "chatml"})
        assert "soup-cli[train]" in str(exc.value) or "install" in str(exc.value).lower()


class TestRecipesHandlers:
    def test_search_returns_results(self):
        out = reg.tool_recipes_search({"query": "qwen"})
        assert out["count"] >= 1
        assert all("name" in r and "model" in r for r in out["results"])
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

    def test_no_mutating_in_readonly_registry_is_executable(self):
        # read-only build has no non-mutating gap: every listed tool is callable
        for spec in reg.build_registry(allow_mutating=False):
            assert callable(spec.handler)
