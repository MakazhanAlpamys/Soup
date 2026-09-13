"""Tests for Quant-Lobotomy Checker (v0.26.0 Part D)."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Verdict classifier
# ---------------------------------------------------------------------------


class TestVerdict:
    @pytest.mark.parametrize("delta, expected", [
        (0.0, "OK"),
        (-0.01, "OK"),
        (-0.019, "OK"),
        (-0.02, "MINOR"),
        (-0.04, "MINOR"),
        (-0.049, "MINOR"),
        (-0.05, "MAJOR"),
        (-0.10, "MAJOR"),
        (0.05, "OK"),  # Improvements are OK
    ])
    def test_classify(self, delta, expected):
        from soup_cli.eval.quant_check import classify_delta

        assert classify_delta(delta) == expected


# ---------------------------------------------------------------------------
# Run (orchestrator)
# ---------------------------------------------------------------------------


class TestRunQuantCheck:
    def _tasks_file(self, tmp_path):
        file = tmp_path / "tasks.jsonl"
        file.write_text(
            '{"prompt": "2+2=", "expected": "4", "scoring": "exact"}\n'
            '{"prompt": "3+3=", "expected": "6", "scoring": "exact"}\n',
            encoding="utf-8",
        )
        return file

    def test_run_happy_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.eval.quant_check import run_quant_check

        tasks_file = self._tasks_file(tmp_path)

        # Before model: perfect
        def before_gen(prompt):
            return {"2+2=": "4", "3+3=": "6"}.get(prompt, "")

        # After model: one regression
        def after_gen(prompt):
            return {"2+2=": "4"}.get(prompt, "")

        result = run_quant_check(
            before_gen=before_gen, after_gen=after_gen,
            tasks_file=str(tasks_file),
        )
        assert result.rows
        row = result.rows[0]
        assert row.before == 1.0
        assert row.after == 0.5
        assert row.delta == -0.5
        assert row.verdict == "MAJOR"

    def test_run_no_regression(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.eval.quant_check import run_quant_check

        tasks_file = self._tasks_file(tmp_path)

        def gen(prompt):
            return {"2+2=": "4", "3+3=": "6"}.get(prompt, "")

        result = run_quant_check(
            before_gen=gen, after_gen=gen,
            tasks_file=str(tasks_file),
        )
        row = result.rows[0]
        assert row.delta == 0.0
        assert row.verdict == "OK"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestQuantCheckCLI:
    def test_help_shows_command(self):
        result = runner.invoke(app, ["eval", "quant-check", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "before" in result.output.lower()
        assert "after" in result.output.lower()

    def test_rejects_missing_before(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        after = tmp_path / "a.safetensors"
        after.write_bytes(b"x")
        tasks = tmp_path / "t.jsonl"
        tasks.write_text('{"prompt": "x", "expected": "y"}\n',
                         encoding="utf-8")
        result = runner.invoke(app, [
            "eval", "quant-check",
            "--before", str(tmp_path / "nope.bin"),
            "--after", str(after),
            "--tasks", str(tasks),
        ])
        assert result.exit_code != 0, (result.output, repr(result.exception))

    def test_rejects_path_traversal_before(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        tasks = tmp_path / "t.jsonl"
        tasks.write_text('{"prompt": "x", "expected": "y"}\n',
                         encoding="utf-8")
        outside = tmp_path.parent / "outside_before.bin"
        outside.write_bytes(b"x")
        try:
            result = runner.invoke(app, [
                "eval", "quant-check",
                "--before", str(outside),
                "--after", str(tmp_path / "a.bin"),
                "--tasks", str(tasks),
            ])
            assert result.exit_code != 0, (result.output, repr(result.exception))
        finally:
            outside.unlink(missing_ok=True)

    def test_rejects_path_traversal_after(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        tasks = tmp_path / "t.jsonl"
        tasks.write_text('{"prompt": "x", "expected": "y"}\n',
                         encoding="utf-8")
        before = tmp_path / "b.bin"
        before.write_bytes(b"x")
        outside = tmp_path.parent / "outside_after.bin"
        outside.write_bytes(b"x")
        try:
            result = runner.invoke(app, [
                "eval", "quant-check",
                "--before", str(before),
                "--after", str(outside),
                "--tasks", str(tasks),
            ])
            assert result.exit_code != 0, (result.output, repr(result.exception))
        finally:
            outside.unlink(missing_ok=True)

    def test_invalid_format_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        tasks = tmp_path / "t.jsonl"
        tasks.write_text('{"prompt": "x"}\n', encoding="utf-8")
        before = tmp_path / "b.bin"
        before.write_bytes(b"x")
        after = tmp_path / "a.bin"
        after.write_bytes(b"x")
        result = runner.invoke(app, [
            "eval", "quant-check",
            "--before", str(before),
            "--after", str(after),
            "--tasks", str(tasks),
            "--format", "xml",
        ])
        assert result.exit_code != 0, (result.output, repr(result.exception))


# ---------------------------------------------------------------------------
# JSON output format
# ---------------------------------------------------------------------------


class TestResolveModelRef:
    def test_missing_entry_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "reg.db"))
        from soup_cli.eval.quant_check import resolve_model_ref

        assert resolve_model_ref("registry://nonexistent_entry") is None

    def test_kind_filter_skips_wrong_kind(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "reg.db"))
        monkeypatch.chdir(tmp_path)
        from soup_cli.eval.quant_check import resolve_model_ref
        from soup_cli.registry.store import RegistryStore

        store = RegistryStore()
        eid = store.push(name="m1", tag="v1", base_model="b", task="sft",
                         run_id=None, config={})
        art = tmp_path / "adapter.bin"
        art.write_bytes(b"x")
        store.add_artifact(entry_id=eid, kind="adapter", path=str(art))
        store.close()

        # Kind filter for 'gguf' should return None (only adapter exists)
        path = resolve_model_ref(f"registry://{eid}", kinds=("gguf",))
        assert path is None

        # Kind filter matching 'adapter' should find it
        path = resolve_model_ref(f"registry://{eid}", kinds=("adapter",))
        assert path is not None


class TestRenderFormats:
    def test_render_markdown(self):
        from soup_cli.eval.quant_check import (
            QuantCheckResult,
            QuantCheckRow,
            render_markdown,
        )

        result = QuantCheckResult(rows=[QuantCheckRow(
            task="t", before=1.0, after=0.9, delta=-0.1, verdict="MAJOR",
        )])
        out = render_markdown(result)
        assert "| Task |" in out
        assert "-0.100" in out
        assert "MAJOR" in out

    def test_render_table_returns_rich_table(self):
        from rich.table import Table

        from soup_cli.eval.quant_check import (
            QuantCheckResult,
            QuantCheckRow,
            render_table,
        )

        result = QuantCheckResult(rows=[QuantCheckRow(
            task="t", before=1.0, after=0.95, delta=-0.05,
            verdict="MAJOR",
        )])
        assert isinstance(render_table(result), Table)


class TestClassifyBoundaries:
    @pytest.mark.parametrize("delta, expected", [
        (-0.02, "MINOR"),
        (-0.05, "MAJOR"),
    ])
    def test_exact_boundaries(self, delta, expected):
        from soup_cli.eval.quant_check import classify_delta

        assert classify_delta(delta) == expected


class TestJSONOutput:
    def test_result_serialises_to_json(self, tmp_path):
        from soup_cli.eval.quant_check import QuantCheckResult, QuantCheckRow

        result = QuantCheckResult(rows=[
            QuantCheckRow(task="math", before=0.72, after=0.69,
                          delta=-0.03, verdict="MINOR"),
        ])
        data = json.loads(result.to_json())
        assert data["rows"][0]["task"] == "math"
        assert data["rows"][0]["verdict"] == "MINOR"
        assert "stub" not in data


class TestQuantCheckIssue809:
    """Acceptance tests for #809 (quant-check exit codes, stubs, and model loading)."""

    def _setup_env(self, tmp_path):
        tasks = tmp_path / "t.jsonl"
        tasks.write_text(
            '{"prompt": "2+2=", "expected": "4", "scoring": "exact"}\n',
            encoding="utf-8",
        )
        before_dir = tmp_path / "before_dir"
        before_dir.mkdir()
        after_dir = tmp_path / "after_dir"
        after_dir.mkdir()
        return tasks, before_dir, after_dir

    def test_load_failure_without_allow_stub_exits_nonzero_and_names_before_side(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.chdir(tmp_path)
        tasks, before_dir, after_dir = self._setup_env(tmp_path)

        result = runner.invoke(app, [
            "eval", "quant-check",
            "--before", str(before_dir),
            "--after", str(after_dir),
            "--tasks", str(tasks),
        ])
        assert result.exit_code != 0
        assert "Failed to load --before model" in result.output

    def test_load_failure_only_after_names_after_side(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.chdir(tmp_path)
        tasks, before_dir, after_dir = self._setup_env(tmp_path)

        def mock_make_gen(path):
            if "before" in path:
                return lambda p: "4"
            raise OSError("Could not load quantized weights")

        monkeypatch.setattr(
            "soup_cli.eval.quant_check.make_model_generator",
            mock_make_gen,
        )

        result = runner.invoke(app, [
            "eval", "quant-check",
            "--before", str(before_dir),
            "--after", str(after_dir),
            "--tasks", str(tasks),
        ])
        assert result.exit_code != 0
        assert "Failed to load --after model" in result.output
        assert "Failed to load --before model" not in result.output

    def test_major_verdict_asserts_exit_code_2(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.chdir(tmp_path)
        tasks, before_dir, after_dir = self._setup_env(tmp_path)

        def mock_make_gen(path):
            if "before" in path:
                return lambda p: "4"
            return lambda p: "wrong"

        monkeypatch.setattr(
            "soup_cli.eval.quant_check.make_model_generator",
            mock_make_gen,
        )

        result = runner.invoke(app, [
            "eval", "quant-check",
            "--before", str(before_dir),
            "--after", str(after_dir),
            "--tasks", str(tasks),
            "--format", "json",
        ])
        assert result.exit_code == 2
        data = json.loads(result.output)
        assert data["rows"][0]["verdict"] == "MAJOR"
        assert data["rows"][0]["delta"] == -1.0

    def test_ok_verdict_asserts_exit_code_0(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.chdir(tmp_path)
        tasks, before_dir, after_dir = self._setup_env(tmp_path)

        monkeypatch.setattr(
            "soup_cli.eval.quant_check.make_model_generator",
            lambda path: (lambda p: "4"),
        )

        result = runner.invoke(app, [
            "eval", "quant-check",
            "--before", str(before_dir),
            "--after", str(after_dir),
            "--tasks", str(tasks),
            "--format", "json",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["rows"][0]["verdict"] == "OK"

    def test_allow_stub_flag_marks_json_and_renders_banner(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.chdir(tmp_path)
        tasks, before_dir, after_dir = self._setup_env(tmp_path)

        # JSON output
        result_json = runner.invoke(app, [
            "eval", "quant-check",
            "--before", str(before_dir),
            "--after", str(after_dir),
            "--tasks", str(tasks),
            "--allow-stub",
            "--format", "json",
        ])
        assert result_json.exit_code == 0
        data = json.loads(result_json.output)
        assert data.get("stub") is True

        # Markdown output
        result_md = runner.invoke(app, [
            "eval", "quant-check",
            "--before", str(before_dir),
            "--after", str(after_dir),
            "--tasks", str(tasks),
            "--allow-stub",
            "--format", "markdown",
        ])
        assert result_md.exit_code == 0
        assert "Deterministic stub scoring" in result_md.output

        # Table output
        result_tbl = runner.invoke(app, [
            "eval", "quant-check",
            "--before", str(before_dir),
            "--after", str(after_dir),
            "--tasks", str(tasks),
            "--allow-stub",
            "--format", "table",
        ])
        assert result_tbl.exit_code == 0
        assert "deterministic stub" in result_tbl.output

    def test_allow_stub_retains_live_before_when_after_fails(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.chdir(tmp_path)
        tasks, before_dir, after_dir = self._setup_env(tmp_path)

        def mock_make_gen(path):
            if "before" in path:
                return lambda p: "4"
            raise OSError("After model corrupt")

        monkeypatch.setattr(
            "soup_cli.eval.quant_check.make_model_generator",
            mock_make_gen,
        )

        result = runner.invoke(app, [
            "eval", "quant-check",
            "--before", str(before_dir),
            "--after", str(after_dir),
            "--tasks", str(tasks),
            "--allow-stub",
            "--format", "json",
        ])
        # Live before got 1.0, stub after got 0.0 -> delta -1.0 (MAJOR) -> exit code 2
        assert result.exit_code == 2
        data = json.loads(result.output)
        assert data["stub"] is True
        assert data["rows"][0]["before"] == 1.0
        assert data["rows"][0]["after"] == 0.0

    def test_refuses_gguf_file_up_front(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.chdir(tmp_path)
        tasks, before_dir, _ = self._setup_env(tmp_path)
        gguf_file = tmp_path / "quantized.q4_k_m.gguf"
        gguf_file.write_bytes(b"mock gguf content")

        result = runner.invoke(app, [
            "eval", "quant-check",
            "--before", str(before_dir),
            "--after", str(gguf_file),
            "--tasks", str(tasks),
        ])
        assert result.exit_code != 0
        assert "standalone GGUF file" in result.output
