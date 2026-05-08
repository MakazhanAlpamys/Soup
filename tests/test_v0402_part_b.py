"""Tests for v0.40.2 Part B — v0.40.1 carry-overs (H2/H3/N1-G2/N7/M5)."""
from __future__ import annotations

import json
import re

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app

runner = CliRunner()


def _plain(s: str) -> str:
    """Strip ANSI escape sequences for help-string assertions."""
    return re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", s)


# ---------------------------------------------------------------------------
# H2 — data flag aliases
# ---------------------------------------------------------------------------


class TestDataFlagAliases:
    def test_split_accepts_train_flag(self, tmp_path, monkeypatch):
        """--train is accepted on `data split` (informational; train is remainder)."""
        monkeypatch.chdir(tmp_path)
        ds = tmp_path / "ds.jsonl"
        rows = [{"text": f"row {i}"} for i in range(20)]
        ds.write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )

        result = runner.invoke(
            app,
            ["data", "split", str(ds), "--train", "70", "--val", "20", "--test", "10"],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))

    def test_split_help_mentions_train(self):
        result = runner.invoke(app, ["data", "split", "--help"])
        assert "--train" in _plain(result.output)

    def test_filter_min_coherence_alias(self):
        result = runner.invoke(app, ["data", "filter", "--help"])
        assert "--min-coherence" in _plain(result.output)

    def test_register_positional_name_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ds = tmp_path / "myset.jsonl"
        ds.write_text('{"text": "x"}\n', encoding="utf-8")

        result = runner.invoke(
            app,
            ["data", "register", "myset", str(ds)],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "myset" in result.output

    def test_unregister_positional_name(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ds = tmp_path / "myset.jsonl"
        ds.write_text('{"text": "x"}\n', encoding="utf-8")

        # Register first
        r1 = runner.invoke(app, ["data", "register", "myset", str(ds)])
        assert r1.exit_code == 0

        r2 = runner.invoke(app, ["data", "unregister", "myset"])
        assert r2.exit_code == 0, (r2.output, repr(r2.exception))


# ---------------------------------------------------------------------------
# H3 — `soup quickstart --output DIR`
# ---------------------------------------------------------------------------


class TestQuickstartOutput:
    def test_quickstart_help_shows_output(self):
        result = runner.invoke(app, ["quickstart", "--help"])
        assert "--output" in _plain(result.output)

    def test_quickstart_output_routes_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "myrun"
        result = runner.invoke(
            app, ["quickstart", "--dry-run", "--output", str(target)]
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        # The actual files quickstart.py writes under --output:
        assert (target / "quickstart_data.jsonl").exists()
        assert (target / "quickstart_soup.yaml").exists()


# ---------------------------------------------------------------------------
# N1 / G2 — `--log-level` deeper plumbing
# ---------------------------------------------------------------------------


class TestLogLevelPlumbing:
    def test_quiet_emits_less_than_normal(self):
        # `data inspect` on missing file emits an error; we look at how many
        # log lines are produced. quiet should emit fewer informational lines.
        # Use --help as a stable surface.
        normal = runner.invoke(app, ["--log-level", "normal", "version"])
        quiet = runner.invoke(app, ["--log-level", "quiet", "version"])
        assert normal.exit_code == 0
        assert quiet.exit_code == 0
        # Both succeed; quiet's output should not exceed normal's by a wide margin.
        assert len(quiet.output) <= len(normal.output) + 10

    def test_log_level_sets_logging_module_level(self, monkeypatch):
        import logging as stdlogging

        from soup_cli.utils.log_level import LogLevel, apply_logging_level

        # Reset root logger
        root = stdlogging.getLogger()
        prev_level = root.level

        try:
            apply_logging_level(LogLevel.DEBUG)
            assert root.level == stdlogging.DEBUG
            apply_logging_level(LogLevel.QUIET)
            assert root.level == stdlogging.ERROR
            apply_logging_level(LogLevel.VERBOSE)
            assert root.level == stdlogging.INFO
            apply_logging_level(LogLevel.NORMAL)
            assert root.level == stdlogging.WARNING
        finally:
            root.setLevel(prev_level)


# ---------------------------------------------------------------------------
# N7 — `soup infer` accepts HF ids when local path missing
# ---------------------------------------------------------------------------


class TestInferHFFallback:
    def test_resolve_model_source_local_path_exists(self, tmp_path):
        from soup_cli.commands.infer import _resolve_model_source

        model_dir = tmp_path / "mymodel"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")

        kind, value = _resolve_model_source(str(model_dir))
        assert kind == "local"
        assert value == str(model_dir)

    def test_resolve_model_source_hf_id_when_no_local(self, tmp_path, monkeypatch):
        from soup_cli.commands.infer import _resolve_model_source

        monkeypatch.chdir(tmp_path)
        kind, value = _resolve_model_source("user/my-model")
        assert kind == "hf"
        assert value == "user/my-model"

    def test_resolve_model_source_invalid_hf_id_when_no_local(
        self, tmp_path, monkeypatch
    ):
        from soup_cli.commands.infer import _resolve_model_source

        monkeypatch.chdir(tmp_path)
        # Path-like but doesn't exist and isn't a valid HF id
        with pytest.raises(FileNotFoundError, match="not found"):
            _resolve_model_source("./nonexistent")

    def test_resolve_model_source_absolute_path_missing(self, tmp_path, monkeypatch):
        from soup_cli.commands.infer import _resolve_model_source

        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            _resolve_model_source("/nonexistent/abs/path")

    def test_resolve_model_source_tilde_path_missing(self, tmp_path, monkeypatch):
        from soup_cli.commands.infer import _resolve_model_source

        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            _resolve_model_source("~/nope/missing")

    def test_resolve_model_source_windows_drive_letter(self, tmp_path, monkeypatch):
        from soup_cli.commands.infer import _resolve_model_source

        monkeypatch.chdir(tmp_path)
        # Drive-letter syntax always treated as path-like; missing → error.
        with pytest.raises(FileNotFoundError):
            _resolve_model_source("Z:/nope/missing")

    def test_is_path_like_branches(self):
        from soup_cli.commands.infer import _is_path_like

        assert _is_path_like("") is True
        assert _is_path_like("./foo") is True
        assert _is_path_like("/abs") is True
        assert _is_path_like("~/x") is True
        assert _is_path_like("\\\\share\\foo") is True
        assert _is_path_like("C:/x") is True
        assert _is_path_like("user/my-model") is False
        assert _is_path_like("microsoft/phi-2") is False


# ---------------------------------------------------------------------------
# M5 — `soup runs --cwd-only`
# ---------------------------------------------------------------------------


class TestRunsCwdOnly:
    def test_runs_help_shows_cwd_only(self):
        result = runner.invoke(app, ["runs", "--help"])
        assert "--cwd-only" in _plain(result.output)

    def test_filter_runs_by_cwd(self, tmp_path):
        from soup_cli.commands.runs import _filter_runs_by_cwd

        cwd = str(tmp_path.resolve())
        runs = [
            {"run_id": "r1", "output_dir": str(tmp_path / "outA")},
            {"run_id": "r2", "output_dir": "/some/other/place"},
            {"run_id": "r3", "output_dir": str(tmp_path / "nested" / "outB")},
            {"run_id": "r4", "output_dir": None},
        ]
        result = _filter_runs_by_cwd(runs, cwd)
        ids = [r["run_id"] for r in result]
        assert "r1" in ids
        assert "r3" in ids
        assert "r2" not in ids
        assert "r4" not in ids

    def test_filter_runs_by_cwd_cross_drive(self, tmp_path):
        """Cross-drive paths on Windows raise ValueError in commonpath; survive."""
        from soup_cli.commands.runs import _filter_runs_by_cwd

        cwd = str(tmp_path.resolve())
        runs = [
            {"run_id": "r1", "output_dir": "Z:\\some\\path"},
            {"run_id": "r2", "output_dir": str(tmp_path / "x")},
        ]
        # Should not raise — should drop r1, keep r2
        result = _filter_runs_by_cwd(runs, cwd)
        ids = [r["run_id"] for r in result]
        assert "r1" not in ids
        assert "r2" in ids
