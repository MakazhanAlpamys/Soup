"""Tests for v0.34.0 Part G — soup tui."""

from __future__ import annotations

import sys

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app

# tui_app imports textual at module-load time. Skip the build-row tests when
# textual isn't available; the CLI surface tests still run.
textual_available = True
try:
    import textual  # noqa: F401
except ImportError:
    textual_available = False


class TestCli:
    def test_tui_help(self):
        runner = CliRunner()
        result = runner.invoke(app, ["tui", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "tui" in result.output.lower() or "dashboard" in result.output.lower()

    def test_refresh_bound_low(self):
        runner = CliRunner()
        result = runner.invoke(app, ["tui", "--refresh", "0.01"])
        assert result.exit_code != 0

    def test_refresh_bound_high(self):
        runner = CliRunner()
        result = runner.invoke(app, ["tui", "--refresh", "999"])
        assert result.exit_code != 0

    def test_limit_bound_low(self):
        runner = CliRunner()
        result = runner.invoke(app, ["tui", "--limit", "0"])
        assert result.exit_code != 0

    def test_missing_textual_friendly_error(self, monkeypatch):
        # Force ImportError on tui_app
        monkeypatch.setitem(sys.modules, "soup_cli.tui_app", None)
        runner = CliRunner()
        result = runner.invoke(app, ["tui"])
        assert result.exit_code != 0
        assert "textual" in result.output.lower() or "install" in result.output.lower()


@pytest.mark.skipif(not textual_available, reason="textual not installed")
class TestRowBuilders:
    def test_build_runs_table_rows(self):
        from soup_cli.tui_app import build_runs_table_rows

        runs = [{
            "run_id": "run_123",
            "experiment_name": "exp",
            "base_model": "meta-llama/Llama-3-8b",
            "task": "sft",
            "status": "completed",
            "final_loss": 0.5,
            "total_steps": 100,
            "cost_usd": 1.23,
        }]
        rows = build_runs_table_rows(runs)
        assert len(rows) == 1
        assert rows[0][0].startswith("run_123")
        assert "0.5" in rows[0][5]
        assert "$1.23" in rows[0][7]

    def test_build_runs_table_rows_missing_fields(self):
        from soup_cli.tui_app import build_runs_table_rows

        rows = build_runs_table_rows([{}])
        assert len(rows) == 1
        # Fields default to "-"
        assert rows[0][0] == "-"
        assert rows[0][7] == "—"  # format_cost_usd(None)

    def test_build_run_detail(self):
        from soup_cli.tui_app import build_run_detail

        run = {
            "run_id": "run_x",
            "status": "completed",
            "base_model": "Qwen2-7b",
            "task": "sft",
            "duration_secs": 60.0,
            "cost_usd": 0.10,
        }
        metrics = [
            {"step": 0, "loss": 2.0},
            {"step": 10, "loss": 1.0},
            {"step": 20, "loss": 0.5},
        ]
        text = build_run_detail(run, metrics)
        assert "run_x" in text
        assert "Qwen2-7b" in text
        assert "2.0000" in text
        assert "0.5000" in text
