"""Operator recovery for stale MCP ``launching`` rows (issue #524)."""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.experiment.tracker import ExperimentTracker

runner = CliRunner()


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "experiments.db"))


def _launching_run(run_id: str, *, age_seconds: int, pid: int | None = None) -> None:
    tracker = ExperimentTracker()
    tracker.launch_run(
        run_id=run_id,
        kind="train",
        config_dict={"mcp_argv": ["soup", "train"]},
        command_digest="digest",
        log_path="mcp.log",
    )
    created_at = (datetime.now() - timedelta(seconds=age_seconds)).isoformat()
    conn = tracker._get_conn()
    conn.execute(
        "UPDATE runs SET created_at = ?, pid = ? WHERE run_id = ?",
        (created_at, pid, run_id),
    )
    conn.commit()


def test_reconcile_expunge_prints_each_stale_run_id_and_keeps_fresh_rows():
    stale_one = "run-stale-one"
    stale_two = "run-stale-two"
    fresh = "run-fresh"
    _launching_run(stale_one, age_seconds=600)
    _launching_run(stale_two, age_seconds=601)
    _launching_run(fresh, age_seconds=30)

    result = runner.invoke(
        app,
        [
            "mcp",
            "runs",
            "reconcile",
            "--expunge-launching",
            "--older-than-seconds",
            "300",
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.output.count(stale_one) == 1
    assert result.output.count(stale_two) == 1
    assert fresh not in result.output
    tracker = ExperimentTracker()
    assert tracker.get_run(stale_one) is None
    assert tracker.get_run(stale_two) is None
    assert tracker.get_run(fresh) is not None


def test_reconcile_refuses_live_pid_without_partially_deleting_other_rows():
    stale_pidless = "run-stale-pidless"
    live = "run-live-pid"
    _launching_run(stale_pidless, age_seconds=600)
    _launching_run(live, age_seconds=600, pid=os.getpid())

    result = runner.invoke(
        app,
        ["mcp", "runs", "reconcile", "--expunge-launching"],
    )

    assert result.exit_code == 1
    assert "refusing" in result.output.lower()
    assert live in result.output
    assert str(os.getpid()) in result.output
    tracker = ExperimentTracker()
    assert tracker.get_run(live) is not None
    assert tracker.get_run(stale_pidless) is not None


def test_reconcile_requires_explicit_expunge_flag():
    stale = "run-stale"
    _launching_run(stale, age_seconds=600)

    result = runner.invoke(app, ["mcp", "runs", "reconcile"])

    assert result.exit_code == 2
    assert "--expunge-launching is required" in result.output
    assert ExperimentTracker().get_run(stale) is not None
