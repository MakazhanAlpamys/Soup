"""The one-active-execution cap sees every launching or running run.

`list_runs()` returns the 50 newest rows, so a live MCP run followed by 50 or
more newer runs dropped out of the capacity check and a second execution could
start beside it. The check now reads every `launching` / `running` row.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

from soup_cli.experiment.tracker import ExperimentTracker
from soup_cli.mcp_server.execution import ExecutionError, ExecutionManager

_PRIOR_RUN = "run-prior"
_ARGV = [sys.executable, "--version"]
_NEWER_RUNS = 60


@pytest.fixture()
def tracker(tmp_path, monkeypatch):
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "experiments.db"))
    monkeypatch.chdir(tmp_path)
    return ExperimentTracker()


def _seed_prior(tracker: ExperimentTracker, *, status: str, pid: int | None) -> None:
    tracker.launch_run(
        run_id=_PRIOR_RUN,
        kind="train",
        config_dict={"mcp_argv": _ARGV},
        command_digest="digest",
        log_path="soup.log",
    )
    conn = tracker._get_conn()
    conn.execute(
        "UPDATE runs SET status = ?, pid = ?, created_at = ? WHERE run_id = ?",
        (status, pid, "2000-01-01T00:00:00", _PRIOR_RUN),
    )
    conn.commit()


def _seed_newer_completed_runs(tracker: ExperimentTracker) -> None:
    for index in range(_NEWER_RUNS):
        tracker.start_run(
            config_dict={"base": "m", "task": "sft"},
            device="cpu",
            device_name="cpu",
            gpu_info={},
            run_id=f"run-newer-{index:03d}",
        )
    conn = tracker._get_conn()
    conn.execute("UPDATE runs SET status = 'completed' WHERE run_id != ?", (_PRIOR_RUN,))
    conn.commit()


def _assert_execution_refused() -> None:
    manager = ExecutionManager()
    assert manager._live_persisted_run() == _PRIOR_RUN
    token = manager.issue(kind="train", argv=_ARGV, display_command="test")
    with patch("subprocess.Popen") as mock_popen:
        with pytest.raises(ExecutionError) as exc:
            manager.execute(token=token, kind="train")
        assert not mock_popen.called
    assert "already active on this machine" in str(exc.value)


def test_running_run_older_than_list_window_blocks(tracker):
    _seed_prior(tracker, status="running", pid=os.getpid())
    _seed_newer_completed_runs(tracker)
    assert _PRIOR_RUN not in {run["run_id"] for run in tracker.list_runs()}
    _assert_execution_refused()


def test_launching_run_older_than_list_window_blocks(tracker):
    _seed_prior(tracker, status="launching", pid=None)
    _seed_newer_completed_runs(tracker)
    assert _PRIOR_RUN not in {run["run_id"] for run in tracker.list_runs()}
    _assert_execution_refused()


def test_list_active_execution_runs_excludes_finished(tracker):
    _seed_prior(tracker, status="running", pid=os.getpid())
    _seed_newer_completed_runs(tracker)
    active = tracker.list_active_execution_runs()
    assert [run["run_id"] for run in active] == [_PRIOR_RUN]
    assert all(run["status"] in ("launching", "running") for run in active)
