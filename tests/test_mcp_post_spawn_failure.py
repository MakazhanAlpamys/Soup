"""A bookkeeping failure after the child starts never leaves it unsupervised.

`mark_running` and the watcher thread ran outside any error handling, so a
locked database or a refused thread start raised past `execute` with the child
still running, no watcher to record its exit, and the in-memory slot held.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

from soup_cli.experiment.tracker import ExperimentTracker
from soup_cli.mcp_server.execution import ExecutionError, ExecutionManager
from soup_cli.utils.process_liveness import process_is_alive

_SLEEPER = [sys.executable, "-c", "import time; time.sleep(60)"]
_REAL_POPEN = subprocess.Popen


@pytest.fixture()
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "experiments.db"))
    monkeypatch.chdir(tmp_path)
    return tmp_path


class _RecordingPopen:
    """Start the real child and keep a handle so the test can always clean up."""

    def __init__(self) -> None:
        self.processes: list[subprocess.Popen] = []

    def __call__(self, *args, **kwargs):
        process = _REAL_POPEN(*args, **kwargs)
        self.processes.append(process)
        return process

    def cleanup(self) -> None:
        for process in self.processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=15)


def _wait_not_alive(pid: int, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process_is_alive(pid):
            return True
        time.sleep(0.05)
    return not process_is_alive(pid)


def _assert_stopped_and_released(manager: ExecutionManager, recorder: _RecordingPopen, exc):
    assert "child process was stopped" in str(exc.value)
    assert len(recorder.processes) == 1
    child = recorder.processes[0]
    assert _wait_not_alive(child.pid), "child process is still running"
    assert manager._active_run_id is None
    run_ids = [plan.run_id for plan in manager._plans.values()]
    assert len(run_ids) == 1
    row = ExperimentTracker().get_run(run_ids[0])
    assert row is not None
    assert row["status"] == "spawn_failed"


def test_mark_running_failure_stops_child_and_frees_slot(project):
    manager = ExecutionManager()
    token = manager.issue(kind="train", argv=_SLEEPER, display_command="sleep")
    recorder = _RecordingPopen()
    try:
        with patch("subprocess.Popen", side_effect=recorder), patch.object(
            ExperimentTracker,
            "mark_running",
            side_effect=sqlite3.OperationalError("database is locked"),
        ):
            with pytest.raises(ExecutionError) as exc:
                manager.execute(token=token, kind="train")
        _assert_stopped_and_released(manager, recorder, exc)
    finally:
        recorder.cleanup()


def test_thread_start_failure_stops_child(project):
    import soup_cli.mcp_server.execution as execution

    manager = ExecutionManager()
    token = manager.issue(kind="train", argv=_SLEEPER, display_command="sleep")
    recorder = _RecordingPopen()
    try:
        with patch("subprocess.Popen", side_effect=recorder), patch.object(
            execution.threading.Thread,
            "start",
            side_effect=RuntimeError("can't start new thread"),
        ):
            with pytest.raises(ExecutionError) as exc:
                manager.execute(token=token, kind="train")
        _assert_stopped_and_released(manager, recorder, exc)
    finally:
        recorder.cleanup()


def test_next_execution_allowed_after_bookkeeping_failure(project):
    manager = ExecutionManager()
    first = manager.issue(kind="train", argv=_SLEEPER, display_command="sleep")
    second = manager.issue(kind="train", argv=_SLEEPER, display_command="sleep")
    recorder = _RecordingPopen()
    try:
        with patch("subprocess.Popen", side_effect=recorder), patch.object(
            ExperimentTracker,
            "mark_running",
            side_effect=sqlite3.OperationalError("database is locked"),
        ):
            with pytest.raises(ExecutionError):
                manager.execute(token=first, kind="train")

        proc = MagicMock()
        proc.pid = 5150
        proc.wait.return_value = 0
        with patch("subprocess.Popen", return_value=proc):
            result = manager.execute(token=second, kind="train")
        assert result["status"] == "running"
    finally:
        recorder.cleanup()
