"""Reconcile-on-read for runs whose watcher never unwound (issue #401).

`ExecutionManager._watch` runs as a daemon thread; when the MCP server exits it
is killed without running `finish_execution`, leaving the run at 'running'
forever. The tracker's read path reconciles such a row by checking whether the
recorded PID is still alive.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

from soup_cli.experiment.tracker import ExperimentTracker

_STATUS_RUNNING = "running"
_STATUS_TERMINATED = "terminated"
_STATUS_COMPLETED = "completed"
_MCP_KIND = "mcp"
_RUN_DEAD = "run-dead"
_RUN_ALIVE = "run-alive"
_RUN_NO_PID = "run-no-pid"
_RUN_DONE = "run-done"


def _tracker() -> ExperimentTracker:
    return ExperimentTracker(db_path=Path(tempfile.mkdtemp()) / "experiments.db")


def _launch(t: ExperimentTracker, run_id: str) -> None:
    t.launch_run(
        run_id=run_id,
        kind=_MCP_KIND,
        config_dict={"lr": 1},
        command_digest="digest",
        log_path="/tmp/soup.log",
    )


def _dead_pid() -> int:
    """A PID that is guaranteed gone: spawn a real process and wait for it."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_dead_watcher_run_is_reconciled_not_reported_running():
    # The exact #401 shape: mark_running records a pid, the watcher never calls
    # finish_execution (daemon thread killed), and the process is gone.
    t = _tracker()
    _launch(t, _RUN_DEAD)
    t.mark_running(_RUN_DEAD, pid=_dead_pid())

    run = t.get_run(_RUN_DEAD)
    assert run["status"] == _STATUS_TERMINATED
    # An unknown outcome must NOT be written as success.
    assert run["status"] != _STATUS_COMPLETED
    assert run["exit_code"] is None
    # Reconciliation is persisted, so list_runs agrees.
    assert t.list_runs()[0]["status"] == _STATUS_TERMINATED


def test_control_a_live_process_is_still_reported_running():
    # CONTROL: a genuinely-running process (this test's own PID) must stay running.
    import os

    t = _tracker()
    _launch(t, _RUN_ALIVE)
    t.mark_running(_RUN_ALIVE, pid=os.getpid())

    assert t.get_run(_RUN_ALIVE)["status"] == _STATUS_RUNNING


def test_running_run_without_a_pid_is_left_untouched():
    # A run with no recorded pid (never MCP-spawned) cannot be liveness-checked,
    # so it must not be reconciled away.
    t = _tracker()
    _launch(t, _RUN_NO_PID)
    # A launched run carries no pid until mark_running; force 'running' with a
    # NULL pid to model a run that was never MCP-spawned.
    conn = t._get_conn()
    conn.execute(
        "UPDATE runs SET status = ?, pid = NULL WHERE run_id = ?",
        (_STATUS_RUNNING, _RUN_NO_PID),
    )
    conn.commit()

    assert t.get_run(_RUN_NO_PID)["status"] == _STATUS_RUNNING


def test_list_runs_reconciles_when_it_is_the_first_read():
    """`list_runs` must reconcile on its own, not inherit `get_run`'s work.

    The assertion at the end of the first test reads `list_runs` only AFTER
    `get_run` has already reconciled and persisted the flip, so it agrees for
    free: deleting the reconcile call from `list_runs` passes that test — and,
    measured, 191 others. This one makes `list_runs` the only read path, so it
    fails if that call site is removed.
    """
    t = _tracker()
    _launch(t, _RUN_DEAD)
    t.mark_running(_RUN_DEAD, pid=_dead_pid())

    runs = t.list_runs()
    assert runs[0]["status"] == _STATUS_TERMINATED
    assert runs[0]["exit_code"] is None


def test_control_list_runs_leaves_a_live_process_running():
    # CONTROL: the test above must not pass by flipping every row to terminal.
    import os

    t = _tracker()
    _launch(t, _RUN_ALIVE)
    t.mark_running(_RUN_ALIVE, pid=os.getpid())

    assert t.list_runs()[0]["status"] == _STATUS_RUNNING


def test_terminal_status_is_never_overwritten_by_reconcile():
    # A finished run keeps its terminal status even though its pid is long gone.
    t = _tracker()
    _launch(t, _RUN_DONE)
    t.mark_running(_RUN_DONE, pid=_dead_pid())
    t.finish_execution(_RUN_DONE, status=_STATUS_COMPLETED, exit_code=0)

    run = t.get_run(_RUN_DONE)
    assert run["status"] == _STATUS_COMPLETED
    assert run["exit_code"] == 0
