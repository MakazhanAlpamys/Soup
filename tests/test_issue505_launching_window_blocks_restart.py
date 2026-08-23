"""A crashed server between Popen() and mark_running() must still block a
restart, not read as free capacity (issue #505).

`ExecutionManager.execute()` inserts a `status='launching'` row with `pid=NULL`
before `subprocess.Popen()`, and only upgrades it to `status='running'` with a
pid afterwards. A server crash in between leaves the row stuck at
`'launching'` forever: it has no pid, so neither #401's reconcile-on-read nor
#402's `_live_persisted_run` liveness check can ever inspect it, and a
restarted server sees free capacity and double-books.
"""
import sys
from unittest.mock import MagicMock, patch

import pytest

from soup_cli.experiment.tracker import ExperimentTracker
from soup_cli.mcp_server.execution import ExecutionError, ExecutionManager

_MCP_KIND = "train"
_ARGV = [sys.executable, "--version"]
_ORPHAN_RUN = "run-orphan-launching"


def _use_temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "experiments.db"))
    monkeypatch.chdir(tmp_path)


def _seed_crash_between_popen_and_mark_running() -> None:
    """Model step 1 (launch_run) having happened and mark_running never running."""
    t = ExperimentTracker()
    t.launch_run(
        run_id=_ORPHAN_RUN,
        kind=_MCP_KIND,
        config_dict={"mcp_argv": _ARGV},
        command_digest="digest",
        log_path="/tmp/soup.log",
    )
    row = t.get_run(_ORPHAN_RUN)
    assert row["status"] == "launching"
    assert row["pid"] is None


def test_stuck_launching_row_blocks_restart_capacity(tmp_path, monkeypatch):
    _use_temp_db(tmp_path, monkeypatch)
    _seed_crash_between_popen_and_mark_running()

    # Fresh server = restart: in-memory slot is None, same fixture #402 uses.
    manager = ExecutionManager()
    assert manager._active_run_id is None
    assert manager._live_persisted_run() == _ORPHAN_RUN


def test_restart_after_crash_window_refuses_a_second_execution(tmp_path, monkeypatch):
    _use_temp_db(tmp_path, monkeypatch)
    _seed_crash_between_popen_and_mark_running()

    manager = ExecutionManager()
    token = manager.issue(kind=_MCP_KIND, argv=_ARGV, display_command="second")
    with patch("subprocess.Popen") as mock_popen:
        proc = MagicMock()
        proc.pid = 99999
        proc.wait.return_value = 0
        mock_popen.return_value = proc
        with pytest.raises(ExecutionError) as exc:
            manager.execute(token=token, kind=_MCP_KIND)
    assert "already active" in str(exc.value).lower()
    mock_popen.assert_not_called()


def test_control_a_running_row_with_a_dead_pid_still_frees_capacity(tmp_path, monkeypatch):
    # CONTROL: the launching-specific branch must not swallow the existing
    # #401/#402 behaviour for a genuine 'running' row.
    import subprocess

    _use_temp_db(tmp_path, monkeypatch)
    t = ExperimentTracker()
    t.launch_run(
        run_id="run-dead-pid",
        kind=_MCP_KIND,
        config_dict={"mcp_argv": _ARGV},
        command_digest="digest",
        log_path="/tmp/soup.log",
    )
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    t.mark_running("run-dead-pid", pid=dead.pid)

    assert ExecutionManager()._live_persisted_run() is None


def test_control_no_runs_yet_reports_free_capacity(tmp_path, monkeypatch):
    _use_temp_db(tmp_path, monkeypatch)
    assert ExecutionManager()._live_persisted_run() is None


def test_stuck_launching_row_still_reads_as_launching(tmp_path, monkeypatch):
    # This fix blocks capacity without guessing at an outcome it cannot
    # verify: reconcile-on-read must not rewrite the row to a terminal
    # status it has no pid to justify.
    _use_temp_db(tmp_path, monkeypatch)
    _seed_crash_between_popen_and_mark_running()

    assert ExperimentTracker().get_run(_ORPHAN_RUN)["status"] == "launching"
    assert ExperimentTracker().list_runs()[0]["status"] == "launching"
