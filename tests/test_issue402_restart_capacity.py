"""One-active-execution cap survives a server restart (issue #402).

`ExecutionManager._active_run_id` is in-memory: a restarted server starts with
it `None`, sees free capacity, and would launch a second training while a child
from the previous server is still alive. The cap is now gated on a persisted
run whose pid is still alive, so a restart cannot double-book — while a stale
record whose process is gone never blocks execution forever.
"""
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from soup_cli.experiment.tracker import ExperimentTracker
from soup_cli.mcp_server.execution import ExecutionError, ExecutionManager

_RUNNING = "running"
_MCP_KIND = "train"
_PRIOR_RUN = "run-prior"
_ARGV = [sys.executable, "--version"]


def _use_temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "experiments.db"))
    monkeypatch.chdir(tmp_path)


def _seed_running_run(pid) -> None:
    t = ExperimentTracker()
    t.launch_run(
        run_id=_PRIOR_RUN,
        kind=_MCP_KIND,
        config_dict={"mcp_argv": _ARGV},
        command_digest="digest",
        log_path="/tmp/soup.log",
    )
    t._get_conn().execute(
        "UPDATE runs SET status = ?, pid = ? WHERE run_id = ?",
        (_RUNNING, pid, _PRIOR_RUN),
    )
    t._get_conn().commit()


def test_restarted_server_refuses_while_prior_child_alive(tmp_path, monkeypatch):
    _use_temp_db(tmp_path, monkeypatch)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        _seed_running_run(child.pid)
        # Fresh server = restart: its in-memory slot is None.
        manager = ExecutionManager()
        assert manager._active_run_id is None
        token = manager.issue(kind=_MCP_KIND, argv=_ARGV, display_command="test")
        with pytest.raises(ExecutionError) as exc:
            manager.execute(token=token, kind=_MCP_KIND)
        assert "already active" in str(exc.value).lower()
    finally:
        child.terminate()
        child.wait()


def test_stale_dead_pid_record_does_not_block(tmp_path, monkeypatch):
    _use_temp_db(tmp_path, monkeypatch)
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()  # process gone; its pid is a stale 'running' record
    _seed_running_run(dead.pid)

    manager = ExecutionManager()
    token = manager.issue(kind=_MCP_KIND, argv=_ARGV, display_command="test")
    with patch("subprocess.Popen") as mock_popen:
        proc = MagicMock()
        proc.pid = 4321
        proc.wait.return_value = 0
        mock_popen.return_value = proc
        res = manager.execute(token=token, kind=_MCP_KIND)
    # A dead prior record must not block a new execution.
    assert res["status"] == _RUNNING


def test_capacity_check_rejects_a_dead_pid_without_help_from_reconcile(
    tmp_path, monkeypatch
):
    """The cap's own liveness check must hold when reconcile-on-read does not.

    #407 rewrites a stale 'running' row on read, so by the time
    `_live_persisted_run` inspects `status` the dead row is already terminal —
    which means its `process_is_alive` guard is never the thing under test.
    Measured: deleting that guard passes all 41 MCP tests. Here reconcile is
    neutralised so the row stays 'running' with a dead pid, leaving the cap's
    own check as the only thing that can reject it. The two are each other's
    fallback; neither may be the only one guarded.
    """
    _use_temp_db(tmp_path, monkeypatch)
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    _seed_running_run(dead.pid)

    with patch.object(
        ExperimentTracker, "_reconcile_orphaned_run", side_effect=lambda run: run
    ):
        assert ExperimentTracker().list_runs()[0]["status"] == _RUNNING  # precondition
        assert ExecutionManager()._live_persisted_run() is None


def test_control_the_capacity_check_still_sees_a_live_pid_without_reconcile(
    tmp_path, monkeypatch
):
    # CONTROL: neutralising reconcile must not make the check reject everything.
    _use_temp_db(tmp_path, monkeypatch)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        _seed_running_run(child.pid)
        with patch.object(
            ExperimentTracker, "_reconcile_orphaned_run", side_effect=lambda run: run
        ):
            assert ExecutionManager()._live_persisted_run() == _PRIOR_RUN
    finally:
        child.terminate()
        child.wait()


def test_live_persisted_run_detects_only_live_pids(tmp_path, monkeypatch):
    _use_temp_db(tmp_path, monkeypatch)
    manager = ExecutionManager()
    # No runs yet.
    assert manager._live_persisted_run() is None
    # Dead pid → not active.
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    _seed_running_run(dead.pid)
    assert manager._live_persisted_run() is None
    # Live pid → active.
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        conn = ExperimentTracker()._get_conn()
        # `status` is restored alongside `pid`, and that is not defensive
        # tidiness: the assertion above reads the row through the tracker, and
        # #401's reconcile-on-read has by then rewritten it from 'running' to
        # 'terminated' precisely because the pid was dead. Updating only `pid`
        # would leave a terminated row that `_live_persisted_run` correctly
        # ignores, and the test would assert against the wrong state.
        conn.execute(
            "UPDATE runs SET pid = ?, status = 'running' WHERE run_id = ?",
            (child.pid, _PRIOR_RUN),
        )
        conn.commit()
        assert manager._live_persisted_run() == _PRIOR_RUN
    finally:
        child.terminate()
        child.wait()
