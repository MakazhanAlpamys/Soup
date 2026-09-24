"""Regression tests for issue #1138: the MCP run-log append must refuse a symlink.

`ExecutionManager.execute()` spawns the child with its stdout/stderr pointed at
`.soup/mcp-runs/<run_id>.log`, opened with a plain `open(path, "ab")`. Unlike
the digest reader in the same file (hardened in v0.75.1) and the audit log
(`open_no_follow` since #820), that open followed symlinks: a link planted at
the log path redirected every byte the child wrote to the link target.
"""

import os
import stat
import sys
from unittest.mock import MagicMock, patch

import pytest

from soup_cli.mcp_server.execution import ExecutionError, ExecutionManager


def _mock_proc(pid: int = 4242) -> MagicMock:
    proc = MagicMock()
    proc.pid = pid
    proc.wait.return_value = 0
    return proc


class TestRunLogSymlinkRefusal:
    @pytest.mark.requires_symlink
    def test_execute_refuses_a_symlink_planted_at_the_run_log_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        manager = ExecutionManager()
        token = manager.issue(
            kind="train",
            argv=[sys.executable, "--version"],
            display_command="test",
            run_id="symrun",
        )
        log_root = tmp_path / ".soup" / "mcp-runs"
        log_root.mkdir(parents=True, exist_ok=True)
        outside = tmp_path / "outside-secret.txt"
        os.symlink(str(outside), str(log_root / "symrun.log"))

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = _mock_proc()
            with pytest.raises(ExecutionError, match="symbolic link or junction"):
                manager.execute(token=token, kind="train")
            # The child must never be spawned through the redirected log.
            assert not mock_popen.called
        # Nothing was created or appended through the symlink.
        assert not outside.exists()

    def test_execute_creates_a_regular_run_log_and_spawns(self, tmp_path, monkeypatch):
        """Control: the normal (non-symlink) path still gets an append-mode log."""
        monkeypatch.chdir(tmp_path)
        manager = ExecutionManager()
        token = manager.issue(
            kind="train",
            argv=[sys.executable, "--version"],
            display_command="test",
            run_id="okrun",
        )
        seen = {}

        def fake_popen(*args, **kwargs):
            # The parent closes the handle right after Popen, so probe it here.
            handle = kwargs["stdout"]
            assert handle.writable()
            handle.write(b"child output\n")
            seen["handle"] = handle
            return _mock_proc(pid=777)

        with patch("subprocess.Popen", side_effect=fake_popen):
            res = manager.execute(token=token, kind="train")
        assert res["status"] == "running"
        assert res["pid"] == 777
        log_path = tmp_path / ".soup" / "mcp-runs" / "okrun.log"
        assert log_path.exists()
        assert not os.path.islink(log_path)
        # Append-mode binary handle that reached the child's stdout slot.
        assert seen["handle"].mode == "ab"
        assert log_path.read_bytes() == b"child output\n"


class TestRunLogOpenFlags:
    """The OS-level flags of the spawn-time log open, pinned by behaviour.

    The control's ``handle.mode == "ab"`` shows the Python-level fdopen mode
    only; each test here fails if the flag it names is dropped from the
    ``open_no_follow`` call.
    """

    def test_reused_run_log_appends_when_the_file_grows_after_open(self, tmp_path, monkeypatch):
        """O_APPEND: the child's bytes land at EOF even when the log grew
        after the handle was opened.

        Without the OS flag a reused run_id's log is written at the offset
        captured at open, clobbering bytes appended in between — fdopen("ab")
        seeks to EOF once at open, not per write.
        """
        monkeypatch.chdir(tmp_path)
        manager = ExecutionManager()
        token = manager.issue(
            kind="train",
            argv=[sys.executable, "--version"],
            display_command="test",
            run_id="appendrun",
        )
        log_path = tmp_path / ".soup" / "mcp-runs" / "appendrun.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_bytes(b"first\n")  # a reused run_id already has a log

        def fake_popen(*args, **kwargs):
            handle = kwargs["stdout"]
            # The log grows between the handle's open and its first write.
            with open(log_path, "ab") as raced:
                raced.write(b"external\n")
            handle.write(b"second\n")
            return _mock_proc(pid=4321)

        with patch("subprocess.Popen", side_effect=fake_popen):
            manager.execute(token=token, kind="train")
        # With O_APPEND the write through the handle reaches EOF; without it,
        # the write lands at offset 6 and "external" is lost.
        assert log_path.read_bytes() == b"first\nexternal\nsecond\n"

    @pytest.mark.skipif(os.name == "nt", reason="os.umask is POSIX-only")
    def test_fresh_log_keeps_plain_open_permissions(self, tmp_path, monkeypatch):
        """Mode 0o666: a fresh log gets plain open()'s permissions under a
        controlled umask (0o644 with umask 0o022), not a tightened 0o600."""
        monkeypatch.chdir(tmp_path)
        manager = ExecutionManager()
        token = manager.issue(
            kind="train",
            argv=[sys.executable, "--version"],
            display_command="test",
            run_id="moderun",
        )
        saved_umask = os.umask(0o022)
        try:
            with patch("subprocess.Popen", return_value=_mock_proc(pid=4322)):
                manager.execute(token=token, kind="train")
        finally:
            os.umask(saved_umask)
        log_path = tmp_path / ".soup" / "mcp-runs" / "moderun.log"
        assert stat.S_IMODE(log_path.stat().st_mode) == 0o644
