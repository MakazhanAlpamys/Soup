"""#1158 — MCP run log directory symlink, junction, and hardlink protections.

PR #1150 / #1138 closed the final-component file symlink at .soup/mcp-runs/<id>.log.
This test suite covers the neighbouring redirections:
1. A directory symlink or junction one level up (.soup/mcp-runs -> evildir).
2. A directory symlink or junction two levels up (.soup -> evildir).
3. A pre-planted hardlink at the log path over an existing file.
"""

from __future__ import annotations

import errno
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from soup_cli.mcp_server.execution import ExecutionError, ExecutionManager
from soup_cli.utils.paths import open_no_follow


def _mock_proc(pid: int = 4242) -> MagicMock:
    proc = MagicMock()
    proc.pid = pid
    proc.wait.return_value = 0
    return proc


class TestRunLogDirectorySymlinkAndHardlink:
    @pytest.mark.requires_symlink
    def test_execute_refuses_a_directory_symlink_one_level_up(self, tmp_path, monkeypatch):
        """A directory symlink at .soup/mcp-runs -> evildir is refused before spawn."""
        monkeypatch.chdir(tmp_path)
        manager = ExecutionManager()
        token = manager.issue(
            kind="train",
            argv=[sys.executable, "--version"],
            display_command="test",
            run_id="dirsymrun",
        )
        evildir = tmp_path / "evildir"
        evildir.mkdir()
        soup_dir = tmp_path / ".soup"
        soup_dir.mkdir()
        os.symlink(str(evildir), str(soup_dir / "mcp-runs"), target_is_directory=True)

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = _mock_proc()
            with pytest.raises(ExecutionError, match="symbolic link or junction"):
                manager.execute(token=token, kind="train")
            assert not mock_popen.called
        assert not (evildir / "dirsymrun.log").exists()

    @pytest.mark.requires_symlink
    def test_execute_refuses_symlink_at_soup_directory_itself(self, tmp_path, monkeypatch):
        """A symlink at .soup itself is refused without creating mcp-runs in target."""
        monkeypatch.chdir(tmp_path)
        manager = ExecutionManager()
        token = manager.issue(
            kind="train",
            argv=[sys.executable, "--version"],
            display_command="test",
            run_id="soupdirsymrun",
        )
        evildir = tmp_path / "evildir"
        evildir.mkdir()
        os.symlink(str(evildir), str(tmp_path / ".soup"), target_is_directory=True)

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = _mock_proc()
            with pytest.raises(ExecutionError, match="symbolic link or junction"):
                manager.execute(token=token, kind="train")
            assert not mock_popen.called
        assert not (evildir / "mcp-runs").exists()

    @pytest.mark.skipif(os.name != "nt", reason="Windows junctions only exist on Windows")
    def test_execute_refuses_a_directory_junction_one_level_up(self, tmp_path, monkeypatch):
        """A directory junction at .soup/mcp-runs -> evildir is refused before spawn."""
        import _winapi

        monkeypatch.chdir(tmp_path)
        manager = ExecutionManager()
        token = manager.issue(
            kind="train",
            argv=[sys.executable, "--version"],
            display_command="test",
            run_id="junctionrun",
        )
        evildir = tmp_path / "evildir"
        evildir.mkdir()
        soup_dir = tmp_path / ".soup"
        soup_dir.mkdir()
        _winapi.CreateJunction(str(evildir), str(soup_dir / "mcp-runs"))

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = _mock_proc()
            with pytest.raises(ExecutionError, match="symbolic link or junction"):
                manager.execute(token=token, kind="train")
            assert not mock_popen.called
        assert not (evildir / "junctionrun.log").exists()

    @pytest.mark.skipif(os.name != "nt", reason="Windows junctions only exist on Windows")
    def test_execute_refuses_junction_at_soup_directory_itself(self, tmp_path, monkeypatch):
        """A junction at .soup itself is refused without creating mcp-runs in target."""
        import _winapi

        monkeypatch.chdir(tmp_path)
        manager = ExecutionManager()
        token = manager.issue(
            kind="train",
            argv=[sys.executable, "--version"],
            display_command="test",
            run_id="soupjunctionrun",
        )
        evildir = tmp_path / "evildir"
        evildir.mkdir()
        _winapi.CreateJunction(str(evildir), str(tmp_path / ".soup"))

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = _mock_proc()
            with pytest.raises(ExecutionError, match="symbolic link or junction"):
                manager.execute(token=token, kind="train")
            assert not mock_popen.called
        assert not (evildir / "mcp-runs").exists()

    @pytest.mark.requires_hardlink
    def test_execute_refuses_a_hardlink_at_the_run_log_path(self, tmp_path, monkeypatch):
        """A hardlink at the log path over an existing file is refused before spawn."""
        monkeypatch.chdir(tmp_path)
        manager = ExecutionManager()
        token = manager.issue(
            kind="train",
            argv=[sys.executable, "--version"],
            display_command="test",
            run_id="hardlinkrun",
        )
        victim = tmp_path / "victim.txt"
        victim.write_bytes(b"SECRET\n")
        log_root = tmp_path / ".soup" / "mcp-runs"
        log_root.mkdir(parents=True, exist_ok=True)
        os.link(str(victim), str(log_root / "hardlinkrun.log"))

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = _mock_proc()
            with pytest.raises(ExecutionError, match="hard link"):
                manager.execute(token=token, kind="train")
            assert not mock_popen.called
        assert victim.read_bytes() == b"SECRET\n"

    def test_execute_ordinary_path_creates_regular_file(self, tmp_path, monkeypatch):
        """Ordinary path creates a regular file and writes child bytes."""
        monkeypatch.chdir(tmp_path)
        manager = ExecutionManager()
        token = manager.issue(
            kind="train",
            argv=[sys.executable, "--version"],
            display_command="test",
            run_id="regularrun",
        )

        def fake_popen(*args, **kwargs):
            handle = kwargs["stdout"]
            handle.write(b"regular output\n")
            return _mock_proc(pid=1234)

        with patch("subprocess.Popen", side_effect=fake_popen):
            res = manager.execute(token=token, kind="train")
        assert res["status"] == "running"
        assert res["pid"] == 1234
        log_path = tmp_path / ".soup" / "mcp-runs" / "regularrun.log"
        assert log_path.exists()
        assert not os.path.islink(log_path)
        assert log_path.read_bytes() == b"regular output\n"


class TestOpenNoFollowHardlinkAndParentChecks:
    @pytest.mark.requires_symlink
    def test_open_no_follow_rejects_parent_directory_symlink(self, tmp_path):
        target_dir = tmp_path / "target"
        target_dir.mkdir()
        link_dir = tmp_path / "link"
        os.symlink(str(target_dir), str(link_dir), target_is_directory=True)
        file_path = link_dir / "test.log"

        with pytest.raises(OSError) as exc_info:
            open_no_follow(
                file_path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o666,
                check_parent=True,
            )
        assert exc_info.value.errno == errno.ELOOP

    @pytest.mark.requires_hardlink
    def test_open_no_follow_rejects_hardlink_with_flag(self, tmp_path):
        src = tmp_path / "orig.txt"
        src.write_bytes(b"content")
        link = tmp_path / "link.txt"
        os.link(str(src), str(link))

        with pytest.raises(OSError) as exc_info:
            open_no_follow(link, os.O_RDWR, refuse_hardlink=True)
        assert exc_info.value.errno == errno.EMLINK
        assert "Hard link not allowed" in str(exc_info.value)

    @pytest.mark.requires_hardlink
    def test_open_no_follow_allows_hardlink_by_default(self, tmp_path):
        """Control: callers like audit log and draft registry do not reject hardlinks."""
        src = tmp_path / "orig_default.txt"
        src.write_bytes(b"content")
        link = tmp_path / "link_default.txt"
        os.link(str(src), str(link))

        fd = open_no_follow(link, os.O_RDWR)
        try:
            assert fd >= 0
        finally:
            os.close(fd)
