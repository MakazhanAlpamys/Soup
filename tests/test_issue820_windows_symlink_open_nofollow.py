"""Tests for Issue #820: Windows draft registry and audit log symlink security.

Verifies:
1. open_no_follow refuses symlinks across all platforms (including Windows).
2. Draft registry lock symlink does not create the target file.
3. Audit log symlink refuses writes and leaves target untouched.
4. Repo-wide ratchet ensuring no new bare getattr(os, "O_NOFOLLOW", 0) idiom.
"""

from __future__ import annotations

import ast
import errno
import os
import stat
import tempfile
from pathlib import Path
from typing import List, Tuple

import pytest

from soup_cli.utils.audit_log import AuditEvent, append_audit_event
from soup_cli.utils.draft import _registry_lock, list_drafts
from soup_cli.utils.paths import open_no_follow

SRC = Path(__file__).resolve().parents[1] / "src" / "soup_cli"


def _can_create_symlinks() -> bool:
    try:
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "src.txt")
            dst = os.path.join(td, "dst.txt")
            with open(src, "w") as f:
                f.write("a")
            os.symlink(src, dst)
            return True
    except OSError:
        return False


CAN_SYMLINK = _can_create_symlinks()
require_symlink = pytest.mark.skipif(
    not CAN_SYMLINK,
    reason="Symlink creation is not permitted on this host / environment",
)


# ===========================================================================
# 1. open_no_follow helper unit tests
# ===========================================================================


class TestOpenNoFollow:
    def test_regular_file_read_write(self, tmp_path: Path) -> None:
        file_path = tmp_path / "normal.txt"
        fd = open_no_follow(file_path, os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(fd, b"hello open_no_follow")
        finally:
            os.close(fd)

        assert file_path.read_bytes() == b"hello open_no_follow"

        fd2 = open_no_follow(file_path, os.O_RDONLY)
        try:
            data = os.read(fd2, 64)
            assert data == b"hello open_no_follow"
        finally:
            os.close(fd2)

    def test_nonexistent_file_without_creat_raises(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.txt"
        with pytest.raises(FileNotFoundError):
            open_no_follow(missing, os.O_RDONLY)

    def test_invalid_path_types_and_values(self) -> None:
        with pytest.raises(TypeError, match="str or Path"):
            open_no_follow(123, os.O_RDONLY)  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="non-empty"):
            open_no_follow("", os.O_RDONLY)

        with pytest.raises(ValueError, match="null bytes"):
            open_no_follow("file\x00.txt", os.O_RDONLY)

    @require_symlink
    def test_symlink_to_existing_file_rejected(self, tmp_path: Path) -> None:
        target = tmp_path / "real.txt"
        target.write_text("secret", encoding="utf-8")
        link = tmp_path / "link.txt"
        os.symlink(str(target), str(link))

        with pytest.raises(OSError) as exc_info:
            open_no_follow(link, os.O_RDONLY)
        assert exc_info.value.errno in (errno.ELOOP, errno.EEXIST)

    @require_symlink
    def test_dangling_symlink_rejected_and_not_created(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "does_not_exist.txt"
        link = tmp_path / "dangling_link.txt"
        os.symlink(str(nonexistent), str(link))

        # Even with O_CREAT, open_no_follow must not follow or create target
        with pytest.raises(OSError) as exc_info:
            open_no_follow(link, os.O_CREAT | os.O_WRONLY, 0o600)
        assert exc_info.value.errno in (errno.ELOOP, errno.EEXIST)
        assert not nonexistent.exists()

    def test_reparse_point_rejection_on_windows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_file = tmp_path / "reparse_candidate.txt"
        real_file.write_text("content", encoding="utf-8")

        fake_reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)

        class MockStatWithReparse:
            st_mode = stat.S_IFREG | 0o644
            st_ino = 100
            st_dev = 200
            st_file_attributes = fake_reparse

        monkeypatch.setattr(os, "name", "nt")
        monkeypatch.setattr(os, "lstat", lambda p: MockStatWithReparse())

        with pytest.raises(OSError) as exc_info:
            open_no_follow(real_file, os.O_RDONLY)
        assert exc_info.value.errno == errno.ELOOP
        assert "Reparse point not allowed" in str(exc_info.value)

    def test_post_open_toctou_swap_detected_on_windows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_file = tmp_path / "swap_candidate.txt"
        real_file.write_text("content", encoding="utf-8")

        monkeypatch.setattr(os, "name", "nt")

        class MockSwappedFstat:
            st_mode = stat.S_IFREG | 0o644
            st_ino = 99999
            st_dev = 88888

        # fstat returns different ino/dev than lstat on existing file
        monkeypatch.setattr(os, "fstat", lambda fd: MockSwappedFstat())

        with pytest.raises(OSError) as exc_info:
            open_no_follow(real_file, os.O_RDONLY)
        assert exc_info.value.errno == errno.ELOOP
        assert "File swapped during open" in str(exc_info.value)

    def test_post_open_symlink_swap_on_create_windows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "new_target.txt"

        monkeypatch.setattr(os, "name", "nt")
        calls = 0

        class MockStatSymlink:
            st_mode = stat.S_IFLNK | 0o777
            st_ino = 123
            st_dev = 456
            st_file_attributes = 0

        def fake_lstat(path):
            nonlocal calls
            calls += 1
            if calls == 1:
                # pre_st: file does not exist initially
                raise FileNotFoundError()
            # post_lst: someone swapped in a symlink right after open
            return MockStatSymlink()

        monkeypatch.setattr(os, "lstat", fake_lstat)

        with pytest.raises(OSError) as exc_info:
            open_no_follow(target, os.O_CREAT | os.O_WRONLY, 0o600)
        assert exc_info.value.errno == errno.ELOOP
        assert "Symbolic link created during open" in str(exc_info.value)


# ===========================================================================
# 2. Draft Registry and Lock Symlink Tests (#820)
# ===========================================================================


class TestDraftRegistrySymlinkDefences:
    @require_symlink
    def test_draft_registry_symlink_read_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        secret = tmp_path / "secret.json"
        secret.write_text(
            '{"drafts": [{"target": "leaked", "draft": "/tmp"}]}',
            encoding="utf-8",
        )
        registry_link = tmp_path / "drafts.json"
        os.symlink(str(secret), str(registry_link))

        monkeypatch.setenv("SOUP_DRAFT_REGISTRY_PATH", str(registry_link))
        assert list_drafts() == []

    @require_symlink
    def test_draft_lock_symlink_does_not_create_target(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        victim = tmp_path / "victim_lock_target.txt"
        assert not victim.exists()

        registry_path = tmp_path / "drafts.json"
        lock_path = tmp_path / "drafts.json.lock"
        os.symlink(str(victim), str(lock_path))

        monkeypatch.setenv("SOUP_DRAFT_REGISTRY_PATH", str(registry_path))

        # Attempt to acquire the registry lock
        with _registry_lock():
            pass

        # The victim file must NOT have been created
        assert not victim.exists(), (
            "Vulnerability #820 reproduced: symlink at drafts.json.lock "
            "caused target file to be created!"
        )


# ===========================================================================
# 3. Audit Log Symlink Tests (#820)
# ===========================================================================


class TestAuditLogSymlinkDefences:
    @require_symlink
    def test_append_audit_event_refuses_symlink_and_leaves_target_untouched(
        self, tmp_path: Path
    ) -> None:
        victim = tmp_path / "victim_audit.txt"
        victim.write_text("PRISTINE_AUDIT_DATA\n", encoding="utf-8")

        link = tmp_path / "audit.jsonl"
        os.symlink(str(victim), str(link))

        ev = AuditEvent(
            timestamp="2026-09-16T12:00:00Z",
            command="version",
            args=(),
            exit_code=0,
            host_id="test_host",
            operator_id="test_user",
        )

        with pytest.raises(OSError) as exc_info:
            append_audit_event(ev, str(link))
        assert exc_info.value.errno in (errno.ELOOP, errno.EEXIST)

        # Ensure victim file content was not appended or modified
        assert victim.read_text(encoding="utf-8") == "PRISTINE_AUDIT_DATA\n"

    @require_symlink
    def test_append_audit_event_dangling_symlink_does_not_create_target(
        self, tmp_path: Path
    ) -> None:
        victim = tmp_path / "victim_audit_uncreated.txt"
        assert not victim.exists()

        link = tmp_path / "audit_dangling.jsonl"
        os.symlink(str(victim), str(link))

        ev = AuditEvent(
            timestamp="2026-09-16T12:00:00Z",
            command="version",
            args=(),
            exit_code=0,
            host_id="test_host",
            operator_id="test_user",
        )

        with pytest.raises(OSError) as exc_info:
            append_audit_event(ev, str(link))
        assert exc_info.value.errno in (errno.ELOOP, errno.EEXIST)
        assert not victim.exists()


# ===========================================================================
# 4. Ratchet: Bare getattr(os, "O_NOFOLLOW", 0) Scanner
# ===========================================================================


def _scan_bare_nofollow(root: Path) -> List[Tuple[str, int]]:
    """Find all occurrences of getattr(os, 'O_NOFOLLOW', ...) in Python files."""
    offenders: List[Tuple[str, int]] = []
    for py_file in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "os"
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == "O_NOFOLLOW"
            ):
                rel = py_file.relative_to(root).as_posix()
                offenders.append((rel, node.lineno))
    return offenders


class TestBareNoFollowRatchet:
    """Ratchet: ensures bare getattr(os, 'O_NOFOLLOW', 0) does not spread,

    and verifies draft.py and audit_log.py are cleanly migrated.
    """

    def test_draft_and_audit_log_contain_no_bare_nofollow(self) -> None:
        sites = _scan_bare_nofollow(SRC)
        site_files = {rel for rel, _line in sites}

        assert "utils/draft.py" not in site_files, (
            "utils/draft.py must use open_no_follow instead of "
            "bare getattr(os, 'O_NOFOLLOW', 0)"
        )
        assert "utils/audit_log.py" not in site_files, (
            "utils/audit_log.py must use open_no_follow instead of "
            "bare getattr(os, 'O_NOFOLLOW', 0)"
        )

    def test_total_bare_nofollow_sites_does_not_exceed_baseline(self) -> None:
        sites = _scan_bare_nofollow(SRC)
        non_helper_sites = [s for s in sites if s[0] != "utils/paths.py"]

        # 38 legacy sites remain outside paths.py; none may be added.
        max_legacy_sites = 38
        assert len(non_helper_sites) <= max_legacy_sites, (
            f"Found {len(non_helper_sites)} bare getattr(os, 'O_NOFOLLOW', 0) sites, "
            f"exceeding ratchet baseline of {max_legacy_sites}. "
            f"New code must use soup_cli.utils.paths.open_no_follow. "
            f"Offenders: {non_helper_sites}"
        )

    def test_scanner_can_actually_fail(self, tmp_path: Path) -> None:
        """Negative proof: verify _scan_bare_nofollow detects bare getattr."""
        test_file = tmp_path / "bad.py"
        test_file.write_text(
            "import os\nflags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)\n",
            encoding="utf-8",
        )
        found = _scan_bare_nofollow(tmp_path)
        assert len(found) == 1
        assert found[0][0] == "bad.py"
        assert found[0][1] == 2
