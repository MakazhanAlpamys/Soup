"""``digest_file`` must open with ``O_NOFOLLOW``, not plain ``open``.

``enforce_under_cwd_and_no_symlink`` is an ``lstat`` check. Between that check
and the read, a symlink can be swapped in at the same path and the digest is
then taken of whatever the link points at — the TOCTOU window the MCP plan /
execute split exists to close (a model swapped after the plan was approved).
The sibling reader ``mcp_server/registry.py::_read_text_under_cwd`` already
opens with ``os.O_RDONLY | os.O_NOFOLLOW`` for exactly this reason.

Windows has no ``os.O_NOFOLLOW``, so there the lstat check is all there is and
the behaviour is unchanged; the flag test below asserts the real flag where the
platform has it and the fallback where it does not.
"""

from __future__ import annotations

import hashlib
import os
import sys

import pytest

from soup_cli.mcp_server import execution as execution_mod
from soup_cli.mcp_server.execution import ExecutionError, digest_file

HAS_NOFOLLOW = hasattr(os, "O_NOFOLLOW")


@pytest.fixture()
def cwd(tmp_path, monkeypatch):
    real = tmp_path / "work"
    real.mkdir()
    monkeypatch.chdir(real)
    return real


def _spy_on_os_open(monkeypatch) -> list[int]:
    """Record the flags every ``os.open`` inside digest_file is called with."""
    seen: list[int] = []
    real_open = os.open

    def _recording_open(path, flags, *args, **kwargs):
        seen.append(flags)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(execution_mod.os, "open", _recording_open)
    return seen


class TestTheOpenIsNoFollow:
    def test_single_file_branch_opens_through_os_open(self, cwd, monkeypatch):
        (cwd / "model.bin").write_bytes(b"weights")
        seen = _spy_on_os_open(monkeypatch)

        digest_file("model.bin", "model")

        assert seen, "digest_file must open through os.open, not builtins.open"

    def test_tree_branch_opens_through_os_open(self, cwd, monkeypatch):
        (cwd / "tree").mkdir()
        (cwd / "tree" / "a.bin").write_bytes(b"a")
        (cwd / "tree" / "b.bin").write_bytes(b"b")
        seen = _spy_on_os_open(monkeypatch)

        digest_file("tree", "model")

        assert len(seen) == 2, f"expected one open per file in the tree, got {seen}"

    @pytest.mark.skipif(not HAS_NOFOLLOW, reason="platform has no O_NOFOLLOW")
    def test_flags_carry_o_nofollow(self, cwd, monkeypatch):
        (cwd / "model.bin").write_bytes(b"weights")
        (cwd / "tree").mkdir()
        (cwd / "tree" / "a.bin").write_bytes(b"a")
        seen = _spy_on_os_open(monkeypatch)

        digest_file("model.bin", "model")
        digest_file("tree", "model")

        assert seen
        for flags in seen:
            assert flags & os.O_NOFOLLOW, f"flags {flags:#o} lack O_NOFOLLOW"

    @pytest.mark.skipif(HAS_NOFOLLOW, reason="fallback path is Windows-only")
    def test_windows_falls_back_to_plain_read_only(self, cwd, monkeypatch):
        """No O_NOFOLLOW here — the open still happens, just without it."""
        (cwd / "model.bin").write_bytes(b"weights")
        seen = _spy_on_os_open(monkeypatch)

        digest_file("model.bin", "model")

        assert seen
        for flags in seen:
            assert flags & os.O_BINARY, "digests must not go through CRLF translation"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlink semantics")
class TestSymlinkRefusedAtReadTime:
    """Simulate the TOCTOU window: the lstat guard passed, then the path became
    a symlink. With the guard neutralised, the open itself must hold the line."""

    def _disable_the_lstat_guard(self, monkeypatch):
        monkeypatch.setattr(
            execution_mod, "enforce_under_cwd_and_no_symlink",
            lambda path, field: None,
        )

    def test_single_file_symlink_is_refused(self, cwd, monkeypatch):
        (cwd / "real.bin").write_bytes(b"weights")
        os.symlink(str(cwd / "real.bin"), str(cwd / "swapped.bin"))
        self._disable_the_lstat_guard(monkeypatch)

        with pytest.raises(ExecutionError, match="unavailable for execution"):
            digest_file("swapped.bin", "model")

    def test_tree_member_symlink_is_refused(self, cwd, monkeypatch):
        (cwd / "real.bin").write_bytes(b"weights")
        (cwd / "tree").mkdir()
        os.symlink(str(cwd / "real.bin"), str(cwd / "tree" / "member.bin"))
        self._disable_the_lstat_guard(monkeypatch)

        with pytest.raises(ExecutionError, match="unavailable for execution"):
            digest_file("tree", "model")


class TestTheNormalPathIsUnchanged:
    def test_single_file_digest_is_the_sha256_of_the_bytes(self, cwd):
        (cwd / "model.bin").write_bytes(b"weights" * 100)

        result = digest_file("model.bin", "model")

        assert result.digest == hashlib.sha256(b"weights" * 100).hexdigest()
        assert result.path == os.path.realpath(str(cwd / "model.bin"))

    def test_tree_digest_is_stable_and_order_independent(self, cwd):
        (cwd / "tree").mkdir()
        (cwd / "tree" / "b.bin").write_bytes(b"bbb")
        (cwd / "tree" / "a.bin").write_bytes(b"aaa")
        (cwd / "tree" / "nested").mkdir()
        (cwd / "tree" / "nested" / "c.bin").write_bytes(b"ccc")

        first = digest_file("tree", "model")
        second = digest_file("tree", "model")

        assert first.digest == second.digest
        assert len(first.digest) == 64

    def test_tree_digest_changes_when_content_changes(self, cwd):
        (cwd / "tree").mkdir()
        (cwd / "tree" / "a.bin").write_bytes(b"aaa")
        before = digest_file("tree", "model").digest
        (cwd / "tree" / "a.bin").write_bytes(b"zzz")

        assert digest_file("tree", "model").digest != before

    def test_byte_cap_still_fires(self, cwd):
        (cwd / "model.bin").write_bytes(b"x" * 100)

        with pytest.raises(ExecutionError, match="maximum byte limit"):
            digest_file("model.bin", "model", max_bytes=10)

    def test_tree_byte_cap_still_fires(self, cwd):
        (cwd / "tree").mkdir()
        (cwd / "tree" / "a.bin").write_bytes(b"x" * 100)

        with pytest.raises(ExecutionError, match="maximum byte limit"):
            digest_file("tree", "model", max_bytes=10)

    def test_file_count_cap_still_fires(self, cwd):
        (cwd / "tree").mkdir()
        for index in range(5):
            (cwd / "tree" / f"{index}.bin").write_bytes(b"x")

        with pytest.raises(ExecutionError, match="maximum file count"):
            digest_file("tree", "model", max_files=2)

    def test_outside_cwd_still_refused(self, cwd, tmp_path):
        outside = tmp_path / "outside.bin"
        outside.write_bytes(b"x")

        with pytest.raises(ExecutionError):
            digest_file(str(outside), "model")
