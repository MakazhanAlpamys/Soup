"""``digest_file`` must open with ``O_NOFOLLOW``, not plain ``open``.

``enforce_under_cwd_and_no_symlink`` is an ``lstat`` check. Between that check
and the read, a symlink can be swapped in at the same path and the digest is
then taken of whatever the link points at — the TOCTOU window the MCP plan /
execute split exists to close (a model swapped after the plan was approved).
The sibling reader ``mcp_server/registry.py::_read_text_under_cwd`` already
opens with ``os.O_RDONLY | os.O_NOFOLLOW`` for exactly this reason.

The open goes through ``soup_cli.utils.paths.open_no_follow`` (#820), which
refuses a symlink in its own pre-open ``lstat`` and passes ``O_NOFOLLOW`` where
the platform has one. Windows has no ``os.O_NOFOLLOW``; there the helper's
``lstat`` / reparse-point check and post-open ``fstat`` cross-check refuse
instead, so the symlink tests below are gated on being able to CREATE a symlink
(``requires_symlink``), not on POSIX. The flag test asserts the real flag where
the platform has it and the fallback where it does not.

Passing the flag is only half of it: the single-file branch used to open
``os.path.realpath(path)``, i.e. a path with the symlink ALREADY RESOLVED, so
``O_NOFOLLOW`` had nothing left to refuse and the swapped-in link's target was
digested. It opens the path as given now, and the recorded ``ProtectedFile.path``
stays the resolved one, because that is what ``_revalidate`` compares.
"""

from __future__ import annotations

import errno
import hashlib
import os

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


def _record_os_open_attempts(monkeypatch) -> list[dict]:
    """Record the path of every ``os.open`` and whether it yielded a descriptor."""
    attempts: list[dict] = []
    real_open = os.open

    def _recording_open(path, flags, *args, **kwargs):
        attempt: dict = {"path": os.fspath(path), "fd": None, "error": None}
        attempts.append(attempt)
        try:
            fd = real_open(path, flags, *args, **kwargs)
        except OSError as exc:
            attempt["error"] = exc
            raise
        attempt["fd"] = fd
        return fd

    monkeypatch.setattr(execution_mod.os, "open", _recording_open)
    return attempts


def _record_no_follow_open_attempts(monkeypatch) -> list[dict]:
    """Record every call ``digest_file`` makes to the no-follow open helper.

    ``digest_file`` opens through ``soup_cli.utils.paths.open_no_follow`` (#820),
    which refuses a symlink in its own pre-open ``lstat``, BEFORE ``os.open`` is
    reached. So for a symlink the open ATTEMPT is the helper call: a spy on
    ``os.open`` records nothing, which is what reddened this test on POSIX when
    it was ported from the release branch onto a ``main`` that opens through
    the helper (#1126).
    """
    attempts: list[dict] = []
    real_open_no_follow = execution_mod.open_no_follow

    def _recording_open_no_follow(path, flags, *args, **kwargs):
        attempt: dict = {"path": os.fspath(path), "fd": None, "error": None}
        attempts.append(attempt)
        try:
            fd = real_open_no_follow(path, flags, *args, **kwargs)
        except OSError as exc:
            attempt["error"] = exc
            raise
        attempt["fd"] = fd
        return fd

    monkeypatch.setattr(execution_mod, "open_no_follow", _recording_open_no_follow)
    return attempts


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


class TestTheSingleFileOpenUsesThePathAsGiven:
    """Runs everywhere, including on an account that cannot create a symlink.

    The regression this pins was invisible to the POSIX-only symlink tests on
    the one platform the maintainer develops on, which is how it reached CI:
    ``O_NOFOLLOW`` was passed, but to a path ``os.path.realpath`` had already
    resolved, so the flag had nothing left to refuse. Checking WHICH path is
    opened needs no symlink and therefore no platform.
    """

    def test_the_opened_path_is_not_the_resolved_one(self, cwd, monkeypatch):
        (cwd / "model.bin").write_bytes(b"weights")
        given = os.path.join(".", "model.bin")
        resolved = os.path.realpath(str(cwd / "model.bin"))
        assert given != resolved, "the fixture must make the two spellings differ"
        attempts = _record_os_open_attempts(monkeypatch)

        result = digest_file(given, "model")

        assert [attempt["path"] for attempt in attempts] == [given], (
            "digest_file must open the path as given, not its realpath"
        )
        # The RESOLVED path is still what is recorded: _revalidate compares it.
        assert result.path == resolved
        assert result.digest == hashlib.sha256(b"weights").hexdigest()


class TestSymlinkRefusedAtReadTime:
    """Simulate the TOCTOU window: the lstat guard passed, then the path became
    a symlink. With the guard neutralised, the open itself must hold the line.

    Gated per test on the capability to create a symlink, not on the platform:
    ``open_no_follow`` refuses on Windows too, so these assertions hold there,
    and the plain-file control needs no symlink at all.
    """

    def _disable_the_lstat_guard(self, monkeypatch):
        monkeypatch.setattr(
            execution_mod, "enforce_under_cwd_and_no_symlink",
            lambda path, field: None,
        )

    @pytest.mark.requires_symlink
    def test_single_file_symlink_is_refused(self, cwd, monkeypatch):
        (cwd / "real.bin").write_bytes(b"weights")
        os.symlink(str(cwd / "real.bin"), str(cwd / "swapped.bin"))
        self._disable_the_lstat_guard(monkeypatch)

        with pytest.raises(ExecutionError, match="unavailable for execution"):
            digest_file("swapped.bin", "model")

    @pytest.mark.requires_symlink
    def test_tree_member_symlink_is_refused(self, cwd, monkeypatch):
        """Two layers can refuse here; the test pins the refusal, not the layer.

        The walk ``lstat``s every member and rejects anything that is not a
        regular file, so on the shipped code that per-member ``S_ISREG`` check
        is what fires first — before the member is ever opened — and the
        message is ``<field> contains non-regular file: <path>``. Were that
        check removed, ``_open_binary_no_follow`` (``open_no_follow``) would still
        refuse, with the path-free ``unavailable for execution``. Either is the
        property under test; matching only one of them turns a defence in depth
        into a brittle assertion about which layer got there first.
        """
        (cwd / "real.bin").write_bytes(b"weights")
        (cwd / "tree").mkdir()
        os.symlink(str(cwd / "real.bin"), str(cwd / "tree" / "member.bin"))
        self._disable_the_lstat_guard(monkeypatch)

        with pytest.raises(
            ExecutionError, match="non-regular file|unavailable for execution"
        ):
            digest_file("tree", "model")

    @pytest.mark.requires_symlink
    def test_single_file_symlink_is_refused_before_the_target_is_read(
        self, cwd, monkeypatch
    ):
        """The refusal must come from the open, with no byte of the target read.

        ``digest_file`` used to open ``os.path.realpath(path)`` — a path with
        the symlink already RESOLVED — so ``O_NOFOLLOW`` could never fire and
        the swapped-in link's target was digested happily. Opening the path as
        given is what makes the refusal reachable, so this pins both halves: the
        refusal, and that it happens before any read.

        The open attempt is counted where the open now happens, at
        ``open_no_follow``: it refuses in a pre-open ``lstat``, so ``os.open``
        is never reached for a symlink. ``os.open`` is still watched for the
        other half: no descriptor on the link or its target may come out of the
        call, which is what a read of the target would need first.
        """
        target = cwd / "real.bin"
        target.write_bytes(b"weights")
        os.symlink(str(target), str(cwd / "swapped.bin"))
        self._disable_the_lstat_guard(monkeypatch)
        raw_opens = _record_os_open_attempts(monkeypatch)
        attempts = _record_no_follow_open_attempts(monkeypatch)

        with pytest.raises(ExecutionError, match="unavailable for execution") as excinfo:
            digest_file("swapped.bin", "model")

        assert len(attempts) == 1, (
            f"expected exactly one open attempt, through open_no_follow, got {attempts}"
        )
        only = attempts[0]
        assert os.path.basename(only["path"]) == "swapped.bin", (
            "digest_file must open the path AS GIVEN; opening the resolved "
            f"target defeats the no-follow refusal (opened {only['path']!r})"
        )
        assert only["fd"] is None and isinstance(only["error"], OSError), (
            "the open must fail, so not a byte of the symlink's target is read"
        )
        assert only["error"].errno == errno.ELOOP, (
            f"the open must fail BECAUSE the path is a symlink, got {only['error']!r}"
        )
        descriptors = [
            attempt for attempt in raw_opens
            if attempt["fd"] is not None
            and os.path.basename(attempt["path"]) in {"swapped.bin", "real.bin"}
        ]
        assert descriptors == [], (
            f"no descriptor on the link or its target may be opened, got {descriptors}"
        )
        # Nothing about the target leaks out: no digest was produced at all, and
        # the message names no path.
        message = str(excinfo.value)
        assert hashlib.sha256(b"weights").hexdigest() not in message
        assert "real.bin" not in message and "swapped.bin" not in message

    def test_a_plain_file_still_digests_under_the_same_conditions(self, cwd, monkeypatch):
        """Control for the test above: the refusal is the symlink, not the setup."""
        (cwd / "plain.bin").write_bytes(b"weights")
        self._disable_the_lstat_guard(monkeypatch)

        result = digest_file("plain.bin", "model")

        assert result.digest == hashlib.sha256(b"weights").hexdigest()
        assert result.path == os.path.realpath(str(cwd / "plain.bin"))


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
