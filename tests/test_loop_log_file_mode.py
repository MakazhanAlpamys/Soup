"""The loop's JSONL logs are created owner-only (0600) on POSIX.

``append_json_line`` created every loop log with mode 0644: ``.soup-loops/iterations.jsonl``,
``versions.jsonl``, ``progress.jsonl``, the holdout log and ``paraphrases.jsonl``, which holds
rewrites of the user's prompts. The serve trace and feedback logs, the PID file and the daemon log
were already 0600. Now they are created 0600, and a log that already exists with wider bits is
brought down to 0600 the next time it is opened for an append (by a file's owner; another user's
file is left alone). On Windows the mode is not set and the folder's permissions apply, so the
mode tests are POSIX only; the call and the tightening step are still checked there.
"""

from __future__ import annotations

import os
import stat

import pytest

import soup_cli.utils.loop_log as loop_log
from soup_cli.utils.loop_log import LoopLog, append_json_line, progress_line

POSIX_ONLY = pytest.mark.skipif(
    os.name == "nt", reason="POSIX permission bits: on Windows the folder's permissions apply"
)


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def test_the_file_is_opened_with_mode_0600(cwd, monkeypatch):
    """Every platform: the mode asked of the OS is owner-only (the bits only act on POSIX)."""
    seen = []
    real_open = loop_log.open_no_follow

    def spy(path, flags, mode=0o600, **kwargs):
        seen.append(mode)
        return real_open(path, flags, mode, **kwargs)

    monkeypatch.setattr(loop_log, "open_no_follow", spy)

    append_json_line(".soup-loops/x.jsonl", {"a": 1})

    assert seen == [0o600]


@POSIX_ONLY
def test_a_new_log_is_created_0600(cwd):
    append_json_line(".soup-loops/paraphrases.jsonl", {"a": 1})

    assert _mode(".soup-loops/paraphrases.jsonl") == 0o600


@POSIX_ONLY
def test_each_loop_log_is_created_0600(cwd):
    log = LoopLog(".soup-loops")

    log.progress(progress_line("iter-1", "harvest", "start"))
    log.iteration({"iteration_id": "iter-1", "outcome": "skipped", "schema_version": 1})
    log.version({"to_version": 1, "cause": "initial", "schema_version": 1})

    for name in ("progress.jsonl", "iterations.jsonl", "versions.jsonl"):
        assert _mode(f".soup-loops/{name}") == 0o600, name


@POSIX_ONLY
def test_a_log_that_is_already_wider_is_tightened_on_the_next_append(cwd):
    path = cwd / ".soup-loops" / "old.jsonl"
    path.parent.mkdir()
    path.write_text('{"a": 0}\n', encoding="utf-8")
    os.chmod(path, 0o644)

    append_json_line(str(path), {"a": 1})

    assert _mode(path) == 0o600
    assert path.read_text(encoding="utf-8").splitlines() == ['{"a": 0}', '{"a": 1}']  # kept


@POSIX_ONLY
def test_a_log_that_is_already_0600_is_not_touched(cwd, monkeypatch):
    path = cwd / "owner.jsonl"
    path.write_text("", encoding="utf-8")
    os.chmod(path, 0o600)
    calls = []
    monkeypatch.setattr(os, "fchmod", lambda *args: calls.append(args))

    append_json_line(str(path), {"a": 1})

    assert calls == []


def test_every_append_goes_through_the_tightening_step(cwd, monkeypatch):
    """Every platform: the step runs on the opened file before each append."""
    seen = []
    monkeypatch.setattr(loop_log, "_tighten_to_owner", lambda fd, **kwargs: seen.append(fd))

    append_json_line(".soup-loops/a.jsonl", {"a": 1})
    append_json_line(".soup-loops/a.jsonl", {"a": 2})

    assert len(seen) == 2 and all(isinstance(fd, int) for fd in seen)


def test_the_tightening_step_lowers_wider_bits_where_chmod_exists(cwd, monkeypatch):
    """Every platform: the step itself, with ``fchmod`` replaced (Windows has none to call)."""
    path = cwd / "f.jsonl"
    path.write_text("", encoding="utf-8")
    calls = []
    monkeypatch.setattr(os, "fchmod", lambda fd, mode: calls.append(mode), raising=False)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    real_fstat = os.fstat

    class WideMode:
        st_mode = stat.S_IFREG | 0o644

    monkeypatch.setattr(os, "fstat", lambda value: WideMode() if value == fd else real_fstat(value))
    try:
        loop_log._tighten_to_owner(fd, posix=True)
        loop_log._tighten_to_owner(fd, posix=False)
    finally:
        os.close(fd)

    assert calls == [0o600]  # only the POSIX call acts


def test_a_file_that_is_not_ours_to_change_is_left_alone(cwd, monkeypatch):
    """``fchmod`` refused (another user's file): the append goes on, nothing is raised."""
    path = cwd / "theirs.jsonl"
    path.write_text("", encoding="utf-8")

    def refuse(fd, mode):
        raise PermissionError(1, "not permitted")

    monkeypatch.setattr(os, "fchmod", refuse, raising=False)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))

    class WideMode:
        st_mode = stat.S_IFREG | 0o666

    real_fstat = os.fstat
    monkeypatch.setattr(os, "fstat", lambda value: WideMode() if value == fd else real_fstat(value))
    try:
        loop_log._tighten_to_owner(fd, posix=True)  # no exception
    finally:
        os.close(fd)
