"""``open_private_append`` does not follow a link: the serve trace and feedback logs.

The two writers realpath their log once, when they are built, and then appended through
``open_private_append``, which opened the path with no ``O_NOFOLLOW``: a link put at the path
later (or at a parent directory) by someone with write access to that directory sent the JSON
lines to any file the server's user can write. It now opens through ``open_no_follow``: a symlink
or a Windows reparse point at the path, a linked parent directory under the working directory and
a hard link are refused by name (``OSError``: ``ELOOP`` or ``EMLINK``). The trace log is passive
and drops the line; the feedback log reports the failure (the endpoint answers 500, never ok).
"""

from __future__ import annotations

import errno
import os
import uuid

import pytest

import soup_cli.monitoring.trace_logger as trace_logger
from soup_cli.monitoring.feedback_log import FeedbackLogWriter
from soup_cli.monitoring.trace_logger import TraceLogWriter, open_private_append


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_it_opens_through_the_no_follow_helper_with_every_guard(cwd, monkeypatch):
    """Every platform: the helper that refuses links is the one that opens the file."""
    seen = []
    real = trace_logger.open_no_follow

    def spy(path, flags, mode=0o600, **kwargs):
        seen.append((mode, kwargs))
        return real(path, flags, mode, **kwargs)

    monkeypatch.setattr(trace_logger, "open_no_follow", spy)

    with open_private_append(cwd / "t.log") as handle:
        handle.write(b"x")

    assert seen == [(0o600, {"check_parent": True, "refuse_hardlink": True})]


def test_an_ordinary_file_is_created_appended_to_and_kept(cwd):
    with open_private_append(cwd / "t.log") as handle:
        handle.write(b"one\n")
    with open_private_append(cwd / "t.log") as handle:
        handle.write(b"two\n")

    assert (cwd / "t.log").read_bytes() == b"one\ntwo\n"


@pytest.mark.requires_symlink
def test_a_symlink_at_the_path_is_refused_by_name_and_not_written_through(cwd):
    target = cwd / "victim.txt"
    target.write_text("keep", encoding="utf-8")
    os.symlink(target, cwd / "t.log")

    with pytest.raises(OSError) as excinfo:
        open_private_append(cwd / "t.log")

    assert excinfo.value.errno == errno.ELOOP
    assert "t.log" in str(excinfo.value)
    assert target.read_text(encoding="utf-8") == "keep"


@pytest.mark.requires_symlink
def test_a_linked_parent_directory_is_refused(cwd):
    (cwd / "real").mkdir()
    os.symlink(cwd / "real", cwd / "linked", target_is_directory=True)

    with pytest.raises(OSError) as excinfo:
        open_private_append(cwd / "linked" / "t.log")

    assert excinfo.value.errno == errno.ELOOP
    assert not (cwd / "real" / "t.log").exists()


def test_a_hard_link_is_refused(cwd):
    (cwd / "victim.txt").write_text("keep", encoding="utf-8")
    try:
        os.link(cwd / "victim.txt", cwd / "t.log")
    except (OSError, NotImplementedError, AttributeError):
        pytest.skip("this file system has no hard links")

    with pytest.raises(OSError) as excinfo:
        open_private_append(cwd / "t.log")

    assert excinfo.value.errno == errno.EMLINK
    assert (cwd / "victim.txt").read_text(encoding="utf-8") == "keep"


@pytest.mark.requires_symlink
def test_the_trace_writer_drops_the_line_when_the_log_was_swapped_for_a_link(cwd):
    writer = TraceLogWriter(str(cwd / "trace.jsonl"))
    victim = cwd / "victim.txt"
    victim.write_text("keep", encoding="utf-8")
    os.symlink(victim, cwd / "trace.jsonl")  # swapped in after the writer was built

    writer.record_conversation(
        response_id=str(uuid.uuid4()), model="m",
        messages=[{"role": "user", "content": "hi"}], response="ok",
    )

    assert victim.read_text(encoding="utf-8") == "keep"


@pytest.mark.requires_symlink
def test_the_feedback_writer_reports_a_log_that_was_swapped_for_a_link(cwd):
    writer = FeedbackLogWriter(str(cwd / "feedback.jsonl"))
    victim = cwd / "victim.txt"
    victim.write_text("keep", encoding="utf-8")
    os.symlink(victim, cwd / "feedback.jsonl")

    with pytest.raises(OSError):
        writer.append({"kind": "thumb"})

    assert victim.read_text(encoding="utf-8") == "keep"
