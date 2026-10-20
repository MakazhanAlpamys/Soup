"""Audit lines written by older versions get the same URL masking as new ones.

``cli._emit_audit_event`` masks option values by option name before a line is
stored; lines stored by older versions still carry them as typed. The writer
(``append_audit_event(..., arg_mask=...)``) applies the same masking to the
stored lines once, guarded by a marker file next to the log. A pass that fails is
not repeated by every command (a record beside the log makes it wait a day), and
temp files a killed pass left behind are cleared by the next one.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import stat
import tempfile
import time
import tracemalloc
import types
from datetime import datetime, timedelta, timezone

import pytest

from soup_cli.utils import audit_log
from soup_cli.utils.argv_redaction import ArgMasker
from soup_cli.utils.audit_log import AuditEvent, append_audit_event, mask_older_lines

URL = "https://hooks.slack.com/services/T0000000/B0000000/abcdefghijklmnopqrstuvwx"
DISCORD_URL = "https://discord.com/api/webhooks/123456789/abcdefghijklmnopqrstuvwxyz"
REDACTED = "<redacted>"


@functools.lru_cache(maxsize=None)
def _chmod_is_honoured() -> bool:
    """Whether this platform stores POSIX permission bits (Windows keeps only read-only)."""
    with tempfile.TemporaryDirectory() as tmp:
        probe = os.path.join(tmp, "probe")
        with open(probe, "w", encoding="utf-8"):
            pass
        os.chmod(probe, 0o640)
        return stat.S_IMODE(os.stat(probe).st_mode) == 0o640


@functools.lru_cache(maxsize=None)
def _open_handle_blocks_replace() -> bool:
    """Whether replacing a file fails while another handle has it open (Windows)."""
    with tempfile.TemporaryDirectory() as tmp:
        held = os.path.join(tmp, "held")
        other = os.path.join(tmp, "other")
        for name in (held, other):
            with open(name, "w", encoding="utf-8"):
                pass
        with open(held, "rb"):
            try:
                os.replace(other, held)
            except OSError:
                return True
    return False


def _record(command: str, args: list[str], exit_code: int = 0) -> dict:
    return {
        "timestamp": "2026-10-01T08:00:00+00:00",
        "command": command,
        "args": list(args),
        "exit_code": exit_code,
        "host_id": "box",
        "operator_id": "me",
    }


def _stored(command: str, args: list[str]) -> bytes:
    """One line exactly as the writer stores it."""
    return (json.dumps(_record(command, args)) + "\n").encode("utf-8")


def _event() -> AuditEvent:
    return AuditEvent(
        timestamp="2026-10-20T00:00:00+00:00",
        command="version",
        args=("--verbose",),
        exit_code=0,
        host_id="box",
        operator_id="me",
    )


@pytest.fixture
def mask():
    import soup_cli.cli as cli

    return cli._audit_arg_masker()


@pytest.fixture
def log(tmp_path):
    return tmp_path / "audit.jsonl"


def _marker(log):
    return str(log) + ".masked-v1"


def _leftovers(log):
    """Temp files the pass may have left next to the log."""
    return [name for name in os.listdir(log.parent) if name.endswith(".tmp")]


@pytest.fixture
def replace_calls(monkeypatch):
    """Every ``os.replace`` made while the test runs (it still goes through)."""
    calls = []
    real = os.replace

    def recording(source, target, *args, **kwargs):
        calls.append((os.fspath(source), os.fspath(target)))
        return real(source, target, *args, **kwargs)

    monkeypatch.setattr(os, "replace", recording)
    return calls


_DAY = 24 * 60 * 60
_HOUR = 60 * 60


class _Clock:
    """A settable stand-in for ``audit_log._utc_now``."""

    def __init__(self):
        self.now = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += timedelta(seconds=seconds)


@pytest.fixture
def clock(monkeypatch):
    fake = _Clock()
    monkeypatch.setattr(audit_log, "_utc_now", fake)
    return fake


def _tried(log):
    """The record a failed pass leaves beside the log."""
    return str(log) + ".masked-v1.tried"


def _tried_text(log):
    with open(_tried(log), encoding="ascii") as fh:
        return fh.read().strip()


@pytest.fixture
def scans(monkeypatch):
    """The files the pass has read through so far (it still runs); empty if it never ran."""
    seen = []
    real = audit_log._mask_file_lines
    monkeypatch.setattr(
        audit_log, "_mask_file_lines", lambda *args: seen.append(args[0]) or real(*args)
    )
    return seen


class _ReplaceRefused:
    """``os.replace`` fails, as it does for a log another process holds open."""

    def __init__(self, monkeypatch):
        self._monkeypatch = monkeypatch
        self._real = os.replace
        monkeypatch.setattr(os, "replace", self._refuse)

    @staticmethod
    def _refuse(*_args, **_kwargs):
        raise PermissionError("the file is in use")

    def lift(self):
        self._monkeypatch.setattr(os, "replace", self._real)


@pytest.fixture
def replace_refused(monkeypatch):
    return _ReplaceRefused(monkeypatch)


# ---------------------------------------------------------------------------
# The pass masks what the writer would have masked
# ---------------------------------------------------------------------------


def test_old_line_is_masked_after_the_next_write(log, mask):
    log.write_bytes(_stored("train", ["--config", "soup.yaml", "--slack-url", URL]))

    append_audit_event(_event(), str(log), arg_mask=mask)

    lines = log.read_bytes().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["args"] == ["--config", "soup.yaml", "--slack-url", REDACTED]
    assert json.loads(lines[1])["command"] == "version"
    assert URL.encode() not in log.read_bytes()


@pytest.mark.parametrize(
    "args, expected",
    [
        (["--slack-url", URL], ["--slack-url", REDACTED]),
        ([f"--slack-url={URL}"], [f"--slack-url={REDACTED}"]),
        (["--discord-url", DISCORD_URL], ["--discord-url", REDACTED]),
        ([f"--discord-url={DISCORD_URL}"], [f"--discord-url={REDACTED}"]),
        (
            ["--config", "a.yaml", "--slack-url", URL, "--discord-url", DISCORD_URL, "--fast"],
            ["--config", "a.yaml", "--slack-url", REDACTED, "--discord-url", REDACTED, "--fast"],
        ),
    ],
)
def test_every_spelling_of_a_webhook_option_is_masked(log, mask, args, expected):
    log.write_bytes(_stored("train", args))

    assert mask_older_lines(str(log), mask) is True

    assert json.loads(log.read_bytes())["args"] == expected


def test_the_cli_writer_masks_older_lines(tmp_path, monkeypatch):
    """``soup <anything>`` is enough: ``_emit_audit_event`` hands its masking to the writer."""
    import soup_cli.cli as cli

    path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("SOUP_AUDIT_LOG_PATH", str(path))
    monkeypatch.delenv("SOUP_NO_AUDIT_LOG", raising=False)
    monkeypatch.setattr(cli, "_audit_disabled", False, raising=False)
    path.write_bytes(_stored("train", ["--slack-url", URL]))

    cli._emit_audit_event(["soup", "version"], 0)

    text = path.read_text(encoding="utf-8")
    assert URL not in text
    assert [json.loads(line)["command"] for line in text.splitlines()] == ["train", "version"]
    assert os.path.exists(_marker(path))


def test_opting_out_of_the_audit_log_leaves_the_file_alone(tmp_path, monkeypatch):
    import soup_cli.cli as cli

    path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("SOUP_AUDIT_LOG_PATH", str(path))
    monkeypatch.setattr(cli, "_audit_disabled", True, raising=False)
    before = _stored("train", ["--slack-url", URL])
    path.write_bytes(before)

    cli._emit_audit_event(["soup", "version"], 0)

    assert path.read_bytes() == before
    assert not os.path.exists(_marker(path))


def test_rotated_backup_is_masked_too(log, mask):
    backup = str(log) + ".1"
    with open(backup, "wb") as fh:
        fh.write(_stored("train", ["--slack-url", URL]))
    log.write_bytes(_stored("train", ["--discord-url", DISCORD_URL]))

    append_audit_event(_event(), str(log), arg_mask=mask)

    assert URL.encode() not in open(backup, "rb").read()
    assert DISCORD_URL.encode() not in log.read_bytes()


def test_masking_is_the_one_new_lines_get(log, mask):
    """The pass reuses ``cli._mask_audit_args``; both agree on every line."""
    import soup_cli.cli as cli

    cases = [
        ("train", ["--slack-url", URL, "--config", "a.yaml"]),
        ("serve", ["--api-key", "VALUE-ONE", "--tool-auth-token=VALUE-TWO", "--max-tokens", "9"]),
        ("push", ["--repo", "u/m", "-t", "VALUE"]),
        ("no-such-command", ["--auth-token", "VALUE"]),
    ]
    log.write_bytes(b"".join(_stored(command, args) for command, args in cases))

    mask_older_lines(str(log), mask)

    stored = [json.loads(line)["args"] for line in log.read_bytes().splitlines()]
    assert stored == [list(cli._mask_audit_args(command, args)) for command, args in cases]
    assert stored[0] == ["--slack-url", REDACTED, "--config", "a.yaml"]


def test_each_stored_line_is_resolved_by_its_own_command(log, mask):
    """`-t` is a credential option of some commands only; `train -t sft` has none."""
    before = _stored("train", ["-t", "sft"])
    log.write_bytes(before)

    mask_older_lines(str(log), mask)

    assert log.read_bytes() == before


def test_a_switch_before_a_webhook_option_is_not_taken_for_its_value(log, mask):
    """`--slack-url` typed with no value: the option after it is still an option."""
    before = _stored("train", ["--slack-url", "--config", "a.yaml"])
    log.write_bytes(before)

    mask_older_lines(str(log), mask)

    assert log.read_bytes() == before


# ---------------------------------------------------------------------------
# Once only
# ---------------------------------------------------------------------------


def test_a_marker_is_written_next_to_the_log(log, mask):
    log.write_bytes(_stored("train", ["--slack-url", URL]))

    append_audit_event(_event(), str(log), arg_mask=mask)

    assert os.path.isfile(str(log) + ".masked-v1")


def test_a_second_write_does_not_rewrite_the_file(log, mask, replace_calls, monkeypatch):
    log.write_bytes(_stored("train", ["--slack-url", URL]))
    append_audit_event(_event(), str(log), arg_mask=mask)
    assert len(replace_calls) == 1  # the first write rewrote it

    # A line stored by another, older process after the first pass. Had the
    # pass run again it would be masked; with the marker it is left as it is.
    late = _stored("train", ["--slack-url", URL])
    with open(log, "ab") as fh:
        fh.write(late)
    scanned = []
    real_scan = audit_log._mask_file_lines
    monkeypatch.setattr(
        audit_log, "_mask_file_lines", lambda *args: scanned.append(args) or real_scan(*args)
    )

    append_audit_event(_event(), str(log), arg_mask=mask)

    assert scanned == []
    assert len(replace_calls) == 1
    assert log.read_bytes().count(late) == 1


def test_no_pass_without_a_mask(log):
    """Callers that give no ``arg_mask`` (the library default) see no change at all."""
    before = _stored("train", ["--slack-url", URL])
    log.write_bytes(before)

    append_audit_event(_event(), str(log))

    assert log.read_bytes().startswith(before)
    assert not os.path.exists(_marker(log))


def test_a_new_log_gets_a_marker_and_is_not_touched(log, mask):
    assert mask_older_lines(str(log), mask) is True

    assert not log.exists()
    assert os.path.isfile(_marker(log))


def test_a_log_already_masked_is_not_rewritten(log, mask, replace_calls):
    original = _stored("train", ["--slack-url", REDACTED]) + _stored("version", [])
    log.write_bytes(original)
    inode = os.stat(log).st_ino

    assert mask_older_lines(str(log), mask) is True

    assert replace_calls == []
    assert _leftovers(log) == []
    assert log.read_bytes() == original
    assert os.stat(log).st_ino == inode
    assert os.path.isfile(_marker(log))


def test_the_pass_is_idempotent(log, mask):
    log.write_bytes(_stored("train", ["--slack-url", URL]) + _stored("version", []))
    mask_older_lines(str(log), mask)
    once = log.read_bytes()
    os.unlink(_marker(log))

    assert mask_older_lines(str(log), mask) is True

    assert log.read_bytes() == once


# ---------------------------------------------------------------------------
# Everything else stays byte for byte
# ---------------------------------------------------------------------------


def test_only_the_args_of_a_changed_line_change(log, mask):
    changed = _record("train", ["--config", "soup.yaml", "--slack-url", URL], exit_code=3)
    changed["host_id"] = "höst"  # non-ASCII, stored escaped
    before = (json.dumps(changed) + "\n").encode("utf-8")
    log.write_bytes(before)

    mask_older_lines(str(log), mask)

    expected = dict(changed, args=["--config", "soup.yaml", "--slack-url", REDACTED])
    assert log.read_bytes() == (json.dumps(expected) + "\n").encode("utf-8")


def test_lines_without_a_change_stay_byte_for_byte(log, mask):
    # Spacing and key order that a parse-and-dump round trip would normalise.
    odd = b'{"exit_code": 0,   "args": ["--config",  "a.yaml"], "command":"train" ,"x":1.50}\n'
    plain = _stored("version", [])
    masked_already = _stored("train", ["--slack-url", REDACTED])
    changed = _stored("train", ["--slack-url", URL])
    log.write_bytes(odd + plain + changed + masked_already + odd)

    mask_older_lines(str(log), mask)

    expected = _stored("train", ["--slack-url", REDACTED])
    assert log.read_bytes() == odd + plain + expected + masked_already + odd


def test_line_order_is_kept(log, mask):
    commands = [f"cmd{index}" for index in range(60)]
    lines = []
    for index, command in enumerate(commands):
        args = ["--slack-url", URL] if index % 7 == 0 else ["--n", str(index)]
        lines.append(_stored(command, args))
    log.write_bytes(b"".join(lines))

    mask_older_lines(str(log), mask)

    stored = [json.loads(line) for line in log.read_bytes().splitlines()]
    assert [record["command"] for record in stored] == commands
    assert [record["args"][1] == REDACTED for record in stored[::7]] == [True] * len(stored[::7])
    assert URL.encode() not in log.read_bytes()


def test_malformed_lines_are_preserved(log, mask):
    junk = [
        b"not json at all\n",
        b"\n",
        b"   \n",
        b'{"command": "train", "args": ["--slack-url", "' + URL.encode() + b'"\n',  # cut off
        b'["--slack-url", "x"]\n',
        b'{"command": "train"}\n',  # no args
        b'{"command": 7, "args": ["--slack-url", "x"]}\n',
        b'{"command": "train", "args": "--slack-url"}\n',
        b'{"command": "train", "args": ["--slack-url", 5]}\n',  # a non-string argument
        b"\xff\xfe\x00 not utf-8\n",
        b"[" * 5000 + b"]" * 5000 + b"\n",  # valid JSON, deeper than the parser's recursion limit
    ]
    good = _stored("train", ["--slack-url", URL])
    log.write_bytes(b"".join(junk[:5]) + good + b"".join(junk[5:]) + good)

    mask_older_lines(str(log), mask)

    expected_good = _stored("train", ["--slack-url", REDACTED])
    expected = b"".join(junk[:5]) + expected_good + b"".join(junk[5:]) + expected_good
    assert log.read_bytes() == expected


def test_line_endings_and_a_missing_final_newline_are_kept(log, mask):
    first = (json.dumps(_record("train", ["--slack-url", URL])) + "\r\n").encode("utf-8")
    last = json.dumps(_record("train", ["--slack-url", URL])).encode("utf-8")  # no newline
    log.write_bytes(first + last)

    mask_older_lines(str(log), mask)

    expected = [_record("train", ["--slack-url", REDACTED])] * 2
    assert log.read_bytes() == (
        (json.dumps(expected[0]) + "\r\n").encode("utf-8") + json.dumps(expected[1]).encode("utf-8")
    )


def test_a_line_longer_than_the_read_limit_is_copied_through(log, mask, monkeypatch):
    monkeypatch.setattr(audit_log, "_MASK_MAX_LINE_BYTES", 256)
    short = _stored("train", ["--slack-url", URL])
    assert len(short) < 256
    huge = b"x" * 1000 + b"\n"
    just_over = b"y" * 255 + b"\n"
    # One line whose first 256 bytes are a complete record: read in pieces it would
    # parse, but as a line it is not JSON and must stay as it is.
    record = _record("train", ["--slack-url", URL])
    record["host_id"] = ""
    record["host_id"] = "h" * (256 - len(json.dumps(record)))
    first_piece = json.dumps(record).encode("utf-8")
    assert len(first_piece) == 256
    one_line = first_piece + b"garbage\n"
    # The tail of an over-long line is not a line of its own either.
    hidden = b"x" * 256 + short
    log.write_bytes(
        short + huge + just_over + one_line + hidden + short + b"z" * 600
    )  # last line has no newline

    mask_older_lines(str(log), mask)

    masked = _stored("train", ["--slack-url", REDACTED])
    assert log.read_bytes() == (
        masked + huge + just_over + one_line + hidden + masked + b"z" * 600
    )


def test_the_replacement_is_given_the_mode_of_the_original(log, mask, monkeypatch):
    """Platform-independent form of the mode test below: what is asked of ``chmod``."""
    log.write_bytes(_stored("train", ["--slack-url", URL]))
    os.chmod(log, 0o640)
    wanted = stat.S_IMODE(os.stat(log).st_mode)
    calls = []
    real_chmod = os.chmod

    def recording(path, mode, *args, **kwargs):
        calls.append((os.path.basename(os.fspath(path)), mode))
        return real_chmod(path, mode, *args, **kwargs)

    monkeypatch.setattr(os, "chmod", recording)

    mask_older_lines(str(log), mask)

    assert [mode for name, mode in calls if name.startswith(".audit-mask-")] == [wanted]


def test_the_replacement_is_written_beside_the_log_and_flushed_to_disk(log, mask, monkeypatch):
    """Same directory (so the swap is one rename on one volume); synced before the swap."""
    log.write_bytes(_stored("train", ["--slack-url", URL]))
    made = []
    synced = []
    real_mkstemp = tempfile.mkstemp
    real_fsync = os.fsync

    def recording_mkstemp(*args, **kwargs):
        made.append(kwargs.get("dir"))
        return real_mkstemp(*args, **kwargs)

    def recording_fsync(descriptor):
        synced.append(descriptor)
        return real_fsync(descriptor)

    monkeypatch.setattr(tempfile, "mkstemp", recording_mkstemp)
    monkeypatch.setattr(os, "fsync", recording_fsync)

    mask_older_lines(str(log), mask)

    assert [os.path.normcase(name) for name in made] == [os.path.normcase(str(log.parent))]
    assert len(synced) == 1


@pytest.mark.skipif(not _chmod_is_honoured(), reason="this platform has no POSIX permission bits")
def test_file_mode_is_preserved(log, mask):
    log.write_bytes(_stored("train", ["--slack-url", URL]))
    os.chmod(log, 0o640)

    mask_older_lines(str(log), mask)

    assert URL.encode() not in log.read_bytes()
    assert stat.S_IMODE(os.stat(log).st_mode) == 0o640


# ---------------------------------------------------------------------------
# Failures leave the file alone and never reach the command
# ---------------------------------------------------------------------------


def test_a_failing_mask_leaves_the_log_untouched_and_is_retried(log, mask, clock):
    before = _stored("version", []) + _stored("train", ["--slack-url", URL])
    log.write_bytes(before)

    def broken(_command, _args):
        raise RuntimeError("masking is unavailable")

    assert mask_older_lines(str(log), broken) is False

    assert log.read_bytes() == before
    assert not os.path.exists(_marker(log))
    assert os.path.isfile(_tried(log))
    assert _leftovers(log) == []

    clock.advance(_DAY)
    assert mask_older_lines(str(log), mask) is True
    assert URL.encode() not in log.read_bytes()
    assert os.path.exists(_marker(log))


def test_a_failing_pass_does_not_stop_the_write(log):
    before = _stored("train", ["--slack-url", URL])
    log.write_bytes(before)

    def broken(_command, _args):
        raise RuntimeError("masking is unavailable")

    append_audit_event(_event(), str(log), arg_mask=broken)

    lines = log.read_bytes().splitlines(keepends=True)
    assert lines[0] == before
    assert json.loads(lines[1])["command"] == "version"


def test_a_failed_replace_leaves_the_log_untouched_and_is_retried(
    log, mask, clock, replace_refused
):
    before = _stored("train", ["--slack-url", URL])
    log.write_bytes(before)

    assert mask_older_lines(str(log), mask) is False
    replace_refused.lift()

    assert log.read_bytes() == before
    assert not os.path.exists(_marker(log))
    assert _leftovers(log) == []

    clock.advance(_DAY)
    assert mask_older_lines(str(log), mask) is True
    assert URL.encode() not in log.read_bytes()


def test_a_read_only_file_is_left_alone_and_the_command_succeeds(
    tmp_path, monkeypatch, replace_calls
):
    import soup_cli.cli as cli

    path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("SOUP_AUDIT_LOG_PATH", str(path))
    monkeypatch.delenv("SOUP_NO_AUDIT_LOG", raising=False)
    monkeypatch.setattr(cli, "_audit_disabled", False, raising=False)
    before = _stored("train", ["--slack-url", URL])
    path.write_bytes(before)
    os.chmod(path, stat.S_IREAD)
    try:
        assert mask_older_lines(str(path), cli._audit_arg_masker()) is False
        # What a command does on its way out: it must not raise, whatever the file is.
        cli._emit_audit_event(["soup", "version"], 0)

        assert path.read_bytes() == before
        assert not os.path.exists(_marker(path))
        assert _leftovers(path) == []
        assert replace_calls == []  # not even attempted
    finally:
        os.chmod(path, stat.S_IREAD | stat.S_IWRITE)


@pytest.mark.skipif(
    not _open_handle_blocks_replace(), reason="an open handle does not block a replace here"
)
def test_a_file_held_open_by_another_process_is_left_alone(log, mask, clock):
    before = _stored("train", ["--slack-url", URL])
    log.write_bytes(before)

    with open(log, "rb"):
        assert mask_older_lines(str(log), mask) is False
        assert log.read_bytes() == before
        assert not os.path.exists(_marker(log))
        assert _leftovers(log) == []

    clock.advance(_DAY)
    assert mask_older_lines(str(log), mask) is True
    assert URL.encode() not in log.read_bytes()


@pytest.mark.requires_symlink
def test_a_symlinked_log_is_never_followed(tmp_path, mask):
    target = tmp_path / "elsewhere.jsonl"
    before = _stored("train", ["--slack-url", URL])
    target.write_bytes(before)
    link = tmp_path / "audit.jsonl"
    os.symlink(str(target), str(link))

    assert mask_older_lines(str(link), mask) is True  # skipped, not a failure

    assert target.read_bytes() == before
    assert os.path.islink(link)


def test_a_directory_in_place_of_the_log_does_not_raise(log, mask):
    log.mkdir()

    assert mask_older_lines(str(log), mask) is False
    assert not os.path.exists(_marker(log))


def test_a_file_that_is_not_a_regular_file_is_not_read(log, mask, monkeypatch):
    """A pipe or device would block a read; only a regular file is ever read."""
    import types

    before = _stored("train", ["--slack-url", URL])
    log.write_bytes(before)
    proxy = types.SimpleNamespace(**{name: getattr(stat, name) for name in dir(stat)})
    proxy.S_ISREG = lambda _mode: False
    monkeypatch.setattr(audit_log, "stat", proxy)

    assert mask_older_lines(str(log), mask) is False

    assert log.read_bytes() == before
    assert not os.path.exists(_marker(log))


def test_lines_appended_while_the_pass_runs_are_kept_and_masked(log, mask):
    """Another soup command may finish while this one is rewriting the log."""
    log.write_bytes(_stored("train", ["--slack-url", URL]))
    appended = []

    def mask_and_append(command, args):
        if not appended:
            appended.append(True)
            with open(log, "ab") as fh:
                fh.write(_stored("late", ["--discord-url", DISCORD_URL]))
        return mask(command, args)

    assert mask_older_lines(str(log), mask_and_append) is True

    stored = [json.loads(line) for line in log.read_bytes().splitlines()]
    assert [record["command"] for record in stored] == ["train", "late"]
    assert stored[1]["args"] == ["--discord-url", REDACTED]
    assert URL.encode() not in log.read_bytes()


def test_a_log_that_shrinks_while_it_is_read_is_left_for_the_next_time(log, mask):
    padding = "p" * 2000
    line = _stored("train", ["--slack-url", URL, "--note", padding])
    log.write_bytes(line * 20)  # well past the reader's buffer

    def mask_and_truncate(command, args):
        with open(log, "r+b") as fh:
            fh.truncate(len(line))
        return mask(command, args)

    assert mask_older_lines(str(log), mask_and_truncate) is False

    assert not os.path.exists(_marker(log))
    assert _leftovers(log) == []


def test_a_log_that_keeps_growing_is_left_for_the_next_time(log, mask):
    before = _stored("train", ["--slack-url", URL])
    log.write_bytes(before)

    def mask_and_grow(command, args):
        # Every line read makes another process add one, so no round ends stable.
        with open(log, "ab") as fh:
            fh.write(_stored("more", ["--n", "1"]))
        return mask(command, args)

    assert mask_older_lines(str(log), mask_and_grow) is False

    assert log.read_bytes().startswith(before)  # the first line is as it was
    assert not os.path.exists(_marker(log))
    assert _leftovers(log) == []


# ---------------------------------------------------------------------------
# A failed pass waits a day before the next one
# ---------------------------------------------------------------------------


def test_a_failed_pass_is_not_repeated_by_the_next_writes(
    log, mask, clock, scans, replace_refused
):
    before = _stored("train", ["--slack-url", URL])
    log.write_bytes(before)

    append_audit_event(_event(), str(log), arg_mask=mask)

    assert scans == [str(log)]  # the first write read the log and could not replace it
    assert os.path.isfile(_tried(log))
    assert not os.path.exists(_marker(log))

    for _ in range(3):
        clock.advance(60)
        append_audit_event(_event(), str(log), arg_mask=mask)

    assert scans == [str(log)]  # the later writes did not read it again
    assert log.read_bytes().startswith(before)
    assert len(log.read_bytes().splitlines()) == 1 + 4
    assert _leftovers(log) == []


def test_the_attempt_record_holds_the_utc_time_of_the_failure(
    log, mask, clock, replace_refused
):
    log.write_bytes(_stored("train", ["--slack-url", URL]))

    assert mask_older_lines(str(log), mask) is False

    assert _tried_text(log) == "2026-10-20T12:00:00+00:00"


def test_a_failed_pass_is_tried_again_after_a_day(log, mask, clock, scans, replace_refused):
    log.write_bytes(_stored("train", ["--slack-url", URL]))
    assert mask_older_lines(str(log), mask) is False
    assert len(scans) == 1

    clock.advance(_DAY - 1)  # a second short of a day
    assert mask_older_lines(str(log), mask) is False
    assert len(scans) == 1

    clock.advance(1)  # a day
    assert mask_older_lines(str(log), mask) is False  # refused again
    assert len(scans) == 2
    assert _tried_text(log) == "2026-10-21T12:00:00+00:00"  # the wait starts over

    clock.advance(_DAY - 1)
    assert mask_older_lines(str(log), mask) is False
    assert len(scans) == 2


def test_a_pass_that_succeeds_removes_the_attempt_record(log, mask, clock, replace_refused):
    log.write_bytes(_stored("train", ["--slack-url", URL]))
    assert mask_older_lines(str(log), mask) is False
    assert os.path.isfile(_tried(log))
    replace_refused.lift()  # the other process let go

    clock.advance(_DAY)
    assert mask_older_lines(str(log), mask) is True

    assert URL.encode() not in log.read_bytes()
    assert os.path.isfile(_marker(log))
    assert not os.path.exists(_tried(log))


def test_a_pass_that_never_failed_leaves_no_attempt_record(log, mask, caplog):
    caplog.set_level(logging.DEBUG, logger="soup_cli.utils.audit_log")
    log.write_bytes(_stored("train", ["--slack-url", URL]))

    assert mask_older_lines(str(log), mask) is True

    assert not os.path.exists(_tried(log))
    assert os.path.isfile(_marker(log))
    assert "failed-attempt record" not in caplog.text  # nothing to remove, nothing to report


def test_the_back_off_uses_the_real_clock_when_none_is_set(log, mask, scans, replace_refused):
    log.write_bytes(_stored("train", ["--slack-url", URL]))

    assert mask_older_lines(str(log), mask) is False
    assert mask_older_lines(str(log), mask) is False

    assert len(scans) == 1


def test_the_back_off_clock_is_utc():
    now = audit_log._utc_now()

    assert now.utcoffset() == timedelta(0)
    assert abs((datetime.now(timezone.utc) - now).total_seconds()) < 60


def test_a_new_attempt_record_replaces_a_longer_one(log, mask, clock, replace_refused):
    log.write_bytes(_stored("train", ["--slack-url", URL]))
    with open(_tried(log), "wb") as fh:
        fh.write(b"x" * 80 + b"\n")  # not a time, so the pass runs and records its failure

    assert mask_older_lines(str(log), mask) is False

    assert _tried_text(log) == "2026-10-20T12:00:00+00:00"


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"1\n",
        b"not a time\n",
        b"\xff\xfe\n",
        b"2026-10-20T11:59:00\n",  # no time zone, so no way to tell how old it is
    ],
    ids=["empty", "marker-text", "text", "not-ascii", "no-zone"],
)
def test_an_attempt_record_that_cannot_be_read_does_not_hold_the_pass_back(
    log, mask, clock, content
):
    log.write_bytes(_stored("train", ["--slack-url", URL]))
    with open(_tried(log), "wb") as fh:
        fh.write(content)

    assert mask_older_lines(str(log), mask) is True

    assert URL.encode() not in log.read_bytes()
    assert not os.path.exists(_tried(log))


def test_an_attempt_dated_in_the_future_does_not_hold_the_pass_back(log, mask, clock):
    log.write_bytes(_stored("train", ["--slack-url", URL]))
    with open(_tried(log), "w", encoding="ascii") as fh:
        fh.write((clock.now + timedelta(hours=2)).isoformat())

    assert mask_older_lines(str(log), mask) is True

    assert URL.encode() not in log.read_bytes()


def test_an_attempt_record_that_cannot_be_read_back_does_not_hold_the_pass_back(
    log, mask, clock, monkeypatch
):
    log.write_bytes(_stored("train", ["--slack-url", URL]))
    with open(_tried(log), "w", encoding="ascii") as fh:
        fh.write(clock.now.isoformat() + "\n")  # recent, if it could be read
    real = os.read

    def refuse(descriptor, size):
        if size == 64:  # the read of the record
            raise OSError("the read failed")
        return real(descriptor, size)

    monkeypatch.setattr(os, "read", refuse)

    assert mask_older_lines(str(log), mask) is True

    assert URL.encode() not in log.read_bytes()


def test_a_recent_attempt_record_holds_the_pass_back(log, mask, clock, scans):
    before = _stored("train", ["--slack-url", URL])
    log.write_bytes(before)
    with open(_tried(log), "w", encoding="ascii") as fh:
        fh.write((clock.now - timedelta(hours=23)).isoformat() + "\n")

    assert mask_older_lines(str(log), mask) is False

    assert scans == []
    assert log.read_bytes() == before
    assert not os.path.exists(_marker(log))


@pytest.mark.requires_symlink
def test_an_attempt_record_that_is_a_symlink_is_not_followed(log, mask, clock, tmp_path):
    log.write_bytes(_stored("train", ["--slack-url", URL]))
    elsewhere = tmp_path / "elsewhere.txt"
    elsewhere.write_text(clock.now.isoformat() + "\n", encoding="ascii")  # recent, if it were read
    os.symlink(str(elsewhere), _tried(log))

    assert mask_older_lines(str(log), mask) is True  # the link was not read

    assert URL.encode() not in log.read_bytes()
    assert not os.path.lexists(_tried(log))  # the link is gone ...
    assert elsewhere.exists()  # ... its target is not


@pytest.mark.requires_symlink
def test_a_failed_pass_does_not_write_through_a_symlink(
    log, mask, clock, tmp_path, replace_refused
):
    log.write_bytes(_stored("train", ["--slack-url", URL]))
    elsewhere = tmp_path / "elsewhere.txt"
    elsewhere.write_text("as it was\n", encoding="ascii")
    os.symlink(str(elsewhere), _tried(log))

    assert mask_older_lines(str(log), mask) is False

    assert elsewhere.read_text(encoding="ascii") == "as it was\n"


def test_an_attempt_record_that_cannot_be_written_does_not_raise(
    log, mask, clock, scans, replace_refused
):
    log.write_bytes(_stored("train", ["--slack-url", URL]))
    os.mkdir(_tried(log))  # a directory where the record would go

    assert mask_older_lines(str(log), mask) is False

    assert mask_older_lines(str(log), mask) is False  # nothing held it back: tried again
    assert len(scans) == 2


def test_an_attempt_record_that_cannot_be_removed_does_not_fail_a_finished_pass(
    log, mask, caplog
):
    caplog.set_level(logging.DEBUG, logger="soup_cli.utils.audit_log")
    log.write_bytes(_stored("train", ["--slack-url", URL]))
    os.mkdir(_tried(log))

    assert mask_older_lines(str(log), mask) is True

    assert URL.encode() not in log.read_bytes()
    assert os.path.isfile(_marker(log))
    assert os.path.isdir(_tried(log))
    assert "could not remove the failed-attempt record" in caplog.text


# ---------------------------------------------------------------------------
# Temp files left by a pass that was killed
# ---------------------------------------------------------------------------


def _aged(path, seconds):
    """Make ``path`` look ``seconds`` old."""
    then = time.time() - seconds
    os.utime(path, (then, then))


def _time_moves_on(monkeypatch, seconds):
    """The pass sees every file ``seconds`` older than it is (nothing is touched on disk)."""
    real = time.time
    monkeypatch.setattr(audit_log, "time", types.SimpleNamespace(time=lambda: real() + seconds))


def test_a_stale_temp_file_from_a_killed_pass_is_removed(log, mask):
    log.write_bytes(_stored("train", ["--slack-url", URL]))
    stale = log.parent / ".audit-mask-abc123.tmp"
    stale.write_bytes(b"half of a copy")
    _aged(stale, 2 * _HOUR)

    assert mask_older_lines(str(log), mask) is True

    assert not stale.exists()
    assert URL.encode() not in log.read_bytes()


def test_the_cleanup_finds_the_temp_files_the_pass_itself_makes(
    log, mask, clock, replace_refused, monkeypatch
):
    names = []
    real = tempfile.mkstemp

    def recording(*args, **kwargs):
        fd, path = real(*args, **kwargs)
        names.append(os.path.basename(path))
        return fd, path

    monkeypatch.setattr(tempfile, "mkstemp", recording)
    log.write_bytes(_stored("train", ["--slack-url", URL]))
    assert mask_older_lines(str(log), mask) is False  # made a temp file, removed it again
    assert len(names) == 1
    left_by_a_killed_pass = log.parent / names[0]
    left_by_a_killed_pass.write_bytes(b"half of a copy")
    _aged(left_by_a_killed_pass, 2 * _HOUR)

    clock.advance(_DAY)
    assert mask_older_lines(str(log), mask) is False  # refused again

    assert not left_by_a_killed_pass.exists()


def test_a_stale_temp_file_is_removed_even_when_the_pass_fails(log, mask, replace_refused):
    log.write_bytes(_stored("train", ["--slack-url", URL]))
    stale = log.parent / ".audit-mask-abc123.tmp"
    stale.write_bytes(b"half of a copy")
    _aged(stale, 2 * _HOUR)

    assert mask_older_lines(str(log), mask) is False

    assert not stale.exists()


def test_only_old_temp_files_of_the_pass_are_removed(log, mask):
    log.write_bytes(_stored("train", ["--slack-url", URL]))
    kept = {}
    for name, age in [
        (".audit-mask-young.tmp", 10 * 60),  # a pass may be writing it right now
        (".audit-mask-almost.tmp", _HOUR - 120),
        (".audit-mask-other.txt", 5 * _HOUR),  # not the pass's suffix
        ("other.tmp", 5 * _HOUR),  # not the pass's prefix
        ("x.audit-mask-y.tmp", 5 * _HOUR),  # the prefix is not at the start
    ]:
        path = log.parent / name
        path.write_bytes(b"data of " + name.encode())
        _aged(path, age)
        kept[name] = path.read_bytes()
    gone = log.parent / ".audit-mask-old.tmp"
    gone.write_bytes(b"old")
    _aged(gone, _HOUR + 120)

    assert mask_older_lines(str(log), mask) is True

    assert not gone.exists()
    for name, data in kept.items():
        assert (log.parent / name).read_bytes() == data, name


@pytest.mark.requires_symlink
def test_a_symlink_named_like_a_temp_file_is_not_followed(log, mask, tmp_path, monkeypatch):
    log.write_bytes(_stored("train", ["--slack-url", URL]))
    target = tmp_path / "elsewhere.bin"
    target.write_bytes(b"not a temp file of the pass")
    link = log.parent / ".audit-mask-link.tmp"
    os.symlink(str(target), str(link))
    _time_moves_on(monkeypatch, 5 * _HOUR)  # the link, like every file, is now hours old

    assert mask_older_lines(str(log), mask) is True

    assert os.path.islink(link)
    assert target.read_bytes() == b"not a temp file of the pass"


def test_a_log_that_needs_no_pass_costs_no_directory_scan(log, mask, clock):
    log.write_bytes(_stored("train", ["--slack-url", REDACTED]))
    stale = log.parent / ".audit-mask-abc123.tmp"
    stale.write_bytes(b"half of a copy")
    _aged(stale, 2 * _HOUR)
    with open(_marker(log), "w", encoding="ascii") as fh:
        fh.write("1\n")

    assert mask_older_lines(str(log), mask) is True  # done already
    assert stale.exists()

    os.unlink(_marker(log))
    with open(_tried(log), "w", encoding="ascii") as fh:
        fh.write(clock.now.isoformat() + "\n")

    assert mask_older_lines(str(log), mask) is False  # waiting out a failed attempt
    assert stale.exists()


def test_a_directory_that_cannot_be_listed_does_not_stop_the_pass(log, mask, monkeypatch):
    log.write_bytes(_stored("train", ["--slack-url", URL]))
    real = os.listdir

    def refuse(path="."):
        if os.path.abspath(path) == os.path.abspath(log.parent):
            raise PermissionError("cannot list the directory")
        return real(path)

    monkeypatch.setattr(os, "listdir", refuse)

    assert mask_older_lines(str(log), mask) is True

    assert URL.encode() not in log.read_bytes()


def test_a_temp_file_that_cannot_be_removed_does_not_stop_the_pass(log, mask, monkeypatch):
    log.write_bytes(_stored("train", ["--slack-url", URL]))
    locked = log.parent / ".audit-mask-locked.tmp"
    removable = log.parent / ".audit-mask-removable.tmp"
    for path in (locked, removable):
        path.write_bytes(b"half of a copy")
        _aged(path, 2 * _HOUR)
    real = os.unlink

    def refuse_one(path, *args, **kwargs):
        if os.path.basename(path) == locked.name:
            raise PermissionError("the file is in use")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", refuse_one)

    assert mask_older_lines(str(log), mask) is True

    assert locked.exists()
    assert not removable.exists()
    assert URL.encode() not in log.read_bytes()


# ---------------------------------------------------------------------------
# Size
# ---------------------------------------------------------------------------


def test_a_large_log_is_masked(log, mask):
    count = 10_000
    lines = []
    for index in range(count):
        args = ["--slack-url", URL] if index % 10 == 0 else ["--n", str(index)]
        lines.append(_stored(f"cmd{index}", args))
    log.write_bytes(b"".join(lines))

    started = time.monotonic()
    assert mask_older_lines(str(log), mask) is True
    elapsed = time.monotonic() - started

    stored = log.read_bytes().splitlines()
    assert len(stored) == count
    assert URL.encode() not in log.read_bytes()
    assert json.loads(stored[0])["args"] == ["--slack-url", REDACTED]
    assert json.loads(stored[1])["args"] == ["--n", "1"]
    assert json.loads(stored[-1])["command"] == f"cmd{count - 1}"
    assert elapsed < 60, elapsed


def _peak_while_masking(path, stub_mask) -> int:
    tracemalloc.start()
    try:
        assert mask_older_lines(str(path), stub_mask) is True
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


def _stub_mask(_command, args):
    return ["<x>" if arg.startswith("https://") else arg for arg in args]


def test_the_log_is_streamed_not_loaded(log):
    """Only the last of 10,000 lines changes, so the whole earlier part is copied over."""
    padding = "p" * 2000
    plain = _stored("train", ["--note", padding])
    changed = _stored("train", ["--slack-url", URL, "--note", padding])
    log.write_bytes(plain * 9_999 + changed)
    size = os.path.getsize(log)
    assert size > 20_000_000

    peak = _peak_while_masking(log, _stub_mask)

    assert URL.encode() not in log.read_bytes()
    expected_last = _stored("train", ["--slack-url", "<x>", "--note", padding])
    assert log.read_bytes() == plain * 9_999 + expected_last
    assert peak < size // 4, (peak, size)


def test_one_enormous_line_is_not_loaded_either(log, monkeypatch):
    """A single line far beyond any real record is read in pieces and copied through."""
    monkeypatch.setattr(audit_log, "_MASK_MAX_LINE_BYTES", 1024 * 1024)
    junk = b"x" * 24_000_000 + b"\n"
    changed = _stored("train", ["--slack-url", URL])
    log.write_bytes(junk + changed)

    peak = _peak_while_masking(log, _stub_mask)

    assert log.read_bytes() == junk + _stored("train", ["--slack-url", "<x>"])
    assert peak < len(junk) // 4, (peak, len(junk))


# ---------------------------------------------------------------------------
# The masking function the pass reuses
# ---------------------------------------------------------------------------


def test_the_bound_masker_walks_the_command_tree_once(monkeypatch):
    import typer

    import soup_cli.cli as cli
    from soup_cli.utils import argv_redaction

    calls = []
    real = typer.main.get_command

    def counting(app_arg):
        calls.append(1)
        return real(app_arg)

    option_walks = []
    real_walk = argv_redaction.all_long_option_strings

    def counting_walk(root):
        option_walks.append(1)
        return real_walk(root)

    monkeypatch.setattr(typer.main, "get_command", counting)
    monkeypatch.setattr(argv_redaction, "all_long_option_strings", counting_walk)
    bound = cli._audit_arg_masker()
    assert calls == [] and option_walks == []  # nothing is walked until a line needs masking

    for _ in range(5):
        assert bound("train", ["--slack-url", URL]) == ("--slack-url", REDACTED)
    assert len(calls) == 1
    assert len(option_walks) == 1


def test_a_given_tree_gives_the_same_answer_as_none():
    import typer

    import soup_cli.cli as cli
    from soup_cli.utils.argv_redaction import all_long_option_strings

    root = typer.main.get_command(cli.app)
    tree = (root, all_long_option_strings(root), {})
    cases = [
        ("train", ["--slack-url", URL, "--config", "a.yaml"]),
        ("serve", ["--api-key", "V", "--model", "m"]),
        ("data", ["-t", "VALUE"]),
        ("push", ["--repo", "u/m", "-t", "VALUE"]),
    ]
    for _round in range(2):  # the second round answers from the memo
        for command, args in cases:
            assert cli._mask_audit_args(command, args, tree=tree) == cli._mask_audit_args(
                command, args
            )


def test_a_failure_while_reading_a_commands_options_falls_back_to_the_whole_app(monkeypatch):
    import typer

    import soup_cli.cli as cli
    from soup_cli.utils import argv_redaction

    root = typer.main.get_command(cli.app)
    memo: dict = {}
    tree = (root, argv_redaction.all_long_option_strings(root), memo)

    def broken(_command):
        raise RuntimeError("the options cannot be read")

    monkeypatch.setattr(argv_redaction, "secret_option_strings", broken)

    masked = cli._mask_audit_args("push", ["--repo", "u/m", "-t", "VALUE"], tree=tree)

    assert masked == ("--repo", "u/m", "-t", REDACTED)
    assert list(memo) == ["*"]


def test_a_group_without_a_subcommand_gets_the_credential_options_of_the_whole_app():
    import typer

    import soup_cli.cli as cli
    from soup_cli.utils.argv_redaction import all_long_option_strings

    root = typer.main.get_command(cli.app)
    tree = (root, all_long_option_strings(root), {})

    for chosen in (None, tree):
        assert cli._mask_audit_args("data", ["-t", "VALUE"], tree=chosen) == ("-t", REDACTED)


def test_a_given_tree_keeps_the_options_it_declares():
    import typer

    import soup_cli.cli as cli
    from soup_cli.utils.argv_redaction import all_long_option_strings

    root = typer.main.get_command(cli.app)
    tree = (root, all_long_option_strings(root), {})
    for command, args in [
        ("train", ["--slack-url", "--config", "a.yaml"]),
        ("train", ["-t", "sft"]),
    ]:
        assert cli._mask_audit_args(command, args, tree=tree) == cli._mask_audit_args(
            command, args
        )


def test_a_commands_options_are_remembered_for_all_its_lines():
    import typer

    import soup_cli.cli as cli
    from soup_cli.utils.argv_redaction import all_long_option_strings, resolve_command

    root = typer.main.get_command(cli.app)
    train = resolve_command(root, ["train"])
    memo: dict = {}
    tree = (root, all_long_option_strings(root), memo)

    first = cli._mask_audit_args("train", ["--slack-url", URL, "--config", "a.yaml"], tree=tree)
    assert first == ("--slack-url", REDACTED, "--config", "a.yaml")
    assert list(memo) == [id(train)]  # stored for the next line of the same command
    assert "--slack-url" in memo[id(train)].credential_opts

    # A later line is answered from what was stored: an entry planted there is used.
    memo[id(train)] = ArgMasker(frozenset({"--planted-flag"}), tree[1])
    line = ["--planted-flag", "v", "--config", "a.yaml"]
    planted = cli._mask_audit_args("train", line, tree=tree)
    assert planted == ("--planted-flag", REDACTED, "--config", "a.yaml")


def test_each_command_gets_one_masker_for_all_its_lines(monkeypatch):
    """The option sets are built once per command (and once for the whole app), not per line."""
    import soup_cli.cli as cli
    from soup_cli.utils import argv_redaction

    built = []
    real = argv_redaction.ArgMasker

    class Counting(real):
        __slots__ = ()

        def __init__(self, credential_opts, known_opts=frozenset()):
            built.append(frozenset(credential_opts))
            super().__init__(credential_opts, known_opts)

    monkeypatch.setattr(argv_redaction, "ArgMasker", Counting)
    bound = cli._audit_arg_masker()
    lines = [
        ("train", ["--slack-url", URL, "--config", "a.yaml"]),
        ("version", []),
        ("data", []),  # a group: masked with the options of the whole app
        ("no-such-command", ["--auth-token", "V"]),  # no command: the same whole-app masker
    ]

    for _round in range(4):
        for command, args in lines:
            bound(command, args)

    assert len(built) == 3, built
    assert len({id(options) for options in built}) == 3


def test_a_masker_keeps_nothing_from_one_call_to_the_next():
    masker = ArgMasker(frozenset({"--auth-token", "-a"}), frozenset({"--port"}))

    assert masker(["--auth-token", "x", "--port", "1"]) == (
        "--auth-token", REDACTED, "--port", "1",
    )
    assert masker(["--auth-token"]) == ("--auth-token",)  # no value: nothing is pending...
    assert masker(["plain", "-a"]) == ("plain", "-a")
    assert masker(["value"]) == ("value",)  # ...for the next call
    assert masker(["-ax"]) == (f"-a{REDACTED}",)
    assert masker(["--auth-token=x"]) == (f"--auth-token={REDACTED}",)


def test_a_masker_masks_the_options_it_was_given_in_every_spelling():
    masker = ArgMasker(frozenset({"--webhook", "-w"}), frozenset({"--sink"}))

    assert masker(["--webhook", "U"]) == ("--webhook", REDACTED)
    assert masker(["--webhook=U"]) == (f"--webhook={REDACTED}",)
    assert masker(["-w", "U"]) == ("-w", REDACTED)
    assert masker(["-wU"]) == (f"-w{REDACTED}",)
    # `--sink` is an option of the command tree, so it is not the value of `--webhook`.
    assert masker(["--webhook", "--sink", "S"]) == ("--webhook", "--sink", "S")
    assert masker(["--webhook", "--other", "S"]) == ("--webhook", REDACTED, "S")


def test_mask_secret_args_gives_the_answers_of_a_masker_built_for_one_call():
    from soup_cli.utils.argv_redaction import mask_secret_args

    options = frozenset({"--webhook", "-w"})
    known = frozenset({"--sink"})
    for args in (
        ["--webhook", "--sink", "S"],
        ["--webhook", "--other", "S"],
        ["--webhook=U", "-wV", "plain"],
    ):
        assert mask_secret_args(args, options, known) == ArgMasker(options, known)(args)
    assert mask_secret_args(["--webhook", "--sink", "S"], options, known) == (
        "--webhook", "--sink", "S",
    )


def test_a_masker_works_from_a_snapshot_of_its_options():
    options = {"--auth-token"}
    masker = ArgMasker(options, {"--port"})
    options.add("--other-flag")  # added after the masker was built

    assert masker.credential_opts == frozenset({"--auth-token"})
    assert masker(["--other-flag", "v"]) == ("--other-flag", "v")
    assert masker(["--auth-token", "v"]) == ("--auth-token", REDACTED)
