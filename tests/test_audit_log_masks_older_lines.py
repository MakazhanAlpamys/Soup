"""Audit lines written by older versions get the same URL masking as new ones.

``cli._emit_audit_event`` masks option values by option name before a line is
stored; lines stored by older versions still carry them as typed. The writer
(``append_audit_event(..., arg_mask=...)``) applies the same masking to the
stored lines once, guarded by a marker file next to the log.
"""

from __future__ import annotations

import functools
import json
import os
import stat
import tempfile
import tracemalloc

import pytest

from soup_cli.utils import audit_log
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


def test_a_failing_mask_leaves_the_log_untouched_and_is_retried(log, mask):
    before = _stored("version", []) + _stored("train", ["--slack-url", URL])
    log.write_bytes(before)

    def broken(_command, _args):
        raise RuntimeError("masking is unavailable")

    assert mask_older_lines(str(log), broken) is False

    assert log.read_bytes() == before
    assert not os.path.exists(_marker(log))
    assert _leftovers(log) == []

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


def test_a_failed_replace_leaves_the_log_untouched_and_is_retried(log, mask, monkeypatch):
    before = _stored("train", ["--slack-url", URL])
    log.write_bytes(before)
    real_replace = os.replace

    def refuse(*_args, **_kwargs):
        raise PermissionError("the file is in use")

    monkeypatch.setattr(os, "replace", refuse)
    assert mask_older_lines(str(log), mask) is False
    monkeypatch.setattr(os, "replace", real_replace)

    assert log.read_bytes() == before
    assert not os.path.exists(_marker(log))
    assert _leftovers(log) == []

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
def test_a_file_held_open_by_another_process_is_left_alone(log, mask):
    before = _stored("train", ["--slack-url", URL])
    log.write_bytes(before)

    with open(log, "rb"):
        assert mask_older_lines(str(log), mask) is False
        assert log.read_bytes() == before
        assert not os.path.exists(_marker(log))
        assert _leftovers(log) == []

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
# Size
# ---------------------------------------------------------------------------


def test_a_large_log_is_masked(log, mask):
    import time

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
    assert "--slack-url" in memo[id(train)]

    # A later line is answered from what was stored: an entry planted there is used.
    memo[id(train)] = frozenset({"--planted-flag"})
    line = ["--planted-flag", "v", "--config", "a.yaml"]
    planted = cli._mask_audit_args("train", line, tree=tree)
    assert planted == ("--planted-flag", REDACTED, "--config", "a.yaml")
