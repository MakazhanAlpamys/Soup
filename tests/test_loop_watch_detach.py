"""``soup loop watch --detach`` leaves a PID file and a log next to ``loop.yaml``.

The detached child used to write to ``/dev/null`` and nothing recorded that it ran, so a
user (or the Soup Zero desktop app) could neither tell whether the daemon was alive nor
read what it printed, and a second ``--detach`` started a second daemon on the same
``loop.yaml``. Now:

* the parent writes ``.soup/loop.pid`` (JSON: ``pid``, ``started_at``, ``log``) once the
  child is spawned, and sends the child's stdout and stderr to ``.soup/loop.log``
  (appended, never truncated);
* a second ``--detach`` is refused while the PID file names a live process, and replaces a
  stale file (the process is gone, or the file is unparsable);
* ``soup loop status`` shows a ``daemon`` row;
* the detached child removes the PID file on exit, but only if it still holds its pid.

Liveness is the repo's ``process_is_alive``; every test replaces it with a fake, so nothing
here ever signals a process. ``subprocess.Popen`` is always mocked: no daemon is spawned.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import types
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import pytest
from typer.testing import CliRunner

import soup_cli.utils.loop_detach as loop_detach
from soup_cli.cli import app
from soup_cli.utils.loop_state import read_state

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

#: A pid the fake liveness never reports alive unless a test says so.
OTHER_PID = 424242


def _clean(text: str) -> str:
    return " ".join(_ANSI.sub("", text).split())


@pytest.fixture()
def initialised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(loop_detach.DETACHED_ENV, raising=False)
    result = runner.invoke(app, ["loop", "init", "m", "--eval", "e.json", "--baseline", "b"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    return tmp_path


@pytest.fixture(autouse=True)
def liveness(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    """Replace the liveness check: only pids put in ``live`` read as running."""
    fake = types.SimpleNamespace(live=set(), asked=[])

    def process_is_alive(pid):
        fake.asked.append(pid)
        return pid in fake.live

    monkeypatch.setattr(loop_detach, "process_is_alive", process_is_alive)
    return fake


def _detach(*extra: str, pid: int = 123, side_effect=None):
    with mock.patch("soup_cli.commands.loop.subprocess.Popen") as popen:
        popen.return_value.pid = pid
        if side_effect is not None:
            popen.side_effect = side_effect
        result = runner.invoke(app, ["loop", "watch", "--detach", *extra])
    return result, popen


def _write_pid_file(root: Path, body: str | bytes) -> Path:
    path = root / ".soup" / "loop.pid"
    path.parent.mkdir(exist_ok=True)
    if isinstance(body, bytes):
        path.write_bytes(body)
    else:
        path.write_text(body, encoding="utf-8")
    return path


def _status() -> str:
    result = runner.invoke(app, ["loop", "status"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    return _clean(result.output)


# ---------------------------------------------------------------------------
# The PID file: where it goes and what it holds
# ---------------------------------------------------------------------------


def test_the_pid_and_log_paths_sit_next_to_the_state_file():
    assert loop_detach.pid_file_path(os.path.join("a", "b", "loop.yaml")) == os.path.join(
        "a", "b", "loop.pid"
    )
    assert loop_detach.log_file_path(os.path.join("a", "b", "loop.yaml")) == os.path.join(
        "a", "b", "loop.log"
    )


def test_the_default_paths_are_beside_the_default_state_file(initialised):
    state_dir = os.path.dirname(os.path.abspath(os.path.join(".soup", "loop.yaml")))
    assert loop_detach.pid_file_path() == os.path.join(state_dir, "loop.pid")
    assert loop_detach.log_file_path() == os.path.join(state_dir, "loop.log")


def test_detach_writes_the_pid_file_as_json(initialised):
    result, popen = _detach()
    out = _clean(result.output)
    assert result.exit_code == 0, (out, repr(result.exception))
    popen.assert_called_once()
    assert "watch detached pid=123" in out
    assert ".soup/loop.pid" in out and ".soup/loop.log" in out

    record = json.loads((initialised / ".soup" / "loop.pid").read_text("utf-8"))
    assert set(record) == {"pid", "started_at", "log"}
    assert record["pid"] == 123
    assert record["log"] == ".soup/loop.log"
    started = datetime.fromisoformat(record["started_at"])
    assert started.utcoffset() == timedelta(0), record["started_at"]
    assert abs(datetime.now(started.tzinfo) - started) < timedelta(minutes=5)


def test_detach_leaves_no_temp_file_behind(initialised):
    result, _ = _detach()
    assert result.exit_code == 0, (result.output, repr(result.exception))
    left = sorted(entry.name for entry in (initialised / ".soup").iterdir())
    assert left == ["loop.log", "loop.pid", "loop.yaml"], left


def test_no_pid_file_is_written_when_the_spawn_fails(initialised):
    result, _ = _detach(side_effect=OSError("no such interpreter"))
    assert result.exit_code != 0
    assert isinstance(result.exception, OSError), repr(result.exception)
    assert not (initialised / ".soup" / "loop.pid").exists()


def test_a_failed_pid_write_after_the_spawn_names_the_running_pid(initialised, monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(loop_detach, "atomic_write_text", refuse)
    result, popen = _detach(pid=777)
    out = _clean(result.output)
    popen.assert_called_once()
    assert result.exit_code == 1, (out, repr(result.exception))
    assert "pid=777" in out and "PID file could not be written" in out and "disk full" in out


def test_the_child_is_marked_as_detached_through_its_environment(initialised):
    result, popen = _detach()
    assert result.exit_code == 0, (result.output, repr(result.exception))
    env = popen.call_args.kwargs["env"]
    assert env[loop_detach.DETACHED_ENV] == "1"
    assert env["PATH"] == os.environ["PATH"], "the child keeps the parent's environment"
    argv = popen.call_args.args[0]
    assert "--foreground" in argv and loop_detach.DETACHED_ENV not in " ".join(argv)


# ---------------------------------------------------------------------------
# The log: appended, and handed to the child as stdout and stderr
# ---------------------------------------------------------------------------


def test_the_log_is_appended_to_and_handed_to_the_child(initialised):
    log = initialised / ".soup" / "loop.log"
    log.write_bytes(b"earlier run\n")
    seen = {}

    def fake_popen(argv, **kwargs):
        handle = kwargs["stdout"]
        seen.update(handle=handle, stderr=kwargs["stderr"], mode=handle.mode)
        handle.write(b"child output\n")
        handle.flush()
        return types.SimpleNamespace(pid=123)

    result, _ = _detach(side_effect=fake_popen)
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert log.read_bytes() == b"earlier run\nchild output\n"
    assert seen["mode"] == "ab", "the handle is an append handle, not a write handle"
    assert seen["stderr"] is subprocess.STDOUT, "stderr follows stdout into the same log"
    assert seen["handle"].closed, "the parent drops its own handle once the child has one"


def test_the_log_is_opened_for_append_and_never_truncated(initialised, monkeypatch):
    flags_seen = []
    real = loop_detach.open_no_follow

    def spy(path, flags, *args, **kwargs):
        if flags & (os.O_WRONLY | os.O_RDWR):
            flags_seen.append(flags)
        return real(path, flags, *args, **kwargs)

    monkeypatch.setattr(loop_detach, "open_no_follow", spy)
    result, _ = _detach()
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert len(flags_seen) == 1, flags_seen
    assert flags_seen[0] & os.O_APPEND, "the log must be opened with O_APPEND"
    assert not flags_seen[0] & os.O_TRUNC, "the log must not be truncated"


def test_the_log_is_created_when_it_does_not_exist(initialised):
    assert not (initialised / ".soup" / "loop.log").exists()
    result, _ = _detach()
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert (initialised / ".soup" / "loop.log").is_file()


@pytest.mark.requires_symlink
def test_a_symlinked_log_is_refused_before_anything_is_spawned(initialised):
    target = initialised / "elsewhere.log"
    target.write_text("keep me\n", encoding="utf-8")
    (initialised / ".soup" / "loop.log").symlink_to(target)

    result, popen = _detach()
    out = _clean(result.output)
    assert result.exit_code == 2, (out, repr(result.exception))
    assert "log file must not be a symlink" in out
    popen.assert_not_called()
    assert target.read_text("utf-8") == "keep me\n"
    assert not (initialised / ".soup" / "loop.pid").exists()


def test_a_hard_linked_log_is_refused(initialised):
    target = initialised / "elsewhere.log"
    target.write_text("keep me\n", encoding="utf-8")
    try:
        os.link(target, initialised / ".soup" / "loop.log")
    except OSError:
        pytest.skip("this filesystem cannot create hard links")

    result, popen = _detach()
    out = _clean(result.output)
    assert result.exit_code == 2, (out, repr(result.exception))
    assert "log file must not be a hard link" in out
    popen.assert_not_called()
    assert target.read_text("utf-8") == "keep me\n"


# ---------------------------------------------------------------------------
# One daemon at a time
# ---------------------------------------------------------------------------


def test_a_second_detach_is_refused_while_the_first_daemon_is_alive(initialised, liveness):
    first, first_popen = _detach(pid=123)
    assert first.exit_code == 0, (first.output, repr(first.exception))
    pid_file = initialised / ".soup" / "loop.pid"
    before = pid_file.read_bytes()
    log = initialised / ".soup" / "loop.log"
    log.write_bytes(b"daemon output\n")
    liveness.live.add(123)

    second, second_popen = _detach(pid=124)
    out = _clean(second.output)
    assert second.exit_code == 2, (out, repr(second.exception))
    assert "already running" in out
    assert "pid 123" in out
    assert ".soup/loop.pid" in out
    assert "watch detached" not in out
    second_popen.assert_not_called()
    assert pid_file.read_bytes() == before, "the live daemon's PID file is left alone"
    assert log.read_bytes() == b"daemon output\n"
    assert 123 in liveness.asked


def test_a_refused_detach_does_not_create_the_log(initialised, liveness):
    _write_pid_file(initialised, json.dumps({"pid": OTHER_PID}))
    liveness.live.add(OTHER_PID)
    result, _ = _detach()
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert not (initialised / ".soup" / "loop.log").exists()


def test_a_stale_pid_file_is_replaced_with_a_notice(initialised, liveness):
    pid_file = _write_pid_file(
        initialised, json.dumps({"pid": OTHER_PID, "started_at": "2020-01-01T00:00:00+00:00"})
    )
    assert OTHER_PID not in liveness.live

    result, popen = _detach(pid=125)
    out = _clean(result.output)
    assert result.exit_code == 0, (out, repr(result.exception))
    popen.assert_called_once()
    assert "replacing stale PID file" in out
    assert f"pid {OTHER_PID} is not running" in out
    assert "watch detached pid=125" in out
    assert json.loads(pid_file.read_text("utf-8"))["pid"] == 125
    assert OTHER_PID in liveness.asked


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b"not json at all", id="not-json"),
        pytest.param(b"\xff\xfe\x00garbage", id="not-utf8"),
        pytest.param(b"[1, 2, 3]", id="list"),
        pytest.param(b'{"started_at": "x"}', id="no-pid"),
        pytest.param(b'{"pid": "123"}', id="pid-is-a-string"),
        pytest.param(b'{"pid": 12.5}', id="pid-is-a-float"),
        pytest.param(b'{"pid": true}', id="pid-is-a-bool"),
        pytest.param(b'{"pid": 0}', id="pid-zero"),
        pytest.param(b'{"pid": -5}', id="pid-negative"),
        pytest.param(b'{"pid": 4294967297}', id="pid-wraps-a-dword"),
        pytest.param(b"[" * 3000, id="deeply-nested"),
        pytest.param(b'{"pid": 5, "pad": "' + b"x" * 5000 + b'"}', id="oversized"),
        # whitespace padding parses once read is cut at the cap: only the size check refuses it
        pytest.param(b'{"pid": 5}' + b" " * 5000, id="oversized-padding"),
    ],
)
def test_an_unparsable_pid_file_is_replaced_and_never_asked_about(initialised, liveness, body):
    pid_file = _write_pid_file(initialised, body)

    result, popen = _detach(pid=126)
    out = _clean(result.output)
    assert result.exit_code == 0, (out, repr(result.exception))
    popen.assert_called_once()
    assert "replacing stale PID file" in out
    assert "not a valid PID file" in out
    assert liveness.asked == [], "an invalid pid must never reach the liveness check"
    assert json.loads(pid_file.read_text("utf-8"))["pid"] == 126


@pytest.mark.requires_symlink
def test_a_symlinked_pid_file_is_refused(initialised):
    target = initialised / "elsewhere.pid"
    target.write_text(json.dumps({"pid": OTHER_PID}), encoding="utf-8")
    (initialised / ".soup" / "loop.pid").symlink_to(target)

    result, popen = _detach()
    out = _clean(result.output)
    assert result.exit_code == 2, (out, repr(result.exception))
    assert "PID file must not be a symlink" in out
    popen.assert_not_called()
    assert json.loads(target.read_text("utf-8")) == {"pid": OTHER_PID}


def test_a_directory_in_place_of_the_pid_file_is_refused_before_the_spawn(initialised):
    (initialised / ".soup" / "loop.pid").mkdir()
    result, popen = _detach()
    out = _clean(result.output)
    assert result.exit_code == 2, (out, repr(result.exception))
    assert "PID file must be a regular file" in out
    popen.assert_not_called()


# ---------------------------------------------------------------------------
# soup loop status
# ---------------------------------------------------------------------------


def test_status_shows_a_running_daemon_with_its_pid_and_log(initialised, liveness):
    _detach(pid=123)
    liveness.live.add(123)
    assert "running (pid 123), log .soup/loop.log" in _status()


def test_status_without_a_pid_file_says_not_running(initialised):
    out = _status()
    assert "daemon" in out
    assert "not running" in out
    assert "stale" not in out
    assert "running (pid" not in out


def test_status_with_a_dead_pid_says_stale(initialised, liveness):
    _write_pid_file(initialised, json.dumps({"pid": OTHER_PID}))
    out = _status()
    assert "not running (stale pid file)" in out
    assert "running (pid" not in out


def test_status_with_an_unparsable_pid_file_says_stale(initialised, liveness):
    _write_pid_file(initialised, b"garbage")
    assert "not running (stale pid file)" in _status()
    assert liveness.asked == []


def test_status_changes_nothing_on_disk(initialised, liveness):
    pid_file = _write_pid_file(initialised, json.dumps({"pid": OTHER_PID}))
    before = pid_file.read_bytes()
    _status()
    assert pid_file.read_bytes() == before, "a stale file is reported, not removed"
    assert not (initialised / ".soup" / "loop.log").exists()


@pytest.mark.requires_symlink
def test_status_does_not_follow_a_symlinked_pid_file(initialised, liveness):
    target = initialised / "elsewhere.pid"
    target.write_text(json.dumps({"pid": OTHER_PID}), encoding="utf-8")
    (initialised / ".soup" / "loop.pid").symlink_to(target)
    liveness.live.add(OTHER_PID)

    out = _status()
    assert "not running (unusable PID file)" in out
    assert "running (pid" not in out
    assert liveness.asked == []


def test_the_pid_module_stays_torch_free():
    code = (
        "import sys\n"
        "import soup_cli.utils.loop_detach\n"
        "import soup_cli.commands.loop\n"
        "assert 'torch' not in sys.modules, 'loop_detach imported torch'\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr[-2000:]


# ---------------------------------------------------------------------------
# Cleanup by the detached child
# ---------------------------------------------------------------------------


def test_cleanup_removes_the_pid_file_when_it_holds_the_own_pid(initialised):
    pid_file = _write_pid_file(initialised, json.dumps({"pid": 77}))
    assert loop_detach.remove_pid_file_if_own(77) is True
    assert not pid_file.exists()


def test_cleanup_leaves_a_pid_file_that_holds_another_pid(initialised):
    pid_file = _write_pid_file(initialised, json.dumps({"pid": 78}))
    assert loop_detach.remove_pid_file_if_own(77) is False
    assert json.loads(pid_file.read_text("utf-8")) == {"pid": 78}


@pytest.mark.parametrize("body", [b"garbage", b"[]", b'{"pid": "77"}'])
def test_cleanup_leaves_a_pid_file_it_cannot_read(initialised, body):
    pid_file = _write_pid_file(initialised, body)
    assert loop_detach.remove_pid_file_if_own(77) is False
    assert pid_file.read_bytes() == body


def test_cleanup_with_no_pid_file_is_a_no_op(initialised):
    assert loop_detach.remove_pid_file_if_own(77) is False


@pytest.mark.requires_symlink
def test_cleanup_never_unlinks_through_a_symlink(initialised):
    target = initialised / "elsewhere.pid"
    target.write_text(json.dumps({"pid": 77}), encoding="utf-8")
    link = initialised / ".soup" / "loop.pid"
    link.symlink_to(target)
    assert loop_detach.remove_pid_file_if_own(77) is False
    assert target.exists() and link.is_symlink()


@pytest.fixture()
def registered(monkeypatch: pytest.MonkeyPatch):
    """Capture ``atexit.register`` calls so no handler outlives the test."""
    calls = []
    monkeypatch.setattr(
        "soup_cli.commands.loop.atexit.register",
        lambda func, *args, **kwargs: calls.append((func, args)),
    )
    monkeypatch.setattr(
        "soup_cli.commands.loop.watch", lambda config: (read_state(), 0)
    )
    return calls


def _foreground_watch():
    result = runner.invoke(
        app, ["loop", "watch", "--foreground", "--max-iterations", "1", "--poll-interval", "1"]
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    return result


def _cleanup_calls(registered):
    return [(func, args) for func, args in registered if func is loop_detach.remove_pid_file_if_own]


def test_a_detached_child_registers_the_cleanup_of_its_own_pid_file(
    initialised, monkeypatch, registered
):
    monkeypatch.setenv(loop_detach.DETACHED_ENV, "1")
    pid_file = _write_pid_file(initialised, json.dumps({"pid": os.getpid()}))

    _foreground_watch()

    calls = _cleanup_calls(registered)
    assert len(calls) == 1, registered
    func, args = calls[0]
    assert args[0] == os.getpid()
    func(*args)  # what the interpreter does at exit
    assert not pid_file.exists()


def test_the_registered_cleanup_leaves_a_pid_file_that_names_another_pid(
    initialised, monkeypatch, registered
):
    monkeypatch.setenv(loop_detach.DETACHED_ENV, "1")
    pid_file = _write_pid_file(initialised, json.dumps({"pid": os.getpid() + 1}))

    _foreground_watch()

    (func, args), = _cleanup_calls(registered)
    func(*args)
    assert json.loads(pid_file.read_text("utf-8")) == {"pid": os.getpid() + 1}


def test_a_plain_foreground_watch_registers_nothing(initialised, registered):
    pid_file = _write_pid_file(initialised, json.dumps({"pid": os.getpid()}))

    _foreground_watch()

    assert _cleanup_calls(registered) == [], "a non-detached watch must not touch the PID file"
    assert pid_file.exists()


def test_a_parent_with_the_marker_set_does_not_register_the_cleanup(
    initialised, monkeypatch, registered
):
    """``--detach`` itself returns at once; only the child it spawns cleans up."""
    monkeypatch.setenv(loop_detach.DETACHED_ENV, "1")
    result, _ = _detach()
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert _cleanup_calls(registered) == []
