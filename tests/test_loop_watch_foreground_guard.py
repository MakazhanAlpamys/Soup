"""A foreground ``soup loop watch`` is refused while a live daemon holds the PID file.

Only ``--detach`` consulted the PID file; a foreground ``soup loop watch`` ran next to the
detached daemon (a user who types it to look at the daemon starts a second one: two trainings
on one card, two harvests of the same rows, two canaries decided from per-cycle reads). The
foreground path now takes the same guard and refuses by name: which pid, the PID file, and how
to stop it. A stale file is not a daemon, an unusable one is noticed and ignored, and the
child that ``--detach`` itself spawned (it carries ``SOUP_LOOP_DETACHED=1`` and its own pid is
in the file) is never refused by its own claim.

``process_is_alive`` is faked and ``watch`` is replaced by a recorder: nothing here signals a
process, trains or starts a daemon.
"""

from __future__ import annotations

import json
import os
import re
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

import soup_cli.utils.loop_detach as loop_detach
from soup_cli.cli import app
from soup_cli.utils.loop_state import read_state

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
DAEMON_PID = 424242


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
    fake = types.SimpleNamespace(live=set())
    monkeypatch.setattr(loop_detach, "process_is_alive", lambda pid: pid in fake.live)
    return fake


@pytest.fixture()
def watch_calls(monkeypatch: pytest.MonkeyPatch) -> list:
    """Replace the daemon loop by a recorder: a call here means the watch was allowed."""
    calls: list = []

    def fake_watch(config):
        calls.append(config)
        return read_state(), 0

    monkeypatch.setattr("soup_cli.commands.loop.watch", fake_watch)
    return calls


def _pid_path(root: Path) -> Path:
    return root / ".soup" / "loop.pid"


def _write_pid(root: Path, pid: int) -> None:
    _pid_path(root).write_text(json.dumps({"pid": pid}), encoding="utf-8")


def _watch(*extra: str):
    return runner.invoke(app, ["loop", "watch", "--max-iterations", "1", *extra])


def test_a_foreground_watch_is_refused_while_a_live_daemon_holds_the_pid_file(
    initialised, liveness, watch_calls
):
    liveness.live.add(DAEMON_PID)
    _write_pid(initialised, DAEMON_PID)

    result = _watch()

    out = _clean(result.output)
    assert result.exit_code == 2, (out, repr(result.exception))
    assert f"a loop daemon is already running: pid {DAEMON_PID}" in out
    assert watch_calls == []


def test_the_refusal_names_the_pid_file_and_how_to_stop_the_daemon(
    initialised, liveness, watch_calls
):
    liveness.live.add(DAEMON_PID)
    _write_pid(initialised, DAEMON_PID)

    out = _clean(_watch().output)

    assert "PID file .soup/loop.pid" in out
    assert "soup loop pause" in out
    assert f"end process {DAEMON_PID}" in out
    assert "delete the file if that pid is not a loop daemon" in out


def test_the_refusal_touches_neither_the_pid_file_nor_loop_yaml(
    initialised, liveness, watch_calls
):
    liveness.live.add(DAEMON_PID)
    _write_pid(initialised, DAEMON_PID)
    state_before = (initialised / ".soup" / "loop.yaml").read_bytes()

    _watch()

    assert json.loads(_pid_path(initialised).read_text(encoding="utf-8")) == {"pid": DAEMON_PID}
    assert (initialised / ".soup" / "loop.yaml").read_bytes() == state_before


def test_the_pre_wired_foreground_watch_is_refused_too(initialised, liveness, watch_calls):
    liveness.live.add(DAEMON_PID)
    _write_pid(initialised, DAEMON_PID)

    result = _watch("--pre-wired")

    assert result.exit_code == 2, (_clean(result.output), repr(result.exception))
    assert f"a loop daemon is already running: pid {DAEMON_PID}" in _clean(result.output)
    assert watch_calls == []


def test_a_foreground_watch_runs_when_there_is_no_pid_file(initialised, watch_calls):
    result = _watch()

    assert result.exit_code == 0, (_clean(result.output), repr(result.exception))
    assert len(watch_calls) == 1
    assert not _pid_path(initialised).exists()  # a foreground watch never creates it


def test_a_stale_pid_file_is_not_a_daemon_and_is_left_alone(initialised, watch_calls):
    _write_pid(initialised, DAEMON_PID)  # the fake says nothing is alive

    result = _watch()

    assert result.exit_code == 0, (_clean(result.output), repr(result.exception))
    assert len(watch_calls) == 1
    assert json.loads(_pid_path(initialised).read_text(encoding="utf-8")) == {"pid": DAEMON_PID}


def test_an_unparsable_pid_file_is_not_a_daemon(initialised, watch_calls):
    _pid_path(initialised).write_text("not json", encoding="utf-8")

    result = _watch()

    assert result.exit_code == 0, (_clean(result.output), repr(result.exception))
    assert len(watch_calls) == 1


def test_an_unusable_pid_file_is_noticed_and_does_not_stop_the_watch(
    initialised, watch_calls
):
    _pid_path(initialised).mkdir()  # a directory where the file should be

    result = _watch()

    out = _clean(result.output)
    assert result.exit_code == 0, (out, repr(result.exception))
    assert "notice: PID file must be a regular file" in out
    assert len(watch_calls) == 1


def test_the_child_that_detach_spawned_is_not_refused_by_its_own_claim(
    initialised, liveness, watch_calls, monkeypatch
):
    # --detach writes the child's pid into the file; the child, which is alive and carries the
    # marker, must start its watch.
    monkeypatch.setenv(loop_detach.DETACHED_ENV, "1")
    liveness.live.add(os.getpid())
    _write_pid(initialised, os.getpid())

    result = _watch()

    assert result.exit_code == 0, (_clean(result.output), repr(result.exception))
    assert len(watch_calls) == 1


def test_detach_is_still_refused_by_the_same_file(initialised, liveness, watch_calls):
    liveness.live.add(DAEMON_PID)
    _write_pid(initialised, DAEMON_PID)

    result = runner.invoke(app, ["loop", "watch", "--detach"])

    assert result.exit_code == 2, (_clean(result.output), repr(result.exception))
    assert f"a loop daemon is already running: pid {DAEMON_PID}" in _clean(result.output)
