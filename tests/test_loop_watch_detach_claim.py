"""``soup loop watch --detach`` claims the PID file BEFORE it spawns the daemon.

The check-spawn-write sequence was not atomic: two ``--detach`` started at the same moment
both saw "no daemon", both spawned one, and the second PID file hid the first daemon from
``soup loop status``. Now the PID file is created exclusively (``O_CREAT | O_EXCL``) holding
the parent's pid; the child's pid replaces it after the spawn; a failed spawn removes it.

``process_is_alive`` is replaced by a fake and ``subprocess.Popen`` is mocked in every test:
nothing here signals a process or starts a daemon.
"""

from __future__ import annotations

import json
import os
import re
import types
from pathlib import Path
from unittest import mock

import pytest
from typer.testing import CliRunner

import soup_cli.utils.loop_detach as loop_detach
from soup_cli.cli import app

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
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
    fake = types.SimpleNamespace(live=set())
    monkeypatch.setattr(loop_detach, "process_is_alive", lambda pid: pid in fake.live)
    monkeypatch.setattr(loop_detach, "_CLAIM_WRITE_GRACE_SEC", 0.0)
    return fake


def _detach(*, pid: int = 123, side_effect=None):
    with mock.patch("soup_cli.commands.loop.subprocess.Popen") as popen:
        popen.return_value.pid = pid
        if side_effect is not None:
            popen.side_effect = side_effect
        result = runner.invoke(app, ["loop", "watch", "--detach"])
    return result, popen


def _pid_path(root: Path) -> Path:
    return root / ".soup" / "loop.pid"


def _pid_in(path: Path) -> int:
    return json.loads(path.read_text(encoding="utf-8"))["pid"]


# ---------------------------------------------------------------------------
# The claim comes first
# ---------------------------------------------------------------------------


def test_the_pid_file_exists_and_names_the_parent_when_the_daemon_is_spawned(initialised):
    seen = {}

    def spawn(*args, **kwargs):
        path = _pid_path(initialised)
        seen["exists"] = path.exists()
        seen["pid"] = _pid_in(path) if path.exists() else None
        return types.SimpleNamespace(pid=123)

    result, _popen = _detach(side_effect=spawn)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert seen == {"exists": True, "pid": os.getpid()}


def test_the_childs_pid_replaces_the_claim_after_the_spawn(initialised):
    result, popen = _detach(pid=777)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert popen.call_count == 1
    assert _pid_in(_pid_path(initialised)) == 777


def test_a_file_that_appears_between_the_check_and_the_write_refuses_the_spawn(
    initialised, liveness, monkeypatch
):
    # The race: another --detach creates its claim at the very moment this one creates its
    # own. The exclusive create finds the file; the live holder wins, nothing is spawned.
    liveness.live.add(OTHER_PID)
    real_open = loop_detach.open_no_follow

    def other_daemon_wins(path, flags, *args, **kwargs):
        _pid_path(initialised).parent.mkdir(exist_ok=True)
        _pid_path(initialised).write_text(json.dumps({"pid": OTHER_PID}), encoding="utf-8")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(loop_detach, "open_no_follow", other_daemon_wins)

    result, popen = _detach()

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert f"a loop daemon is already running: pid {OTHER_PID}" in _clean(result.output)
    assert popen.call_count == 0
    assert _pid_in(_pid_path(initialised)) == OTHER_PID


def test_a_second_claim_while_the_first_is_in_flight_is_refused(initialised, liveness):
    # The claim holds the first process's own pid, which is alive: a parallel --detach that
    # finds the file mid-spawn refuses instead of treating it as stale.
    liveness.live.add(os.getpid())
    first = loop_detach.claim_pid_file()
    assert first is None

    with pytest.raises(loop_detach.PidFileHeldError) as excinfo:
        loop_detach.claim_pid_file()

    assert excinfo.value.pid == os.getpid()


def test_a_failed_spawn_removes_the_claim(initialised):
    result, _popen = _detach(side_effect=OSError("exec failed"))

    assert result.exit_code != 0
    assert not _pid_path(initialised).exists()


def test_a_log_that_cannot_be_opened_removes_the_claim(initialised, monkeypatch):
    def refuse(path=None):
        raise ValueError("log file must not be a symlink")

    monkeypatch.setattr("soup_cli.commands.loop.open_log", refuse)

    result, popen = _detach()

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert popen.call_count == 0
    assert not _pid_path(initialised).exists()


def test_the_claim_never_removes_a_file_another_daemon_wrote_meanwhile(initialised, liveness):
    liveness.live.add(OTHER_PID)
    claimed = loop_detach.claim_pid_file()
    assert claimed is None
    _pid_path(initialised).write_text(json.dumps({"pid": OTHER_PID}), encoding="utf-8")

    assert loop_detach.remove_pid_file_if_own(os.getpid()) is False
    assert _pid_in(_pid_path(initialised)) == OTHER_PID


# ---------------------------------------------------------------------------
# Stale and unusable files
# ---------------------------------------------------------------------------


def test_a_stale_file_is_replaced_and_described(initialised, liveness):
    _pid_path(initialised).write_text(json.dumps({"pid": OTHER_PID}), encoding="utf-8")

    replaced = loop_detach.claim_pid_file()

    assert replaced is not None and replaced.state == loop_detach.STALE
    assert replaced.pid == OTHER_PID
    assert _pid_in(_pid_path(initialised)) == os.getpid()


def test_an_unparsable_file_is_replaced_after_the_grace(initialised):
    _pid_path(initialised).write_text("not json", encoding="utf-8")

    replaced = loop_detach.claim_pid_file()

    assert replaced is not None and replaced.pid is None
    assert _pid_in(_pid_path(initialised)) == os.getpid()


def test_a_claim_still_being_written_is_not_mistaken_for_a_stale_file(
    initialised, liveness, monkeypatch
):
    # Another claimant created the file but has not written its pid yet. During the grace
    # wait it does; this claimant must then see a live holder and refuse.
    liveness.live.add(OTHER_PID)
    _pid_path(initialised).write_text("", encoding="utf-8")

    def writer_finishes(_seconds):
        _pid_path(initialised).write_text(json.dumps({"pid": OTHER_PID}), encoding="utf-8")

    monkeypatch.setattr(loop_detach.time, "sleep", writer_finishes)

    with pytest.raises(loop_detach.PidFileHeldError) as excinfo:
        loop_detach.claim_pid_file()

    assert excinfo.value.pid == OTHER_PID


@pytest.mark.requires_symlink
def test_a_symlinked_pid_file_is_still_refused(initialised):
    target = initialised / "elsewhere"
    target.write_text("x", encoding="utf-8")
    os.symlink(target, _pid_path(initialised))

    with pytest.raises(ValueError, match="symlink"):
        loop_detach.claim_pid_file()

    assert target.read_text(encoding="utf-8") == "x"


def test_the_claim_is_created_owner_only(initialised, monkeypatch):
    seen = []
    real_open = loop_detach.open_no_follow

    def spy(path, flags, mode=0o600, **kwargs):
        seen.append((flags & os.O_EXCL, mode))
        return real_open(path, flags, mode, **kwargs)

    monkeypatch.setattr(loop_detach, "open_no_follow", spy)

    loop_detach.claim_pid_file()

    assert seen == [(os.O_EXCL, 0o600)]
