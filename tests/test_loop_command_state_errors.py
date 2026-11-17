"""The loop commands that take the ``loop.yaml`` lock report its errors in one line.

``soup loop pause|resume|set|canary|rollback`` write ``loop.yaml`` under the state lock. A lock
held for the whole wait (``TimeoutError``) or a value that makes the state invalid
(``soup loop set trace_dir <5000 characters>``: ``ValueError`` from ``LoopState``) escaped as a
raw Typer traceback. Now each prints one red line naming the lock file or the field and exits
(1 for a locked file, 2 for a refused value, as the commands' other refusals do).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

import soup_cli.utils.loop_state as loop_state
from soup_cli.commands.loop import app
from soup_cli.utils.loop_state import LoopState, read_state, write_state

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _clean(text: str) -> str:
    return " ".join(_ANSI.sub("", text).split())


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(loop_state, "STATE_LOCK_TIMEOUT_SEC", 0.2)
    return tmp_path


def _path(root: Path) -> str:
    return str(root / ".soup" / "loop.yaml")


def _state(**kwargs) -> LoopState:
    base = LoopState(
        served_model="base", eval_suite="e.json", baseline="b", status="running",
        canary_active="cand", canary_traffic_pct=5.0, canary_rollout_id="r1",
        canary_adapter="out/cand", canary_started_ts=1_800_000_000.0,
    )
    return replace(base, **kwargs)


def _hold_the_lock(root: Path, monkeypatch) -> Path:
    lock = Path(_path(root) + ".lock")
    lock.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
    monkeypatch.setattr(loop_state, "process_is_alive", lambda pid: True)
    return lock


def _run(*args):
    result = CliRunner().invoke(app, list(args))
    return result, _clean(result.output)


LOCKED_COMMANDS = [
    pytest.param(("pause",), {}, id="pause"),
    pytest.param(("resume",), {"status": "paused"}, id="resume"),
    pytest.param(("set", "base_model", "org/other"), {}, id="set"),
    pytest.param(("canary", "newcand", "--traffic", "10%"), {}, id="canary"),
    pytest.param(("rollback",), {}, id="rollback-withdraw"),
]


class TestALockedFile:
    @pytest.mark.parametrize(("args", "overrides"), LOCKED_COMMANDS)
    def test_one_line_names_the_lock_file_and_nothing_is_written(
        self, cwd, monkeypatch, args, overrides
    ):
        write_state(_state(**overrides), _path(cwd))
        before = Path(_path(cwd)).read_bytes()
        lock = _hold_the_lock(cwd, monkeypatch)

        result, text = _run(*args)

        assert result.exit_code == 1, (text, repr(result.exception))
        assert "loop.yaml is locked by another process" in text
        assert "loop.yaml.lock" in text
        assert "Traceback" not in text
        assert result.exception is None or isinstance(result.exception, SystemExit)
        assert Path(_path(cwd)).read_bytes() == before
        assert lock.exists()  # a live holder's lock is left alone


class TestAnInvalidValue:
    def test_a_trace_dir_over_the_cap_names_the_field_and_exits_2(self, cwd):
        write_state(_state(), _path(cwd))
        too_long = "d" * 5000

        result, text = _run("set", "trace_dir", too_long)

        assert result.exit_code == 2, (text[:200], repr(result.exception))
        assert "trace_dir" in text and "exceeds" in text
        assert "Traceback" not in text
        assert read_state(_path(cwd)).trace_dir is None
        assert not Path(_path(cwd) + ".lock").exists()
