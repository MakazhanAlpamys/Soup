"""The watch daemon survives, or stops with a named reason, when ``loop.yaml`` misbehaves.

A ``watch`` that read a gone or invalid ``loop.yaml`` used to ``break`` with no line anywhere,
a ``PermissionError`` (Windows, another process mid-``os.replace``) or the ``TimeoutError`` of a
locked file killed it after hours of training, and ``merge_state`` fell back to the stale
iteration copy. Now:

* an unreadable or invalid file at the top of a cycle, or one that cannot be merged after an
  iteration, stops the daemon with a recorded reason (``WatchAbortedError``; the CLI exits 1);
* transient I/O (``OSError``: a permission error, a lock timeout) logs one line and goes on;
* the iteration's record is written either way, with a note saying what was not saved.
"""

from __future__ import annotations

import builtins
import json
import logging
import re
import signal
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

import soup_cli.commands.loop as loop_command
import soup_cli.utils.loop_daemon as daemon
import soup_cli.utils.loop_state as loop_state
from soup_cli.utils.loop_daemon import WatchAbortedError, WatchConfig, watch
from soup_cli.utils.loop_state import LoopState, read_state, write_state

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _clean(text: str) -> str:
    return " ".join(_ANSI.sub("", text).split())


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _path(root: Path) -> str:
    return str(root / ".soup" / "loop.yaml")


def _state() -> LoopState:
    return LoopState(
        served_model="base", eval_suite="e.json", baseline="b", status="running",
        production_name="v5", production_adapter="v5", production_version=2, latest_version=2,
    )


def _config(root: Path, *, during_train=lambda: None, **kwargs) -> WatchConfig:
    def train(state, harvest):
        during_train()
        return {"run_id": "run-1", "adapter_path": "out", "notes": "trained"}

    return WatchConfig(
        poll_interval_sec=1.0, max_iterations=1, state_path=_path(root),
        iteration_dir=str(root / ".soup-loops"),
        harvest_fn=lambda state: {"pairs_harvested": 3, "traces_collected": 3},
        train_fn=train,
        gate_fn=lambda state, train_out: {"gate_verdict": "OK"},
        deploy_fn=lambda state, outs: {"deployed": False},
        cost_fn=lambda state: 0.0,
        **kwargs,
    )


def _records(root: Path) -> list:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (root / ".soup-loops").rglob("iteration.json")
    ]


class TestTheFileIsUnreadable:
    def test_a_file_that_cannot_be_merged_after_an_iteration_stops_the_daemon_with_a_reason(
        self, cwd
    ):
        write_state(_state(), _path(cwd))

        with pytest.raises(WatchAbortedError) as caught:
            watch(_config(cwd, during_train=lambda: Path(_path(cwd)).unlink()))

        assert "loop.yaml cannot be read" in str(caught.value)
        assert "not written" in str(caught.value)
        assert caught.value.iterations == 1
        assert not Path(_path(cwd)).exists()  # nothing was written back

    def test_the_iteration_that_ran_is_still_recorded_with_a_note(self, cwd):
        write_state(_state(), _path(cwd))

        with pytest.raises(WatchAbortedError):
            watch(_config(cwd, during_train=lambda: Path(_path(cwd)).unlink()))

        notes = [record["notes"] for record in _records(cwd)]
        assert len(notes) == 1
        assert "loop.yaml could not be updated" in notes[0]

    def test_a_corrupt_file_at_the_top_of_a_cycle_stops_the_daemon_with_a_reason(self, cwd):
        write_state(_state(), _path(cwd))

        def corrupt():
            Path(_path(cwd)).write_text("{ not json", encoding="utf-8")

        with pytest.raises(WatchAbortedError) as caught:
            watch(_config(cwd, on_cycle=corrupt))

        assert "not valid JSON" in str(caught.value)
        assert caught.value.iterations == 0
        assert Path(_path(cwd)).read_text(encoding="utf-8") == "{ not json"

    def test_the_reason_goes_to_the_log_and_to_the_progress_file(self, cwd, caplog):
        write_state(_state(), _path(cwd))

        with caplog.at_level(logging.ERROR, logger=daemon.__name__):
            with pytest.raises(WatchAbortedError):
                watch(_config(cwd, on_cycle=lambda: Path(_path(cwd)).unlink()))

        assert any("loop.yaml" in record.getMessage() for record in caplog.records)
        lines = [
            json.loads(line)
            for line in (cwd / ".soup-loops" / "progress.jsonl").read_text("utf-8").splitlines()
        ]
        assert [(line["stage"], line["event"], line.get("error")) for line in lines][-1] == (
            "daemon", "error", "state_unreadable",
        )

    def test_the_cli_exits_1_and_says_why(self, cwd, monkeypatch):
        write_state(_state(), _path(cwd))

        def aborted(config):
            raise WatchAbortedError("loop.yaml cannot be read (gone)", _state(), 3)

        monkeypatch.setattr(loop_command, "watch", aborted)
        result = CliRunner().invoke(loop_command.app, ["watch", "--foreground"])

        text = _clean(result.output)
        assert result.exit_code == 1, (text, repr(result.exception))
        assert "watch stopped after 3 iteration(s)" in text
        assert "loop.yaml cannot be read (gone)" in text


class TestTransientErrors:
    def test_a_permission_error_reading_the_file_is_retried_next_cycle(self, cwd, monkeypatch):
        write_state(_state(), _path(cwd))
        real_read = daemon.read_state
        calls = []

        def flaky(path=None):
            calls.append(1)
            if len(calls) == 2:  # the read at the top of the first cycle
                raise PermissionError("another process is replacing the file")
            return real_read(path)

        monkeypatch.setattr(daemon, "read_state", flaky)

        state, ran = watch(_config(cwd))

        assert ran == 1  # the next cycle ran the iteration
        assert state.iteration_count == 1

    def test_a_lock_timeout_while_saving_does_not_kill_the_daemon(self, cwd, monkeypatch):
        write_state(_state(), _path(cwd))

        def locked(base, new, path=None, *, skipped=None):
            raise TimeoutError("loop.yaml is locked by another process (pid 1)")

        monkeypatch.setattr(daemon, "merge_state", locked)

        state, ran = watch(_config(cwd))

        assert ran == 1
        notes = [record["notes"] for record in _records(cwd)]
        assert len(notes) == 1 and "loop.yaml could not be updated" in notes[0]
        assert "locked by another process" in notes[0]

    def test_an_os_error_in_the_final_status_write_is_swallowed(self, cwd, monkeypatch):
        write_state(_state(), _path(cwd))

        def locked(mutate, path=None):
            raise TimeoutError("loop.yaml is locked by another process (pid 1)")

        monkeypatch.setattr(daemon, "update_state", locked)

        _state_out, ran = watch(_config(cwd))

        assert ran == 1


class TestAConflictingChangeIsNamed:
    def test_the_note_names_the_field_loop_yaml_kept(self, cwd, monkeypatch):
        write_state(_state(), _path(cwd))

        def manual_promotion():
            loop_state.update_state(
                lambda fresh: replace(fresh, production_version=9, latest_version=9),
                _path(cwd),
            )

        real_run_once = daemon.run_once

        def run_once_bumping(state, cfg, **kwargs):
            # the iteration itself moves latest_version to 3, which cannot sit next to 9
            new_state, record, decision = real_run_once(state, cfg, **kwargs)
            return replace(new_state, latest_version=3), record, decision

        monkeypatch.setattr(daemon, "run_once", run_once_bumping)

        final, _ran = watch(_config(cwd, during_train=manual_promotion))

        assert read_state(_path(cwd)).latest_version == 9
        assert final.production_version == 9
        notes = [record["notes"] for record in _records(cwd)]
        assert "loop.yaml changed during the run; not overwritten: latest_version" in notes[0]


class TestReadStateRetriesAPermissionError:
    def _flaky_open(self, monkeypatch, failures: int):
        real_open = builtins.open
        seen = {"left": failures}

        def flaky(file, mode="r", *args, **kwargs):
            if str(file).endswith("loop.yaml") and "b" in mode and seen["left"] > 0:
                seen["left"] -= 1
                raise PermissionError("sharing violation")
            return real_open(file, mode, *args, **kwargs)

        monkeypatch.setattr(loop_state, "open", flaky, raising=False)
        return seen

    def test_where_it_is_retried_a_short_denial_is_ridden_out(self, cwd, monkeypatch):
        write_state(_state(), _path(cwd))
        monkeypatch.setattr(loop_state, "_PERMISSION_RETRY", True)
        seen = self._flaky_open(monkeypatch, failures=3)

        assert read_state(_path(cwd)).production_name == "v5"
        assert seen["left"] == 0

    def test_a_denial_that_does_not_end_is_raised(self, cwd, monkeypatch):
        write_state(_state(), _path(cwd))
        monkeypatch.setattr(loop_state, "_PERMISSION_RETRY", True)
        self._flaky_open(monkeypatch, failures=10_000)

        with pytest.raises(PermissionError):
            read_state(_path(cwd))

    def test_where_it_is_not_retried_the_first_denial_is_raised(self, cwd, monkeypatch):
        write_state(_state(), _path(cwd))
        monkeypatch.setattr(loop_state, "_PERMISSION_RETRY", False)
        seen = self._flaky_open(monkeypatch, failures=1)

        with pytest.raises(PermissionError):
            read_state(_path(cwd))
        assert seen["left"] == 0


class TestSignalHandlers:
    def test_watch_puts_the_signal_handlers_back(self, cwd):
        write_state(_state(), _path(cwd))
        before = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))

        watch(_config(cwd))

        assert (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)) == before
