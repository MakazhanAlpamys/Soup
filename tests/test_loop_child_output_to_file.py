"""The train and ship children write their output to a file, not into the daemon's memory.

``capture_output=True`` held everything a 24 h training run printed (progress bars, a repeated
warning) in the daemon until the child exited. Now stdout and stderr go to one log file next to
the run (``.soup-loops/adapters/<run>.train.log``, ``.soup-loops/verdicts/<run>.ship.log``),
stdin is the null device, and only the last lines (capped) go into the failure note; the
iteration record names the file (``train_log`` / ``verify_log``).
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone

import pytest

import soup_cli.utils.loop_daemon as daemon
import soup_cli.utils.loop_stages as ls
from soup_cli.utils.loop_daemon import WatchConfig, run_once
from soup_cli.utils.loop_state import LoopState


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "registry.db"))
    for name in ("SOUP_LOOP_TRACE_DIR", "SOUP_LOOP_THUMBS_DB", "SOUP_LOOP_SERVE_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "pairs.jsonl").write_text(
        json.dumps({"prompt": "q", "chosen": "a", "rejected": "b"}) + "\n", encoding="utf-8"
    )
    return tmp_path


def _state() -> LoopState:
    return LoopState(
        served_model="org/base", eval_suite="e", baseline="b", base_model="org/base"
    )


class _Result:
    def __init__(self, returncode: int) -> None:
        self.returncode = returncode
        self.stderr = None
        self.stdout = None


class _NoisyChild:
    """A child that prints ~3 MiB of progress, then an error, to the handle it was given."""

    def __init__(self, returncode: int = 1) -> None:
        self.returncode = returncode
        self.kwargs: dict = {}

    def __call__(self, argv, **kwargs):
        self.kwargs = kwargs
        handle = kwargs["stdout"]
        for number in range(30_000):
            handle.write(f"step {number} " .ljust(100, "#").encode() + b"\n")
        handle.write(b"RuntimeError: CUDA out of memory\n")
        return _Result(self.returncode)


class TestTrain:
    def _run(self, child, monkeypatch):
        monkeypatch.setattr(ls, "_TRAIN_RUNNER", child)
        return ls.train_dpo_from_pairs(
            _state(), {"pairs_path": "pairs.jsonl", "pairs_harvested": 1}
        )

    def test_the_child_gets_a_file_a_null_stdin_and_no_capture(self, cwd, monkeypatch):
        child = _NoisyChild(returncode=0)

        self._run(child, monkeypatch)

        assert "capture_output" not in child.kwargs
        assert child.kwargs["stderr"] == subprocess.STDOUT
        assert child.kwargs["stdin"] == subprocess.DEVNULL
        assert child.kwargs["stdout"].name.endswith(".train.log")

    def test_the_whole_output_is_in_the_file_and_the_note_has_only_the_tail(
        self, cwd, monkeypatch
    ):
        out = self._run(_NoisyChild(), monkeypatch)

        assert out["failed"] is True
        assert "exit code 1" in out["notes"] and "CUDA out of memory" in out["notes"]
        assert "step 0 " not in out["notes"] and len(out["notes"]) <= 700
        log_path = out["train_log"]
        assert log_path.endswith(".train.log") and os.path.isfile(log_path)
        assert os.path.getsize(log_path) > 3_000_000
        assert "step 0 " in open(log_path, encoding="utf-8").read(200)

    def test_a_successful_run_names_its_log_too(self, cwd, monkeypatch):
        out = self._run(_NoisyChild(returncode=0), monkeypatch)

        assert out["skipped"] is False and out["train_log"].endswith(".train.log")

    def test_a_runner_that_prints_to_stderr_only_still_gets_its_tail_recorded(
        self, cwd, monkeypatch
    ):
        class Old:
            returncode = 3
            stderr = b"boom: old style"

        out = self._run(lambda argv, **kw: Old(), monkeypatch)

        assert "exit code 3" in out["notes"] and "boom: old style" in out["notes"]


class TestShip:
    def test_the_ship_child_writes_to_a_file_too(self, cwd, monkeypatch):
        adapter = cwd / "adapter"
        adapter.mkdir()
        seen: dict = {}

        def ship(argv, **kwargs):
            seen.update(kwargs)
            kwargs["stdout"].write(b"progress\nship exploded: bad suite\n")
            return _Result(2)

        monkeypatch.setattr(ls, "_SHIP_RUNNER", ship)
        out = ls.verify_with_ship(
            _state(), {"adapter_path": str(adapter), "run_id": "run-1", "skipped": False}
        )

        assert "capture_output" not in seen and seen["stdin"] == subprocess.DEVNULL
        assert seen["stdout"].name.endswith(".ship.log")
        assert out["gate_verdict"] == "SKIPPED"
        assert "soup ship wrote no verdict (exit 2): ship exploded: bad suite" in out["notes"]
        assert out["verify_log"].endswith("run-1.ship.log")
        assert "progress" in open(out["verify_log"], encoding="utf-8").read()


class TestTheIterationRecordNamesTheFiles:
    def test_the_record_carries_both_paths(self, cwd):
        config = WatchConfig(
            poll_interval_sec=1.0, max_iterations=1,
            state_path=str(cwd / ".soup" / "loop.yaml"),
            iteration_dir=str(cwd / ".soup-loops"),
            harvest_fn=lambda state: {"pairs_harvested": 1, "traces_collected": 1},
            train_fn=lambda state, harvest: {
                "run_id": "run-1", "adapter_path": "out",
                "train_log": ".soup-loops/adapters/run-1.train.log",
            },
            gate_fn=lambda state, train_out: {
                "gate_verdict": "OK", "verify_log": ".soup-loops/verdicts/run-1.ship.log",
            },
            deploy_fn=lambda state, outs: {"deployed": False},
            cost_fn=lambda state: 0.0,
        )

        _state_out, record, _decision = run_once(
            LoopState(served_model="base", eval_suite="e.json", baseline="b",
                      status="running"),
            config, now=datetime(2026, 11, 14, 12, 0, tzinfo=timezone.utc),
        )

        assert record.train_log == ".soup-loops/adapters/run-1.train.log"
        assert record.verify_log == ".soup-loops/verdicts/run-1.ship.log"

    def test_a_path_with_control_characters_is_left_out(self):
        assert daemon._log_ref("a\x1b[2Jb.log") is None
        assert daemon._log_ref(None) is None
        assert daemon._log_ref("x" * 600) is None
        assert daemon._log_ref(".soup-loops/a.log") == ".soup-loops/a.log"
