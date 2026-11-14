"""A failed training run records why: the exit code and the tail of its stderr.

The train stage used to return ``skipped: True`` with no reason (the subprocess output was
dropped) and the iteration log said only "a stage failed". Now a failed train returns
``failed: True`` (never ``skipped``) with a note holding the exit code and the last lines of
stderr, capped and stripped of control characters; the iteration record carries the note, the
loop's outcome is ``failed``, and ``soup loop status`` shows it as ``last_failure``.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace

import pytest
from typer.testing import CliRunner

import soup_cli.utils.loop_stages as ls
from soup_cli.cli import app
from soup_cli.utils.loop_state import LoopState, read_state, write_state

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


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


def _train(runner_fn, monkeypatch):
    monkeypatch.setattr(ls, "_TRAIN_RUNNER", runner_fn)
    return ls.train_dpo_from_pairs(_state(), {"pairs_path": "pairs.jsonl", "pairs_harvested": 1})


class _Result:
    def __init__(self, returncode: int, stderr: bytes) -> None:
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = b""


class TestTrainFailure:
    def test_the_exit_code_and_the_stderr_tail_are_recorded(self, cwd, monkeypatch):
        stderr = b"\n".join(f"line {number}".encode() for number in range(1, 40))
        stderr += b"\nRuntimeError: CUDA out of memory"

        out = _train(lambda argv, **kw: _Result(3, stderr), monkeypatch)

        assert out["failed"] is True and out["run_id"] is None
        assert not out.get("skipped"), "a failure is not a skip"
        assert "exit code 3" in out["notes"]
        assert "CUDA out of memory" in out["notes"]
        assert "line 1\n" not in out["notes"] + "\n", "only the tail is kept"
        assert len(out["notes"]) <= 700

    def test_control_characters_in_stderr_are_stripped(self, cwd, monkeypatch):
        stderr = b"boom \x1b]0;PWNED\x07 \x1b[31mred\x1b[0m\x08\x0b"

        out = _train(lambda argv, **kw: _Result(1, stderr), monkeypatch)

        for char in "\x1b\x07\x08\x0b":
            assert char not in out["notes"]
        assert "boom" in out["notes"] and "PWNED" in out["notes"]

    def test_bytes_that_are_not_utf8_do_not_crash(self, cwd, monkeypatch):
        out = _train(lambda argv, **kw: _Result(1, b"caf\xe9 \xff\xfe broke"), monkeypatch)

        assert out["failed"] is True and "broke" in out["notes"]

    def test_a_runner_without_stderr_still_reports_the_code(self, cwd, monkeypatch):
        class Bare:
            returncode = 5

        out = _train(lambda argv, **kw: Bare(), monkeypatch)

        assert out["failed"] is True and "exit code 5" in out["notes"]

    def test_an_exception_from_the_runner_is_named(self, cwd, monkeypatch):
        def explode(argv, **kw):
            raise TimeoutError("took 24h")

        out = _train(explode, monkeypatch)

        assert out["failed"] is True and not out.get("skipped")
        assert "TimeoutError" in out["notes"]

    def test_a_clean_run_has_no_failure_keys(self, cwd, monkeypatch):
        out = _train(lambda argv, **kw: _Result(0, b""), monkeypatch)

        assert out["skipped"] is False and not out.get("failed")


class TestStatusShowsTheFailure:
    def test_last_failure_is_a_status_row(self, cwd):
        write_state(
            replace(_state(), last_failure="train failed (exit code 3): CUDA out of memory"),
            str(cwd / ".soup" / "loop.yaml"),
        )

        result = runner.invoke(app, ["loop", "status"])

        out = " ".join(_ANSI.sub("", result.output).split())
        assert result.exit_code == 0, (out, repr(result.exception))
        assert "last_failure" in out and "exit code 3" in out

    def test_a_loop_without_a_failure_has_no_row(self, cwd):
        write_state(_state(), str(cwd / ".soup" / "loop.yaml"))

        result = runner.invoke(app, ["loop", "status"])

        assert "last_failure" not in result.output

    def test_a_hostile_last_failure_prints_no_control_bytes(self, cwd):
        write_state(
            replace(_state(), last_failure="boom \x1b]0;PWNED\x07"),
            str(cwd / ".soup" / "loop.yaml"),
        )

        result = runner.invoke(app, ["loop", "status"])

        assert "\x1b" not in result.output and "\x07" not in result.output
        assert read_state(str(cwd / ".soup" / "loop.yaml")).last_failure.startswith("boom")
