"""A stage that raises is a failed cycle, not the end of the daemon.

``run_once`` called the harvest, train, gate and deploy stages with no guard and ``watch`` had only
a ``finally``: any exception that escaped a stage (a locked thumbs database, a directory that
went away, a bug in a custom stage) stopped the daemon for good (status ``stopped``, a traceback
in ``.soup/loop.log``, nobody to restart it). Now the failure is recorded with the stage's name
and the exception's type (the iteration record, its manifest and line, ``last_failure``), the
same wait-then-hold rule as for a failed train applies (5 minutes after the first failure, 30
after the second, held after the third until ``soup loop resume``), and the daemon keeps
running. ``KeyboardInterrupt`` and ``SystemExit`` are not ``Exception`` and still stop it.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import datetime

import pytest

from soup_cli.utils.loop_daemon import WatchConfig, run_once, watch
from soup_cli.utils.loop_log import LoopLog
from soup_cli.utils.loop_state import LoopState, read_state, write_state

PAST = "2000-01-01T00:00:00+00:00"
LOCKED = "thumbs database could not be read (OperationalError: database is locked)"


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SOUP_LOOP_SERVE_ENDPOINT", raising=False)
    return tmp_path


def _state(**kwargs) -> LoopState:
    return LoopState(served_model="base", eval_suite="e", baseline="b", status="running", **kwargs)


class Pipeline:
    """Counting stages; ``raising`` names the stage that raises ``error``."""

    def __init__(
        self,
        raising: "str | None" = None,
        error: "BaseException | None" = None,
        train_result: "dict | None" = None,
        gate_verdict: str = "OK",
        deploy_result: "dict | None" = None,
    ):
        self.raising = raising
        self.error = error or ValueError(LOCKED)
        self.train_result = train_result
        self.gate_verdict = gate_verdict
        self.deploy_result = deploy_result
        self.calls: "list[str]" = []

    def _step(self, name: str):
        self.calls.append(name)
        if self.raising == name:
            raise self.error

    def harvest(self, state):
        self._step("harvest")
        return {"pairs_harvested": 3}

    def train(self, state, harvested):
        self._step("train")
        if self.train_result is not None:
            return self.train_result
        return {"run_id": "run-1", "skipped": False, "adapter_path": "adapters/a"}

    def gate(self, state, trained):
        self._step("gate")
        return {"gate_verdict": self.gate_verdict}

    def deploy(self, state, trained):
        self._step("deploy")
        return self.deploy_result or {"deployed": True}

    def cost(self, state):
        self._step("cost")
        return 0.0

    def config(self, root=None, **kwargs) -> WatchConfig:
        paths = {}
        if root is not None:
            paths = {
                "state_path": str(root / ".soup" / "loop.yaml"),
                "iteration_dir": str(root / ".soup-loops"),
            }
        return WatchConfig(
            max_iterations=kwargs.pop("max_iterations", 1), poll_interval_sec=1.0,
            harvest_fn=self.harvest, train_fn=self.train, gate_fn=self.gate,
            deploy_fn=self.deploy, cost_fn=self.cost, **paths, **kwargs,
        )


def _lines(root, name="iterations.jsonl"):
    path = root / ".soup-loops" / name
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


STAGES = ("cost", "harvest", "train", "gate", "deploy")
HELD_STAGES = ("cost", "harvest", "train", "gate")  # a deploy has its own retry (3 attempts)


@pytest.mark.parametrize("stage", STAGES)
def test_a_stage_that_raises_does_not_escape_run_once(cwd, stage):
    pipeline = Pipeline(raising=stage)

    new_state, record, _ = run_once(_state(), pipeline.config())

    assert f"the {stage} stage raised ValueError" in record.notes
    assert f"the {stage} stage raised ValueError" in new_state.last_failure
    assert "database is locked" in new_state.last_failure  # the message helps, the type leads


@pytest.mark.parametrize("stage", STAGES)
def test_the_failure_gets_an_iteration_line_a_counter_and_a_failed_outcome(cwd, stage):
    pipeline = Pipeline(raising=stage)
    log = LoopLog(str(cwd / ".soup-loops"))

    new_state, record, decision = run_once(_state(), pipeline.config(), log=log)

    assert decision.proceed is True  # a failed iteration is written, as a failed train is
    assert [line["outcome"] for line in _lines(cwd)] == ["promoted", "failed"]
    assert f"the {stage} stage raised ValueError" in _lines(cwd)[-1]["reason"]
    assert (new_state.iteration_count, new_state.runs_today) == (1, 1)


@pytest.mark.parametrize("stage", ("cost", "harvest", "train"))
def test_no_later_stage_runs_after_an_earlier_one_raised(cwd, stage):
    pipeline = Pipeline(raising=stage)

    run_once(_state(), pipeline.config())

    order = ["cost", "harvest", "train", "gate", "deploy"]
    assert pipeline.calls == order[: order.index(stage) + 1]


def test_a_gate_that_raised_deploys_nothing(cwd):
    pipeline = Pipeline(raising="gate")

    new_state, _, _ = run_once(_state(), pipeline.config())

    assert "deploy" not in pipeline.calls
    assert new_state.attempt_cursor_ts is None and new_state.pending_deploy is None


def test_a_deploy_that_raised_is_retried_like_a_refused_deploy(cwd):
    pipeline = Pipeline(raising="deploy")

    new_state, record, _ = run_once(_state(), pipeline.config())

    assert new_state.pending_deploy is not None and new_state.pending_deploy["attempts"] == 1
    assert "deploy failed (attempt 1 of 3): the deploy stage raised ValueError" in record.notes
    assert new_state.failed_train is None  # the deploy has its own retry, not the train back-off


@pytest.mark.parametrize("stage", HELD_STAGES)
def test_the_failure_is_remembered_for_the_wait_then_hold_rule(cwd, stage):
    pipeline = Pipeline(raising=stage)

    new_state, _, _ = run_once(_state(), pipeline.config())

    remembered = new_state.failed_stage
    assert new_state.failed_train is None
    assert remembered["attempts"] == 1 and remembered["held"] is False
    assert remembered["retry_after"] is not None
    assert remembered["reason"].startswith(f"stage error: the {stage} stage raised ValueError")


@pytest.mark.parametrize("stage", HELD_STAGES)
def test_the_same_stage_failing_again_waits_then_holds(cwd, stage):
    pipeline = Pipeline(raising=stage)
    state = _state()
    config = pipeline.config()

    state, _, _ = run_once(state, config)  # first failure: waits
    calls_after_first = len(pipeline.calls)
    state, _, waiting = run_once(state, config)
    assert len(pipeline.calls) == calls_after_first  # nothing ran: still waiting
    assert waiting.proceed is False
    assert "stage_backoff" in state.last_idle["reason"]
    assert f"the {stage} stage raised ValueError" in state.last_idle["reason"]
    assert f"the {stage} stage raised ValueError" in state.last_failure  # still in view

    for expected_attempts in (2, 3):
        state = replace(state, failed_stage={**state.failed_stage, "retry_after": PAST})
        before = len(pipeline.calls)
        state, _, _ = run_once(state, config)
        assert len(pipeline.calls) > before  # the wait was over: it tried again
        assert state.failed_stage["attempts"] == expected_attempts

    assert state.failed_stage["held"] is True and state.failed_stage["retry_after"] is None
    before = len(pipeline.calls)
    state, _, held = run_once(state, config)
    assert len(pipeline.calls) == before
    assert held.proceed is False
    assert "stage_held" in state.last_idle["reason"]
    assert "soup loop resume" in state.last_idle["reason"]


def test_the_first_wait_is_five_minutes_and_the_second_thirty(cwd):
    pipeline = Pipeline(raising="harvest")
    state = _state()
    config = pipeline.config()

    state, _, _ = run_once(state, config)
    first = state.failed_stage
    state = replace(state, failed_stage={**first, "retry_after": PAST})
    state, _, _ = run_once(state, config)
    second = state.failed_stage

    def wait_seconds(record):
        last = datetime.fromisoformat(record["last_at"])
        return (datetime.fromisoformat(record["retry_after"]) - last).total_seconds()

    assert wait_seconds(first) == 300.0 and wait_seconds(second) == 1800.0


def test_a_different_stage_failure_continues_the_loop_wide_count(cwd):
    state = _state()
    state, _, _ = run_once(state, Pipeline(raising="harvest").config())
    state = replace(state, failed_stage={**state.failed_stage, "retry_after": PAST})

    state, _, _ = run_once(state, Pipeline(raising="gate").config())

    assert state.failed_stage["attempts"] == 2
    assert "the gate stage raised" in state.failed_stage["reason"]


def test_alternating_stage_errors_reach_the_loop_hold(cwd):
    state = _state()
    for stage in ("harvest", "train", "harvest"):
        state, _, _ = run_once(state, Pipeline(raising=stage).config())
        if state.failed_stage is not None and not state.failed_stage["held"]:
            state = replace(state, failed_stage={**state.failed_stage, "retry_after": PAST})

    assert state.failed_stage is not None
    assert state.failed_stage["attempts"] == 3
    assert state.failed_stage["held"] is True
    assert state.failed_stage["retry_after"] is None
    assert "the harvest stage raised" in state.failed_stage["reason"]


def test_stage_error_does_not_replace_pending_train_failure(cwd):
    train_failure = {
        "inputs": "a" * 64, "attempts": 2, "last_at": PAST, "retry_after": PAST,
        "held": False, "reason": "train failed (exit code 1)",
    }

    state, _, _ = run_once(_state(failed_train=train_failure), Pipeline(raising="harvest").config())

    assert state.failed_train == train_failure
    assert state.failed_stage is not None
    assert state.failed_stage["attempts"] == 1


def test_training_retry_after_stage_error_keeps_its_own_attempts_and_hold(cwd):
    train_failure = {
        "inputs": "a" * 64, "attempts": 2, "last_at": PAST, "retry_after": PAST,
        "held": False, "reason": "train failed (exit code 1)",
    }
    state, _, _ = run_once(_state(failed_train=train_failure), Pipeline(raising="harvest").config())
    state = replace(state, failed_stage={**state.failed_stage, "retry_after": PAST})

    failed_training = Pipeline(train_result={
        "run_id": None, "skipped": False, "failed": True, "train_inputs": "a" * 64,
        "failure": "train failed (exit code 1)", "notes": "train failed",
    })
    state, _, _ = run_once(state, failed_training.config())

    assert state.failed_train["attempts"] == 3 and state.failed_train["held"] is True
    assert state.failed_stage is not None and state.failed_stage["attempts"] == 2


def test_train_failure_harvest_error_and_failed_train_retry_reach_loop_hold(cwd):
    failed_train = {
        "run_id": None, "skipped": False, "failed": True, "train_inputs": "a" * 64,
        "failure": "train failed", "notes": "train failed",
    }
    state, _, _ = run_once(_state(), Pipeline(train_result=failed_train).config())
    assert state.failed_stage is not None and state.failed_stage["attempts"] == 1

    state = replace(
        state,
        failed_stage={**state.failed_stage, "retry_after": PAST},
        failed_train={**state.failed_train, "retry_after": PAST},
    )
    state, _, _ = run_once(state, Pipeline(raising="harvest").config())
    assert state.failed_stage is not None and state.failed_stage["attempts"] == 2

    state = replace(state, failed_stage={**state.failed_stage, "retry_after": PAST})
    state, _, _ = run_once(state, Pipeline(train_result=failed_train).config())

    assert state.failed_stage is not None
    assert state.failed_stage["attempts"] == 3 and state.failed_stage["held"] is True
    assert state.failed_train["attempts"] == 2


def test_idle_polls_between_raised_errors_clear_the_streak(cwd):
    idle_train = {"run_id": None, "skipped": True, "notes": "not enough new data"}
    for stage in ("harvest", "train", "gate"):
        state, _, _ = run_once(_state(), Pipeline(raising=stage).config())
        assert state.failed_stage is not None
        assert state.failed_stage["attempts"] == 1 and state.failed_stage["held"] is False

        state = replace(state, failed_stage={**state.failed_stage, "retry_after": PAST})
        state, _, _ = run_once(state, Pipeline(train_result=idle_train).config())
        assert state.failed_stage is None

    state, _, _ = run_once(_state(), Pipeline(raising="harvest").config())
    assert state.failed_stage["attempts"] == 1 and state.failed_stage["held"] is False


def test_three_raised_errors_separated_by_clean_idle_polls_never_hold(cwd):
    idle_train = {"run_id": None, "skipped": True, "notes": "not enough new data"}
    state = _state()

    for stage in ("harvest", "train", "gate"):
        state, _, _ = run_once(state, Pipeline(raising=stage).config())
        assert state.failed_stage is not None
        assert state.failed_stage["attempts"] == 1
        assert state.failed_stage["held"] is False
        state = replace(state, failed_stage={**state.failed_stage, "retry_after": PAST})
        state, _, _ = run_once(state, Pipeline(train_result=idle_train).config())
        assert state.failed_stage is None

    assert state.failed_stage is None


@pytest.mark.parametrize(("train_result", "expected_attempts"), [
    ({"run_id": None, "skipped": False, "failed": True, "train_inputs": "a" * 64,
      "failure": "train failed", "notes": "train failed"}, 2),
    ({"run_id": None, "skipped": True, "train_held": True, "notes": "training held"}, 1),
])
def test_training_failure_counts_but_training_hold_preserves_stage_streak(
    cwd, train_result, expected_attempts
):
    state, _, _ = run_once(_state(), Pipeline(raising="harvest").config())
    state = replace(state, failed_stage={**state.failed_stage, "retry_after": PAST})

    state, _, _ = run_once(state, Pipeline(train_result=train_result).config())

    assert state.failed_stage is not None
    assert state.failed_stage["attempts"] == expected_attempts


def test_v078_stage_failure_state_is_migrated_without_losing_attempts(cwd):
    import hashlib

    old_record = {
        "inputs": hashlib.sha256(b"loop stage error:harvest").hexdigest(), "attempts": 2,
        "last_at": PAST,
        "retry_after": PAST, "held": False,
        "reason": "stage error: the harvest stage raised ValueError",
    }

    state, _, _ = run_once(_state(failed_train=old_record), Pipeline(raising="train").config())

    assert state.failed_train is None
    assert state.failed_stage is not None
    assert state.failed_stage["attempts"] == 3
    assert state.failed_stage["held"] is True
    assert "the train stage raised" in state.failed_stage["reason"]


def test_v078_state_without_failed_stage_field_loads_with_default(cwd):
    path = cwd / ".soup" / "loop.yaml"
    write_state(_state(), str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    data.pop("failed_stage")
    path.write_text(json.dumps(data), encoding="utf-8")

    assert read_state(str(path)).failed_stage is None


def test_v078_held_stage_error_file_roundtrips_runs_no_stage_and_shows_status(cwd):
    import hashlib

    from typer.testing import CliRunner

    from soup_cli.commands.loop import app as loop_app

    path = cwd / ".soup" / "loop.yaml"
    old_record = {
        "inputs": hashlib.sha256(b"loop stage error:harvest").hexdigest(),
        "attempts": 3,
        "last_at": PAST,
        "retry_after": None,
        "held": True,
        "reason": "stage error: the harvest stage raised ValueError",
    }
    write_state(_state(failed_train=old_record), str(path))
    original = json.loads(path.read_text(encoding="utf-8"))
    original.pop("failed_stage", None)
    path.write_text(json.dumps(original), encoding="utf-8")

    state = read_state(str(path))
    assert state.failed_stage is None and state.failed_train == old_record
    write_state(state, str(path))
    assert read_state(str(path)).failed_train == old_record

    pipeline = Pipeline()
    held_state, _, decision = run_once(state, pipeline.config())
    assert decision.proceed is False
    assert pipeline.calls == []
    assert "stage_held" in held_state.last_idle["reason"]

    result = CliRunner().invoke(loop_app, ["status"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "stage_retry" in result.output and "held" in result.output.lower()
    assert "train_retry" not in result.output


def test_a_cycle_that_gets_through_every_stage_clears_the_record(cwd):
    pipeline = Pipeline(raising="harvest")
    state = _state()
    state, _, _ = run_once(state, pipeline.config())
    state = replace(state, failed_stage={**state.failed_stage, "retry_after": PAST})

    state, _, _ = run_once(state, Pipeline().config())

    assert state.failed_stage is None


def test_clean_cycle_resets_streak_before_a_new_stage_failure(cwd):
    state = _state()
    state, _, _ = run_once(state, Pipeline(raising="harvest").config())
    state = replace(state, failed_stage={**state.failed_stage, "retry_after": PAST})
    state, _, _ = run_once(state, Pipeline().config())
    assert state.failed_stage is None

    state, _, _ = run_once(state, Pipeline(raising="harvest").config())
    assert state.failed_stage["attempts"] == 1
    assert state.failed_stage["held"] is False


def test_two_failures_clean_cycle_then_failure_starts_a_new_streak(cwd):
    state = _state()
    for stage in ("harvest", "gate"):
        state, _, _ = run_once(state, Pipeline(raising=stage).config())
        assert state.failed_stage is not None
        state = replace(state, failed_stage={**state.failed_stage, "retry_after": PAST})

    state, _, _ = run_once(state, Pipeline().config())
    assert state.failed_stage is None

    state, _, _ = run_once(state, Pipeline(raising="harvest").config())
    assert state.failed_stage is not None
    assert state.failed_stage["attempts"] == 1
    assert state.failed_stage["held"] is False


def test_non_ok_gate_clears_stage_streak_without_counting_as_failure(cwd):
    state, _, _ = run_once(_state(), Pipeline(raising="harvest").config())
    state = replace(state, failed_stage={**state.failed_stage, "retry_after": PAST})

    state, _, _ = run_once(state, Pipeline(gate_verdict="DON'T SHIP").config())

    assert state.failed_stage is None


def test_pending_deploy_retry_clears_stage_streak_and_keeps_deploy_retry(cwd):
    refused = {"deployed": False, "failed": True, "notes": "serve refused candidate"}
    state, _, _ = run_once(_state(), Pipeline(deploy_result=refused).config())
    assert state.pending_deploy is not None
    state = replace(
        state,
        failed_stage={
            "inputs": "a" * 64,
            "attempts": 1,
            "last_at": PAST,
            "retry_after": PAST,
            "held": False,
            "reason": "stage error: the harvest stage raised ValueError",
        },
    )

    state, _, _ = run_once(state, Pipeline(deploy_result=refused).config())

    assert state.pending_deploy is not None
    assert state.pending_deploy["attempts"] == 2
    assert state.failed_stage is None


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"reason": "harvest failed"}, "reason must start with 'stage error: '"),
        ({"extra": "unexpected"}, "must hold exactly"),
    ],
)
def test_failed_stage_validation_rejects_invalid_reason_and_keys(changes, message):
    record = {
        "inputs": "a" * 64, "attempts": 1, "last_at": PAST,
        "retry_after": PAST, "held": False,
        "reason": "stage error: the harvest stage raised ValueError",
    }
    record.update(changes)

    with pytest.raises(ValueError, match=message):
        _state(failed_stage=record)


def test_a_train_failure_record_is_not_taken_for_a_stage_error(cwd):
    """The hold of a failed train is the train stage's own (by dataset); stages still run."""
    record = {
        "inputs": "a" * 64, "attempts": 3, "last_at": PAST, "retry_after": None, "held": True,
        "reason": "train failed",
    }
    pipeline = Pipeline()

    run_once(_state(failed_train=record), pipeline.config())

    assert pipeline.calls[:2] == ["cost", "harvest"]


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(3)], ids=["ctrl-c", "exit"])
@pytest.mark.parametrize("stage", STAGES)
def test_keyboard_interrupt_and_system_exit_still_stop_the_daemon(cwd, stage, error):
    pipeline = Pipeline(raising=stage, error=error)

    with pytest.raises(type(error)):
        run_once(_state(), pipeline.config())


def test_a_raising_harvest_does_not_stop_a_watching_daemon(cwd):
    write_state(_state(), str(cwd / ".soup" / "loop.yaml"))
    pipeline = Pipeline(raising="harvest")

    final_state, ran = watch(pipeline.config(cwd, max_iterations=3))

    assert ran == 3  # it went on polling
    assert pipeline.calls.count("harvest") == 1  # the next two polls waited out the back-off
    saved = read_state(str(cwd / ".soup" / "loop.yaml"))
    assert saved.status == "stopped"  # the natural end (max_iterations), not a crash
    assert "the harvest stage raised ValueError" in saved.last_failure
    assert saved.failed_stage["attempts"] == 1
    assert [line["outcome"] for line in _lines(cwd)] == ["promoted", "failed"]
    assert len([name for name in os.listdir(cwd / ".soup-loops") if name.startswith("iter-")]) == 1
    assert final_state.failed_stage["attempts"] == 1


def test_the_traceback_goes_to_the_daemon_log(cwd, caplog):
    pipeline = Pipeline(raising="harvest")

    with caplog.at_level("WARNING"):
        run_once(_state(), pipeline.config())

    entry = next(item for item in caplog.records if "harvest stage raised" in item.getMessage())
    assert entry.exc_info is not None and entry.exc_info[0] is ValueError


def test_the_message_is_one_clean_short_line(cwd):
    noisy = RuntimeError("line one\nline two\x1b[31m " + "x" * 600)
    pipeline = Pipeline(raising="harvest", error=noisy)

    new_state, _, _ = run_once(_state(), pipeline.config())

    reason = new_state.failed_stage["reason"]
    assert "\n" not in reason and "\x1b" not in reason and len(reason) <= 300
    assert len(new_state.last_failure) <= 500 and "\x1b" not in new_state.last_failure


class TestWhatStatusSays:
    """``soup loop status`` shows training and raised-stage retries independently."""

    def test_a_waiting_stage_error_is_described_by_stage(self, cwd):
        from soup_cli.utils.loop_retry import describe

        state, _, _ = run_once(_state(), Pipeline(raising="harvest").config())

        row = describe(state.failed_stage)

        assert row.startswith("waiting: the harvest stage raised ValueError")
        assert "(1 time in a row)" in row and "next attempt after" in row
        assert "same inputs" not in row  # it is not a failed training

    def test_a_held_stage_error_says_how_to_clear_it(self, cwd):
        from soup_cli.utils.loop_retry import describe

        state = _state()
        config = Pipeline(raising="gate").config()
        for _ in range(3):
            state, _, _ = run_once(state, config)
            state = replace(state, failed_stage={**state.failed_stage, "retry_after": PAST})

        row = describe(state.failed_stage)

        assert row.startswith("held: the gate stage raised ValueError")
        assert "soup loop resume clears it" in row
        assert "the data, the recipe or the base model" not in row

    def test_a_failed_training_is_still_described_as_before(self, cwd):
        from soup_cli.utils.loop_retry import describe

        record = {
            "inputs": "a" * 64, "attempts": 2, "last_at": PAST, "retry_after": None,
            "held": True, "reason": "train failed (exit code 1)",
        }

        row = describe(record)

        assert row == (
            "held: failed 2 times on the same inputs (train failed (exit code 1)); no new "
            "attempt until the data, the recipe or the base model changes, or soup loop "
            "resume clears it"
        )
