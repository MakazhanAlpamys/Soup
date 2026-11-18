"""A training run that failed is not repeated on the same inputs (review 5, F3).

The daemon polled every 10 s and started a failed train again on each poll, on an identical
dataset: 4 attempts in 35 s, each with its own ``iter-*`` directory. Now ``loop.yaml`` remembers
the failure against a fingerprint of the dataset, the method, the base and the recipe
(``failed_train``): the same inputs wait 5 min after the first failure, 30 min after the second,
and after the third the loop holds until the inputs change or ``soup loop resume`` clears it.
``soup loop status`` shows the ``train_retry`` row.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
from typer.testing import CliRunner

import soup_cli.utils.loop_retry as retry
import soup_cli.utils.loop_stages as ls
from soup_cli.commands.loop import app as loop_app
from soup_cli.utils.loop_daemon import WatchConfig, run_once
from soup_cli.utils.loop_state import LoopState, read_state, write_state

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
NOW = datetime(2026, 11, 16, 12, 0, 0, tzinfo=timezone.utc)
ROW = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}


def _plain(text: str) -> str:
    return " ".join(_ANSI.sub("", text).replace("│", " ").split())


def _state(**kwargs) -> LoopState:
    base = dict(
        served_model="org/base", eval_suite="e", baseline="b", base_model="org/base",
        train_method="sft", status="running",
    )
    base.update(kwargs)
    return LoopState(**base)


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "registry.db"))
    for name in ("SOUP_LOOP_TRACE_DIR", "SOUP_LOOP_THUMBS_DB", "SOUP_LOOP_SERVE_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "sft.jsonl").write_text(json.dumps(ROW) + "\n", encoding="utf-8")
    return tmp_path


class _Child:
    """A ``soup train`` stand-in that counts how often it was started."""

    def __init__(self, returncode: int) -> None:
        self.returncode = returncode
        self.calls = 0

    def __call__(self, argv, **kwargs):
        self.calls += 1
        kwargs["stdout"].write(b"Loading config from x.yaml...\n")
        return type("R", (), {"returncode": self.returncode, "stderr": None})()


def _print(fp="a" * 64, attempts=1, held=False, retry_after="2026-11-16T12:05:00+00:00"):
    return {
        "inputs": fp, "attempts": attempts, "last_at": "2026-11-16T12:00:00+00:00",
        "retry_after": None if held else retry_after, "held": held,
        "reason": "train failed (exit code 1): Hardware-fit gate",
    }


class TestFingerprint:
    def _write(self, tmp_path, rows, name="d.jsonl"):
        path = tmp_path / name
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return str(path)

    def test_the_same_inputs_have_the_same_fingerprint_wherever_the_file_is(self, tmp_path):
        first = self._write(tmp_path, [ROW])
        other = tmp_path / "copy.jsonl"
        other.write_bytes((tmp_path / "d.jsonl").read_bytes())

        left = retry.train_inputs_fingerprint(_state(), first, "sft")
        right = retry.train_inputs_fingerprint(_state(), str(other), "sft")

        assert left == right and left is not None

    def test_another_row_is_another_fingerprint(self, tmp_path):
        one = self._write(tmp_path, [ROW])
        two = self._write(tmp_path, [ROW, {**ROW, "source": "more"}], name="two.jsonl")

        first = retry.train_inputs_fingerprint(_state(), one, "sft")
        second = retry.train_inputs_fingerprint(_state(), two, "sft")

        assert first != second

    @pytest.mark.parametrize(
        "changed",
        [
            {"base_model": "org/other"},
            {"sft_recipe": {"lr": 0.0001}},
            {"sft_recipe": {"epochs": 3}},
        ],
    )
    def test_a_changed_base_or_recipe_is_another_fingerprint(self, tmp_path, changed):
        path = self._write(tmp_path, [ROW])

        before = retry.train_inputs_fingerprint(_state(), path, "sft")
        after = retry.train_inputs_fingerprint(_state(**changed), path, "sft")

        assert before != after

    def test_the_method_is_part_of_it(self, tmp_path):
        path = self._write(tmp_path, [ROW])

        sft = retry.train_inputs_fingerprint(_state(), path, "sft")
        kto = retry.train_inputs_fingerprint(_state(), path, "kto")

        assert sft != kto

    def test_an_unreadable_dataset_has_none(self, tmp_path):
        assert retry.train_inputs_fingerprint(_state(), str(tmp_path / "missing"), "sft") is None


class TestAfterFailure:
    def test_the_first_failure_waits_five_minutes(self):
        record = retry.after_failure(None, "f" * 64, NOW, "train failed (exit code 1)")

        assert record["attempts"] == 1 and record["held"] is False
        assert record["retry_after"] == (NOW + timedelta(seconds=300)).isoformat()
        assert record["last_at"] == NOW.isoformat() and record["inputs"] == "f" * 64

    def test_the_second_failure_on_the_same_inputs_waits_half_an_hour(self):
        first = retry.after_failure(None, "f" * 64, NOW, "x")

        second = retry.after_failure(first, "f" * 64, NOW, "x")

        assert second["attempts"] == 2 and second["held"] is False
        assert second["retry_after"] == (NOW + timedelta(seconds=1800)).isoformat()

    def test_the_third_holds_and_has_no_retry_time(self):
        record = None
        for _ in range(retry.ATTEMPT_LIMIT):
            record = retry.after_failure(record, "f" * 64, NOW, "x")

        assert retry.ATTEMPT_LIMIT == 3
        assert record["attempts"] == 3 and record["held"] is True and record["retry_after"] is None

    def test_other_inputs_start_the_count_again(self):
        held = None
        for _ in range(3):
            held = retry.after_failure(held, "a" * 64, NOW, "x")

        fresh = retry.after_failure(held, "b" * 64, NOW, "x")

        assert fresh["attempts"] == 1 and fresh["held"] is False

    def test_the_reason_is_one_short_line(self):
        record = retry.after_failure(None, "f" * 64, NOW, "a\n\tb  " + "c" * 1000)

        assert record["reason"].startswith("a b ccc") and len(record["reason"]) == 300


class TestHoldReason:
    def test_nothing_remembered_goes_ahead(self):
        assert retry.hold_reason(None, "a" * 64, NOW) is None
        assert retry.hold_reason(_print(), None, NOW) is None

    def test_other_inputs_go_ahead(self):
        assert retry.hold_reason(_print(fp="a" * 64), "b" * 64, NOW) is None

    def test_inside_the_wait_it_says_when_the_next_attempt_is(self):
        note = retry.hold_reason(_print(), "a" * 64, NOW)

        assert note == (
            "train_backoff: the same inputs failed 1 time (last 2026-11-16T12:00:00+00:00: "
            "train failed (exit code 1): Hardware-fit gate); next attempt after "
            "2026-11-16T12:05:00+00:00"
        )

    def test_after_the_wait_it_goes_ahead(self):
        assert retry.hold_reason(_print(), "a" * 64, NOW + timedelta(minutes=6)) is None

    def test_held_says_what_clears_it(self):
        note = retry.hold_reason(_print(attempts=3, held=True), "a" * 64, NOW)

        assert note.startswith("train_held: the same inputs failed 3 times")
        assert (
            "until the data, the recipe or the base model changes, or soup loop resume clears it"
            in note
        )

    def test_held_stays_held_whatever_the_clock(self):
        assert retry.hold_reason(
            _print(attempts=3, held=True), "a" * 64, NOW + timedelta(days=30)
        ) is not None


class TestTheStage:
    def _ctx(self):
        return {"pairs_path": "sft.jsonl", "pairs_harvested": 1}

    def _fp(self, state):
        return retry.train_inputs_fingerprint(state, "sft.jsonl", "sft")

    def test_a_failed_run_reports_its_fingerprint_and_reason(self, cwd, monkeypatch):
        child = _Child(returncode=1)
        monkeypatch.setattr(ls, "_TRAIN_RUNNER", child)
        state = _state()

        out = ls.train_dpo_from_pairs(state, self._ctx())

        assert out["failed"] is True and child.calls == 1
        assert out["train_inputs"] == self._fp(state)
        assert out["failure"].startswith("train failed (exit code 1)")

    def test_the_same_inputs_inside_the_wait_do_not_start_a_child(self, cwd, monkeypatch):
        child = _Child(returncode=1)
        monkeypatch.setattr(ls, "_TRAIN_RUNNER", child)
        state = _state()
        fp = self._fp(state)
        soon = (datetime.now(timezone.utc) + timedelta(minutes=5)).replace(microsecond=0)
        state = replace(state, failed_train=_print(fp=fp, retry_after=soon.isoformat()))

        out = ls.train_dpo_from_pairs(state, self._ctx())

        assert child.calls == 0
        assert out["skipped"] is True and out["train_held"] is True
        assert out["notes"].startswith("train_backoff: the same inputs failed 1 time")
        assert "failed" not in out and out["run_id"] is None

    def test_held_inputs_do_not_start_a_child_whatever_the_clock(self, cwd, monkeypatch):
        child = _Child(returncode=1)
        monkeypatch.setattr(ls, "_TRAIN_RUNNER", child)
        state = _state()
        state = replace(state, failed_train=_print(fp=self._fp(state), attempts=3, held=True))

        out = ls.train_dpo_from_pairs(state, self._ctx())

        assert child.calls == 0 and out["train_held"] is True
        assert out["notes"].startswith("train_held:")

    def test_after_the_wait_the_run_is_tried_again(self, cwd, monkeypatch):
        child = _Child(returncode=0)
        monkeypatch.setattr(ls, "_TRAIN_RUNNER", child)
        state = _state()
        state = replace(
            state, failed_train=_print(fp=self._fp(state), retry_after="2026-01-01T00:00:00+00:00")
        )

        out = ls.train_dpo_from_pairs(state, self._ctx())

        assert child.calls == 1 and out["run_id"]

    def test_new_data_is_tried_at_once_even_when_held(self, cwd, monkeypatch):
        child = _Child(returncode=0)
        monkeypatch.setattr(ls, "_TRAIN_RUNNER", child)
        state = _state()
        state = replace(state, failed_train=_print(fp=self._fp(state), attempts=3, held=True))
        (cwd / "sft.jsonl").write_text(
            json.dumps(ROW) + "\n" + json.dumps({**ROW, "source": "new"}) + "\n", encoding="utf-8"
        )

        out = ls.train_dpo_from_pairs(state, self._ctx())

        assert child.calls == 1 and out["run_id"]

    def test_a_changed_recipe_is_tried_at_once_even_when_held(self, cwd, monkeypatch):
        child = _Child(returncode=0)
        monkeypatch.setattr(ls, "_TRAIN_RUNNER", child)
        state = _state()
        held = replace(state, failed_train=_print(fp=self._fp(state), attempts=3, held=True))

        out = ls.train_dpo_from_pairs(replace(held, sft_recipe={"lr": 0.0001}), self._ctx())

        assert child.calls == 1 and out["run_id"]


class TestTheDaemon:
    def _config(self, root, train_fn):
        return WatchConfig(
            poll_interval_sec=1.0, max_iterations=1,
            state_path=str(root / ".soup" / "loop.yaml"),
            canary_stats_path=str(root / ".soup" / "canary-stats.json"),
            iteration_dir=str(root / ".soup-loops"),
            harvest_fn=lambda state: {
                "pairs_harvested": 1, "traces_collected": 1, "pairs_path": "sft.jsonl",
                "new_pairs": 1,
            },
            train_fn=train_fn,
            gate_fn=lambda state, ctx: {"gate_verdict": "SKIPPED"},
            deploy_fn=lambda state, ctx: {"deployed": False},
            cost_fn=lambda state: 0.0,
        )

    def test_three_failures_then_a_hold_and_the_child_ran_three_times(self, cwd, monkeypatch):
        child = _Child(returncode=1)
        monkeypatch.setattr(ls, "_TRAIN_RUNNER", child)
        config = self._config(cwd, ls.train_dpo_from_pairs)
        state = _state()

        history = []
        for _ in range(7):
            state, record, _ = run_once(state, config)
            history.append(state.failed_train and state.failed_train["attempts"])
            # the wait is over: bring the retry time into the past, as an elapsed wait does
            if state.failed_train and not state.failed_train["held"]:
                state = replace(
                    state,
                    failed_train={**state.failed_train, "retry_after": "2026-01-01T00:00:00+00:00"},
                )

        assert child.calls == 3, history
        assert history == [1, 2, 3, 3, 3, 3, 3]
        assert state.failed_train["held"] is True
        assert "train_held:" in record.notes

    def test_a_poll_inside_the_wait_changes_nothing_and_says_why(self, cwd, monkeypatch):
        child = _Child(returncode=1)
        monkeypatch.setattr(ls, "_TRAIN_RUNNER", child)
        config = self._config(cwd, ls.train_dpo_from_pairs)

        state, _, _ = run_once(_state(), config)
        before = state.failed_train
        state, record, _ = run_once(state, config)

        assert child.calls == 1
        assert state.failed_train == before
        assert record.notes.startswith("train_backoff:")

    def test_a_run_that_trained_clears_the_record(self, cwd, monkeypatch):
        failing = _Child(returncode=1)
        monkeypatch.setattr(ls, "_TRAIN_RUNNER", failing)
        config = self._config(cwd, ls.train_dpo_from_pairs)
        state, _, _ = run_once(_state(), config)
        assert state.failed_train is not None
        state = replace(
            state, failed_train={**state.failed_train, "retry_after": "2026-01-01T00:00:00+00:00"}
        )
        monkeypatch.setattr(ls, "_TRAIN_RUNNER", _Child(returncode=0))

        state, _, _ = run_once(state, config)

        assert state.failed_train is None

    def test_a_custom_train_stage_without_a_fingerprint_is_never_held(self, cwd):
        def train(state, ctx):
            return {"run_id": None, "skipped": False, "adapter_path": None, "failed": True,
                    "notes": "train failed (exit code 1)"}

        config = self._config(cwd, train)
        state = _state()
        for _ in range(5):
            state, _, _ = run_once(state, config)

        assert state.failed_train is None


class TestStateField:
    def test_it_round_trips_through_loop_yaml(self, cwd):
        path = str(cwd / ".soup" / "loop.yaml")
        write_state(_state(failed_train=_print()), path)

        assert read_state(path).failed_train == _print()

    @pytest.mark.parametrize(
        "bad",
        [
            "x", [], {}, {**_print(), "attempts": True}, {**_print(), "attempts": 0},
            {**_print(), "attempts": 101}, {**_print(), "inputs": "nothex"},
            {**_print(), "held": "yes"}, {**_print(), "reason": "a\x1bb"},
            {**_print(), "extra": 1}, {**_print(), "last_at": 5},
            {**_print(), "retry_after": 5}, {**_print(), "reason": ""},
        ],
    )
    def test_a_bad_record_is_refused(self, bad):
        with pytest.raises((ValueError, TypeError)):
            _state(failed_train=bad)


class TestStatusAndResume:
    def _runner(self):
        return CliRunner()

    def _status(self, cwd, state):
        write_state(state, str(cwd / ".soup" / "loop.yaml"))
        result = self._runner().invoke(loop_app, ["status"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        return _plain(result.output)

    def test_status_shows_a_waiting_retry(self, cwd):
        soon = (datetime.now(timezone.utc) + timedelta(minutes=5)).replace(microsecond=0)

        text = self._status(cwd, _state(failed_train=_print(retry_after=soon.isoformat())))

        assert "train_retry" in text
        assert "waiting: failed 1 time on the same inputs" in text
        assert f"next attempt after {soon.isoformat()}" in text

    def test_status_shows_a_hold_and_what_clears_it(self, cwd):
        text = self._status(cwd, _state(failed_train=_print(attempts=3, held=True)))

        assert "held: failed 3 times on the same inputs" in text
        assert "soup loop resume clears it" in text

    def test_status_shows_nothing_without_a_failure(self, cwd):
        assert "train_retry" not in self._status(cwd, _state())

    def test_status_shows_train_and_stage_retries_separately(self, cwd):
        stage = {**_print(), "reason": "stage error: the harvest stage raised ValueError"}

        text = self._status(cwd, _state(failed_train=_print(), failed_stage=stage))

        assert "train_retry" in text and "failed 1 time on the same inputs" in text
        assert "stage_retry" in text and "the harvest stage raised ValueError" in text

    def test_a_hostile_reason_prints_no_control_bytes(self, cwd):
        # the state refuses control characters, so the reason reaches the terminal clean
        write_state(_state(failed_train=_print()), str(cwd / ".soup" / "loop.yaml"))
        result = self._runner().invoke(loop_app, ["status"])

        assert "\x1b]" not in result.output

    def test_resume_clears_a_hold_on_a_running_loop(self, cwd):
        path = str(cwd / ".soup" / "loop.yaml")
        stage = {**_print(attempts=3, held=True), "reason": "stage error: harvest failed"}
        write_state(_state(failed_train=_print(attempts=3, held=True), failed_stage=stage), path)

        result = self._runner().invoke(loop_app, ["resume"])

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "retry hold cleared" in _plain(result.output)
        assert read_state(path).failed_train is None
        assert read_state(path).failed_stage is None
        assert read_state(path).status == "running"

    def test_resume_of_a_paused_loop_resumes_and_clears(self, cwd):
        path = str(cwd / ".soup" / "loop.yaml")
        stage = {**_print(attempts=2), "reason": "stage error: the harvest stage raised ValueError"}
        write_state(_state(status="paused", failed_train=_print(), failed_stage=stage), path)

        result = self._runner().invoke(loop_app, ["resume"])

        assert result.exit_code == 0, (result.output, repr(result.exception))
        text = _plain(result.output)
        assert "loop resumed" in text and "retry hold cleared" in text
        resumed = read_state(path)
        assert resumed.status == "running"
        assert resumed.failed_train is None and resumed.failed_stage is None

    def test_resume_without_a_hold_is_as_before(self, cwd):
        path = str(cwd / ".soup" / "loop.yaml")
        write_state(_state(), path)

        result = self._runner().invoke(loop_app, ["resume"])

        assert "loop is running, not paused" in _plain(result.output)
        assert "hold cleared" not in _plain(result.output)
