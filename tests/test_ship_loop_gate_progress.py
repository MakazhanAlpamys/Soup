"""``soup ship --gate loop-v2`` says where it is while it asks its questions (review 5, F8).

The gate asks BOTH versions every correction paraphrase and every held-out question, one
generation at a time, and printed nothing between "asking both versions N question(s)" and the
verdict: about 40 minutes on a 0.6B model for 52 + 18 questions, which the loop (a 6 hour limit
for a ship run) cannot tell from a hang. Now a progress line is printed at least every
``EVERY_ANSWERS`` answers or ``EVERY_SECONDS`` seconds, and one when the last answer is in. The
lines hold only counts and a clock reading (no model or user text).
"""

from __future__ import annotations

import io

import pytest
from rich.console import Console

import soup_cli.commands.ship as ship
from soup_cli.utils.ship_loop_gate import (
    EVERY_ANSWERS,
    EVERY_SECONDS,
    CorrectionEdit,
    GateProgress,
    HeldOutEdit,
)
from soup_cli.utils.ship_verdict import ShipVerdict


class Clock:
    """A monotonic clock the test moves by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _reporter(total, clock=None, **kwargs):
    lines: list = []
    clock = clock or Clock()
    return GateProgress(total, lines.append, clock=clock, **kwargs), lines, clock


class TestReporter:
    def test_the_limits_are_five_answers_and_thirty_seconds(self):
        assert (EVERY_ANSWERS, EVERY_SECONDS) == (5, 30.0)

    def test_a_line_every_five_answers_and_one_at_the_end(self):
        progress, lines, _ = _reporter(12)

        for _ in range(12):
            progress.step()

        assert [line.split(" answers")[0] for line in lines] == [
            "Gate loop-v2: 5/12", "Gate loop-v2: 10/12", "Gate loop-v2: 12/12",
        ]

    def test_the_last_answer_is_not_reported_twice(self):
        progress, lines, _ = _reporter(10)

        for _ in range(10):
            progress.step()

        assert len(lines) == 2 and lines[-1].startswith("Gate loop-v2: 10/10")

    def test_a_slow_answer_gets_a_line_before_the_fifth(self):
        progress, lines, clock = _reporter(100)

        progress.step()
        assert lines == []
        clock.now += 31.0
        progress.step()

        assert len(lines) == 1 and lines[0].startswith("Gate loop-v2: 2/100")

    def test_no_stretch_of_answers_goes_unreported_for_longer_than_the_limits(self):
        progress, lines, clock = _reporter(1000)
        last_report_at, last_report_count = clock.now, 0
        answers_since, seconds_since = 0, 0.0

        for count in range(1, 400):
            clock.now += 7.0  # a slow generation: 7 s each
            before = len(lines)
            progress.step()
            if len(lines) > before:
                answers_since, seconds_since = 0, 0.0
                last_report_at, last_report_count = clock.now, count
            else:
                answers_since += 1
                seconds_since = clock.now - last_report_at
            assert answers_since < EVERY_ANSWERS and seconds_since < EVERY_SECONDS, count
        assert last_report_count > 0

    def test_the_line_carries_the_elapsed_time_and_no_text_but_counts(self):
        progress, lines, clock = _reporter(10)
        for _ in range(4):
            clock.now += 1.0
            progress.step()
        clock.now += 301.0
        progress.step()

        assert lines == ["Gate loop-v2: 5/10 answers asked (5 min 5 s elapsed)"]

    def test_wrapping_a_generator_counts_every_call_and_returns_its_answer(self):
        progress, lines, _ = _reporter(10)
        calls: list = []

        def generate(prompt):
            calls.append(prompt)
            return prompt.upper()

        wrapped = progress.wrap(generate)

        assert [wrapped(f"q{index}") for index in range(5)] == ["Q0", "Q1", "Q2", "Q3", "Q4"]
        assert calls == ["q0", "q1", "q2", "q3", "q4"] and len(lines) == 1

    def test_a_failing_generator_is_not_counted_and_its_error_goes_through(self):
        progress, lines, _ = _reporter(10)

        def generate(prompt):
            raise RuntimeError("out of memory")

        with pytest.raises(RuntimeError, match="out of memory"):
            progress.wrap(generate)("q")

        assert lines == []

    def test_a_total_of_zero_prints_nothing(self):
        progress, lines, _ = _reporter(0)
        progress.finish()

        assert lines == []


class TestInTheShipRun:
    def _verdict(self):
        from soup_cli.utils.ship_verdict import (
            DECISION_SHIP,
            BenchmarkDelta,
            TaskWin,
        )

        delta = BenchmarkDelta(
            name="mini_mmlu", base=0.5, tuned=0.5, delta=0.0, regressed=False
        )
        return ShipVerdict(
            decision=DECISION_SHIP,
            task_win=TaskWin(mode="metric", base=0.2, tuned=0.8, won=True),
            benchmark_deltas=(delta,), failed_rule=None, forgetting_threshold=0.05,
            soup_version="0",
        )

    def test_apply_loop_gate_prints_progress_while_it_asks(self, monkeypatch):
        from soup_cli.utils.ship_loop_gate import GateInputs

        buffer = io.StringIO()
        monkeypatch.setattr(ship, "console", Console(file=buffer, width=200, force_terminal=False))
        corrections = tuple(
            CorrectionEdit(
                edit_id=f"e{number}", original="old answer", expected="new fact here",
                paraphrases=("p1", "p2", "p3"),
            )
            for number in range(4)
        )
        held_out = tuple(
            HeldOutEdit(edit_id=f"h{number}", prompt="q", original="old", expected="new fact")
            for number in range(3)
        )
        asked: list = []

        def old(prompt):
            asked.append(("old", prompt))
            return "old answer"

        def new(prompt):
            asked.append(("new", prompt))
            return "the new fact here"

        verdict = self._verdict()
        assert isinstance(verdict, ShipVerdict)

        ship._apply_loop_gate(
            verdict, GateInputs(corrections=corrections, held_out=held_out), old, new,
            task_mode="exact", judge_model=None,
        )

        text = " ".join(buffer.getvalue().split())
        assert len(asked) == 2 * (12 + 3)
        assert "Gate loop-v2: 5/30 answers asked" in text
        assert "Gate loop-v2: 30/30 answers asked" in text
        assert text.count("answers asked") == 6  # every 5 of 30, the 30th included
