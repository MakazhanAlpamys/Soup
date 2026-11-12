"""Comparison deltas must be representable before they reach the SDK or CLI."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.eval.leaderboard import compare_runs
from soup_cli.experiment.tracker import ExperimentTracker
from tests.conftest import strip_ansi

RUN_A = "run-[bold]a[/]"
RUN_B = "run-b"
BENCHMARK = "bench-[green]name[/]"
ERROR = f"runs {RUN_A!r} and {RUN_B!r}, benchmark {BENCHMARK!r}: score delta must be finite"
OVERFLOW_SCORES = [(-1e308, 1e308), (1e308, -1e308)]


@pytest.fixture
def tracker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[ExperimentTracker]:
    instance = ExperimentTracker(db_path=tmp_path / "synthetic.db")
    for run in (RUN_A, RUN_B):
        instance.start_run({}, "cpu", "synthetic", {}, run_id=run)
    monkeypatch.setattr("soup_cli.experiment.tracker.ExperimentTracker", lambda: instance)
    try:
        yield instance
    finally:
        instance.close()


@pytest.mark.parametrize("left,right", OVERFLOW_SCORES, ids=["positive", "negative"])
def test_compare_refuses_overflowing_delta(
    tracker: ExperimentTracker, left: float, right: float,
) -> None:
    tracker.save_eval_result("synthetic", BENCHMARK, left, {}, run_id=RUN_A)
    tracker.save_eval_result("synthetic", BENCHMARK, right, {}, run_id=RUN_B)
    with pytest.raises(ValueError) as caught:
        compare_runs(tracker, RUN_A, RUN_B)
    assert str(caught.value) == ERROR


@pytest.mark.parametrize("left,right", OVERFLOW_SCORES, ids=["positive", "negative"])
def test_cli_refuses_overflow_before_rendering_comparison(
    tracker: ExperimentTracker, left: float, right: float,
) -> None:
    for run, score in ((RUN_A, left), (RUN_B, right)):
        tracker.save_eval_result("synthetic", "a-finite-control", 0.5, {}, run_id=run)
        tracker.save_eval_result("synthetic", BENCHMARK, score, {}, run_id=run)

    result = CliRunner().invoke(app, ["eval", "compare", RUN_A, RUN_B])

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert " ".join(strip_ansi(result.output).split()) == ERROR


@pytest.mark.parametrize(
    "left,right,expected,regressed",
    [
        (0.0, 1e308, 1e308, False),
        (0.9, 0.7, -0.2, True),
        (0.01, 0.0, -0.01, False),
        (0.0, None, None, False),
    ],
    ids=["large-finite", "regression", "at-threshold", "missing-benchmark"],
)
def test_finite_or_missing_delta_remains_serializable(
    tracker: ExperimentTracker, left: float, right: float | None,
    expected: float | None, regressed: bool,
) -> None:
    tracker.save_eval_result("synthetic", BENCHMARK, left, {}, run_id=RUN_A)
    if right is not None:
        tracker.save_eval_result("synthetic", BENCHMARK, right, {}, run_id=RUN_B)

    comparison = compare_runs(tracker, RUN_A, RUN_B)

    assert comparison["comparisons"][0]["delta"] == pytest.approx(expected)
    assert comparison["has_regressions"] is regressed
    json.dumps(comparison, allow_nan=False)
