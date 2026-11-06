"""Selected evaluation evidence must be finite before comparison or display."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.eval.leaderboard import build_leaderboard_from_tracker, compare_runs
from soup_cli.experiment.tracker import ExperimentTracker
from tests.conftest import strip_ansi

RUN_A = "run-[bold]a[/]"
RUN_B = "run-b"
MODEL = "model-[red]name[/]"
BENCHMARK = "bench-[green]name[/]"
CASES = [(float("inf"), float("inf")), (float("inf"), 0.5), (0.5, -float("inf"))]
CASE_IDS = ["both-infinite", "left-infinite", "right-infinite"]


def normalized(text: str) -> str:
    return " ".join(strip_ansi(text).split())


@pytest.fixture
def tracker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[ExperimentTracker]:
    instance = ExperimentTracker(db_path=tmp_path / "synthetic.db")
    for run_id in (RUN_A, RUN_B):
        instance.start_run({}, "cpu", "synthetic", {}, run_id=run_id)
    monkeypatch.setattr("soup_cli.experiment.tracker.ExperimentTracker", lambda: instance)
    yield instance
    instance.close()


def save(tracker: ExperimentTracker, run_id: str, score: float, benchmark: str = BENCHMARK) -> None:
    tracker.save_eval_result(MODEL, benchmark, score, {}, run_id=run_id)


def message(context: str, score: float) -> str:
    return f"{context}: stored score {score!r} must be finite"


@pytest.mark.parametrize("left,right", CASES, ids=CASE_IDS)
def test_compare_refuses_selected_nonfinite(
    tracker: ExperimentTracker, left: float, right: float,
) -> None:
    save(tracker, RUN_A, left)
    save(tracker, RUN_B, right)
    run_id, score = (RUN_A, left) if left == float("inf") else (RUN_B, right)
    with pytest.raises(ValueError) as caught:
        compare_runs(tracker, RUN_A, RUN_B)
    assert str(caught.value) == message(f"run {run_id!r}, benchmark {BENCHMARK!r}", score)
    assert tracker.get_eval_results(run_id=run_id)[0]["score"] == score


@pytest.mark.parametrize("left,right", CASES, ids=CASE_IDS)
def test_cli_compare_refuses_nonfinite_without_partial_output(
    tracker: ExperimentTracker, left: float, right: float,
) -> None:
    save(tracker, RUN_A, left)
    save(tracker, RUN_B, right)
    run_id, score = (RUN_A, left) if left == float("inf") else (RUN_B, right)
    result = CliRunner().invoke(app, ["eval", "compare", RUN_A, RUN_B])
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit), repr(result.exception)
    assert normalized(result.output) == message(f"run {run_id!r}, benchmark {BENCHMARK!r}", score)


@pytest.mark.parametrize("score", [float("inf"), -float("inf")], ids=["positive", "negative"])
def test_leaderboard_refuses_selected_nonfinite(tracker: ExperimentTracker, score: float) -> None:
    save(tracker, RUN_A, score)
    with pytest.raises(ValueError) as caught:
        build_leaderboard_from_tracker(tracker)
    assert str(caught.value) == message(f"model {MODEL!r}, benchmark {BENCHMARK!r}", score)


@pytest.mark.parametrize("fmt", ["table", "json", "csv"])
@pytest.mark.parametrize("score", [float("inf"), -float("inf")], ids=["positive", "negative"])
def test_cli_leaderboard_refuses_nonfinite_without_partial_output(
    tracker: ExperimentTracker, score: float, fmt: str,
) -> None:
    save(tracker, RUN_A, score)
    result = CliRunner().invoke(app, ["eval", "leaderboard", "--format", fmt])
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit), repr(result.exception)
    assert normalized(result.output) == message(f"model {MODEL!r}, benchmark {BENCHMARK!r}", score)


@pytest.mark.parametrize("consumer", ["compare", "leaderboard"])
def test_nonfinite_custom_tracker_nan_is_refused(consumer: str) -> None:
    tracker = Mock()
    tracker.get_eval_results.return_value = [
        {"model_path": MODEL, "benchmark": BENCHMARK, "score": float("nan")},
    ]
    with pytest.raises(ValueError) as caught:
        if consumer == "compare":
            compare_runs(tracker, RUN_A, RUN_B)
        else:
            build_leaderboard_from_tracker(tracker)
    context = f"run {RUN_A!r}" if consumer == "compare" else f"model {MODEL!r}"
    assert str(caught.value) == message(f"{context}, benchmark {BENCHMARK!r}", float("nan"))


@pytest.mark.parametrize("left,right,regressed", [
    (0.9, 0.7, True), (0.0, 0.0, False), (0.01, 0.0, False), (0.02, 0.0, True),
])
def test_finite_comparison_and_cli_exit_are_unchanged(
    tracker: ExperimentTracker, left: float, right: float, regressed: bool,
) -> None:
    save(tracker, RUN_A, left)
    save(tracker, RUN_B, right)
    comparison = compare_runs(tracker, RUN_A, RUN_B)
    assert comparison["comparisons"][0]["delta"] == pytest.approx(right - left)
    assert comparison["has_regressions"] is regressed
    result = CliRunner().invoke(app, ["eval", "compare", RUN_A, RUN_B])
    assert result.exit_code == 0
    assert ("Regressions detected:" in normalized(result.output)) is regressed


def test_partial_benchmark_coverage_is_preserved(tracker: ExperimentTracker) -> None:
    save(tracker, RUN_A, 0.0)
    comparison = compare_runs(tracker, RUN_A, RUN_B)
    assert comparison["comparisons"] == [{
        "benchmark": BENCHMARK, "run_1_score": 0.0, "run_2_score": None, "delta": None,
    }]
    assert not comparison["has_regressions"]
    result = CliRunner().invoke(app, ["eval", "compare", RUN_A, RUN_B])
    assert result.exit_code == 0


def test_one_sided_nonfinite_benchmark_is_refused(tracker: ExperimentTracker) -> None:
    save(tracker, RUN_A, float("inf"))
    expected = message(f"run {RUN_A!r}, benchmark {BENCHMARK!r}", float("inf"))
    with pytest.raises(ValueError) as caught:
        compare_runs(tracker, RUN_A, RUN_B)
    assert str(caught.value) == expected
    result = CliRunner().invoke(app, ["eval", "compare", RUN_A, RUN_B])
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit), repr(result.exception)
    assert normalized(result.output) == expected


@pytest.mark.parametrize("consumer", ["compare", "leaderboard"])
def test_newest_invalid_score_never_falls_back(tracker: ExperimentTracker, consumer: str) -> None:
    save(tracker, RUN_A, 0.7)
    save(tracker, RUN_A, float("inf"))
    save(tracker, RUN_B, 0.0)
    with pytest.raises(ValueError):
        if consumer == "compare":
            compare_runs(tracker, RUN_A, RUN_B)
        else:
            build_leaderboard_from_tracker(tracker, run_id=RUN_A)


@pytest.mark.parametrize("fmt", ["table", "json", "csv"])
def test_newest_finite_score_over_old_infinite_is_preserved(
    tracker: ExperimentTracker, fmt: str,
) -> None:
    save(tracker, RUN_A, float("inf"))
    save(tracker, RUN_A, 0.0)
    save(tracker, RUN_B, 0.0)
    result = compare_runs(tracker, RUN_A, RUN_B)
    assert result["comparisons"][0]["run_1_score"] == 0.0
    assert result["comparisons"][0]["delta"] == 0.0
    leaderboard = build_leaderboard_from_tracker(tracker)
    assert leaderboard.models[MODEL][BENCHMARK] == 0.0
    cli = CliRunner().invoke(app, ["eval", "leaderboard", "--format", fmt])
    assert cli.exit_code == 0
    assert "Infinity" not in cli.output and "inf" not in cli.output
