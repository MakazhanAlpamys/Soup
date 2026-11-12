"""Finite stored scores keep a finite mean without changing ordinary rounding."""

from __future__ import annotations

import csv
import io
import json
import math
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.eval.leaderboard import build_leaderboard_from_tracker, export_leaderboard
from soup_cli.experiment.tracker import ExperimentTracker
from tests.conftest import strip_ansi

MODEL = "synthetic-model"
EXTREMES = [
    ([1e308, 1e308], 1e308),
    ([-1e308, -1e308], -1e308),
    ([1e308, 1e308, -1e308, -1e308], 0.0),
    ([sys.float_info.max] * 3, sys.float_info.max),
    ([-sys.float_info.max] * 3, -sys.float_info.max),
]
ORDINARY = [([0.1, 0.1, 0.1], 0.10000000000000002), ([0.0], 0.0), ([0.75], 0.75)]
SURFACES = ["function", "json", "csv", "cli-table", "cli-json"]


def normalized(text: str) -> str:
    return " ".join(strip_ansi(text).split())


def reject_constant(value: str) -> None:
    raise ValueError(value)


@pytest.fixture
def tracker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[ExperimentTracker]:
    instance = ExperimentTracker(db_path=tmp_path / "synthetic.db")
    instance.start_run({}, "cpu", "synthetic", {}, run_id="synthetic")
    monkeypatch.setattr("soup_cli.experiment.tracker.ExperimentTracker", lambda: instance)
    monkeypatch.setattr("soup_cli.commands.eval.console", Console(width=5000))
    yield instance
    instance.close()


def save(tracker: ExperimentTracker, values: list[float], model: str = MODEL) -> None:
    for index, score in enumerate(values):
        tracker.save_eval_result(model, f"bench-{index}", score, {}, run_id="synthetic")


def assert_average(tracker: ExperimentTracker, expected: float, surface: str) -> None:
    board = build_leaderboard_from_tracker(tracker)
    if surface == "function":
        average = board.get_sorted_models()[0][2]
        assert math.isfinite(average)
        assert average == expected
    elif surface == "csv":
        row = next(csv.DictReader(io.StringIO(export_leaderboard(board, fmt="csv"))))
        assert row["average"] == f"{expected:.4f}"
    else:
        if surface.startswith("cli-"):
            fmt = "table" if surface == "cli-table" else "json"
            result = CliRunner().invoke(app, ["eval", "leaderboard", "--format", fmt])
            assert result.exit_code == 0, (result.output, repr(result.exception))
            output = normalized(result.output)
        else:
            output = export_leaderboard(board)
        if surface == "cli-table":
            assert "inf" not in output.lower()
            assert f"{expected:.4f}" in output
        else:
            rows = json.loads(output, parse_constant=reject_constant)
            assert rows[0]["average"] == round(expected, 4)
    assert all(math.isfinite(row["score"]) for row in tracker.get_eval_results())


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize("values,expected", EXTREMES,
                         ids=["positive", "negative", "mixed", "max-positive", "max-negative"])
def test_extreme_finite_scores_keep_finite_average(
    tracker: ExperimentTracker, values: list[float], expected: float, surface: str,
) -> None:
    save(tracker, values)
    assert_average(tracker, expected, surface)


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize("values,expected", ORDINARY, ids=["last-bit", "zero", "single"])
def test_ordinary_average_keeps_original_result(
    tracker: ExperimentTracker, values: list[float], expected: float, surface: str,
) -> None:
    save(tracker, values)
    assert_average(tracker, expected, surface)


def test_larger_finite_model_ranks_above_overflowing_sum(tracker: ExperimentTracker) -> None:
    save(tracker, [1e308, 1e308])
    save(tracker, [1.5e308], model="higher")
    rows = build_leaderboard_from_tracker(tracker).get_sorted_models()
    assert [(model, average) for model, _, average in rows] == [
        ("higher", 1.5e308), (MODEL, 1e308),
    ]


def test_sort_by_benchmark_keeps_selected_score(tracker: ExperimentTracker) -> None:
    save(tracker, [0.1, 0.9])
    save(tracker, [0.2, 0.0], model="higher")
    rows = build_leaderboard_from_tracker(tracker).get_sorted_models(sort_by="bench-0")
    assert [(model, score) for model, _, score in rows] == [("higher", 0.2), (MODEL, 0.1)]
