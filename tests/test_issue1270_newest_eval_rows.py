"""#1270 — eval comparisons must use the newest row per benchmark."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from soup_cli.eval.leaderboard import build_leaderboard_from_tracker, compare_runs
from soup_cli.eval.results import newest_eval_rows
from soup_cli.registry.diff import eval_delta


def _row(benchmark: str, score: float, created_at: str, rowid: int) -> dict:
    return {
        "benchmark": benchmark,
        "score": score,
        "created_at": created_at,
        "id": rowid,
        "model_path": "model",
    }


def test_newest_eval_rows_breaks_equal_timestamps_by_insertion_order():
    rows = [
        _row("mmlu", 0.30, "2026-01-01T00:00:00", 1),
        _row("mmlu", 0.45, "2026-01-01T00:00:00", 2),
    ]

    assert [row["score"] for row in newest_eval_rows(rows)] == [0.45]


def test_registry_diff_uses_newest_rows_on_both_sides():
    left = [
        _row("mmlu", 0.45, "2026-01-02T00:00:00", 2),
        _row("mmlu", 0.30, "2026-01-01T00:00:00", 1),
    ]
    right = [
        _row("mmlu", 0.40, "2026-01-02T00:00:00", 4),
        _row("mmlu", 0.50, "2026-01-01T00:00:00", 3),
    ]

    result = eval_delta(left, right)

    assert result == [{
        "benchmark": "mmlu",
        "left": 0.45,
        "right": 0.40,
        "delta": pytest.approx(-0.05),
    }]


def test_eval_compare_uses_newest_rows_on_both_sides():
    tracker = MagicMock()
    tracker.get_eval_results.side_effect = [
        [
            _row("mmlu", 0.45, "2026-01-02T00:00:00", 2),
            _row("mmlu", 0.30, "2026-01-01T00:00:00", 1),
        ],
        [
            _row("mmlu", 0.40, "2026-01-02T00:00:00", 4),
            _row("mmlu", 0.50, "2026-01-01T00:00:00", 3),
        ],
    ]

    result = compare_runs(tracker, "run-a", "run-b")

    assert result["comparisons"] == [{
        "benchmark": "mmlu",
        "run_1_score": 0.45,
        "run_2_score": 0.40,
        "delta": pytest.approx(-0.05),
    }]
    assert result["regressions"] == ["mmlu"]


def test_leaderboard_uses_newest_row_per_model_and_benchmark():
    tracker = MagicMock()
    tracker.get_eval_results.return_value = [
        _row("mmlu", 0.45, "2026-01-02T00:00:00", 2),
        _row("mmlu", 0.30, "2026-01-01T00:00:00", 1),
    ]

    leaderboard = build_leaderboard_from_tracker(tracker)

    assert leaderboard.models == {"model": {"mmlu": 0.45}}


def test_leaderboard_keeps_the_newest_row_for_every_model():
    tracker = MagicMock()
    tracker.get_eval_results.return_value = [
        {**_row("mmlu", 0.45, "2026-01-02T00:00:00", 3), "model_path": "model-a"},
        {**_row("mmlu", 0.40, "2026-01-02T00:00:00", 2), "model_path": "model-b"},
        {**_row("mmlu", 0.30, "2026-01-01T00:00:00", 1), "model_path": "model-a"},
    ]

    leaderboard = build_leaderboard_from_tracker(tracker)

    assert leaderboard.models == {
        "model-a": {"mmlu": 0.45},
        "model-b": {"mmlu": 0.40},
    }


def test_newest_eval_rows_skips_empty_benchmark_names():
    rows = [_row("", 0.5, "2026-01-01T00:00:00", 1)]

    assert newest_eval_rows(rows) == []
