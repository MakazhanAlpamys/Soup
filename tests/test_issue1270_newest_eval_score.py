"""#1270 — registry diff / eval compare must use newest score per benchmark.

Dict-overwrite of ``{benchmark: score}`` kept the *last* row seen. With
``ORDER BY created_at DESC`` that last row is the oldest, so a re-measurement
was ignored (same class of bug as #1224 / #1260).
"""

from __future__ import annotations

from soup_cli.eval.leaderboard import compare_runs
from soup_cli.eval.newest_row import newest_score_per_benchmark
from soup_cli.registry.diff import eval_delta


def _rows(*specs: tuple[str, float, str, int]) -> list[dict]:
    """Build eval_results-like rows: (benchmark, score, created_at, id)."""
    return [
        {"benchmark": b, "score": s, "created_at": ts, "id": i}
        for b, s, ts, i in specs
    ]


class TestNewestScorePerBenchmark:
    def test_keeps_newest_created_at(self):
        rows = _rows(
            ("mmlu", 0.30, "2026-01-01T00:00:00", 1),
            ("mmlu", 0.45, "2026-01-02T00:00:00", 2),
        )
        assert newest_score_per_benchmark(rows) == {"mmlu": 0.45}

    def test_tie_on_created_at_uses_higher_id(self):
        rows = _rows(
            ("mmlu", 0.30, "2026-01-01T00:00:00", 1),
            ("mmlu", 0.45, "2026-01-01T00:00:00", 2),
        )
        assert newest_score_per_benchmark(rows) == {"mmlu": 0.45}


class TestEvalDeltaNewest:
    def test_example_from_issue_reports_regression(self):
        # Run A: 0.30 then re-measure 0.45; Run B: 0.50 then 0.40.
        # Oldest-vs-oldest = +0.20; newest-vs-newest = -0.05 regression.
        left = _rows(
            ("bench", 0.30, "2026-01-01T00:00:00", 1),
            ("bench", 0.45, "2026-01-02T00:00:00", 2),
        )
        right = _rows(
            ("bench", 0.50, "2026-01-01T00:00:00", 3),
            ("bench", 0.40, "2026-01-02T00:00:00", 4),
        )
        deltas = eval_delta(left, right)
        assert len(deltas) == 1
        assert deltas[0]["left"] == 0.45
        assert deltas[0]["right"] == 0.40
        assert abs(deltas[0]["delta"] - (-0.05)) < 1e-9


class _FakeTracker:
    def __init__(self, by_run: dict[str, list[dict]]):
        self._by_run = by_run

    def get_eval_results(self, run_id=None):
        return list(self._by_run.get(run_id, []))


class TestCompareRunsNewest:
    def test_example_from_issue_flags_regression(self):
        tracker = _FakeTracker(
            {
                "A": _rows(
                    ("bench", 0.30, "2026-01-01T00:00:00", 1),
                    ("bench", 0.45, "2026-01-02T00:00:00", 2),
                ),
                "B": _rows(
                    ("bench", 0.50, "2026-01-01T00:00:00", 3),
                    ("bench", 0.40, "2026-01-02T00:00:00", 4),
                ),
            }
        )
        result = compare_runs(tracker, "A", "B")
        assert result["comparisons"][0]["run_1_score"] == 0.45
        assert result["comparisons"][0]["run_2_score"] == 0.40
        assert abs(result["comparisons"][0]["delta"] - (-0.05)) < 1e-9
        assert result["has_regressions"] is True
        assert "bench" in result["regressions"]

    def test_created_at_tie_uses_insertion_id(self):
        tracker = _FakeTracker(
            {
                "A": _rows(
                    ("bench", 0.10, "2026-01-01T00:00:00", 1),
                    ("bench", 0.90, "2026-01-01T00:00:00", 9),
                ),
                "B": _rows(
                    ("bench", 0.50, "2026-01-01T00:00:00", 2),
                ),
            }
        )
        result = compare_runs(tracker, "A", "B")
        assert result["comparisons"][0]["run_1_score"] == 0.90
