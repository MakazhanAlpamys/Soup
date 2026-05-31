"""Leaderboard — aggregate eval results from experiments.db."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LeaderboardEntry:
    """A single entry in the leaderboard."""

    model_path: str
    benchmark: str
    score: float
    run_id: Optional[str] = None
    created_at: str = ""


@dataclass
class Leaderboard:
    """Aggregated leaderboard across all evaluated models."""

    entries: list[LeaderboardEntry] = field(default_factory=list)
    models: dict[str, dict[str, float]] = field(default_factory=dict)

    def compute(self) -> None:
        """Aggregate entries by model, computing per-benchmark scores."""
        self.models = {}
        for entry in self.entries:
            model = entry.model_path
            if model not in self.models:
                self.models[model] = {}
            # Keep the latest score per benchmark per model
            self.models[model][entry.benchmark] = entry.score

    def get_sorted_models(
        self, sort_by: Optional[str] = None,
    ) -> list[tuple[str, dict[str, float], float]]:
        """Return models sorted by average score or specific benchmark.

        Returns list of (model_path, benchmark_scores, sort_score).
        """
        result = []
        for model, scores in self.models.items():
            if sort_by and sort_by in scores:
                sort_score = scores[sort_by]
            else:
                sort_score = (
                    sum(scores.values()) / len(scores) if scores else 0.0
                )
            result.append((model, scores, sort_score))

        result.sort(key=lambda item: item[2], reverse=True)
        return result


def build_leaderboard_from_tracker(
    tracker: object,
    run_id: Optional[str] = None,
) -> Leaderboard:
    """Build a leaderboard from experiment tracker eval results."""
    eval_results = tracker.get_eval_results(run_id=run_id)

    entries = []
    for row in eval_results:
        entries.append(LeaderboardEntry(
            model_path=row.get("model_path", ""),
            benchmark=row.get("benchmark", ""),
            score=row.get("score", 0.0),
            run_id=row.get("run_id"),
            created_at=row.get("created_at", ""),
        ))

    leaderboard = Leaderboard(entries=entries)
    leaderboard.compute()
    return leaderboard


def compare_runs(
    tracker: object,
    run_id_1: str,
    run_id_2: str,
) -> dict:
    """Compare eval results between two runs.

    Returns dict with per-benchmark comparison and deltas.
    """
    results_1 = tracker.get_eval_results(run_id=run_id_1)
    results_2 = tracker.get_eval_results(run_id=run_id_2)

    scores_1: dict[str, float] = {}
    for row in results_1:
        scores_1[row["benchmark"]] = row["score"]

    scores_2: dict[str, float] = {}
    for row in results_2:
        scores_2[row["benchmark"]] = row["score"]

    all_benchmarks = sorted(set(scores_1.keys()) | set(scores_2.keys()))

    comparisons = []
    regressions = []
    for bench in all_benchmarks:
        score_a = scores_1.get(bench)
        score_b = scores_2.get(bench)
        delta = None
        if score_a is not None and score_b is not None:
            delta = score_b - score_a
            if delta < -0.01:
                regressions.append(bench)

        comparisons.append({
            "benchmark": bench,
            "run_1_score": score_a,
            "run_2_score": score_b,
            "delta": delta,
        })

    return {
        "run_1": run_id_1,
        "run_2": run_id_2,
        "comparisons": comparisons,
        "regressions": regressions,
        "has_regressions": len(regressions) > 0,
    }


def export_leaderboard(
    leaderboard: Leaderboard,
    fmt: str = "json",
) -> str:
    """Export leaderboard to JSON or CSV string."""
    sorted_models = leaderboard.get_sorted_models()

    if fmt == "csv":
        # Collect all benchmarks
        all_benchmarks = set()
        for _, scores, _ in sorted_models:
            all_benchmarks.update(scores.keys())
        benchmarks = sorted(all_benchmarks)

        lines = ["model," + ",".join(benchmarks) + ",average"]
        for model, scores, avg in sorted_models:
            vals = [str(scores.get(bench, "")) for bench in benchmarks]
            lines.append(f"{model},{','.join(vals)},{avg:.4f}")
        return "\n".join(lines)

    # Default: JSON
    data = []
    for model, scores, avg in sorted_models:
        data.append({
            "model": model,
            "scores": scores,
            "average": round(avg, 4),
        })
    return json.dumps(data, indent=2)
