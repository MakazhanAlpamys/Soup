"""Catastrophic forgetting probe (v0.56.0).

Extends v0.25 ``eval/forgetting.py``: compares per-task accuracy on a
held-out reference vs the adapter and converts the Δ into a [0, 1]
preservation score (1.0 = no forgetting).
"""

from __future__ import annotations

from typing import Mapping

from soup_cli.utils.diagnose._common import (
    merge_evidence,
    require_finite_unit,
)
from soup_cli.utils.diagnose.report import FailureScore, classify_score


def score_forgetting(
    base_accuracy: Mapping[str, float],
    adapter_accuracy: Mapping[str, float],
    *,
    tolerance: float = 0.02,
) -> FailureScore:
    """Score forgetting from per-task base + adapter accuracies.

    ``tolerance`` is the per-task accuracy drop considered "no
    regression"; defaults to 2 % (mirrors v0.26 Quant-Lobotomy OK band).
    Score = mean(min(1.0, 1.0 - max(0, base - adapter - tolerance))).
    """
    if not isinstance(base_accuracy, Mapping):
        raise TypeError("base_accuracy must be Mapping[str, float]")
    if not isinstance(adapter_accuracy, Mapping):
        raise TypeError("adapter_accuracy must be Mapping[str, float]")
    # `require_finite_unit` rejects bool / non-numeric / NaN / ±Inf /
    # out-of-range with one call (code-review MEDIUM fix replacing the
    # manual chain that silently accepted NaN).
    tol = require_finite_unit(tolerance, "tolerance")
    shared = sorted(set(base_accuracy) & set(adapter_accuracy))
    if not shared:
        # Cannot compare → return neutral OK with explanatory evidence.
        return FailureScore(
            mode="forgetting",
            score=1.0,
            verdict="OK",
            evidence="no shared tasks; nothing to compare",
        )
    drops = []
    worst_task = ""
    worst_delta = 0.0
    for task in shared:
        base = require_finite_unit(base_accuracy[task], f"base_accuracy[{task!r}]")
        adapter = require_finite_unit(
            adapter_accuracy[task], f"adapter_accuracy[{task!r}]"
        )
        delta = max(0.0, base - adapter - tol)
        drops.append(min(1.0, delta))
        if delta > worst_delta:
            worst_delta = delta
            worst_task = task
    score = max(0.0, 1.0 - (sum(drops) / len(drops)))
    verdict = classify_score(score)
    evidence = merge_evidence(
        {
            "tasks": len(shared),
            "worst_task": worst_task or "—",
            "worst_drop": worst_delta,
            "tolerance": tol,
        }
    )
    return FailureScore(
        mode="forgetting",
        score=score,
        verdict=verdict,
        evidence=evidence,
    )
