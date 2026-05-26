"""Active-learning sampler — surface uncertain prod traces for review.

v0.63.0 Part C — picks the rows the model is *least confident* about so
humans only review what the policy itself thinks is borderline. Reduces
human-eval cost by 5-10x in practice.

Three modes via the input data shape:

1. Single reward-model score (``rm_score``) — uncertainty via max-entropy:
   ``1 - |2 * score - 1|``. Score 0.5 -> uncertainty 1.0 (peak),
   scores 0.0 or 1.0 -> uncertainty 0.0.
2. Two reward-model scores (``rm_scores: [s1, s2]``) — disagreement via
   ``|s1 - s2|``. Bigger gap -> higher uncertainty.
3. K reward-model scores (``rm_scores: [s1, ..., sK]``) for ``3 <= K <= 32``
   — population variance scaled by 4 (max disagreement = 1.0 when half the
   RMs score 0 and half score 1). Monotone-correct: adding a fresh RM score
   equal to the running mean *decreases* the score (the new contribution to
   sum-of-squares is zero while the denominator grows), so consensus on
   redundant evidence can never spike uncertainty. Replaces the v0.63.0
   ``max(scores) - min(scores)`` fallback which was monotone-broken — closes
   #206.

Composes with v0.19 human eval (the output JSONL is a drop-in human-eval
prompt set) and v0.58 ``soup loop watch`` (which can run this nightly).
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from soup_cli.utils.paths import is_under_cwd

_MAX_BUDGET: Final[int] = 100_000
_MAX_INPUT_ROWS: Final[int] = 10_000_000  # 10M — production-scale day of traces
# Cap on K reward-model scores per row. 32 is generous (most ensembles use
# 3-8 RMs) and bounds the inner O(K) variance loop at well under a
# microsecond per row. K>32 raises ValueError (DoS defence) — closes #206.
# Final[int] so a caller cannot silently disable the cap by rebinding it.
_MAX_RM_SCORES: Final[int] = 32


@dataclass(frozen=True)
class ActiveLearningPlan:
    """Result of an active-learning pass."""

    rows_in: int
    rows_selected: int
    budget: int
    mean_uncertainty: float

    def __post_init__(self) -> None:
        if self.rows_in < 0:
            raise ValueError("rows_in must be >= 0")
        if self.rows_selected < 0:
            raise ValueError("rows_selected must be >= 0")
        if self.budget < 1:
            raise ValueError("budget must be >= 1")
        if self.rows_selected > self.rows_in:
            raise ValueError(
                f"rows_selected ({self.rows_selected}) cannot exceed "
                f"rows_in ({self.rows_in})"
            )
        # NaN/Inf guard on the mean (code-review LOW fix v0.63.0 — matches
        # project-wide finite-only policy for every other numeric field).
        if not math.isfinite(self.mean_uncertainty):
            raise ValueError("mean_uncertainty must be finite (no NaN / Inf)")


def validate_budget(value: object) -> int:
    """Validate ``budget`` is a positive int within sane bounds.

    Mirrors v0.41.0 / v0.62.0 numeric validator policy: bool-first
    rejection, non-int -> TypeError, range -> ValueError.
    """
    if isinstance(value, bool):
        raise TypeError("budget must be int, not bool")
    if not isinstance(value, int):
        raise TypeError(
            f"budget must be int, got {type(value).__name__}"
        )
    if value < 1:
        raise ValueError(f"budget must be >= 1, got {value}")
    if value > _MAX_BUDGET:
        raise ValueError(
            f"budget must be <= {_MAX_BUDGET}, got {value}"
        )
    return value


def _validate_score(score: object, *, idx: int) -> float:
    if isinstance(score, bool):
        raise TypeError(f"scores[{idx}] must be number, not bool")
    if not isinstance(score, (int, float)):
        raise TypeError(
            f"scores[{idx}] must be number, got {type(score).__name__}"
        )
    f_score = float(score)
    if not math.isfinite(f_score):
        raise ValueError(f"scores[{idx}] must be finite (no NaN / Inf)")
    if not (0.0 <= f_score <= 1.0):
        raise ValueError(
            f"scores[{idx}] must be in [0.0, 1.0], got {f_score}"
        )
    return f_score


def score_uncertainty(*, scores: Sequence[float | int]) -> float:
    """Compute uncertainty from K reward-model scores (``1 <= K <= 32``).

    - K=1: max-entropy distance from 0.5 (peak at 0.5 -> uncertainty 1.0)
    - K=2: pairwise disagreement (``|s1 - s2|``)
    - K>=3: population variance scaled by 4, clamped to ``[0, 1]``

    For scores in ``[0, 1]`` the population variance is bounded by 0.25
    (achieved when half the RMs score 0 and half score 1), so the 4x scale
    keeps the K>=3 path inside the unit interval and consistent with the
    K<=2 forms.

    Validation rejects bool-as-int, non-finite values (NaN / +/-Inf), and
    out-of-range scores at every K — see ``_validate_score``.
    """
    if not isinstance(scores, Sequence) or isinstance(scores, str):
        raise TypeError(
            f"scores must be a sequence, got {type(scores).__name__}"
        )
    if len(scores) == 0:
        return 0.0
    if len(scores) > _MAX_RM_SCORES:
        raise ValueError(
            f"score_uncertainty supports at most {_MAX_RM_SCORES} RM scores, "
            f"got {len(scores)} (DoS cap)"
        )
    validated = [_validate_score(s, idx=i) for i, s in enumerate(scores)]
    k = len(validated)
    if k == 1:
        # 1 - |2*s - 1|  -> peak at s=0.5
        return 1.0 - abs(2.0 * validated[0] - 1.0)
    if k == 2:
        # Disagreement closed-form. Preserved verbatim for K=2 even though
        # variance gives the same answer up to scaling — existing operator
        # dashboards / thresholds depend on the |s1 - s2| value.
        return abs(validated[0] - validated[1])
    # K>=3: 4 * population variance, clamped into the unit interval.
    # Population variance (divide by N, not N-1) keeps the bound tight at
    # 0.25 for scores in [0, 1] and gives the monotonicity invariant
    # described in the module docstring. ``math.fsum`` uses compensated
    # summation to keep accumulated rounding error sub-ULP even at K=32 —
    # the variance formula is sensitive to it because we square the deltas.
    mean = math.fsum(validated) / k
    var = math.fsum((s - mean) ** 2 for s in validated) / k
    # max(0.0, ...) is defensive: floating-point subtraction can yield -epsilon
    # even though population variance is mathematically non-negative.
    return max(0.0, min(1.0, 4.0 * var))


def _row_uncertainty(row: Mapping[str, object]) -> float:
    """Compute uncertainty for a single row.

    Priority: explicit ``uncertainty`` field > ``rm_scores`` list >
    ``rm_score`` scalar > 0.0.
    """
    if not isinstance(row, Mapping):
        raise TypeError(
            f"row must be a Mapping, got {type(row).__name__}"
        )
    explicit = row.get("uncertainty")
    if isinstance(explicit, (int, float)) and not isinstance(explicit, bool):
        f_val = float(explicit)
        if not math.isfinite(f_val):
            return 0.0
        return max(0.0, min(1.0, f_val))
    scores_field = row.get("rm_scores")
    if isinstance(scores_field, Sequence) and not isinstance(scores_field, str):
        scores_list: list[float] = []
        try:
            for i, s in enumerate(scores_field):
                scores_list.append(_validate_score(s, idx=i))
        except (TypeError, ValueError):
            return 0.0
        # Route every K through ``score_uncertainty`` so the variance path
        # (K>=3) and the closed forms (K=1, K=2) share a single source of
        # truth. K>32 raises ValueError on the cap — isolate the row by
        # returning 0.0 instead of breaking the whole batch (matches the
        # existing _validate_score isolation policy two lines above).
        try:
            return score_uncertainty(scores=scores_list)
        except (TypeError, ValueError):
            return 0.0
    scalar = row.get("rm_score")
    if isinstance(scalar, (int, float)) and not isinstance(scalar, bool):
        try:
            validated = _validate_score(scalar, idx=0)
        except (TypeError, ValueError):
            return 0.0
        return score_uncertainty(scores=[validated])
    return 0.0


def pick_top_uncertain(
    rows: Iterable[Mapping[str, object]],
    *,
    budget: int,
) -> list[Mapping[str, object]]:
    """Pick the top-N rows by uncertainty.

    Stable on ties — earlier rows win to make the output deterministic.
    """
    n_budget = validate_budget(budget)
    materialised: list[Mapping[str, object]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError(
                f"rows must yield Mapping, got {type(row).__name__}"
            )
        materialised.append(row)
        if len(materialised) >= _MAX_INPUT_ROWS:
            break
    if not materialised:
        return []
    scored = [
        (idx, _row_uncertainty(row), row) for idx, row in enumerate(materialised)
    ]
    # Sort: highest uncertainty first, ties broken by original order.
    scored.sort(key=lambda triple: (-triple[1], triple[0]))
    return [row for (_, _, row) in scored[:n_budget]]


def sample_uncertain_rows(
    input_path: str,
    *,
    output_path: str,
    budget: int,
) -> ActiveLearningPlan:
    """Read JSONL, pick top-uncertainty rows, write out, return summary."""
    n_budget = validate_budget(budget)
    if not isinstance(input_path, str):
        raise TypeError(
            f"input_path must be str, got {type(input_path).__name__}"
        )
    if not isinstance(output_path, str):
        raise TypeError(
            f"output_path must be str, got {type(output_path).__name__}"
        )
    if not input_path or not output_path:
        raise ValueError("input/output paths must be non-empty")
    if "\x00" in input_path or "\x00" in output_path:
        raise ValueError("paths must not contain null bytes")
    if not is_under_cwd(input_path):
        raise ValueError(f"input_path {input_path!r} is outside cwd")
    if not is_under_cwd(output_path):
        raise ValueError(f"output_path {output_path!r} is outside cwd")
    if not os.path.isfile(input_path):
        raise FileNotFoundError(input_path)

    rows: list[Mapping[str, object]] = []
    with open(input_path, encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
            if len(rows) >= _MAX_INPUT_ROWS:
                break

    top = pick_top_uncertain(rows, budget=n_budget)

    # Compute mean uncertainty of selected rows for the report.
    mean_unc = 0.0
    if top:
        total = sum(_row_uncertainty(r) for r in top)
        mean_unc = total / len(top)

    with open(output_path, "w", encoding="utf-8") as fh_out:
        for row in top:
            fh_out.write(json.dumps(row, ensure_ascii=False) + "\n")

    return ActiveLearningPlan(
        rows_in=len(rows),
        rows_selected=len(top),
        budget=n_budget,
        mean_uncertainty=mean_unc,
    )


__all__ = [
    "ActiveLearningPlan",
    "pick_top_uncertain",
    "sample_uncertain_rows",
    "score_uncertainty",
    "validate_budget",
]
