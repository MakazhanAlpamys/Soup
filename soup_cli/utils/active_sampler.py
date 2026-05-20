"""Active-learning sampler — surface uncertain prod traces for review.

v0.63.0 Part C — picks the rows the model is *least confident* about so
humans only review what the policy itself thinks is borderline. Reduces
human-eval cost by 5-10x in practice.

Two modes via the input data shape:
1. Single reward-model score (`rm_score`) — uncertainty via max-entropy:
   ``1 - |2 * score - 1|``. Score 0.5 -> uncertainty 1.0 (peak),
   scores 0.0 or 1.0 -> uncertainty 0.0.
2. Two reward-model scores (`rm_scores: [s1, s2]`) — disagreement via
   ``|s1 - s2|``. Bigger gap -> higher uncertainty.

Composes with v0.19 human eval (the output JSONL is a drop-in human-eval
prompt set) and v0.58 `soup loop watch` (which can run this nightly).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence

from soup_cli.utils.paths import is_under_cwd

_MAX_BUDGET = 100_000
_MAX_INPUT_ROWS = 10_000_000  # 10M — production-scale day of traces


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


def score_uncertainty(*, scores: Sequence[float]) -> float:
    """Compute uncertainty from one or two reward-model scores.

    1 score: max-entropy distance from 0.5 (peak at 0.5 -> uncertainty 1.0)
    2 scores: pairwise disagreement (|s1 - s2|)
    """
    if not isinstance(scores, Sequence) or isinstance(scores, str):
        raise TypeError(
            f"scores must be a sequence, got {type(scores).__name__}"
        )
    if len(scores) == 0:
        return 0.0
    if len(scores) > 2:
        raise ValueError(
            "score_uncertainty supports 1 or 2 RM scores (v0.63.0). "
            "K>2 RMs deferred to a future release."
        )
    validated = [_validate_score(s, idx=i) for i, s in enumerate(scores)]
    if len(validated) == 1:
        # 1 - |2*s - 1|  -> peak at s=0.5
        return 1.0 - abs(2.0 * validated[0] - 1.0)
    # len == 2 -> pairwise disagreement
    return abs(validated[0] - validated[1])


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
        scores_list: List[float] = []
        try:
            for i, s in enumerate(scores_field):
                scores_list.append(_validate_score(s, idx=i))
        except (TypeError, ValueError):
            return 0.0
        if len(scores_list) <= 2:
            return score_uncertainty(scores=scores_list)
        # >2 scores: silently fall back to disagreement = max - min
        return max(scores_list) - min(scores_list)
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
) -> List[Mapping[str, object]]:
    """Pick the top-N rows by uncertainty.

    Stable on ties — earlier rows win to make the output deterministic.
    """
    n_budget = validate_budget(budget)
    materialised: List[Mapping[str, object]] = []
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

    rows: List[Mapping[str, object]] = []
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
