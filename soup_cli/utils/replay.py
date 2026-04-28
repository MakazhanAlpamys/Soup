"""Replay metric history (v0.34.0 Part E).

Pure-Python replay logic. The CLI command is a thin wrapper that pulls
metric rows from the experiment tracker and asks this module to render
a summary table + (optional) loss curve. Kept separate from the live
``TrainingDisplay`` so the replay path doesn't accidentally pull in
``rich.live`` / threading just to print a static table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

# CLI hard-cap: cap rendered points to keep the terminal renderer responsive
# even on long runs that logged tens of thousands of steps.
MAX_PLOT_POINTS = 2000

# Minimum step delta to keep when sub-sampling. Otherwise large stretches
# can collapse into "step 100, 100, 100" duplicates after coarsening.
_MIN_STEP_DELTA = 1


@dataclass(frozen=True)
class ReplaySummary:
    total_rows: int
    first_step: Optional[int]
    last_step: Optional[int]
    initial_loss: Optional[float]
    final_loss: Optional[float]
    min_loss: Optional[float]
    min_loss_step: Optional[int]


def _finite(value) -> bool:
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value)


def summarise(metrics: Sequence[dict]) -> ReplaySummary:
    """Reduce a metric series into headline numbers."""
    if not metrics:
        return ReplaySummary(0, None, None, None, None, None, None)
    finite = [row for row in metrics if _finite(row.get("loss"))]
    if not finite:
        first_step = metrics[0].get("step")
        last_step = metrics[-1].get("step")
        return ReplaySummary(len(metrics), first_step, last_step, None, None, None, None)
    initial_loss = finite[0]["loss"]
    final_loss = finite[-1]["loss"]
    min_row = min(finite, key=lambda row: row["loss"])
    return ReplaySummary(
        total_rows=len(metrics),
        first_step=metrics[0].get("step"),
        last_step=metrics[-1].get("step"),
        initial_loss=float(initial_loss),
        final_loss=float(final_loss),
        min_loss=float(min_row["loss"]),
        min_loss_step=min_row.get("step"),
    )


def downsample(
    metrics: Sequence[dict],
    *,
    max_points: int = MAX_PLOT_POINTS,
) -> List[dict]:
    """Return at most ``max_points`` rows preserving order + endpoints.

    Uses uniform stride sampling so chart shape is preserved.
    """
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    rows = list(metrics)
    if len(rows) <= max_points:
        return rows
    stride = max(_MIN_STEP_DELTA, len(rows) // max_points)
    sampled = rows[::stride]
    # Always pin the last row so the final loss is visible.
    if sampled[-1] is not rows[-1]:
        sampled.append(rows[-1])
    return sampled
