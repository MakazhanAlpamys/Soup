"""Convergence / plateau detection (v0.32.0 Part F).

Pure helpers. No torch import. Used by the monitoring callback to surface
"loss has plateaued for N steps — consider early stop or LR cut" advice.
"""

from __future__ import annotations

import math
from statistics import fmean, pstdev
from typing import Literal, Sequence

MIN_WINDOW = 1
MAX_WINDOW = 10_000
MAX_REL_TOL = 1.0


def detect_plateau(
    losses: Sequence[float], window: int = 50, rel_tol: float = 0.005,
) -> bool:
    """True when relative range of the last ``window`` losses < ``rel_tol``.

    Relative range = (max - min) / mean. If the window's mean is non-positive
    or non-finite, returns False (refuses to assess).
    """
    if not (MIN_WINDOW <= window <= MAX_WINDOW):
        raise ValueError(
            f"window must be in [{MIN_WINDOW}, {MAX_WINDOW}], got {window}"
        )
    if not (0.0 <= rel_tol <= MAX_REL_TOL):
        raise ValueError(
            f"rel_tol must be in [0, {MAX_REL_TOL}], got {rel_tol}"
        )
    if len(losses) < window:
        return False
    tail = losses[-window:]
    mean = fmean(tail)
    if not math.isfinite(mean) or mean <= 0:
        return False
    rng = max(tail) - min(tail)
    return (rng / mean) < rel_tol


def recommend_action(
    losses: Sequence[float],
    window: int = 50,
    rel_tol: float = 0.005,
    osc_cv: float = 0.10,
) -> Literal["continue", "early_stop", "lower_lr"]:
    """Map current loss curve to a single advice string.

    - Plateau (low variance, no descent): ``early_stop``
    - High coefficient-of-variation in window with no clear trend: ``lower_lr``
    - Otherwise: ``continue``
    """
    if len(losses) < max(window, 4):
        return "continue"
    if detect_plateau(losses, window=window, rel_tol=rel_tol):
        return "early_stop"

    tail = losses[-window:]
    mean = fmean(tail)
    if not math.isfinite(mean) or mean <= 0:
        return "continue"

    cv = pstdev(tail) / mean
    # Slope = (last - first) / first; oscillation = high CV but no real descent.
    slope = (tail[-1] - tail[0]) / tail[0] if tail[0] > 0 else 0.0
    if cv > osc_cv and abs(slope) < osc_cv:
        return "lower_lr"
    return "continue"
