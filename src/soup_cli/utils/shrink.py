"""soup shrink — depth-prune + distill-heal (v0.71.29, arXiv:2403.17887).

"The Unreasonable Ineffectiveness of the Deeper Layers" (Gromov et al.): rank a
model's decoder layers by the angular distance of the residual stream across a
contiguous block over a calibration set, drop the least-important block, then
optionally *heal* by distilling the original model into the pruned student.

This module has two halves:

* a **pure** verdict half (frozen dataclasses + ``decide_shrink`` +
  ``render_shrink_panel`` + ``shrink_verdict_to_dict``) with NO top-level torch
  import, so it is fully CPU-testable and cheap to import; and
* a **torch-lazy** prune / importance half (``compute_layer_importance``,
  ``select_drop_block``, ``prune_model_layers``, arch allowlist) whose heavy
  imports happen inside the functions.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from rich.panel import Panel

from soup_cli import __version__

DECISION_SHIP = "SHIP"
DECISION_DONT_SHIP = "DON'T SHIP"
DEFAULT_TOLERANCE = 0.10
MAX_TOLERANCE = 5.0

# Verdict ratio epsilon so an exact-boundary drop (ratio-1 == tolerance) SHIPs.
_RATIO_EPS = 1e-9


# ---------------------------------------------------------------------------
# Frozen dataclasses (pure)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LayerImportance:
    """One candidate contiguous block, ranked by residual angular distance."""

    start: int  # first dropped decoder layer (0-indexed)
    block_size: int  # number of layers in the block
    angular_distance: float  # mean per-token angular distance (lower = safer to drop)


@dataclass(frozen=True)
class ShrinkVerdict:
    """The binary shrink decision plus the evidence that produced it."""

    decision: str  # DECISION_SHIP | DECISION_DONT_SHIP
    ppl_original: float
    ppl_final: float
    ppl_ratio: float  # ppl_final / ppl_original
    tolerance: float
    layers_before: int
    layers_after: int
    params_saved_pct: float
    healed: bool
    soup_version: str


# ---------------------------------------------------------------------------
# Verdict (pure)
# ---------------------------------------------------------------------------
def _finite_positive(value: object, name: str) -> float:
    """Coerce ``value`` to a finite, strictly-positive float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return out


def decide_shrink(
    ppl_original: object,
    ppl_final: object,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    layers_before: int,
    layers_after: int,
    params_saved_pct: float = 0.0,
    healed: bool = False,
    soup_version: str = __version__,
) -> ShrinkVerdict:
    """SHIP iff ``ppl_final / ppl_original - 1 <= tolerance``.

    ``decide_ship`` (soup ship) would trivially reject every shrink because
    pruning always raises perplexity — so shrink has its own rule: the pruned
    (and optionally healed) model ships when its perplexity regression stays
    within ``tolerance`` (absolute ratio, default 10 %).
    """
    orig = _finite_positive(ppl_original, "ppl_original")
    final = _finite_positive(ppl_final, "ppl_final")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise ValueError("tolerance must be a number")
    tol = float(tolerance)
    if not math.isfinite(tol) or not (0.0 <= tol <= MAX_TOLERANCE):
        raise ValueError(f"tolerance must be in [0.0, {MAX_TOLERANCE}]")
    ratio = final / orig
    decision = DECISION_SHIP if (ratio - 1.0) <= tol + _RATIO_EPS else DECISION_DONT_SHIP
    return ShrinkVerdict(
        decision=decision,
        ppl_original=round(orig, 4),
        ppl_final=round(final, 4),
        ppl_ratio=round(ratio, 4),
        tolerance=tol,
        layers_before=int(layers_before),
        layers_after=int(layers_after),
        params_saved_pct=round(float(params_saved_pct), 2),
        healed=bool(healed),
        soup_version=str(soup_version),
    )


def shrink_verdict_to_dict(verdict: ShrinkVerdict) -> dict:
    """Plain-dict view of a ``ShrinkVerdict`` (JSON-serialisable)."""
    return asdict(verdict)


def render_shrink_panel(verdict: ShrinkVerdict) -> Panel:
    """One-screen Rich panel summarising the shrink verdict."""
    color = "green" if verdict.decision == DECISION_SHIP else "red"
    body = (
        f"[bold]{verdict.decision}[/]\n\n"
        f"Layers: {verdict.layers_before} -> {verdict.layers_after}  "
        f"(params saved {verdict.params_saved_pct:.1f}%)\n"
        f"Perplexity: {verdict.ppl_original:.3f} -> {verdict.ppl_final:.3f}  "
        f"(x{verdict.ppl_ratio:.3f}, tolerance {verdict.tolerance:.0%})\n"
        f"Healed: {'yes' if verdict.healed else 'no'}"
    )
    return Panel(body, title="soup shrink", border_style=color)
