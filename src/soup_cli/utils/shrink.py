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
import re
from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence

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


def shrink_verdict_to_dict(verdict: ShrinkVerdict) -> dict[str, Any]:
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


# ---------------------------------------------------------------------------
# Arch allowlist + prune (torch-lazy)
# ---------------------------------------------------------------------------
# Note: real SmolLM/SmolLM2 checkpoints report model_type="llama", so the
# "smollm" entry is aspirational (any HF model that literally reports a "smol"
# model_type) — "llama" is matched first and wins for the released checkpoints.
_ARCH_PATTERNS = {
    "llama": re.compile(r"llama", re.I),
    "qwen": re.compile(r"qwen", re.I),
    "smollm": re.compile(r"smol", re.I),
}
SUPPORTED_SHRINK_ARCHS = tuple(_ARCH_PATTERNS)


def shrink_arch_of(model: object) -> str:
    """Return the supported family name for ``model`` or raise ``ValueError``.

    Detection is over ``config.model_type`` + ``config.architectures`` with
    regex word-family matching (mirrors ``longlora.is_*_model``). Only the v1
    families in :data:`SUPPORTED_SHRINK_ARCHS` (Llama / Qwen / SmolLM — all of
    which expose ``model.model.layers`` + ``config.num_hidden_layers``) are
    accepted; anything else is a friendly reject.
    """
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", "") or ""
    architectures = list(getattr(config, "architectures", []) or [])
    haystack = " ".join([str(model_type), *[str(a) for a in architectures]])
    for family, pattern in _ARCH_PATTERNS.items():
        if pattern.search(haystack):
            return family
    raise ValueError(
        f"soup shrink v1 supports {SUPPORTED_SHRINK_ARCHS}; got "
        f"model_type={model_type!r} (unsupported). Open an issue to add it."
    )


def layer_list(model: object) -> Any:
    """Return ``model.model.layers`` (the decoder ``ModuleList``), arch-guarded."""
    shrink_arch_of(model)  # raises on unsupported arch
    try:
        return model.model.layers  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise ValueError("model has no .model.layers ModuleList") from exc


def prune_model_layers(model: object, start: int, block_size: int) -> None:
    """Drop decoder layers ``[start, start + block_size)`` in place.

    Slices ``model.model.layers`` and patches ``config.num_hidden_layers``. The
    first and last decoder layers are protected (they carry the most residual
    transformation, per the paper), so the dropped block must stay within
    ``[1, num_layers - 1)``. Callers MUST reload the model from the saved dir
    before measuring/generating — slicing leaves each surviving layer's
    ``self_attn.layer_idx`` stale, which ``from_pretrained`` reconstructs
    correctly.
    """
    import torch.nn as nn

    layers = layer_list(model)
    n_total = len(layers)
    if not isinstance(start, int) or isinstance(start, bool):
        raise ValueError("start must be an int")
    if not isinstance(block_size, int) or isinstance(block_size, bool):
        raise ValueError("block_size must be an int")
    if block_size < 1 or block_size >= n_total:
        raise ValueError(f"block_size must be in [1, {n_total - 1}], got {block_size}")
    end = start + block_size  # exclusive
    if start < 1 or end > n_total - 1:
        raise ValueError(
            f"dropped block [{start}, {end}) must stay within [1, {n_total - 1}) "
            "(the first and last layer are protected)"
        )
    kept = [layers[i] for i in range(n_total) if not (start <= i < end)]
    model.model.layers = nn.ModuleList(kept)  # type: ignore[attr-defined]
    model.config.num_hidden_layers = len(kept)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Importance scan + block selection + drop-count resolution
# ---------------------------------------------------------------------------
_IMPORTANCE_MAX_LENGTH = 512
_DEFAULT_MAX_PROMPTS = 256


def resolve_drop_count(
    num_layers: int, *, drop_ratio: Optional[float], drop_layers: Optional[int]
) -> int:
    """Resolve the block **count** from exactly one of ratio / explicit count.

    ``drop_layers = round(drop_ratio * num_layers)`` when a ratio is given.
    Validated against the position bound ``1 <= count <= num_layers - 2`` (the
    first and last layer are always protected).
    """
    if (drop_ratio is None) == (drop_layers is None):
        raise ValueError("set exactly one of --drop-ratio / --drop-layers")
    if drop_layers is not None:
        if isinstance(drop_layers, bool) or not isinstance(drop_layers, int):
            raise ValueError("drop_layers must be an int")
        count = drop_layers
    else:
        if isinstance(drop_ratio, bool) or not isinstance(drop_ratio, (int, float)):
            raise ValueError("drop_ratio must be a number")
        if not (0.0 < float(drop_ratio) < 1.0):
            raise ValueError("drop_ratio must be in (0, 1)")
        count = round(float(drop_ratio) * num_layers)
    max_count = num_layers - 2  # protect first + last
    if not (1 <= count <= max_count):
        raise ValueError(
            f"drop count {count} out of range [1, {max_count}] for {num_layers} "
            "layers (first and last layer protected)"
        )
    return count


def compute_layer_importance(
    model: object,
    tokenizer: object,
    prompts: Sequence[str],
    *,
    block_size: int,
    device: str,
    max_prompts: int = _DEFAULT_MAX_PROMPTS,
) -> list[LayerImportance]:
    """Rank every position-valid contiguous block by residual angular distance.

    One ``output_hidden_states=True`` forward per calibration prompt captures
    every layer boundary at once (``hidden_states`` has ``num_layers + 1``
    entries; index 0 is the embedding output, index ``k`` the output of decoder
    layer ``k - 1``). For a block ``[L, L + block_size)`` the residual entering
    the block is ``hidden_states[L]`` and the residual leaving it is
    ``hidden_states[L + block_size]``. The block importance is the mean, over
    every non-pad token across the whole calib set, of the per-token angular
    distance ``arccos(cos) / pi`` between those two residuals (Gromov et al.).

    Only position-valid starts ``L in [1, num_layers - block_size - 1]`` are
    scored (first and last layer protected). Returns the blocks sorted by
    ascending distance (safest to drop first).
    """
    import torch

    n_layers = int(model.config.num_hidden_layers)  # type: ignore[attr-defined]
    valid_starts = list(range(1, n_layers - block_size))  # L in [1, n-bs-1]
    if not valid_starts:
        raise ValueError(
            f"block_size {block_size} leaves no position-valid block for "
            f"{n_layers} layers (first and last layer protected)"
        )
    prompt_list = [p for p in prompts if isinstance(p, str) and p.strip()][:max_prompts]
    if not prompt_list:
        raise ValueError("calib prompts must contain at least one non-empty string")

    sums = {s: 0.0 for s in valid_starts}
    counts = {s: 0 for s in valid_starts}
    model.eval()  # type: ignore[attr-defined]
    with torch.no_grad():
        for text in prompt_list:
            raw = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=_IMPORTANCE_MAX_LENGTH,
            )
            inputs = {k: v.to(device) for k, v in dict(raw).items()}
            out = model(**inputs, output_hidden_states=True)  # type: ignore[operator]
            hidden = out.hidden_states
            if len(hidden) != n_layers + 1:
                raise ValueError(
                    f"expected {n_layers + 1} hidden states, got {len(hidden)}"
                )
            mask = inputs.get("attention_mask")
            for start in valid_starts:
                h_in = hidden[start][0].to(torch.float32)  # [seq, D]
                h_out = hidden[start + block_size][0].to(torch.float32)
                cos = torch.nn.functional.cosine_similarity(h_in, h_out, dim=-1)
                cos = cos.clamp(-1.0, 1.0)
                dist = torch.arccos(cos) / torch.pi  # [seq]
                if mask is not None:
                    keep = mask[0].to(torch.bool)
                    dist = dist[keep]
                sums[start] += float(dist.sum().item())
                counts[start] += int(dist.numel())

    imps = [
        LayerImportance(
            start=s,
            block_size=block_size,
            angular_distance=(sums[s] / counts[s]) if counts[s] else float("inf"),
        )
        for s in valid_starts
    ]
    imps.sort(key=lambda x: x.angular_distance)
    return imps


def select_drop_block(importances: Sequence[LayerImportance]) -> LayerImportance:
    """Return the least-important (min angular-distance) candidate block."""
    if not importances:
        raise ValueError("no importance scores to select from")
    return min(importances, key=lambda x: x.angular_distance)
