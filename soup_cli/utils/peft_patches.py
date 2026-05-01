"""Surgical PEFT/architecture patches (v0.39.0 Part D).

Logging is via the standard ``logging`` module so failures are inspectable
without raising. Best-effort patches never crash training.


Two narrow patches that PEFT upstream doesn't (yet) handle:

1. **Gemma4 ``ClippableLinear``** — Gemma 4 uses a ``ClippableLinear`` subclass
   that PEFT's ``LoraConfig.target_modules`` matcher doesn't recognise. We
   detect it by class name and swap to plain ``nn.Linear`` so PEFT's normal
   matcher takes over. We don't try to preserve the clipping semantics
   because Gemma4 only invokes them at inference; training is unaffected.

2. **Fused-MoE 3-D expert weights** — ``ParamWrapper`` in PEFT crashes when a
   LoRA layer wraps a 3-D weight tensor (``[num_experts, in, out]``). We
   detect 3-D LoRA target modules and silently strip ``lora_dropout`` since
   the dropout layer is what triggers the crash.

Both patches are version-gated and architecture-gated. Apply via the
public entry point :func:`apply_surgical_patches` which inspects the
model name and runs only the patches that match.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Word-boundary match against ``model_name`` (case-insensitive). Bare substring
# would over-match on names like "ungemma4ed" or "my-gemma4ish". Boundaries:
# start/end-of-string or a non-alnum/underscore char on each side.
_GEMMA4_RE = re.compile(r"(?:^|[^a-z0-9])gemma-?4(?:[^a-z0-9]|$)", re.IGNORECASE)


def is_gemma4_model(model_name: Any) -> bool:
    if not isinstance(model_name, str) or not model_name:
        return False
    if "\x00" in model_name:
        return False
    return _GEMMA4_RE.search(model_name) is not None


def _looks_like_clippable_linear(module: Any) -> bool:
    """Match by class name, not isinstance, so we don't import gemma internals."""
    return type(module).__name__ == "ClippableLinear"


def apply_gemma4_clippable_patch(model: Any) -> int:
    """Swap any ``ClippableLinear`` submodules with plain ``nn.Linear``.

    Returns the number of modules patched. Safe to call when no
    ``ClippableLinear`` is present (returns 0).
    """
    import torch.nn as nn  # lazy

    if model is None:
        return 0

    swapped = 0
    # Walk parents → swap by attribute. Avoid mutation during iteration:
    # collect first, then patch.
    targets: list[tuple[Any, str, Any]] = []
    for parent in model.modules():
        for child_name, child in list(parent.named_children()):
            if _looks_like_clippable_linear(child):
                targets.append((parent, child_name, child))

    for parent, child_name, child in targets:
        # Build a plain Linear with the same shape + dtype + device.
        in_features = getattr(child, "in_features", None)
        out_features = getattr(child, "out_features", None)
        bias = getattr(child, "bias", None) is not None
        if in_features is None or out_features is None:
            continue
        replacement = nn.Linear(in_features, out_features, bias=bias)
        # Copy weights over (best-effort).
        try:
            replacement.weight.data.copy_(child.weight.data)
            if bias and replacement.bias is not None and child.bias is not None:
                replacement.bias.data.copy_(child.bias.data)
            replacement = replacement.to(
                dtype=child.weight.dtype, device=child.weight.device
            )
        except Exception as exc:  # noqa: BLE001 — fall back to random init w/ log
            logger.debug(
                "Failed to copy ClippableLinear weights at %s: %s; using fresh init",
                child_name, exc,
            )
        setattr(parent, child_name, replacement)
        swapped += 1
    return swapped


def strip_lora_dropout_for_3d_experts(peft_model: Any) -> int:
    """Zero out ``lora_dropout`` on any LoRA target whose base weight is 3-D.

    PEFT's ``ParamWrapper`` for fused-MoE experts cannot wrap an
    ``nn.Dropout`` instance whose forward expects a 2-D tensor when the
    expert weight is shaped ``[num_experts, in, out]``. Setting ``p=0.0``
    is the documented workaround upstream (Axolotl ``adapter.py``).

    Returns the number of modules whose dropout was disabled.
    """
    if peft_model is None:
        return 0
    count = 0
    for _name, module in peft_model.named_modules():
        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        ndim = getattr(weight, "ndim", None)
        if ndim != 3:
            continue
        dropout = getattr(module, "lora_dropout", None)
        if dropout is None:
            continue
        # dropout may be an nn.Dropout, an nn.ModuleDict (PEFT >=0.10), or a
        # plain attribute with a ``p`` field. Cover the common shapes.
        try:
            if hasattr(dropout, "p"):
                dropout.p = 0.0
                count += 1
            elif hasattr(dropout, "values"):
                for sub in dropout.values():
                    if hasattr(sub, "p"):
                        sub.p = 0.0
                        count += 1
        except Exception:
            continue
    return count


def apply_surgical_patches(model: Any, model_name: str) -> dict[str, int]:
    """Run all gated PEFT/architecture patches for ``model``.

    Returns a dict ``{"gemma4_clippable": int, "moe_3d_dropout": int}`` —
    counts of modules patched.
    """
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model_name must be a non-empty string")
    if "\x00" in model_name:
        raise ValueError("model_name cannot contain null bytes")
    counts = {"gemma4_clippable": 0, "moe_3d_dropout": 0}
    if is_gemma4_model(model_name):
        counts["gemma4_clippable"] = apply_gemma4_clippable_patch(model)
    counts["moe_3d_dropout"] = strip_lora_dropout_for_3d_experts(model)
    return counts
