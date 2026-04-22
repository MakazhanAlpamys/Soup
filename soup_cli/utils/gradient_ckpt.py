"""Gradient checkpointing tiers — selective / medium / full / auto.

Gradient checkpointing trades compute for memory by re-computing activations
during the backward pass instead of storing them. Tiers control *how much*
re-computation happens:

- ``False`` / ``None``      — disabled (no memory savings)
- ``True`` / ``"full"``     — every transformer block (~30% slow, biggest save)
- ``"medium"``              — every other block (balance)
- ``"selective"``           — attention only (~10% slow, modest save)
- ``"auto"``                — pick based on detected VRAM headroom

``resolve_gradient_checkpointing`` returns a kwargs-dict suitable for
``TrainingArguments(**kwargs)``. Granularity (medium / selective) is a separate
concept — query ``resolve_granularity`` for the chosen tier so callers can
install the correct downstream hooks without polluting HF's kwargs surface.
"""

from __future__ import annotations

from typing import Any, Union

TierLike = Union[bool, str, None]


# Heuristic VRAM thresholds (GB) for tier=auto. Tuned for a 7-8B LoRA run at
# bf16 + 4bit quant + max_length≈4k.
# Below 24 GB: full checkpoint (largest memory savings).
# 24-80 GB:    medium (every other block) — balance.
# >80 GB:      selective (attention only) — minimize slowdown.
AUTO_FULL_THRESHOLD_GB = 24.0
AUTO_SELECTIVE_THRESHOLD_GB = 80.0


def resolve_granularity(
    tier: TierLike, gpu_memory_gb: float | None = None,
) -> str | None:
    """Return the granularity string the wrapper should install hooks for.

    One of: ``"full"`` | ``"medium"`` | ``"selective"`` | ``None`` (disabled).
    ``"auto"`` is resolved to full/medium/selective based on ``gpu_memory_gb``.
    """
    if not tier:
        return None
    if tier is True or tier == "full":
        return "full"
    if tier in ("medium", "selective"):
        return tier  # type: ignore[return-value]
    if tier == "auto":
        if gpu_memory_gb is None:
            return "full"
        if gpu_memory_gb < AUTO_FULL_THRESHOLD_GB:
            return "full"
        if gpu_memory_gb <= AUTO_SELECTIVE_THRESHOLD_GB:
            return "medium"
        return "selective"
    return None


def resolve_gradient_checkpointing(
    tier: TierLike, gpu_memory_gb: float | None = None,
) -> dict[str, Any]:
    """Resolve a gradient_checkpointing setting into TrainingArguments kwargs.

    Only returns keys that HuggingFace's ``TrainingArguments`` actually accepts.
    Granularity (medium/selective) is not represented here; query
    ``resolve_granularity`` for that.

    Args:
        tier: TrainingConfig.gradient_checkpointing value (bool or tier string).
        gpu_memory_gb: GPU memory (GB) used by ``"auto"`` tier. If None, falls
            back to full checkpointing on auto.

    Returns:
        Dict of kwargs suitable for ``TrainingArguments(**kwargs)``:
        - ``gradient_checkpointing`` (bool)
        - ``gradient_checkpointing_kwargs`` (dict)
    """
    granularity = resolve_granularity(tier, gpu_memory_gb=gpu_memory_gb)
    if granularity is None:
        return {}

    # All granularities use HF's standard non-reentrant checkpointing at the
    # TrainingArguments level. Selective / medium installation happens inside
    # the wrapper via torch-level hooks (deferred to v0.28.1 wiring), without
    # leaking markers into HF's kwargs surface.
    return {
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
    }


def describe_tier(tier: TierLike, gpu_memory_gb: float | None = None) -> str:
    """Return a short human-readable description of the selected tier."""
    if not tier:
        return "off"
    if tier is True or tier == "full":
        return "full (every block)"
    if tier == "medium":
        return "medium (every other block)"
    if tier == "selective":
        return "selective (attention only)"
    if tier == "auto":
        if gpu_memory_gb is None:
            return "auto → full (unknown VRAM)"
        if gpu_memory_gb < AUTO_FULL_THRESHOLD_GB:
            return f"auto → full (VRAM {gpu_memory_gb:.0f}GB)"
        if gpu_memory_gb <= AUTO_SELECTIVE_THRESHOLD_GB:
            return f"auto → medium (VRAM {gpu_memory_gb:.0f}GB)"
        return f"auto → selective (VRAM {gpu_memory_gb:.0f}GB)"
    return str(tier)
