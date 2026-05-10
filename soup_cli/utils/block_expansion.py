"""LLaMA Pro block expansion — v0.41.0 Part C (schema + helper).

LLaMA Pro adds ``N`` zero-initialised transformer blocks to a base model and
freezes the original blocks, training only the new blocks. This implements
the schema validators and a thin block-insertion helper that operates on
HF causal-LM model objects (lazy import).

Live wiring (trainer-side) lands in v0.41.1 follow-up — the schema gate
ensures users cannot enable a stub-then-live combination silently.

References:
- LlamaFactory ``freeze_trainable_layers`` (positive = train top-N, negative
  = train bottom-N) + ``expand_layers`` (block count).
"""

from __future__ import annotations

from typing import Any

_MAX_EXPAND_LAYERS = 64
_MIN_EXPAND_LAYERS = 1


def validate_expand_layers(value: object) -> int:
    """Validate ``training.expand_layers`` (LLaMA Pro)."""
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"expand_layers must be int, got {type(value).__name__}"
        )
    if value < _MIN_EXPAND_LAYERS or value > _MAX_EXPAND_LAYERS:
        raise ValueError(
            f"expand_layers must be in [{_MIN_EXPAND_LAYERS}, "
            f"{_MAX_EXPAND_LAYERS}], got {value}"
        )
    return int(value)


def validate_freeze_trainable_layers(value: object) -> int:
    """Signed int — positive = train top-N, negative = train bottom-N."""
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"freeze_trainable_layers must be int, got {type(value).__name__}"
        )
    if abs(value) > 1000:
        raise ValueError(
            f"freeze_trainable_layers magnitude must be <= 1000, got {value}"
        )
    return int(value)


def expand_model_blocks(model: Any, num_new_blocks: int) -> int:
    """Insert ``num_new_blocks`` zero-init transformer layers at the end.

    Returns the total number of layers after expansion. The helper is
    schema-only in v0.41.0 — full wiring (zero-init weights, freeze
    original layers, route through HF Trainer) lands in v0.41.1.

    Passing ``num_new_blocks=0`` is the no-op path — useful for callers
    that want to query layer count without triggering the deferred
    NotImplementedError.
    """
    if num_new_blocks is None or num_new_blocks == 0:
        return _count_layers(model)
    validate_expand_layers(num_new_blocks)
    raise NotImplementedError(
        "expand_model_blocks live wiring is deferred to v0.41.1 — schema "
        "fields ship in v0.41.0 to lock the surface."
    )


def _count_layers(model: Any) -> int:
    """Best-effort count of decoder layers on an HF causal-LM."""
    inner = getattr(model, "model", None) or model
    layers = getattr(inner, "layers", None)
    if layers is None and hasattr(inner, "decoder"):
        layers = getattr(inner.decoder, "layers", None)
    if layers is None or not hasattr(layers, "__len__"):
        return 0
    return len(layers)
