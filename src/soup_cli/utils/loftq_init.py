"""LoftQ init — v0.41.0 Part C.

Quantization-aware LoRA initialization. PEFT supports LoftQ via
``init_lora_weights="loftq"`` plus a ``loftq_config`` carrying iteration count
and bits. We surface a thin wrapper so trainers can opt in via
``training.lora.init_strategy='loftq'`` without each wrapper re-implementing
the branch.

Schema validation:
- ``loftq_iter`` ∈ [1, 10] (default 1).
- ``loftq_bits`` ∈ {2, 4, 8} (default 4).

Live wiring:
- peft >= 0.7 ships ``LoftQConfig``; we lazy-import to keep the CLI cold-start
  fast. Older peft versions raise a friendly error.
"""

from __future__ import annotations

from typing import Any

_VALID_LOFTQ_BITS = (2, 4, 8)
_LOFTQ_ITER_MIN = 1
_LOFTQ_ITER_MAX = 10


def validate_loftq_iter(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"loftq_iter must be int, got {type(value).__name__}"
        )
    if value < _LOFTQ_ITER_MIN or value > _LOFTQ_ITER_MAX:
        raise ValueError(
            f"loftq_iter must be in [{_LOFTQ_ITER_MIN}, {_LOFTQ_ITER_MAX}], "
            f"got {value}"
        )
    return int(value)


def validate_loftq_bits(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"loftq_bits must be int, got {type(value).__name__}"
        )
    if value not in _VALID_LOFTQ_BITS:
        raise ValueError(
            f"loftq_bits must be one of {_VALID_LOFTQ_BITS}, got {value}"
        )
    return int(value)


def build_loftq_config(loftq_iter: int = 1, loftq_bits: int = 4) -> Any:
    """Lazy-construct a peft.LoftQConfig.

    Raises ImportError with an actionable message if peft is too old.
    """
    iter_v = validate_loftq_iter(loftq_iter)
    bits_v = validate_loftq_bits(loftq_bits)
    try:
        from peft import LoftQConfig  # type: ignore
    except ImportError as exc:  # pragma: no cover — covered indirectly
        raise ImportError(
            "LoftQ requires peft >= 0.7. "
            "pip install --upgrade peft"
        ) from exc
    return LoftQConfig(loftq_bits=bits_v, loftq_iter=iter_v)
