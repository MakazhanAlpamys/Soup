"""Auto warmup scheduling (v0.32.0 Part D).

Computes a sensible ``warmup_steps`` from dataset size, batch, grad_accum,
epochs, and a warmup ratio. Clamped to a safe range so users with tiny
datasets still get some warmup, and users with huge datasets don't burn
half a million steps doing nothing useful.
"""

from __future__ import annotations

import inspect
import math
from typing import Any

try:
    from transformers import TrainingArguments
except ImportError:  # pragma: no cover
    TrainingArguments = None  # type: ignore[misc,assignment]

MIN_WARMUP = 10
MAX_WARMUP = 1000
DEFAULT_RATIO = 0.03
MAX_RATIO = 0.5


def compute_warmup_steps(
    num_examples: int,
    batch_size: int,
    grad_accum: int,
    epochs: int,
    ratio: float = DEFAULT_RATIO,
) -> int:
    """Return warmup steps clamped to [MIN_WARMUP, MAX_WARMUP].

    Special case: ``ratio == 0`` means "no warmup" and returns 0 (the
    schema allows ``warmup_ratio=0.0`` so this matches HF Trainer's
    convention).
    """
    if num_examples < 1:
        raise ValueError(f"num_examples must be >= 1, got {num_examples}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if grad_accum < 1:
        raise ValueError(f"grad_accum must be >= 1, got {grad_accum}")
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if not (0.0 <= ratio <= MAX_RATIO):
        raise ValueError(f"ratio must be in [0, {MAX_RATIO}], got {ratio}")
    if ratio == 0.0:
        return 0
    effective_batch = batch_size * grad_accum
    steps_per_epoch = max(1, math.ceil(num_examples / effective_batch))
    total_steps = steps_per_epoch * epochs
    raw = int(round(total_steps * ratio))
    return max(MIN_WARMUP, min(MAX_WARMUP, raw))


def inspect_training_arguments_warmup_support(
    training_args_cls: type | Any = TrainingArguments,
) -> bool:
    """Inspect whether warmup_ratio is accepted by training_args_cls.__init__."""
    if training_args_cls is None:
        try:
            from transformers import TrainingArguments as HFArgs

            training_args_cls = HFArgs
        except ImportError:
            return False
    try:
        sig = inspect.signature(training_args_cls.__init__)
        return "warmup_ratio" in sig.parameters
    except (TypeError, ValueError):
        return False


def resolve_trainer_warmup_steps(
    warmup_ratio: float | int | None = None,
    warmup_steps: int | None = None,
    total_training_steps: int | None = None,
) -> float | int:
    """Return the warmup steps or fraction to pass to HF TrainingArguments(warmup_steps=...).

    - If ``warmup_steps`` is explicitly set, use it.
    - If only ``warmup_ratio`` is set and ``total_training_steps`` is provided (> 0),
      dynamically calculate ``warmup_steps = max(1, int(total_training_steps * warmup_ratio))``.
    - If only ``warmup_ratio`` is set and ``total_training_steps`` is None, return warmup_ratio
      as a float in [0, 1) which HuggingFace TrainingArguments dynamically resolves over
      actual optimizer steps: ``warmup_steps = math.ceil(num_training_steps * warmup_steps)``.
    """
    if warmup_steps is not None:
        int_steps = int(warmup_steps)
        if int_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        return int_steps

    if warmup_ratio is None:
        return 0.0

    val = float(warmup_ratio)
    if not (0.0 <= val < 1.0):
        raise ValueError(f"warmup_ratio must be in [0.0, 1.0), got {val}")

    if val == 0.0:
        return 0 if total_training_steps is not None else 0.0

    if total_training_steps is not None and total_training_steps > 0:
        return max(1, int(total_training_steps * val))

    return val


def build_warmup_kwargs(
    warmup_ratio: float | int | None = None,
    warmup_steps: int | None = None,
    total_training_steps: int | None = None,
    training_args_cls: type | Any = TrainingArguments,
) -> dict[str, Any]:
    """Translate warmup configuration into backward-compatible TrainingArguments kwargs."""
    accepts_warmup_ratio = inspect_training_arguments_warmup_support(training_args_cls)

    if warmup_steps is not None:
        steps_val = int(warmup_steps)
        if steps_val < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        kwargs: dict[str, Any] = {"warmup_steps": steps_val}
        if accepts_warmup_ratio:
            kwargs["warmup_ratio"] = 0.0
        return kwargs

    if warmup_ratio is None:
        kwargs = {"warmup_steps": 0}
        if accepts_warmup_ratio:
            kwargs["warmup_ratio"] = 0.0
        return kwargs

    val = float(warmup_ratio)
    if not (0.0 <= val < 1.0):
        raise ValueError(f"warmup_ratio must be in [0.0, 1.0), got {val}")

    if val == 0.0:
        kwargs = {"warmup_steps": 0}
        if accepts_warmup_ratio:
            kwargs["warmup_ratio"] = 0.0
        return kwargs

    if total_training_steps is not None and total_training_steps > 0:
        calc_steps: float | int = max(1, int(total_training_steps * val))
    else:
        calc_steps = val

    if accepts_warmup_ratio:
        return {
            "warmup_ratio": val,
            "warmup_steps": calc_steps if isinstance(calc_steps, int) else 0,
        }

    return {"warmup_steps": calc_steps}


def sanitize_training_kwargs(
    kwargs: dict[str, Any],
    total_training_steps: int | None = None,
    training_args_cls: type | Any = TrainingArguments,
) -> dict[str, Any]:
    """Ensure training_kwargs is compatible with training_args_cls without raising TypeError."""
    accepts_warmup_ratio = inspect_training_arguments_warmup_support(training_args_cls)
    if not accepts_warmup_ratio and "warmup_ratio" in kwargs:
        ratio = kwargs.pop("warmup_ratio")
        if "warmup_steps" not in kwargs or kwargs["warmup_steps"] is None:
            if ratio is not None:
                val = float(ratio)
                if not (0.0 <= val < 1.0):
                    raise ValueError(f"warmup_ratio must be in [0.0, 1.0), got {val}")
                if val == 0.0:
                    kwargs["warmup_steps"] = 0
                elif total_training_steps is not None and total_training_steps > 0:
                    kwargs["warmup_steps"] = max(1, int(total_training_steps * val))
                else:
                    kwargs["warmup_steps"] = val
            else:
                kwargs["warmup_steps"] = 0
    return kwargs
