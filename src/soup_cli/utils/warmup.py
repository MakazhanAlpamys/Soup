"""Auto warmup scheduling (v0.32.0 Part D).

Computes a sensible ``warmup_steps`` from dataset size, batch, grad_accum,
epochs, and a warmup ratio. Clamped to a safe range so users with tiny
datasets still get some warmup, and users with huge datasets don't burn
half a million steps doing nothing useful.
"""

from __future__ import annotations

import math

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
        raise ValueError(
            f"ratio must be in [0, {MAX_RATIO}], got {ratio}"
        )
    if ratio == 0.0:
        return 0
    effective_batch = batch_size * grad_accum
    steps_per_epoch = max(1, math.ceil(num_examples / effective_batch))
    total_steps = steps_per_epoch * epochs
    raw = int(round(total_steps * ratio))
    return max(MIN_WARMUP, min(MAX_WARMUP, raw))
