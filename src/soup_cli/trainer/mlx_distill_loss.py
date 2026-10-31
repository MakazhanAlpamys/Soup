"""FP32 token-distillation sums for complete token-weighted optimizer windows.

No backend import at module scope. The caller supplies active shifted indices
from CPU-side collation, avoiding dynamic boolean indexing or per-chunk host
synchronization. Chunking bounds one divergence operation, not the full logits
or all backward intermediates; native checkpointing recomputes the latter.
"""

from __future__ import annotations

import math
from typing import Any


def token_loss_sums(
    student: Any,
    teacher: Any,
    labels: Any,
    active_indices: Any,
    *,
    divergence: str = "forward_kl",
    temperature: float = 2.0,
    chunk_size: int | None = None,
    recompute: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Return ``(0.5*CE_sum + 0.5*KD_sum, metrics)`` over causal targets.

    ``active_indices`` indexes flattened ``labels[:, 1:]`` excluding -100.
    The runtime builds it from the exact supported token masks before tracing.
    Normalize these sums once by the full optimizer window's active count.
    Teacher logits are stopped even when this kernel is differentiated directly.
    An empty selection returns a differentiable FP32 zero; the trainer refuses
    empty rows/windows. Finite flags concern active logits only; the runtime
    evaluates them with loss/gradients and refuses a bad update.
    """
    import mlx.core as mx
    import mlx.nn as nn

    from soup_cli.utils.distill import validate_distill_temperature, validate_divergence

    divergence = validate_divergence(divergence)
    temperature = validate_distill_temperature(temperature)
    if chunk_size is not None and (type(chunk_size) is not int or chunk_size < 1):
        raise ValueError("distill_chunk_size must be a positive int or None")
    if type(recompute) is not bool:
        raise ValueError("distill_checkpoint must be bool")
    if student.ndim != 3 or teacher.shape != student.shape or labels.shape != student.shape[:2]:
        raise ValueError("MLX distillation requires matching B/S/V logits and B/S labels")
    if active_indices.ndim != 1 or active_indices.dtype not in (mx.int32, mx.int64):
        raise ValueError("MLX distillation active indices must be a one-dimensional integer array")

    vocab = student.shape[-1]
    s = student[:, :-1].reshape(-1, vocab)[active_indices].astype(mx.float32)
    t = mx.stop_gradient(teacher[:, :-1].reshape(-1, vocab)[active_indices]).astype(mx.float32)
    targets = labels[:, 1:].reshape(-1)[active_indices]
    finite = {"student_finite": mx.all(mx.isfinite(s)), "teacher_finite": mx.all(mx.isfinite(t))}
    count = active_indices.size
    if count == 0:
        zero = mx.sum(s)
        return zero, {**finite, "ce_sum": zero, "kd_sum": zero}

    ce = nn.losses.cross_entropy(s, targets, reduction="sum")

    def divergence_sum(student_chunk: Any, teacher_chunk: Any) -> Any:
        log_s = nn.log_softmax(student_chunk / temperature, axis=-1)
        log_t = nn.log_softmax(teacher_chunk / temperature, axis=-1)
        if divergence == "forward_kl":
            values = mx.exp(log_t) * (log_t - log_s)
        elif divergence == "reverse_kl":
            values = mx.exp(log_s) * (log_s - log_t)
        else:
            log_m = mx.logaddexp(log_s, log_t) - math.log(2)
            values = (mx.exp(log_t) * (log_t - log_m) + mx.exp(log_s) * (log_s - log_m)) / 2
        return mx.sum(values)

    compute = mx.checkpoint(divergence_sum) if recompute else divergence_sum
    width = chunk_size or count
    kd = mx.array(0, dtype=mx.float32)
    for start in range(0, count, width):
        kd = kd + compute(s[start:start + width], t[start:start + width])
    kd = kd * temperature**2
    return (ce + kd) / 2, {**finite, "ce_sum": ce, "kd_sum": kd}
