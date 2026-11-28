"""Shared final-save validation for non-finite training metrics and parameters."""

from __future__ import annotations

import math
from typing import Any

_FINITE_TRAINING_METRICS = (
    "loss",
    "grad_norm",
    "entropy",
    "train_loss",
    "eval_loss",
)


def assert_finite_training_state(
    log_history: list[dict[str, Any]],
    model: Any | None = None,
    *,
    metric_key: str = "loss",
) -> None:
    """Refuse the final save when metrics or trainable weights are non-finite.

    Only the most recently logged value of each metric is checked. ``log_history``
    accumulates one entry per log call over the whole run, and a metric that was
    transiently non-finite earlier (e.g. a GradScaler warm-up nan) but recovered
    says nothing about the final state; checking every past entry made a
    self-corrected run indistinguishable from a genuinely corrupted one.
    """
    metrics_to_check = (
        _FINITE_TRAINING_METRICS
        if metric_key in _FINITE_TRAINING_METRICS
        else (*_FINITE_TRAINING_METRICS, metric_key)
    )
    for metric in metrics_to_check:
        for entry in reversed(log_history):
            if not isinstance(entry, dict) or metric not in entry:
                continue
            value = entry[metric]
            try:
                finite = math.isfinite(float(value))
            except (TypeError, ValueError):
                break
            if finite:
                break
            step = entry.get("step", "unknown")
            raise RuntimeError(
                f"non-finite training metric {metric}={value} at step {step}; "
                "refusing to save the final model because its weights may be corrupted"
            )
    if model is None:
        return

    import torch

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or getattr(parameter, "is_meta", False):
            continue
        if not parameter.is_floating_point():
            continue
        if torch.isfinite(parameter.detach()).all().item():
            continue
        raise RuntimeError(
            f"non-finite trainable parameter {name!r}; refusing to save the final "
            "model because its weights are corrupted"
        )
