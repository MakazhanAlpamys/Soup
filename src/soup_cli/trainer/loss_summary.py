"""Shared extraction for training-completion loss summaries."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def _finite_loss(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        loss = float(value)
    except (TypeError, ValueError):
        return None
    return loss if math.isfinite(loss) else None


def summarize_training_loss(
    log_history: Sequence[Any], final_metrics: Mapping[str, Any] | None = None
) -> dict[str, float | str]:
    """Return result fields without inventing a loss delta.

    Hugging Face writes per-step values under ``loss`` and the final run mean
    under ``train_loss``. A run with fewer steps than ``logging_steps`` can
    therefore have only the mean. Keep the existing numeric result fields for
    trackers and sweep ranking, while recording whether the live summary owns
    a real two-point delta.
    """
    per_step: list[float] = []
    for entry in log_history:
        if not isinstance(entry, Mapping) or "loss" not in entry:
            continue
        loss = _finite_loss(entry["loss"])
        if loss is not None:
            per_step.append(loss)

    if len(per_step) >= 2:
        return {
            "initial_loss": per_step[0],
            "final_loss": per_step[-1],
            "loss_summary_kind": "delta",
        }
    if len(per_step) == 1:
        return {
            "initial_loss": per_step[0],
            "final_loss": per_step[0],
            "loss_summary_kind": "single",
        }

    for entry in reversed(log_history):
        if not isinstance(entry, Mapping) or "train_loss" not in entry:
            continue
        mean_loss = _finite_loss(entry["train_loss"])
        if mean_loss is not None:
            return {
                "initial_loss": mean_loss,
                "final_loss": mean_loss,
                "loss_summary_kind": "mean",
            }

    if final_metrics is not None and "train_loss" in final_metrics:
        mean_loss = _finite_loss(final_metrics["train_loss"])
        if mean_loss is not None:
            return {
                "initial_loss": mean_loss,
                "final_loss": mean_loss,
                "loss_summary_kind": "mean",
            }

    return {
        "initial_loss": 0.0,
        "final_loss": 0.0,
        "loss_summary_kind": "unavailable",
    }
