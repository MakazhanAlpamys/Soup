"""Shared extraction for training-completion loss summaries."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from soup_cli.experiment.row_kinds import (
    LOSS_SUMMARY_MEAN,
    LOSS_SUMMARY_PER_STEP,
    LOSS_SUMMARY_UNAVAILABLE,
)

#: The trainers' own ``loss_summary_kind`` -> the value ``runs.loss_summary_kind`` stores.
_STORED_KIND = {
    "delta": LOSS_SUMMARY_PER_STEP,
    "single": LOSS_SUMMARY_PER_STEP,
    "mean": LOSS_SUMMARY_MEAN,
    "unavailable": LOSS_SUMMARY_UNAVAILABLE,
}


def _numeric_loss(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _finite_loss(value: object) -> float | None:
    loss = _numeric_loss(value)
    if loss is None:
        return None
    return loss if math.isfinite(loss) else None


def summarize_training_loss(
    log_history: Sequence[Any],
    final_metrics: Mapping[str, Any] | None = None,
    loss_key: str = "loss",
) -> dict[str, float | str]:
    """Return result fields without inventing a loss delta.

    Hugging Face writes per-step values under ``loss`` and the final run mean
    under ``train_loss``. A run with fewer steps than ``logging_steps`` can
    therefore have only the mean. Keep the existing numeric result fields for
    trackers and sweep ranking, while recording whether the live summary owns
    a real two-point delta.

    Trainers whose framework logs per-step losses under a different key pass
    it explicitly via ``loss_key`` -- trl's PPO trainer (0.29) logs
    ``loss/policy_avg``. The key is chosen by the caller, never guessed, and
    a non-default key is recorded in the result so a PPO policy loss is not
    mistaken for an SFT cross-entropy.
    """
    per_step: list[float] = []
    for entry in log_history:
        if not isinstance(entry, Mapping) or loss_key not in entry:
            continue
        loss = _numeric_loss(entry[loss_key])
        if loss is not None:
            per_step.append(loss)

    def _tag(result: dict[str, float | str]) -> dict[str, float | str]:
        if loss_key != "loss":
            result["loss_key"] = loss_key
        return result

    if len(per_step) >= 2:
        return _tag(
            {
                "initial_loss": per_step[0],
                "final_loss": per_step[-1],
                "loss_summary_kind": "delta",
            }
        )
    if len(per_step) == 1:
        return _tag(
            {
                "initial_loss": per_step[0],
                "final_loss": per_step[0],
                "loss_summary_kind": "single",
            }
        )

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


def _stored_number(value: object) -> float | None:
    """A number as a float; None for anything else (bool, str and bytes included).

    Unlike :func:`_numeric_loss` this does not parse text: a ``"0"`` in a result dict
    is not a measurement. Anything else that converts with ``float()`` (a numpy or
    torch scalar a trainer left in its result) still counts, as it did when the
    panel formatted the value directly.
    """
    if isinstance(value, (bool, str, bytes)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def measured_run_losses(result: Mapping[str, Any]) -> tuple[float | None, float | None]:
    """The ``(initial_loss, final_loss)`` pair a finished run measured, as the trainer left it.

    :func:`summarize_training_loss` reports ``0.0`` / ``0.0`` with
    ``loss_summary_kind == "unavailable"`` for a run that observed no loss. That
    placeholder keeps the trainers' numeric result fields stable, but it is not a
    measurement, so this gives ``(None, None)`` for it. A genuine measured zero (any other
    kind, or no kind) stays ``0.0``; a non-finite tail (NaN or an infinity) is returned
    as it was measured, because the completion panel shows a diverged run as one; a
    ``"mean"`` or ``"single"`` summary keeps its number as before (#908). A result that
    has no numeric loss at all (a missing key, ``None``, a bool or a string) gives
    ``None`` for that end instead of raising ``KeyError`` after the adapter is saved.

    This is what the panel formats; what the database row keeps is
    :func:`stored_run_losses`.
    """
    if result.get("loss_summary_kind") == "unavailable":
        return None, None
    return _stored_number(result.get("initial_loss")), _stored_number(result.get("final_loss"))


def stored_run_losses(result: Mapping[str, Any]) -> tuple[float | None, float | None]:
    """The ``(initial_loss, final_loss)`` pair the ``runs`` row keeps for a finished run.

    :func:`measured_run_losses` made storable: a database row of ``0.0`` -> ``0.0``
    reads as a run that reached a perfect loss, so a run that observed no loss stores
    NULL for both. The pair is stored whole or not at all: ``runs.loss_summary_kind``
    says ``unavailable`` only when both columns are NULL and ``per_step`` / ``mean``
    only when both hold a number, so a result with one usable end, or an end that is
    NaN, stores NULL for both. SQLite turns a NaN into NULL by itself, which left a
    NULL ``final_loss`` beside ``per_step``; the NaN is refused here instead, and
    :func:`stored_loss_summary_kind` follows. An infinity is kept: SQLite holds it, and
    the readers already show a diverged run (#2750).
    """
    initial, final = measured_run_losses(result)
    if initial is None or final is None or math.isnan(initial) or math.isnan(final):
        return None, None
    return initial, final


def stored_loss_summary_kind(result: Mapping[str, Any]) -> str | None:
    """The ``runs.loss_summary_kind`` value for a finished run: what its two losses are.

    The trainers' ``delta`` and ``single`` kinds are both ``per_step`` (measured
    per-step losses); ``mean`` and ``unavailable`` keep their names. A result with no
    numeric loss at all is ``unavailable`` whatever its kind claims. A result whose
    trainer named no (or an unknown) kind but does carry numbers is not guessed at:
    None, the value of a run finished before the column existed.
    """
    stored_initial, stored_final = stored_run_losses(result)
    if stored_initial is None and stored_final is None:
        return LOSS_SUMMARY_UNAVAILABLE
    kind = result.get("loss_summary_kind")
    return _STORED_KIND.get(kind) if isinstance(kind, str) else None
