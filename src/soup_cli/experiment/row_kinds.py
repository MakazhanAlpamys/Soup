"""The closed value sets of two nullable ``kind`` columns of the experiment database.

Both columns are add-only: a database written before they existed gets them as NULL
(``ExperimentTracker`` adds them in place), and an old row stays NULL -- it is not
backfilled, because NULL says "written before this column existed". Code that writes a
row states its kind; the strings below are the contract, and are pinned by
``tests/test_nullable_loss_contract.py``.

``runs.loss_summary_kind`` -- what ``runs.initial_loss`` / ``runs.final_loss`` are:

* ``per_step``    measured per-step losses: the first and the last one logged (or the
                  one step, when the run logged a single step);
* ``mean``        a run too short to log a step: both columns hold the run-level
                  ``train_loss`` mean, which is not a per-step loss (#908);
* ``unavailable`` the run observed no loss; both columns are NULL.

``metrics.kind`` -- what a ``metrics`` row (and the live ``metric`` event that mirrors it)
is:

* ``train``       a training log: its ``loss`` / ``lr`` / ``grad_norm`` were measured by
                  that log;
* ``eval``        an evaluation: it carries a validation number (``val_loss`` or
                  ``val_reward``), and ``loss`` / ``lr`` are the last measured training
                  values carried into it (tests/test_issue23_mlx_val_loss_producer.py),
                  not a measurement of the evaluation;
* ``summary``     a run-level summary: HF's final runtime event, which measures no
                  per-step loss. The callback stores its throughput
                  (``train_steps_per_second``) as ``speed`` and nothing else: ``loss``,
                  ``lr``, ``grad_norm`` and the validation columns are NULL, never the
                  last training values carried forward, and the ``train_loss`` mean is
                  not stored. A reader of the training curve skips these rows.
"""

from __future__ import annotations

from typing import Any, Optional

LOSS_SUMMARY_PER_STEP = "per_step"
LOSS_SUMMARY_MEAN = "mean"
LOSS_SUMMARY_UNAVAILABLE = "unavailable"
LOSS_SUMMARY_KINDS = (LOSS_SUMMARY_PER_STEP, LOSS_SUMMARY_MEAN, LOSS_SUMMARY_UNAVAILABLE)

ROW_KIND_TRAIN = "train"
ROW_KIND_EVAL = "eval"
ROW_KIND_SUMMARY = "summary"
METRIC_ROW_KINDS = (ROW_KIND_TRAIN, ROW_KIND_EVAL, ROW_KIND_SUMMARY)


def _check(value: Any, allowed: tuple[str, ...], name: str) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str) and value in allowed:
        return value
    raise ValueError(f"{name} must be one of {', '.join(allowed)} (or None); got {value!r}")


def check_loss_summary_kind(value: Any) -> Optional[str]:
    """``value`` if it is None or one of :data:`LOSS_SUMMARY_KINDS`, else ``ValueError``."""
    return _check(value, LOSS_SUMMARY_KINDS, "loss_summary_kind")


def check_metric_row_kind(value: Any) -> Optional[str]:
    """``value`` if it is None or one of :data:`METRIC_ROW_KINDS`, else ``ValueError``."""
    return _check(value, METRIC_ROW_KINDS, "kind")


def metric_row_kind(*, val_loss: Optional[float] = None, val_reward: Optional[float] = None) -> str:
    """``eval`` for a row that carries a validation number, ``train`` for any other.

    A measured ``0.0`` is a validation number (``is not None``, not truthiness).
    """
    if val_loss is not None or val_reward is not None:
        return ROW_KIND_EVAL
    return ROW_KIND_TRAIN
