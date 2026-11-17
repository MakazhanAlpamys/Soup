"""Which trainer log keys are a run's training reward, and what the tracker stores for them.

Two nullable, add-only columns of the ``metrics`` table hold the per-step reward of a
reinforcement-style run (``experiment/tracker.py``): ``metrics.reward``, the mean reward over
the logging window that ends at the step, and ``metrics.reward_std``, where the trainer
logs one (GRPO: the window mean of the per-batch standard deviations). NULL
means "not measured": every non-RL run, a row whose log carried no reward, a value that is
not a number (a ``bool``, a string) and a value that is not finite.

This module is the one place that decides which log entry is which column. The allowlist is
exact on purpose. A key that merely contains "reward" is another number: DPO's implicit
``rewards/chosen`` / ``rewards/margins``, the KL-shaped ``objective/rlhf_reward``, a
per-function ``rewards/<fn>/mean``, and an evaluation's ``eval_reward`` (its own column,
``val_reward``) are all left alone. The keys below were read from the installed TRL
(0.29.1) source; ``tests/test_metrics_reward_column.py`` pins them:

* ``reward`` -> ``reward``: ``GRPOTrainer`` (``grpo_trainer.py``, the mean of the summed
  per-function rewards over the logging window);
* ``objective/scores`` -> ``reward``: the experimental ``PPOTrainer`` (the mean score of the
  reward model or function, before the KL penalty) and the experimental
  ``OnlineDPOTrainer`` when it scores completions with a reward model or function. It is
  absent when a pairwise judge ranks completions, because a ranking is not a score;
* ``reward_std`` -> ``reward_std``: ``GRPOTrainer`` (the standard deviation of the batch's
  summed rewards; NaN for a batch of one, which is stored as NULL).

When a log carries both ``reward`` and ``objective/scores``, the earlier key in
:data:`REWARD_LOG_KEYS` (``reward``) wins.
"""

from __future__ import annotations

import math
import numbers
from typing import Any, Mapping, Optional

#: ``(trainer log key, metrics column)`` in precedence order. Exact key names only.
REWARD_LOG_KEYS: tuple[tuple[str, str], ...] = (
    ("reward", "reward"),
    ("objective/scores", "reward"),
    ("reward_std", "reward_std"),
)

#: The columns :data:`REWARD_LOG_KEYS` can fill.
REWARD_COLUMNS: tuple[str, ...] = ("reward", "reward_std")


def _is_number(value: Any) -> bool:
    """A real number as logged: not a ``bool`` and not a string (``numbers.Real`` covers
    the numpy scalars a trainer may leave in its logs)."""
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def finite_reward(value: Any) -> Optional[float]:
    """``value`` as a float when it is a finite number, else None.

    A ``bool`` is not a number (SQLite would store ``True`` as 1.0), a numeric string is
    not a measurement, and NaN / an infinity cannot be plotted or sent as JSON; an integer
    beyond float range is no finite number either.
    """
    if not _is_number(value):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None


def reward_from_logs(logs: Mapping[str, Any]) -> dict[str, Optional[float]]:
    """The reward columns a trainer log measured, as ``{column: finite float or None}``.

    A column is present when an allowlisted key carried a number: its value is the number,
    or None when that number is not finite (the step happened, its reward is unusable).
    A key whose value is not a number at all measured nothing and leaves its column out,
    so an empty dict means the log carried no reward.
    """
    found: dict[str, Optional[float]] = {}
    for key, column in REWARD_LOG_KEYS:
        if column in found:
            continue
        value = logs.get(key)
        if _is_number(value):
            found[column] = finite_reward(value)
    return found
