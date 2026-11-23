"""Optimizer steps per second, measured the same way for the panel and the record."""

import math
from typing import Any, Optional


def is_finite_number(value: Any) -> bool:
    """Accept measured numbers, excluding booleans and unrepresentable integers."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


class StepRate:
    """The run average since the first logged step: ``Δstep / Δtime``.

    Hugging Face sets ``train_steps_per_second`` only in the final summary log,
    so a per-step rate has to be measured here (#2791). The live panel and the
    trainer callback each keep one and feed it the same steps, so the speed on
    screen is the speed in the record.
    """

    def __init__(self) -> None:
        self._first: Optional[tuple[float, float]] = None

    def reset(self) -> None:
        """Forget the first step; the next update starts measuring again."""
        self._first = None

    def update(self, step: Any, now: float) -> Optional[float]:
        """Return the rate at ``step`` and wall-clock ``now``, or None when unknown.

        None on the first step (there is no interval yet), while neither the
        step nor the clock has moved, and for a step that is not a finite
        non-negative number (it is ignored and the first step is kept). A step
        below the first one means the counter restarted, so measuring starts
        again from it.
        """
        if not is_finite_number(step) or step < 0:
            return None
        if self._first is None or step < self._first[0]:
            self._first = (step, now)
            return None
        first_step, first_time = self._first
        if step > first_step and now > first_time:
            return (step - first_step) / (now - first_time)
        return None
