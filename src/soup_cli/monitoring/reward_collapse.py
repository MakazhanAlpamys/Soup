"""Observe TRL's measured fraction of groups with no reward spread."""

from dataclasses import dataclass
from numbers import Real
from typing import Optional


def reward_group_fraction(value: object) -> Optional[float]:
    """Accept a finite measured fraction, including a genuine zero."""
    if isinstance(value, Real) and not isinstance(value, bool) and 0 <= value <= 1:
        return float(value)
    return None


@dataclass
class RewardCollapseMonitor:
    """Warn once per run after five distinct high measurement events.

    The 50%/five-event rule is a diagnostic heuristic, not a quality gate.
    Missing metrics do not invent measurements; invalid ones break the streak.
    """

    fraction: Optional[float] = None
    _last_step: int = -1
    _streak: int = 0
    _warned: bool = False

    def observe(self, step: int, value: object) -> bool:
        """Return true once when the high-fraction streak reaches five."""
        if step == self._last_step:
            return False
        if step < self._last_step:
            self._streak = 0
        self._last_step = step
        self.fraction = reward_group_fraction(value)
        self._streak = self._streak + 1 if self.fraction is not None and self.fraction > .5 else 0
        if self._streak >= 5 and not self._warned:
            self._warned = True
            return True
        return False
