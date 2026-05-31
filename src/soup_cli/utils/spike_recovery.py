"""Loss spike auto-recovery strategy (v0.32.0 Part E).

When the watchdog fires:
1. If ``attempts < max_attempts`` and a checkpoint exists: rollback,
   decay the LR, resume.
2. Otherwise: stop training (existing watchdog behaviour).

This module provides the policy. Live trainer-state mutation
(rollback + LR change + resume) is wired in via the existing
``SoupTrainerCallback`` in v0.32.0; the auto-rollback to checkpoint is
advisory in v0.32.0 (issues warning + recommends manual resume), with
full live wiring tracked for v0.32.1 — the same pattern used by
v0.30.0 ``--auto-quant`` and v0.30.0 structured-output flags.
"""

from __future__ import annotations

from dataclasses import dataclass

MIN_LR_FLOOR = 1e-9
MAX_ATTEMPTS_CAP = 10


@dataclass(frozen=True)
class SpikeRecoveryStrategy:
    """Policy for loss-spike recovery decisions."""

    max_attempts: int = 3
    lr_decay: float = 0.5
    min_lr: float = MIN_LR_FLOOR

    def __post_init__(self) -> None:
        if not (1 <= self.max_attempts <= MAX_ATTEMPTS_CAP):
            raise ValueError(
                f"max_attempts must be in [1, {MAX_ATTEMPTS_CAP}], "
                f"got {self.max_attempts}"
            )
        if not (0 < self.lr_decay < 1):
            raise ValueError(
                f"lr_decay must be in (0, 1), got {self.lr_decay}"
            )
        if self.min_lr <= 0:
            raise ValueError(f"min_lr must be > 0, got {self.min_lr}")

    def should_recover(self, attempts: int) -> bool:
        """True when the recovery budget hasn't been exhausted."""
        return attempts < self.max_attempts

    def compute_new_lr(self, current_lr: float) -> float:
        """Decay the LR by ``lr_decay``, floored at ``min_lr``."""
        if current_lr <= 0:
            raise ValueError(f"current_lr must be > 0, got {current_lr}")
        new = current_lr * self.lr_decay
        return max(new, self.min_lr)
