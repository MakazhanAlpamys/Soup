"""Live gradient accumulation auto-tuning (v0.32.0 Part B).

Pure helpers — no torch / GPU touching at module load. The HF Trainer
callback wiring is intentionally left to v0.32.1 because mid-run mutation of
``gradient_accumulation_steps`` requires DataLoader rebuild which is not
safe to do inside ``on_step_end`` without changing TRL internals.

For v0.32.0 the monitor is wired as an *advisory*:
- It records peak memory each step and flags if pressure is above
  ``threshold`` (default 0.92 of total VRAM).
- It returns a recommended ``(batch, accum)`` pair the user can apply on
  the next run (also surfaced by the autopilot extension).
"""

from __future__ import annotations

from dataclasses import dataclass, field

MIN_THRESHOLD = 0.05
MAX_THRESHOLD = 0.99
MAX_ACCUM = 1024  # HF Trainer accepts higher but DataLoader prefetch degrades


@dataclass
class GradAccumMonitor:
    """Tracks VRAM pressure and recommends batch / accum adjustments."""

    total_vram_gb: float
    threshold: float = 0.92
    peak_used_gb: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if not (self.total_vram_gb > 0):
            raise ValueError(
                f"total_vram_gb must be > 0, got {self.total_vram_gb}"
            )
        if not (MIN_THRESHOLD < self.threshold < MAX_THRESHOLD):
            raise ValueError(
                f"threshold must be in ({MIN_THRESHOLD}, {MAX_THRESHOLD}), "
                f"got {self.threshold}"
            )

    def observe(self, used_vram_gb: float) -> None:
        """Record a memory observation; updates running peak."""
        if used_vram_gb < 0:
            raise ValueError(f"used_vram_gb must be >= 0, got {used_vram_gb}")
        if used_vram_gb > self.peak_used_gb:
            self.peak_used_gb = used_vram_gb

    def should_adjust(self, used_vram_gb: float) -> bool:
        """True when used VRAM crosses the pressure threshold."""
        if used_vram_gb < 0:
            raise ValueError(f"used_vram_gb must be >= 0, got {used_vram_gb}")
        return (used_vram_gb / self.total_vram_gb) >= self.threshold

    def recommend(
        self, current_batch: int, current_accum: int,
    ) -> tuple[int, int]:
        """Halve batch and double accum, preserving effective batch.

        Floors batch at 1 — if batch is already 1, accum is left untouched.
        Caps the new accum at ``MAX_ACCUM`` (1024); past that DataLoader
        prefetch degrades and the user is better off cutting ``max_length``
        or moving to gradient checkpointing.
        """
        if current_batch < 1 or current_accum < 1:
            raise ValueError(
                f"current_batch and current_accum must be >= 1, "
                f"got ({current_batch}, {current_accum})"
            )
        if current_batch == 1:
            return current_batch, current_accum
        new_accum = min(current_accum * 2, MAX_ACCUM)
        return current_batch // 2, new_accum
