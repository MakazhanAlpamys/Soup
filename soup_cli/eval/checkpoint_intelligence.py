"""Checkpoint intelligence — pick best checkpoint by quality, not loss (Part G).

HF Trainer's "best_model" is determined by loss, but lower loss does not
always correlate with better real-world quality. This module runs a quality
metric during training and tracks which checkpoint truly performs best —
plus prunes lower-quality checkpoints to save disk.
"""

from __future__ import annotations

import os
import shutil
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _abort_on_symlink(_func, path, exc_info):
    """``shutil.rmtree`` onerror callback: re-raise to abort recursive walk
    if a symlink (or any error condition) is encountered mid-walk.

    Defence-in-depth: ``shutil.rmtree`` already does not follow symlinks by
    default (it removes the link itself), but if a future Python version or a
    crafted directory structure changes that, aborting here keeps the
    invariant that prune never traverses outside the checkpoint subtree.
    """
    # Prefer lstat to avoid following the symlink during inspection.
    try:
        if stat.S_ISLNK(os.lstat(path).st_mode):
            raise OSError(
                f"prune_checkpoints aborted: symlink encountered mid-walk: {path}"
            )
    except OSError:
        # Re-raise the original exc_info so the caller sees the real error.
        raise
    # Re-raise the original failure if it wasn't a symlink hazard.
    exc_type, exc_val, _exc_tb = exc_info
    if exc_val is not None:
        raise exc_val
    raise OSError(f"prune_checkpoints failed: {path}")

# Weighting for the composite metric
COMPOSITE_WEIGHTS = {"judge": 0.5, "mmlu": 0.3, "custom": 0.2}


def compute_composite(
    judge: Optional[float] = None,
    mmlu: Optional[float] = None,
    custom: Optional[float] = None,
) -> float:
    """Weighted average of available quality metrics (ignores None)."""
    total_weight = 0.0
    total_score = 0.0
    for name, value in (("judge", judge), ("mmlu", mmlu), ("custom", custom)):
        if value is None:
            continue
        weight = COMPOSITE_WEIGHTS[name]
        total_score += value * weight
        total_weight += weight
    if total_weight == 0:
        return 0.0
    return total_score / total_weight


@dataclass
class CheckpointEval:
    """One checkpoint quality evaluation."""

    step: int
    score: float
    metric: str = "composite"
    is_best: bool = False


@dataclass
class CheckpointTracker:
    """Tracks checkpoint quality evaluations and determines the best one."""

    metric: str = "composite"
    keep_top: int = 3
    patience: int = 2
    history: list[CheckpointEval] = field(default_factory=list)

    @property
    def best(self) -> Optional[CheckpointEval]:
        if not self.history:
            return None
        return max(self.history, key=lambda e: e.score)

    def record(self, step: int, score: float) -> CheckpointEval:
        """Record a new checkpoint eval."""
        evaluation = CheckpointEval(step=step, score=score, metric=self.metric)
        self.history.append(evaluation)
        best = self.best
        if best is not None and best.step == step:
            for ev in self.history:
                ev.is_best = ev.step == step
        return evaluation

    def should_early_stop(self) -> bool:
        """Return True if quality regressed for ``patience`` consecutive evals."""
        if len(self.history) <= self.patience:
            return False
        window = self.history[-(self.patience + 1):]
        for i in range(1, len(window)):
            if window[i].score >= window[i - 1].score:
                return False
        return True

    def top_n_steps(self) -> list[int]:
        """Return step numbers of the top-N checkpoints by score."""
        sorted_hist = sorted(self.history, key=lambda e: e.score, reverse=True)
        return [e.step for e in sorted_hist[: self.keep_top]]

    def prune_checkpoints(self, output_dir: Path) -> list[int]:
        """Delete checkpoint-{step} directories not in the top-N.

        Only removes directories whose resolved path is strictly inside
        ``output_dir`` and whose name matches ``checkpoint-<int>``. Never
        follows symlinks outside the output dir.
        """
        output_dir = Path(output_dir).resolve()
        if not output_dir.exists():
            return []

        keep = set(self.top_n_steps())
        removed: list[int] = []

        for child in output_dir.iterdir():
            # TOCTOU-safe symlink check via os.lstat (does not follow links).
            try:
                child_stat = os.lstat(str(child))
            except OSError:
                continue
            if stat.S_ISLNK(child_stat.st_mode):
                continue
            if not stat.S_ISDIR(child_stat.st_mode):
                continue
            name = child.name
            if not name.startswith("checkpoint-"):
                continue
            try:
                step = int(name.split("-", 1)[1])
            except (ValueError, IndexError):
                continue
            if step in keep:
                continue
            # Safety: double-check path stays inside output_dir.
            try:
                child.resolve().relative_to(output_dir)
            except ValueError:
                continue
            try:
                shutil.rmtree(child, onerror=_abort_on_symlink)
            except OSError:
                # Symlink encountered mid-walk OR permission error — skip and
                # continue with other checkpoints rather than aborting the
                # whole prune pass.
                continue
            removed.append(step)

        return removed
