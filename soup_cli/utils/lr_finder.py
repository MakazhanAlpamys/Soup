"""LR Range Finder (v0.32.0 Part A) — fast.ai-style sweep.

Pure helpers — runs no actual training. The driver in ``commands/train.py``
plugs them into a short HF Trainer loop and writes a JSON report.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, Sequence, TypedDict

from soup_cli.utils.paths import is_under_cwd


class LRFinderResult(TypedDict):
    """Structured result from ``find_optimal_lr``."""

    recommended_lr: float
    min_loss_lr: float
    diverged_at: Optional[float]
    smoothed_losses: list[float]

# Bounds prevent runaway sweeps and silly inputs.
MAX_NUM_STEPS = 10_000
MIN_NUM_STEPS = 2
DIVERGENCE_FACTOR = 4.0
SMOOTHING_BETA = 0.98


def compute_lr_schedule(
    start_lr: float, end_lr: float, num_steps: int,
) -> list[float]:
    """Geometric (log-linear) LR sweep from ``start_lr`` to ``end_lr``."""
    if not (start_lr > 0 and math.isfinite(start_lr)):
        raise ValueError(f"start_lr must be positive finite, got {start_lr}")
    if not (end_lr > 0 and math.isfinite(end_lr)):
        raise ValueError(f"end_lr must be positive finite, got {end_lr}")
    if end_lr <= start_lr:
        raise ValueError(f"end_lr ({end_lr}) must be > start_lr ({start_lr})")
    if num_steps < MIN_NUM_STEPS or num_steps > MAX_NUM_STEPS:
        raise ValueError(
            f"num_steps must be in [{MIN_NUM_STEPS}, {MAX_NUM_STEPS}], got {num_steps}"
        )
    log_start = math.log(start_lr)
    log_end = math.log(end_lr)
    step = (log_end - log_start) / (num_steps - 1)
    return [math.exp(log_start + i * step) for i in range(num_steps)]


def _smooth(losses: Sequence[float], beta: float = SMOOTHING_BETA) -> list[float]:
    """Exponential moving average with bias correction (Smith 2017)."""
    smoothed: list[float] = []
    avg = 0.0
    for index, loss in enumerate(losses, start=1):
        avg = beta * avg + (1 - beta) * loss
        smoothed.append(avg / (1 - beta ** index))
    return smoothed


def find_optimal_lr(
    lrs: Sequence[float], losses: Sequence[float],
) -> LRFinderResult:
    """Pick the LR with the steepest negative gradient before divergence.

    Edge case: when the smoothed loss is monotonically increasing from the
    start (``min_idx <= 1``), there is no meaningful descent region. The
    function returns the lowest LR (``lrs[0]``) as ``recommended_lr`` so
    callers get a deterministic fallback rather than ``None``.
    """
    if len(lrs) != len(losses):
        raise ValueError(
            f"lrs and losses must have equal length (got {len(lrs)} vs {len(losses)})"
        )
    if len(lrs) < 4:
        raise ValueError(f"Need at least 4 (lr, loss) pairs, got {len(lrs)}")

    smoothed = _smooth(losses)

    # Find min smoothed loss.
    min_idx = min(range(len(smoothed)), key=lambda index: smoothed[index])
    min_loss_lr = lrs[min_idx]

    # Detect divergence: first index after min where |loss| > DIVERGENCE_FACTOR * |min|.
    # ``abs`` keeps the check correct if a custom log-prob style loss goes negative.
    diverged_at: Optional[float] = None
    threshold = abs(smoothed[min_idx]) * DIVERGENCE_FACTOR
    for index in range(min_idx + 1, len(smoothed)):
        if abs(smoothed[index]) > threshold:
            diverged_at = lrs[index]
            break

    # Compute steepest negative gradient (in log-LR space) up to min_idx.
    upper = max(min_idx, 1)
    best_grad = 0.0
    best_idx = 0
    for index in range(1, upper + 1):
        d_lr = math.log(lrs[index]) - math.log(lrs[index - 1])
        d_loss = smoothed[index] - smoothed[index - 1]
        if d_lr > 0:
            grad = d_loss / d_lr
            if grad < best_grad:
                best_grad = grad
                best_idx = index

    # Step back one — recommend LR slightly before the steepest descent end.
    rec_idx = max(0, best_idx - 1) if best_grad < 0 else 0
    recommended_lr = lrs[rec_idx]

    return {
        "recommended_lr": recommended_lr,
        "min_loss_lr": min_loss_lr,
        "diverged_at": diverged_at,
        "smoothed_losses": smoothed,
    }


def _finite_or_reject(values: Sequence[float], label: str) -> list[float]:
    """Reject NaN / Infinity floats so the JSON report is parser-safe."""
    cleaned: list[float] = []
    for value in values:
        as_float = float(value)
        if not math.isfinite(as_float):
            raise ValueError(
                f"{label} contains non-finite value ({value!r}); "
                "NaN / Infinity are rejected to keep the JSON report valid."
            )
        cleaned.append(as_float)
    return cleaned


def run_lr_sweep(
    *, model, dataloader, schedule, optimizer_factory, device: str = "cpu",
) -> list[float]:
    """Run an in-process LR-sweep training loop (#56, v0.33.0).

    For each LR in ``schedule``, pulls the next batch from ``dataloader``,
    runs a forward + backward + optimizer step with that LR, records the
    loss. Diverged batches (NaN/Inf loss) terminate the sweep early so the
    report's ``diverged_at`` is honest.

    Args:
        model: a torch ``nn.Module`` returning a dict with ``loss`` field
            (HF causal-LM contract).
        dataloader: any iterable producing kwargs dicts for ``model(**batch)``.
        schedule: LR sweep from :func:`compute_lr_schedule`.
        optimizer_factory: callable ``(params) -> Optimizer`` so we can
            instantiate without depending on a specific optimizer here.
        device: ``"cpu"`` / ``"cuda"`` / ``"mps"``.

    Returns:
        list of per-step losses, length <= ``len(schedule)``.

    Raises:
        ValueError: if the schedule is empty.

    Notes:
        - We mutate ``param_group["lr"]`` per step (standard LR-finder
          pattern, no scheduler interference).
        - Loss is captured as a Python float to break the autograd graph.
        - The loop is bounded by the schedule length and the dataloader
          length — whichever is shorter.
    """
    if not schedule:
        raise ValueError("schedule must be non-empty")

    optimizer = optimizer_factory(model.parameters())
    losses: list[float] = []

    iterator = iter(dataloader)
    for lr in schedule:
        try:
            batch = next(iterator)
        except StopIteration:
            break
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Move tensor batch values onto the right device when possible.
        # Stays import-free here; ``v.to(device)`` is duck-typed against any
        # tensor-like object so we don't need a hard torch dependency.
        if isinstance(batch, dict):
            batch = {
                k: (v.to(device) if hasattr(v, "to") else v)
                for k, v in batch.items()
            }

        optimizer.zero_grad(set_to_none=True)
        out = model(**batch) if isinstance(batch, dict) else model(batch)
        loss = out["loss"] if isinstance(out, dict) else out.loss
        loss_value = float(loss.detach().item()) if hasattr(loss, "detach") else float(loss)
        if not math.isfinite(loss_value):
            break
        losses.append(loss_value)
        loss.backward()
        optimizer.step()
    return losses


def save_lr_finder_report(
    lrs: Sequence[float], losses: Sequence[float], output_path: Path | str,
) -> None:
    """Write a JSON report with the sweep + recommended LR."""
    output = Path(output_path)
    if not is_under_cwd(output):
        raise ValueError(f"Report path must stay under cwd: {output}")

    report_lrs = _finite_or_reject(lrs, "lrs")
    report_losses = _finite_or_reject(losses, "losses")
    summary = find_optimal_lr(report_lrs, report_losses)
    payload = {
        "lrs": report_lrs,
        "losses": report_losses,
        "smoothed_losses": summary["smoothed_losses"],
        "recommended_lr": summary["recommended_lr"],
        "min_loss_lr": summary["min_loss_lr"],
        "diverged_at": summary["diverged_at"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # ``allow_nan=False`` is belt-and-braces: report_* are already finite,
    # but ``smoothed_losses`` could carry a non-finite if the input loss
    # somehow drifted. Reject rather than emit ``NaN`` (invalid JSON).
    output.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
