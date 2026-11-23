"""Duration estimates from comparable SFT runs in the local tracker."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import median

MIN_MATCHES = 3


@dataclass(frozen=True)
class RunSettings:
    model: str
    backend: str
    quantization: str
    batch_size: int
    accumulation: int
    max_length: int

    def planned_steps(self, rows: int, epochs: int) -> int:
        rows_per_step = self.batch_size * self.accumulation
        return ((rows + rows_per_step - 1) // rows_per_step) * epochs


def cost_history_eligible(
    *,
    task: str,
    resume_from: str | None,
    device: str,
    gpu_count: int | None,
    distributed: bool,
    fsdp: str | None,
    ds_config_path: str | None,
) -> bool:
    """Record launch conditions that permit an elapsed-time comparison."""
    return (
        task == "sft"
        and resume_from is None
        and device == "cuda"
        and gpu_count == 1
        and not distributed
        and not fsdp
        and ds_config_path is None
    )


def fixed_sft_settings(config: dict) -> RunSettings | None:
    """Identify configurations where a row count can estimate optimizer steps."""
    data = config.get("data")
    training = config.get("training")
    if not isinstance(data, dict) or not isinstance(training, dict):
        return None
    backend = config.get("backend")
    if (
        config.get("task") != "sft"
        or backend not in ("transformers", "unsloth")
        or config.get("modality") != "text"
        or any(data.get(flag) for flag in ("streaming", "interleave", "replay"))
        or any(
            training.get(flag)
            for flag in ("packing", "multipack", "grad_accum_auto_tune", "stream_layers")
        )
    ):
        return None
    model = config.get("base")
    quantization = training.get("quantization")
    if not isinstance(model, str) or not isinstance(quantization, str):
        return None
    batch = training.get("batch_size")
    accumulation = training.get("gradient_accumulation_steps")
    length = data.get("max_length")
    if any(type(value) is not int or value < 1 for value in (batch, accumulation, length)):
        return None
    return RunSettings(model, backend, quantization, batch, accumulation, length)


def matching_step_times(
    runs: Iterable[dict], settings: RunSettings, gpu_label: str
) -> list[float]:
    """Return elapsed seconds per optimizer step for completed, verified runs."""
    times = []
    for run in runs:
        if (
            run.get("status") != "completed"
            or run.get("run_kind") != "train"
            or run.get("device") != "cuda"
            or run.get("cost_history_eligible") != 1
            or run.get("cost_gpu_label") != gpu_label
        ):
            continue
        steps = run.get("total_steps")
        duration = run.get("duration_secs")
        if (
            type(steps) is not int or steps < 1
            or type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0
        ):
            continue
        try:
            config = json.loads(run["config_json"])
        except (KeyError, TypeError, ValueError):
            continue
        if isinstance(config, dict) and fixed_sft_settings(config) == settings:
            times.append(duration / steps)
    return times


def confidence_for(count: int, relative_spread: float) -> str:
    """Describe agreement among measured runs with a fixed heuristic."""
    if count >= 8 and relative_spread <= 0.20:
        return "high"
    if count >= 5 and relative_spread <= 0.50:
        return "medium"
    return "low"


def estimate_from_step_times(times: list[float], planned_steps: int) -> dict | None:
    """Scale the observed range and median to the new optimizer step count."""
    count = len(times)
    if count < MIN_MATCHES or planned_steps < 1:
        return None
    middle = median(times)
    lower, upper = min(times), max(times)
    spread = (upper - lower) / middle
    return {
        "matches": count,
        "optimizer_steps": planned_steps,
        "median_seconds_per_step": middle,
        "min_seconds_per_step": lower,
        "max_seconds_per_step": upper,
        "relative_spread": spread,
        "confidence": confidence_for(count, spread),
        "min_duration_hrs": lower * planned_steps / 3600,
        "max_duration_hrs": upper * planned_steps / 3600,
    }
