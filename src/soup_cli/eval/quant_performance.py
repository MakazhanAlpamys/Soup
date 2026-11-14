"""Opt-in generation measurements for quant-check; quality alone gates exit status."""

from __future__ import annotations

import gc
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ModelPerformance:
    device: str
    dtype: str
    latency_ms: float
    tokens_per_second: float
    generated_tokens: int
    peak_gpu_memory_bytes: int | None
    repeat_latency_ms: list[float]
    latency_samples_ms: list[list[float]]
    generated_token_samples: list[list[int]]


@dataclass(frozen=True)
class PerformanceComparison:
    before: ModelPerformance
    after: ModelPerformance
    latency_change_pct: float | None
    tokens_per_second_change_pct: float | None
    peak_gpu_memory_change_pct: float | None
    latency_ranges_overlap: bool | None
    warmup: int
    repeats: int
    max_new_tokens: int = 256
    timing_scope: str = (
        "synchronized prompt generation including tokenize/copy/decode; excludes load"
    )
    quality_samples: str = "first measured answer per task; warmups and other repeats are unscored"
    memory_scope: str = "CUDA allocated peak after warmup, resident weights included; CPU/MPS null"


def validate_budgets(warmup: int, repeats: int) -> None:
    for name, value, minimum in (("warmup", warmup, 0), ("repeats", repeats, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"performance {name} must be an integer >= {minimum}")


def resolve_performance_device(device: str) -> str:
    """Pin a single concrete device; never let auto placement split the comparison."""
    from soup_cli.utils.live_eval import resolve_device

    normalized = resolve_device(device).strip().lower()
    if not re.fullmatch(r"cpu|cuda(?::\d+)?|mps", normalized):
        raise ValueError("--device must be auto, cpu, cuda, cuda:<index> or mps")
    if normalized == "cpu":
        return normalized
    import torch

    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError(f"Requested --device {normalized} but CUDA is unavailable")
        index = int(normalized.split(":")[1]) if ":" in normalized else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise ValueError(f"Requested CUDA index {index} is unavailable")
        return f"cuda:{index}"
    if not torch.backends.mps.is_available():
        raise ValueError("Requested --device mps but MPS is unavailable")
    return normalized


class _MemoryProbe:
    def __init__(self, device: str) -> None:
        self.device = device

    def synchronize(self) -> None:
        if self.device.startswith("cuda"):
            import torch

            torch.cuda.synchronize(self.device)
        elif self.device == "mps":
            import torch

            torch.mps.synchronize()

    def reset(self) -> None:
        if self.device.startswith("cuda"):
            import torch

            torch.cuda.reset_peak_memory_stats(self.device)

    def peak(self) -> int | None:
        if self.device.startswith("cuda"):
            import torch

            return int(torch.cuda.max_memory_allocated(self.device))
        return None

    def release(self) -> None:
        gc.collect()
        if self.device.startswith("cuda"):
            import torch

            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()


def measure_model(
    factory: Callable[[], Any], tasks: list, *, device: str, warmup: int, repeats: int,
) -> tuple[float, ModelPerformance]:
    """One model lifetime, including cleanup if loading/generation/scoring fails."""
    from soup_cli.eval.custom import score_task

    probe = _MemoryProbe(device)
    generator = None
    try:
        generator = factory()
        for _ in range(warmup):
            generator(tasks[0].prompt)
        probe.synchronize()
        probe.reset()
        latencies: list[list[float]] = []
        tokens: list[list[int]] = []
        score = 0.0
        for task in tasks:
            task_times, task_tokens = [], []
            for repeat in range(repeats):
                probe.synchronize()
                started = time.perf_counter()
                output = generator(task.prompt)
                probe.synchronize()
                elapsed = time.perf_counter() - started
                count = getattr(generator, "last_generated_tokens", None)
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise ValueError("Performance requires actual generated token-ID counts")
                if not math.isfinite(elapsed) or elapsed <= 0:
                    raise ValueError("Generation timer must produce a finite positive duration")
                task_times.append(elapsed * 1000)
                task_tokens.append(count)
                if repeat == 0:
                    score += float(score_task(task, output).score)
            latencies.append(task_times)
            tokens.append(task_tokens)
        total_ms = sum(sum(row) for row in latencies)
        total_tokens = sum(sum(row) for row in tokens)
        metrics = ModelPerformance(
            device=str(generator.device), dtype=str(generator.dtype),
            latency_ms=total_ms / (len(tasks) * repeats),
            tokens_per_second=total_tokens * 1000 / total_ms,
            generated_tokens=total_tokens, peak_gpu_memory_bytes=probe.peak(),
            repeat_latency_ms=[sum(row[i] for row in latencies) / len(tasks)
                               for i in range(repeats)],
            latency_samples_ms=latencies, generated_token_samples=tokens,
        )
        return score / len(tasks), metrics
    finally:
        try:
            if generator is not None:
                generator.close()
        finally:
            generator = None
            probe.release()


def percent_change(before: float | None, after: float | None) -> float | None:
    if before is None or after is None or before == 0:
        return None
    return (after / before - 1) * 100


def compare_performance(
    before: ModelPerformance, after: ModelPerformance, *, warmup: int, repeats: int,
) -> PerformanceComparison:
    overlap = None
    if repeats > 1:
        # A descriptive observed range, not a confidence interval/significance test.
        overlap = max(min(before.repeat_latency_ms), min(after.repeat_latency_ms)) <= min(
            max(before.repeat_latency_ms), max(after.repeat_latency_ms)
        )
    return PerformanceComparison(
        before=before, after=after,
        latency_change_pct=percent_change(before.latency_ms, after.latency_ms),
        tokens_per_second_change_pct=percent_change(
            before.tokens_per_second, after.tokens_per_second,
        ),
        peak_gpu_memory_change_pct=percent_change(
            before.peak_gpu_memory_bytes, after.peak_gpu_memory_bytes,
        ),
        latency_ranges_overlap=overlap, warmup=warmup, repeats=repeats,
    )


def performance_rows(performance: PerformanceComparison) -> list[tuple[str, str, str, str]]:
    rows = []
    for label, field, change in (
        ("Mean prompt latency (ms)", "latency_ms", performance.latency_change_pct),
        ("Generated tokens/s", "tokens_per_second", performance.tokens_per_second_change_pct),
        ("CUDA allocated peak (MiB)", "peak_gpu_memory_bytes",
         performance.peak_gpu_memory_change_pct),
    ):
        values = [getattr(side, field) for side in (performance.before, performance.after)]
        if field == "peak_gpu_memory_bytes":
            values = [None if value is None else value / (1024 ** 2) for value in values]
        rows.append((label, *("N/A" if value is None else f"{value:.3f}" for value in values),
                     "N/A" if change is None else f"{change:+.1f}%"))
    return rows


def performance_note(performance: PerformanceComparison) -> str:
    a, b = performance.before, performance.after
    if performance.latency_ranges_overlap is None:
        noise = "one repeat: noise cannot be estimated"
    elif performance.latency_ranges_overlap:
        noise = "observed repeat latency ranges overlap; speed difference may be noise"
    else:
        noise = "observed repeat latency ranges do not overlap (not a significance test)"
    ranges = "; ".join(
        f"{name} {min(side.repeat_latency_ms):.3f}..{max(side.repeat_latency_ms):.3f} ms"
        for name, side in (("before", a), ("after", b))
    )
    return (
        f"Informational performance; {a.device}/{a.dtype} -> {b.device}/{b.dtype}; "
        f"warmup={performance.warmup}, repeats={performance.repeats}, "
        f"max_new_tokens={performance.max_new_tokens}; {ranges}; {noise}. "
        "Quality alone determines the exit code."
    )
