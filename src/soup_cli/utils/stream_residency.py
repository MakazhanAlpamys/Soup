"""Setup-only planning for immutable streamed decoder weights in spare VRAM.

This is a capacity policy, not a claim that the largest resident set is the
fastest one. Its two forward/backward probes never perform an optimizer step.
They measure the configured shape before training and preserve training RNG,
adapter values, and module training flags. Optimizer state is estimated because
the Trainer has not created it yet.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Union

RESIDENCY_HEADROOM = 0.85
RESIDENCY_RESERVE_BYTES = 64 * 2**20
# Two fp32 Adam moments plus a conservative fp32 step workspace. Weights and
# gradients are already present in the measured forward/backward peak.
OPTIMIZER_RESERVE_BYTES_PER_PARAM = 12


@dataclass(frozen=True)
class ResidencyBudget:
    cap_bytes: int
    available_bytes: int
    bound: str


@dataclass(frozen=True)
class ResidencyPlan:
    mode: Union[int, str]
    indices: tuple[int, ...] = ()
    resident_bytes: int = 0
    baseline_peak_allocated_bytes: int = 0
    baseline_peak_reserved_bytes: int = 0
    candidate_peak_allocated_bytes: int = 0
    candidate_peak_reserved_bytes: int = 0
    cap_bytes: int = 0
    reserve_bytes: int = 0
    optimizer_reserve_bytes: int = 0
    setup_seconds: float = 0.0
    preload_seconds: float = 0.0
    reason: str = "disabled"

    @property
    def selected_indices(self) -> tuple[int, ...]:
        return self.indices

    @property
    def enabled(self) -> bool:
        return bool(self.indices)


def residency_budget(
    *,
    total_bytes: int,
    free_bytes: int,
    reserved_bytes: int,
    peak_reserved_bytes: int,
    reserve_bytes: int = RESIDENCY_RESERVE_BYTES,
    optimizer_reserve_bytes: int = 0,
    override_bytes: Optional[int] = None,
) -> ResidencyBudget:
    """A shared cap after existing allocations and explicit setup reserves."""
    numbers = (total_bytes, free_bytes, reserved_bytes, peak_reserved_bytes,
               reserve_bytes, optimizer_reserve_bytes)
    if any(type(value) is not int or value < 0 for value in numbers):
        raise ValueError("memory byte counts must be non-negative integers")
    if override_bytes is not None and (type(override_bytes) is not int or override_bytes < 1):
        raise ValueError("override_bytes must be a positive integer or None")
    caps = {
        "card": math.floor(RESIDENCY_HEADROOM * total_bytes),
        "free": free_bytes + reserved_bytes,
    }
    if override_bytes is not None:
        caps["override"] = override_bytes
    bound = min(caps, key=caps.get)
    cap = caps[bound]
    available = max(0, cap - peak_reserved_bytes - reserve_bytes - optimizer_reserve_bytes)
    return ResidencyBudget(cap, available, bound)


def centered_layer_indices(n_layers: int, count: int) -> tuple[int, ...]:
    """A deterministic block, matching the existing residency experiment."""
    if type(n_layers) is not int or type(count) is not int or not 0 <= count <= n_layers:
        raise ValueError("count must be an integer between zero and n_layers")
    start = (n_layers - count) // 2
    return tuple(range(start, start + count))


def plan_centered_layers(
    layer_bytes: Sequence[int],
    budget_bytes: int,
    *,
    peak_allocated_bytes: int = 0,
    peak_reserved_bytes: int = 0,
) -> tuple[int, ...]:
    """Largest nested centred block whose charged payload fits the budget."""
    if any(type(size) is not int or size <= 0 for size in layer_bytes):
        raise ValueError("each resident layer must occupy a positive integer number of bytes")
    if type(budget_bytes) is not int or budget_bytes < 0:
        raise ValueError("budget_bytes must be a non-negative integer")
    selected: tuple[int, ...] = ()
    for count in range(1, len(layer_bytes) + 1):
        indices = centered_layer_indices(len(layer_bytes), count)
        charge = sum(layer_bytes[idx] for idx in indices)
        if peak_reserved_bytes > peak_allocated_bytes > 0:
            charge = -(-charge * peak_reserved_bytes // peak_allocated_bytes)
        if charge > budget_bytes:
            break
        selected = indices
    return selected


class _CudaMemory:
    def __init__(self, device: Any):
        self.device = device

    def total(self) -> int:
        import torch

        return int(torch.cuda.get_device_properties(self.device).total_memory)

    def free(self) -> int:
        import torch

        return int(torch.cuda.mem_get_info(self.device)[0])

    def reserved(self) -> int:
        import torch

        return int(torch.cuda.memory_reserved(self.device))

    def synchronize(self) -> None:
        import torch

        torch.cuda.synchronize(self.device)

    def release(self) -> None:
        import torch

        torch.cuda.empty_cache()


def _measure_preserving_state(
    model: Any,
    measure: Callable[..., Any],
    *,
    rows: int,
    seq_len: int,
    vocab_size: int,
    device: Any,
) -> Any:
    import torch

    parameters = [param for param in model.parameters() if param.requires_grad]
    if any(param.grad is not None for param in model.parameters()):
        raise ValueError("stream residency planning requires pristine gradients before training")
    # Streamed wrappers deliberately hide themselves from named_modules();
    # walk the actual tree so their mode is preserved as well as their inner's.
    pending = [model]
    seen = set()
    states = []
    while pending:
        module = pending.pop()
        if id(module) in seen:
            continue
        seen.add(id(module))
        states.append((module, module.training))
        pending.extend(child for child in module._modules.values() if child is not None)
    # Host copies do not inflate the measured VRAM peak and do not create a
    # second GPU adapter allocation. No optimizer exists or is stepped here.
    values = [param.detach().to(device="cpu", copy=True) for param in parameters]
    where = torch.device(device)
    devices = []
    if where.type == "cuda":
        devices = [where.index if where.index is not None else torch.cuda.current_device()]
    try:
        with torch.random.fork_rng(devices=devices):
            return measure(model, rows=rows, seq_len=seq_len, vocab_size=vocab_size, device=device)
    finally:
        with torch.no_grad():
            for param, original in zip(parameters, values):
                param.copy_(original)
        for param in model.parameters():
            param.grad = None
        for module, training in states:
            module.training = training


def configure_stream_residency(
    model: Any,
    runtime: Any,
    mode: Union[int, str],
    *,
    rows: int,
    seq_len: int,
    vocab_size: int,
    override_bytes: Optional[int] = None,
    notify: Optional[Callable[[str], Any]] = None,
    memory: Any = None,
    measure: Optional[Callable[..., Any]] = None,
    reserve_bytes: int = RESIDENCY_RESERVE_BYTES,
    optimizer_reserve_bytes: Optional[int] = None,
) -> ResidencyPlan:
    """Install and measure an opt-in RAM-tier resident set before training.

    ``auto`` safely keeps the ordinary pool when capacity cannot be established.
    Explicit positive counts refuse an unsupported or non-fitting configuration.
    A non-OOM CUDA/probe error always propagates. ``memory`` and ``measure`` are
    injectable instruments; neither changes the arithmetic being trained.
    """
    if type(mode) is not int and mode != "auto":
        raise ValueError("stream_resident_layers must be a non-negative integer or 'auto'")
    if type(mode) is int and mode < 0:
        raise ValueError("stream_resident_layers must be non-negative")
    if mode == 0:
        return ResidencyPlan(mode=mode)

    started = time.perf_counter()
    state: dict[str, Any] = {"mode": mode, "reserve_bytes": reserve_bytes}

    def result(reason: str, **updates: Any) -> ResidencyPlan:
        plan = ResidencyPlan(**(state | updates | {
            "reason": reason, "setup_seconds": time.perf_counter() - started,
        }))
        if notify is not None:
            notify(
                f"training.stream_resident_layers={mode!r}: {reason}; "
                f"resident layers {list(plan.indices)}, payload {plan.resident_bytes} bytes, "
                f"cap {plan.cap_bytes} bytes, setup reserve "
                f"{plan.reserve_bytes + plan.optimizer_reserve_bytes} bytes "
                "(optimizer state estimated; setup forward/backward, no optimizer step)."
            )
        return plan

    def refuse(reason: str) -> ResidencyPlan:
        if mode != "auto":
            raise ValueError(f"training.stream_resident_layers={mode}: {reason}")
        return result(reason)

    pool = runtime.pool
    if getattr(runtime, "tier", None) != "ram":
        return refuse("ordinary streaming kept: residency currently supports the RAM tier only")
    if not getattr(pool, "is_cuda", False):
        return refuse("ordinary streaming kept: no CUDA pool to measure")
    if min(rows, seq_len, vocab_size) < 1:
        return refuse("ordinary streaming kept: no configured batch/context/vocabulary to measure")
    n_layers = runtime.n_layers
    if type(n_layers) is not int or n_layers < 1:
        return refuse("ordinary streaming kept: no decoder layers")
    if type(mode) is int and mode > n_layers:
        raise ValueError(f"requested {mode} resident layers but the decoder has {n_layers}")
    if getattr(pool, "resident_slots", {}):
        raise ValueError("stream residency is already configured")

    from soup_cli.utils.layer_stream_runtime import (
        _is_out_of_memory,
        clear_stream_substitution_cache,
        measure_step_peak_bytes,
    )

    memory = _CudaMemory(pool.device) if memory is None else memory
    measure = measure_step_peak_bytes if measure is None else measure
    if optimizer_reserve_bytes is None:
        optimizer_reserve_bytes = OPTIMIZER_RESERVE_BYTES_PER_PARAM * sum(
            int(param.numel()) for param in model.parameters() if param.requires_grad
        )
    state["optimizer_reserve_bytes"] = optimizer_reserve_bytes

    def probe() -> Any:
        peak = _measure_preserving_state(
            model, measure, rows=rows, seq_len=seq_len, vocab_size=vocab_size, device=pool.device,
        )
        if peak is not None and getattr(peak, "failed", False):
            raise RuntimeError(f"stream residency probe failed: {peak.error}")
        return peak

    def clear() -> None:
        # Wait before dropping private tensors: cached Params4bit and tensor
        # views otherwise keep the allocation alive after the pool shrinks.
        memory.synchronize()
        clear_stream_substitution_cache(model)
        pool.clear_resident_layers()
        memory.release()

    try:
        baseline = probe()
    except Exception as exc:
        if not _is_out_of_memory(exc):
            raise
        return refuse("ordinary streaming kept: baseline probe ran out of memory")
    if baseline is None:
        return refuse("ordinary streaming kept: baseline peak could not be measured")
    if getattr(baseline, "oom", False):
        return refuse("ordinary streaming kept: baseline probe ran out of memory")
    state.update(
        baseline_peak_allocated_bytes=int(baseline.peak_bytes),
        baseline_peak_reserved_bytes=int(baseline.reserved_bytes),
    )
    budget = residency_budget(
        total_bytes=memory.total(), free_bytes=memory.free(), reserved_bytes=memory.reserved(),
        peak_reserved_bytes=int(baseline.reserved_bytes), reserve_bytes=reserve_bytes,
        optimizer_reserve_bytes=optimizer_reserve_bytes, override_bytes=override_bytes,
    )
    state["cap_bytes"] = budget.cap_bytes
    sizes = [int(pool.resident_slot_bytes(idx)) for idx in range(n_layers)]
    indices = (
        plan_centered_layers(
            sizes, budget.available_bytes, peak_allocated_bytes=int(baseline.peak_bytes),
            peak_reserved_bytes=int(baseline.reserved_bytes),
        ) if mode == "auto" else centered_layer_indices(n_layers, mode)
    )
    if not indices:
        return refuse("ordinary streaming kept: no complete decoder layer fits the spare budget")
    payload = sum(sizes[idx] for idx in indices)
    if mode != "auto" and payload > budget.available_bytes:
        return refuse("requested resident layers exceed the measured spare budget")
    preload_started = time.perf_counter()
    try:
        clear_stream_substitution_cache(model)
        pool.reserve_layers(indices, runtime.source, getattr(runtime.prefetcher, "stream", None))
        memory.synchronize()
        state["preload_seconds"] = time.perf_counter() - preload_started
        candidate = probe()
        if candidate is None:
            clear()
            return refuse("ordinary streaming kept: resident candidate peak could not be measured")
        state.update(
            candidate_peak_allocated_bytes=int(candidate.peak_bytes),
            candidate_peak_reserved_bytes=int(candidate.reserved_bytes),
        )
        if getattr(candidate, "oom", False):
            clear()
            return refuse("ordinary streaming kept: resident candidate ran out of memory")
        required = int(candidate.reserved_bytes) + reserve_bytes + optimizer_reserve_bytes
        if required > budget.cap_bytes:
            clear()
            return refuse("ordinary streaming kept: resident candidate exceeded the shared cap")
    except Exception as exc:
        clear()
        if not _is_out_of_memory(exc):
            raise
        return refuse("ordinary streaming kept: resident allocation/probe ran out of memory")
    return result(
        "measured resident candidate fits the shared cap",
        indices=indices, resident_bytes=payload,
    )
