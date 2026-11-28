"""Qualify activation retention after the immutable RAM/GPU caches are installed.

The native retention planner estimates a suffix from one layer. Cached disk
sources may be worth probing too, but their tier name is not speed evidence.
This adapter shares the residency cap and verifies the complete selected suffix.
It is a capacity policy; the benchmark must decide whether retention pays off.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from soup_cli.utils.stream_residency import (
    OPTIMIZER_RESERVE_BYTES_PER_PARAM,
    RESIDENCY_RESERVE_BYTES,
    _measure_preserving_state,
    residency_budget,
)


@dataclass(frozen=True)
class CacheRecomputePlan:
    retained_layers: int = 0
    cap_bytes: int = 0
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0
    reserve_bytes: int = 0
    optimizer_reserve_bytes: int = 0
    setup_seconds: float = 0.0
    reason: str = "not measured"
    record_only_seconds: float = 0.0


def configure_cached_stream_recompute(
    model: Any,
    runtime: Any,
    *,
    rows: int,
    seq_len: int,
    vocab_size: int,
    override_bytes: Optional[int] = None,
    notify: Optional[Callable[[str], Any]] = None,
    memory: Any = None,
    planner: Optional[Callable[..., int]] = None,
    measure: Optional[Callable[..., Any]] = None,
    reserve_bytes: int = RESIDENCY_RESERVE_BYTES,
    optimizer_reserve_bytes: Optional[int] = None,
    retain: Optional[int] = None,
) -> CacheRecomputePlan:
    """Probe native retention with cache allocations and future AdamW charged.

    Run only at pristine setup, after GPU residency and RAM-cache promotion.
    Automatic capacity failure restores recomputation; other execution errors
    propagate. No weights, gradients, RNG or module training modes are changed.
    The production caller is CUDA-only; injectable instruments allow CPU controls.
    A recorded suffix skips the one-layer estimate and runs the full probe only.
    """
    from soup_cli.utils.layer_stream_runtime import (
        CudaMemoryReader,
        _is_out_of_memory,
        clear_stream_substitution_cache,
        measure_step_peak_bytes,
        plan_stream_recompute_auto,
        set_stream_recompute,
    )

    started = time.perf_counter()
    state: dict[str, Any] = {}
    emit = notify if notify is not None else lambda _message: None
    if optimizer_reserve_bytes is None:
        optimizer_reserve_bytes = OPTIMIZER_RESERVE_BYTES_PER_PARAM * sum(
            int(param.numel()) for param in model.parameters() if param.requires_grad
        )
    for value in (reserve_bytes, optimizer_reserve_bytes):
        if type(value) is not int or value < 0:
            raise ValueError("setup and optimizer reserves must be non-negative integer bytes")
    if retain is not None and (type(retain) is not int or retain < 1):
        raise ValueError("recorded activation retention must be a positive layer count")
    state.update(reserve_bytes=reserve_bytes, optimizer_reserve_bytes=optimizer_reserve_bytes)

    def result(reason: str, **updates: Any) -> CacheRecomputePlan:
        state.update(updates)
        plan = CacheRecomputePlan(
            **state, reason=reason, setup_seconds=time.perf_counter() - started,
        )
        emit(
            "[cyan]Cached stream activation plan:[/] "
            f"{plan.retained_layers} retained layers; {reason}; "
            f"shared cap {plan.cap_bytes} bytes, "
            f"setup/optimizer reserves {reserve_bytes}/{optimizer_reserve_bytes} bytes"
        )
        return plan

    def fallback(reason: str) -> CacheRecomputePlan:
        set_stream_recompute(model, True)
        clear_stream_substitution_cache(model)
        memory.synchronize()
        memory.release()
        return result(reason)

    if memory is None:
        if not getattr(runtime.pool, "is_cuda", False):
            return result("no CUDA pool to measure")
        memory = CudaMemoryReader(runtime.pool.device)
    planner = plan_stream_recompute_auto if planner is None else planner
    measure = measure_step_peak_bytes if measure is None else measure
    budget = residency_budget(
        total_bytes=memory.total(), free_bytes=memory.free(),
        reserved_bytes=memory.reserved(), peak_reserved_bytes=0,
        reserve_bytes=reserve_bytes, optimizer_reserve_bytes=optimizer_reserve_bytes,
        override_bytes=override_bytes,
    )
    state["cap_bytes"] = budget.cap_bytes
    if budget.cap_bytes <= reserve_bytes + optimizer_reserve_bytes:
        return fallback("setup/optimizer reserves leave no activation budget")
    shape = dict(rows=rows, seq_len=seq_len, vocab_size=vocab_size, device=runtime.pool.device)

    def select(current: Any, **_shape: Any) -> int:
        return planner(
            current, rows=rows, seq_len=seq_len, vocab_size=vocab_size,
            notify=emit, memory=memory,
            override_bytes=budget.cap_bytes - reserve_bytes - optimizer_reserve_bytes,
        )

    try:
        if retain is None:
            planning_started = time.perf_counter()
            count = _measure_preserving_state(model, select, **shape)
            state["record_only_seconds"] = time.perf_counter() - planning_started
            if count == 0:
                return fallback("native one-layer probe selected no retained layers")
        else:
            count = retain
            if set_stream_recompute(model, False, retain=count) < count:
                return fallback("recorded retained suffix exceeds the decoder layer count")
        peak = _measure_preserving_state(model, measure, **shape)
        if peak is not None and getattr(peak, "failed", False):
            raise RuntimeError(f"cached activation probe failed: {peak.error}")
        if peak is None or getattr(peak, "oom", False):
            return fallback("complete retained suffix could not fit or be measured")
        state.update(
            peak_allocated_bytes=int(peak.peak_bytes), peak_reserved_bytes=int(peak.reserved_bytes),
        )
        current = residency_budget(
            total_bytes=memory.total(), free_bytes=memory.free(),
            reserved_bytes=memory.reserved(), peak_reserved_bytes=int(peak.reserved_bytes),
            reserve_bytes=reserve_bytes, optimizer_reserve_bytes=optimizer_reserve_bytes,
            override_bytes=budget.cap_bytes,
        )
        state["cap_bytes"] = current.cap_bytes
        required = int(peak.reserved_bytes) + reserve_bytes + optimizer_reserve_bytes
        if required > current.cap_bytes:
            return fallback("complete retained suffix exceeded the shared cap including reserves")
    except Exception as exc:
        set_stream_recompute(model, True)
        clear_stream_substitution_cache(model)
        memory.synchronize()
        memory.release()
        if not _is_out_of_memory(exc):
            raise
        return result("cached activation allocation/probe ran out of memory")
    return result("complete retained suffix fits the shared cap", retained_layers=count)
