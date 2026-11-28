"""Research-only persistent dense decoder slots; no config or arithmetic changes.

The shipped pool and its CUDA event ordering remain responsible for every copy.
Selected layers receive private storage once; all other layers retain the original
rotating slots. The host store remains intact. Install before training, after setup.
"""

from __future__ import annotations

import time
from types import MethodType
from typing import Any, Sequence


def install_persistent_layers(runtime: Any, indices: Sequence[int]) -> dict:
    """Give selected frozen dense layers dedicated, never-recycled pool slots."""
    import torch

    pool = runtime.pool
    if getattr(pool, "_persistent_layers", None) is not None:
        raise ValueError("persistent layers already installed")
    chosen = tuple(indices)
    if any(type(idx) is not int for idx in chosen):
        raise ValueError("layer indices must be integers")
    if len(set(chosen)) != len(chosen) or tuple(sorted(chosen)) != chosen:
        raise ValueError("layer indices must be unique and sorted")
    if any(idx < 0 or idx >= runtime.n_layers for idx in chosen):
        raise ValueError("layer index outside decoder stack")
    if len(chosen) >= runtime.n_layers:
        raise ValueError("research requires a partially streamed decoder stack")
    if runtime.tier != "ram":
        raise ValueError("research supports the RAM tier only")
    if any(not value.is_floating_point() for value in pool.buffers[0].values()):
        raise ValueError("research supports dense floating-point weights only")
    base_slots = pool.n
    if not len(pool.buffers) == len(pool.owner) == len(pool.generation) == base_slots:
        raise ValueError("unexpected pool layout")
    if pool.is_cuda:
        torch.cuda.synchronize()
    started = time.perf_counter()
    original_slot = pool.slot_for
    original_load = pool.load_async
    mapping = {idx: base_slots + position for position, idx in enumerate(chosen)}
    added_bytes = 0
    for idx in chosen:
        buffers = {name: torch.empty_like(value) for name, value in pool.buffers[0].items()}
        pool.buffers.append(buffers)
        pool.owner.append(None)
        pool.generation.append(0)
        if pool.is_cuda:
            pool.events.append(torch.cuda.Event())
        added_bytes += sum(value.numel() * value.element_size() for value in buffers.values())
    pool.nbytes += added_bytes
    state = {
        "indices": list(chosen), "dedicated_slots": mapping,
        "rotating_slots": base_slots, "resident_bytes": added_bytes,
        "preload_loads": 0, "resident_load_calls_skipped": 0,
        "dynamic_loads": 0, "dynamic_logical_bytes": 0,
        "arithmetic_changed": False, "host_store_retained": True,
    }
    pool._persistent_layers = state

    def slot_for(self: Any, idx: int) -> int:
        return mapping[idx] if idx in mapping else original_slot(idx)

    pool.slot_for = MethodType(slot_for, pool)
    # Invoke the original copy implementation before installing the skip wrapper.
    # Its stream-wait and source-release events are kept exactly as shipped.
    for idx in chosen:
        original_load(idx, runtime.source, runtime.prefetcher.stream)
        state["preload_loads"] += 1
    if pool.is_cuda:
        torch.cuda.synchronize()
    for idx, slot in mapping.items():
        if pool.owner[slot] != idx:
            raise RuntimeError("persistent slot preload failed")

    def load_async(self: Any, idx: int, source: Any, stream: Any = None) -> int:
        if idx in mapping:
            slot = mapping[idx]
            if self.owner[slot] != idx:
                raise RuntimeError("persistent slot owner changed")
            state["resident_load_calls_skipped"] += 1
            return slot
        before = self.loads
        result = original_load(idx, source, stream)
        if self.loads != before:
            state["dynamic_loads"] += 1
            keys = (
                self.active_keys_by_layer[0] if len(self.active_keys_by_layer) == 1
                else self.active_keys_by_layer[idx]
            )
            buffers = self.buffers[self.slot_for(idx)]
            state["dynamic_logical_bytes"] += sum(
                buffers[key].numel() * buffers[key].element_size() for key in keys
            )
        return result

    pool.load_async = MethodType(load_async, pool)
    state["preload_seconds"] = time.perf_counter() - started
    return state
