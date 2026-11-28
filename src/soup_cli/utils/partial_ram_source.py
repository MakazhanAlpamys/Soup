"""Bounded, immutable decoder-layer residency beside the existing disk reader.

The budget covers owned tensor allocations and one mapped layer during loading.
It is not a process-RSS cap: the caller must reserve activations, other buffers,
Python/allocator overhead and safety headroom, including other live host allocations.
"""

import os
import threading
import time
from dataclasses import dataclass
from typing import Any

#: Total seconds one close() waits for disk reader/range workers before it gives up and
#: keeps the lease charged; a later close() waits again.
CLOSE_JOIN_SECONDS = 10.0


def _bytes(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be non-negative integer bytes")


@dataclass(frozen=True)
class PartialRamBudget:
    """Explicit shared-host envelope; cache_bytes is an allocation cap, not a target."""

    limit_bytes: int
    reserved_bytes: int
    cache_bytes: int

    def __post_init__(self) -> None:
        for name in ("limit_bytes", "reserved_bytes", "cache_bytes"):
            _bytes(getattr(self, name), name)
        if self.limit_bytes == 0 or self.reserved_bytes > self.limit_bytes:
            raise ValueError("RAM budget must be positive and cover all reservations")


class PartialRamSource:
    """One consumer, immutable whole-layer cache, unchanged async disk source.

    Only nonresident paths are handed to AsyncDiskSource, so lookahead cannot
    accidentally read cached layers. Global indices are mapped in both get and
    release. Cache loading reads each selected shard once before training.
    """

    def __init__(
        self,
        shard_dir: str,
        n_layers: int,
        spec: Any,
        *,
        budget: PartialRamBudget,
        pin: bool = True,
        shard_paths: Any = None,
        read_ahead: int = 2,
        layer_roots: Any = None,
        cacheable_layers: int | None = None,
        host_budget: Any = None,
    ) -> None:
        from soup_cli.utils.async_disk_source import AsyncDiskSource
        from soup_cli.utils.host_memory_budget import HostMemoryBudget
        from soup_cli.utils.layer_stream_runtime import RamSource, _spec_bytes, plan_pinned_arenas

        if not isinstance(budget, PartialRamBudget):
            raise TypeError("An explicit PartialRamBudget is required")
        if type(n_layers) is not int or n_layers < 1:
            raise ValueError("n_layers must be positive")
        count = n_layers if cacheable_layers is None else cacheable_layers
        if type(count) is not int or not 0 <= count <= n_layers:
            raise ValueError("cacheable_layers must name a decoder prefix")
        specs = RamSource._normalize_layer_specs(spec, n_layers)
        paths = RamSource._normalize_shard_paths(shard_dir, n_layers, shard_paths)
        roots = AsyncDiskSource._normalize_layer_roots(layer_roots, n_layers)
        sizes = [_spec_bytes(entry) for entry in specs]
        self._closed = False
        self._closing = False
        self._disk_events: list[Any] = []
        self._disk = None
        self._ram = None
        self._events: list[Any] = []
        self._lease = None
        self.n_layers = n_layers
        self.pinned = bool(pin)
        self.read_ahead = read_ahead
        self.budget = budget
        self.host_budget = host_budget or HostMemoryBudget(
            budget.limit_bytes, budget.reserved_bytes
        )
        if (
            not isinstance(self.host_budget, HostMemoryBudget)
            or self.host_budget.limit_bytes != budget.limit_bytes
            or self.host_budget.reserved_bytes != budget.reserved_bytes
        ):
            raise ValueError(
                "shared host budget must match the explicit limit and fixed reservation"
            )
        self.resident_tensor_bytes_served = 0
        chosen = []
        tensor_sizes: list[int] = []
        allocation = 0
        # Stable first-fit in decoder order. Whole layers only, including NF4 sidecars.
        for idx in range(count):
            candidate = tensor_sizes + [
                _spec_bytes({key: value}) for key, value in specs[idx].items()
            ]
            amount = plan_pinned_arenas(candidate).pinned_bytes if pin else sum(candidate)
            if sizes[idx] > 0 and amount <= budget.cache_bytes:
                chosen.append(idx)
                tensor_sizes = candidate
                allocation = amount
        self.resident_indices = tuple(chosen)
        self.disk_indices = tuple(idx for idx in range(n_layers) if idx not in chosen)
        self._ram_index = {idx: local for local, idx in enumerate(self.resident_indices)}
        self._disk_index = {idx: local for local, idx in enumerate(self.disk_indices)}
        self.resident_bytes = sum(sizes[idx] for idx in chosen)
        # safe_open can map the source while copying into the final RAM store.
        transient = max((os.path.getsize(paths[idx]) for idx in chosen), default=0)
        try:
            # Claim the currently available envelope atomically before any allocation;
            # shrink to owned buffers after loading. Other consumers cannot race the check.
            self._lease = self.host_budget.reserve(
                "stream.weights", self.host_budget.available_bytes
            )
            remaining = self._lease.nbytes - allocation - transient
            if remaining < 0:
                raise ValueError(
                    "RAM budget cannot cover cache plus loading transient/reservations"
                )
            if self.disk_indices:
                self._disk = AsyncDiskSource(
                    shard_dir,
                    len(self.disk_indices),
                    [specs[idx] for idx in self.disk_indices],
                    shard_paths=[paths[idx] for idx in self.disk_indices],
                    pin=pin,
                    read_ahead=read_ahead,
                    layer_roots=[roots[idx] for idx in self.disk_indices],
                    max_host_bytes=remaining,
                )
            if chosen:
                self._ram = RamSource(
                    shard_dir,
                    len(chosen),
                    [specs[idx] for idx in chosen],
                    pin=pin,
                    shard_paths=[paths[idx] for idx in chosen],
                )
            disk_allocated = getattr(self._disk, "host_allocated_bytes", 0)
            self.host_allocated_bytes = allocation + disk_allocated
            self.accounted_peak_bytes = (
                self.host_budget.used_bytes
                - self._lease.nbytes
                + self.host_allocated_bytes
                + transient
            )
            self.pinned_bytes = getattr(self._ram, "pinned_bytes", 0) + getattr(
                self._disk, "pinned_bytes", 0
            )
            self.nbytes = self.resident_bytes + getattr(self._disk, "nbytes", 0)
            self.staging_bytes = getattr(self._disk, "staging_bytes", 0)
            self.disk_bytes = sum(sizes[idx] for idx in self.disk_indices)
            self._lease.resize(self.host_allocated_bytes)
        except BaseException:
            try:
                self.close()
            except BaseException:  # noqa: BLE001 - lease stays charged; report the cause
                pass
            raise

    def _check(self, idx: int) -> None:
        if self._closed or self._closing:
            raise RuntimeError("partial RAM source is closed or closing")
        if type(idx) is not int:
            raise TypeError("layer index must be an integer")
        if not 0 <= idx < self.n_layers:
            raise IndexError(idx)

    def get(self, idx: int, name: str) -> Any:
        self._check(idx)
        if idx in self._ram_index:
            tensor = self._ram.get(self._ram_index[idx], name)
            self.resident_tensor_bytes_served += tensor.numel() * tensor.element_size()
            return tensor
        return self._disk.get(self._disk_index[idx], name)

    def release(self, idx: int, event: Any = None) -> None:
        self._check(idx)
        if idx in self._disk_index:
            self._disk.release(self._disk_index[idx], event)
        elif event is not None:
            # Keep pinned source storage alive until every outstanding H2D completes.
            self._events = [held for held in self._events if not held.query()]
            self._events.append(event)

    @staticmethod
    def _disk_threads(disk: Any) -> list[Any]:
        threads = [getattr(disk, "_thread", None)]
        for pool in getattr(disk, "_pools", {}).values():
            threads.extend(getattr(pool, "_threads", ()))
        return [thread for thread in threads if thread is not None]

    def close(self) -> None:
        """Release the weight lease only after every user of its memory has finished.

        AsyncDiskSource.close joins with a timeout, so a reader or range worker can
        outlive it while still writing into staging, and it drops the slots' pending
        disk-copy events without waiting on them. Those events are captured first and
        synchronized here, as are cached-layer transfer events. If a worker is still
        alive or an event fails, this raises and keeps buffers and lease charged;
        no further get/release is accepted, and a later close() resumes the drain.
        """
        if self._closed:
            return
        self._closing = True
        disk = self._disk
        if disk is not None:
            ready = getattr(disk, "_ready", None)
            if ready is not None:
                with ready:
                    self._disk_events.extend(
                        event for event in getattr(disk, "_drain", ()) if event is not None
                    )
            disk.close()
            deadline = time.monotonic() + CLOSE_JOIN_SECONDS
            for thread in self._disk_threads(disk):
                if thread.ident is not None and thread is not threading.current_thread():
                    thread.join(timeout=max(0.0, deadline - time.monotonic()))
            alive = [thread.name for thread in self._disk_threads(disk) if thread.is_alive()]
            if alive:
                raise RuntimeError(
                    f"disk staging is still owned by {alive}; the stream.weights lease is "
                    "retained until a later close() finds them finished"
                )
        while self._disk_events:
            self._disk_events[0].synchronize()
            self._disk_events.pop(0)
        while self._events:
            self._events[0].synchronize()
            self._events.pop(0)
        self._ram = None
        self._disk = None
        self.nbytes = 0
        self.host_allocated_bytes = 0
        self.pinned_bytes = 0
        if self._lease is not None:
            self._lease.close()
        self._closed = True
