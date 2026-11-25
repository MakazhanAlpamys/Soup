"""C3/A4 shared allocation bridge for uncompressed L2L activations.

An explicit shared allocator wraps the existing resident/spill store. Construct
activations before optional C3 residency to preserve A4's priority. This is not
a process-RSS limit. The public L2L path uses a separately reserved RAM partition.
"""

from collections.abc import Callable, Sequence
from typing import Any

from soup_cli.utils.host_memory_budget import HostMemoryBudget
from soup_cli.utils.l2l_activations import (
    DEFAULT_RING,
    SECTOR_BYTES,
    ActivationStore,
    SpillBackend,
)


def activation_allocation_bytes(
    *, n_layers: int, k: int, chunk_bytes: int, spill_layers: int, ring: int, pin: bool
) -> int:
    """A4 uncompressed layout: resident chunks plus TWO aligned staging rings.

    Matches StoreLayout.host_bytes at A4 commit d6f7f733 (floor bytes = 0).
    Charge alignment slack before rounding each actual pinned allocation.
    """
    from soup_cli.utils.layer_stream_runtime import _next_power_of_two

    if any(type(value) is not int or value < 1 for value in (n_layers, k, chunk_bytes, ring)):
        raise ValueError("activation dimensions, chunk bytes and ring must be positive integers")
    if type(spill_layers) is not int or not 0 <= spill_layers <= n_layers:
        raise ValueError("spill_layers must be an integer within the layer count")
    if type(pin) is not bool:
        raise ValueError("pin must be a boolean")
    block = _next_power_of_two(chunk_bytes) if pin else chunk_bytes
    total = (n_layers - spill_layers) * k * block
    if spill_layers:
        request = chunk_bytes + SECTOR_BYTES
        total += 2 * ring * (_next_power_of_two(request) if pin else request)
    return total


class BudgetedActivationStore(ActivationStore):
    def __init__(
        self,
        *,
        host_budget: HostMemoryBudget,
        n_layers: int,
        k: int,
        chunk_shape: tuple[int, ...],
        dtype: Any,
        device: Any,
        pin: bool,
        mb_shapes: Sequence[tuple[int, ...]] | None = None,
        spill_layers: int = 0,
        spill_factory: Callable[[int], SpillBackend] | None = None,
        ring: int = DEFAULT_RING,
    ) -> None:
        import math

        import torch

        dimensions = (n_layers, k, *chunk_shape)
        if not chunk_shape or any(type(value) is not int or value < 1 for value in dimensions):
            raise ValueError("activation dimensions must be positive integers")
        chunk_bytes = math.prod(chunk_shape) * torch.empty((), dtype=dtype).element_size()
        allocation = activation_allocation_bytes(
            n_layers=n_layers,
            k=k,
            chunk_bytes=chunk_bytes,
            spill_layers=spill_layers,
            ring=ring,
            pin=pin,
        )
        if spill_layers and spill_factory is None:
            raise ValueError("spill_layers > 0 needs a spill_factory")
        self._budget_closed = False
        self._budget_closing = False
        self._budget_lease = host_budget.reserve("a4.activations", allocation)
        try:
            super().__init__(
                n_layers=n_layers,
                k=k,
                chunk_shape=chunk_shape,
                dtype=dtype,
                device=device,
                pin=pin,
                mb_shapes=mb_shapes,
                spill_layers=spill_layers,
                spill_factory=spill_factory,
                ring=ring,
            )
        except BaseException:
            self.close()
            raise

    def _raise_if_failed(self) -> None:
        if self._budget_closed or self._budget_closing:
            raise RuntimeError("budgeted activation store is closed or closing")
        super()._raise_if_failed()

    def begin_backward(self) -> None:
        self._raise_if_failed()
        super().begin_backward()

    def close(self) -> None:
        """Drain workers and device transfers BEFORE freeing buffers or their lease.

        A failed join/synchronization/backend close retains the allocations and
        reservation; a later close can retry cleanup. No new work is accepted.
        Also handles a constructor that only allocated some of its buffers.
        Caller must serialize close with put/get (same contract as the source).
        """
        if self._budget_closed:
            return
        self._budget_closing = True
        self._closing = True
        for name in ("_writes", "_reads"):
            pending = getattr(self, name, None)
            if pending is not None:
                pending.put(None)
        condition = getattr(self, "_cond", None)
        if condition is not None:
            with condition:
                condition.notify_all()
        for worker in getattr(self, "_threads", ()):
            if worker.ident is not None:
                worker.join(timeout=30)
        if any(worker.is_alive() for worker in getattr(self, "_threads", ())):
            raise RuntimeError("activation spill worker still owns buffers; reservation retained")
        if getattr(self, "is_cuda", False):
            import torch

            # Includes resident copies and H2D events held by read-ring buffers.
            torch.cuda.synchronize(self.device)
        spill = getattr(self, "_spill", None)
        if spill is not None:
            spill.close(remove=True)
            self.stats.spill_left_behind = spill.left_behind
            self._spill = None
        for name in ("_resident", "_ready", "_requested", "_written"):
            held = getattr(self, name, None)
            if held is not None:
                held.clear()
        # Queues/rings can hold tensors even after both worker threads exit.
        self._write_ring = self._read_ring = None
        self._writes = self._reads = None
        self._threads = []
        # A worker exception traceback may retain its last staging tensor.
        self._error = None
        self._budget_lease.close()
        self._budget_closed = True
