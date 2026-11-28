"""Byte layout and a static host cache for packed NF4 expert transfers.

The byte layout for the single-slot ``ExpertBufferPool``. This module holds only CPU logic:
how the stored tensors of one expert lie in one aligned byte block (so one host-to-device
copy moves all of them), how to write them there, and a static host cache of such blocks
charged against an explicit byte budget. The copy to the device lives in
``expert_stream.ExpertBufferPool``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Hashable, Iterable, Mapping, Optional, Tuple

#: Match the CUDA allocator's alignment so vectorised NF4 kernels see aligned packed weights.
_ALIGNMENT = 256


def layout_rows(items: Iterable[Tuple[str, tuple, Any]]) -> Tuple[list, int]:
    """``(rows, size)`` for ``(name, shape, dtype)`` items placed in order in one byte block.

    A row is ``(name, offset, nbytes, dtype, shape)``. Every region starts on a multiple of
    ``max(256, element size)``, scalars and tiny sidecars included.
    """
    rows, cursor = [], 0
    for name, shape, dtype in items:
        width = _element_size(dtype)
        alignment = max(_ALIGNMENT, width)
        cursor = -(-cursor // alignment) * alignment
        count = math.prod(shape) * width
        rows.append((name, cursor, count, dtype, tuple(shape)))
        cursor += count
    if not rows:
        raise ValueError("an expert transfer must be nonempty")
    return rows, cursor


def _element_size(dtype: Any) -> int:
    import torch

    return torch.empty((), dtype=dtype).element_size()


def packing_layout(tensors: Mapping[str, Any]) -> Tuple[list, int]:
    """``layout_rows`` of real tensors, refusing anything but nonempty contiguous CPU memory."""
    if not tensors:
        raise ValueError("an expert transfer must be nonempty")
    for tensor in tensors.values():
        if tensor.device.type != "cpu" or not tensor.is_contiguous() or tensor.numel() == 0:
            raise ValueError("expert transfer requires nonempty contiguous CPU tensors")
    return layout_rows((name, tuple(t.shape), t.dtype) for name, t in tensors.items())


def pack_into(block: Any, rows: Iterable[tuple], tensors: Mapping[str, Any]) -> int:
    """Copy each tensor's exact bytes to its row in ``block``; return the bytes copied.

    ``block`` is a contiguous CPU uint8 tensor. A tensor whose dtype or shape differs from its
    row is refused: a converting copy would change the stored bytes.
    """
    import torch

    copied = 0
    for name, offset, count, dtype, shape in rows:
        tensor = tensors[name]
        if tensor.dtype != dtype or tuple(tensor.shape) != tuple(shape):
            raise ValueError(f"{name}: stored tensor does not match its packed layout row")
        block[offset : offset + count].copy_(tensor.reshape(-1).view(torch.uint8))
        copied += count
    return copied


class PackedHostHotset:
    """Static first-fit host cache of packed byte blocks, charged to one byte budget.

    A block is charged in full, padding included, before it is copied. There is no eviction:
    once the budget is spent later keys are rejected, so a hit never depends on access order.
    Admitted blocks are owned copies and are never mutated.
    """

    def __init__(self, budget_bytes: int) -> None:
        if type(budget_bytes) is not int or budget_bytes < 1:
            raise ValueError("budget_bytes must be a positive integer")
        self.budget_bytes = budget_bytes
        self._blocks: Dict[Hashable, Any] = {}
        self.payload_bytes = 0
        self.hits = 0
        self.misses = 0
        self.rejected = 0

    def get(self, key: Hashable) -> Optional[Any]:
        block = self._blocks.get(key)
        if block is None:
            self.misses += 1
            return None
        self.hits += 1
        return block

    def holds(self, key: Hashable) -> bool:
        """Whether ``key`` is cached, without touching the hit and miss counters."""
        return key in self._blocks

    def put(self, key: Hashable, block: Any) -> bool:
        """Admit a copy of ``block`` when ``key`` is new and the whole block fits."""
        if key in self._blocks or block.numel() > self.budget_bytes - self.payload_bytes:
            self.rejected += 1
            return False
        self._blocks[key] = block.clone()
        self.payload_bytes += block.numel()
        return True

    def stats(self) -> Dict[str, int]:
        return {
            "budget_bytes": self.budget_bytes, "payload_bytes": self.payload_bytes,
            "entries": len(self._blocks), "hits": self.hits, "misses": self.misses,
            "rejected": self.rejected,
        }
