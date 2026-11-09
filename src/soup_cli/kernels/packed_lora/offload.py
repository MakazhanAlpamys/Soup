# SPDX-License-Identifier: Apache-2.0
#
# Origin: Soup Kernel Challenge (Decentrathon 6.0), the submission of team Luck
# (BAITC-Hacks/hack-e79529fc-luck@b3f1225f); the int8 gate/up packs follow ideas seen in the
# solutions of teams Null Pointers, 21, DarkDaur and OrbiTeam.
# Modified for Soup: the module-level stash and registry became a per-run :class:`PinnedPool`
# that unlocks everything it locked when closed; the background pinning threads, the
# ``/proc/self/maps`` unregister shortcut and the module switches of the original are not ported.
"""What each layer's backward needs, between forward and backward: on the GPU where it has room,
else in pinned host memory.

Keeping a layer's saved tensors on the GPU costs memory at the peak; recomputing them costs
matmuls (gradient checkpointing). Here a layer's saved tensors either stay on the GPU (resident:
nothing is copied, nothing recomputed) or, for the layers that do not fit, the forward copies
them to pinned CPU memory on a separate CUDA stream while it goes on computing, and the backward
copies each layer's tensors back (two or three copies per layer, gate and up first) one layer
ahead of use. Which layers are resident is decided per step (:meth:`Stash.plan_count`, from the
memory plan, or :meth:`Stash.plan`: as many of the LAST layers as fit the free GPU memory). The
resident layers are the ones whose backward runs first, so the layers that do travel have the
longest to come back. The copy engine works in parallel with the compute units, so the transfers
hide behind the matmuls. The values are the same tensors, bit for bit, except gate and up (73 %
of the bytes), which the int8 backward reads anyway: they are stored as int8 packs (``q8``,
1.88x fewer bytes).

The pinned buffers (one per offloaded layer, sized for the largest batch) are plain CPU buffers
of the exact size, page-locked with ``cudaHostRegister``: PyTorch's pinned allocator rounds every
buffer up to a power of two, which at 9 GB of buffers is a lot of host memory. The pool
unregisters them when the run closes.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor

import torch

from .layout import ALIGN, layer_bytes, step_reserve_bytes  # noqa: F401  (re-exported)

IN_FLIGHT = 4  # saves whose device->host copies may still be running (4 per layer at most)
SAFETY_BYTES = 1 << 30  # GPU memory left alone when the resident layers are counted
#: Offload and the host-parked tables move gigabytes a step over PCIe. Below this speed the link
#: is not worth planning around (a healthy gen3 x16 link does 10-12 GB/s; one of eight Kaggle
#: sessions measured 2.4 GB/s and the run came out slower than the stock step).
LINK_MIN_BYTES_PER_S = 4e9
LINK_BUDGET_S = 0.2  # copy time allowed before calling the link slow


def measure_link(n_bytes=32 << 20):
    """Pinned host <-> GPU copy speed in bytes/s: the slower direction's fastest copy. Copies go
    on, both ways, until each direction has had one copy above ``LINK_MIN_BYTES_PER_S`` or
    ``LINK_BUDGET_S`` of copying has passed: a few ms on a healthy link. Any error answers None
    (unknown), never stops training."""
    if not torch.cuda.is_available():
        return None
    try:
        stream = torch.cuda.current_stream()
        src = torch.empty(n_bytes, dtype=torch.uint8, pin_memory=True)  # a scratch buffer
        gpu = torch.empty(n_bytes, device=stream.device, dtype=torch.uint8)
        best = {True: 0.0, False: 0.0}
        spent = 0.0
        while spent < LINK_BUDGET_S and min(best.values()) < LINK_MIN_BYTES_PER_S:
            for to_gpu in (True, False):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(stream)
                if to_gpu:
                    gpu.copy_(src, non_blocking=True)
                else:
                    src.copy_(gpu, non_blocking=True)
                end.record(stream)
                end.synchronize()
                seconds = start.elapsed_time(end) / 1e3
                spent += seconds
                best[to_gpu] = max(best[to_gpu], n_bytes / seconds)
        del gpu
        return min(best.values())
    except Exception:  # noqa: BLE001 - a failed probe must never stop training
        return None


def is_slow(link_bytes_per_s):
    """Whether a measured link speed (None: unknown) is below ``LINK_MIN_BYTES_PER_S``."""
    return link_bytes_per_s is not None and link_bytes_per_s < LINK_MIN_BYTES_PER_S


class PinnedPool:
    """Page-locked host buffers of one run; closing the pool unlocks every one it locked."""

    def __init__(self):
        self._registered = {}  # data_ptr -> bytes, until unregistered

    def register(self, buffer):
        """Page-lock a CPU tensor's memory in place; False if the driver refused."""
        n_bytes = buffer.numel() * buffer.element_size()
        ptr = buffer.data_ptr()
        ok = int(torch.cuda.cudart().cudaHostRegister(ptr, n_bytes, 0)) == 0
        if ok:
            self._registered[ptr] = n_bytes
        return ok

    def unregister(self, ptr):
        """Page-unlock a buffer this pool locked (once: a second call, or an address the pool
        never saw, does nothing). True if the driver unpinned it."""
        if self._registered.pop(ptr, None) is None:
            return False
        return int(torch.cuda.cudart().cudaHostUnregister(ptr)) == 0

    def alloc(self, n_bytes):
        """A page-locked uint8 CPU buffer of exactly ``n_bytes``."""
        buffer = torch.empty(n_bytes, dtype=torch.uint8)
        if self.register(buffer):
            return buffer
        return torch.empty(n_bytes, dtype=torch.uint8, pin_memory=True)

    def locked_bytes(self):
        """Bytes this pool has page-locked and not yet released."""
        return sum(self._registered.values())

    def close(self):
        """Unlock everything this pool locked (the tensors stay valid, as pageable memory)."""
        for ptr in list(self._registered):
            self.unregister(ptr)


class Stash:
    """Saved tensors of every layer between its forward and its backward."""

    def __init__(self, n_layers, capacity_bytes, pool, n_host=None, eager=True, workers=1):
        """Pinned buffers for the first ``n_host`` layers (all, by default) now (``eager``) or at
        first use; the layers past them get theirs at first use if a step ever offloads them."""
        self.capacity = capacity_bytes
        self.pool = pool
        n_host = n_layers if n_host is None else n_host
        if eager and n_host:
            with ThreadPoolExecutor(workers) as executor:
                self.host = list(executor.map(pool.alloc, [capacity_bytes] * n_host))
            self.host += [None] * (n_layers - n_host)
        else:
            self.host = [None] * n_layers
        self.resident_from = n_layers  # layers from this index on keep their tensors on the GPU
        self.kept = [[] for _ in range(n_layers)]
        self.layout = [[] for _ in range(n_layers)]  # (offset, bytes, dtype, shape) per tensor
        self.used = [0] * n_layers
        self.stream = None  # the copy stream
        self.main = None  # the compute stream, read once per step (begin_step)
        self.device = None
        self.pending = {}
        self.in_flight = deque()
        self.held = deque()  # (copy done event, tensors) per save, until release()
        self.saves = 0
        # CUDA events made once and reused: "compute queued so far" for the copy stream to
        # wait on, one per save still in flight (+1), one per copy back (layer, tensors)
        self.queued = self.copied = None
        self.arrived = {}

    @classmethod
    def resident_only(cls, n_layers, pool):
        """A stash without pinned buffers: every layer stays on the GPU."""
        stash = cls(n_layers, 0, pool, eager=False)
        stash.resident_from = 0
        return stash

    def close(self):
        """Unlock and drop the pinned buffers."""
        hosts, self.host = self.host, []
        for host in hosts:
            if host is not None:
                self.pool.unregister(host.data_ptr())
        hosts.clear()
        self.kept = [[] for _ in self.kept]
        self.pending.clear()

    def fits(self, n_layers, bytes_per_layer):
        """Whether the buffers can hold a step of ``bytes_per_layer`` bytes a layer."""
        return len(self.host) == n_layers and bytes_per_layer <= self.capacity

    def plan(self, mode, per_layer_bytes, reserve_bytes, device, tokens_ok=True):
        """Decide this step's resident layers from the free GPU memory; sets ``resident_from``.
        mode: "offload" | "resident" | "auto". ``per_layer_bytes``: what one layer's saved tensors
        take on the GPU; ``reserve_bytes``: what the step needs next to them (transients,
        optimizer state). ``tokens_ok`` False (the pinned buffers cannot hold a layer of this
        step): every layer stays on the GPU."""
        n = len(self.host)
        if mode == "resident" or not tokens_ok:
            self.resident_from = 0
        elif mode == "offload":
            self.resident_from = n
        else:
            room = free_gpu_bytes(device) - reserve_bytes - SAFETY_BYTES
            count = max(0, min(n, int(room // max(per_layer_bytes, 1))))
            self.resident_from = n - count
        return self.resident_from

    def plan_count(self, count):
        """This step's resident layers = the last ``count`` (the memory plan, budget.py)."""
        n = len(self.host)
        self.resident_from = n - max(0, min(n, count))
        return self.resident_from

    def _copy_stream(self, device):
        if self.stream is None:
            self.stream = torch.cuda.Stream(device)
        return self.stream

    def begin_step(self, device):
        """Called before the step's first save: forget the last step, note the compute stream
        (autograd runs the backward on the forward's stream, so it holds for the whole step)."""
        if self.queued is None:
            self.queued = torch.cuda.Event()
            self.copied = [torch.cuda.Event() for _ in range(IN_FLIGHT + 1)]
        self.release()  # (nothing left from the last step's forward, normally)
        self.device = device
        self.main = torch.cuda.current_stream(device)
        for i in range(len(self.layout)):
            self.layout[i], self.used[i] = [], 0
            self.kept[i] = []
        self.pending.clear()

    def _on_copy_stream(self):
        """Switch to the copy stream, after everything queued so far on the compute stream."""
        stream = self._copy_stream(self.device)
        self.queued.record(self.main)
        stream.wait_event(self.queued)
        torch.cuda.set_stream(stream)
        return stream

    def save(self, index, *tensors):
        """Copy the tensors of layer ``index`` to the host, asynchronously; the caller may drop
        them right away (the stash keeps them until release()). A resident layer keeps the
        tensors themselves, on the GPU, until they are loaded."""
        if index >= self.resident_from:
            layout = self.layout[index]
            for t in tensors:
                if not t.is_contiguous():
                    # k[:, ::group] / v[:, ::group] are views of the GQA-repeated tensors: keeping
                    # the view keeps all `group` copies alive (7x for Qwen2.5-7B); keep only the
                    # rows that are needed, as the offloaded copy does
                    t = t.contiguous()
                layout.append((None, t.nbytes, t.dtype, t.shape))
                self.kept[index].append(t)
            return
        host = self.host[index]
        if host is None:
            host = self.host[index] = self.pool.alloc(self.capacity)
        at, layout = self.used[index], self.layout[index]
        # events are reused in turn: the one recorded IN_FLIGHT + 1 saves ago was waited for
        copied = self.copied[self.saves % (IN_FLIGHT + 1)]
        self.saves += 1
        stream = self._on_copy_stream()
        try:
            for t in tensors:
                n = t.nbytes
                host[at:at + n].view(t.dtype).view(t.shape).copy_(t, non_blocking=True)
                layout.append((at, n, t.dtype, t.shape))
                at += -(-n // ALIGN) * ALIGN
            copied.record(stream)
        finally:
            torch.cuda.set_stream(self.main)
        self.held.append((copied, tensors))
        self.used[index] = at
        # bound how far the host runs ahead of the copies: it waits for the copy issued
        # IN_FLIGHT saves ago (the GPU still has everything queued since then to do)
        self.in_flight.append(copied)
        while len(self.in_flight) > IN_FLIGHT:
            self.in_flight.popleft().synchronize()

    def release(self, keep=0):
        """Drop the tensors of all saves but the newest ``keep``. The compute stream first waits
        (on the GPU, the host does not) for their copies, so their blocks go back to the allocator
        at once, for work queued after this point on that stream; called where the copies have
        normally long finished. (Instead of record_stream, which frees a block only once the host
        sees its copy done, and the host runs ahead of the GPU.) At most IN_FLIGHT saves are held,
        so their events are not recorded again before this waits on them."""
        n = len(self.held) - keep
        if n <= 0:
            return
        self.main.wait_event(self.held[n - 1][0])  # copies run in order: the newest one's
        for _ in range(n):
            self.held.popleft()

    def prefetch(self, index, device, first=0, last=None):
        """Start copying tensors first..last-1 (in save order) of layer ``index`` back to the GPU,
        in one transfer into a buffer of their own (no-op if started)."""
        key = (index, first, last)
        if index < 0 or index >= self.resident_from or key in self.pending:
            return
        layout = self.layout[index][first:last]
        start, stop = layout[0][0], layout[-1][0] + layout[-1][1]
        buffer = torch.empty(stop - start, device=device, dtype=torch.uint8)
        done = self.arrived.get(key)
        if done is None:
            done = self.arrived[key] = torch.cuda.Event()
        # the copy stream waits for the compute stream: the new block may still be read by
        # earlier work queued there
        stream = self._on_copy_stream()
        try:
            buffer.copy_(self.host[index][start:stop], non_blocking=True)
            done.record(stream)
        finally:
            torch.cuda.set_stream(self.main)
        self.pending[key] = (buffer, done, start)

    def load(self, index, device, first=0, last=None):
        """Tensors first..last-1 of layer ``index`` on the GPU (views of one buffer), in the order
        saved. A resident layer gives its tensors up (the stash holds no reference afterwards)."""
        if index >= self.resident_from:
            kept = self.kept[index]
            out = tuple(kept[first:last])
            for i in range(first, last):
                kept[i] = None
            return out
        self.prefetch(index, device, first, last)
        buffer, done, start = self.pending.pop((index, first, last))
        self.main.wait_event(done)
        return tuple(buffer[at - start:at - start + n].view(dtype).view(shape)
                     for at, n, dtype, shape in self.layout[index][first:last])


def free_gpu_bytes(device):
    """Bytes a new allocation can still count on: the driver's free memory plus what PyTorch's
    allocator holds cached."""
    free, _ = torch.cuda.mem_get_info(device)
    return free + torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)


def stash_capacity(config, save_x=False, tokens=4096, qk_norm=False):
    """Bytes per layer for a batch of up to ``tokens`` tokens, from a transformers model config."""
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    return layer_bytes(tokens, config.hidden_size, config.intermediate_size,
                       config.num_attention_heads * head_dim, config.num_key_value_heads * head_dim,
                       save_x, True, qk_norm)


__all__ = ["ALIGN", "PinnedPool", "Stash", "free_gpu_bytes", "is_slow", "layer_bytes",
           "measure_link", "stash_capacity", "step_reserve_bytes"]
