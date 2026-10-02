#!/usr/bin/env python3
"""Host-RAM store for the L2L probe's boundary activations, with an NVMe spill.

Layer-major micro-batching (plan.md § "Giant MoE", step 0) keeps the input of
every decoder layer for every micro-batch between the forward walk and the
backward walk: ``n_layers x k`` chunks of ``(1, seq, hidden)``. On the 70B
shape one chunk is 512 x 8192 x 2 B = 8 MiB, an exact power of two, so pinned
chunks do not pay #901's power-of-two rounding.

Layers ``[0, spill_layers)`` go to a spill file instead. They are written first
in the forward and read last in the backward, so their I/O has the most compute
to hide behind. ``DirectSpillFile`` bypasses the page cache, so a spilled chunk
really reaches the drive; it is filled and drained from sector-aligned pinned
staging rings, by one writer and one reader thread. ``BufferedSpillFile``
exists for the CPU tests only, and the probe refuses a backend whose ``direct``
is False.

Benchmark harness code, not shipped. Top level imports the standard library
only; torch is imported inside functions.
"""

from __future__ import annotations

import io
import math
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

SECTOR_BYTES = 4096
DEFAULT_RING = 8
_FILE_SHARE_READ = 0x00000001
_FILE_SHARE_WRITE = 0x00000002
_FILE_FLAG_NO_BUFFERING = 0x20000000


def aligned_buffer(nbytes: int, *, pin: bool) -> Any:
    """A contiguous CPU uint8 tensor of exactly ``nbytes`` at a sector-aligned address.

    One sector is over-allocated and sliced off, because torch's caching host
    allocator can hand out pinned blocks at 512-byte granularity (#1531).
    """
    import torch

    raw = torch.empty(nbytes + SECTOR_BYTES, dtype=torch.uint8, pin_memory=pin)
    shift = (-raw.data_ptr()) % SECTOR_BYTES
    view = raw[shift : shift + nbytes]
    if view.data_ptr() % SECTOR_BYTES:
        raise RuntimeError("could not align a staging buffer to the sector size")
    return view


def _write_all(handle: io.FileIO, offset: int, buf: Any) -> None:
    handle.seek(offset)
    view = memoryview(buf.numpy())
    done = 0
    while done < len(view):
        got = handle.write(view[done:])
        if not got:
            raise OSError(f"short write, {done} of {len(view)} bytes at offset {offset}")
        done += got


def _read_all(handle: io.FileIO, offset: int, buf: Any) -> None:
    handle.seek(offset)
    view = memoryview(buf.numpy())
    done = 0
    while done < len(view):
        got = handle.readinto(view[done:])
        if not got:
            raise OSError(f"short read, {done} of {len(view)} bytes at offset {offset}")
        done += got


class SpillBackend:
    """Positional writes and reads of whole chunks on one file."""

    direct = False
    path = ""
    left_behind: Optional[str] = None

    def _remove(self) -> None:
        """Best effort: an indexer or scanner holding the file must not crash the arm."""
        if not os.path.exists(self.path):
            return
        try:
            os.remove(self.path)
        except OSError:
            self.left_behind = self.path

    def write(self, offset: int, buf: Any) -> None:
        raise NotImplementedError

    def read(self, offset: int, buf: Any) -> None:
        raise NotImplementedError

    def close(self, remove: bool = True) -> None:
        raise NotImplementedError


class BufferedSpillFile(SpillBackend):
    """Page-cache I/O, for the CPU tests. A measured arm never uses it."""

    def __init__(self, path: str, size: int):
        self.path = str(path)
        with open(self.path, "wb") as handle:
            handle.truncate(size)
        self._writer: Optional[io.FileIO] = io.FileIO(self.path, "r+b")
        self._reader: Optional[io.FileIO] = io.FileIO(self.path, "rb")

    def write(self, offset: int, buf: Any) -> None:
        _write_all(self._writer, offset, buf)
        self._writer.flush()

    def read(self, offset: int, buf: Any) -> None:
        _read_all(self._reader, offset, buf)

    def close(self, remove: bool = True) -> None:
        for handle in (self._writer, self._reader):
            if handle is not None:
                handle.close()
        self._writer = self._reader = None
        if remove:
            self._remove()


def _open_unbuffered(path: str) -> io.FileIO:
    """A read-write handle with the page cache bypassed (Windows, Linux, macOS)."""
    if sys.platform == "win32":
        import _winapi
        import msvcrt

        handle = _winapi.CreateFile(
            path,
            _winapi.GENERIC_READ | _winapi.GENERIC_WRITE,
            _FILE_SHARE_READ | _FILE_SHARE_WRITE,
            _winapi.NULL,
            _winapi.OPEN_EXISTING,
            _FILE_FLAG_NO_BUFFERING,
            _winapi.NULL,
        )
        try:
            fd = msvcrt.open_osfhandle(handle, os.O_RDWR | os.O_BINARY)
        except OSError:
            _winapi.CloseHandle(handle)
            raise
        return io.FileIO(fd, "r+b", closefd=True)
    direct = getattr(os, "O_DIRECT", None)
    if direct is not None:
        return io.FileIO(os.open(path, os.O_RDWR | direct), "r+b", closefd=True)
    if sys.platform == "darwin":
        import fcntl

        fd = os.open(path, os.O_RDWR)
        try:
            fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
        except OSError:
            os.close(fd)
            raise
        return io.FileIO(fd, "r+b", closefd=True)
    raise OSError(f"no unbuffered file I/O on {sys.platform}")


def _check_aligned(offset: int, buf: Any) -> None:
    if offset % SECTOR_BYTES or buf.numel() % SECTOR_BYTES or buf.data_ptr() % SECTOR_BYTES:
        raise ValueError(
            "unbuffered I/O needs a sector-aligned offset, length and address; got offset "
            f"{offset}, length {buf.numel()}, address {buf.data_ptr()}"
        )


class DirectSpillFile(SpillBackend):
    """The page cache bypassed, so a spilled chunk really goes to the drive.

    The file is filled with zeros once at construction: an unbuffered write past
    the valid-data length makes the filesystem zero-fill the gap synchronously,
    which would land inside the first timed step instead of here.
    """

    direct = True

    def __init__(self, path: str, size: int, *, fill_chunk: int = 8 << 20):
        if size <= 0 or size % SECTOR_BYTES:
            raise ValueError(
                f"spill size must be a positive multiple of {SECTOR_BYTES}; got {size}"
            )
        self.path = str(path)
        self._writer: Optional[io.FileIO] = None
        self._reader: Optional[io.FileIO] = None
        with open(self.path, "wb") as handle:
            handle.truncate(size)
        try:
            self._writer = _open_unbuffered(self.path)
            zeros = aligned_buffer(fill_chunk, pin=False)
            zeros.zero_()
            for offset in range(0, size, fill_chunk):
                _write_all(self._writer, offset, zeros[: min(fill_chunk, size - offset)])
            self._reader = _open_unbuffered(self.path)
        except BaseException:
            self.close()
            raise

    def write(self, offset: int, buf: Any) -> None:
        _check_aligned(offset, buf)
        _write_all(self._writer, offset, buf)

    def read(self, offset: int, buf: Any) -> None:
        _check_aligned(offset, buf)
        _read_all(self._reader, offset, buf)

    def close(self, remove: bool = True) -> None:
        for handle in (self._writer, self._reader):
            if handle is not None:
                handle.close()
        self._writer = self._reader = None
        if remove:
            self._remove()


class _Ring:
    """Staging buffers handed out in turn. A buffer comes back together with the
    CUDA event that must complete before anyone may overwrite it."""

    def __init__(self, buffers: List[Any]):
        self._free: "queue.Queue[Tuple[Any, Any]]" = queue.Queue()
        for buf in buffers:
            self._free.put((buf, None))

    def acquire(self, abort: Callable[[], bool]) -> Optional[Any]:
        """The next free buffer, or None once ``abort()`` says the store failed or closed."""
        while True:
            try:
                buf, event = self._free.get(timeout=0.5)
                break
            except queue.Empty:
                if abort():
                    return None
        if event is not None:
            event.synchronize()
        return buf

    def release(self, buf: Any, event: Any = None) -> None:
        self._free.put((buf, event))


@dataclass
class StoreStats:
    chunk_bytes: int
    resident_chunks: int
    spilled_chunks: int
    spill_layers: int
    pinned: bool
    direct: Optional[bool]
    bytes_written: int = 0
    bytes_read: int = 0
    put_wait_s: float = 0.0
    get_wait_s: float = 0.0
    spill_left_behind: Optional[str] = None


Key = Tuple[int, int]


class ActivationStore:
    """``n_layers x k`` boundary activations: pinned chunks, plus the lowest
    ``spill_layers`` layers through a spill backend."""

    def __init__(
        self,
        *,
        n_layers: int,
        k: int,
        chunk_shape: Tuple[int, ...],
        dtype: Any,
        device: Any,
        pin: bool,
        spill_layers: int = 0,
        spill_factory: Optional[Callable[[int], SpillBackend]] = None,
        ring: int = DEFAULT_RING,
    ):
        import torch

        if k < 1 or n_layers < 1:
            raise ValueError(f"need n_layers >= 1 and k >= 1; got {n_layers} and {k}")
        if not 0 <= spill_layers <= n_layers:
            raise ValueError(f"spill_layers must be in [0, {n_layers}]; got {spill_layers}")
        if spill_layers and spill_factory is None:
            raise ValueError("spill_layers > 0 needs a spill_factory")
        self.n_layers = int(n_layers)
        self.k = int(k)
        self.shape = tuple(int(dim) for dim in chunk_shape)
        self.dtype = dtype
        self.device = torch.device(device)
        self.is_cuda = self.device.type == "cuda"
        self.spill_layers = int(spill_layers)
        self.chunk_bytes = math.prod(self.shape) * torch.empty((), dtype=dtype).element_size()
        self._error: Optional[BaseException] = None
        self._closing = False
        self._resident: Dict[Key, Any] = {}
        for layer in range(self.spill_layers, self.n_layers):
            for mb in range(self.k):
                self._resident[(layer, mb)] = torch.empty(self.shape, dtype=dtype, pin_memory=pin)
        self.stats = StoreStats(
            chunk_bytes=self.chunk_bytes,
            resident_chunks=len(self._resident),
            spilled_chunks=self.spill_layers * self.k,
            spill_layers=self.spill_layers,
            pinned=bool(pin),
            direct=None,
        )
        self._spill: Optional[SpillBackend] = None
        if not self.spill_layers:
            return
        self._spill = spill_factory(self.spill_layers * self.k * self.chunk_bytes)
        self.stats.direct = bool(self._spill.direct)
        self._write_ring = _Ring([aligned_buffer(self.chunk_bytes, pin=pin) for _ in range(ring)])
        self._read_ring = _Ring([aligned_buffer(self.chunk_bytes, pin=pin) for _ in range(ring)])
        self._writes: "queue.Queue[Any]" = queue.Queue()
        self._reads: "queue.Queue[Any]" = queue.Queue()
        self._cond = threading.Condition()
        self._written: Set[Key] = set()
        self._requested: Set[Key] = set()
        self._ready: Dict[Key, Any] = {}
        self._threads = [
            threading.Thread(target=self._write_loop, name="l2l-spill-writer", daemon=True),
            threading.Thread(target=self._read_loop, name="l2l-spill-reader", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    # -- placement -----------------------------------------------------------
    def is_spilled(self, layer: int) -> bool:
        return layer < self.spill_layers

    def _offset(self, key: Key) -> int:
        layer, mb = key
        return (layer * self.k + mb) * self.chunk_bytes

    def _typed(self, buf: Any) -> Any:
        return buf.view(self.dtype).view(self.shape)

    def _record(self) -> Any:
        if not self.is_cuda:
            return None
        import torch

        event = torch.cuda.Event()
        event.record()
        return event

    # -- errors --------------------------------------------------------------
    def _fail(self, exc: BaseException) -> None:
        with self._cond:
            if self._error is None:
                self._error = exc
            self._cond.notify_all()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            path = self._spill.path if self._spill is not None else "<none>"
            raise RuntimeError(
                f"activation spill to {path} failed: {self._error!r}"
            ) from self._error

    # -- the forward ---------------------------------------------------------
    def put(self, layer: int, mb: int, tensor: Any) -> None:
        self._raise_if_failed()
        key = (layer, mb)
        if not self.is_spilled(layer):
            self._resident[key].copy_(tensor, non_blocking=True)
            return
        started = time.perf_counter()
        buf = self._write_ring.acquire(lambda: self._error is not None or self._closing)
        self.stats.put_wait_s += time.perf_counter() - started
        self._raise_if_failed()
        if buf is None:
            raise RuntimeError("the activation store was closed during a put")
        self._typed(buf).copy_(tensor, non_blocking=True)
        event = self._record()
        with self._cond:
            self._written.discard(key)
        self._writes.put((key, buf, event))

    def _write_loop(self) -> None:
        try:
            while True:
                item = self._writes.get()
                if item is None:
                    return
                key, buf, event = item
                if event is not None:
                    event.synchronize()
                self._spill.write(self._offset(key), buf)
                with self._cond:
                    self.stats.bytes_written += self.chunk_bytes
                    self._written.add(key)
                    self._cond.notify_all()
                self._write_ring.release(buf, None)
        except BaseException as exc:  # noqa: BLE001 — surfaced on the next put/get
            self._fail(exc)

    # -- the backward --------------------------------------------------------
    def _request(self, key: Key) -> None:
        with self._cond:
            if key in self._requested:
                return
            self._requested.add(key)
        self._reads.put(key)

    def begin_backward(self) -> None:
        """Queue every spilled chunk's read in the order the backward consumes them."""
        for layer in range(self.spill_layers - 1, -1, -1):
            for mb in range(self.k):
                self._request((layer, mb))

    def _read_loop(self) -> None:
        try:
            while True:
                key = self._reads.get()
                if key is None:
                    return
                with self._cond:
                    while key not in self._written:
                        if self._closing or self._error is not None:
                            return
                        self._cond.wait(0.5)
                buf = self._read_ring.acquire(lambda: self._closing or self._error is not None)
                if buf is None:
                    return
                self._spill.read(self._offset(key), buf)
                with self._cond:
                    self.stats.bytes_read += self.chunk_bytes
                    self._ready[key] = buf
                    self._cond.notify_all()
        except BaseException as exc:  # noqa: BLE001 — surfaced on the next put/get
            self._fail(exc)

    def get(self, layer: int, mb: int) -> Any:
        """The stored activation as a NEW tensor on the store's device."""
        self._raise_if_failed()
        key = (layer, mb)
        if not self.is_spilled(layer):
            return self._resident[key].to(self.device, non_blocking=True, copy=True)
        self._request(key)
        started = time.perf_counter()
        with self._cond:
            while key not in self._ready:
                if self._error is not None:
                    break
                self._cond.wait(0.5)
            buf = self._ready.pop(key, None)
            self._requested.discard(key)
        self.stats.get_wait_s += time.perf_counter() - started
        self._raise_if_failed()
        out = self._typed(buf).to(self.device, non_blocking=True, copy=True)
        self._read_ring.release(buf, self._record())
        return out

    def end_step(self) -> None:
        """Every spilled chunk requested this step must have been consumed."""
        self._raise_if_failed()
        if not self.spill_layers:
            return
        with self._cond:
            pending = sorted(self._requested | set(self._ready))
        if pending:
            raise RuntimeError(
                f"{len(pending)} spilled activation chunk(s) were not consumed this step, "
                f"first {pending[:4]}: the schedule and the store disagree"
            )

    def close(self) -> None:
        if self._spill is None:
            return
        self._closing = True
        self._writes.put(None)
        self._reads.put(None)
        with self._cond:
            self._cond.notify_all()
        for thread in self._threads:
            thread.join(timeout=30)
        self._spill.close(remove=True)
        self.stats.spill_left_behind = self._spill.left_behind
        self._spill = None
