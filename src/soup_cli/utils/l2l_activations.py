"""Host-RAM store for the L2L step's boundary activations, with an NVMe spill.

Layer-major micro-batching keeps the input of
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

``chunk_shape`` is a capacity. A tensor with fewer elements (a shorter window)
is stored in the front of its slot and comes back from ``get`` in its own shape.
Every slot of the spill file and every staging buffer is the chunk rounded up to
the sector, so unbuffered I/O stays aligned for any chunk size.

Top level imports the standard library only; torch is imported inside functions.
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
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

SECTOR_BYTES = 4096
DEFAULT_RING = 8
_FILE_SHARE_READ = 0x00000001
_FILE_SHARE_WRITE = 0x00000002
_FILE_FLAG_NO_BUFFERING = 0x20000000
# CPython caps Windows FileIO requests at INT_MAX, which is not sector-aligned.
_RAW_IO_MAX_BYTES = (1 << 31) - 1


def sector_round_up(nbytes: int) -> int:
    """``nbytes`` rounded up to a whole number of sectors."""
    return -(-int(nbytes) // SECTOR_BYTES) * SECTOR_BYTES


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
    max_request = _RAW_IO_MAX_BYTES // SECTOR_BYTES * SECTOR_BYTES
    done = 0
    while done < len(view):
        got = handle.write(view[done : done + max_request])
        if not got:
            raise OSError(f"short write, {done} of {len(view)} bytes at offset {offset}")
        done += got


def _read_all(handle: io.FileIO, offset: int, buf: Any) -> None:
    handle.seek(offset)
    view = memoryview(buf.numpy())
    max_request = _RAW_IO_MAX_BYTES // SECTOR_BYTES * SECTOR_BYTES
    done = 0
    while done < len(view):
        got = handle.readinto(view[done : done + max_request])
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
    which would land inside the first timed step instead of here. A construction
    that fails (a full drive included) removes the file it created.
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
        created = False
        try:
            with open(self.path, "wb") as handle:
                created = True
                handle.truncate(size)
            self._writer = _open_unbuffered(self.path)
            zeros = aligned_buffer(fill_chunk, pin=False)
            zeros.zero_()
            for offset in range(0, size, fill_chunk):
                _write_all(self._writer, offset, zeros[: min(fill_chunk, size - offset)])
            self._reader = _open_unbuffered(self.path)
        except BaseException:
            self.close(remove=created)
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
            try:
                event.synchronize()
            except BaseException:
                self._free.put((buf, event))
                raise
        return buf

    def release(self, buf: Any, event: Any = None) -> None:
        self._free.put((buf, event))

    def close(self) -> None:
        """Finish outstanding copies before releasing the host buffers they read."""
        while True:
            try:
                buf, event = self._free.get_nowait()
            except queue.Empty:
                return
            if event is not None:
                try:
                    event.synchronize()
                except BaseException:
                    self._free.put((buf, event))
                    raise


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
    ``spill_layers`` layers through a spill backend.

    ``chunk_shape`` sizes the largest chunk and the spill slots. ``mb_shapes`` may
    give each micro-batch a smaller capacity. ``put`` accepts up to that capacity;
    ``get`` returns each layer's activation in the shape it was put with.
    """

    def __init__(
        self,
        *,
        n_layers: int,
        k: int,
        chunk_shape: Tuple[int, ...],
        dtype: Any,
        device: Any,
        pin: bool,
        mb_shapes: Optional[Sequence[Tuple[int, ...]]] = None,
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
        #: Resident capacity of each micro-batch's chunks. Without ``mb_shapes`` every
        #: micro-batch gets ``chunk_shape``; with it, each gets its own (rows of different
        #: lengths), and ``chunk_shape`` must hold the largest (it sizes the spill chunk).
        self.mb_shapes = (
            [self.shape] * self.k
            if mb_shapes is None
            else [tuple(int(dim) for dim in shape) for shape in mb_shapes]
        )
        if len(self.mb_shapes) != self.k:
            raise ValueError(f"{len(self.mb_shapes)} mb_shapes for k={self.k}")
        if any(math.prod(shape) > math.prod(self.shape) for shape in self.mb_shapes):
            raise ValueError(f"an mb_shape is larger than chunk_shape {self.shape}")
        self.dtype = dtype
        self.device = torch.device(device)
        self.is_cuda = self.device.type == "cuda"
        self.spill_layers = int(spill_layers)
        self.capacity = math.prod(self.shape)
        self._itemsize = torch.empty((), dtype=dtype).element_size()
        self.chunk_bytes = self.capacity * self._itemsize
        #: One spilled chunk's place in the file and in a staging buffer: unbuffered
        #: I/O needs sector-aligned offsets and lengths whatever the chunk size.
        self.slot_bytes = sector_round_up(self.chunk_bytes)
        self._error: Optional[BaseException] = None
        self._closing = False
        self._closed = False
        self._shapes: Dict[Key, Tuple[int, ...]] = {}
        self._resident: Dict[Key, Any] = {}
        self._spill: Optional[SpillBackend] = None
        self.spill_path: Optional[str] = None
        self._write_ring: Optional[_Ring] = None
        self._read_ring: Optional[_Ring] = None
        self._writes: Optional[queue.Queue[Any]] = None
        self._reads: Optional[queue.Queue[Any]] = None
        self._cond: Optional[threading.Condition] = None
        self._written: Set[Key] = set()
        self._requested: Set[Key] = set()
        self._ready: Dict[Key, Any] = {}
        self._step_keys: Set[Key] = set()
        self._threads: List[threading.Thread] = []
        self._worker_done: List[threading.Event] = []
        self.stats = StoreStats(
            chunk_bytes=self.chunk_bytes,
            resident_chunks=(self.n_layers - self.spill_layers) * self.k,
            spilled_chunks=self.spill_layers * self.k,
            spill_layers=self.spill_layers,
            pinned=bool(pin),
            direct=None,
        )
        try:
            for layer in range(self.spill_layers, self.n_layers):
                for mb in range(self.k):
                    self._resident[(layer, mb)] = torch.empty(
                        math.prod(self.mb_shapes[mb]), dtype=dtype, pin_memory=pin
                    )
            if not self.spill_layers:
                return
            self._spill = spill_factory(self.spill_layers * self.k * self.slot_bytes)
            self.spill_path = self._spill.path
            self.stats.direct = bool(self._spill.direct)
            self._write_ring = _Ring(
                [aligned_buffer(self.slot_bytes, pin=pin) for _ in range(ring)]
            )
            self._read_ring = _Ring(
                [aligned_buffer(self.slot_bytes, pin=pin) for _ in range(ring)]
            )
            self._writes = queue.Queue()
            self._reads = queue.Queue()
            self._cond = threading.Condition()
            self._worker_done = [threading.Event(), threading.Event()]
            self._threads = [
                threading.Thread(target=self._write_loop, name="l2l-spill-writer", daemon=True),
                threading.Thread(target=self._read_loop, name="l2l-spill-reader", daemon=True),
            ]
            for thread in self._threads:
                thread.start()
        except BaseException:
            # A failed constructor never reaches the caller's store assignment/finally.
            self.close()
            raise

    # -- placement -----------------------------------------------------------
    def is_spilled(self, layer: int) -> bool:
        return layer < self.spill_layers

    def _offset(self, key: Key) -> int:
        layer, mb = key
        return (layer * self.k + mb) * self.slot_bytes

    def _fit(self, mb: int, tensor: Any) -> Tuple[int, ...]:
        """The stored shape; refuses one larger than this micro-batch's capacity."""
        shape = tuple(int(dim) for dim in tensor.shape)
        capacity_shape = self.mb_shapes[mb]
        if shape != capacity_shape and math.prod(shape) > math.prod(capacity_shape):
            raise ValueError(
                f"an activation of shape {shape} does not fit micro-batch {mb}'s "
                f"{capacity_shape} chunks"
            )
        return shape

    def _view(self, chunk: Any, shape: Tuple[int, ...]) -> Any:
        """The front of a resident chunk, as a tensor of ``shape``."""
        if chunk.shape == shape:
            return chunk
        return chunk.view(-1)[: math.prod(shape)].view(shape)

    def _typed(self, buf: Any, shape: Tuple[int, ...]) -> Any:
        flat = buf[: self.chunk_bytes].view(self.dtype)
        if shape == self.shape:
            return flat.view(self.shape)
        return flat[: math.prod(shape)].view(shape)

    def _io_bytes(self, shape: Tuple[int, ...]) -> int:
        """Bytes one transfer of a ``shape`` chunk moves: its own size, sector-rounded."""
        return sector_round_up(math.prod(shape) * self._itemsize)

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
        if self._error is None:
            return
        from soup_cli.utils.l2l_spill import SpillDiskFullError, disk_full_message, is_disk_full

        path = self.spill_path or "<none>"
        full = is_disk_full(self._error)
        if full is not None:
            # The file is removed before the error leaves, so the message can say so.
            self.close()
            raise SpillDiskFullError(
                disk_full_message(
                    path,
                    file_bytes=self.spill_layers * self.k * self.slot_bytes,
                    during="writing",
                    written=self.stats.bytes_written,
                    left_behind=self.stats.spill_left_behind,
                    error=full,
                )
            ) from self._error
        raise RuntimeError(f"activation spill to {path} failed: {self._error!r}") from self._error

    # -- the forward ---------------------------------------------------------
    def put(self, layer: int, mb: int, tensor: Any) -> None:
        self._raise_if_failed()
        key = (layer, mb)
        shape = self._fit(mb, tensor)
        if not self.is_spilled(layer):
            self._view(self._resident[key], shape).copy_(tensor, non_blocking=self.is_cuda)
            self._shapes[key] = shape
            return
        started = time.perf_counter()
        buf = self._write_ring.acquire(lambda: self._error is not None or self._closing)
        self.stats.put_wait_s += time.perf_counter() - started
        self._raise_if_failed()
        if buf is None:
            raise RuntimeError("the activation store was closed during a put")
        # Only CUDA has events protecting the writer from a pending host transfer.
        self._typed(buf, shape).copy_(tensor, non_blocking=self.is_cuda)
        event = self._record()
        with self._cond:
            self._written.discard(key)
            self._shapes[key] = shape
            self._step_keys.add(key)
        self._writes.put((key, buf, event, self._io_bytes(shape)))

    def _write_loop(self) -> None:
        try:
            while True:
                item = self._writes.get()
                if item is None:
                    return
                key, buf, event, nbytes = item
                if event is not None:
                    try:
                        event.synchronize()
                    except BaseException:
                        self._write_ring.release(buf, event)
                        raise
                self._spill.write(self._offset(key), buf[:nbytes])
                with self._cond:
                    self.stats.bytes_written += nbytes
                    self._written.add(key)
                    self._cond.notify_all()
                self._write_ring.release(buf, None)
        except BaseException as exc:  # noqa: BLE001 — surfaced on the next put/get
            self._fail(exc)
        finally:
            self._worker_done[0].set()

    # -- the backward --------------------------------------------------------
    def _request(self, key: Key) -> None:
        with self._cond:
            if key in self._requested:
                return
            self._requested.add(key)
        self._reads.put(key)

    def begin_backward(self) -> None:
        """Queue the read of every spilled chunk put this step, in the order the backward
        consumes them. A shorter window than ``k`` puts fewer micro-batches; a chunk it did
        not put is not read (it may never have been written)."""
        if not self.spill_layers:
            return
        with self._cond:
            put = set(self._step_keys)
        for layer in range(self.spill_layers - 1, -1, -1):
            for mb in range(self.k):
                if (layer, mb) in put:
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
                    nbytes = self._io_bytes(self._shapes.get(key, self.mb_shapes[key[1]]))
                buf = self._read_ring.acquire(lambda: self._closing or self._error is not None)
                if buf is None:
                    return
                self._spill.read(self._offset(key), buf[:nbytes])
                with self._cond:
                    self.stats.bytes_read += nbytes
                    self._ready[key] = buf
                    self._cond.notify_all()
        except BaseException as exc:  # noqa: BLE001 — surfaced on the next put/get
            self._fail(exc)
        finally:
            self._worker_done[1].set()

    def get(self, layer: int, mb: int) -> Any:
        """The stored activation as a NEW tensor on the store's device, in its put shape."""
        self._raise_if_failed()
        key = (layer, mb)
        shape = self._shapes.get(key, self.mb_shapes[mb])
        if not self.is_spilled(layer):
            return self._view(self._resident[key], shape).to(
                self.device, non_blocking=self.is_cuda, copy=True
            )
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
        out = self._typed(buf, shape).to(self.device, non_blocking=self.is_cuda, copy=True)
        self._read_ring.release(buf, self._record())
        return out

    def end_step(self) -> None:
        """Every spilled chunk requested this step must have been consumed."""
        self._raise_if_failed()
        if not self.spill_layers:
            return
        with self._cond:
            pending = sorted(self._requested | set(self._ready))
            self._step_keys.clear()
        if pending:
            raise RuntimeError(
                f"{len(pending)} spilled activation chunk(s) were not consumed this step, "
                f"first {pending[:4]}: the schedule and the store disagree"
            )

    def close(self) -> None:
        """Stop I/O and release every host allocation, retaining only store metadata."""
        if self._closed:
            return
        self._closing = True
        if self._writes is not None:
            self._writes.put(None)
        if self._reads is not None:
            self._reads.put(None)
        if self._cond is not None:
            with self._cond:
                self._cond.notify_all()
        for thread, done in zip(self._threads, self._worker_done):
            # An interrupted CPython join may report stopped before the target exits.
            if thread.ident is not None:
                thread.join()
                done.wait()
        self._threads.clear()
        self._worker_done.clear()
        try:
            if self.is_cuda and self._resident:
                import torch

                # Resident copies have no ring event and may use another CUDA stream.
                torch.cuda.synchronize(self.device)
            if self._writes is not None:
                while True:
                    try:
                        item = self._writes.get_nowait()
                    except queue.Empty:
                        break
                    if item is not None and item[2] is not None:
                        try:
                            item[2].synchronize()
                        except BaseException:
                            self._writes.put(item)
                            raise
            for ring in (self._write_ring, self._read_ring):
                if ring is not None:
                    ring.close()
        finally:
            if self._spill is not None:
                try:
                    self._spill.close(remove=True)
                finally:
                    self.stats.spill_left_behind = self._spill.left_behind
                self._spill = None
        # A failed synchronization leaves its source queued for a later close retry.
        self._write_ring = self._read_ring = None
        self._writes = self._reads = None
        self._ready.clear()
        self._resident.clear()
        self._shapes.clear()
        self._written.clear()
        self._requested.clear()
        self._step_keys.clear()
        # Chained backend exceptions can retain the transfer buffer in several frames.
        errors = [self._error]
        seen: Set[int] = set()
        while errors:
            error = errors.pop()
            if error is None or id(error) in seen:
                continue
            seen.add(id(error))
            error.__traceback__ = None
            errors.extend((error.__cause__, error.__context__))
        self._closed = True
