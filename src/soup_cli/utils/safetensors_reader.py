"""Parse a safetensors header without mapping the file.

``safe_open`` memory-maps, and on Windows a mapping charges commit for the
file's whole size (#926, measured: 48.99 GB mapped raised the commit charge
46.17 GB). The streaming disk tier holds one shard per decoder layer for a
whole run, so mapping is what we are avoiding, not an implementation detail.

The format is: 8 bytes little-endian header length, that many bytes of JSON
mapping name -> {dtype, shape, data_offsets}, then the tensor bytes. Offsets in
the JSON are relative to the end of the header; ``TensorRange`` stores them
ABSOLUTE so a caller can seek directly.

NO top-level torch or safetensors: this module is import-light by design.
"""

import io
import json
import logging
import math
import os
import struct
import sys
from dataclasses import dataclass
from typing import IO, TYPE_CHECKING, Dict, List, Tuple

if TYPE_CHECKING:  # pragma: no cover — annotations only; the module stays torch-free
    import torch

logger = logging.getLogger(__name__)

# A real header is a few hundred KB. The cap turns a corrupt length field into
# a named refusal instead of a 16 EB allocation attempt.
_MAX_HEADER_BYTES = 100_000_000

#: The unit direct I/O demands of a file offset, a request length and a buffer
#: address: a multiple of every sector size in use (512-byte and 4 KiB volumes
#: alike). Measured on the dev box (#974): an unaligned LENGTH is refused with
#: EINVAL; an unaligned address happened to be tolerated and is not relied on.
SECTOR_BYTES = 4096

# Win32 flags for ``open_direct``; ``_winapi`` exports GENERIC_READ and
# OPEN_EXISTING but not these.
_FILE_SHARE_READ = 0x00000001
_FILE_SHARE_WRITE = 0x00000002
_FILE_FLAG_NO_BUFFERING = 0x20000000
_FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000

# Safetensors' spelling -> Soup's, matching layer_stream_runtime._SAFETENSORS_DTYPES.
_DTYPES: Dict[str, str] = {
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
    "F64": "float64",
    "I8": "int8",
    "I16": "int16",
    "I32": "int32",
    "I64": "int64",
    "U8": "uint8",
    "BOOL": "bool",
}

_ITEMSIZE: Dict[str, int] = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
    "float64": 8,
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "uint8": 1,
    "bool": 1,
}


@dataclass(frozen=True)
class ShardIdentity:
    """Which FILE a set of byte ranges was read off.

    ``read_header`` closes the file, so between the parse and every later read
    there is no handle, no inode pin and no fingerprint — and ``read_into``
    checks how many bytes arrived, never that they came from the same file.
    A shard replaced in place by one of the SAME SIZE and a different layout
    therefore reads at stale offsets and trains on garbage with no error
    anywhere. ``layer_shard`` re-shards into the same directory with
    ``os.replace`` whenever the base's fingerprint moves, so the window is the
    whole run rather than a microsecond.

    Four fields because no one of them is sufficient: size alone misses a
    same-size rewrite, mtime alone misses a preserved timestamp, and
    ``(ino, dev)`` alone misses an in-place rewrite that keeps the inode.
    This type only REPORTS identity; the policy and the message belong to the
    caller that knows what the file is for.
    """

    size: int
    mtime_ns: int
    ino: int
    dev: int


def identity_of(handle: "object") -> ShardIdentity:
    """The identity of the file behind an OPEN handle (``os.fstat``, not a path).

    Taking it off the handle rather than the path is what makes it atomic with
    the bytes read through that handle: a path-based ``stat`` could describe a
    different file than the one the caller is about to read.
    """
    st = os.fstat(handle.fileno())
    return ShardIdentity(
        size=st.st_size, mtime_ns=st.st_mtime_ns, ino=st.st_ino, dev=st.st_dev
    )


@dataclass(frozen=True)
class TensorRange:
    """One tensor's identity and its ABSOLUTE byte range in the shard."""

    name: str
    dtype: str
    shape: Tuple[int, ...]
    start: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start


def read_header(path: str) -> Dict[str, TensorRange]:
    """Every tensor in ``path``, with absolute byte ranges. Never maps the file."""
    return read_header_with_identity(path)[0]


def read_header_with_identity(path: str) -> Tuple[Dict[str, TensorRange], ShardIdentity]:
    """``read_header``, plus the identity of the file the ranges were read off.

    The identity comes from an ``os.fstat`` on the SAME handle the header is
    read through, so a caller that keeps the ranges for the rest of a run can
    prove, at every later open, that it is still addressing the file it parsed.
    """
    with open(path, "rb") as handle:
        identity = identity_of(handle)
        size = identity.size
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"{path}: header is truncated (not even a length field)")
        (length,) = struct.unpack("<Q", raw_length)
        if length > _MAX_HEADER_BYTES:
            raise ValueError(
                f"{path}: header claims {length} bytes, above the "
                f"{_MAX_HEADER_BYTES} cap; refusing rather than allocating it"
            )
        body = handle.read(length)
    if len(body) != length:
        raise ValueError(f"{path}: header is truncated ({len(body)} of {length} bytes)")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: header is not valid JSON ({exc})") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: header is not a JSON object")

    base = 8 + length
    entries: Dict[str, TensorRange] = {}
    for name, meta in payload.items():
        if name == "__metadata__":
            continue
        if not isinstance(meta, dict):
            raise ValueError(f"{path}: entry {name!r} is not an object")
        raw_dtype = meta.get("dtype")
        if raw_dtype not in _DTYPES:
            raise ValueError(
                f"{path}: tensor {name!r} has unsupported dtype {raw_dtype!r}; "
                f"supported: {', '.join(sorted(_DTYPES))}"
            )
        dtype = _DTYPES[raw_dtype]
        # No default: ``dict.get(key, default)`` substitutes only when the KEY is
        # absent, so a stored ``null`` would slip a default past this check and
        # raise a bare TypeError at the comprehension. Mirrors data_offsets below.
        raw_shape = meta.get("shape")
        if not isinstance(raw_shape, (list, tuple)):
            raise ValueError(f"{path}: tensor {name!r} has no valid shape")
        shape = tuple(int(dim) for dim in raw_shape)
        if any(dim < 0 for dim in shape):
            raise ValueError(f"{path}: tensor {name!r} has a negative dimension")
        offsets = meta.get("data_offsets")
        if not isinstance(offsets, (list, tuple)) or len(offsets) != 2:
            raise ValueError(f"{path}: tensor {name!r} has no valid data_offsets")
        start, end = base + int(offsets[0]), base + int(offsets[1])
        if start < base:
            raise ValueError(
                f"{path}: tensor {name!r} has data_offsets starting before the "
                f"tensor-data region (byte {start}, header ends at {base})"
            )
        if start > end:
            raise ValueError(f"{path}: tensor {name!r} has a reversed byte range")
        if end > size:
            raise ValueError(
                f"{path}: tensor {name!r} ends at byte {end}, past the end of the "
                f"file ({size} bytes)"
            )
        expected = math.prod(shape) * _ITEMSIZE[dtype]
        if end - start != expected:
            raise ValueError(
                f"{path}: tensor {name!r} declares shape {shape} of {dtype} "
                f"({expected} bytes) but its byte range holds {end - start}"
            )
        entries[name] = TensorRange(
            name=name, dtype=dtype, shape=shape, start=start, end=end
        )
    return entries, identity


def read_into(handle: "object", entry: TensorRange, tensor: "object") -> None:
    """Fill ``tensor`` from ``handle`` at ``entry``'s byte range.

    ``tensor`` is a pre-allocated CPU tensor of the right shape and dtype —
    allocating here would defeat the pool. The read goes through a ``uint8``
    view of the SAME memory, because ``numpy()`` refuses bfloat16 and a
    ``frombuffer`` round trip would copy.

    The ``reshape(-1)`` comes BEFORE that view, and is not cosmetic: torch
    refuses a dtype-``view`` on a 0-dimensional tensor ("self.dim() cannot be 0
    to view Float as Byte"), and NF4 double quantisation stores a SCALAR
    ``::nested_offset`` per quantised weight — 7 of the 30 tensors in a real
    decoder-layer shard. Viewing first made every 4-bit layer unreadable
    through this path. Reshaping a contiguous tensor returns a view, so the
    uint8 flat still aliases the destination's storage and ``readinto`` writes
    through to it; the contiguity refusal above is what keeps that true, which
    is why it stays ahead of this line rather than being folded into it.

    If this raises, ``tensor`` holds undefined contents — a short read leaves
    whatever prefix bytes arrived and does not zero or roll back the rest —
    and must not be used until a later call fills it successfully.
    """
    import torch

    if not tensor.is_contiguous():
        raise ValueError(
            f"tensor {entry.name!r}: destination must be contiguous to be read into"
        )
    if tensor.device.type != "cpu":
        raise ValueError(
            f"tensor {entry.name!r}: destination must live on the CPU, "
            f"got {tensor.device}"
        )
    held = tensor.numel() * tensor.element_size()
    if held != entry.nbytes:
        raise ValueError(
            f"tensor {entry.name!r}: shard holds {entry.nbytes} bytes but the "
            f"destination holds {held}"
        )
    flat = tensor.reshape(-1).view(torch.uint8)
    handle.seek(entry.start)
    got = handle.readinto(memoryview(flat.numpy()))
    if got != entry.nbytes:
        raise OSError(
            f"tensor {entry.name!r}: short read, {got} of {entry.nbytes} bytes "
            f"at offset {entry.start}"
        )


class DirectIOUnavailableError(OSError):
    """This platform offers no way to read a file past the page cache."""


def aligned_span(lo: int, hi: int, *, sector: int = SECTOR_BYTES) -> Tuple[int, int]:
    """The sector-aligned superset of the byte range ``[lo, hi)``.

    A safetensors data section starts wherever its header ends (byte 2976 on
    the 70B-shaped store), so a direct read covers a little more than the
    tensors and the caller places them at ``start - aligned_lo``.
    """
    return lo - lo % sector, -(-hi // sector) * sector


def plan_ranges(
    lo: int, hi: int, parts: int, *, sector: int = SECTOR_BYTES
) -> List[Tuple[int, int]]:
    """Split the aligned span ``[lo, hi)`` into at most ``parts`` byte ranges.

    Contiguous, non-empty, every boundary on a sector — so each range can be
    one direct-I/O request into a sector-aligned slice of the same buffer. A
    span shorter than ``parts`` sectors yields fewer ranges rather than empty
    ones.
    """
    if isinstance(parts, bool) or int(parts) < 1:
        raise ValueError(f"parts must be a positive int; got {parts!r}")
    if lo % sector or hi % sector:
        raise ValueError(f"span [{lo}, {hi}) is not aligned to {sector}-byte sectors")
    if hi <= lo:
        raise ValueError(f"span [{lo}, {hi}) is empty")
    sectors = (hi - lo) // sector
    parts = min(int(parts), sectors)
    return [
        (lo + (sectors * part // parts) * sector, lo + (sectors * (part + 1) // parts) * sector)
        for part in range(parts)
    ]


def read_range_into(
    handle: IO[bytes], start: int, view: "torch.Tensor", expected: int
) -> None:
    """Read ``expected`` bytes at ``start`` into the FRONT of ``view``.

    ``view`` is a contiguous CPU ``uint8`` tensor at least ``expected`` long —
    usually longer: it is the sector-aligned slice of a staging region, and the
    last range of a layer runs past the end of the file, where the request
    returns short and the loop stops at ``expected`` rather than demanding the
    padding. A ``readinto`` that returns FEWER bytes than asked is asked again:
    ``io.FileIO`` is one syscall, and POSIX lets it return short with more still
    to come (Linux documents exactly that for ``O_DIRECT``). Only a ZERO return
    is the end of the data, and one that comes before ``expected`` is a file
    shorter than its header said, refused as a short read. Every request keeps
    the aligned length ``view`` has from ``done`` on, which a direct-I/O handle
    requires; a short return that leaves ``done`` off a sector boundary makes
    the next request one the handle refuses — an ``OSError`` either way, never
    a silent partial fill.

    If this raises, ``view`` holds undefined contents.
    """
    import torch

    if expected < 0:
        raise ValueError(f"expected byte count must not be negative; got {expected}")
    if view.dtype != torch.uint8 or view.device.type != "cpu" or not view.is_contiguous():
        raise ValueError("destination must be a contiguous CPU uint8 tensor")
    if view.numel() < expected:
        raise ValueError(
            f"destination holds {view.numel()} bytes but {expected} were expected "
            f"at offset {start}"
        )
    handle.seek(start)
    buffer = memoryview(view.numpy())
    done = 0
    while done < expected:
        got = handle.readinto(buffer[done:])
        if not got:
            raise OSError(f"short read, {done} of {expected} bytes at offset {start}")
        done += got


def _owning_fileio(fd: int) -> io.FileIO:
    """Wrap ``fd`` so the descriptor is closed if the wrapper itself cannot be built."""
    try:
        return io.FileIO(fd, "rb", closefd=True)
    except BaseException:
        os.close(fd)
        raise


def open_direct(path: str) -> io.FileIO:
    """Open ``path`` for reading with the page cache bypassed.

    Measured on the dev box (#974, Samsung PM9B1 NVMe, Windows 11): buffered
    reads of a store larger than RAM top out at 1.2-2.9 GB/s cold whatever the
    request pattern, unbuffered reads of the same layers into a 4 KiB-aligned
    buffer reach 3.5-5.65 GB/s. Windows: ``FILE_FLAG_NO_BUFFERING`` on a
    handle wrapped as a CRT descriptor; Linux: ``O_DIRECT``; macOS:
    ``F_NOCACHE``. The caller owes the alignment (:data:`SECTOR_BYTES`) of
    every offset, length and address it reads with. Raises ``OSError`` where
    the filesystem refuses (tmpfs has no ``O_DIRECT``) and
    :class:`DirectIOUnavailableError` where the platform has no such mode at
    all; the reader falls back to a buffered ``open`` in both cases.
    """
    if sys.platform == "win32":
        import _winapi
        import msvcrt

        try:
            handle = _winapi.CreateFile(
                path,
                _winapi.GENERIC_READ,
                _FILE_SHARE_READ | _FILE_SHARE_WRITE,
                _winapi.NULL,
                _winapi.OPEN_EXISTING,
                _FILE_FLAG_NO_BUFFERING | _FILE_FLAG_SEQUENTIAL_SCAN,
                _winapi.NULL,
            )
        except OSError as exc:
            # ``_winapi`` raises without the path; a missing shard must name
            # itself the way ``open`` does, or the operator gets "[WinError 2]".
            raise type(exc)(
                exc.errno, exc.strerror, path, getattr(exc, "winerror", None)
            ) from None
        try:
            fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
        except OSError:
            _winapi.CloseHandle(handle)
            raise
        return _owning_fileio(fd)
    cloexec = getattr(os, "O_CLOEXEC", 0)
    direct = getattr(os, "O_DIRECT", None)
    if direct is not None:
        return _owning_fileio(os.open(path, os.O_RDONLY | direct | cloexec))
    if sys.platform == "darwin":
        import fcntl

        nocache = getattr(fcntl, "F_NOCACHE", None)
        if nocache is not None:
            fd = os.open(path, os.O_RDONLY | cloexec)
            try:
                fcntl.fcntl(fd, nocache, 1)
            except OSError:
                os.close(fd)
                raise
            return _owning_fileio(fd)
    raise DirectIOUnavailableError(f"no direct I/O mode on {sys.platform}")
