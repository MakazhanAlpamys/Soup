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

import json
import logging
import math
import os
import struct
from dataclasses import dataclass
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# A real header is a few hundred KB. The cap turns a corrupt length field into
# a named refusal instead of a 16 EB allocation attempt.
_MAX_HEADER_BYTES = 100_000_000

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
