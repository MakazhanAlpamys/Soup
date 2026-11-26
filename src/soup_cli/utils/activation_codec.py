"""Per-token activation codecs shared by the activation store and its benchmarks.

The numerical operations and their order are those of the original codec.
The float32 scale is the row absmax, not an absmax/levels step. Dividing by absmax
before multiplying by levels preserves finite bf16 extremes and denormals.
Torch stays lazy so the byte arithmetic is usable by memory planners.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

IDENTITY = "bf16"
INT8 = "int8"
FP8 = "fp8_e4m3"
COARSE = "int4"
CANDIDATES = (INT8, FP8)
SCALE_BYTES = 4
SCALE_DTYPE = "float32"
BF16_BYTES = 2


@dataclass(frozen=True)
class Codec:
    """One codec. ``levels`` is the largest stored magnitude; 0 means identity."""

    name: str
    levels: int
    payload_dtype: Optional[str]
    packed: bool = False

    @property
    def identity(self) -> bool:
        return self.levels == 0


CODECS: Dict[str, Codec] = {
    IDENTITY: Codec(IDENTITY, 0, None),
    INT8: Codec(INT8, 127, "int8"),
    FP8: Codec(FP8, 448, "float8_e4m3fn"),
    COARSE: Codec(COARSE, 7, "uint8", packed=True),
}


class NonFiniteActivationError(ValueError):
    """A NaN or Inf makes a per-row absmax scale non-finite."""


def codec_of(name: str) -> Codec:
    try:
        return CODECS[name]
    except KeyError:
        raise ValueError(f"unknown codec {name!r}; choose from {sorted(CODECS)}") from None


def rows_of(shape: Tuple[int, ...]) -> int:
    """Tokens in a chunk: every dimension but the hidden one."""
    return math.prod(int(dim) for dim in shape[:-1])


def payload_shape(shape: Tuple[int, ...], codec: str) -> Tuple[int, ...]:
    spec = codec_of(codec)
    hidden = int(shape[-1])
    last = (hidden + 1) // 2 if spec.packed else hidden
    return tuple(int(dim) for dim in shape[:-1]) + (last,)


def payload_chunk_bytes(
    shape: Tuple[int, ...], codec: str, *, source_element_bytes: int = BF16_BYTES
) -> int:
    """Bytes of one payload, using the source element size for the identity."""
    spec = codec_of(codec)
    element = source_element_bytes if spec.identity else 1
    return math.prod(payload_shape(shape, codec)) * element


def scale_chunk_bytes(shape: Tuple[int, ...], codec: str) -> int:
    """One float32 per row, none for the identity."""
    return 0 if codec_of(codec).identity else rows_of(shape) * SCALE_BYTES


def compressed_chunk_bytes(
    shape: Tuple[int, ...], codec: str, *, source_element_bytes: int = BF16_BYTES
) -> int:
    """Logical payload plus scales, before any pinned allocator rounding."""
    return payload_chunk_bytes(
        shape, codec, source_element_bytes=source_element_bytes
    ) + scale_chunk_bytes(shape, codec)


def bytes_per_token_per_layer(
    hidden: int, codec: str, *, source_element_bytes: int = BF16_BYTES
) -> int:
    return compressed_chunk_bytes(
        (1, int(hidden)), codec, source_element_bytes=source_element_bytes
    )


def tokens_at_budget(budget_bytes: int, *, n_layers: int, hidden: int, codec: str) -> int:
    """Floor arithmetic before allocator rounding, not a measured throughput."""
    return int(budget_bytes) // (int(n_layers) * bytes_per_token_per_layer(hidden, codec))


def encode(x: Any, codec: str, *, check_finite: bool = True) -> Tuple[Any, Optional[Any]]:
    """Return device-local payload and float32 row absmax scales.

    The identity returns ``x`` and None. A lossy codec rejects non-finite rows
    with a host synchronization unless its caller opts into a deferred check.
    """
    import torch

    spec = codec_of(codec)
    if spec.identity:
        return x, None
    with torch.no_grad():
        values = x.float()
        absmax = values.abs().amax(dim=-1, keepdim=True)
        if check_finite and not bool(torch.isfinite(absmax).all()):
            bad = int((~torch.isfinite(absmax)).sum())
            raise NonFiniteActivationError(
                f"{codec}: {bad} of {absmax.numel()} rows hold a NaN or an Inf; a per-row "
                "absmax codec cannot encode them")
        safe = torch.where(absmax > 0, absmax, torch.ones_like(absmax))
        unit = values / safe
        if spec.name == FP8:
            payload = (unit * spec.levels).clamp_(-spec.levels, spec.levels).to(
                torch.float8_e4m3fn
            )
        else:
            levels = torch.round(unit * spec.levels).clamp_(-spec.levels, spec.levels)
            if spec.packed:
                payload = _pack_nibbles(levels)
            else:
                payload = levels.to(torch.int8)
        return payload, absmax.squeeze(-1)


def decode(payload: Any, scale: Optional[Any], codec: str, dtype: Any, hidden: int) -> Any:
    """Restore in ``dtype`` on the payload device, in the order ``encode`` inverts.

    Raises ``ValueError`` naming the field when ``hidden``, ``scale`` or the payload shape do
    not fit the codec, before anything is allocated.
    """
    import torch

    spec = codec_of(codec)
    if spec.identity:
        return payload
    if isinstance(hidden, bool) or not isinstance(hidden, int) or hidden <= 0:
        raise ValueError(f"hidden must be a positive integer; got {hidden!r}")
    if scale is None:
        raise ValueError(f"scale is required to decode the lossy codec {codec}; got None")
    width = int(payload.shape[-1]) if len(payload.shape) else 0
    expected = (hidden + 1) // 2 if spec.packed else hidden
    if width != expected:
        raise ValueError(
            f"hidden={hidden} needs a payload width of {expected} for {codec}; "
            f"payload shape is {tuple(payload.shape)}"
        )
    if tuple(scale.shape) != tuple(payload.shape[:-1]):
        raise ValueError(
            f"scale shape {tuple(scale.shape)} does not match the payload rows "
            f"{tuple(payload.shape[:-1])}"
        )
    with torch.no_grad():
        if spec.packed:
            levels = _unpack_nibbles(payload, hidden)
        else:
            levels = payload.float()
        return (levels / spec.levels * scale.unsqueeze(-1)).to(dtype)


def _pack_nibbles(levels: Any) -> Any:
    """Pack signed levels with the even index low, padding an odd width with level 0."""
    import torch

    biased = (levels + 8).to(torch.uint8)
    if biased.shape[-1] % 2:
        pad = torch.full(biased.shape[:-1] + (1,), 8, dtype=torch.uint8, device=biased.device)
        biased = torch.cat([biased, pad], dim=-1)
    return biased[..., 0::2] | (biased[..., 1::2] << 4)


def _unpack_nibbles(packed: Any, hidden: int) -> Any:
    import torch

    low = (packed & 0x0F).float() - 8
    high = (packed >> 4).float() - 8
    out = torch.stack([low, high], dim=-1).flatten(-2)
    return out[..., :hidden]


def round_trip(x: Any, codec: str) -> Any:
    payload, scale = encode(x, codec)
    return decode(payload, scale, codec, x.dtype, int(x.shape[-1]))


def store_compress(x: Any, codec: str) -> Tuple[Any, Any, Any]:
    """Encode without a host sync, returning a device-local finite flag as well.

    The identity codec stores the bytes unchanged, so it has no scale and nothing to refuse:
    it returns ``(x, None, True)`` with the flag as a device-local boolean scalar.
    """
    import torch

    payload, scale = encode(x, codec, check_finite=False)
    if scale is None:
        return payload, None, torch.ones((), dtype=torch.bool, device=payload.device)
    return payload, scale, torch.isfinite(scale).all()
