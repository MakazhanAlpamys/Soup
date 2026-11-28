"""Conservative cache budget for a new, single-disk per-expert NF4 store.

Reads only bounded JSON headers, including FP8, without memory-mapping weights.
The budget includes an additional largest layer for atomic writes and reserves
space for headers, the index, code tables and a possible tied vocabulary copy.
"""

import json
import math
import os
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

_SIZES = {"F8_E4M3": 1, "F8_E5M2": 1, "BF16": 2, "F16": 2, "F32": 4}
_EXPERT = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.\d+\.(gate|up|down)_proj\.weight$")
_LAYER = re.compile(r"model\.layers\.(\d+)\.")


def read_source_header(handle: Any, file_bytes: int) -> Tuple[bytes, Dict[str, Any]]:
    """Read a bounded header from a local or ranged read-only source handle."""
    raw = handle.read(8)
    if len(raw) != 8:
        raise ValueError("truncated safetensors header")
    length = int.from_bytes(raw, "little")
    if not 2 <= length <= 64 * 2**20:
        raise ValueError("safetensors header exceeds budget reader limit")
    if length > file_bytes - 8:
        raise ValueError("truncated safetensors header")
    data = handle.read(length)
    if len(data) != length:
        raise ValueError("truncated safetensors header")
    header = json.loads(data)
    if not isinstance(header, dict):
        raise ValueError("invalid safetensors header")
    return data, header


def new_expert_cache_budget(
    source_files: Sequence[str], *, quant_suffixes: Sequence[str] = (),
    n_layers: Optional[int] = None,
) -> int:
    """Budget bytes; refuse unsupported inputs rather than extrapolating a ratio.

    Valid for NF4 with blocksize 64 and no nested statistics. The writer checks
    codec sidecars against their declared shape and dtype before any write.
    Existing cache replacement and striping require a separate placement plan.
    Optional suffixes/layer count must match the actual writer configuration;
    without them all non-expert values retain the original FP32 upper bound.
    """
    if not source_files:
        raise ValueError("expert cache budget requires complete source files")
    def headers() -> Iterable[Tuple[int, int, Dict[str, Any]]]:
        for path in source_files:
            with open(path, "rb") as handle:
                size = os.fstat(handle.fileno()).st_size
                raw, header = read_source_header(handle, size)
            yield size, len(raw), header

    return expert_cache_budget_from_headers(
        headers(), quant_suffixes=quant_suffixes, n_layers=n_layers,
    )


def expert_cache_budget_from_headers(
    headers: Iterable[Tuple[int, int, Dict[str, Any]]], *,
    quant_suffixes: Sequence[str] = (), n_layers: Optional[int] = None,
) -> int:
    """Apply the same conservative bound to validated remote header records.

    NF4 dense tensors are budgeted only when explicitly named by the writer's
    quantization selection. Vocab/extras stay at FP32, and all source headers,
    including layers the model drops, still undergo shape/offset validation.
    """
    if n_layers is not None and (type(n_layers) is not int or n_layers <= 0):
        raise ValueError("budget layer count must be a positive integer")
    suffixes = set(quant_suffixes)
    if any(not isinstance(key, str) or not key.endswith(".weight") for key in suffixes):
        raise ValueError("budget quantization suffixes must name weight tensors")
    seen = set()
    layers = defaultdict(int)
    total = count = largest = expert_count = 0
    for file_bytes, length, header in headers:
        if (type(file_bytes) is not int or type(length) is not int
                or not 2 <= length <= min(64 * 2**20, file_bytes - 8)
                or not isinstance(header, dict)):
            raise ValueError("invalid safetensors header record")
        for key, value in header.items():
            if key == "__metadata__":
                continue
            if key in seen or len(key.encode("utf-8")) > 1024:
                raise ValueError("duplicate or oversized source tensor key")
            seen.add(key)
            shape, dtype = value["shape"], value["dtype"]
            if (dtype not in _SIZES or not isinstance(shape, list) or len(shape) > 8
                    or any(type(dim) is not int or dim < 0 for dim in shape)):
                raise ValueError("unsupported tensor shape or dtype for expert cache budget")
            elements = math.prod(shape)
            offsets = value["data_offsets"]
            if (len(offsets) != 2 or any(type(pos) is not int for pos in offsets)
                    or offsets[0] < 0 or offsets[1] - offsets[0] != elements * _SIZES[dtype]
                    or offsets[1] > file_bytes - length - 8):
                raise ValueError("source tensor offsets do not fit the file")
            count += 1
            layer = _LAYER.match(key)
            if layer and n_layers is not None and int(layer[1]) >= n_layers:
                continue
            if key.endswith("weight_scale_inv"):
                continue
            expert = _EXPERT.fullmatch(key)
            selected = layer and key[layer.end():] in suffixes
            if expert or selected:
                # Gate/up may be concatenated. Extra blocks bound that rounding,
                # and every source matrix gets its own metadata allowance below.
                size = (elements + 1) // 2 + 8 * ((elements + 63) // 64 + 1)
                expert_count += bool(expert)
            else:
                size = 4 * elements
                largest = max(largest, size)
            total += size
            if not expert:
                largest = max(largest, 4 * elements)
            if layer:
                layers[layer[1]] += size
    if not expert_count:
        raise ValueError("source contains no supported routed experts")
    # Four stored entries per source tensor, each with generous JSON/alignment
    # allowance, plus index/temp reserve and a possible duplicated tied head.
    metadata = 4 * count * 16384
    return total + max(layers.values(), default=0) + largest + metadata + 2**30
