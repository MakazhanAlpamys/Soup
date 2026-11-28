"""Triton NF4 restore over a group of expert slots."""

from __future__ import annotations

from functools import lru_cache
from math import prod
from typing import Any


@lru_cache(maxsize=1)
def build_group_decoder() -> Any:
    import triton
    import triton.language as tl

    @triton.jit
    def decode(
        packed_ptrs, stats_ptrs, nested_ptrs, offset_ptrs, output_ptrs,
        code_ptr, nested_code_ptr,
        elements: tl.constexpr, quant_block: tl.constexpr,
        nested_block: tl.constexpr, nested: tl.constexpr, block: tl.constexpr,
    ):
        expert = tl.program_id(1)
        index = tl.program_id(0) * block + tl.arange(0, block)
        valid = index < elements
        packed = tl.load(packed_ptrs + expert).to(tl.pointer_type(tl.uint8))
        output = tl.load(output_ptrs + expert).to(tl.pointer_type(tl.bfloat16))
        value = tl.load(packed + index // 2, valid, other=0).to(tl.int32)
        nibble = tl.where(index % 2 == 0, value >> 4, value & 15)
        code = tl.load(code_ptr + nibble)
        stat_index = index // quant_block
        if nested:
            stats = tl.load(stats_ptrs + expert).to(tl.pointer_type(tl.uint8))
            nested_stats = tl.load(nested_ptrs + expert).to(tl.pointer_type(tl.float32))
            offset = tl.load(offset_ptrs + expert).to(tl.pointer_type(tl.float32))
            stat_code = tl.load(stats + stat_index, valid, other=0).to(tl.int32)
            multiplier = tl.load(nested_stats + stat_index // nested_block, valid, other=0)
            scale = tl.load(nested_code_ptr + stat_code) * multiplier
            scale = scale + tl.load(offset)
        else:
            stats = tl.load(stats_ptrs + expert).to(tl.pointer_type(tl.float32))
            scale = tl.load(stats + stat_index, valid, other=0)
        tl.store(output + index, (code * scale).to(tl.bfloat16), valid)

    return decode


def restore_group(slots: list[dict], shapes: dict, specs: dict, device: Any) -> None:
    """Decode a projection for all leased slots with one launch."""
    import torch

    first = slots[0]
    for part, spec in specs.items():
        names = [f"{part}:0", f"{part}:1"]
        names.extend((f"{part}:2", f"{part}:3") if spec.nested else (part, part))
        names.append(part)
        elements = prod(shapes[part])
        with torch.cuda.device(device):
            pointers = torch.tensor(
                [[slot[name].data_ptr() for slot in slots] for name in names],
                dtype=torch.int64, device=device,
            )
            build_group_decoder()[((elements + 1023) // 1024, len(slots))](
                *pointers.unbind(), first["code"], first.get("nested_code", first["code"]),
                elements, spec.blocksize, spec.nested_blocksize if spec.nested else 1,
                spec.nested, 1024, num_warps=4, enable_fp_fusion=False,
            )
