"""Grouped packed NF4 GEMM for CUDA BF16 inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_KERNEL = None


@dataclass(frozen=True)
class PackedNF4:
    """One frozen projection, in the existing bitsandbytes nibble layout."""

    packed: Any
    scales: Any
    code: Any
    shape: tuple[int, int]
    blocksize: int = 64
    nested_scales: Any = None
    nested_code: Any = None
    offset: Any = None
    nested_blocksize: int = 256

    @property
    def nested(self) -> bool:
        return self.nested_scales is not None


def _aligned(size: int) -> int:
    return (size + 511) // 512 * 512


def plan_fused_group(
    rows: list[int], shape: tuple[int, int], *, transpose: bool = False,
    native_small_rows: bool = False,
) -> dict[str, int]:
    """Bound device outputs/descriptor storage; packed weights are caller-owned."""
    if not isinstance(rows, list) or not 1 <= len(rows) <= 4:
        raise ValueError("one through four expert row counts required")
    if any(type(m) is not int or not 0 <= m <= 512 for m in rows):
        raise ValueError("expert rows must be integers from zero through 512")
    if (not isinstance(shape, tuple) or len(shape) != 2
            or any(type(d) is not int or not 1 <= d < 2**31 or d % 32 for d in shape)):
        raise ValueError("projection dimensions must be positive 32-aligned integers")
    if type(transpose) is not bool:
        raise ValueError("transpose must be boolean")
    if type(native_small_rows) is not bool:
        raise ValueError("native_small_rows must be boolean")
    if native_small_rows and shape not in ((2048, 2048), (2048, 1024)):
        raise ValueError("hybrid WMMA is restricted to OLMoE projection geometry")
    output_columns = shape[1] if transpose else shape[0]
    output = sum(_aligned(m * output_columns * 2) for m in rows)
    fused = sum(m > (128 if native_small_rows else 0) for m in rows)
    fallback = sum(0 < m <= 128 for m in rows) if native_small_rows else 0
    descriptors = _aligned(9 * fused * 8) if fused else 0
    dense = _aligned(shape[0] * shape[1] * 2) if fallback else 0
    return {"output_bytes": output, "descriptor_bytes": descriptors,
            "call_bytes": output + descriptors + dense, "dense_weight_bytes": dense,
            "fused_experts": fused, "fallback_experts": fallback,
            "max_rows": max(rows), "groups": len(rows)}


def validate_fused_group(
    inputs: list, weights: list[PackedNF4], *, transpose: bool = False,
    call_budget_bytes: int, backend: str = "triton",
) -> dict[str, int]:
    """Validate all descriptors before output allocation or device launch."""
    import torch

    if type(call_budget_bytes) is not int or call_budget_bytes < 1:
        raise ValueError("positive integer call budget required")
    if backend not in ("triton", "wmma", "hybrid_wmma"):
        raise ValueError("unknown fused expert backend")
    if (not isinstance(inputs, list) or not isinstance(weights, list)
            or not 1 <= len(inputs) <= 4 or len(inputs) != len(weights)
            or any(not isinstance(w, PackedNF4) for w in weights)):
        raise ValueError("one through four matching input/packed-weight pairs required")
    if any(not isinstance(x, torch.Tensor) or x.ndim != 2 for x in inputs):
        raise ValueError("two-dimensional tensor inputs required")
    first = weights[0]
    if (type(first.blocksize) is not int or not 16 <= first.blocksize <= 4096
            or first.blocksize & (first.blocksize - 1)):
        raise ValueError("invalid NF4 blocksize")
    plan = plan_fused_group([x.shape[0] for x in inputs], first.shape, transpose=transpose,
                            native_small_rows=backend == "hybrid_wmma")
    if plan["fallback_experts"]:
        # Conservative bound for both decoded and offset-adjusted absmax arrays.
        blocks = (first.shape[0] * first.shape[1] + first.blocksize - 1) // first.blocksize
        scratch = 2 * _aligned(blocks * 4) if first.nested else 0
        plan["call_bytes"] += scratch
    if plan["call_bytes"] > call_budget_bytes:
        raise ValueError("fused group exceeds call budget before allocation")
    for w in weights:
        if ((w.shape, w.blocksize, w.nested, w.nested_blocksize)
                != (first.shape, first.blocksize, first.nested, first.nested_blocksize)):
            raise ValueError("group must share projection geometry and NF4 block metadata")
        if (type(w.blocksize) is not int or not 16 <= w.blocksize <= 4096
                or w.blocksize & (w.blocksize - 1)):
            raise ValueError("invalid NF4 blocksize")
        if (type(w.nested_blocksize) is not int or not 16 <= w.nested_blocksize <= 4096
                or w.nested_blocksize & (w.nested_blocksize - 1)):
            raise ValueError("invalid nested NF4 blocksize")
        n, k = w.shape
        blocks = (n * k + w.blocksize - 1) // w.blocksize
        specs = [(w.packed, (n * k // 2,), torch.uint8),
                 (w.scales, (blocks,), torch.uint8 if w.nested else torch.float32),
                 (w.code, (16,), torch.float32)]
        if w.nested:
            specs.extend([
                (w.nested_scales, ((blocks + w.nested_blocksize - 1)
                                   // w.nested_blocksize,), torch.float32),
                (w.nested_code, (256,), torch.float32), (w.offset, (1,), torch.float32),
            ])
        elif any(v is not None for v in (w.nested_code, w.offset)):
            raise ValueError("partial nested NF4 state")
        for tensor, shape, dtype in specs:
            if (not isinstance(tensor, torch.Tensor) or tensor.dtype != dtype
                    or tuple(tensor.shape) != shape or not tensor.is_contiguous()
                    or tensor.device != inputs[0].device or tensor.requires_grad):
                raise ValueError("invalid frozen NF4 tensor extent, dtype, device or layout")
    for x in inputs:
        if (x.dtype != torch.bfloat16 or x.device != inputs[0].device
                or x.shape[1] != first.shape[0 if transpose else 1]
                or not x.is_contiguous() or x.requires_grad):
            raise ValueError("contiguous frozen BF16 inputs with matching dimensions required")
    return plan


def _native_small_linear(x: Any, w: PackedNF4, transpose: bool, output: Any) -> None:
    """Declared small-row path, never reported as a fused expert execution."""
    import bitsandbytes.functional as functional
    import torch

    state2 = functional.QuantState(
        absmax=w.nested_scales, code=w.nested_code, blocksize=w.nested_blocksize,
        dtype=torch.float32,
    ) if w.nested else None
    state = functional.QuantState(
        absmax=w.scales, code=w.code, shape=w.shape, blocksize=w.blocksize,
        quant_type="nf4", dtype=torch.bfloat16, state2=state2,
        offset=w.offset.reshape(()) if w.nested else None,
    )
    dense = functional.dequantize_4bit(w.packed.reshape(-1, 1), state)
    torch.mm(x, dense if transpose else dense.t(), out=output)


def _kernel() -> Any:
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def grouped_packed_nf4(
        table, groups: tl.constexpr, n: tl.constexpr, k: tl.constexpr,
        quant_block: tl.constexpr, nested: tl.constexpr, nested_block: tl.constexpr,
        transpose: tl.constexpr, bm: tl.constexpr, bn: tl.constexpr, bk: tl.constexpr,
    ):
        expert = tl.program_id(2)
        m = tl.load(table + 8 * groups + expert)
        if tl.program_id(0) * bm < m:
            x = tl.load(table + expert).to(tl.pointer_type(tl.bfloat16))
            packed = tl.load(table + groups + expert).to(tl.pointer_type(tl.uint8))
            code = tl.load(table + 3 * groups + expert).to(tl.pointer_type(tl.float32))
            y = tl.load(table + 7 * groups + expert).to(tl.pointer_type(tl.bfloat16))
            rows = (tl.program_id(0) * bm + tl.arange(0, bm)).to(tl.int64)
            columns = (tl.program_id(1) * bn + tl.arange(0, bn)).to(tl.int64)
            reduction = tl.arange(0, bk)
            reduce_size: tl.constexpr = n if transpose else k
            output_size: tl.constexpr = k if transpose else n
            # Native linear is expressed as W @ X.T in cuBLAS column-major
            # coordinates. Preserve that operand orientation in the MMA too.
            acc = tl.full((bn, bm), 0, tl.float32)
            for start in range(tl.cdiv(reduce_size, bk)):
                inner = (start * bk + reduction).to(tl.int64)
                a = tl.load(x + rows[:, None] * reduce_size + inner[None, :],
                            (rows[:, None] < m) & (inner[None, :] < reduce_size), other=0)
                if transpose:
                    pos = inner[:, None] * k + columns[None, :]
                else:
                    pos = columns[None, :] * k + inner[:, None]
                valid = (inner[:, None] < reduce_size) & (columns[None, :] < output_size)
                byte = tl.load(packed + pos // 2, valid, other=0).to(tl.int32)
                nibble = tl.where(pos % 2 == 0, byte >> 4, byte & 15)
                value = tl.load(code + nibble)
                index = pos // quant_block
                if nested:
                    scales = tl.load(table + 2 * groups + expert).to(tl.pointer_type(tl.uint8))
                    scales2 = tl.load(table + 4 * groups + expert).to(tl.pointer_type(tl.float32))
                    code2 = tl.load(table + 5 * groups + expert).to(tl.pointer_type(tl.float32))
                    offset = tl.load(table + 6 * groups + expert).to(tl.pointer_type(tl.float32))
                    encoded = tl.load(scales + index, valid, other=0).to(tl.int32)
                    magnitude = tl.load(code2 + encoded) * tl.load(
                        scales2 + index // nested_block, valid, other=0)
                    magnitude = magnitude + tl.load(offset)
                else:
                    scales = tl.load(table + 2 * groups + expert).to(tl.pointer_type(tl.float32))
                    magnitude = tl.load(scales + index, valid, other=0)
                # Preserve the original BF16 weight rounding before the MMA.
                b = tl.where(valid, (value * magnitude).to(tl.bfloat16), 0)
                acc = tl.dot(tl.trans(b), tl.trans(a), acc, input_precision="ieee")
            acc = tl.trans(acc)
            tl.store(y + rows[:, None] * output_size + columns[None, :], acc,
                     (rows[:, None] < m) & (columns[None, :] < output_size))

    _KERNEL = grouped_packed_nf4
    return _KERNEL


def fused_group_linear(
    inputs: list, weights: list[PackedNF4], *, transpose: bool = False,
    call_budget_bytes: int, backend: str = "triton",
) -> list:
    """Launch one fused kernel over a group. No autograd graph is constructed.

    ``transpose=True`` computes dX through the same packed matrix. A training
    wrapper must reload frozen weights inside a valid lease before calling it.
    Native-cuBLAS parity is qualified per source, device and finite recipe;
    it is not an unconditional API guarantee.
    """
    import torch

    plan = validate_fused_group(inputs, weights, transpose=transpose,
                                call_budget_bytes=call_budget_bytes, backend=backend)
    device = inputs[0].device
    if device.type != "cuda" or torch.cuda.get_device_capability(device)[0] < 8:
        raise ValueError("fused NF4 expert GEMM requires CUDA compute capability 8.0 or newer")
    if backend == "hybrid_wmma":
        from soup_cli.utils.expert_group_gemm import require_reference_policy

        require_reference_policy()
    for w in weights:
        check = getattr(w.packed, "_soup_stream_owner_check", None)
        if check is not None:
            check()
    columns = weights[0].shape[1 if transpose else 0]
    outputs = [torch.empty((x.shape[0], columns), dtype=x.dtype, device=device) for x in inputs]
    if not plan["max_rows"]:
        return outputs
    selected = list(range(len(inputs)))
    if backend == "hybrid_wmma":
        selected = [i for i, x in enumerate(inputs) if x.shape[0] > 128]
        with torch.no_grad():
            for i, x in enumerate(inputs):
                if 0 < x.shape[0] <= 128:
                    _native_small_linear(x, weights[i], transpose, outputs[i])
        if not selected:
            return outputs
    # Preserve output order; only known eligible experts enter the fused launch.
    inputs = [inputs[i] for i in selected]
    weights = [weights[i] for i in selected]
    fused_outputs = [outputs[i] for i in selected]
    pointers = [
        [x.data_ptr() for x in inputs], [w.packed.data_ptr() for w in weights],
        [w.scales.data_ptr() for w in weights], [w.code.data_ptr() for w in weights],
        [(w.nested_scales if w.nested else w.code).data_ptr() for w in weights],
        [(w.nested_code if w.nested else w.code).data_ptr() for w in weights],
        [(w.offset if w.nested else w.code).data_ptr() for w in weights],
        [y.data_ptr() for y in fused_outputs], [x.shape[0] for x in inputs],
    ]
    with torch.cuda.device(device), torch.no_grad():
        table = torch.tensor(pointers, dtype=torch.int64, device=device)
        n, k = weights[0].shape
        if backend in ("wmma", "hybrid_wmma"):
            from soup_cli.utils.expert_group_fused_wmma import launch_grouped

            launch_grouped(table, (n, k), groups=len(weights), max_rows=plan["max_rows"],
                           blocksize=weights[0].blocksize, nested=weights[0].nested,
                           nested_blocksize=weights[0].nested_blocksize, transpose=transpose)
            return outputs
        _kernel()[((plan["max_rows"] + 15) // 16, (columns + 31) // 32, len(weights))](
            table, len(weights), n, k, weights[0].blocksize, weights[0].nested,
            weights[0].nested_blocksize, transpose, 16, 32, 32,
            num_warps=4, enable_fp_fusion=False,
        )
    return outputs
