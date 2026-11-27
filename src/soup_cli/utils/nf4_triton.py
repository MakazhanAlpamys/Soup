# SPDX-License-Identifier: Apache-2.0
#
# Adapted from the Soup Kernel Challenge entry of AG TEAM (Apache-2.0),
# github.com/BAITC-Hacks/hack-2227fe74-ag-team, solution/nf4.py.
# ruff: noqa: N803, N806
"""Optional single-launch NF4 decoding for resident bitsandbytes 4-bit weights."""

from __future__ import annotations

import os
import sys
from contextlib import ExitStack, contextmanager, nullcontext
from functools import cache, wraps
from math import prod
from threading import Lock

ENV = "SOUP_NF4_TRITON"
_TILE = 2048  # packed bytes per program
_BLOCK_SIZES = (32, 64, 128, 256, 512, 1024, 2048, 4096)
_patch_lock = Lock()
_patch_users = 0
_restore_hooks = ExitStack()
tl = None  # Set by _kernel. Triton's interpreter resolves language symbols as globals.


def _nf4_kernel(
    packed_ptr, scale_ptr, scale2_ptr, code2_ptr, offset_ptr, code_ptr, out_ptr, n_bytes,
    NESTED: tl.constexpr, BLOCKSIZE: tl.constexpr, BLOCKSIZE2: tl.constexpr,
    TILE: tl.constexpr, INTERPRET: tl.constexpr,
):
    at = tl.program_id(0).to(tl.int64) * TILE + tl.arange(0, TILE)
    mask = at < n_bytes
    byte = tl.load(packed_ptr + at, mask=mask, other=0).to(tl.int32)
    block = at // (BLOCKSIZE // 2)
    if NESTED:
        index = tl.load(scale_ptr + block, mask=mask, other=0).to(tl.int32)
        code = tl.load(code2_ptr + index, mask=mask, other=0.0)
        scale2 = tl.load(scale2_ptr + block // BLOCKSIZE2, mask=mask, other=0.0)
        if INTERPRET:
            product = code * scale2
        else:
            # Round the product separately, even if the compiler enables FP fusion.
            product = tl.inline_asm_elementwise(
                "mul.rn.f32 $0, $1, $2;", "=r,r,r", [code, scale2],
                dtype=tl.float32, is_pure=True, pack=1,
            )
        scale = product + tl.load(offset_ptr)
    else:
        scale = tl.load(scale_ptr + block, mask=mask, other=0.0)
    high = tl.load(code_ptr + (byte >> 4), mask=mask, other=0.0) * scale
    low = tl.load(code_ptr + (byte & 15), mask=mask, other=0.0) * scale
    pairs = tl.join(high, low)
    if INTERPRET and out_ptr.dtype.element_ty == tl.bfloat16:
        # The interpreter truncates FP32 to BF16. Match hardware round-to-nearest-even.
        bits = pairs.to(tl.int32, bitcast=True)
        rounded = (bits + (((bits >> 16) & 1) + 0x7FFF)) >> 16
        pairs = rounded.to(tl.int16).to(tl.bfloat16, bitcast=True)
    else:
        pairs = pairs.to(out_ptr.dtype.element_ty)
    dst = tl.program_id(0).to(tl.int64) * (2 * TILE) + tl.arange(0, 2 * TILE)
    tl.store(out_ptr + dst, tl.reshape(pairs, [2 * TILE]), mask=dst < 2 * n_bytes)


@cache
def _kernel():
    global tl
    import triton
    from triton import language as tl

    return triton.jit(_nf4_kernel)


@cache
def _nf4_code(device):
    from bitsandbytes.functional import get_4bit_type

    # bitsandbytes' NF4 decoders select the canonical table by quant_type.
    return get_4bit_type("nf4", device=device).float().contiguous()


def _interpreted():
    return os.environ.get("TRITON_INTERPRET") == "1"


@cache
def _cuda_supported(device):
    import torch
    import triton

    if not torch.version.cuda or torch.cuda.get_device_capability(device) < (8, 0):
        return False
    try:
        with torch.cuda.device(device):
            return triton.runtime.driver.active.get_current_target().backend == "cuda"
    except (AttributeError, OSError, RuntimeError):
        return False


def _launch(packed, out, scale, blocksize, nested):
    import torch
    import triton

    scale2, code2, offset, blocksize2 = nested or (scale, scale, scale, 1)
    n_bytes = packed.numel()
    device_scope = torch.cuda.device(packed.device) if packed.is_cuda else nullcontext()
    with device_scope:
        _kernel()[(triton.cdiv(n_bytes, _TILE),)](
            packed, scale, scale2, code2, offset, _nf4_code(packed.device), out, n_bytes,
            NESTED=nested is not None, BLOCKSIZE=blocksize, BLOCKSIZE2=blocksize2,
            TILE=_TILE, INTERPRET=_interpreted(), num_warps=4, enable_fp_fusion=False,
        )
    return out


def _tensor_matches(value, device, dtype, count):
    import torch

    return (
        isinstance(value, torch.Tensor)
        and value.device == device
        and value.dtype == dtype
        and value.layout == torch.strided
        and value.is_contiguous()
        and value.numel() == count
        and not value.requires_grad
    )


def _decode(packed, shape, dtype, blocksize, scale, nested=None, out=None):
    """Return the dense weight, or None if the call belongs to stock bitsandbytes."""
    import torch
    import triton

    # bitsandbytes builds its lookup table in torch's current default dtype.
    if torch.compiler.is_compiling() or torch.get_default_dtype() not in (
        torch.float32, torch.float64,
    ):
        return None
    device = packed.device
    if device.type == "cpu":
        if not _interpreted():
            return None
    elif not packed.is_cuda or _interpreted() or not _cuda_supported(device):
        return None
    if shape is None or len(shape) != 2 or min(shape) <= 0 or blocksize not in _BLOCK_SIZES:
        return None
    count = prod(shape)
    if (
        dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or count % 2
        or packed.ndim not in (1, 2)
        or not _tensor_matches(packed, device, torch.uint8, count // 2)
    ):
        return None
    blocks = (count + blocksize - 1) // blocksize
    if not _tensor_matches(scale, device, torch.uint8 if nested else torch.float32, blocks):
        return None
    if nested:
        scale2, code2, offset, blocksize2 = nested
        if (
            blocksize2 not in _BLOCK_SIZES
            or not _tensor_matches(scale2, device, torch.float32,
                                   (blocks + blocksize2 - 1) // blocksize2)
            or not _tensor_matches(code2, device, torch.float32, 256)
            or not _tensor_matches(offset, device, torch.float32, 1)
            or offset.ndim > 1
        ):
            return None
    if out is not None:
        if not _tensor_matches(out, device, dtype, count) or out.shape != shape:
            return None
        inputs = (packed, scale, *nested[:3]) if nested else (packed, scale)
        if any(out.untyped_storage().data_ptr() == t.untyped_storage().data_ptr() for t in inputs):
            return None
    dense = out if out is not None else torch.empty(tuple(shape), device=device, dtype=dtype)
    try:
        _launch(packed, dense, scale, blocksize, nested)
    except triton.TritonError:
        return None
    if out is not None:
        torch.autograd.graph.increment_version(out)
    return dense


def _wrap_dequantize(original):
    @wraps(original)
    def dequantize(A, quant_state=None, absmax=None, out=None, blocksize=None, quant_type="fp4"):
        state = quant_state
        if state is not None and state.quant_type == "nf4":
            nested = None
            if state.nested:
                import torch

                second = state.state2
                if second is not None and second.dtype == torch.float32 and not second.nested:
                    nested = (second.absmax, second.code, state.offset, second.blocksize)
            if not state.nested or nested is not None:
                result = _decode(
                    A, state.shape, state.dtype, state.blocksize, state.absmax, nested, out,
                )
                if result is not None:
                    # bitsandbytes also accepts a row of packed bytes and returns a transpose.
                    return result.t() if A.shape[0] == 1 else result
        return original(A, quant_state, absmax, out, blocksize, quant_type)

    return dequantize


def _wrap_linear(original):
    @wraps(original)
    def dequant_linear(
        A, B, shapeB, absmax, blocksize, quant_type, bias=None,
        absmax_8bit=None, absmax_code=None, absmax_offset=None,
    ):
        if quant_type == "nf4" and A.device == B.device:
            if absmax_8bit is None:
                result = _decode(B, shapeB, A.dtype, blocksize, absmax)
            else:
                # bitsandbytes' CUDA linear fallback fixes the second block size at 256.
                nested = (absmax, absmax_code, absmax_offset, 256)
                result = _decode(B, shapeB, A.dtype, blocksize, absmax_8bit, nested)
            if result is not None:
                from torch.nn.functional import linear

                return linear(A, result, bias)
        return original(
            A, B, shapeB, absmax, blocksize, quant_type, bias,
            absmax_8bit=absmax_8bit, absmax_code=absmax_code, absmax_offset=absmax_offset,
        )

    return dequant_linear


@contextmanager
def patch_from_env(device=None):
    """Temporarily patch both bitsandbytes entry points when explicitly enabled."""
    global _patch_users
    if os.environ.get(ENV) != "1":
        yield False
        return
    try:
        import triton  # noqa: F401
        from bitsandbytes import functional
    except (ImportError, OSError):
        yield False
        return
    if device is not None:
        import torch

        device = torch.device(device)
        if (
            device.type != "cuda"
            or _interpreted()
            or not torch.cuda.is_available()
            or not _cuda_supported(device)
        ):
            yield False
            return

    with _patch_lock:
        if _patch_users == 0:
            original = functional.dequantize_4bit
            _restore_hooks.callback(setattr, functional, "dequantize_4bit", original)
            functional.dequantize_4bit = _wrap_dequantize(original)
            # bitsandbytes registers its CUDA backend during import when CUDA is available.
            ops = sys.modules.get("bitsandbytes.backends.cuda.ops")
            original_linear = getattr(ops, "_dequant_linear_fallback", None)
            if original_linear is not None:
                _restore_hooks.callback(setattr, ops, "_dequant_linear_fallback", original_linear)
                ops._dequant_linear_fallback = _wrap_linear(original_linear)
        _patch_users += 1
    try:
        yield True
    finally:
        with _patch_lock:
            _patch_users -= 1
            if not _patch_users:
                _restore_hooks.close()
