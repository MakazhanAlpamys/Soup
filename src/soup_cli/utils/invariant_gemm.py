"""Batch-invariant GEMM for the routed experts of streamed MoE models.

Native GEMM can choose its kernel and reduction order based on the number of rows.
Route-preserving MBR joins rows from several micro-batches into one expert call.
``linear`` uses a fixed tile configuration and a sequential K loop with one fp32
accumulator per output tile. The input gradient uses the same kernel. The expert
weights are frozen. The approach follows Thinking Machines' batch-invariant matmul
(https://github.com/thinking-machines-lab/batch_invariant_ops, MIT) with small tiles.

The kernel runs on CUDA with Triton. BF16 requires compute capability 8.0+.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Tuple

EXPERT_GEMMS = ("native", "batch_invariant", "segmented")


@dataclass(frozen=True)
class TileConfig:
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int


TILE = TileConfig(block_m=32, block_n=64, block_k=64, num_warps=4, num_stages=3)


@lru_cache(maxsize=None)
def _kernel() -> Any:
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=["m"])
    def invariant_gemm_kernel(
        a_ptr, b_ptr, c_ptr, m, n, k,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
    ):
        # Offsets in int64: row * stride overflows int32 well before the tensors do.
        rows = (tl.program_id(0) * block_m + tl.arange(0, block_m)).to(tl.int64)
        cols = (tl.program_id(1) * block_n + tl.arange(0, block_n)).to(tl.int64)
        depth = tl.arange(0, block_k).to(tl.int64)
        a_ptrs = a_ptr + rows[:, None] * stride_am + depth[None, :] * stride_ak
        b_ptrs = b_ptr + depth[:, None] * stride_bk + cols[None, :] * stride_bn
        acc = tl.zeros((block_m, block_n), dtype=tl.float32)
        for start in range(0, k, block_k):
            left = k - start
            a = tl.load(a_ptrs, mask=(rows[:, None] < m) & (depth[None, :] < left), other=0.0)
            b = tl.load(b_ptrs, mask=(depth[:, None] < left) & (cols[None, :] < n), other=0.0)
            acc = tl.dot(a, b, acc, input_precision="ieee")
            a_ptrs += block_k * stride_ak
            b_ptrs += block_k * stride_bk
        c_ptrs = c_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn
        mask = (rows[:, None] < m) & (cols[None, :] < n)
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)

    return invariant_gemm_kernel


def check_supported(device: Any, dtype: Any) -> None:
    """Refuse a device/dtype pair the kernel cannot run (raises ``ValueError``)."""
    import torch

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(
            "training.stream_expert_gemm='batch_invariant' needs a CUDA device (Triton GEMM), "
            f"got {device.type}"
        )
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"training.stream_expert_gemm='batch_invariant' does not support {dtype}")
    if dtype == torch.bfloat16 and torch.cuda.get_device_capability(device) < (8, 0):
        raise ValueError(
            "training.stream_expert_gemm='batch_invariant' with bfloat16 needs "
            "compute capability 8.0+"
        )
    try:
        import triton  # noqa: F401
    except ImportError as exc:
        raise ValueError(
            "training.stream_expert_gemm='batch_invariant' needs Triton installed"
        ) from exc


def matmul(a: Any, b: Any) -> Any:
    """``a @ b`` for 2-D ``a [M, K]`` and any strided ``b [K, N]``, M-invariant per row."""
    import torch

    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError(f"invariant matmul needs [M, K] @ [K, N], got {a.shape} @ {b.shape}")
    if a.dtype != b.dtype or a.device != b.device:
        raise ValueError("invariant matmul operands must share dtype and device")
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    if m == 0 or n == 0:
        return out
    if k == 0:
        return out.zero_()
    grid = (-(-m // TILE.block_m), -(-n // TILE.block_n))
    _kernel()[grid](
        a, b, out, m, n, k, *a.stride(), *b.stride(), *out.stride(),
        block_m=TILE.block_m, block_n=TILE.block_n, block_k=TILE.block_k,
        num_warps=TILE.num_warps, num_stages=TILE.num_stages,
    )
    return out


@lru_cache(maxsize=None)
def _linear_function() -> Any:
    import torch

    class InvariantLinear(torch.autograd.Function):
        """``x @ w.T`` with a gradient for ``x`` only; ``w`` is a frozen expert weight."""

        @staticmethod
        def forward(ctx: Any, x: Any, weight: Any) -> Any:
            ctx.save_for_backward(weight)
            flat = x.reshape(-1, x.shape[-1])
            return matmul(flat, weight.t()).reshape(*x.shape[:-1], weight.shape[0])

        @staticmethod
        def backward(ctx: Any, grad_out: Any) -> Tuple[Any, None]:
            if ctx.needs_input_grad[1]:
                raise RuntimeError("batch-invariant expert GEMM has no weight gradient")
            if not ctx.needs_input_grad[0]:
                return None, None
            (weight,) = ctx.saved_tensors
            flat = grad_out.reshape(-1, grad_out.shape[-1]).to(weight.dtype)
            grad = matmul(flat, weight)
            return grad.reshape(*grad_out.shape[:-1], weight.shape[1]), None

    return InvariantLinear


def linear(x: Any, weight: Any) -> Any:
    """``F.linear(x, weight)`` without bias, batch-invariant; differentiable in ``x`` only.

    Under autocast both operands are cast to the autocast dtype first, as ``F.linear`` does,
    so the result dtype matches the native path.
    """
    import torch

    device_type = x.device.type
    if torch.is_autocast_enabled(device_type):
        dtype = torch.get_autocast_dtype(device_type)
        x, weight = x.to(dtype), weight.to(dtype)
    elif x.dtype != weight.dtype:
        raise ValueError(f"expert input {x.dtype} and weight {weight.dtype} differ")
    with torch.autocast(device_type, enabled=False):
        return _linear_function().apply(x, weight)


#: Columns per step of the per-row reduction in ``scale_rows``' weight gradient.
ROW_BLOCK = 1024


@lru_cache(maxsize=None)
def _row_dot_kernel() -> Any:
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=["m"])
    def row_dot_kernel(
        a_ptr, b_ptr, out_ptr, m, n, stride_am, stride_an, stride_bm, stride_bn,
        block: tl.constexpr,
    ):
        # One program per row, columns in fixed blocks in order: no dependence on M.
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, block).to(tl.int64)
        acc = tl.zeros((block,), dtype=tl.float32)
        for start in range(0, n, block):
            mask = cols < n - start
            a = tl.load(a_ptr + row * stride_am + (start + cols) * stride_an, mask=mask, other=0.0)
            b = tl.load(b_ptr + row * stride_bm + (start + cols) * stride_bn, mask=mask, other=0.0)
            # Native autograd forms ``grad * values`` in the operands' dtype, then sums.
            acc += (a * b).to(tl.float32)
        tl.store(out_ptr + row, tl.sum(acc, axis=0).to(out_ptr.dtype.element_ty))

    return row_dot_kernel


def row_dot(a: Any, b: Any, out_dtype: Any) -> Any:
    """``(a * b).sum(-1, keepdim=True)`` for ``[M, N]`` operands, M-invariant per row."""
    import torch

    if a.ndim != 2 or a.shape != b.shape or a.device != b.device:
        raise ValueError(f"row_dot needs two [M, N] tensors, got {a.shape} and {b.shape}")
    m, n = a.shape
    out = torch.empty((m,), dtype=out_dtype, device=a.device)
    if m == 0:
        return out.view(m, 1)
    if n == 0:
        return out.zero_().view(m, 1)
    _row_dot_kernel()[(m,)](
        a, b, out, m, n, *a.stride(), *b.stride(), block=ROW_BLOCK, num_warps=4,
    )
    return out.view(m, 1)


@lru_cache(maxsize=None)
def _scale_function() -> Any:
    import torch

    class ScaleRows(torch.autograd.Function):
        """``values * weights`` for ``values [M, N]`` and ``weights [M, 1]``.

        Native autograd reduces ``d(weights)`` over N with a CUDA reduction whose tree
        depends on M; here that reduction is ``row_dot``, one fixed order per row.
        """

        @staticmethod
        def forward(ctx: Any, values: Any, weights: Any) -> Any:
            ctx.save_for_backward(values, weights)
            return values * weights

        @staticmethod
        def backward(ctx: Any, grad_out: Any) -> Tuple[Any, Any]:
            values, weights = ctx.saved_tensors
            grad_values = grad_out * weights if ctx.needs_input_grad[0] else None
            grad_weights = None
            if ctx.needs_input_grad[1]:
                # Native mul promotes, then sums in the promoted dtype and casts back.
                dtype = torch.promote_types(grad_out.dtype, values.dtype)
                grad_weights = row_dot(
                    grad_out.to(dtype), values.to(dtype), weights.dtype
                )
            return grad_values, grad_weights

    return ScaleRows


def scale_rows(values: Any, weights: Any) -> Any:
    """``values * weights`` (routing weight per expert row) with an M-invariant weight VJP."""
    if values.ndim != 2 or weights.shape != (values.shape[0], 1):
        raise ValueError(f"scale_rows needs [M, N] * [M, 1], got {values.shape} {weights.shape}")
    if not values.is_cuda:
        return values * weights
    return _scale_function().apply(values, weights)
