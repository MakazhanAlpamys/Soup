"""Optional Triton SwiGLU activation for grouped experts."""

from __future__ import annotations

from typing import Any


def plan_swiglu(rows: int, intermediate: int) -> dict:
    """Activation-only storage model; not the peak of a whole expert/model.

    Both paths retain the existing BF16 gate/up GEMM output. Eager SiLU then
    multiply creates two I-wide tensors; fusion creates only the final one.
    Allocations are rounded independently to PyTorch's 512-byte small bins.
    """
    if (type(rows) is not int or rows < 0 or type(intermediate) is not int
            or intermediate < 1):
        raise ValueError("nonnegative rows and positive intermediate dimension required")
    input_bytes = (rows * intermediate * 4 + 511) // 512 * 512
    output_bytes = (rows * intermediate * 2 + 511) // 512 * 512
    return {"input_bytes": input_bytes, "output_bytes": output_bytes,
            "eager_live_bytes": input_bytes + 2 * output_bytes,
            "fused_live_bytes": input_bytes + output_bytes,
            "removed_temporary_bytes": output_bytes,
            "scope": "activation-only allocation model; CUDA peak must be measured"}


_KERNEL = None


def _kernel() -> Any:
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def activate(source, output, elements: tl.constexpr, width: tl.constexpr,
                 block: tl.constexpr):
        index = tl.program_id(0) * block + tl.arange(0, block)
        valid = index < elements
        row, col = index // width, index % width
        gate = tl.load(source + row * (2 * width) + col, valid, other=0).to(tl.float32)
        up = tl.load(source + row * (2 * width) + width + col, valid, other=0).to(tl.float32)
        # Native SiLU: float32 exp/division, then BF16 rounding BEFORE multiply.
        # tl.exp is approximate; do not substitute it or remove this rounding.
        silu = tl.div_rn(gate, 1.0 + libdevice.exp(-gate))
        rounded = silu.to(tl.bfloat16).to(tl.float32)
        tl.store(output + index, (rounded * up).to(tl.bfloat16), valid)

    _KERNEL = activate
    return _KERNEL


def fused_swiglu(matrix: Any, *, output_budget_bytes: int) -> Any:
    """Forward-only BF16 CUDA candidate, with explicit output allocation budget.

    Does not mutate its input or change PyTorch math policy. No autograd promise:
    callers must be under no_grad and operands must not carry a gradient graph.
    """
    import torch

    if type(output_budget_bytes) is not int or output_budget_bytes < 1:
        raise ValueError("positive integer output budget required")
    if (not isinstance(matrix, torch.Tensor) or matrix.ndim != 2
            or not matrix.shape[1] or matrix.shape[1] % 2
            or matrix.dtype != torch.bfloat16 or not matrix.is_contiguous()):
        raise ValueError("contiguous two-dimensional BF16 gate/up matrix required")
    if matrix.device.type != "cuda":
        raise ValueError("Triton SwiGLU requires CUDA")
    if torch.is_grad_enabled() or matrix.requires_grad:
        raise ValueError("experimental SwiGLU is forward-only and requires no_grad")
    width = matrix.shape[1] // 2
    plan = plan_swiglu(matrix.shape[0], width)
    if plan["output_bytes"] > output_budget_bytes:
        raise ValueError("SwiGLU output exceeds declared budget")
    kernel = _kernel() if matrix.shape[0] else None
    output = torch.empty((matrix.shape[0], width), device=matrix.device, dtype=matrix.dtype)
    if kernel is not None:
        import triton

        with torch.cuda.device(matrix.device):
            kernel[(triton.cdiv(output.numel(), 256),)](
                matrix, output, output.numel(), width, 256, enable_fp_fusion=False,
            )
    return output
