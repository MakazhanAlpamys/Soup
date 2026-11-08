# SPDX-License-Identifier: Apache-2.0
"""Start-up checks of the kernels themselves, run once per process on the first use.

Support is decided by capability, not by version tables: instead of reading the versions of
torch, Triton and bitsandbytes, the kernels are run on tiny inputs against PyTorch, and the NF4
decoder is compared with bitsandbytes' own on the model's first layer. A failure is a list of
reasons, so the run stops (or falls back to the stock step) before the first step, not in the
middle of it.
"""

from __future__ import annotations

import torch

_PASSED: dict = {}  # (device index, dtype) -> True, once the tiny-kernel checks passed here


def _close(a, b, rtol, atol):
    return bool(torch.allclose(a.float(), b.float(), rtol=rtol, atol=atol))


def kernel_problems(dtype, device=None):  # pragma: no cover - needs Triton and CUDA (gpu tests)
    """Reasons the Triton kernels do not produce what PyTorch does on this machine (empty list:
    they do). Tiny inputs; cached per device and dtype."""
    device = torch.device(device or "cuda")
    key = (device.index if device.index is not None else torch.cuda.current_device(), dtype)
    if key in _PASSED:
        return []
    from .triton.norms import rmsnorm_bwd, rmsnorm_fwd
    from .triton.q8 import dequant_q8, quant_q8

    problems = []
    gen = torch.Generator(device=device).manual_seed(0)
    # RMSNorm forward and backward against the Qwen2 formula in fp32
    rows, cols, eps = 6, 192, 1e-6
    x = torch.randn(rows, cols, device=device, generator=gen)
    w = torch.rand(cols, device=device, generator=gen) + 0.5
    y, rstd = rmsnorm_fwd(x, w, eps, dtype=dtype)
    ref_rstd = torch.rsqrt(x.pow(2).mean(-1) + eps)
    ref_y = (w * (x * ref_rstd[:, None])).to(dtype)
    if not (_close(rstd, ref_rstd, 1e-5, 1e-6) and _close(y, ref_y, 1e-2, 1e-2)):
        problems.append("the RMSNorm kernel disagrees with PyTorch")
    dy = torch.randn(rows, cols, device=device, generator=gen)
    xr = x.clone().requires_grad_(True)
    ((w * (xr * torch.rsqrt(xr.pow(2).mean(-1, keepdim=True) + eps))) * dy).sum().backward()
    if not _close(rmsnorm_bwd(dy, x, w, rstd), xr.grad, 1e-4, 1e-5):
        problems.append("the RMSNorm backward kernel disagrees with PyTorch")
    # int8 pack round trip: within one step of the group's scale
    rows, cols = 8, 256
    t = torch.randn(rows, cols, device=device, generator=gen).to(dtype)
    back = dequant_q8(quant_q8(t), rows, cols, dtype).float()
    step = t.float().abs().view(rows, cols // 64, 64).amax(2, keepdim=True) / 127
    if not bool(((back - t.float()).abs().view(rows, cols // 64, 64) <= step * 1.01 + 1e-6).all()):
        problems.append("the int8 pack kernel does not round-trip within one quantisation step")
    # PyTorch's private memory-efficient attention against scaled_dot_product_attention
    problems.extend(_attention_problems(dtype, device, gen))
    if not problems:
        _PASSED[key] = True
    return problems


def _attention_problems(dtype, device, gen):  # pragma: no cover - needs CUDA (gpu tests)
    q = torch.randn(1, 2, 24, 2, 32, device=device, generator=gen).to(dtype)  # [1, rows, T, H, D]
    q, k, v = q[:, 0], q[:, 1], q[:, 1].flip(1)
    cu = torch.tensor([0, 24], dtype=torch.int32, device=device)
    try:
        out, lse, seed, offset, _, _ = torch.ops.aten._efficient_attention_forward(
            q, k, v, None, cu, cu, 24, 24, 0.0, 1, True, scale=32 ** -0.5)
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        ).transpose(1, 2)
        torch.ops.aten._efficient_attention_backward(
            torch.ones_like(out), q, k, v, None, out, cu, cu, 24, 24, lse, 0.0, seed, offset, 1,
            False, scale=32 ** -0.5, num_splits_key=None)
    except Exception as exc:  # noqa: BLE001 - any failure here is the answer
        return [f"torch's private memory-efficient attention failed on a tiny input: {exc!r}"]
    if not _close(out, ref, 2e-2, 2e-2):
        return ["torch's private memory-efficient attention disagrees with scaled_dot_product"
                "_attention on a tiny input"]
    return []


def decode_problems(layer, weight):  # pragma: no cover - needs Triton and CUDA (gpu tests)
    """Reasons the NF4 decoder does not reproduce bitsandbytes' on this model: layer 0's q
    projection, decoded by the pack and by ``bitsandbytes.functional.dequantize_4bit``, must be
    bit-identical."""
    import bitsandbytes as bnb

    (mine,) = layer.pack.dequant(0, 1)
    try:
        theirs = bnb.functional.dequantize_4bit(weight.data, weight.quant_state).to(mine.dtype)
    except Exception as exc:  # noqa: BLE001 - any failure here is the answer
        return [f"bitsandbytes could not decode the first 4-bit weight for comparison: {exc!r}"]
    if mine.shape != theirs.shape or not torch.equal(mine, theirs):
        return ["the NF4 decoder is not bit-identical to bitsandbytes' on the first 4-bit weight"]
    return []
