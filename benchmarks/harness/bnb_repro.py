"""Standalone reproducer for bitsandbytes issue #2034.

This is the upstream minimal reproducer used to demonstrate that
``MatMul4Bit`` keeps the packed weight and quantization state on the
autograd context as ordinary attributes instead of saved tensors.

Requirements
------------
- CUDA GPU required.
- No model download is required.
- Historical environment: torch 2.13.0+cu130, bitsandbytes 0.50.0.
- The original reproducer uses one CUDA device and takes about one minute.

Protocol
--------
Compare three arms using gradient checkpointing:

- NF4 with recycled buffers: expected to reproduce the defect.
- NF4 with private buffers: reference.
- bf16 with recycled buffers: control.

The trainable parameter is upstream of the 4-bit matmul so its gradient
depends on the value consumed by ``MatMul4Bit.backward``.
"""

from __future__ import annotations

import sys

import bitsandbytes as bnb
import torch
import torch.nn.functional as functional
from bitsandbytes.functional import quantize_4bit
from torch.utils.checkpoint import checkpoint

DEV = "cuda"
DTYPE = torch.bfloat16
N_LAYERS = 8
DIM = 4096
TOKENS = 256
POOL_SLOTS = 2


class SlotPool:
    """Round-robin device buffers used by the recycled-buffer arm."""

    def __init__(self, n_slots, like):
        self.slots = [torch.empty_like(like) for _ in range(n_slots)]
        self.cursor = 0

    def acquire(self, src):
        slot = self.slots[self.cursor % len(self.slots)]
        self.cursor += 1
        slot.copy_(src)
        return slot


def run(quant, recycle, params_init):
    torch.manual_seed(0)
    weights = [
        torch.randn(DIM, DIM, device=DEV, dtype=DTYPE) / 32
        for _ in range(N_LAYERS)
    ]

    if quant == "nf4":
        sources, states = [], []
        for weight in weights:
            packed, state = quantize_4bit(
                weight,
                blocksize=64,
                quant_type="nf4",
                compress_statistics=True,
            )
            sources.append(packed)
            states.append(state)
    else:
        sources, states = weights, [None] * N_LAYERS

    pool = SlotPool(POOL_SLOTS, sources[0]) if recycle else None
    params = [param.clone().requires_grad_(True) for param in params_init]

    def body(idx):
        def fn(x, scale):
            hidden = x * scale

            # The buffer refill occurs inside the checkpointed region.
            weight = pool.acquire(sources[idx]) if recycle else sources[idx]

            if quant == "nf4":
                return bnb.matmul_4bit(
                    hidden,
                    weight.t(),
                    quant_state=states[idx],
                )

            return functional.linear(hidden, weight)

        return fn

    x = torch.randn(
        TOKENS,
        DIM,
        device=DEV,
        dtype=DTYPE,
        requires_grad=True,
    )

    for index in range(N_LAYERS):
        x = checkpoint(
            body(index),
            x,
            params[index],
            use_reentrant=False,
        )

    x.float().pow(2).mean().backward()

    return [param.grad.detach().float().clone() for param in params]


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: CUDA is required for bnb_repro.py")
        return 0

    print(f"torch         {torch.__version__}")
    print(f"bitsandbytes  {bnb.__version__}")
    print(f"gpu           {torch.cuda.get_device_name(0)}")
    print(
        f"shape         {N_LAYERS} layers x {DIM} x {DIM}, "
        f"M={TOKENS}, dtype={DTYPE}"
    )

    torch.manual_seed(1234)
    init = [
        torch.ones(DIM, device=DEV, dtype=DTYPE)
        + 0.01 * torch.randn(DIM, device=DEV, dtype=DTYPE)
        for _ in range(N_LAYERS)
    ]

    reference = run("nf4", recycle=False, params_init=init)
    recycled = run("nf4", recycle=True, params_init=init)
    bf16_recycled = run("bf16", recycle=True, params_init=init)
    bf16_reference = run("bf16", recycle=False, params_init=init)

    print()
    print(
        f"{'layer':>5}  "
        f"{'nf4 recycled vs private':>24}  "
        f"{'bf16 recycled vs private':>25}"
    )

    for index in range(N_LAYERS):
        nf4_diff = (
            recycled[index] - reference[index]
        ).abs().max().item()
        bf16_diff = (
            bf16_recycled[index] - bf16_reference[index]
        ).abs().max().item()

        nf4_status = "MISMATCH" if nf4_diff else "ok"
        bf16_status = "MISMATCH" if bf16_diff else "ok"

        print(
            f"{index:>5}  "
            f"{nf4_diff:>16.6e} {nf4_status:>9}  "
            f"{bf16_diff:>16.6e} {bf16_status:>9}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
