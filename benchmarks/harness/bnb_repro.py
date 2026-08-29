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

Negative control
---------------
``--bypass-pool`` deliberately disables the recycled-buffer allocation
for the NF4 candidate arm. The NF4 mismatch must then disappear. The
harness treats that as a detected mutation and exits non-zero, proving
that the reproduction depends on the recycled-buffer path.

The normal invocation also requires the NF4 recycled/private comparison
to mismatch. If ``pool.acquire()`` is bypassed or removed by a mutation,
that comparison becomes equal and the harness fails instead of silently
returning success.
"""

from __future__ import annotations

import argparse
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


def run(quant, recycle, params_init, bypass_pool=False):
    """Run one arm and return gradients for the trainable scale parameters."""
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
    elif quant == "bf16":
        sources, states = weights, [None] * N_LAYERS
    else:
        raise ValueError(f"unsupported quantisation: {quant!r}")

    pool = SlotPool(POOL_SLOTS, sources[0]) if recycle else None
    params = [param.clone().requires_grad_(True) for param in params_init]

    def body(idx):
        def fn(x, scale):
            hidden = x * scale

            if recycle and not bypass_pool:
                # The buffer refill occurs inside the checkpointed region.
                weight = pool.acquire(sources[idx])
            else:
                # Negative control: deliberately bypass recycled buffers.
                weight = sources[idx]

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


def max_gradient_diff(left, right):
    """Return the largest absolute gradient difference between two arms."""
    return max(
        (candidate - reference).abs().max().item()
        for candidate, reference in zip(left, right)
    )


def print_comparison(reference, candidate, label):
    """Print per-layer differences and return the number of mismatches."""
    print()
    print(f"{'layer':>5}  {label:>24}")

    mismatches = 0

    for index in range(N_LAYERS):
        diff = (
            candidate[index] - reference[index]
        ).abs().max().item()

        status = "MISMATCH" if diff else "ok"

        if diff:
            mismatches += 1

        print(
            f"{index:>5}  "
            f"{diff:>16.6e} {status:>9}"
        )

    return mismatches


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproduce bitsandbytes issue #2034."
    )
    parser.add_argument(
        "--bypass-pool",
        action="store_true",
        help=(
            "negative control: bypass recycled-buffer allocation for the "
            "NF4 candidate arm. The expected NF4 mismatch must disappear; "
            "the harness then exits non-zero because the mutation was "
            "detected."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

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

    if args.bypass_pool:
        print("mode          NEGATIVE CONTROL: pool reuse bypassed")

    torch.manual_seed(1234)
    init = [
        torch.ones(DIM, device=DEV, dtype=DTYPE)
        + 0.01 * torch.randn(DIM, device=DEV, dtype=DTYPE)
        for _ in range(N_LAYERS)
    ]

    reference = run(
        "nf4",
        recycle=False,
        params_init=init,
    )

    recycled = run(
        "nf4",
        recycle=True,
        params_init=init,
        bypass_pool=args.bypass_pool,
    )

    bf16_recycled = run(
        "bf16",
        recycle=True,
        params_init=init,
    )

    bf16_reference = run(
        "bf16",
        recycle=False,
        params_init=init,
    )

    nf4_mismatches = print_comparison(
        reference,
        recycled,
        "nf4 recycled vs private",
    )

    bf16_mismatches = print_comparison(
        bf16_reference,
        bf16_recycled,
        "bf16 recycled vs private",
    )

    nf4_diff = max_gradient_diff(recycled, reference)
    bf16_diff = max_gradient_diff(
        bf16_recycled,
        bf16_reference,
    )

    print()

    if args.bypass_pool:
        # The mutation deliberately removes the recycled-buffer mechanism.
        # A successful reproduction here would mean the harness is not
        # sensitive to the mechanism it claims to test.
        if nf4_diff != 0:
            print(
                "ERROR: bypassing pool reuse still produced an NF4 mismatch."
            )
            print(
                "RESULT: negative control was NOT caught."
            )
            return 2

        print(
            "NEGATIVE CONTROL: bypassing pool reuse removed the NF4 mismatch."
        )
        print("RESULT: mutation detected; exiting non-zero.")
        return 1

    # Normal reproduction: the recycled NF4 arm must differ from the
    # private-buffer reference. This is the mutation guard: if pool.acquire()
    # is removed or bypassed, the comparison becomes equal and the harness
    # fails rather than silently returning success.
    if nf4_diff == 0:
        print(
            "ERROR: expected NF4 recycled/private mismatch was not reproduced."
        )
        return 1

    # The bf16 arm is the control: recycling should not introduce the
    # corresponding NF4 stale-gradient mismatch.
    if bf16_diff != 0:
        print(
            "ERROR: bf16 recycled/private control unexpectedly mismatched."
        )
        return 1

    print(
        f"RESULT: NF4 mismatch detected across "
        f"{nf4_mismatches}/{N_LAYERS} layers; "
        f"bf16 control matched across "
        f"{N_LAYERS - bf16_mismatches}/{N_LAYERS} layers."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
