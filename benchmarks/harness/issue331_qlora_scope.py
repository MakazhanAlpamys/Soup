"""Does the #331 defect reach ORDINARY QLoRA?

The claim we are about to publish is "ordinary QLoRA is not affected". It is NOT
stated anywhere in benchmarks/gate-h100-validation.md -- the word QLoRA does not
occur in that file once -- so it is an inference, and an inference published as a
mechanism is exactly what version 3 of the preprint had to retract. This measures
it instead.

Mechanism, as recorded: bitsandbytes.MatMul4Bit stashes the packed weight and
quant_state on ctx as plain attributes rather than through save_for_backward, so
gradient checkpointing cannot discard and recompute them. The stashed reference
ALIASES whatever storage the weight lives in. The defect therefore needs a
consumer that REFILLS that storage between the forward and the backward -- layer
streaming's pooled buffer does; a private per-weight allocation does not.

Arms, all in one process, same input, same seeds, gradients compared on the
TRAINABLE LoRA parameters (a QLoRA run's actual output) as well as on dL/dx:

  REF  private buffer, no checkpointing            the reference
  A    private buffer, checkpointing ON            <- ORDINARY QLoRA
  A10  arm A repeated 10 times                     <- does it drift with use?
  B    pooled buffer REFILLED after the forward    <- positive control

B is not decoration. Without it, "A matches REF" is indistinguishable from a
harness that cannot detect anything at all. A10 exists because the record says
the streamed defect usually appears from the SECOND backward onward, so a
single-shot exact result would not settle the question.

Run:  python qlora_scope.py
"""

import sys

import bitsandbytes as bnb
import torch
from torch.utils.checkpoint import checkpoint

DEV = "cuda"
DT = torch.bfloat16
DIM = 4096
BATCH = 512
RANK = 16
REPEATS = 10


class QLoRALinear(torch.nn.Module):
    """A frozen NF4 base with a trainable LoRA pair -- i.e. QLoRA, minimally."""

    def __init__(self, weight: torch.Tensor, lora_a: torch.Tensor, lora_b: torch.Tensor):
        super().__init__()
        self.base = bnb.nn.Linear4bit(
            DIM, DIM, bias=False, compute_dtype=DT, quant_type="nf4",
            compress_statistics=True,
        )
        self.base.weight = bnb.nn.Params4bit(
            data=weight, requires_grad=False, quant_type="nf4", compress_statistics=True,
        )
        self.lora_a = torch.nn.Parameter(lora_a.clone())
        self.lora_b = torch.nn.Parameter(lora_b.clone())

    def forward(self, x):
        return self.base(x) + (x @ self.lora_a.T) @ self.lora_b.T


def build(weight, lora_a, lora_b) -> QLoRALinear:
    return QLoRALinear(weight, lora_a, lora_b).to(DEV)


def run_arm(mod, x, use_checkpoint: bool, refill_with=None, passes: int = 1):
    """Returns (grad_x, grad_lora_a, grad_lora_b) of the LAST pass."""
    for _ in range(passes):
        mod.zero_grad(set_to_none=True)
        xin = x.detach().clone().requires_grad_(True)

        out = checkpoint(mod, xin, use_reentrant=False) if use_checkpoint else mod(xin)
        loss = out.float().pow(2).sum()

        if refill_with is not None:
            # The pool slot is handed to the next layer. Streaming does this
            # between the forward and the backward; ordinary QLoRA never does.
            mod.base.weight.data.copy_(refill_with)

        loss.backward()

    return (
        xin.grad.detach().float(),
        mod.lora_a.grad.detach().float(),
        mod.lora_b.grad.detach().float(),
    )


def worst(a, b) -> float:
    return max((x - y).abs().max().item() for x, y in zip(a, b))


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA -- this measures nothing without a GPU")
        return 1

    print(f"torch {torch.__version__} | bitsandbytes {bnb.__version__}")
    print(f"gpu   {torch.cuda.get_device_name(0)}")
    print(f"shape {DIM}x{DIM} NF4 base, LoRA r={RANK}, batch {BATCH}")

    torch.manual_seed(0)
    w_this = torch.randn(DIM, DIM, dtype=DT) / DIM**0.5
    w_next = torch.randn(DIM, DIM, dtype=DT) / DIM**0.5
    lora_a = torch.randn(RANK, DIM, dtype=DT) / DIM**0.5
    lora_b = torch.randn(DIM, RANK, dtype=DT) / RANK**0.5
    x = torch.randn(BATCH, DIM, device=DEV, dtype=DT) / DIM**0.5

    ref = run_arm(build(w_this, lora_a, lora_b), x, use_checkpoint=False)
    arm_a = run_arm(build(w_this, lora_a, lora_b), x, use_checkpoint=True)
    arm_a10 = run_arm(build(w_this, lora_a, lora_b), x, use_checkpoint=True, passes=REPEATS)

    ctrl = build(w_this, lora_a, lora_b)
    packed_next = build(w_next, lora_a, lora_b).base.weight.data.detach().clone()
    assert packed_next.shape == ctrl.base.weight.data.shape, "packed shapes must match"
    arm_b = run_arm(ctrl, x, use_checkpoint=True, refill_with=packed_next)

    scale = max(t.abs().max().item() for t in ref)
    da, da10, db = worst(arm_a, ref), worst(arm_a10, ref), worst(arm_b, ref)

    print()
    print(f"reference |grad| max                      {scale:.6e}")
    print(f"A    private buffer + checkpointing       {da:.6e}   <- ordinary QLoRA")
    print(f"A10  same, {REPEATS} consecutive backwards       {da10:.6e}   <- no drift with use")
    print(f"B    pooled buffer refilled (control)     {db:.6e}   <- must be non-zero")
    print()

    if db <= 0.0:
        print("INCONCLUSIVE: the control did not fire, so this harness cannot")
        print("detect the defect and arm A proves nothing.")
        return 2

    if da == 0.0 and da10 == 0.0:
        print("RESULT: ordinary QLoRA is NOT affected.")
        print("  Bit-exact on dL/dx and on both LoRA gradients, singly and over")
        print(f"  {REPEATS} consecutive backwards, in a process where the control arm")
        print(f"  diverged by {db:.3e} through the identical code path.")
        return 0

    print("RESULT: arm A is NOT exact -- the claim would be WRONG as stated.")
    return 3


if __name__ == "__main__":
    sys.exit(main())
