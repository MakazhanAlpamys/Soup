# SPDX-License-Identifier: Apache-2.0
#
# Origin: Soup Kernel Challenge (Decentrathon 6.0), the submission of team Luck
# (BAITC-Hacks/hack-e79529fc-luck@b3f1225f); the int8 gate/up packs follow ideas seen in the
# solutions of teams Null Pointers, 21, DarkDaur and OrbiTeam. Modified for Soup: the planner
# is pure arithmetic with no module state, the 4-bit layers always stay on the GPU (the
# layer-streaming half of the original plan is not ported), and a budget below the floor is
# refused with the minimum instead of running the smallest plan.
"""A GPU memory budget for the whole training step (``training.packed_lora.gpu_budget_mib``).

With a budget, the planner decides once the model and the batch size are known what the budget
is spent on, in this order, by the PCIe traffic saved per byte of GPU memory (everything not on
the GPU is copied over the link every step, behind the compute, on a side stream):

  0. the floor (always): the 4-bit layers' packs, the LoRA parameters + Adam state, and the
     step's transient tensors (the backward of the MLP, about 16.6 B x tokens x intermediate);
  1. the saved activations of the last ``k`` layers (a layer's 78 KB per token for Qwen2.5-7B;
     the others are copied to pinned host memory in the forward and back, one layer ahead, in
     the backward): resident layers are those whose backward runs first;
  2. only when every layer is resident: the lm_head (otherwise walked in slabs from the host),
     then the embedding.

The estimate of the peak is ``fixed + max(D0, D_res + k x layer_bytes)``. ``fixed`` is counted
from the model's shapes (checked to the MiB against ``before_mib`` of a 7B run: predicted 563 +
7 for 570 measured). ``D0 = (16.6 I + 13.8 H) T`` bytes is the transient peak of a step with
every layer offloaded and ``D_res = 0.79 D0`` the one with the last layers resident (I =
intermediate size, H = hidden size, T = tokens): both are fits to measurements of the RTX 5070
Laptop on the Qwen2.5-7B and 1.5B shapes. A margin (2 % of the budget + 64 MiB) covers the
allocator's rounding and what the fit leaves out, and :func:`observe` corrects the plan from the
peak the run actually reached (one step late).

The planner is a fit, not a proof: other LoRA target sets, other models and the bf16 stash
(``int8_backward: off``) can differ by a few percent. A budget under the floor is refused with
the smallest budget that works.

Nothing here changes a number the training computes, only where tensors live. No torch import.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

from .layout import layer_bytes

SAFETY_FRAC = 0.02  # of the budget
SAFETY_BYTES = 64 << 20
# transient peak of a step, bytes per token: A_INTER x intermediate + A_HIDDEN x hidden (see above)
A_INTER = 16.6
A_HIDDEN = 13.8
RES_FACTOR = 0.79  # the transient with the last layers resident, relative to all offloaded
HEAD_CHUNK = 256  # tokens per logits chunk of the head kernel (triton/head.py CHUNK)
HEAD_SLAB = 8192  # rows of one lm_head slab of the host-parked head (triton/head.py SLAB)
FALLBACK_ASSUMED_TOKENS = 8192

MIB = 1 << 20


class BudgetTooSmallError(ValueError):
    """The budget cannot hold the model at this batch size; ``minimum_mib`` is the smallest
    ``gpu_budget_mib`` that can."""

    def __init__(self, message: str, minimum_mib: int) -> None:
        super().__init__(message)
        self.minimum_mib = minimum_mib


def limit_of(budget: int) -> int:
    """The estimate the plan may reach: the budget minus the safety margin."""
    return int(budget * (1 - SAFETY_FRAC)) - SAFETY_BYTES


class Shape:
    """Sizes of the model that the plan needs (bytes unless noted)."""

    def __init__(self, layers, hidden, inter, q_dim, kv_dim, vocab, lora_params, save_x=True,
                 tied=False, dt_bytes=2, qk_norm=False):
        self.qk_norm = qk_norm
        self.layers, self.hidden, self.inter = layers, hidden, inter
        self.q_dim, self.kv_dim, self.vocab = q_dim, kv_dim, vocab
        self.lora_params, self.save_x = lora_params, save_x
        self.tied, self.dt_bytes = tied, dt_bytes
        weights = [hidden * q_dim, hidden * kv_dim, hidden * kv_dim, q_dim * hidden,
                   hidden * inter, hidden * inter, hidden * inter]
        numel = sum(weights)
        codes, scales = numel // 2, numel // 64
        self.slot = -(-codes // 256) * 256 + scales  # triton/nf4.Pack.slot_bytes
        groups = sum(-(-(n // 64) // 256) for n in weights)
        self.aux = 8 * groups  # a pack's fp32 second-level scales + offsets
        self.table = vocab * hidden * dt_bytes
        # fp32 norms, q/k/v biases (fp32 + compute-dtype copy), final norm
        self.small = layers * (8 * hidden + 6 * (q_dim + 2 * kv_dim)) + 4 * hidden

    @classmethod
    def of_model(cls, model, save_x=True) -> "Shape":
        """From a peft-wrapped (or plain) causal LM: the trainable parameters are the LoRA."""
        config = model.config
        heads, kv = config.num_attention_heads, config.num_key_value_heads
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // heads
        lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
        try:
            tied = model.get_input_embeddings().weight is model.get_output_embeddings().weight
        except AttributeError:
            tied = bool(getattr(config, "tie_word_embeddings", False))
        inner = getattr(model, "model", None) or model.get_decoder()
        qk_norm = hasattr(inner.layers[0].self_attn, "q_norm")
        return cls(config.num_hidden_layers, config.hidden_size, config.intermediate_size,
                   heads * head_dim, kv * head_dim, config.vocab_size, lora, save_x, tied, 2,
                   qk_norm)

    @classmethod
    def of_config(cls, config, rank=16) -> "Shape":
        """Before the LoRA exists: rank-``rank`` adapters on all seven projections (an upper
        bound for the recipes this is meant for)."""
        heads, kv = config.num_attention_heads, config.num_key_value_heads
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // heads
        h, i, qd, kd = config.hidden_size, config.intermediate_size, heads * head_dim, kv * head_dim
        per_layer = rank * ((h + qd) + 2 * (h + kd) + (qd + h) + 3 * (h + i))
        return cls(config.num_hidden_layers, h, i, qd, kd, config.vocab_size,
                   config.num_hidden_layers * per_layer, True,
                   bool(getattr(config, "tie_word_embeddings", False)), 2,
                   getattr(config, "model_type", "") == "qwen3")

    # -- sizes -------------------------------------------------------------------------------
    def fixed(self, head_gpu: bool, embed_gpu: bool) -> int:
        """Bytes on the GPU between steps: LoRA (fp32 weights, compute-dtype copies, Adam m and
        v), every layer's 4-bit pack, and the tables that live there."""
        total = self.lora_params * (4 + 2 + 8) + self.small + self.layers * self.aux
        total += self.layers * self.slot
        if self.tied:
            total += self.table if (head_gpu or embed_gpu) else 0
        else:
            total += self.table * (int(bool(head_gpu)) + int(bool(embed_gpu)))
        return total

    def d0(self, tokens: int) -> float:
        """Transient peak of a step with every layer's activations offloaded."""
        return (A_INTER * self.inter + A_HIDDEN * self.hidden) * tokens

    def d_res(self, tokens: int, head_gpu: bool = False) -> float:
        """Transient at the turn of the step (end of forward, start of backward), on top of the
        resident layers: the larger of the first backward's and the head's."""
        head = 12 * tokens * self.hidden + 2 * HEAD_CHUNK * self.vocab * self.dt_bytes
        if not head_gpu:  # slab buffers of head_forward_streamed
            head += 2 * HEAD_SLAB * self.hidden * self.dt_bytes + 2 * tokens * HEAD_SLAB
        return max(RES_FACTOR * self.d0(tokens), head)

    def layer(self, tokens: int, int8: bool = True) -> int:
        """Bytes of one layer's saved tensors (``layout.layer_bytes``)."""
        return layer_bytes(tokens, self.hidden, self.inter, self.q_dim, self.kv_dim,
                           self.save_x, int8, self.qk_norm)

    def peak(self, head_gpu: bool, embed_gpu: bool, k: int, tokens: int,
             int8: bool = True) -> float:
        """The estimated torch peak (allocated bytes) with the last ``k`` layers resident."""
        pl = self.layer(tokens, int8)
        extra = max(self.d0(tokens), self.d_res(tokens, head_gpu) + k * pl)
        return self.fixed(head_gpu, embed_gpu) + extra


def minimum_budget_mib(shape: Shape, tokens: int, int8: bool = True) -> int:
    """The smallest ``gpu_budget_mib`` whose limit holds the floor: every pack resident, the
    tables on the host, no layer's activations resident."""
    floor = shape.peak(False, False, 0, tokens, int8) + SAFETY_BYTES
    return int(math.ceil(floor / (1 - SAFETY_FRAC) / MIB))


class Plan:
    """The decision: how many of the last layers keep their saved activations on the GPU and where
    the lm_head and the embedding live; ``acts_resident(tokens)`` for every step. The 4-bit packs
    always stay on the GPU.

    ``acts_fixed`` pins the activations ("offload": none resident, "resident": all of them) as
    ``training.packed_lora.activations`` does. Raises :class:`BudgetTooSmallError` for a budget the
    floor does not fit, and for ``acts_fixed="resident"`` when all layers do not fit."""

    def __init__(self, shape: Shape, budget: int, tokens: int, int8: bool,
                 acts_fixed: Optional[str] = None):
        self.shape, self.budget, self.tokens, self.int8 = shape, budget, tokens, int8
        self.limit = limit_of(budget)
        self.acts_fixed = acts_fixed if acts_fixed in ("offload", "resident") else None
        self.cap = shape.layers  # lowered if a step exceeds the budget (observe)
        self.base_peak = None  # max_memory_allocated when the plan was made
        self.seen_peak = None
        s, t, layers = shape, tokens, shape.layers
        if s.peak(False, False, 0, t, int8) > self.limit:
            minimum = minimum_budget_mib(s, t, int8)
            raise BudgetTooSmallError(
                f"training.packed_lora.gpu_budget_mib={budget // MIB} is below the minimum of "
                f"{minimum} MiB for this model at {t} tokens per step (4-bit weights, LoRA "
                f"state and the step's transients stay on the GPU). Raise the budget or "
                f"lower the batch size or data.max_length.", minimum)
        head_gpu = embed_gpu = False
        pl = s.layer(t, int8)
        self.k = self._count(t, pl, head_gpu, embed_gpu)
        # the tables, only once every layer's activations are resident too (the peak at k = layers
        # is over the limit otherwise)
        if self.acts_fixed != "offload":
            if s.peak(True, False, layers, t, int8) <= self.limit:
                head_gpu = True
                if s.peak(True, True, layers, t, int8) <= self.limit:
                    embed_gpu = True
                self.k = self._count(t, pl, head_gpu, embed_gpu)
        self.head_gpu, self.embed_gpu = head_gpu, embed_gpu
        self.est_peak = s.peak(head_gpu, embed_gpu, self.k, t, int8)
        if self.acts_fixed == "resident" and self.est_peak > self.limit:
            raise BudgetTooSmallError(
                f"training.packed_lora.activations: resident keeps all {layers} layers' saved "
                f"tensors on the GPU, which needs about {int(self.est_peak) // MIB} MiB at "
                f"{t} tokens per step; the budget is {budget // MIB} MiB. Use activations: auto "
                f"or raise the budget.", int(math.ceil(self.est_peak / (1 - SAFETY_FRAC) / MIB)))

    def set_int8(self, int8: bool) -> None:
        """The int8 gate/up stash turned out on / off (the start-up check): the layers' size
        changes."""
        int8 = bool(int8)
        if int8 == self.int8:
            return
        self.int8 = int8
        pl = self.shape.layer(self.tokens, int8)
        self.k = self._count(self.tokens, pl, self.head_gpu, self.embed_gpu)
        self.est_peak = self.shape.peak(self.head_gpu, self.embed_gpu, self.k, self.tokens, int8)

    def _count(self, tokens: int, pl: int, head_gpu: bool, embed_gpu: bool) -> int:
        layers = self.shape.layers
        if self.acts_fixed == "offload":
            return 0
        if self.acts_fixed == "resident":
            return layers
        s = self.shape
        base = s.fixed(head_gpu, embed_gpu)
        room = self.limit - base - s.d_res(tokens, head_gpu)
        k = int(room // max(pl, 1)) if room > 0 else 0
        return max(0, min(layers, k))

    def acts_resident(self, tokens: int, per_layer: Optional[int] = None) -> int:
        """Resident layers (the last ones) for a step of ``tokens`` tokens."""
        pl = per_layer or self.shape.layer(tokens, self.int8)
        return min(self._count(tokens, pl, self.head_gpu, self.embed_gpu), self.cap)

    def host_bytes(self) -> int:
        """Pinned host memory the plan needs: the offloaded layers' buffers and the tables."""
        s = self.shape
        pl = s.layer(self.tokens, self.int8)
        tables = 0 if (self.head_gpu and self.embed_gpu) else s.table * (
            1 if s.tied else int(not self.head_gpu) + int(not self.embed_gpu))
        return (s.layers - self.k) * pl + tables

    def pcie_bytes(self) -> tuple[int, int]:
        """(host->GPU, GPU->host) bytes a full-size step moves."""
        s = self.shape
        down = (s.layers - self.k) * s.layer(self.tokens, self.int8)
        up = down + (0 if self.head_gpu else s.table)
        return up, down

    def describe(self) -> str:
        """One line for the log."""
        s = self.shape
        tables = ("lm_head+embedding on GPU" if self.head_gpu and self.embed_gpu else
                  "lm_head on GPU, embedding on host" if self.head_gpu else
                  "lm_head+embedding on host")
        h2d, d2h = self.pcie_bytes()
        return (f"plan: weights resident, {tables}, activations {self.k}/{s.layers} resident"
                f"{' (int8 gate/up stash)' if self.int8 else ''}, est peak "
                f"{self.est_peak / MIB:.0f} MiB of budget {self.budget / MIB:.0f} MiB "
                f"({self.tokens} tokens), host {self.host_bytes() / MIB:.0f} MiB pinned, "
                f"PCIe/step ~{h2d / 2**30:.1f} GiB up, ~{d2h / 2**30:.1f} GiB down")


def observe(plan: Plan, peak: int, log: Callable[[str], None]) -> None:
    """The guard, called before each step's forward with ``torch.cuda.max_memory_allocated()``:
    if the peak since the training began went over the budget while this plan was running, hold
    fewer layers resident from now on."""
    if plan.base_peak is None:
        plan.base_peak = plan.seen_peak = peak  # whatever happened before the first step
        return
    if peak > plan.budget and peak > plan.seen_peak:
        pl = plan.shape.layer(plan.tokens, plan.int8)
        cut = int(math.ceil((peak - plan.budget) / pl)) + 1
        current = plan.acts_resident(plan.tokens)
        plan.cap = max(0, min(plan.cap, current) - cut)
        log(f"budget exceeded: peak {peak / MIB:.0f} MiB > {plan.budget / MIB:.0f} MiB (the plan "
            f"estimated {plan.est_peak / MIB:.0f}); resident activation layers {current} -> "
            f"{plan.cap} from the next step")
    plan.seen_peak = max(plan.seen_peak, peak)
