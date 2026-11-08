# SPDX-License-Identifier: Apache-2.0
#
# Origin: Soup Kernel Challenge (Decentrathon 6.0), the submission of team Luck
# (BAITC-Hacks/hack-e79529fc-luck@b3f1225f); the int8 gate/up packs follow ideas seen in the
# solutions of teams Null Pointers, 21, DarkDaur and OrbiTeam.
# Modified for Soup: the first-step setup of the original (``patches._packed_model``) and its
# module singletons are one object that belongs to one training run and releases everything it
# holds when closed.
"""Everything one packed-LoRA training run owns, in one object.

A :class:`PackedRun` holds the policy, the packed layers of the model, the activation stash, the
host-parked tables, the memory plan, the int8 verdict and the counters. Nothing is global: two
runs in one process, one after the other, do not see each other, and :meth:`PackedRun.close` gives
the pinned host memory back and puts every table on the GPU again.

The trainer wiring (a hook on the causal LM's ``forward``) lives outside this package::

    run = PackedRun(Policy.from_config(config), log=console_line)
    run.prepare(causal_lm, torch.bfloat16, tokens=batch_size * max_length)   # after get_peft_model
    ...
    result = run.forward(input_ids, attention_mask, labels, num_items_in_batch)
    if result is None:   # not right-padded: run the stock forward for this batch
        run.note_stock("a batch that is not right-padded")
    ...
    run.restore_tables()   # before eval, generation, merge and save
    run.park_tables()      # and back afterwards
    run.close()
"""

import os

import torch

from . import budget, hostembed, int8check, offload, selftest
from . import model as packed_model
from .layout import step_reserve_bytes
from .offload import PinnedPool, Stash
from .support import OPEN_ARCHITECTURES


class PackedRun:
    """The state of one training run on the packed kernels."""

    def __init__(self, policy, log=None):
        self.policy = policy
        self.log = log or (lambda message: None)
        self.rounding = policy.rounding()
        self.int8 = policy.int8_switch()
        self.pool = PinnedPool()
        self.stash = None
        self.tables = None
        self.plan = None
        self.packed = None
        self.model = None
        self.device = None
        self.dtype = None  # the compute dtype of the layers (a trainer's autocast must match it)
        self.tokens = None
        self.park_pending = False  # the plan keeps a table off the GPU and none is parked yet
        self.link = None  # measured PCIe speed, bytes/s, when something crosses it
        self.steps_packed = 0
        self.steps_stock = 0
        self.stock_reasons = {}
        self.closed = False

    # -- setup ----------------------------------------------------------------------------------
    def prepare(self, model, dtype, tokens, architectures=OPEN_ARCHITECTURES, park=True):
        """Read the model, run the start-up checks, make the memory plan and place what it says.

        ``tokens``: the most tokens a step can hold (batch size x max length). ``park=False``
        leaves the tables the plan keeps off the GPU where they are until :meth:`park_tables`
        (a trainer that has not yet been built or prepared by accelerate must not see a
        parameter on the host). Raises
        :class:`~soup_cli.kernels.packed_lora.model.UnsupportedError` for a model the kernels do not
        handle, and :class:`~soup_cli.kernels.packed_lora.budget.BudgetTooSmallError` for a budget
        the floor does not fit; the caller turns either into an error or into the stock step
        (``policy.on_unsupported``). Returns the packed model."""
        causal = packed_model.unwrap(model)
        packed = packed_model.prepare(causal, dtype, architectures)
        self.model, self.packed, self.tokens = causal, packed, int(tokens)
        self.dtype = dtype
        self.device = packed.layers[0].pack.scale2.device
        self.log(packed_model.describe(packed))
        problems = selftest.kernel_problems(dtype, self.device) + selftest.decode_problems(
            packed.layers[0], packed.layers[0].pack.weights[0])
        if problems:
            raise packed_model.UnsupportedError(
                "the kernel self-test failed: " + "; ".join(problems))
        if self.int8.needs_check:
            int8check.decide(packed, self.int8, self.rounding, self.log)
        n_layers = len(packed.layers)
        int8_on = self.int8.enabled() and self.tokens > 16
        if self.policy.gpu_budget_bytes is not None:
            shape = budget.Shape.of_model(causal, packed.save_x)
            acts = self.policy.activations if self.policy.activations != "auto" else None
            self.plan = budget.Plan(shape, self.policy.gpu_budget_bytes, self.tokens, int8_on,
                                    acts)
            if self.plan.head_gpu and self.plan.embed_gpu:
                self.log("all tables stay on the GPU")
            elif park:
                self._park_tables()
            else:
                self.park_pending = True
        n_host = 0
        if self.policy.activations != "resident":
            n_host = n_layers if self.plan is None else n_layers - self.plan.k
        if n_host or self.park_pending or (self.tables is not None and self.tables.any_active()):
            self.link = offload.measure_link()  # something crosses PCIe: how fast?
            if offload.is_slow(self.link):
                self.log(f"PCIe is slow ({self.link / 1e9:.1f} GB/s): "
                         + ("the plan keeps its host copies, expect a slower step"
                            if self.plan is not None else "activations stay on the GPU"))
        if n_host and not (self.plan is None and offload.is_slow(self.link)):
            per_layer = packed_model_layer_bytes(packed, self.tokens, int8_on)
            workers = max(1, min(4, (os.cpu_count() or 2) // 2))
            self.stash = Stash(n_layers, per_layer, self.pool, n_host=n_host, workers=workers)
        if self.plan is not None:
            self.log(self.plan.describe())
        return packed

    def _tables_args(self):
        causal = self.model
        return causal.get_input_embeddings().weight, causal.get_output_embeddings().weight

    def _park_tables(self):
        """Move the tables the plan keeps off the GPU into pinned host memory."""
        plan, packed = self.plan, self.packed
        keep_embed = plan.embed_gpu or (packed.tied and plan.head_gpu)
        if self.tables is None:
            self.tables = hostembed.HostTables(self.pool, packed.tied)
        embed_p, head_p = self._tables_args()
        if not self.tables.park(embed_p, head_p, embed=not keep_embed, head=not plan.head_gpu):
            self.log("a table could not move to host memory: it stays on the GPU")

    # -- one step -------------------------------------------------------------------------------
    def stash_for_step(self, packed, n_tokens, per_layer, int8_stash, device):
        """The stash for this step, with this step's resident layers decided (``model.forward``)."""
        n_layers = len(packed.layers)
        mode = self.policy.activations
        count = None
        if self.plan is not None:
            # a GPU budget: the plan says how many of the last layers stay resident
            self.plan.int8 = int8_stash
            budget.observe(self.plan, torch.cuda.max_memory_allocated(), self.log)
            count = self.plan.acts_resident(n_tokens, per_layer)
            mode = "resident" if count >= n_layers else "budget"
        elif mode == "auto" and offload.is_slow(self.link):
            mode = "resident"  # slow PCIe: nothing crosses it
        stash = self.stash if self.stash is not None and self.stash.fits(n_layers, per_layer) \
            else None
        if stash is None and mode != "resident":
            # no pinned buffers big enough (this step is bigger than the run said): layers that do
            # not fit the GPU can still be offloaded, to buffers page-locked at first use
            if self.stash is not None:
                self.stash.close()
            stash = self.stash = Stash(n_layers, per_layer + per_layer // 4, self.pool,
                                       eager=False)
        if stash is None:  # every layer stays on the GPU
            return Stash.resident_only(n_layers, self.pool)
        if count is not None:
            stash.plan_count(count)
        else:
            stash.plan(mode, per_layer, step_reserve_bytes(
                n_tokens, packed.hidden, packed.intermediate, packed.lora_bytes * 3 // 2), device)
        return stash

    def forward(self, input_ids, attention_mask, labels, n_items=None):
        """(loss, stats) for one training batch, or None if the batch is not right-padded (the
        caller then runs the stock forward). Counts the step."""
        result = packed_model.forward(self.packed, self, input_ids, attention_mask, labels, n_items)
        if result is not None:
            self.steps_packed += 1
        return result

    def note_stock(self, reason):
        """Count a step that ran the stock forward, and say why once."""
        self.steps_stock += 1
        if reason not in self.stock_reasons:
            self.stock_reasons[reason] = 0
            self.log(f"stock forward for a training step: {reason}")
        self.stock_reasons[reason] += 1

    def summary(self):
        """One line for the end of training."""
        extra = "; ".join(f"{n}x {reason}" for reason, n in self.stock_reasons.items())
        return (f"packed_lora: {self.steps_packed} packed, {self.steps_stock} stock steps"
                + (f" ({extra})" if extra else ""))

    # -- the tables, around eval / generation / save --------------------------------------------
    def restore_tables(self):
        """Every host-parked table back on the GPU (the stock forward and checkpoint saves read
        them there). The host copies are kept for :meth:`park_tables`."""
        if self.tables is not None and self.tables.any_active():
            embed_p, head_p = self._tables_args()
            self.tables.restore(embed_p, head_p, self.device)

    def park_tables(self):
        """Park the tables the plan keeps off the GPU: the first time after ``prepare(park=False)``,
        and again after :meth:`restore_tables`."""
        if self.plan is not None and not self.closed and (
                self.tables is not None or self.park_pending):
            self.park_pending = False
            self._park_tables()

    # -- end ------------------------------------------------------------------------------------
    def close(self):
        """Give the pinned host memory back and put every table on the GPU. Safe to call twice."""
        if self.closed:
            return
        self.closed = True
        if self.tables is not None:
            self.restore_tables()
            self.tables.close()
        if self.stash is not None:
            self.stash.close()
        self.pool.close()
        self.stash = self.tables = self.plan = self.packed = None


def packed_model_layer_bytes(packed, tokens, int8):
    """Bytes one layer's saved tensors take for ``tokens`` tokens (``layout.layer_bytes``)."""
    return offload.layer_bytes(tokens, packed.hidden, packed.intermediate, packed.q_dim,
                               packed.kv_dim, packed.save_x, int8, packed.qk_norm)
