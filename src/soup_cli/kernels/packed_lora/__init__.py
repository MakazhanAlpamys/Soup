# SPDX-License-Identifier: Apache-2.0
#
# Origin: Soup Kernel Challenge (Decentrathon 6.0), the submission of team Luck
# (BAITC-Hacks/hack-e79529fc-luck@b3f1225f); the int8 gate/up packs follow ideas seen in the
# solutions of teams Null Pointers, 21, DarkDaur and OrbiTeam. Modified for Soup: see the
# header of each module.
"""``training.packed_lora``: Triton kernels for the Qwen2 QLoRA SFT step.

Scope: Qwen2 QLoRA SFT. Qwen3 is ported and tested at layer level; its gate is closed (see
``support.OPEN_ARCHITECTURES``).

The pure half (no torch, no Triton; safe to import from the config and the CLI preflight)::

    config.py   the settings of one run (a future ``training.packed_lora``) and which runs the
                kernels cannot take; nothing in ``config/schema.py`` refers to it yet
    layout.py   byte sizes shared by the planner, the stash and the int8 kernels
    policy.py   the settings -> a frozen per-run policy, rounding and int8 state
    budget.py   the GPU memory planner
    support.py  what the kernels support, as lists of reasons

The GPU half (torch, bitsandbytes and Triton, imported on first use)::

    model.py    ``prepare`` (a peft Qwen2 model -> packed layers) and the packed ``forward``
    layers.py   the decoder layer and the head as autograd functions
    offload.py  saved activations: on the GPU, or in pinned host memory
    hostembed.py the embedding and lm_head tables parked in pinned host memory
    int8check.py the start-up gradient check of the int8 backward
    selftest.py  the start-up check of the kernels themselves
    triton/     the kernels (:func:`soup_cli.kernels.packed_lora.loader.load_kernels`)

A step takes the stock batch (right-padded rows), drops the padding inside it, and runs every
layer on the remaining tokens; the loss and its gradient come out of a head kernel that never
builds the logits.

Numbers (RTX 5070 Laptop; ``benchmarks/packed-lora-gpu-tests-2026-11-08.md`` has the rows).
Against the stock forward and backward on tiny random models the loss agrees to 8e-5 relative
(the bar; 6e-5 measured), and the worst LoRA-gradient error over every parameter is 1.0 to 2.4 %
in bf16 and 0.3 to 1.5 % in fp16. Those rows round the int8 gradients to nearest, which is not the
default; their bars are 3 % (exact mode) and 4 % (int8 backward) in bf16 and 0.6 % / 2.5 % in
fp16. The default numerics (``int8_backward: auto``, ``int8_rounding: grad``, stochastic rounding
of the gradients) have one tiny-model row, 2.32 % against a bar of 5 %: unbiased but noisier per
step. On Qwen2.5-1.5B the gradient cosine to the stock step is 0.9990 to 0.9994 (exact mode) and
0.9941 to 0.9955 (default).
"""
