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
builds the logits. Numbers: the loss matches the stock forward to 8e-5 relative, the LoRA
gradients to 1.5 % (exact mode) or 3.5 % (int8 backward) on the layer checks.
"""
