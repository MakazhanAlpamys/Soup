# SPDX-License-Identifier: Apache-2.0
#
# Origin: Soup Kernel Challenge (Decentrathon 6.0), the submission of team Luck
# (BAITC-Hacks/hack-e79529fc-luck@b3f1225f); the int8 gate/up packs follow ideas seen in the
# solutions of teams Null Pointers, 21, DarkDaur and OrbiTeam. Modified for Soup: byte sizes
# split out of the kernels so the planner can use them without Triton.
"""Byte sizes of what the packed step stores. Pure arithmetic: no torch, no Triton.

The planner (``budget.py``), the activation stash (``offload.py``) and the int8 kernels
(``triton/q8.py``) must agree on these numbers, so each is written once, here.
"""

from __future__ import annotations

#: Every tensor a layer saves starts on a multiple of this many bytes of its buffer.
ALIGN = 256

#: Values per scale of an int8 pack (``triton/q8.py``): one fp32 scale per group of a row.
Q8_GROUP = 64


def q8_usable_width(n_cols: int) -> bool:
    """Whether rows of ``n_cols`` values can be packed: whole groups of ``Q8_GROUP``."""
    return n_cols % Q8_GROUP == 0


def q8_pack_bytes(n_cols: int) -> int:
    """Bytes of a pack per row of ``n_cols`` values: int8 values and one fp32 scale per group."""
    return n_cols + 4 * (n_cols // Q8_GROUP)


def layer_bytes(tokens: int, hidden: int, inter: int, q_dim: int, kv_dim: int,
                save_x: bool = False, int8: bool = True, qk_norm: bool = False) -> int:
    """Bytes one layer's saved tensors take for ``tokens`` tokens, on the host (the pinned
    buffer) or on the GPU (resident): fp32 residual, q, k, v (one copy per kv head), attention
    output, x (with a gate / up LoRA), gate and up (int8 packs where the int8 backward uses
    them, more than 16 tokens; else in the compute dtype, 2 bytes), plus alignment."""
    mlp = 2 * 2 * inter
    if int8 and tokens > 16 and q8_usable_width(inter):
        mlp = 2 * q8_pack_bytes(inter)
    per_token = 4 * hidden + 2 * q_dim + 2 * 2 * kv_dim + 2 * q_dim + mlp
    if save_x:
        per_token += 2 * hidden
    if qk_norm:  # Qwen3: x_hat of q and of k (compute dtype) for the norm backward
        per_token += 2 * (q_dim + kv_dim)
    return tokens * per_token + 8 * ALIGN


def step_reserve_bytes(tokens: int, hidden: int, inter: int, lora_param_bytes: int = 0) -> int:
    """What a step needs on the GPU next to its saved tensors: the backward's largest transients
    (a few [tokens, intermediate] int32 / int8 / compute-dtype tensors and the decoded MLP
    weights), the fp32 residual-stream tensors, and gradients plus Adam state of the LoRA
    parameters."""
    return 24 * tokens * inter + 16 * tokens * hidden + 6 * inter * hidden + 3 * lora_param_bytes
