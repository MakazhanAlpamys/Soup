# SPDX-License-Identifier: Apache-2.0
#
# Origin: Soup Kernel Challenge (Decentrathon 6.0), the submission of team Luck
# (BAITC-Hacks/hack-e79529fc-luck@b3f1225f); the int8 gate/up packs follow ideas seen in the
# solutions of teams Null Pointers, 21, DarkDaur and OrbiTeam.
"""The Triton kernels of ``training.packed_lora``.

Importing any module here imports ``triton``. Nothing outside ``soup_cli.kernels.packed_lora``
should do that directly: go through :func:`soup_cli.kernels.packed_lora.loader.load`, which
checks that Triton is importable and says why when it is not.

``nf4``     NF4 + double-quant decode of bitsandbytes 4-bit weights (bit-exact), or to int8
``norms``   RMSNorm over the fp32 residual stream
``rope``    the q/k/v step between the fused matmul and attention (LoRA epilogue, optional Qwen3
            q/k norms, rotary embedding, GQA repeat), forward and backward
``swiglu``  elementwise SwiGLU
``q8``      int8 packs of the saved gate/up activations, the fused MLP forward, SwiGLU backward
``int8``    int8 tensor-core helpers of the backward's input-gradient GEMMs
``head``    lm_head + cross-entropy in chunks, without ever holding the logits
"""
