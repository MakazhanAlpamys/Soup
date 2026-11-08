# SPDX-License-Identifier: Apache-2.0
#
# Origin: Soup Kernel Challenge (Decentrathon 6.0), the submission of team Luck
# (BAITC-Hacks/hack-e79529fc-luck@b3f1225f); the int8 gate/up packs follow ideas seen in the
# solutions of teams Null Pointers, 21, DarkDaur and OrbiTeam. Modified for Soup: the
# checks the original made one by one inside ``model.prepare`` are collected here as
# functions that return reasons, so the preflight, the setup and the tests share them.
"""What the packed-LoRA kernels support, as lists of reasons (empty list = supported).

Every function takes plain values or duck-typed objects and imports torch only when it must,
so the CLI preflight and the CPU test suite use them without a GPU. Support is decided by
capability, never by a version table: the private ``aten`` attention signature is read from
the installed torch, and the kernels' own start-up checks (``int8check.py``, ``selftest.py``)
run on the first use.

Qwen3: gate closed. Its layers are ported (per-head q/k norms in the rotary kernel) and match the
stock forward on a tiny random model and on Qwen3-1.7B, but no training run on a real Qwen3 has
matched plain Soup's loss curve, so :data:`OPEN_ARCHITECTURES` lists Qwen2 only.
"""

from __future__ import annotations

from typing import Any, Optional

#: The decoder-only architectures the kernels were written and tested for, by ``model_type``.
PORTED_ARCHITECTURES = {"qwen2": "Qwen2ForCausalLM", "qwen3": "Qwen3ForCausalLM"}

#: The ones a run may use. Qwen3 is ported (per-head q/k norms in the rotary kernel) and its
#: layers match the stock forward on a tiny random model and on Qwen3-1.7B, but no training
#: curve on a real Qwen3 has matched plain Soup's yet, so the gate stays closed.
OPEN_ARCHITECTURES = ("Qwen2ForCausalLM",)

#: Compute capability (major) from which bf16 and the int8 tensor-core GEMMs are available.
MIN_COMPUTE_MAJOR = 8


def environment_problems(*, system: str, has_triton: bool,
                         compute_capability: Optional[tuple[int, int]],
                         bf16_supported: bool, world_size: int = 1,
                         sharding: Optional[str] = None) -> list[str]:
    """Reasons the machine cannot run the kernels. ``system`` is ``platform.system()``;
    ``compute_capability`` is ``None`` without a CUDA device; ``sharding`` names the FSDP or
    DeepSpeed setting of a single-process run, if it has one."""
    problems: list[str] = []
    if system != "Linux":
        problems.append(
            f"runs on Linux (and WSL2) only; this is {system or 'an unknown system'}. Triton "
            f"ships with the Linux CUDA build of torch."
        )
    if not has_triton:
        problems.append(
            "needs the triton package (it comes with the Linux CUDA build of torch); it is not "
            "importable here."
        )
    if compute_capability is None:
        problems.append("needs a CUDA GPU; none is available.")
    else:
        major, minor = compute_capability
        if major < MIN_COMPUTE_MAJOR:
            problems.append(
                f"needs a GPU of compute capability {MIN_COMPUTE_MAJOR}.0 or newer (Ampere, "
                f"bf16 and int8 tensor cores); this one is {major}.{minor}. fp16 on a T4 "
                f"(7.5) is not validated with the current kernels yet."
            )
        elif not bf16_supported:
            problems.append("needs bf16 support on the GPU; torch reports none.")
    if world_size != 1:
        problems.append(
            f"supports one GPU only; WORLD_SIZE is {world_size}. FSDP, DeepSpeed and DDP are "
            f"not supported."
        )
    if sharding:
        problems.append(
            f"supports one GPU only; {sharding} is set. FSDP, DeepSpeed and DDP are not "
            f"supported."
        )
    return problems


def model_config_problems(config: Any, architectures: tuple[str, ...] = OPEN_ARCHITECTURES
                          ) -> list[str]:
    """Reasons a model, known only by its ``transformers`` config, cannot run. Checked before
    anything is downloaded. ``architectures``: the classes to accept (the tests of the Qwen3
    layers pass ``tuple(PORTED_ARCHITECTURES.values())``)."""
    problems: list[str] = []
    model_type = getattr(config, "model_type", None)
    cls_name = PORTED_ARCHITECTURES.get(model_type)
    if cls_name is None:
        listed = ", ".join(OPEN_ARCHITECTURES)
        problems.append(f"supports {listed} only; this model has model_type={model_type!r}.")
        return problems
    if cls_name not in architectures:
        if cls_name == "Qwen3ForCausalLM":
            problems.append(
                "Qwen3 is ported but not enabled: its layers match the stock forward, but no "
                "training run on a real Qwen3 has matched plain Soup's loss curve yet."
            )
        else:
            problems.append(f"{cls_name} is not enabled for this run.")
        return problems
    if getattr(config, "use_sliding_window", False) or "sliding_attention" in (
            getattr(config, "layer_types", None) or ()):
        problems.append("uses sliding-window attention; the kernels run full causal attention.")
    if getattr(config, "hidden_act", "silu") != "silu":
        problems.append(
            f"uses the {config.hidden_act!r} activation; the kernels run SwiGLU (silu)."
        )
    heads = getattr(config, "num_attention_heads", 0)
    kv_heads = getattr(config, "num_key_value_heads", 0)
    hidden = getattr(config, "hidden_size", 0)
    head_dim = getattr(config, "head_dim", None) or (hidden // heads if heads else 0)
    if not heads or not kv_heads or heads % kv_heads or kv_heads & (kv_heads - 1):
        problems.append(
            f"has {heads} attention heads and {kv_heads} kv heads; the rotary kernel needs the "
            f"kv heads to divide the heads and to be a power of two."
        )
    if head_dim <= 0 or head_dim % 2:
        problems.append(f"has head size {head_dim}; the rotary kernel needs an even head size.")
    if getattr(config, "intermediate_size", 0) % 64:
        problems.append(
            f"has intermediate size {config.intermediate_size}; the int8 packs need a multiple "
            f"of 64."
        )
    return problems


#: What the kernels pass to ``aten._efficient_attention_forward`` / ``_backward``, in order. The
#: ops are private: a torch that renames, reorders or drops an argument is caught here, at
#: start-up, instead of failing in the middle of a step.
ATTENTION_FORWARD_ARGS = (
    "query", "key", "value", "bias", "cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q",
    "max_seqlen_k", "dropout_p", "custom_mask_type", "compute_log_sumexp",
)
ATTENTION_FORWARD_KWARGS = ("scale",)
ATTENTION_FORWARD_OUTPUTS = 6  # output, logsumexp, philox_seed, philox_offset, two lengths
ATTENTION_BACKWARD_ARGS = (
    "grad_out_", "query", "key", "value", "bias", "out", "cu_seqlens_q", "cu_seqlens_k",
    "max_seqlen_q", "max_seqlen_k", "logsumexp", "dropout_p", "philox_seed", "philox_offset",
    "custom_mask_type", "bias_requires_grad",
)
ATTENTION_BACKWARD_KWARGS = ("scale", "num_splits_key")
ATTENTION_BACKWARD_OUTPUTS = 4  # dq, dk, dv, dbias


def _signature_problem(name: str, schema: Any, positional: tuple[str, ...],
                       keyword: tuple[str, ...], outputs: int) -> Optional[str]:
    arguments = list(schema.arguments)
    have_positional = tuple(a.name for a in arguments if not a.kwarg_only)[:len(positional)]
    have_keyword = {a.name for a in arguments if a.kwarg_only}
    if have_positional != positional:
        return (f"torch's private aten.{name} takes ({', '.join(have_positional)}, ...); the "
                f"kernels call it with ({', '.join(positional)}).")
    missing = [k for k in keyword if k not in have_keyword]
    if missing:
        return f"torch's private aten.{name} has no keyword argument {', '.join(missing)}."
    if len(schema.returns) != outputs:
        return (f"torch's private aten.{name} returns {len(schema.returns)} values; the "
                f"kernels read {outputs}.")
    return None


def attention_signature_problems() -> list[str]:
    """Reasons the installed torch's private memory-efficient attention ops do not have the
    signatures the kernels call. Needs torch; no GPU."""
    try:
        import torch

        forward = torch.ops.aten._efficient_attention_forward.default._schema
        backward = torch.ops.aten._efficient_attention_backward.default._schema
    except (ImportError, AttributeError) as exc:
        return [f"torch's private memory-efficient attention ops are not available ({exc})."]
    found = (
        _signature_problem("_efficient_attention_forward", forward, ATTENTION_FORWARD_ARGS,
                           ATTENTION_FORWARD_KWARGS, ATTENTION_FORWARD_OUTPUTS),
        _signature_problem("_efficient_attention_backward", backward, ATTENTION_BACKWARD_ARGS,
                           ATTENTION_BACKWARD_KWARGS, ATTENTION_BACKWARD_OUTPUTS),
    )
    return [problem for problem in found if problem]


def host_memory_problems(needed_bytes: int, available_bytes: Optional[int]) -> list[str]:
    """Reasons the pinned host memory the plan asks for is not there. ``available_bytes`` is
    the machine's free RAM (``None``: not measurable, nothing to say)."""
    if available_bytes is None or needed_bytes <= available_bytes:
        return []
    gib = 1 << 30
    return [
        f"needs about {needed_bytes / gib:.1f} GiB of pinned host memory for the offloaded "
        f"activations and the host-parked tables; {available_bytes / gib:.1f} GiB of RAM is "
        f"free. A host that swaps trains at a fraction of the speed: raise "
        f"training.packed_lora.gpu_budget_mib or lower the batch size."
    ]


def gpu_memory_problems(floor_bytes: int, free_bytes: Optional[int]) -> list[str]:
    """Reasons the GPU cannot hold the plan's floor right now."""
    if free_bytes is None or floor_bytes <= free_bytes:
        return []
    mib = 1 << 20
    return [
        f"needs about {floor_bytes // mib} MiB of free GPU memory for its floor (4-bit weights, "
        f"LoRA state and the step's transients); {free_bytes // mib} MiB is free."
    ]
