"""FSDP2 (Fully Sharded Data Parallel) configuration templates.

FSDP2 is PyTorch's native distributed training solution, an alternative to DeepSpeed.
It shards model parameters, gradients, and optimizer states across GPUs with
tighter integration into PyTorch's autograd engine.

FSDP2 advantages over DeepSpeed:
- Native PyTorch (no external dependency)
- Better composability with torch.compile
- Simpler configuration for most use cases
- Built-in mixed precision via torch.amp

Requires: torch >= 2.2.0, accelerate >= 0.27.0
"""

from __future__ import annotations

import copy
from functools import update_wrapper
from types import FunctionType, MethodType
from typing import Any

# FSDP2 Full Shard: shards params + gradients + optimizer states (like ZeRO-3)
FSDP_FULL_SHARD = {
    "fsdp": "full_shard auto_wrap",
    "fsdp_config": {
        "backward_prefetch": "backward_pre",
        "forward_prefetch": True,
        "use_orig_params": True,
        "limit_all_gathers": True,
        "sync_module_states": True,
    },
}

# FSDP2 Shard Grad Op: shards gradients + optimizer states only (like ZeRO-2)
FSDP_SHARD_GRAD_OP = {
    "fsdp": "shard_grad_op auto_wrap",
    "fsdp_config": {
        "backward_prefetch": "backward_pre",
        "forward_prefetch": True,
        "use_orig_params": True,
        "limit_all_gathers": True,
        "sync_module_states": True,
    },
}

# FSDP2 Full Shard with CPU offload (memory-constrained setups)
FSDP_FULL_SHARD_OFFLOAD = {
    "fsdp": "full_shard auto_wrap offload",
    "fsdp_config": {
        "backward_prefetch": "backward_pre",
        "forward_prefetch": True,
        "use_orig_params": True,
        "limit_all_gathers": True,
        "sync_module_states": True,
    },
}

FSDP_CONFIGS = {
    "full_shard": FSDP_FULL_SHARD,
    "shard_grad": FSDP_SHARD_GRAD_OP,
    "full_offload": FSDP_FULL_SHARD_OFFLOAD,
}


def prepare_fsdp2_tied_embeddings(trainer: Any) -> bool:
    """Keep frozen tied input/output weights in the same public FSDP2 group.

    Accelerate 1.14 can carve the embedding into a tail group while the tied
    head remains in the root. Torch 2.13 rejects that at its first forward.
    RAM-efficient loading must capture the full weights before joint placement.
    In that mode install an instance-local preparation hook at Accelerate's
    post-capture auto-wrap boundary; other modes keep the measured eager path.
    """
    accelerator = getattr(trainer, "accelerator", None)
    plugin = getattr(getattr(accelerator, "state", None), "fsdp_plugin", None)
    if getattr(plugin, "fsdp_version", None) != 2:
        return False
    if getattr(accelerator, "_soup_fsdp2_tied_embeddings", False):
        return False
    model = trainer.model
    if not _has_frozen_tied_embeddings(model):
        return False
    if getattr(plugin, "cpu_ram_efficient_loading", False):
        _defer_fsdp2_tied_embeddings(accelerator)
        return True
    return _shard_fsdp2_tied_embeddings(accelerator, model)


def _has_frozen_tied_embeddings(model: Any) -> bool:
    embed = getattr(model, "get_input_embeddings", lambda: None)()
    head = getattr(model, "get_output_embeddings", lambda: None)()
    weight = getattr(embed, "weight", None)
    if weight is None or weight is not getattr(head, "weight", None):
        return False
    return not any(parameter.requires_grad for module in (embed, head)
                   for parameter in module.parameters())


def _shard_fsdp2_tied_embeddings(accelerator: Any, model: Any) -> bool:
    if not _has_frozen_tied_embeddings(model):
        return False
    embed = model.get_input_embeddings()
    head = model.get_output_embeddings()
    plugin = accelerator.state.fsdp_plugin

    from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard

    if isinstance(embed, FSDPModule) or isinstance(head, FSDPModule):
        return False
    mesh = getattr(accelerator, "torch_device_mesh", None)
    if mesh is not None:
        mesh = mesh[tuple(accelerator.parallelism_config.fsdp_dim_names)]
    fully_shard(
        [embed, head],
        mesh=mesh,
        mp_policy=getattr(plugin, "mixed_precision_policy", None) or MixedPrecisionPolicy(),
        offload_policy=getattr(plugin, "cpu_offload", None),
        reshard_after_forward=False,
    )
    return True


def _clone_with_globals(function: Any, replacements: dict[str, Any]) -> Any:
    """Reuse upstream code with local bindings, without patching its module."""
    if not isinstance(function, FunctionType) or not all(
        name in function.__code__.co_names and name in function.__globals__
        for name in replacements
    ):
        raise RuntimeError(
            "Unsupported Accelerate FSDP2 preparation API for RAM-efficient tied weights. "
            "Disable cpu_ram_efficient_loading or use the validated Accelerate 1.14 stack."
        )
    bindings = {**function.__globals__, **replacements}
    clone = FunctionType(function.__code__, bindings, function.__name__,
                         function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = function.__kwdefaults__
    return update_wrapper(clone, function)


def _defer_fsdp2_tied_embeddings(accelerator: Any) -> None:
    """Place tied modules after full-state capture and meta conversion.

    Accelerate 1.14's auto-wrap-policy factory is called after both operations
    and before any child sharding. Rebind that factory in a copy of the exact
    upstream preparation function, then bind its caller to this accelerator
    only. Other accelerators and Accelerate module globals remain untouched.
    """
    from accelerate.utils import fsdp_utils

    prepare = getattr(getattr(accelerator, "_prepare_fsdp2", None), "__func__", None)
    upstream = getattr(prepare, "__globals__", {}).get("fsdp2_prepare_model")
    factory = getattr(upstream, "__globals__", {}).get("fsdp2_prepare_auto_wrap_policy")
    if factory is None or factory is not getattr(
        fsdp_utils, "fsdp2_prepare_auto_wrap_policy", None,
    ):
        raise RuntimeError(
            "Unsupported Accelerate FSDP2 auto-wrap-policy boundary. "
            "Disable cpu_ram_efficient_loading or use the validated Accelerate 1.14 stack."
        )

    def prepare_policy(plugin: Any, model: Any) -> Any:
        if plugin is accelerator.state.fsdp_plugin:
            _shard_fsdp2_tied_embeddings(accelerator, model)
        return factory(plugin, model)

    local_prepare_model = _clone_with_globals(
        upstream, {"fsdp2_prepare_auto_wrap_policy": prepare_policy},
    )
    local_prepare = _clone_with_globals(prepare, {"fsdp2_prepare_model": local_prepare_model})
    accelerator._prepare_fsdp2 = MethodType(local_prepare, accelerator)
    accelerator._soup_fsdp2_tied_embeddings = True


def get_fsdp_config(preset: str) -> dict:
    """Get FSDP config dict by preset name.

    Args:
        preset: One of 'full_shard', 'shard_grad', 'full_offload'.

    Returns:
        Deep copy of the FSDP config dict.

    Raises:
        ValueError: If preset is not recognized.
    """
    if preset not in FSDP_CONFIGS:
        raise ValueError(
            f"Unknown FSDP config: {preset}. "
            f"Options: {', '.join(FSDP_CONFIGS.keys())}"
        )
    return copy.deepcopy(FSDP_CONFIGS[preset])


def get_fsdp_training_args(preset: str) -> dict:
    """Get FSDP kwargs to pass to TrainingArguments.

    Args:
        preset: FSDP preset name.

    Returns:
        Dict of kwargs to unpack into TrainingArguments.
    """
    config = get_fsdp_config(preset)
    return {
        "fsdp": config["fsdp"],
        "fsdp_config": config["fsdp_config"],
    }


def apply_fsdp_training_kwargs(
    training_kwargs: dict,
    fsdp_config: dict | None,
    use_fsdp2_compile: bool,
) -> dict:
    """Mutate and return ``training_kwargs`` with FSDP + optional torch.compile.

    Centralizes the "FSDP-block" logic from the trainer wrappers so it can
    be unit-tested directly without mocking an entire model load.

    Args:
        training_kwargs: The dict being built for ``TrainingArguments``.
        fsdp_config: Either ``None`` or a dict with keys ``fsdp`` /
            ``fsdp_config`` (as returned by :func:`get_fsdp_training_args`).
        use_fsdp2_compile: Whether ``training.use_fsdp2_compile`` is set.

    Returns:
        The same ``training_kwargs`` dict (mutated in place).

    Raises:
        ValueError: If ``fsdp_config`` contains unexpected keys.
    """
    if not fsdp_config:
        return training_kwargs

    allowed = {"fsdp", "fsdp_config"}
    unexpected = set(fsdp_config.keys()) - allowed
    if unexpected:
        raise ValueError(f"Unexpected FSDP config keys: {unexpected}")
    training_kwargs.update(fsdp_config)
    if use_fsdp2_compile:
        training_kwargs["torch_compile"] = True
    return training_kwargs


def is_fsdp_available() -> bool:
    """Check if FSDP2 requirements are met (torch >= 2.2, accelerate >= 0.27)."""
    try:
        import torch

        parts = torch.__version__.split(".")[:2]
        torch_version = tuple(
            int(p.split("+")[0].split("a")[0].split("b")[0].split("rc")[0])
            for p in parts
        )
        if torch_version < (2, 2):
            return False
    except (ImportError, ValueError):
        return False

    try:
        import accelerate  # noqa: F401

        return True
    except ImportError:
        return False


def validate_fsdp2_compile_config(
    use_compile: bool,
    fsdp_preset: str | None,
    backend: str,
    device: str,
    deepspeed_config: str | None = None,
) -> list[str]:
    """Validate FSDP2 + ``torch.compile`` combination.

    Args:
        use_compile: Whether ``training.use_fsdp2_compile`` is enabled.
        fsdp_preset: FSDP preset name, or None if FSDP is disabled.
        backend: Training backend (transformers / unsloth / mlx).
        device: Training device (cuda / cpu / mps).
        deepspeed_config: DeepSpeed config path, or None. If set together with
            ``use_compile``, we reject — DeepSpeed owns its own compile path and
            combining the two produces a cryptic runtime error.

    Returns:
        List of error messages. Empty list means valid.
    """
    if not use_compile:
        return []

    errors: list[str] = []
    if deepspeed_config:
        errors.append(
            "use_fsdp2_compile is incompatible with --deepspeed. "
            "DeepSpeed owns its own torch.compile integration; "
            "mixing the two crashes at runtime. Pick one."
        )
    if not fsdp_preset:
        errors.append(
            "use_fsdp2_compile requires FSDP to be enabled. "
            "Pass --fsdp full_shard (or shard_grad / full_offload)."
        )
    if device != "cuda":
        errors.append(
            f"use_fsdp2_compile requires CUDA GPUs. Current device: {device}."
        )
    if backend != "transformers":
        errors.append(
            f"use_fsdp2_compile is only supported with backend=transformers "
            f"(got {backend!r}); unsloth bakes its own compile path."
        )
    if fsdp_preset and not is_fsdp_available():
        errors.append(
            "use_fsdp2_compile requires torch >= 2.2.0 and accelerate >= 0.27.0. "
            "Upgrade with: pip install -U torch accelerate"
        )
    return errors


def validate_fsdp_config(
    fsdp_preset: str | None,
    deepspeed_config: str | None,
    backend: str,
    device: str,
) -> list[str]:
    """Validate FSDP configuration and return error messages.

    Args:
        fsdp_preset: FSDP preset name, or None if not using FSDP.
        deepspeed_config: DeepSpeed config path, or None.
        backend: Training backend (transformers/unsloth).
        device: Training device (cuda/cpu/mps).

    Returns:
        List of error messages. Empty list means valid.
    """
    errors: list[str] = []

    if not fsdp_preset:
        return errors

    if deepspeed_config:
        errors.append(
            "Cannot use FSDP and DeepSpeed together. Choose one: "
            "--fsdp or --deepspeed."
        )

    if device != "cuda":
        errors.append(
            "FSDP requires CUDA GPUs. "
            f"Current device: {device}."
        )

    if backend == "unsloth":
        errors.append(
            "FSDP is not compatible with the unsloth backend. "
            "Use backend: transformers."
        )

    if not is_fsdp_available():
        errors.append(
            "FSDP2 requires torch >= 2.2.0 and accelerate >= 0.27.0. "
            "Upgrade with: pip install -U torch accelerate"
        )

    if fsdp_preset not in FSDP_CONFIGS:
        errors.append(
            f"Unknown FSDP preset: {fsdp_preset}. "
            f"Options: {', '.join(FSDP_CONFIGS.keys())}"
        )

    return errors
