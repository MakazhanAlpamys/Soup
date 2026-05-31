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
