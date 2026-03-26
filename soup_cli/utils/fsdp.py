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
    "fsdp_full_shard": FSDP_FULL_SHARD,
    "fsdp_shard_grad": FSDP_SHARD_GRAD_OP,
    "fsdp_full_offload": FSDP_FULL_SHARD_OFFLOAD,
}


def get_fsdp_config(preset: str) -> dict:
    """Get FSDP config dict by preset name.

    Args:
        preset: One of 'fsdp_full_shard', 'fsdp_shard_grad', 'fsdp_full_offload'.

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
