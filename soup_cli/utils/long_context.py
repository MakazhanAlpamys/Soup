"""Long-context fine-tuning utilities — 128k+ token support.

Configures RoPE (Rotary Position Embedding) scaling to extend model context
windows beyond their pre-training length. Supports multiple scaling strategies:

- linear: Simple linear interpolation (PI) — good baseline
- dynamic: NTK-aware Dynamic scaling — better for large extensions
- yarn: YaRN (Yet another RoPE extensioN) — best quality for 4-8x extension
- longrope: LongRoPE — progressive extension with search-based factors

Also handles gradient checkpointing configuration for memory efficiency
when training on very long sequences.
"""

from __future__ import annotations

# Supported RoPE scaling methods
ROPE_SCALING_TYPES = ("linear", "dynamic", "yarn", "longrope")

# Default context lengths for known model families
MODEL_DEFAULT_CONTEXT: dict[str, int] = {
    "llama-3": 8192,
    "llama-2": 4096,
    "mistral": 32768,
    "mixtral": 32768,
    "qwen2": 32768,
    "qwen3": 32768,
    "phi-3": 4096,
    "phi-4": 16384,
    "gemma": 8192,
    "gemma-2": 8192,
    "deepseek": 4096,
    "codellama": 16384,
}


def get_model_default_context(model_name: str) -> int:
    """Estimate the default context length for a model based on its name.

    Args:
        model_name: HuggingFace model name/path.

    Returns:
        Estimated default context length in tokens.
    """
    model_lower = model_name.lower()
    for family, ctx_len in MODEL_DEFAULT_CONTEXT.items():
        if family in model_lower:
            return ctx_len
    # Conservative default for unknown models
    return 4096


def get_rope_scaling_config(
    scaling_type: str,
    target_length: int,
    original_length: int,
) -> dict:
    """Build RoPE scaling configuration for extending context.

    Args:
        scaling_type: One of 'linear', 'dynamic', 'yarn', 'longrope'.
        target_length: Desired context length (e.g., 131072 for 128k).
        original_length: Model's pre-trained context length.

    Returns:
        Dict to pass as `rope_scaling` in model config.

    Raises:
        ValueError: If scaling_type is not supported.
    """
    if scaling_type not in ROPE_SCALING_TYPES:
        raise ValueError(
            f"Unknown RoPE scaling type: {scaling_type}. "
            f"Options: {', '.join(ROPE_SCALING_TYPES)}"
        )

    factor = target_length / original_length
    if factor <= 1.0:
        # No scaling needed — target is within original context
        return {}

    if scaling_type == "linear":
        return {
            "type": "linear",
            "factor": float(factor),
        }
    elif scaling_type == "dynamic":
        return {
            "type": "dynamic",
            "factor": float(factor),
        }
    elif scaling_type == "yarn":
        return {
            "type": "yarn",
            "factor": float(factor),
            "original_max_position_embeddings": original_length,
        }
    else:  # longrope — guaranteed by Literal constraint in schema
        return {
            "type": "longrope",
            "factor": float(factor),
            "original_max_position_embeddings": original_length,
        }


def apply_long_context_config(
    model_config,
    target_length: int,
    rope_scaling_type: str = "dynamic",
    model_name: str = "",
) -> dict | None:
    """Apply long-context configuration to a model config object.

    Modifies the model config to extend the context window using RoPE scaling
    and returns the scaling config that was applied.

    Args:
        model_config: The model's config object (from model.config).
        target_length: Desired max sequence length.
        rope_scaling_type: RoPE scaling strategy.
        model_name: Model name (for estimating original context length).

    Returns:
        The rope_scaling dict that was applied, or None if no scaling needed.
    """
    # Determine original context length
    original_length = getattr(
        model_config,
        "max_position_embeddings",
        get_model_default_context(model_name),
    )

    if target_length <= original_length:
        return None

    rope_config = get_rope_scaling_config(
        scaling_type=rope_scaling_type,
        target_length=target_length,
        original_length=original_length,
    )

    if not rope_config:
        return None

    # Apply to model config
    model_config.rope_scaling = rope_config
    model_config.max_position_embeddings = target_length

    return rope_config


def validate_long_context_config(
    max_length: int,
    rope_scaling_type: str | None,
    use_gradient_checkpointing: bool,
) -> list[str]:
    """Validate long-context configuration.

    Args:
        max_length: Target sequence length.
        rope_scaling_type: RoPE scaling type, or None if not specified.
        use_gradient_checkpointing: Whether gradient checkpointing is enabled.

    Returns:
        List of warning/error messages.
    """
    errors: list[str] = []

    if rope_scaling_type and rope_scaling_type not in ROPE_SCALING_TYPES:
        errors.append(
            f"Unknown RoPE scaling type: {rope_scaling_type}. "
            f"Options: {', '.join(ROPE_SCALING_TYPES)}"
        )

    if max_length >= 65536 and not use_gradient_checkpointing:
        errors.append(
            f"Training with max_length={max_length} without gradient checkpointing "
            "will likely cause OOM. Set gradient_checkpointing: true in config."
        )

    return errors
