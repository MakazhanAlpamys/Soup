"""GaLore (Gradient Low-Rank Projection) — memory-efficient full-parameter training."""


def get_galore_optimizer_and_params(
    galore_rank: int = 128,
    galore_update_proj_gap: int = 200,
    galore_scale: float = 0.25,
) -> dict:
    """Build GaLore optimizer kwargs for HuggingFace Trainer.

    Returns dict of kwargs to pass to TrainingArguments.

    GaLore applies low-rank gradient projection to linear layers,
    reducing optimizer memory from O(n*m) to O(n*r + m*r) where r << min(n,m).
    """
    # Enforce types defensively — protect against upstream coercion bypasses
    rank = int(galore_rank)
    gap = int(galore_update_proj_gap)
    scale = float(galore_scale)
    if rank < 1 or gap < 1 or scale <= 0:
        raise ValueError("Invalid GaLore parameters: rank/gap must be >=1, scale must be >0")

    return {
        "optim": "galore_adamw",
        "optim_target_modules": ["attn", "mlp"],
        "optim_args": f"rank={rank},update_proj_gap={gap},scale={scale}",
    }


def validate_galore_config(
    use_galore: bool,
    quantization: str,
    backend: str,
) -> list[str]:
    """Validate GaLore configuration and return list of error messages."""
    errors: list[str] = []

    if not use_galore:
        return errors

    if quantization in ("4bit", "8bit"):
        errors.append(
            "GaLore is incompatible with quantization. "
            "Set quantization: none when using GaLore."
        )

    if backend == "unsloth":
        errors.append(
            "GaLore is not compatible with the unsloth backend. "
            "Use backend: transformers."
        )

    return errors
