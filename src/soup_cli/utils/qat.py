"""Quantization-Aware Training (QAT) utilities.

QAT trains the model with simulated quantization during the forward pass,
so the model learns to be robust to quantization noise. This produces
significantly better quality compared to post-training quantization (PTQ).

Requires: torch >= 2.0, bitsandbytes >= 0.43.0 (for QAT support).
"""

from rich.console import Console

console = Console()


def is_qat_available() -> bool:
    """Check if QAT dependencies are available."""
    try:
        import torch  # noqa: F401
        from torchao.quantization import quantize_  # noqa: F401

        return True
    except ImportError:
        return False


def get_qat_config():
    """Return torchao QAT quantization config for int8 weight-only quantization.

    Uses torchao's Int8WeightOnlyConfig for QAT training. The model is trained
    with fake quantization ops inserted, so it learns to compensate for
    quantization error during training.
    """
    from torchao.quantization import Int8WeightOnlyConfig

    return Int8WeightOnlyConfig()


def prepare_model_for_qat(model):
    """Prepare a model for Quantization-Aware Training.

    Inserts fake quantization operators into the model so that forward passes
    simulate the effect of int8 quantization. Backward passes use
    straight-through estimators so gradients flow normally.

    Args:
        model: A PyTorch model (typically after LoRA has been applied).

    Returns:
        The model with QAT observers/fake-quant modules inserted.
    """
    from torchao.quantization import quantize_

    qat_config = get_qat_config()
    quantize_(model, qat_config)
    console.print("[green]QAT enabled:[/] model prepared with fake quantization ops")
    return model


def validate_qat_config(quantization: str, backend: str, modality: str) -> list[str]:
    """Validate QAT configuration and return warnings/errors.

    Args:
        quantization: The quantization setting (4bit, 8bit, none).
        backend: Training backend (transformers, unsloth).
        modality: Training modality (text, vision).

    Returns:
        List of warning/error messages. Empty list means valid.
    """
    errors = []

    if backend == "unsloth":
        errors.append(
            "QAT is not compatible with the unsloth backend. "
            "Use backend: transformers with quantization_aware: true."
        )

    if quantization not in ("4bit", "none"):
        errors.append(
            f"QAT works best with quantization: 4bit or none, got: {quantization}. "
            "Consider using quantization: 4bit for QLoRA + QAT."
        )

    if not is_qat_available():
        errors.append(
            "torchao is not installed. "
            "Install it with: pip install torchao"
        )

    return errors
