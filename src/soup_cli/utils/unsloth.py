"""Unsloth backend utilities — detection, model loading, LoRA patching."""

from __future__ import annotations


def is_unsloth_available() -> bool:
    """Check if unsloth is installed and importable."""
    try:
        import unsloth  # noqa: F401

        return True
    except ImportError:
        return False


def get_unsloth_version() -> str | None:
    """Return unsloth version string, or None if not installed."""
    try:
        import unsloth

        return getattr(unsloth, "__version__", "unknown")
    except ImportError:
        return None


def load_model_and_tokenizer(
    model_name: str,
    max_seq_length: int,
    quantization: str = "4bit",
    lora_r: int = 64,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: str | list[str] | None = "auto",
):
    """Load model + tokenizer via unsloth FastLanguageModel with LoRA already applied.

    Returns (model, tokenizer) — model already has LoRA adapters attached.
    Unsloth handles quantization, LoRA patching, and kernel optimization internally.
    """
    from unsloth import FastLanguageModel

    load_in_4bit = quantization == "4bit"

    # Unsloth's FastLanguageModel.from_pretrained handles quantization internally
    dtype = None  # auto-detect
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    # Resolve target_modules for LoRA
    if target_modules == "auto" or target_modules is None:
        # Unsloth default: all linear layers for maximum performance
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    elif isinstance(target_modules, str):
        target_modules = [target_modules]

    # Apply LoRA via unsloth's optimized path
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",  # 2x longer context for free
    )

    return model, tokenizer
