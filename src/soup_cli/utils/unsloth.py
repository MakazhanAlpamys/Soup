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


def resolve_unsloth_gradient_checkpointing(
    gradient_checkpointing: bool | str | None, *, stream_layers: bool = False
) -> bool | str:
    """Map ``training.gradient_checkpointing`` to Unsloth's ``use_gradient_checkpointing`` value.

    Unsloth's ``get_peft_model`` takes either ``"unsloth"`` (its own optimized
    checkpointing kernel) or ``False`` to disable checkpointing
    entirely. This follows the same rule as the HF path
    (``should_enable_hf_gradient_checkpointing``): checkpointing is off when the
    config disables it or when layer streaming already checkpoints every layer.

    The resolver is imported lazily — ``layer_stream`` pulls in ``rich`` at module
    top, and ``unsloth`` helpers must stay import-light (repo rule: heavy deps
    lazy-imported, never at module top).
    """
    from soup_cli.utils.layer_stream import should_enable_hf_gradient_checkpointing

    if should_enable_hf_gradient_checkpointing(gradient_checkpointing, stream_layers=stream_layers):
        return "unsloth"
    return False


def load_model_and_tokenizer(
    model_name: str,
    max_seq_length: int,
    quantization: str = "4bit",
    lora_r: int = 64,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: str | list[str] | None = "auto",
    use_rslora: bool = False,
    use_gradient_checkpointing: bool | str = "unsloth",
):
    """Load model + tokenizer via unsloth FastLanguageModel with LoRA already applied.

    Returns (model, tokenizer) — model already has LoRA adapters attached.
    Unsloth handles quantization, LoRA patching, and kernel optimization internally.

    ``use_rslora`` is ``lora.use_rslora``: Unsloth's ``get_peft_model`` takes it
    directly and scales the adapter by ``alpha / sqrt(r)`` instead of
    ``alpha / r``. It is forwarded on every call so the config's value, not
    Unsloth's default, decides (#1853: the flag loaded and was never passed).

    ``use_gradient_checkpointing`` is ``training.gradient_checkpointing`` as
    resolved by ``resolve_unsloth_gradient_checkpointing``: ``"unsloth"`` enables
    Unsloth's checkpointing kernel, ``False`` disables checkpointing entirely
    (#797: the flag was hardcoded to ``"unsloth"``, so disabling it was ignored).
    """
    if quantization == "8bit":
        # #1811: the config refuses this pairing; name it here too rather than
        # loading the model in 16-bit if the loader is ever reached another way.
        raise ValueError(
            "quantization 8bit is not supported by the Unsloth loader "
            "(it passes load_in_4bit only); use 4bit or none"
        )

    from unsloth import FastLanguageModel

    load_in_4bit = quantization == "4bit"

    # Unsloth's FastLanguageModel.from_pretrained builds its OWN BitsAndBytesConfig
    # internally and hardcodes bnb_4bit_use_double_quant=True; it exposes no
    # passthrough (no double-quant kwarg, no quantization_config=). So
    # training.bnb_4bit_use_double_quant cannot be honoured on the Unsloth path —
    # #321 covers only the paths where Soup itself constructs the BnB config
    # (resident load, streamed load, save/merge). Threading the flag here would
    # be silently discarded, so it is deliberately not passed.
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
        use_gradient_checkpointing=use_gradient_checkpointing,  # #797: honoured, not hardcoded
        use_rslora=use_rslora,
    )

    return model, tokenizer
