"""Import-light validation for the native Sesame CSM trainer (#265)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from soup_cli.config.schema import SoupConfig


def validate_csm_config(cfg: SoupConfig) -> None:
    """Refuse text-SFT substitutes and knobs the native trainer cannot honour."""
    if cfg.backend != "transformers":
        raise ValueError("Sesame CSM requires backend='transformers'")
    if cfg.data.format != "audio":
        raise ValueError(
            "Sesame CSM requires data.format='audio' with a transcript and a raw audio clip; "
            "pre-encoded codec-string chat is not a native CSM training example."
        )
    tcfg = cfg.training
    if tcfg.quantization != "none":
        raise ValueError("Sesame CSM requires training.quantization='none'")
    if not isinstance(tcfg.gradient_checkpointing, bool):
        raise ValueError("Sesame CSM supports boolean training.gradient_checkpointing only")

    # This is a separate Trainer path: accepting an SFT optimization without
    # implementing it would silently ignore an operator's requested behaviour.
    supported = {
        "epochs", "lr", "batch_size", "gradient_accumulation_steps", "seed", "data_seed",
        "warmup_ratio", "weight_decay", "max_grad_norm", "optimizer", "scheduler",
        "save_steps", "logging_steps", "eval_steps", "lora", "quantization", "tts_family",
        "gradient_checkpointing",
    }
    defaults = type(tcfg)().model_dump()
    for name, value in tcfg.model_dump().items():
        if name not in supported and value != defaults[name]:
            raise ValueError(f"Sesame CSM does not support training.{name}")
    lora_defaults = type(tcfg.lora)().model_dump()
    for name, value in tcfg.lora.model_dump().items():
        if name not in {"r", "alpha", "dropout"} and value != lora_defaults[name]:
            raise ValueError(f"Sesame CSM does not support training.lora.{name}")
    data_defaults = type(cfg.data)(train=cfg.data.train).model_dump()
    for name, value in cfg.data.model_dump().items():
        if name not in {"train", "format", "val_split", "max_length", "audio_dir"}:
            if value != data_defaults[name]:
                raise ValueError(f"Sesame CSM does not support data.{name}")


def validate_csm_row(row: object) -> tuple[str, str, str]:
    """One clip and its exact transcript, with a native speaker role (0 or 1).

    Multiple messages paired with just one audio path are ambiguous: no
    transcript/context concatenation or speaker inference is performed.
    """
    if not isinstance(row, dict):
        raise ValueError("CSM row must be a dict")
    path = row.get("audio")
    if not isinstance(path, str) or not path.strip() or "\x00" in path:
        raise ValueError("CSM row needs a non-empty local audio path")
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 1:
        raise ValueError("CSM row needs exactly one transcript message for its audio clip")
    message = messages[0]
    if not isinstance(message, dict) or set(message) != {"role", "content"}:
        raise ValueError("CSM transcript message must contain only role and content")
    speaker = message["role"]
    if not isinstance(speaker, str) or speaker not in {"0", "1"}:
        raise ValueError("CSM transcript role must be the speaker id '0' or '1'")
    text = message["content"]
    if not isinstance(text, str) or not text.strip() or "\x00" in text:
        raise ValueError("CSM transcript content must be non-empty text")
    return path, speaker, text


def validate_csm_dataset(dataset: dict) -> None:
    """Validate every paired transcript before downloading weights or taking a step."""
    for split in ("train", "val"):
        for index, row in enumerate(dataset.get(split) or []):
            try:
                validate_csm_row(row)
            except ValueError as exc:
                raise ValueError(f"CSM {split} row {index + 1}: {exc}") from exc
    if not dataset.get("train"):
        raise ValueError("CSM requires at least one training row")
