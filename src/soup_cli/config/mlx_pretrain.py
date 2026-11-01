"""Import-light scope validation for the bounded native raw-corpus trainer."""

from __future__ import annotations

from pathlib import Path


def refuse_unsupported_pretrain(cfg) -> None:
    """Name settings that this loop cannot execute, before any model allocation."""
    if cfg.modality != "text":
        raise ValueError("MLX pretrain requires modality: text")
    data, training = cfg.data, cfg.training
    if data.format != "plaintext":
        raise ValueError("MLX pretrain requires data.format: plaintext")
    for name in ("train", "validation"):
        source = getattr(data, name)
        if source is not None and (
            not isinstance(source, str) or Path(source).suffix.lower() not in {".txt", ".jsonl"}
            or "://" in source
        ):
            raise ValueError(f"MLX pretrain data.{name} requires one local .txt or .jsonl file")
    if training.lora.r < 1:
        raise ValueError("MLX pretrain requires training.lora.r >= 1 (frozen base with LoRA)")
    if training.quantization not in {"none", "4bit"}:
        raise ValueError("MLX pretrain training.quantization accepts none or pre-quantized 4bit")
    # Only fields actually implemented by this loop (or consumed by its CLI).
    # Compare defaults so unrelated task knobs do not make every minimal config fail.
    supported = {
        "epochs", "lr", "batch_size", "gradient_accumulation_steps", "seed", "data_seed",
        "warmup_ratio", "weight_decay", "max_grad_norm", "lora", "quantization", "optimizer",
        "scheduler", "save_steps", "logging_steps", "eval_steps", "hub", "rewind_log",
    }
    data_fields = {"train", "validation", "format", "val_split", "max_length",
                   "train_on_responses_only"}
    lora_fields = {"r", "alpha", "dropout", "target_modules", "use_rslora"}
    for obj, allowed, prefix in ((data, data_fields, "data"),
                                  (training, supported, "training"),
                                  (training.lora, lora_fields, "training.lora")):
        for name, info in type(obj).model_fields.items():
            if name not in allowed and getattr(obj, name) != info.get_default(
                call_default_factory=True,
            ):
                raise ValueError(
                    f"MLX pretrain does not support {prefix}.{name}; remove it or use "
                    "backend: transformers"
                )
    if ("train_on_responses_only" in data.model_fields_set and data.train_on_responses_only):
        raise ValueError("MLX pretrain does not support data.train_on_responses_only: true")
    if "rewind_log" in training.model_fields_set and training.rewind_log:
        raise ValueError("MLX pretrain does not support training.rewind_log: true")
    # These are SFT defaults, not raw-corpus behavior. Resolve them in the
    # effective config so serializing a supported config remains loadable.
    if "train_on_responses_only" not in data.model_fields_set:
        cfg.data = data.model_copy(update={"train_on_responses_only": False})
    if "rewind_log" not in training.model_fields_set:
        cfg.training = training.model_copy(update={"rewind_log": False})
    from soup_cli.trainer.mlx_optim import plan_optimizer

    plan_optimizer(lr=training.lr, optimizer=training.optimizer, scheduler=training.scheduler,
                   warmup_ratio=training.warmup_ratio, weight_decay=training.weight_decay,
                   total_updates=1)
