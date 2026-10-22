"""Portable validation for the qualified native token-distillation slice."""

from soup_cli.trainer.mlx_optim import plan_optimizer


def validate_native_distill(config) -> None:
    """Refuse unsupported, non-default modifiers before loading either model.

    Defaults for other tasks have no distillation meaning. Explicit non-default
    modifiers must be consumed here or refused; they must not silently change
    into the plain token objective. New common settings require qualification.
    """
    tcfg = config.training
    supported = {
        "epochs", "batch_size", "gradient_accumulation_steps", "lr", "optimizer",
        "scheduler", "warmup_ratio", "weight_decay", "max_grad_norm", "gradient_checkpointing",
        "seed", "data_seed", "logging_steps", "save_steps", "eval_steps", "lora",
        "teacher_model", "distill_divergence", "distill_temperature", "distill_chunk_size",
        "distill_checkpoint", "quantization", "rewind_log",
    }
    defaults = type(tcfg)().model_dump()
    refused = [f"training.{key}" for key, value in tcfg.model_dump().items()
               if key not in supported and value != defaults[key]]
    if config.modality != "text":
        refused.append("modality (only text causal models)")
    if tcfg.distill_mode != "token":
        refused.append("training.distill_mode (only token)")
    if tcfg.quantization != "none":
        refused.append("training.quantization (only none)")
    if tcfg.lora.r < 1:
        refused.append("training.lora.r (ordinary LoRA is required)")
    lora_defaults = type(tcfg.lora)().model_dump()
    refused.extend(f"training.lora.{key}" for key, value in tcfg.lora.model_dump().items()
                   if key not in {"r", "alpha", "dropout", "target_modules"}
                   and value != lora_defaults[key])
    if tcfg.optimizer not in {"adamw", "adamw_torch", "adamw_hf"}:
        refused.append("training.optimizer (qualified AdamW aliases: adamw, adamw_torch, adamw_hf)")
    if not isinstance(tcfg.gradient_checkpointing, bool):
        refused.append("training.gradient_checkpointing (only bool)")
    refused.extend(f"data.{name}" for name in (
        "mask_history", "train_on_messages_with_train_field", "train_on_prompt", "chat_template",
        "tokenized_path", "streaming", "skip_prepare_dataset", "remove_unused_columns",
        "prompt_strategy", "extend_conversation", "split_thinking", "eval_on_each_dataset",
    ) if getattr(config.data, name))
    if refused:
        raise ValueError("MLX token distillation does not support: " + ", ".join(refused))
    plan_optimizer(
        lr=tcfg.lr, optimizer=tcfg.optimizer, scheduler=tcfg.scheduler,
        warmup_ratio=tcfg.warmup_ratio, weight_decay=tcfg.weight_decay, total_updates=1,
    )
