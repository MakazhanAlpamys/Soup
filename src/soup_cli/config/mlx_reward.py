"""The native reward loop's applied fields, checked after existing schema refusals."""

from __future__ import annotations

import math
from typing import Any

# The loop and its shared SFT LoRA/optimizer helpers consume these fields.
# Other changed training settings must not silently become no-ops (#2003).
APPLIED_TRAINING = frozenset(
    {
        "epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "lr",
        "optimizer",
        "scheduler",
        "warmup_ratio",
        "weight_decay",
        "max_grad_norm",
        "gradient_checkpointing",
        "save_steps",
        "logging_steps",
        "seed",
        "data_seed",
        "quantization",
        "lora",
        "eval_gate",
        "rewind_log",
    }
)
APPLIED_DATA = frozenset(
    {"train", "format", "val_split", "max_length", "interleave",
     "replay", "replay_ratio", "replay_seed", "streaming", "tokenized_path"}
)
APPLIED_LORA = frozenset({"r", "alpha", "dropout", "target_modules", "use_rslora"})
TARGETS = frozenset(
    {f"self_attn.{name}_proj" for name in ("q", "k", "v", "o")}
    | {f"mlp.{name}_proj" for name in ("gate", "up", "down")}
)


def _refuse(field: str, value: Any, allowed: str) -> None:
    from soup_cli.utils.mlx_reward_bundle import label

    raise ValueError(
        f"MLX reward_model does not support {field}={label(value)}. Set {field} to {allowed}."
    )


def validate_reward_scope(cfg: Any) -> None:
    """Extra restrictions describe this scalar dense slice, not a global MLX policy."""
    training, data = cfg.training, cfg.data
    if cfg.modality != "text":
        _refuse("modality", cfg.modality, "text (raw preference pairs)")
    if training.quantization != "none":
        _refuse("training.quantization", training.quantization, "none for a dense base")
    if training.rewind_log and "rewind_log" in training.model_fields_set:
        _refuse("training.rewind_log", True, "false; per-row token-loss recording is SFT-only")
    if "rewind_log" not in training.model_fields_set:
        cfg.training = training = training.model_copy(update={"rewind_log": False})
    if training.quantize_reward_model:
        _refuse("training.quantize_reward_model", True, "false")
    if training.eval_gate is not None and training.eval_gate.enabled:
        raise ValueError(
            "MLX reward_model does not support training.eval_gate.enabled. "
            "Set training.eval_gate.enabled: false; use the reward trainer validation metrics."
        )
    if cfg.eval is not None and cfg.eval.auto_eval:
        raise ValueError(
            "MLX reward_model does not support eval.auto_eval. "
            "Set eval.auto_eval: false; held-out preference evaluation runs in the reward trainer."
        )
    if data.tokenized_path is not None:
        _refuse("data.tokenized_path", data.tokenized_path, "null; supply raw preference rows")
    if data.format not in ("auto", "dpo"):
        _refuse("data.format", data.format, "auto or dpo (raw prompt/chosen/rejected strings)")
    if data.streaming:
        _refuse("data.streaming", True, "false; native reward epochs shuffle retained pairs")
    for name, info in type(data).model_fields.items():
        value = getattr(data, name)
        if name not in APPLIED_DATA and value != info.get_default(call_default_factory=True):
            _refuse(f"data.{name}", value, repr(info.get_default(call_default_factory=True)))
    if isinstance(training.gradient_checkpointing, str):
        _refuse("training.gradient_checkpointing", training.gradient_checkpointing, "true or false")
    lora = training.lora
    if not 1 <= lora.r <= 4096:
        _refuse("training.lora.r", lora.r, "an integer in [1, 4096]")
    if not math.isfinite(lora.alpha) or lora.alpha <= 0:
        _refuse("training.lora.alpha", lora.alpha, "a finite positive number")
    if not math.isfinite(lora.dropout) or not 0 <= lora.dropout < 1:
        _refuse("training.lora.dropout", lora.dropout, "a finite number in [0, 1)")
    from soup_cli.trainer.mlx_sft import resolve_mlx_target_keys

    targets = resolve_mlx_target_keys(lora)
    if not targets or len(set(targets)) != len(targets) or not set(targets) <= TARGETS:
        _refuse(
            "training.lora.target_modules",
            lora.target_modules,
            "auto or a nonempty unique subset of " + ", ".join(sorted(TARGETS)),
        )
    # rsLoRA and dropout are consumed by _apply_lora, exactly as for MLX SFT.
    # DoRA/alternative initialization are outside this ordinary-LoRA bundle slice.
    for name, info in type(lora).model_fields.items():
        value = getattr(lora, name)
        if name not in APPLIED_LORA and value != info.get_default(call_default_factory=True):
            _refuse(
                f"training.lora.{name}", value, repr(info.get_default(call_default_factory=True))
            )
    for name, info in type(training).model_fields.items():
        value = getattr(training, name)
        if name not in APPLIED_TRAINING and value != info.get_default(call_default_factory=True):
            _refuse(f"training.{name}", value, repr(info.get_default(call_default_factory=True)))
    # Keep exactly the shared SFT optimizer/schedule support and diagnostics.
    from soup_cli.trainer.mlx_optim import plan_optimizer

    plan_optimizer(
        lr=training.lr,
        optimizer=training.optimizer,
        scheduler=training.scheduler,
        warmup_ratio=training.warmup_ratio,
        weight_decay=training.weight_decay,
        total_updates=1,
    )
