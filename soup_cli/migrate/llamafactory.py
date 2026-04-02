"""LLaMA-Factory config → Soup config migration."""

from pathlib import Path
from typing import Any, Dict, List

import yaml

from soup_cli.migrate.common import to_number

# LLaMA-Factory stage → Soup task mapping
_STAGE_MAP = {
    "sft": "sft",
    "pt": "pretrain",
    "rm": "reward_model",
    "dpo": "dpo",
    "kto": "kto",
    "ppo": "ppo",
}

# LLaMA-Factory pref_loss → Soup task override (when stage=dpo)
_PREF_LOSS_MAP = {
    "sigmoid": "dpo",
    "orpo": "orpo",
    "simpo": "simpo",
}

# Task → default data format
_TASK_FORMAT_MAP = {
    "sft": "auto",
    "pretrain": "plaintext",
    "dpo": "dpo",
    "kto": "kto",
    "orpo": "dpo",
    "simpo": "dpo",
    "ipo": "dpo",
    "reward_model": "dpo",
    "ppo": "auto",
}


def migrate_llamafactory(config_path: Path) -> Dict[str, Any]:
    """Parse a LLaMA-Factory YAML config and return a Soup config dict.

    Returns a dict suitable for config_to_yaml(). Includes a ``_warnings``
    key with a list of human-readable migration notes.
    """
    raw_text = config_path.read_text(encoding="utf-8")
    raw = yaml.safe_load(raw_text)
    if not raw or not isinstance(raw, dict):
        raise ValueError("Config file is empty or not a valid YAML mapping")

    if "model_name_or_path" not in raw:
        raise ValueError("Missing required key: model_name_or_path")

    warnings: List[str] = []

    # --- Base model ---
    base = raw["model_name_or_path"]

    # --- Task ---
    stage = raw.get("stage", "sft")
    task = _STAGE_MAP.get(stage, "sft")

    # Override task if pref_loss is specified (LF unifies under stage:dpo)
    pref_loss = raw.get("pref_loss")
    if pref_loss and pref_loss in _PREF_LOSS_MAP:
        task = _PREF_LOSS_MAP[pref_loss]

    data_format = _TASK_FORMAT_MAP.get(task, "auto")

    # --- LoRA ---
    finetuning_type = raw.get("finetuning_type", "lora")
    include_lora = finetuning_type == "lora"

    if finetuning_type == "freeze":
        warnings.append(
            "finetuning_type: freeze is not supported in Soup. "
            "Using LoRA instead."
        )
        include_lora = True

    if finetuning_type == "full":
        warnings.append(
            "finetuning_type: full — no LoRA will be used. "
            "Soup will train all parameters."
        )

    lora_section = {}
    if include_lora:
        lora_section["r"] = raw.get("lora_rank", 64)
        lora_section["alpha"] = raw.get("lora_alpha", 16)
        if "lora_dropout" in raw:
            lora_section["dropout"] = raw["lora_dropout"]
        # lora_target: "all" → auto
        lora_target = raw.get("lora_target")
        if lora_target == "all" or lora_target is None:
            lora_section["target_modules"] = "auto"
        else:
            lora_section["target_modules"] = lora_target
        if raw.get("use_dora"):
            lora_section["use_dora"] = True

    # --- Training ---
    training: Dict[str, Any] = {}
    if "num_train_epochs" in raw:
        training["epochs"] = raw["num_train_epochs"]
    if "learning_rate" in raw:
        training["lr"] = to_number(raw["learning_rate"])
    if "per_device_train_batch_size" in raw:
        training["batch_size"] = raw["per_device_train_batch_size"]
    if "gradient_accumulation_steps" in raw:
        training["gradient_accumulation_steps"] = raw["gradient_accumulation_steps"]
    if "lr_scheduler_type" in raw:
        training["scheduler"] = raw["lr_scheduler_type"]
    if "warmup_ratio" in raw:
        training["warmup_ratio"] = raw["warmup_ratio"]

    # Quantization
    quant_bit = raw.get("quantization_bit")
    if quant_bit == 4:
        training["quantization"] = "4bit"
    elif quant_bit == 8:
        training["quantization"] = "8bit"
    elif quant_bit is not None:
        training["quantization"] = "none"

    # Task-specific params
    pref_beta = raw.get("pref_beta")
    if pref_beta is not None:
        if task == "dpo":
            training["dpo_beta"] = pref_beta
        elif task == "kto":
            training["kto_beta"] = pref_beta
        elif task == "orpo":
            training["orpo_beta"] = pref_beta

    if "reward_model" in raw:
        training["reward_model"] = raw["reward_model"]

    # LoRA+
    if "loraplus_lr_ratio" in raw:
        training["loraplus_lr_ratio"] = raw["loraplus_lr_ratio"]

    # Add lora section
    if include_lora and lora_section:
        training["lora"] = lora_section

    # --- Data ---
    data: Dict[str, Any] = {}
    dataset_name = raw.get("dataset")
    if dataset_name:
        data["train"] = f"./{dataset_name}.jsonl"
        warnings.append(
            f"Dataset '{dataset_name}' is a LLaMA-Factory dataset registry name. "
            "You need to provide the actual file path in data.train."
        )
    else:
        data["train"] = "./data/train.jsonl"
        warnings.append("No dataset specified. Using placeholder path ./data/train.jsonl")

    data["format"] = data_format
    if "cutoff_len" in raw:
        data["max_length"] = raw["cutoff_len"]

    # --- Output ---
    output = raw.get("output_dir", "./output")

    # --- Comments for unmapped fields ---
    if raw.get("bf16") or raw.get("fp16"):
        warnings.append("bf16/fp16 is auto-detected in Soup (no manual setting needed)")
    if raw.get("deepspeed"):
        warnings.append(
            f"DeepSpeed config: {raw['deepspeed']}. Use --deepspeed flag with soup train."
        )
    if raw.get("report_to"):
        report_to = raw["report_to"]
        warnings.append(f"report_to: {report_to}. Use --wandb or --tensorboard flag.")
    if raw.get("neftune_noise_alpha") is not None:
        warnings.append(
            f"neftune_noise_alpha: {raw['neftune_noise_alpha']}. "
            "Add training.neftune_alpha in soup.yaml if supported."
        )
    template_name = raw.get("template")
    if template_name:
        warnings.append(f"Original template: {template_name} (auto-detected in Soup)")

    result: Dict[str, Any] = {
        "base": base,
        "task": task,
        "data": data,
        "training": training,
        "output": output,
        "_warnings": warnings,
    }

    return result
