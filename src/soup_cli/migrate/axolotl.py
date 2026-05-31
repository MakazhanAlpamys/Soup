"""Axolotl config → Soup config migration."""

from pathlib import Path
from typing import Any, Dict, List

import yaml

from soup_cli.migrate.common import to_number

# Axolotl rl → Soup task mapping
_RL_MAP = {
    "dpo": "dpo",
    "kto": "kto",
    "grpo": "grpo",
    "gdpo": "grpo",
}

# Axolotl dataset type → Soup format
_FORMAT_MAP = {
    "alpaca": "alpaca",
    "sharegpt": "sharegpt",
    "chat_template": "chatml",
    "completion": "auto",
}

# Optimizer name mapping
_OPTIMIZER_MAP = {
    "adamw_8bit": "adamw_bnb_8bit",
}

# Task → default data format
_TASK_FORMAT_MAP = {
    "sft": None,  # Use dataset type
    "dpo": "dpo",
    "kto": "kto",
    "grpo": "auto",
}


def migrate_axolotl(config_path: Path) -> Dict[str, Any]:
    """Parse an Axolotl YAML config and return a Soup config dict.

    Returns a dict suitable for config_to_yaml(). Includes a ``_warnings``
    key with a list of human-readable migration notes.
    """
    raw_text = config_path.read_text(encoding="utf-8")
    raw = yaml.safe_load(raw_text)
    if not raw or not isinstance(raw, dict):
        raise ValueError("Config file is empty or not a valid YAML mapping")

    if "base_model" not in raw:
        raise ValueError("Missing required key: base_model")

    warnings: List[str] = []

    # --- Base model ---
    base = raw["base_model"]

    # --- Task ---
    rl_type = raw.get("rl")
    task = _RL_MAP.get(rl_type, "sft") if rl_type else "sft"

    # --- Data ---
    data: Dict[str, Any] = {}
    datasets = raw.get("datasets", [])
    if datasets:
        first_ds = datasets[0]
        data["train"] = first_ds.get("path", "./data/train.jsonl")
        ds_type = first_ds.get("type", "auto")
        # Use task-specific format or dataset type
        task_fmt = _TASK_FORMAT_MAP.get(task)
        if task_fmt:
            data["format"] = task_fmt
        else:
            data["format"] = _FORMAT_MAP.get(ds_type, ds_type)
        if len(datasets) > 1:
            warnings.append(
                f"Multiple datasets found ({len(datasets)}). "
                "Only the first dataset was imported. "
                "Soup uses a single data.train path — merge datasets with "
                "'soup data merge' first."
            )
    else:
        data["train"] = "./data/train.jsonl"
        data["format"] = "auto"
        warnings.append("No datasets specified. Using placeholder path.")

    if "sequence_len" in raw:
        data["max_length"] = raw["sequence_len"]

    # --- LoRA ---
    adapter = raw.get("adapter")
    lora_section: Dict[str, Any] = {}
    include_lora = adapter in ("lora", "qlora")

    if include_lora:
        if "lora_r" in raw:
            lora_section["r"] = raw["lora_r"]
        if "lora_alpha" in raw:
            lora_section["alpha"] = raw["lora_alpha"]
        if "lora_dropout" in raw:
            lora_section["dropout"] = raw["lora_dropout"]
        if raw.get("lora_target_linear"):
            lora_section["target_modules"] = "auto"
        elif "lora_target_modules" in raw:
            lora_section["target_modules"] = raw["lora_target_modules"]

    # --- Training ---
    training: Dict[str, Any] = {}
    if "micro_batch_size" in raw:
        training["batch_size"] = raw["micro_batch_size"]
    if "gradient_accumulation_steps" in raw:
        training["gradient_accumulation_steps"] = raw["gradient_accumulation_steps"]
    if "num_epochs" in raw:
        training["epochs"] = raw["num_epochs"]
    if "learning_rate" in raw:
        training["lr"] = to_number(raw["learning_rate"])
    if "optimizer" in raw:
        opt = raw["optimizer"]
        training["optimizer"] = _OPTIMIZER_MAP.get(opt, opt)
    if "lr_scheduler" in raw:
        training["scheduler"] = raw["lr_scheduler"]
    if "warmup_ratio" in raw:
        training["warmup_ratio"] = raw["warmup_ratio"]

    # Quantization
    if adapter == "qlora" or raw.get("load_in_4bit"):
        training["quantization"] = "4bit"
    elif raw.get("load_in_8bit"):
        training["quantization"] = "8bit"

    # Flash attention
    if raw.get("flash_attention"):
        training["use_flash_attn"] = True

    # Add lora section
    if include_lora and lora_section:
        training["lora"] = lora_section

    # --- Output ---
    output = raw.get("output_dir", "./output")

    # --- Unsupported features ---
    if raw.get("sample_packing"):
        warnings.append(
            "sample_packing is not supported in Soup. "
            "Sequences will be padded individually."
        )
    if raw.get("wandb_project"):
        warnings.append(
            f"wandb_project: {raw['wandb_project']}. Use --wandb flag with soup train."
        )

    result: Dict[str, Any] = {
        "base": base,
        "task": task,
        "data": data,
        "training": training,
        "output": output,
        "_warnings": warnings,
    }

    return result
