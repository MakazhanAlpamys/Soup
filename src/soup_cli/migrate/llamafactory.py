"""LLaMA-Factory config → Soup config migration."""

from pathlib import Path
from typing import Any, Dict, List

from soup_cli.migrate.common import (
    split_module_list,
    to_number,
    unmapped_keys_warning,
    validation_split,
)
from soup_cli.utils.yaml_limits import MAX_CONFIG_BYTES, load_yaml_file_limited

# LLaMA-Factory's own LoRA defaults (src/llamafactory/hparams/finetuning_args.py:
# ``lora_rank`` 8, ``lora_alpha = lora_alpha or lora_rank * 2``). A migrated config
# without the keys must train what LLaMA-Factory trained, not Soup's r 64 / alpha 16,
# which is an 8x larger adapter at an 8x smaller alpha / r (#1807).
_LF_DEFAULT_LORA_RANK = 8

# Every top-level key this migrator reads, mapped or warned about by name. Any
# other key in the source is listed in one "not migrated" warning (#1807).
_HANDLED_KEYS = frozenset({
    "model_name_or_path", "stage", "pref_loss", "finetuning_type", "lora_rank",
    "lora_alpha", "lora_dropout", "lora_target", "use_dora", "num_train_epochs",
    "learning_rate", "per_device_train_batch_size", "gradient_accumulation_steps",
    "lr_scheduler_type", "warmup_ratio", "quantization_bit", "quantization_method",
    "pref_beta", "simpo_gamma", "reward_model", "loraplus_lr_ratio", "dataset",
    "cutoff_len", "val_size", "output_dir", "bf16", "fp16", "deepspeed", "report_to",
    "neftune_noise_alpha", "template",
})

# LLaMA-Factory stage → Soup task mapping
_STAGE_MAP = {
    "sft": "sft",
    "pt": "pretrain",
    "rm": "reward_model",
    "dpo": "dpo",
    "kto": "kto",
    "ppo": "ppo",
}

# LLaMA-Factory pref_loss → Soup task override (when stage=dpo). #1214: a value
# with no Soup task (hinge, kto_pair) is refused by name, never defaulted to dpo.
_PREF_LOSS_MAP = {
    "sigmoid": "dpo",
    "ipo": "ipo",
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


def _lora_int(value: Any, key: str, warnings: List[str]) -> Any:
    """``lora_rank: "8"`` (quoted in the YAML) is the integer 8.

    LLaMA-Factory's parser does not coerce it, but the author meant 8, and the
    ``alpha = 2 * rank`` default below needs the integer. Anything else that is
    not an integer is copied as written and named, so the missing alpha default
    is not a silent one (#1807).
    """
    if isinstance(value, bool) or not isinstance(value, str):
        return value
    text = value.strip()
    if text.isdigit():
        return int(text)
    note = ", and lora_alpha was not defaulted from it" if key == "lora_rank" else ""
    warnings.append(f"{key}: {value!r} is not an integer; copied as written{note}.")
    return value


def migrate_llamafactory(config_path: Path) -> Dict[str, Any]:
    """Parse a LLaMA-Factory YAML config and return a Soup config dict.

    Returns a dict suitable for config_to_yaml(). Includes a ``_warnings``
    key with a list of human-readable migration notes.
    """
    raw = load_yaml_file_limited(
        config_path, f"Config file {config_path}", max_bytes=MAX_CONFIG_BYTES
    )
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

    # Override task if pref_loss is specified (LF unifies under stage:dpo, and
    # reads pref_loss only there).
    pref_loss = raw.get("pref_loss")
    if stage == "dpo" and pref_loss is not None:
        if not isinstance(pref_loss, str):
            raise ValueError(
                f"LLaMA-Factory pref_loss must name one loss as a string, got "
                f"{type(pref_loss).__name__}: {pref_loss!r}."
            )
        if pref_loss not in _PREF_LOSS_MAP:
            supported = ", ".join(sorted(_PREF_LOSS_MAP))
            raise ValueError(
                f"No Soup task matches LLaMA-Factory pref_loss: {pref_loss}. "
                f"Supported pref_loss values are {supported}."
            )
        task = _PREF_LOSS_MAP[pref_loss]

    data_format = _TASK_FORMAT_MAP.get(task, "auto")

    # --- LoRA / Full Fine-tuning ---
    finetuning_type = raw.get("finetuning_type", "lora")
    include_lora = finetuning_type == "lora"
    lora_section: Dict[str, Any] = {}

    if finetuning_type == "freeze":
        warnings.append(
            "finetuning_type: freeze is not supported in Soup. "
            "Using LoRA instead."
        )
        include_lora = True

    elif finetuning_type == "oft":
        warnings.append(
            "finetuning_type: oft is not supported in Soup. "
            "Using LoRA instead."
        )
        include_lora = True

    elif finetuning_type == "full":
        if task in ("sft", "embedding"):
            warnings.append(
                "finetuning_type: full — no LoRA will be used. "
                "Soup will train all parameters."
            )
            lora_section = {"r": 0}
        else:
            raise ValueError(
                f"finetuning_type: full is requested, but stage '{stage}' (task '{task}') "
                "does not support full fine-tuning in Soup. "
                "Only sft and embedding support full fine-tuning (lora.r=0)."
            )

    if include_lora:
        # LLaMA-Factory's defaults, not Soup's: r 8, alpha 2 * r (#1807).
        rank = raw.get("lora_rank")
        if rank is None:
            rank = _LF_DEFAULT_LORA_RANK
        else:
            rank = _lora_int(rank, "lora_rank", warnings)
        lora_section["r"] = rank
        alpha = raw.get("lora_alpha")
        if alpha is not None:
            alpha = _lora_int(alpha, "lora_alpha", warnings)
        if alpha is None and isinstance(rank, int) and not isinstance(rank, bool):
            alpha = 2 * rank
        if alpha is not None:
            lora_section["alpha"] = alpha
        if "lora_dropout" in raw:
            lora_section["dropout"] = raw["lora_dropout"]
        # lora_target: "all" (and LLaMA-Factory's default when the key is absent)
        # means every linear layer; peft's spelling of that is ``all-linear``.
        # ``auto`` would resolve to q_proj / v_proj on Llama, two of seven (#1807).
        lora_target = raw.get("lora_target")
        if lora_target == "all" or lora_target is None:
            lora_section["target_modules"] = "all-linear"
        else:
            lora_section["target_modules"] = split_module_list(lora_target)
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
    quant_bit_raw = raw.get("quantization_bit")
    quant_bit = None
    if quant_bit_raw is not None:
        if isinstance(quant_bit_raw, int):
            quant_bit = quant_bit_raw
        elif isinstance(quant_bit_raw, str) and quant_bit_raw.strip().isdigit():
            quant_bit = int(quant_bit_raw.strip())
        else:
            raise ValueError(f"Invalid quantization_bit value: {quant_bit_raw!r}")

    quant_method = raw.get("quantization_method")
    if isinstance(quant_method, str):
        quant_method = quant_method.lower().strip()

    if finetuning_type == "full":
        training["quantization"] = "none"
    elif quant_bit is None:
        training["quantization"] = "none"
    elif quant_method in (None, "bitsandbytes", "bnb"):
        if quant_bit == 4:
            training["quantization"] = "4bit"
        elif quant_bit == 8:
            training["quantization"] = "8bit"
        else:
            method_name = quant_method or "bitsandbytes"
            raise ValueError(
                f"Unsupported quantization_bit={quant_bit} for method '{method_name}'"
            )
    elif quant_method == "hqq":
        if quant_bit in (1, 2, 3, 4, 5, 6, 8):
            training["quantization"] = f"hqq:{quant_bit}bit"
        else:
            raise ValueError(
                f"Unsupported quantization_bit={quant_bit} for method 'hqq'"
            )
    elif quant_method == "eetq":
        if quant_bit == 8:
            training["quantization"] = "eetq"
        else:
            raise ValueError(
                f"Unsupported quantization_bit={quant_bit} for method 'eetq' (expected 8)"
            )
    else:
        raise ValueError(f"Unsupported quantization_method: '{quant_method}'")

    # Task-specific params
    pref_beta = raw.get("pref_beta")
    if pref_beta is not None:
        if task == "dpo":
            training["dpo_beta"] = pref_beta
        elif task == "kto":
            training["kto_beta"] = pref_beta
        elif task == "orpo":
            training["orpo_beta"] = pref_beta
        elif task == "ipo":
            # One knob under three names, shown rather than assumed (#1807):
            # LLaMA-Factory 1fded91 src/llamafactory/train/dpo/trainer.py:79-80 keeps
            # pref_beta as self.beta and pref_loss as self.loss_type, and for ipo
            # (:205-207, use_ref_model is True) hands both to trl's dpo_loss;
            # trl 0.29.1 trl/trainer/dpo_trainer.py:1241-1252 computes the IPO loss
            # as (delta - 1 / (2 * self.beta)) ** 2, and trl/trainer/dpo_config.py:
            # 252-256 documents that beta as IPO's regularisation parameter (tau);
            # Soup's trainer/ipo.py:173-174 passes training.ipo_tau as that beta.
            # All three default to 0.1 (finetuning_args.py:171-172, dpo_config.py:253,
            # schema.py ipo_tau), so an absent pref_beta stays absent here and Soup's
            # default applies.
            training["ipo_tau"] = pref_beta
        elif task == "simpo":
            # Soup's SimPO trainer leaves trl's beta at its default and has no
            # field for it, so the value cannot be carried; say so (#1807).
            warnings.append(
                f"pref_beta: {pref_beta} under pref_loss: simpo has no Soup equivalent "
                "(task: simpo takes training.simpo_gamma and training.cpo_alpha; trl's "
                "SimPO beta stays at its default). Dropped."
            )
        else:
            warnings.append(
                f"pref_beta: {pref_beta} is not read by LLaMA-Factory for stage: {stage}, "
                "and Soup has no equivalent on task: "
                f"{task}. Dropped."
            )
    simpo_gamma = raw.get("simpo_gamma")
    if simpo_gamma is not None:
        if task == "simpo":
            training["simpo_gamma"] = simpo_gamma
        else:
            warnings.append(
                f"simpo_gamma: {simpo_gamma} applies to pref_loss: simpo only "
                f"(this config migrates to task: {task}). Dropped."
            )

    if "reward_model" in raw:
        training["reward_model"] = raw["reward_model"]

    # LoRA+
    if "loraplus_lr_ratio" in raw:
        training["loraplus_lr_ratio"] = raw["loraplus_lr_ratio"]

    # Add lora section
    if lora_section:
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
    # val_size: LLaMA-Factory's default is 0 (train on every row); without this
    # mapping every migrated config got Soup's default of 0.1 (#1807).
    val_split = validation_split(raw.get("val_size"), "val_size", warnings)
    if val_split is not None:
        data["val_split"] = val_split

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
    dropped = unmapped_keys_warning(raw, _HANDLED_KEYS, "LLaMA-Factory")
    if dropped is not None:
        warnings.append(dropped)

    result: Dict[str, Any] = {
        "base": base,
        "task": task,
        "data": data,
        "training": training,
        "output": output,
        "_warnings": warnings,
    }

    return result
