"""Unsloth notebook (.ipynb) → Soup config migration.

Uses AST parsing only — never exec/eval on notebook code.
"""

import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Trainer class → Soup task mapping
_TRAINER_MAP = {
    "SFTTrainer": "sft",
    "DPOTrainer": "dpo",
    "GRPOTrainer": "grpo",
    "KTOTrainer": "kto",
    "ORPOTrainer": "orpo",
    "PPOTrainer": "ppo",
}

# Trainer config class → Soup task mapping
_CONFIG_MAP = {
    "DPOConfig": "dpo",
    "GRPOConfig": "grpo",
    "KTOConfig": "kto",
    "ORPOConfig": "orpo",
    "PPOConfig": "ppo",
}

# Task → default data format
_TASK_FORMAT_MAP = {
    "sft": "auto",
    "dpo": "dpo",
    "grpo": "auto",
    "kto": "kto",
    "orpo": "dpo",
    "ppo": "auto",
}


def migrate_unsloth(notebook_path: Path) -> Dict[str, Any]:
    """Parse an Unsloth .ipynb notebook and return a Soup config dict.

    Extracts parameters from function calls using AST parsing only.
    Returns a dict suitable for config_to_yaml(). Includes a ``_warnings``
    key with a list of human-readable migration notes.
    """
    try:
        raw_text = notebook_path.read_text(encoding="utf-8")
        notebook = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in notebook file: {exc}")

    cells = notebook.get("cells", [])
    code_cells = [
        cell for cell in cells
        if cell.get("cell_type") == "code"
    ]

    if not code_cells:
        raise ValueError("No code cells found in notebook")

    # Combine all code cell sources for AST parsing
    all_source = ""
    for cell in code_cells:
        source = cell.get("source", [])
        if isinstance(source, list):
            all_source += "".join(source) + "\n"
        else:
            all_source += source + "\n"

    # Parse AST — safe, no execution
    try:
        tree = ast.parse(all_source)
    except SyntaxError:
        raise ValueError("Could not parse notebook code (SyntaxError)")

    warnings: List[str] = []
    base: Optional[str] = None
    max_seq_length: Optional[int] = None
    load_in_4bit: Optional[bool] = None
    lora_params: Dict[str, Any] = {}
    training_params: Dict[str, Any] = {}
    task = "sft"
    output_dir = "./output"

    # Walk AST to extract function call arguments
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        func_name = _get_func_name(node)
        if func_name is None:
            continue

        if func_name == "from_pretrained":
            # FastLanguageModel.from_pretrained(...)
            kwargs = _extract_kwargs(node)
            if "model_name" in kwargs:
                base = kwargs["model_name"]
            if "max_seq_length" in kwargs:
                max_seq_length = kwargs["max_seq_length"]
            if "load_in_4bit" in kwargs:
                load_in_4bit = kwargs["load_in_4bit"]

        elif func_name == "get_peft_model":
            # FastLanguageModel.get_peft_model(...)
            kwargs = _extract_kwargs(node)
            if "r" in kwargs:
                lora_params["r"] = kwargs["r"]
            if "lora_alpha" in kwargs:
                lora_params["alpha"] = kwargs["lora_alpha"]
            if "lora_dropout" in kwargs:
                lora_params["dropout"] = kwargs["lora_dropout"]
            if "target_modules" in kwargs:
                lora_params["target_modules"] = kwargs["target_modules"]
            if "use_dora" in kwargs:
                lora_params["use_dora"] = kwargs["use_dora"]
            if "use_rslora" in kwargs and kwargs["use_rslora"]:
                lora_params["use_rslora"] = True

        elif func_name in _TRAINER_MAP:
            # SFTTrainer(...), DPOTrainer(...), etc.
            task = _TRAINER_MAP[func_name]
            kwargs = _extract_kwargs(node)
            if kwargs.get("packing"):
                warnings.append(
                    "packing=True is not supported in Soup. "
                    "Sequences will be padded individually."
                )

        elif func_name == "TrainingArguments" or func_name in _CONFIG_MAP:
            # TrainingArguments(...), DPOConfig(...), etc.
            if func_name in _CONFIG_MAP:
                task = _CONFIG_MAP[func_name]
            kwargs = _extract_kwargs(node)
            if "per_device_train_batch_size" in kwargs:
                training_params["batch_size"] = kwargs["per_device_train_batch_size"]
            if "num_train_epochs" in kwargs:
                training_params["epochs"] = kwargs["num_train_epochs"]
            if "learning_rate" in kwargs:
                training_params["lr"] = kwargs["learning_rate"]
            if "optim" in kwargs:
                training_params["optimizer"] = kwargs["optim"]
            if "lr_scheduler_type" in kwargs:
                training_params["scheduler"] = kwargs["lr_scheduler_type"]
            if "output_dir" in kwargs:
                output_dir = kwargs["output_dir"]
            if "beta" in kwargs:
                if task == "dpo":
                    training_params["dpo_beta"] = kwargs["beta"]
                elif task == "kto":
                    training_params["kto_beta"] = kwargs["beta"]
            if "max_steps" in kwargs:
                warnings.append(
                    f"max_steps={kwargs['max_steps']}. "
                    "Soup uses epochs; set training.epochs instead."
                )

    if base is None:
        raise ValueError("No FastLanguageModel.from_pretrained() call found in notebook")

    # Build result
    data_format = _TASK_FORMAT_MAP.get(task, "auto")
    training: Dict[str, Any] = {**training_params}

    if load_in_4bit:
        training["quantization"] = "4bit"

    if lora_params:
        training["lora"] = lora_params

    data: Dict[str, Any] = {
        "train": "./data/train.jsonl",
        "format": data_format,
    }
    if max_seq_length:
        data["max_length"] = max_seq_length

    warnings.append("data.train is a placeholder — set the actual dataset path.")

    result: Dict[str, Any] = {
        "base": base,
        "task": task,
        "data": data,
        "training": training,
        "output": output_dir,
        "_warnings": warnings,
    }

    return result


def _get_func_name(node: ast.Call) -> Optional[str]:
    """Extract the function name from a Call node."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    elif isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _extract_kwargs(node: ast.Call) -> Dict[str, Any]:
    """Extract keyword arguments from a function Call node as Python values.

    Only extracts simple literal values (str, int, float, bool, list, None).
    """
    result: Dict[str, Any] = {}
    for kw in node.keywords:
        if kw.arg is None:
            continue  # **kwargs — skip
        value = _ast_to_value(kw.value)
        if value is not _SENTINEL:
            result[kw.arg] = value
    return result


class _SentinelType:
    """Sentinel for values that cannot be extracted."""
    pass


_SENTINEL = _SentinelType()


def _ast_to_value(node: ast.AST) -> Any:
    """Convert an AST node to a Python value. Returns _SENTINEL if not a literal."""
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.List):
        items = [_ast_to_value(elt) for elt in node.elts]
        if any(isinstance(item, _SentinelType) for item in items):
            return _SENTINEL
        return items
    elif isinstance(node, ast.Tuple):
        items = [_ast_to_value(elt) for elt in node.elts]
        if any(isinstance(item, _SentinelType) for item in items):
            return _SENTINEL
        return items
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        val = _ast_to_value(node.operand)
        if isinstance(val, _SentinelType):
            return _SENTINEL
        return -val
    elif isinstance(node, ast.NameConstant):  # Python 3.7 compat
        return node.value
    return _SENTINEL
