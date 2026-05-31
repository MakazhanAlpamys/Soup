"""v0.44.0 Part B — Studio-style onboarding wizard.

Pure-Python: takes the answers as a dict and renders a complete `soup.yaml`.
The interactive prompt loop lives in `commands/init.py` and calls
`render_onboarding_yaml(answers)` here.

Five questions:
1. base model (HF repo id or local path)
2. dataset (local JSONL path or HF dataset name)
3. task (sft / dpo / preference)
4. quantization (4bit / 8bit / none)
5. epochs (1-10)
"""

from __future__ import annotations

from typing import Any, Dict

import yaml

from soup_cli.utils.paths import is_under_cwd

VALID_TASKS = frozenset(
    {"sft", "dpo", "kto", "orpo", "simpo", "ipo", "bco", "preference"}
)
VALID_QUANT = frozenset({"4bit", "8bit", "none"})

_MAX_BASE_LEN = 256
_MAX_DATASET_LEN = 512
_MAX_OUTPUT_LEN = 512


def _check_string(value: Any, *, field: str, max_len: int) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be str")
    if not value:
        raise ValueError(f"{field} must be non-empty")
    if "\x00" in value:
        raise ValueError(f"{field} contains NUL byte")
    if len(value) > max_len:
        raise ValueError(f"{field} exceeds {max_len} chars")
    return value


def render_onboarding_yaml(answers: Dict[str, Any]) -> str:
    """Render a complete `soup.yaml` from a 5-answer dict.

    Required keys: base, dataset, task, quantization, epochs.
    Optional: output (default `./out`), batch_size (default `auto`).
    """
    if not isinstance(answers, dict):
        raise TypeError("answers must be dict")
    base = _check_string(answers.get("base"), field="base", max_len=_MAX_BASE_LEN)
    dataset = _check_string(
        answers.get("dataset"), field="dataset", max_len=_MAX_DATASET_LEN
    )
    task = answers.get("task")
    if task not in VALID_TASKS:
        raise ValueError(
            f"task must be one of {sorted(VALID_TASKS)}; got {task!r}"
        )
    quant = answers.get("quantization", "4bit")
    if quant not in VALID_QUANT:
        raise ValueError(
            f"quantization must be one of {sorted(VALID_QUANT)}; got {quant!r}"
        )
    epochs = answers.get("epochs")
    if isinstance(epochs, bool) or not isinstance(epochs, int):
        raise TypeError("epochs must be int")
    if not (1 <= epochs <= 10):
        raise ValueError("epochs must be in [1, 10]")
    output = _check_string(
        answers.get("output", "./out"), field="output", max_len=_MAX_OUTPUT_LEN
    )
    if not is_under_cwd(output):
        # Match the project policy of leaking only the basename in errors.
        import os

        raise ValueError(
            f"output must stay under cwd: {os.path.basename(output)}"
        )
    batch_size: Any = answers.get("batch_size", "auto")
    if isinstance(batch_size, bool):
        raise TypeError("batch_size must be int or 'auto'")
    if not (batch_size == "auto" or (isinstance(batch_size, int) and batch_size > 0)):
        raise ValueError("batch_size must be a positive int or 'auto'")
    config = {
        "base": base,
        "task": task,
        "data": {"train": dataset, "format": "auto"},
        "training": {
            "epochs": epochs,
            "lr": 2e-4,
            "batch_size": batch_size,
            "quantization": quant,
        },
        "output": output,
    }
    return yaml.safe_dump(config, sort_keys=False, default_flow_style=False)
