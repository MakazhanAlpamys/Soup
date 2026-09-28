"""Issue #1325: ``training.expand_layers`` must refuse unsupported routes.

LLaMA Pro block expansion is wired only in the transformers text
SFT/pretrain ``_setup_transformers`` paths.  Every other task, backend or
modality previously accepted the fields and silently trained the unexpanded
model; these tests pin the load-time refusal and the two accepted routes.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from soup_cli.config.loader import load_config_from_string


def _yaml(task: str, backend: str = "transformers", modality: str = "") -> str:
    modality_line = f"modality: {modality}\n" if modality else ""
    return (
        "base: meta-llama/Llama-3.1-8B\n"
        f"task: {task}\n"
        f"backend: {backend}\n"
        f"{modality_line}"
        "data:\n  train: data.jsonl\n"
        "training:\n"
        "  expand_layers: 2\n"
        "  freeze_trainable_layers: 2\n"
    )


@pytest.mark.parametrize(
    "task",
    ["dpo", "kto", "grpo", "reward_model", "embedding"],
)
def test_expand_layers_refused_off_sft_pretrain_tasks(task: str):
    with pytest.raises((ValidationError, ValueError), match="expand_layers"):
        load_config_from_string(_yaml(task))


def test_expand_layers_refused_on_unsloth_backend():
    with pytest.raises((ValidationError, ValueError), match="backend='unsloth'"):
        load_config_from_string(_yaml("sft", backend="unsloth"))


def test_expand_layers_refused_on_vision_modality():
    with pytest.raises((ValidationError, ValueError), match="modality='vision'"):
        load_config_from_string(_yaml("sft", modality="vision"))


def test_expand_layers_accepted_on_supported_routes():
    for task in ("sft", "pretrain"):
        cfg = load_config_from_string(_yaml(task))
        assert cfg.training.expand_layers == 2


def test_error_message_names_actual_route():
    with pytest.raises((ValidationError, ValueError), match="task='dpo'"):
        load_config_from_string(_yaml("dpo"))
