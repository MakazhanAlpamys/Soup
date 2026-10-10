"""#1618 — ``auto_mixed_precision`` is applied by ``task='sft'`` (and 'tts') only.

The field loaded on all 23 tasks while only the SFT trainer read it
(sft.py:_resolve_mixed_precision, inherited by tts.py through super()), so
every other run trained at its trainer's default precision with nothing said.
The task gate now refuses the field outside ``AMP_APPLYING_TASKS``; these
tests pin the refusal per task, the applying set's continued acceptance, and
the ``false`` default's continued acceptance on every task.
"""

from __future__ import annotations

import typing

import pytest
import yaml
from pydantic import ValidationError

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import SoupConfig

# The minimum each task needs to load at all (the same table as PR #1532's tests).
_TOP = {"tts": {"modality": "audio_out"}}
_DATA = {
    "dpo": {"format": "dpo"},
    "kto": {"format": "kto"},
    "pretrain": {"format": "plaintext"},
    "unlearn": {"forget_set": "./f.jsonl"},
}
_TRAINING = {
    "preference": {"preference_loss": "dpo"},
    "tts": {"tts_family": "orpheus"},
    "classifier": {"num_labels": 2},
    "reranker": {"num_labels": 2},
    "cross_encoder": {"num_labels": 2},
    "unlearn": {"unlearn_method": "npo"},
    "moe_lora_routing": {"mole_task_adapters": ["./a", "./b"]},
    "distill": {"teacher_model": "org/t"},
    "online_dpo": {"online_dpo_judge": "pairrm"},
}

TASKS = typing.get_args(SoupConfig.model_fields["task"].annotation)
# Listed literally, not read from AMP_APPLYING_TASKS: dropping a task from the
# constant must not also drop its control here.
APPLYING = ["sft", "tts"]


def _config_for(task: str, training_extra: dict) -> str:
    raw = {
        "base": "org/m",
        "task": task,
        **_TOP.get(task, {}),
        "data": {"train": "./x.jsonl", **_DATA.get(task, {})},
        "training": {**_TRAINING.get(task, {}), **training_extra},
    }
    return yaml.safe_dump(raw)


@pytest.mark.parametrize("task", [t for t in TASKS if t not in APPLYING])
def test_amp_true_is_refused_outside_the_applying_tasks(task):
    with pytest.raises((ValidationError, ValueError)) as exc:
        load_config_from_string(_config_for(task, {"auto_mixed_precision": True}))
    msg = str(exc.value)
    assert "auto_mixed_precision" in msg
    assert task in msg


@pytest.mark.parametrize("task", [t for t in TASKS if t not in APPLYING])
def test_the_same_configs_load_without_the_field(task):
    cfg = load_config_from_string(_config_for(task, {}))
    assert cfg.task == task


@pytest.mark.parametrize("task", APPLYING)
def test_the_applying_tasks_still_load_with_amp_true(task):
    cfg = load_config_from_string(_config_for(task, {"auto_mixed_precision": True}))
    assert cfg.training.auto_mixed_precision is True
    assert cfg.task == task


@pytest.mark.parametrize("task", TASKS)
def test_amp_false_loads_on_every_task(task):
    cfg = load_config_from_string(_config_for(task, {"auto_mixed_precision": False}))
    assert cfg.training.auto_mixed_precision is False
