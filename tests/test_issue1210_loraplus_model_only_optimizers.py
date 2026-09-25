"""#1210 — LoRA+ with an optimizer transformers can only build from the model.

`attach_loraplus_optimizer` asks `Trainer.get_optimizer_cls_and_kwargs(args)` for the
optimizer without a model. For apollo_adamw, lomo and adalomo transformers needs one,
so such a config passed load and failed only after the base model had loaded. It is
now refused at config load, as is LoRA+ with GaLore.

Every case runs under several tasks: the refusal is a TrainingConfig check and must not
depend on the task, which a task-scoped refactor could otherwise break unnoticed.
"""

from __future__ import annotations

import pytest

from soup_cli.config.loader import load_config_from_string

_BASE = """
base: some-org/some-model
task: {task}
data:
  train: ./train.jsonl
training:
  lora:
    r: 8
    alpha: 16
"""

_TASKS = ["sft", "dpo", "ppo"]


def _config(task: str, **training: object) -> str:
    lines = "".join(f"  {key}: {value}\n" for key, value in training.items())
    return _BASE.format(task=task) + lines


@pytest.mark.parametrize("task", _TASKS)
@pytest.mark.parametrize("optimizer", ["apollo_adamw", "lomo", "adalomo"])
def test_loraplus_with_a_model_only_optimizer_is_refused_at_load(task, optimizer):
    with pytest.raises(ValueError) as caught:
        load_config_from_string(_config(task, loraplus_lr_ratio=16.0, optimizer=optimizer))
    message = str(caught.value)
    assert "training.loraplus_lr_ratio" in message
    assert f"training.optimizer={optimizer!r}" in message


@pytest.mark.parametrize("task", _TASKS)
def test_optimizer_name_is_matched_case_insensitively(task):
    with pytest.raises(ValueError, match="training.loraplus_lr_ratio"):
        load_config_from_string(_config(task, loraplus_lr_ratio=16.0, optimizer="LOMO"))


@pytest.mark.parametrize("task", _TASKS)
def test_loraplus_with_galore_is_refused_at_load(task):
    with pytest.raises(ValueError) as caught:
        load_config_from_string(_config(task, loraplus_lr_ratio=16.0, use_galore="true"))
    message = str(caught.value)
    assert "training.loraplus_lr_ratio" in message
    assert "training.use_galore" in message


@pytest.mark.parametrize("task", _TASKS)
@pytest.mark.parametrize(
    "training",
    [
        {"loraplus_lr_ratio": 16.0},
        {"loraplus_lr_ratio": 16.0, "optimizer": "adamw_torch"},
        {"loraplus_lr_ratio": 16.0, "optimizer": "adamw_8bit"},
        {"optimizer": "apollo_adamw"},
        {"optimizer": "lomo"},
        {"optimizer": "adalomo"},
        {"use_galore": "true"},
    ],
    ids=[
        "loraplus-default-optimizer",
        "loraplus-adamw_torch",
        "loraplus-adamw_8bit",
        "apollo_adamw-alone",
        "lomo-alone",
        "adalomo-alone",
        "galore-alone",
    ],
)
def test_each_setting_alone_still_loads(task, training):
    # Controls: only the combination is refused.
    load_config_from_string(_config(task, **training))
