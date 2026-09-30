"""Issue #1325: ``training.expand_layers`` must refuse unsupported routes.

LLaMA Pro block expansion is applied only by the transformers text
SFT/pretrain ``_setup_transformers`` paths. Every other task, backend or
modality accepted the fields and trained without them; these tests pin the
load-time refusal and the two accepted routes.

Every refused config is first loaded WITHOUT ``expand_layers``, so a refusal
can only come from ``expand_layers`` and not from something else wrong with
the config. ``quantization: none`` keeps the scope refusal separate from the
quantized-base refusal (#1237).
"""

from __future__ import annotations

import typing

import pytest

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import SoupConfig

_SUPPORTED_TASKS = ("sft", "pretrain")
_ALL_TASKS = typing.get_args(SoupConfig.model_fields["task"].annotation)
_OTHER_TASKS = tuple(task for task in _ALL_TASKS if task not in _SUPPORTED_TASKS)

# The least each task needs to load on backend: transformers at all. A new task
# that needs more fails the baseline load below, which names it.
_ROOT = {"tts": "modality: audio_out\n"}
_DATA = {
    "asr": "  format: asr\n",
    "unlearn": "  forget_set: forget.jsonl\n  retain_set: retain.jsonl\n",
}
_TRAINING = {
    "preference": "  preference_loss: dpo\n",
    "online_dpo": "  online_dpo_judge: https://judge.example.com/v1\n",
    "ppo": "  reward_model: org/reward\n",
    "distill": "  teacher_model: meta-llama/Llama-3.1-70B\n",
    "tts": "  tts_family: orpheus\n",
    "classifier": "  num_labels: 2\n",
    "reranker": "  num_labels: 2\n",
    "cross_encoder": "  num_labels: 2\n",
    "unlearn": "  unlearn_method: npo\n",
    "moe_lora_routing": "  mole_task_adapters:\n  - ./a1\n  - ./a2\n",
}
_EXPAND = "  expand_layers: 2\n  freeze_trainable_layers: 2\n"


def _yaml(task, backend="transformers", modality=None, expand=True):
    root = f"modality: {modality}\n" if modality else _ROOT.get(task, "")
    return (
        "base: meta-llama/Llama-3.1-8B\n"
        f"task: {task}\n"
        f"backend: {backend}\n"
        f"{root}"
        "data:\n  train: data.jsonl\n"
        + _DATA.get(task, "")
        + "training:\n  quantization: none\n"
        + _TRAINING.get(task, "")
        + (_EXPAND if expand else "")
    )


def test_the_task_list_comes_from_the_schema():
    assert len(_OTHER_TASKS) == len(_ALL_TASKS) - 2
    assert "dpo" in _OTHER_TASKS and "sft" not in _OTHER_TASKS


@pytest.mark.parametrize("task", _OTHER_TASKS)
def test_refused_on_every_other_task(task):
    load_config_from_string(_yaml(task, expand=False))
    with pytest.raises(ValueError, match=rf"expand_layers.*task='{task}'"):
        load_config_from_string(_yaml(task))


@pytest.mark.parametrize(
    "task,backend",
    [("sft", "unsloth"), ("pretrain", "unsloth"), ("sft", "mlx")],
)
def test_refused_off_transformers(task, backend):
    load_config_from_string(_yaml(task, backend=backend, expand=False))
    with pytest.raises(ValueError, match=rf"expand_layers.*backend='{backend}'"):
        load_config_from_string(_yaml(task, backend=backend))


@pytest.mark.parametrize("modality", ["vision", "audio", "audio_out"])
@pytest.mark.parametrize("task", _SUPPORTED_TASKS)
def test_refused_off_text(task, modality):
    load_config_from_string(_yaml(task, modality=modality, expand=False))
    with pytest.raises(ValueError, match=rf"expand_layers.*modality='{modality}'"):
        load_config_from_string(_yaml(task, modality=modality))


@pytest.mark.parametrize("task", _SUPPORTED_TASKS)
def test_supported_routes_still_load(task):
    cfg = load_config_from_string(_yaml(task, modality="text"))
    assert cfg.training.expand_layers == 2
    assert cfg.training.freeze_trainable_layers == 2
