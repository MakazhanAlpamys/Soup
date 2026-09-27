"""#1264 -- ``training.moe_lora`` with ``backend: unsloth`` is refused with a reason
that is true for the task, and the refusals that apply on every backend come first.

Before the fix, one message ("unsloth attaches its own fixed attention targets")
was given on every task. It is false on the ten tasks whose trainer has no unsloth
setup at all: there ``backend: unsloth`` is what goes unapplied, and the flag is
read. And because the unsloth check ran first, ``asr`` / ``prm`` /
``moe_lora_routing`` and SFT vision / audio were told to switch to
``backend: transformers``, which only surfaced a second refusal.
"""

from __future__ import annotations

import importlib
import typing

import pytest
import yaml

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import SoupConfig

# What each task needs besides base / task / data.train to load at all.
_TOP = {"tts": {"modality": "audio_out"}}
_DATA = {
    "dpo": {"format": "dpo"}, "kto": {"format": "kto"}, "pretrain": {"format": "plaintext"},
    "unlearn": {"forget_set": "./f.jsonl"},
}
_TRAINING = {
    "preference": {"preference_loss": "dpo"},
    "tts": {"tts_family": "orpheus"},
    "classifier": {"num_labels": 2, "classifier_lora": True},
    "reranker": {"num_labels": 2, "classifier_lora": True},
    "cross_encoder": {"num_labels": 2, "classifier_lora": True},
    "unlearn": {"unlearn_method": "npo"},
    "moe_lora_routing": {"mole_task_adapters": ["./a", "./b"]},
    "distill": {"teacher_model": "org/t"},
    "online_dpo": {"online_dpo_judge": "pairrm"},
}

# The trainer each task dispatches to in commands/train.py.
_WRAPPERS = {
    "sft": "sft:SFTTrainerWrapper", "dpo": "dpo:DPOTrainerWrapper",
    "grpo": "grpo:GRPOTrainerWrapper", "ppo": "ppo:PPOTrainerWrapper",
    "reward_model": "reward_model:RewardModelTrainerWrapper",
    "kto": "kto:KTOTrainerWrapper", "orpo": "orpo:ORPOTrainerWrapper",
    "simpo": "simpo:SimPOTrainerWrapper", "ipo": "ipo:IPOTrainerWrapper",
    "bco": "bco:BCOTrainerWrapper", "preference": "preference:PreferenceTrainerWrapper",
    "pretrain": "pretrain:PretrainTrainerWrapper",
    "embedding": "embedding:EmbeddingTrainerWrapper", "prm": "prm:PRMTrainerWrapper",
    "tts": "tts:TTSTrainerWrapper", "classifier": "classifier:ClassifierTrainerWrapper",
    "reranker": "classifier:ClassifierTrainerWrapper",
    "cross_encoder": "classifier:ClassifierTrainerWrapper",
    "distill": "distill:DistillTrainerWrapper", "unlearn": "unlearn:UnlearnTrainerWrapper",
    "moe_lora_routing": "mole_routing:MoleRoutingTrainerWrapper",
    "online_dpo": "online_dpo:OnlineDPOTrainerWrapper", "asr": "asr:AsrTrainerWrapper",
}
# preference builds one of these per preference_loss (trainer/preference.py).
_PREFERENCE_DELEGATES = ("dpo", "simpo", "orpo", "ipo", "bco")

_WITH_UNSLOTH = (
    "sft", "dpo", "grpo", "ppo", "kto", "orpo", "simpo", "ipo", "bco", "preference",
    "pretrain", "embedding", "tts",
)
# No unsloth setup, and moe_lora is read on transformers: the unsloth refusal
# reaches these. prm / moe_lora_routing / asr get their task refusal instead.
_WITHOUT_UNSLOTH = (
    "reward_model", "classifier", "reranker", "cross_encoder", "unlearn", "distill",
    "online_dpo",
)
_TASK_REFUSED = ("prm", "moe_lora_routing", "asr")


def _yaml(task, backend, moe_lora=True, modality=None):
    top = dict(_TOP.get(task, {}))
    if modality:
        top["modality"] = modality
    data = {"train": "./x.jsonl", **_DATA.get(task, {})}
    if modality:
        data["format"] = {"vision": "llava", "audio": "audio"}[modality]
    training = {"moe_lora": moe_lora}
    if task != "prm":  # prm refuses any lora block: it fine-tunes every parameter
        training["lora"] = {"r": 4, "dropout": 0.0}
    training.update(_TRAINING.get(task, {}))
    raw = {"base": "org/m", "task": task, "backend": backend, **top, "data": data,
           "training": training}
    return yaml.safe_dump(raw)


def _refusal(task, backend, modality=None):
    with pytest.raises(ValueError) as info:
        load_config_from_string(_yaml(task, backend, modality=modality))
    return " ".join(str(info.value).split())


class TestTheReasonIsTrueForTheTask:
    @pytest.mark.parametrize("task", _WITH_UNSLOTH)
    def test_a_task_with_an_unsloth_setup_is_told_about_its_fixed_targets(self, task):
        message = _refusal(task, "unsloth")

        assert "fixed attention targets" in message, message
        assert "no unsloth setup" not in message

    @pytest.mark.parametrize("task", _WITHOUT_UNSLOTH)
    def test_a_task_without_one_is_told_unsloth_does_not_run_it(self, task):
        message = _refusal(task, "unsloth")

        assert f"task={task!r} has no unsloth setup" in message, message
        assert "fixed attention targets" not in message

    @pytest.mark.parametrize("task", _WITH_UNSLOTH + _WITHOUT_UNSLOTH)
    def test_the_advice_to_use_transformers_holds(self, task):
        """Both messages say to use backend: transformers; there, the config loads."""
        cfg = load_config_from_string(_yaml(task, "transformers"))

        assert cfg.training.moe_lora is True and cfg.backend == "transformers"


class TestTheTaskRefusalComesFirst:
    @pytest.mark.parametrize("task", _TASK_REFUSED)
    def test_one_refusal_whatever_the_backend(self, task):
        on_unsloth = _refusal(task, "unsloth")

        assert f"not applied by task={task!r}" in on_unsloth, on_unsloth
        assert on_unsloth == _refusal(task, "transformers")

    @pytest.mark.parametrize("modality", ["vision", "audio"])
    def test_sft_vision_and_audio_too(self, modality):
        on_unsloth = _refusal("sft", "unsloth", modality=modality)

        assert f"modality={modality!r}" in on_unsloth, on_unsloth
        assert on_unsloth == _refusal("sft", "transformers", modality=modality)

    @pytest.mark.parametrize("task", ["classifier", "reranker", "cross_encoder"])
    def test_the_classifier_family_without_its_adapter_too(self, task):
        """Without classifier_lora the flag has nothing to select on any backend,
        so that refusal must come before the unsloth one, whose advice would be false."""

        def refusal(backend):
            raw = yaml.safe_load(_yaml(task, backend))
            del raw["training"]["classifier_lora"]
            with pytest.raises(ValueError) as info:
                load_config_from_string(yaml.safe_dump(raw))
            return " ".join(str(info.value).split())

        on_unsloth = refusal("unsloth")

        assert f"not applied by task={task!r} unless" in on_unsloth, on_unsloth
        assert on_unsloth == refusal("transformers")


class TestTheSplitMatchesTheTrainers:
    """The task lists in the schema must follow the trainers: a task whose wrapper
    gains or loses ``_setup_unsloth`` has to move, or its message turns false."""

    def test_every_task_is_classified(self):
        tasks = set(typing.get_args(SoupConfig.model_fields["task"].annotation))

        assert tasks == set(_WRAPPERS)
        assert tasks == set(_WITH_UNSLOTH) | set(_WITHOUT_UNSLOTH) | set(_TASK_REFUSED)

    @staticmethod
    def _has_unsloth_setup(task):
        if task == "preference":
            return all(_TestHelpers.has_setup(t) for t in _PREFERENCE_DELEGATES)
        return _TestHelpers.has_setup(task)

    @pytest.mark.parametrize("task", sorted(_WRAPPERS))
    def test_the_schema_list_matches_the_wrapper(self, task):
        from soup_cli.config.schema import UNSLOTH_SETUP_TASKS

        assert (task in UNSLOTH_SETUP_TASKS) == self._has_unsloth_setup(task)


class _TestHelpers:
    @staticmethod
    def has_setup(task):
        module, name = _WRAPPERS[task].split(":")
        wrapper = getattr(importlib.import_module(f"soup_cli.trainer.{module}"), name)
        return hasattr(wrapper, "_setup_unsloth")
