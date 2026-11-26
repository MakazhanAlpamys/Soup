"""``training.frozen_prefix_cache`` schema rules."""

import pytest

from soup_cli.config.schema import SoupConfig


def make_config(**training_overrides):
    training = {
        "frozen_prefix_cache": True,
        "freeze_layers": 16,
        "lora": {"r": 8},
    }
    training.update(training_overrides)
    return {
        "task": "sft",
        "backend": "transformers",
        "base": "mistralai/Mistral-7B-v0.1",
        "training": training,
        "data": {"train": "dummy.jsonl"},
    }


def test_off_by_default():
    config = SoupConfig.model_validate(
        {
            "base": "mistralai/Mistral-7B-v0.1",
            "data": {"train": "dummy.jsonl"},
        }
    )

    assert config.training.frozen_prefix_cache is False


def test_valid_with_freeze_layers():
    config = SoupConfig.model_validate(make_config())

    assert config.training.frozen_prefix_cache is True
    assert config.training.freeze_layers == 16


def test_valid_with_freeze_ratio():
    config = SoupConfig.model_validate(
        make_config(freeze_layers=None, freeze_ratio=0.75)
    )

    assert config.training.freeze_ratio == 0.75


def test_off_leaves_every_other_setting_alone():
    config = SoupConfig.model_validate(
        make_config(
            frozen_prefix_cache=False,
            freeze_layers=None,
            packing=True,
            stream_layers=False,
        )
    )

    assert config.training.frozen_prefix_cache is False


@pytest.mark.parametrize("value", [1, 0, "true", "yes", 1.0])
def test_a_non_boolean_switch_is_refused(value):
    with pytest.raises(ValueError, match="training.frozen_prefix_cache must be bool"):
        SoupConfig.model_validate(make_config(frozen_prefix_cache=value))


@pytest.mark.parametrize(
    "overrides, message",
    [
        (
            {"freeze_layers": None},
            "requires training.freeze_layers or training.freeze_ratio",
        ),
        ({"lora": {"r": 0}}, "requires training.lora.r >= 1"),
        ({"stream_layers": True}, "does not support training.stream_layers"),
        ({"packing": True}, "does not support training.packing or training.multipack"),
        ({"multipack": True}, "does not support training.packing or training.multipack"),
        (
            {"gradient_checkpointing": True},
            "requires training.gradient_checkpointing=false",
        ),
        (
            {"gradient_checkpointing": "auto"},
            "requires training.gradient_checkpointing=false",
        ),
        (
            {"lora": {"r": 8, "use_vera": True}},
            "does not support training.lora.use_vera",
        ),
        (
            {"lora": {"r": 8, "target_parameters": ["experts.w1"]}},
            "does not support training.lora.target_parameters",
        ),
    ],
)
def test_incompatible_settings_are_refused_by_name(overrides, message):
    with pytest.raises(ValueError, match=message):
        SoupConfig.model_validate(make_config(**overrides))


@pytest.mark.parametrize(
    "top, message",
    [
        ({"task": "dpo"}, "requires task='sft'"),
        ({"backend": "unsloth"}, "requires backend='transformers'"),
        ({"modality": "vision"}, "requires modality='text'"),
    ],
)
def test_task_backend_and_modality_gates_are_named(top, message):
    raw = make_config()
    raw.update(top)

    with pytest.raises(ValueError, match=message):
        SoupConfig.model_validate(raw)


def test_unknown_key_still_refuses_the_load():
    with pytest.raises(ValueError, match="frozen_prefix_cash"):
        SoupConfig.model_validate(make_config(frozen_prefix_cash=True))
