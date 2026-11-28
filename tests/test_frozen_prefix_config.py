"""``training.frozen_prefix_cache`` schema rules."""

import pytest

from soup_cli.config.loader import config_from_mapping
from soup_cli.config.schema import SoupConfig


def make_config(**training_overrides):
    training = {
        "frozen_prefix_cache": True,
        "freeze_layers": 16,
        "batch_size": 1,
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
        (
            {"quantization": "none", "lora": {"r": 0}},
            "requires training.lora.r >= 1",
        ),
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
            {
                "quantization": "none",
                "lora": {"r": 8, "dropout": 0, "target_parameters": ["experts.w1"]},
            },
            "does not support training.lora.target_parameters",
        ),
        (
            {"neftune_alpha": 5.0},
            "cannot be combined with training.neftune_alpha",
        ),
        (
            {"neftune_alpha": 0.0},
            "cannot be combined with training.neftune_alpha",
        ),
    ],
)
def test_incompatible_settings_are_refused_by_name(overrides, message):
    with pytest.raises(ValueError, match=message):
        SoupConfig.model_validate(make_config(**overrides))


@pytest.mark.parametrize("batch_size", [2, 8, "auto"])
def test_batches_above_one_need_a_fixed_data_order(batch_size):
    with pytest.raises(ValueError, match="needs a fixed data order") as caught:
        SoupConfig.model_validate(make_config(batch_size=batch_size))

    assert "training.batch_size: 1" in str(caught.value)
    assert "training.curriculum_metric: length" in str(caught.value)


def test_batch_of_one_needs_no_fixed_order():
    config = SoupConfig.model_validate(make_config(batch_size=1))

    assert config.training.batch_size == 1


def test_a_length_curriculum_is_the_fixed_order_for_larger_batches():
    config = SoupConfig.model_validate(
        make_config(batch_size=4, curriculum=True, curriculum_metric="length")
    )

    assert config.training.batch_size == 4


@pytest.mark.parametrize("metric", ["perplexity", "loss"])
def test_other_curriculum_metrics_do_not_fix_the_order(metric):
    with pytest.raises(ValueError, match="needs a fixed data order"):
        SoupConfig.model_validate(
            make_config(batch_size=4, curriculum=True, curriculum_metric=metric)
        )


def test_task_gate_is_named_where_the_freeze_check_lets_it_through():
    raw = make_config(tts_family="orpheus")
    raw.update({"task": "tts", "modality": "audio_out"})

    with pytest.raises(ValueError, match="requires task='sft'; got task='tts'"):
        SoupConfig.model_validate(raw)


@pytest.mark.parametrize(
    "top, training, message",
    [
        ({"task": "dpo"}, {}, "freeze_layers is not applied by task='dpo'"),
        ({"backend": "unsloth"}, {}, "not applied by task='sft' with backend='unsloth'"),
        ({"modality": "vision"}, {}, "not applied by task='sft' with modality='vision'"),
        (
            {},
            {"stream_layers": True},
            "not applied by task='sft' with training.stream_layers=true",
        ),
    ],
)
def test_unsupported_paths_are_refused_by_the_freeze_check_first(top, training, message):
    raw = make_config(**training)
    raw.update(top)

    with pytest.raises(ValueError, match=message):
        SoupConfig.model_validate(raw)


def test_unknown_key_still_refuses_the_load():
    with pytest.raises(ValueError, match="unknown config key 'training.frozen_prefix_cash'"):
        config_from_mapping(make_config(frozen_prefix_cash=True))
