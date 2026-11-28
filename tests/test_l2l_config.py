"""``training.l2l`` and its options in the config schema.

Seam: ``load_config_from_string``. Every refusal names the field and the gate it hits.
"""

from __future__ import annotations

import re

import pytest
import yaml

from soup_cli.config.loader import load_config_from_string


def _yaml(task="sft", data=None, **training):
    fields = {
        "batch_size": 1,
        "quantization": "none",
        "stream_layers": True,
        "gradient_accumulation_steps": 2,
        "lora": {"r": 4, "alpha": 8, "dropout": 0.0},
    }
    fields.update(training)
    return yaml.safe_dump(
        {
            "base": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "task": task,
            "data": {"train": "data.jsonl", **(data or {})},
            "training": fields,
        }
    )


def test_defaults_leave_an_existing_stream_layers_config_alone():
    cfg = load_config_from_string(_yaml())
    assert cfg.training.l2l is False
    assert cfg.training.l2l_spill_dir is None
    assert cfg.training.l2l_activation_codec == "none"
    assert cfg.training.l2l_auto_k is False


def test_l2l_loads_with_stream_layers_and_accumulation():
    cfg = load_config_from_string(_yaml(l2l=True))
    assert cfg.training.l2l is True
    assert cfg.training.gradient_accumulation_steps == 2


def test_l2l_with_k_equal_one_is_accepted():
    assert load_config_from_string(_yaml(l2l=True, gradient_accumulation_steps=1)).training.l2l


def test_l2l_without_stream_layers_names_the_field_and_the_gate():
    with pytest.raises(ValueError, match=r"training\.l2l requires training\.stream_layers=true"):
        load_config_from_string(_yaml(l2l=True, stream_layers=False))


def test_l2l_for_a_task_other_than_sft_names_the_task_gate():
    pattern = r"training\.l2l supports task in \['sft', 'dpo'\]; got task='ipo'"
    with pytest.raises(ValueError, match=pattern):
        load_config_from_string(_yaml(task="ipo", l2l=True))


@pytest.mark.parametrize("value", [1, 0, "yes", "true"])
def test_l2l_switches_take_a_bool_only(value):
    for key in ("l2l", "l2l_auto_k"):
        with pytest.raises(ValueError, match=rf"training\.{key} must be true or false"):
            load_config_from_string(_yaml(**{key: value}))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("l2l_spill_dir", "spill"),
        ("l2l_activation_codec", "fp8_e4m3"),
        ("l2l_auto_k", True),
    ],
)
def test_an_option_set_while_l2l_is_off_is_refused_with_the_field_named(key, value):
    with pytest.raises(
        ValueError,
        match=rf"training\.{key} is set but training\.l2l is false.*Set l2l=true or drop {key}",
    ):
        load_config_from_string(_yaml(**{key: value}))


def test_options_set_without_stream_layers_are_still_refused_with_the_field_named():
    pattern = r"training\.l2l_spill_dir is set but training\.l2l is false"
    with pytest.raises(ValueError, match=pattern):
        load_config_from_string(_yaml(stream_layers=False, l2l_spill_dir="spill"))


def test_the_fp8_codec_is_accepted_by_the_schema_and_closed_at_launch():
    # The gate lives in utils/l2l_recipe.AVAILABLE_CODECS (tests/test_l2l_recipe.py).
    cfg = load_config_from_string(_yaml(l2l=True, l2l_activation_codec="fp8_e4m3"))
    assert cfg.training.l2l_activation_codec == "fp8_e4m3"


def test_an_unknown_codec_is_refused():
    with pytest.raises(ValueError, match="l2l_activation_codec"):
        load_config_from_string(_yaml(l2l=True, l2l_activation_codec="int4"))


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("", "training.l2l_spill_dir must be a non-empty path"),
        ("   ", "training.l2l_spill_dir must be a non-empty path"),
        ("a" * 4097, "training.l2l_spill_dir must be <= 4096 chars"),
        ("sp\x00ill", "training.l2l_spill_dir must not contain null bytes"),
    ],
)
def test_l2l_spill_dir_shape_is_validated(value, message):
    with pytest.raises(ValueError, match=message.replace(".", r"\.")):
        load_config_from_string(yaml.safe_dump({
            "base": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "task": "sft",
            "data": {"train": "data.jsonl"},
            "training": {
                "batch_size": 1, "quantization": "none", "stream_layers": True,
                "l2l": True, "l2l_spill_dir": value,
            },
        }))


def test_l2l_spill_dir_is_kept_when_l2l_is_on(tmp_path):
    spill = str(tmp_path / "spill")
    cfg = load_config_from_string(_yaml(l2l=True, l2l_spill_dir=spill))
    assert cfg.training.l2l_spill_dir == spill


@pytest.mark.parametrize(
    ("training", "data", "named"),
    [
        ({"prefix_tree": True}, {}, "training.prefix_tree"),
        ({"stream_loss_chunk": 256}, {}, "training.stream_loss_chunk"),
        ({}, {"format": "raft"}, "data.format='raft'"),
    ],
)
def test_l2l_refuses_what_builds_its_own_trainer(training, data, named):
    pattern = r"training\.l2l cannot be combined with .*" + re.escape(named)
    with pytest.raises(ValueError, match=pattern):
        load_config_from_string(_yaml(data=data, l2l=True, **training))


def test_the_moe_fields_stay_unknown_keys():
    for key in (
        "l2l_two_read_experts",
        "l2l_grouped_expert_routes",
        "l2l_route_preserving_mbr",
        "l2l_mbr_group_size",
    ):
        with pytest.raises(ValueError, match=key):
            load_config_from_string(_yaml(l2l=True, **{key: True}))
