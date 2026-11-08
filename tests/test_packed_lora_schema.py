"""``training.packed_lora`` in ``soup.yaml``: the key, through ``SoupConfig``.

The model and the sentence for each refused setting are tested in
``test_packed_lora_config.py``; here: that the schema carries them, that an unknown key in the
block is reported, and that the policy is built from a config. Every refusal is asserted on
the sentence a user would grep for, and each one is reached with exactly one
offending setting so that removing its branch fails exactly one test.
"""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import PackedLoraConfig, SoupConfig, TrainingConfig


def _cfg(*, packed=None, task="sft", backend="transformers", modality="text", data=None,
         **training):
    """A supported config; keyword arguments override the ``training`` block."""
    lora = {"dropout": 0.0, **training.pop("lora", {})}
    block = {"enabled": True} if packed is None else packed
    return SoupConfig(
        base="test/model",
        task=task,
        backend=backend,
        modality=modality,
        data={"train": "tests/fixtures/sample_train.jsonl", **(data or {})},
        training={"quantization": "4bit", "lora": lora, "packed_lora": block, **training},
    )


def _refusal(**kwargs) -> str:
    with pytest.raises(ValidationError) as excinfo:
        _cfg(**kwargs)
    return str(excinfo.value)


# --- the block and its defaults -----------------------------------------------------------


class TestTheBlock:
    def test_it_is_off_unless_asked_for(self):
        assert TrainingConfig().packed_lora is None
        assert PackedLoraConfig().enabled is False

    def test_the_defaults(self):
        block = PackedLoraConfig()
        assert block.gpu_budget_mib is None
        assert block.activations == "auto"
        assert block.int8_backward == "auto"
        # stochastic rounding of the gradient quantisers: round-to-nearest is biased (see config.py)
        assert block.int8_rounding == "grad"
        assert block.on_unsupported == "error"

    def test_the_supported_slice_loads(self):
        cfg = _cfg()
        assert cfg.training.packed_lora.enabled is True

    def test_disabled_block_loads_on_any_task(self):
        cfg = _cfg(packed={"enabled": False}, task="dpo", data={"format": "dpo"})
        assert cfg.training.packed_lora.enabled is False

    def test_every_tunable_loads_with_its_non_default_value(self):
        cfg = _cfg(packed={
            "enabled": True, "gpu_budget_mib": 7000, "activations": "offload",
            "int8_backward": "stash", "int8_rounding": "stochastic", "on_unsupported": "stock",
        })
        block = cfg.training.packed_lora
        assert (block.gpu_budget_mib, block.activations) == (7000, "offload")
        assert (block.int8_backward, block.int8_rounding) == ("stash", "stochastic")
        assert block.on_unsupported == "stock"

    def test_gradient_checkpointing_is_accepted(self):
        # the stash replaces it; wiring prints one line (PR 2), the config does not refuse it
        assert _cfg(gradient_checkpointing=True).training.packed_lora.enabled is True

    def test_a_dumped_config_loads_again(self):
        cfg = _cfg(packed={"enabled": True, "gpu_budget_mib": 7000})
        again = SoupConfig(**cfg.model_dump())
        assert again.training.packed_lora == cfg.training.packed_lora

    def test_a_default_dump_with_the_block_off_loads_again(self):
        # recipes and sweeps re-validate model_dump(); defaults must not trip the footgun guard
        cfg = _cfg(packed={"enabled": False})
        assert SoupConfig(**cfg.model_dump()).training.packed_lora.enabled is False


# --- bounds and types of the block's own fields -------------------------------------------


class TestTheFields:
    @pytest.mark.parametrize("value", [0, 511, -1, 1_048_577])
    def test_budget_bounds(self, value):
        message = _refusal(packed={"enabled": True, "gpu_budget_mib": value})
        assert "gpu_budget_mib" in message

    @pytest.mark.parametrize("value", [512, 1_048_576])
    def test_budget_bounds_are_inclusive(self, value):
        assert _cfg(packed={"enabled": True, "gpu_budget_mib": value}).training.packed_lora

    def test_budget_rejects_a_bool_as_an_int(self):
        message = _refusal(packed={"enabled": True, "gpu_budget_mib": True})
        assert "training.packed_lora.gpu_budget_mib must be an int, not bool" in message

    @pytest.mark.parametrize("value", [7000.5, 7000.0])
    def test_budget_rejects_a_float(self, value):
        message = _refusal(packed={"enabled": True, "gpu_budget_mib": value})
        assert f"training.packed_lora.gpu_budget_mib must be an int; got {value!r}" in message

    @pytest.mark.parametrize("field,value", [
        ("activations", "gpu"),
        ("int8_backward", "maybe"),
        ("int8_rounding", "sloppy"),
        ("on_unsupported", "ignore"),
    ])
    def test_choices_are_closed(self, field, value):
        assert field in _refusal(packed={"enabled": True, field: value})

    def test_an_unknown_key_in_the_block_is_reported(self):
        text = yaml.safe_dump({
            "base": "test/model", "task": "sft",
            "data": {"train": "tests/fixtures/sample_train.jsonl"},
            "training": {"quantization": "4bit", "lora": {"dropout": 0},
                         "packed_lora": {"enabled": True, "gpu_budget": 7000}},
        })
        from soup_cli.config.unknown_keys import find_unknown_config_keys

        found = find_unknown_config_keys(yaml.safe_load(text))
        assert [item.path for item in found] == ["training.packed_lora.gpu_budget"]

    def test_the_loader_refuses_an_unknown_key_in_the_block(self):
        text = (
            "base: test/model\ntask: sft\n"
            "data:\n  train: tests/fixtures/sample_train.jsonl\n"
            "training:\n  quantization: 4bit\n  lora:\n    dropout: 0\n"
            "  packed_lora:\n    enabled: true\n    gpu_budget: 7000\n"
        )
        with pytest.raises(Exception) as excinfo:
            load_config_from_string(text)
        assert "training.packed_lora.gpu_budget" in str(excinfo.value)

    @pytest.mark.parametrize("yaml_value,expected", [("off", "off"), ("on", "on")])
    def test_yaml_booleans_are_mapped_back_for_int8_backward(self, yaml_value, expected):
        # YAML 1.1 reads an unquoted on/off as a bool: the bool agrees with the mode
        text = (
            "base: test/model\ntask: sft\n"
            "data:\n  train: tests/fixtures/sample_train.jsonl\n"
            "training:\n  quantization: 4bit\n  lora:\n    dropout: 0\n"
            f"  packed_lora:\n    enabled: true\n    int8_backward: {yaml_value}\n"
        )
        cfg = load_config_from_string(text)
        assert cfg.training.packed_lora.int8_backward == expected

    def test_int8_backward_refuses_a_number(self):
        message = _refusal(packed={"enabled": True, "int8_backward": 1})
        assert "training.packed_lora.int8_backward must be one of" in message


# --- a tunable set while the block is off -------------------------------------------------


class TestTheFootgunGuard:
    @pytest.mark.parametrize("field,value", [
        ("gpu_budget_mib", 7000),
        ("activations", "offload"),
        ("int8_backward", "off"),
        ("int8_rounding", "nearest"),
        ("on_unsupported", "stock"),
    ])
    def test_a_tunable_while_disabled_is_refused(self, field, value):
        message = _refusal(packed={"enabled": False, field: value})
        assert f"training.packed_lora.{field}" in message
        assert "set but packed_lora is not enabled" in message
        assert "enabled: true" in message

    def test_all_the_offenders_are_named(self):
        message = _refusal(packed={"enabled": False, "gpu_budget_mib": 7000,
                                   "activations": "offload"})
        assert "gpu_budget_mib" in message and "activations" in message


# --- what the kernels cannot honour -------------------------------------------------------


class TestWhatItRefuses:
    def test_backend(self):
        message = _refusal(backend="unsloth")
        assert "training.packed_lora needs backend='transformers'; got backend='unsloth'" in message

    def test_modality(self):
        message = _refusal(modality="vision")
        assert "training.packed_lora supports modality='text' only; got modality='vision'" in (
            message)

    @pytest.mark.parametrize("task", ["dpo", "grpo", "pretrain"])
    def test_task(self, task):
        data = {"format": "dpo"} if task == "dpo" else None
        if task == "grpo":
            data = {"format": "chatml"}
        if task == "pretrain":
            data = {"format": "plaintext"}
        message = _refusal(task=task, data=data)
        assert f"training.packed_lora supports task='sft' only; got task={task!r}" in message

    def test_full_fine_tuning(self):
        message = _refusal(lora={"r": 0})
        assert "training.packed_lora needs a LoRA; lora.r=0 trains the full model" in message

    @pytest.mark.parametrize("quantization", ["none", "8bit"])
    def test_quantization(self, quantization):
        message = _refusal(quantization=quantization)
        assert "training.packed_lora needs quantization: 4bit" in message
        assert f"got quantization={quantization!r}" in message

    def test_double_quantization_off(self):
        message = _refusal(bnb_4bit_use_double_quant=False)
        assert "training.packed_lora needs double quantization" in message
        assert "bnb_4bit_use_double_quant" in message

    def test_the_default_dropout_is_refused_with_the_way_out(self):
        message = _refusal(lora={"dropout": 0.05})
        assert "training.packed_lora needs lora.dropout: 0; got 0.05" in message
        assert "Soup's default is 0.05" in message

    def test_the_default_lora_block_is_refused_for_its_dropout(self):
        with pytest.raises(ValidationError) as excinfo:
            SoupConfig(
                base="test/model", task="sft",
                data={"train": "tests/fixtures/sample_train.jsonl"},
                training={"quantization": "4bit", "packed_lora": {"enabled": True}},
            )
        assert "needs lora.dropout: 0" in str(excinfo.value)

    @pytest.mark.parametrize("lora,name", [
        ({"use_dora": True}, "lora.use_dora"),
        ({"use_vera": True}, "lora.use_vera"),
    ])
    def test_adapters_that_are_not_plain_lora(self, lora, name):
        message = _refusal(lora=lora)
        assert f"training.packed_lora cannot be combined with {name}" in message

    @pytest.mark.parametrize("field,value,name", [
        ("use_liger", True, "use_liger"),
        ("use_cut_ce", True, "use_cut_ce"),
        ("fast_lora", True, "fast_lora"),
        ("use_flash_attn", True, "use_flash_attn"),
        ("packing", True, "packing"),
        ("multipack", True, "multipack"),
        ("neftune_alpha", 5.0, "neftune_alpha"),
        ("stream_layers", True, "stream_layers"),
        ("activation_offloading", "cpu", "activation_offloading"),
        ("use_galore", True, "use_galore"),
        ("use_lorafa", True, "use_lorafa"),
        ("lisa_enabled", True, "lisa_enabled"),
    ])
    def test_training_switches_that_clash(self, field, value, name):
        message = _refusal(**{field: value})
        assert f"training.packed_lora cannot be combined with training.{name}" in message

    @pytest.mark.parametrize("training,name", [
        ({"unfrozen_parameters": ["^model.layers.0"]}, "training.unfrozen_parameters"),
        ({"quantization_aware": "fp8"}, "training.quantization_aware"),
        ({"lora": {"dropout": 0.0, "target_parameters": ["experts"]}}, "lora.target_parameters"),
    ])
    def test_modes_that_train_other_weights(self, training, name):
        message = _refusal(**training)
        assert f"training.packed_lora cannot be combined with {name}" in message

    def test_raft_rows(self):
        message = _refusal(data={"format": "raft"})
        assert "training.packed_lora cannot be combined with data.format: raft" in message

    def test_each_refusal_comes_with_one_reason_only(self):
        message = _refusal(quantization="none", lora={"dropout": 0.05})
        assert message.count("training.packed_lora") == 1


# --- the generated reference and the recipe snapshot know the new field -------------------


def test_the_field_is_documented():
    description = TrainingConfig.model_fields["packed_lora"].description
    assert "Triton" in description and "off by default" in description.lower()


# --- the policy of a run, from a config ---------------------------------------------------


class TestPolicyFromAConfig:
    def test_defaults(self):
        from soup_cli.kernels.packed_lora.policy import Policy

        assert Policy.from_config(_cfg()) == Policy()

    def test_every_field_is_carried(self):
        from soup_cli.kernels.packed_lora.policy import Policy

        cfg = _cfg(packed={"enabled": True, "gpu_budget_mib": 7000, "activations": "offload",
                           "int8_backward": "stash", "int8_rounding": "stochastic",
                           "on_unsupported": "stock"})
        pol = Policy.from_config(cfg)
        assert pol.gpu_budget_bytes == 7000 * (1 << 20)
        assert (pol.activations, pol.int8_backward) == ("offload", "stash")
        assert (pol.int8_rounding, pol.on_unsupported) == ("stochastic", "stock")

    def test_a_config_without_the_kernels_is_refused(self):
        from soup_cli.kernels.packed_lora.policy import Policy

        with pytest.raises(ValueError, match="training.packed_lora.enabled is not true"):
            Policy.from_config(SoupConfig(base="test/model", data={"train": "x.jsonl"}))
        with pytest.raises(ValueError, match="training.packed_lora.enabled is not true"):
            Policy.from_config(_cfg(packed={"enabled": False}))


def test_the_schema_uses_the_packages_model():
    from soup_cli.kernels.packed_lora import config as package_config

    assert PackedLoraConfig is package_config.PackedLoraConfig
