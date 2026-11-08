"""``soup_cli.kernels.packed_lora.config``: the settings of the packed-LoRA kernels (CPU).

The model is the type of ``training.packed_lora`` (tested through ``SoupConfig`` in
``test_packed_lora_schema.py``). What is checked here: the block's defaults and bounds, and
:func:`packed_lora_problem`, the sentence for each setting of a run that the kernels cannot honour.
Every refusal is asserted on the words a user would grep for, and each one is reached with exactly
one offending setting, so that removing its branch fails exactly one test.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from soup_cli.config.schema import TrainingConfig
from soup_cli.kernels.packed_lora.config import (
    MAX_BUDGET_MIB,
    MIN_BUDGET_MIB,
    PackedLoraConfig,
    packed_lora_problem,
)


def _problem(*, packed=None, task="sft", backend="transformers", modality="text", data=None,
             **training):
    """The kernels' answer for a supported config with the overrides given."""
    lora = {"dropout": 0.0, **training.pop("lora", {})}
    config = SimpleNamespace(
        backend=backend, modality=modality, task=task,
        data=SimpleNamespace(format=(data or {}).get("format", "alpaca")),
        training=TrainingConfig(
            **{"quantization": "4bit", "batch_size": 2, "lora": lora, **training}))
    block = PackedLoraConfig(**({"enabled": True} if packed is None else packed))
    return packed_lora_problem(config, block)


# --- the block and its defaults -----------------------------------------------------------


class TestTheBlock:
    def test_it_is_off_unless_asked_for(self):
        assert PackedLoraConfig().enabled is False

    def test_the_defaults(self):
        block = PackedLoraConfig()
        assert block.gpu_budget_mib is None
        assert block.activations == "auto"
        assert block.int8_backward == "auto"
        # stochastic rounding of the gradient quantisers: round-to-nearest is biased (see config.py)
        assert block.int8_rounding == "grad"
        assert block.on_unsupported == "error"

    def test_every_tunable_loads_with_its_non_default_value(self):
        block = PackedLoraConfig(
            enabled=True, gpu_budget_mib=7000, activations="offload", int8_backward="stash",
            int8_rounding="stochastic", on_unsupported="stock")
        assert (block.gpu_budget_mib, block.activations) == (7000, "offload")
        assert (block.int8_backward, block.int8_rounding) == ("stash", "stochastic")
        assert block.on_unsupported == "stock"

    def test_a_dumped_block_loads_again(self):
        block = PackedLoraConfig(enabled=True, gpu_budget_mib=7000)
        assert PackedLoraConfig(**block.model_dump()) == block

    def test_a_default_dump_with_the_block_off_loads_again(self):
        assert PackedLoraConfig(**PackedLoraConfig().model_dump()).enabled is False


# --- bounds and types of the block's own fields -------------------------------------------


class TestTheFields:
    @pytest.mark.parametrize("value", [0, MIN_BUDGET_MIB - 1, -1, MAX_BUDGET_MIB + 1])
    def test_budget_bounds(self, value):
        with pytest.raises(ValidationError) as excinfo:
            PackedLoraConfig(enabled=True, gpu_budget_mib=value)
        assert "gpu_budget_mib" in str(excinfo.value)

    @pytest.mark.parametrize("value", [MIN_BUDGET_MIB, MAX_BUDGET_MIB])
    def test_budget_bounds_are_inclusive(self, value):
        assert PackedLoraConfig(enabled=True, gpu_budget_mib=value).gpu_budget_mib == value

    def test_the_bounds(self):
        assert (MIN_BUDGET_MIB, MAX_BUDGET_MIB) == (512, 1_048_576)

    def test_budget_rejects_a_bool_as_an_int(self):
        with pytest.raises(ValidationError) as excinfo:
            PackedLoraConfig(enabled=True, gpu_budget_mib=True)
        assert "training.packed_lora.gpu_budget_mib must be an int, not bool" in str(excinfo.value)

    @pytest.mark.parametrize("value", [7000.5, 7000.0])
    def test_budget_rejects_a_float(self, value):
        with pytest.raises(ValidationError) as excinfo:
            PackedLoraConfig(enabled=True, gpu_budget_mib=value)
        assert f"training.packed_lora.gpu_budget_mib must be an int; got {value!r}" in str(
            excinfo.value)

    @pytest.mark.parametrize("field,value", [
        ("activations", "gpu"),
        ("int8_backward", "maybe"),
        ("int8_rounding", "sloppy"),
        ("on_unsupported", "ignore"),
    ])
    def test_choices_are_closed(self, field, value):
        with pytest.raises(ValidationError) as excinfo:
            PackedLoraConfig(enabled=True, **{field: value})
        assert field in str(excinfo.value)

    @pytest.mark.parametrize("value,expected", [(True, "on"), (False, "off")])
    def test_yaml_booleans_are_mapped_back_for_int8_backward(self, value, expected):
        # YAML 1.1 reads an unquoted on/off as a bool: the bool agrees with the mode
        assert PackedLoraConfig(enabled=True, int8_backward=value).int8_backward == expected

    @pytest.mark.parametrize("value", [1, 0, 2.5])
    def test_int8_backward_refuses_a_number(self, value):
        with pytest.raises(ValidationError) as excinfo:
            PackedLoraConfig(enabled=True, int8_backward=value)
        assert "training.packed_lora.int8_backward must be one of" in str(excinfo.value)


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
        with pytest.raises(ValidationError) as excinfo:
            PackedLoraConfig(enabled=False, **{field: value})
        message = str(excinfo.value)
        assert f"training.packed_lora.{field}" in message
        assert "set but packed_lora is not enabled" in message
        assert "enabled: true" in message

    def test_all_the_offenders_are_named(self):
        with pytest.raises(ValidationError) as excinfo:
            PackedLoraConfig(enabled=False, gpu_budget_mib=7000, activations="offload")
        assert "gpu_budget_mib" in str(excinfo.value) and "activations" in str(excinfo.value)


# --- what the kernels cannot honour -------------------------------------------------------


class TestWhatItRefuses:
    def test_the_supported_slice_has_no_problem(self):
        assert _problem() is None

    def test_every_tunable_value_is_fine(self):
        assert _problem(packed={"enabled": True, "gpu_budget_mib": 7000, "activations": "offload",
                                "int8_backward": "off", "int8_rounding": "nearest"}) is None

    def test_gradient_checkpointing_is_accepted(self):
        # the stash replaces it; the wiring prints one line, the check does not refuse it
        assert _problem(gradient_checkpointing=True) is None

    def test_a_block_that_is_off_refuses_nothing(self):
        assert _problem(packed={"enabled": False}, backend="mlx", task="dpo") is None
        config = SimpleNamespace(backend="mlx", training=None)
        assert packed_lora_problem(config, None) is None

    def test_backend(self):
        assert _problem(backend="unsloth").startswith(
            "training.packed_lora needs backend='transformers'; got backend='unsloth'")

    def test_modality(self):
        assert _problem(modality="vision").startswith(
            "training.packed_lora supports modality='text' only; got modality='vision'")

    @pytest.mark.parametrize("task", ["dpo", "grpo", "pretrain"])
    def test_task(self, task):
        assert _problem(task=task).startswith(
            f"training.packed_lora supports task='sft' only; got task={task!r}")

    def test_full_fine_tuning(self):
        assert _problem(lora={"r": 0}).startswith(
            "training.packed_lora needs a LoRA; lora.r=0 trains the full model")

    @pytest.mark.parametrize("quantization", ["none", "8bit"])
    def test_quantization(self, quantization):
        message = _problem(quantization=quantization)
        assert message.startswith("training.packed_lora needs quantization: 4bit")
        assert f"got quantization={quantization!r}" in message

    def test_double_quantization_off(self):
        message = _problem(bnb_4bit_use_double_quant=False)
        assert message.startswith("training.packed_lora needs double quantization")
        assert "bnb_4bit_use_double_quant" in message

    def test_the_default_dropout_is_refused_with_the_way_out(self):
        message = _problem(lora={"dropout": 0.05})
        assert message.startswith("training.packed_lora needs lora.dropout: 0; got 0.05")
        assert "Soup's default is 0.05" in message

    def test_the_default_lora_block_is_refused_for_its_dropout(self):
        config = SimpleNamespace(
            backend="transformers", modality="text", task="sft",
            data=SimpleNamespace(format="alpaca"),
            training=TrainingConfig(quantization="4bit", batch_size=2))
        problem = packed_lora_problem(config, PackedLoraConfig(enabled=True))
        assert "needs lora.dropout: 0" in problem

    def test_an_auto_batch_size_is_refused_with_the_reason(self):
        message = _problem(batch_size="auto")
        assert message.startswith(
            "training.packed_lora needs an explicit training.batch_size; got 'auto'")
        assert "memory plan is made for the batch size" in message

    def test_the_default_batch_size_is_auto_and_is_refused(self):
        config = SimpleNamespace(
            backend="transformers", modality="text", task="sft",
            data=SimpleNamespace(format="alpaca"),
            training=TrainingConfig(quantization="4bit", lora={"dropout": 0.0}))
        assert "explicit training.batch_size" in packed_lora_problem(
            config, PackedLoraConfig(enabled=True))

    @pytest.mark.parametrize("lora,name", [
        ({"use_dora": True}, "lora.use_dora"),
        ({"use_vera": True}, "lora.use_vera"),
    ])
    def test_adapters_that_are_not_plain_lora(self, lora, name):
        assert _problem(lora=lora).startswith(
            f"training.packed_lora cannot be combined with {name}")

    @pytest.mark.parametrize("field,value", [
        ("use_liger", True), ("use_cut_ce", True), ("fast_lora", True), ("use_flash_attn", True),
        ("packing", True), ("multipack", True), ("neftune_alpha", 5.0), ("stream_layers", True),
        ("activation_offloading", "cpu"), ("use_galore", True), ("use_lorafa", True),
        ("lisa_enabled", True),
    ])
    def test_training_switches_that_clash(self, field, value):
        assert _problem(**{field: value}).startswith(
            f"training.packed_lora cannot be combined with training.{field}")

    @pytest.mark.parametrize("training,name", [
        ({"unfrozen_parameters": ["^model.layers.0"]}, "training.unfrozen_parameters"),
        ({"quantization_aware": "fp8"}, "training.quantization_aware"),
        ({"lora": {"dropout": 0.0, "target_parameters": ["experts"]}}, "lora.target_parameters"),
    ])
    def test_modes_that_train_other_weights(self, training, name):
        assert _problem(**training).startswith(
            f"training.packed_lora cannot be combined with {name}")

    def test_raft_rows(self):
        assert _problem(data={"format": "raft"}).startswith(
            "training.packed_lora cannot be combined with data.format: raft")

    def test_each_refusal_comes_with_one_reason_only(self):
        message = _problem(quantization="none", lora={"dropout": 0.05})
        assert message.count("training.packed_lora") == 1
