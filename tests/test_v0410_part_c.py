"""v0.41.0 Part C — PEFT methods (LoftQ + LLaMA Pro + MoD + 8/16-bit aliases) tests."""

from __future__ import annotations

import typing

import pytest
from pydantic import ValidationError

from soup_cli.config.schema import LoraConfig, SoupConfig, TrainingConfig
from soup_cli.utils.block_expansion import (
    _count_layers,
    expand_model_blocks,
    validate_expand_layers,
    validate_freeze_trainable_layers,
)
from soup_cli.utils.loftq_init import (
    build_loftq_config,
    validate_loftq_bits,
    validate_loftq_iter,
)


def _base_data():
    return {"train": "/tmp/x.jsonl"}


# Every quantization value except "none", read from the schema, so a new Quant
# Menu format is refused -- and tested -- without editing a list here.
_QUANTIZED = tuple(
    value
    for value in typing.get_args(TrainingConfig.model_fields["quantization"].annotation)
    if value != "none"
)


def _sft(**training):
    return SoupConfig(base="fake-org/fake-model", task="sft", data=_base_data(), training=training)


# ---------- LoftQ ----------

class TestLoftqValidators:
    def test_iter_default(self):
        assert validate_loftq_iter(1) == 1

    def test_iter_bool_rejected(self):
        with pytest.raises(ValueError, match="must be int"):
            validate_loftq_iter(True)

    def test_iter_oob_rejected(self):
        with pytest.raises(ValueError, match="must be in"):
            validate_loftq_iter(0)
        with pytest.raises(ValueError, match="must be in"):
            validate_loftq_iter(11)

    def test_bits_valid(self):
        assert validate_loftq_bits(2) == 2
        assert validate_loftq_bits(4) == 4
        assert validate_loftq_bits(8) == 8

    def test_bits_invalid(self):
        with pytest.raises(ValueError, match="must be one of"):
            validate_loftq_bits(3)

    def test_bits_bool_rejected(self):
        with pytest.raises(ValueError, match="must be int"):
            validate_loftq_bits(True)


class TestLoraInitStrategyLoftq:
    def test_loftq_accepted(self):
        cfg = LoraConfig(init_strategy="loftq")
        assert cfg.init_strategy == "loftq"
        assert cfg.loftq_iter == 1
        assert cfg.loftq_bits == 4

    def test_loftq_with_dora_rejected(self):
        with pytest.raises(ValidationError, match="loftq.*incompatible.*use_dora"):
            LoraConfig(init_strategy="loftq", use_dora=True)

    def test_loftq_with_vera_rejected(self):
        with pytest.raises(ValidationError, match="loftq.*incompatible.*use_vera"):
            LoraConfig(init_strategy="loftq", use_vera=True)

    def test_loftq_iter_bounds(self):
        with pytest.raises(ValidationError):
            LoraConfig(init_strategy="loftq", loftq_iter=0)

    def test_loftq_bits_invalid(self):
        with pytest.raises(ValidationError):
            LoraConfig(init_strategy="loftq", loftq_bits=3)


# ---------- LLaMA Pro / block expansion ----------

class TestBlockExpansion:
    def test_expand_layers_validation(self):
        assert validate_expand_layers(None) == 0
        assert validate_expand_layers(4) == 4

    def test_expand_layers_bool_rejected(self):
        with pytest.raises(ValueError, match="must be int"):
            validate_expand_layers(True)

    def test_expand_layers_oob(self):
        with pytest.raises(ValueError, match="must be in"):
            validate_expand_layers(0)
        with pytest.raises(ValueError, match="must be in"):
            validate_expand_layers(65)

    def test_freeze_trainable_layers(self):
        assert validate_freeze_trainable_layers(None) == 0
        assert validate_freeze_trainable_layers(4) == 4
        assert validate_freeze_trainable_layers(-4) == -4

    def test_freeze_trainable_layers_oob(self):
        with pytest.raises(ValueError, match="magnitude"):
            validate_freeze_trainable_layers(1001)
        with pytest.raises(ValueError, match="magnitude"):
            validate_freeze_trainable_layers(-1001)

    def test_freeze_trainable_layers_bool_rejected(self):
        with pytest.raises(ValueError, match="must be int"):
            validate_freeze_trainable_layers(True)

    def test_expand_model_blocks_zero_returns_layer_count(self):
        class Stub:
            class Inner:
                layers = [object(), object(), object()]

            model = Inner()

        assert expand_model_blocks(Stub(), 0) == 3

    def test_expand_model_blocks_rejects_object_without_layers(self):
        # v0.53.4 #83 lifted the deferred stub. Calling on a bare ``object()``
        # now raises ValueError because no decoder layers can be discovered.
        with pytest.raises(ValueError, match="decoder layers"):
            expand_model_blocks(object(), 4)


class TestSchemaBlockExpansion:
    def test_expand_layers_requires_freeze(self):
        with pytest.raises(ValidationError, match="requires freeze_trainable_layers"):
            TrainingConfig(expand_layers=4)

    def test_expand_layers_with_freeze_accepted(self):
        cfg = TrainingConfig(
            expand_layers=4, freeze_trainable_layers=4, quantization="none"
        )
        assert cfg.expand_layers == 4
        assert cfg.freeze_trainable_layers == 4

    @pytest.mark.parametrize("freeze", [2, -4, 0])
    def test_freeze_trainable_layers_alone_refused(self, freeze):
        # #1396: without expand_layers nothing reads the field, so every value
        # (positive, negative or zero) trained as if it were unset.
        with pytest.raises(
            ValidationError,
            match="freeze_trainable_layers only applies together with expand_layers",
        ):
            TrainingConfig(freeze_trainable_layers=freeze)

    @pytest.mark.parametrize("task", ["sft", "pretrain"])
    def test_freeze_trainable_layers_alone_refused_at_load(self, task):
        # The issue's own case, through the loader `soup train` uses.
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(
            ValueError,
            match="only applies together with expand_layers.*Remove it",
        ):
            load_config_from_string(
                "base: org/model\n"
                f"task: {task}\n"
                "data:\n  train: ./data.jsonl\n"
                "training:\n  freeze_trainable_layers: 2\n"
            )

    @pytest.mark.parametrize("quantization", _QUANTIZED)
    @pytest.mark.parametrize("freeze", [4, -4, 0])
    def test_expand_layers_refused_on_quantized_base(self, quantization, freeze):
        # The refusal lives on SoupConfig now (it reads the resolved
        # quantization), so build a full config rather than a bare TrainingConfig.
        # #1409's equality check runs first (TrainingConfig before SoupConfig),
        # so -4 and 0 are refused for not matching expand_layers instead.
        match = (
            "requires training.quantization: none"
            if freeze == 4
            else r"freeze_trainable_layers: -?\d+ does not match expand_layers: 4"
        )
        with pytest.raises(ValidationError, match=match):
            _sft(
                expand_layers=4,
                freeze_trainable_layers=freeze,
                quantization=quantization,
            )

    def test_expand_layers_refused_on_default_quantization(self):
        with pytest.raises(ValidationError, match="requires training.quantization: none"):
            _sft(expand_layers=2, freeze_trainable_layers=2)

    def test_load_in_16bit_alias_satisfies_the_requirement(self):
        # docs/peft-and-efficiency.md: load_in_16bit: true == quantization: none
        cfg = _sft(expand_layers=2, freeze_trainable_layers=2, load_in_16bit=True)
        assert cfg.training.quantization == "none"
        assert cfg.training.expand_layers == 2

    def test_load_in_8bit_alias_is_refused(self):
        with pytest.raises(ValidationError, match="requires training.quantization: none"):
            _sft(expand_layers=2, freeze_trainable_layers=2, load_in_8bit=True)

    @pytest.mark.parametrize("task", ["sft", "pretrain"])
    def test_default_quantization_refused_through_the_loader(self, task):
        # #1237's acceptance box: task sft AND pretrain, through the loader `soup train` uses.
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="requires training.quantization: none"):
            load_config_from_string(
                "base: org/model\n"
                f"task: {task}\n"
                "data:\n  train: ./data.jsonl\n"
                "training:\n  expand_layers: 2\n  freeze_trainable_layers: 2\n"
            )

    @pytest.mark.parametrize(
        "task,backend", [("dpo", "transformers"), ("sft", "mlx"), ("sft", "unsloth")]
    )
    def test_scope_refusal_precedes_the_quantization_refusal(self, task, backend):
        with pytest.raises(ValidationError, match="only wired for task sft/pretrain"):
            SoupConfig(
                base="fake-org/fake-model",
                task=task,
                backend=backend,
                data=_base_data(),
                training={"expand_layers": 2, "freeze_trainable_layers": 2},
            )

    def test_expand_layers_explicit_quant_still_refused_on_unhonoured_task(self):
        # A quant-menu value the resolver never rewrites (gptq) is still refused on an
        # unhonoured task, and with THAT task's reason: this check must not mask it.
        # (Fails if the check is defined above the resolver.)
        with pytest.raises(ValidationError, match="task='prm' does not apply"):
            SoupConfig(
                base="fake-org/fake-model",
                task="prm",
                data=_base_data(),
                training={
                    "expand_layers": 2,
                    "freeze_trainable_layers": 2,
                    "quantization": "gptq",
                },
            )

    def test_freeze_magnitude_oob(self):
        with pytest.raises(ValidationError, match="magnitude"):
            TrainingConfig(freeze_trainable_layers=1500)


# ---------- Mixture-of-Depths ----------

class TestUseMod:
    def test_default_off(self):
        assert TrainingConfig().use_mod is False

    def test_can_enable(self):
        cfg = TrainingConfig(use_mod=True)
        assert cfg.use_mod is True


# ---------- 8/16-bit aliases ----------

class TestLoadInAliases:
    def test_default_none(self):
        cfg = TrainingConfig()
        assert cfg.load_in_8bit is None
        assert cfg.load_in_16bit is None

    def test_load_in_8bit_remaps(self):
        cfg = TrainingConfig(load_in_8bit=True, quantization="none")
        assert cfg.quantization == "8bit"

    def test_load_in_16bit_remaps(self):
        cfg = TrainingConfig(load_in_16bit=True, quantization="4bit")
        assert cfg.quantization == "none"

    def test_mutually_exclusive(self):
        with pytest.raises(ValidationError, match="mutually exclusive"):
            TrainingConfig(load_in_8bit=True, load_in_16bit=True)

    def test_both_false_no_op(self):
        cfg = TrainingConfig(
            load_in_8bit=False, load_in_16bit=False, quantization="4bit"
        )
        assert cfg.quantization == "4bit"

    def test_alias_with_quant_menu_rejected(self):
        with pytest.raises(ValidationError, match="cannot be combined"):
            TrainingConfig(load_in_8bit=True, quantization="gptq")


# ---------- Full SoupConfig integration ----------

class TestSoupConfigIntegration:
    def test_full_yaml_loftq(self):
        cfg = SoupConfig(
            base="meta-llama/Llama-3.1-8B",
            data=_base_data(),
            training={
                "quantization": "none",
                "lora": {"init_strategy": "loftq", "loftq_iter": 2, "loftq_bits": 4},
                "optimizer": "badam",
                "lr_groups": {"q_proj": 1e-4},
            },
        )
        assert cfg.training.lora.init_strategy == "loftq"
        assert cfg.training.optimizer == "badam"
        assert len(cfg.training.lr_groups) == 1

    def test_expand_layers_field_validator_rejects_bool(self):
        # Pydantic Field(ge=1, le=64) accepts True (=1) — the explicit
        # validator must reject bool to match project bool-as-int policy.
        with pytest.raises(ValidationError, match="must be int"):
            TrainingConfig(expand_layers=True, freeze_trainable_layers=4)


class TestCountLayers:
    def test_decoder_path(self):
        class Stub:
            class Decoder:
                layers = [object(), object()]

            class Inner:
                decoder = None  # set below

            model = Inner()

        Stub.Inner.decoder = Stub.Decoder()
        assert _count_layers(Stub()) == 2

    def test_no_layers_returns_zero(self):
        assert _count_layers(object()) == 0

    def test_layers_no_len(self):
        class NoLen:
            pass

        class Stub:
            class Inner:
                layers = NoLen()

            model = Inner()

        assert _count_layers(Stub()) == 0

    def test_expand_zero_with_none(self):
        class Stub:
            class Inner:
                layers = [1, 2]

            model = Inner()

        assert expand_model_blocks(Stub(), None) == 2


class TestBuildLoftqConfig:
    def test_invalid_iter_rejected(self):
        with pytest.raises(ValueError, match="loftq_iter"):
            build_loftq_config(loftq_iter=0, loftq_bits=4)

    def test_invalid_bits_rejected(self):
        with pytest.raises(ValueError, match="loftq_bits"):
            build_loftq_config(loftq_iter=1, loftq_bits=3)

    def test_happy_path(self):
        # peft is a hard dependency (core dep), so this should succeed
        # in the test environment. If peft is missing, ImportError with
        # actionable message is the contract.
        try:
            cfg = build_loftq_config(loftq_iter=2, loftq_bits=4)
        except ImportError as exc:
            assert "peft" in str(exc).lower()
            return
        # Confirm peft.LoftQConfig was constructed with the right values.
        assert getattr(cfg, "loftq_bits", None) == 4
        assert getattr(cfg, "loftq_iter", None) == 2
