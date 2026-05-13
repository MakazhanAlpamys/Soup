"""v0.41.0 Part C — PEFT methods (LoftQ + LLaMA Pro + MoD + 8/16-bit aliases) tests."""

from __future__ import annotations

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
        cfg = TrainingConfig(expand_layers=4, freeze_trainable_layers=4)
        assert cfg.expand_layers == 4
        assert cfg.freeze_trainable_layers == 4

    def test_freeze_trainable_layers_alone_ok(self):
        cfg = TrainingConfig(freeze_trainable_layers=-4)
        assert cfg.freeze_trainable_layers == -4
        assert cfg.expand_layers is None

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
