"""Tests for v0.39.0 Part A — PiSSA init + init_strategy field on LoraConfig."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from soup_cli.config.schema import LoraConfig
from soup_cli.utils.peft_builder import build_peft_config


class TestInitStrategyField:
    def test_default_is_random(self):
        cfg = LoraConfig()
        assert cfg.init_strategy == "random"

    def test_accepts_pissa(self):
        cfg = LoraConfig(init_strategy="pissa")
        assert cfg.init_strategy == "pissa"

    def test_accepts_olora(self):
        cfg = LoraConfig(init_strategy="olora")
        assert cfg.init_strategy == "olora"

    def test_rejects_unknown(self):
        with pytest.raises(ValidationError):
            LoraConfig(init_strategy="loftq")

    def test_rejects_non_string(self):
        with pytest.raises(ValidationError):
            LoraConfig(init_strategy=42)


class TestBackcompatNoMutation:
    def test_backcompat_does_not_mutate_input_dict(self):
        """v0.39.0 security fix — _backcompat_align_olora must copy."""
        original = {"use_olora": True}
        LoraConfig(**original)
        assert "init_strategy" not in original, (
            "Validator mutated caller's dict in-place"
        )


class TestInitStrategyOloraBackcompat:
    def test_use_olora_true_alone_still_works(self):
        cfg = LoraConfig(use_olora=True)
        assert cfg.use_olora is True
        assert cfg.init_strategy == "olora"  # auto-aligned for back-compat

    def test_init_strategy_olora_alone_works(self):
        cfg = LoraConfig(init_strategy="olora")
        assert cfg.init_strategy == "olora"
        # use_olora may be True or False; effect is via init_strategy

    def test_both_set_consistent_ok(self):
        cfg = LoraConfig(use_olora=True, init_strategy="olora")
        assert cfg.init_strategy == "olora"

    def test_use_olora_true_with_pissa_rejected(self):
        with pytest.raises(ValidationError, match="init_strategy"):
            LoraConfig(use_olora=True, init_strategy="pissa")

    def test_use_olora_true_with_random_rejected(self):
        # explicit conflict — user said both random and olora; loud-fail
        with pytest.raises(ValidationError, match="init_strategy"):
            LoraConfig(use_olora=True, init_strategy="random")


class TestInitStrategyMutualExclusion:
    def test_pissa_with_dora_rejected(self):
        # PiSSA + DoRA isn't supported (PEFT init_lora_weights conflicts with use_dora init)
        with pytest.raises(ValidationError, match="init_strategy"):
            LoraConfig(use_dora=True, init_strategy="pissa")

    def test_pissa_with_vera_rejected(self):
        # VeRA doesn't have init_lora_weights — PiSSA meaningless
        with pytest.raises(ValidationError, match="init_strategy"):
            LoraConfig(use_vera=True, init_strategy="pissa")

    def test_pissa_with_rslora_ok(self):
        # rsLoRA only changes scaling factor — orthogonal to init
        cfg = LoraConfig(use_rslora=True, init_strategy="pissa")
        assert cfg.init_strategy == "pissa"
        assert cfg.use_rslora is True


class TestPeftBuilderInitStrategy:
    def test_random_does_not_set_init_lora_weights(self):
        cfg = LoraConfig(init_strategy="random")
        spec = build_peft_config(cfg, target_modules="auto", task_type="CAUSAL_LM")
        assert spec["peft_cls"] == "LoraConfig"
        assert "init_lora_weights" not in spec["init_kwargs"]

    def test_pissa_sets_init_lora_weights(self):
        cfg = LoraConfig(init_strategy="pissa")
        spec = build_peft_config(cfg, target_modules="auto", task_type="CAUSAL_LM")
        assert spec["init_kwargs"]["init_lora_weights"] == "pissa"

    def test_olora_via_init_strategy(self):
        cfg = LoraConfig(init_strategy="olora")
        spec = build_peft_config(cfg, target_modules="auto", task_type="CAUSAL_LM")
        assert spec["init_kwargs"]["init_lora_weights"] == "olora"

    def test_olora_via_use_olora_legacy(self):
        cfg = LoraConfig(use_olora=True)
        spec = build_peft_config(cfg, target_modules="auto", task_type="CAUSAL_LM")
        assert spec["init_kwargs"]["init_lora_weights"] == "olora"

    def test_vera_ignores_init_strategy_random(self):
        cfg = LoraConfig(use_vera=True, init_strategy="random")
        spec = build_peft_config(cfg, target_modules="auto", task_type="CAUSAL_LM")
        assert spec["peft_cls"] == "VeraConfig"
        assert "init_lora_weights" not in spec["init_kwargs"]


class TestInstantiatePeftConfig:
    def test_instantiate_lora_config(self):
        from soup_cli.utils.peft_builder import instantiate_peft_config
        cfg = LoraConfig(init_strategy="pissa")
        spec = build_peft_config(cfg, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
        result = instantiate_peft_config(spec)
        # Verify it really is a peft.LoraConfig with the right kwargs
        import peft
        assert isinstance(result, peft.LoraConfig)
        assert result.r == 64
        assert result.init_lora_weights == "pissa"

    def test_instantiate_with_rank_pattern(self):
        from soup_cli.utils.peft_builder import instantiate_peft_config
        cfg = LoraConfig(rank_pattern={"q_proj": 8})
        spec = build_peft_config(cfg, target_modules=["q_proj"], task_type="CAUSAL_LM")
        result = instantiate_peft_config(spec)
        assert result.rank_pattern == {"q_proj": 8}
