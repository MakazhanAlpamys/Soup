"""Tests for v0.39.0 Part C — per-pattern LoRA rank/alpha."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from soup_cli.config.schema import LoraConfig
from soup_cli.utils.peft_builder import build_peft_config


class TestRankPatternSchema:
    def test_default_none(self):
        cfg = LoraConfig()
        assert cfg.rank_pattern is None
        assert cfg.alpha_pattern is None

    def test_rank_pattern_dict_accepted(self):
        cfg = LoraConfig(rank_pattern={"q_proj": 8, "v_proj": 16})
        assert cfg.rank_pattern == {"q_proj": 8, "v_proj": 16}

    def test_alpha_pattern_dict_accepted(self):
        cfg = LoraConfig(alpha_pattern={"q_proj": 16, "v_proj": 32})
        assert cfg.alpha_pattern == {"q_proj": 16, "v_proj": 32}

    def test_rank_pattern_rejects_non_int_value(self):
        with pytest.raises(ValidationError):
            LoraConfig(rank_pattern={"q_proj": "high"})

    def test_rank_pattern_rejects_negative(self):
        with pytest.raises(ValidationError):
            LoraConfig(rank_pattern={"q_proj": -1})

    def test_rank_pattern_rejects_zero(self):
        with pytest.raises(ValidationError):
            LoraConfig(rank_pattern={"q_proj": 0})

    def test_rank_pattern_rejects_too_large(self):
        with pytest.raises(ValidationError):
            LoraConfig(rank_pattern={"q_proj": 10_000})

    def test_rank_pattern_rejects_empty_key(self):
        with pytest.raises(ValidationError):
            LoraConfig(rank_pattern={"": 8})

    def test_rank_pattern_rejects_null_byte_key(self):
        with pytest.raises(ValidationError):
            LoraConfig(rank_pattern={"q\x00proj": 8})

    def test_rank_pattern_rejects_too_many_keys(self):
        # Cap at 256 patterns to prevent absurd configs
        big = {f"k{i}": 8 for i in range(257)}
        with pytest.raises(ValidationError):
            LoraConfig(rank_pattern=big)

    def test_rank_pattern_rejects_bool_value(self):
        # bool is subclass of int in Python — exclude explicitly
        with pytest.raises(ValidationError):
            LoraConfig(rank_pattern={"q_proj": True})


class TestRankPatternMutualExclusion:
    def test_rank_pattern_with_vera_rejected(self):
        with pytest.raises(ValidationError, match="rank_pattern"):
            LoraConfig(use_vera=True, rank_pattern={"q_proj": 8})

    def test_alpha_pattern_with_vera_rejected(self):
        with pytest.raises(ValidationError, match="alpha_pattern"):
            LoraConfig(use_vera=True, alpha_pattern={"q_proj": 16})

    def test_rank_pattern_with_dora_ok(self):
        # DoRA still uses standard LoraConfig; rank_pattern works
        cfg = LoraConfig(use_dora=True, rank_pattern={"q_proj": 8})
        assert cfg.rank_pattern == {"q_proj": 8}


class TestPeftBuilderRankPattern:
    def test_rank_pattern_propagated(self):
        cfg = LoraConfig(rank_pattern={"q_proj": 8, "v_proj": 16})
        spec = build_peft_config(cfg, target_modules="auto", task_type="CAUSAL_LM")
        assert spec["init_kwargs"]["rank_pattern"] == {"q_proj": 8, "v_proj": 16}

    def test_alpha_pattern_propagated(self):
        cfg = LoraConfig(alpha_pattern={"q_proj": 16})
        spec = build_peft_config(cfg, target_modules="auto", task_type="CAUSAL_LM")
        assert spec["init_kwargs"]["alpha_pattern"] == {"q_proj": 16}

    def test_neither_pattern_omitted_when_none(self):
        cfg = LoraConfig()
        spec = build_peft_config(cfg, target_modules="auto", task_type="CAUSAL_LM")
        assert "rank_pattern" not in spec["init_kwargs"]
        assert "alpha_pattern" not in spec["init_kwargs"]

    def test_vera_path_ignores_patterns_when_unset(self):
        cfg = LoraConfig(use_vera=True)
        spec = build_peft_config(cfg, target_modules="auto", task_type="CAUSAL_LM")
        assert spec["peft_cls"] == "VeraConfig"
        assert "rank_pattern" not in spec["init_kwargs"]
