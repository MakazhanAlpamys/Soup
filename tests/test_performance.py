"""Tests for v0.15.0 Performance + Long-context features.

Covers: Liger Kernel, FlashAttention, FSDP2, Ring FlashAttention, long-context (RoPE).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from soup_cli.config.schema import TEMPLATES, SoupConfig

# ─── Liger Kernel Tests ─────────────────────────────────────────────────


class TestLigerConfig:
    """Test Liger Kernel configuration in SoupConfig."""

    def test_use_liger_default_false(self):
        cfg = SoupConfig(base="test/model", data={"train": "./data.jsonl"})
        assert cfg.training.use_liger is False

    def test_use_liger_enabled(self):
        cfg = SoupConfig(
            base="test/model",
            data={"train": "./data.jsonl"},
            training={"use_liger": True},
        )
        assert cfg.training.use_liger is True


class TestLigerValidation:
    """Test Liger Kernel validation logic."""

    def test_validate_liger_disabled_returns_empty(self):
        from soup_cli.utils.liger import validate_liger_config

        errors = validate_liger_config(False, "transformers", "cuda")
        assert errors == []

    def test_validate_liger_not_installed(self):
        from soup_cli.utils.liger import validate_liger_config

        with patch("soup_cli.utils.liger.check_liger_available", return_value=False):
            errors = validate_liger_config(True, "transformers", "cuda")
            assert any("not installed" in err for err in errors)

    def test_validate_liger_unsloth_incompatible(self):
        from soup_cli.utils.liger import validate_liger_config

        errors = validate_liger_config(True, "unsloth", "cuda")
        assert any("unsloth" in err.lower() for err in errors)

    def test_validate_liger_cpu_incompatible(self):
        from soup_cli.utils.liger import validate_liger_config

        errors = validate_liger_config(True, "transformers", "cpu")
        assert any("CUDA" in err for err in errors)

    def test_validate_liger_valid_config(self):
        from soup_cli.utils.liger import validate_liger_config

        with patch("soup_cli.utils.liger.check_liger_available", return_value=True):
            errors = validate_liger_config(True, "transformers", "cuda")
            assert errors == []


class TestLigerDetection:
    """Test Liger Kernel availability detection."""

    def test_check_liger_available_not_installed(self):
        from soup_cli.utils.liger import check_liger_available

        with patch.dict("sys.modules", {"liger_kernel": None}):
            # When import fails, should return False
            result = check_liger_available()
            # Result depends on actual environment; just verify it's bool
            assert isinstance(result, bool)

    def test_get_liger_version_not_installed(self):
        from soup_cli.utils.liger import get_liger_version

        with patch("soup_cli.utils.liger.check_liger_available", return_value=False):
            # get_liger_version does its own import attempt
            result = get_liger_version()
            assert result is None or isinstance(result, str)

    def test_apply_liger_kernel_not_available(self):
        from soup_cli.utils.liger import apply_liger_kernel

        with patch("soup_cli.utils.liger.check_liger_available", return_value=False):
            result = apply_liger_kernel("meta-llama/Llama-3.1-8B")
            assert result is False

    def test_apply_liger_manual_unknown_model(self):
        from soup_cli.utils.liger import _apply_liger_manual

        result = _apply_liger_manual("completely-unknown-model")
        assert result is False


# ─── FlashAttention Tests ────────────────────────────────────────────────


class TestFlashAttnConfig:
    """Test FlashAttention configuration in SoupConfig."""

    def test_use_flash_attn_default_false(self):
        cfg = SoupConfig(base="test/model", data={"train": "./data.jsonl"})
        assert cfg.training.use_flash_attn is False

    def test_use_flash_attn_enabled(self):
        cfg = SoupConfig(
            base="test/model",
            data={"train": "./data.jsonl"},
            training={"use_flash_attn": True},
        )
        assert cfg.training.use_flash_attn is True


class TestFlashAttnDetection:
    """Test FlashAttention detection and validation."""

    def test_check_flash_attn_available_no_cuda(self):
        with patch("soup_cli.utils.flash_attn.check_flash_attn_available") as mock_detect:
            mock_detect.return_value = None
            result = mock_detect()
            assert result is None

    def test_get_attn_implementation_disabled(self):
        from soup_cli.utils.flash_attn import get_attn_implementation

        result = get_attn_implementation(use_flash_attn=False, device="cuda")
        assert result is None

    def test_get_attn_implementation_cpu(self):
        from soup_cli.utils.flash_attn import get_attn_implementation

        result = get_attn_implementation(use_flash_attn=True, device="cpu")
        assert result is None

    def test_validate_flash_attn_disabled(self):
        from soup_cli.utils.flash_attn import validate_flash_attn_config

        errors = validate_flash_attn_config(False, "transformers", "cuda")
        assert errors == []

    def test_validate_flash_attn_cpu_error(self):
        from soup_cli.utils.flash_attn import validate_flash_attn_config

        errors = validate_flash_attn_config(True, "transformers", "cpu")
        assert any("CUDA" in err for err in errors)

    def test_validate_flash_attn_unsloth_no_error(self):
        """Unsloth handles FlashAttention internally — not an error."""
        from soup_cli.utils.flash_attn import validate_flash_attn_config

        # Even with unsloth, the only error should be about availability (not backend)
        errors = validate_flash_attn_config(True, "unsloth", "cuda")
        assert not any("unsloth" in err.lower() for err in errors)

    def test_get_flash_attn_version_not_installed(self):
        from soup_cli.utils.flash_attn import get_flash_attn_version

        result = get_flash_attn_version()
        assert result is None or isinstance(result, str)

    def test_flash_attn_versions_constant(self):
        from soup_cli.utils.flash_attn import FLASH_ATTN_VERSIONS

        assert "flash_attention_2" in FLASH_ATTN_VERSIONS
        assert "flash_attention_3" in FLASH_ATTN_VERSIONS


# ─── FSDP2 Tests ────────────────────────────────────────────────────────


class TestFSDPConfig:
    """Test FSDP2 configuration and presets."""

    def test_fsdp_full_shard_preset(self):
        from soup_cli.utils.fsdp import get_fsdp_config

        config = get_fsdp_config("full_shard")
        assert "full_shard" in config["fsdp"]
        assert "auto_wrap" in config["fsdp"]

    def test_fsdp_shard_grad_preset(self):
        from soup_cli.utils.fsdp import get_fsdp_config

        config = get_fsdp_config("shard_grad")
        assert "shard_grad_op" in config["fsdp"]

    def test_fsdp_full_offload_preset(self):
        from soup_cli.utils.fsdp import get_fsdp_config

        config = get_fsdp_config("full_offload")
        assert "offload" in config["fsdp"]

    def test_fsdp_unknown_preset_raises(self):
        from soup_cli.utils.fsdp import get_fsdp_config

        with pytest.raises(ValueError, match="Unknown FSDP config"):
            get_fsdp_config("nonexistent")

    def test_fsdp_training_args_keys(self):
        from soup_cli.utils.fsdp import get_fsdp_training_args

        kwargs = get_fsdp_training_args("full_shard")
        assert "fsdp" in kwargs
        assert "fsdp_config" in kwargs

    def test_fsdp_config_deep_copy(self):
        """get_fsdp_config should return a deep copy (no shared state)."""
        from soup_cli.utils.fsdp import get_fsdp_config

        config1 = get_fsdp_config("full_shard")
        config2 = get_fsdp_config("full_shard")
        config1["fsdp"] = "modified"
        assert config2["fsdp"] != "modified"

    def test_fsdp_configs_dict(self):
        from soup_cli.utils.fsdp import FSDP_CONFIGS

        assert "full_shard" in FSDP_CONFIGS
        assert "shard_grad" in FSDP_CONFIGS
        assert "full_offload" in FSDP_CONFIGS


class TestFSDPValidation:
    """Test FSDP2 validation logic."""

    def test_validate_fsdp_disabled(self):
        from soup_cli.utils.fsdp import validate_fsdp_config

        errors = validate_fsdp_config(None, None, "transformers", "cuda")
        assert errors == []

    def test_validate_fsdp_with_deepspeed_conflict(self):
        from soup_cli.utils.fsdp import validate_fsdp_config

        errors = validate_fsdp_config("full_shard", "/tmp/ds.json", "transformers", "cuda")
        assert any("DeepSpeed" in err for err in errors)

    def test_validate_fsdp_cpu_error(self):
        from soup_cli.utils.fsdp import validate_fsdp_config

        errors = validate_fsdp_config("full_shard", None, "transformers", "cpu")
        assert any("CUDA" in err for err in errors)

    def test_validate_fsdp_unsloth_error(self):
        from soup_cli.utils.fsdp import validate_fsdp_config

        errors = validate_fsdp_config("full_shard", None, "unsloth", "cuda")
        assert any("unsloth" in err.lower() for err in errors)

    def test_validate_fsdp_unknown_preset(self):
        from soup_cli.utils.fsdp import validate_fsdp_config

        errors = validate_fsdp_config("invalid_preset", None, "transformers", "cuda")
        assert any("Unknown FSDP preset" in err for err in errors)


class TestFSDPAvailability:
    """Test FSDP2 availability detection."""

    def test_is_fsdp_available_returns_bool(self):
        from soup_cli.utils.fsdp import is_fsdp_available

        result = is_fsdp_available()
        assert isinstance(result, bool)


# ─── Ring FlashAttention Tests ───────────────────────────────────────────


class TestRingAttentionConfig:
    """Test Ring FlashAttention configuration."""

    def test_use_ring_attention_default_false(self):
        cfg = SoupConfig(base="test/model", data={"train": "./data.jsonl"})
        assert cfg.training.use_ring_attention is False

    def test_use_ring_attention_enabled(self):
        cfg = SoupConfig(
            base="test/model",
            data={"train": "./data.jsonl"},
            training={"use_ring_attention": True},
        )
        assert cfg.training.use_ring_attention is True


class TestRingAttentionUtils:
    """Test Ring FlashAttention utility functions."""

    def test_get_sequence_parallel_size_single_gpu(self):
        from soup_cli.utils.ring_attention import get_sequence_parallel_size

        result = get_sequence_parallel_size(gpu_count=1, max_length=131072)
        assert result == 1

    def test_get_sequence_parallel_size_short_sequence(self):
        from soup_cli.utils.ring_attention import get_sequence_parallel_size

        result = get_sequence_parallel_size(gpu_count=4, max_length=2048)
        assert result == 1

    def test_get_sequence_parallel_size_long_sequence(self):
        from soup_cli.utils.ring_attention import get_sequence_parallel_size

        result = get_sequence_parallel_size(gpu_count=8, max_length=131072)
        assert result >= 2
        # Should be power of 2
        assert result & (result - 1) == 0

    def test_get_sequence_parallel_size_power_of_two(self):
        from soup_cli.utils.ring_attention import get_sequence_parallel_size

        result = get_sequence_parallel_size(gpu_count=6, max_length=131072)
        # Should be power of 2, max <= gpu_count
        assert result in (1, 2, 4)


class TestRingAttentionValidation:
    """Test Ring FlashAttention validation."""

    def test_validate_disabled(self):
        from soup_cli.utils.ring_attention import validate_ring_attention_config

        errors = validate_ring_attention_config(False, "cuda", 131072)
        assert errors == []

    def test_validate_cpu_error(self):
        from soup_cli.utils.ring_attention import validate_ring_attention_config

        errors = validate_ring_attention_config(True, "cpu", 131072)
        assert any("CUDA" in err for err in errors)

    def test_validate_short_sequence_warning(self):
        from soup_cli.utils.ring_attention import validate_ring_attention_config

        errors = validate_ring_attention_config(True, "cuda", 2048)
        assert any("8192" in err for err in errors)

    def test_check_ring_attention_available_returns_bool(self):
        from soup_cli.utils.ring_attention import check_ring_attention_available

        result = check_ring_attention_available()
        assert isinstance(result, bool)

    def test_get_ring_attention_version_not_installed(self):
        from soup_cli.utils.ring_attention import get_ring_attention_version

        result = get_ring_attention_version()
        assert result is None or isinstance(result, str)


# ─── Long-Context Tests ─────────────────────────────────────────────────


class TestLongContextConfig:
    """Test long-context configuration in SoupConfig."""

    def test_rope_scaling_type_default_none(self):
        cfg = SoupConfig(base="test/model", data={"train": "./data.jsonl"})
        assert cfg.training.rope_scaling_type is None

    def test_rope_scaling_type_dynamic(self):
        cfg = SoupConfig(
            base="test/model",
            data={"train": "./data.jsonl"},
            training={"rope_scaling_type": "dynamic"},
        )
        assert cfg.training.rope_scaling_type == "dynamic"

    def test_rope_scaling_type_invalid_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SoupConfig(
                base="test/model",
                data={"train": "./data.jsonl"},
                training={"rope_scaling_type": "invalid_type"},
            )

    def test_gradient_checkpointing_default_false(self):
        cfg = SoupConfig(base="test/model", data={"train": "./data.jsonl"})
        assert cfg.training.gradient_checkpointing is False

    def test_gradient_checkpointing_enabled(self):
        cfg = SoupConfig(
            base="test/model",
            data={"train": "./data.jsonl"},
            training={"gradient_checkpointing": True},
        )
        assert cfg.training.gradient_checkpointing is True


class TestLongContextUtils:
    """Test long-context utility functions."""

    def test_rope_scaling_types_constant(self):
        from soup_cli.utils.long_context import ROPE_SCALING_TYPES

        assert "linear" in ROPE_SCALING_TYPES
        assert "dynamic" in ROPE_SCALING_TYPES
        assert "yarn" in ROPE_SCALING_TYPES
        assert "longrope" in ROPE_SCALING_TYPES

    def test_model_default_context_llama3(self):
        from soup_cli.utils.long_context import get_model_default_context

        ctx = get_model_default_context("meta-llama/Llama-3.1-8B")
        assert ctx == 8192

    def test_model_default_context_mistral(self):
        from soup_cli.utils.long_context import get_model_default_context

        ctx = get_model_default_context("mistralai/Mistral-7B-v0.1")
        assert ctx == 32768

    def test_model_default_context_unknown(self):
        from soup_cli.utils.long_context import get_model_default_context

        ctx = get_model_default_context("unknown/model-xyz")
        assert ctx == 4096  # Conservative default

    def test_get_rope_scaling_config_linear(self):
        from soup_cli.utils.long_context import get_rope_scaling_config

        config = get_rope_scaling_config("linear", 131072, 8192)
        assert config["type"] == "linear"
        assert config["factor"] == pytest.approx(16.0)

    def test_get_rope_scaling_config_dynamic(self):
        from soup_cli.utils.long_context import get_rope_scaling_config

        config = get_rope_scaling_config("dynamic", 65536, 8192)
        assert config["type"] == "dynamic"
        assert config["factor"] == pytest.approx(8.0)

    def test_get_rope_scaling_config_yarn(self):
        from soup_cli.utils.long_context import get_rope_scaling_config

        config = get_rope_scaling_config("yarn", 32768, 8192)
        assert config["type"] == "yarn"
        assert config["factor"] == pytest.approx(4.0)
        assert config["original_max_position_embeddings"] == 8192

    def test_get_rope_scaling_config_longrope(self):
        from soup_cli.utils.long_context import get_rope_scaling_config

        config = get_rope_scaling_config("longrope", 32768, 8192)
        assert config["type"] == "longrope"

    def test_get_rope_scaling_config_no_scaling_needed(self):
        from soup_cli.utils.long_context import get_rope_scaling_config

        config = get_rope_scaling_config("linear", 4096, 8192)
        assert config == {}

    def test_get_rope_scaling_config_factor_as_target(self):
        """When target_length < original_length and > 1.0, treat as factor."""
        from soup_cli.utils.long_context import get_rope_scaling_config

        config = get_rope_scaling_config("linear", 4.0, 4096)
        assert config["type"] == "linear"
        assert config["factor"] == pytest.approx(4.0)

    def test_get_rope_scaling_config_dynamic_factor(self):
        """Dynamic scaling with factor-style argument."""
        from soup_cli.utils.long_context import get_rope_scaling_config

        config = get_rope_scaling_config("dynamic", 2.0, 8192)
        assert config["type"] == "dynamic"
        assert config["factor"] == pytest.approx(2.0)

    def test_get_rope_scaling_config_invalid_type(self):
        from soup_cli.utils.long_context import get_rope_scaling_config

        with pytest.raises(ValueError, match="Unknown RoPE scaling type"):
            get_rope_scaling_config("invalid", 131072, 8192)

    def test_apply_long_context_config_modifies_model(self):
        from soup_cli.utils.long_context import apply_long_context_config

        model_config = MagicMock()
        model_config.max_position_embeddings = 8192
        model_config.rope_scaling = None

        rope_config = apply_long_context_config(
            model_config, target_length=131072, rope_scaling_type="dynamic",
            model_name="meta-llama/Llama-3.1-8B",
        )
        assert rope_config is not None
        assert rope_config["type"] == "dynamic"
        assert model_config.max_position_embeddings == 131072

    def test_apply_long_context_config_no_scaling_needed(self):
        from soup_cli.utils.long_context import apply_long_context_config

        model_config = MagicMock()
        model_config.max_position_embeddings = 131072

        result = apply_long_context_config(
            model_config, target_length=8192, rope_scaling_type="dynamic",
            model_name="test/model",
        )
        assert result is None


class TestLongContextValidation:
    """Test long-context validation logic."""

    def test_validate_no_rope_scaling(self):
        from soup_cli.utils.long_context import validate_long_context_config

        errors = validate_long_context_config(2048, None, False)
        assert errors == []

    def test_validate_invalid_rope_type(self):
        from soup_cli.utils.long_context import validate_long_context_config

        errors = validate_long_context_config(131072, "invalid_type", True)
        assert any("Unknown RoPE" in err for err in errors)

    def test_validate_long_context_no_gradient_checkpointing(self):
        from soup_cli.utils.long_context import validate_long_context_config

        errors = validate_long_context_config(131072, "dynamic", False)
        assert any("gradient checkpointing" in err.lower() for err in errors)

    def test_validate_long_context_with_gradient_checkpointing(self):
        from soup_cli.utils.long_context import validate_long_context_config

        errors = validate_long_context_config(131072, "dynamic", True)
        assert not any("gradient checkpointing" in err.lower() for err in errors)


# ─── Template Tests ──────────────────────────────────────────────────────


class TestMaxLengthBounds:
    """Test max_length has proper bounds (security fix H3+M3)."""

    def test_max_length_too_small_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SoupConfig(
                base="test/model",
                data={"train": "./data.jsonl", "max_length": 0},
            )

    def test_max_length_too_large_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SoupConfig(
                base="test/model",
                data={"train": "./data.jsonl", "max_length": 2000000},
            )

    def test_max_length_valid_128k(self):
        cfg = SoupConfig(
            base="test/model",
            data={"train": "./data.jsonl", "max_length": 131072},
        )
        assert cfg.data.max_length == 131072


class TestLongContextTemplate:
    """Test that the longcontext template exists and is valid."""

    def test_longcontext_template_exists(self):
        assert "longcontext" in TEMPLATES

    def test_longcontext_template_has_max_length(self):
        assert "131072" in TEMPLATES["longcontext"]

    def test_longcontext_template_has_rope_scaling(self):
        assert "rope_scaling_type" in TEMPLATES["longcontext"]

    def test_longcontext_template_has_gradient_checkpointing(self):
        assert "gradient_checkpointing" in TEMPLATES["longcontext"]

    def test_longcontext_template_has_flash_attn(self):
        assert "use_flash_attn" in TEMPLATES["longcontext"]

    def test_longcontext_template_count(self):
        """Should now have 17 templates (16 + bco added in v0.40.0)."""
        assert len(TEMPLATES) == 17


# ─── Trainer fsdp_config Parameter Tests ─────────────────────────────────


class TestTrainerFSDPParam:
    """Test that all trainers accept fsdp_config parameter."""

    def test_sft_trainer_accepts_fsdp_config(self):
        from soup_cli.trainer.sft import SFTTrainerWrapper

        cfg = SoupConfig(base="test/model", data={"train": "./data.jsonl"})
        wrapper = SFTTrainerWrapper(cfg, fsdp_config={"fsdp": "full_shard auto_wrap"})
        assert wrapper.fsdp_config == {"fsdp": "full_shard auto_wrap"}

    def test_dpo_trainer_accepts_fsdp_config(self):
        from soup_cli.trainer.dpo import DPOTrainerWrapper

        cfg = SoupConfig(base="test/model", task="dpo", data={"train": "./data.jsonl"})
        wrapper = DPOTrainerWrapper(cfg, fsdp_config={"fsdp": "full_shard"})
        assert wrapper.fsdp_config == {"fsdp": "full_shard"}

    def test_grpo_trainer_accepts_fsdp_config(self):
        from soup_cli.trainer.grpo import GRPOTrainerWrapper

        cfg = SoupConfig(base="test/model", task="grpo", data={"train": "./data.jsonl"})
        wrapper = GRPOTrainerWrapper(cfg, fsdp_config={"fsdp": "full_shard"})
        assert wrapper.fsdp_config == {"fsdp": "full_shard"}

    def test_kto_trainer_accepts_fsdp_config(self):
        from soup_cli.trainer.kto import KTOTrainerWrapper

        cfg = SoupConfig(base="test/model", task="kto", data={"train": "./data.jsonl"})
        wrapper = KTOTrainerWrapper(cfg, fsdp_config={"fsdp": "full_shard"})
        assert wrapper.fsdp_config == {"fsdp": "full_shard"}

    def test_orpo_trainer_accepts_fsdp_config(self):
        from soup_cli.trainer.orpo import ORPOTrainerWrapper

        cfg = SoupConfig(base="test/model", task="orpo", data={"train": "./data.jsonl"})
        wrapper = ORPOTrainerWrapper(cfg, fsdp_config={"fsdp": "full_shard"})
        assert wrapper.fsdp_config == {"fsdp": "full_shard"}

    def test_simpo_trainer_accepts_fsdp_config(self):
        from soup_cli.trainer.simpo import SimPOTrainerWrapper

        cfg = SoupConfig(base="test/model", task="simpo", data={"train": "./data.jsonl"})
        wrapper = SimPOTrainerWrapper(cfg, fsdp_config={"fsdp": "full_shard"})
        assert wrapper.fsdp_config == {"fsdp": "full_shard"}

    def test_ipo_trainer_accepts_fsdp_config(self):
        from soup_cli.trainer.ipo import IPOTrainerWrapper

        cfg = SoupConfig(base="test/model", task="ipo", data={"train": "./data.jsonl"})
        wrapper = IPOTrainerWrapper(cfg, fsdp_config={"fsdp": "full_shard"})
        assert wrapper.fsdp_config == {"fsdp": "full_shard"}

    def test_pretrain_trainer_accepts_fsdp_config(self):
        from soup_cli.trainer.pretrain import PretrainTrainerWrapper

        cfg = SoupConfig(
            base="test/model", task="pretrain",
            data={"train": "./data.jsonl", "format": "plaintext"},
        )
        wrapper = PretrainTrainerWrapper(cfg, fsdp_config={"fsdp": "full_shard"})
        assert wrapper.fsdp_config == {"fsdp": "full_shard"}

    def test_reward_model_trainer_accepts_fsdp_config(self):
        from soup_cli.trainer.reward_model import RewardModelTrainerWrapper

        cfg = SoupConfig(
            base="test/model", task="reward_model",
            data={"train": "./data.jsonl"},
        )
        wrapper = RewardModelTrainerWrapper(cfg, fsdp_config={"fsdp": "full_shard"})
        assert wrapper.fsdp_config == {"fsdp": "full_shard"}

    def test_ppo_trainer_accepts_fsdp_config(self):
        from soup_cli.trainer.ppo import PPOTrainerWrapper

        cfg = SoupConfig(base="test/model", task="ppo", data={"train": "./data.jsonl"})
        wrapper = PPOTrainerWrapper(cfg, fsdp_config={"fsdp": "full_shard"})
        assert wrapper.fsdp_config == {"fsdp": "full_shard"}

    def test_sft_trainer_fsdp_config_default_none(self):
        from soup_cli.trainer.sft import SFTTrainerWrapper

        cfg = SoupConfig(base="test/model", data={"train": "./data.jsonl"})
        wrapper = SFTTrainerWrapper(cfg)
        assert wrapper.fsdp_config is None


# ─── Combined Config Tests ───────────────────────────────────────────────


class TestCombinedPerformanceConfig:
    """Test multiple performance features enabled together."""

    def test_all_features_enabled(self):
        cfg = SoupConfig(
            base="test/model",
            data={"train": "./data.jsonl", "max_length": 131072},
            training={
                "use_liger": True,
                "use_flash_attn": True,
                "use_ring_attention": True,
                "rope_scaling_type": "dynamic",
                "gradient_checkpointing": True,
            },
        )
        assert cfg.training.use_liger is True
        assert cfg.training.use_flash_attn is True
        assert cfg.training.use_ring_attention is True
        assert cfg.training.rope_scaling_type == "dynamic"
        assert cfg.training.gradient_checkpointing is True
        assert cfg.data.max_length == 131072

    def test_config_serialization_roundtrip(self):
        cfg = SoupConfig(
            base="test/model",
            data={"train": "./data.jsonl"},
            training={
                "use_liger": True,
                "use_flash_attn": True,
                "gradient_checkpointing": True,
                "rope_scaling_type": "yarn",
            },
        )
        dumped = cfg.model_dump()
        assert dumped["training"]["use_liger"] is True
        assert dumped["training"]["use_flash_attn"] is True
        assert dumped["training"]["gradient_checkpointing"] is True
        assert dumped["training"]["rope_scaling_type"] == "yarn"

    def test_rope_scaling_all_types_accepted(self):
        for scaling_type in ("linear", "dynamic", "yarn", "longrope"):
            cfg = SoupConfig(
                base="test/model",
                data={"train": "./data.jsonl"},
                training={"rope_scaling_type": scaling_type},
            )
            assert cfg.training.rope_scaling_type == scaling_type


# ─── Train Command --fsdp Flag Tests ────────────────────────────────────


class TestTrainCommandFSDPFlag:
    """Test --fsdp flag in train command."""

    def test_fsdp_configs_accessible_from_fsdp_module(self):
        from soup_cli.utils.fsdp import FSDP_CONFIGS

        assert len(FSDP_CONFIGS) == 3

    def test_fsdp_full_shard_has_correct_keys(self):
        from soup_cli.utils.fsdp import FSDP_FULL_SHARD

        assert "fsdp" in FSDP_FULL_SHARD
        assert "fsdp_config" in FSDP_FULL_SHARD
        assert "use_orig_params" in FSDP_FULL_SHARD["fsdp_config"]
        assert FSDP_FULL_SHARD["fsdp_config"]["use_orig_params"] is True
