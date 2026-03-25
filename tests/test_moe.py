"""Tests for MoE model detection, ScatterMoE LoRA target modules, and MoE info."""

from unittest.mock import MagicMock

from soup_cli.utils.moe import (
    MOE_CONFIG_KEYS,
    MOE_EXPERT_PATTERNS,
    STANDARD_TARGET_MODULES,
    detect_moe_model,
    get_moe_info,
    get_moe_target_modules,
)

# ─── MoE Detection Tests ──────────────────────────────────────────────────


class TestDetectMoEModel:
    """Test MoE model detection from model config."""

    def _make_model(self, **config_attrs):
        """Create a mock model with given config attributes."""
        model = MagicMock()
        config = MagicMock()
        # Reset all MOE_CONFIG_KEYS to None by default
        for key in MOE_CONFIG_KEYS:
            setattr(config, key, None)
        config.model_type = ""
        # Set requested attributes
        for key, value in config_attrs.items():
            setattr(config, key, value)
        model.config = config
        return model

    def test_detect_mixtral(self):
        """Should detect Mixtral as MoE (num_local_experts > 1)."""
        model = self._make_model(num_local_experts=8, model_type="mixtral")
        assert detect_moe_model(model) is True

    def test_detect_qwen3_moe(self):
        """Should detect Qwen3-MoE (num_experts > 1)."""
        model = self._make_model(num_experts=128, model_type="qwen3_moe")
        assert detect_moe_model(model) is True

    def test_detect_deepseek_v3(self):
        """Should detect DeepSeek V3 (n_routed_experts > 1)."""
        model = self._make_model(n_routed_experts=256, model_type="deepseek_v3")
        assert detect_moe_model(model) is True

    def test_detect_by_model_type_alone(self):
        """Should detect MoE by model_type even without expert count keys."""
        model = self._make_model(model_type="mixtral")
        assert detect_moe_model(model) is True

    def test_detect_by_config_key_alone(self):
        """Should detect MoE by config key even without known model_type."""
        model = self._make_model(num_local_experts=4, model_type="custom_moe")
        assert detect_moe_model(model) is True

    def test_non_moe_model(self):
        """Standard model should not be detected as MoE."""
        model = self._make_model(model_type="llama")
        assert detect_moe_model(model) is False

    def test_model_without_config(self):
        """Model without config attribute should return False."""
        model = MagicMock(spec=[])
        assert detect_moe_model(model) is False

    def test_single_expert_not_moe(self):
        """num_local_experts=1 should not be considered MoE."""
        model = self._make_model(num_local_experts=1, model_type="llama")
        assert detect_moe_model(model) is False

    def test_detect_dbrx(self):
        """Should detect DBRX as MoE."""
        model = self._make_model(model_type="dbrx")
        assert detect_moe_model(model) is True

    def test_detect_olmoe(self):
        """Should detect OLMoE as MoE."""
        model = self._make_model(model_type="olmoe")
        assert detect_moe_model(model) is True

    def test_detect_case_insensitive(self):
        """model_type detection should be case-insensitive."""
        model = self._make_model(model_type="Mixtral")
        assert detect_moe_model(model) is True


# ─── MoE Target Modules Tests ─────────────────────────────────────────────


class TestGetMoETargetModules:
    """Test ScatterMoE LoRA target module discovery."""

    def _make_moe_model(self, named_modules=None):
        """Create a mock MoE model with expert module names."""
        model = MagicMock()
        config = MagicMock()
        config.num_local_experts = 8
        config.model_type = "mixtral"
        for key in MOE_CONFIG_KEYS:
            if key != "num_local_experts":
                setattr(config, key, None)
        model.config = config

        if named_modules is None:
            named_modules = [
                ("model.layers.0.self_attn.q_proj", MagicMock()),
                ("model.layers.0.block_sparse_moe.experts.0.gate_proj", MagicMock()),
                ("model.layers.0.block_sparse_moe.experts.0.up_proj", MagicMock()),
                ("model.layers.0.block_sparse_moe.experts.0.down_proj", MagicMock()),
            ]
        model.named_modules.return_value = named_modules
        return model

    def test_returns_target_modules_for_moe(self):
        """Should return a list of target modules for MoE models."""
        model = self._make_moe_model()
        targets = get_moe_target_modules(model)
        assert targets is not None
        assert isinstance(targets, list)
        assert len(targets) > 0

    def test_includes_attention_modules(self):
        """Target modules should include standard attention modules."""
        model = self._make_moe_model()
        targets = get_moe_target_modules(model)
        for attn_module in STANDARD_TARGET_MODULES:
            assert attn_module in targets

    def test_includes_expert_modules(self):
        """Target modules should include discovered expert FFN modules."""
        model = self._make_moe_model()
        targets = get_moe_target_modules(model)
        assert "gate_proj" in targets
        assert "up_proj" in targets
        assert "down_proj" in targets

    def test_returns_none_for_non_moe(self):
        """Should return None for non-MoE models."""
        model = MagicMock()
        config = MagicMock()
        for key in MOE_CONFIG_KEYS:
            setattr(config, key, None)
        config.model_type = "llama"
        model.config = config
        targets = get_moe_target_modules(model)
        assert targets is None

    def test_fallback_expert_modules(self):
        """If no expert modules found, should include fallback patterns."""
        model = self._make_moe_model(named_modules=[
            ("model.layers.0.self_attn.q_proj", MagicMock()),
            ("model.layers.0.some_custom_layer", MagicMock()),
        ])
        targets = get_moe_target_modules(model)
        # Should fall back to standard expert patterns
        assert "gate_proj" in targets
        assert "up_proj" in targets
        assert "down_proj" in targets


# ─── MoE Info Tests ───────────────────────────────────────────────────────


class TestGetMoEInfo:
    """Test MoE architecture info extraction."""

    def test_mixtral_info(self):
        """Should extract info from Mixtral config."""
        model = MagicMock()
        config = MagicMock()
        config.num_local_experts = 8
        config.num_experts_per_tok = 2
        config.model_type = "mixtral"
        # Set other keys to None
        for key in ("num_experts", "n_routed_experts", "moe_num_experts",
                     "num_experts_per_token", "num_selected_experts"):
            setattr(config, key, None)
        model.config = config

        info = get_moe_info(model)
        assert info["num_experts"] == 8
        assert info["num_active_experts"] == 2
        assert info["model_type"] == "mixtral"

    def test_non_moe_info_empty(self):
        """Should return empty dict for non-MoE model."""
        model = MagicMock()
        config = MagicMock()
        for key in MOE_CONFIG_KEYS:
            setattr(config, key, None)
        # Also set active-expert keys to None to prevent MagicMock auto-creation
        for key in ("num_experts_per_tok", "num_experts_per_token",
                     "num_selected_experts"):
            setattr(config, key, None)
        config.model_type = "llama"
        model.config = config

        info = get_moe_info(model)
        assert info == {}

    def test_model_without_config(self):
        """Should return empty dict for model without config."""
        model = MagicMock(spec=[])
        info = get_moe_info(model)
        assert info == {}

    def test_qwen3_moe_info(self):
        """Should extract info from Qwen3-MoE config."""
        model = MagicMock()
        config = MagicMock()
        config.num_experts = 128
        config.num_experts_per_tok = 8
        config.model_type = "qwen3_moe"
        for key in ("num_local_experts", "n_routed_experts", "moe_num_experts",
                     "num_experts_per_token", "num_selected_experts"):
            setattr(config, key, None)
        model.config = config

        info = get_moe_info(model)
        assert info["num_experts"] == 128
        assert info["num_active_experts"] == 8

    def test_deepseek_info(self):
        """Should extract info from DeepSeek V3 config."""
        model = MagicMock()
        config = MagicMock()
        config.n_routed_experts = 256
        config.num_experts_per_token = 8
        config.model_type = "deepseek_v3"
        for key in ("num_local_experts", "num_experts", "moe_num_experts",
                     "num_experts_per_tok", "num_selected_experts"):
            setattr(config, key, None)
        model.config = config

        info = get_moe_info(model)
        assert info["num_experts"] == 256
        assert info["num_active_experts"] == 8


# ─── Constants Tests ──────────────────────────────────────────────────────


class TestMoEConstants:
    """Test that MoE constants are properly defined."""

    def test_moe_config_keys_non_empty(self):
        assert len(MOE_CONFIG_KEYS) > 0

    def test_moe_expert_patterns_non_empty(self):
        assert len(MOE_EXPERT_PATTERNS) > 0

    def test_standard_target_modules_has_attention(self):
        assert "q_proj" in STANDARD_TARGET_MODULES
        assert "v_proj" in STANDARD_TARGET_MODULES

    def test_expert_patterns_has_ffn(self):
        assert "gate_proj" in MOE_EXPERT_PATTERNS
        assert "up_proj" in MOE_EXPERT_PATTERNS
        assert "down_proj" in MOE_EXPERT_PATTERNS
