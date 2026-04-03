"""Tests for freeze training: freeze_layers, freeze_ratio config and layer freezing."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from soup_cli.config.schema import SoupConfig, TrainingConfig

# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestFreezeConfig:
    """Tests for freeze training config fields."""

    def test_freeze_layers_default_none(self):
        """freeze_layers defaults to None."""
        cfg = TrainingConfig()
        assert cfg.freeze_layers is None

    def test_freeze_ratio_default_none(self):
        """freeze_ratio defaults to None."""
        cfg = TrainingConfig()
        assert cfg.freeze_ratio is None

    def test_freeze_layers_valid(self):
        """freeze_layers accepts positive int."""
        cfg = TrainingConfig(freeze_layers=24)
        assert cfg.freeze_layers == 24

    def test_freeze_layers_zero_rejected(self):
        """freeze_layers must be >= 1."""
        with pytest.raises(Exception):
            TrainingConfig(freeze_layers=0)

    def test_freeze_layers_negative_rejected(self):
        """freeze_layers must be positive."""
        with pytest.raises(Exception):
            TrainingConfig(freeze_layers=-5)

    def test_freeze_ratio_valid(self):
        """freeze_ratio accepts float in (0, 1)."""
        cfg = TrainingConfig(freeze_ratio=0.75)
        assert cfg.freeze_ratio == 0.75

    def test_freeze_ratio_zero_rejected(self):
        """freeze_ratio must be > 0."""
        with pytest.raises(Exception):
            TrainingConfig(freeze_ratio=0.0)

    def test_freeze_ratio_one_rejected(self):
        """freeze_ratio must be < 1 (can't freeze everything)."""
        with pytest.raises(Exception):
            TrainingConfig(freeze_ratio=1.0)

    def test_freeze_ratio_over_one_rejected(self):
        """freeze_ratio must be < 1."""
        with pytest.raises(Exception):
            TrainingConfig(freeze_ratio=1.5)

    def test_freeze_layers_and_ratio_both_set(self):
        """Both freeze_layers and freeze_ratio can be set (layers takes priority)."""
        cfg = TrainingConfig(freeze_layers=10, freeze_ratio=0.5)
        assert cfg.freeze_layers == 10
        assert cfg.freeze_ratio == 0.5

    def test_freeze_in_yaml_roundtrip(self):
        """freeze fields survive YAML round-trip via SoupConfig."""
        cfg = SoupConfig(
            base="test/model",
            data={"train": "data.jsonl"},
            training={"freeze_layers": 16},
        )
        assert cfg.training.freeze_layers == 16

    def test_freeze_ratio_in_yaml_roundtrip(self):
        """freeze_ratio survives YAML round-trip."""
        cfg = SoupConfig(
            base="test/model",
            data={"train": "data.jsonl"},
            training={"freeze_ratio": 0.5},
        )
        assert cfg.training.freeze_ratio == 0.5


# ---------------------------------------------------------------------------
# Layer freezing logic
# ---------------------------------------------------------------------------


class TestFreezeModelLayers:
    """Tests for freeze_model_layers utility function."""

    def _make_mock_model(self, num_layers: int = 32):
        """Create a mock model with named_parameters."""
        model = MagicMock()
        params = []
        for layer_idx in range(num_layers):
            parts = [
                "self_attn.q_proj.weight",
                "self_attn.v_proj.weight",
                "mlp.up_proj.weight",
            ]
            for part in parts:
                param = MagicMock()
                param.requires_grad = True
                name = f"model.layers.{layer_idx}.{part}"
                params.append((name, param))
        # Add non-layer params (embed, lm_head)
        embed_param = MagicMock()
        embed_param.requires_grad = True
        params.append(("model.embed_tokens.weight", embed_param))

        head_param = MagicMock()
        head_param.requires_grad = True
        params.append(("lm_head.weight", head_param))

        model.named_parameters.return_value = params
        return model, params

    def test_freeze_by_layer_count(self):
        """freeze_model_layers freezes first N layers."""
        from soup_cli.utils.freeze import freeze_model_layers

        model, params = self._make_mock_model(32)
        frozen_count = freeze_model_layers(model, freeze_layers=24)

        # First 24 layers' params should have requires_grad = False
        for name, param in params:
            if "layers." in name:
                layer_idx = int(name.split("layers.")[1].split(".")[0])
                if layer_idx < 24:
                    assert param.requires_grad is False
                else:
                    assert param.requires_grad is True

        assert frozen_count > 0

    def test_freeze_by_ratio(self):
        """freeze_model_layers freezes by ratio."""
        from soup_cli.utils.freeze import freeze_model_layers

        model, params = self._make_mock_model(32)
        frozen_count = freeze_model_layers(model, freeze_ratio=0.75)

        # 75% of 32 = 24 layers frozen
        for name, param in params:
            if "layers." in name:
                layer_idx = int(name.split("layers.")[1].split(".")[0])
                if layer_idx < 24:
                    assert param.requires_grad is False

        assert frozen_count > 0

    def test_freeze_layers_priority_over_ratio(self):
        """freeze_layers takes priority when both specified."""
        from soup_cli.utils.freeze import freeze_model_layers

        model, params = self._make_mock_model(32)
        freeze_model_layers(model, freeze_layers=10, freeze_ratio=0.75)

        # Should freeze 10 layers, not 24
        for name, param in params:
            if "layers." in name:
                layer_idx = int(name.split("layers.")[1].split(".")[0])
                if layer_idx < 10:
                    assert param.requires_grad is False
                else:
                    assert param.requires_grad is True

    def test_freeze_does_not_freeze_embeddings(self):
        """Embeddings are not frozen (they're not layer params)."""
        from soup_cli.utils.freeze import freeze_model_layers

        model, params = self._make_mock_model(32)
        freeze_model_layers(model, freeze_layers=24)

        # embed_tokens and lm_head should remain trainable
        for name, param in params:
            if "embed_tokens" in name or "lm_head" in name:
                assert param.requires_grad is True

    def test_freeze_more_than_total_layers(self):
        """Freezing more layers than model has freezes all layers."""
        from soup_cli.utils.freeze import freeze_model_layers

        model, params = self._make_mock_model(8)
        freeze_model_layers(model, freeze_layers=100)

        # All 8 layers frozen
        for name, param in params:
            if "layers." in name:
                assert param.requires_grad is False

    def test_freeze_returns_count(self):
        """freeze_model_layers returns number of frozen parameters."""
        from soup_cli.utils.freeze import freeze_model_layers

        model, params = self._make_mock_model(32)
        frozen = freeze_model_layers(model, freeze_layers=16)
        # 16 layers × 3 params each = 48
        assert frozen == 48

    def test_no_freeze_when_none(self):
        """No freezing when both are None."""
        from soup_cli.utils.freeze import freeze_model_layers

        model, params = self._make_mock_model(8)
        frozen = freeze_model_layers(model, freeze_layers=None, freeze_ratio=None)
        assert frozen == 0

    def test_detect_num_layers(self):
        """_detect_num_layers extracts layer count from model params."""
        from soup_cli.utils.freeze import _detect_num_layers

        model = MagicMock()
        params = [
            (f"model.layers.{idx}.self_attn.weight", MagicMock())
            for idx in range(32)
        ]
        model.named_parameters.return_value = params
        assert _detect_num_layers(model) == 32

    def test_detect_num_layers_no_layers(self):
        """_detect_num_layers returns 0 for models without numbered layers."""
        from soup_cli.utils.freeze import _detect_num_layers

        model = MagicMock()
        model.named_parameters.return_value = [
            ("embed.weight", MagicMock()),
        ]
        assert _detect_num_layers(model) == 0

    def test_detect_num_layers_gpt2_style(self):
        """_detect_num_layers handles GPT-2 style 'transformer.h.N.' naming."""
        from soup_cli.utils.freeze import _detect_num_layers

        model = MagicMock()
        params = [
            (f"transformer.h.{idx}.attn.weight", MagicMock())
            for idx in range(12)
        ]
        model.named_parameters.return_value = params
        assert _detect_num_layers(model) == 12


# ---------------------------------------------------------------------------
# Sweep integration
# ---------------------------------------------------------------------------


class TestFreezeSweep:
    """Tests for freeze fields in sweep param support."""

    def test_freeze_layers_in_sweep(self):
        """freeze_layers is a valid sweep param."""
        from soup_cli.commands.sweep import _parse_sweep_params

        params = _parse_sweep_params(["training.freeze_layers=8,16,24"])
        assert "training.freeze_layers" in params
        assert len(params["training.freeze_layers"]) == 3

    def test_freeze_ratio_in_sweep(self):
        """freeze_ratio is a valid sweep param."""
        from soup_cli.commands.sweep import _parse_sweep_params

        params = _parse_sweep_params(["training.freeze_ratio=0.25,0.5,0.75"])
        assert "training.freeze_ratio" in params
        assert len(params["training.freeze_ratio"]) == 3
