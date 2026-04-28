"""Tests for FP8 recipe support (v0.28.1).

Covers:
- Schema: fp8_recipe field accepts valid literals, rejects invalid strings
- Schema: fp8_recipe requires quantization_aware='fp8' when non-default
- Schema: default recipe is 'tensorwise' (backward compat with v0.28.0)
- Dispatch: apply_fp8_training passes correct recipe to Float8LinearConfig
- Integration: fp8_recipe with non-SFT tasks rejected (via quantization_aware gate)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from soup_cli.config.schema import SoupConfig, TrainingConfig

# ─── Schema: fp8_recipe field ──────────────────────────────────────────────


class TestFP8RecipeSchema:
    """fp8_recipe accepts 'tensorwise', 'rowwise', 'rowwise_with_gw_hp'."""

    def test_fp8_recipe_default_tensorwise(self):
        cfg = SoupConfig(base="test/model", data={"train": "./data.jsonl"})
        assert cfg.training.fp8_recipe == "tensorwise"

    def test_fp8_recipe_tensorwise_explicit(self):
        cfg = SoupConfig(
            base="test/model",
            data={"train": "./data.jsonl"},
            training={"quantization_aware": "fp8", "fp8_recipe": "tensorwise"},
        )
        assert cfg.training.fp8_recipe == "tensorwise"

    def test_fp8_recipe_rowwise(self):
        cfg = SoupConfig(
            base="test/model",
            data={"train": "./data.jsonl"},
            training={"quantization_aware": "fp8", "fp8_recipe": "rowwise"},
        )
        assert cfg.training.fp8_recipe == "rowwise"

    def test_fp8_recipe_rowwise_with_gw_hp(self):
        cfg = SoupConfig(
            base="test/model",
            data={"train": "./data.jsonl"},
            training={
                "quantization_aware": "fp8",
                "fp8_recipe": "rowwise_with_gw_hp",
            },
        )
        assert cfg.training.fp8_recipe == "rowwise_with_gw_hp"

    def test_fp8_recipe_invalid_string_rejected(self):
        """Only the three literal values are accepted."""
        with pytest.raises(ValidationError) as exc:
            SoupConfig(
                base="test/model",
                data={"train": "./data.jsonl"},
                training={
                    "quantization_aware": "fp8",
                    "fp8_recipe": "delayed",
                },
            )
        assert "fp8_recipe" in str(exc.value)

    def test_fp8_recipe_invalid_empty_string_rejected(self):
        with pytest.raises(ValidationError):
            SoupConfig(
                base="test/model",
                data={"train": "./data.jsonl"},
                training={"quantization_aware": "fp8", "fp8_recipe": ""},
            )

    def test_fp8_recipe_invalid_int_rejected(self):
        with pytest.raises(ValidationError):
            SoupConfig(
                base="test/model",
                data={"train": "./data.jsonl"},
                training={"quantization_aware": "fp8", "fp8_recipe": 42},
            )


# ─── Schema: fp8_recipe requires quantization_aware='fp8' ─────────────────


class TestFP8RecipeRequiresFP8:
    """Non-default fp8_recipe without quantization_aware='fp8' is rejected."""

    def test_default_recipe_allowed_without_fp8(self):
        """tensorwise (default) is fine even without fp8 — it's the default."""
        cfg = SoupConfig(
            base="test/model",
            data={"train": "./data.jsonl"},
            training={"fp8_recipe": "tensorwise"},
        )
        assert cfg.training.fp8_recipe == "tensorwise"
        assert cfg.training.quantization_aware is False

    def test_rowwise_without_fp8_rejected(self):
        with pytest.raises(ValidationError) as exc:
            SoupConfig(
                base="test/model",
                data={"train": "./data.jsonl"},
                training={"fp8_recipe": "rowwise"},
            )
        assert "quantization_aware" in str(exc.value)

    def test_rowwise_with_gw_hp_without_fp8_rejected(self):
        with pytest.raises(ValidationError) as exc:
            SoupConfig(
                base="test/model",
                data={"train": "./data.jsonl"},
                training={"fp8_recipe": "rowwise_with_gw_hp"},
            )
        assert "quantization_aware" in str(exc.value)

    def test_rowwise_with_bool_true_qat_rejected(self):
        """Bool True = int8 QAT, not FP8 — recipe should be rejected."""
        with pytest.raises(ValidationError) as exc:
            SoupConfig(
                base="test/model",
                data={"train": "./data.jsonl"},
                training={
                    "quantization_aware": True,
                    "fp8_recipe": "rowwise",
                },
            )
        assert "quantization_aware" in str(exc.value)


# ─── Dispatch: apply_fp8_training recipe parameter ────────────────────────


class TestFP8RecipeDispatch:
    """apply_fp8_training passes recipe to Float8LinearConfig.from_recipe_name."""

    def test_apply_fp8_dispatches_tensorwise(self):
        """Default recipe passes 'tensorwise' to from_recipe_name."""
        mock_config = MagicMock()
        mock_from_recipe = MagicMock(return_value=mock_config)
        mock_convert = MagicMock()

        fake_float8 = MagicMock()
        fake_float8.convert_to_float8_training = mock_convert
        fake_config_mod = MagicMock()
        fake_config_mod.Float8LinearConfig.from_recipe_name = mock_from_recipe

        with patch.dict(
            "sys.modules",
            {
                "torchao": MagicMock(),
                "torchao.float8": fake_float8,
                "torchao.float8.config": fake_config_mod,
            },
        ):
            # Need to reimport to pick up the mocked modules
            import importlib

            import soup_cli.utils.fp8 as fp8_mod

            importlib.reload(fp8_mod)

            # Mock is_fp8_available to return True
            with patch.object(fp8_mod, "is_fp8_available", return_value=True):
                model = MagicMock()
                result = fp8_mod.apply_fp8_training(model, recipe="tensorwise")

        mock_from_recipe.assert_called_once_with("tensorwise")
        mock_convert.assert_called_once_with(model, config=mock_config)
        assert result is True

    def test_apply_fp8_dispatches_rowwise(self):
        """Rowwise recipe passes 'rowwise' to from_recipe_name."""
        mock_config = MagicMock()
        mock_from_recipe = MagicMock(return_value=mock_config)
        mock_convert = MagicMock()

        fake_float8 = MagicMock()
        fake_float8.convert_to_float8_training = mock_convert
        fake_config_mod = MagicMock()
        fake_config_mod.Float8LinearConfig.from_recipe_name = mock_from_recipe

        with patch.dict(
            "sys.modules",
            {
                "torchao": MagicMock(),
                "torchao.float8": fake_float8,
                "torchao.float8.config": fake_config_mod,
            },
        ):
            import importlib

            import soup_cli.utils.fp8 as fp8_mod

            importlib.reload(fp8_mod)

            with patch.object(fp8_mod, "is_fp8_available", return_value=True):
                model = MagicMock()
                result = fp8_mod.apply_fp8_training(model, recipe="rowwise")

        mock_from_recipe.assert_called_once_with("rowwise")
        assert result is True

    def test_apply_fp8_dispatches_rowwise_with_gw_hp(self):
        """rowwise_with_gw_hp recipe passes through correctly."""
        mock_config = MagicMock()
        mock_from_recipe = MagicMock(return_value=mock_config)
        mock_convert = MagicMock()

        fake_float8 = MagicMock()
        fake_float8.convert_to_float8_training = mock_convert
        fake_config_mod = MagicMock()
        fake_config_mod.Float8LinearConfig.from_recipe_name = mock_from_recipe

        with patch.dict(
            "sys.modules",
            {
                "torchao": MagicMock(),
                "torchao.float8": fake_float8,
                "torchao.float8.config": fake_config_mod,
            },
        ):
            import importlib

            import soup_cli.utils.fp8 as fp8_mod

            importlib.reload(fp8_mod)

            with patch.object(fp8_mod, "is_fp8_available", return_value=True):
                model = MagicMock()
                result = fp8_mod.apply_fp8_training(
                    model, recipe="rowwise_with_gw_hp"
                )

        mock_from_recipe.assert_called_once_with("rowwise_with_gw_hp")
        assert result is True

    def test_apply_fp8_returns_false_when_unavailable(self):
        """When FP8 deps are missing, apply_fp8_training returns False."""
        from soup_cli.utils.fp8 import apply_fp8_training

        with patch("soup_cli.utils.fp8.is_fp8_available", return_value=False):
            model = MagicMock()
            assert apply_fp8_training(model, recipe="rowwise") is False

    def test_apply_fp8_default_recipe_is_tensorwise(self):
        """Calling without recipe= uses 'tensorwise' (v0.28.0 compat)."""
        mock_config = MagicMock()
        mock_from_recipe = MagicMock(return_value=mock_config)
        mock_convert = MagicMock()

        fake_float8 = MagicMock()
        fake_float8.convert_to_float8_training = mock_convert
        fake_config_mod = MagicMock()
        fake_config_mod.Float8LinearConfig.from_recipe_name = mock_from_recipe

        with patch.dict(
            "sys.modules",
            {
                "torchao": MagicMock(),
                "torchao.float8": fake_float8,
                "torchao.float8.config": fake_config_mod,
            },
        ):
            import importlib

            import soup_cli.utils.fp8 as fp8_mod

            importlib.reload(fp8_mod)

            with patch.object(fp8_mod, "is_fp8_available", return_value=True):
                model = MagicMock()
                fp8_mod.apply_fp8_training(model)

        # Default should be tensorwise
        mock_from_recipe.assert_called_once_with("tensorwise")


# ─── Integration: fp8_recipe + non-SFT tasks ──────────────────────────────


class TestFP8RecipeNonSFT:
    """FP8 recipe on non-SFT tasks: accepted on transformer backends (v0.35.0)."""

    def test_fp8_recipe_on_dpo_accepted(self):
        """DPO + fp8 + recipe is valid (all transformer trainers wired in v0.35.0)."""
        cfg = SoupConfig(
            base="m",
            task="dpo",
            data={"train": "./d.jsonl", "format": "dpo"},
            training={
                "quantization_aware": "fp8",
                "fp8_recipe": "rowwise",
            },
        )
        assert cfg.training.fp8_recipe == "rowwise"

    def test_fp8_recipe_on_grpo_accepted(self):
        cfg = SoupConfig(
            base="m",
            task="grpo",
            data={"train": "./d.jsonl"},
            training={
                "quantization_aware": "fp8",
                "fp8_recipe": "rowwise_with_gw_hp",
            },
        )
        assert cfg.training.fp8_recipe == "rowwise_with_gw_hp"

    def test_fp8_recipe_on_mlx_rejected(self):
        """MLX backend does not support FP8 — rejected at config level."""
        with pytest.raises(ValidationError):
            SoupConfig(
                base="m",
                task="sft",
                backend="mlx",
                data={"train": "./d.jsonl"},
                training={
                    "quantization_aware": "fp8",
                    "fp8_recipe": "rowwise",
                },
            )

    def test_fp8_recipe_on_sft_accepted(self):
        """SFT + fp8 + recipe is valid."""
        cfg = SoupConfig(
            base="m",
            task="sft",
            data={"train": "./d.jsonl"},
            training={
                "quantization_aware": "fp8",
                "fp8_recipe": "rowwise",
            },
        )
        assert cfg.training.fp8_recipe == "rowwise"
        assert cfg.training.quantization_aware == "fp8"

    def test_all_recipes_accepted_on_sft(self):
        """All three recipes are valid on SFT with fp8."""
        for recipe in ("tensorwise", "rowwise", "rowwise_with_gw_hp"):
            cfg = SoupConfig(
                base="m",
                task="sft",
                data={"train": "./d.jsonl"},
                training={
                    "quantization_aware": "fp8",
                    "fp8_recipe": recipe,
                },
            )
            assert cfg.training.fp8_recipe == recipe


# ─── Backward compatibility ───────────────────────────────────────────────


class TestFP8RecipeBackwardCompat:
    """v0.28.0 configs without fp8_recipe still work (defaults to tensorwise)."""

    def test_v028_config_no_recipe_field(self):
        """Config with quantization_aware='fp8' but no fp8_recipe is valid."""
        cfg = SoupConfig(
            base="test/model",
            data={"train": "./data.jsonl"},
            training={"quantization_aware": "fp8"},
        )
        assert cfg.training.quantization_aware == "fp8"
        assert cfg.training.fp8_recipe == "tensorwise"

    def test_v028_bool_true_unaffected(self):
        """Bool True (int8 QAT) is unaffected by fp8_recipe field."""
        cfg = SoupConfig(
            base="test/model",
            data={"train": "./data.jsonl"},
            training={"quantization_aware": True},
        )
        assert cfg.training.quantization_aware is True
        assert cfg.training.fp8_recipe == "tensorwise"  # default, unused

    def test_training_config_fp8_recipe_default(self):
        """TrainingConfig alone defaults fp8_recipe to tensorwise."""
        tcfg = TrainingConfig()
        assert tcfg.fp8_recipe == "tensorwise"
