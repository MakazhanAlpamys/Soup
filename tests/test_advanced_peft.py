"""Tests for v0.12.0 Advanced PEFT — DoRA, LoRA+, GaLore."""

import pytest
from pydantic import ValidationError

from soup_cli.config.schema import SoupConfig

# ─── DoRA Config Tests ──────────────────────────────────────────────────────


class TestDoRAConfig:
    """Test DoRA (use_dora) config validation."""

    def test_use_dora_default_false(self):
        cfg = SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl"},
        )
        assert cfg.training.lora.use_dora is False

    def test_use_dora_enabled(self):
        cfg = SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl"},
            training={"lora": {"use_dora": True}},
        )
        assert cfg.training.lora.use_dora is True

    def test_dora_with_all_tasks(self):
        """DoRA should be valid for all task types."""
        for task in ["sft", "dpo", "kto", "orpo", "simpo", "ipo", "grpo"]:
            cfg = SoupConfig(
                base="some-model",
                task=task,
                data={"train": "./data.jsonl"},
                training={"lora": {"use_dora": True}},
            )
            assert cfg.training.lora.use_dora is True

    def test_dora_with_custom_lora_params(self):
        cfg = SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl"},
            training={
                "lora": {
                    "r": 128,
                    "alpha": 32,
                    "dropout": 0.1,
                    "use_dora": True,
                }
            },
        )
        assert cfg.training.lora.use_dora is True
        assert cfg.training.lora.r == 128
        assert cfg.training.lora.alpha == 32

    @pytest.mark.parametrize("backend", ["unsloth", "mlx"])
    def test_use_dora_is_refused_off_the_transformers_backend(self, backend: str):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="use_dora|dora"):
            load_config_from_string(
                f"base: org/model\ntask: sft\nbackend: {backend}\n"
                "data: {train: ./x.jsonl}\n"
                "training:\n  quantization: 4bit\n  lora: {r: 8, alpha: 16, use_dora: true}\n"
                "output: ./out\n"
            )

    def test_use_dora_is_accepted_on_transformers_backend(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(
            "base: org/model\ntask: sft\nbackend: transformers\n"
            "data: {train: ./x.jsonl}\n"
            "training:\n  quantization: 4bit\n  lora: {r: 8, alpha: 16, use_dora: true}\n"
            "output: ./out\n"
        )
        assert cfg.training.lora.use_dora is True

    def test_dora_yaml_round_trip(self):
        from soup_cli.config.loader import load_config_from_string

        yaml_str = """
base: some-model
task: sft
data:
  train: ./data.jsonl
training:
  lora:
    r: 64
    alpha: 16
    use_dora: true
"""
        cfg = load_config_from_string(yaml_str)
        assert cfg.training.lora.use_dora is True


# ─── LoRA+ Config Tests ─────────────────────────────────────────────────────


class TestLoraPlusConfig:
    """Test LoRA+ (loraplus_lr_ratio) config validation."""

    def test_loraplus_default_none(self):
        cfg = SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl"},
        )
        assert cfg.training.loraplus_lr_ratio is None

    def test_loraplus_custom_ratio(self):
        cfg = SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl"},
            training={"loraplus_lr_ratio": 16.0},
        )
        assert cfg.training.loraplus_lr_ratio == pytest.approx(16.0)

    def test_loraplus_must_be_positive(self):
        with pytest.raises(ValidationError):
            SoupConfig(
                base="some-model",
                data={"train": "./data.jsonl"},
                training={"loraplus_lr_ratio": 0},
            )

    def test_loraplus_negative_rejected(self):
        with pytest.raises(ValidationError):
            SoupConfig(
                base="some-model",
                data={"train": "./data.jsonl"},
                training={"loraplus_lr_ratio": -1.0},
            )

    def test_loraplus_yaml_round_trip(self):
        from soup_cli.config.loader import load_config_from_string

        yaml_str = """
base: some-model
task: sft
data:
  train: ./data.jsonl
training:
  loraplus_lr_ratio: 16.0
"""
        cfg = load_config_from_string(yaml_str)
        assert cfg.training.loraplus_lr_ratio == pytest.approx(16.0)

    def test_loraplus_with_dora_combined(self):
        """LoRA+ and DoRA can be used together."""
        cfg = SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl"},
            training={
                "loraplus_lr_ratio": 8.0,
                "lora": {"use_dora": True},
            },
        )
        assert cfg.training.loraplus_lr_ratio == pytest.approx(8.0)
        assert cfg.training.lora.use_dora is True


# ─── GaLore Config Tests ────────────────────────────────────────────────────


class TestGaLoreConfig:
    """Test GaLore config validation."""

    def test_galore_default_disabled(self):
        cfg = SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl"},
        )
        assert cfg.training.use_galore is False

    def test_galore_enabled(self):
        cfg = SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl"},
            training={"use_galore": True, "quantization": "none"},
        )
        assert cfg.training.use_galore is True

    def test_galore_rank_default(self):
        cfg = SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl"},
        )
        assert cfg.training.galore_rank == 128

    def test_galore_rank_custom(self):
        cfg = SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl"},
            training={"galore_rank": 256},
        )
        assert cfg.training.galore_rank == 256

    def test_galore_rank_must_be_positive(self):
        with pytest.raises(ValidationError):
            SoupConfig(
                base="some-model",
                data={"train": "./data.jsonl"},
                training={"galore_rank": 0},
            )

    def test_galore_update_proj_gap_default(self):
        cfg = SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl"},
        )
        assert cfg.training.galore_update_proj_gap == 200

    def test_galore_scale_default(self):
        cfg = SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl"},
        )
        assert cfg.training.galore_scale == pytest.approx(0.25)

    def test_galore_scale_must_be_positive(self):
        with pytest.raises(ValidationError):
            SoupConfig(
                base="some-model",
                data={"train": "./data.jsonl"},
                training={"galore_scale": 0},
            )

    def test_galore_yaml_round_trip(self):
        from soup_cli.config.loader import load_config_from_string

        yaml_str = """
base: some-model
task: sft
data:
  train: ./data.jsonl
training:
  use_galore: true
  galore_rank: 64
  galore_update_proj_gap: 100
  galore_scale: 0.5
  quantization: none
"""
        cfg = load_config_from_string(yaml_str)
        assert cfg.training.use_galore is True
        assert cfg.training.galore_rank == 64
        assert cfg.training.galore_update_proj_gap == 100
        assert cfg.training.galore_scale == pytest.approx(0.5)


# ─── GaLore Validation Tests ────────────────────────────────────────────────


class TestGaLoreValidation:
    """Test GaLore config validation helper."""

    def test_galore_incompatible_with_4bit(self):
        from soup_cli.utils.galore import validate_galore_config

        errors = validate_galore_config(
            use_galore=True, quantization="4bit", backend="transformers",
        )
        assert len(errors) == 1
        assert "quantization" in errors[0].lower()

    def test_galore_incompatible_with_8bit(self):
        from soup_cli.utils.galore import validate_galore_config

        errors = validate_galore_config(
            use_galore=True, quantization="8bit", backend="transformers",
        )
        assert len(errors) == 1

    def test_galore_incompatible_with_unsloth(self):
        from soup_cli.utils.galore import validate_galore_config

        errors = validate_galore_config(
            use_galore=True, quantization="none", backend="unsloth",
        )
        assert len(errors) == 1
        assert "unsloth" in errors[0].lower()

    def test_galore_valid_config_no_errors(self):
        from soup_cli.utils.galore import validate_galore_config

        errors = validate_galore_config(
            use_galore=True, quantization="none", backend="transformers",
        )
        assert len(errors) == 0

    def test_galore_disabled_no_errors(self):
        from soup_cli.utils.galore import validate_galore_config

        errors = validate_galore_config(
            use_galore=False, quantization="4bit", backend="unsloth",
        )
        assert len(errors) == 0

    def test_galore_multiple_errors(self):
        """Both quantization and unsloth should be flagged."""
        from soup_cli.utils.galore import validate_galore_config

        errors = validate_galore_config(
            use_galore=True, quantization="4bit", backend="unsloth",
        )
        assert len(errors) == 2


# ─── GaLore Optimizer Helper Tests ──────────────────────────────────────────


class TestGaLoreOptimizerHelper:
    """Test the get_galore_optimizer_and_params helper."""

    def test_returns_galore_optimizer_name(self):
        from soup_cli.utils.galore import get_galore_optimizer_and_params

        result = get_galore_optimizer_and_params(
        )
        assert result["optim"] == "galore_adamw"

    def test_returns_target_modules(self):
        from soup_cli.utils.galore import get_galore_optimizer_and_params

        result = get_galore_optimizer_and_params(
        )
        assert "optim_target_modules" in result
        assert isinstance(result["optim_target_modules"], list)

    def test_returns_optim_args_with_rank(self):
        from soup_cli.utils.galore import get_galore_optimizer_and_params

        result = get_galore_optimizer_and_params(galore_rank=64)
        assert "rank=64" in result["optim_args"]

    def test_returns_optim_args_with_update_gap(self):
        from soup_cli.utils.galore import get_galore_optimizer_and_params

        result = get_galore_optimizer_and_params(galore_update_proj_gap=100)
        assert "update_proj_gap=100" in result["optim_args"]

    def test_returns_optim_args_with_scale(self):
        from soup_cli.utils.galore import get_galore_optimizer_and_params

        result = get_galore_optimizer_and_params(galore_scale=0.5)
        assert "scale=0.5" in result["optim_args"]

    def test_invalid_rank_raises_value_error(self):
        from soup_cli.utils.galore import get_galore_optimizer_and_params

        with pytest.raises(ValueError, match="Invalid GaLore"):
            get_galore_optimizer_and_params(galore_rank=0)

    def test_invalid_scale_raises_value_error(self):
        from soup_cli.utils.galore import get_galore_optimizer_and_params

        with pytest.raises(ValueError, match="Invalid GaLore"):
            get_galore_optimizer_and_params(galore_scale=0)


# ─── Sweep Shortcut Tests ───────────────────────────────────────────────────


class TestAdvancedPEFTSweepParams:
    """Test sweep shortcuts for DoRA, LoRA+, GaLore."""

    def test_use_dora_shortcut(self):
        from soup_cli.commands.sweep import _set_nested_param

        config = {}
        _set_nested_param(config, "use_dora", True)
        assert config["training"]["lora"]["use_dora"] is True

    def test_loraplus_lr_ratio_shortcut(self):
        from soup_cli.commands.sweep import _set_nested_param

        config = {}
        _set_nested_param(config, "loraplus_lr_ratio", 16.0)
        assert config["training"]["loraplus_lr_ratio"] == pytest.approx(16.0)

    def test_use_galore_shortcut(self):
        from soup_cli.commands.sweep import _set_nested_param

        config = {}
        _set_nested_param(config, "use_galore", True)
        assert config["training"]["use_galore"] is True

    def test_galore_rank_shortcut(self):
        from soup_cli.commands.sweep import _set_nested_param

        config = {}
        _set_nested_param(config, "galore_rank", 64)
        assert config["training"]["galore_rank"] == 64


# ─── Experiment Name Validation Tests ────────────────────────────────────────


class TestExperimentNameValidation:
    """Test experiment_name path traversal protection."""

    def test_normal_name_accepted(self):
        cfg = SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl"},
            experiment_name="my_experiment_1",
        )
        assert cfg.experiment_name == "my_experiment_1"

    def test_none_name_accepted(self):
        cfg = SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl"},
            experiment_name=None,
        )
        assert cfg.experiment_name is None

    def test_forward_slash_rejected(self):
        with pytest.raises(ValidationError):
            SoupConfig(
                base="some-model",
                data={"train": "./data.jsonl"},
                experiment_name="../../etc/passwd",
            )

    def test_backslash_rejected(self):
        with pytest.raises(ValidationError):
            SoupConfig(
                base="some-model",
                data={"train": "./data.jsonl"},
                experiment_name="..\\..\\secret",
            )

    def test_colon_rejected(self):
        with pytest.raises(ValidationError):
            SoupConfig(
                base="some-model",
                data={"train": "./data.jsonl"},
                experiment_name="C:\\evil",
            )

    def test_null_byte_rejected(self):
        with pytest.raises(ValidationError):
            SoupConfig(
                base="some-model",
                data={"train": "./data.jsonl"},
                experiment_name="name\x00evil",
            )
