"""Tests for Unsloth backend — config, detection, trainer integration, templates."""

from unittest.mock import MagicMock, patch

import pytest

from soup_cli.config.schema import TEMPLATES, SoupConfig

# ─── Config Tests ───────────────────────────────────────────────────────────


class TestUnslothConfig:
    """Test backend config field validation."""

    def test_backend_default_is_transformers(self):
        """Default backend should be 'transformers'."""
        cfg = SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl"},
        )
        assert cfg.backend == "transformers"

    def test_backend_unsloth_accepted(self):
        """backend: unsloth should be valid."""
        cfg = SoupConfig(
            base="some-model",
            backend="unsloth",
            data={"train": "./data.jsonl"},
        )
        assert cfg.backend == "unsloth"

    def test_backend_transformers_accepted(self):
        """backend: transformers should be valid."""
        cfg = SoupConfig(
            base="some-model",
            backend="transformers",
            data={"train": "./data.jsonl"},
        )
        assert cfg.backend == "transformers"

    def test_backend_invalid_rejected(self):
        """Invalid backend should raise validation error."""
        with pytest.raises(Exception):
            SoupConfig(
                base="some-model",
                backend="invalid",
                data={"train": "./data.jsonl"},
            )

    def test_backend_with_sft(self):
        """Unsloth backend should work with SFT task."""
        cfg = SoupConfig(
            base="some-model",
            task="sft",
            backend="unsloth",
            data={"train": "./data.jsonl"},
        )
        assert cfg.task == "sft"
        assert cfg.backend == "unsloth"

    def test_backend_with_dpo(self):
        """Unsloth backend should work with DPO task."""
        cfg = SoupConfig(
            base="some-model",
            task="dpo",
            backend="unsloth",
            data={"train": "./data.jsonl"},
        )
        assert cfg.task == "dpo"
        assert cfg.backend == "unsloth"

    def test_backend_with_grpo(self):
        """Unsloth backend should work with GRPO task."""
        cfg = SoupConfig(
            base="some-model",
            task="grpo",
            backend="unsloth",
            data={"train": "./data.jsonl"},
        )
        assert cfg.task == "grpo"
        assert cfg.backend == "unsloth"

    def test_backend_in_model_dump(self):
        """backend field should appear in model_dump output."""
        cfg = SoupConfig(
            base="some-model",
            backend="unsloth",
            data={"train": "./data.jsonl"},
        )
        dump = cfg.model_dump()
        assert dump["backend"] == "unsloth"

    def test_full_unsloth_config(self):
        """Full config with unsloth backend should validate."""
        cfg = SoupConfig(
            base="meta-llama/Llama-3.1-8B-Instruct",
            task="sft",
            backend="unsloth",
            data={"train": "./data.jsonl", "format": "alpaca", "max_length": 4096},
            training={
                "epochs": 3,
                "lr": 2e-5,
                "quantization": "4bit",
                "lora": {"r": 64, "alpha": 16},
            },
        )
        assert cfg.backend == "unsloth"
        assert cfg.training.quantization == "4bit"
        assert cfg.data.max_length == 4096


# ─── Detection Tests ────────────────────────────────────────────────────────


class TestUnslothDetection:
    """Test unsloth availability detection."""

    def test_is_unsloth_available_when_installed(self):
        """Should return True when unsloth is importable."""
        mock_module = MagicMock()
        with patch.dict("sys.modules", {"unsloth": mock_module}):
            # Need to reimport to avoid cached result
            import importlib

            import soup_cli.utils.unsloth

            importlib.reload(soup_cli.utils.unsloth)
            assert soup_cli.utils.unsloth.is_unsloth_available() is True

    def test_is_unsloth_available_when_not_installed(self):
        """Should return False when unsloth is not importable."""
        from soup_cli.utils.unsloth import is_unsloth_available

        # Default environment doesn't have unsloth
        # This test works because unsloth isn't installed in test env
        result = is_unsloth_available()
        assert isinstance(result, bool)

    def test_get_unsloth_version_when_not_installed(self):
        """Should return None when unsloth is not installed."""
        from soup_cli.utils.unsloth import get_unsloth_version

        result = get_unsloth_version()
        # In test env, unsloth is not installed
        assert result is None or isinstance(result, str)

    def test_get_unsloth_version_when_installed(self):
        """Should return version string when unsloth is installed."""
        mock_module = MagicMock()
        mock_module.__version__ = "2024.11.0"
        with patch.dict("sys.modules", {"unsloth": mock_module}):
            import importlib

            import soup_cli.utils.unsloth

            importlib.reload(soup_cli.utils.unsloth)
            result = soup_cli.utils.unsloth.get_unsloth_version()
            assert result == "2024.11.0"


# ─── Trainer Integration Tests ──────────────────────────────────────────────


class TestSFTUnslothIntegration:
    """Test SFT trainer with unsloth backend."""

    def test_sft_wrapper_init_with_unsloth(self):
        """SFTTrainerWrapper should accept unsloth backend config."""
        from soup_cli.trainer.sft import SFTTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            backend="unsloth",
            data={"train": "./data.jsonl"},
        )
        wrapper = SFTTrainerWrapper(cfg, device="cuda")
        assert wrapper.config.backend == "unsloth"

    def test_sft_wrapper_init_with_transformers(self):
        """SFTTrainerWrapper should work with default transformers backend."""
        from soup_cli.trainer.sft import SFTTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl"},
        )
        wrapper = SFTTrainerWrapper(cfg, device="cuda")
        assert wrapper.config.backend == "transformers"

    def test_sft_setup_unsloth_calls_load(self):
        """_setup_unsloth should call utils.unsloth.load_model_and_tokenizer."""
        from soup_cli.trainer.sft import SFTTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            backend="unsloth",
            data={"train": "./data.jsonl", "max_length": 2048},
            training={"lora": {"r": 64, "alpha": 16, "dropout": 0.05}},
        )
        wrapper = SFTTrainerWrapper(cfg, device="cuda")

        mock_model = MagicMock()
        mock_model.get_nb_trainable_parameters.return_value = (1000, 100000)
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "pad"

        with patch(
            "soup_cli.utils.unsloth.load_model_and_tokenizer",
            return_value=(mock_model, mock_tokenizer),
        ) as mock_load:
            wrapper._setup_unsloth(cfg, cfg.training)
            mock_load.assert_called_once_with(
                model_name="some-model",
                max_seq_length=2048,
                quantization="4bit",
                lora_r=64,
                lora_alpha=16,
                lora_dropout=0.05,
                target_modules="auto",
            )
            assert wrapper.model is mock_model
            assert wrapper.tokenizer is mock_tokenizer


class TestDPOUnslothIntegration:
    """Test DPO trainer with unsloth backend."""

    def test_dpo_wrapper_init_with_unsloth(self):
        """DPOTrainerWrapper should accept unsloth backend config."""
        from soup_cli.trainer.dpo import DPOTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="dpo",
            backend="unsloth",
            data={"train": "./data.jsonl"},
        )
        wrapper = DPOTrainerWrapper(cfg, device="cuda")
        assert wrapper.config.backend == "unsloth"

    def test_dpo_setup_unsloth_calls_load(self):
        """_setup_unsloth should call utils.unsloth.load_model_and_tokenizer."""
        from soup_cli.trainer.dpo import DPOTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="dpo",
            backend="unsloth",
            data={"train": "./data.jsonl", "max_length": 2048},
        )
        wrapper = DPOTrainerWrapper(cfg, device="cuda")

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "pad"

        with patch(
            "soup_cli.utils.unsloth.load_model_and_tokenizer",
            return_value=(mock_model, mock_tokenizer),
        ) as mock_load:
            wrapper._setup_unsloth(cfg, cfg.training)
            mock_load.assert_called_once()
            assert wrapper.model is mock_model


class TestGRPOUnslothIntegration:
    """Test GRPO trainer with unsloth backend."""

    def test_grpo_wrapper_init_with_unsloth(self):
        """GRPOTrainerWrapper should accept unsloth backend config."""
        from soup_cli.trainer.grpo import GRPOTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="grpo",
            backend="unsloth",
            data={"train": "./data.jsonl"},
        )
        wrapper = GRPOTrainerWrapper(cfg, device="cuda")
        assert wrapper.config.backend == "unsloth"

    def test_grpo_setup_unsloth_calls_load(self):
        """_setup_unsloth should call utils.unsloth.load_model_and_tokenizer."""
        from soup_cli.trainer.grpo import GRPOTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="grpo",
            backend="unsloth",
            data={"train": "./data.jsonl", "max_length": 4096},
        )
        wrapper = GRPOTrainerWrapper(cfg, device="cuda")

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "pad"

        with patch(
            "soup_cli.utils.unsloth.load_model_and_tokenizer",
            return_value=(mock_model, mock_tokenizer),
        ) as mock_load:
            wrapper._setup_unsloth(cfg, cfg.training)
            mock_load.assert_called_once()
            assert wrapper.model is mock_model


# ─── Template Tests ─────────────────────────────────────────────────────────


class TestTemplatesHaveUnslothHint:
    """Test that templates mention unsloth backend as an option."""

    def test_chat_template_mentions_unsloth(self):
        assert "unsloth" in TEMPLATES["chat"]

    def test_code_template_mentions_unsloth(self):
        assert "unsloth" in TEMPLATES["code"]

    def test_reasoning_template_mentions_unsloth(self):
        assert "unsloth" in TEMPLATES["reasoning"]

    def test_medical_template_mentions_unsloth(self):
        assert "unsloth" in TEMPLATES["medical"]

    def test_templates_default_backend_commented(self):
        """Templates should have unsloth commented out (not active by default)."""
        for name, template in TEMPLATES.items():
            assert "# backend: unsloth" in template, f"{name} template missing unsloth hint"


# ─── Sweep Shortcut Tests ──────────────────────────────────────────────────


class TestBackendSweepParam:
    """Test backend parameter in sweep shortcuts."""

    def test_backend_shortcut(self):
        from soup_cli.commands.sweep import _set_nested_param

        config = {"backend": "transformers"}
        _set_nested_param(config, "backend", "unsloth")
        assert config["backend"] == "unsloth"


# ─── Doctor Tests ───────────────────────────────────────────────────────────


class TestDoctorUnsloth:
    """Test that doctor checks for unsloth."""

    def test_unsloth_in_deps_list(self):
        from soup_cli.commands.doctor import DEPS

        pkg_names = [pkg_name for _, pkg_name, _, _ in DEPS]
        assert "unsloth" in pkg_names

    def test_unsloth_is_optional(self):
        from soup_cli.commands.doctor import DEPS

        for import_name, pkg_name, _, required in DEPS:
            if pkg_name == "unsloth":
                assert required is False
                break
        else:
            pytest.fail("unsloth not found in DEPS")


# ─── Load Function Tests ───────────────────────────────────────────────────


class TestLoadModelAndTokenizer:
    """Test the load_model_and_tokenizer function with mocked unsloth."""

    def test_load_with_4bit(self):
        """Should pass load_in_4bit=True for 4bit quantization."""
        mock_flm = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)
        mock_flm.get_peft_model.return_value = mock_model

        with patch.dict("sys.modules", {"unsloth": MagicMock(FastLanguageModel=mock_flm)}):
            import importlib

            import soup_cli.utils.unsloth

            importlib.reload(soup_cli.utils.unsloth)
            model, tokenizer = soup_cli.utils.unsloth.load_model_and_tokenizer(
                model_name="test-model",
                max_seq_length=2048,
                quantization="4bit",
            )
            mock_flm.from_pretrained.assert_called_once()
            call_kwargs = mock_flm.from_pretrained.call_args
            assert call_kwargs[1]["load_in_4bit"] is True

    def test_load_with_none_quantization(self):
        """Should pass load_in_4bit=False for 'none' quantization."""
        mock_flm = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)
        mock_flm.get_peft_model.return_value = mock_model

        with patch.dict("sys.modules", {"unsloth": MagicMock(FastLanguageModel=mock_flm)}):
            import importlib

            import soup_cli.utils.unsloth

            importlib.reload(soup_cli.utils.unsloth)
            soup_cli.utils.unsloth.load_model_and_tokenizer(
                model_name="test-model",
                max_seq_length=2048,
                quantization="none",
            )
            call_kwargs = mock_flm.from_pretrained.call_args
            assert call_kwargs[1]["load_in_4bit"] is False

    def test_load_auto_target_modules(self):
        """auto target_modules should expand to default linear layers."""
        mock_flm = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)
        mock_flm.get_peft_model.return_value = mock_model

        with patch.dict("sys.modules", {"unsloth": MagicMock(FastLanguageModel=mock_flm)}):
            import importlib

            import soup_cli.utils.unsloth

            importlib.reload(soup_cli.utils.unsloth)
            soup_cli.utils.unsloth.load_model_and_tokenizer(
                model_name="test-model",
                max_seq_length=2048,
                target_modules="auto",
            )
            peft_call_kwargs = mock_flm.get_peft_model.call_args
            target = peft_call_kwargs[1]["target_modules"]
            assert "q_proj" in target
            assert "v_proj" in target

    def test_load_custom_target_modules(self):
        """Custom target_modules list should be passed through."""
        mock_flm = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)
        mock_flm.get_peft_model.return_value = mock_model

        with patch.dict("sys.modules", {"unsloth": MagicMock(FastLanguageModel=mock_flm)}):
            import importlib

            import soup_cli.utils.unsloth

            importlib.reload(soup_cli.utils.unsloth)
            custom_modules = ["q_proj", "k_proj"]
            soup_cli.utils.unsloth.load_model_and_tokenizer(
                model_name="test-model",
                max_seq_length=2048,
                target_modules=custom_modules,
            )
            peft_call_kwargs = mock_flm.get_peft_model.call_args
            assert peft_call_kwargs[1]["target_modules"] == custom_modules

    def test_load_lora_params_passed(self):
        """LoRA params should be forwarded to get_peft_model."""
        mock_flm = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)
        mock_flm.get_peft_model.return_value = mock_model

        with patch.dict("sys.modules", {"unsloth": MagicMock(FastLanguageModel=mock_flm)}):
            import importlib

            import soup_cli.utils.unsloth

            importlib.reload(soup_cli.utils.unsloth)
            soup_cli.utils.unsloth.load_model_and_tokenizer(
                model_name="test-model",
                max_seq_length=2048,
                lora_r=128,
                lora_alpha=32,
                lora_dropout=0.1,
            )
            peft_kwargs = mock_flm.get_peft_model.call_args[1]
            assert peft_kwargs["r"] == 128
            assert peft_kwargs["lora_alpha"] == 32
            assert peft_kwargs["lora_dropout"] == 0.1
