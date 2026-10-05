"""Tests for ORPO training — config, data format, template, routing, sweep."""

from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

import pytest
from pydantic import ValidationError

from soup_cli.config.schema import TEMPLATES, SoupConfig

# ─── Config Tests ───────────────────────────────────────────────────────────


class TestORPOConfig:
    """Test ORPO task config validation."""

    def test_orpo_task_accepted(self):
        """ORPO task should be a valid task type."""
        cfg = SoupConfig(
            base="some-model",
            task="orpo",
            data={"train": "./data.jsonl"},
        )
        assert cfg.task == "orpo"

    def test_orpo_beta_default(self):
        """orpo_beta should default to 0.1."""
        cfg = SoupConfig(
            base="some-model",
            task="orpo",
            data={"train": "./data.jsonl"},
        )
        assert cfg.training.orpo_beta == 0.1

    def test_orpo_beta_custom(self):
        """Custom orpo_beta should be accepted."""
        cfg = SoupConfig(
            base="some-model",
            task="orpo",
            data={"train": "./data.jsonl"},
            training={"orpo_beta": 0.05},
        )
        assert cfg.training.orpo_beta == pytest.approx(0.05)

    def test_orpo_beta_must_be_positive(self):
        """orpo_beta must be > 0."""
        with pytest.raises(ValidationError):
            SoupConfig(
                base="some-model",
                task="orpo",
                data={"train": "./data.jsonl"},
                training={"orpo_beta": 0},
            )

    def test_orpo_full_config(self):
        """Full ORPO config should validate correctly."""
        cfg = SoupConfig(
            base="meta-llama/Llama-3.1-8B-Instruct",
            task="orpo",
            data={"train": "./data.jsonl", "format": "dpo", "max_length": 2048},
            training={
                "epochs": 3,
                "lr": 1e-5,
                "orpo_beta": 0.2,
                "lora": {"r": 64, "alpha": 16},
                "quantization": "4bit",
            },
        )
        assert cfg.task == "orpo"
        assert cfg.training.orpo_beta == pytest.approx(0.2)
        assert cfg.data.max_length == 2048

    def test_orpo_uses_dpo_data_format(self):
        """ORPO should work with the DPO data format."""
        cfg = SoupConfig(
            base="some-model",
            task="orpo",
            data={"train": "./data.jsonl", "format": "dpo"},
        )
        assert cfg.data.format == "dpo"


# ─── Data Format Tests ─────────────────────────────────────────────────────


class TestORPODataFormat:
    """Test that ORPO uses DPO data format correctly."""

    def test_dpo_format_works_for_orpo(self):
        """DPO format signature should detect preference data."""
        from soup_cli.data.formats import detect_format

        data = [{"prompt": "Q", "chosen": "A", "rejected": "B"}]
        assert detect_format(data) == "dpo"

    def test_convert_dpo_row_for_orpo(self):
        """format_to_messages should convert DPO rows correctly."""
        from soup_cli.data.formats import format_to_messages

        row = {"prompt": "What is 2+2?", "chosen": "4", "rejected": "Fish"}
        result = format_to_messages(row, "dpo")
        assert result["prompt"] == "What is 2+2?"
        assert result["chosen"] == "4"
        assert result["rejected"] == "Fish"


# ─── Template Tests ──────────────────────────────────────────────────────────


class TestORPOTemplate:
    """Test the ORPO template."""

    def test_orpo_template_exists(self):
        assert "orpo" in TEMPLATES

    def test_orpo_template_valid_yaml(self):
        import yaml

        config = yaml.safe_load(TEMPLATES["orpo"])
        assert config["task"] == "orpo"
        assert config["training"]["orpo_beta"] == 0.1
        assert config["data"]["format"] == "dpo"

    def test_orpo_template_valid_config(self):
        import yaml

        raw = yaml.safe_load(TEMPLATES["orpo"])
        cfg = SoupConfig(**raw)
        assert cfg.task == "orpo"
        assert cfg.training.orpo_beta == 0.1


# ─── Train Command Routing Tests ─────────────────────────────────────────────


class TestORPOTrainRouting:
    """Test that train command routes to ORPO trainer."""

    def test_orpo_import_exists(self):
        """ORPOTrainerWrapper should be importable."""
        from soup_cli.trainer.orpo import ORPOTrainerWrapper

        assert ORPOTrainerWrapper is not None

    def test_orpo_wrapper_init(self):
        """ORPOTrainerWrapper should initialize without error."""
        from soup_cli.trainer.orpo import ORPOTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="orpo",
            data={"train": "./data.jsonl"},
        )
        wrapper = ORPOTrainerWrapper(cfg, device="cpu")
        assert wrapper.config.task == "orpo"
        assert wrapper.device == "cpu"
        assert wrapper.model is None
        assert wrapper.trainer is None

    def test_orpo_wrapper_init_with_options(self):
        """ORPOTrainerWrapper should accept all constructor options."""
        from soup_cli.trainer.orpo import ORPOTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="orpo",
            data={"train": "./data.jsonl"},
        )
        wrapper = ORPOTrainerWrapper(
            cfg,
            device="cuda",
            report_to="wandb",
            deepspeed_config="ds.json",
        )
        assert wrapper.report_to == "wandb"
        assert wrapper.deepspeed_config == "ds.json"


# ─── Sweep Shortcut Tests ────────────────────────────────────────────────────


class TestORPOSweepParams:
    """Test ORPO parameter shortcuts in sweep."""

    def test_orpo_beta_shortcut(self):
        from soup_cli.commands.sweep import _set_nested_param

        config = {"training": {"orpo_beta": 0.1}}
        _set_nested_param(config, "orpo_beta", 0.05)
        assert config["training"]["orpo_beta"] == 0.05

    def test_orpo_beta_shortcut_creates_nested_key(self):
        from soup_cli.commands.sweep import _set_nested_param

        config = {}
        _set_nested_param(config, "orpo_beta", 0.2)
        assert config["training"]["orpo_beta"] == pytest.approx(0.2)

    def test_sweep_run_single_routes_to_orpo_trainer(self):
        """_run_single should instantiate ORPOTrainerWrapper for orpo task."""
        from soup_cli.commands.sweep import _run_single

        cfg = SoupConfig(
            base="some-model",
            task="orpo",
            data={"train": "./data.jsonl"},
        )

        fake_dataset = {
            "train": [
                {"prompt": "Q?", "chosen": "A", "rejected": "B"},
            ]
        }
        fake_result = {
            "initial_loss": 1.0,
            "final_loss": 0.5,
            "total_steps": 10,
            "duration_secs": 60.0,
            "output_dir": "./output",
            "duration": "1m",
        }

        fake_gpu_info = {"memory_total": "0 MB", "memory_total_bytes": 0}
        with (
            mock_patch("soup_cli.data.loader.load_dataset", return_value=fake_dataset),
            mock_patch("soup_cli.utils.gpu.detect_device", return_value=("cpu", "CPU")),
            mock_patch("soup_cli.utils.gpu.get_gpu_info", return_value=fake_gpu_info),
            mock_patch("soup_cli.experiment.tracker.ExperimentTracker") as mock_tracker_cls,
            mock_patch("soup_cli.monitoring.display.TrainingDisplay"),
            mock_patch("soup_cli.trainer.orpo.ORPOTrainerWrapper.setup"),
            mock_patch(
                "soup_cli.trainer.orpo.ORPOTrainerWrapper.train", return_value=fake_result
            ) as mock_train,
        ):
            mock_tracker = MagicMock()
            mock_tracker.start_run.return_value = "run-orpo-1"
            mock_tracker_cls.return_value = mock_tracker

            result = _run_single(cfg, {}, "orpo_run_1", None)

        mock_train.assert_called_once()
        assert result["run_id"] == "run-orpo-1"


# ─── Config Validation Edge Cases ────────────────────────────────────────────


class TestORPOConfigEdgeCases:
    """Additional config validation edge cases for ORPO."""

    def test_orpo_beta_negative_rejected(self):
        with pytest.raises(ValidationError):
            SoupConfig(
                base="some-model",
                task="orpo",
                data={"train": "./data.jsonl"},
                training={"orpo_beta": -0.1},
            )

    def test_orpo_beta_very_large_accepted(self):
        cfg = SoupConfig(
            base="some-model",
            task="orpo",
            data={"train": "./data.jsonl"},
            training={"orpo_beta": 10.0},
        )
        assert cfg.training.orpo_beta == pytest.approx(10.0)

    def test_orpo_config_unsloth_backend(self):
        cfg = SoupConfig(
            base="some-model",
            task="orpo",
            backend="unsloth",
            data={"train": "./data.jsonl"},
        )
        assert cfg.backend == "unsloth"
        assert cfg.task == "orpo"

    def test_orpo_tokenizer_none_before_setup(self):
        from soup_cli.trainer.orpo import ORPOTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="orpo",
            data={"train": "./data.jsonl"},
        )
        wrapper = ORPOTrainerWrapper(cfg)
        assert wrapper.tokenizer is None

    def test_orpo_output_dir_none_before_setup(self):
        from soup_cli.trainer.orpo import ORPOTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="orpo",
            data={"train": "./data.jsonl"},
        )
        wrapper = ORPOTrainerWrapper(cfg)
        assert wrapper._output_dir is None


# ─── Train Method Guard Test ──────────────────────────────────────────────────


class TestORPOTrainGuard:
    """Test the RuntimeError guard when train() is called before setup()."""

    def test_train_before_setup_raises_runtime_error(self):
        from soup_cli.trainer.orpo import ORPOTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="orpo",
            data={"train": "./data.jsonl"},
        )
        wrapper = ORPOTrainerWrapper(cfg)
        with pytest.raises(RuntimeError, match="setup\\(dataset\\) first"):
            wrapper.train()


# ─── Train Method Result Structure ───────────────────────────────────────────


class TestORPOTrainResults:
    """Test the result dict returned by train() using a mocked trainer."""

    def _make_wrapper_with_mock_trainer(self, log_history=None, global_step=20):
        from soup_cli.trainer.orpo import ORPOTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="orpo",
            data={"train": "./data.jsonl"},
            output="./output",
        )
        wrapper = ORPOTrainerWrapper(cfg, device="cpu")
        mock_trainer = MagicMock()
        mock_trainer.train = MagicMock()
        mock_trainer.state.log_history = log_history if log_history is not None else []
        mock_trainer.state.global_step = global_step
        mock_trainer.save_model = MagicMock()
        wrapper.trainer = mock_trainer
        wrapper.tokenizer = MagicMock()
        wrapper._output_dir = "./output"
        return wrapper, mock_trainer

    def test_train_returns_expected_keys(self):
        wrapper, _ = self._make_wrapper_with_mock_trainer(
            log_history=[{"loss": 1.5}, {"loss": 0.8}], global_step=10
        )
        result = wrapper.train()
        assert "initial_loss" in result
        assert "final_loss" in result
        assert "duration" in result
        assert "duration_secs" in result
        assert "output_dir" in result
        assert "total_steps" in result

    def test_train_result_losses_from_log_history(self):
        wrapper, _ = self._make_wrapper_with_mock_trainer(
            log_history=[{"loss": 2.0}, {"loss": 1.0}, {"loss": 0.5}], global_step=30
        )
        result = wrapper.train()
        assert result["initial_loss"] == pytest.approx(2.0)
        assert result["final_loss"] == pytest.approx(0.5)

    def test_train_result_empty_log_history(self):
        wrapper, _ = self._make_wrapper_with_mock_trainer(log_history=[], global_step=0)
        result = wrapper.train()
        assert result["initial_loss"] == 0
        assert result["final_loss"] == 0

    def test_train_calls_save_model(self):
        wrapper, mock_trainer = self._make_wrapper_with_mock_trainer(
            log_history=[{"loss": 1.0}], global_step=5
        )
        wrapper.train()
        mock_trainer.save_model.assert_called_once_with("./output")

    def test_train_passes_resume_checkpoint(self):
        wrapper, mock_trainer = self._make_wrapper_with_mock_trainer(
            log_history=[{"loss": 1.0}], global_step=5
        )
        wrapper.train(resume_from_checkpoint="/ckpt/checkpoint-50")
        mock_trainer.train.assert_called_once_with(resume_from_checkpoint="/ckpt/checkpoint-50")


# ─── CLI Init ORPO Template Tests ─────────────────────────────────────────────


class TestORPOInitTemplate:
    """Test that soup init produces correct output for ORPO."""

    def test_init_orpo_template_creates_file(self, tmp_path):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        output = tmp_path / "soup.yaml"
        result = runner.invoke(app, ["init", "--template", "orpo", "--output", str(output)])
        assert result.exit_code == 0
        assert output.exists()
        content = output.read_text()
        assert "task: orpo" in content
        assert "orpo_beta" in content
        assert "format: dpo" in content

    def test_init_orpo_template_produces_valid_config(self, tmp_path):
        from pathlib import Path

        from typer.testing import CliRunner

        from soup_cli.cli import app
        from soup_cli.config.loader import load_config

        runner = CliRunner()
        output = tmp_path / "soup.yaml"
        runner.invoke(app, ["init", "--template", "orpo", "--output", str(output)])
        cfg = load_config(Path(output))
        assert cfg.task == "orpo"
        assert cfg.training.orpo_beta == pytest.approx(0.1)


# ─── Wizard ORPO Path Tests ────────────────────────────────────────────────────


class TestORPOWizardPath:
    """Test the interactive wizard auto-sets format for ORPO task."""

    def test_wizard_orpo_task_sets_dpo_format(self):
        from soup_cli.commands.init import _interactive_wizard

        with mock_patch(
            "soup_cli.commands.init.Prompt.ask",
            side_effect=[
                "some-model",
                "orpo",
                "./data.jsonl",
                "3",
                "yes",
            ],
        ):
            config_text = _interactive_wizard()

        assert "task: orpo" in config_text
        assert "format: dpo" in config_text
        assert "orpo_beta: 0.1" in config_text

    def test_wizard_orpo_does_not_prompt_for_format(self):
        from soup_cli.commands.init import _interactive_wizard

        prompt_calls = []

        def record_prompt(question, **kwargs):
            prompt_calls.append(question)
            answers = {
                "Base model": "some-model",
                "Task": "orpo",
                "Training data path": "./data.jsonl",
                "Epochs": "3",
                "Use QLoRA (4-bit)?": "yes",
            }
            return answers.get(question, kwargs.get("default", ""))

        with mock_patch("soup_cli.commands.init.Prompt.ask", side_effect=record_prompt):
            _interactive_wizard()

        assert not any("format" in call.lower() for call in prompt_calls)


# ─── Config Loader Round-trip Tests ──────────────────────────────────────────


class TestORPOConfigLoaderRoundTrip:
    """Test ORPO template YAML survives round-trip."""

    def test_orpo_template_round_trip(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(TEMPLATES["orpo"])
        assert cfg.task == "orpo"
        assert cfg.training.orpo_beta == pytest.approx(0.1)
        assert cfg.data.format == "dpo"

    def test_orpo_custom_yaml_round_trip(self):
        from soup_cli.config.loader import load_config_from_string

        yaml_str = """
base: custom-model/llama-7b
task: orpo

data:
  train: ./orpo_data.jsonl
  format: dpo
  max_length: 1024

training:
  epochs: 5
  lr: 5e-6
  orpo_beta: 0.05
  quantization: none

output: ./orpo_output
"""
        cfg = load_config_from_string(yaml_str)
        assert cfg.task == "orpo"
        assert cfg.training.orpo_beta == pytest.approx(0.05)
        assert cfg.training.epochs == 5
