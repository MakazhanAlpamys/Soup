"""Tests for IPO training — config, data format, template, routing, sweep."""

from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

import pytest
from pydantic import ValidationError

from soup_cli.config.schema import TEMPLATES, SoupConfig

# ─── Config Tests ───────────────────────────────────────────────────────────


class TestIPOConfig:
    """Test IPO task config validation."""

    def test_ipo_task_accepted(self):
        cfg = SoupConfig(
            base="some-model",
            task="ipo",
            data={"train": "./data.jsonl"},
        )
        assert cfg.task == "ipo"

    def test_ipo_tau_default(self):
        cfg = SoupConfig(
            base="some-model",
            task="ipo",
            data={"train": "./data.jsonl"},
        )
        assert cfg.training.ipo_tau == 0.1

    def test_ipo_tau_custom(self):
        cfg = SoupConfig(
            base="some-model",
            task="ipo",
            data={"train": "./data.jsonl"},
            training={"ipo_tau": 0.5},
        )
        assert cfg.training.ipo_tau == pytest.approx(0.5)

    def test_ipo_tau_must_be_positive(self):
        with pytest.raises(ValidationError):
            SoupConfig(
                base="some-model",
                task="ipo",
                data={"train": "./data.jsonl"},
                training={"ipo_tau": 0},
            )

    def test_ipo_full_config(self):
        cfg = SoupConfig(
            base="meta-llama/Llama-3.1-8B-Instruct",
            task="ipo",
            data={"train": "./data.jsonl", "format": "dpo", "max_length": 2048},
            training={
                "epochs": 3,
                "lr": 1e-5,
                "ipo_tau": 0.2,
                "lora": {"r": 64, "alpha": 16},
                "quantization": "4bit",
            },
        )
        assert cfg.task == "ipo"
        assert cfg.training.ipo_tau == pytest.approx(0.2)

    def test_ipo_uses_dpo_data_format(self):
        cfg = SoupConfig(
            base="some-model",
            task="ipo",
            data={"train": "./data.jsonl", "format": "dpo"},
        )
        assert cfg.data.format == "dpo"


# ─── Template Tests ──────────────────────────────────────────────────────────


class TestIPOTemplate:
    """Test the IPO template."""

    def test_ipo_template_exists(self):
        assert "ipo" in TEMPLATES

    def test_ipo_template_valid_yaml(self):
        import yaml

        config = yaml.safe_load(TEMPLATES["ipo"])
        assert config["task"] == "ipo"
        assert config["training"]["ipo_tau"] == 0.1
        assert config["data"]["format"] == "dpo"

    def test_ipo_template_valid_config(self):
        import yaml

        raw = yaml.safe_load(TEMPLATES["ipo"])
        cfg = SoupConfig(**raw)
        assert cfg.task == "ipo"
        assert cfg.training.ipo_tau == 0.1


# ─── Train Command Routing Tests ─────────────────────────────────────────────


class TestIPOTrainRouting:
    """Test that train command routes to IPO trainer."""

    def test_ipo_import_exists(self):
        from soup_cli.trainer.ipo import IPOTrainerWrapper

        assert IPOTrainerWrapper is not None

    def test_ipo_wrapper_init(self):
        from soup_cli.trainer.ipo import IPOTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="ipo",
            data={"train": "./data.jsonl"},
        )
        wrapper = IPOTrainerWrapper(cfg, device="cpu")
        assert wrapper.config.task == "ipo"
        assert wrapper.device == "cpu"
        assert wrapper.model is None
        assert wrapper.trainer is None

    def test_ipo_wrapper_init_with_options(self):
        from soup_cli.trainer.ipo import IPOTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="ipo",
            data={"train": "./data.jsonl"},
        )
        wrapper = IPOTrainerWrapper(
            cfg,
            device="cuda",
            report_to="wandb",
            deepspeed_config="ds.json",
        )
        assert wrapper.report_to == "wandb"
        assert wrapper.deepspeed_config == "ds.json"


# ─── Sweep Shortcut Tests ────────────────────────────────────────────────────


class TestIPOSweepParams:
    """Test IPO parameter shortcuts in sweep."""

    def test_ipo_tau_shortcut(self):
        from soup_cli.commands.sweep import _set_nested_param

        config = {"training": {"ipo_tau": 0.1}}
        _set_nested_param(config, "ipo_tau", 0.5)
        assert config["training"]["ipo_tau"] == 0.5

    def test_ipo_tau_shortcut_creates_nested_key(self):
        from soup_cli.commands.sweep import _set_nested_param

        config = {}
        _set_nested_param(config, "ipo_tau", 0.2)
        assert config["training"]["ipo_tau"] == pytest.approx(0.2)

    def test_sweep_run_single_routes_to_ipo_trainer(self):
        from soup_cli.commands.sweep import _run_single

        cfg = SoupConfig(
            base="some-model",
            task="ipo",
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
            mock_patch("soup_cli.trainer.ipo.IPOTrainerWrapper.setup"),
            mock_patch(
                "soup_cli.trainer.ipo.IPOTrainerWrapper.train", return_value=fake_result
            ) as mock_train,
        ):
            mock_tracker = MagicMock()
            mock_tracker.start_run.return_value = "run-ipo-1"
            mock_tracker_cls.return_value = mock_tracker

            result = _run_single(cfg, {}, "ipo_run_1", None)

        mock_train.assert_called_once()
        assert result["run_id"] == "run-ipo-1"


# ─── Config Validation Edge Cases ────────────────────────────────────────────


class TestIPOConfigEdgeCases:
    def test_ipo_tau_negative_rejected(self):
        with pytest.raises(ValidationError):
            SoupConfig(
                base="some-model",
                task="ipo",
                data={"train": "./data.jsonl"},
                training={"ipo_tau": -0.1},
            )

    def test_ipo_tau_very_large_accepted(self):
        cfg = SoupConfig(
            base="some-model",
            task="ipo",
            data={"train": "./data.jsonl"},
            training={"ipo_tau": 10.0},
        )
        assert cfg.training.ipo_tau == pytest.approx(10.0)

    def test_ipo_config_unsloth_backend(self):
        cfg = SoupConfig(
            base="some-model",
            task="ipo",
            backend="unsloth",
            data={"train": "./data.jsonl"},
        )
        assert cfg.backend == "unsloth"
        assert cfg.task == "ipo"

    def test_ipo_tokenizer_none_before_setup(self):
        from soup_cli.trainer.ipo import IPOTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="ipo",
            data={"train": "./data.jsonl"},
        )
        wrapper = IPOTrainerWrapper(cfg)
        assert wrapper.tokenizer is None

    def test_ipo_output_dir_none_before_setup(self):
        from soup_cli.trainer.ipo import IPOTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="ipo",
            data={"train": "./data.jsonl"},
        )
        wrapper = IPOTrainerWrapper(cfg)
        assert wrapper._output_dir is None


# ─── Train Method Guard Test ──────────────────────────────────────────────────


class TestIPOTrainGuard:
    def test_train_before_setup_raises_runtime_error(self):
        from soup_cli.trainer.ipo import IPOTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="ipo",
            data={"train": "./data.jsonl"},
        )
        wrapper = IPOTrainerWrapper(cfg)
        with pytest.raises(RuntimeError, match="setup\\(dataset\\) first"):
            wrapper.train()


# ─── Train Method Result Structure ───────────────────────────────────────────


class TestIPOTrainResults:
    def _make_wrapper_with_mock_trainer(self, log_history=None, global_step=20):
        from soup_cli.trainer.ipo import IPOTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="ipo",
            data={"train": "./data.jsonl"},
            output="./output",
        )
        wrapper = IPOTrainerWrapper(cfg, device="cpu")
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
        assert "output_dir" in result
        assert "total_steps" in result

    def test_train_result_losses(self):
        wrapper, _ = self._make_wrapper_with_mock_trainer(
            log_history=[{"loss": 2.0}, {"loss": 0.5}], global_step=20
        )
        result = wrapper.train()
        assert result["initial_loss"] == pytest.approx(2.0)
        assert result["final_loss"] == pytest.approx(0.5)

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


# ─── CLI Init IPO Template Tests ─────────────────────────────────────────────


class TestIPOInitTemplate:
    def test_init_ipo_template_creates_file(self, tmp_path):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        output = tmp_path / "soup.yaml"
        result = runner.invoke(app, ["init", "--template", "ipo", "--output", str(output)])
        assert result.exit_code == 0
        assert output.exists()
        content = output.read_text()
        assert "task: ipo" in content
        assert "ipo_tau" in content
        assert "format: dpo" in content

    def test_init_ipo_template_produces_valid_config(self, tmp_path):
        from pathlib import Path

        from typer.testing import CliRunner

        from soup_cli.cli import app
        from soup_cli.config.loader import load_config

        runner = CliRunner()
        output = tmp_path / "soup.yaml"
        runner.invoke(app, ["init", "--template", "ipo", "--output", str(output)])
        cfg = load_config(Path(output))
        assert cfg.task == "ipo"
        assert cfg.training.ipo_tau == pytest.approx(0.1)


# ─── Wizard IPO Path Tests ────────────────────────────────────────────────────


class TestIPOWizardPath:
    def test_wizard_ipo_task_sets_dpo_format(self):
        from soup_cli.commands.init import _interactive_wizard

        with mock_patch(
            "soup_cli.commands.init.Prompt.ask",
            side_effect=[
                "some-model",
                "ipo",
                "./data.jsonl",
                "3",
                "yes",
            ],
        ):
            config_text = _interactive_wizard()

        assert "task: ipo" in config_text
        assert "format: dpo" in config_text
        assert "ipo_tau: 0.1" in config_text


# ─── Config Loader Round-trip Tests ──────────────────────────────────────────


class TestIPOConfigLoaderRoundTrip:
    def test_ipo_template_round_trip(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(TEMPLATES["ipo"])
        assert cfg.task == "ipo"
        assert cfg.training.ipo_tau == pytest.approx(0.1)
        assert cfg.data.format == "dpo"

    def test_ipo_custom_yaml_round_trip(self):
        from soup_cli.config.loader import load_config_from_string

        yaml_str = """
base: custom-model/llama-7b
task: ipo

data:
  train: ./ipo_data.jsonl
  format: dpo
  max_length: 1024

training:
  epochs: 5
  lr: 5e-6
  ipo_tau: 0.05
  quantization: none

output: ./ipo_output
"""
        cfg = load_config_from_string(yaml_str)
        assert cfg.task == "ipo"
        assert cfg.training.ipo_tau == pytest.approx(0.05)
        assert cfg.training.epochs == 5
