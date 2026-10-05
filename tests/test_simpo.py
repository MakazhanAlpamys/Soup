"""Tests for SimPO training — config, data format, template, routing, sweep."""

from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

import pytest
from pydantic import ValidationError

from soup_cli.config.schema import TEMPLATES, SoupConfig

# ─── Config Tests ───────────────────────────────────────────────────────────


class TestSimPOConfig:
    """Test SimPO task config validation."""

    def test_simpo_task_accepted(self):
        cfg = SoupConfig(
            base="some-model",
            task="simpo",
            data={"train": "./data.jsonl"},
        )
        assert cfg.task == "simpo"

    def test_simpo_gamma_default(self):
        cfg = SoupConfig(
            base="some-model",
            task="simpo",
            data={"train": "./data.jsonl"},
        )
        assert cfg.training.simpo_gamma == 0.5

    def test_cpo_alpha_default(self):
        cfg = SoupConfig(
            base="some-model",
            task="simpo",
            data={"train": "./data.jsonl"},
        )
        assert cfg.training.cpo_alpha == 1.0

    def test_simpo_gamma_custom(self):
        cfg = SoupConfig(
            base="some-model",
            task="simpo",
            data={"train": "./data.jsonl"},
            training={"simpo_gamma": 1.0},
        )
        assert cfg.training.simpo_gamma == pytest.approx(1.0)

    def test_cpo_alpha_must_be_positive(self):
        with pytest.raises(ValidationError):
            SoupConfig(
                base="some-model",
                task="simpo",
                data={"train": "./data.jsonl"},
                training={"cpo_alpha": 0},
            )

    def test_simpo_gamma_zero_accepted(self):
        """simpo_gamma can be 0 (no margin)."""
        cfg = SoupConfig(
            base="some-model",
            task="simpo",
            data={"train": "./data.jsonl"},
            training={"simpo_gamma": 0},
        )
        assert cfg.training.simpo_gamma == 0

    def test_simpo_full_config(self):
        cfg = SoupConfig(
            base="meta-llama/Llama-3.1-8B-Instruct",
            task="simpo",
            data={"train": "./data.jsonl", "format": "dpo", "max_length": 2048},
            training={
                "epochs": 3,
                "lr": 1e-5,
                "simpo_gamma": 1.5,
                "cpo_alpha": 0.5,
                "lora": {"r": 64, "alpha": 16},
                "quantization": "4bit",
            },
        )
        assert cfg.task == "simpo"
        assert cfg.training.simpo_gamma == pytest.approx(1.5)
        assert cfg.training.cpo_alpha == pytest.approx(0.5)


# ─── Template Tests ──────────────────────────────────────────────────────────


class TestSimPOTemplate:
    """Test the SimPO template."""

    def test_simpo_template_exists(self):
        assert "simpo" in TEMPLATES

    def test_simpo_template_valid_yaml(self):
        import yaml

        config = yaml.safe_load(TEMPLATES["simpo"])
        assert config["task"] == "simpo"
        assert config["training"]["simpo_gamma"] == 0.5
        assert config["training"]["cpo_alpha"] == 1.0
        assert config["data"]["format"] == "dpo"

    def test_simpo_template_valid_config(self):
        import yaml

        raw = yaml.safe_load(TEMPLATES["simpo"])
        cfg = SoupConfig(**raw)
        assert cfg.task == "simpo"
        assert cfg.training.simpo_gamma == 0.5


# ─── Train Command Routing Tests ─────────────────────────────────────────────


class TestSimPOTrainRouting:
    """Test that train command routes to SimPO trainer."""

    def test_simpo_import_exists(self):
        from soup_cli.trainer.simpo import SimPOTrainerWrapper

        assert SimPOTrainerWrapper is not None

    def test_simpo_wrapper_init(self):
        from soup_cli.trainer.simpo import SimPOTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="simpo",
            data={"train": "./data.jsonl"},
        )
        wrapper = SimPOTrainerWrapper(cfg, device="cpu")
        assert wrapper.config.task == "simpo"
        assert wrapper.device == "cpu"
        assert wrapper.model is None
        assert wrapper.trainer is None

    def test_simpo_wrapper_init_with_options(self):
        from soup_cli.trainer.simpo import SimPOTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="simpo",
            data={"train": "./data.jsonl"},
        )
        wrapper = SimPOTrainerWrapper(
            cfg,
            device="cuda",
            report_to="wandb",
            deepspeed_config="ds.json",
        )
        assert wrapper.report_to == "wandb"
        assert wrapper.deepspeed_config == "ds.json"


# ─── Sweep Shortcut Tests ────────────────────────────────────────────────────


class TestSimPOSweepParams:
    """Test SimPO parameter shortcuts in sweep."""

    def test_simpo_gamma_shortcut(self):
        from soup_cli.commands.sweep import _set_nested_param

        config = {"training": {"simpo_gamma": 0.5}}
        _set_nested_param(config, "simpo_gamma", 1.0)
        assert config["training"]["simpo_gamma"] == 1.0

    def test_cpo_alpha_shortcut(self):
        from soup_cli.commands.sweep import _set_nested_param

        config = {}
        _set_nested_param(config, "cpo_alpha", 0.5)
        assert config["training"]["cpo_alpha"] == pytest.approx(0.5)

    def test_sweep_run_single_routes_to_simpo_trainer(self):
        from soup_cli.commands.sweep import _run_single

        cfg = SoupConfig(
            base="some-model",
            task="simpo",
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
            mock_patch("soup_cli.trainer.simpo.SimPOTrainerWrapper.setup"),
            mock_patch(
                "soup_cli.trainer.simpo.SimPOTrainerWrapper.train", return_value=fake_result
            ) as mock_train,
        ):
            mock_tracker = MagicMock()
            mock_tracker.start_run.return_value = "run-simpo-1"
            mock_tracker_cls.return_value = mock_tracker

            result = _run_single(cfg, {}, "simpo_run_1", None)

        mock_train.assert_called_once()
        assert result["run_id"] == "run-simpo-1"


# ─── Train Method Guard Test ──────────────────────────────────────────────────


class TestSimPOTrainGuard:
    def test_train_before_setup_raises_runtime_error(self):
        from soup_cli.trainer.simpo import SimPOTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="simpo",
            data={"train": "./data.jsonl"},
        )
        wrapper = SimPOTrainerWrapper(cfg)
        with pytest.raises(RuntimeError, match="setup\\(dataset\\) first"):
            wrapper.train()


# ─── Train Method Result Structure ───────────────────────────────────────────


class TestSimPOTrainResults:
    def _make_wrapper_with_mock_trainer(self, log_history=None, global_step=20):
        from soup_cli.trainer.simpo import SimPOTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="simpo",
            data={"train": "./data.jsonl"},
            output="./output",
        )
        wrapper = SimPOTrainerWrapper(cfg, device="cpu")
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


# ─── CLI Init SimPO Template Tests ─────────────────────────────────────────────


class TestSimPOInitTemplate:
    def test_init_simpo_template_creates_file(self, tmp_path):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        output = tmp_path / "soup.yaml"
        result = runner.invoke(app, ["init", "--template", "simpo", "--output", str(output)])
        assert result.exit_code == 0
        assert output.exists()
        content = output.read_text()
        assert "task: simpo" in content
        assert "simpo_gamma" in content

    def test_init_simpo_template_produces_valid_config(self, tmp_path):
        from pathlib import Path

        from typer.testing import CliRunner

        from soup_cli.cli import app
        from soup_cli.config.loader import load_config

        runner = CliRunner()
        output = tmp_path / "soup.yaml"
        runner.invoke(app, ["init", "--template", "simpo", "--output", str(output)])
        cfg = load_config(Path(output))
        assert cfg.task == "simpo"


# ─── Wizard SimPO Path Tests ────────────────────────────────────────────────────


class TestSimPOWizardPath:
    def test_wizard_simpo_task_sets_dpo_format(self):
        from soup_cli.commands.init import _interactive_wizard

        with mock_patch(
            "soup_cli.commands.init.Prompt.ask",
            side_effect=[
                "some-model",
                "simpo",
                "./data.jsonl",
                "3",
                "yes",
            ],
        ):
            config_text = _interactive_wizard()

        assert "task: simpo" in config_text
        assert "format: dpo" in config_text
        assert "simpo_gamma: 0.5" in config_text
        assert "cpo_alpha: 1.0" in config_text


# ─── Config Loader Round-trip Tests ──────────────────────────────────────────


class TestSimPOConfigLoaderRoundTrip:
    def test_simpo_template_round_trip(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(TEMPLATES["simpo"])
        assert cfg.task == "simpo"
        assert cfg.training.simpo_gamma == pytest.approx(0.5)
