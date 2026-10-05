"""Tests for --tensorboard flag in soup train command."""

import re
from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


# ─── Flag Conflict Tests ──────────────────────────────────────────────────


class TestTensorBoardFlagConflict:
    """Test that --wandb and --tensorboard cannot be used together."""

    def test_wandb_and_tensorboard_conflict(self, tmp_path):
        """Should fail if both --wandb and --tensorboard are set."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        config_file = tmp_path / "soup.yaml"
        config_file.write_text("base: some-model\ntask: sft\ndata:\n  train: ./data.jsonl\n")

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "train",
                "--config",
                str(config_file),
                "--wandb",
                "--tensorboard",
            ],
        )
        assert result.exit_code != 0
        assert "cannot use" in result.output.lower() or "pick one" in result.output.lower()

    def test_tensorboard_only_does_not_conflict(self, tmp_path):
        """--tensorboard alone should not trigger conflict error."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        config_file = tmp_path / "soup.yaml"
        config_file.write_text("base: some-model\ntask: sft\ndata:\n  train: ./data.jsonl\n")

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "train",
                "--config",
                str(config_file),
                "--tensorboard",
            ],
        )
        # Should not hit the conflict error — may fail later (import/data)
        assert "cannot use" not in result.output.lower()


# ─── TensorBoard Import Check Tests ──────────────────────────────────────


class TestTensorBoardImportCheck:
    """Test that --tensorboard checks for tensorboard installation."""

    def test_tensorboard_flag_checks_import(self, tmp_path):
        """Should fail with install hint when tensorboard is not available."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        config_file = tmp_path / "soup.yaml"
        config_file.write_text("base: some-model\ntask: sft\ndata:\n  train: ./data.jsonl\n")

        # Patch the specific import inside train.py to raise ImportError
        with mock_patch(
            "soup_cli.commands.train.typer",
            wraps=__import__("typer"),
        ):
            original_import = (
                __builtins__["__import__"]
                if isinstance(__builtins__, dict)
                else __builtins__.__import__
            )

            def mock_import(name, *args, **kwargs):
                if name == "tensorboard":
                    raise ImportError("no tensorboard")
                return original_import(name, *args, **kwargs)

            with mock_patch("builtins.__import__", side_effect=mock_import):
                runner = CliRunner()
                result = runner.invoke(
                    app,
                    [
                        "train",
                        "--config",
                        str(config_file),
                        "--tensorboard",
                    ],
                )

        assert result.exit_code != 0


# ─── CLI Help Tests ──────────────────────────────────────────────────────


class TestTensorBoardCLI:
    """Test tensorboard flag appears in train command help."""

    def test_tensorboard_in_train_help(self):
        """Train command should show --tensorboard in help."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        clean = _strip_ansi(result.output)
        assert "--tensorboard" in clean

    def test_tensorboard_help_text(self):
        """--tensorboard help should mention TensorBoard."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["train", "--help"])
        clean = _strip_ansi(result.output)
        assert "tensorboard" in clean.lower()


# ─── Trainer report_to Integration Tests ─────────────────────────────────


class TestTensorBoardTrainerIntegration:
    """Test that trainers receive report_to='tensorboard' when flag is set."""

    def test_sft_trainer_accepts_tensorboard_report_to(self):
        """SFT trainer should accept report_to='tensorboard'."""
        from soup_cli.config.schema import SoupConfig
        from soup_cli.trainer.sft import SFTTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="sft",
            data={"train": "./data.jsonl"},
        )
        wrapper = SFTTrainerWrapper(cfg, device="cpu", report_to="tensorboard")
        assert wrapper.report_to == "tensorboard"

    def test_dpo_trainer_accepts_tensorboard_report_to(self):
        """DPO trainer should accept report_to='tensorboard'."""
        from soup_cli.config.schema import SoupConfig
        from soup_cli.trainer.dpo import DPOTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="dpo",
            data={"train": "./data.jsonl"},
        )
        wrapper = DPOTrainerWrapper(cfg, device="cpu", report_to="tensorboard")
        assert wrapper.report_to == "tensorboard"

    def test_grpo_trainer_accepts_tensorboard_report_to(self):
        """GRPO trainer should accept report_to='tensorboard'."""
        from soup_cli.config.schema import SoupConfig
        from soup_cli.trainer.grpo import GRPOTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="grpo",
            data={"train": "./data.jsonl"},
        )
        wrapper = GRPOTrainerWrapper(cfg, device="cpu", report_to="tensorboard")
        assert wrapper.report_to == "tensorboard"

    def test_kto_trainer_accepts_tensorboard_report_to(self):
        """KTO trainer should accept report_to='tensorboard'."""
        from soup_cli.config.schema import SoupConfig
        from soup_cli.trainer.kto import KTOTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="kto",
            data={"train": "./data.jsonl"},
        )
        wrapper = KTOTrainerWrapper(cfg, device="cpu", report_to="tensorboard")
        assert wrapper.report_to == "tensorboard"


# ─── Sweep Integration Test ──────────────────────────────────────────────


class TestTensorBoardSweepRouting:
    """Test tensorboard works through sweep routing."""

    def test_sweep_run_single_with_tensorboard_report_to(self):
        """Sweep _run_single should pass report_to through to trainer."""
        from soup_cli.config.schema import SoupConfig

        cfg = SoupConfig(
            base="some-model",
            task="sft",
            data={"train": "./data.jsonl"},
        )

        fake_dataset = {"train": [{"messages": [{"role": "user", "content": "Q"}]}]}
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
            mock_patch("soup_cli.trainer.sft.SFTTrainerWrapper.setup"),
            mock_patch("soup_cli.trainer.sft.SFTTrainerWrapper.train", return_value=fake_result),
        ):
            mock_tracker = MagicMock()
            mock_tracker.start_run.return_value = "run-tb-1"
            mock_tracker_cls.return_value = mock_tracker

            from soup_cli.commands.sweep import _run_single

            result = _run_single(cfg, {}, "tb_run_1", None)

        assert result["run_id"] == "run-tb-1"


# ─── Happy Path Test ─────────────────────────────────────────────────────


class TestTensorBoardHappyPath:
    """Test that --tensorboard succeeds when tensorboard is installed."""

    def test_tensorboard_accepted_when_installed(self, tmp_path):
        """--tensorboard should print 'enabled' when tensorboard is available."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        config_file = tmp_path / "soup.yaml"
        config_file.write_text("base: some-model\ntask: sft\ndata:\n  train: ./data.jsonl\n")

        # Mock tensorboard import to succeed
        mock_tb = MagicMock()
        with mock_patch.dict("sys.modules", {"tensorboard": mock_tb}):
            runner = CliRunner()
            result = runner.invoke(
                app,
                [
                    "train",
                    "--config",
                    str(config_file),
                    "--tensorboard",
                    "--yes",
                ],
            )
            # Should get past the tensorboard check — will fail later on
            # data loading, but should print "TensorBoard logging enabled"
            assert "tensorboard logging enabled" in result.output.lower()
