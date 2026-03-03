"""Tests for resume training and W&B integration."""

from pathlib import Path

from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands.train import _resolve_checkpoint

runner = CliRunner()


# --- _resolve_checkpoint ---

def test_resolve_checkpoint_auto_finds_latest(tmp_path: Path):
    """auto should find the latest checkpoint by number."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "checkpoint-100").mkdir()
    (output_dir / "checkpoint-200").mkdir()
    (output_dir / "checkpoint-50").mkdir()

    result = _resolve_checkpoint("auto", str(output_dir))
    assert result == str(output_dir / "checkpoint-200")


def test_resolve_checkpoint_auto_with_experiment_name(tmp_path: Path):
    """auto should look inside experiment_name subdirectory."""
    output_dir = tmp_path / "output"
    exp_dir = output_dir / "my-experiment"
    exp_dir.mkdir(parents=True)
    (exp_dir / "checkpoint-100").mkdir()
    (exp_dir / "checkpoint-300").mkdir()

    result = _resolve_checkpoint("auto", str(output_dir), experiment_name="my-experiment")
    assert result == str(exp_dir / "checkpoint-300")


def test_resolve_checkpoint_auto_no_checkpoints(tmp_path: Path):
    """auto should return None if no checkpoints exist."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    result = _resolve_checkpoint("auto", str(output_dir))
    assert result is None


def test_resolve_checkpoint_auto_missing_dir(tmp_path: Path):
    """auto should return None if output dir doesn't exist."""
    result = _resolve_checkpoint("auto", str(tmp_path / "nonexistent"))
    assert result is None


def test_resolve_checkpoint_direct_path(tmp_path: Path):
    """Direct path to checkpoint should be returned as-is."""
    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()

    result = _resolve_checkpoint(str(checkpoint), str(tmp_path))
    assert result == str(checkpoint)


def test_resolve_checkpoint_direct_path_nonexistent():
    """Nonexistent direct path should return None."""
    result = _resolve_checkpoint("/nonexistent/checkpoint-100", "/output")
    assert result is None


def test_resolve_checkpoint_auto_ignores_non_checkpoint_dirs(tmp_path: Path):
    """auto should ignore directories that don't start with checkpoint-."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "checkpoint-100").mkdir()
    (output_dir / "logs").mkdir()
    (output_dir / "runs").mkdir()

    result = _resolve_checkpoint("auto", str(output_dir))
    assert result == str(output_dir / "checkpoint-100")


# --- CLI flags ---

def test_train_resume_flag_in_help():
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "--resume" in result.output


def test_train_wandb_flag_in_help():
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "--wandb" in result.output


def test_train_resume_nonexistent_checkpoint(tmp_path: Path):
    """Resume with nonexistent checkpoint path should fail (after config validation)."""
    config_file = tmp_path / "soup.yaml"
    config_file.write_text("""
base: meta-llama/Llama-3.1-8B
data:
  train: ./data.jsonl
""")
    result = runner.invoke(
        app, ["train", "--config", str(config_file), "--resume", "/nonexistent/checkpoint"]
    )
    assert result.exit_code == 1
    assert "no checkpoint found" in result.output.lower()
