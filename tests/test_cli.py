"""Tests for CLI commands."""

from typer.testing import CliRunner

from soup_cli import __version__
from soup_cli.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_version_full():
    result = runner.invoke(app, ["version", "--full"])
    assert result.exit_code == 0
    assert __version__ in result.output
    assert "Python" in result.output


def test_version_full_short_flag():
    result = runner.invoke(app, ["version", "-f"])
    assert result.exit_code == 0
    assert "Python" in result.output


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Fine-tune" in result.output


def test_init_unknown_template():
    result = runner.invoke(app, ["init", "--template", "nonexistent"])
    assert result.exit_code == 1


def test_train_missing_config():
    result = runner.invoke(app, ["train", "--config", "nonexistent.yaml"])
    assert result.exit_code == 1


def test_chat_missing_model():
    result = runner.invoke(app, ["chat", "--model", "nonexistent_path"])
    assert result.exit_code == 1


def test_push_missing_model():
    result = runner.invoke(
        app, ["push", "--model", "nonexistent_path", "--repo", "user/model"]
    )
    assert result.exit_code == 1


def test_push_not_a_directory(tmp_path):
    # Create a file (not a directory)
    fake_file = tmp_path / "model.bin"
    fake_file.write_text("not a model")
    result = runner.invoke(
        app, ["push", "--model", str(fake_file), "--repo", "user/model"]
    )
    assert result.exit_code == 1


def test_push_invalid_model_dir(tmp_path):
    # Create an empty directory (no adapter_config.json or config.json)
    model_dir = tmp_path / "empty_model"
    model_dir.mkdir()
    result = runner.invoke(
        app, ["push", "--model", str(model_dir), "--repo", "user/model"]
    )
    assert result.exit_code == 1


def test_help_shows_all_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "chat" in result.output
    assert "push" in result.output
    assert "train" in result.output
    assert "init" in result.output
    assert "export" in result.output
    assert "merge" in result.output
