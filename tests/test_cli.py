"""Tests for CLI commands."""

from typer.testing import CliRunner

from soup_cli.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


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
