"""Tests for eval command (basic CLI validation, not full eval runs)."""

from typer.testing import CliRunner

from soup_cli.cli import app

runner = CliRunner()


def test_eval_missing_model():
    """soup eval with nonexistent model path should fail."""
    result = runner.invoke(app, ["eval", "--model", "nonexistent_path", "--benchmarks", "mmlu"])
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_eval_help():
    """soup eval --help should show usage info."""
    result = runner.invoke(app, ["eval", "--help"])
    assert result.exit_code == 0
    assert "benchmarks" in result.output.lower()
    assert "model" in result.output.lower()
