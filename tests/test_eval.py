"""Tests for eval command (basic CLI validation, not full eval runs)."""

from typer.testing import CliRunner

from soup_cli.cli import app

runner = CliRunner()


def test_eval_benchmark_missing_model():
    """soup eval benchmark with nonexistent model path should fail."""
    result = runner.invoke(
        app,
        ["eval", "benchmark", "--model", "nonexistent_path", "--benchmarks", "mmlu"],
    )
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_eval_help():
    """soup eval --help should show subcommands."""
    result = runner.invoke(app, ["eval", "--help"])
    assert result.exit_code == 0
    assert "benchmark" in result.output.lower()
    assert "custom" in result.output.lower()
    assert "judge" in result.output.lower()
    assert "leaderboard" in result.output.lower()
