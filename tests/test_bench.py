"""Tests for soup bench CLI command."""

from typer.testing import CliRunner
from soup_cli.cli import app

runner = CliRunner()


def test_bench_model_not_found():
    """soup bench with nonexistent model should fail gracefully."""
    result = runner.invoke(app, ["bench", "nonexistent_model_path"])
    assert result.exit_code == 1
    assert "not found" in result.output.lower()
