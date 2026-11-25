"""Tests for soup quickstart command."""

import json

import pytest
import yaml
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands.quickstart import DEMO_CONFIG, DEMO_DATA

runner = CliRunner()


def test_quickstart_help():
    """soup quickstart --help works."""
    result = runner.invoke(app, ["quickstart", "--help"])
    assert result.exit_code == 0
    assert "demo" in result.output.lower() or "quickstart" in result.output.lower()


def test_quickstart_dry_run(tmp_path, monkeypatch):
    """soup quickstart --dry-run creates files but does not train."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["quickstart", "--dry-run"])
    assert result.exit_code == 0
    assert "Dry run" in result.output

    # Check files were created
    data_path = tmp_path / "quickstart_data.jsonl"
    config_path = tmp_path / "quickstart_soup.yaml"
    assert data_path.exists()
    assert config_path.exists()

    # Validate data
    with open(data_path) as fh:
        lines = fh.readlines()
    assert len(lines) == len(DEMO_DATA)
    first = json.loads(lines[0])
    assert "instruction" in first
    assert "output" in first


@pytest.mark.parametrize(
    "name",
    ["demo", "run #1", "run 🧪", "run\u2028next"],
    ids=["plain", "yaml-comment", "emoji", "unicode-line-separator"],
)
def test_quickstart_output_config_preserves_directory(tmp_path, monkeypatch, name):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "soup_cli.commands.quickstart._pick_quickstart_model",
        lambda: ("test/model", None),
    )
    directory = tmp_path / name
    result = runner.invoke(app, ["quickstart", "--dry-run", "--output", name])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    directory = directory.resolve()
    data_path = directory / "quickstart_data.jsonl"
    assert data_path.is_file()
    rendered = (directory / "quickstart_soup.yaml").read_text(encoding="utf-8")
    assert (directory / "soup.yaml").read_text(encoding="utf-8") == rendered
    config = yaml.safe_load(rendered)
    assert config["base"] == "test/model"
    assert config["data"]["train"] == str(data_path)
    assert config["output"] == str(directory / "quickstart_output")


def test_quickstart_dry_run_existing_files(tmp_path, monkeypatch):
    """soup quickstart --dry-run does not overwrite existing files."""
    monkeypatch.chdir(tmp_path)

    # Create existing files
    data_path = tmp_path / "quickstart_data.jsonl"
    data_path.write_text("existing data\n")
    config_path = tmp_path / "quickstart_soup.yaml"
    config_path.write_text("existing config\n")

    result = runner.invoke(app, ["quickstart", "--dry-run"])
    assert result.exit_code == 0
    assert "already exists" in result.output

    # Files should NOT be overwritten
    assert data_path.read_text() == "existing data\n"
    assert config_path.read_text() == "existing config\n"


def test_quickstart_cancel(tmp_path, monkeypatch):
    """soup quickstart can be cancelled."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["quickstart"], input="n\n")
    assert result.exit_code == 0
    assert "Cancelled" in result.output


def test_demo_data_valid():
    """Demo data is valid alpaca format."""
    for entry in DEMO_DATA:
        assert "instruction" in entry
        assert "output" in entry
        assert "input" in entry
        assert len(entry["instruction"]) > 0
        assert len(entry["output"]) > 0


def test_demo_data_count():
    """Demo data has 20 examples."""
    assert len(DEMO_DATA) == 20


def test_demo_config_valid():
    """Demo config is valid YAML."""
    config = yaml.safe_load(DEMO_CONFIG)
    assert config["base"] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert config["task"] == "sft"
    assert config["data"]["train"] == "./quickstart_data.jsonl"
    assert config["data"]["format"] == "alpaca"
    assert config["training"]["epochs"] == 1


def test_quickstart_yes_dry_run(tmp_path, monkeypatch):
    """soup quickstart --yes --dry-run skips confirmation."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["quickstart", "--yes", "--dry-run"])
    assert result.exit_code == 0
    assert "Dry run" in result.output
    assert (tmp_path / "quickstart_data.jsonl").exists()
    assert (tmp_path / "quickstart_soup.yaml").exists()
