"""Tests for soup chat command."""

import json
from pathlib import Path

from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands.chat import _detect_base_model

runner = CliRunner()


def test_chat_missing_model_path():
    """Chat with nonexistent path should fail."""
    result = runner.invoke(app, ["chat", "--model", "/nonexistent/path"])
    assert result.exit_code == 1


def test_detect_base_model_valid(tmp_path: Path):
    """Should read base_model_name_or_path from adapter_config.json."""
    config_path = tmp_path / "adapter_config.json"
    config_path.write_text(
        json.dumps(
            {
                "base_model_name_or_path": "meta-llama/Llama-3.1-8B-Instruct",
                "r": 64,
                "lora_alpha": 16,
            }
        )
    )
    result = _detect_base_model(config_path)
    assert result == "meta-llama/Llama-3.1-8B-Instruct"


def test_detect_base_model_missing_key(tmp_path: Path):
    """Should return None if base_model_name_or_path is missing."""
    config_path = tmp_path / "adapter_config.json"
    config_path.write_text(json.dumps({"r": 64}))
    result = _detect_base_model(config_path)
    assert result is None


def test_detect_base_model_invalid_json(tmp_path: Path):
    """Should return None for malformed JSON."""
    config_path = tmp_path / "adapter_config.json"
    config_path.write_text("not valid json {{{")
    result = _detect_base_model(config_path)
    assert result is None


def test_detect_base_model_missing_file(tmp_path: Path):
    """Should return None for nonexistent file."""
    config_path = tmp_path / "nonexistent.json"
    result = _detect_base_model(config_path)
    assert result is None


def test_chat_adapter_without_base_model(tmp_path: Path):
    """Chat with adapter that has no base model info should fail."""
    # Create fake adapter directory with adapter_config.json but no base model
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    config = adapter_dir / "adapter_config.json"
    config.write_text(json.dumps({"r": 64}))

    result = runner.invoke(app, ["chat", "--model", str(adapter_dir)])
    assert result.exit_code == 1
    assert "Cannot detect base model" in result.output


def test_chat_non_adapter_directory(tmp_path: Path):
    """Chat with directory that has no adapter_config.json skips base model detection."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    # No adapter_config.json → not an adapter, will try to load directly
    # This will fail because there's no actual model, but it should get past validation
    result = runner.invoke(app, ["chat", "--model", str(model_dir)])
    # Should fail during model loading, not during validation
    assert result.exit_code == 1
