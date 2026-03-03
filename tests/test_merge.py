"""Tests for soup merge command."""

import json
from pathlib import Path

from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands.merge import _detect_base_model, _format_size

runner = CliRunner()


# --- _format_size ---

def test_format_size_bytes():
    assert _format_size(512) == "512.0 B"


def test_format_size_kb():
    assert _format_size(2048) == "2.0 KB"


def test_format_size_mb():
    assert _format_size(5 * 1024 * 1024) == "5.0 MB"


def test_format_size_gb():
    assert _format_size(3 * 1024**3) == "3.0 GB"


# --- _detect_base_model ---

def test_detect_base_model(tmp_path: Path):
    config = tmp_path / "adapter_config.json"
    config.write_text(json.dumps({
        "base_model_name_or_path": "meta-llama/Llama-3.1-8B",
    }))
    assert _detect_base_model(config) == "meta-llama/Llama-3.1-8B"


def test_detect_base_model_missing_key(tmp_path: Path):
    config = tmp_path / "adapter_config.json"
    config.write_text(json.dumps({"r": 64}))
    assert _detect_base_model(config) is None


def test_detect_base_model_bad_json(tmp_path: Path):
    config = tmp_path / "adapter_config.json"
    config.write_text("not valid json")
    assert _detect_base_model(config) is None


def test_detect_base_model_missing_file(tmp_path: Path):
    config = tmp_path / "nonexistent.json"
    assert _detect_base_model(config) is None


# --- CLI validation ---

def test_merge_missing_adapter():
    result = runner.invoke(app, ["merge", "--adapter", "/nonexistent"])
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_merge_not_a_lora_adapter(tmp_path: Path):
    """Directory without adapter_config.json should fail."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    result = runner.invoke(app, ["merge", "--adapter", str(model_dir)])
    assert result.exit_code == 1
    assert "not a lora adapter" in result.output.lower()


def test_merge_no_base_model_detected(tmp_path: Path):
    """Adapter with empty config (no base_model_name_or_path) and no --base flag."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(json.dumps({"r": 64}))
    result = runner.invoke(app, ["merge", "--adapter", str(adapter_dir)])
    assert result.exit_code == 1
    assert "base" in result.output.lower()


def test_merge_invalid_dtype(tmp_path: Path):
    """Invalid dtype should fail."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "meta-llama/Llama-3.1-8B",
    }))
    result = runner.invoke(
        app, ["merge", "--adapter", str(adapter_dir), "--dtype", "int8"]
    )
    assert result.exit_code == 1
    assert "invalid dtype" in result.output.lower()


def test_merge_help():
    result = runner.invoke(app, ["merge", "--help"])
    assert result.exit_code == 0
    assert "adapter" in result.output.lower()
    assert "base" in result.output.lower()
