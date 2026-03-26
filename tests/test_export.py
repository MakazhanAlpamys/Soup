"""Tests for soup export command."""

import json
from pathlib import Path

from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands.export import (
    GGUF_QUANT_TYPES,
    SUPPORTED_FORMATS,
    _detect_base_model,
    _find_quantize_binary,
    _format_size,
)

runner = CliRunner()


# --- _format_size ---

def test_format_size_bytes():
    assert _format_size(100) == "100.0 B"


def test_format_size_gb():
    assert _format_size(2 * 1024**3) == "2.0 GB"


# --- _detect_base_model ---

def test_detect_base_model_valid(tmp_path: Path):
    config = tmp_path / "adapter_config.json"
    config.write_text(json.dumps({
        "base_model_name_or_path": "meta-llama/Llama-3.1-8B",
    }))
    assert _detect_base_model(config) == "meta-llama/Llama-3.1-8B"


def test_detect_base_model_no_key(tmp_path: Path):
    config = tmp_path / "adapter_config.json"
    config.write_text(json.dumps({"r": 64}))
    assert _detect_base_model(config) is None


def test_detect_base_model_bad_json(tmp_path: Path):
    config = tmp_path / "adapter_config.json"
    config.write_text("bad json")
    assert _detect_base_model(config) is None


# --- _find_quantize_binary ---

def test_find_quantize_binary_not_found(tmp_path: Path):
    """Should return None if no quantize binary exists."""
    assert _find_quantize_binary(tmp_path) is None


def test_find_quantize_binary_in_build(tmp_path: Path):
    """Should find binary in build/bin/."""
    bin_dir = tmp_path / "build" / "bin"
    bin_dir.mkdir(parents=True)
    quantize = bin_dir / "llama-quantize"
    quantize.write_text("fake binary")
    assert _find_quantize_binary(tmp_path) == quantize


# --- Constants ---

def test_supported_formats():
    assert "gguf" in SUPPORTED_FORMATS


def test_gguf_quant_types():
    assert "q4_k_m" in GGUF_QUANT_TYPES
    assert "f16" in GGUF_QUANT_TYPES
    assert len(GGUF_QUANT_TYPES) >= 4


# --- CLI validation ---

def test_export_missing_model():
    result = runner.invoke(app, ["export", "--model", "/nonexistent"])
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_export_unsupported_format(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    result = runner.invoke(
        app, ["export", "--model", str(model_dir), "--format", "safetensors"]
    )
    assert result.exit_code == 1
    assert "unsupported format" in result.output.lower()


def test_export_unsupported_quant(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    result = runner.invoke(
        app, ["export", "--model", str(model_dir), "--quant", "q2_k"]
    )
    assert result.exit_code == 1
    assert "unsupported quantization" in result.output.lower()


def test_export_adapter_no_base(tmp_path: Path):
    """LoRA adapter without detectable base model and no --base flag."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(json.dumps({"r": 64}))
    result = runner.invoke(app, ["export", "--model", str(adapter_dir)])
    assert result.exit_code == 1
    assert "base" in result.output.lower()


def test_export_help():
    result = runner.invoke(app, ["export", "--help"])
    assert result.exit_code == 0
    assert "gguf" in result.output.lower()
    assert "quant" in result.output.lower()
