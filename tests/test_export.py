"""Tests for soup export command."""

import json
import re
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

def test_find_quantize_binary_not_found(tmp_path: Path, monkeypatch):
    """Should return None if no quantize binary exists."""
    monkeypatch.setattr("soup_cli.utils.tool_paths.find_tool", lambda _name, path=None: None)
    assert _find_quantize_binary(tmp_path) is None


def test_find_quantize_binary_in_build(tmp_path: Path):
    """Should find binary in build/bin/."""
    bin_dir = tmp_path / "build" / "bin"
    bin_dir.mkdir(parents=True)
    quantize = bin_dir / "llama-quantize"
    quantize.write_text("fake binary")
    assert _find_quantize_binary(tmp_path) == quantize


def test_find_quantize_binary_on_path(tmp_path: Path, monkeypatch):
    """Should retain production discovery of a globally installed binary."""
    global_binary = tmp_path.parent / "global-bin" / "llama-quantize"
    monkeypatch.setattr(
        "soup_cli.utils.tool_paths.find_tool",
        lambda name, path=None: str(global_binary) if name == "llama-quantize" else None,
    )

    assert _find_quantize_binary(tmp_path) == global_binary


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


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Help text with Rich colour removed and line wraps collapsed to one space."""
    return " ".join(_ANSI_RE.sub("", text).split())


def test_export_help_names_every_supported_format():
    """The --format help lists every accepted format (torchao and gguf-ud were missing)."""
    result = runner.invoke(app, ["export", "--help"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    out = _plain(result.output)
    formats_line = out.split("Export format:", 1)
    assert len(formats_line) == 2, out
    # Only the --format sentence: it ends at the deprecation note's closing bracket,
    # before the next option's help (which also says "torchao").
    listed = formats_line[1].split(")", 1)[0]
    for fmt in SUPPORTED_FORMATS:
        assert re.search(rf"(?<![\w-]){re.escape(fmt)}(?![\w-])", listed), (
            f"--format help does not list {fmt!r}: {listed[:200]}"
        )
