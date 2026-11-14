"""Tests for low-memory merge dependency handling (#2227)."""

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi


def test_merge_low_memory_missing_transformers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    output_dir = tmp_path / "output"

    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 8,
                "lora_alpha": 16,
                "base_model_name_or_path": str(base_dir),
            }
        )
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "transformers", None)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "merge",
            "--adapter",
            str(adapter_dir),
            "--base",
            str(base_dir),
            "--output",
            str(output_dir),
            "--low-memory",
        ],
    )

    clean_output = " ".join(strip_ansi(result.output).split())
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "Missing dependency" in clean_output
    assert "pip install torch transformers safetensors" in clean_output
    assert "UnboundLocalError" not in clean_output
