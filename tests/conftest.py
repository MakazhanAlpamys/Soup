"""Shared test fixtures."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create a temp directory with sample training data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def sample_alpaca_data(tmp_data_dir: Path) -> Path:
    """Create a sample alpaca-format JSONL file."""
    path = tmp_data_dir / "train.jsonl"
    samples = [
        {"instruction": "What is Python?", "input": "", "output": "Python is a programming language."},
        {"instruction": "Explain gravity", "input": "", "output": "Gravity is a fundamental force."},
        {"instruction": "Translate hello to Spanish", "input": "hello", "output": "hola"},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    return path


@pytest.fixture
def sample_config(tmp_path: Path, sample_alpaca_data: Path) -> Path:
    """Create a sample soup.yaml config."""
    config_path = tmp_path / "soup.yaml"
    config_path.write_text(
        f"""base: meta-llama/Llama-3.1-8B-Instruct
task: sft
data:
  train: {sample_alpaca_data}
  format: alpaca
  val_split: 0.1
training:
  epochs: 1
  lr: 2e-5
  batch_size: 1
  lora:
    r: 8
    alpha: 16
  quantization: 4bit
output: {tmp_path / 'output'}
""",
        encoding="utf-8",
    )
    return config_path
