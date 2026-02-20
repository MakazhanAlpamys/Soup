"""Tests for config loading and validation."""

from pathlib import Path

import pytest

from soup_cli.config.loader import load_config
from soup_cli.config.schema import SoupConfig


def test_load_valid_config(sample_config: Path):
    """Valid config should parse without errors."""
    cfg = load_config(sample_config)
    assert isinstance(cfg, SoupConfig)
    assert cfg.base == "meta-llama/Llama-3.1-8B-Instruct"
    assert cfg.task == "sft"
    assert cfg.training.epochs == 1
    assert cfg.training.lora.r == 8
    assert cfg.training.quantization == "4bit"


def test_config_defaults():
    """Config should fill in defaults for optional fields."""
    cfg = SoupConfig(
        base="some-model",
        data={"train": "./data.jsonl"},
    )
    assert cfg.task == "sft"
    assert cfg.training.epochs == 3
    assert cfg.training.lr == 2e-5
    assert cfg.training.batch_size == "auto"
    assert cfg.training.lora.r == 64
    assert cfg.training.quantization == "4bit"
    assert cfg.output == "./output"


def test_config_invalid_task():
    """Invalid task should raise validation error."""
    with pytest.raises(Exception):
        SoupConfig(
            base="some-model",
            task="invalid_task",
            data={"train": "./data.jsonl"},
        )


def test_config_val_split_bounds():
    """val_split must be between 0 and 0.5."""
    with pytest.raises(Exception):
        SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl", "val_split": 0.9},
        )
