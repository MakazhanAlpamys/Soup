"""Tests for data tools: convert, merge, dedup, stats, and reverse format conversion."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.data.formats import (
    format_to_messages,
    messages_to_format,
)
from soup_cli.data.validator import extended_stats

runner = CliRunner()


# --- Reverse format conversion tests ---


def test_messages_to_alpaca():
    """Convert messages → alpaca format."""
    row = {"messages": [
        {"role": "user", "content": "What is AI?"},
        {"role": "assistant", "content": "AI is artificial intelligence."},
    ]}
    result = messages_to_format(row, "alpaca")
    assert result["instruction"] == "What is AI?"
    assert result["output"] == "AI is artificial intelligence."


def test_messages_to_alpaca_with_system():
    """System message should be preserved in alpaca format."""
    row = {"messages": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ]}
    result = messages_to_format(row, "alpaca")
    assert result["system"] == "You are helpful."
    assert result["instruction"] == "Hello"
    assert result["output"] == "Hi!"


def test_messages_to_sharegpt():
    """Convert messages → sharegpt format."""
    row = {"messages": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ]}
    result = messages_to_format(row, "sharegpt")
    assert result["conversations"][0]["from"] == "human"
    assert result["conversations"][0]["value"] == "Hello"
    assert result["conversations"][1]["from"] == "gpt"
    assert result["conversations"][1]["value"] == "Hi!"


def test_messages_to_chatml():
    """Convert messages → chatml should be identity."""
    row = {"messages": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ]}
    result = messages_to_format(row, "chatml")
    assert result == row


def test_roundtrip_alpaca_to_chatml_to_alpaca():
    """Converting alpaca → chatml → alpaca should preserve content."""
    original = {"instruction": "Explain X", "input": "", "output": "X is amazing."}
    chatml = format_to_messages(original, "alpaca")
    assert chatml is not None
    back = messages_to_format(chatml, "alpaca")
    assert back["instruction"] == original["instruction"]
    assert back["output"] == original["output"]


def test_roundtrip_sharegpt_to_chatml_to_sharegpt():
    """Converting sharegpt → chatml → sharegpt should preserve content."""
    original = {"conversations": [
        {"from": "human", "value": "Hello"},
        {"from": "gpt", "value": "Hi!"},
    ]}
    chatml = format_to_messages(original, "sharegpt")
    assert chatml is not None
    back = messages_to_format(chatml, "sharegpt")
    assert back["conversations"][0]["from"] == "human"
    assert back["conversations"][0]["value"] == "Hello"


def test_messages_to_invalid_format():
    """Invalid target format should raise ValueError."""
    row = {"messages": [{"role": "user", "content": "Hello"}]}
    with pytest.raises(ValueError, match="Cannot convert"):
        messages_to_format(row, "dpo")


# --- Extended stats tests ---


def test_extended_stats_basic():
    """Extended stats should compute percentiles and token counts."""
    data = [
        {"instruction": "short", "output": "ok"},
        {"instruction": "a bit longer instruction here", "output": "response text here"},
        {"instruction": "x" * 200, "output": "y" * 100},
    ]
    result = extended_stats(data)
    assert result["total"] == 3
    assert result["length_p50"] > 0
    assert result["avg_tokens"] > 0
    assert result["min_tokens"] >= 1
    assert len(result["lengths"]) == 3


def test_extended_stats_empty():
    """Extended stats on empty data should return zeros."""
    result = extended_stats([])
    assert result["total"] == 0
    assert result["avg_tokens"] == 0


# --- CLI data command tests ---


@pytest.fixture
def sample_alpaca_file(tmp_path: Path) -> Path:
    """Create a sample alpaca JSONL file."""
    data = [
        {"instruction": "What is AI?", "input": "", "output": "AI is..."},
        {"instruction": "Explain ML", "input": "", "output": "ML is..."},
        {"instruction": "What is DL?", "input": "", "output": "DL is..."},
    ]
    filepath = tmp_path / "sample.jsonl"
    with open(filepath, "w") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")
    return filepath


@pytest.fixture
def sample_sharegpt_file(tmp_path: Path) -> Path:
    """Create a sample sharegpt JSONL file."""
    data = [
        {"conversations": [
            {"from": "human", "value": "Hello"},
            {"from": "gpt", "value": "Hi!"},
        ]},
        {"conversations": [
            {"from": "human", "value": "How are you?"},
            {"from": "gpt", "value": "I'm good!"},
        ]},
    ]
    filepath = tmp_path / "sharegpt.jsonl"
    with open(filepath, "w") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")
    return filepath


def test_convert_command(sample_alpaca_file, tmp_path):
    """soup data convert should convert alpaca → sharegpt."""
    output = str(tmp_path / "output.jsonl")
    result = runner.invoke(
        app, ["data", "convert", str(sample_alpaca_file), "--to", "sharegpt", "-o", output]
    )
    assert result.exit_code == 0
    assert "Converted" in result.output

    # Verify output file
    with open(output) as f:
        lines = f.readlines()
    assert len(lines) == 3
    first = json.loads(lines[0])
    assert "conversations" in first


def test_convert_same_format(sample_alpaca_file):
    """Converting to same format should warn and exit."""
    result = runner.invoke(
        app, ["data", "convert", str(sample_alpaca_file), "--to", "alpaca"]
    )
    assert result.exit_code == 0
    assert "Nothing to convert" in result.output


def test_convert_missing_file():
    """Converting nonexistent file should fail."""
    result = runner.invoke(app, ["data", "convert", "nonexistent.jsonl", "--to", "chatml"])
    assert result.exit_code == 1


def test_merge_command(sample_alpaca_file, sample_sharegpt_file, tmp_path):
    """soup data merge should merge files."""
    output = str(tmp_path / "merged.jsonl")
    result = runner.invoke(
        app,
        ["data", "merge", str(sample_alpaca_file), str(sample_sharegpt_file), "-o", output],
    )
    assert result.exit_code == 0
    assert "Merged" in result.output
    assert "5 rows" in result.output  # 3 + 2

    with open(output) as f:
        lines = f.readlines()
    assert len(lines) == 5


def test_merge_missing_file(sample_alpaca_file):
    """Merge with nonexistent file should fail."""
    result = runner.invoke(
        app, ["data", "merge", str(sample_alpaca_file), "nonexistent.jsonl"]
    )
    assert result.exit_code == 1


def test_stats_command(sample_alpaca_file):
    """soup data stats should show extended statistics."""
    result = runner.invoke(app, ["data", "stats", str(sample_alpaca_file)])
    assert result.exit_code == 0
    assert "p50" in result.output
    assert "Tokens" in result.output


def test_stats_missing_file():
    """Stats on nonexistent file should fail."""
    result = runner.invoke(app, ["data", "stats", "nonexistent.jsonl"])
    assert result.exit_code == 1
