"""Tests for data loading, format detection, and validation."""

from pathlib import Path

from soup_cli.data.formats import detect_format, format_to_messages
from soup_cli.data.loader import load_raw_data
from soup_cli.data.validator import validate_and_stats


def test_load_jsonl(sample_alpaca_data: Path):
    data = load_raw_data(sample_alpaca_data)
    assert len(data) == 3
    assert "instruction" in data[0]
    assert "output" in data[0]


def test_detect_alpaca_format():
    data = [{"instruction": "test", "input": "", "output": "result"}]
    assert detect_format(data) == "alpaca"


def test_detect_sharegpt_format():
    data = [{"conversations": [{"from": "human", "value": "hi"}]}]
    assert detect_format(data) == "sharegpt"


def test_detect_chatml_format():
    data = [{"messages": [{"role": "user", "content": "hi"}]}]
    assert detect_format(data) == "chatml"


def test_convert_alpaca():
    row = {"instruction": "Explain AI", "input": "", "output": "AI is..."}
    result = format_to_messages(row, "alpaca")
    assert result is not None
    assert len(result["messages"]) == 2
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][1]["role"] == "assistant"


def test_convert_alpaca_with_input():
    row = {"instruction": "Translate", "input": "hello", "output": "привет"}
    result = format_to_messages(row, "alpaca")
    assert "hello" in result["messages"][0]["content"]


def test_convert_sharegpt():
    row = {
        "conversations": [
            {"from": "human", "value": "What is 2+2?"},
            {"from": "gpt", "value": "4"},
        ]
    }
    result = format_to_messages(row, "sharegpt")
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][1]["role"] == "assistant"


def test_validate_stats(sample_alpaca_data: Path):
    data = load_raw_data(sample_alpaca_data)
    stats = validate_and_stats(data)
    assert stats["total"] == 3
    assert "instruction" in stats["columns"]
    assert stats["avg_length"] > 0


def test_validate_with_format(sample_alpaca_data: Path):
    data = load_raw_data(sample_alpaca_data)
    stats = validate_and_stats(data, expected_format="alpaca")
    assert stats["valid_rows"] == 3
    assert len(stats["issues"]) == 0  # no issues for valid data


def test_detect_dpo_format():
    data = [{"prompt": "What is AI?", "chosen": "AI is...", "rejected": "I don't know"}]
    assert detect_format(data) == "dpo"


def test_convert_dpo():
    row = {"prompt": "Explain gravity", "chosen": "Gravity is a force", "rejected": "No idea"}
    result = format_to_messages(row, "dpo")
    assert result is not None
    assert result["prompt"] == "Explain gravity"
    assert result["chosen"] == "Gravity is a force"
    assert result["rejected"] == "No idea"
