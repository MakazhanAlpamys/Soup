"""Tests for data/formats.py — reverse conversion + edge cases."""

import pytest

from soup_cli.data.formats import (
    detect_format,
    format_to_messages,
    messages_to_format,
)

# --- detect_format edge cases ---


def test_detect_format_empty():
    """Empty dataset should raise ValueError."""
    with pytest.raises(ValueError, match="Empty"):
        detect_format([])


def test_detect_format_unknown_keys():
    """Unrecognized keys should raise ValueError."""
    with pytest.raises(ValueError, match="Cannot detect"):
        detect_format([{"foo": "bar", "baz": 123}])


# --- format_to_messages edge cases ---


def test_format_to_messages_unknown_format():
    """Unknown format should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown format"):
        format_to_messages({"a": "b"}, "unknown_fmt")


def test_format_to_messages_bad_row():
    """Missing required keys should return None (gracefully handled)."""
    result = format_to_messages({"wrong": "keys"}, "alpaca")
    assert result is None


def test_convert_chatml_passthrough():
    """ChatML format should pass through messages directly."""
    row = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}
    result = format_to_messages(row, "chatml")
    assert result["messages"] == row["messages"]


def test_convert_alpaca_with_system():
    """Alpaca row with system field should include system message."""
    row = {
        "instruction": "Translate",
        "input": "",
        "output": "Done",
        "system": "You are a translator.",
    }
    result = format_to_messages(row, "alpaca")
    assert len(result["messages"]) == 3
    assert result["messages"][0]["role"] == "system"
    assert result["messages"][0]["content"] == "You are a translator."


def test_convert_sharegpt_with_system():
    """ShareGPT with system turn should map correctly."""
    row = {
        "conversations": [
            {"from": "system", "value": "Be helpful."},
            {"from": "human", "value": "Hello"},
            {"from": "gpt", "value": "Hi there!"},
        ]
    }
    result = format_to_messages(row, "sharegpt")
    assert result["messages"][0]["role"] == "system"
    assert result["messages"][1]["role"] == "user"
    assert result["messages"][2]["role"] == "assistant"


# --- reverse conversion: messages_to_format ---


def test_messages_to_alpaca():
    """Convert messages back to alpaca format."""
    row = {
        "messages": [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "A programming language."},
        ]
    }
    result = messages_to_format(row, "alpaca")
    assert result is not None
    assert result["instruction"] == "What is Python?"
    assert result["output"] == "A programming language."
    assert result["input"] == ""


def test_messages_to_alpaca_with_system():
    """Convert messages with system to alpaca format."""
    row = {
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "Artificial Intelligence."},
        ]
    }
    result = messages_to_format(row, "alpaca")
    assert result["system"] == "Be brief."
    assert result["instruction"] == "What is AI?"
    assert result["output"] == "Artificial Intelligence."


def test_messages_to_sharegpt():
    """Convert messages to sharegpt format."""
    row = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
    }
    result = messages_to_format(row, "sharegpt")
    assert result is not None
    convs = result["conversations"]
    assert len(convs) == 2
    assert convs[0]["from"] == "human"
    assert convs[0]["value"] == "Hello"
    assert convs[1]["from"] == "gpt"
    assert convs[1]["value"] == "Hi!"


def test_messages_to_chatml():
    """Convert to chatml should return the row as-is."""
    row = {
        "messages": [
            {"role": "user", "content": "Test"},
            {"role": "assistant", "content": "OK"},
        ]
    }
    result = messages_to_format(row, "chatml")
    assert result is row  # should be the same object (passthrough)


def test_messages_to_unknown_format():
    """Unknown target format should raise ValueError."""
    row = {"messages": [{"role": "user", "content": "hi"}]}
    with pytest.raises(ValueError, match="Cannot convert"):
        messages_to_format(row, "unknown")


def test_messages_to_format_bad_row():
    """Broken row should return None."""
    result = messages_to_format({"bad": "data"}, "alpaca")
    assert result is None


# --- round-trip tests ---


def test_roundtrip_alpaca():
    """alpaca → messages → alpaca should preserve data."""
    original = {"instruction": "Explain ML", "input": "", "output": "ML is..."}
    messages = format_to_messages(original, "alpaca")
    back = messages_to_format(messages, "alpaca")
    assert back["instruction"] == original["instruction"]
    assert back["output"] == original["output"]


def test_roundtrip_sharegpt():
    """sharegpt → messages → sharegpt should preserve data."""
    original = {
        "conversations": [
            {"from": "human", "value": "Hi"},
            {"from": "gpt", "value": "Hello!"},
        ]
    }
    messages = format_to_messages(original, "sharegpt")
    back = messages_to_format(messages, "sharegpt")
    assert back["conversations"][0]["from"] == "human"
    assert back["conversations"][0]["value"] == "Hi"
    assert back["conversations"][1]["from"] == "gpt"
    assert back["conversations"][1]["value"] == "Hello!"
