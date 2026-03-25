"""Tests for data loading (loader.py)."""

import json
from pathlib import Path

import pytest

from soup_cli.data.loader import load_raw_data


def test_load_jsonl(sample_alpaca_data: Path):
    """Load JSONL file should return list of dicts."""
    data = load_raw_data(sample_alpaca_data)
    assert len(data) == 3
    assert data[0]["instruction"] == "What is Python?"


def test_load_json(tmp_path: Path):
    """Load JSON array file."""
    path = tmp_path / "data.json"
    records = [
        {"instruction": "Q1", "input": "", "output": "A1"},
        {"instruction": "Q2", "input": "", "output": "A2"},
    ]
    path.write_text(json.dumps(records))
    data = load_raw_data(path)
    assert len(data) == 2
    assert data[0]["instruction"] == "Q1"


def test_load_json_not_array(tmp_path: Path):
    """JSON file with object (not array) should raise ValueError."""
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"key": "value"}))
    with pytest.raises(ValueError, match="list"):
        load_raw_data(path)


def test_load_csv(tmp_path: Path):
    """Load CSV file with headers."""
    path = tmp_path / "data.csv"
    path.write_text("instruction,input,output\nWhat is AI,,AI is...\nExplain ML,,ML is...\n")
    data = load_raw_data(path)
    assert len(data) == 2
    assert data[0]["instruction"] == "What is AI"
    assert data[1]["output"] == "ML is..."


def test_load_nonexistent_file(tmp_path: Path):
    """Loading nonexistent file should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_raw_data(tmp_path / "nonexistent.jsonl")


def test_load_unsupported_extension(tmp_path: Path):
    """Unsupported file extension should raise ValueError."""
    path = tmp_path / "data.xyz"
    path.write_text("hello")
    with pytest.raises(ValueError, match="Unsupported"):
        load_raw_data(path)


def test_load_jsonl_with_empty_lines(tmp_path: Path):
    """JSONL loader should skip empty lines."""
    path = tmp_path / "data.jsonl"
    content = (
        json.dumps({"instruction": "Q1", "output": "A1"}) + "\n"
        + "\n"
        + json.dumps({"instruction": "Q2", "output": "A2"}) + "\n"
        + "\n"
    )
    path.write_text(content)
    data = load_raw_data(path)
    assert len(data) == 2


def test_load_jsonl_with_invalid_line(tmp_path: Path):
    """JSONL loader should skip invalid JSON lines with a warning."""
    path = tmp_path / "data.jsonl"
    content = (
        json.dumps({"instruction": "Q1", "output": "A1"}) + "\n"
        + "this is not json\n"
        + json.dumps({"instruction": "Q2", "output": "A2"}) + "\n"
    )
    path.write_text(content)
    data = load_raw_data(path)
    assert len(data) == 2
