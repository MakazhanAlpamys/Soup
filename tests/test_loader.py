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


@pytest.mark.parametrize("suffix", ["json", "jsonl", "csv", "txt"])
@pytest.mark.parametrize("bom", [False, True], ids=["plain", "bom"])
def test_utf8_bom_datasets_match_plain_utf8(tmp_path: Path, suffix: str, bom: bool) -> None:
    """A BOM is an encoding signature, not a field name or document character."""
    rows = [{"instruction": "Сұрақ", "input": "", "output": "Жауап"},
            {"instruction": "Question", "input": "", "output": "Answer"}]
    bodies = {
        "json": json.dumps(rows, ensure_ascii=False),
        "jsonl": "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        "csv": "instruction,input,output\r\nСұрақ,,Жауап\r\nQuestion,,Answer\r\n",
        "txt": "Сұрақ\n\nQuestion\n",
    }
    expected = [{"text": "Сұрақ"}, {"text": "Question"}] if suffix == "txt" else rows
    path = tmp_path / f"data.{suffix}"
    path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + bodies[suffix].encode("utf-8"))
    assert load_raw_data(path) == expected


def test_excel_utf8_bom_csv_is_detected_as_alpaca(tmp_path: Path) -> None:
    from soup_cli.data.formats import detect_format, format_to_messages

    path = tmp_path / "excel.csv"
    path.write_bytes(b"\xef\xbb\xbf" + (
        'instruction,input,output\r\n"Question, with comma",context,Answer\r\n'
    ).encode("utf-8"))
    rows = load_raw_data(path)
    assert detect_format(rows) == "alpaca"
    assert format_to_messages(rows[0], "alpaca") == {"messages": [
        {"role": "user", "content": "Question, with comma\ncontext"},
        {"role": "assistant", "content": "Answer"},
    ]}


@pytest.mark.parametrize("suffix, body", [
    ("json", '[{"text":"inner\ufeffmarker"}]'),
    ("csv", "text\r\ninner\ufeffmarker\r\n"),
    ("txt", "inner\ufeffmarker\n"),
])
def test_bom_codec_preserves_embedded_character(tmp_path: Path, suffix: str, body: str) -> None:
    path = tmp_path / f"data.{suffix}"
    path.write_bytes(b"\xef\xbb\xbf" + body.encode("utf-8"))
    assert load_raw_data(path) == [{"text": "inner\ufeffmarker"}]


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


def test_load_parquet_preserves_nested_lists(tmp_path: Path):
    """Parquet nested messages should load as Python lists."""
    import pandas as pd

    records = [
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ]
        }
    ]

    path = tmp_path / "chatml.parquet"
    pd.DataFrame(records).to_parquet(path)

    loaded = load_raw_data(path)

    assert len(loaded) == 1
    assert isinstance(loaded[0]["messages"], list)
