"""Tests for issue #3061: row text length must measure text, not the repr of list values.

`_compute_row_text_length` counted `len(str(value))` for every top-level
value, so a chat row's length was the length of the Python repr of its
messages list. Flat rows were already correct and must not move.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.data.validator import (
    _compute_row_text_length,
    extended_stats,
    validate_and_stats,
)
from tests.conftest import strip_ansi

runner = CliRunner()

LONG_QUESTION = "a much longer question about fine tuning datasets and their preparation"


def _len(row: dict) -> int:
    return _compute_row_text_length(row)[0]


# --- One row per format: the number reads the text -------------------------


def test_chatml_row_measures_message_text():
    assert _len({"messages": [{"role": "user", "content": "hi"}]}) == 2
    assert _len({"messages": [{"role": "user", "content": LONG_QUESTION}]}) == 71


def test_chatml_two_messages_join_with_one_separator():
    row = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
    }
    assert _len(row) == 14  # 5 + 8 + 1 separator


def test_sharegpt_row_measures_turn_values():
    row = {
        "conversations": [
            {"from": "human", "value": "hello"},
            {"from": "gpt", "value": "hi there"},
        ]
    }
    assert _len(row) == 14


def test_vision_row_measures_text_parts_only():
    row = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "cat.jpg"},
                    {"type": "text", "text": "What is this?"},
                ],
            },
            {"role": "assistant", "content": "A cat."},
        ]
    }
    assert _len(row) == 20  # 13 + 6 + 1 separator


def test_tool_calling_row_measures_content_and_call_arguments():
    row = {
        "messages": [
            {"role": "user", "content": "What is the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        }
                    }
                ],
            },
        ]
    }
    assert _len(row) == 38  # 20 + 17 + 1 separator; the call name is not text


def test_preference_row_measures_chosen_and_rejected_text():
    row = {
        "prompt": "Name a color.",
        "chosen": [{"role": "assistant", "content": "Blue."}],
        "rejected": [{"role": "assistant", "content": "I cannot answer."}],
    }
    assert _len(row) == 36  # 13 + 5 + 16 + 2 separators


# --- Flat rows do not move ---------------------------------------------------


def test_flat_row_with_extra_non_text_column_is_pinned():
    row = {"instruction": "abcd", "id": 123, "output": "ef"}
    assert _len(row) == 11  # 4 + 3 + 2 + 2 separators, before and after the fix


def test_flat_alpaca_row_unchanged():
    assert _len({"instruction": "hello", "output": "hi there"}) == 14


def test_dict_value_still_counts_as_before():
    # Only list values change; a dict value keeps its previous treatment.
    assert _len({"text": "ab", "meta": {"a": 1}}) == 11  # 2 + len("{'a': 1}") + 1


# --- List edge cases ---------------------------------------------------------


def test_empty_messages_list_counts_nothing():
    assert _compute_row_text_length({"messages": []}) == (0, 0)


def test_message_with_empty_content_contributes_no_text():
    row = {"messages": [{"role": "user", "content": ""}], "id": 7}
    assert _len(row) == 1  # only the id counts


def test_content_parts_with_non_text_part():
    row = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "abc"},
                    {"type": "image", "image": "i.png"},
                ],
            }
        ]
    }
    assert _len(row) == 3


def test_nested_list_without_text_counts_nothing():
    assert _len({"tags": [["a", "b"]]}) == 0


def test_numeric_list_counts_no_text():
    assert _len({"text": "hello", "embedding": [0.1, 0.2]}) == 5


def test_string_list_measures_its_strings():
    assert _len({"completions": ["ab", "cde"]}) == 6  # 2 + 3 + 1 separator


# --- Both callers ------------------------------------------------------------


def test_validate_and_stats_uses_text_length_and_counts_short_chat_rows():
    data = [
        {"messages": [{"role": "user", "content": "hi"}]},
        {"messages": [{"role": "user", "content": LONG_QUESTION}]},
        {"messages": [{"role": "user", "content": "short"}]},
    ]
    stats = validate_and_stats(data)
    assert stats["min_length"] == 2
    assert stats["max_length"] == 71
    assert stats["avg_length"] == 26
    assert "2 samples are very short (<10 chars)" in stats["issues"]


def test_validate_and_stats_mixed_flat_and_chat_rows():
    data = [
        {"instruction": "abcd", "output": "ef"},
        {"messages": [{"role": "user", "content": "hi"}]},
    ]
    stats = validate_and_stats(data)
    assert stats["min_length"] == 2
    assert stats["max_length"] == 7
    assert stats["avg_length"] == 4


def test_extended_stats_percentiles_and_tokens_follow_text_length():
    data = [
        {"messages": [{"role": "user", "content": "hi"}]},
        {"messages": [{"role": "user", "content": LONG_QUESTION}]},
        {"messages": [{"role": "user", "content": "What is the weather?"}]},
        {"messages": [{"role": "user", "content": "x" * 40}]},
    ]
    stats = extended_stats(data)
    assert stats["lengths"] == [2, 71, 20, 40]
    assert stats["length_p10"] == 2
    assert stats["length_p25"] == 20
    assert stats["length_p50"] == 40
    assert stats["length_p75"] == 71
    assert stats["length_p90"] == 71
    assert stats["token_counts"] == [1, 17, 5, 10]
    assert stats["min_tokens"] == 1
    assert stats["max_tokens"] == 17


# --- The command ---------------------------------------------------------------


def test_inspect_reports_text_lengths_for_tiny_chat_file(tmp_path: Path):
    path = tmp_path / "tiny.jsonl"
    rows = [
        {"messages": [{"role": "user", "content": "hi"}]},
        {"messages": [{"role": "user", "content": LONG_QUESTION}]},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    result = runner.invoke(app, ["data", "inspect", str(path)])

    assert result.exit_code == 0, result.output
    out = strip_ansi(result.output)
    assert re.search(r"Avg length \(chars\)\s*│\s*36\s*│", out)
    assert re.search(r"Min length\s*│\s*2\s*│", out)
    assert re.search(r"Max length\s*│\s*71\s*│", out)
