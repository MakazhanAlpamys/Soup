"""Unit tests for Feature 3: Automated Dataset Cleaning & Sanity Repair Pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.data_clean import (
    CleanReport,
    clean_dataset,
    clean_row,
    is_echo_turn,
    repair_code_fences,
    repair_json_string,
    sanitize_text,
    strip_boilerplate,
)

runner = CliRunner()


def _hash_file(path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def _create_sample_dirty_file(directory: Path) -> Path:
    """Create a temporary dirty JSONL file for CLI testing."""
    file_path = directory / "dirty_data.jsonl"
    data = [
        {
            "messages": [
                {"role": "user", "content": "How to add in Python?\u200b"},
                {
                    "role": "assistant",
                    "content": (
                        "Certainly! As an AI language model, here is the function:\n\n"
                        "```python\ndef add(a, b):\n    return a + b\n"
                    ),
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a programming language."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Empty question"},
                {"role": "assistant", "content": "   \n\u200b   "},
            ]
        },
    ]
    with open(file_path, "w", encoding="utf-8") as file_handle:
        for row in data:
            file_handle.write(json.dumps(row) + "\n")
    return file_path


def test_sanitize_text_control_characters():
    """Verify zero-width spaces, C0 controls, and CRLF are properly cleaned."""
    dirty_text = "Hello\u200b world\x00\x07!\r\nSecond line."
    cleaned, modified = sanitize_text(dirty_text)
    assert modified is True
    assert cleaned == "Hello world!\nSecond line."


def test_sanitize_text_clean_noop():
    """Verify clean text returns modified=False."""
    clean_text = "Standard clean text with\nnewlines and tabs\there."
    cleaned, modified = sanitize_text(clean_text)
    assert modified is False
    assert cleaned == clean_text


def test_repair_code_fences_unclosed():
    """Verify unclosed markdown code blocks are safely closed."""
    unclosed = "Here is the code:\n```python\ndef add(a, b):\n    return a + b\n"
    cleaned, modified = repair_code_fences(unclosed)
    assert modified is True
    assert cleaned.endswith("\n```")
    assert cleaned.count("```") == 2


def test_repair_code_fences_already_balanced():
    """Verify balanced code blocks are left intact."""
    balanced = "Here is the code:\n```python\ndef add(a, b):\n    return a + b\n```"
    cleaned, modified = repair_code_fences(balanced)
    assert modified is False
    assert cleaned == balanced


def test_strip_boilerplate_prefixes():
    """Verify common LLM preamble is stripped."""
    text = "Certainly! As an AI language model, I can explain that. Quantum computing uses qubits."
    cleaned, modified = strip_boilerplate(text)
    assert modified is True
    assert cleaned == "Quantum computing uses qubits."


def test_strip_boilerplate_suffixes():
    """Verify common LLM sign-offs are stripped."""
    text = "Here is the answer. I hope this helps! Please let me know if you have any questions."
    cleaned, modified = strip_boilerplate(text)
    assert modified is True
    assert cleaned == "Here is the answer."


def test_repair_json_string_trailing_comma():
    """Verify trailing commas in JSON are repaired."""
    invalid_json = '{"name": "test", "value": 42,}'
    repaired, modified = repair_json_string(invalid_json)
    assert modified is True
    parsed = json.loads(repaired)
    assert parsed["name"] == "test"
    assert parsed["value"] == 42


def test_repair_json_string_markdown_wrapped():
    """Verify markdown-wrapped JSON is extracted cleanly."""
    wrapped_json = '```json\n{"action": "search", "query": "orders"}\n```'
    repaired, modified = repair_json_string(wrapped_json)
    assert modified is True
    parsed = json.loads(repaired)
    assert parsed["action"] == "search"
    assert parsed["query"] == "orders"


def test_is_echo_turn():
    """Verify echo turns where assistant repeats prompt verbatim are flagged."""
    assert is_echo_turn("What is your return policy?", "what is your return policy?") is True
    assert is_echo_turn("What is your return policy?", "Our return policy is 30 days.") is False


def test_clean_row_chatml_with_heuristics():
    """Test cleaning a full ChatML row when opt-in heuristics are enabled."""
    row = {
        "messages": [
            {"role": "user", "content": "How do I add in Python?\u200b"},
            {
                "role": "assistant",
                "content": (
                    "Sure! As an AI language model, here is the function:\n\n"
                    "```python\ndef add(a, b):\n    return a + b\n"
                ),
            },
        ]
    }
    cleaned, rules = clean_row(
        row,
        "chatml",
        strip_ai_boilerplate=True,
        repair_code=True,
    )
    assert cleaned is not None
    assert "Invisible & Control Chars" in rules
    assert "Boilerplate Disclaimers" in rules
    assert "Markdown Code Fence Repair" in rules

    assistant_msg = cleaned["messages"][1]["content"]
    assert "As an AI" not in assistant_msg
    assert assistant_msg.endswith("```")


def test_clean_row_alpaca():
    """Test cleaning an Alpaca row with opt-in boilerplate stripping."""
    row = {
        "instruction": "Explain gravity\u200b",
        "input": "",
        "output": "Certainly! Gravity is a fundamental force. I hope this helps!",
    }
    cleaned, rules = clean_row(row, "alpaca", strip_ai_boilerplate=True)
    assert cleaned is not None
    assert "Boilerplate Disclaimers" in rules
    assert cleaned["output"] == "Gravity is a fundamental force."


def test_clean_row_sharegpt():
    """Test cleaning a ShareGPT row with opt-in boilerplate stripping."""
    row = {
        "conversations": [
            {"from": "human", "value": "Hello"},
            {"from": "gpt", "value": "Sure! How can I help you today?"},
        ]
    }
    cleaned, rules = clean_row(row, "sharegpt", strip_ai_boilerplate=True)
    assert cleaned is not None
    assert cleaned["conversations"][1]["value"] == "How can I help you today?"


def test_clean_row_tool_calling():
    """Test repairing JSON in tool_calls arguments with opt-in repair_json."""
    row = {
        "messages": [{"role": "user", "content": "Fetch order 123"}],
        "tool_calls": [
            {"name": "get_order", "arguments": '{"order_id": 123,}'}
        ],
    }
    cleaned, rules = clean_row(row, "tool-calling", repair_json=True)
    assert cleaned is not None
    assert "Malformed JSON in Tools" in rules
    assert cleaned["tool_calls"][0]["arguments"] == '{"order_id": 123}'


def test_clean_row_empty_dropped():
    """Test that rows with empty assistant turns are dropped by default."""
    row = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "   \n\u200b   "},
        ]
    }
    cleaned, rules = clean_row(row, "chatml", min_tokens=1)
    assert cleaned is None
    assert "Empty / Whitespace Turns" in rules


def test_echo_preserved_by_default_and_pruned_with_opt_in():
    """Test that echo rows are preserved by default, and pruned only with prune_echo=True."""
    row = {
        "messages": [
            {"role": "user", "content": "Repeat after me"},
            {"role": "assistant", "content": "Repeat after me"},
        ]
    }
    # Default behavior: preserved as clean
    cleaned_default, rules_default = clean_row(row, "chatml", prune_echo=False)
    assert cleaned_default is not None
    assert len(rules_default) == 0
    assert cleaned_default["messages"][1]["content"] == "Repeat after me"

    # Opt-in behavior: dropped
    cleaned_opt_in, rules_opt_in = clean_row(row, "chatml", prune_echo=True)
    assert cleaned_opt_in is None
    assert "Target Leakage / Echo" in rules_opt_in


def test_byte_identity_clean_rows_and_arithmetic_closure():
    """Verify byte-identity for clean rows and arithmetic closure of report accounting."""
    clean_row_data = {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 is 4."},
        ]
    }
    single_zwsp_row = {
        "messages": [
            {"role": "user", "content": "Hello\u200b world"},
            {"role": "assistant", "content": "Hello there!"},
        ]
    }
    empty_row = {
        "messages": [
            {"role": "user", "content": "Silent question"},
            {"role": "assistant", "content": ""},
        ]
    }
    echo_row = {
        "messages": [
            {"role": "user", "content": "Echo test"},
            {"role": "assistant", "content": "Echo test"},
        ]
    }

    corpus = [clean_row_data, single_zwsp_row, empty_row, echo_row]

    # Run with default flags (all heuristics off, prune_echo off)
    cleaned_data, report = clean_dataset(corpus, "chatml")

    assert isinstance(report, CleanReport)
    assert report.total_scanned == 4
    assert report.total_clean == 2  # clean_row_data and echo_row
    assert report.total_modified == 1  # single_zwsp_row
    assert report.total_dropped == 1  # empty_row

    # Exact arithmetic closure check
    assert report.total_scanned == report.total_clean + report.total_modified + report.total_dropped

    # Clean rows must be 100% byte-for-byte identical to input
    assert json.dumps(cleaned_data[0]) == json.dumps(clean_row_data)
    assert json.dumps(cleaned_data[1]) == json.dumps(echo_row)

    # Modified row must not be clean and must record the exact rule
    assert "Invisible & Control Chars" in report.rule_counts
    assert report.rule_counts["Invisible & Control Chars"] == 1


def test_input_file_never_modified_under_any_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verify that soup data clean NEVER modifies its input file in place under any flags."""
    monkeypatch.chdir(tmp_path)
    dirty_file = _create_sample_dirty_file(tmp_path)
    initial_hash = _hash_file(dirty_file)

    # 1. Default live run
    out1 = tmp_path / "out1.jsonl"
    res1 = runner.invoke(app, ["data", "clean", str(dirty_file), "-o", str(out1)])
    assert res1.exit_code == 0
    assert _hash_file(dirty_file) == initial_hash

    # 2. Live run with all heuristic flags active
    out2 = tmp_path / "out2.jsonl"
    res2 = runner.invoke(
        app,
        [
            "data",
            "clean",
            str(dirty_file),
            "-o",
            str(out2),
            "--strip-boilerplate",
            "--repair-code",
            "--repair-json",
            "--prune-echo",
        ],
    )
    assert res2.exit_code == 0
    assert _hash_file(dirty_file) == initial_hash

    # 3. Dry-run mode
    res3 = runner.invoke(app, ["data", "clean", str(dirty_file), "--dry-run"])
    assert res3.exit_code == 0
    assert _hash_file(dirty_file) == initial_hash


def test_dry_run_provably_write_free(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify that --dry-run creates zero files and makes zero modifications to disk."""
    monkeypatch.chdir(tmp_path)
    dirty_file = _create_sample_dirty_file(tmp_path)

    # Snapshot directory state
    dir_snapshot_before = {
        p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in tmp_path.iterdir()
    }

    result = runner.invoke(app, ["data", "clean", str(dirty_file), "--dry-run"])
    assert result.exit_code == 0

    dir_snapshot_after = {
        p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in tmp_path.iterdir()
    }

    assert dir_snapshot_before == dir_snapshot_after


def test_cli_rejects_inplace_output_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify that setting output path equal to input path is rejected."""
    monkeypatch.chdir(tmp_path)
    dirty_file = _create_sample_dirty_file(tmp_path)
    initial_hash = _hash_file(dirty_file)

    result = runner.invoke(
        app,
        ["data", "clean", str(dirty_file), "-o", str(dirty_file)],
    )
    assert result.exit_code == 1
    assert "Output path cannot be the same as input path" in result.output
    assert _hash_file(dirty_file) == initial_hash


def test_cli_clean_command_live(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test full soup data clean CLI command with default options."""
    monkeypatch.chdir(tmp_path)
    dirty_file = _create_sample_dirty_file(tmp_path)
    output_file = tmp_path / "cleaned_output.jsonl"

    result = runner.invoke(
        app,
        ["data", "clean", str(dirty_file), "-o", str(output_file)],
    )
    assert result.exit_code == 0
    assert output_file.exists()

    with open(output_file, encoding="utf-8") as file_handle:
        lines = [json.loads(line) for line in file_handle if line.strip()]

    # Out of 3 rows in sample:
    # 1: sanitized \u200b (modified)
    # 2: clean (clean)
    # 3: empty assistant (dropped)
    # -> 2 output rows
    assert len(lines) == 2


def test_cli_clean_command_json_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test soup data clean with --json flag."""
    monkeypatch.chdir(tmp_path)
    dirty_file = _create_sample_dirty_file(tmp_path)

    result = runner.invoke(
        app,
        ["data", "clean", str(dirty_file), "--dry-run", "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["total_scanned"] == 3
    assert payload["output_rows"] == 2
    assert payload["total_dropped"] == 1
