"""The data-validation preview emits one receipt and preserves existing exits."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from csv import Error as CsvError
from pathlib import Path
from types import SimpleNamespace

import pytest

from soup_cli.commands import data as data_command
from soup_cli.data.formats import VALID_FORMATS

from .conftest import make_cli_runner, strip_ansi

GOOD_ROW = '{"instruction":"Summarize the record","output":"A useful summary."}'
PRIVATE_ROW = "secret-content-ml-test"
REPO = Path(__file__).resolve().parents[1]

_CHAT_TURNS = [
    {"role": "user", "content": "Summarize the record"},
    {"role": "assistant", "content": "A useful summary."},
]
_SHAREGPT_TURNS = [
    {"from": "human", "value": "Summarize the record"},
    {"from": "gpt", "value": "A useful summary."},
]
_FORMAT_ROWS = {
    "alpaca": json.loads(GOOD_ROW),
    "asr": {"audio": "sample.wav", "text": "A useful summary."},
    "audio": {"audio": "sample.wav", "messages": _CHAT_TURNS},
    "chatml": {"messages": _CHAT_TURNS},
    "cross_encoder": {
        "text_a": "A useful summary.",
        "text_b": "A concise summary.",
        "label": 1,
    },
    "dpo": {"prompt": "Summarize the record", "chosen": "A useful summary.", "rejected": "No."},
    "embedding": {"anchor": "A useful summary.", "positive": "A concise summary."},
    "input_output": {"segments": [{"text": "A useful summary.", "label": True}]},
    "kto": {"prompt": "Summarize the record", "completion": "A useful summary.", "label": True},
    "llava": {"image": "sample.png", "conversations": _SHAREGPT_TURNS},
    "multimodal": {"messages": _CHAT_TURNS},
    "plaintext": {"text": "A useful summary."},
    "pre_tokenized": {"input_ids": [1, 2], "labels": [1, 2]},
    "prm": {
        "prompt": "Summarize the record",
        "completions": ["A useful summary."],
        "labels": [True],
    },
    "raft": {
        "query": "Summarize the record",
        "golden_doc": "A short record to summarize.",
        "answer": "A useful summary.",
    },
    "sharegpt": {"conversations": _SHAREGPT_TURNS},
    "sharegpt4v": {"image": "sample.png", "conversations": _SHAREGPT_TURNS},
    "tool-calling": {
        "messages": [
            {"role": "user", "content": "Find the record"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_record",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_record", "content": "The record."},
        ],
        "tools": [
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}},
            },
        ],
    },
    "video": {"video": "sample.mp4", "messages": _CHAT_TURNS},
}


def _invoke(path, *args):
    return make_cli_runner().invoke(data_command.app, ["validate", str(path), *args])


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


def _check_receipt(code, stdout, *, status):
    assert stdout.count(b"\n") == 1 and stdout.endswith(b"\n"), stdout
    receipt = json.loads(stdout.decode("utf-8"))
    assert receipt["schema"] == "soup.command_result"
    assert receipt["schema_version"] == 1
    assert receipt["command"] == "data.validate"
    assert receipt["phase"] == "validate"
    assert receipt["status"] == status
    assert receipt["exit_code"] == code
    assert receipt["soup_version"]
    assert receipt["started_at"].endswith(("Z", "+00:00"))
    assert receipt["duration_s"] >= 0
    assert receipt["run_id"] is None
    assert receipt["child"] is None
    assert receipt["artifacts"] == []
    assert receipt["truncated"] == {}
    assert PRIVATE_ROW.encode() not in stdout
    return receipt


@pytest.mark.parametrize(
    ("fmt", "args"),
    [*((fmt, ("--format", fmt)) for fmt in VALID_FORMATS), ("tool-calling", ())],
    ids=[*VALID_FORMATS, "tool-calling-auto"],
)
def test_preview_accepts_each_supported_format(tmp_path, fmt, args):
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(_FORMAT_ROWS[fmt]) + "\n", encoding="utf-8")
    result = _invoke(path, *args, "--result-json")

    assert result.exit_code == 0, result.output
    receipt = _check_receipt(0, result.stdout_bytes, status="ok")
    assert receipt["decision"] == "valid"
    assert receipt["reasons"] == []
    assert receipt["data"] == {
        "format": fmt,
        "total": 1,
        "valid": 1,
        "invalid": 0,
        "valid_fraction": 1.0,
    }
    stderr = _plain(result.stderr)
    assert f"1/1 rows valid for {fmt} format" in stderr
    if not args:
        assert f"Auto-detected format: {fmt}" in stderr


@pytest.mark.parametrize(
    ("minimum", "code", "status", "decision"),
    [("0.5", 0, "ok", "valid"), ("0.75", 2, "blocked", "invalid")],
)
def test_preview_uses_the_legacy_denominator_and_exit(tmp_path, minimum, code, status, decision):
    path = tmp_path / "private-name.jsonl"
    path.write_text(f"{GOOD_ROW}\n{json.dumps(PRIVATE_ROW)}\n", encoding="utf-8")
    args = ("--format", "alpaca", "--min-valid-fraction", minimum)
    legacy = _invoke(path, *args)
    result = _invoke(path, *args, "--result-json")

    assert legacy.exit_code == result.exit_code == code, result.output
    assert "1/2 rows valid" in _plain(legacy.stdout)
    assert legacy.stderr == ""
    receipt = _check_receipt(code, result.stdout_bytes, status=status)
    stderr = _plain(result.stderr)
    assert "1/2 rows valid" in stderr
    assert "Warning: line 2 is a JSON string, not an object" in stderr
    assert path.name not in result.stdout
    assert receipt["decision"] == decision
    assert receipt["data"] == {
        "format": "alpaca",
        "total": 2,
        "valid": 1,
        "invalid": 1,
        "valid_fraction": 0.5,
    }
    assert receipt["warnings"][0]["code"] == "soup.data.rows_skipped"
    assert receipt["warnings"][0]["details"] == {"skipped": 1, "total": 2}
    if code == 2:
        reason, = receipt["reasons"]
        assert reason["code"] == "soup.data.valid_fraction_below_minimum"
        assert reason["details"] == {"actual": 0.5, "minimum": 0.75}
    else:
        assert receipt["reasons"] == []


def test_preview_reports_validator_warnings_for_retained_rows(tmp_path):
    path = tmp_path / "short.jsonl"
    path.write_text('{"instruction":"a","output":"b"}\n', encoding="utf-8")
    result = _invoke(path, "--format", "alpaca", "--result-json")

    assert result.exit_code == 0, result.output
    receipt = _check_receipt(0, result.stdout_bytes, status="ok")
    assert receipt["data"]["valid"] == 1
    assert [warning["code"] for warning in receipt["warnings"]] == [
        "soup.data.validation_warnings",
    ]
    assert "1 samples are very short" in _plain(result.stderr)


def test_preview_does_not_add_an_empty_dataset_warning_when_all_rows_were_skipped(tmp_path):
    path = tmp_path / "skipped.jsonl"
    path.write_text("42\n", encoding="utf-8")
    result = _invoke(path, "--format", "alpaca", "--result-json")

    assert result.exit_code == 2, result.output
    receipt = _check_receipt(2, result.stdout_bytes, status="blocked")
    assert [warning["code"] for warning in receipt["warnings"]] == ["soup.data.rows_skipped"]
    assert receipt["warnings"][0]["details"] == {"skipped": 1, "total": 1}
    assert "no object rows remain" in _plain(result.stderr)


@pytest.mark.parametrize(
    ("name", "content", "fmt", "code", "status", "reason", "diagnostic"),
    [
        ("missing.jsonl", None, "alpaca", 3, "refused", "soup.data.source_missing",
         "File not found"),
        ("present.jsonl", GOOD_ROW, PRIVATE_ROW, 3, "refused", "soup.data.format_unsupported",
         "Unknown --format"),
        ("unknown.jsonl", '{"unrecognized":"value"}', "auto", 3, "refused",
         "soup.data.format_undetected", "Cannot detect format"),
        ("broken.json", "{", "alpaca", 1, "failed", "soup.data.load_failed",
         "Could not load dataset"),
        ("invalid.jsonl", '{"instruction":null,"output":null}', "alpaca", 2, "blocked",
         "soup.data.no_usable_rows", "Validation failed: no usable rows remain."),
    ],
)
def test_preview_has_a_safe_reason_for_each_handled_error(
    tmp_path, name, content, fmt, code, status, reason, diagnostic,
):
    path = tmp_path / name
    if content is not None:
        path.write_text(content + "\n", encoding="utf-8")
    result = _invoke(path, "--format", fmt, "--result-json")

    assert result.exit_code == code, result.output
    payload = _check_receipt(code, result.stdout_bytes, status=status)
    assert [item["code"] for item in payload["reasons"]] == [reason]
    assert payload["decision"] == ("invalid" if code == 2 else None)
    if code != 2:
        assert payload["data"] is None
    assert diagnostic in _plain(result.stderr)


@pytest.mark.parametrize("suffix", ["jsonl", "txt"])
def test_explicit_empty_format_has_null_fraction_and_keeps_exit_zero(tmp_path, suffix):
    path = tmp_path / f"[red]{PRIVATE_ROW}.{suffix}"
    path.touch()
    fmt = "plaintext" if suffix == "txt" else "alpaca"
    result = _invoke(path, "--format", fmt, "--result-json", "--min-valid-fraction", "1.0")

    assert result.exit_code == 0, result.output
    data = _check_receipt(0, result.stdout_bytes, status="ok")["data"]
    assert data["format"] == fmt
    assert data["total"] == data["valid"] == data["invalid"] == 0
    assert data["valid_fraction"] is None
    stderr = _plain(result.stderr)
    assert "Dataset is empty" in stderr
    assert PRIVATE_ROW not in stderr


def test_inaccessible_path_receipt_keeps_backend_text_private(tmp_path, monkeypatch):
    path = tmp_path / "inaccessible.jsonl"
    original_exists = Path.exists

    def check_exists(self):
        if self == path:
            raise PermissionError(f"{path}: {PRIVATE_ROW}")
        return original_exists(self)

    monkeypatch.setattr(Path, "exists", check_exists)
    result = _invoke(path, "--result-json")

    assert result.exit_code == 1, result.output
    payload = _check_receipt(1, result.stdout_bytes, status="failed")
    assert [reason["code"] for reason in payload["reasons"]] == ["soup.data.load_failed"]
    stderr = _plain(result.stderr)
    assert "Could not load dataset" in stderr
    assert str(path) not in stderr
    assert PRIVATE_ROW not in stderr


@pytest.mark.parametrize(
    ("suffix", "error", "diagnostic"),
    [
        ("parquet", ImportError, "pip install pyarrow"),
        ("csv", CsvError, "Could not load dataset"),
    ],
)
def test_load_failures_keep_actionable_diagnostics_without_backend_text(
    tmp_path, monkeypatch, suffix, error, diagnostic,
):
    path = tmp_path / f"private-name.{suffix}"
    path.touch()

    def fail(*args, **kwargs):
        raise error(f"{path}: {PRIVATE_ROW}")

    monkeypatch.setattr(data_command, "load_raw_data", fail)
    result = _invoke(path, "--format", "alpaca", "--result-json")

    assert result.exit_code == 1, result.output
    payload = _check_receipt(1, result.stdout_bytes, status="failed")
    assert [reason["code"] for reason in payload["reasons"]] == ["soup.data.load_failed"]
    assert payload["data"] is None
    stderr = _plain(result.stderr)
    assert diagnostic in stderr
    assert path.name not in result.stdout and path.name not in stderr
    assert PRIVATE_ROW not in stderr


@pytest.mark.parametrize(
    ("target", "error", "code", "status", "reason", "diagnostic"),
    [
        ("load_raw_data", RecursionError, 1, "failed", "soup.cli.execution_failed",
         "Dataset validation failed unexpectedly (RecursionError)."),
        ("validate_and_stats", RuntimeError, 1, "failed", "soup.cli.execution_failed",
         "Dataset validation failed unexpectedly (RuntimeError)."),
        ("validate_and_stats", KeyboardInterrupt, 130, "cancelled", "soup.cli.interrupted",
         "Interrupted."),
    ],
)
def test_preview_handles_unexpected_errors_and_interrupts(
    tmp_path, monkeypatch, target, error, code, status, reason, diagnostic,
):
    path = tmp_path / "private-name.jsonl"
    path.write_text(GOOD_ROW + "\n", encoding="utf-8")

    def fail(*args, **kwargs):
        raise error(f"{path}: {PRIVATE_ROW}")

    monkeypatch.setattr(data_command, target, fail)
    result = _invoke(path, "--format", "alpaca", "--result-json")

    assert result.exit_code == code, result.output
    payload = _check_receipt(code, result.stdout_bytes, status=status)
    assert [item["code"] for item in payload["reasons"]] == [reason]
    if status == "failed":
        assert payload["reasons"][0]["message"] == "The command could not complete."
    assert payload["data"] is None and payload["decision"] is None
    stderr = _plain(result.stderr)
    assert diagnostic in stderr
    assert path.name not in result.stdout and path.name not in stderr
    assert PRIVATE_ROW not in stderr


@pytest.mark.parametrize("reason", ["soup.data.source_missing", "soup.data.format_unsupported"])
def test_preview_does_not_retry_a_failed_stdout_write(tmp_path, monkeypatch, reason):
    path = tmp_path / "rows.jsonl"
    fmt = "alpaca"
    if reason == "soup.data.format_unsupported":
        path.write_text(GOOD_ROW + "\n", encoding="utf-8")
        fmt = "unknown"
    attempted = []

    def fail_write(payload):
        attempted.append(payload)
        raise BrokenPipeError("fixture rejected the receipt")

    buffer = SimpleNamespace(write=fail_write)
    monkeypatch.setattr(data_command, "sys", SimpleNamespace(stdout=SimpleNamespace(buffer=buffer)))
    result = _invoke(path, "--format", fmt, "--result-json")

    assert isinstance(result.exception, BrokenPipeError)
    assert result.stdout_bytes == b""
    (first_attempt,) = attempted
    payload = _check_receipt(3, first_attempt, status="refused")
    assert [item["code"] for item in payload["reasons"]] == [reason]


def test_parse_failure_keeps_cli_usage_exit_and_does_not_emit_receipt(tmp_path):
    result = _invoke(
        tmp_path / "data.jsonl", "--result-json", "--min-valid-fraction", "nan",
    )
    assert result.exit_code == 3
    assert result.stdout_bytes == b""
    assert "min-valid-fraction" in _plain(result.stderr)


def test_main_entry_point_separates_utf8_stdout_from_loader_warnings(tmp_path):
    path = tmp_path / "donn\u00e9es.jsonl"
    path.write_text(f"{GOOD_ROW}\n{PRIVATE_ROW}\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "soup_cli", "data", "validate", str(path), "--result-json"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(REPO / "src"), "FORCE_COLOR": "1"},
        capture_output=True,
        check=False,
        timeout=45,
    )

    assert result.returncode == 0, (result.stdout, result.stderr)
    receipt = _check_receipt(0, result.stdout, status="ok")
    assert receipt["data"]["valid"] == 1
    assert receipt["data"]["total"] == 2
    assert "Warning: invalid JSON on line 2" in _plain(result.stderr.decode("utf-8"))
