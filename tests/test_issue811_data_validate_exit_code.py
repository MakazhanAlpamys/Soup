"""Regression tests for issue #811: soup data validate exit code contract.

Validates that:
1. When total > 0 and valid_rows == 0, soup data validate exits 2.
2. Partial drops (e.g. 1/2 valid) default to exit 0.
3. --strict exits 2 on any dropped row, and exits 0 when 100% of rows are valid.
4. --min-valid-fraction exits 2 when valid fraction is below threshold, and 0 at or above.
5. Out-of-range --min-valid-fraction values exit 1.
6. Missing files and undetectable formats maintain exit 1.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi


def test_zero_valid_rows_exits_2(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    rows = [{"instruction": None, "output": None} for _ in range(5)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    result = CliRunner().invoke(app, ["data", "validate", str(path), "--format", "alpaca"])
    clean_out = strip_ansi(result.output)

    assert result.exit_code == 2, result.output
    assert "0/5 rows valid for alpaca format" in clean_out


def test_partial_valid_rows_defaults_to_exit_0(tmp_path: Path) -> None:
    path = tmp_path / "partial.jsonl"
    rows = [
        {"instruction": "say hi", "output": "hello"},
        {"instruction": None, "output": None},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    result = CliRunner().invoke(app, ["data", "validate", str(path), "--format", "alpaca"])
    clean_out = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "1/2 rows valid for alpaca format" in clean_out


def test_strict_mode_exits_2_on_dropped_rows(tmp_path: Path) -> None:
    path = tmp_path / "partial.jsonl"
    rows = [
        {"instruction": "say hi", "output": "hello"},
        {"instruction": None, "output": None},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    result = CliRunner().invoke(
        app, ["data", "validate", str(path), "--format", "alpaca", "--strict"]
    )
    clean_out = strip_ansi(result.output)

    assert result.exit_code == 2, result.output
    assert "Strict validation failed" in clean_out


def test_strict_mode_exits_0_when_all_rows_valid(tmp_path: Path) -> None:
    path = tmp_path / "all_good.jsonl"
    rows = [
        {"instruction": "say hi", "output": "hello"},
        {"instruction": "say bye", "output": "goodbye"},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    result = CliRunner().invoke(
        app, ["data", "validate", str(path), "--format", "alpaca", "--strict"]
    )
    clean_out = strip_ansi(result.output)

    assert result.exit_code == 0, result.output
    assert "2/2 rows valid for alpaca format" in clean_out


def test_min_valid_fraction_enforces_threshold(tmp_path: Path) -> None:
    path = tmp_path / "partial.jsonl"
    # 2 out of 5 valid = 40%
    rows = [
        {"instruction": "say hi", "output": "hello"},
        {"instruction": "say bye", "output": "goodbye"},
        {"instruction": None, "output": None},
        {"instruction": None, "output": None},
        {"instruction": None, "output": None},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    # 40% < 50% -> exit 2
    res_fail = CliRunner().invoke(
        app,
        ["data", "validate", str(path), "--format", "alpaca", "--min-valid-fraction", "0.5"],
    )
    assert res_fail.exit_code == 2, res_fail.output
    assert "Validation threshold failed" in strip_ansi(res_fail.output)

    # 40% >= 40% -> exit 0
    res_pass = CliRunner().invoke(
        app,
        ["data", "validate", str(path), "--format", "alpaca", "--min-valid-fraction", "0.4"],
    )
    assert res_pass.exit_code == 0, res_pass.output
    assert "2/5 rows valid for alpaca format" in strip_ansi(res_pass.output)


def test_min_valid_fraction_out_of_range_exits_1(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    rows = [{"instruction": "say hi", "output": "hello"}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    res1 = CliRunner().invoke(
        app,
        ["data", "validate", str(path), "--format", "alpaca", "--min-valid-fraction", "1.5"],
    )
    assert res1.exit_code == 1, res1.output

    res2 = CliRunner().invoke(
        app,
        ["data", "validate", str(path), "--format", "alpaca", "--min-valid-fraction", "-0.1"],
    )
    assert res2.exit_code == 1, res2.output


def test_missing_file_and_undetectable_format_maintain_exit_1(tmp_path: Path) -> None:
    # Missing file
    res_missing = CliRunner().invoke(app, ["data", "validate", str(tmp_path / "nonexistent.jsonl")])
    assert res_missing.exit_code == 1, res_missing.output
    assert "File not found" in strip_ansi(res_missing.output)

    # Undetectable format
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("", encoding="utf-8")
    res_undetectable = CliRunner().invoke(app, ["data", "validate", str(empty_file)])
    assert res_undetectable.exit_code == 1, res_undetectable.output
