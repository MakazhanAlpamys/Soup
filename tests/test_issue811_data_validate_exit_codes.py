"""Regression tests for ``soup data validate`` exit codes (#811)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import Result
from typer.testing import CliRunner

from soup_cli.cli import app

from .conftest import strip_ansi


def _write_alpaca(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _invoke_validate(path: Path, *extra_args: str) -> Result:
    return CliRunner().invoke(
        app,
        ["data", "validate", str(path), "--format", "alpaca", *extra_args],
    )


def test_validate_exits_two_when_all_rows_are_unusable(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    _write_alpaca(path, [{"instruction": None, "output": None}] * 5)

    result = _invoke_validate(path)
    output = strip_ansi(result.output)

    assert result.exit_code == 2
    assert "0/5 rows valid" in output
    assert "no usable rows remain" in output


def test_validate_allows_partially_valid_data_by_default(tmp_path: Path) -> None:
    path = tmp_path / "partial.jsonl"
    _write_alpaca(
        path,
        [
            {"instruction": "Summarize this", "output": "A summary"},
            {"instruction": None, "output": None},
        ],
    )

    result = _invoke_validate(path)

    assert result.exit_code == 0, result.output
    assert "1/2 rows valid" in strip_ansi(result.output)


@pytest.mark.parametrize(
    ("minimum", "expected_exit_code"),
    [("0.75", 2), ("0.5", 0)],
)
def test_validate_applies_minimum_fraction(
    tmp_path: Path,
    minimum: str,
    expected_exit_code: int,
) -> None:
    path = tmp_path / "partial.jsonl"
    _write_alpaca(
        path,
        [
            {"instruction": "Summarize this", "output": "A summary"},
            {"instruction": None, "output": None},
        ],
    )

    result = _invoke_validate(path, "--min-valid-fraction", minimum)

    assert result.exit_code == expected_exit_code, result.output
    if expected_exit_code == 2:
        assert "valid fraction 0.500 is below --min-valid-fraction 0.750" in strip_ansi(
            result.output
        )


def test_validate_keeps_input_errors_at_exit_one(tmp_path: Path) -> None:
    missing = _invoke_validate(tmp_path / "missing.jsonl")

    assert missing.exit_code == 1
    assert "File not found" in strip_ansi(missing.output)

    unknown = tmp_path / "unknown.jsonl"
    unknown.write_text('{"unrecognized": "shape"}\n', encoding="utf-8")
    undetectable = CliRunner().invoke(app, ["data", "validate", str(unknown)])

    assert undetectable.exit_code == 1
    assert "Cannot detect format" in strip_ansi(undetectable.output)
