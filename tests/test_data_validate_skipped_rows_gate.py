"""#1887 — ``soup data validate`` counts the rows the loader skipped as invalid rows.

#1873 made the loader skip every row that is not a JSON object (and it already
skipped lines that are not valid JSON). ``validate`` listed those skips but its
numbers never saw them: ``total`` and ``valid`` came from the retained objects
only, so a file of junk rows exited 0 at "0/0 rows valid" and ``[good, 42]``
passed ``--min-valid-fraction 1.0`` at "1/1 rows valid". The rule now: a skipped
row is an invalid row for the ``N/M rows valid`` line, for ``--min-valid-fraction``
and for the exit code. The listing and the retained-object statistics are unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.commands.data import app
from tests.conftest import strip_ansi

GOOD = {"instruction": "A sufficiently long question", "output": "A useful answer"}
GOOD2 = {**GOOD, "instruction": "Another sufficiently long question"}
JUNK = [42, "abc", [1, 2], None]

pytestmark = pytest.mark.parametrize("extension", ["jsonl", "json"])


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


def _write(tmp_path: Path, extension: str, rows: list, *, raw_lines: list[str] | None = None):
    path = tmp_path / f"data.{extension}"
    if extension == "jsonl":
        lines = [json.dumps(row) for row in rows] + list(raw_lines or [])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        assert raw_lines is None, "a .json array cannot hold an invalid line"
        path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def _validate(path: Path, *extra: str):
    result = CliRunner().invoke(app, ["validate", str(path), *extra])
    return result.exit_code, _plain(result.output)


# ---------------------------------------------------------------------------
# The issue's reproductions
# ---------------------------------------------------------------------------


def test_a_file_of_only_junk_rows_fails_the_gate(tmp_path, extension):
    path = _write(tmp_path, extension, JUNK)
    code, out = _validate(path, "--format", "alpaca")
    assert code == 2, out
    assert "0/4 rows valid" in out
    assert "no usable rows remain" in out
    assert "Dataset is valid!" not in out


def test_good_plus_junk_fails_a_minimum_of_one(tmp_path, extension):
    path = _write(tmp_path, extension, [GOOD, 42])
    code, out = _validate(path, "--format", "alpaca", "--min-valid-fraction", "1.0")
    assert code == 2, out
    assert "1/2 rows valid" in out
    assert "valid fraction 0.500 is below --min-valid-fraction 1.000" in out


def test_the_minimum_is_strict_so_half_passes_at_half(tmp_path, extension):
    path = _write(tmp_path, extension, [GOOD, 42])
    code, out = _validate(path, "--format", "alpaca", "--min-valid-fraction", "0.5")
    assert code == 0, out
    assert "1/2 rows valid" in out


# ---------------------------------------------------------------------------
# Controls: what must keep passing
# ---------------------------------------------------------------------------


def test_control_a_clean_file_passes_a_minimum_of_one(tmp_path, extension):
    path = _write(tmp_path, extension, [GOOD, GOOD2])
    code, out = _validate(path, "--format", "alpaca", "--min-valid-fraction", "1.0")
    assert code == 0, out
    assert "2/2 rows valid" in out
    assert "Dataset is valid!" in out


def test_a_partial_drop_still_passes_without_a_minimum(tmp_path, extension):
    path = _write(tmp_path, extension, [GOOD, 42])
    code, out = _validate(path, "--format", "alpaca")
    assert code == 0, out
    assert "1/2 rows valid" in out


def test_the_listing_cap_and_the_count_agree(tmp_path, extension):
    path = _write(tmp_path, extension, [GOOD] + [42] * 6)
    code, out = _validate(path, "--format", "alpaca")
    assert code == 0, out
    assert "1/7 rows valid" in out
    assert "... and 3 more" in out


# ---------------------------------------------------------------------------
# The "0 rows" cases keep their codes
# ---------------------------------------------------------------------------


def test_a_zero_byte_file_is_still_zero_of_zero(tmp_path, extension):
    path = tmp_path / f"empty.{extension}"
    path.write_text("" if extension == "jsonl" else "[]", encoding="utf-8")
    code, out = _validate(path, "--format", "alpaca")
    assert code == 0, out
    assert "0/0 rows valid" in out


def test_an_all_junk_file_without_a_format_is_still_a_usage_error(tmp_path, extension):
    path = _write(tmp_path, extension, JUNK)
    code, out = _validate(path)
    assert code == 3, out
    # The refusal says what happened rather than calling a 4-row file empty.
    assert "no object rows remain" in out
    assert "4 row(s) skipped" in out
    assert "--format" in out


def test_the_all_junk_issue_line_names_the_skips_not_an_empty_dataset(tmp_path, extension):
    path = _write(tmp_path, extension, JUNK)
    code, out = _validate(path, "--format", "alpaca")
    assert code == 2, out
    assert "no object rows remain: 4 row(s) skipped" in out
    assert "Dataset is empty" not in out


# ---------------------------------------------------------------------------
# Lines that are not JSON at all count the same way (.jsonl only)
# ---------------------------------------------------------------------------


def test_an_invalid_json_line_is_an_invalid_row(tmp_path, extension):
    if extension == "json":
        pytest.skip("a .json array is parsed whole; an invalid file is a load error")
    path = _write(tmp_path, extension, [GOOD], raw_lines=["{not json"])
    code, out = _validate(path, "--format", "alpaca", "--min-valid-fraction", "1.0")
    assert code == 2, out
    assert "1/2 rows valid" in out
    assert "invalid JSON on line 2" in out
    assert "Dataset is valid!" not in out


def test_an_invalid_json_line_is_listed_once(tmp_path, extension):
    if extension == "json":
        pytest.skip("a .json array is parsed whole; an invalid file is a load error")
    path = _write(tmp_path, extension, [GOOD], raw_lines=["{not json"])
    code, out = _validate(path, "--format", "alpaca")
    assert code == 0, out
    listing = out.split("Issues found:", 1)[1]
    assert listing.count("invalid JSON on line 2") == 1
