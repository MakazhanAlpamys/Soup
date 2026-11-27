"""#3376: the soup data scoring commands must not drop rows without saying so.

``soup data score``, ``decontaminate``, ``toxicity``, ``langdetect``, ``pii``
and ``educational`` read their input (and ``decontaminate`` its
``--benchmark-file``) through the tolerant ``load_jsonl_rows``. It stopped at
1,000,000 rows and skipped lines that are not JSON objects, both silently, so
a ``Flagged: 0`` or a ``decontaminate`` output could describe part of a file.

Over the cap the run now exits 1 before any output is written. A skipped line
gets the warning ``soup data clean`` gives, and the panel's ``Input:`` (or the
scorecard's ``Skipped lines`` row) counts it. A file of object rows under the
cap is reported as before. The tests patch ``_MAX_ROWS`` down to 3.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils import data_score
from tests.conftest import strip_ansi

BOX = chr(0x2502)
BENCH_TEXT = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima"
ALPACA = [
    {"instruction": "Say hello", "input": "", "output": "Hello there"},
    {"instruction": "Mail the report", "input": "", "output": "Sent it to alice@example.com"},
]

# (command, its output file or None, the arguments a successful run also needs)
COMMANDS = [
    pytest.param("score", None, (), id="score"),
    pytest.param(
        "decontaminate", "clean.jsonl",
        ("--benchmark-file", "bench.jsonl", "-o", "clean.jsonl"), id="decontaminate",
    ),
    pytest.param("toxicity", "toxicity.jsonl", ("-o", "toxicity.jsonl"), id="toxicity"),
    pytest.param("langdetect", "langdetect.jsonl", ("-o", "langdetect.jsonl"), id="langdetect"),
    pytest.param("pii", "pii.jsonl", ("-o", "pii.jsonl"), id="pii"),
    pytest.param(
        "educational", "educational.jsonl", ("-o", "educational.jsonl"), id="educational",
    ),
]
PANEL_COMMANDS = [param for param in COMMANDS if param.id != "score"]


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path, monkeypatch):
    # Soup refuses paths outside the working directory.
    monkeypatch.chdir(tmp_path)
    _write("bench.jsonl", [json.dumps({"text": BENCH_TEXT})])


def _write(name: str, lines: list[str]) -> None:
    with open(name, "w", encoding="utf-8") as fh:
        fh.write("".join(line + "\n" for line in lines))


def _rows(count: int) -> list[str]:
    return [
        json.dumps({"text": f"plain row number {index} for the scorer"})
        for index in range(count)
    ]


def _run(command: str, *args: str):
    result = CliRunner().invoke(app, ["data", command, "-i", "in.jsonl", *args])
    output = " ".join(strip_ansi(result.output).replace(BOX, " ").split())
    return result, output


def _assert_refused(result, output: str, message: str) -> None:
    assert result.exit_code == 1, output
    assert isinstance(result.exception, SystemExit), result.exception
    assert message in output
    assert "Traceback" not in output
    assert "Input:" not in output and "Data Scorecard" not in output


@pytest.fixture
def cap_of_three(monkeypatch):
    monkeypatch.setattr(data_score, "_MAX_ROWS", 3)


# --- the row cap ----------------------------------------------------------


def test_limit_error_is_a_value_error():
    # Both handlers in commands/data_score.py catch ValueError, so the limit
    # error is reported as a message, never as a traceback.
    assert issubclass(data_score.JsonlLimitError, ValueError)


def test_tolerant_reader_refuses_over_the_cap(tmp_path, cap_of_three):
    _write("in.jsonl", _rows(4))
    with pytest.raises(data_score.JsonlLimitError, match="file exceeds 3 rows"):
        data_score.load_jsonl_rows("in.jsonl")


@pytest.mark.parametrize(("command", "output_file", "extra"), COMMANDS)
def test_over_cap_is_refused(tmp_path, cap_of_three, command, output_file, extra):
    _write("in.jsonl", _rows(4))
    result, output = _run(command, *extra)
    _assert_refused(result, output, "file exceeds 3 rows")
    if output_file is not None:
        assert not (tmp_path / output_file).exists()


def test_benchmark_file_over_cap_is_refused(tmp_path, cap_of_three):
    _write("in.jsonl", _rows(1))
    _write("bench.jsonl", [json.dumps({"text": BENCH_TEXT})] * 4)
    result, output = _run(
        "decontaminate", "--benchmark-file", "bench.jsonl", "-o", "clean.jsonl",
    )
    _assert_refused(result, output, "--benchmark-file: file exceeds 3 rows")
    assert not (tmp_path / "clean.jsonl").exists()


@pytest.mark.parametrize(("command", "output_file", "extra"), PANEL_COMMANDS)
def test_file_at_the_cap_is_reported_as_before(
    tmp_path, cap_of_three, command, output_file, extra,
):
    _write("in.jsonl", _rows(3))
    result, output = _run(command, *extra)
    assert result.exit_code == 0, output
    assert "Input: 3 " in output
    assert "skipped" not in output and "Warning" not in output
    assert (tmp_path / output_file).exists()


def test_scorecard_at_the_cap_is_reported_as_before(cap_of_three):
    _write("in.jsonl", _rows(3))
    result, output = _run("score")
    assert result.exit_code == 0, output
    assert "Total rows 3 " in output
    assert "Skipped lines" not in output and "Warning" not in output


def test_skipped_line_after_the_last_row_does_not_count_toward_the_cap(cap_of_three):
    _write("in.jsonl", [*_rows(3), "not json"])
    result, output = _run("pii", "-o", "pii.jsonl")
    assert result.exit_code == 0, output
    assert "Warning: invalid JSON on line 4" in output
    assert "Input: 3 (1 line skipped)" in output


# --- skipped lines --------------------------------------------------------


def _bad_file() -> None:
    # The 4-line file from the issue: 2 alpaca rows, then `not json`, then `[1]`.
    _write("in.jsonl", [*(json.dumps(row) for row in ALPACA), "not json", "[1]"])


@pytest.mark.parametrize(("command", "output_file", "extra"), PANEL_COMMANDS)
def test_skipped_lines_are_reported(tmp_path, command, output_file, extra):
    _bad_file()
    result, output = _run(command, *extra)
    assert result.exit_code == 0, output
    assert "Warning: invalid JSON on line 3: Expecting value" in output
    assert "Warning: line 4 is a JSON array, not an object" in output
    assert "Input: 2 (2 lines skipped)" in output
    assert len((tmp_path / output_file).read_text(encoding="utf-8").splitlines()) <= 2


def test_scorecard_counts_skipped_lines():
    _bad_file()
    result, output = _run("score")
    assert result.exit_code == 0, output
    assert "Warning: invalid JSON on line 3" in output
    assert "Warning: line 4 is a JSON array, not an object" in output
    assert "Total rows 2 " in output
    assert "Skipped lines 2 " in output


def test_pii_still_flags_the_rows_it_reads():
    _bad_file()
    result, output = _run("pii", "-o", "pii.jsonl")
    assert result.exit_code == 0, output
    assert "Flagged: 1 " in output


def test_skipped_warnings_are_capped_per_kind():
    _write("in.jsonl", [_rows(1)[0], *(f"bad {index}" for index in range(5))])
    result, output = _run("pii", "-o", "pii.jsonl")
    assert result.exit_code == 0, output
    assert output.count("Warning: invalid JSON on line") == 3
    assert "Warning: ... and 2 more lines skipped (invalid JSON)" in output
    assert "Input: 1 (5 lines skipped)" in output


def test_benchmark_file_skipped_lines_are_reported():
    _write("in.jsonl", _rows(1))
    _write("bench.jsonl", [json.dumps({"text": BENCH_TEXT}), "not json"])
    result, output = _run(
        "decontaminate", "--benchmark-file", "bench.jsonl", "-o", "clean.jsonl",
    )
    assert result.exit_code == 0, output
    assert "Warning: invalid JSON on line 2" in output
    assert "benchmark texts from bench.jsonl (1 line skipped)" in output
    assert "Input: 1 Kept: 1" in output


@pytest.mark.parametrize(("command", "output_file", "extra"), COMMANDS)
def test_all_lines_skipped_says_so(tmp_path, command, output_file, extra):
    _write("in.jsonl", ["not json", "[1]"])
    result, output = _run(command, *extra)
    _assert_refused(result, output, "no object rows remain: 2 row(s) skipped (listed above)")
    assert "no extractable text" not in output
    if output_file is not None:
        assert not (tmp_path / output_file).exists()


def test_benchmark_file_with_only_skipped_lines_says_so(tmp_path):
    _write("in.jsonl", _rows(1))
    _write("bench.jsonl", ["not json"])
    result, output = _run(
        "decontaminate", "--benchmark-file", "bench.jsonl", "-o", "clean.jsonl",
    )
    _assert_refused(
        result, output, "--benchmark-file: no object rows remain: 1 row(s) skipped",
    )
    assert not (tmp_path / "clean.jsonl").exists()


# --- controls -------------------------------------------------------------


@pytest.mark.parametrize(("command", "output_file", "extra"), PANEL_COMMANDS)
def test_clean_file_has_no_warning(command, output_file, extra):
    _write("in.jsonl", [json.dumps(row) for row in ALPACA])
    result, output = _run(command, *extra)
    assert result.exit_code == 0, output
    assert "Warning" not in output and "skipped" not in output
    assert "Input: 2 " in output


def test_on_skip_names_each_kind_without_row_contents():
    _write("in.jsonl", [_rows(1)[0], "secret-token-123", '"a string"', "[1]"])
    seen = []
    rows = data_score.load_jsonl_rows(
        "in.jsonl", on_skip=lambda kind, message: seen.append((kind, message)),
    )
    assert len(rows) == 1
    assert [kind for kind, _ in seen] == ["invalid_json", "non_object", "non_object"]
    assert seen[1][1] == "line 3 is a JSON string, not an object"
    assert seen[2][1] == "line 4 is a JSON array, not an object"
    assert all("secret-token-123" not in message for _, message in seen)


def test_strict_reader_is_unchanged(cap_of_three):
    _write("in.jsonl", [*_rows(3), "not json"])
    with pytest.raises(data_score.JsonlLimitError, match="file exceeds 3 rows"):
        data_score.load_jsonl_rows("in.jsonl", strict=True)
