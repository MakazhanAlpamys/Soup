"""Skipped-row warnings are capped per kind, with an exact count kept."""

import json
import re

import pytest
from typer.testing import CliRunner

from soup_cli.commands.data import app
from soup_cli.data.loader import SkippedRows, load_raw_data
from soup_cli.data.validator import _MAX_REASON_SAMPLES as N
from tests.conftest import strip_ansi

GOOD = {"instruction": "A sufficiently long question", "output": "A useful answer"}
PAIR = {"prompt": "A sufficiently long prompt", "chosen": "Good", "rejected": "Bad"}


def _plain(text):
    return " ".join(strip_ansi(text).split())


def _write_jsonl(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _non_object_warnings(text, unit="line"):
    return re.findall(rf"Warning: {unit} \d+ is a JSON \w+, not an object", _plain(text))


def _bad_json_warnings(text):
    return re.findall(r"Warning: invalid JSON on line \d+", _plain(text))


def test_jsonl_non_objects_are_capped_with_one_summary(tmp_path, capsys):
    path = tmp_path / "junk.jsonl"
    _write_jsonl(path, [json.dumps(GOOD)] + [str(i) for i in range(N + 5)])
    issues = SkippedRows()
    assert load_raw_data(path, issues=issues) == [GOOD]
    out = capsys.readouterr().out
    assert len(_non_object_warnings(out)) == N
    assert _plain(out).count("... and 5 more lines skipped") == 1
    assert issues.total == N + 5
    assert len(issues) == N


def test_json_array_non_objects_are_capped_with_one_summary(tmp_path, capsys):
    path = tmp_path / "junk.json"
    path.write_text(json.dumps([GOOD] + list(range(N + 5))), encoding="utf-8")
    issues = SkippedRows()
    assert load_raw_data(path, issues=issues) == [GOOD]
    out = capsys.readouterr().out
    assert len(_non_object_warnings(out, "row")) == N
    assert _plain(out).count("... and 5 more rows skipped") == 1
    assert issues.total == N + 5
    assert len(issues) == N


def test_invalid_json_lines_are_capped_with_one_summary(tmp_path, capsys):
    path = tmp_path / "junk.jsonl"
    _write_jsonl(path, [json.dumps(GOOD)] + [f"{{not json {i}" for i in range(N + 5)])
    assert load_raw_data(path) == [GOOD]
    out = capsys.readouterr().out
    assert len(_bad_json_warnings(out)) == N
    assert _plain(out).count("... and 5 more lines skipped (invalid JSON)") == 1


def test_each_kind_gets_its_own_cap_and_summary(tmp_path, capsys):
    path = tmp_path / "mixed.jsonl"
    lines = [json.dumps(GOOD)]
    lines += [str(i) for i in range(N + 2)]
    lines += [f"{{not json {i}" for i in range(N + 2)]
    _write_jsonl(path, lines)
    issues = SkippedRows()
    assert load_raw_data(path, issues=issues) == [GOOD]
    out = capsys.readouterr().out
    assert len(_non_object_warnings(out)) == N
    assert len(_bad_json_warnings(out)) == N
    assert _plain(out).count("... and 2 more lines skipped (JSON values") == 1
    assert _plain(out).count("... and 2 more lines skipped (invalid JSON)") == 1
    # every skipped row, of either kind, is counted (#1887) and 2 * N samples are kept
    assert issues.total == 2 * (N + 2)
    assert len(issues) == 2 * N


@pytest.mark.parametrize("junk", [N - 1, N])
def test_up_to_the_cap_prints_every_warning_and_no_summary(tmp_path, capsys, junk):
    path = tmp_path / "few.jsonl"
    _write_jsonl(path, [json.dumps(GOOD)] + [str(i) for i in range(junk)])
    issues = SkippedRows()
    load_raw_data(path, issues=issues)
    out = capsys.readouterr().out
    assert len(_non_object_warnings(out)) == junk
    assert "more" not in _plain(out)
    assert issues.total == junk
    assert len(issues) == junk


def test_plain_list_issues_still_works_and_is_bounded(tmp_path):
    path = tmp_path / "junk.jsonl"
    _write_jsonl(path, [json.dumps(GOOD)] + [str(i) for i in range(N + 5)])
    issues = []
    load_raw_data(path, issues=issues)
    assert len(issues) == N


def test_validate_reports_capped_listing_with_exact_remainder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "junk.jsonl"
    _write_jsonl(path, [json.dumps(GOOD)] + [str(i) for i in range(N + 5)])
    result = CliRunner().invoke(app, ["validate", str(path), "--format", "alpaca"])
    assert result.exit_code == 0, (result.output, result.exception)
    out = _plain(result.output)
    assert len(_non_object_warnings(out)) == N  # loader warnings
    assert len(re.findall(r"! line \d+ is a JSON number", out)) == N  # issue listing
    assert out.count("... and 5 more lines skipped") == 1  # loader summary
    assert out.count("! ... and 5 more") == 1  # listing remainder


def test_review_caps_warnings_and_still_shows_pairs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "pairs.jsonl"
    _write_jsonl(path, [str(i) for i in range(N + 5)] + [json.dumps(PAIR)])
    result = CliRunner().invoke(app, ["review", str(path)])
    assert result.exit_code == 0, (result.output, result.exception)
    out = _plain(result.output)
    assert len(_non_object_warnings(out)) == N
    assert out.count("... and 5 more lines skipped") == 1
    assert "Pair 1" in out


def test_large_junk_file_is_bounded(tmp_path, capsys):
    path = tmp_path / "junk20k.jsonl"
    _write_jsonl(path, [json.dumps(GOOD)] + [str(i) for i in range(20000)])
    issues = SkippedRows()
    assert load_raw_data(path, issues=issues) == [GOOD]
    out = capsys.readouterr().out
    assert len(_non_object_warnings(out)) == N
    assert issues.total == 20000
    assert len(issues) == N


@pytest.mark.parametrize("extension", ["jsonl", "json"])
def test_validate_counts_every_skipped_row_not_just_the_sampled_ones(tmp_path, extension):
    # #1887 counts skipped rows as invalid rows; with the warnings capped, the count
    # must come from the exact total, not from the few retained samples.
    junk = N + 17
    path = tmp_path / f"data.{extension}"
    rows = [GOOD, {**GOOD, "instruction": "Another sufficiently long question"}]
    if extension == "jsonl":
        _write_jsonl(path, [json.dumps(r) for r in rows] + [str(i) for i in range(junk)])
    else:
        path.write_text(json.dumps(rows + list(range(junk))), encoding="utf-8")
    total = 2 + junk

    result = CliRunner().invoke(app, ["validate", str(path), "--format", "alpaca"])
    out = _plain(result.output)
    assert f"2/{total} rows valid" in out, out

    gated = CliRunner().invoke(
        app, ["validate", str(path), "--format", "alpaca", "--min-valid-fraction", "0.5"],
    )
    assert gated.exit_code == 2, _plain(gated.output)


def test_validate_counts_non_object_and_invalid_json_lines_together(tmp_path):
    path = tmp_path / "mixed.jsonl"
    _write_jsonl(
        path,
        [json.dumps(GOOD)] + [str(i) for i in range(N + 4)] + ["{bad"] * (N + 6),
    )
    skipped = (N + 4) + (N + 6)
    result = CliRunner().invoke(app, ["validate", str(path), "--format", "alpaca"])
    assert f"1/{1 + skipped} rows valid" in _plain(result.output), _plain(result.output)


def test_validate_listing_shows_samples_of_both_kinds(tmp_path):
    path = tmp_path / "both.jsonl"
    _write_jsonl(
        path,
        [json.dumps(GOOD)] + [str(i) for i in range(N)] + ["{bad"] * (N + 2),
    )
    result = CliRunner().invoke(app, ["validate", str(path), "--format", "alpaca"])
    out = _plain(result.output)
    # the issue listing under "Issues found" keeps samples of both kinds
    listing = out.split("Issues found", 1)[1]
    assert listing.count("invalid JSON on line") == N, out
    assert listing.count("not an object") == N, out
    assert "... and 2 more" in listing, out
