"""Non-object dataset rows are named and skipped, with a canary first-row exception."""

import json

import pytest
from typer.testing import CliRunner

from soup_cli.commands.data import app
from soup_cli.commands.data_canary import app as canary_app
from soup_cli.config.schema import DataConfig
from soup_cli.data.loader import load_dataset, load_raw_data
from tests.conftest import strip_ansi

GOOD = {"instruction": "A sufficiently long question", "output": "A useful answer"}
GOOD2 = {**GOOD, "instruction": "Another sufficiently long question"}
BAD_KINDS = [("abc", "string"), ([1, 2], "array"), (42, "number"),
             (None, "null"), (True, "boolean"), (1.5, "number")]


def _plain(text):
    return " ".join(strip_ansi(text).split())


def _write(path, rows):
    content = ("\n".join(json.dumps(row) for row in rows)
               if path.suffix == ".jsonl" else json.dumps(rows))
    path.write_text(content, encoding="utf-8")
    return content


@pytest.mark.parametrize("bad,kind", BAD_KINDS)
@pytest.mark.parametrize("extension", ["jsonl", "json"])
def test_raw_loader_skips_non_objects_and_names_their_kind(tmp_path, capsys, bad, kind, extension):
    path = tmp_path / f"mixed.{extension}"
    _write(path, [GOOD, bad, GOOD2])
    issues = []
    assert load_raw_data(path, issues=issues) == [GOOD, GOOD2]
    unit = "line" if extension == "jsonl" else "row"
    assert issues == [f"{unit} 2 is a JSON {kind}, not an object"]
    assert issues[0] in _plain(capsys.readouterr().out)


@pytest.mark.parametrize("bad,kind", BAD_KINDS)
@pytest.mark.parametrize("command", ["review", "inspect", "validate", "stats", "dedup"])
@pytest.mark.parametrize("position", [1, 2])
def test_commands_skip_junk_at_first_or_middle_line_and_continue(
    tmp_path, monkeypatch, bad, kind, command, position,
):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "mixed.jsonl"
    rows = [GOOD, GOOD2]
    rows.insert(position - 1, bad)
    _write(path, rows)
    args = [command, str(path)]
    if command == "validate":
        args += ["--format", "alpaca"]
    if command == "dedup":
        args += ["--output", "deduped.jsonl"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, (result.output, result.exception)
    output = _plain(result.output)
    assert f"line {position} is a JSON {kind}, not an object" in output
    if command == "review":
        assert "Pair 2" in output
    if command == "validate":
        assert "Issues found:" in output
        assert "Dataset is valid!" not in output


def test_blank_lines_keep_the_physical_line_and_hide_contents(tmp_path, capsys):
    path = tmp_path / "blank.jsonl"
    path.write_text('\n\n"private content"\n' + json.dumps(GOOD), encoding="utf-8")
    assert load_raw_data(path) == [GOOD]
    output = _plain(capsys.readouterr().out)
    assert "line 3 is a JSON string, not an object" in output
    assert "private content" not in output


@pytest.mark.parametrize("bad,kind", BAD_KINDS)
@pytest.mark.parametrize("extension", ["jsonl", "json"])
@pytest.mark.parametrize("explicit_format", [False, True])
def test_canary_insert_refuses_non_object_first_row_without_writing(
    tmp_path, monkeypatch, bad, kind, extension, explicit_format,
):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / f"mixed.{extension}"
    content = _write(path, [bad, GOOD])
    args = ["insert", str(path), "--manifest", "manifest.json", "-o", "canaried.jsonl"]
    if explicit_format:
        args += ["--format", "alpaca"]
    result = CliRunner().invoke(canary_app, args)
    assert result.exit_code == 1
    assert f"the first row is a JSON {kind}, not an object" in _plain(result.output)
    assert not (tmp_path / "canaried.jsonl").exists()
    assert not (tmp_path / "manifest.json").exists()
    assert path.read_text(encoding="utf-8") == content


@pytest.mark.parametrize("extension", ["jsonl", "json"])
def test_canary_insert_preserves_later_non_object_rows(tmp_path, monkeypatch, extension):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / f"mixed.{extension}"
    _write(path, [GOOD, 42, GOOD2])
    issues = []
    assert load_raw_data(path, issues=issues, skip_non_object=False) == [GOOD, 42, GOOD2]
    assert issues == []
    result = CliRunner().invoke(canary_app, [
        "insert", str(path), "--manifest", "manifest.json", "-o", "canaried.jsonl", "--count", "2",
    ])
    assert result.exit_code == 0, (result.output, result.exception)
    written = [json.loads(line) for line in (tmp_path / "canaried.jsonl").read_text().splitlines()]
    assert len(written) == 5
    assert [row for row in written if row in [GOOD, 42, GOOD2]] == [GOOD, 42, GOOD2]


def test_training_loader_warns_before_conversion_instead_of_old_drop_error(tmp_path, capsys):
    path = tmp_path / "train.jsonl"
    _write(path, [GOOD, "abc", GOOD2])
    dataset = load_dataset(DataConfig(train=str(path), format="alpaca", val_split=0.0))
    assert len(dataset["train"]) == 2
    output = _plain(capsys.readouterr().out)
    assert "line 2 is a JSON string, not an object" in output
    assert "string indices must be integers" not in output


def test_validate_caps_the_repeated_issue_listing(tmp_path):
    path = tmp_path / "many.jsonl"
    _write(path, [GOOD] + [42] * 6)
    result = CliRunner().invoke(app, ["validate", str(path), "--format", "alpaca"])
    assert result.exit_code == 0, (result.output, result.exception)
    listing = _plain(result.output).split("Issues found:", 1)[1]
    for line in (2, 3, 4):
        assert f"line {line} is a JSON number, not an object" in listing
    assert "line 5 is a JSON" not in listing
    assert "... and 3 more" in listing
