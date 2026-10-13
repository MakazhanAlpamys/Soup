"""Dataset commands report non-object JSON rows instead of crashing on dict methods."""

import json

import pytest
from typer.testing import CliRunner

from soup_cli.commands.data import app
from soup_cli.data.loader import load_raw_data

GOOD = {"instruction": "A sufficiently long question", "output": "A useful answer"}
BAD = ["abc", [1, 2], 42, None]


@pytest.mark.parametrize("bad", BAD)
@pytest.mark.parametrize("extension", ["jsonl", "json"])
def test_raw_loader_skips_non_object_rows_and_collects_positions(tmp_path, capsys, bad, extension):
    path = tmp_path / f"mixed.{extension}"
    rows = [GOOD, bad, GOOD]
    content = (
        "\n".join(json.dumps(row) for row in rows)
        if extension == "jsonl" else json.dumps(rows)
    )
    path.write_text(content, encoding="utf-8")
    issues = []
    assert load_raw_data(path, issues=issues) == [GOOD, GOOD]
    unit = "line" if extension == "jsonl" else "row"
    assert len(issues) == 1
    assert issues[0].startswith(f"{unit} 2 is a JSON")
    assert "not an object" in issues[0]
    assert issues[0] in capsys.readouterr().out


@pytest.mark.parametrize("bad", BAD)
@pytest.mark.parametrize("command", ["review", "inspect", "validate", "stats", "dedup"])
def test_commands_warn_and_keep_usable_rows(tmp_path, monkeypatch, bad, command):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "mixed.jsonl"
    rows = [GOOD, bad, {**GOOD, "instruction": "Another sufficiently long question"}]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    args = [command, str(path)]
    if command == "validate":
        args += ["--format", "alpaca"]
    if command == "dedup":
        args += ["--output", "deduped.jsonl"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, (result.output, result.exception)
    assert "line 2 is a JSON" in result.output
    assert "not an object" in result.output
    if command == "validate":
        assert "Issues found:" in result.output
        assert "Dataset is valid!" not in result.output


def test_blank_lines_do_not_change_reported_physical_line(tmp_path, capsys):
    path = tmp_path / "blank.jsonl"
    path.write_text('\n\n"private content"\n' + json.dumps(GOOD), encoding="utf-8")
    assert load_raw_data(path) == [GOOD]
    output = capsys.readouterr().out
    assert "line 3 is a JSON string, not an object" in output
    assert "private content" not in output
