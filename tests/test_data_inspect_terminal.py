"""Display of paths, columns and sample values through soup data inspect."""

import json

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from tests._windows_filenames import skip_colon_on_windows
from tests.conftest import strip_ansi


def _inspect(name, *args):
    result = CliRunner().invoke(app, ["data", "inspect", name, *args])
    lines = [" ".join(strip_ansi(line).split()) for line in result.output.splitlines()]
    return result.exit_code, lines


def test_dataset_text_is_literal_in_stats_headers_and_sample_cells(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    key = "field:rocket:[key]\u202e"
    value = r"reply :rocket: \[bold]" + "\x07" + " end"
    (tmp_path / "data.jsonl").write_text(json.dumps({key: value}) + "\n", encoding="utf-8")

    code, lines = _inspect("data.jsonl")
    assert code == 0, lines
    assert sum("field:rocket:[key]" in line for line in lines) == 2, lines
    assert any(r"reply :rocket: \[bold] end" in line for line in lines), lines
    assert all("\u202e" not in line and "\x07" not in line for line in lines)


def test_sample_is_cut_before_controls_are_stripped(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    value = "a" * 58 + "\x07" + "XY"
    (tmp_path / "data.jsonl").write_text(json.dumps({"v": value}) + "\n", encoding="utf-8")
    code, lines = _inspect("data.jsonl")
    assert code == 0, lines
    assert any("a" * 58 + "X" in line for line in lines), lines
    assert all("XY" not in line for line in lines)


@pytest.mark.parametrize("name", [
    "data[bold]\u202e.jsonl",
    pytest.param("data:rocket:.jsonl", marks=skip_colon_on_windows),
])
def test_missing_and_existing_paths(name, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    displayed = name.replace("\u202e", "")
    code, lines = _inspect(name)
    assert code == 1
    assert f"File not found: {displayed}" in lines

    (tmp_path / name).write_text('{"text":"hello"}\n', encoding="utf-8")
    code, lines = _inspect(name)
    assert code == 0, lines
    assert f"Inspecting {displayed}..." in lines


def test_plain_file_statistics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "plain.jsonl").write_text('{"text":"hi"}\n' * 2, encoding="utf-8")
    code, lines = _inspect("plain.jsonl")
    assert code == 0, lines
    for label, value in [("Total samples", "2"), ("Columns", "text"), ("Duplicates", "1")]:
        assert any(label in line and value in line.split() for line in lines), lines
    assert "Sample rows (2):" in lines
