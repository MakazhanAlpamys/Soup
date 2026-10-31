"""soup quickstart prints the directory and files it creates as typed, not as Rich markup."""

import re
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi

runner = CliRunner()

# A colon is not a legal file-name character on Windows.
COLON_NAME = pytest.param(
    "demo:warning:",
    marks=pytest.mark.skipif(sys.platform == "win32", reason="':' is not valid in a Windows name"),
)
DIRECTORIES = ["[red]proj", "[/]proj", COLON_NAME]


def _plain(result) -> str:
    return " ".join(strip_ansi(result.output).split())


@pytest.fixture(autouse=True)
def _wide_terminal(monkeypatch):
    monkeypatch.setenv("COLUMNS", "500")


@pytest.fixture(autouse=True)
def _default_model(monkeypatch):
    """No test here may read this machine's GPU."""
    monkeypatch.setattr(
        "soup_cli.commands.quickstart._pick_quickstart_model",
        lambda: ("test/model", None),
    )


@pytest.mark.parametrize("name", DIRECTORIES)
def test_quickstart_names_the_working_directory_as_typed(tmp_path, monkeypatch, name):
    workdir = tmp_path / name
    workdir.mkdir(parents=True)
    monkeypatch.chdir(workdir)

    result = runner.invoke(app, ["quickstart", "--dry-run", "--yes"])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert (workdir / "quickstart_data.jsonl").exists()
    text = _plain(result)
    shown = text.replace("\\", "/")
    # The confirmation panel, which is the only place a raw Rich Panel is used.
    escaped = re.escape(name)
    assert re.search(rf"1\. Create \S*{escaped}/quickstart_data\.jsonl \(20 examples\)", shown)
    assert re.search(rf"2\. Create \S*{escaped}/quickstart_soup\.yaml config", shown)
    assert f"{name}/quickstart_soup.yaml" in shown.split("soup train --config")[1]


def test_quickstart_strips_controls_only_from_the_displayed_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    directory = Path("[bold]out\x9b\u202e")

    result = runner.invoke(
        app, ["quickstart", "--dry-run", "--yes", "--output", str(directory)]
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert (directory / "quickstart_data.jsonl").exists()
    text = _plain(result)
    assert "[bold]out/quickstart_data.jsonl" in text.replace("\\", "/")
    assert "\x9b" not in text and "\u202e" not in text


@pytest.mark.parametrize("name", DIRECTORIES)
def test_quickstart_names_existing_files_as_typed(tmp_path, monkeypatch, name):
    workdir = tmp_path / name
    workdir.mkdir(parents=True)
    monkeypatch.chdir(workdir)
    (workdir / "quickstart_data.jsonl").write_text("keep\n", encoding="utf-8")
    (workdir / "quickstart_soup.yaml").write_text("keep: me\n", encoding="utf-8")

    result = runner.invoke(app, ["quickstart", "--dry-run", "--yes"])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert (workdir / "quickstart_data.jsonl").read_text(encoding="utf-8") == "keep\n"
    text = _plain(result).replace("\\", "/")
    assert "Data file already exists:" in text
    assert f"{name}/quickstart_data.jsonl" in text
    assert "Config file already exists:" in text
    assert f"{name}/quickstart_soup.yaml" in text


@pytest.mark.parametrize("name", ["../[red]elsewhere", "../[/]x", "../gone:warning:"])
def test_quickstart_refused_output_is_named_as_typed(tmp_path, monkeypatch, name):
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    result = runner.invoke(app, ["quickstart", "--dry-run", "--yes", "--output", name])

    assert result.exit_code == 2
    assert f"got: {name}" in _plain(result)
