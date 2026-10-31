"""soup migrate prints the paths it was given as typed, not as Rich markup."""

import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi
from tests.test_migrate import LLAMA_FACTORY_SFT

runner = CliRunner()

# A colon is not a legal file-name character on Windows.
COLON_NAME = pytest.param(
    "missing:warning:.yaml",
    marks=pytest.mark.skipif(sys.platform == "win32", reason="':' is not valid in a Windows name"),
)


def _plain(result) -> str:
    return " ".join(strip_ansi(result.output).split())


def _migrate(*args: str, **kwargs):
    return runner.invoke(app, ["migrate", "--from", "llamafactory", *args], **kwargs)


@pytest.fixture
def source(tmp_path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text(LLAMA_FACTORY_SFT, encoding="utf-8")
    return path


@pytest.mark.parametrize("name", ["soup.yaml", "[red]soup.yaml", "[/].yaml", COLON_NAME])
def test_migrate_reports_literal_output_path_after_writing(source, name):
    output = Path(name)
    output.parent.mkdir(parents=True, exist_ok=True)

    result = _migrate(str(source), "--output", name, "--yes")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "meta-llama/Llama-3.1-8B-Instruct" in output.read_text(encoding="utf-8")
    text = _plain(result)
    assert f"Config written to {output}" in text
    assert f"soup train --config {output}" in text


def test_migrate_strips_controls_only_from_displayed_output_path(source):
    output = Path("[bold]config\x9b\u202e.yaml")

    result = _migrate(str(source), "--output", str(output), "--yes")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "meta-llama/Llama-3.1-8B-Instruct" in output.read_text(encoding="utf-8")
    text = _plain(result)
    assert "Config written to [bold]config.yaml" in text
    assert "soup train --config [bold]config.yaml" in text
    assert "\x9b" not in text and "\u202e" not in text


def test_migrate_overwrite_prompt_shows_the_path_as_typed(source):
    output = Path("[bold]x\x9b.yaml")
    output.write_text("keep: me\n", encoding="utf-8")

    result = _migrate(str(source), "--output", str(output), input="n\n")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert output.read_text(encoding="utf-8") == "keep: me\n"
    text = _plain(result)
    assert "File '[bold]x.yaml' already exists. Overwrite?" in text
    assert "\x9b" not in text
    assert "Aborted." in text


@pytest.mark.parametrize("name", ["[red]missing.yaml", "[/].yaml", COLON_NAME])
def test_migrate_missing_input_is_named_literally(source, name):
    result = _migrate(name, "--output", "soup.yaml", "--yes")

    assert result.exit_code == 1
    # Windows prints the path with backslashes, as Path does.
    assert f"Input file not found: {Path(name)}" in _plain(result)
    assert not Path("soup.yaml").exists()


def test_migrate_strips_controls_from_a_refused_input_path(source):
    result = _migrate("../[bold]out\x9b\u202e.yaml", "--output", "soup.yaml", "--yes")

    assert result.exit_code == 1
    text = _plain(result)
    assert "Input path is outside the current directory:" in text
    assert "[bold]out.yaml" in text
    assert "\x9b" not in text and "\u202e" not in text


def test_migrate_refused_output_path_is_named_literally(source):
    result = _migrate(str(source), "--output", "../[red]out.yaml", "--yes")

    assert result.exit_code == 1
    text = _plain(result)
    assert "Output path is outside the current directory:" in text
    assert "[red]out.yaml" in text


def test_migrate_jsonl_refusal_names_the_file_literally(source):
    data = Path("[red]data.jsonl")
    data.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")

    result = _migrate(str(data), "--output", "soup.yaml", "--yes")

    assert result.exit_code == 2
    assert "got JSONL ([red]data.jsonl)" in _plain(result)
