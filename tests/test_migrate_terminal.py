"""soup migrate prints the paths it was given as typed, not as Rich markup."""

from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from tests._windows_filenames import skip_colon_on_windows
from tests.conftest import strip_ansi
from tests.test_migrate import LLAMA_FACTORY_SFT

runner = CliRunner()


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch):
    """Rich wraps at the terminal width, which CI sets differently from a laptop."""
    monkeypatch.setattr("soup_cli.commands.migrate.console", Console(width=500))


# A colon is not a legal file-name character on Windows.
COLON_NAME = pytest.param(
    "missing:warning:.yaml",
    marks=skip_colon_on_windows,
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
    # The receipt echoes the CLI argument, including its original separator spelling.
    assert f"Config written to {name}" in text
    assert f"soup train --config {name}" in text


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


def test_migrate_strips_controls_from_a_refused_output_path(source):
    result = _migrate(str(source), "--output", "../x[bold]out\x9b\u202e.yaml", "--yes")

    assert result.exit_code == 1
    text = _plain(result)
    assert "Output path is outside the current directory:" in text
    assert "x[bold]out.yaml" in text
    assert "\x9b" not in text and "\u202e" not in text


def test_migrate_jsonl_refusal_names_the_file_literally(source):
    data = Path("[red]data.jsonl")
    data.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")

    result = _migrate(str(data), "--output", "soup.yaml", "--yes")

    assert result.exit_code == 2
    assert "got JSONL ([red]data.jsonl)" in _plain(result)


@skip_colon_on_windows
def test_migrate_jsonl_refusal_shows_an_emoji_code_in_the_name_as_typed(source):
    data = Path("data:warning:.jsonl")
    data.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")

    result = _migrate(str(data), "--output", "soup.yaml", "--yes")

    assert result.exit_code == 2
    assert "got JSONL (data:warning:.jsonl)" in _plain(result)


@skip_colon_on_windows
def test_migrate_refused_output_path_shows_an_emoji_code_as_typed(source):
    result = _migrate(str(source), "--output", "../x:warning:.yaml", "--yes")

    assert result.exit_code == 1
    assert "Output path is outside the current directory: ../x:warning:.yaml" in _plain(result)


_CONTROLS = "\x9b\u202e"


@pytest.mark.parametrize("name", ["x[red]s", "x[/]s", "s:warning:"])
def test_migrate_unknown_source_is_named_as_typed(source, name):
    result = runner.invoke(app, ["migrate", "--from", name, str(source)])

    assert result.exit_code == 1
    assert f"Unknown source: {name}" in _plain(result)


def test_migrate_unknown_source_strips_controls_from_the_display(source):
    result = runner.invoke(app, ["migrate", "--from", f"x[bold]s{_CONTROLS}", str(source)])

    assert result.exit_code == 1
    text = _plain(result)
    assert "Unknown source: x[bold]s" in text
    assert "\x9b" not in text and "\u202e" not in text


@pytest.mark.parametrize(
    "message", ["bad [/] value", "[red]refused", "stop:warning:", f"x[bold]m{_CONTROLS}"]
)
def test_migrate_failure_message_is_shown_as_typed(source, monkeypatch, message):
    def _refuse(path):
        raise ValueError(message)

    monkeypatch.setattr("soup_cli.migrate.llamafactory.migrate_llamafactory", _refuse)

    result = _migrate(str(source), "--output", "soup.yaml", "--yes")

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Migration failed: {message.replace(_CONTROLS, '')}" in text
    assert "\x9b" not in text and "\u202e" not in text


@pytest.mark.parametrize(
    "message", ["bad [/] value", "[red]refused", "stop:warning:", f"x[bold]m{_CONTROLS}"]
)
def test_migrate_schema_failure_message_is_shown_as_typed(source, monkeypatch, message):
    def _refuse(yaml_str):
        raise ValueError(message)

    monkeypatch.setattr("soup_cli.config.loader.load_config_from_string", _refuse)

    result = _migrate(str(source), "--output", "soup.yaml", "--yes")

    assert result.exit_code == 1
    text = _plain(result)
    shown = message.replace(_CONTROLS, "")
    assert f"Generated config failed schema validation: {shown}" in text
    assert "\x9b" not in text and "\u202e" not in text


@pytest.mark.parametrize(
    "warning", ["bad [/] value", "[red]careful", "stop:warning:", f"x[bold]w{_CONTROLS}"]
)
def test_migrate_warnings_are_shown_as_typed(source, monkeypatch, warning):
    from soup_cli.migrate import llamafactory

    real = llamafactory.migrate_llamafactory
    monkeypatch.setattr(
        llamafactory, "migrate_llamafactory", lambda path: {**real(path), "_warnings": [warning]}
    )

    result = _migrate(str(source), "--output", "soup.yaml", "--yes")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"! {warning.replace(_CONTROLS, '')}" in text
    assert "\x9b" not in text and "\u202e" not in text
