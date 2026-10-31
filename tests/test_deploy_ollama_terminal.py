"""soup deploy ollama prints model names, paths and prompts as typed, not as Rich markup."""

import re
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi

runner = CliRunner()

_OLLAMA = "soup_cli.utils.ollama"

# A colon is not a legal file-name character on Windows.
COLON_NAME = pytest.param(
    "m:warning:",
    marks=pytest.mark.skipif(sys.platform == "win32", reason="':' is not valid in a Windows name"),
)


def _plain(result) -> str:
    return " ".join(strip_ansi(result.output).split()).replace("\\", "/")


@pytest.fixture(autouse=True)
def _ollama(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COLUMNS", "500")
    monkeypatch.setattr(f"{_OLLAMA}.detect_ollama", lambda: "0.5.0")
    monkeypatch.setattr(f"{_OLLAMA}.deploy_to_ollama", lambda name, modelfile: (True, "ok"))


def _gguf(directory: str = ".") -> Path:
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "model.gguf"
    path.write_bytes(b"fake")
    return path


def _deploy(*args: str, **kwargs):
    return runner.invoke(app, ["deploy", "ollama", "--name", "soup-test", *args], **kwargs)


@pytest.mark.parametrize("directory", ["[red]gguf", "[/]gguf", COLON_NAME])
def test_deploy_plan_names_the_gguf_path_as_typed(directory):
    path = _gguf(directory)

    result = _deploy("--model", str(path), "--yes")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert re.search(rf"GGUF: +\S*{re.escape(directory)}/model\.gguf", text)


@pytest.mark.parametrize("system", ["Use [INST] and [/INST] tags", "[red]careful", "hi :warning:"])
def test_deploy_plan_shows_the_system_prompt_as_typed(system):
    path = _gguf()

    result = _deploy("--model", str(path), "--system", system, "--yes")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert f"System: {system}" in _plain(result)


@pytest.mark.parametrize("value", ["[red]x", "[/]x", "x:warning:"])
def test_deploy_plan_shows_parameters_as_typed(value):
    path = _gguf()

    result = _deploy("--model", str(path), "--parameter", f"stop={value}", "--yes")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert f"'stop': '{value}'" in _plain(result)


def test_deploy_plan_strips_controls_from_the_display_only():
    path = _gguf()

    result = _deploy("--model", str(path), "--system", "[bold]hi\x9b\u202e", "--yes")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert "System: [bold]hi" in text
    assert "\x9b" not in text and "\u202e" not in text


@pytest.mark.parametrize("name", ["soup-a:warning:", "a:warning:"])
def test_deploy_names_the_model_as_typed(name):
    path = _gguf()

    result = runner.invoke(
        app, ["deploy", "ollama", "--model", str(path), "--name", name, "--yes"]
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"Model: {name}" in text
    assert f"Model created: {name}" in text
    assert f"ollama run {name}" in text


@pytest.mark.parametrize("template", ["[red]x", "[/]x", "x:warning:"])
def test_deploy_unknown_template_is_named_as_typed(template):
    path = _gguf()

    result = _deploy("--model", str(path), "--template", template, "--yes")

    assert result.exit_code == 1
    assert f"Unknown template: {template}" in _plain(result)


@pytest.mark.parametrize("param", ["[red]x", "[/]x", "x:warning:"])
def test_deploy_invalid_parameter_is_named_as_typed(param):
    path = _gguf()

    result = _deploy("--model", str(path), "--parameter", param, "--yes")

    assert result.exit_code == 1
    assert f"Invalid parameter format: {param}" in _plain(result)


@pytest.mark.parametrize("name", ["[red]weights.gguf", "[/].gguf", "gone:warning:.gguf"])
def test_deploy_missing_gguf_is_named_as_typed(name):
    result = _deploy("--model", name, "--yes")

    assert result.exit_code == 1
    assert f"GGUF file not found: {Path(name).as_posix()}" in _plain(result)


@pytest.mark.parametrize("message", ["bad [/] value", "[red]refused", "stop:warning:"])
def test_deploy_failure_message_is_shown_as_typed(monkeypatch, message):
    monkeypatch.setattr(f"{_OLLAMA}.deploy_to_ollama", lambda name, modelfile: (False, message))
    path = _gguf()

    result = _deploy("--model", str(path), "--yes")

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert f"Deploy failed: {message}" in _plain(result)


@pytest.mark.parametrize("message", ["bad [/] value", "[red]refused"])
def test_remove_failure_message_is_shown_as_typed(monkeypatch, message):
    monkeypatch.setattr(f"{_OLLAMA}.remove_model", lambda name: (False, message))

    result = runner.invoke(app, ["deploy", "ollama", "--remove", "soup-test", "--yes"])

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert message in _plain(result)
