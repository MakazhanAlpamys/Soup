"""soup deploy hf-space prints typed and remote text as typed, not as Rich markup."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.hf import validate_repo_id
from tests._windows_filenames import skip_colon_on_windows
from tests.conftest import strip_ansi

runner = CliRunner()

_CONTROLS = "\x9b\u202e"


# One spelling per way the old code went wrong: a tag, a closing tag with
# nothing to close (a MarkupError), an emoji code, and a tag with control
# characters, so the strip and the escape are pinned separately.
_SPELLINGS = [
    "x[red]n",
    "[/]n",
    pytest.param("n:warning:", id="emoji-code"),
    f"x[bold]n{_CONTROLS}",
]
_PATH_SPELLINGS = [
    "x[red]n",
    "[/]n",
    pytest.param("n:warning:", marks=skip_colon_on_windows, id="emoji-code"),
    f"x[bold]n{_CONTROLS}",
]

# Repo ids that validate_repo_id refuses, in the spellings that matter. A bare
# "[/]" cannot be one component, so the closing tag rides on a ".." segment.
_BAD_REPO_IDS = [
    "a[red]b",
    "..[/]n",
    pytest.param("a:warning:b", id="emoji-code"),
    f"a[bold]b{_CONTROLS}",
]


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    """Rich wraps at the terminal width, which CI sets differently from a laptop."""
    monkeypatch.setattr("soup_cli.commands.deploy.console", Console(width=500, height=100))
    monkeypatch.setattr("soup_cli.utils.hf.resolve_token", lambda: "test-token")
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.setenv("COLUMNS", "500")
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)


def _plain(result) -> str:
    return " ".join(strip_ansi(result.output).split())


def _shown(value: str) -> str:
    return value.replace(_CONTROLS, "")


def _not_shown(text: str) -> bool:
    return "\x9b" not in text and "\u202e" not in text


def _deploy(*args: str, model: str = "u/m", space: str = "u/s"):
    return runner.invoke(
        app, ["deploy", "hf-space", "--model", model, "--space", space, *args]
    )


def _refusal(repo_id: str) -> str:
    with pytest.raises(ValueError) as raised:
        validate_repo_id(repo_id)
    return str(raised.value)


def _api(monkeypatch, api):
    monkeypatch.setattr("soup_cli.utils.hf.get_hf_api", lambda **kwargs: api)


@pytest.mark.parametrize("value", _SPELLINGS)
def test_unknown_template_is_named_as_typed(value):
    result = _deploy("--template", value)

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Unknown template: {_shown(value)} Available: gradio-chat, streamlit-chat" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _BAD_REPO_IDS)
def test_refused_space_id_is_shown_as_typed(value):
    result = _deploy(space=value)

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Invalid --space repo id: {_refusal(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _BAD_REPO_IDS)
def test_refused_model_id_is_shown_as_typed(value):
    result = _deploy(model=value)

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Template render failed: {_refusal(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _PATH_SPELLINGS)
def test_template_dir_outside_cwd_is_shown_as_typed(name):
    typed = f"../{name}"

    result = _deploy("--template-dir", typed)

    assert result.exit_code == 1
    text = _plain(result)
    assert (
        "Template render failed: template-dir must stay under the current working "
        f"directory; got: {typed!r}"
    ) in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _PATH_SPELLINGS)
def test_missing_template_dir_is_named_as_typed(name):
    result = _deploy("--template-dir", name)

    assert result.exit_code == 1
    text = _plain(result)
    assert (
        "Template directory error: template-dir does not exist or is not a directory: "
        f"{_shown(name)}"
    ) in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _PATH_SPELLINGS)
def test_template_dir_is_named_as_typed_in_the_plan(monkeypatch, name):
    template = Path(name)
    template.mkdir(parents=True)
    (template / "app.py").write_text("print('{MODEL_REPO}')\n", encoding="utf-8")
    (template / "README.md").write_text("# {MODEL_REPO}\n", encoding="utf-8")
    _api(monkeypatch, MagicMock())

    result = _deploy("--template-dir", name, "--yes")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"Template: {_shown(name)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _SPELLINGS)
def test_remote_error_is_shown_as_typed(monkeypatch, value):
    api = MagicMock()
    api.create_repo.side_effect = RuntimeError(f"hub said {value}")
    _api(monkeypatch, api)

    result = _deploy("--yes")

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Space deploy failed: hub said {_shown(value)}" in text
    assert _not_shown(text)
