"""soup license-advisor prints a typed target and license as typed, not as Rich markup."""

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi

runner = CliRunner()

_CONTROLS = "\x9b\u202e"

# One spelling per way the old code went wrong: a tag, a closing tag with
# nothing to close (a MarkupError), an emoji code, and a tag with control
# characters, so the strip and the escape are pinned separately.
_VALUES = [
    "x[red]n",
    "[/]n",
    pytest.param("n:warning:", id="emoji-code"),
    f"x[bold]n{_CONTROLS}",
]


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    """Rich wraps at the terminal width, which CI sets differently from a laptop."""
    monkeypatch.setattr(
        "soup_cli.commands.license_advisor.console", Console(width=500, height=100)
    )
    monkeypatch.setenv("COLUMNS", "500")


def _plain(result) -> str:
    return " ".join(strip_ansi(result.output).split())


def _shown(value: str) -> str:
    return "".join(ch for ch in value if ch not in _CONTROLS)


def _not_shown(text: str) -> bool:
    return "\x9b" not in text and "\u202e" not in text


def _advise(*args: str):
    return runner.invoke(app, ["license-advisor", *args])


@pytest.mark.parametrize("value", _VALUES)
def test_an_unknown_target_is_shown_as_typed(value):
    result = _advise("--target", value)

    assert result.exit_code == 2
    text = _plain(result)
    assert f"unknown target {value!r}; known: b2c, defense, embedded" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_an_unknown_license_is_quoted_as_typed_in_the_reason(value):
    result = _advise("--target", "b2c", "--license", value)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"unknown license {value.lower().strip()!r}; verify with legal" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_the_risk_title_names_the_license_as_typed(value):
    result = _advise("--target", "b2c", "--license", value)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    titles = [line for line in strip_ansi(result.output).splitlines() if "Risk check:" in line]
    assert len(titles) == 1
    assert f"Risk check: {_shown(value)} " in " ".join(titles[0].split())
    assert _not_shown(titles[0])
