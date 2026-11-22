"""A refused soup why run reference is displayed literally, with controls removed."""

import pytest
from rich.console import Console

from soup_cli.cli import app
from tests.conftest import make_cli_runner, strip_ansi

runner = make_cli_runner()


@pytest.fixture(autouse=True)
def _isolated_console_and_tracker(tmp_path, monkeypatch):
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "runs.db"))
    monkeypatch.setattr("soup_cli.commands.why.console", Console(width=200, height=40))


@pytest.mark.parametrize(
    ("run_id", "shown"),
    [
        pytest.param("missing-run", "missing-run", id="plain-control"),
        pytest.param("missing[red]run", "missing[red]run", id="bracket-control"),
        pytest.param("missing[/]run", "missing[/]run", id="closing-tag-control"),
        pytest.param("missing:warning:run", "missing:warning:run", id="emoji-code"),
        pytest.param(
            "missing[bold]run\x9b\u202e", "missing[bold]run", id="c1-bidi-with-tag"
        ),
    ],
)
def test_refused_run_reference_is_shown_as_typed(run_id, shown):
    result = runner.invoke(app, ["why", run_id])

    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit), repr(result.exception)
    assert result.exception.code == 1
    # Check the captured bytes' characters before any ANSI cleaning can hide a leak.
    assert "\x9b" not in result.stdout
    assert "\u202e" not in result.stdout
    assert strip_ansi(result.stdout) == f"Run not found: {shown}\n"
