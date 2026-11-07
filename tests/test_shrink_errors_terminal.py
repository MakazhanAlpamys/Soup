"""Shrink's early errors keep typed emoji codes and strip terminal controls."""

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import shrink as shrink_cmd
from soup_cli.utils.paths import quote_path
from tests.conftest import strip_ansi

runner = CliRunner()

_CONTROLS = "\x9b\u202e"

# No file with these names is created. Two characters before ':' avoid a
# Windows drive, and a prefix before '[' avoids backslash-markup interactions.
_CALIB_NAMES = [
    pytest.param("xn[red]calib.jsonl", id="tag-guard"),
    pytest.param("xn[/]calib.jsonl", id="closing-tag-guard"),
    pytest.param("xn:warning:.jsonl", id="emoji-code"),
    pytest.param(f"xn[bold]calib{_CONTROLS}.jsonl", id="repr-controls-guard"),
]
_OUTPUT_NAMES = [
    pytest.param("xn[red]out", id="tag-guard"),
    pytest.param("xn[/]out", id="closing-tag-guard"),
    pytest.param("xn:warning:out", id="emoji-code"),
    pytest.param(f"xn[bold]out{_CONTROLS}", id="tag-and-controls"),
]


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setattr(shrink_cmd, "console", Console(width=500, height=100))
    monkeypatch.setenv("COLUMNS", "500")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "registry.db"))
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    def _no_model_load(*args, **kwargs):
        raise AssertionError("early shrink validation must not inspect or load a model")

    # Both refusals precede even the trust-remote-code/model metadata probe.
    monkeypatch.setattr(shrink_cmd, "_resolve_trc", _no_model_load)
    monkeypatch.setattr(shrink_cmd, "_load_for_shrink", _no_model_load)


def _error_text(result) -> str:
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert isinstance(result.exception, SystemExit), repr(result.exception)
    # Inspect the captured bytes before any ANSI cleaning or whitespace folding.
    assert "\x9b" not in result.output
    assert "\u202e" not in result.output
    return " ".join(strip_ansi(result.output).split())


@pytest.mark.parametrize("name", _CALIB_NAMES)
def test_unreadable_calibration_is_named_as_typed(name):
    result = runner.invoke(
        app, ["shrink", "--model", "unused", "--drop-layers", "1", "--calib", name]
    )

    text = _error_text(result)
    assert "Error: calib path unreadable:" in text
    # OSError already repr-quotes its filename, escaping both controls. This is
    # a guard, not evidence that the new strip changes this producer's output.
    assert repr(name) in text


@pytest.mark.parametrize("name", _OUTPUT_NAMES)
def test_outside_output_is_named_as_typed(name):
    output = f"../{name}"
    result = runner.invoke(
        app,
        [
            "shrink", "--model", "unused", "--drop-layers", "1",
            "--calib", "missing.jsonl", "--output-dir", output,
        ],
    )

    text = _error_text(result)
    # quote_path escapes C1 as text, but leaves U+202E for the display to strip.
    quoted = quote_path(output).replace("\u202e", "")
    assert f"Error: --output-dir {quoted} must stay under cwd" in text
