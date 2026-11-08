"""Missing export model paths remain literal in the early refusal."""

from pathlib import Path

import pytest
from rich.console import Console

from soup_cli.cli import app
from soup_cli.commands import export
from tests.conftest import make_cli_runner, strip_ansi


@pytest.mark.parametrize(
    ("path", "display"),
    [
        pytest.param("missing-model", "missing-model", id="plain"),
        pytest.param("xn[red]model", "xn[red]model", id="markup"),
        pytest.param("xn[/]model", "xn[/]model", id="closing-markup"),
        pytest.param(r"missing\[model", r"missing\[model", id="backslash"),
        pytest.param("xn:warning:model", "xn:warning:model", id="emoji-alias"),
        pytest.param("xn\x9b\u202emodel", "xnmodel", id="controls"),
    ],
)
def test_export_missing_model_path_is_literal(path, display, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(export, "console", Console(width=200, height=50))
    # The cleaned display spelling is a real directory. It must not become
    # the model lookup target when the typed path contains controls.
    (tmp_path / "xnmodel").mkdir()

    def unexpected_preflight(*args, **kwargs):
        pytest.fail("a missing model must refuse before inspecting the converter")

    monkeypatch.setattr(export, "_check_gguf_capabilities", unexpected_preflight)
    result = make_cli_runner().invoke(app, ["export", "--model", path, "--format", "gguf"])

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert isinstance(result.exception, SystemExit), repr(result.exception)
    assert "\x9b" not in result.stdout
    assert "\u202e" not in result.stdout
    assert strip_ansi(result.stdout) == f"Model path not found: {Path(display)}\n"
    assert not list(tmp_path.glob("*.gguf"))
