"""Display a missing explicit llama.cpp path literally during GGUF preflight."""

import subprocess
from pathlib import Path

import pytest
from rich.console import Console

from soup_cli.cli import app
from soup_cli.commands import export
from tests.conftest import make_cli_runner, strip_ansi


@pytest.mark.parametrize(
    ("path", "display"),
    [
        pytest.param("missing-llama.cpp", "missing-llama.cpp", id="plain"),
        pytest.param("xn[red]llama.cpp", "xn[red]llama.cpp", id="markup"),
        pytest.param("xn[/]llama.cpp", "xn[/]llama.cpp", id="closing-markup"),
        pytest.param(r"missing\[unclosed.cpp", r"missing\[unclosed.cpp", id="backslash"),
        pytest.param("xn:warning:llama.cpp", "xn:warning:llama.cpp", id="emoji-alias"),
        pytest.param("xn\x9b\u202ellama.cpp", "xnllama.cpp", id="controls"),
    ],
)
def test_export_missing_llama_cpp_path_is_literal(path, display, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(export, "console", Console(width=200, height=50))
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        '{"architectures":["LlamaForCausalLM"]}', encoding="utf-8"
    )
    # In the controls case the display spelling names a different, existing
    # path. Cleaning the lookup as well would skip the missing-path refusal.
    (tmp_path / "xnllama.cpp").mkdir()

    def unexpected_subprocess(*args, **kwargs):
        pytest.fail("missing explicit --llama-cpp must refuse before running subprocesses")

    monkeypatch.setattr(subprocess, "run", unexpected_subprocess)
    result = make_cli_runner().invoke(
        app,
        ["export", "--model", "model", "--format", "gguf", "--llama-cpp", path],
    )

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert isinstance(result.exception, SystemExit), repr(result.exception)
    assert "\x9b" not in result.stdout
    assert "\u202e" not in result.stdout
    assert strip_ansi(result.stdout) == f"llama.cpp not found at: {Path(display)}\n"
    assert not list(tmp_path.glob("*.gguf"))
