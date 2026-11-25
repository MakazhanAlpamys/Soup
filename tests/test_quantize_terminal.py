"""The export command printed by quantize can be copied without changing its arguments."""

import shlex

import pytest
from rich.console import Console

from soup_cli.cli import app
from tests.conftest import make_cli_runner, strip_ansi

runner = make_cli_runner()
VALUES = ["plain", "x[red]n", "[/]n", "n:warning:", "x\\[n", "n\\\\", "x\x9b\u202en"]


@pytest.mark.parametrize("value", VALUES)
@pytest.mark.parametrize("fmt", ["gguf", "onnx", "tensorrt"])
@pytest.mark.parametrize("width", [48, 500])
def test_export_arguments_survive_the_printed_command(monkeypatch, value, fmt, width):
    monkeypatch.setattr("soup_cli.commands.quantize.console", Console(width=width, height=100))
    model = f"model {value}" + "m" * 80
    output = f"output {value}" + "o" * 80

    result = runner.invoke(app, ["quantize", model, "--to", fmt, "--output", output])

    assert result.exit_code == 0, result.output
    command = next(
        line.strip()
        for line in strip_ansi(result.output).splitlines()
        if line.strip().startswith("soup export ")
    )
    expected = ["soup", "export", "--model", model.replace("\x9b\u202e", ""), "--format", fmt]
    if fmt == "gguf":
        expected += ["--quant", "q4_K_M"]
    expected += ["--output", output.replace("\x9b\u202e", "")]
    assert shlex.split(command) == expected
    assert "\x9b" not in result.output and "\u202e" not in result.output


@pytest.mark.parametrize("value", VALUES)
def test_invalid_format_is_named_literally(monkeypatch, value):
    monkeypatch.setattr("soup_cli.commands.quantize.console", Console(width=500, height=100))

    result = runner.invoke(app, ["quantize", "model", "--to", value])

    assert result.exit_code == 2
    assert strip_ansi(result.output).strip() == (
        f"--to must be one of ['gguf', 'onnx', 'tensorrt']; got {value!r}"
    )


def test_command_and_refusal_keep_their_styles(monkeypatch):
    monkeypatch.setattr(
        "soup_cli.commands.quantize.console",
        Console(
            width=500, height=100, force_terminal=True, no_color=False, color_system="standard"
        ),
    )
    good = runner.invoke(app, ["quantize", "model"])
    bad = runner.invoke(app, ["quantize", "model", "--to", "nope"])
    assert good.exit_code == 0 and bad.exit_code == 2
    assert "\x1b[1m  soup export --model model --format gguf --quant q4_K_M\x1b[0m" in good.output
    assert (
        "\x1b[31m--to must be one of ['gguf', 'onnx', 'tensorrt']; got 'nope'\x1b[0m" in bad.output
    )
