"""Early draft refusals show typed text literally without terminal controls."""

import builtins
import socket

import pytest
from rich.console import Console

from soup_cli.cli import app
from tests._windows_filenames import skip_colon_on_windows
from tests.conftest import make_cli_runner, strip_ansi

runner = make_cli_runner()


@pytest.fixture(autouse=True)
def _early_refusal_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "runs.db"))
    monkeypatch.setattr("soup_cli.commands.draft.console", Console(width=200, height=40))
    attempted = []

    def unexpected_work(*args, **kwargs):
        attempted.append((args, kwargs))
        raise AssertionError("An early refusal must not resolve trust, load or run a model")

    for name in (
        "_resolve_trust", "_vocab_size_of", "_load_pair_member", "_run_distill",
        "register_draft",
    ):
        monkeypatch.setattr(f"soup_cli.commands.draft.{name}", unexpected_work)
    monkeypatch.setattr(socket.socket, "connect", unexpected_work)
    original_import = builtins.__import__

    def no_heavy_import(name, *args, **kwargs):
        if name.split(".")[0] in {
            "torch", "transformers", "peft", "trl", "mlx", "datasets", "accelerate",
            "bitsandbytes",
        }:
            attempted.append(name)
            raise AssertionError(f"An early refusal must not import {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_heavy_import)
    yield
    assert attempted == []


def _assert_refusal(result, expected):
    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit), repr(result.exception)
    assert result.exception.code == 1
    # Check raw output before ANSI cleanup could hide a control-character leak.
    assert "\x9b" not in result.stdout
    assert "\u202e" not in result.stdout
    assert strip_ansi(result.stdout) == expected + "\n"


@pytest.mark.parametrize("command", ["distill", "measure"])
@pytest.mark.parametrize(
    ("name", "shown"),
    [
        pytest.param("plain.jsonl", "plain.jsonl", id="plain-control"),
        pytest.param("x[red]n.jsonl", "x[red]n.jsonl", id="bracket-control"),
        pytest.param("[/]n.jsonl", "[/]n.jsonl", id="closing-tag-control"),
        pytest.param(
            "x:warning:n.jsonl", "x:warning:n.jsonl", id="emoji-code",
            marks=skip_colon_on_windows,
        ),
        pytest.param("x[bold]n\x9b\u202e.jsonl", "x[bold]n.jsonl", id="c1-bidi-with-tag"),
    ],
)
def test_empty_input_path_is_shown_as_typed(tmp_path, command, name, shown):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    if command == "distill":
        args = ["--draft-base", "unused-base", "--data", name, "-o", "draft-out", "--plan-only"]
        message = f"data file has no usable rows: {shown}"
    else:
        args = ["--draft", "unused-draft", "--prompts", name]
        message = f"prompts file yielded no usable prompt text: {shown}"

    result = runner.invoke(app, ["draft", command, "--target", "unused-target", *args])

    _assert_refusal(result, message)
    assert path.read_text(encoding="utf-8") == ""
    assert not (tmp_path / "draft-out").exists()


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("k", id="plain-control"),
        pytest.param("k[red]n", id="bracket-control"),
        pytest.param("k[/]n", id="closing-tag-control"),
        pytest.param("k:warning:", id="emoji-code"),
        pytest.param("k[bold]n\x9b\u202e", id="repr-escaped-control"),
    ],
)
def test_rejected_sweep_value_is_shown_as_typed(value):
    result = runner.invoke(
        app,
        ["draft", "measure", "--target", "unused-target", "--draft", "unused-draft",
         "--prompts", "never-read.jsonl", "--sweep-k", value],
    )

    # The parser already quotes this value with repr(), which escapes controls
    # but leaves emoji aliases intact. This case pins emoji handling on Windows too.
    _assert_refusal(result, f"--sweep-k values must be integers, got {value!r}")
