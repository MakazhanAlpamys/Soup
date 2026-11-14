"""Diff's display preserves literal text without changing generation or JSONL."""

import copy
import io
import json
import re
from pathlib import Path

import pytest
import typer
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import diff as command
from tests._windows_filenames import skip_colon_on_windows
from tests.conftest import strip_ansi

VALUES = [
    pytest.param("ordinary", "ordinary", id="plain-control"),
    pytest.param(":rocket:", ":rocket:", id="emoji-alias"),
    pytest.param("x[red]name", "x[red]name", id="markup"),
    pytest.param(r"x\[unfinished", r"x\[unfinished", id="backslash"),
    pytest.param("x\x1b\x07\x9b\u202e[end]", "x[end]", id="controls"),
]
FILE_VALUES = [
    pytest.param(*case.values, id=case.id,
                 marks=skip_colon_on_windows if ":" in case.values[0] else ())
    for case in VALUES
]


def _plain(text):
    text = strip_ansi(text)
    assert not any(char in text for char in ("\x1b", "\x07", "\x9b", "\u202e")), repr(text)
    return " ".join(text.split())


def _filename(value):
    # C0 is tested in messages; it cannot occur in a real Windows filename.
    return "item-" + value.replace("\x1b", "").replace("\x07", "")


@pytest.fixture
def printed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    stream = io.StringIO()
    monkeypatch.setattr(command, "console", Console(
        file=stream, width=700, height=200, force_terminal=True,
        color_system="standard", no_color=False,
    ))
    return stream


@pytest.mark.parametrize("value,shown", FILE_VALUES)
@pytest.mark.parametrize("kind", ["model-a", "model-b", "missing", "directory", "empty", "read"])
def test_diff_file_refusals_are_literal_and_do_not_load_models(
    printed, monkeypatch, kind, value, shown
):
    path = Path(_filename(value))
    display = str(Path(_filename(shown)))
    a, b = "model-a", "model-b"
    Path(a).mkdir()
    Path(b).mkdir()
    args = ["diff", "--model-a", a, "--model-b", b]
    if kind in {"model-a", "model-b"}:
        args[args.index("--" + kind) + 1] = str(path)
        expected = f"Model {kind[-1].upper()} not found: {display}"
    else:
        args += ["--prompts", str(path)]
        if kind == "directory":
            path.mkdir(parents=True)
            expected = f"Prompts path is not a file: {display}"
        elif kind == "missing":
            expected = f"Prompts file not found: {display}"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
            if kind == "read":
                def unreadable(*args):
                    raise OSError(f"Unreadable {value}")

                monkeypatch.setattr(command, "_collect_prompts", unreadable)
                expected = f"Cannot read prompts file {display}: Unreadable {shown}"
            else:
                expected = f"No prompts found in file: {display}"

    def unexpected(*args, **kwargs):
        pytest.fail("Refused input must not load a model")

    monkeypatch.setattr(command, "_load_model", unexpected)
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1, (result.output, result.exception)
    assert _plain(printed.getvalue()) == expected
    assert "\x1b[31m" in printed.getvalue()
    if kind in {"read", "empty"}:
        assert path.read_bytes() == b""


@pytest.mark.parametrize("value,shown", VALUES)
def test_diff_prompts_and_replies_are_literal_but_generation_and_jsonl_keep_originals(
    printed, monkeypatch, value, shown
):
    loads, generations = [], []
    Path("a").mkdir()
    Path("b").mkdir()
    prompt = f"Prompt {value}"
    replies = [f"A {value}", f"B {value}"]

    def load(path, base, device, trust):
        loads.append((path, base, device, trust))
        return path, "tokenizer"

    def generate(model, tokenizer, messages, **options):
        generations.append((model, tokenizer, copy.deepcopy(messages), options))
        return replies[len(generations) - 1]

    monkeypatch.setattr(command, "_load_model", load)
    monkeypatch.setattr(command, "_generate", generate)
    result = CliRunner().invoke(app, [
        "diff", "-a", "a", "-b", "b", "--prompt", prompt, "--device", "cpu",
        "--base-a", value, "--base-b", value, "--max-tokens", "9",
        "--temperature", "0", "--output", "result.jsonl", "--trust-remote-code",
    ])
    assert result.exit_code == 0, (result.output, result.exception)
    text = _plain(printed.getvalue())
    for expected in [f"Prompt {shown}", f"A {shown}", f"B {shown}",
                     "Model A: a", "Model B: b", "Model A (a)", "Model B (b)",
                     "Results saved to result.jsonl"]:
        assert expected in text
    assert loads == [("a", value, "cpu", True), ("b", value, "cpu", True)]
    assert generations == [
        (name, "tokenizer", [{"role": "user", "content": prompt}],
         {"max_tokens": 9, "temperature": 0.0})
        for name in ["a", "b"]
    ]
    assert json.loads(Path("result.jsonl").read_text()) == {
        "prompt": prompt, "response_a": replies[0], "response_b": replies[1],
        "metrics": command._compute_metrics(*replies),
    }
    assert "\x1b[2mPrompt " in printed.getvalue()


@pytest.mark.parametrize("value,shown", FILE_VALUES)
def test_diff_model_titles_summary_and_output_receipt_keep_literal_paths(
    printed, monkeypatch, value, shown
):
    paths = [Path(prefix + _filename(value)) for prefix in ["a-", "b-"]]
    output = Path("out-" + _filename(value) + ".jsonl")
    for path in paths:
        path.mkdir(parents=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    loads = []

    def load(path, *args):
        loads.append(path)
        return object(), object()

    monkeypatch.setattr(command, "_load_model", load)
    monkeypatch.setattr(command, "_generate", lambda *args, **kwargs: "")
    result = CliRunner().invoke(app, [
        "diff", "-a", str(paths[0]), "-b", str(paths[1]), "--device", "cpu",
        "--prompt", "plain", "--output", str(output),
    ])
    assert result.exit_code == 0, (result.output, result.exception)
    text = _plain(printed.getvalue())
    for label, path in zip(["A", "B"], paths):
        displayed = str(path).replace(_filename(value), _filename(shown))
        basename = path.name.replace(_filename(value), _filename(shown))
        # One occurrence in the plan and one in the response-panel title.
        assert text.count(f"Model {label}: {displayed}") == 2
        assert f"Model {label} ({basename})" in text
    assert f"Results saved to {str(output).replace(_filename(value), _filename(shown))}" in text
    assert "<empty>" in text
    assert loads == list(map(str, paths))
    assert json.loads(output.read_text())["response_a"] == ""
    spans = re.findall(r"\x1b\[([\d;]+)m([^\x1b]*)", printed.getvalue())
    for label, colour in [("A", "34"), ("B", "32")]:
        assert any(colour in codes.split(";") and f"Model {label}:" in text
                   for codes, text in spans)


@pytest.mark.parametrize("value,shown", FILE_VALUES)
def test_diff_missing_adapter_base_refusal_preserves_path(printed, value, shown):
    path = Path(_filename(value))
    path.mkdir(parents=True)
    metadata = path / "adapter_config.json"
    metadata.write_text("{}", encoding="utf-8")
    with pytest.raises(typer.Exit) as error:
        command._load_model(str(path), None, "cpu")
    assert error.value.exit_code == 1
    assert _plain(printed.getvalue()) == (
        f"Cannot detect base model for {str(path).replace(_filename(value), _filename(shown))}. "
        "Use --base-a/--base-b."
    )
    assert metadata.read_bytes() == b"{}"
