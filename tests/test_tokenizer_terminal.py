"""soup tokenizer train shows typed paths, file names and operating-system errors as typed."""

import json
import os
import re
import sys
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import tokenizer as tokenizer_cmd
from tests._windows_filenames import skip_colon_on_windows
from tests.conftest import strip_ansi

runner = CliRunner()

_CONTROLS = "\x9b\u202e"

# A backslash separates directories on Windows, so a name that holds one cannot be a file there.
_NO_BACKSLASH_ON_WINDOWS = pytest.mark.skipif(
    sys.platform == "win32", reason="a backslash separates directories on Windows"
)

# One spelling per way the old code went wrong: a tag, a closing tag with
# nothing to close (a MarkupError), an emoji code, and a tag with control
# characters, so the strip and the escape are pinned separately. A backslash before
# an unmatched bracket and a trailing pair of backslashes are what markup rewrites.
_TEXTS = [
    "x[red]n",
    "[/]n",
    pytest.param("n:warning:", id="emoji-code"),
    pytest.param("x\\[n", id="backslash-bracket"),
    pytest.param("n\\\\", id="trailing-backslashes"),
    f"x[bold]n{_CONTROLS}",
]
# The same spellings typed as a path that does not exist. Windows reads `n:` as a drive.
_VALUES = [
    "x[red]n",
    "[/]n",
    pytest.param("n:warning:", marks=skip_colon_on_windows, id="emoji-code"),
    pytest.param("x\\[n", id="backslash-bracket"),
    pytest.param("n\\\\", id="trailing-backslashes"),
    f"x[bold]n{_CONTROLS}",
]
# A file name cannot hold a slash, so no closing tag with nothing to close.
_NAMES = [
    "x[red]n",
    pytest.param("n:warning:", marks=skip_colon_on_windows, id="emoji-code"),
    pytest.param("x\\[n", marks=_NO_BACKSLASH_ON_WINDOWS, id="backslash-bracket"),
    pytest.param("n\\\\", marks=_NO_BACKSLASH_ON_WINDOWS, id="trailing-backslashes"),
    f"x[bold]n{_CONTROLS}",
]

_CORPUS = ["hello world number 1", "hello world number 2", "the quick brown fox", "number 3"]


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    """Rich wraps at the terminal width, which CI sets differently from a laptop."""
    monkeypatch.setattr(tokenizer_cmd, "console", Console(width=500, height=100))
    monkeypatch.setenv("COLUMNS", "500")
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)


@pytest.fixture
def tokenizers_lib():
    """The commands that reach the trainer need the ``tokenizers`` package."""
    return pytest.importorskip("tokenizers")


def _coloured(monkeypatch):
    monkeypatch.setattr(
        tokenizer_cmd,
        "console",
        Console(
            width=500, height=100, force_terminal=True, no_color=False, color_system="standard"
        ),
    )


def _write_corpus(path: str) -> None:
    rows = "".join(json.dumps({"text": text}) + "\n" for text in _CORPUS)
    Path(path).write_text(rows, encoding="utf-8")


def _train(*, input_path: str = "corpus.jsonl", output: str = "out"):
    return runner.invoke(
        app,
        [
            "tokenizer", "train", "--input", input_path, "--vocab-size", "300",
            "--output", output,
        ],
    )


def _plain(result) -> str:
    return " ".join(strip_ansi(result.output).split())


def _lines(result) -> list:
    """Every output line, its whitespace collapsed and the panel's side borders cut off."""
    lines = strip_ansi(result.output).splitlines()
    return [" ".join(line.strip("\u2502").split()) for line in lines]


def _shown(value: str) -> str:
    return "".join(ch for ch in value if ch not in _CONTROLS)


def _not_shown(text: str) -> bool:
    return "\x9b" not in text and "\u202e" not in text


class _Os:
    """The ``os`` module as the command sees it, with an ``lstat`` that fails for one name."""

    def __init__(self, name: str, message: str) -> None:
        self._name = name
        self._message = message

    def __getattr__(self, attribute):
        return getattr(os, attribute)

    def lstat(self, path, *args, **kwargs):
        if Path(path).name == self._name:
            raise OSError(self._message)
        return os.lstat(path, *args, **kwargs)


@pytest.mark.parametrize("value", _VALUES)
def test_a_missing_input_file_is_shown_as_typed(value):
    result = _train(input_path=value)

    assert result.exit_code == 1, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"Input file not found: {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_a_file_as_the_output_directory_is_shown_as_typed(name):
    _write_corpus("corpus.jsonl")
    Path(name).write_text("keep me\n", encoding="utf-8")

    result = _train(output=name)

    assert result.exit_code == 1, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"Output path is not a directory: {_shown(name)}" in text
    assert "BPE Tokenizer Training" not in text
    assert _not_shown(text)
    assert Path(name).read_text(encoding="utf-8") == "keep me\n"


@pytest.mark.usefixtures("tokenizers_lib")
@pytest.mark.parametrize("name", _NAMES)
def test_an_empty_corpus_is_shown_by_its_file_name_as_typed(name):
    Path("data").mkdir()
    Path("data", f"{name}.jsonl").write_text("", encoding="utf-8")

    result = _train(input_path=f"data/{name}.jsonl")

    assert result.exit_code == 1, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"No usable training text found in {_shown(name)}.jsonl" in text
    assert "BPE Tokenizer Training" not in text
    assert _not_shown(text)


@pytest.mark.usefixtures("tokenizers_lib")
@pytest.mark.parametrize("name", _NAMES)
def test_the_plan_panel_shows_the_file_and_directory_names_as_typed(name):
    Path("data").mkdir()
    _write_corpus(f"data/{name}.jsonl")

    result = _train(input_path=f"data/{name}.jsonl", output=f"build/out-{name}")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    lines = _lines(result)
    shown = _shown(name)
    assert f"Input: {shown}.jsonl" in lines
    assert "Texts: 4" in lines
    assert "Vocab size: 300" in lines
    assert f"Output dir: out-{shown}" in lines
    assert _not_shown(_plain(result))


@pytest.mark.usefixtures("tokenizers_lib")
@pytest.mark.parametrize("name", _NAMES)
def test_the_receipt_shows_the_output_directory_as_typed(name):
    _write_corpus("corpus.jsonl")

    result = _train(output=f"build/out-{name}")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    directory = Path("build", f"out-{name}")
    tokens = len(json.loads((directory / "vocab.json").read_text(encoding="utf-8")))
    receipt = f"Trained tokenizer with {tokens} tokens -> out-{_shown(name)}/tokenizer.json"
    assert receipt in _lines(result)
    assert _not_shown(_plain(result))


@pytest.mark.usefixtures("tokenizers_lib")
@pytest.mark.parametrize("value", _TEXTS)
def test_an_unreadable_output_directory_is_shown_as_typed(monkeypatch, value):
    _write_corpus("corpus.jsonl")
    monkeypatch.setattr(tokenizer_cmd, "os", _Os("out", f"stat said {value}"))

    result = _train()

    assert result.exit_code == 1, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"Output directory not accessible: stat said {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.usefixtures("tokenizers_lib")
def test_a_plain_run_prints_the_same_panel_and_receipt():
    _write_corpus("corpus.jsonl")

    result = _train()

    assert result.exit_code == 0, (result.output, repr(result.exception))
    lines = strip_ansi(result.output).splitlines()
    assert any("BPE Tokenizer Training" in line for line in lines)
    assert "Input:        corpus.jsonl" in strip_ansi(result.output)
    assert "Texts:        4" in strip_ansi(result.output)
    assert "Vocab size:   300" in strip_ansi(result.output)
    assert "Output dir:   out" in strip_ansi(result.output)
    receipt = r"^Trained tokenizer with \d+ tokens -> out/tokenizer\.json$"
    assert re.search(receipt, strip_ansi(result.output), re.M)
    assert (Path("out") / "tokenizer.json").is_file()
    assert (Path("out") / "vocab.json").is_file()


def test_a_plain_missing_input_file_and_a_plain_file_as_output_keep_their_text():
    _write_corpus("corpus.jsonl")
    Path("old.json").write_text("{}\n", encoding="utf-8")

    missing = _train(input_path="missing.jsonl")
    clash = _train(output="old.json")

    assert strip_ansi(missing.output) == "Input file not found: missing.jsonl\n"
    assert strip_ansi(clash.output) == "Output path is not a directory: old.json\n"
    assert (missing.exit_code, clash.exit_code) == (1, 1)


# --- styles ----------------------------------------------------------------


def test_the_refusals_keep_their_red_labels(monkeypatch):
    _coloured(monkeypatch)
    _write_corpus("corpus.jsonl")
    Path("old.json").write_text("{}\n", encoding="utf-8")

    missing = _train(input_path="missing.jsonl")
    clash = _train(output="old.json")

    assert missing.output == "\x1b[31mInput file not found:\x1b[0m missing.jsonl\n"
    assert clash.output == "\x1b[31mOutput path is not a directory:\x1b[0m old.json\n"


@pytest.mark.usefixtures("tokenizers_lib")
def test_the_empty_corpus_and_unreadable_directory_refusals_keep_their_red_labels(monkeypatch):
    _coloured(monkeypatch)
    Path("empty.jsonl").write_text("", encoding="utf-8")
    _write_corpus("corpus.jsonl")

    empty = _train(input_path="empty.jsonl")
    monkeypatch.setattr(tokenizer_cmd, "os", _Os("out", "stat said no"))
    unreadable = _train()

    assert empty.output == "\x1b[31mNo usable training text found in\x1b[0m empty.jsonl\n"
    assert unreadable.output.endswith(
        "\x1b[31mOutput directory not accessible:\x1b[0m stat said no\n"
    )


@pytest.mark.usefixtures("tokenizers_lib")
def test_the_panel_and_the_receipt_keep_their_bold_values_and_green_label(monkeypatch):
    _coloured(monkeypatch)
    _write_corpus("corpus.jsonl")

    result = _train()

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "Input:        \x1b[1mcorpus.jsonl\x1b[0m" in result.output
    assert "Texts:        \x1b[1m4\x1b[0m" in result.output
    assert "Vocab size:   \x1b[1m300\x1b[0m" in result.output
    assert "Output dir:   \x1b[1mout\x1b[0m" in result.output
    assert re.search(
        r"\x1b\[32mTrained tokenizer with\x1b\[0m \x1b\[1m\d+\x1b\[0m tokens -> "
        r"\x1b\[1mout/tokenizer\.json\x1b\[0m\n",
        result.output,
    )
