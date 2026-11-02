"""soup data topics prints paths, the embedding model and errors as typed, not as Rich markup."""

import json
import sys
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi

runner = CliRunner()

_CONTROLS = "\x9b\u202e"

# A colon is not a legal file-name character on Windows.
_NO_COLON_ON_WINDOWS = pytest.mark.skipif(
    sys.platform == "win32", reason="':' is not valid in a Windows name"
)

_NAMES = [
    "x[red]n.json",
    "x[bold]n.json",
    pytest.param("n:warning:.json", marks=_NO_COLON_ON_WINDOWS, id="emoji-code"),
]

_MISSING = [
    "x[red]n.jsonl",
    "x[bold]n.jsonl",
    pytest.param("n:warning:.jsonl", marks=_NO_COLON_ON_WINDOWS, id="emoji-code"),
]


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch):
    """Rich wraps at the terminal width, which CI sets differently from a laptop."""
    monkeypatch.setattr("soup_cli.commands.data_topics.console", Console(width=500))


@pytest.fixture(autouse=True)
def _workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def dataset() -> str:
    rows = ["red apples and pears", "green pears and plums", "stock market trading volume"]
    Path("data.jsonl").write_text(
        "\n".join(json.dumps({"text": row}) for row in rows) + "\n", encoding="utf-8"
    )
    return "data.jsonl"


@pytest.fixture
def embeddings(monkeypatch):
    monkeypatch.setattr(
        "soup_cli.commands.data_topics.embed_texts",
        lambda texts, **kwargs: [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
    )


def _plain(result) -> str:
    return " ".join(strip_ansi(result.output).split())


def _topics(*args: str):
    return runner.invoke(app, ["data", "topics", *args])


def _not_shown(text: str) -> bool:
    return "\x9b" not in text and "\u202e" not in text


@pytest.mark.parametrize("name", _MISSING)
def test_missing_dataset_is_named_as_typed(name):
    result = _topics(name)

    assert result.exit_code == 1
    assert f"File not found: {name}" in _plain(result)


def test_missing_dataset_strips_controls_from_the_display():
    result = _topics(f"x[bold]n{_CONTROLS}.jsonl")

    assert result.exit_code == 1
    text = _plain(result)
    assert "File not found: x[bold]n.jsonl" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_output_outside_cwd_is_named_as_typed(dataset, name):
    result = _topics(dataset, "--output", f"../{name}")

    assert result.exit_code == 1
    assert f"Output path is outside the working directory: ../{name}" in _plain(result)


def test_output_outside_cwd_strips_controls_from_the_display(dataset):
    result = _topics(dataset, "--output", f"../x[bold]o{_CONTROLS}.json")

    assert result.exit_code == 1
    text = _plain(result)
    assert "Output path is outside the working directory: ../x[bold]o.json" in text
    assert _not_shown(text)


@pytest.mark.parametrize(
    "model", ["x[red]m", "x[/]m", pytest.param("m:warning:", id="emoji-code")]
)
def test_embedding_model_is_named_as_typed(dataset, embeddings, model):
    result = _topics(dataset, "--embed-model", model)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert f"Embedding 3 rows with {model}..." in _plain(result)


def test_embedding_model_strips_controls_from_the_display(dataset, embeddings):
    result = _topics(dataset, "--embed-model", f"x[bold]m{_CONTROLS}")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert "Embedding 3 rows with x[bold]m..." in text
    assert _not_shown(text)


@pytest.mark.parametrize(
    "message", ["bad [/] model", "[red]refused", "stop:warning:", f"x[bold]m{_CONTROLS}"]
)
def test_embedding_error_is_shown_as_typed(dataset, monkeypatch, message):
    def _refuse(texts, **kwargs):
        raise ValueError(message)

    monkeypatch.setattr("soup_cli.commands.data_topics.embed_texts", _refuse)

    result = _topics(dataset)

    assert result.exit_code == 1
    text = _plain(result)
    assert message.replace(_CONTROLS, "") in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", [*_NAMES, f"x[bold]n{_CONTROLS}.json"])
def test_written_report_path_is_named_as_typed(dataset, embeddings, name):
    result = _topics(dataset, "--output", name)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"Report written: {name.replace(_CONTROLS, '')}" in text
    assert _not_shown(text)
    assert Path(name).is_file()
