"""soup docs search shows the typed query, the indexed text and its errors as typed."""

import json
import re

import pytest
from rich.console import Console

import soup_cli.docs.retriever as retriever
from soup_cli.cli import app
from soup_cli.commands import docs as docs_cmd
from tests.conftest import make_cli_runner, strip_ansi

runner = make_cli_runner()

_CONTROLS = "\x9b\u202e"

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


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    """Rich wraps at the terminal width, which CI sets differently from a laptop."""
    monkeypatch.setenv("COLUMNS", "500")
    monkeypatch.setattr(docs_cmd, "console", Console(width=500, height=100))
    monkeypatch.setattr(docs_cmd, "err_console", Console(width=500, height=100, stderr=True))


def _coloured(monkeypatch):
    settings = {
        "width": 500, "height": 100, "force_terminal": True, "no_color": False,
        "color_system": "standard",
    }
    monkeypatch.setattr(docs_cmd, "console", Console(**settings))
    monkeypatch.setattr(docs_cmd, "err_console", Console(stderr=True, **settings))


class _Index:
    """Stand in for the retriever, so the display is tested without an index."""

    def __init__(self, monkeypatch, results=None, error=None):
        self.calls = []
        self._results = [] if results is None else results
        self._error = error
        monkeypatch.setattr(retriever, "search_docs", self._search)

    def _search(self, query, *, top_k):
        self.calls.append((query, top_k))
        if self._error is not None:
            raise self._error
        return self._results


def _row(doc_name="guide", section="Intro", text="some text", score=1.2345):
    return {"doc_name": doc_name, "section": section, "text": text, "score": score}


def _search(*args: str):
    return runner.invoke(app, ["docs", "search", *args])


def _lines(result) -> list:
    """Every stdout line with its whitespace collapsed."""
    return [" ".join(line.split()) for line in strip_ansi(result.stdout).splitlines()]


def _shown(value: str) -> str:
    return "".join(ch for ch in value if ch not in _CONTROLS)


def _not_shown(*streams: str) -> bool:
    return all("\x9b" not in s and "\u202e" not in s for s in streams)


# --- no match ------------------------------------------------------------------


@pytest.mark.parametrize("value", _TEXTS)
def test_the_no_match_line_shows_the_query_as_typed(monkeypatch, value):
    index = _Index(monkeypatch)

    result = _search(value)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert _lines(result) == [f"No matching documentation found for: {_shown(value)}"]
    assert _not_shown(result.stdout)
    assert index.calls == [(value, 5)]


# --- the table -----------------------------------------------------------------


@pytest.mark.parametrize("value", _TEXTS)
def test_the_table_title_shows_the_query_as_typed(monkeypatch, value):
    index = _Index(monkeypatch, [_row()])

    result = _search(value, "--top-k", "3")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert f"Documentation Search: '{_shown(value)}'" in _lines(result)
    assert _not_shown(result.stdout)
    assert index.calls == [(value, 3)]


@pytest.mark.parametrize("value", _TEXTS)
def test_the_cells_show_the_indexed_text_as_stored(monkeypatch, value):
    _Index(monkeypatch, [_row(f"doc-{value}", f"sec-{value}", f"text {value}")])

    result = _search("q")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    shown = _shown(value)
    row = f"\u2502 doc-{shown} \u2502 sec-{shown} \u2502 1.2345 \u2502 text {shown} \u2502"
    assert row in _lines(result)
    assert _not_shown(result.stdout)


def test_a_long_excerpt_is_cut_before_its_controls_are_removed(monkeypatch):
    # 116 letters, a C1 control at the cut, then more text: cutting first keeps the control
    # inside the 117 characters, so the excerpt ends after the letters; stripping first
    # would let one more letter in.
    _Index(monkeypatch, [_row(text="x" * 116 + "\x9b" + "yz" * 20)])

    result = _search("q")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    row = [line for line in _lines(result) if "1.2345" in line]
    assert row == [f"\u2502 guide \u2502 Intro \u2502 1.2345 \u2502 {'x' * 116}... \u2502"]


def test_a_plain_table_keeps_its_shape(monkeypatch):
    _Index(monkeypatch, [_row("models", "The extras table", "Add pip install", 22.4966)])

    result = _search("pip install", "-k", "2")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    lines = strip_ansi(result.stdout).splitlines()
    assert lines[0].strip() == "Documentation Search: 'pip install'"
    # The header row uses a heavy bar that Rich swaps for a light one where the terminal cannot
    # draw it (Windows), so read the cell names whichever bar separates them.
    header = [" ".join(cell.split()) for cell in re.split("[\u2502\u2503]", lines[2])]
    assert [cell for cell in header if cell] == ["Doc", "Section", "Score", "Excerpt"]
    assert " ".join(lines[4].split()) == (
        "\u2502 models \u2502 The extras table \u2502 22.4966 \u2502 Add pip install \u2502"
    )


# --- refusals --------------------------------------------------------------------


@pytest.mark.parametrize("value", _TEXTS)
def test_an_unknown_format_is_shown_as_typed(monkeypatch, value):
    index = _Index(monkeypatch)

    result = _search("q", "--format", value)

    assert result.exit_code == 2, (result.output, repr(result.exception))
    error = " ".join(strip_ansi(result.stderr).split())
    assert error == f"Error: Unknown --format: '{_shown(value)}'. Choose 'table' or 'json'."
    assert _not_shown(result.stderr, result.stdout)
    assert index.calls == []


@pytest.mark.parametrize("error", [RuntimeError, ValueError, FileNotFoundError, OSError])
@pytest.mark.parametrize("value", _TEXTS)
def test_a_retriever_error_is_shown_as_typed(monkeypatch, error, value):
    _Index(monkeypatch, error=error(f"docs said {value}"))

    result = _search("q")

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert " ".join(strip_ansi(result.stderr).split()) == f"Error: docs said {_shown(value)}"
    assert _not_shown(result.stderr, result.stdout)


def test_a_zero_top_k_keeps_its_refusal(monkeypatch):
    index = _Index(monkeypatch)

    result = _search("q", "--top-k", "0")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert strip_ansi(result.stderr).strip() == (
        "Error: --top-k must be greater than or equal to 1, got 0."
    )
    assert index.calls == []


def test_the_json_output_is_the_results_untouched(monkeypatch):
    rows = [_row("x[red]n", "n:warning:", "a\u202eb")]
    _Index(monkeypatch, rows)

    result = _search("q", "--format", "json")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert json.loads(result.stdout) == rows


# --- styles ------------------------------------------------------------------------


def test_the_error_labels_are_bold_red_and_the_no_match_line_dim_and_italic(monkeypatch):
    _coloured(monkeypatch)
    _Index(monkeypatch)

    empty = _search("my query")
    bad_format = _search("q", "--format", "yaml")

    assert empty.stdout == (
        "\x1b[2mNo matching documentation found for:\x1b[0m \x1b[3mmy query\x1b[0m\n"
    )
    assert bad_format.stderr == (
        "\x1b[1;31mError:\x1b[0m Unknown --format: 'yaml'. Choose 'table' or 'json'.\n"
    )


def test_a_retriever_error_label_is_bold_red(monkeypatch):
    _coloured(monkeypatch)
    _Index(monkeypatch, error=RuntimeError("index broken"))

    result = _search("q")

    assert result.stderr == "\x1b[1;31mError:\x1b[0m index broken\n"


def test_the_table_title_is_italic_and_the_columns_keep_their_styles(monkeypatch):
    _coloured(monkeypatch)
    _Index(monkeypatch, [_row("models", "The extras table", "Add pip install", 22.4966)])

    result = _search("pip install")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    title = result.stdout.splitlines()[0]
    assert "\x1b[3m" in title
    assert "Documentation Search: 'pip install'" in strip_ansi(title)
    body = result.stdout.splitlines()[4]
    assert "\x1b[36mmodels\x1b[0m" in body
    assert "\x1b[1;32mThe extras table\x1b[0m" in body
    assert "\x1b[33m22.4966\x1b[0m" in body
    assert "\x1b[2mAdd pip install\x1b[0m" in body
