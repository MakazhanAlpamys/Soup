"""soup eval citation shows the typed style, the typed output path and their errors as typed."""

import json
import sys
from pathlib import Path

import pytest
import typer
from rich.console import Console
from typer.testing import CliRunner

import soup_cli.utils.citation_faithful as citation_faithful
import soup_cli.utils.paths as paths_module
from soup_cli.commands._eval_v07110 import register
from soup_cli.utils.paths import quote_path
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
# A file name cannot hold a slash, so no closing tag with nothing to close.
_NAMES = [
    "x[red]n",
    pytest.param("n:warning:", marks=skip_colon_on_windows, id="emoji-code"),
    pytest.param("x\\[n", marks=_NO_BACKSLASH_ON_WINDOWS, id="backslash-bracket"),
    pytest.param("n\\\\", marks=_NO_BACKSLASH_ON_WINDOWS, id="trailing-backslashes"),
    f"x[bold]n{_CONTROLS}",
]

_ROWS = [
    {"predicted": "see [doc-1]", "expected_ids": ["doc-1"]},
    {"predicted": "see [doc-9]", "expected_ids": ["doc-1"]},
]


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    """Rich wraps at the terminal width, which CI sets differently from a laptop."""
    monkeypatch.setenv("COLUMNS", "500")
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    rows = "".join(json.dumps(row) + "\n" for row in _ROWS)
    (workdir / "data.jsonl").write_text(rows, encoding="utf-8")


def _app(console: Console) -> typer.Typer:
    app = typer.Typer()

    @app.callback()
    def main() -> None:
        """Keep ``citation`` a named sub-command."""

    register(app, console)
    return app


def _plain_console() -> Console:
    return Console(width=500, height=100)


def _coloured_console() -> Console:
    return Console(
        force_terminal=True, no_color=False, width=500, height=100, color_system="standard"
    )


def _run(*args: str, console: Console = None):
    return runner.invoke(_app(console or _plain_console()), ["citation", "data.jsonl", *args])


def _plain(result) -> str:
    return " ".join(strip_ansi(result.output).split())


def _shown(value: str) -> str:
    return "".join(ch for ch in value if ch not in _CONTROLS)


def _not_shown(text: str) -> bool:
    return "\x9b" not in text and "\u202e" not in text


# --- --style -----------------------------------------------------------------


@pytest.mark.parametrize("value", _TEXTS)
def test_a_refused_style_is_shown_as_typed(value):
    result = _run("--style", value)

    assert result.exit_code == 2, (result.output, repr(result.exception))
    text = _plain(result)
    # The validator quotes the typed style with repr(), which already escapes controls.
    refusal = (
        f"Invalid --style: unknown citation_style {value!r}; supported: bracket, footnote, inline"
    )
    assert refusal in text
    assert _not_shown(text)


@pytest.mark.parametrize("error", [TypeError, ValueError])
@pytest.mark.parametrize("value", _TEXTS)
def test_a_style_refusal_with_any_text_is_shown_as_typed(monkeypatch, error, value):
    def refuse(style):
        raise error(f"style said {value}")

    monkeypatch.setattr(citation_faithful, "validate_citation_style", refuse)

    result = _run()

    assert result.exit_code == 2, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"Invalid --style: style said {_shown(value)}" in text
    assert _not_shown(text)


# --- --output ----------------------------------------------------------------


@pytest.mark.parametrize("name", _NAMES)
def test_a_refused_output_path_is_shown_as_typed(tmp_path, name):
    target = str(tmp_path / "elsewhere" / f"{name}.json")

    result = _run("--output", target)

    assert result.exit_code == 2, (result.output, repr(result.exception))
    text = _plain(result)
    # quote_path escapes C0 and C1 controls and leaves U+202E and aliases alone.
    quoted = _shown(quote_path(target))
    expected = f"Invalid --output: citation output {quoted} must stay under cwd"
    assert expected in text
    assert _not_shown(text)
    assert not (tmp_path / "elsewhere").exists()


@pytest.mark.parametrize("error", [TypeError, ValueError])
@pytest.mark.parametrize("value", _TEXTS)
def test_an_output_refusal_with_any_text_is_shown_as_typed(monkeypatch, error, value):
    real = paths_module.enforce_under_cwd_and_no_symlink

    def refuse(path, field):
        if field == "citation output":
            raise error(f"output said {value}")
        return real(path, field)

    monkeypatch.setattr(paths_module, "enforce_under_cwd_and_no_symlink", refuse)

    result = _run("--output", "report.json")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"Invalid --output: output said {_shown(value)}" in text
    assert _not_shown(text)
    assert not Path("report.json").exists()


@pytest.mark.parametrize("name", _NAMES)
def test_the_receipt_shows_the_output_path_as_typed(name):
    Path("reports").mkdir()

    result = _run("--output", f"reports/{name}.json")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    # The receipt shows the path as typed, directory included, not just the file name.
    assert f"Wrote citation report -> reports/{_shown(name)}.json" in text
    assert _not_shown(text)
    report = json.loads(Path("reports", f"{name}.json").read_text(encoding="utf-8"))
    assert report["n_rows"] == 2
    assert report["style"] == "bracket"


# --- lines that stay ---------------------------------------------------------


def test_a_plain_run_keeps_its_table_panel_and_receipt():
    result = _run("--output", "report.json")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    lines = strip_ansi(result.output).splitlines()
    assert lines[-1] == "Wrote citation report -> report.json"
    text = _plain(result)
    assert "Rows scored: 2" in text
    assert "Style: bracket" in text
    assert "Mean precision: 0.500" in text
    assert "Mean recall: 0.500" in text
    assert "Mean F1: 0.500" in text
    assert json.loads(Path("report.json").read_text(encoding="utf-8"))["aggregate"]["f1"] == 0.5


def test_a_plain_run_without_output_prints_no_receipt():
    result = _run("--style", "BRACKET")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "Wrote" not in strip_ansi(result.output)
    assert "Style: bracket" in _plain(result)


def test_a_skipped_row_keeps_its_text():
    Path("data.jsonl").write_text(
        json.dumps({"predicted": "x", "expected_ids": ["a\u0000b"]}) + "\n"
        + json.dumps(_ROWS[0]) + "\n",
        encoding="utf-8",
    )

    result = _run()

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "Row 0 skipped: expected_ids[0] must not contain null bytes" in _plain(result)


# --- styles ------------------------------------------------------------------


def test_the_refusals_keep_their_red_labels():
    style = _run("--style", "nope", console=_coloured_console())
    output = _run("--output", "../x.json", console=_coloured_console())

    assert style.exit_code == 2
    assert output.exit_code == 2
    assert style.output == (
        "\x1b[31mInvalid --style:\x1b[0m unknown citation_style 'nope'; supported: bracket, "
        "footnote, inline\n"
    )
    assert output.output.endswith(
        "\x1b[31mInvalid --output:\x1b[0m citation output '../x.json' must stay under cwd\n"
    )


def test_the_receipt_has_no_style_of_its_own():
    result = _run("--output", "report.json", console=_coloured_console())

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert result.output.endswith("\nWrote citation report -> report.json\n")
