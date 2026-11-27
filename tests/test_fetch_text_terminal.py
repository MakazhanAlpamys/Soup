"""soup fetch shows namespaces, entry names and paths as typed, backslashes included."""

import os
import sys
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import fetch as fetch_cmd
from soup_cli.utils.fetch_examples import fetch_examples_dir, get_entry
from tests._windows_filenames import skip_colon_on_windows
from tests.conftest import strip_ansi

runner = CliRunner()

_ENTRY = "llama-3.1-8b-lora"
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


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    """Rich wraps at the terminal width, which CI sets differently from a laptop."""
    monkeypatch.setenv("COLUMNS", "500")
    monkeypatch.setattr(fetch_cmd, "console", Console(width=500, height=100))
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)


def _coloured(monkeypatch):
    monkeypatch.setattr(
        fetch_cmd,
        "console",
        Console(
            width=500, height=100, force_terminal=True, no_color=False, color_system="standard"
        ),
    )


def _fetch(*args: str):
    return runner.invoke(app, ["fetch", *args])


def _plain(result) -> str:
    return " ".join(strip_ansi(result.output).split())


def _shown(value: str) -> str:
    return "".join(ch for ch in value if ch not in _CONTROLS)


def _not_shown(text: str) -> bool:
    return "\x9b" not in text and "\u202e" not in text


class _Os:
    """The ``os`` module as the command sees it, with an ``lstat`` that is refused for one name."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, attribute):
        return getattr(os, attribute)

    def lstat(self, path, *args, **kwargs):
        if os.path.basename(str(path)) == self._name:
            raise PermissionError("denied")
        return os.lstat(path, *args, **kwargs)


# --- the listing --------------------------------------------------------------


@pytest.mark.parametrize("value", _TEXTS)
def test_an_unknown_namespace_is_quoted_as_typed(value):
    result = _fetch(value)

    assert result.exit_code == 2, (result.output, repr(result.exception))
    text = _plain(result)
    # The validator quotes the namespace with repr(), which already escapes controls.
    valid = "['configs', 'deepspeed_configs', 'examples']"
    assert f"namespace must be one of {valid}; got {value!r}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _TEXTS)
def test_a_listing_refusal_with_any_text_is_shown_as_typed(monkeypatch, value):
    def refuse(namespace):
        raise ValueError(f"listing said {value}")

    monkeypatch.setattr(fetch_cmd, "list_entries", refuse)

    result = _fetch("examples")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"listing said {_shown(value)}" in text
    assert _not_shown(text)


# --- an unknown entry ---------------------------------------------------------


@pytest.mark.parametrize("value", _TEXTS)
def test_an_unknown_namespace_with_an_entry_is_shown_as_typed(value):
    result = _fetch(value, "nm")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"Unknown {_shown(value)} entry: nm." in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _TEXTS)
def test_an_unknown_entry_is_shown_as_typed(value):
    result = _fetch("examples", value)

    assert result.exit_code == 2, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"Unknown examples entry: {_shown(value)}." in text
    assert _not_shown(text)


# --- the destination ----------------------------------------------------------


@pytest.mark.parametrize("name", _NAMES)
def test_an_output_outside_the_working_directory_is_shown_by_its_name_as_typed(tmp_path, name):
    result = _fetch("examples", _ENTRY, "--output", str(tmp_path / "elsewhere" / f"{name}.yaml"))

    assert result.exit_code == 2, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"--output must stay under cwd: {_shown(name)}.yaml." in text
    assert _not_shown(text)
    assert not (tmp_path / "elsewhere").exists()


@pytest.mark.parametrize("name", _NAMES)
def test_an_existing_output_is_shown_by_its_name_as_typed(name):
    Path("sub").mkdir()
    Path("sub", f"{name}.yaml").write_text("keep", encoding="utf-8")

    result = _fetch("examples", _ENTRY, "--output", f"sub/{name}.yaml")

    assert result.exit_code == 1, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"{_shown(name)}.yaml already exists. Use --force to overwrite." in text
    assert "sub" not in text
    assert _not_shown(text)
    assert Path("sub", f"{name}.yaml").read_text(encoding="utf-8") == "keep"


@pytest.mark.requires_symlink
@pytest.mark.parametrize("name", _NAMES)
def test_a_symlinked_output_is_shown_by_the_name_it_resolves_to(name):
    Path(f"{name}.yaml").write_text("keep", encoding="utf-8")
    os.symlink(f"{name}.yaml", "link.yaml")

    result = _fetch("examples", _ENTRY, "--output", "link.yaml", "--force")

    assert result.exit_code == 1, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"Refusing to overwrite symlink at {_shown(name)}.yaml" in text
    assert _not_shown(text)
    assert Path(f"{name}.yaml").read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("name", _NAMES)
def test_an_output_that_cannot_be_inspected_is_shown_by_its_name_as_typed(monkeypatch, name):
    monkeypatch.setattr(fetch_cmd, "os", _Os(f"{name}.yaml"))

    result = _fetch("examples", _ENTRY, "--output", f"{name}.yaml")

    assert result.exit_code == 1, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"Cannot stat target {_shown(name)}.yaml: PermissionError" in text
    assert _not_shown(text)
    assert not Path(f"{name}.yaml").exists()


# --- the receipt --------------------------------------------------------------


@pytest.mark.parametrize("name", _NAMES)
def test_the_receipt_shows_the_written_path_as_typed(name):
    entry = get_entry(_ENTRY)

    result = _fetch("examples", _ENTRY, "--output", f"{name}.yaml")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    lines = [" ".join(line.split()) for line in strip_ansi(result.output).splitlines()]
    assert lines == [f"Wrote {_shown(os.path.realpath(f'{name}.yaml'))}", entry.description]
    source = Path(fetch_examples_dir(), entry.filename)
    assert Path(f"{name}.yaml").read_bytes() == source.read_bytes()


# --- lines that stay ---------------------------------------------------------


def test_the_listing_of_a_namespace_keeps_its_table():
    result = _fetch("examples")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert _ENTRY in text
    assert get_entry(_ENTRY).description in text


def test_a_namespace_without_entries_keeps_its_line():
    result = _fetch("configs")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert strip_ansi(result.output) == "No entries in namespace 'configs'.\n"


def test_a_directory_as_the_output_keeps_its_refusal():
    Path("out[dir]").mkdir()

    result = _fetch("examples", _ENTRY, "--output", "out[dir]")

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "Cannot write --output out[dir]: destination is a directory." in _plain(result)


# --- styles ------------------------------------------------------------------


def test_the_refusals_are_red_as_a_whole(monkeypatch):
    _coloured(monkeypatch)
    Path("old.yaml").write_text("keep", encoding="utf-8")

    unknown = _fetch("examples", "nope")
    namespace = _fetch("nope")
    exists = _fetch("examples", _ENTRY, "--output", "old.yaml")
    outside = _fetch("examples", _ENTRY, "--output", "../old.yaml")

    assert unknown.output == "\x1b[31mUnknown examples entry: nope.\x1b[0m\n"
    assert namespace.output.startswith("\x1b[31mnamespace must be one of ['configs', ")
    assert namespace.output.endswith("got 'nope'\x1b[0m\n")
    assert exists.output == "\x1b[31mold.yaml already exists. Use --force to overwrite.\x1b[0m\n"
    assert outside.output == "\x1b[31m--output must stay under cwd: old.yaml.\x1b[0m\n"


def test_the_receipt_has_a_green_label_and_a_dim_description(monkeypatch):
    _coloured(monkeypatch)
    entry = get_entry(_ENTRY)

    result = _fetch("examples", _ENTRY, "--output", "new.yaml")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    target = os.path.realpath("new.yaml")
    assert result.output == f"\x1b[32mWrote\x1b[0m {target}\n\x1b[2m{entry.description}\x1b[0m\n"


def test_the_stat_refusal_is_red_as_a_whole(monkeypatch):
    _coloured(monkeypatch)
    monkeypatch.setattr(fetch_cmd, "os", _Os("new.yaml"))

    result = _fetch("examples", _ENTRY, "--output", "new.yaml")

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert result.output == "\x1b[31mCannot stat target new.yaml: PermissionError\x1b[0m\n"


@pytest.mark.requires_symlink
def test_the_symlink_refusal_is_red_as_a_whole(monkeypatch):
    _coloured(monkeypatch)
    Path("real.yaml").write_text("keep", encoding="utf-8")
    os.symlink("real.yaml", "link.yaml")

    result = _fetch("examples", _ENTRY, "--output", "link.yaml", "--force")

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert result.output == "\x1b[31mRefusing to overwrite symlink at real.yaml\x1b[0m\n"
