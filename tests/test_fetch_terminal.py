"""soup fetch prints namespaces, entry names and output paths as typed, not as Rich markup."""

import os
import re
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from tests._windows_filenames import skip_colon_on_windows
from tests.conftest import strip_ansi

runner = CliRunner()

_ENTRY = "llama-3.1-8b-lora"
_CONTROLS = "\x9b\u202e"


_FILE_NAMES = [
    "x[red]o.yaml",
    "x[bold]o.yaml",
    pytest.param("o:warning:.yaml", marks=skip_colon_on_windows, id="emoji-code"),
]


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch):
    """Rich wraps at the terminal width, which CI sets differently from a laptop."""
    monkeypatch.setattr("soup_cli.commands.fetch.console", Console(width=500))


@pytest.fixture(autouse=True)
def _workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _plain(result) -> str:
    return " ".join(strip_ansi(result.output).split())


def _fetch(*args: str):
    return runner.invoke(app, ["fetch", *args])


def _not_shown(text: str) -> bool:
    return "\x9b" not in text and "\u202e" not in text


@pytest.mark.parametrize(
    "namespace", ["x[red]ns", "x[/]ns", pytest.param("ns:warning:", id="emoji-code")]
)
def test_unknown_namespace_listing_names_it_as_typed(namespace):
    result = _fetch(namespace)

    assert result.exit_code == 2
    assert f"got '{namespace}'" in _plain(result)


@pytest.mark.parametrize(
    ("namespace", "name"),
    [
        ("examples", "x[red]n"),
        ("examples", "x[/]n"),
        ("examples", "n:warning:"),
        ("x[red]ns", "nm"),
        ("x[/]ns", "nm"),
        ("ns:warning:", "nm"),
    ],
)
def test_unknown_entry_is_named_as_typed(namespace, name):
    result = _fetch(namespace, name)

    assert result.exit_code == 2
    assert f"Unknown {namespace} entry: {name}." in _plain(result)


def test_unknown_entry_strips_controls_from_the_display():
    result = _fetch(f"x[bold]ns{_CONTROLS}", f"x[bold]nm{_CONTROLS}")

    assert result.exit_code == 2
    text = _plain(result)
    assert "Unknown x[bold]ns entry: x[bold]nm." in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _FILE_NAMES)
def test_output_outside_cwd_is_named_as_typed(name):
    result = _fetch("examples", _ENTRY, "--output", f"../{name}")

    assert result.exit_code == 2
    assert f"--output must stay under cwd: {name}." in _plain(result)


def test_output_outside_cwd_strips_controls_from_the_display():
    result = _fetch("examples", _ENTRY, "--output", f"../x[bold]o{_CONTROLS}.yaml")

    assert result.exit_code == 2
    text = _plain(result)
    assert "--output must stay under cwd: x[bold]o.yaml." in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _FILE_NAMES)
def test_existing_output_is_named_as_typed(name):
    Path(name).write_text("keep", encoding="utf-8")

    result = _fetch("examples", _ENTRY, "--output", name)

    assert result.exit_code == 1
    assert f"{name} already exists. Use --force to overwrite." in _plain(result)
    assert Path(name).read_text(encoding="utf-8") == "keep"


def test_existing_output_strips_controls_from_the_display():
    name = f"x[bold]o{_CONTROLS}.yaml"
    Path(name).write_text("keep", encoding="utf-8")

    result = _fetch("examples", _ENTRY, "--output", name)

    assert result.exit_code == 1
    text = _plain(result)
    assert "x[bold]o.yaml already exists. Use --force to overwrite." in text
    assert _not_shown(text)


@pytest.mark.requires_symlink
@pytest.mark.parametrize("name", [*_FILE_NAMES, f"x[bold]o{_CONTROLS}.yaml"])
def test_symlinked_output_is_named_as_typed(name):
    Path(name).write_text("keep", encoding="utf-8")
    os.symlink(name, "link.yaml")

    result = _fetch("examples", _ENTRY, "--output", "link.yaml", "--force")

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Refusing to overwrite symlink at {name.replace(_CONTROLS, '')}" in text
    assert _not_shown(text)
    assert Path(name).read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("name", [*_FILE_NAMES, f"x[bold]o{_CONTROLS}.yaml"])
def test_unstatable_output_is_named_as_typed(monkeypatch, name):
    real_lstat = os.lstat

    def _deny(path, *args, **kwargs):
        if os.path.basename(str(path)) == name:
            raise PermissionError("denied")
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr("soup_cli.commands.fetch.os.lstat", _deny)

    result = _fetch("examples", _ENTRY, "--output", name)

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Cannot stat target {name.replace(_CONTROLS, '')}: PermissionError" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", [*_FILE_NAMES, f"x[bold]o{_CONTROLS}.yaml"])
def test_written_output_path_is_named_as_typed(name):
    result = _fetch("examples", _ENTRY, "--output", name)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert re.search(rf"Wrote \S*{re.escape(name.replace(_CONTROLS, ''))}", text)
    assert _not_shown(text)
    assert Path(name).is_file()
