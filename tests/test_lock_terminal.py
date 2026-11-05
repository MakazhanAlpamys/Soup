"""soup lock prints typed paths, models and lock-file text as typed, not as Rich markup."""

import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli import __version__
from soup_cli.cli import app
from soup_cli.utils.exit_codes import EXIT_GATE_FAILED, EXIT_USAGE_ERROR
from soup_cli.utils.paths import quote_path
from soup_cli.utils.soup_lock import SoupLock, compute_lock_closure
from tests.conftest import strip_ansi

runner = CliRunner()

_CONTROLS = "\x9b\u202e"
_DASH = chr(0x2014)
_BASE_SHA, _DATA_SHA, _ENV_HASH = "a" * 64, "b" * 64, "c" * 64

# A colon is not a legal file-name character on Windows, and "n:" there is a drive.
_NO_COLON_ON_WINDOWS = pytest.mark.skipif(
    sys.platform == "win32", reason="':' is not valid in a Windows name"
)

# One spelling per way the old code went wrong: a tag, an emoji code, and a tag
# with control characters, so the strip and the escape are pinned separately.
_NAMES = [
    "x[red]n.lock",
    pytest.param("n:warning:.lock", marks=_NO_COLON_ON_WINDOWS, id="emoji-code"),
    f"x[bold]n{_CONTROLS}.lock",
]
# Values that are not file names can also carry a closing tag with nothing to close.
_VALUES = [
    "x[red]n",
    "[/]n",
    pytest.param("n:warning:", id="emoji-code"),
    f"x[bold]n{_CONTROLS}",
]


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    """Rich wraps at the terminal width, which CI sets differently from a laptop."""
    monkeypatch.setattr("soup_cli.commands.lock.console", Console(width=500, height=100))
    monkeypatch.setenv("COLUMNS", "500")
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)


def _plain(result) -> str:
    return " ".join(strip_ansi(result.output).split())


def _shown(value: str) -> str:
    return value.replace(_CONTROLS, "")


def _not_shown(text: str) -> bool:
    return "\x9b" not in text and "\u202e" not in text


def _lock_file(name: str, **fields: str) -> str:
    values = {
        "soup_version": "0.0.1",
        "base_model": "org/model",
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    values.update(fields)
    lock = SoupLock(
        base_model_sha=_BASE_SHA,
        dataset_sha=_DATA_SHA,
        env_hash=_ENV_HASH,
        closure_sha=compute_lock_closure(
            base_model_sha=_BASE_SHA, dataset_sha=_DATA_SHA, env_hash=_ENV_HASH
        ),
        **values,
    )
    Path(name).write_text(json.dumps(asdict(lock)), encoding="utf-8")
    return name


def _write(*args: str):
    return runner.invoke(app, ["lock", "write", *args])


def _write_args(**overrides: str) -> list[str]:
    options = {
        "--base-model": "org/model",
        "--base-sha": _BASE_SHA,
        "--dataset-sha": _DATA_SHA,
        "--env-hash": _ENV_HASH,
    }
    options.update(overrides)
    return [part for item in options.items() for part in item]


def _check(path: str, **overrides: str):
    options = {
        "--base-model": "org/model",
        "--base-sha": _BASE_SHA,
        "--dataset-sha": _DATA_SHA,
        "--env-hash": _ENV_HASH,
    }
    options.update(overrides)
    args = [part for item in options.items() for part in item]
    return runner.invoke(app, ["lock", "check", path, *args])


def _two_lines(first: str, second: str) -> str:
    # Two lines of a panel are separated by its border: a box character, or "|".
    return re.escape(first) + r"(?: \W)+ " + re.escape(second)


def _outside(name: str) -> str:
    # quote_path shows C0/C1 controls as \xNN text but leaves the bidi override.
    return quote_path(f"../{name}").replace("\u202e", "")


@pytest.mark.parametrize("name", _NAMES)
def test_write_names_the_output_as_typed(name):
    result = _write(*_write_args(), "--output", name)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"Lock: {_shown(name)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_write_names_the_base_model_as_typed(value):
    result = _write(*_write_args(**{"--base-model": value}))

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"Base: {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_write_names_a_missing_env_lock_as_typed(name):
    args = [part for part in _write_args() if part != _ENV_HASH]
    args.remove("--env-hash")

    result = _write(*args, "--env-lock", name)

    assert result.exit_code == 2
    text = _plain(result)
    assert f"--env-hash not provided and {_shown(name)} not found." in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_write_shows_an_env_lock_outside_cwd_as_typed(name):
    args = [part for part in _write_args() if part != _ENV_HASH]
    args.remove("--env-hash")

    result = _write(*args, "--env-lock", f"../{name}")

    assert result.exit_code == 2
    text = _plain(result)
    assert f"lock {f'../{name}'!r} is outside cwd" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_write_shows_an_output_outside_cwd_as_typed(name):
    result = _write(*_write_args(), "--output", f"../{name}")

    assert result.exit_code == 2
    text = _plain(result)
    assert f"lock path {_outside(name)} must stay under cwd" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_write_shows_a_refused_sha_as_typed(value):
    result = _write(*_write_args(**{"--base-sha": value}))

    assert result.exit_code == 2
    text = _plain(result)
    assert f"base_model_sha must be 64 hex chars (got {value!r})" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
@pytest.mark.parametrize("field, label", [
    pytest.param("soup_version", "Soup version", id="version"),
    pytest.param("base_model", "Base model", id="base-model"),
    pytest.param("created_at", "Created at", id="created-at"),
])
def test_show_prints_the_lock_text_as_typed(field, label, value):
    _lock_file("soup.lock", **{field: value})

    result = runner.invoke(app, ["lock", "show"])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"{label}: {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_show_names_the_lock_path_in_its_title_as_typed(name):
    _lock_file(name)

    result = runner.invoke(app, ["lock", "show", name])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"soup.lock {_DASH} {_shown(name)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_show_names_a_missing_lock_as_typed(name):
    result = runner.invoke(app, ["lock", "show", name])

    assert result.exit_code == EXIT_USAGE_ERROR
    text = _plain(result)
    assert f"lock file not found: {name!r}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_show_names_a_lock_outside_cwd_as_typed(name):
    result = runner.invoke(app, ["lock", "show", f"../{name}"])

    assert result.exit_code == EXIT_USAGE_ERROR
    text = _plain(result)
    assert f"lock path {_outside(name)} must stay under cwd" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_check_names_the_lock_when_it_matches_as_typed(name):
    _lock_file(name)

    result = _check(name)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert re.search(_two_lines(f"Lock: {_shown(name)}", "Status: OK"), text)
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_check_names_the_lock_when_it_drifted_as_typed(name):
    _lock_file(name)

    result = _check(name, **{"--dataset-sha": "d" * 64})

    assert result.exit_code == EXIT_GATE_FAILED
    text = _plain(result)
    assert re.search(_two_lines(f"Lock: {_shown(name)}", "Status: DRIFT"), text)
    assert "- dataset_sha drifted" in text
    assert _not_shown(text)


@pytest.mark.parametrize(
    ("overrides", "status", "exit_code"),
    [
        pytest.param({}, "OK", 0, id="matching"),
        pytest.param({"--dataset-sha": "d" * 64}, "DRIFT", EXIT_GATE_FAILED, id="drifted"),
    ],
)
def test_check_keeps_backslash_before_unmatched_bracket(overrides, status, exit_code):
    # A literal backslash on POSIX and a directory separator on Windows.
    displayed = r"nested\[unclosed.lock"
    name = displayed.replace(".lock", f"{_CONTROLS}.lock")
    path = Path(name)
    path.parent.mkdir(exist_ok=True)
    _lock_file(name)
    before = path.read_bytes()

    result = _check(name, **overrides)

    assert result.exit_code == exit_code, (result.output, repr(result.exception))
    assert _not_shown(result.stdout)
    assert re.search(_two_lines(f"Lock: {displayed}", f"Status: {status}"), _plain(result))
    assert path.read_bytes() == before


@pytest.mark.parametrize("value", _VALUES)
def test_check_shows_the_locked_soup_version_as_typed(value):
    _lock_file("soup.lock", soup_version=value)

    result = _check("soup.lock", **{"--dataset-sha": "d" * 64})

    assert result.exit_code == EXIT_GATE_FAILED
    text = _plain(result)
    assert f"Note: soup version changed ({_shown(value)} -> {__version__})" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_check_names_a_missing_lock_as_typed(name):
    result = _check(name)

    assert result.exit_code == EXIT_USAGE_ERROR
    text = _plain(result)
    assert f"lock file not found: {name!r}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_check_shows_a_refused_sha_as_typed(value):
    _lock_file("soup.lock")

    result = _check("soup.lock", **{"--base-sha": value})

    assert result.exit_code == EXIT_USAGE_ERROR
    text = _plain(result)
    assert f"base_model_sha must be 64 hex chars (got {value!r})" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_check_names_a_lock_outside_cwd_as_typed(name):
    result = _check(f"../{name}")

    assert result.exit_code == EXIT_USAGE_ERROR
    text = _plain(result)
    assert f"lock path {_outside(name)} must stay under cwd" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_write_shows_a_refused_env_lock_entry_as_typed(value):
    entry = {"name": "torch", "version": "2.0.0", "source": value}
    Path("soup-env.lock").write_text(json.dumps({"entries": [entry]}), encoding="utf-8")
    args = [part for part in _write_args() if part != _ENV_HASH]
    args.remove("--env-hash")

    result = _write(*args)

    assert result.exit_code == 2
    text = _plain(result)
    assert f"got {value!r}" in text
    assert _not_shown(text)
