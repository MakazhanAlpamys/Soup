"""soup runs prints run names and curriculum-history file names as typed, not as Rich markup."""

import os
import sys
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.experiment.tracker import ExperimentTracker
from tests.conftest import strip_ansi

runner = CliRunner()

_CONTROLS = "\x9b\u202e"

# A colon is not a legal file-name character on Windows.
_NO_COLON_ON_WINDOWS = pytest.mark.skipif(
    sys.platform == "win32", reason="':' is not valid in a Windows name"
)


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch):
    """Rich wraps at the terminal width, which CI sets differently from a laptop."""
    monkeypatch.setattr("soup_cli.commands.runs.console", Console(width=500))


def _plain(result) -> str:
    return " ".join(strip_ansi(result.output).split())


@pytest.fixture
def db(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "experiments.db"
    monkeypatch.setenv("SOUP_DB_PATH", str(path))
    monkeypatch.setenv("COLUMNS", "500")
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    return path


def _start(db: Path, *, run_id=None, name=None, base="org/model", task="sft") -> str:
    tracker = ExperimentTracker(db_path=db)
    try:
        return tracker.start_run(
            config_dict={"base": base, "task": task},
            device="cpu",
            device_name="CPU",
            gpu_info={},
            experiment_name=name,
            run_id=run_id,
        )
    finally:
        tracker.close()


@pytest.mark.parametrize("value", ["x[red]v", "x[/]v", "v:warning:"])
@pytest.mark.parametrize("field", ["run_id", "name", "base", "task"])
def test_runs_list_shows_each_cell_as_typed(db, field, value):
    _start(db, **{field: value})

    result = runner.invoke(app, ["runs"])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert value in _plain(result)


@pytest.mark.parametrize("field", ["run_id", "name", "base", "task"])
def test_runs_list_strips_controls_from_each_cell(db, field):
    _start(db, **{field: f"x[bold]v{_CONTROLS}"})

    result = runner.invoke(app, ["runs"])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert "x[bold]v" in text
    assert "\x9b" not in text and "\u202e" not in text


def _curve(db: Path, history: str):
    run_id = _start(db)
    return runner.invoke(app, ["runs", "curriculum-curve", run_id, "--history", history])


@pytest.mark.parametrize(
    "name", ["x[red]h.jsonl", pytest.param("h:warning:.jsonl", marks=_NO_COLON_ON_WINDOWS)]
)
def test_curriculum_curve_names_a_missing_history_file_as_typed(db, name):
    result = _curve(db, name)

    assert result.exit_code == 1
    assert f"curriculum_history.jsonl not found: {name}" in _plain(result)


def test_curriculum_curve_strips_controls_from_a_missing_history_file(db):
    result = _curve(db, f"x[bold]h{_CONTROLS}.jsonl")

    assert result.exit_code == 1
    text = _plain(result)
    assert "curriculum_history.jsonl not found: x[bold]h.jsonl" in text
    assert "\x9b" not in text and "\u202e" not in text


_SYMLINK_NAMES = ["x[red]l.jsonl", "l:warning:.jsonl", f"x[bold]l{_CONTROLS}.jsonl"]


@pytest.mark.skipif(sys.platform == "win32", reason="creating a symlink needs a privilege")
@pytest.mark.parametrize("name", _SYMLINK_NAMES)
def test_curriculum_curve_names_a_symlinked_history_file_as_typed(db, name):
    Path("real.jsonl").write_text("{}", encoding="utf-8")
    os.symlink("real.jsonl", name)

    result = _curve(db, name)

    assert result.exit_code == 2
    text = _plain(result)
    assert f"rejected for safety): {name.replace(_CONTROLS, '')}" in text
    assert "\x9b" not in text and "\u202e" not in text


_UNREADABLE_NAMES = ["x[red]s.jsonl", "s:warning:.jsonl", f"x[bold]s{_CONTROLS}.jsonl"]


@pytest.mark.parametrize("name", _UNREADABLE_NAMES)
def test_curriculum_curve_names_an_unreadable_history_file_as_typed(db, monkeypatch, name):
    real_lstat = os.lstat

    def _deny(path, *args, **kwargs):
        if os.path.basename(str(path)) == name:
            raise PermissionError("denied")
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr("soup_cli.commands.runs.os.lstat", _deny)

    result = _curve(db, name)

    assert result.exit_code == 2
    text = _plain(result)
    assert f"history path is not stat-able: {name.replace(_CONTROLS, '')}" in text
    assert "\x9b" not in text and "\u202e" not in text
