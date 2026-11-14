"""soup runs show prints stored run fields as typed, not as Rich markup."""

import json
import re
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.experiment.tracker import ExperimentTracker
from tests.conftest import strip_ansi

runner = CliRunner()

_CONTROLS = "\x9b\u202e"
_RUN_ID = "run_show_terminal"

# One spelling per way the old code went wrong: a tag, a closing tag with
# nothing to close (a MarkupError), an emoji code, and a tag with control
# characters, so the strip and the escape are pinned separately.
_SPELLINGS = [
    "x[red]n",
    "[/]n",
    pytest.param("n:warning:", id="emoji-code"),
    f"x[bold]n{_CONTROLS}",
]

# (start_run keyword, the line the panel prints for it)
_PANEL_FIELDS = [
    pytest.param("name", "Name: {}", id="name"),
    pytest.param("base", "Model: {}", id="model"),
    pytest.param("task", "Task: {}", id="task"),
    pytest.param("device_name", "Device: {} (cpu)", id="device-name"),
    pytest.param("device", "Device: CPU ({})", id="device"),
    pytest.param("gpu", "GPU Memory: {}", id="gpu-memory"),
    pytest.param("output_dir", "Output: {}", id="output-dir"),
    pytest.param("error", "Error: {}", id="error"),
]


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch):
    """Rich wraps at the terminal width, which CI sets differently from a laptop."""
    monkeypatch.setattr("soup_cli.commands.runs.console", Console(width=500))


@pytest.fixture
def db(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "experiments.db"
    monkeypatch.setenv("SOUP_DB_PATH", str(path))
    monkeypatch.setenv("COLUMNS", "500")
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    return path


def _plain(result) -> str:
    return " ".join(strip_ansi(result.output).split())


def _shown(value: str) -> str:
    return value.replace(_CONTROLS, "")


def _not_shown(text: str) -> bool:
    return "\x9b" not in text and "\u202e" not in text


def _start(
    db: Path,
    *,
    run_id: str = _RUN_ID,
    name: str = "exp",
    base: str = "org/model",
    task: str = "sft",
    device: str = "cpu",
    device_name: str = "CPU",
    gpu: str = "",
    note: str = "n",
    output_dir: str | None = None,
    error: str | None = None,
) -> str:
    tracker = ExperimentTracker(db_path=db)
    try:
        tracker.start_run(
            config_dict={"base": base, "task": task, "note": note},
            device=device,
            device_name=device_name,
            gpu_info={"memory_total": gpu},
            experiment_name=name,
            run_id=run_id,
        )
        if output_dir is not None:
            conn = tracker._get_conn()
            conn.execute("UPDATE runs SET output_dir = ? WHERE run_id = ?", (output_dir, run_id))
            conn.commit()
        if error is not None:
            tracker.fail_run(run_id, error=error)
    finally:
        tracker.close()
    return run_id


def _show(run_id: str):
    return runner.invoke(app, ["runs", "show", run_id])


@pytest.mark.parametrize("value", _SPELLINGS)
@pytest.mark.parametrize("keyword, line", _PANEL_FIELDS)
def test_show_panel_field_is_shown_as_typed(db, keyword, line, value):
    _start(db, **{keyword: value})

    result = _show(_RUN_ID)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert line.format(_shown(value)) in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _SPELLINGS)
def test_show_names_the_run_id_as_typed(db, value):
    run_id = _start(db, run_id=value)

    result = _show(run_id)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"Run ID: {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _SPELLINGS)
def test_show_config_panel_is_shown_as_typed(db, value):
    _start(db, note=value)

    result = _show(_RUN_ID)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f'"note": {json.dumps(value)}' in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _SPELLINGS)
def test_show_eval_table_names_the_benchmark_as_typed(db, value):
    _start(db)
    tracker = ExperimentTracker(db_path=db)
    try:
        tracker.save_eval_result(
            model_path="model", benchmark=value, score=0.5, details={}, run_id=_RUN_ID
        )
    finally:
        tracker.close()

    result = _show(_RUN_ID)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    # \W is the cell border: a box character, or "|" on a terminal without them.
    assert re.search(re.escape(_shown(value)) + r" \W 0\.5000", text)
    assert _not_shown(text)
