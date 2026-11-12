"""soup runs delete, clean and replay print a stored run id as typed, not as Rich markup."""

from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.experiment.tracker import ExperimentTracker
from tests.conftest import strip_ansi

runner = CliRunner()

_CONTROLS = "\x9b\u202e"

# One spelling per way the old code went wrong: a tag, a closing tag with
# nothing to close (a MarkupError), an emoji code, and a tag with control
# characters, so the strip and the escape are pinned separately.
_SPELLINGS = [
    "x[red]n",
    "[/]n",
    pytest.param("n:warning:", id="emoji-code"),
    f"x[bold]n{_CONTROLS}",
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


def _start(db: Path, run_id: str, **columns: str) -> str:
    tracker = ExperimentTracker(db_path=db)
    try:
        tracker.start_run(
            config_dict={"base": "org/model", "task": "sft"},
            device="cpu",
            device_name="CPU",
            gpu_info={},
            run_id=run_id,
        )
        for column, value in columns.items():
            conn = tracker._get_conn()
            conn.execute(f"UPDATE runs SET {column} = ? WHERE run_id = ?", (value, run_id))
            conn.commit()
    finally:
        tracker.close()
    return run_id


@pytest.mark.parametrize("value", _SPELLINGS)
def test_delete_names_the_deleted_run_as_typed(db, value):
    _start(db, value)

    result = runner.invoke(app, ["runs", "delete", value, "--force"])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"Deleted run: {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _SPELLINGS)
def test_delete_prompt_names_the_run_as_typed(db, value):
    _start(db, value)

    result = runner.invoke(app, ["runs", "delete", value], input="n\n")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"Delete run {_shown(value)}?" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _SPELLINGS)
def test_clean_names_the_run_in_its_fallback_note_as_typed(db, value):
    out = Path("out")
    (out / "checkpoint-1").mkdir(parents=True)
    (out / "checkpoint-2").mkdir()
    # absolute, as finish_run records it; clean skips a relative one (#2725)
    _start(db, value, output_dir=str(out.absolute()))

    result = runner.invoke(app, ["runs", "clean", value, "--dry-run"])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert (
        f"{_shown(value)}: no checkpoint has a recorded loss; "
        "keeping the latest, checkpoint-2, whole."
    ) in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _SPELLINGS)
def test_replay_names_the_run_id_as_typed(db, value):
    _start(db, value)

    result = runner.invoke(app, ["runs", "replay", value, "--no-plot"])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"Run ID: {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _SPELLINGS)
def test_replay_names_the_status_as_typed(db, value):
    _start(db, "run_replay_status", status=value)

    result = runner.invoke(app, ["runs", "replay", "run_replay_status", "--no-plot"])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"Status: {_shown(value)}" in text
    assert _not_shown(text)
