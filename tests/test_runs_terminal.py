"""soup runs prints run ids and checkpoint paths as typed, not as Rich markup."""

import sys
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.experiment.tracker import ExperimentTracker
from tests.conftest import strip_ansi

runner = CliRunner()


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch):
    """Rich wraps at the terminal width, which CI sets differently from a laptop."""
    monkeypatch.setattr("soup_cli.commands.runs.console", Console(width=500))


NOT_FOUND_NAMES = [
    "[red]nope",
    "[/]nope",
    pytest.param("gone:warning:", id="emoji-code"),
]


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


@pytest.mark.parametrize("name", NOT_FOUND_NAMES)
def test_delete_names_a_missing_run_as_typed(db, name):
    result = runner.invoke(app, ["runs", "delete", name])

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert f"Run not found: {name}" in _plain(result)


@pytest.mark.parametrize("name", NOT_FOUND_NAMES)
def test_clean_names_a_missing_run_as_typed(db, name):
    result = runner.invoke(app, ["runs", "clean", name])

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert f"Run not found: {name}" in _plain(result)


@pytest.mark.parametrize("position", [0, 1])
@pytest.mark.parametrize("name", NOT_FOUND_NAMES)
def test_compare_names_a_missing_run_as_typed(db, name, position):
    tracker = ExperimentTracker(db_path=db)
    try:
        existing = tracker.start_run(config_dict={}, device="cpu", device_name="CPU", gpu_info={})
    finally:
        tracker.close()
    args = [existing, existing]
    args[position] = name

    result = runner.invoke(app, ["runs", "compare", *args])

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert f"Run not found: {name}" in _plain(result)


@pytest.mark.parametrize("command", ["delete", "clean"])
def test_missing_run_display_strips_controls(db, command):
    result = runner.invoke(app, ["runs", command, "[bold]gone\x9b\u202e"])

    assert result.exit_code == 1
    text = _plain(result)
    assert "Run not found: [bold]gone" in text
    assert "\x9b" not in text and "\u202e" not in text


def _run_with_checkpoints(db: Path, output: Path) -> str:
    tracker = ExperimentTracker(db_path=db)
    try:
        run_id = tracker.start_run(config_dict={}, device="cpu", device_name="CPU", gpu_info={})
        for step in (1, 2):
            checkpoint = output / f"checkpoint-{step}"
            checkpoint.mkdir(parents=True)
            for name in ("model.safetensors", "optimizer.pt", "scheduler.pt"):
                (checkpoint / name).write_text("x" * 64, encoding="utf-8")
        tracker.finish_run(
            run_id=run_id,
            initial_loss=1.0,
            final_loss=0.5,
            total_steps=2,
            duration_secs=1.0,
            output_dir=str(output),
        )
        tracker.log_metrics(run_id, step=1, loss=0.9)
        tracker.log_metrics(run_id, step=2, loss=0.4)
    finally:
        tracker.close()
    return run_id


@pytest.mark.parametrize("directory", ["[red]out", "[/]out", "[bold]out\x9b\u202e"])
def test_clean_dry_run_lists_paths_as_typed(db, tmp_path, directory):
    output = tmp_path / "work" / directory
    run_id = _run_with_checkpoints(db, output)

    result = runner.invoke(app, ["runs", "clean", run_id, "--dry-run"])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    shown = directory.replace("\x9b", "").replace("\u202e", "")
    assert "Delete file:" in text
    assert f"{shown}/checkpoint-1/optimizer.pt" in text.replace("\\", "/")
    assert "\x9b" not in text and "\u202e" not in text
    assert (output / "checkpoint-1" / "optimizer.pt").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_clean_failed_delete_warning_is_shown_as_typed(db, tmp_path, monkeypatch):
    output = tmp_path / "work" / "[red]out"
    run_id = _run_with_checkpoints(db, output)

    def _refuse(self, *args, **kwargs):
        raise OSError("cannot [/] remove")

    monkeypatch.setattr(Path, "unlink", _refuse)

    result = runner.invoke(app, ["runs", "clean", run_id, "--force"])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert "Failed to delete file" in text
    assert "[red]out" in text
    assert "cannot [/] remove" in text
