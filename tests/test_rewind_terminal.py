"""Rewind prints recorded data literally without changing its JSON output."""

import json
from pathlib import Path

import pytest
from rich.console import Console

from soup_cli.cli import app
from soup_cli.commands import rewind
from soup_cli.experiment.tracker import ExperimentTracker
from soup_cli.monitoring.rewind_log import REWIND_LOG_VERSION, RewindLog
from soup_cli.utils import paths
from soup_cli.utils import rewind as engine
from tests.conftest import make_cli_runner, strip_ansi

RAW = r"text:warning:[red]\[end"
STORED = RAW + "\x9b\u202e"


@pytest.fixture(autouse=True)
def cwd_and_console(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "experiments.db"))
    monkeypatch.setattr(
        rewind,
        "console",
        Console(
            width=500,
            height=100,
            force_terminal=True,
            no_color=False,
            color_system="standard",
            legacy_windows=False,
        ),
    )


def run(*args):
    return make_cli_runner().invoke(app, ["rewind", *args])


@pytest.fixture
def log(monkeypatch):
    directory = Path("recorded-[weights]\x9b\u202e")
    directory.mkdir()
    path = directory / RewindLog.FILENAME
    header = {
        "kind": "header",
        "version": REWIND_LOG_VERSION,
        "backend": STORED,
        "task": "task-" + STORED,
        "dataset_fingerprint": "example",
    }
    batch = {
        "kind": "batch",
        "step": 1,
        "micro": 0,
        "rows": [0],
        "row_loss": [1.0],
        "row_tokens": [8],
    }
    path.write_text(json.dumps(header) + "\n" + json.dumps(batch) + "\n", encoding="utf-8")
    target = {"run_id": STORED, "output_dir": str(directory), "config_json": "{}"}

    def get_run(self, run_id):
        assert run_id == STORED
        return target

    monkeypatch.setattr(ExperimentTracker, "get_run", get_run)
    return path, header


@pytest.mark.parametrize(
    ("suffixes", "message"),
    [
        pytest.param((), f"Run not found: {RAW}", id="no-match"),
        pytest.param(
            ("1", "2"),
            f"Run id prefix '{RAW}' matches 2 runs; give more characters",
            id="ambiguous",
        ),
    ],
)
def test_run_lookup_failure_keeps_literal_id(suffixes, message):
    if suffixes:
        tracker = ExperimentTracker()
        try:
            for suffix in suffixes:
                tracker.start_run({}, "cpu", "CPU", {}, run_id=STORED + suffix)
        finally:
            tracker.close()
    result = run(STORED)
    assert result.exit_code == 1, (result.output, result.exception)
    assert result.stdout == f"\x1b[31m{message}\x1b[0m\n"


@pytest.mark.parametrize("kind", ["missing", "invalid", "empty", "no-step"])
def test_log_errors_preserve_path_and_styles(log, kind):
    path, header = log
    if kind == "missing":
        path.unlink()
    elif kind == "invalid":
        path.write_text("{broken", encoding="utf-8")
    elif kind == "empty":
        path.write_text(json.dumps(header) + "\n", encoding="utf-8")
    result = run(STORED, "--step", "2" if kind == "no-step" else "1")
    assert result.exit_code == 1, (result.output, result.exception)
    shown = str(path).replace("\x9b\u202e", "")
    prefix = {
        "missing": "\x1b[31mNo rewind log at\x1b[0m ",
        "invalid": "\x1b[31mCannot read rewind log:\x1b[0m rewind log ",
        "empty": "\x1b[33mNo micro-batches were recorded\x1b[0m in ",
        "no-step": "\x1b[31mNo records for step 2\x1b[0m in ",
    }[kind]
    assert prefix + shown in result.output
    if kind == "missing":
        assert "\x1b[1mtraining.rewind_log: false\x1b[0m" in result.output
    if kind == "empty":
        assert (
            "\x1b[1mRewind log off\x1b[0m or \x1b[1mRewind recorder disabled\x1b[0m"
            in result.output
        )


def test_panel_preview_and_json_keep_original_meaning(log, monkeypatch):
    monkeypatch.setattr(engine, "load_row_previews", lambda *args, **kwargs: {0: STORED})
    output = Path("report-[draft]\u202e.json")
    result = run(STORED, "--step", "1", "--json", str(output))
    assert result.exit_code == 0, (result.output, result.exception)
    plain = strip_ansi(result.output)
    rows = [
        line.removeprefix("│ ").removesuffix("│").rstrip()
        for line in plain.splitlines()
        if line.startswith("│")
    ]
    assert rows[:2] == [f"Run: {RAW}", f"Backend: {RAW} | Task: task-{RAW} | Steps recorded: 1"]
    assert f"Run: \x1b[1m{RAW}\x1b[0m" in result.output
    assert f"│ {RAW} " in plain
    assert "\x1b[2mWrote report-[draft].json\x1b[0m" in result.output
    assert json.loads(output.read_text(encoding="utf-8"))["run_id"] == STORED


@pytest.mark.parametrize("kind", ["output", "write", "preview"])
def test_external_diagnostics_keep_literal_error(log, monkeypatch, kind):
    def fail(*args, **kwargs):
        raise ValueError(STORED)

    if kind == "preview":
        monkeypatch.setattr(engine, "load_row_previews", fail)
        args = []
        expected = f"\x1b[2mCould not rebuild the dataset for previews: {RAW}\x1b[0m"
        code = 0
    else:
        function = "check_report_output_path" if kind == "output" else "atomic_write_text"
        monkeypatch.setattr(paths, function, fail)
        args = ["--json", "out.json", "--no-preview"]
        label = "Invalid output:" if kind == "output" else "Cannot write the JSON:"
        expected = f"\x1b[31m{label}\x1b[0m {RAW}"
        code = 2 if kind == "output" else 1
    result = run(STORED, "--step", "1", *args)
    assert result.exit_code == code, (result.output, result.exception)
    assert expected in result.output
    assert not Path("out.json").exists()
