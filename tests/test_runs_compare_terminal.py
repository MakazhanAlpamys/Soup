"""soup runs compare prints stored run text as typed, not as Rich markup."""

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
_A = "run_cmp_a"
_B = "run_cmp_b"

# One spelling per way the old code went wrong: a tag, a closing tag with
# nothing to close (a MarkupError), an emoji code, and a tag with control
# characters, so the strip and the escape are pinned separately.
_SPELLINGS = [
    "x[red]n",
    "[/]n",
    pytest.param("n:warning:", id="emoji-code"),
    f"x[bold]n{_CONTROLS}",
]

# (start_run keyword, the row label compare prints for it)
_ROW_FIELDS = [
    pytest.param("name", "Name", id="name"),
    pytest.param("base", "Model", id="model"),
    pytest.param("task", "Task", id="task"),
    pytest.param("device_name", "Device", id="device"),
    pytest.param("quantization", "Quantization", id="quantization"),
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
    run_id: str,
    *,
    name: str = "exp",
    base: str = "org/model",
    task: str = "sft",
    device_name: str = "CPU",
    quantization: str = "4bit",
) -> str:
    tracker = ExperimentTracker(db_path=db)
    try:
        return tracker.start_run(
            config_dict={"base": base, "task": task, "training": {"quantization": quantization}},
            device="cpu",
            device_name=device_name,
            gpu_info={},
            experiment_name=name,
            run_id=run_id,
        )
    finally:
        tracker.close()


def _compare(first: str, second: str):
    return runner.invoke(app, ["runs", "compare", first, second])


def _row(label: str, *cells: str) -> str:
    # \W is the cell border: a box character, or "|" on a terminal without them.
    return rf"{label} \W " + r" \W ".join(re.escape(cell) for cell in cells) + r" \W"


@pytest.mark.parametrize("value", _SPELLINGS)
@pytest.mark.parametrize("keyword, label", _ROW_FIELDS)
def test_compare_shows_a_differing_cell_as_typed(db, keyword, label, value):
    _start(db, _A, **{keyword: value})
    _start(db, _B)

    result = _compare(_A, _B)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert re.search(_row(label, _shown(value)), text)
    assert _not_shown(text)


@pytest.mark.parametrize("value", _SPELLINGS)
@pytest.mark.parametrize("keyword, label", _ROW_FIELDS)
def test_compare_shows_a_matching_cell_as_typed(db, keyword, label, value):
    _start(db, _A, **{keyword: value})
    _start(db, _B, **{keyword: value})

    result = _compare(_A, _B)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert re.search(_row(label, _shown(value), _shown(value)), text)
    assert _not_shown(text)


@pytest.mark.parametrize("position", [0, 1])
@pytest.mark.parametrize("value", _SPELLINGS)
def test_compare_names_a_run_id_column_as_typed(db, value, position):
    ids = ["run_cmp_plain", "run_cmp_plain2"]
    ids[position] = value
    _start(db, ids[0])
    _start(db, ids[1])

    result = _compare(*ids)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    shown = [_shown(run_id) for run_id in ids]
    assert re.search(_row("Metric", *shown), text)
    assert _not_shown(text)


def test_compare_still_highlights_only_the_differing_rows(db, monkeypatch):
    monkeypatch.setattr(
        "soup_cli.commands.runs.console",
        Console(
            width=500,
            force_terminal=True,
            color_system="standard",
            legacy_windows=False,
            no_color=False,
        ),
    )
    _start(db, _A, base="org/one")
    _start(db, _B, base="org/two")

    result = _compare(_A, _B)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    # Right-justified cells keep their padding inside the coloured span.
    assert re.search(r"\x1b\[33m\s*org/one\x1b\[0m", result.output)
    assert re.search(r"\x1b\[33m\s*org/two\x1b\[0m", result.output)
    assert not re.search(r"\x1b\[33m\s*sft", result.output)
