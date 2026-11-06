"""Tunability diagnostics and report paths keep literal user text."""

import json
import sys
from pathlib import Path

import pytest
from rich.console import Console

from soup_cli.cli import app
from tests.conftest import make_cli_runner, strip_ansi

runner = make_cli_runner()
VALUES = ["plain", "x[red]n", "[/]n", "n:warning:", "x\\[n", "n\\\\", "x\x9b\u202en"]


@pytest.fixture(autouse=True)
def environment(monkeypatch, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setattr("soup_cli.commands.tunability.console", Console(width=500, height=100))
    Path("input.jsonl").write_text('{"text":"hello"}\n', encoding="utf-8")


def plain(result):
    return " ".join(strip_ansi(result.output).split())


@pytest.mark.parametrize("value", VALUES)
@pytest.mark.parametrize("field", ["dataset", "output"])
def test_outside_paths_are_refused_literally(value, field):
    path = f"../out-{value}.json"
    args = ["tunability", "--dataset", path if field == "dataset" else "input.jsonl"]
    if field == "output":
        args += ["--output", path]

    result = runner.invoke(app, args)

    assert result.exit_code == 2
    assert f"{field} {path!r} is outside cwd" in plain(result)
    assert not Path(path).exists()


@pytest.mark.parametrize("value", VALUES)
@pytest.mark.parametrize("stage", ["candidates", "probe", "write"])
def test_refusals_keep_error_text(monkeypatch, value, stage):
    import typer

    def fail(*args, **kwargs):
        if stage == "candidates":
            raise typer.BadParameter(value)
        raise ValueError(value)

    functions = {
        "candidates": "_resolve_candidates",
        "probe": "run_tunability",
        "write": "write_report",
    }
    monkeypatch.setattr(f"soup_cli.commands.tunability.{functions[stage]}", fail)

    result = runner.invoke(app, ["tunability", "--dataset", "input.jsonl", "-o", "report.json"])

    assert result.exit_code == 2
    assert value.replace("\x9b\u202e", "") in plain(result)
    assert not Path("report.json").exists()


@pytest.mark.parametrize(
    "value",
    [
        "plain",
        "x[red]n",
        "x\\[n",
        "x\x9b\u202en",
        pytest.param(
            "n:warning:",
            marks=pytest.mark.skipif(sys.platform == "win32", reason="colon in filename"),
        ),
    ],
)
def test_receipt_names_the_actual_written_file(value):
    path = Path(f"out-{value}.json")
    path.parent.mkdir(parents=True, exist_ok=True)

    result = runner.invoke(app, ["tunability", "--dataset", "input.jsonl", "-o", str(path)])

    assert result.exit_code == 0, result.output
    assert f"Report written to {str(path).replace(chr(155) + chr(8238), '')}" in plain(result)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["results"]) == 8


def test_receipt_and_failure_keep_their_styles(monkeypatch):
    monkeypatch.setattr(
        "soup_cli.commands.tunability.console",
        Console(
            width=500, height=100, force_terminal=True, no_color=False, color_system="standard"
        ),
    )
    good = runner.invoke(app, ["tunability", "--dataset", "input.jsonl", "-o", "report.json"])
    bad = runner.invoke(app, ["tunability", "--dataset", "../missing.jsonl"])
    assert good.exit_code == 0 and bad.exit_code == 2
    assert "\x1b[32mReport written to report.json\x1b[0m" in good.output
    assert "\x1b[31mdataset '../missing.jsonl' is outside cwd\x1b[0m" in bad.output
