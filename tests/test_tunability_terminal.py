"""Tunability diagnostics and report paths keep literal user text."""

import json
from pathlib import Path

import pytest
from rich.console import Console

from soup_cli.cli import app
from tests._windows_filenames import skip_colon_on_windows
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
        pytest.param("n:warning:", marks=skip_colon_on_windows),
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


@pytest.mark.parametrize("value", VALUES)
def test_an_unknown_candidate_is_named_as_typed(value):
    result = runner.invoke(app, ["tunability", "--dataset", "input.jsonl", "--candidates", value])

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert f"unknown candidate {value!r}; known:" in plain(result)


def _coloured(monkeypatch):
    monkeypatch.setattr(
        "soup_cli.commands.tunability.console",
        Console(
            width=500, height=100, force_terminal=True, no_color=False, color_system="standard"
        ),
    )


def test_receipt_and_failure_keep_their_styles(monkeypatch):
    _coloured(monkeypatch)
    good = runner.invoke(app, ["tunability", "--dataset", "input.jsonl", "-o", "report.json"])
    bad = runner.invoke(app, ["tunability", "--dataset", "../missing.jsonl"])
    assert good.exit_code == 0 and bad.exit_code == 2
    assert "\x1b[32mReport written to report.json\x1b[0m" in good.output
    assert "\x1b[31mdataset '../missing.jsonl' is outside cwd\x1b[0m" in bad.output


@pytest.mark.parametrize("stage", ["candidates", "probe", "write"])
def test_every_error_line_is_red(monkeypatch, stage):
    import typer

    _coloured(monkeypatch)
    functions = {
        "candidates": "_resolve_candidates",
        "probe": "run_tunability",
        "write": "write_report",
    }

    def fail(*_args, **_kwargs):
        if stage == "candidates":
            raise typer.BadParameter("boom")
        raise ValueError("boom")

    monkeypatch.setattr(f"soup_cli.commands.tunability.{functions[stage]}", fail)

    result = runner.invoke(app, ["tunability", "--dataset", "input.jsonl", "-o", "report.json"])

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert "\x1b[31mboom\x1b[0m" in result.output


def test_the_output_refusal_is_red(monkeypatch):
    _coloured(monkeypatch)

    result = runner.invoke(app, ["tunability", "--dataset", "input.jsonl", "-o", "../report.json"])

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert "\x1b[31moutput '../report.json' is outside cwd\x1b[0m" in result.output
