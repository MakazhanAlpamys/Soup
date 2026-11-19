"""Cost refusals display typed values literally on the appropriate output stream."""

import json

import pytest
from rich.console import Console

from soup_cli.cli import app
from tests.conftest import make_cli_runner, strip_ansi

runner = make_cli_runner()

VALUES = [
    ("missing", "missing"),
    ("x[red]n", "x[red]n"),
    ("[/]n", "[/]n"),
    ("n:warning:", "n:warning:"),
    ("x[bold]n\x9b\u202e", "x[bold]n"),
]


@pytest.fixture(autouse=True)
def _fixed_consoles(monkeypatch):
    # Set both dimensions: Rich ignores width alone on a forced-color dumb terminal.
    monkeypatch.setattr("soup_cli.commands.cost.console", Console(width=500, height=100))
    monkeypatch.setattr(
        "soup_cli.commands.cost.err_console", Console(stderr=True, width=500, height=100)
    )


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "soup.yaml"
    path.write_text("base: tiny\ndata:\n  train: missing.jsonl\n", encoding="utf-8")
    return path


def _refusal(args, json_mode):
    result = runner.invoke(app, ["cost", *args, *(["--json"] if json_mode else [])])
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert isinstance(result.exception, SystemExit)
    diagnostic = result.stderr if json_mode else result.stdout
    assert (result.stdout if json_mode else result.stderr) == ""
    assert "\x9b" not in diagnostic and "\u202e" not in diagnostic
    return " ".join(strip_ansi(diagnostic).split())


@pytest.mark.parametrize("json_mode", [False, True])
@pytest.mark.parametrize(("value", "shown"), VALUES)
def test_missing_config_is_displayed_literally(config, value, shown, json_mode):
    argument = value + ".yaml"
    assert not (config.parent / argument).exists()

    text = _refusal(["--config", argument], json_mode)

    assert f"Config file not found: {shown}.yaml" in text


@pytest.mark.parametrize("json_mode", [False, True])
@pytest.mark.parametrize(("value", "shown"), VALUES)
def test_unknown_gpu_is_displayed_literally(config, value, shown, json_mode):
    text = _refusal(["--config", str(config), "--gpu", value], json_mode)

    assert f"No matching GPUs found for: {shown}" in text


@pytest.mark.parametrize("json_mode", [False, True])
def test_success_keeps_the_selected_gpu_and_streams(config, json_mode):
    result = runner.invoke(
        app,
        ["cost", "--config", str(config), "--gpu", "H100", *(["--json"] if json_mode else [])],
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert result.stderr == ""
    if json_mode:
        rows = json.loads(result.stdout)
        assert len(rows) == 1
        assert rows[0]["gpu"] == "H100"
        assert rows[0]["total_cost"] >= 0
    else:
        assert "Training Cost Estimate" in result.stdout
        assert "H100" in result.stdout
        assert "RTX 4090" not in result.stdout
