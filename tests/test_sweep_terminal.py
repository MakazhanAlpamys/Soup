"""soup sweep prints config paths, parameters and run names as typed, not as Rich markup."""

import re
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi

runner = CliRunner()

_CONFIG = """\
base: hf/model
task: sft
data:
  train: ./t.jsonl
  format: auto
output: ./o
"""

# A colon is not a legal file-name character on Windows.
COLON_NAME = pytest.param(
    "demo:warning:",
    marks=pytest.mark.skipif(sys.platform == "win32", reason="':' is not valid in a Windows name"),
)
NAMES = ["[red]cfg", "[/]cfg", COLON_NAME]


def _plain(result) -> str:
    return " ".join(strip_ansi(result.output).split()).replace("\\", "/")


@pytest.fixture(autouse=True)
def _wide_terminal(monkeypatch):
    monkeypatch.setenv("COLUMNS", "500")


def _config(tmp_path, directory: str = ""):
    folder = tmp_path / directory
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "soup.yaml"
    path.write_text(_CONFIG, encoding="utf-8")
    return path


@pytest.mark.parametrize("name", ["[red]soup.yaml", "[/]soup.yaml", "gone:warning:.yaml"])
def test_sweep_missing_config_is_named_as_typed(tmp_path, monkeypatch, name):
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["sweep", "--config", name, "--param", "lr=1e-5,2e-5"])

    assert result.exit_code == 1
    # Windows prints the path with backslashes, as Path does.
    assert f"Config not found: {Path(name).as_posix()}" in _plain(result)


@pytest.mark.parametrize("strategy", ["[red]x", "[/]x", "x:warning:"])
def test_sweep_invalid_strategy_is_named_as_typed(tmp_path, strategy):
    config = _config(tmp_path)

    result = runner.invoke(
        app, ["sweep", "--config", str(config), "--param", "lr=1e-5,2e-5", "--strategy", strategy]
    )

    assert result.exit_code == 1
    assert f"Invalid strategy: {strategy}. Must be grid or random." in _plain(result)


@pytest.mark.parametrize("key", ["[red]nope", "[/]nope", "nope:warning:"])
def test_sweep_unknown_parameter_is_named_as_typed(tmp_path, key):
    config = _config(tmp_path)

    result = runner.invoke(
        app, ["sweep", "--config", str(config), "--param", f"{key}=1,2", "--dry-run"]
    )

    assert result.exit_code == 1
    assert key in _plain(result)


@pytest.mark.parametrize("param", ["[red]nope", "[/]nope", "nope:warning:"])
def test_sweep_skipped_parameter_is_named_as_typed(tmp_path, param):
    config = _config(tmp_path)

    result = runner.invoke(
        app,
        [
            "sweep", "--config", str(config), "--param", param,
            "--param", "lr=1e-5,2e-5", "--dry-run",
        ],
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert f"Skipping invalid param: {param} (missing '=')" in _plain(result)


@pytest.mark.parametrize("name", NAMES)
def test_sweep_plan_shows_the_config_path_and_values_as_typed(tmp_path, name):
    config = _config(tmp_path, name)

    result = runner.invoke(
        app,
        ["sweep", "--config", str(config), "--param", "lr=[red]a,b:warning:", "--dry-run"],
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert re.search(rf"Config: +\S*{re.escape(name)}/soup\.yaml", text)
    assert "[red]a" in text
    assert "b:warning:" in text


def test_sweep_plan_strips_controls_from_the_display_only(tmp_path):
    config = _config(tmp_path)

    result = runner.invoke(
        app,
        ["sweep", "--config", str(config), "--param", "lr=[bold]a\x9b\u202e,b", "--dry-run"],
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert "[bold]a" in text
    assert "\x9b" not in text and "\u202e" not in text


@pytest.mark.parametrize("run_name", ["[red]n", "[/]n", "n:warning:"])
def test_sweep_runs_and_best_run_are_named_as_typed(tmp_path, monkeypatch, run_name):
    config = _config(tmp_path)
    monkeypatch.setattr(
        "soup_cli.commands.sweep._run_single",
        lambda base_cfg, combo, name, config_path: {"final_loss": 0.5, "run_id": "run-1"},
    )

    result = runner.invoke(
        app,
        [
            "sweep", "--config", str(config), "--param", "lr=[red]a,b:warning:",
            "--name", run_name, "--yes",
        ],
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"Run 1/2: {run_name}_1" in text
    assert "lr = [red]a" in text
    assert "lr = b:warning:" in text
    assert f"Best run: {run_name}_1" in text
    # The results table lists the arm's name and parameter values too.
    assert text.count(f"{run_name}_1") >= 3
