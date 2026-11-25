"""soup sweep prints config paths, parameters and run names as typed, not as Rich markup."""

import re
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from tests._windows_filenames import skip_colon_on_windows
from tests.conftest import strip_ansi

runner = CliRunner()


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch):
    """Set both dimensions: Rich ignores width alone on a forced-color dumb terminal."""
    monkeypatch.setattr("soup_cli.commands.sweep.console", Console(width=500, height=100))


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
    marks=skip_colon_on_windows,
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
    # lr=[red]a is arm 1, the best arm: its own line and the best-run line.
    assert text.count("lr = [red]a") == 2
    # Grid row, arm line, results row, best-run line; b:warning: has no best-run line.
    assert text.count("[red]a") == 4
    assert text.count("b:warning:") == 3


@pytest.mark.parametrize("run_name", ["x[red]n", "[/]n"])
def test_sweep_failed_run_is_named_as_typed(tmp_path, monkeypatch, run_name):
    config = _config(tmp_path)

    def _fail(base_cfg, combo, name, config_path):
        raise RuntimeError("boom")

    monkeypatch.setattr("soup_cli.commands.sweep._run_single", _fail)

    result = runner.invoke(
        app,
        ["sweep", "--config", str(config), "--param", "lr=1e-5,2e-5", "--name", run_name, "--yes"],
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    text = _plain(result)
    assert f"Run {run_name}_1 failed: boom" in text
    assert "No sweep run completed: 2 of 2" in text


# A key under a dict-valued field (here ``rank_pattern``) is free text, so it can carry
# markup into the plan, the grid, the arm lines, the results table and the best-run line.
_DICT_CONFIG = _CONFIG + "training:\n  lora:\n    rank_pattern:\n      q_proj: 4\n"


@pytest.mark.parametrize("suffix", ["[red]x", "x:warning:"])
def test_sweep_dict_key_is_shown_as_typed_everywhere(tmp_path, monkeypatch, suffix):
    config = tmp_path / "soup.yaml"
    config.write_text(_DICT_CONFIG, encoding="utf-8")
    key = f"training.lora.rank_pattern.{suffix}"
    monkeypatch.setattr(
        "soup_cli.commands.sweep._run_single",
        lambda base_cfg, combo, name, config_path: {
            "final_loss": 0.4 if combo[key] == 16 else 0.5,
            "run_id": "run-1",
        },
    )

    result = runner.invoke(
        app, ["sweep", "--config", str(config), "--param", f"{key}=8,16", "--yes"]
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"Params: {key}" in text
    # Plan, grid header, two arm lines, results header and the best-run line.
    assert text.count(key) == 6
    assert f"{key} = 16" in text


def test_sweep_best_run_shows_an_emoji_code_in_the_value_as_typed(tmp_path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.setattr(
        "soup_cli.commands.sweep._run_single",
        lambda base_cfg, combo, name, config_path: {
            "final_loss": 0.4 if combo["lr"] == "b:warning:" else 0.5,
            "run_id": "run-1",
        },
    )

    result = runner.invoke(
        app,
        ["sweep", "--config", str(config), "--param", "lr=[red]a,b:warning:", "--yes"],
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    # Grid row, arm line, results row and the best-run line.
    assert text.count("b:warning:") == 4
    assert text.count("lr = b:warning:") == 2


def test_sweep_failed_run_shows_the_error_and_an_emoji_code_as_typed(tmp_path, monkeypatch):
    config = _config(tmp_path)

    def _fail(base_cfg, combo, name, config_path):
        raise RuntimeError("bo[/]om x:warning:")

    monkeypatch.setattr("soup_cli.commands.sweep._run_single", _fail)

    result = runner.invoke(
        app,
        ["sweep", "--config", str(config), "--param", "lr=1e-5,2e-5", "--name", "n:warning:",
         "--yes"],
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "Run n:warning:_1 failed: bo[/]om x:warning:" in _plain(result)


def test_sweep_results_table_strips_controls_from_the_run_name(tmp_path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.setattr(
        "soup_cli.commands.sweep._run_single",
        lambda base_cfg, combo, name, config_path: {"final_loss": 0.5, "run_id": "run-1"},
    )

    result = runner.invoke(
        app,
        ["sweep", "--config", str(config), "--param", "lr=1e-5,2e-5", "--name", "n\x9b\u202e",
         "--yes"],
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert "\x9b" not in text and "\u202e" not in text
    assert text.count("n_1") >= 3


def test_sweep_dict_key_has_its_controls_stripped_from_the_display_only(tmp_path, monkeypatch):
    config = tmp_path / "soup.yaml"
    config.write_text(_DICT_CONFIG, encoding="utf-8")
    key = "training.lora.rank_pattern.k\x9b\u202e"
    monkeypatch.setattr(
        "soup_cli.commands.sweep._run_single",
        lambda base_cfg, combo, name, config_path: {
            "final_loss": 0.4 if combo[key] == 16 else 0.5,
            "run_id": "run-1",
        },
    )

    result = runner.invoke(
        app, ["sweep", "--config", str(config), "--param", f"{key}=8,16", "--yes"]
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert "\x9b" not in text and "\u202e" not in text
    # Plan, grid header, two arm lines, results header and the best-run line.
    assert text.count("training.lora.rank_pattern.k") == 6
