"""Autopilot shows literal inputs while writing their original values."""

import json
import os
import sys

import pytest
import yaml
from rich.console import Console

from soup_cli.autopilot import generate_config
from soup_cli.autopilot.analyzer import HardwareProfile
from soup_cli.cli import app
from soup_cli.commands import autopilot
from tests.conftest import make_cli_runner, strip_ansi

SPELLINGS = [
    (":warning:", ":warning:"),
    ("x[red]n", "x[red]n"),
    ("x\\[n", "x\\[n"),
    ("x\x9b\u202en", "xn"),
]


@pytest.fixture
def input_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(autopilot, "console", Console(width=300, height=100))
    data = tmp_path / "rows.jsonl"
    data.write_text(
        json.dumps({"instruction": "hello", "output": "world"}) + "\n", encoding="utf-8"
    )
    return data


def _invoke(data, *args, model="owner/model-1B", goal="chat"):
    return make_cli_runner().invoke(
        app,
        ["autopilot", "--model", model, "--data", str(data), "--goal", goal, *args],
    )


@pytest.mark.parametrize("value,shown", SPELLINGS)
def test_model_and_hardware_names_in_decision_panel(input_file, monkeypatch, value, shown):
    model = f"owner/model-1B-{value}"
    profile = HardwareProfile("cpu", f"GPU-{value}", 24.0, 0.0, 32.0)
    monkeypatch.setattr(autopilot, "analyze_hardware", lambda: profile)
    monkeypatch.setattr(generate_config, "analyze_hardware", lambda: profile)

    result = _invoke(input_file, "--yes", model=model)

    assert result.exit_code == 0, (result.output, result.exception)
    plain = " ".join(strip_ansi(result.output).split())
    assert f"Model: owner/model-1B-{shown}" in plain
    assert f"Hardware: GPU-{shown}" in plain
    assert f"Dataset: {input_file.name}" in plain
    assert "Goal: chat -> task: sft" in plain
    config = yaml.safe_load(input_file.with_name("soup.yaml").read_text(encoding="utf-8"))
    assert config["base"] == model
    assert config["data"]["train"] == os.path.realpath(input_file)


@pytest.mark.parametrize("value,shown", SPELLINGS)
def test_dataset_and_output_path_keep_literal_names(input_file, monkeypatch, value, shown):
    if sys.platform == "win32" and ":" in value:
        pytest.skip("colon is not a valid Windows filename character")
    data = input_file.parent / f"rows-{value}.jsonl"
    data.parent.mkdir(parents=True, exist_ok=True)
    input_file.rename(data)
    output = input_file.parent / f"config-{value}.yaml"
    profile = HardwareProfile("cpu", "none", 24.0, 0.0, 32.0)
    monkeypatch.setattr(autopilot, "analyze_hardware", lambda: profile)
    monkeypatch.setattr(generate_config, "analyze_hardware", lambda: profile)

    result = _invoke(data, "--output", str(output), "--yes")

    assert result.exit_code == 0, (result.output, result.exception)
    plain = " ".join(strip_ansi(result.output).split())
    expected_data = (input_file.parent / f"rows-{shown}.jsonl").name
    expected_output = os.path.realpath(input_file.parent / f"config-{shown}.yaml")
    assert f"Dataset: {expected_data}" in plain
    assert f"Config written: {expected_output}" in plain
    assert yaml.safe_load(output.read_text(encoding="utf-8"))["data"]["train"] == os.path.realpath(
        data
    )


@pytest.mark.parametrize("value,shown", SPELLINGS)
def test_unknown_goal_stays_literal(input_file, value, shown):
    result = _invoke(input_file, goal=value)

    assert result.exit_code == 1, result.exception
    assert f"Unknown goal '{shown}'. Options:" in " ".join(strip_ansi(result.output).split())


@pytest.mark.parametrize("stage", ["parse_gpu_budget", "analyze_dataset", "build_soup_config"])
def test_diagnostic_preserves_error_and_red_style(input_file, monkeypatch, stage):
    def fail(*args, **kwargs):
        raise ValueError("try :warning: x\\[n A\x9bB\u202eC")

    monkeypatch.setattr(autopilot, stage, fail)
    monkeypatch.setattr(
        autopilot,
        "console",
        Console(
            width=300, height=100, force_terminal=True, no_color=False, color_system="standard"
        ),
    )
    result = _invoke(input_file, "--gpu-budget", "24GB", "--yes")

    assert result.exit_code == 1, result.exception
    assert "\x1b[31mtry :warning: x\\[n ABC\x1b[0m" in result.output


def test_missing_dataset_stays_literal(input_file):
    path = input_file.with_name("missing[red]\x9b\u202e.jsonl")
    result = _invoke(path)

    assert result.exit_code == 1, result.exception
    shown = os.path.realpath(input_file.with_name("missing[red].jsonl"))
    assert f"Data file not found: {shown}" in " ".join(strip_ansi(result.output).split())
