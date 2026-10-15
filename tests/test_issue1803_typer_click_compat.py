"""Exercise Soup's Click-sensitive behavior on the installed, real Typer tree."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import autopilot, ship
from soup_cli.utils.argv_redaction import all_secret_option_strings, iter_commands
from tests.conftest import strip_ansi


@pytest.mark.parametrize("command", [["ship"], ["expect"], ["lock", "check"]])
def test_gate_bad_flags_keep_exit_three(command: list[str]) -> None:
    result = CliRunner().invoke(app, [*command, "--no-such-flag"])
    assert result.exit_code == 3, (result.output, result.exception)
    assert "No such option" in result.output


def test_ordinary_bad_flags_keep_exit_two() -> None:
    result = CliRunner().invoke(app, ["version", "--no-such-flag"])
    assert result.exit_code == 2, (result.output, result.exception)
    assert "No such option" in result.output


@pytest.mark.parametrize(
    ("arguments", "missing"),
    [
        ([], "--model"),
        (["--model", "model"], "--data"),
        (["--model", "model", "--data", "data.jsonl"], "--goal"),
    ],
)
@pytest.mark.parametrize("color", [False, True], ids=["plain", "ansi"])
def test_noninteractive_autopilot_reports_the_missing_option(
    monkeypatch: pytest.MonkeyPatch, arguments: list[str], missing: str, color: bool,
) -> None:
    monkeypatch.setattr(autopilot, "_stdin_isatty", lambda: False)
    result = CliRunner().invoke(
        app, ["autopilot", *arguments], color=color,
        env={"FORCE_COLOR": "1" if color else "0"},
    )
    assert result.exit_code == 2, (result.output, result.exception)
    output = " ".join(strip_ansi(result.output).split())
    assert "Missing option" in output and missing in output


@pytest.mark.parametrize("explicit", [False, True])
def test_ship_uses_config_defaults_but_preserves_explicit_flags(
    monkeypatch: pytest.MonkeyPatch, explicit: bool,
) -> None:
    configured = SimpleNamespace(
        task_eval=None, task_mode="metric", general_suite=None, judge_model=None,
        baseline=None, forgetting_threshold=0.42, noise_floor=None,
    )
    monkeypatch.setattr(ship, "_parse_ship_config", lambda path: (None, configured))
    observed = []

    def stop_after_threshold(value: float) -> None:
        observed.append(value)
        raise typer.Exit(0)

    monkeypatch.setattr(ship, "_validate_threshold_flag", stop_after_threshold)
    arguments = ["ship", "--config", "soup.yaml"]
    if explicit:
        arguments.extend(["--forgetting-threshold", "0.17"])
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0, (result.output, result.exception)
    assert observed == [0.17 if explicit else 0.42]


def test_ship_unknown_parameter_cannot_be_overridden() -> None:
    command = typer.main.get_command(app).commands["ship"]
    with command.make_context("ship", []) as context:
        assert not ship._flag_is_default(context, "unknown_parameter")


@pytest.mark.parametrize("explicit", [False, True])
def test_push_keeps_hf_env_token_out_of_other_hubs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit: bool,
) -> None:
    from soup_cli.utils import hubs

    monkeypatch.chdir(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "hf-token-for-hf-only")
    monkeypatch.setenv("MODELSCOPE_API_TOKEN", "modelscope-env-token")
    uploads = []
    monkeypatch.setattr(hubs, "upload_repo", lambda *args, **kwargs: uploads.append(kwargs))
    arguments = ["push", "--model", "model", "--repo", "user/model", "--hub", "modelscope"]
    if explicit:
        arguments.extend(["--token", "typed-modelscope-token"])
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0, (result.output, result.exception)
    assert len(uploads) == 1
    assert uploads[0]["token"] == (
        "typed-modelscope-token" if explicit else "modelscope-env-token"
    )


@pytest.mark.parametrize(
    "arguments",
    [["bench", "model", "--help"], ["bench", "--p50", "model", "--help"]],
)
@pytest.mark.parametrize("color", [False, True], ids=["plain", "ansi"])
def test_bench_legacy_routing_reaches_infer_help(arguments: list[str], color: bool) -> None:
    result = CliRunner().invoke(
        app, arguments, color=color, env={"COLUMNS": "200", "FORCE_COLOR": "1" if color else "0"},
    )
    assert result.exit_code == 0, (result.output, result.exception)
    output = " ".join(strip_ansi(result.output).split())
    assert "Path to model" in output and "--max-tokens" in output


def test_redaction_walks_the_real_tree_and_finds_nested_credentials() -> None:
    root = typer.main.get_command(app)
    commands = list(iter_commands(root))
    assert len(commands) > 200
    assert root.commands["push"] in commands
    assert {"--token", "-t", "--api-key", "--tool-auth-token"} <= all_secret_option_strings(root)
