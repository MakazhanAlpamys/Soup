"""Plan/apply display external text literally without changing persisted inputs."""

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import plan
from soup_cli.utils import terraform_plan
from tests._tiny_checkpoint import write_tiny_model
from tests._windows_filenames import skip_colon_on_windows
from tests.conftest import strip_ansi

VALUES = [
    pytest.param("ordinary-name", "ordinary-name", id="plain-control"),
    pytest.param(":rocket:", ":rocket:", id="emoji-alias"),
    pytest.param("[red]name[/]", "[red]name[/]", id="markup"),
    pytest.param(r"name\[unfinished", r"name\[unfinished", id="backslash"),
    pytest.param("name\x1b\x07\x9b\u202e[end]", "name[end]", id="controls"),
]


def _plain(result, exit_code=0):
    assert result.exit_code == exit_code, (result.output, repr(result.exception))
    text = strip_ansi(result.output)
    assert not any(char in text for char in ("\x1b", "\x07", "\x9b", "\u202e")), repr(text)
    return " ".join(text.split())


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        plan, "console",
        Console(width=300, force_terminal=True, color_system="standard", no_color=False),
    )
    return tmp_path


def _plan():
    return terraform_plan.TrainingPlan(
        base="tiny", task="sft", config_sha="a" * 64, dataset_sha="b" * 64,
        estimated_cost_usd=0.5, estimated_minutes=10.0, peak_vram_gb=2.0,
        spot_price_usd_per_hour=0.3,
    )


@pytest.mark.parametrize("value,shown", VALUES)
@pytest.mark.parametrize(
    "stage",
    ["plan-load", "plan-build", "plan-write", "apply-load", "apply-read",
     "apply-build", "apply-write"],
)
def test_each_caught_error_is_literal_and_leaves_state_untouched(
    workdir, monkeypatch, stage, value, shown
):
    state = terraform_plan.TrainingState(_plan(), False, None, None)
    original = b"operator's saved state\n"
    Path("state.json").write_bytes(original)
    monkeypatch.setattr(plan, "_load_yaml_config", lambda *a, **k: ({}, None))
    monkeypatch.setattr(plan, "build_plan", lambda *a, **k: state.plan)
    monkeypatch.setattr(terraform_plan, "build_plan", lambda *a, **k: state.plan)
    monkeypatch.setattr(terraform_plan, "read_state", lambda *a, **k: state)

    def fail(*args, **kwargs):
        raise ValueError(f"Cannot use {value}")

    targets = {
        "plan-load": (plan, "_load_yaml_config"),
        "plan-build": (plan, "build_plan"),
        "plan-write": (plan, "write_state"),
        "apply-load": (plan, "_load_yaml_config"),
        "apply-read": (terraform_plan, "read_state"),
        "apply-build": (terraform_plan, "build_plan"),
        "apply-write": (plan, "write_state"),
    }
    monkeypatch.setattr(*targets[stage], fail)
    result = CliRunner().invoke(app, [stage.split("-")[0], "--state", "state.json"])

    assert _plain(result, 2) == f"Cannot use {shown}"
    assert "\x1b[31m" in result.output
    assert Path("state.json").read_bytes() == original
    assert sorted(path.name for path in workdir.iterdir()) == ["state.json"]


@pytest.mark.parametrize("value,shown", VALUES)
@pytest.mark.parametrize("command", ["plan", "apply"])
def test_missing_config_names_the_original_path_literally(
    workdir, monkeypatch, command, value, shown
):
    received = []

    def missing(path, **kwargs):
        received.append(path)
        raise FileNotFoundError(path)

    monkeypatch.setattr(plan, "_load_yaml_config", missing)
    result = CliRunner().invoke(app, [command, "--config", value])

    assert _plain(result, 1) == f"Config not found: {shown}"
    assert received == [value]
    assert not list(workdir.iterdir())


@pytest.mark.parametrize("value,shown", VALUES)
def test_missing_state_names_the_original_path_literally(workdir, monkeypatch, value, shown):
    received = []
    monkeypatch.setattr(plan, "_load_yaml_config", lambda *a, **k: ({}, None))

    def missing(path):
        received.append(path)
        raise FileNotFoundError(path)

    monkeypatch.setattr(terraform_plan, "read_state", missing)
    result = CliRunner().invoke(app, ["apply", "--state", value])

    assert _plain(result, 1) == f"No state file at {shown}; run `soup plan` first."
    assert received == [value]
    assert not list(workdir.iterdir())


@pytest.mark.parametrize("value,shown", VALUES)
@pytest.mark.parametrize("field", ["base", "task", "peak_vram_reason"])
def test_plan_table_is_literal_and_persists_the_original_values(
    workdir, monkeypatch, field, value, shown
):
    planned = replace(
        _plan(), **{field: value},
        **({"peak_vram_gb": None} if field == "peak_vram_reason" else {}),
    )
    config = {"base": planned.base, "task": planned.task}
    monkeypatch.setattr(plan, "_load_yaml_config", lambda *a, **k: (config, None))
    monkeypatch.setattr(plan, "build_plan", lambda *a, **k: planned)

    result = CliRunner().invoke(app, ["plan"])

    assert shown in _plain(result)
    assert terraform_plan.read_state("soup.tfstate").plan == planned


@pytest.mark.parametrize(
    "suffix,shown",
    [
        pytest.param("plain", "plain", id="plain-control"),
        pytest.param("[bold]name\u202e[end]", "[bold]name[end]", id="markup-and-bidi"),
        pytest.param(":rocket:", ":rocket:", id="emoji-alias", marks=skip_colon_on_windows),
        pytest.param(
            r"name\[unfinished", r"name\[unfinished", id="backslash",
            marks=pytest.mark.skipif(sys.platform == "win32", reason="POSIX backslash filename"),
        ),
    ],
)
def test_real_plan_apply_round_trip_keeps_paths_hashes_and_state(workdir, suffix, shown):
    base = f"model-{suffix}"
    config = f"config-{suffix}.yaml"
    state_file = f"state-{suffix}.json"
    write_tiny_model(workdir / base)
    document = {
        "base": base, "task": "sft", "data": {"train": "train.jsonl", "max_length": 512},
        "training": {"batch_size": 2, "quantization": "none"},
    }
    Path("train.jsonl").write_text('{"instruction":"hi","output":"hello"}\n', encoding="utf-8")
    Path(config).write_text(yaml.safe_dump(document), encoding="utf-8")
    original_config = Path(config).read_bytes()
    original_data = Path("train.jsonl").read_bytes()
    expected = terraform_plan.build_plan(document)
    runner = CliRunner()
    args = ["--config", config, "--state", state_file]

    planned = runner.invoke(app, ["plan", *args])
    text = _plain(planned)
    assert f"model-{shown}" in text
    assert f"Plan written to state-{shown}.json." in text
    saved = terraform_plan.read_state(state_file)
    assert saved.plan == expected and saved.applied is False
    before = Path(state_file).read_bytes()

    checked = runner.invoke(app, ["apply", *args, "--dry-run"])
    assert "Dry-run. No drift detected; would proceed." in _plain(checked)
    assert Path(state_file).read_bytes() == before

    applied = runner.invoke(app, ["apply", *args])
    text = _plain(applied)
    assert f"Plan locked in state-{shown}.json." in text
    assert f"Now run: soup train --config config-{shown}.yaml" in text
    assert "This records the plan; it does not start training." in text
    assert "\x1b[32m" in applied.output and "\x1b[1m" in applied.output
    saved = terraform_plan.read_state(state_file)
    assert saved.plan == expected and saved.applied is True
    assert saved.applied_at is not None and saved.run_id is None
    assert Path(config).read_bytes() == original_config
    assert Path("train.jsonl").read_bytes() == original_data
    assert {path.name for path in workdir.iterdir()} == {base, config, state_file, "train.jsonl"}

    # A real config change refuses with exit 3 and keeps the applied state intact.
    before = Path(state_file).read_bytes()
    changed = {**document, "training": {**document["training"], "lr": 2e-5}}
    Path(config).write_text(yaml.safe_dump(changed), encoding="utf-8")
    changed_config = Path(config).read_bytes()
    refused = runner.invoke(app, ["apply", *args])
    assert (
        "Drift detected. Plan no longer matches state: config_sha"
    ) in _plain(refused, 3)
    assert "Run `soup plan` again to refresh, then re-apply." in _plain(refused, 3)
    assert "\x1b[31mDrift detected." in refused.output
    assert Path(state_file).read_bytes() == before
    assert Path(config).read_bytes() == changed_config
    assert Path("train.jsonl").read_bytes() == original_data


def test_known_peak_keeps_the_offload_advisory_dim(workdir, monkeypatch):
    monkeypatch.setattr(
        plan, "_load_yaml_config",
        lambda *a, **k: ({"training": {"activation_offloading": "cpu"}}, None),
    )
    monkeypatch.setattr(plan, "build_plan", lambda *a, **k: _plan())
    result = CliRunner().invoke(app, ["plan"])

    text = " ".join(_plain(result).replace("│", " ").split())
    assert (
        "training.activation_offloading: cpu is not modelled; activations are charged as resident"
    ) in text
    assert "\x1b[2mtraining.activation_offloading" in result.output
