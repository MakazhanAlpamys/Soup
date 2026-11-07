"""``soup plan``'s peak VRAM: the train pre-flight's estimate, or unknown with a reason (#2049).

The plan used to record a fixed 8.0 GB whatever the run. It now records the shared
estimate when it can read the model, and ``null`` with ``peak_vram_reason`` when it
cannot; a state file written before the change still loads.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands.train import _build_hardware_fit_input
from soup_cli.config.loader import load_config_from_string
from soup_cli.utils.hardware_fit import estimate_peak_vram_gb
from soup_cli.utils.terraform_plan import (
    TrainingPlan,
    TrainingState,
    build_plan,
    detect_drift,
    read_state,
    write_state,
)
from tests._tiny_checkpoint import write_tiny_model
from tests.conftest import strip_ansi


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    write_tiny_model(tmp_path / "tiny")
    return tmp_path


def _config(base: str, **training) -> dict:
    return {
        "base": base,
        "task": "sft",
        "data": {"train": "train.jsonl", "max_length": 512},
        "training": {"batch_size": 2, "quantization": "none", **training},
    }


def test_a_readable_model_gets_the_pre_flight_s_number(workdir):
    config = _config(str(workdir / "tiny"))
    plan = build_plan(config)

    import yaml

    inp = _build_hardware_fit_input(load_config_from_string(yaml.safe_dump(config)))
    assert plan.peak_vram_gb == round(estimate_peak_vram_gb(inp).total_gb, 3)
    assert plan.peak_vram_reason is None
    assert plan.peak_vram_gb != 8.0


@pytest.mark.parametrize(
    ("config", "reason"),
    [
        (_config("someone/unreadable-7B"), "config.json"),
        (_config("TINY", batch_size="auto"), "batch_size auto"),
        (_config("TINY", stream_layers=True), "layer streaming"),
        ({**_config("TINY"), "backend": "mlx"}, "backend mlx: no CUDA estimate"),
    ],
)
def test_a_run_it_cannot_price_is_unknown_with_a_reason(workdir, config, reason):
    if config["base"] == "TINY":
        config["base"] = str(workdir / "tiny")
    plan = build_plan(config)
    assert plan.peak_vram_gb is None
    assert reason in plan.peak_vram_reason


def test_unknown_round_trips_through_the_state_file(workdir):
    plan = build_plan(_config("someone/unreadable-7B"))
    write_state(TrainingState(plan=plan, applied=False, applied_at=None, run_id=None), "s.json")
    raw = json.loads(Path("s.json").read_text(encoding="utf-8"))["plan"]
    assert raw["peak_vram_gb"] is None
    assert raw["peak_vram_reason"] == plan.peak_vram_reason
    assert read_state("s.json").plan == plan


def test_a_known_peak_writes_no_reason_key(workdir):
    plan = build_plan(_config(str(workdir / "tiny")))
    write_state(TrainingState(plan=plan, applied=False, applied_at=None, run_id=None), "s.json")
    raw = json.loads(Path("s.json").read_text(encoding="utf-8"))["plan"]
    assert isinstance(raw["peak_vram_gb"], float)
    assert "peak_vram_reason" not in raw


def _old_state(peak) -> dict:
    return {
        "schema_version": "1",
        "plan": {
            "base": "m",
            "task": "sft",
            "config_sha": "a" * 64,
            "dataset_sha": "b" * 64,
            "estimated_cost_usd": 0.5,
            "estimated_minutes": 10.0,
            "peak_vram_gb": peak,
            "spot_price_usd_per_hour": 0.30,
        },
        "applied": False,
        "applied_at": None,
        "run_id": None,
    }


def test_a_state_file_from_before_the_change_still_loads(workdir):
    Path("old.json").write_text(json.dumps(_old_state(8.0)), encoding="utf-8")
    plan = read_state("old.json").plan
    assert plan.peak_vram_gb == 8.0
    assert plan.peak_vram_reason is None


def test_a_null_peak_without_a_reason_is_refused(workdir):
    Path("bad.json").write_text(json.dumps(_old_state(None)), encoding="utf-8")
    with pytest.raises((TypeError, ValueError), match="peak_vram_reason"):
        read_state("bad.json")


def test_the_peak_is_still_not_part_of_drift(workdir):
    before = build_plan(_config("someone/unreadable-7B"))
    after = TrainingPlan(**{**before.__dict__, "peak_vram_gb": 3.0, "peak_vram_reason": None})
    state = TrainingState(plan=before, applied=False, applied_at=None, run_id=None)
    assert not detect_drift(state, after).has_drift


def test_soup_plan_prints_unknown_and_the_reason(workdir, monkeypatch):
    import yaml
    from rich.console import Console

    # Wide enough for the whole reason on one line: at 80 columns Rich wraps the
    # cell, and the check would match only up to wherever the wrap fell.
    monkeypatch.setattr("soup_cli.commands.plan.console", Console(width=200))

    Path("soup.yaml").write_text(
        yaml.safe_dump(_config("someone/unreadable-7B")), encoding="utf-8"
    )
    result = CliRunner().invoke(app, ["plan", "--config", "soup.yaml"])
    assert result.exit_code == 0, result.output
    plain = " ".join(strip_ansi(result.output).split())
    assert "unknown: could not read the model's config.json (local, cache or Hub)" in plain
