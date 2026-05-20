"""v0.64.0 Part B — `soup plan` / `apply` (Terraform UX) tests."""

from __future__ import annotations

import dataclasses
import os
import sys

import pytest
from typer.testing import CliRunner

runner = CliRunner()


# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------


def test_module_imports():
    from soup_cli.utils import terraform_plan

    assert hasattr(terraform_plan, "TrainingPlan")
    assert hasattr(terraform_plan, "TrainingState")
    assert hasattr(terraform_plan, "build_plan")
    assert hasattr(terraform_plan, "write_state")
    assert hasattr(terraform_plan, "read_state")
    assert hasattr(terraform_plan, "detect_drift")
    assert hasattr(terraform_plan, "compute_config_sha")
    assert hasattr(terraform_plan, "DEFAULT_STATE_FILE")


# ---------------------------------------------------------------------------
# compute_config_sha
# ---------------------------------------------------------------------------


def test_compute_config_sha_deterministic():
    from soup_cli.utils.terraform_plan import compute_config_sha

    a = compute_config_sha({"a": 1, "b": 2})
    b = compute_config_sha({"b": 2, "a": 1})  # key order shouldn't matter
    assert a == b
    assert len(a) == 64  # SHA-256 hex


def test_compute_config_sha_changes_with_content():
    from soup_cli.utils.terraform_plan import compute_config_sha

    assert compute_config_sha({"a": 1}) != compute_config_sha({"a": 2})


def test_compute_config_sha_rejects_non_dict():
    from soup_cli.utils.terraform_plan import compute_config_sha

    with pytest.raises(TypeError):
        compute_config_sha("not a dict")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TrainingPlan
# ---------------------------------------------------------------------------


def test_training_plan_frozen():
    from soup_cli.utils.terraform_plan import TrainingPlan

    p = TrainingPlan(
        base="meta-llama/Llama-3.2-1B",
        task="sft",
        config_sha="a" * 64,
        dataset_sha="b" * 64,
        estimated_cost_usd=0.50,
        estimated_minutes=10.0,
        peak_vram_gb=8.0,
        spot_price_usd_per_hour=0.30,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.base = "other"  # type: ignore[misc]


def test_training_plan_rejects_short_sha():
    from soup_cli.utils.terraform_plan import TrainingPlan

    with pytest.raises(ValueError, match="sha"):
        TrainingPlan(
            base="m",
            task="sft",
            config_sha="short",
            dataset_sha="b" * 64,
            estimated_cost_usd=0.50,
            estimated_minutes=10.0,
            peak_vram_gb=8.0,
            spot_price_usd_per_hour=0.30,
        )


def test_training_plan_rejects_non_finite_cost():
    from soup_cli.utils.terraform_plan import TrainingPlan

    with pytest.raises(ValueError, match="finite"):
        TrainingPlan(
            base="m",
            task="sft",
            config_sha="a" * 64,
            dataset_sha="b" * 64,
            estimated_cost_usd=float("nan"),
            estimated_minutes=10.0,
            peak_vram_gb=8.0,
            spot_price_usd_per_hour=0.30,
        )


def test_training_plan_rejects_bool_cost():
    from soup_cli.utils.terraform_plan import TrainingPlan

    with pytest.raises(TypeError, match="bool"):
        TrainingPlan(
            base="m",
            task="sft",
            config_sha="a" * 64,
            dataset_sha="b" * 64,
            estimated_cost_usd=True,  # type: ignore[arg-type]
            estimated_minutes=10.0,
            peak_vram_gb=8.0,
            spot_price_usd_per_hour=0.30,
        )


def test_training_plan_rejects_negative_cost():
    from soup_cli.utils.terraform_plan import TrainingPlan

    with pytest.raises(ValueError, match="negative"):
        TrainingPlan(
            base="m",
            task="sft",
            config_sha="a" * 64,
            dataset_sha="b" * 64,
            estimated_cost_usd=-1.0,
            estimated_minutes=10.0,
            peak_vram_gb=8.0,
            spot_price_usd_per_hour=0.30,
        )


# ---------------------------------------------------------------------------
# build_plan
# ---------------------------------------------------------------------------


def test_build_plan_happy(tmp_path, monkeypatch):
    from soup_cli.utils.terraform_plan import build_plan

    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"prompt": "a", "completion": "b"}\n')

    config = {
        "base": "meta-llama/Llama-3.2-1B",
        "task": "sft",
        "data": {"train": str(dataset)},
        "training": {"epochs": 1, "lr": 5e-5, "batch_size": 4},
    }
    plan = build_plan(config)
    assert plan.base == "meta-llama/Llama-3.2-1B"
    assert plan.task == "sft"
    assert len(plan.config_sha) == 64
    assert plan.estimated_cost_usd >= 0
    assert plan.estimated_minutes >= 0


def test_build_plan_rejects_missing_base(tmp_path, monkeypatch):
    from soup_cli.utils.terraform_plan import build_plan

    monkeypatch.chdir(tmp_path)
    config = {"task": "sft", "data": {"train": "x.jsonl"}}
    with pytest.raises(ValueError, match="base"):
        build_plan(config)


def test_build_plan_rejects_non_dict():
    from soup_cli.utils.terraform_plan import build_plan

    with pytest.raises(TypeError):
        build_plan("not a dict")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# write_state / read_state
# ---------------------------------------------------------------------------


def test_write_state_roundtrip(tmp_path, monkeypatch):
    from soup_cli.utils.terraform_plan import (
        TrainingPlan,
        TrainingState,
        read_state,
        write_state,
    )

    monkeypatch.chdir(tmp_path)
    plan = TrainingPlan(
        base="m",
        task="sft",
        config_sha="a" * 64,
        dataset_sha="b" * 64,
        estimated_cost_usd=0.50,
        estimated_minutes=10.0,
        peak_vram_gb=8.0,
        spot_price_usd_per_hour=0.30,
    )
    state = TrainingState(plan=plan, applied=False, applied_at=None, run_id=None)
    out = tmp_path / "soup.tfstate"
    write_state(state, str(out))

    loaded = read_state(str(out))
    assert loaded.plan.base == "m"
    assert loaded.applied is False


def test_write_state_outside_cwd_rejected(tmp_path, monkeypatch):
    from soup_cli.utils.terraform_plan import TrainingPlan, TrainingState, write_state

    monkeypatch.chdir(tmp_path)
    plan = TrainingPlan(
        base="m", task="sft", config_sha="a" * 64, dataset_sha="b" * 64,
        estimated_cost_usd=0.5, estimated_minutes=10.0, peak_vram_gb=8.0,
        spot_price_usd_per_hour=0.30,
    )
    state = TrainingState(plan=plan, applied=False, applied_at=None, run_id=None)
    outside = tmp_path.parent / "evil.tfstate"
    with pytest.raises(ValueError, match="cwd"):
        write_state(state, str(outside))


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks")
def test_write_state_symlink_rejected(tmp_path, monkeypatch):
    from soup_cli.utils.terraform_plan import TrainingPlan, TrainingState, write_state

    monkeypatch.chdir(tmp_path)
    plan = TrainingPlan(
        base="m", task="sft", config_sha="a" * 64, dataset_sha="b" * 64,
        estimated_cost_usd=0.5, estimated_minutes=10.0, peak_vram_gb=8.0,
        spot_price_usd_per_hour=0.30,
    )
    state = TrainingState(plan=plan, applied=False, applied_at=None, run_id=None)
    target = tmp_path / "real.tfstate"
    target.write_text("{}")
    link = tmp_path / "link.tfstate"
    os.symlink(target, link)
    with pytest.raises(ValueError, match="symlink"):
        write_state(state, str(link))


def test_read_state_missing(tmp_path, monkeypatch):
    from soup_cli.utils.terraform_plan import read_state

    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        read_state(str(tmp_path / "nope.tfstate"))


# ---------------------------------------------------------------------------
# detect_drift
# ---------------------------------------------------------------------------


def test_detect_drift_clean(tmp_path, monkeypatch):
    from soup_cli.utils.terraform_plan import (
        TrainingState,
        build_plan,
        detect_drift,
    )

    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n")
    config = {
        "base": "m",
        "task": "sft",
        "data": {"train": str(dataset)},
        "training": {"epochs": 1, "lr": 5e-5, "batch_size": 4},
    }
    plan = build_plan(config)
    state = TrainingState(plan=plan, applied=False, applied_at=None, run_id=None)
    # Rebuild plan from same config → no drift
    plan_now = build_plan(config)
    drift = detect_drift(state, plan_now)
    assert drift.has_drift is False


def test_detect_drift_dirty(tmp_path, monkeypatch):
    from soup_cli.utils.terraform_plan import (
        TrainingState,
        build_plan,
        detect_drift,
    )

    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n")
    config_v1 = {
        "base": "m",
        "task": "sft",
        "data": {"train": str(dataset)},
        "training": {"epochs": 1, "lr": 5e-5, "batch_size": 4},
    }
    plan_v1 = build_plan(config_v1)
    state = TrainingState(plan=plan_v1, applied=False, applied_at=None, run_id=None)
    # Mutate config
    config_v2 = dict(config_v1)
    config_v2["training"] = {"epochs": 2, "lr": 5e-5, "batch_size": 4}
    plan_v2 = build_plan(config_v2)
    drift = detect_drift(state, plan_v2)
    assert drift.has_drift is True
    assert "config_sha" in drift.changed_fields


def test_detect_drift_rejects_non_state():
    from soup_cli.utils.terraform_plan import detect_drift

    with pytest.raises(TypeError):
        detect_drift("not state", "not plan")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def _write_minimal_config(path):
    path.write_text(
        "base: meta-llama/Llama-3.2-1B\n"
        "task: sft\n"
        "data:\n"
        "  train: ./data.jsonl\n"
        "training:\n"
        "  epochs: 1\n"
        "  lr: 0.00005\n"
        "  batch_size: 4\n"
    )


def test_cli_plan_help():
    from soup_cli.cli import app

    result = runner.invoke(app, ["plan", "--help"])
    assert result.exit_code == 0, (result.output, repr(result.exception))


def test_cli_apply_help():
    from soup_cli.cli import app

    result = runner.invoke(app, ["apply", "--help"])
    assert result.exit_code == 0, (result.output, repr(result.exception))


def test_cli_plan_happy(tmp_path, monkeypatch):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "soup.yaml"
    _write_minimal_config(cfg)
    (tmp_path / "data.jsonl").write_text("{}\n")
    result = runner.invoke(app, ["plan", "--config", str(cfg)])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert (tmp_path / "soup.tfstate").exists()


def test_cli_plan_outside_cwd_config(tmp_path, monkeypatch):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    outside_cfg = tmp_path.parent / "evil.yaml"
    _write_minimal_config(outside_cfg)
    result = runner.invoke(app, ["plan", "--config", str(outside_cfg)])
    assert result.exit_code != 0


def test_cli_apply_drift_refused(tmp_path, monkeypatch):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "soup.yaml"
    _write_minimal_config(cfg)
    (tmp_path / "data.jsonl").write_text("{}\n")
    # plan first
    r = runner.invoke(app, ["plan", "--config", str(cfg)])
    assert r.exit_code == 0, (r.output, repr(r.exception))
    # mutate config
    cfg.write_text(cfg.read_text().replace("epochs: 1", "epochs: 99"))
    # apply must refuse on drift
    r2 = runner.invoke(app, ["apply", "--config", str(cfg)])
    assert r2.exit_code != 0
    assert "drift" in r2.output.lower() or "mismatch" in r2.output.lower()


def test_cli_apply_clean(tmp_path, monkeypatch):
    """apply with --dry-run after a plan should succeed clean."""
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "soup.yaml"
    _write_minimal_config(cfg)
    (tmp_path / "data.jsonl").write_text("{}\n")
    r = runner.invoke(app, ["plan", "--config", str(cfg)])
    assert r.exit_code == 0, (r.output, repr(r.exception))
    r2 = runner.invoke(app, ["apply", "--config", str(cfg), "--dry-run"])
    assert r2.exit_code == 0, (r2.output, repr(r2.exception))


# ---------------------------------------------------------------------------
# DriftReport frozen
# ---------------------------------------------------------------------------


def test_drift_report_frozen():
    from soup_cli.utils.terraform_plan import DriftReport

    d = DriftReport(has_drift=False, changed_fields=())
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.has_drift = True  # type: ignore[misc]


def test_drift_report_changed_fields_is_tuple():
    from soup_cli.utils.terraform_plan import DriftReport

    with pytest.raises(TypeError, match="tuple"):
        DriftReport(has_drift=True, changed_fields=["x"])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Source-wiring regression
# ---------------------------------------------------------------------------


def test_cli_registers_plan_apply():
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "soup_cli" / "cli.py"
    text = src.read_text(encoding="utf-8")
    assert '"plan"' in text or "'plan'" in text or "name=\"plan\"" in text


def test_no_heavy_top_level_imports():
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "soup_cli" / "utils" / "terraform_plan.py"
    text = src.read_text(encoding="utf-8")
    import re
    for bad in ["^import torch", "^from torch", "^import transformers", "^from transformers"]:
        assert not re.search(bad, text, re.MULTILINE)
