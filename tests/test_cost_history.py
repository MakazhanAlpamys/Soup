"""Verify history estimates without a GPU or a training run."""

import json
import sqlite3

import pytest

from soup_cli.cli import app
from soup_cli.config.loader import load_config
from soup_cli.experiment.tracker import ExperimentTracker
from soup_cli.utils.cost_history import (
    confidence_for,
    fixed_sft_settings,
    matching_step_times,
)
from tests.conftest import make_cli_runner, strip_ansi

runner = make_cli_runner()


@pytest.fixture
def history_config(tmp_path, monkeypatch):
    history_path = tmp_path / "experiments.db"
    monkeypatch.setenv("SOUP_DB_PATH", str(history_path))
    train = tmp_path / "train.jsonl"
    train.write_text('{"instruction":"short","output":"answer"}\n' * 33, encoding="utf-8")
    config = tmp_path / "soup.yaml"
    config.write_text(
        "base: Qwen/Qwen2.5-1.5B-Instruct\n"
        "backend: transformers\ntask: sft\n"
        f"data:\n  train: {json.dumps(train.as_posix())}\n  max_length: 256\n  val_split: 0\n"
        "training:\n  batch_size: 4\n  gradient_accumulation_steps: 2\n  epochs: 2\n",
        encoding="utf-8",
    )
    return config, history_path


def _record(store, config, run_id, duration, *, name="NVIDIA A100-SXM4-80GB", eligible=True):
    run_id = store.start_run(
        config, "cuda", name, {}, run_id=run_id, cost_history_eligible=eligible,
    )
    store.finish_run(run_id, 2.0, 1.0, 10, duration, "")
    return run_id


def test_cost_history_prices_both_quotes_for_one_gpu(history_config):
    config, history_path = history_config
    settings = load_config(config).model_dump()
    store = ExperimentTracker(history_path)
    for index, duration in enumerate((20, 30, 40)):
        _record(store, settings, f"comparable-{index}", duration)
    _record(store, settings, "resumed", 1, eligible=False)
    _record(store, settings, "legacy-unknown", 75, eligible=None)
    _record(store, settings, "other-gpu", 70, name="NVIDIA A100-SXM4-40GB")
    settings["training"]["batch_size"] = "auto"
    _record(store, settings, "auto-batch", 100)
    store.close()

    args = ["cost", "--config", str(config), "--history", "--gpu", "A100 80G"]
    result = runner.invoke(app, [*args, "--json"])
    assert result.exit_code == 0, (result.output, result.exception)
    quotes = json.loads(result.stdout)
    assert [row["provider"] for row in quotes] == ["RunPod", "CoreWeave"]
    for quote in quotes:
        assert quote["estimate_source"] == "history"
        assert quote["duration_hrs"] == pytest.approx(30 / 3600)
        assert quote["total_cost"] == pytest.approx(quote["duration_hrs"] * quote["cost_per_hr"])
        assert quote["history"] == pytest.approx({
            "matches": 3,
            "optimizer_steps": 10,
            "median_seconds_per_step": 3,
            "min_seconds_per_step": 2,
            "max_seconds_per_step": 4,
            "relative_spread": 2 / 3,
            "confidence": "low",
            "min_duration_hrs": 20 / 3600,
            "max_duration_hrs": 40 / 3600,
        })

    table = runner.invoke(app, args)
    assert table.exit_code == 0
    output = " ".join(strip_ansi(table.stdout).split())
    assert (
        "History estimate from 3 matching runs. Median 3.00 s/step, observed range 2.00 to "
        "4.00 s/step, relative spread 66.7%, confidence low."
    ) in output
    assert "For 10 estimated optimizer steps, the range is 0.33 to 0.67 min." in output


@pytest.mark.parametrize(
    ("gpu", "reason"),
    [
        ([], "Select a single GPU model with --gpu to use history."),
        (["--gpu", "A100"], "Select a single GPU model with --gpu to use history."),
        (["--gpu", "H100"], "Found 0 matching runs for H100; need 3."),
    ],
)
def test_cost_history_fallback_preserves_model_estimate(history_config, gpu, reason):
    config, history_path = history_config
    args = ["cost", "--config", str(config), "--json", *gpu]
    original = runner.invoke(app, args)
    assert original.exit_code == 0
    assert not history_path.exists()
    result = runner.invoke(app, [*args, "--history"])
    assert result.exit_code == 0, (result.output, result.exception)
    assert json.loads(result.stdout) == [
        dict(row, estimate_source="model", estimate_reason=reason)
        for row in json.loads(original.stdout)
    ]
    assert history_path.exists() == (gpu == ["--gpu", "H100"])


@pytest.mark.parametrize(
    ("old", "new", "reason"),
    [
        ("task: sft", "task: dpo", "History is available for SFT only."),
        ("batch_size: 4", "batch_size: auto", "History requires text SFT"),
        ("train.jsonl", "missing.jsonl", "The training row count is unavailable."),
    ],
)
def test_cost_history_reports_unsupported_inputs(history_config, old, new, reason):
    config, history_path = history_config
    config.write_text(config.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
    result = runner.invoke(app, ["cost", "--config", str(config), "--gpu", "H100", "--history"])
    assert result.exit_code == 0, (result.output, result.exception)
    assert f"Model-based estimate. {reason}" in " ".join(strip_ansi(result.stdout).split())
    assert not history_path.exists()


def test_matching_requires_a_completed_run_with_known_provenance(history_config):
    config, history_path = history_config
    payload = load_config(config).model_dump()
    settings = fixed_sft_settings(payload)
    assert settings is not None
    store = ExperimentTracker(history_path)
    run_id = _record(store, payload, "matching", 20, name="NVIDIA H100 80GB HBM3")
    assert matching_step_times(store.list_runs(), settings, "H100") == [2]
    same = store.get_run(run_id)
    mismatches = [
        dict(same, status="failed"),
        dict(same, run_kind="execution"),
        dict(same, device="cpu"),
        dict(same, cost_history_eligible=0),
        dict(same, total_steps=0),
        dict(same, duration_secs=float("inf")),
        dict(same, config_json="broken"),
    ]
    for section, key, value in (
        (None, "base", "other"),
        (None, "backend", "unsloth"),
        ("training", "quantization", "none"),
        ("training", "gradient_accumulation_steps", 3),
        ("training", "packing", True),
        ("data", "max_length", 128),
    ):
        changed = json.loads(same["config_json"])
        (changed[section] if section else changed)[key] = value
        mismatches.append(dict(same, config_json=json.dumps(changed)))
    assert matching_step_times(mismatches, settings, "H100") == []

    # Reusing a row loses the earlier verification until launch conditions are recorded again.
    store.start_run(payload, "cuda", "NVIDIA H100 80GB HBM3", {}, run_id=run_id)
    assert store.get_run(run_id)["cost_history_eligible"] is None
    store.close()


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, True),
        ({"task": "dpo"}, False),
        ({"resume_from": "output/checkpoint-10"}, False),
        ({"device": "cpu"}, False),
        ({"gpu_count": 2}, False),
        ({"gpu_count": None}, False),
        ({"distributed": True}, False),
        ({"fsdp": "full_shard"}, False),
        ({"ds_config_path": "ds.json"}, False),
    ],
    ids=["fresh", "task", "resume", "device", "two-gpus", "unknown-gpus", "ddp", "fsdp", "ds"],
)
def test_training_launch_must_be_fresh_and_single_gpu(overrides, expected):
    from soup_cli.utils.cost_history import cost_history_eligible

    launch = {
        "task": "sft",
        "resume_from": None,
        "device": "cuda",
        "gpu_count": 1,
        "distributed": False,
        "fsdp": None,
        "ds_config_path": None,
    }
    assert cost_history_eligible(**(launch | overrides)) is expected


def test_legacy_history_migrates_in_place_and_old_run_does_not_match(history_config):
    config, history_path = history_config
    payload = load_config(config).model_dump()
    settings = fixed_sft_settings(payload)
    assert settings is not None

    # Create a finished, otherwise comparable row before the eligibility column existed.
    conn = sqlite3.connect(history_path)
    try:
        conn.executescript("""
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'running',
                config_json TEXT NOT NULL,
                device TEXT,
                total_steps INTEGER,
                duration_secs REAL,
                cost_gpu_label TEXT,
                run_kind TEXT NOT NULL DEFAULT 'train'
            );
        """)
        conn.execute(
            """INSERT INTO runs (run_id, created_at, status, config_json, device,
                    total_steps, duration_secs, cost_gpu_label)
                VALUES ('legacy', '2026-01-01T00:00:00', 'completed', ?, 'cuda', 10, 20, 'H100')""",
            (json.dumps(payload),),
        )
        conn.commit()
    finally:
        conn.close()

    store = ExperimentTracker(history_path)
    try:
        row = store.get_run("legacy")
        assert row["cost_history_eligible"] is None
        assert matching_step_times([row], settings, "H100") == []
        assert matching_step_times([dict(row, cost_history_eligible=1)], settings, "H100") == [2]
        schema_version = store._get_conn().execute("PRAGMA schema_version").fetchone()[0]
    finally:
        store.close()

    reopened = ExperimentTracker(history_path)
    try:
        assert reopened.get_run("legacy") == row
        assert reopened._get_conn().execute("PRAGMA schema_version").fetchone()[0] == schema_version
    finally:
        reopened.close()


@pytest.mark.parametrize(
    ("count", "spread", "confidence"),
    [(3, 0, "low"), (5, 0.50, "medium"), (5, 0.51, "low"), (8, 0.20, "high"), (8, 0.21, "medium")],
)
def test_confidence_has_fixed_count_and_spread_cutoffs(count, spread, confidence):
    assert confidence_for(count, spread) == confidence
