"""Candidate-bound live mix scores, exercised against the real SQLite tracker."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from soup_cli.experiment.tracker import ExperimentTracker
from soup_cli.utils import mix_proxy


@pytest.fixture
def tracker(tmp_path):
    instance = ExperimentTracker(db_path=tmp_path / "runs.sqlite")
    try:
        yield instance
    finally:
        instance.close()


def _candidate(tracker, output, *, run_id="candidate", final_loss=0.001, steps=5):
    output = str(output)
    tracker.start_run(
        {"base": "test-base", "task": "sft", "output": output},
        "cpu", "CPU", {}, run_id=run_id,
    )
    tracker.finish_run(
        run_id, initial_loss=9.0, final_loss=final_loss, total_steps=steps,
        duration_secs=1.0, output_dir=output,
    )
    return run_id


def _read(tracker, output, run_id="candidate"):
    return mix_proxy._read_final_eval_loss(str(tracker.db_path), run_id, str(output))


@pytest.fixture
def base_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("a.jsonl", "b.jsonl"):
        (tmp_path / name).write_text('{"text":"one"}\n{"text":"two"}\n')
    path = tmp_path / "soup.yaml"
    path.write_text(yaml.safe_dump({
        "base": "test-base", "task": "sft", "backend": "transformers",
        "data": {"train": "a.jsonl", "format": "plaintext", "val_split": 0.2},
        "training": {"epochs": 1}, "output": "original-output",
    }))
    return path


def test_public_proxy_cannot_score_an_unrelated_completed_run(base_yaml, tracker, monkeypatch):
    """Regression reproducer: old code silently returned this stale training loss."""
    _candidate(tracker, "unrelated-output", run_id="unrelated", final_loss=0.001)
    monkeypatch.setattr(
        mix_proxy.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=b"", stderr=b""),
    )
    with pytest.raises(RuntimeError):
        mix_proxy.proxy_run_for_weights(
            [0.5, 0.5], ["a.jsonl", "b.jsonl"], str(base_yaml),
            tracker_db_path=str(tracker.db_path),
        )


def test_reader_uses_final_validation_loss_not_training_summary(tmp_path, tracker):
    output = tmp_path / "candidate-output"
    run_id = _candidate(tracker, output, final_loss=0.001)
    tracker.log_metrics(run_id, step=5, loss=0.001, val_loss=2.75, kind="eval")
    assert _read(tracker, output) == pytest.approx(2.75)


def test_reader_accepts_genuinely_measured_zero(tmp_path, tracker):
    output = tmp_path / "candidate-output"
    run_id = _candidate(tracker, output)
    tracker.log_metrics(run_id, step=5, val_loss=0.0, kind="eval")
    assert _read(tracker, output) == 0.0


def test_reader_refuses_prefix_fallback(tmp_path, tracker):
    output = tmp_path / "candidate-output"
    run_id = _candidate(tracker, output, run_id="candidate-longer-id")
    tracker.log_metrics(run_id, step=5, val_loss=0.5, kind="eval")
    # get_run itself permits a unique prefix, but evidence must name the exact run.
    assert tracker.get_run("candidate")["run_id"] == run_id
    with pytest.raises(RuntimeError):
        _read(tracker, output, "candidate")


@pytest.mark.parametrize("status", ["running", "failed", "terminated"])
def test_reader_requires_completed_candidate(tmp_path, tracker, status):
    output = tmp_path / "candidate-output"
    run_id = _candidate(tracker, output)
    tracker.log_metrics(run_id, step=5, val_loss=0.25, kind="eval")
    tracker._get_conn().execute("UPDATE runs SET status=? WHERE run_id=?", (status, run_id))
    tracker._get_conn().commit()
    with pytest.raises(RuntimeError):
        _read(tracker, output)


def test_reader_refuses_missing_validation_even_with_completed_training(tmp_path, tracker):
    output = tmp_path / "candidate-output"
    run_id = _candidate(tracker, output, final_loss=0.0)
    tracker.log_metrics(run_id, step=5, loss=0.0, kind="train")
    with pytest.raises(RuntimeError):
        _read(tracker, output)


@pytest.mark.parametrize("kind", [None, "train", "summary"])
def test_reader_requires_explicit_evaluation_row_kind(tmp_path, tracker, kind):
    output = tmp_path / "candidate-output"
    run_id = _candidate(tracker, output)
    tracker.log_metrics(run_id, step=5, val_loss=0.01, kind=kind)
    with pytest.raises(RuntimeError):
        _read(tracker, output)


@pytest.mark.parametrize("step", [0, 4, 6])
def test_reader_refuses_validation_not_at_final_completed_step(tmp_path, tracker, step):
    output = tmp_path / "candidate-output"
    run_id = _candidate(tracker, output)
    tracker.log_metrics(run_id, step=step, val_loss=0.25, kind="eval")
    with pytest.raises(RuntimeError):
        _read(tracker, output)


@pytest.mark.parametrize("bad_loss", [None, float("nan"), float("inf"), -0.5, "not-a-loss"])
def test_reader_does_not_fall_back_from_invalid_latest_evaluation(
    tmp_path, tracker, bad_loss,
):
    output = tmp_path / "candidate-output"
    run_id = _candidate(tracker, output)
    tracker.log_metrics(run_id, step=5, val_loss=0.25, kind="eval")
    tracker.log_metrics(run_id, step=5, val_loss=bad_loss, kind="eval")
    with pytest.raises(RuntimeError):
        _read(tracker, output)


def test_reader_uses_arrival_order_not_largest_step(tmp_path, tracker):
    output = tmp_path / "candidate-output"
    run_id = _candidate(tracker, output)
    tracker.log_metrics(run_id, step=5, val_loss=0.25, kind="eval")
    tracker.log_metrics(run_id, step=4, val_loss=0.01, kind="eval")
    # A later conflicting evaluation must not be hidden by get_metrics' step sort.
    with pytest.raises(RuntimeError):
        _read(tracker, output)


@pytest.mark.parametrize("field", ["output_dir", "config_json"])
def test_reader_binds_both_declared_and_completed_output(tmp_path, tracker, field):
    output = tmp_path / "candidate-output"
    run_id = _candidate(tracker, output)
    tracker.log_metrics(run_id, step=5, val_loss=0.25, kind="eval")
    if field == "output_dir":
        tracker._get_conn().execute(
            "UPDATE runs SET output_dir=? WHERE run_id=?", (str(tmp_path / "other"), run_id),
        )
    else:
        tracker._get_conn().execute(
            "UPDATE runs SET config_json=? WHERE run_id=?", ('{"output":"different"}', run_id),
        )
    tracker._get_conn().commit()
    with pytest.raises(RuntimeError):
        _read(tracker, output)


def test_reader_canonicalizes_output_and_leaves_environment_untouched(
    tmp_path, tracker, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_DB_PATH", "parent-db-must-not-change")
    output = tmp_path / "candidate-output"
    run_id = _candidate(tracker, "./candidate-output")
    tracker.log_metrics(run_id, step=5, val_loss=0.25, kind="eval")
    assert _read(tracker, output / ".." / "candidate-output") == pytest.approx(0.25)
    assert os.environ["SOUP_DB_PATH"] == "parent-db-must-not-change"


@pytest.mark.parametrize("block,key,value", [
    (None, "task", "dpo"),
    (None, "backend", "mlx"),
    (None, "modality", "vision"),
    ("data", "streaming", True),
    ("data", "val_split", 0.0),
    ("data", "val_split", float("nan")),
    ("data", "format", "pre_tokenized"),
])
def test_live_proxy_config_rejects_unsupported_or_unmeasured_objectives(
    base_yaml, block, key, value,
):
    config = yaml.safe_load(base_yaml.read_text())
    (config if block is None else config[block])[key] = value
    if block == "data" and key == "format" and value == "pre_tokenized":
        config["data"]["tokenized_path"] = "cache"
    base_yaml.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        mix_proxy.validate_proxy_config(str(base_yaml))


def test_live_proxy_config_accepts_measured_sft_development_split(base_yaml):
    assert mix_proxy.validate_proxy_config(str(base_yaml)) is None


def _write_child_result(env, config, *, loss, add_unrelated=False):
    db = ExperimentTracker(db_path=Path(env["SOUP_DB_PATH"]))
    try:
        run_id = env["SOUP_MCP_RUN_ID"]
        db.start_run(config, "cpu", "CPU", {}, run_id=run_id)
        db.log_metrics(run_id, step=3, loss=999.0, val_loss=loss, kind="eval")
        db.finish_run(run_id, 999.0, 999.0, 3, 0.01, config["output"])
        if add_unrelated:
            other = _candidate(db, "unrelated-output", run_id="concurrent-unrelated")
            db.log_metrics(other, step=5, val_loss=0.0001, kind="eval")
    finally:
        db.close()


@pytest.mark.parametrize("experiment_name", ["caller-experiment", ".."])
def test_public_candidates_isolate_run_database_output_and_parent_environment(
    base_yaml, monkeypatch, experiment_name,
):
    monkeypatch.setenv("SOUP_MCP_RUN_ID", "parent-mcp-run")
    monkeypatch.setenv("SOUP_DB_PATH", "parent-tracker.sqlite")
    base_config = yaml.safe_load(base_yaml.read_text())
    base_config["experiment_name"] = experiment_name
    base_yaml.write_text(yaml.safe_dump(base_config))
    original_yaml = base_yaml.read_bytes()
    original = Path("original-output")
    original.mkdir()
    sentinel = original / "adapter.safetensors"
    sentinel.write_bytes(b"original-model-must-survive")
    observed = []

    def subprocess_stub(argv, **kwargs):
        env = kwargs["env"]
        path = Path(argv[argv.index("--config") + 1])
        config = yaml.safe_load(path.read_text())
        # SFT appends experiment_name to output; ".." is currently schema-valid.
        # The proxy must clear it to bind the actual artifact directory exactly.
        assert config.get("experiment_name") is None
        output = Path(config["output"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "candidate-artifact").write_text("temporary")
        observed.append((env.copy(), config, path))
        _write_child_result(env, config, loss=0.25 * len(observed))
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(mix_proxy.subprocess, "run", subprocess_stub)
    results = [
        mix_proxy.proxy_run_for_weights(weights, ["a.jsonl", "b.jsonl"], str(base_yaml))
        for weights in ([0.25, 0.75], [0.75, 0.25])
    ]
    assert results == pytest.approx([0.25, 0.5])
    assert len({env["SOUP_MCP_RUN_ID"] for env, _, _ in observed}) == 2
    assert len({env["SOUP_DB_PATH"] for env, _, _ in observed}) == 2
    assert len({config["output"] for _, config, _ in observed}) == 2
    for env, config, path in observed:
        assert env["SOUP_MCP_RUN_ID"] != "parent-mcp-run"
        assert env["SOUP_DB_PATH"] != "parent-tracker.sqlite"
        assert os.path.realpath(config["output"]) != os.path.realpath(original)
        assert not path.exists()
        assert not Path(config["output"]).exists()
    assert observed[0][1]["data"]["interleave"]["probs"] == [0.25, 0.75]
    assert observed[1][1]["data"]["interleave"]["probs"] == [0.75, 0.25]
    assert os.environ["SOUP_MCP_RUN_ID"] == "parent-mcp-run"
    assert os.environ["SOUP_DB_PATH"] == "parent-tracker.sqlite"
    assert sentinel.read_bytes() == b"original-model-must-survive"
    assert base_yaml.read_bytes() == original_yaml


def test_explicit_shared_tracker_still_scores_only_the_exact_child(
    base_yaml, tracker, monkeypatch,
):
    def subprocess_stub(argv, **kwargs):
        env = kwargs["env"]
        config = yaml.safe_load(Path(argv[argv.index("--config") + 1]).read_text())
        assert os.path.realpath(env["SOUP_DB_PATH"]) == os.path.realpath(tracker.db_path)
        _write_child_result(env, config, loss=1.5, add_unrelated=True)
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(mix_proxy.subprocess, "run", subprocess_stub)
    assert mix_proxy.proxy_run_for_weights(
        [0.5, 0.5], ["a.jsonl", "b.jsonl"], str(base_yaml),
        tracker_db_path=str(tracker.db_path),
    ) == pytest.approx(1.5)


def test_live_proxy_evaluates_tiny_local_model_end_to_end(
    tmp_path, monkeypatch, record_property,
):
    """Offline CPU training proves orchestration, not useful model quality."""
    from tests.test_issue1223_evaluate_val_split import (
        _requires_train_extra,
        _tiny_llama_dir,
    )

    _requires_train_extra()
    monkeypatch.chdir(tmp_path)
    for name, value in {
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "WANDB_DISABLED": "true",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "SOUP_DB_PATH": str(tmp_path / "parent-tracker.sqlite"),
        "SOUP_MCP_RUN_ID": "parent-run-must-not-be-reused",
    }.items():
        monkeypatch.setenv(name, value)
    weights = _tiny_llama_dir(tmp_path)
    sources = []
    for name, prompt in (("a.jsonl", "hello"), ("b.jsonl", "hi")):
        path = tmp_path / name
        rows = [{
            "messages": [
                {"role": "user", "content": f"{prompt} {i}"},
                {"role": "assistant", "content": "good answer"},
            ],
        } for i in range(8)]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        sources.append(str(path))

    original = tmp_path / "original-output"
    original.mkdir()
    sentinel = original / "adapter.safetensors"
    sentinel.write_bytes(b"original-model-must-survive")
    base = tmp_path / "soup.yaml"
    base.write_text(yaml.safe_dump({
        "base": weights, "task": "sft", "backend": "transformers",
        "data": {
            "train": sources[0], "format": "chatml", "chat_template": "chatml",
            "max_length": 64, "val_split": 0.25,
        },
        "training": {
            "epochs": 1, "batch_size": 2, "gradient_accumulation_steps": 1,
            "logging_steps": 1, "save_steps": 1000, "quantization": "none",
            "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
        },
        "experiment_name": "caller-experiment", "output": str(original),
    }), encoding="utf-8")
    original_yaml = base.read_bytes()

    # No subprocess/trainer/tracker stubs: the installed CLI trains and evaluates
    # a randomly initialized local checkpoint, then reads its actual SQLite row.
    score = mix_proxy.proxy_run_for_weights([0.5, 0.5], sources, str(base))

    assert math.isfinite(score) and score >= 0.0
    record_property("tiny_cpu_proxy_validation_loss", score)
    assert sentinel.read_bytes() == b"original-model-must-survive"
    assert sorted(path.name for path in original.iterdir()) == [sentinel.name]
    assert base.read_bytes() == original_yaml
    assert not list(tmp_path.glob(".soup_mix_proxy.*"))
    assert not (tmp_path / "parent-tracker.sqlite").exists()
    assert os.environ["SOUP_DB_PATH"] == str(tmp_path / "parent-tracker.sqlite")
    assert os.environ["SOUP_MCP_RUN_ID"] == "parent-run-must-not-be-reused"
