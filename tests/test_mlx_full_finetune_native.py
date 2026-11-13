"""#1992: real offline Llama updates, weighted reference, and standalone reload."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import numpy as np  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.tuner.datasets import CacheDataset  # noqa: E402

from soup_cli.config.loader import load_config_from_string  # noqa: E402
from soup_cli.trainer.mlx_full import prepare_full_model  # noqa: E402
from soup_cli.trainer.mlx_masking import (  # noqa: E402
    MaskedChatDataset,
    masked_iterate_batches,
)
from soup_cli.trainer.mlx_routing import resolve_trainer  # noqa: E402
from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper  # noqa: E402
from tests.conftest import strip_ansi  # noqa: E402
from tests.test_mlx_dora_native import _checkpoint  # noqa: E402


def _config(base, output, **training):
    return load_config_from_string(yaml.safe_dump({
        "base": str(base), "backend": "mlx", "task": "sft", "output": str(output),
        "data": {"train": "./train.jsonl", "max_length": 64, "val_split": 0},
        "training": {
            "quantization": "none", "lora": {"r": 0}, "batch_size": 1,
            "epochs": 1, "gradient_accumulation_steps": 2, "lr": 0.01,
            "scheduler": "constant", "warmup_ratio": 0, "weight_decay": 0,
            "logging_steps": 2, "save_steps": 2, "max_grad_norm": 0.1,
            "seed": 17, "data_seed": 19, "rewind_log": False, **training,
        },
    }))


def _rows():
    return {"train": [{"messages": [
        {"role": "user", "content": "gamma beta"},
        {"role": "assistant", "content": completion},
    ]} for completion in ["alpha", "beta gamma alpha beta", "gamma alpha", "beta gamma alpha"]]}


def _capture_optimizer(monkeypatch):
    from soup_cli.trainer import mlx_optim

    build = mlx_optim.build_optimizer
    captured = []

    def capture(plan):
        optimizer = build(plan)
        captured.append(optimizer)
        return optimizer

    monkeypatch.setattr(mlx_optim, "build_optimizer", capture)
    return captured


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("freezing", [{}, {"freeze_layers": 1}, {"freeze_ratio": 0.5}])
def test_public_update_freeze_counts_complete_save_and_fresh_reload(
    tmp_path, monkeypatch, capsys, tied, freezing,
):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    base = _checkpoint(tmp_path / "base", layers=2, tied=tied)
    fixture_hash = hashlib.sha256((base / "model.safetensors").read_bytes()).hexdigest()
    output = tmp_path / "trained"
    config = _config(base, output, **freezing)
    trainer_type, kwargs = resolve_trainer(config)
    wrapper = trainer_type(config, **kwargs)
    wrapper.setup(_rows())
    before = dict(tree_flatten(wrapper.model.parameters()))
    mx.eval(before)
    from soup_cli.trainer import mlx_full

    save = mlx_full.save_full_model
    saved_logits = {}
    probe = mx.array([[3, 5, 6, 7]])

    def observe_save(path, model, tokenizer, model_config, metadata):
        saved_logits[path.name] = model(probe)
        mx.eval(saved_logits[path.name])
        save(path, model, tokenizer, model_config, metadata)

    monkeypatch.setattr(mlx_full, "save_full_model", observe_save)
    optimizer = _capture_optimizer(monkeypatch)
    mx.reset_peak_memory()
    result = wrapper.train()
    peak_bytes = mx.get_peak_memory()
    after = dict(tree_flatten(wrapper.model.parameters()))
    assert after.keys() == before.keys()
    cutoff = 1 if freezing else 0
    frozen_names = {name for name in before if cutoff and name.startswith("model.layers.0.")}
    deltas = {}
    for name, initial in before.items():
        delta = mx.abs(after[name] - initial).max().item()
        deltas[name] = delta
        if name in frozen_names:
            assert mx.array_equal(after[name], initial).item(), name
        else:
            assert delta > 0, name
    assert any(name.startswith("model.embed_tokens.") for name in before)
    assert any(name.startswith("model.norm.") for name in before)
    assert ("lm_head.weight" not in before) == tied
    states = dict(tree_flatten(optimizer[0].state))
    moment_names = {name.removesuffix(".m") for name in states if name.endswith(".m")}
    assert moment_names == before.keys() - frozen_names
    assert all(states[f"{name}.m"].dtype == after[name].dtype for name in moment_names)
    metadata = json.loads((output / "training_config.json").read_text())
    trainable_count = sum(before[name].size for name in before.keys() - frozen_names)
    frozen_count = sum(before[name].size for name in frozen_names)
    assert metadata["fine_tune_type"] == "full"
    assert metadata["trainable_parameters"] == trainable_count
    assert metadata["frozen_parameters"] == frozen_count
    assert metadata["trainable_names"] == sorted(before.keys() - frozen_names)
    assert metadata["frozen_names"] == sorted(frozen_names)
    assert metadata["completed_updates"] == optimizer[0].step.item() == 2
    assert metadata["resume_training_state"] is False
    text = " ".join(strip_ansi(capsys.readouterr().out).split())
    assert f"{trainable_count:,} trainable parameters" in text
    assert f"{frozen_count:,} frozen parameters" in text
    wrapper.model.eval()
    expected_logits = wrapper.model(probe)
    mx.eval(expected_logits)
    checkpoints = sorted(output.glob("checkpoint-*"))
    assert [path.name for path in checkpoints] == ["checkpoint-0000002", "checkpoint-0000004"]
    for artifact in [output, *checkpoints]:
        weights = mx.load(str(artifact / "model.safetensors"))
        assert weights.keys() == before.keys()
        index = json.loads((artifact / "model.safetensors.index.json").read_text())
        assert index["weight_map"].keys() == before.keys()
        assert not (artifact / "adapters.safetensors").exists()
        assert not (artifact / "adapter_config.json").exists()
        for name in frozen_names:
            assert mx.array_equal(weights[name], before[name]).item(), name
    # The source is removed before a new interpreter imports MLX and loads each artifact.
    shutil.rmtree(base)
    script = '''
import sys
import mlx.core as mx
from mlx_lm import load
model, tokenizer = load(sys.argv[1])
model.eval()
assert tokenizer.encode("alpha beta")
mx.save_safetensors(sys.argv[2], {"logits": model(mx.array([[3, 5, 6, 7]]))})
'''
    for artifact in [output, *checkpoints]:
        logits_path = tmp_path / f"{artifact.name}-logits.safetensors"
        child = subprocess.run(
            [sys.executable, "-c", script, str(artifact), str(logits_path)],
            capture_output=True, text=True, timeout=60,
        )
        assert child.returncode == 0, child.stdout + child.stderr
        reloaded_logits = mx.load(str(logits_path))["logits"]
        assert mx.array_equal(reloaded_logits, saved_logits[artifact.name]).item()
    assert mx.array_equal(saved_logits[output.name], expected_logits).item()
    evidence_dir = os.environ.get("SOUP_1992_EVIDENCE_DIR")
    if evidence_dir:
        Path(evidence_dir).mkdir(parents=True, exist_ok=True)
        evidence_name = f"tied-{tied}-freeze-{cutoff}-{next(iter(freezing), 'none')}.json"
        (Path(evidence_dir) / evidence_name).write_text(
            json.dumps({
                "fixture_sha256": fixture_hash, "fixture": "offline seeded native Llama",
                "seed": 7, "data": _rows(), "config": config.model_dump(mode="json"),
                "training": result, "peak_bytes": peak_bytes, "metadata": metadata,
                "parameter_max_abs_deltas": deltas,
                "optimizer_state_bytes": sum(value.nbytes for value in states.values()),
                "moment_dtypes": sorted({str(states[f"{name}.m"].dtype) for name in moment_names}),
                "reload_max_abs_error": 0.0,
            }, indent=2) + "\n"
        )


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_accumulated_masked_update_matches_independent_reference(
    tmp_path, monkeypatch, tied, checkpointing,
):
    import mlx.optimizers as optim

    base = _checkpoint(tmp_path / "base", layers=2, tied=tied)
    config = _config(base, tmp_path / "out", gradient_checkpointing=checkpointing, optimizer="sgd")
    wrapper = MLXSFTTrainerWrapper(config)
    wrapper.setup(_rows())
    reference, _ = load(str(base))
    reference.unfreeze()
    reference.train()
    dataset = CacheDataset(MaskedChatDataset(_rows()["train"], wrapper.tokenizer))
    np.random.seed(config.training.data_seed)
    batches = list(masked_iterate_batches(dataset, 1, 64))
    counts = [int(mask[:, 1:].sum().item()) for _, mask in batches]
    assert len(set(counts[:2])) > 1  # Equal micro-batch means must give the wrong update.

    def oracle(model, pair):
        total, count = mx.array(0.0), mx.array(0)
        for tokens, mask in pair:
            logits = model(tokens[:, :-1]).astype(mx.float32)
            targets = tokens[:, 1:]
            losses = mx.logsumexp(logits, axis=-1) - mx.take_along_axis(
                logits, targets[..., None], axis=-1,
            ).squeeze(-1)
            supervised = mask[:, 1:]
            total = total + (losses * supervised).sum()
            count = count + supervised.sum()
        return total / count

    optimizer = optim.SGD(learning_rate=config.training.lr)
    gradient = nn.value_and_grad(reference, oracle)
    layer_call = type(wrapper.model.layers[0]).__call__
    for start in range(0, len(batches), 2):
        _, gradients = gradient(reference, batches[start:start + 2])
        flat = tree_flatten(gradients)
        norm = mx.sqrt(sum((value.astype(mx.float32) ** 2).sum() for _, value in flat))
        assert norm.item() > config.training.max_grad_norm
        from mlx.utils import tree_map

        clipped = tree_map(
            lambda value: value * config.training.max_grad_norm / (norm + 1e-6), gradients,
        )
        optimizer.update(reference, clipped)
        mx.eval(reference.state, optimizer.state)
    wrapper.train()
    assert type(wrapper.model.layers[0]).__call__ is layer_call
    expected = dict(tree_flatten(reference.parameters()))
    for name, value in tree_flatten(wrapper.model.parameters()):
        assert mx.allclose(value, expected[name], atol=2e-7, rtol=2e-6).item(), name


@pytest.mark.parametrize("problem,match", [
    ("quantized", "quantized module"), ("layout", "unknown layer layout"),
    ("out_of_range", "out of range"), ("all_frozen", "nothing trainable"),
])
def test_loaded_model_refusals(tmp_path, monkeypatch, problem, match):
    base = _checkpoint(tmp_path / "base", layers=2, quantized=problem == "quantized")
    config = _config(base, tmp_path / "out", **({"freeze_layers": 3}
                                              if problem == "out_of_range" else {}))
    if problem == "layout":
        from soup_cli.utils import mlx as mlx_utils

        model, tokenizer, model_config = load(str(base), return_config=True)
        model.model_type = "unknown"
        monkeypatch.setattr(mlx_utils, "load_mlx_model",
                            lambda *args, **kwargs: (model, tokenizer, model_config))
    if problem == "all_frozen":
        from mlx_lm.models.llama import Model

        monkeypatch.setattr(Model, "trainable_parameters", lambda self: {})
    with pytest.raises(ValueError, match=match):
        MLXSFTTrainerWrapper(config).setup(_rows())


def test_full_resume_refused_before_training(tmp_path):
    config = _config("./unavailable-base", tmp_path / "out")
    with pytest.raises(ValueError, match="--resume.*optimizer/RNG-state"):
        MLXSFTTrainerWrapper(config).train(resume_from_checkpoint="checkpoint")
    assert not (tmp_path / "out").exists()


def test_whole_layer_freeze_keeps_embedding_norm_and_head_trainable(tmp_path):
    base = _checkpoint(tmp_path / "base", layers=2)
    model, _ = load(str(base))
    plan = prepare_full_model(model, _config(base, tmp_path / "out", freeze_layers=2))
    assert set(plan["trainable_names"]) == {
        "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
    }


def test_freeze_layers_takes_precedence_over_ratio(tmp_path):
    base = _checkpoint(tmp_path / "base", layers=3)
    model, _ = load(str(base))
    plan = prepare_full_model(model, _config(base, tmp_path / "out", freeze_layers=1,
                                           freeze_ratio=0.9))
    assert plan["freeze_cutoff"] == 1
    assert not any(name.startswith("model.layers.1.") for name in plan["frozen_names"])


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_low_precision_updates_and_adam_moment_dtype(tmp_path, monkeypatch, dtype):
    base = _checkpoint(tmp_path / "base", layers=2)
    path = str(base / "model.safetensors")
    converted = {name: value.astype(dtype) for name, value in mx.load(path).items()}
    mx.eval(converted)  # Materialize casts before replacing their memory-mapped source file.
    mx.save_safetensors(path, converted)
    config = _config(base, tmp_path / "out", freeze_layers=1)
    wrapper = MLXSFTTrainerWrapper(config)
    wrapper.setup(_rows())
    initial = dict(tree_flatten(wrapper.model.parameters()))
    mx.eval(initial)
    captured = _capture_optimizer(monkeypatch)
    wrapper.train()
    updated = dict(tree_flatten(wrapper.model.parameters()))
    for name in wrapper._full_model_metadata["trainable_names"]:
        assert updated[name].dtype == mx.float32
        assert not mx.array_equal(updated[name], initial[name]).item(), name
        assert mx.all(mx.isfinite(updated[name])).item(), name
    states = dict(tree_flatten(captured[0].state))
    moments = [value for name, value in states.items() if name.endswith((".m", ".v"))]
    assert moments and all(value.dtype == mx.float32 for value in moments)
    for name in wrapper._full_model_metadata["frozen_names"]:
        assert updated[name].dtype == dtype
        assert mx.array_equal(updated[name], initial[name]).item(), name
    restored, _ = load(str(tmp_path / "out"))
    for name, value in tree_flatten(restored.parameters()):
        assert mx.array_equal(value, updated[name]).item(), name


@pytest.mark.parametrize("previous", ["adapter", "intermediate"])
def test_existing_artifact_is_refused_without_overwriting(tmp_path, previous):
    base = _checkpoint(tmp_path / "base", layers=2)
    wrapper = MLXSFTTrainerWrapper(_config(base, tmp_path / "out"))
    wrapper.setup(_rows())
    output = tmp_path / "out"
    output.mkdir()
    if previous == "intermediate":
        output = output / "checkpoint-0000002"
        output.mkdir()
    artifact = output / "adapters.safetensors"
    artifact.write_bytes(b"preserve")
    with pytest.raises(ValueError, match="already contains"):
        wrapper.train()
    assert artifact.read_bytes() == b"preserve"


def test_public_cli_trains_and_reports_standalone_model(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "runs.db"))
    base = _checkpoint(tmp_path / "base", layers=2)
    config = _config(base, tmp_path / "out", freeze_layers=1)
    config.training.rewind_log = True
    data = tmp_path / "train.jsonl"
    data.write_text("\n".join(json.dumps(row) for row in _rows()["train"]) + "\n")
    config.data.train = str(data)
    path = tmp_path / "soup.yaml"
    path.write_text(yaml.safe_dump(config.model_dump(mode="json")))
    result = CliRunner().invoke(app, ["train", "--config", str(path), "--yes"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = " ".join(strip_ansi(result.output).split())
    assert "standalone model; no soup merge is needed" in text
    rewind = [json.loads(line) for line in (tmp_path / "out" / "rewind.jsonl")
              .read_text().splitlines()]
    assert len([record for record in rewind if "row_tokens" in record]) == 4
    restored, _ = load(str(tmp_path / "out"))
    assert mx.isfinite(restored(mx.array([[3, 5, 6]]))).all().item()
    initial = mx.load(str(base / "model.safetensors"))
    for name, value in tree_flatten(restored.parameters()):
        if name.startswith("model.layers.0."):
            assert mx.array_equal(value, initial[name]).item(), name
        else:
            assert not mx.array_equal(value, initial[name]).item(), name


def test_dense_watchdog_evaluation_and_mid_group_save(tmp_path):
    base = _checkpoint(tmp_path / "base", layers=2)
    config = _config(base, tmp_path / "out", save_steps=3, logging_steps=1,
                     loss_watchdog=True, loss_watchdog_threshold=0.01,
                     loss_watchdog_patience=2)
    config.training.eval_steps = 1
    config.data.val_split = 0.2
    config = load_config_from_string(yaml.safe_dump(config.model_dump(mode="json")))
    wrapper = MLXSFTTrainerWrapper(config)
    rows = _rows()
    rows["val"] = rows["train"][:1]
    wrapper.setup(rows)
    result = wrapper.train()
    assert result["stop_reason"] == "loss_watchdog"
    assert result["completed_updates"] == 2
    assert result["eval_updates"] == [1, 2]
    assert [row["optimizer_step"] for row in wrapper._eval_log] == [1, 2]
    assert [path.name for path in (tmp_path / "out").glob("checkpoint-*")] == [
        "checkpoint-0000004",
    ]
    metadata = json.loads((tmp_path / "out" / "training_config.json").read_text())
    assert metadata["stop_reason"] == "loss_watchdog"
    assert metadata["completed_iterations"] == 4
    saved, _ = load(str(tmp_path / "out"))
    for name, value in tree_flatten(wrapper.model.parameters()):
        assert mx.array_equal(value, dict(tree_flatten(saved.parameters()))[name]).item(), name


@pytest.mark.parametrize("invalid", ["zero_tokens", "non_finite"])
def test_invalid_loss_refused_before_update_and_checkpoint_patch_restored(
    tmp_path, monkeypatch, invalid,
):
    from soup_cli.trainer import mlx_masking

    base = _checkpoint(tmp_path / "base", layers=2)
    config = _config(base, tmp_path / "out", gradient_checkpointing=True)
    wrapper = MLXSFTTrainerWrapper(config)
    wrapper.setup(_rows())
    original_loss = mlx_masking.masked_loss
    captured = _capture_optimizer(monkeypatch)
    original_call = type(wrapper.model.layers[0]).__call__

    def invalid_loss(model, *batch):
        value, count = original_loss(model, *batch)
        if invalid == "zero_tokens":
            return value * 0, mx.array(0)
        return value * float("nan"), count

    monkeypatch.setattr(mlx_masking, "masked_loss", invalid_loss)
    with pytest.raises(FloatingPointError, match="empty or non-finite supervised loss"):
        wrapper.train()
    assert captured[0].step.item() == 0
    assert type(wrapper.model.layers[0]).__call__ is original_call
    assert not (tmp_path / "out" / "model.safetensors").exists()
