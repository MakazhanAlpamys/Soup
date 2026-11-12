"""Actual compiled MLX uninterrupted versus fresh-process continuation (#1977)."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from tests.test_mlx_dora_native import _checkpoint  # noqa: E402


@pytest.mark.parametrize(
    "quantized,accumulation,scheduler,checkpoint,masked,mid_window,cut",
    [
        (False, 1, "constant", False, False, False, 1),
        (False, 1, "cosine", False, False, False, 1),
        (True, 2, "cosine", False, False, False, 2),
        (False, 2, "linear", False, False, True, 1),
        (True, 1, "constant_with_warmup", False, False, False, 7),
        (False, 2, "cosine", True, True, True, 1),
        (True, 2, "linear", True, True, False, 3),
    ],
)
def test_fresh_process_exact_continuation(
    tmp_path, quantized, accumulation, scheduler, checkpoint, masked, mid_window, cut, monkeypatch
):
    import mlx.core as mx

    monkeypatch.chdir(tmp_path)
    shadow = tmp_path / 'src' / 'soup_cli'
    shadow.mkdir(parents=True)
    (shadow / '__init__.py').write_text("raise RuntimeError('unrelated cwd checkout imported')")
    base = _checkpoint(tmp_path / "base", quantized)
    rows = {
        "train": [
            {"text": text}
            for text in [
                "alpha beta gamma",
                "beta gamma alpha beta",
                "gamma alpha",
                "alpha beta gamma beta",
            ]
        ],
        "val": [{"text": "gamma beta alpha"}],
    }
    if masked:
        rows = {
            split: [
                {
                    "messages": [
                        {"role": "user", "content": row["text"]},
                        {"role": "assistant", "content": "beta gamma"},
                        {"role": "user", "content": "alpha beta"},
                        {"role": "assistant", "content": "gamma alpha"},
                    ]
                }
                for row in data
            ]
            for split, data in rows.items()
        }
    row_file = tmp_path / "rows.json"
    row_file.write_text(json.dumps(rows))
    cfg = {
        "base": str(base),
        "backend": "mlx",
        "data": {"train": str(row_file), "train_on_responses_only": masked, "max_length": 64},
        "training": {
            "mlx_full_state": True,
            "epochs": 2,
            "batch_size": 1,
            "gradient_accumulation_steps": accumulation,
            "gradient_checkpointing": checkpoint,
            "seed": 7,
            "data_seed": 0,
            "rewind_log": False,
            "save_steps": 1,
            "logging_steps": 1,
            "lr": 0.004,
            "scheduler": scheduler,
            "warmup_ratio": 0.25,
            "weight_decay": 0 if checkpoint else 0.07,
            "optimizer": "adamw_hf" if checkpoint else "adamw_torch",
            "eval_steps": 1,
            "quantization": "4bit" if quantized else "none",
            "lora": {"r": 4, "alpha": 8, "dropout": 0.2},
        },
    }
    total = 8 // accumulation
    worker = Path(__file__).with_name("_mlx_full_state_worker.py")

    def run(name, output, resume=None, stop=None, stop_micro=None):
        config = tmp_path / f"{name}.json"
        config.write_text(json.dumps({**cfg, "output": str(output)}))
        result = tmp_path / f"{name}-result.json"
        cmd = [
            sys.executable,
            str(worker),
            "--config",
            str(config),
            "--rows",
            str(row_file),
            "--result",
            str(result),
        ]
        if resume:
            cmd += ["--resume", str(resume)]
        if stop:
            cmd += ["--stop", str(stop)]
        if stop_micro:
            cmd += ["--stop-micro", str(stop_micro)]
        env = {
            **os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src"),
            "HF_HUB_OFFLINE": "1",
        }
        completed = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=90)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        return json.loads(result.read_text())

    full = tmp_path / "full"
    split = tmp_path / "split"
    uninterrupted = run("uninterrupted", full)
    stopped = run(
        "stopped",
        split,
        stop=None if mid_window else cut,
        stop_micro=cut * accumulation + 1 if mid_window else None,
    )
    assert stopped["result"]["deliberate_interruption"]
    resumed = run("resumed", split, resume=split / f"mlx-state-{cut}")
    assert resumed["result"]["resumed_from_update"] == cut
    assert resumed["result"]["completed_updates"] == total
    assert resumed["result"]["total_steps"] == 8
    for name in ["adapters.safetensors", "optimizer.safetensors", "random.safetensors"]:
        expected = mx.load(str(full / f"mlx-state-{total}" / name))
        actual = mx.load(str(split / f"mlx-state-{total}" / name))
        assert expected.keys() == actual.keys()
        for key in expected:
            assert mx.array_equal(expected[key], actual[key]).item(), (name, key)
    assert json.loads((full / f"mlx-state-{total}" / "state.json").read_text()) == json.loads(
        (split / f"mlx-state-{total}" / "state.json").read_text()
    )

    saved_host_state = json.loads((full / f"mlx-state-{total}" / "state.json").read_text())
    assert saved_host_state["numpy"][2] != 624  # Advanced beyond the seeded state.
    assert saved_host_state["python"][2] is not None  # Advanced Gaussian cache retained.

    def losses(run):
        return [(x["step"], x["loss"]) for x in run["trace"] if x.get("loss") is not None]

    # Evaluations may repeat the last training value; compare each actual update.
    full_loss = dict(losses(uninterrupted))
    split_loss = dict(losses(stopped) + losses(resumed))
    assert set(full_loss) >= set(range(accumulation, 9, accumulation))
    assert full_loss == split_loss

    # Checkpoint hooks observe the real batches/masks, rates and advanced random draws.
    boundary = cut * accumulation
    assert uninterrupted["batches"] == (
        [b for b in stopped["batches"] if b["microbatch"] <= boundary] + resumed["batches"]
    )
    assert uninterrupted["updates"] == stopped["updates"] + resumed["updates"]
    # Independent scalar oracle, including the intentional zero-rate first warmup update.
    peak, warm = cfg["training"]["lr"], int(total * cfg["training"]["warmup_ratio"])
    expected_rates = []
    for update in range(total):
        if update < warm:
            rate = peak * update / warm
        elif scheduler in {"constant", "constant_with_warmup"}:
            rate = peak
        else:
            fraction = (update - warm) / max(1, total - warm)
            rate = peak * (
                (1 + math.cos(math.pi * fraction)) / 2 if scheduler == "cosine" else 1 - fraction
            )
        expected_rates.append(rate)
    assert [update["lr"] for update in uninterrupted["updates"]] == pytest.approx(
        expected_rates, rel=2e-6, abs=1e-9
    )
    assert all(run["frozen_unchanged"] for run in [uninterrupted, stopped, resumed])
    from mlx.utils import tree_flatten
    from mlx_lm import load

    reloaded, _ = load(str(base), adapter_path=str(split))
    saved_adapter = mx.load(str(split / f"mlx-state-{total}" / "adapters.safetensors"))
    restored_tensors = dict(tree_flatten(reloaded.parameters()))
    assert all(
        mx.array_equal(value, restored_tensors[key]).item() for key, value in saved_adapter.items()
    )
    assert mx.isfinite(reloaded(mx.array([[5, 6, 7]]))).all().item()
    finished = run("finished", split, resume=split / f"mlx-state-{total}")
    assert finished["result"]["resumed_from_update"] == total
    assert finished["result"]["completed_updates"] == total
    assert not finished["updates"] and not finished["batches"]


def test_public_cli_auto_explicit_and_finished_resume(tmp_path):
    import mlx.core as mx

    base = _checkpoint(tmp_path / "base")
    data = tmp_path / "data.jsonl"
    data.write_text(
        "\n".join(
            json.dumps({"text": text})
            for text in [
                "alpha beta gamma",
                "beta gamma alpha beta",
                "gamma beta",
                "alpha gamma beta alpha",
                "beta alpha gamma",
            ]
        )
        + "\n"
    )
    rows = tmp_path / "unused-rows.json"
    rows.write_text("{}")
    worker = Path(__file__).with_name("_mlx_full_state_worker.py")

    def run(name, output, resume=None, stop=None):
        cfg = {
            "base": str(base),
            "backend": "mlx",
            "output": str(output),
            "data": {
                "train": str(data),
                "format": "plaintext",
                "val_split": 0,
                "train_on_responses_only": False,
                "max_length": 64,
            },
            "training": {
                "mlx_full_state": True,
                "quantization": "none",
                "epochs": 3,
                "batch_size": 2,
                "gradient_accumulation_steps": 2,
                "optimizer": "adamw_torch_fused",
                "seed": 7,
                "data_seed": 0,
                "lr": 0.004,
                "scheduler": "cosine",
                "warmup_ratio": 0.4,
                "save_steps": 1,
                "logging_steps": 1,
                "rewind_log": False,
                "lora": {"r": 4, "alpha": 8, "dropout": 0.2},
            },
        }
        config = tmp_path / f"{name}.json"
        config.write_text(json.dumps(cfg))
        result = tmp_path / f"{name}-result.json"
        cmd = [
            sys.executable,
            str(worker),
            "--cli",
            "--config",
            str(config),
            "--rows",
            str(rows),
            "--result",
            str(result),
        ]
        if resume:
            cmd += ["--resume", str(resume)]
        if stop:
            cmd += ["--stop", str(stop)]
        completed = subprocess.run(
            cmd,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=90,
            env={
                **os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src"),
                "HF_HUB_OFFLINE": "1",
            },
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        return json.loads(result.read_text())

    full, split = tmp_path / "full", tmp_path / "split"
    continuous = run("continuous", full)
    interrupted = run("interrupted", split, stop=2)
    previous = (split / "mlx-state-2" / "manifest.json").read_bytes()
    resumed = run("resumed", split, resume="auto")
    total = continuous["result"]["completed_updates"]
    assert total == 4  # nine planned microbatches round down to four complete windows.
    assert resumed["result"]["resumed_from_update"] == 2
    assert resumed["result"]["completed_updates"] == total
    assert "full-state resume" in resumed["console"]
    assert "original schedule/budget preserved" in " ".join(resumed["console"].split())
    assert previous == (split / "mlx-state-2" / "manifest.json").read_bytes()
    assert continuous["batches"] == interrupted["batches"] + resumed["batches"]
    assert continuous["updates"] == interrupted["updates"] + resumed["updates"]
    for name in ["adapters.safetensors", "optimizer.safetensors", "random.safetensors"]:
        expected = mx.load(str(full / f"mlx-state-{total}" / name))
        actual = mx.load(str(split / f"mlx-state-{total}" / name))
        assert expected.keys() == actual.keys()
        assert all(mx.array_equal(value, actual[key]).item() for key, value in expected.items())
    finished = run("finished", split, resume=split / f"mlx-state-{total}")
    assert finished["result"]["resumed_from_update"] == total
    assert not finished["batches"] and not finished["updates"]
    metadata = json.loads((split / "adapter_config.json").read_text())
    assert metadata["resume_mode"] == "full-state"
    assert metadata["completed_updates"] == total


def test_legacy_weights_only_mode_and_reload_remain_explicit(tmp_path, capsys):
    from mlx_lm import load

    from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper
    from tests.test_mlx_dora_native import _config, _rows

    base = _checkpoint(tmp_path / "base")
    cfg = _config(base, tmp_path / "legacy", dora=False)
    cfg.training.rewind_log = False
    fresh = MLXSFTTrainerWrapper(cfg)
    fresh.setup(_rows())
    assert fresh.train()["resume_mode"] == "fresh"
    warm = MLXSFTTrainerWrapper(cfg)
    warm.setup(_rows())
    result = warm.train(resume_from_checkpoint=str(tmp_path / "legacy/adapters.safetensors"))
    assert result["resume_mode"] == "weights-only"
    assert "weights-only warm start" in capsys.readouterr().out
    metadata = json.loads((tmp_path / "legacy/adapter_config.json").read_text())
    assert metadata["resume_mode"] == "weights-only"
    import mlx.core as mx

    model, _ = load(str(base), adapter_path=str(tmp_path / "legacy"))
    assert mx.isfinite(model(mx.array([[5, 6, 7]]))).all().item()
