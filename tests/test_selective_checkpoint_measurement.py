"""The measurement selector and closure must fail closed without changing defaults."""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load(name):
    path = Path(__file__).parents[1] / "benchmarks/harness" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_original_pair_runner_harness_remains_the_default(monkeypatch):
    module = _load("stream_ab_pairs")
    captured = []
    monkeypatch.setattr(module, "measure", lambda args: captured.append(args) or 0)
    monkeypatch.setattr(
        sys, "argv", ["runner", "--out", "o", "--tree", "t", "--weights", "w", "--shards", "s"]
    )
    assert module.main() == 0
    assert captured[0].harness == "issue361_nf4_throughput.py"


@pytest.mark.parametrize("selected", ["../../../outside.py", "../../other.py", "../README.md"])
def test_pair_runner_refuses_harness_outside_its_directory(tmp_path, selected):
    module = _load("stream_ab_pairs")
    args = argparse.Namespace(tree=str(tmp_path), out=str(tmp_path), common="", harness=selected)
    with pytest.raises(ValueError, match="inside"):
        module.run_arm(args, 0, "A", "")


def test_pair_runner_selects_requested_harness_without_changing_recipe(tmp_path, monkeypatch):
    module = _load("stream_ab_pairs")
    calls = []

    def execute(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="out", stderr="err")

    monkeypatch.setattr(module.subprocess, "run", execute)
    args = argparse.Namespace(
        tree=str(tmp_path),
        out=str(tmp_path),
        common="--steps 50",
        harness="issue2050_selective_checkpoint.py",
        weights="weights",
        shards="shards",
    )
    assert module.run_arm(args, 1, "B", "--policy qkv")[0] == 0
    command, kwargs = calls[0]
    assert command[1] == os.path.realpath(tmp_path / "benchmarks/harness" / args.harness)
    assert command[-4:] == ["--steps", "50", "--policy", "qkv"]
    assert kwargs["env"]["PYTHONPATH"] == str(tmp_path / "src")


def test_tensor_hash_handles_scalar_loss_and_bfloat16():
    torch = pytest.importorskip("torch")
    module = _load("issue2050_selective_checkpoint")
    for dtype in (torch.float32, torch.bfloat16):
        scalar = torch.tensor(1.0, dtype=dtype)
        raw = scalar.reshape(1).view(torch.uint8).numpy().tobytes()
        assert module.tensor_hash(scalar) == hashlib.sha256(raw).hexdigest()
        assert module.tensor_hash(scalar) != module.tensor_hash(scalar + 1)


def test_bootstrap_requires_all_five_pairs_and_retains_units():
    module = _load("issue2050_audit")
    with pytest.raises(ValueError, match="five"):
        module.bootstrap([2.0] * 4)
    assert module.bootstrap([3.0] * 5) == {
        "median_pct": 3.0,
        "ci95": [3.0, 3.0],
        "ci98_75": [3.0, 3.0],
    }


def test_incomplete_campaign_cannot_be_closed(tmp_path):
    module = _load("issue2050_audit")
    (tmp_path / "status.json").write_text(json.dumps({"phase": "running"}))
    with pytest.raises(AssertionError, match="incomplete"):
        module.audit(tmp_path)
