"""soup ship refuses an all-zero lora_B adapter before any model load (#2410).

No GPU: _resolve_generators is mocked; the adapter is a synthetic safetensors file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from soup_cli.commands import ship as ship_cmd
from soup_cli.utils.ship_verdict import FAILED_ADAPTER_NOOP, FAILED_TASK_WIN
from tests.cli_runner import CliRunner
from tests.conftest import strip_ansi

runner = CliRunner()

Q_A = "base_model.model.layers.0.self_attn.q_proj.lora_A.weight"
Q_B = "base_model.model.layers.0.self_attn.q_proj.lora_B.weight"


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


def _write_adapter(dir_path: Path, weights: dict) -> None:
    pytest.importorskip("safetensors")
    from safetensors.numpy import save_file

    dir_path.mkdir()
    save_file(weights, str(dir_path / "adapter_model.safetensors"))
    (dir_path / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 2, "lora_alpha": 4}), encoding="utf-8"
    )


def _lora(b_value: float) -> dict:
    return {
        Q_A: np.ones((2, 4), dtype=np.float32),
        Q_B: np.full((4, 2), b_value, dtype=np.float32),
    }


def _run(monkeypatch, weights: dict, *extra: str, write=_write_adapter):
    loads = []

    def _fake_resolve(base, tuned, adapter, device, quantization=None):
        loads.append(adapter)
        return (lambda p: "hi", lambda p: "hi")

    monkeypatch.setattr(ship_cmd, "_resolve_generators", _fake_resolve)
    with runner.isolated_filesystem():
        write(Path("lora"), weights)
        Path("task.jsonl").write_text(
            json.dumps({"prompt": "say hi", "expected": "hi", "scoring": "contains"}) + "\n",
            encoding="utf-8",
        )
        res = runner.invoke(
            ship_cmd.app,
            ["--base", "m", "--adapter", "lora", "--task-eval", "task.jsonl",
             "--device", "cpu", "--output", "out.json", *extra],
        )
        out = (
            json.loads(Path("out.json").read_text(encoding="utf-8"))
            if Path("out.json").exists() else None
        )
        evidence_written = Path("ev.json").exists()
    return res, loads, out, evidence_written


def test_all_zero_lora_b_refused_before_any_model_load(monkeypatch):
    res, loads, out, evidence_written = _run(
        monkeypatch, _lora(0.0), "--emit-evidence", "ev.json"
    )
    assert res.exit_code == 2, (res.output, repr(res.exception))
    assert loads == [], (res.output, repr(res.exception))
    assert out["failed_rule"] == FAILED_ADAPTER_NOOP
    assert out["failed_rule"] == "adapter_noop"
    plain = _plain(res.output)
    assert "lora_B" in plain
    assert "no scores were measured; evidence not written" in plain
    assert not evidence_written


def test_trained_adapter_runs_the_live_eval(monkeypatch):
    res, loads, out, _ = _run(monkeypatch, _lora(0.5))
    assert loads == ["lora"], (res.output, repr(res.exception))
    assert res.exit_code == 2, (res.output, repr(res.exception))
    assert out["failed_rule"] == FAILED_TASK_WIN


def test_adapter_without_lora_b_keys_is_left_alone(monkeypatch):
    vera = {"base_model.model.layers.0.self_attn.q_proj.vera_lambda_b": np.zeros(4, np.float32)}
    res, loads, out, _ = _run(monkeypatch, vera)
    assert loads == ["lora"], (res.output, repr(res.exception))
    assert res.exit_code == 2, (res.output, repr(res.exception))
    assert out["failed_rule"] == FAILED_TASK_WIN


def test_non_zero_standalone_tensor_is_left_alone(monkeypatch):
    weights = _lora(0.0)
    weights["base_model.model.score.weight"] = np.ones((2, 4), dtype=np.float32)
    res, loads, out, _ = _run(monkeypatch, weights)
    assert loads == ["lora"], (res.output, repr(res.exception))
    assert res.exit_code == 2, (res.output, repr(res.exception))
    assert out["failed_rule"] == FAILED_TASK_WIN


def test_unreadable_weights_warn_and_continue(monkeypatch):
    def _bin_adapter(dir_path: Path, weights: dict) -> None:
        dir_path.mkdir()
        (dir_path / "adapter_model.bin").write_bytes(b"\x00" * 16)

    res, loads, out, _ = _run(monkeypatch, {}, write=_bin_adapter)
    assert loads == ["lora"], (res.output, repr(res.exception))
    assert res.exit_code == 2, (res.output, repr(res.exception))
    assert out["failed_rule"] == FAILED_TASK_WIN
    assert "could not read the adapter weights" in _plain(res.output)


def test_partly_zero_lora_b_runs_the_live_eval(monkeypatch):
    weights = _lora(0.0)
    weights[Q_A.replace("q_proj", "v_proj")] = np.ones((2, 4), dtype=np.float32)
    weights[Q_B.replace("q_proj", "v_proj")] = np.full((4, 2), 0.5, dtype=np.float32)
    res, loads, out, _ = _run(monkeypatch, weights)
    assert loads == ["lora"], (res.output, repr(res.exception))
    assert res.exit_code == 2, (res.output, repr(res.exception))
    assert out["failed_rule"] == FAILED_TASK_WIN


def test_adapter_that_is_not_a_local_directory_is_skipped_silently(monkeypatch):
    res, loads, out, _ = _run(monkeypatch, {}, write=lambda dir_path, weights: None)
    assert loads == ["lora"], (res.output, repr(res.exception))
    assert res.exit_code == 2, (res.output, repr(res.exception))
    assert out["failed_rule"] == FAILED_TASK_WIN
    assert "could not read the adapter weights" not in _plain(res.output)


@pytest.mark.parametrize("exc_type", [ValueError, TypeError, OSError, RuntimeError, ImportError])
def test_read_failure_warns_with_the_reason_and_continues(monkeypatch, exc_type):
    from soup_cli.utils import adapter_check

    def _raise(path):
        raise exc_type("bad\x00\x9b [tag] header")

    monkeypatch.setattr(adapter_check, "check_adapter", _raise)
    res, loads, out, _ = _run(monkeypatch, _lora(0.0))
    assert loads == ["lora"], (res.output, repr(res.exception))
    assert res.exit_code == 2, (res.output, repr(res.exception))
    assert out["failed_rule"] == FAILED_TASK_WIN
    assert "\x00" not in res.output and "\x9b" not in res.output
    plain = _plain(res.output)
    assert "could not read the adapter weights" in plain
    assert "bad [tag] header" in plain
    assert "\\[tag]" not in plain
