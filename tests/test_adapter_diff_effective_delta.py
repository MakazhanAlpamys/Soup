"""``soup adapters diff`` compares LoRA modules on their effective delta.

The report's "ΔW Frobenius" column must be ``||ΔW_a - ΔW_b||_F`` with
``ΔW = scaling · B @ A`` per module: LoRA factors are not unique, two adapters
of different rank share no factor shape, and the same factors under a different
``lora_alpha`` are a different ``ΔW``.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

pytest.importorskip("safetensors")

_STEM = "base_model.model.model.layers.0.self_attn.q_proj"
_A = f"{_STEM}.lora_A.weight"
_B = f"{_STEM}.lora_B.weight"
_DIM = 6


def _factors(r, value, slot=0):
    a = np.zeros((r, _DIM), dtype=np.float32)
    b = np.zeros((_DIM, r), dtype=np.float32)
    a[slot, slot] = 1.0
    b[slot, slot] = value
    return a, b


def _write(path, tensors, **config):
    from safetensors.numpy import save_file

    path.mkdir(parents=True)
    save_file(tensors, str(path / "adapter_model.safetensors"))
    cfg = {"peft_type": "LORA", "target_modules": ["q_proj"], **config}
    (path / "adapter_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return path.name


def _diff(tmp_path, monkeypatch, left, right):
    from soup_cli.utils.adapter_diff import compute_adapter_diff

    monkeypatch.chdir(tmp_path)
    return compute_adapter_diff(left, right)


def test_lora_alpha_change_is_a_change(tmp_path, monkeypatch):
    a, b = _factors(2, 1.0)
    _write(tmp_path / "x", {_A: a, _B: b}, r=2, lora_alpha=2)  # ΔW = 1 · e0 e0^T
    _write(tmp_path / "y", {_A: a, _B: b}, r=2, lora_alpha=4)  # ΔW = 2 · e0 e0^T
    report = _diff(tmp_path, monkeypatch, "x", "y")
    assert [d.name for d in report.per_layer] == [_STEM]
    row = report.per_layer[0]
    assert row.norm_a == pytest.approx(1.0)
    assert row.norm_b == pytest.approx(2.0)
    assert row.frobenius == pytest.approx(1.0)
    assert row.relative == pytest.approx(0.5)


def test_different_ranks_are_compared(tmp_path, monkeypatch):
    a2, b2 = _factors(2, 1.0)
    a4, b4 = _factors(4, 3.0)
    _write(tmp_path / "x", {_A: a2, _B: b2}, r=2, lora_alpha=2)  # ΔW = e0 e0^T
    _write(tmp_path / "y", {_A: a4, _B: b4}, r=4, lora_alpha=4)  # ΔW = 3 e0 e0^T
    report = _diff(tmp_path, monkeypatch, "x", "y")
    assert report.shared_layers == 1
    assert report.per_layer[0].frobenius == pytest.approx(2.0)
    assert report.top_changed == (_STEM,)
    assert report.only_in_a == () and report.only_in_b == ()


def test_reparameterised_factors_are_no_change(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, _DIM)).astype(np.float32)
    b = rng.standard_normal((_DIM, 4)).astype(np.float32)
    # (2B, A/2) is the same ΔW as (B, A); both scale exactly in float32.
    _write(tmp_path / "x", {_A: a, _B: b}, r=4, lora_alpha=8)
    _write(tmp_path / "y", {_A: a / 2, _B: b * 2}, r=4, lora_alpha=8)
    report = _diff(tmp_path, monkeypatch, "x", "y")
    row = report.per_layer[0]
    assert row.frobenius <= 1e-6 * row.norm_a
    expected = 2.0 * np.linalg.norm(b.astype(np.float64) @ a.astype(np.float64))
    assert row.norm_a == pytest.approx(expected)


def test_rslora_scaling_is_applied(tmp_path, monkeypatch):
    a, b = _factors(4, 1.0)
    _write(tmp_path / "x", {_A: a, _B: b}, r=4, lora_alpha=4)  # 4 / 4 = 1
    _write(tmp_path / "y", {_A: a, _B: b}, r=4, lora_alpha=4, use_rslora=True)  # 4 / 2 = 2
    report = _diff(tmp_path, monkeypatch, "x", "y")
    assert report.per_layer[0].norm_b == pytest.approx(2.0)
    assert report.per_layer[0].frobenius == pytest.approx(1.0)


def test_identical_adapters_diff_to_exact_zero(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    tensors = {
        _A: rng.standard_normal((4, _DIM)).astype(np.float32),
        _B: rng.standard_normal((_DIM, 4)).astype(np.float32),
    }
    _write(tmp_path / "x", tensors, r=4, lora_alpha=8)
    _write(tmp_path / "y", tensors, r=4, lora_alpha=8)
    report = _diff(tmp_path, monkeypatch, "x", "y")
    assert report.per_layer[0].frobenius == 0.0


def test_untrained_adapter_has_zero_effective_rank(tmp_path, monkeypatch):
    rng = np.random.default_rng(2)
    a = rng.standard_normal((4, _DIM)).astype(np.float32)
    zero_b = np.zeros((_DIM, 4), dtype=np.float32)
    _write(tmp_path / "x", {_A: a, _B: zero_b}, r=4, lora_alpha=8)  # ΔW = 0
    _write(tmp_path / "y", {_A: a, _B: zero_b}, r=4, lora_alpha=8)
    report = _diff(tmp_path, monkeypatch, "x", "y")
    assert report.effective_rank_a == 0.0
    assert report.per_layer[0].norm_a == 0.0


def test_other_tensors_and_half_pairs_are_diffed_by_name(tmp_path, monkeypatch):
    a, b = _factors(2, 1.0)
    bias = "base_model.model.model.layers.0.self_attn.q_proj.bias"
    _write(
        tmp_path / "x", {_A: a, _B: b, bias: np.ones(_DIM, dtype=np.float32)},
        r=2, lora_alpha=2,
    )
    # y has only the lora_A half of the module, so there is no ΔW to compare.
    _write(tmp_path / "y", {_A: a, bias: np.zeros(_DIM, dtype=np.float32)}, r=2, lora_alpha=2)
    report = _diff(tmp_path, monkeypatch, "x", "y")
    assert [d.name for d in report.per_layer] == [bias, _A]
    assert report.per_layer[0].frobenius == pytest.approx(np.sqrt(_DIM))
    assert report.per_layer[1].frobenius == 0.0
    assert report.only_in_a == (_B,)


def test_cli_json_reports_the_module(tmp_path):
    from typer.testing import CliRunner

    from soup_cli.cli import app

    a2, b2 = _factors(2, 1.0)
    a4, b4 = _factors(4, 3.0)
    _write(tmp_path / "x", {_A: a2, _B: b2}, r=2, lora_alpha=2)
    _write(tmp_path / "y", {_A: a4, _B: b4}, r=4, lora_alpha=4)
    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        res = CliRunner().invoke(
            app, ["adapters", "diff", "x", "y", "--format", "json", "-o", "d.json"]
        )
    finally:
        os.chdir(old)
    assert res.exit_code == 0, res.output
    payload = json.loads((tmp_path / "d.json").read_text(encoding="utf-8"))
    assert payload["shared_layers"] == 1
    assert payload["top_changed"] == [_STEM]
    assert payload["per_layer"][0]["frobenius"] == pytest.approx(2.0)
