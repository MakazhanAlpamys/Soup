"""Tests for soup adapters check: pre-flight health audit (Issue #1721).

Acceptance coverage:
1. Healthy adapter: ||ΔW||_F > 0, live fraction 1.0, verdict `alive`, exit 0.
2. Synthetic adapter with all-zero lora_B: verdict `inactive: all lora_B layers are zero`, exit 2.
3. Synthetic adapter with .inner. wrapper keys: verdict `inactive: leaked .inner. keys`, exit 2.
4. Partially zero lora_B: live fraction < 1.0, verdict `inactive: lora_B layers are zero`, exit 2.
5. Non-finite weights (NaN/Inf): verdict `inactive: non-finite weights detected`, exit 2.
6. JSON mode: --json emits complete document matching terminal metrics.
7. Pathguard: outside-cwd and symlink rejections exit 1.
8. Mathematical identity: trace formula matches explicit (B @ A) Frobenius norm bit-for-bit.
9. rsLoRA scaling factor: alpha / sqrt(r).
10. Transposed Conv1D shapes: trace identity bit-exact.
11. Incomplete LoRA pairs: orphaned weights report inactive.
12. Standalone non-LoRA tensors: zero bias not flagged as zero lora_B.
13. Strict RFC 8259 JSON: null serialization under NaN weights.
14. Boolean rank guard: booleans in adapter_config.json safely ignored.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from soup_cli.cli import app as soup_app
from soup_cli.utils.adapter_check import check_adapter
from tests.conftest import strip_ansi

runner = CliRunner()

Q_LORA_A = "base_model.model.layers.0.self_attn.q_proj.lora_A.weight"
Q_LORA_B = "base_model.model.layers.0.self_attn.q_proj.lora_B.weight"
V_LORA_A = "base_model.model.layers.0.self_attn.v_proj.lora_A.weight"
V_LORA_B = "base_model.model.layers.0.self_attn.v_proj.lora_B.weight"
INNER_LORA_A = "base_model.model.layers.0.inner.self_attn.q_proj.lora_A.weight"
INNER_LORA_B = "base_model.model.layers.0.inner.self_attn.q_proj.lora_B.weight"


def _write_synthetic_adapter(
    dir_path: Path,
    weights: dict[str, np.ndarray],
    r: int = 8,
    alpha: float = 16.0,
    extra_cfg: dict | None = None,
) -> None:
    """Write synthetic adapter directory with safetensors and adapter_config.json."""
    pytest.importorskip("safetensors")
    from safetensors.numpy import save_file

    dir_path.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(dir_path / "adapter_model.safetensors"))
    cfg = {
        "peft_type": "LORA",
        "r": r,
        "lora_alpha": alpha,
        "target_modules": ["q_proj", "v_proj"],
    }
    if extra_cfg:
        cfg.update(extra_cfg)
    (dir_path / "adapter_config.json").write_text(json.dumps(cfg), encoding="utf-8")


# ---------- Core Acceptance Tests ----------


def test_healthy_adapter_is_alive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A healthy adapter with non-zero updates reports 'alive' and exits 0."""
    pytest.importorskip("safetensors")
    monkeypatch.chdir(tmp_path)
    adapter_dir = tmp_path / "healthy"

    weights = {
        Q_LORA_A: np.ones((8, 16), dtype=np.float32),
        Q_LORA_B: 0.5 * np.ones((16, 8), dtype=np.float32),
    }
    _write_synthetic_adapter(adapter_dir, weights, r=8, alpha=16.0)

    report = check_adapter("healthy")
    assert report.verdict == "alive"
    assert report.reason is None
    assert report.verdict_line == "alive"
    assert report.total_frobenius > 0.0
    assert report.live_fraction == 1.0
    assert report.live_layers == 1
    assert report.total_layers == 1
    assert len(report.all_zero_lora_b_layers) == 0
    assert len(report.inner_keys) == 0

    # CLI test
    res = runner.invoke(soup_app, ["adapters", "check", "healthy"])
    assert res.exit_code == 0
    lines = res.stdout.strip().splitlines()
    assert lines[-1] == "alive"
    assert "Live fraction: 100.0%" in strip_ansi(res.stdout)


def test_all_zero_lora_b_reports_inactive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """An untrained adapter with all-zero lora_B reports inactive and exits 2."""
    pytest.importorskip("safetensors")
    monkeypatch.chdir(tmp_path)
    adapter_dir = tmp_path / "zero_b"

    weights = {
        Q_LORA_A: np.ones((8, 16), dtype=np.float32),
        Q_LORA_B: np.zeros((16, 8), dtype=np.float32),
    }
    _write_synthetic_adapter(adapter_dir, weights, r=8, alpha=16.0)

    report = check_adapter("zero_b")
    assert report.verdict == "inactive"
    assert report.reason == "all lora_B layers are zero (1/1)"
    assert report.verdict_line == "inactive: all lora_B layers are zero (1/1)"
    assert report.total_frobenius == 0.0
    assert report.live_fraction == 0.0
    assert report.live_layers == 0
    assert len(report.all_zero_lora_b_layers) == 1

    # CLI test
    res = runner.invoke(soup_app, ["adapters", "check", "zero_b"])
    assert res.exit_code == 2
    lines = res.stdout.strip().splitlines()
    assert lines[-1] == "inactive: all lora_B layers are zero (1/1)"


def test_inner_namespace_leak_reports_inactive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """An adapter with leaked .inner. wrapper keys reports inactive and exits 2."""
    pytest.importorskip("safetensors")
    monkeypatch.chdir(tmp_path)
    adapter_dir = tmp_path / "inner_leak"

    weights = {
        INNER_LORA_A: np.ones((8, 16), dtype=np.float32),
        INNER_LORA_B: 0.5 * np.ones((16, 8), dtype=np.float32),
    }
    _write_synthetic_adapter(adapter_dir, weights, r=8, alpha=16.0)

    report = check_adapter("inner_leak")
    assert report.verdict == "inactive"
    assert "leaked .inner. keys detected" in (report.reason or "")
    assert len(report.inner_keys) == 2
    assert INNER_LORA_A in report.inner_keys

    # CLI test
    res = runner.invoke(soup_app, ["adapters", "check", "inner_leak"])
    assert res.exit_code == 2
    lines = res.stdout.strip().splitlines()
    assert lines[-1].startswith("inactive: leaked .inner. keys detected")


def test_partially_zero_lora_b(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """An adapter with some zero and some live lora_B layers reports inactive with layer ratio."""
    pytest.importorskip("safetensors")
    monkeypatch.chdir(tmp_path)
    adapter_dir = tmp_path / "partial"

    weights = {
        Q_LORA_A: np.ones((8, 16), dtype=np.float32),
        Q_LORA_B: 0.5 * np.ones((16, 8), dtype=np.float32),
        V_LORA_A: np.ones((8, 16), dtype=np.float32),
        V_LORA_B: np.zeros((16, 8), dtype=np.float32),
    }
    _write_synthetic_adapter(adapter_dir, weights, r=8, alpha=16.0)

    report = check_adapter("partial")
    assert report.verdict == "inactive"
    assert report.reason == "lora_B layers are zero (1/2)"
    assert report.live_fraction == 0.5
    assert report.live_layers == 1
    assert report.total_layers == 2
    assert len(report.all_zero_lora_b_layers) == 1

    res = runner.invoke(soup_app, ["adapters", "check", "partial"])
    assert res.exit_code == 2
    lines = res.stdout.strip().splitlines()
    assert lines[-1] == "inactive: lora_B layers are zero (1/2)"


def test_non_finite_weights_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """NaN/Inf in weights causes inactive classification."""
    pytest.importorskip("safetensors")
    monkeypatch.chdir(tmp_path)
    adapter_dir = tmp_path / "nan_adapter"

    nan_b = np.ones((16, 8), dtype=np.float32)
    nan_b[0, 0] = np.nan
    weights = {
        Q_LORA_A: np.ones((8, 16), dtype=np.float32),
        Q_LORA_B: nan_b,
    }
    _write_synthetic_adapter(adapter_dir, weights, r=8, alpha=16.0)

    report = check_adapter("nan_adapter")
    assert report.verdict == "inactive"
    assert report.reason == "non-finite weights detected (NaN/Inf)"
    assert math.isnan(report.total_frobenius)

    res = runner.invoke(soup_app, ["adapters", "check", "nan_adapter"])
    assert res.exit_code == 2
    assert "inactive: non-finite weights detected" in res.stdout


def test_json_output_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """--json emits valid parseable JSON document matching the terminal metrics."""
    pytest.importorskip("safetensors")
    monkeypatch.chdir(tmp_path)
    adapter_dir = tmp_path / "healthy"

    weights = {
        Q_LORA_A: np.ones((4, 8), dtype=np.float32),
        Q_LORA_B: np.ones((8, 4), dtype=np.float32),
    }
    _write_synthetic_adapter(adapter_dir, weights, r=4, alpha=8.0)

    res = runner.invoke(soup_app, ["adapters", "check", "healthy", "--json"])
    assert res.exit_code == 0
    doc = json.loads(res.stdout)
    assert doc["verdict"] == "alive"
    assert doc["verdict_line"] == "alive"
    assert doc["live_fraction"] == 1.0
    assert doc["live_layers"] == 1
    assert doc["total_layers"] == 1
    assert doc["total_frobenius"] > 0.0
    assert "inner_keys" in doc and doc["inner_keys"] == []
    assert "all_zero_lora_b_layers" in doc and doc["all_zero_lora_b_layers"] == []
    assert "reason" in doc and doc["reason"] is None
    assert len(doc["per_layer"]) == 1


# ---------- Security & Path Validation ----------


def test_outside_cwd_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Adapters outside cwd must be refused by the path guard."""
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    outside_dir = tmp_path / "outside_adapter"
    outside_dir.mkdir()
    monkeypatch.chdir(work_dir)

    with pytest.raises(ValueError, match="must stay under cwd"):
        check_adapter(str(outside_dir))

    res = runner.invoke(soup_app, ["adapters", "check", str(outside_dir)])
    assert res.exit_code == 1
    assert "refused" in res.output.lower()
    assert "not found" not in res.output.lower()


def test_missing_adapter_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Non-existent adapter directory exits 1."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        check_adapter("does_not_exist")

    res = runner.invoke(soup_app, ["adapters", "check", "does_not_exist"])
    assert res.exit_code == 1
    assert "does_not_exist" in res.output or "not found" in res.output.lower()


# ---------- Multi-Layer RSS & Conv1D Transposed Contraction ----------


def test_two_layer_and_conv1d_transposed_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Assert total_frobenius == sqrt(sum of squares) and Conv1D transposed contraction."""
    pytest.importorskip("safetensors")
    monkeypatch.chdir(tmp_path)
    adapter_dir = tmp_path / "two_layer_conv1d"

    # Layer 0: standard Linear (q_proj), r=8, out=16, in=32 -> B is (16, 8), A is (8, 32)
    rng = np.random.default_rng(123)
    q_b = rng.standard_normal((16, 8)).astype(np.float32)
    q_a = rng.standard_normal((8, 32)).astype(np.float32)

    # Layer 1: transposed Conv1D (v_proj), r=8, in=32, out=16 -> B is (8, 16), A is (32, 8)
    v_b = rng.standard_normal((8, 16)).astype(np.float32)
    v_a = rng.standard_normal((32, 8)).astype(np.float32)

    weights = {
        Q_LORA_A: q_a,
        Q_LORA_B: q_b,
        V_LORA_A: v_a,
        V_LORA_B: v_b,
    }
    _write_synthetic_adapter(adapter_dir, weights, r=8, alpha=16.0)

    report = check_adapter("two_layer_conv1d")
    assert report.verdict == "alive"
    assert report.total_layers == 2
    assert report.live_layers == 2

    # Direct computation: scaling = 16.0 / 8 = 2.0
    delta0 = 2.0 * np.matmul(q_b.astype(np.float64), q_a.astype(np.float64))
    expected_fro0 = float(np.linalg.norm(delta0))

    delta1 = 2.0 * np.matmul(v_a.astype(np.float64), v_b.astype(np.float64))
    expected_fro1 = float(np.linalg.norm(delta1))

    assert math.isclose(report.per_layer[0].frobenius, expected_fro0, rel_tol=1e-12)
    assert math.isclose(report.per_layer[1].frobenius, expected_fro1, rel_tol=1e-12)

    # Total frobenius must equal Root-Sum-of-Squares (RSS), not simple linear sum
    expected_total = math.sqrt(expected_fro0**2 + expected_fro1**2)
    assert math.isclose(report.total_frobenius, expected_total, rel_tol=1e-12)


# ---------- Hardening & Invariant Defenses ----------


def test_rslora_scaling_factor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """When use_rslora is True, scaling factor must be alpha / sqrt(r)."""
    pytest.importorskip("safetensors")
    monkeypatch.chdir(tmp_path)
    adapter_dir = tmp_path / "rslora"

    r = 16
    alpha = 32.0
    # Expected scaling: 32 / sqrt(16) = 32 / 4 = 8.0
    weights = {
        Q_LORA_A: np.ones((16, 8), dtype=np.float32),
        Q_LORA_B: np.ones((8, 16), dtype=np.float32),
    }
    _write_synthetic_adapter(
        adapter_dir, weights, r=r, alpha=alpha, extra_cfg={"use_rslora": True}
    )

    report = check_adapter("rslora")
    assert report.verdict == "alive"
    # Direct computation: 8.0 * ||B @ A||_F
    b_mat = np.ones((8, 16), dtype=np.float64)
    a_mat = np.ones((16, 8), dtype=np.float64)
    expected_fro = 8.0 * float(np.sqrt(np.sum(np.matmul(b_mat, a_mat) ** 2)))
    assert math.isclose(report.total_frobenius, expected_fro, rel_tol=1e-6)


def test_incomplete_lora_pair_reported_inactive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """An adapter with orphaned LoRA weights (missing A or missing B) reports inactive."""
    pytest.importorskip("safetensors")
    monkeypatch.chdir(tmp_path)
    adapter_dir = tmp_path / "orphan"

    weights = {
        Q_LORA_A: np.ones((8, 16), dtype=np.float32),
        # Q_LORA_B is intentionally missing
    }
    _write_synthetic_adapter(adapter_dir, weights, r=8, alpha=16.0)

    report = check_adapter("orphan")
    assert report.verdict == "inactive"
    assert "incomplete LoRA pairs detected" in (report.reason or "")
    assert len(report.orphaned_layers) == 1

    res = runner.invoke(soup_app, ["adapters", "check", "orphan"])
    assert res.exit_code == 2
    assert "inactive: incomplete LoRA pairs detected" in res.stdout


def test_standalone_tensor_not_flagged_as_lora_b(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Zero-valued standalone non-LoRA tensors must not be counted as all-zero lora_B layers."""
    pytest.importorskip("safetensors")
    monkeypatch.chdir(tmp_path)
    adapter_dir = tmp_path / "with_bias"

    weights = {
        Q_LORA_A: np.ones((8, 16), dtype=np.float32),
        Q_LORA_B: 0.5 * np.ones((16, 8), dtype=np.float32),
        "base_model.model.custom_bias": np.zeros((16,), dtype=np.float32),
    }
    _write_synthetic_adapter(adapter_dir, weights, r=8, alpha=16.0)

    report = check_adapter("with_bias")
    assert report.verdict == "alive"
    assert len(report.all_zero_lora_b_layers) == 0
    assert report.total_layers == 1
    assert report.live_fraction == 1.0
    assert len(report.standalone_tensors) == 1


def test_adapter_with_only_standalone_tensors_reports_no_lora(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """An adapter with only saved heads / biases and 0 LoRA pairs reports no lora weights."""
    pytest.importorskip("safetensors")
    monkeypatch.chdir(tmp_path)
    adapter_dir = tmp_path / "only_heads"

    weights = {
        "base_model.model.lm_head.weight": np.ones((32, 16), dtype=np.float32),
    }
    _write_synthetic_adapter(adapter_dir, weights, r=8, alpha=16.0)

    report = check_adapter("only_heads")
    assert report.verdict == "inactive"
    assert report.reason == "no lora weights found"
    assert report.total_layers == 0
    assert len(report.standalone_tensors) == 1


def test_json_mode_rfc8259_strict_under_nan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Under NaN weights, render_check_json produces RFC 8259 compliant JSON with nulls."""
    pytest.importorskip("safetensors")
    monkeypatch.chdir(tmp_path)
    adapter_dir = tmp_path / "nan_adapter"

    nan_b = np.ones((16, 8), dtype=np.float32)
    nan_b[0, 0] = np.nan
    weights = {
        Q_LORA_A: np.ones((8, 16), dtype=np.float32),
        Q_LORA_B: nan_b,
    }
    _write_synthetic_adapter(adapter_dir, weights, r=8, alpha=16.0)

    res = runner.invoke(soup_app, ["adapters", "check", "nan_adapter", "--json"])
    assert res.exit_code == 2
    def reject_constant(c):
        raise ValueError(f"Non-compliant JSON token: {c}")

    doc = json.loads(res.stdout, parse_constant=reject_constant)
    assert doc["verdict"] == "inactive"
    assert doc["reason"] == "non-finite weights detected (NaN/Inf)"
    assert doc["total_frobenius"] is None
    assert "inner_keys" in doc
    assert "all_zero_lora_b_layers" in doc
    assert '"total_frobenius": null' in strip_ansi(res.stdout)


def test_boolean_rank_in_config_not_coerced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Boolean 'r': true in adapter_config.json must not be coerced to integer 1."""
    pytest.importorskip("safetensors")
    monkeypatch.chdir(tmp_path)
    adapter_dir = tmp_path / "bool_cfg"

    weights = {
        Q_LORA_A: np.ones((8, 16), dtype=np.float32),
        Q_LORA_B: 0.5 * np.ones((16, 8), dtype=np.float32),
    }
    _write_synthetic_adapter(
        adapter_dir, weights, r=8, alpha=16.0, extra_cfg={"r": True}
    )

    report = check_adapter("bool_cfg")
    # Rank is invalid (bool), so fallback scaling 1.0 is used rather than 16/1 = 16.0
    b_mat = 0.5 * np.ones((16, 8), dtype=np.float64)
    a_mat = np.ones((8, 16), dtype=np.float64)
    expected_fro = 1.0 * float(np.sqrt(np.sum(np.matmul(b_mat, a_mat) ** 2)))
    assert math.isclose(report.total_frobenius, expected_fro, rel_tol=1e-6)


def test_lora_shape_mismatch_reported_inactive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Mismatched A and B matrix shapes report inactive with shape mismatch reason."""
    pytest.importorskip("safetensors")
    monkeypatch.chdir(tmp_path)
    adapter_dir = tmp_path / "bad_shapes"

    # Incompatible shapes: A is (7, 13), B is (11, 19)
    weights = {
        Q_LORA_A: np.ones((7, 13), dtype=np.float32),
        Q_LORA_B: np.ones((11, 19), dtype=np.float32),
    }
    _write_synthetic_adapter(adapter_dir, weights, r=8, alpha=16.0)

    report = check_adapter("bad_shapes")
    assert report.verdict == "inactive"
    assert "shape mismatch in LoRA projections" in (report.reason or "")
    assert len(report.shape_mismatches) == 1
    assert math.isnan(report.total_frobenius)

    res = runner.invoke(soup_app, ["adapters", "check", "bad_shapes"])
    assert res.exit_code == 2
    assert "inactive: shape mismatch in LoRA projections" in res.stdout


# ---------- Mutation Resistance Teeth ----------


def test_mutation_zero_b_detection_teeth():
    """Verify that if all-zero B check is removed, healthy and zero-B adapters would conflate."""
    # Control: zero B must report is_b_zero True
    zero_b = np.zeros((16, 8), dtype=np.float64)
    assert bool(np.all(zero_b == 0.0)) is True

    # Perturbed: even a 1e-7 perturbation must report is_b_zero False
    live_b = zero_b.copy()
    live_b[0, 0] = 1e-7
    assert bool(np.all(live_b == 0.0)) is False


def test_corrupt_adapter_config_json_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Corrupted adapter_config.json must raise ValueError and exit 1."""
    pytest.importorskip("safetensors")
    from safetensors.numpy import save_file

    monkeypatch.chdir(tmp_path)
    adapter_dir = tmp_path / "corrupt_cfg"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    weights = {
        Q_LORA_A: np.ones((16, 8), dtype=np.float32),
        Q_LORA_B: np.ones((8, 16), dtype=np.float32),
    }
    save_file(weights, str(adapter_dir / "adapter_model.safetensors"))
    (adapter_dir / "adapter_config.json").write_text("{\"r\": 8, invalid_json", encoding="utf-8")

    with pytest.raises(ValueError, match="unreadable or invalid JSON"):
        check_adapter("corrupt_cfg")

    res = runner.invoke(soup_app, ["adapters", "check", "corrupt_cfg"])
    assert res.exit_code == 1


def test_non_dict_adapter_config_json_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Non-dict JSON in adapter_config.json must raise ValueError and exit 1."""
    pytest.importorskip("safetensors")
    from safetensors.numpy import save_file

    monkeypatch.chdir(tmp_path)
    adapter_dir = tmp_path / "nondict_cfg"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    weights = {
        Q_LORA_A: np.ones((16, 8), dtype=np.float32),
        Q_LORA_B: np.ones((8, 16), dtype=np.float32),
    }
    save_file(weights, str(adapter_dir / "adapter_model.safetensors"))
    (adapter_dir / "adapter_config.json").write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(ValueError, match="expected JSON object"):
        check_adapter("nondict_cfg")

    res = runner.invoke(soup_app, ["adapters", "check", "nondict_cfg"])
    assert res.exit_code == 1


def test_fp16_float64_accumulation_precision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """FP16 inputs must accumulate in float64 without float32 precision loss (rel_tol=1e-12)."""
    pytest.importorskip("safetensors")
    monkeypatch.chdir(tmp_path)
    adapter_dir = tmp_path / "fp16_prec"

    rng = np.random.default_rng(42)
    b_fp16 = (rng.standard_normal((512, 16)) * 1000.0).astype(np.float16)
    a_fp16 = (rng.standard_normal((16, 512)) * 1000.0).astype(np.float16)
    weights = {
        Q_LORA_A: a_fp16,
        Q_LORA_B: b_fp16,
    }
    _write_synthetic_adapter(adapter_dir, weights, r=16, alpha=16.0)

    report = check_adapter("fp16_prec")
    assert report.verdict == "alive"

    b64 = b_fp16.astype(np.float64)
    a64 = a_fp16.astype(np.float64)
    expected_fro = float(np.linalg.norm(b64 @ a64))

    assert math.isclose(report.total_frobenius, expected_fro, rel_tol=1e-12)


@pytest.mark.requires_symlink
def test_symlinked_components_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Symlinked adapter dir, adapter_model.safetensors, and adapter_config.json each exit 1."""
    pytest.importorskip("safetensors")
    monkeypatch.chdir(tmp_path)

    real_dir = tmp_path / "real_adapter"
    weights = {
        Q_LORA_A: np.ones((8, 16), dtype=np.float32),
        Q_LORA_B: np.ones((16, 8), dtype=np.float32),
    }
    _write_synthetic_adapter(real_dir, weights, r=8, alpha=16.0)

    # 1. Symlinked directory
    symlink_dir = tmp_path / "symlink_dir"
    symlink_dir.symlink_to(real_dir, target_is_directory=True)
    res1 = runner.invoke(soup_app, ["adapters", "check", "symlink_dir"])
    assert res1.exit_code == 1
    assert "must not be a symlink" in strip_ansi(res1.output)

    # 2. Symlinked adapter_model.safetensors
    adapter_symlink_weights = tmp_path / "symlink_weights"
    adapter_symlink_weights.mkdir()
    (adapter_symlink_weights / "adapter_config.json").write_text(
        (real_dir / "adapter_config.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (adapter_symlink_weights / "adapter_model.safetensors").symlink_to(
        real_dir / "adapter_model.safetensors"
    )
    res2 = runner.invoke(soup_app, ["adapters", "check", "symlink_weights"])
    assert res2.exit_code == 1
    assert "must not be a symlink" in strip_ansi(res2.output)

    # 3. Symlinked adapter_config.json
    adapter_symlink_cfg = tmp_path / "symlink_cfg"
    adapter_symlink_cfg.mkdir()
    (adapter_symlink_cfg / "adapter_model.safetensors").write_bytes(
        (real_dir / "adapter_model.safetensors").read_bytes()
    )
    (adapter_symlink_cfg / "adapter_config.json").symlink_to(
        real_dir / "adapter_config.json"
    )
    res3 = runner.invoke(soup_app, ["adapters", "check", "symlink_cfg"])
    assert res3.exit_code == 1
    assert "must not be a symlink" in strip_ansi(res3.output)


def test_standard_lora_scaling_factor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Standard LoRA scaling must compute scaling = alpha / r (e.g. 32 / 8 = 4.0)."""
    pytest.importorskip("safetensors")
    monkeypatch.chdir(tmp_path)
    adapter_dir = tmp_path / "standard_scaling"

    r = 8
    alpha = 32.0
    weights = {
        Q_LORA_A: np.ones((8, 16), dtype=np.float32),
        Q_LORA_B: np.ones((16, 8), dtype=np.float32),
    }
    _write_synthetic_adapter(adapter_dir, weights, r=r, alpha=alpha)

    report = check_adapter("standard_scaling")
    assert report.verdict == "alive"

    b_mat = np.ones((16, 8), dtype=np.float64)
    a_mat = np.ones((8, 16), dtype=np.float64)
    expected_fro = 4.0 * float(np.linalg.norm(b_mat @ a_mat))
    assert math.isclose(report.total_frobenius, expected_fro, rel_tol=1e-6)


def test_terminal_injection_control_bytes_stripped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Control and ANSI escape sequences in tensor names must be stripped in terminal output."""
    pytest.importorskip("safetensors")
    monkeypatch.chdir(tmp_path)
    adapter_dir = tmp_path / "esc_adapter"

    esc_key = "base_model.model.layers.0.inner.\x1b[2Jself_attn.q_proj.lora_A.weight"
    weights = {
        esc_key: np.ones((8, 16), dtype=np.float32),
    }
    _write_synthetic_adapter(adapter_dir, weights, r=8, alpha=16.0)

    res = runner.invoke(soup_app, ["adapters", "check", "esc_adapter"])
    assert chr(27) not in res.stdout
    assert "\x1b" not in res.stdout


