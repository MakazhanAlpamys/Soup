"""Tests for v0.60.0 Part A — `soup adapters scan` (weight-space backdoor detector).

Coverage:
- Spectral feature kernels (rank-1 dominance, energy concentration)
- ``ScanFinding`` / ``ScanReport`` frozen dataclasses
- ``scan_adapter_weights`` pure-function happy + flagged paths
- ``scan_adapter`` containment + load
- CLI smoke (``soup adapters scan``)
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from soup_cli.cli import app

# ---------- Spectral kernels ----------


class TestSpectralKernels:
    def test_imports(self):
        from soup_cli.utils.adapter_scan import (
            ScanFinding,
            ScanReport,
            compute_spectral_features,
            scan_adapter_weights,
        )
        assert callable(compute_spectral_features)
        assert callable(scan_adapter_weights)
        assert dataclasses.is_dataclass(ScanFinding)
        assert dataclasses.is_dataclass(ScanReport)

    def test_compute_spectral_features_uniform_matrix_no_dominance(self):
        from soup_cli.utils.adapter_scan import compute_spectral_features

        rng = np.random.default_rng(seed=0)
        # Roughly isotropic matrix — top SV ratio should be small.
        m = rng.standard_normal((32, 32))
        feats = compute_spectral_features(m)
        assert feats["top_sv_ratio"] < 5.0
        assert 0.0 <= feats["energy_top1"] <= 1.0
        assert 0.0 <= feats["effective_rank"] <= 32.0

    def test_compute_spectral_features_rank1_dominance(self):
        from soup_cli.utils.adapter_scan import compute_spectral_features

        # Pure rank-1 matrix — top SV captures all energy.
        u = np.ones((32, 1))
        v = np.ones((1, 32))
        m = u @ v
        feats = compute_spectral_features(m)
        assert feats["top_sv_ratio"] > 100.0
        assert feats["energy_top1"] > 0.99
        assert feats["effective_rank"] < 1.5

    def test_compute_spectral_features_rejects_non_2d(self):
        from soup_cli.utils.adapter_scan import compute_spectral_features

        with pytest.raises(TypeError):
            compute_spectral_features("not-a-matrix")

    def test_compute_spectral_features_empty_returns_zeros(self):
        from soup_cli.utils.adapter_scan import compute_spectral_features

        m = np.zeros((4, 4))
        feats = compute_spectral_features(m)
        assert feats["top_sv_ratio"] == 0.0
        assert feats["energy_top1"] == 0.0


# ---------- Pure-function scanner ----------


class TestScanAdapterWeights:
    def test_clean_weights_pass(self):
        from soup_cli.utils.adapter_scan import scan_adapter_weights

        rng = np.random.default_rng(seed=42)
        weights = {
            f"layer_{i}.lora_A.weight": rng.standard_normal((16, 64)) * 0.01
            for i in range(4)
        }
        weights.update({
            f"layer_{i}.lora_B.weight": rng.standard_normal((64, 16)) * 0.01
            for i in range(4)
        })
        report = scan_adapter_weights(weights, adapter_name="clean")
        assert report.overall == "OK"
        # No FAIL findings.
        assert all(f.severity != "FAIL" for f in report.findings)

    def test_rank1_perturbation_flagged(self):
        from soup_cli.utils.adapter_scan import scan_adapter_weights

        rng = np.random.default_rng(seed=0)
        # Start clean, then inject a high-magnitude rank-1 outer product on one layer.
        weights = {
            f"layer_{i}.lora_A.weight": rng.standard_normal((16, 64)) * 0.01
            for i in range(4)
        }
        u = np.ones((16, 1)) * 5.0
        v = np.ones((1, 64)) * 5.0
        weights["layer_evil.lora_A.weight"] = u @ v
        report = scan_adapter_weights(weights, adapter_name="evil")
        # Should flag rank-1 dominance OR frobenius outlier.
        kinds = {f.kind for f in report.findings}
        assert "rank1_dominance" in kinds or "frobenius_outlier" in kinds
        assert report.overall in ("WARN", "FAIL")

    def test_nan_in_weights_flagged_as_fail(self):
        from soup_cli.utils.adapter_scan import scan_adapter_weights

        m = np.zeros((4, 4))
        m[0, 0] = float("nan")
        weights = {"nan_layer.lora_A.weight": m}
        report = scan_adapter_weights(weights, adapter_name="nan")
        assert report.overall == "FAIL"
        assert any(f.kind == "nan_inf" for f in report.findings)

    def test_inf_in_weights_flagged_as_fail(self):
        from soup_cli.utils.adapter_scan import scan_adapter_weights

        m = np.zeros((4, 4))
        m[0, 0] = float("inf")
        weights = {"inf_layer.lora_A.weight": m}
        report = scan_adapter_weights(weights, adapter_name="inf")
        assert report.overall == "FAIL"

    def test_scan_report_frozen(self):
        from soup_cli.utils.adapter_scan import scan_adapter_weights

        weights = {"a.weight": np.zeros((2, 2))}
        report = scan_adapter_weights(weights, adapter_name="x")
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.overall = "evil"  # type: ignore[misc]

    def test_scan_finding_frozen(self):
        from soup_cli.utils.adapter_scan import ScanFinding

        finding = ScanFinding(
            layer="L", kind="rank1_dominance", severity="WARN",
            value=42.0, threshold=10.0, message="ok",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            finding.severity = "FAIL"  # type: ignore[misc]

    def test_rejects_non_mapping(self):
        from soup_cli.utils.adapter_scan import scan_adapter_weights

        with pytest.raises(TypeError):
            scan_adapter_weights("not-a-dict", adapter_name="x")

    def test_rejects_null_byte_name(self):
        from soup_cli.utils.adapter_scan import scan_adapter_weights

        with pytest.raises(ValueError):
            scan_adapter_weights({}, adapter_name="bad\x00name")

    def test_empty_weights_returns_ok(self):
        from soup_cli.utils.adapter_scan import scan_adapter_weights

        report = scan_adapter_weights({}, adapter_name="empty")
        assert report.overall == "OK"

    def test_kind_in_allowlist(self):
        from soup_cli.utils.adapter_scan import _VALID_KINDS

        assert "rank1_dominance" in _VALID_KINDS
        assert "frobenius_outlier" in _VALID_KINDS
        assert "nan_inf" in _VALID_KINDS
        assert "energy_concentration" in _VALID_KINDS

    def test_severity_in_allowlist(self):
        from soup_cli.utils.adapter_scan import ScanFinding

        with pytest.raises(ValueError):
            ScanFinding(
                layer="L", kind="rank1_dominance", severity="UNKNOWN_SEV",
                value=0.0, threshold=0.0, message="",
            )

    def test_unknown_kind_rejected(self):
        from soup_cli.utils.adapter_scan import ScanFinding

        with pytest.raises(ValueError):
            ScanFinding(
                layer="L", kind="bogus_kind", severity="WARN",
                value=0.0, threshold=0.0, message="",
            )

    def test_scan_finding_rejects_bool_value(self):
        """bool-as-int rejection on numeric fields (project policy)."""
        from soup_cli.utils.adapter_scan import ScanFinding

        with pytest.raises(ValueError):
            ScanFinding(
                layer="L", kind="rank1_dominance", severity="WARN",
                value=True, threshold=0.0, message="",  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError):
            ScanFinding(
                layer="L", kind="rank1_dominance", severity="WARN",
                value=0.0, threshold=False, message="",  # type: ignore[arg-type]
            )

    def test_scan_finding_rejects_non_finite_value(self):
        from soup_cli.utils.adapter_scan import ScanFinding

        with pytest.raises(ValueError):
            ScanFinding(
                layer="L", kind="rank1_dominance", severity="WARN",
                value=float("nan"), threshold=0.0, message="",
            )
        with pytest.raises(ValueError):
            ScanFinding(
                layer="L", kind="rank1_dominance", severity="WARN",
                value=float("inf"), threshold=0.0, message="",
            )


# ---------- File loader ----------


def _make_safetensors_adapter(tmpdir: Path, weights: dict) -> Path:
    """Create a minimal adapter dir with adapter_model.safetensors + config."""
    pytest.importorskip("safetensors")
    from safetensors.numpy import save_file

    target = tmpdir / "adapter"
    target.mkdir()
    save_file(weights, str(target / "adapter_model.safetensors"))
    (target / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 8}), encoding="utf-8"
    )
    return target


class TestScanAdapterFromDisk:
    def test_scan_adapter_clean_safetensors(self, tmp_path, monkeypatch):
        pytest.importorskip("safetensors")
        monkeypatch.chdir(tmp_path)
        rng = np.random.default_rng(seed=0)
        weights = {
            "lora_A.weight": rng.standard_normal((16, 64)).astype("float32") * 0.01,
        }
        adapter = _make_safetensors_adapter(tmp_path, weights)
        from soup_cli.utils.adapter_scan import scan_adapter

        report = scan_adapter(str(adapter))
        assert report.overall in ("OK", "WARN")

    def test_scan_adapter_outside_cwd_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        outside = tmp_path.parent / "elsewhere_adapter"
        from soup_cli.utils.adapter_scan import scan_adapter

        with pytest.raises(ValueError):
            scan_adapter(str(outside))

    def test_scan_adapter_missing_safetensors(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        empty = tmp_path / "empty_adapter"
        empty.mkdir()
        from soup_cli.utils.adapter_scan import scan_adapter

        with pytest.raises((FileNotFoundError, RuntimeError)):
            scan_adapter(str(empty))

    @pytest.mark.skipif(os.name == "nt", reason="POSIX symlink semantics")
    def test_scan_adapter_symlink_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        real = tmp_path / "real"
        real.mkdir()
        (real / "adapter_model.safetensors").write_bytes(b"x")
        link = tmp_path / "linked"
        os.symlink(str(real), str(link))
        from soup_cli.utils.adapter_scan import scan_adapter

        with pytest.raises(ValueError):
            scan_adapter(str(link))


# ---------- CLI ----------


class TestScanCli:
    def test_scan_help(self):
        runner = CliRunner()
        result = runner.invoke(app, ["adapters", "scan", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "scan" in result.output.lower()

    def test_scan_clean(self, tmp_path, monkeypatch):
        pytest.importorskip("safetensors")
        monkeypatch.chdir(tmp_path)
        rng = np.random.default_rng(seed=0)
        weights = {
            "lora_A.weight": rng.standard_normal((16, 64)).astype("float32") * 0.01,
        }
        adapter = _make_safetensors_adapter(tmp_path, weights)
        runner = CliRunner()
        result = runner.invoke(app, ["adapters", "scan", str(adapter.relative_to(tmp_path))])
        # Exit code 0 (OK) or 1 (WARN) is acceptable for clean.
        assert result.exit_code in (0, 1), (result.output, repr(result.exception))

    def test_scan_outside_cwd_exits_nonzero(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(app, ["adapters", "scan", "/nonexistent/outside"])
        assert result.exit_code != 0


# ---------- Source wiring ----------


class TestSourceWiring:
    def test_module_imports_clean(self):
        import soup_cli.utils.adapter_scan as m

        assert hasattr(m, "scan_adapter")
        assert hasattr(m, "scan_adapter_weights")
        assert hasattr(m, "ScanReport")
        assert hasattr(m, "ScanFinding")

    def test_no_top_level_torch(self):
        src = (
            Path(__file__).resolve().parent.parent
            / "src" / "soup_cli" / "utils" / "adapter_scan.py"
        )
        text = src.read_text(encoding="utf-8")
        # numpy is fine; torch must be lazy
        assert "import torch" not in text or "    import torch" in text or "  import torch" in text
