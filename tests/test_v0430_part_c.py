"""Tests for v0.43.0 Part C — Profiling extras + VSCode setup."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from soup_cli.utils.profiling_v0_43 import (
    BandwidthExpectation,
    detect_anomaly_context,
    expected_bandwidth,
    memory_snapshot_context,
    nccl_bandwidth_check,
    resolve_snapshot_path,
)
from soup_cli.utils.vscode_setup import build_launch_json, write_vscode_launch

# ----------------- snapshot path -----------------

class TestResolveSnapshotPath:
    def test_happy(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = resolve_snapshot_path("run-123")
        assert path.endswith(os.path.join("profiles", "run-123.snapshot.pickle"))
        assert os.path.realpath(str(tmp_path)) in path

    @pytest.mark.parametrize("bad", ["", ".", "..", "a/b", "a\\b", "a\x00b"])
    def test_invalid_run_id(self, bad, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError):
            resolve_snapshot_path(bad)

    def test_non_string_run_id(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="must be a string"):
            resolve_snapshot_path(123)  # type: ignore[arg-type]

    def test_invalid_base_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError):
            resolve_snapshot_path("run-1", base_dir="")
        with pytest.raises(ValueError, match="null"):
            resolve_snapshot_path("run-1", base_dir="prof\x00iles")

    def test_absolute_base_dir_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="relative"):
            resolve_snapshot_path("run-1", base_dir=str(tmp_path / "abs"))

    def test_dotdot_in_base_dir_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match=r"'\.\.'"):
            resolve_snapshot_path("run-1", base_dir="../escape")

    def test_dot_base_dir_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match=r"'\.\.'"):
            resolve_snapshot_path("run-1", base_dir="..")


# ----------------- memory_snapshot_context -----------------

class TestMemorySnapshotContext:
    def test_no_torch_yields_none(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # Force ImportError for torch.
        import builtins

        real_import = builtins.__import__

        def fake(name, *a, **kw):
            if name == "torch":
                raise ImportError("torch not installed")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", fake)
        with memory_snapshot_context("run-x") as path:
            assert path is None

    def test_invalid_max_entries(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError):
            with memory_snapshot_context("run-x", max_entries=0):
                pass
        with pytest.raises(ValueError):
            with memory_snapshot_context("run-x", max_entries=True):  # type: ignore[arg-type]
                pass

    def test_invalid_run_id_propagates(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError):
            with memory_snapshot_context(""):
                pass


class TestDetectAnomalyContext:
    def test_no_torch_yields_false(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake(name, *a, **kw):
            if name == "torch":
                raise ImportError("torch not installed")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", fake)
        with detect_anomaly_context() as enabled:
            assert enabled is False

    def test_with_real_torch_or_skip(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch not installed in this environment")
        with detect_anomaly_context() as enabled:
            # On any torch with autograd module the context is active.
            assert enabled is True


# ----------------- nccl_bandwidth_check -----------------

class TestExpectedBandwidth:
    @pytest.mark.parametrize(
        "gpu,link",
        [
            ("h100", "nvlink"),
            ("h100", "pcie"),
            ("a100", "nvlink"),
            ("a100", "pcie"),
            ("v100", "nvlink"),
            ("rtx4090", "pcie"),
        ],
    )
    def test_known_pairs(self, gpu, link):
        bw = expected_bandwidth(gpu, link)
        assert bw is not None and bw > 0

    def test_case_insensitive(self):
        assert expected_bandwidth("H100", "NVLINK") == 450.0

    def test_unknown_returns_none(self):
        assert expected_bandwidth("evil", "pcie") is None
        assert expected_bandwidth("h100", "carrier-pigeon") is None

    def test_non_string_returns_none(self):
        assert expected_bandwidth(None, "nvlink") is None  # type: ignore[arg-type]
        assert expected_bandwidth(123, 456) is None  # type: ignore[arg-type]

    def test_dataclass_frozen(self):
        be = BandwidthExpectation(gpu="h100", link="nvlink", expected_gb_per_sec=450.0)
        with pytest.raises(Exception):
            be.gpu = "a100"  # type: ignore[misc]


class TestNcclBandwidthCheck:
    def test_ok(self):
        result = nccl_bandwidth_check(
            gpu="h100", link="nvlink", measured_gb_per_sec=400.0
        )
        assert result["status"] == "OK"
        assert result["expected_gb_per_sec"] == 450.0
        assert 0.88 <= result["ratio"] <= 0.89

    def test_minor(self):
        result = nccl_bandwidth_check(
            gpu="h100", link="nvlink", measured_gb_per_sec=300.0
        )
        assert result["status"] == "MINOR"

    def test_major(self):
        result = nccl_bandwidth_check(
            gpu="h100", link="nvlink", measured_gb_per_sec=100.0
        )
        assert result["status"] == "MAJOR"

    def test_unknown_pair(self):
        result = nccl_bandwidth_check(
            gpu="evil", link="pcie", measured_gb_per_sec=5.0
        )
        assert result["status"] == "UNKNOWN"
        assert result["expected_gb_per_sec"] is None
        assert result["ratio"] is None

    def test_negative_measured_rejected(self):
        with pytest.raises(ValueError):
            nccl_bandwidth_check(
                gpu="h100", link="nvlink", measured_gb_per_sec=-1.0
            )

    def test_nonfinite_measured_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            nccl_bandwidth_check(
                gpu="h100",
                link="nvlink",
                measured_gb_per_sec=float("inf"),
            )

    def test_bool_measured_rejected(self):
        with pytest.raises(ValueError):
            nccl_bandwidth_check(
                gpu="h100", link="nvlink", measured_gb_per_sec=True  # type: ignore[arg-type]
            )


# ----------------- VSCode launch.json -----------------

class TestBuildLaunchJson:
    def test_default(self):
        payload = build_launch_json()
        assert payload["version"] == "0.2.0"
        configs = payload["configurations"]
        assert len(configs) >= 2
        train = next(c for c in configs if c["name"] == "soup train")
        assert "soup.yaml" in train["args"]
        assert train["module"] == "soup_cli.cli"

    def test_custom_config_path(self):
        payload = build_launch_json(config_path="my-cfg.yaml")
        train = next(c for c in payload["configurations"] if c["name"] == "soup train")
        assert "my-cfg.yaml" in train["args"]

    def test_invalid_config_path(self):
        with pytest.raises(ValueError):
            build_launch_json(config_path="")
        with pytest.raises(ValueError, match="null"):
            build_launch_json(config_path="x\x00y.yaml")
        with pytest.raises(ValueError, match="newlines"):
            build_launch_json(config_path="a\nb.yaml")
        with pytest.raises(ValueError):
            build_launch_json(config_path="a" * 1024)


class TestWriteVscodeLaunch:
    def test_writes_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        out = write_vscode_launch()
        assert Path(out).is_file()
        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        assert data["version"] == "0.2.0"

    def test_refuses_overwrite_without_force(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_vscode_launch()
        with pytest.raises(FileExistsError):
            write_vscode_launch()

    def test_force_overwrites(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_vscode_launch()
        # Mutate file then re-write with force=True.
        path = Path(tmp_path) / ".vscode" / "launch.json"
        path.write_text("garbage", encoding="utf-8")
        out = write_vscode_launch(force=True)
        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        assert data["version"] == "0.2.0"

    def test_outside_cwd_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="under cwd"):
            write_vscode_launch(target_dir=str(tmp_path.parent / "evil"))

    def test_invalid_target_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError):
            write_vscode_launch(target_dir="")
        with pytest.raises(ValueError, match="null"):
            write_vscode_launch(target_dir="ev\x00il")

    def test_force_must_be_bool(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError):
            write_vscode_launch(force="yes")  # type: ignore[arg-type]

    def test_symlink_target_rejected(self, tmp_path, monkeypatch):
        # Skip on Windows without dev-mode where symlinks need privilege.
        if os.name == "nt":
            pytest.skip("symlink creation needs admin/dev mode on Windows")
        monkeypatch.chdir(tmp_path)
        vscode_dir = tmp_path / ".vscode"
        vscode_dir.mkdir()
        outside = tmp_path / "outside.json"
        outside.write_text("{}", encoding="utf-8")
        os.symlink(str(outside), str(vscode_dir / "launch.json"))
        with pytest.raises(ValueError, match="symlink"):
            write_vscode_launch(force=True)
