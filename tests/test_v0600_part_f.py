"""Tests for v0.60.0 Part F — ``soup airgap-bundle``.

Coverage:
- ``AirgapBundlePlan`` frozen dataclass + bundle-size cap
- ``build_airgap_bundle`` tarball assembly + signed manifest
- ``inspect_airgap_bundle`` round-trip
- CLI smoke (``soup airgap-bundle``)
"""

from __future__ import annotations

import dataclasses
import json
import os
import tarfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app


def _make_model_dir(tmp_path: Path, name: str = "model") -> Path:
    target = tmp_path / name
    target.mkdir()
    (target / "config.json").write_text('{"model_type": "llama"}', encoding="utf-8")
    (target / "weights.safetensors").write_bytes(b"safe-weight-bytes")
    return target


def _make_dataset_dir(tmp_path: Path, name: str = "dataset") -> Path:
    target = tmp_path / name
    target.mkdir()
    (target / "train.jsonl").write_text(
        '{"text": "hello"}\n{"text": "world"}\n', encoding="utf-8"
    )
    return target


class TestPlan:
    def test_imports(self):
        from soup_cli.utils.airgap_bundle import (
            AirgapBundlePlan,
            BundleManifest,
            build_airgap_bundle,
            inspect_airgap_bundle,
        )
        assert callable(build_airgap_bundle)
        assert callable(inspect_airgap_bundle)
        assert dataclasses.is_dataclass(AirgapBundlePlan)
        assert dataclasses.is_dataclass(BundleManifest)

    def test_plan_frozen(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        model = _make_model_dir(tmp_path)
        from soup_cli.utils.airgap_bundle import AirgapBundlePlan

        plan = AirgapBundlePlan(
            output=str(tmp_path / "out.tar"),
            model_dir=str(model),
            dataset_dirs=(),
            wheel_dirs=(),
            kernel_dirs=(),
            bundle_size_cap_bytes=100 * 1024 * 1024 * 1024,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            plan.output = "evil"  # type: ignore[misc]

    def test_plan_bundle_cap_must_be_positive(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        model = _make_model_dir(tmp_path)
        from soup_cli.utils.airgap_bundle import AirgapBundlePlan

        with pytest.raises(ValueError):
            AirgapBundlePlan(
                output=str(tmp_path / "out.tar"),
                model_dir=str(model),
                dataset_dirs=(),
                wheel_dirs=(),
                kernel_dirs=(),
                bundle_size_cap_bytes=0,
            )

    def test_plan_rejects_bool_cap(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        model = _make_model_dir(tmp_path)
        from soup_cli.utils.airgap_bundle import AirgapBundlePlan

        with pytest.raises(ValueError):
            AirgapBundlePlan(
                output=str(tmp_path / "out.tar"),
                model_dir=str(model),
                dataset_dirs=(),
                wheel_dirs=(),
                kernel_dirs=(),
                bundle_size_cap_bytes=True,  # type: ignore[arg-type]
            )


class TestBuild:
    def test_build_minimal_bundle(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        model = _make_model_dir(tmp_path)
        from soup_cli.utils.airgap_bundle import (
            AirgapBundlePlan,
            build_airgap_bundle,
        )
        plan = AirgapBundlePlan(
            output=str(tmp_path / "bundle.tar"),
            model_dir=str(model),
            dataset_dirs=(),
            wheel_dirs=(),
            kernel_dirs=(),
            bundle_size_cap_bytes=10 * 1024 * 1024,
        )
        manifest = build_airgap_bundle(plan)
        assert os.path.isfile(plan.output)
        assert manifest.model_dir.endswith("model")
        # Tar contains manifest.json
        with tarfile.open(plan.output) as tar:
            names = tar.getnames()
            assert any(n.endswith("manifest.json") for n in names)
            assert any("weights.safetensors" in n for n in names)

    def test_build_with_datasets(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        model = _make_model_dir(tmp_path)
        dataset = _make_dataset_dir(tmp_path)
        from soup_cli.utils.airgap_bundle import (
            AirgapBundlePlan,
            build_airgap_bundle,
        )
        plan = AirgapBundlePlan(
            output=str(tmp_path / "bundle.tar"),
            model_dir=str(model),
            dataset_dirs=(str(dataset),),
            wheel_dirs=(),
            kernel_dirs=(),
            bundle_size_cap_bytes=10 * 1024 * 1024,
        )
        manifest = build_airgap_bundle(plan)
        assert len(manifest.datasets) == 1
        with tarfile.open(plan.output) as tar:
            names = tar.getnames()
            assert any("train.jsonl" in n for n in names)

    def test_build_refuses_oversize(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        model = _make_model_dir(tmp_path)
        from soup_cli.utils.airgap_bundle import (
            AirgapBundlePlan,
            build_airgap_bundle,
        )
        # Cap = 1 byte; the model has more.
        plan = AirgapBundlePlan(
            output=str(tmp_path / "bundle.tar"),
            model_dir=str(model),
            dataset_dirs=(),
            wheel_dirs=(),
            kernel_dirs=(),
            bundle_size_cap_bytes=1,
        )
        with pytest.raises(ValueError, match="(?i)cap|exceeds|size"):
            build_airgap_bundle(plan)

    def test_build_output_outside_cwd_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        model = _make_model_dir(tmp_path)
        from soup_cli.utils.airgap_bundle import (
            AirgapBundlePlan,
            build_airgap_bundle,
        )
        outside_out = str(tmp_path.parent / "bundle.tar")
        plan = AirgapBundlePlan(
            output=outside_out,
            model_dir=str(model),
            dataset_dirs=(),
            wheel_dirs=(),
            kernel_dirs=(),
            bundle_size_cap_bytes=10 * 1024 * 1024,
        )
        with pytest.raises(ValueError):
            build_airgap_bundle(plan)

    def test_build_model_outside_cwd_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.airgap_bundle import (
            AirgapBundlePlan,
            build_airgap_bundle,
        )
        plan = AirgapBundlePlan(
            output=str(tmp_path / "bundle.tar"),
            model_dir=str(tmp_path.parent / "outside_model"),
            dataset_dirs=(),
            wheel_dirs=(),
            kernel_dirs=(),
            bundle_size_cap_bytes=10 * 1024 * 1024,
        )
        with pytest.raises(ValueError):
            build_airgap_bundle(plan)


class TestInspect:
    def test_inspect_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        model = _make_model_dir(tmp_path)
        from soup_cli.utils.airgap_bundle import (
            AirgapBundlePlan,
            build_airgap_bundle,
            inspect_airgap_bundle,
        )
        plan = AirgapBundlePlan(
            output=str(tmp_path / "bundle.tar"),
            model_dir=str(model),
            dataset_dirs=(),
            wheel_dirs=(),
            kernel_dirs=(),
            bundle_size_cap_bytes=10 * 1024 * 1024,
        )
        build_airgap_bundle(plan)
        manifest = inspect_airgap_bundle(plan.output)
        assert manifest.model_dir.endswith("model")
        # Manifest's bundled-files list includes the model files.
        names = [entry.name for entry in manifest.files]
        assert any("weights.safetensors" in n for n in names)

    def test_inspect_outside_cwd_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.airgap_bundle import inspect_airgap_bundle

        with pytest.raises(ValueError):
            inspect_airgap_bundle(str(tmp_path.parent / "outside.tar"))

    def test_inspect_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.airgap_bundle import inspect_airgap_bundle

        with pytest.raises(FileNotFoundError):
            inspect_airgap_bundle(str(tmp_path / "nope.tar"))

    def test_inspect_non_tar_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bad = tmp_path / "not-a-tar.tar"
        bad.write_text("not actually a tarball", encoding="utf-8")
        from soup_cli.utils.airgap_bundle import inspect_airgap_bundle

        with pytest.raises((tarfile.ReadError, ValueError)):
            inspect_airgap_bundle(str(bad))


class TestAirgapCli:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(app, ["airgap-bundle", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "airgap" in result.output.lower() or "bundle" in result.output.lower()

    def test_build_via_cli(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        model = _make_model_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            app, [
                "airgap-bundle",
                "--model", str(model.relative_to(tmp_path)),
                "--output", "bundle.tar",
            ]
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert (tmp_path / "bundle.tar").is_file()

    def test_cap_rejection_via_cli(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        model = _make_model_dir(tmp_path)
        # Pad the model so the cap can be exercised even after rounding to
        # an integer byte count.
        (model / "weights.safetensors").write_bytes(b"x" * 10_000)
        runner = CliRunner()
        result = runner.invoke(
            app, [
                "airgap-bundle",
                "--model", str(model.relative_to(tmp_path)),
                "--output", "bundle.tar",
                "--bundle-size-cap", "0.000001",  # ~1073 bytes < 10 000
            ]
        )
        assert result.exit_code != 0


class TestSecurityReviewFixes:
    """Regression guards for the v0.60.0 Part F security-review fixes."""

    def test_inspect_uses_data_filter_when_available(self, tmp_path, monkeypatch):
        """Project policy: tarfile extraction filter must be set on py3.12+.

        This is a defence-in-depth control — ``inspect_airgap_bundle``
        only reads ``manifest.json`` today, but the filter assignment
        guards future maintainers who add ``tar.extractall``.
        """
        import sys
        if sys.version_info < (3, 12):
            pytest.skip("data_filter ships in Python 3.12+")
        # Just confirm the helper is referenced in the source — that's
        # the regression-proof bit.
        src = (
            Path(__file__).resolve().parent.parent
            / "src" / "soup_cli" / "utils" / "airgap_bundle.py"
        )
        text = src.read_text(encoding="utf-8")
        assert "tarfile.data_filter" in text

    @pytest.mark.skipif(os.name == "nt", reason="POSIX symlink semantics")
    def test_build_rejects_symlink_at_output(self, tmp_path, monkeypatch):
        """Pre-placed symlink at plan.output is rejected (TOCTOU defence)."""
        monkeypatch.chdir(tmp_path)
        model = _make_model_dir(tmp_path)
        target = tmp_path / "real.tar"
        target.write_bytes(b"")
        link = tmp_path / "bundle.tar"
        os.symlink(str(target), str(link))
        from soup_cli.utils.airgap_bundle import (
            AirgapBundlePlan,
            build_airgap_bundle,
        )
        plan = AirgapBundlePlan(
            output=str(link),
            model_dir=str(model),
            dataset_dirs=(),
            wheel_dirs=(),
            kernel_dirs=(),
            bundle_size_cap_bytes=10 * 1024 * 1024,
        )
        with pytest.raises(ValueError, match="(?i)symlink"):
            build_airgap_bundle(plan)

    def test_dataset_ordering_is_stable(self, tmp_path, monkeypatch):
        """Reordering dataset_dirs must NOT change the manifest file list."""
        monkeypatch.chdir(tmp_path)
        model = _make_model_dir(tmp_path)
        da = _make_dataset_dir(tmp_path, name="da")
        db = _make_dataset_dir(tmp_path, name="db")
        from soup_cli.utils.airgap_bundle import (
            AirgapBundlePlan,
            build_airgap_bundle,
        )
        plan1 = AirgapBundlePlan(
            output=str(tmp_path / "b1.tar"),
            model_dir=str(model),
            dataset_dirs=(str(da), str(db)),
            wheel_dirs=(),
            kernel_dirs=(),
            bundle_size_cap_bytes=10 * 1024 * 1024,
        )
        plan2 = AirgapBundlePlan(
            output=str(tmp_path / "b2.tar"),
            model_dir=str(model),
            dataset_dirs=(str(db), str(da)),
            wheel_dirs=(),
            kernel_dirs=(),
            bundle_size_cap_bytes=10 * 1024 * 1024,
        )
        m1 = build_airgap_bundle(plan1)
        m2 = build_airgap_bundle(plan2)
        names1 = sorted(e.name for e in m1.files)
        names2 = sorted(e.name for e in m2.files)
        assert names1 == names2

    def test_inspect_rejects_oversized_manifest(self, tmp_path, monkeypatch):
        """Crafted bundles with multi-GiB manifest.json are rejected."""
        import io as _io
        import tarfile as _tarfile
        monkeypatch.chdir(tmp_path)
        bad = tmp_path / "evil_bundle.tar"
        with _tarfile.open(bad, "w") as tar:
            # 65 MiB > 64 MiB cap.
            payload = b'{"name": "evil"}' + (b"x" * (65 * 1024 * 1024))
            info = _tarfile.TarInfo(name="manifest.json")
            info.size = len(payload)
            tar.addfile(info, _io.BytesIO(payload))
        from soup_cli.utils.airgap_bundle import inspect_airgap_bundle

        with pytest.raises(ValueError, match="(?i)exceeds"):
            inspect_airgap_bundle(str(bad))


class TestSourceWiring:
    def test_module_imports(self):
        from soup_cli.utils import airgap_bundle as m

        assert hasattr(m, "build_airgap_bundle")
        assert hasattr(m, "inspect_airgap_bundle")
        assert hasattr(m, "AirgapBundlePlan")

    def test_manifest_has_soup_version(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        model = _make_model_dir(tmp_path)
        from soup_cli.utils.airgap_bundle import (
            AirgapBundlePlan,
            build_airgap_bundle,
        )
        plan = AirgapBundlePlan(
            output=str(tmp_path / "bundle.tar"),
            model_dir=str(model),
            dataset_dirs=(),
            wheel_dirs=(),
            kernel_dirs=(),
            bundle_size_cap_bytes=10 * 1024 * 1024,
        )
        manifest = build_airgap_bundle(plan)
        # Manifest carries soup_version + created_at for audit trail.
        with tarfile.open(plan.output) as tar:
            member = tar.getmember("manifest.json")
            extracted = tar.extractfile(member)
            assert extracted is not None
            payload = json.loads(extracted.read().decode("utf-8"))
        assert "soup_version" in payload
        assert "created_at" in payload
        assert payload["soup_version"] == manifest.soup_version
