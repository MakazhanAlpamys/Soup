"""Tests for v0.60.0 Part B — `soup adapters sign / verify`.

Coverage:
- ``AdapterManifest`` / ``SignatureRecord`` frozen dataclasses
- ``compute_adapter_manifest`` deterministic SHA-256 over file list
- ``sign_adapter`` with UNSIGNED backend (live) + SIGSTORE deferred stub
- ``verify_adapter`` strict + lenient modes
- CLI smoke (sign / verify)
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app


def _make_adapter(tmp_path: Path, name: str = "adapter") -> Path:
    target = tmp_path / name
    target.mkdir()
    (target / "adapter_model.safetensors").write_bytes(b"fake-weights-bytes")
    (target / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 8}), encoding="utf-8"
    )
    return target


class TestManifest:
    def test_imports(self):
        from soup_cli.utils.adapter_sign import (
            AdapterManifest,
            SignatureRecord,
            compute_adapter_manifest,
            sign_adapter,
            verify_adapter,
        )
        assert callable(compute_adapter_manifest)
        assert callable(sign_adapter)
        assert callable(verify_adapter)
        assert dataclasses.is_dataclass(AdapterManifest)
        assert dataclasses.is_dataclass(SignatureRecord)

    def test_manifest_deterministic(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapter = _make_adapter(tmp_path)
        from soup_cli.utils.adapter_sign import compute_adapter_manifest

        m1 = compute_adapter_manifest(str(adapter))
        m2 = compute_adapter_manifest(str(adapter))
        assert m1.adapter == m2.adapter
        assert m1.files == m2.files
        assert m1.merkle_root == m2.merkle_root
        # Stable hash across runs.
        assert len(m1.merkle_root) == 64

    def test_manifest_outside_cwd_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        outside = tmp_path.parent / "outside_adapter"
        from soup_cli.utils.adapter_sign import compute_adapter_manifest

        with pytest.raises(ValueError):
            compute_adapter_manifest(str(outside))

    def test_manifest_missing_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.adapter_sign import compute_adapter_manifest

        with pytest.raises(FileNotFoundError):
            compute_adapter_manifest(str(tmp_path / "nope"))

    def test_manifest_frozen(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapter = _make_adapter(tmp_path)
        from soup_cli.utils.adapter_sign import compute_adapter_manifest

        manifest = compute_adapter_manifest(str(adapter))
        with pytest.raises(dataclasses.FrozenInstanceError):
            manifest.merkle_root = "evil"  # type: ignore[misc]

    def test_manifest_changes_when_weights_change(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapter = _make_adapter(tmp_path)
        from soup_cli.utils.adapter_sign import compute_adapter_manifest

        m1 = compute_adapter_manifest(str(adapter))
        # Mutate weights file
        (adapter / "adapter_model.safetensors").write_bytes(b"mutated")
        m2 = compute_adapter_manifest(str(adapter))
        assert m1.merkle_root != m2.merkle_root

    def test_manifest_includes_all_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapter = _make_adapter(tmp_path)
        from soup_cli.utils.adapter_sign import compute_adapter_manifest

        manifest = compute_adapter_manifest(str(adapter))
        names = {entry.name for entry in manifest.files}
        assert "adapter_model.safetensors" in names
        assert "adapter_config.json" in names

    @pytest.mark.skipif(os.name == "nt", reason="POSIX symlink semantics")
    def test_manifest_symlink_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "real_adapter"
        target.mkdir()
        link = tmp_path / "linked"
        os.symlink(str(target), str(link))
        from soup_cli.utils.adapter_sign import compute_adapter_manifest

        with pytest.raises(ValueError):
            compute_adapter_manifest(str(link))


class TestSign:
    def test_sign_unsigned_backend(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapter = _make_adapter(tmp_path)
        from soup_cli.utils.adapter_sign import sign_adapter

        record = sign_adapter(str(adapter), backend="unsigned")
        assert record.backend == "unsigned"
        assert record.signature == ""
        sig_file = adapter / ".soup-signature.json"
        assert sig_file.is_file()
        payload = json.loads(sig_file.read_text(encoding="utf-8"))
        assert payload["backend"] == "unsigned"
        assert "merkle_root" in payload

    def test_sign_sigstore_deferred(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapter = _make_adapter(tmp_path)
        from soup_cli.utils.adapter_sign import sign_adapter

        with pytest.raises(NotImplementedError, match="v0.60.1"):
            sign_adapter(str(adapter), backend="sigstore")

    def test_sign_ed25519_deferred(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapter = _make_adapter(tmp_path)
        from soup_cli.utils.adapter_sign import sign_adapter

        with pytest.raises(NotImplementedError, match="v0.60.1"):
            sign_adapter(str(adapter), backend="ed25519")

    def test_sign_unknown_backend(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapter = _make_adapter(tmp_path)
        from soup_cli.utils.adapter_sign import sign_adapter

        with pytest.raises(ValueError):
            sign_adapter(str(adapter), backend="weird_unknown")

    def test_sign_signature_record_frozen(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapter = _make_adapter(tmp_path)
        from soup_cli.utils.adapter_sign import sign_adapter

        record = sign_adapter(str(adapter), backend="unsigned")
        with pytest.raises(dataclasses.FrozenInstanceError):
            record.backend = "evil"  # type: ignore[misc]


class TestVerify:
    def test_verify_signed_adapter_passes(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapter = _make_adapter(tmp_path)
        from soup_cli.utils.adapter_sign import sign_adapter, verify_adapter

        sign_adapter(str(adapter), backend="unsigned")
        report = verify_adapter(str(adapter))
        assert report.valid is True
        assert report.backend == "unsigned"

    def test_verify_unsigned_adapter_lenient(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapter = _make_adapter(tmp_path)
        from soup_cli.utils.adapter_sign import verify_adapter

        report = verify_adapter(str(adapter), strict=False)
        assert report.valid is False
        assert report.reason
        # Lenient mode does not raise; just reports invalid.

    def test_verify_unsigned_adapter_strict_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapter = _make_adapter(tmp_path)
        from soup_cli.utils.adapter_sign import verify_adapter

        with pytest.raises(ValueError, match="(?i)signed|signature"):
            verify_adapter(str(adapter), strict=True)

    def test_verify_tampered_weights_fails(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapter = _make_adapter(tmp_path)
        from soup_cli.utils.adapter_sign import sign_adapter, verify_adapter

        sign_adapter(str(adapter), backend="unsigned")
        # Tamper with weights AFTER signing
        (adapter / "adapter_model.safetensors").write_bytes(b"tampered-bytes")
        report = verify_adapter(str(adapter))
        assert report.valid is False
        # Strict mode raises on tamper.
        with pytest.raises(ValueError):
            verify_adapter(str(adapter), strict=True)

    def test_verify_outside_cwd_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.adapter_sign import verify_adapter

        with pytest.raises(ValueError):
            verify_adapter(str(tmp_path.parent / "outside_adapter"))

    def test_verify_unsigned_strict_distinct_message(self, tmp_path, monkeypatch):
        """Strict mode message must explicitly mention 'signed' (CI grep)."""
        monkeypatch.chdir(tmp_path)
        adapter = _make_adapter(tmp_path)
        from soup_cli.utils.adapter_sign import verify_adapter

        try:
            verify_adapter(str(adapter), strict=True)
            pytest.fail("expected ValueError")
        except ValueError as exc:
            msg = str(exc).lower()
            assert "signed" in msg or "signature" in msg


class TestSignVerifyCli:
    def test_sign_help(self):
        runner = CliRunner()
        result = runner.invoke(app, ["adapters", "sign", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))

    def test_verify_help(self):
        runner = CliRunner()
        result = runner.invoke(app, ["adapters", "verify", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))

    def test_sign_then_verify_cli(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapter = _make_adapter(tmp_path)
        rel = str(adapter.relative_to(tmp_path))
        runner = CliRunner()
        sign_result = runner.invoke(app, ["adapters", "sign", rel])
        assert sign_result.exit_code == 0, (sign_result.output, repr(sign_result.exception))
        verify_result = runner.invoke(app, ["adapters", "verify", rel])
        assert verify_result.exit_code == 0, (verify_result.output, repr(verify_result.exception))

    def test_verify_unsigned_strict_cli_exit3(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapter = _make_adapter(tmp_path)
        rel = str(adapter.relative_to(tmp_path))
        runner = CliRunner()
        result = runner.invoke(app, ["adapters", "verify", rel, "--strict"])
        assert result.exit_code == 3


class TestSecurityReviewFixes:
    """Regression guards for the v0.60.0 Part B security-review fixes."""

    def test_oversized_signature_rejected(self, tmp_path, monkeypatch):
        """Verify rejects a signature file > 16 MiB cap."""
        monkeypatch.chdir(tmp_path)
        adapter = _make_adapter(tmp_path)
        sig = adapter / ".soup-signature.json"
        # Write a 17 MiB JSON-shaped string — over the 16 MiB cap.
        big = '{"x": "' + ("a" * (17 * 1024 * 1024)) + '"}'
        sig.write_text(big, encoding="utf-8")
        from soup_cli.utils.adapter_sign import verify_adapter

        with pytest.raises(ValueError, match="(?i)exceeds"):
            verify_adapter(str(adapter))

    @pytest.mark.skipif(os.name == "nt", reason="POSIX symlink semantics")
    def test_symlinked_signature_file_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapter = _make_adapter(tmp_path)
        # Place a symlink at the signature location.
        target = tmp_path / "outside.json"
        target.write_text('{"backend": "evil"}', encoding="utf-8")
        os.symlink(str(target), str(adapter / ".soup-signature.json"))
        from soup_cli.utils.adapter_sign import verify_adapter

        with pytest.raises(ValueError, match="(?i)symlink"):
            verify_adapter(str(adapter))


class TestSourceWiring:
    def test_module_imports_clean(self):
        import soup_cli.utils.adapter_sign as m

        assert hasattr(m, "sign_adapter")
        assert hasattr(m, "verify_adapter")
        assert hasattr(m, "AdapterManifest")

    def test_signature_filename_constant(self):
        from soup_cli.utils.adapter_sign import _SIGNATURE_FILENAME

        assert _SIGNATURE_FILENAME == ".soup-signature.json"
