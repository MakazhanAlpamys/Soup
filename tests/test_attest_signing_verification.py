"""v0.71.2 #179 — Sigstore + ed25519 signing and verification tests for soup attest.

Tests cover:
- sign_attestation with sigstore backend returns a populated bundle string.
- verify_cmd raises typer.Exit(0) on valid sigstore verification flow.
- ed25519 happy-path: generate key, sign payload, verify it succeeds.
- ed25519 missing-key: sidecar without public_key rejects with exit 1.
- ed25519 mismatched-signature: tampered payload or wrong key rejects with exit 1.
"""

from __future__ import annotations

import json
import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock


def _make_sigstore_sign_stub() -> ModuleType:
    """Build a minimal ``sigstore.sign`` module stub with Signer.for_identity."""
    mock_bundle = MagicMock()
    bundle_json = json.dumps({
        "version": "v1",
        "signatures": [{"keyId": "test-key-id", "sig": "deadbeef"}],
        "integratedTime": 1234567890,
    }).encode("utf-8")
    mock_bundle.to_json.return_value = bundle_json

    mock_cert = MagicMock()

    def _pub_bytes(encoding=None):
        return b"-----BEGIN CERTIFICATE-----\ntest-cert\n-----END CERTIFICATE-----"

    mock_cert.public_bytes = _pub_bytes

    mock_result = MagicMock()
    mock_result.bundle = mock_bundle
    mock_result.signature = b"deadbeefcafebabe"
    mock_result.cert = mock_cert

    mock_signer_instance = MagicMock()
    mock_signer_instance.sign.return_value = mock_result

    def for_identity_factory(cls: Any, *args: Any, **kwargs: Any) -> MagicMock:
        return mock_signer_instance

    mod = ModuleType("sigstore.sign")
    signer_cls = MagicMock()
    signer_cls.for_identity = staticmethod(for_identity_factory)
    mod.Signer = signer_cls
    return mod


def _make_sigstore_verify_stub() -> ModuleType:
    """Build a minimal ``sigstore.verify`` module stub with Verifier + verify()."""
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
    mock_ctx.__exit__ = MagicMock(return_value=False)

    verifier_cls = MagicMock()
    verifier_cls.production.return_value = mock_ctx

    def verify_func(*args: Any, **kwargs: Any) -> None:
        pass  # no-op — success path

    mod = ModuleType("sigstore.verify")
    mod.Verifier = verifier_cls
    mod.verify = verify_func
    return mod


class TestSignAttestationSigstore:
    def test_sigstore_returns_populated_bundle_string(self):
        """Mocked sigstore signing should return a non-empty bundle JSON string."""
        # sigstore is not installed in the test env; inject stubs via sys.modules.
        _orig_sign = sys.modules.get("sigstore.sign")
        _orig_pkg = sys.modules.get("sigstore")

        try:
            sigstore_pkg = ModuleType("sigstore")
            sigstore_sign = _make_sigstore_sign_stub()
            sigstore_pkg.sign = sigstore_sign
            sigstore_pkg.oidc = ModuleType("sigstore.oidc")
            sigstore_pkg.oidc.Issuer = MagicMock(
                prod=MagicMock(
                    return_value=MagicMock(
                        find_identity_token=MagicMock(return_value="fake-token")
                    )
                )
            )
            sys.modules["sigstore"] = sigstore_pkg
            sys.modules["sigstore.sign"] = sigstore_sign
            sys.modules["sigstore.oidc"] = sigstore_pkg.oidc

            from soup_cli.utils.attest import sign_attestation

            result = sign_attestation(b"test-payload", backend="sigstore")

            assert result["backend"] == "sigstore"
            assert isinstance(result["bundle"], str)
            assert len(result["bundle"]) > 0
            parsed = json.loads(result["bundle"])
            assert parsed["version"] == "v1"
            assert len(parsed["signatures"]) == 1
            assert result["signature"] != ""
            assert result["certificate"] != ""
        finally:
            for key in ("sigstore.sign", "sigstore.oidc", "sigstore"):
                if key in sys.modules and sys.modules[key] not in (
                    _orig_sign, _orig_pkg
                ):
                    del sys.modules[key]
            if _orig_sign is not None:
                sys.modules["sigstore.sign"] = _orig_sign
            if _orig_pkg is not None:
                sys.modules["sigstore"] = _orig_pkg


class TestVerifyCmdSigstore:
    def test_verify_cmd_valid_sigstore_flow_exits_zero(self, tmp_path, monkeypatch):
        """verify_cmd should raise typer.Exit(0) on a valid sigstore verification."""
        from typer.testing import CliRunner

        from soup_cli.commands.attest import app

        monkeypatch.chdir(tmp_path)
        runner = CliRunner()

        # Create a minimal in-toto statement file.
        stmt = {
            "_type": "https://in-toto.io/Statement/v1",
            "subject": [{"name": "model.bin", "digest": {"sha256": "a" * 64}}],
            "predicateType": "https://slsa.dev/provenance/v1",
            "predicate": {},
        }
        stmt_path = tmp_path / "statement.json"
        stmt_path.write_text(
            json.dumps(stmt, indent=2, sort_keys=True), encoding="utf-8"
        )

        # Create a sigstore sidecar with valid-looking data.
        bundle_json_str = json.dumps({
            "version": "v1",
            "signatures": [{"keyId": "test-key-id", "sig": "deadbeef"}],
            "integratedTime": 1234567890,
        })
        sidecar = {
            "backend": "sigstore",
            "signature": "",
            "certificate": "-----BEGIN CERTIFICATE-----\ntest-cert\n-----END CERTIFICATE-----",
            "bundle": bundle_json_str,
            "identity_token_issuer": "https://github.com",
        }
        sidecar_path = tmp_path / "statement.json.sig"
        sidecar_path.write_text(
            json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8"
        )

        # Inject sigstore.verify stub via sys.modules.
        _orig_verify = sys.modules.get("sigstore.verify")
        _orig_pkg = sys.modules.get("sigstore")

        try:
            sigstore_pkg = ModuleType("sigstore")
            sigstore_verify = _make_sigstore_verify_stub()
            sigstore_pkg.verify = sigstore_verify
            sigstore_pkg.oidc = ModuleType("sigstore.oidc")
            sigstore_pkg.oidc.Issuer = MagicMock(
                prod=MagicMock(
                    return_value=MagicMock(identity=MagicMock(return_value="fake-id"))
                )
            )
            sys.modules["sigstore"] = sigstore_pkg
            sys.modules["sigstore.verify"] = sigstore_verify
            sys.modules["sigstore.oidc"] = sigstore_pkg.oidc

            result = runner.invoke(
                app,
                [
                    "verify",
                    str(stmt_path),
                ],
            )

            assert result.exit_code == 0, result.output
            assert "valid" in result.output.lower()
        finally:
            for key in ("sigstore.verify", "sigstore.oidc", "sigstore"):
                if key in sys.modules and sys.modules[key] not in (
                    _orig_verify, _orig_pkg
                ):
                    del sys.modules[key]
            if _orig_verify is not None:
                sys.modules["sigstore.verify"] = _orig_verify
            if _orig_pkg is not None:
                sys.modules["sigstore"] = _orig_pkg

    def test_verify_cmd_sigstore_missing_bundle_exits_one(self, tmp_path, monkeypatch):
        """verify_cmd should raise typer.Exit(1) when bundle data is missing."""
        from typer.testing import CliRunner

        from soup_cli.commands.attest import app

        monkeypatch.chdir(tmp_path)
        runner = CliRunner()

        stmt = {
            "_type": "https://in-toto.io/Statement/v1",
            "subject": [{"name": "model.bin", "digest": {"sha256": "a" * 64}}],
            "predicateType": "https://slsa.dev/provenance/v1",
            "predicate": {},
        }
        stmt_path = tmp_path / "statement.json"
        stmt_path.write_text(
            json.dumps(stmt, indent=2, sort_keys=True), encoding="utf-8"
        )

        # Sidecar with empty bundle — should fail verification.
        sidecar = {
            "backend": "sigstore",
            "signature": "",
            "certificate": "-----BEGIN CERTIFICATE-----\ntest-cert\n-----END CERTIFICATE-----",
          "bundle": "",
            "identity_token_issuer": "https://github.com",
        }
        sidecar_path = tmp_path / "statement.json.sig"
        sidecar_path.write_text(
            json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8"
        )

        result = runner.invoke(
            app,
            [
                "verify",
                str(stmt_path),
            ],
        )

        assert result.exit_code == 1, result.output
        assert "missing certificate or bundle" in result.output.lower()


class TestSigstoreImportError:
    def test_sigstore_missing_raises_friendly_import_error(self):
        """sign_attestation should raise a friendly ImportError when sigstore is not installed."""
        _orig_sigstore = sys.modules.get("sigstore")

        try:
            # Remove cached modules so the next import gets a fresh module
            # instance (avoids importlib.reload class-identity issues).
            for key in list(sys.modules):
                if key.startswith("soup_cli.utils.attest"):
                    del sys.modules[key]
            if "sigstore" in sys.modules:
                del sys.modules["sigstore"]
            if "sigstore.sign" in sys.modules:
                del sys.modules["sigstore.sign"]
            if "sigstore.oidc" in sys.modules:
                del sys.modules["sigstore.oidc"]

            from soup_cli.utils.attest import sign_attestation

            try:
                sign_attestation(b"test-payload", backend="sigstore")
                assert False, "Expected ImportError to be raised"
            except ImportError as exc:
                msg = str(exc)
                assert "sigstore" in msg.lower()
                assert "pip install" in msg or "install" in msg.lower()
        finally:
            if _orig_sigstore is not None:
                sys.modules["sigstore"] = _orig_sigstore


class TestEd25519HappyPath:
    def test_sign_and_verify_ed25519_exits_zero(self, tmp_path, monkeypatch):
        """ed25519 emit + verify should succeed with matching key."""
        from typer.testing import CliRunner

        from soup_cli.commands.attest import app

        monkeypatch.chdir(tmp_path)
        runner = CliRunner()

        # Generate an ed25519 private key.
        from soup_cli.utils.signing import generate_ed25519_private_pem

        priv_pem = generate_ed25519_private_pem()
        priv_path = tmp_path / "priv.pem"
        priv_path.write_text(priv_pem, encoding="utf-8")

        # Emit an attestation with ed25519 signing.
        stmt_out = runner.invoke(
            app,
            [
                "emit",
                "--stage", "train",
                "--subject", "model.bin",
                "--sha", "a" * 64,
                "--sign", "ed25519",
                "--key", str(priv_path),
                "--output", str(tmp_path / "statement.json"),
            ],
        )
        assert stmt_out.exit_code == 0, stmt_out.output

        # Verify the attestation using the embedded public key from sidecar.
        verify_out = runner.invoke(
            app,
            [
                "verify",
                str(tmp_path / "statement.json"),
            ],
        )
        assert verify_out.exit_code == 0, verify_out.output
        assert "valid" in verify_out.output.lower()

    def test_verify_with_trusted_public_key_exits_zero(self, tmp_path, monkeypatch):
        """verify_cmd with --public-key matching the embedded key should succeed."""
        from typer.testing import CliRunner

        from soup_cli.commands.attest import app

        monkeypatch.chdir(tmp_path)
        runner = CliRunner()

        # Generate an ed25519 private key.
        from soup_cli.utils.signing import (
            generate_ed25519_private_pem,
            private_key_from_pem,
            public_key_pem,
        )

        priv_pem = generate_ed25519_private_pem()
        priv_key = private_key_from_pem(priv_pem)
        pub_pem = public_key_pem(priv_key)

        priv_path = tmp_path / "priv.pem"
        priv_path.write_text(priv_pem, encoding="utf-8")
        pub_path = tmp_path / "pub.pem"
        pub_path.write_text(pub_pem, encoding="utf-8")

        # Emit an attestation with ed25519 signing.
        stmt_out = runner.invoke(
            app,
            [
                "emit",
                "--stage", "extract",
                "--subject", "data.bin",
                "--sha", "b" * 64,
                "--sign", "ed25519",
                "--key", str(priv_path),
                "--output", str(tmp_path / "statement.json"),
            ],
        )
        assert stmt_out.exit_code == 0, stmt_out.output

        # Verify with --public-key pointing to the matching trusted key.
        verify_out = runner.invoke(
            app,
            [
                "verify",
                str(tmp_path / "statement.json"),
                "--public-key", str(pub_path),
            ],
        )
        assert verify_out.exit_code == 0, verify_out.output
        assert "valid" in verify_out.output.lower()


class TestEd25519MissingKey:
    def test_verify_sidecar_without_public_key_exits_one(self, tmp_path, monkeypatch):
        """verify_cmd should reject when sidecar has no public_key and --public-key is absent."""
        from typer.testing import CliRunner

        from soup_cli.commands.attest import app

        monkeypatch.chdir(tmp_path)
        runner = CliRunner()

        # Create a sigstore-like statement file.
        stmt = {
            "_type": "https://in-toto.io/Statement/v1",
            "subject": [{"name": "model.bin", "digest": {"sha256": "c" * 64}}],
            "predicateType": "https://slsa.dev/provenance/v1",
            "predicate": {},
        }
        stmt_path = tmp_path / "statement.json"
        stmt_path.write_text(
            json.dumps(stmt, indent=2, sort_keys=True), encoding="utf-8"
        )

        # Create a sidecar with backend=ed25519 but NO public_key field.
        sidecar = {
            "backend": "ed25519",
            "signature": "deadbeefcafebabe",
            "public_key": "",
        }
        sidecar_path = tmp_path / "statement.json.sig"
        sidecar_path.write_text(
            json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8"
        )

        result = runner.invoke(
            app,
            [
                "verify",
                str(stmt_path),
            ],
        )

        assert result.exit_code == 1, result.output
        assert "no public key" in result.output.lower()


class TestEd25519MismatchedSignature:
    def test_tampered_payload_rejects(self, tmp_path, monkeypatch):
        """verify_cmd should reject when the payload has been tampered."""
        from typer.testing import CliRunner

        from soup_cli.commands.attest import app

        monkeypatch.chdir(tmp_path)
        runner = CliRunner()

        # Generate an ed25519 private key.
        from soup_cli.utils.signing import generate_ed25519_private_pem

        priv_pem = generate_ed25519_private_pem()
        priv_path = tmp_path / "priv.pem"
        priv_path.write_text(priv_pem, encoding="utf-8")

        # Emit an attestation with ed25519 signing.
        stmt_out = runner.invoke(
            app,
            [
                "emit",
                "--stage", "eval",
                "--subject", "model.bin",
                "--sha", "d" * 64,
                "--sign", "ed25519",
                "--key", str(priv_path),
                "--output", str(tmp_path / "statement.json"),
            ],
        )
        assert stmt_out.exit_code == 0, stmt_out.output

        # Tamper with the statement file (change a character).
        stmt_json = json.loads((tmp_path / "statement.json").read_text(encoding="utf-8"))
        stmt_json["subject"][0]["name"] = "TAMPERED.bin"
        (tmp_path / "statement.json").write_text(
            json.dumps(stmt_json, indent=2, sort_keys=True), encoding="utf-8"
        )

        # Verify the tampered statement — should fail.
        verify_out = runner.invoke(
            app,
            [
                "verify",
                str(tmp_path / "statement.json"),
            ],
        )
        assert verify_out.exit_code == 1, verify_out.output
        assert "INVALID" in verify_out.output

    def test_wrong_public_key_rejects(self, tmp_path, monkeypatch):
        """verify_cmd with --public-key of a different key should reject."""
        from typer.testing import CliRunner

        from soup_cli.commands.attest import app

        monkeypatch.chdir(tmp_path)
        runner = CliRunner()

        # Generate two DIFFERENT ed25519 keys.
        from soup_cli.utils.signing import generate_ed25519_private_pem

        priv_pem_a = generate_ed25519_private_pem()
        priv_pem_b = generate_ed25519_private_pem()

        priv_path_a = tmp_path / "priv_a.pem"
        priv_path_a.write_text(priv_pem_a, encoding="utf-8")
        pub_path_b = tmp_path / "pub_b.pem"

        # Derive the public key of key B from its PEM.
        from cryptography.hazmat.primitives import serialization

        priv_key_b = serialization.load_pem_private_key(
            priv_pem_b.encode("utf-8"), password=None
        )
        assert isinstance(
            priv_key_b,
            __import__(
                "cryptography.hazmat.primitives.asymmetric.ed25519",
                fromlist=["Ed25519PrivateKey"],
            ).Ed25519PrivateKey,
        )
        pub_pem_b = priv_key_b.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("utf-8")

        pub_path_b.write_text(pub_pem_b, encoding="utf-8")

        # Emit an attestation signed with key A.
        stmt_out = runner.invoke(
            app,
            [
                "emit",
                "--stage", "export",
                "--subject", "model.bin",
                "--sha", "e" * 64,
                "--sign", "ed25519",
                "--key", str(priv_path_a),
                "--output", str(tmp_path / "statement.json"),
            ],
        )
        assert stmt_out.exit_code == 0, stmt_out.output

        # Verify with --public-key of key B (mismatched) — should fail.
        verify_out = runner.invoke(
            app,
            [
                "verify",
                str(tmp_path / "statement.json"),
                "--public-key", str(pub_path_b),
            ],
        )
        assert verify_out.exit_code == 1, verify_out.output
        assert "untrusted key" in verify_out.output.lower()
