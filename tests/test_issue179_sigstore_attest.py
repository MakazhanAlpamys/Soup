"""#179: Sigstore keyless signing for in-toto/SLSA attestations."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _patch_signer(monkeypatch, bundle='{"bundle":"ok"}'):
    from soup_cli.utils import sigstore_signing

    seen = {}

    def sign(payload, *, interactive=False):
        seen["payload"] = payload
        seen["interactive"] = interactive
        return bundle

    monkeypatch.setattr(sigstore_signing, "sign_payload_sigstore", sign)
    return seen


def _emit_args(output):
    return [
        "emit",
        "--stage",
        "train",
        "--subject",
        "adapter",
        "--sha",
        "a" * 64,
        "--sign",
        "sigstore",
        "--output",
        str(output),
    ]


def _write_verify_pair(tmp_path):
    statement = {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": [{"name": "x", "digest": {"sha256": "a" * 64}}],
        "predicateType": "https://slsa.dev/provenance/v1",
        "predicate": {},
    }
    statement_path = tmp_path / "att.json"
    statement_path.write_text(
        json.dumps(statement, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    sidecar_path = tmp_path / "att.json.sig"
    sidecar_path.write_text(
        json.dumps(
            {
                "backend": "sigstore",
                "signature": "",
                "public_key": "",
                "sigstore_bundle": '{"bundle":"ok"}',
            }
        ),
        encoding="utf-8",
    )
    return statement_path, sidecar_path


class TestAttestationSigstoreBackend:
    def test_sign_returns_portable_bundle(self, monkeypatch):
        from soup_cli.utils.attest import sign_attestation

        seen = _patch_signer(monkeypatch)
        result = sign_attestation(b'{"hello":"world"}', backend="sigstore")

        assert result == {
            "signature": "",
            "backend": "sigstore",
            "sigstore_bundle": '{"bundle":"ok"}',
        }
        assert seen["payload"] == b'{"hello":"world"}'


    def test_verify_wrapper_threads_identity_and_issuer(self, monkeypatch):
        from soup_cli.utils import sigstore_signing
        from soup_cli.utils.attest import verify_sigstore_attestation

        seen = {}

        def verify(payload, bundle, *, identity, issuer):
            seen.update(
                payload=payload,
                bundle=bundle,
                identity=identity,
                issuer=issuer,
            )

        monkeypatch.setattr(sigstore_signing, "verify_payload_sigstore", verify)
        assert verify_sigstore_attestation(
            b"statement",
            '{"bundle":"ok"}',
            identity="trusted@example.com",
            issuer="https://issuer.example",
        )
        assert seen == {
            "payload": b"statement",
            "bundle": '{"bundle":"ok"}',
            "identity": "trusted@example.com",
            "issuer": "https://issuer.example",
        }


class TestAttestSigstoreCli:
    def test_emit_writes_bundle_sidecar(self, tmp_path, monkeypatch):
        from soup_cli.commands.attest import app

        monkeypatch.chdir(tmp_path)
        _patch_signer(monkeypatch)
        output = tmp_path / "attestation.json"

        result = CliRunner().invoke(app, _emit_args(output))
        assert result.exit_code == 0, result.output
        assert output.exists()
        sidecar = json.loads(
            (tmp_path / "attestation.json.sig").read_text(encoding="utf-8")
        )
        assert sidecar["backend"] == "sigstore"
        assert sidecar["signature"] == ""
        assert sidecar["public_key"] == ""
        assert sidecar["sigstore_bundle"] == '{"bundle":"ok"}'

    def test_sigstore_failure_never_downgrades_to_unsigned(
        self, tmp_path, monkeypatch
    ):
        from soup_cli.commands.attest import app
        from soup_cli.utils import sigstore_signing

        monkeypatch.chdir(tmp_path)

        monkeypatch.setattr(
            sigstore_signing,
            "sign_payload_sigstore",
            lambda payload, *, interactive=False: (_ for _ in ()).throw(
                RuntimeError("OIDC unavailable")
            ),
        )
        output = tmp_path / "attestation.json"
        result = CliRunner().invoke(app, _emit_args(output))

        assert result.exit_code == 1
        assert "OIDC unavailable" in result.output
        assert not output.exists()
        assert not (tmp_path / "attestation.json.sig").exists()

    def test_verify_requires_out_of_band_identity(self, tmp_path, monkeypatch):
        from soup_cli.commands.attest import app

        monkeypatch.chdir(tmp_path)
        statement, sidecar = _write_verify_pair(tmp_path)
        result = CliRunner().invoke(
            app,
            ["verify", str(statement), "--signature", str(sidecar)],
        )
        assert result.exit_code == 3
        assert "--cert-identity" in result.output
        assert "--cert-oidc-issuer" in result.output

    def test_identity_without_issuer_is_usage_error(self, tmp_path, monkeypatch):
        from soup_cli.commands.attest import app

        monkeypatch.chdir(tmp_path)
        statement, sidecar = _write_verify_pair(tmp_path)
        result = CliRunner().invoke(
            app,
            [
                "verify",
                str(statement),
                "--signature",
                str(sidecar),
                "--cert-identity",
                "trusted@example.com",
            ],
        )
        assert result.exit_code == 2
        assert "--cert-identity requires --cert-oidc-issuer" in result.output

    def test_issuer_without_identity_is_usage_error(self, tmp_path, monkeypatch):
        from soup_cli.commands.attest import app

        monkeypatch.chdir(tmp_path)
        statement, sidecar = _write_verify_pair(tmp_path)
        result = CliRunner().invoke(
            app,
            [
                "verify",
                str(statement),
                "--signature",
                str(sidecar),
                "--cert-oidc-issuer",
                "https://issuer.example",
            ],
        )
        assert result.exit_code == 2
        assert "--cert-oidc-issuer requires --cert-identity" in result.output

    def test_verify_passes_identity_policy(self, tmp_path, monkeypatch):
        import soup_cli.commands.attest as attest_cmd

        monkeypatch.chdir(tmp_path)
        statement, sidecar = _write_verify_pair(tmp_path)
        seen = {}

        def verify(payload, bundle, *, identity, issuer=None):
            seen.update(
                payload=payload,
                bundle=bundle,
                identity=identity,
                issuer=issuer,
            )
            return True

        monkeypatch.setattr(attest_cmd, "verify_sigstore_attestation", verify)
        result = CliRunner().invoke(
            attest_cmd.app,
            [
                "verify",
                str(statement),
                "--signature",
                str(sidecar),

                "--cert-identity",
                "trusted@example.com",
                "--cert-oidc-issuer",
                "https://issuer.example",
            ],
        )
        assert result.exit_code == 0, result.output
        assert seen["identity"] == "trusted@example.com"
        assert seen["issuer"] == "https://issuer.example"
        assert seen["bundle"] == '{"bundle":"ok"}'
        assert b"https://in-toto.io/Statement/v1" in seen["payload"]

    def test_identity_mismatch_is_invalid(self, tmp_path, monkeypatch):
        import soup_cli.commands.attest as attest_cmd

        monkeypatch.chdir(tmp_path)
        statement, sidecar = _write_verify_pair(tmp_path)

        def reject(*args, **kwargs):
            raise ValueError("certificate identity mismatch")

        monkeypatch.setattr(attest_cmd, "verify_sigstore_attestation", reject)
        result = CliRunner().invoke(
            attest_cmd.app,
            [
                "verify",
                str(statement),
                "--signature",
                str(sidecar),
                "--cert-identity",
                "wrong@example.com",
                "--cert-oidc-issuer",
                "https://issuer.example",
            ],
        )
        assert result.exit_code == 3
        assert "identity mismatch" in result.output


    def test_oversized_signature_sidecar_is_rejected_before_json_load(
        self, tmp_path, monkeypatch
    ):
        import soup_cli.commands.attest as attest_cmd

        monkeypatch.chdir(tmp_path)
        statement, sidecar = _write_verify_pair(tmp_path)
        monkeypatch.setattr(attest_cmd, "_MAX_SIGNATURE_SIDECAR_BYTES", 8)
        result = CliRunner().invoke(
            attest_cmd.app,
            ["verify", str(statement), "--signature", str(sidecar)],
        )
        assert result.exit_code == 2
        assert "exceeds 8 bytes" in _strip_ansi(result.output)


class TestAttestSigstoreReviewFollowups:
    def test_empty_sigstore_bundle_is_an_error(self, monkeypatch):
        from soup_cli.utils import sigstore_signing
        from soup_cli.utils.attest import sign_attestation

        monkeypatch.setattr(
            sigstore_signing,
            "sign_payload_sigstore",
            lambda payload, *, interactive=False: "",
        )
        with pytest.raises(RuntimeError, match="empty bundle"):
            sign_attestation(b"statement", backend="sigstore")

    def test_emit_threads_interactive_oidc(self, tmp_path, monkeypatch):
        from soup_cli.commands.attest import app

        monkeypatch.chdir(tmp_path)
        seen = _patch_signer(monkeypatch)
        output = tmp_path / "att.json"
        args = _emit_args(output) + ["--interactive-oidc"]
        result = CliRunner().invoke(app, args)

        assert result.exit_code == 0, result.output
        assert seen["interactive"] is True

    @pytest.mark.parametrize(
        ("sign_value", "extra_args"),
        [
            ("sigstore", []),
            ("SIGSTORE", []),
            ("Sigstore", []),
            ("sigstore", ["--attach-to-registry", "entry-1"]),
            ("sigstore", ["--output", "../outside.json"]),
        ],
    )
    def test_invalid_local_args_refuse_before_sigstore_signing(
        self, tmp_path, monkeypatch, sign_value, extra_args
    ):
        import soup_cli.commands.attest as attest_cmd

        monkeypatch.chdir(tmp_path)
        called = {"sign": False}

        def should_not_sign(*args, **kwargs):
            called["sign"] = True
            raise AssertionError("Sigstore signing ran before local validation")

        monkeypatch.setattr(attest_cmd, "sign_attestation", should_not_sign)
        args = [
            "emit",
            "--stage",
            "train",
            "--subject",
            "adapter",
            "--sha",
            "a" * 64,
            "--sign",
            sign_value,
            *extra_args,
        ]
        result = CliRunner().invoke(attest_cmd.app, args)

        assert result.exit_code == 2, result.output
        assert called["sign"] is False

    @pytest.mark.requires_symlink
    @pytest.mark.parametrize("which", ["output", "sidecar"])
    def test_symlink_output_paths_refuse_before_sigstore_signing(
        self, tmp_path, monkeypatch, which
    ):
        import soup_cli.commands.attest as attest_cmd

        monkeypatch.chdir(tmp_path)
        called = {"sign": False}

        def should_not_sign(*args, **kwargs):
            called["sign"] = True
            raise AssertionError("Sigstore signing ran before symlink validation")

        monkeypatch.setattr(attest_cmd, "sign_attestation", should_not_sign)
        output = tmp_path / "att.json"
        target = tmp_path / "target"
        target.write_text("target", encoding="utf-8")
        link = output if which == "output" else tmp_path / "att.json.sig"
        link.symlink_to(target)

        result = CliRunner().invoke(
            attest_cmd.app, _emit_args(output)
        )
        assert result.exit_code == 2, result.output
        assert called["sign"] is False

    @pytest.mark.parametrize("which", ["output", "sidecar"])
    def test_directory_output_paths_refuse_before_sigstore_signing(
        self, tmp_path, monkeypatch, which
    ):
        import soup_cli.commands.attest as attest_cmd

        monkeypatch.chdir(tmp_path)
        called = {"sign": False}

        def should_not_sign(*args, **kwargs):
            called["sign"] = True
            raise AssertionError("Sigstore signing ran before directory validation")

        monkeypatch.setattr(attest_cmd, "sign_attestation", should_not_sign)
        output = tmp_path / "att.json"
        directory = output if which == "output" else tmp_path / "att.json.sig"
        directory.mkdir()

        result = CliRunner().invoke(
            attest_cmd.app, _emit_args(output)
        )
        assert result.exit_code == 2, result.output
        assert called["sign"] is False

    def test_empty_bundle_cli_fails_without_writing_files(self, tmp_path, monkeypatch):
        from soup_cli.commands.attest import app
        from soup_cli.utils import sigstore_signing

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            sigstore_signing,
            "sign_payload_sigstore",
            lambda payload, *, interactive=False: "",
        )
        output = tmp_path / "att.json"
        result = CliRunner().invoke(app, _emit_args(output))

        assert result.exit_code == 1, result.output
        assert "empty bundle" in result.output
        assert not output.exists()
        assert not (tmp_path / "att.json.sig").exists()

    def test_sigstore_with_key_refuses_before_signing(self, tmp_path, monkeypatch):
        from soup_cli.commands.attest import app

        monkeypatch.chdir(tmp_path)
        seen = _patch_signer(monkeypatch)
        output = tmp_path / "att.json"
        result = CliRunner().invoke(
            app,
            _emit_args(output) + ["--key", "signing.pem"],
        )

        assert result.exit_code == 2, result.output
        assert "payload" not in seen
        assert not output.exists()

    def test_public_key_is_refused_on_a_sigstore_record(self, tmp_path, monkeypatch):
        import soup_cli.commands.attest as attest_cmd

        monkeypatch.chdir(tmp_path)
        statement, sidecar = _write_verify_pair(tmp_path)
        (tmp_path / "trusted.pub").write_text("unused", encoding="utf-8")
        called = {"verify": False}

        def verify(*args, **kwargs):
            called["verify"] = True
            return True

        monkeypatch.setattr(attest_cmd, "verify_sigstore_attestation", verify)
        result = CliRunner().invoke(
            attest_cmd.app,
            [
                "verify",
                str(statement),
                "--signature",
                str(sidecar),
                "--public-key",
                "trusted.pub",
                "--cert-identity",
                "trusted@example.com",
                "--cert-oidc-issuer",
                "https://issuer.example",
            ],
        )

        assert result.exit_code == 2, result.output
        assert called["verify"] is False

    def test_verify_hands_the_verifier_the_bytes_emit_signed(self, tmp_path, monkeypatch):
        import soup_cli.commands.attest as attest_cmd

        monkeypatch.chdir(tmp_path)
        signed = _patch_signer(monkeypatch)
        output = tmp_path / "att.json"
        emitted = CliRunner().invoke(attest_cmd.app, _emit_args(output))
        assert emitted.exit_code == 0, emitted.output

        document = json.loads(output.read_text(encoding="utf-8"))
        output.write_bytes(
            (json.dumps(document, separators=(",", ":")) + "\r\n").encode("utf-8")
        )
        verified = {}

        def verify(payload, bundle, *, identity, issuer):
            verified["payload"] = payload
            return True

        monkeypatch.setattr(attest_cmd, "verify_sigstore_attestation", verify)
        result = CliRunner().invoke(
            attest_cmd.app,
            [
                "verify",
                str(output),
                "--signature",
                f"{output}.sig",
                "--cert-identity",
                "trusted@example.com",
                "--cert-oidc-issuer",
                "https://issuer.example",
            ],
        )

        assert result.exit_code == 0, result.output
        assert verified["payload"] == signed["payload"]

    def test_ed25519_sidecar_omits_empty_sigstore_field(self, tmp_path, monkeypatch):
        from soup_cli.commands.attest import app
        from soup_cli.utils.signing import generate_ed25519_private_pem

        monkeypatch.chdir(tmp_path)
        key = tmp_path / "signing.pem"
        key.write_text(generate_ed25519_private_pem(), encoding="utf-8")
        output = tmp_path / "att.json"

        result = CliRunner().invoke(
            app,
            [
                "emit",
                "--stage",
                "train",
                "--subject",
                "adapter",
                "--sha",
                "a" * 64,
                "--sign",
                "ed25519",
                "--key",
                str(key),
                "--output",
                str(output),
            ],
        )
        assert result.exit_code == 0, result.output
        sidecar = json.loads(
            (tmp_path / "att.json.sig").read_text(encoding="utf-8")
        )
        assert sidecar["backend"] == "ed25519"
        assert "sigstore_bundle" not in sidecar

    @pytest.mark.parametrize(
        "backend", ["ed25519", "SIGSTORE", "sigstore ", "", "unsigned"]
    )
    def test_cert_identity_rejects_every_non_sigstore_backend_with_valid_ed25519(
        self, tmp_path, monkeypatch, backend
    ):
        from soup_cli.commands.attest import app
        from soup_cli.utils.signing import generate_ed25519_private_pem

        monkeypatch.chdir(tmp_path)
        key = tmp_path / "signing.pem"
        key.write_text(generate_ed25519_private_pem(), encoding="utf-8")
        output = tmp_path / "att.json"
        emit = CliRunner().invoke(
            app,
            [
                "emit",
                "--stage",
                "train",
                "--subject",
                "adapter",
                "--sha",
                "a" * 64,
                "--sign",
                "ed25519",
                "--key",
                str(key),
                "--output",
                str(output),
            ],
        )
        assert emit.exit_code == 0, emit.output

        sidecar_path = tmp_path / "att.json.sig"
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sidecar["backend"] = backend
        sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

        result = CliRunner().invoke(
            app,
            [
                "verify",
                str(output),
                "--signature",
                str(sidecar_path),
                "--cert-identity",
                "trusted@example.com",
                "--cert-oidc-issuer",
                "https://issuer.example",
            ],
        )
        assert result.exit_code == 2, result.output
        assert "applies only to Sigstore signatures" in result.output

    def test_bundle_error_strips_terminal_control_bytes(self, tmp_path, monkeypatch):
        import soup_cli.commands.attest as attest_cmd

        monkeypatch.chdir(tmp_path)
        statement, sidecar = _write_verify_pair(tmp_path)
        attack = "\x1b]0;SPOOFED-TITLE\x07\x1b[2J\x1b[H\x1b[32mValid: True\x1b[0m"

        def reject(*args, **kwargs):
            raise ValueError(attack)

        monkeypatch.setattr(attest_cmd, "verify_sigstore_attestation", reject)
        result = CliRunner().invoke(
            attest_cmd.app,
            [
                "verify",
                str(statement),
                "--signature",
                str(sidecar),
                "--cert-identity",
                "trusted@example.com",
                "--cert-oidc-issuer",
                "https://issuer.example",
            ],
        )

        assert result.exit_code == 3, result.output
        out = _strip_ansi(result.output)
        assert "\x1b" not in out
        assert "\x07" not in out
        assert "SPOOFED-TITLE" in out

    def test_missing_sigstore_extra_uses_unavailable_exit(self, tmp_path, monkeypatch):
        import builtins

        import soup_cli.commands.attest as attest_cmd

        monkeypatch.chdir(tmp_path)
        statement, sidecar = _write_verify_pair(tmp_path)
        real_import = builtins.__import__

        def force_missing_sigstore(name, *args, **kwargs):
            if name == "sigstore" or name.startswith("sigstore."):
                raise ImportError("forced missing sigstore")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", force_missing_sigstore)
        result = CliRunner().invoke(
            attest_cmd.app,
            [
                "verify",
                str(statement),
                "--signature",
                str(sidecar),
                "--cert-identity",
                "trusted@example.com",
                "--cert-oidc-issuer",
                "https://issuer.example",
            ],
        )

        assert result.exit_code == 1, result.output
        out = _strip_ansi(result.output)
        assert "Sigstore verification unavailable" in out
        assert "soup-cli[sigstore]" in out

    def test_interactive_oidc_refuses_non_sigstore_backend(
        self, tmp_path, monkeypatch
    ):
        from soup_cli.commands.attest import app

        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(
            app,
            [
                "emit",
                "--stage",
                "train",
                "--subject",
                "adapter",
                "--sha",
                "a" * 64,
                "--sign",
                "unsigned",
                "--interactive-oidc",
            ],
        )
        assert result.exit_code == 2, result.output
        assert "--interactive-oidc requires --sign sigstore" in result.output


def test_verifier_production_failure_is_unavailable_not_invalid(tmp_path, monkeypatch):
    import sys
    from types import ModuleType

    import soup_cli.commands.attest as attest_cmd

    monkeypatch.chdir(tmp_path)
    statement, sidecar = _write_verify_pair(tmp_path)
    models = ModuleType("sigstore.models")
    verify_mod = ModuleType("sigstore.verify")
    policy_mod = ModuleType("sigstore.verify.policy")

    class Bundle:
        @classmethod
        def from_json(cls, _text):
            return cls()

    class Identity:
        def __init__(self, *, identity, issuer):
            self.identity = identity
            self.issuer = issuer

    class Verifier:
        @classmethod
        def production(cls):
            raise RuntimeError(
                "TUF metadata unavailable\x1b]0;SPOOFED-TITLE\x07\x1b[2J"
            )

    models.Bundle = Bundle
    verify_mod.Verifier = Verifier
    policy_mod.Identity = Identity
    monkeypatch.setitem(sys.modules, "sigstore.models", models)
    monkeypatch.setitem(sys.modules, "sigstore.verify", verify_mod)
    monkeypatch.setitem(sys.modules, "sigstore.verify.policy", policy_mod)

    result = CliRunner().invoke(
        attest_cmd.app,
        [
            "verify", str(statement), "--signature", str(sidecar),
            "--cert-identity", "trusted@example.com",
            "--cert-oidc-issuer", "https://issuer.example",
        ],
    )
    assert result.exit_code == 1, result.output
    out = _strip_ansi(result.output)
    flattened = " ".join(out.split())
    assert "verification unavailable" in flattened.lower()
    assert "TUF metadata unavailable" in flattened
    assert "\x1b" not in out
    assert "\x07" not in out


def test_emit_write_oserror_is_usage_failure(tmp_path, monkeypatch):
    import soup_cli.commands.attest as attest_cmd

    monkeypatch.chdir(tmp_path)
    _patch_signer(monkeypatch)
    output = tmp_path / "att.json"

    def fail_write(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(attest_cmd, "write_attestation", fail_write)
    result = CliRunner().invoke(attest_cmd.app, _emit_args(output))
    assert result.exit_code == 2, result.output
    cleaned = _strip_ansi(result.output)
    assert "Write failed" in cleaned
    assert "disk full" in cleaned


def test_attest_docs_document_unavailable_exit():
    docs_path = (
        Path(__file__).resolve().parents[1] / "docs" / "adapters-and-governance.md"
    )
    docs = docs_path.read_text(encoding="utf-8")
    assert "exits 1 when Sigstore" in docs
