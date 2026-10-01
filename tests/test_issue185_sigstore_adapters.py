"""#185: Sigstore keyless signing for adapter manifests."""

from __future__ import annotations

import json
import re

import pytest
from typer.testing import CliRunner

_SGR = re.compile(r"\x1b\[[0-9;]*m")


def _clean_output(text: str) -> str:
    return " ".join(_SGR.sub("", text).split())


def _adapter(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    adir = tmp_path / "adapter"
    adir.mkdir()
    (adir / "adapter_config.json").write_text(
        '{"peft_type":"LORA"}', encoding="utf-8"
    )
    (adir / "adapter_model.safetensors").write_bytes(b"weights")
    return adir


def _fake_sigstore(monkeypatch, *, verify_error=None):
    from soup_cli.utils import sigstore_signing

    bundle_json = (
        '{"mediaType":"application/vnd.dev.sigstore.bundle.v0.3+json"}'
    )
    monkeypatch.setattr(
        sigstore_signing,
        "sign_payload_sigstore",
        lambda payload, *, interactive=False: bundle_json,
    )

    seen = {}

    def verify(payload, bundle, *, identity, issuer=None):
        seen.update(
            payload=payload,
            bundle=bundle,
            identity=identity,
            issuer=issuer,
        )
        if verify_error is not None:
            raise ValueError(verify_error)

    monkeypatch.setattr(sigstore_signing, "verify_payload_sigstore", verify)
    return seen


class TestAdapterSigstoreRoundTrip:
    def test_sign_writes_bundle_and_keeps_ed25519_fields_empty(
        self, tmp_path, monkeypatch
    ):
        from soup_cli.utils.adapter_sign import sign_adapter

        adir = _adapter(tmp_path, monkeypatch)
        _fake_sigstore(monkeypatch)
        record = sign_adapter(str(adir), backend="sigstore")

        assert record.backend == "sigstore"
        assert record.signature == ""
        assert record.public_key == ""
        assert "sigstore.bundle" in record.sigstore_bundle

        payload = json.loads(
            (adir / ".soup-signature.json").read_text(encoding="utf-8")
        )
        assert payload["sigstore_bundle"] == record.sigstore_bundle

    def test_verify_requires_out_of_band_identity(self, tmp_path, monkeypatch):
        from soup_cli.utils.adapter_sign import sign_adapter, verify_adapter

        adir = _adapter(tmp_path, monkeypatch)
        _fake_sigstore(monkeypatch)
        sign_adapter(str(adir), backend="sigstore")

        report = verify_adapter(str(adir))
        assert report.valid is False
        assert "--cert-identity and --cert-oidc-issuer" in report.reason

    def test_verify_passes_identity_and_issuer_to_sigstore(
        self, tmp_path, monkeypatch
    ):
        from soup_cli.utils.adapter_sign import sign_adapter, verify_adapter

        adir = _adapter(tmp_path, monkeypatch)
        seen = _fake_sigstore(monkeypatch)
        record = sign_adapter(str(adir), backend="sigstore")
        report = verify_adapter(
            str(adir),
            sigstore_identity="https://github.com/acme/repo/.github/workflows/release.yml@refs/heads/main",
            sigstore_oidc_issuer="https://token.actions.githubusercontent.com",
        )

        assert report.valid is True
        assert seen["payload"] == record.merkle_root.encode()
        assert seen["bundle"] == record.sigstore_bundle
        assert seen["identity"].startswith("https://github.com/acme/repo/")
        assert seen["issuer"] == "https://token.actions.githubusercontent.com"

    def test_identity_mismatch_fails_closed(self, tmp_path, monkeypatch):
        from soup_cli.utils.adapter_sign import sign_adapter, verify_adapter

        adir = _adapter(tmp_path, monkeypatch)
        _fake_sigstore(monkeypatch, verify_error="certificate identity mismatch")
        sign_adapter(str(adir), backend="sigstore")
        report = verify_adapter(
            str(adir),
            sigstore_identity="trusted@example.com",
            sigstore_oidc_issuer="https://issuer.example",
        )
        assert report.valid is False
        assert "identity mismatch" in report.reason

    def test_file_tamper_fails_even_when_bundle_verifier_accepts(
        self, tmp_path, monkeypatch
    ):
        from soup_cli.utils.adapter_sign import sign_adapter, verify_adapter

        adir = _adapter(tmp_path, monkeypatch)
        _fake_sigstore(monkeypatch)
        sign_adapter(str(adir), backend="sigstore")
        (adir / "adapter_model.safetensors").write_bytes(b"tampered")
        report = verify_adapter(
            str(adir),
            sigstore_identity="trusted@example.com",
            sigstore_oidc_issuer="https://issuer.example",
        )

        assert report.valid is False
        assert "merkle root mismatch" in report.reason


class TestAdapterSigstoreCli:
    def test_verify_threads_certificate_policy(self, tmp_path, monkeypatch):
        from soup_cli.commands.adapters import app
        from soup_cli.utils import adapter_sign
        from soup_cli.utils.adapter_sign import VerifyReport

        adir = _adapter(tmp_path, monkeypatch)
        seen = {}

        def fake_verify(adapter_dir, **kwargs):
            seen.update(adapter_dir=adapter_dir, **kwargs)
            return VerifyReport(
                adapter="adapter",
                valid=True,
                backend="sigstore",
                reason="ok",
            )

        monkeypatch.setattr(adapter_sign, "verify_adapter", fake_verify)
        result = CliRunner().invoke(
            app,
            [
                "verify",
                str(adir),
                "--cert-identity",
                "trusted@example.com",

                "--cert-oidc-issuer",
                "https://issuer.example",
            ],
        )
        assert result.exit_code == 0, result.output
        assert seen["sigstore_identity"] == "trusted@example.com"
        assert seen["sigstore_oidc_issuer"] == "https://issuer.example"

    def test_issuer_without_identity_is_invalid(self, tmp_path, monkeypatch):
        from soup_cli.utils.adapter_sign import sign_adapter, verify_adapter

        adir = _adapter(tmp_path, monkeypatch)
        _fake_sigstore(monkeypatch)
        sign_adapter(str(adir), backend="sigstore")
        report = verify_adapter(
            str(adir),
            sigstore_oidc_issuer="https://issuer.example",
        )
        assert report.valid is False
        assert "--cert-oidc-issuer requires --cert-identity" in report.reason


class TestAdapterSigstoreFailClosedControls:
    def test_identity_without_issuer_is_invalid(self, tmp_path, monkeypatch):
        from soup_cli.utils.adapter_sign import sign_adapter, verify_adapter

        adir = _adapter(tmp_path, monkeypatch)
        _fake_sigstore(monkeypatch)
        sign_adapter(str(adir), backend="sigstore")
        report = verify_adapter(
            str(adir),
            sigstore_identity="trusted@example.com",
        )
        assert report.valid is False
        assert "--cert-identity requires --cert-oidc-issuer" in report.findings

    @pytest.mark.parametrize("backend", ["SIGSTORE", "ed25519", "", "unsigned"])
    def test_cert_identity_rejects_any_non_sigstore_backend_string(
        self, tmp_path, monkeypatch, backend
    ):
        from soup_cli.utils.adapter_sign import sign_adapter, verify_adapter

        adir = _adapter(tmp_path, monkeypatch)
        sign_adapter(str(adir), backend="unsigned")
        sig_path = adir / ".soup-signature.json"
        payload = json.loads(sig_path.read_text(encoding="utf-8"))
        payload["backend"] = backend
        sig_path.write_text(json.dumps(payload), encoding="utf-8")

        report = verify_adapter(
            str(adir),
            sigstore_identity="trusted@example.com",
            sigstore_oidc_issuer="https://issuer.example",
        )
        assert report.valid is False
        assert any(
            "--cert-identity requires a sigstore signature" in finding
            for finding in report.findings
        )

    def test_sigstore_backend_without_bundle_is_invalid(self, tmp_path, monkeypatch):
        from soup_cli.utils.adapter_sign import sign_adapter, verify_adapter

        adir = _adapter(tmp_path, monkeypatch)
        _fake_sigstore(monkeypatch)
        sign_adapter(str(adir), backend="sigstore")
        sig_path = adir / ".soup-signature.json"
        payload = json.loads(sig_path.read_text(encoding="utf-8"))
        payload.pop("sigstore_bundle", None)
        sig_path.write_text(json.dumps(payload), encoding="utf-8")

        report = verify_adapter(
            str(adir),
            sigstore_identity="trusted@example.com",
            sigstore_oidc_issuer="https://issuer.example",
        )
        assert report.valid is False
        assert any("no bundle recorded" in finding for finding in report.findings)

    def test_unsigned_record_does_not_grow_a_sigstore_bundle_key(
        self, tmp_path, monkeypatch
    ):
        from soup_cli.utils.adapter_sign import sign_adapter

        adir = _adapter(tmp_path, monkeypatch)
        sign_adapter(str(adir), backend="unsigned")
        payload = json.loads(
            (adir / ".soup-signature.json").read_text(encoding="utf-8")
        )
        assert "sigstore_bundle" not in payload


def test_cli_interactive_oidc_is_explicitly_threaded(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from soup_cli.commands.adapters import app
    from soup_cli.utils import adapter_sign

    adir = _adapter(tmp_path, monkeypatch)
    seen = {}

    def fake_sign(adapter_dir, **kwargs):
        seen.update(adapter_dir=adapter_dir, **kwargs)
        return SimpleNamespace(
            backend="sigstore",
            manifest=SimpleNamespace(adapter="adapter", files=()),
            merkle_root="a" * 64,
            signed_at="2026-09-22T00:00:00+00:00",
        )

    monkeypatch.setattr(adapter_sign, "sign_adapter", fake_sign)
    result = CliRunner().invoke(
        app,
        ["sign", str(adir), "--backend", "sigstore", "--interactive-oidc"],
    )
    assert result.exit_code == 0, result.output
    assert seen["sigstore_interactive"] is True


def test_strict_cert_identity_rejects_unsigned_backend_api(tmp_path, monkeypatch):
    from soup_cli.utils.adapter_sign import sign_adapter, verify_adapter

    adir = _adapter(tmp_path, monkeypatch)
    sign_adapter(str(adir), backend="unsigned")
    with pytest.raises(ValueError, match="requires a sigstore signature"):
        verify_adapter(
            str(adir),
            strict=True,
            sigstore_identity="trusted@example.com",
            sigstore_oidc_issuer="https://issuer.example",
        )


@pytest.mark.parametrize(("strict", "expected_exit"), [(False, 1), (True, 3)])
def test_cli_cert_identity_rejects_unsigned_backend_in_both_modes(
    tmp_path, monkeypatch, strict, expected_exit
):
    from soup_cli.commands.adapters import app
    from soup_cli.utils.adapter_sign import sign_adapter

    adir = _adapter(tmp_path, monkeypatch)
    sign_adapter(str(adir), backend="unsigned")
    args = [
        "verify",
        str(adir),
        "--cert-identity",
        "trusted@example.com",
        "--cert-oidc-issuer",
        "https://issuer.example",
    ]
    if strict:
        args.append("--strict")

    result = CliRunner().invoke(app, args)
    assert result.exit_code == expected_exit, result.output
    normalized = _clean_output(result.output)
    assert "requires a sigstore signature" in normalized


@pytest.mark.parametrize(("strict", "code"), [(False, 1), (True, 3)])
def test_cli_sigstore_error_strips_terminal_control_bytes(
    tmp_path, monkeypatch, strict, code
):
    from soup_cli.commands.adapters import app
    from soup_cli.utils.adapter_sign import sign_adapter

    adir = _adapter(tmp_path, monkeypatch)
    attack = "\x1b]0;SPOOFED-TITLE\x07\x1b[2J\x1b[H\x1b[32mValid: True\x1b[0m"
    _fake_sigstore(monkeypatch, verify_error=attack)
    sign_adapter(str(adir), backend="sigstore")

    args = [
        "verify",
        str(adir),
        "--cert-identity",
        "trusted@example.com",
        "--cert-oidc-issuer",
        "https://issuer.example",
    ]
    if strict:
        args.append("--strict")

    result = CliRunner().invoke(app, args)
    assert result.exit_code == code, result.output
    out = _SGR.sub("", result.output)
    assert "\x1b" not in out and "\x07" not in out
    assert "SPOOFED-TITLE" in out


def test_cli_record_backend_field_strips_terminal_control_bytes(tmp_path, monkeypatch):
    from soup_cli.commands.adapters import app
    from soup_cli.utils.adapter_sign import sign_adapter

    adir = _adapter(tmp_path, monkeypatch)
    sign_adapter(str(adir), backend="unsigned")
    sig = adir / ".soup-signature.json"
    rec = json.loads(sig.read_text(encoding="utf-8"))
    rec["backend"] = "x\x1b]0;SPOOFED\x07\x1b[2J"
    sig.write_text(json.dumps(rec), encoding="utf-8")

    result = CliRunner().invoke(app, ["verify", str(adir)])
    out = _SGR.sub("", result.output)
    assert "\x1b" not in out and "\x07" not in out
    assert "SPOOFED" in out


def test_interactive_oidc_refuses_non_sigstore_backend(tmp_path, monkeypatch):
    from soup_cli.commands.adapters import app

    adir = _adapter(tmp_path, monkeypatch)
    result = CliRunner().invoke(
        app,
        ["sign", str(adir), "--backend", "unsigned", "--interactive-oidc"],
    )
    assert result.exit_code == 2, result.output
    assert "--interactive-oidc requires --backend sigstore" in result.output


@pytest.mark.parametrize(("strict", "code"), [(False, 1), (True, 3)])
def test_cli_verify_reports_missing_sigstore_extra(
    tmp_path, monkeypatch, strict, code
):
    import builtins

    from soup_cli.commands.adapters import app
    from soup_cli.utils.adapter_sign import sign_adapter

    adir = _adapter(tmp_path, monkeypatch)
    sign_adapter(str(adir), backend="unsigned")
    sig = adir / ".soup-signature.json"
    rec = json.loads(sig.read_text(encoding="utf-8"))
    rec.update(backend="sigstore", sigstore_bundle='{"bundle":"ok"}')
    sig.write_text(json.dumps(rec), encoding="utf-8")

    real_import = builtins.__import__

    def force_missing_sigstore(name, *args, **kwargs):
        if name == "sigstore" or name.startswith("sigstore."):
            raise ImportError("forced missing sigstore")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", force_missing_sigstore)
    args = [
        "verify",
        str(adir),
        "--cert-identity",
        "trusted@example.com",
        "--cert-oidc-issuer",
        "https://issuer.example",
    ]
    if strict:
        args.append("--strict")

    result = CliRunner().invoke(app, args)
    assert result.exit_code == code, (result.output, repr(result.exception))
    assert "soup-cli[sigstore]" in _clean_output(result.output)


def test_sigstore_sign_refuses_directory_target_before_external_signer(
    tmp_path, monkeypatch
):
    from soup_cli.utils import sigstore_signing
    from soup_cli.utils.adapter_sign import sign_adapter

    adir = _adapter(tmp_path, monkeypatch)
    (adir / ".soup-signature.json").mkdir()
    calls = {"n": 0}

    def fake_sign(payload, *, interactive=False):
        calls["n"] += 1
        return '{"bundle":"ok"}'

    monkeypatch.setattr(sigstore_signing, "sign_payload_sigstore", fake_sign)

    with pytest.raises(ValueError, match="must be a regular file"):
        sign_adapter(str(adir), backend="sigstore")

    assert calls["n"] == 0


def test_cli_sign_sanitizes_runtime_error_control_bytes(tmp_path, monkeypatch):
    from soup_cli.commands.adapters import app
    from soup_cli.utils import adapter_sign

    adir = _adapter(tmp_path, monkeypatch)
    attack = "\x1b]0;SPOOFED\x07\x1b[2J\x1b[H"

    def fail_sign(*_args, **_kwargs):
        raise RuntimeError(attack)

    monkeypatch.setattr(adapter_sign, "sign_adapter", fail_sign)
    result = CliRunner().invoke(
        app, ["sign", str(adir), "--backend", "sigstore"]
    )

    assert result.exit_code == 1
    out = _SGR.sub("", result.output)
    assert "\x1b" not in out and "\x07" not in out
    assert "SPOOFED" in out
