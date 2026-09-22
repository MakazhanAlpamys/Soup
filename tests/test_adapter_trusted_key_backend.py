"""adapters verify --public-key accepts only ed25519 signatures by that key."""

from __future__ import annotations

import json
import re

import pytest
from typer.testing import CliRunner

runner = CliRunner()


def _clean(text: str) -> str:
    return " ".join(re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text).split())


@pytest.fixture
def signed(tmp_path, monkeypatch):
    """An ed25519-signed adapter plus the signer's public key as trusted.pub."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SOUP_SIGNING_KEY", raising=False)
    from soup_cli.utils.adapter_sign import sign_adapter
    from soup_cli.utils.signing import (
        generate_ed25519_private_pem,
        load_private_key_file,
        public_key_pem,
    )

    adir = tmp_path / "adapter"
    adir.mkdir()
    (adir / "adapter_config.json").write_text(json.dumps({"r": 8}), encoding="utf-8")
    (adir / "adapter_model.safetensors").write_bytes(b"original-weights")
    key = tmp_path / "priv.pem"
    key.write_text(generate_ed25519_private_pem(), encoding="utf-8")
    sign_adapter(str(adir), backend="ed25519", key_path=str(key))
    trusted = tmp_path / "trusted.pub"
    trusted.write_text(public_key_pem(load_private_key_file(str(key))), encoding="utf-8")
    return adir, trusted


def _resign_unsigned_after_edit(adir):
    from soup_cli.utils.adapter_sign import sign_adapter

    (adir / "adapter_model.safetensors").write_bytes(b"replaced-weights")
    sign_adapter(str(adir))  # default backend: unsigned, recomputed Merkle root


def test_trusted_key_rejects_unsigned_record(signed):
    from soup_cli.utils.adapter_sign import verify_adapter

    adir, trusted = signed
    _resign_unsigned_after_edit(adir)
    report = verify_adapter(str(adir), trusted_public_key=str(trusted))
    assert report.valid is False
    assert report.backend == "unsigned"
    assert any("requires an ed25519 signature" in f for f in report.findings)


def test_trusted_key_strict_raises(signed):
    from soup_cli.utils.adapter_sign import verify_adapter

    adir, trusted = signed
    _resign_unsigned_after_edit(adir)
    with pytest.raises(ValueError, match="requires an ed25519 signature"):
        verify_adapter(str(adir), strict=True, trusted_public_key=str(trusted))


@pytest.mark.parametrize("backend", ["ED25519", "sigstore", "", "unsigned"])
def test_trusted_key_rejects_any_non_ed25519_backend_string(signed, backend):
    from soup_cli.utils.adapter_sign import verify_adapter

    adir, trusted = signed
    sig_path = adir / ".soup-signature.json"
    payload = json.loads(sig_path.read_text(encoding="utf-8"))
    payload["backend"] = backend
    sig_path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        report = verify_adapter(str(adir), trusted_public_key=str(trusted))
    except ValueError:
        return  # a record the loader refuses outright is also not accepted
    assert report.valid is False


def test_trusted_key_missing_file_with_unsigned_record_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from soup_cli.utils.adapter_sign import sign_adapter, verify_adapter

    adir = tmp_path / "ad"
    adir.mkdir()
    (adir / "adapter_model.safetensors").write_bytes(b"v1")
    sign_adapter(str(adir))
    report = verify_adapter(str(adir), trusted_public_key="no-such-key.pub")
    assert report.valid is False


def test_without_trusted_key_unsigned_still_verifies(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from soup_cli.utils.adapter_sign import sign_adapter, verify_adapter

    adir = tmp_path / "ad"
    adir.mkdir()
    (adir / "adapter_model.safetensors").write_bytes(b"v1")
    sign_adapter(str(adir))
    assert verify_adapter(str(adir)).valid is True


def test_trusted_key_matching_ed25519_still_valid(signed):
    from soup_cli.utils.adapter_sign import verify_adapter

    adir, trusted = signed
    assert verify_adapter(str(adir), trusted_public_key=str(trusted)).valid is True


def test_cli_strict_exit_3_names_reason(signed):
    from soup_cli.cli import app

    adir, trusted = signed
    _resign_unsigned_after_edit(adir)
    result = runner.invoke(
        app, ["adapters", "verify", str(adir), "--public-key", str(trusted), "--strict"]
    )
    assert result.exit_code == 3, (result.output, repr(result.exception))
    assert "requires an ed25519 signature" in _clean(result.output)


def test_cli_lenient_exit_1(signed):
    from soup_cli.cli import app

    adir, trusted = signed
    _resign_unsigned_after_edit(adir)
    result = runner.invoke(app, ["adapters", "verify", str(adir), "--public-key", str(trusted)])
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "requires an ed25519 signature" in _clean(result.output)
