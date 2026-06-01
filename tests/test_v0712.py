"""v0.71.2 — "Governance & supply-chain live" (no GPU).

Closes (in this patch): #186, #187, #190, #191, #192 fully; #179 / #185 get the
ed25519 half live (Sigstore keyless stays infra-blocked — needs OIDC + network).

Test organisation (one class per concern):
- TestSigningPrimitives        — utils/signing.py ed25519 helpers (#179/#185 core)
- TestAdapterSignEd25519       — adapter_sign.py ed25519 backend (#185)
- TestAttestEd25519            — attest.py ed25519 backend (#179)
- TestNamespacePinConcurrency  — WAL + cross-process lock (#191)
- TestDownloadRepoNamespacePin — download_repo gate wiring (#186)
- TestLicenseExtraction        — adapter_config.json / model-card license (#187)
- TestLicenseOverrideAudit      — license-override -> audit-log (#190)
- TestMergeScanGate            — adapters merge refuses on scan FAIL (#192)
- TestCli                      — CLI smoke for the new flags
"""

from __future__ import annotations

import json
import os
import sys

import pytest

POSIX_ONLY = pytest.mark.skipif(
    sys.platform == "win32", reason="symlink creation needs elevation on Windows"
)


def _strip_ansi(text: str) -> str:
    """Remove ANSI SGR escapes from Rich-rendered help output.

    Under color (CI sets ``FORCE_COLOR``) Rich inserts ANSI codes *between* the
    two dashes of an option name, so ``--key`` becomes
    ``\x1b[1;36m-\x1b[0m\x1b[1;36m-key\x1b[0m`` and a raw substring check fails.
    Strip the codes before asserting (mirrors the v0.71.1 ``test_serve`` fix).
    """
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", text)


# ---------------------------------------------------------------------------
# #179 / #185 — ed25519 signing primitives (utils/signing.py)
# ---------------------------------------------------------------------------
class TestSigningPrimitives:
    def test_is_signing_available(self):
        from soup_cli.utils.signing import is_signing_available

        # cryptography is in the dev env; this must be True here.
        assert is_signing_available() is True

    def test_generate_and_roundtrip(self):
        from soup_cli.utils.signing import (
            generate_ed25519_private_pem,
            private_key_from_pem,
            public_key_pem,
            sign_payload,
            verify_payload,
        )

        pem = generate_ed25519_private_pem()
        assert "PRIVATE KEY" in pem
        key = private_key_from_pem(pem)
        pub = public_key_pem(key)
        assert "PUBLIC KEY" in pub
        sig = sign_payload(key, b"the merkle root")
        assert isinstance(sig, str) and len(sig) == 128  # 64-byte sig hex
        assert verify_payload(pub, b"the merkle root", sig) is True

    def test_verify_rejects_tampered_payload(self):
        from soup_cli.utils.signing import (
            generate_ed25519_private_pem,
            private_key_from_pem,
            public_key_pem,
            sign_payload,
            verify_payload,
        )

        key = private_key_from_pem(generate_ed25519_private_pem())
        pub = public_key_pem(key)
        sig = sign_payload(key, b"original")
        assert verify_payload(pub, b"tampered", sig) is False

    def test_verify_rejects_wrong_key(self):
        from soup_cli.utils.signing import (
            generate_ed25519_private_pem,
            private_key_from_pem,
            public_key_pem,
            sign_payload,
            verify_payload,
        )

        signer = private_key_from_pem(generate_ed25519_private_pem())
        other = private_key_from_pem(generate_ed25519_private_pem())
        sig = sign_payload(signer, b"x")
        assert verify_payload(public_key_pem(other), b"x", sig) is False

    def test_verify_bad_signature_hex_returns_false(self):
        from soup_cli.utils.signing import (
            generate_ed25519_private_pem,
            private_key_from_pem,
            public_key_pem,
            verify_payload,
        )

        pub = public_key_pem(private_key_from_pem(generate_ed25519_private_pem()))
        assert verify_payload(pub, b"x", "not-hex-zz") is False
        assert verify_payload(pub, b"x", "ab") is False  # wrong length

    def test_sign_payload_rejects_non_bytes(self):
        from soup_cli.utils.signing import (
            generate_ed25519_private_pem,
            private_key_from_pem,
            sign_payload,
        )

        key = private_key_from_pem(generate_ed25519_private_pem())
        with pytest.raises(TypeError):
            sign_payload(key, "a string")  # type: ignore[arg-type]

    def test_private_key_from_pem_rejects_non_ed25519(self):
        # RSA key PEM (generated inline) must be rejected — ed25519 only.
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        from soup_cli.utils.signing import private_key_from_pem

        rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        rsa_pem = rsa_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode("utf-8")
        with pytest.raises(ValueError, match="ed25519"):
            private_key_from_pem(rsa_pem)

    def test_private_key_from_pem_rejects_garbage(self):
        from soup_cli.utils.signing import private_key_from_pem

        with pytest.raises(ValueError):
            private_key_from_pem("not a pem")

    def test_load_private_key_file(self, tmp_path):
        from soup_cli.utils.signing import (
            generate_ed25519_private_pem,
            load_private_key_file,
            public_key_pem,
            sign_payload,
            verify_payload,
        )

        pem = generate_ed25519_private_pem()
        key_path = tmp_path / "priv.pem"
        key_path.write_text(pem, encoding="utf-8")
        key = load_private_key_file(str(key_path))
        sig = sign_payload(key, b"z")
        assert verify_payload(public_key_pem(key), b"z", sig) is True

    def test_load_private_key_file_missing(self, tmp_path):
        from soup_cli.utils.signing import load_private_key_file

        with pytest.raises(FileNotFoundError):
            load_private_key_file(str(tmp_path / "nope.pem"))

    def test_load_private_key_file_null_byte(self):
        from soup_cli.utils.signing import load_private_key_file

        with pytest.raises(ValueError):
            load_private_key_file("a\x00b.pem")

    @POSIX_ONLY
    def test_load_private_key_file_symlink_rejected(self, tmp_path):
        from soup_cli.utils.signing import (
            generate_ed25519_private_pem,
            load_private_key_file,
        )

        real = tmp_path / "real.pem"
        real.write_text(generate_ed25519_private_pem(), encoding="utf-8")
        link = tmp_path / "link.pem"
        os.symlink(real, link)
        with pytest.raises(ValueError, match="symlink"):
            load_private_key_file(str(link))

    def test_resolve_signing_key_explicit(self, tmp_path):
        from soup_cli.utils.signing import (
            generate_ed25519_private_pem,
            resolve_signing_key,
        )

        key_path = tmp_path / "k.pem"
        key_path.write_text(generate_ed25519_private_pem(), encoding="utf-8")
        key = resolve_signing_key(str(key_path), env={})
        assert key is not None

    def test_resolve_signing_key_env(self, tmp_path):
        from soup_cli.utils.signing import (
            generate_ed25519_private_pem,
            resolve_signing_key,
        )

        key_path = tmp_path / "k.pem"
        key_path.write_text(generate_ed25519_private_pem(), encoding="utf-8")
        key = resolve_signing_key(None, env={"SOUP_SIGNING_KEY": str(key_path)})
        assert key is not None

    def test_resolve_signing_key_missing_raises(self):
        from soup_cli.utils.signing import resolve_signing_key

        with pytest.raises(ValueError, match="private key"):
            resolve_signing_key(None, env={})


def _write_fake_adapter(tmp_path):
    """Write a minimal adapter dir (config + a tiny safetensors-ish blob)."""
    adir = tmp_path / "adapter"
    adir.mkdir()
    (adir / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 8}), encoding="utf-8"
    )
    (adir / "adapter_model.safetensors").write_bytes(b"\x00\x01\x02fake-weights")
    return adir


# ---------------------------------------------------------------------------
# #185 — ed25519 backend for adapter_sign
# ---------------------------------------------------------------------------
class TestAdapterSignEd25519:
    def _key(self, tmp_path):
        from soup_cli.utils.signing import generate_ed25519_private_pem

        kp = tmp_path / "priv.pem"
        kp.write_text(generate_ed25519_private_pem(), encoding="utf-8")
        return str(kp)

    def test_sign_ed25519_writes_real_signature(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.adapter_sign import sign_adapter

        adir = _write_fake_adapter(tmp_path)
        key = self._key(tmp_path)
        rec = sign_adapter(str(adir), backend="ed25519", key_path=key)
        assert rec.backend == "ed25519"
        assert rec.signature  # non-empty hex
        assert "PUBLIC KEY" in rec.public_key
        sig_file = json.loads((adir / ".soup-signature.json").read_text())
        assert sig_file["backend"] == "ed25519"
        assert sig_file["signature"]
        assert "PUBLIC KEY" in sig_file["public_key"]

    def test_verify_ed25519_passes_on_clean_adapter(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.adapter_sign import sign_adapter, verify_adapter

        adir = _write_fake_adapter(tmp_path)
        sign_adapter(str(adir), backend="ed25519", key_path=self._key(tmp_path))
        report = verify_adapter(str(adir))
        assert report.valid is True
        assert report.backend == "ed25519"

    def test_verify_ed25519_fails_on_tamper(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.adapter_sign import sign_adapter, verify_adapter

        adir = _write_fake_adapter(tmp_path)
        sign_adapter(str(adir), backend="ed25519", key_path=self._key(tmp_path))
        # Tamper a weight byte AFTER signing.
        (adir / "adapter_model.safetensors").write_bytes(b"\xff\xff\xfftampered")
        report = verify_adapter(str(adir))
        assert report.valid is False

    def test_verify_ed25519_fails_when_signature_corrupted(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.adapter_sign import sign_adapter, verify_adapter

        adir = _write_fake_adapter(tmp_path)
        sign_adapter(str(adir), backend="ed25519", key_path=self._key(tmp_path))
        # Flip the recorded signature so the ed25519 verify fails even though
        # the merkle root still matches.
        sig_path = adir / ".soup-signature.json"
        payload = json.loads(sig_path.read_text())
        payload["signature"] = "00" * 64
        sig_path.write_text(json.dumps(payload), encoding="utf-8")
        report = verify_adapter(str(adir))
        assert report.valid is False
        assert any("ed25519" in f.lower() for f in report.findings)

    def test_verify_with_matching_trusted_key(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.adapter_sign import sign_adapter, verify_adapter
        from soup_cli.utils.signing import load_private_key_file, public_key_pem

        adir = _write_fake_adapter(tmp_path)
        key_path = self._key(tmp_path)
        sign_adapter(str(adir), backend="ed25519", key_path=key_path)
        # Trusted pubkey == signer's pubkey -> verify passes authentication.
        trusted = tmp_path / "trusted.pub"
        trusted.write_text(
            public_key_pem(load_private_key_file(key_path)), encoding="utf-8"
        )
        report = verify_adapter(str(adir), trusted_public_key=str(trusted))
        assert report.valid is True

    def test_verify_with_untrusted_key_fails(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.adapter_sign import sign_adapter, verify_adapter
        from soup_cli.utils.signing import (
            generate_ed25519_private_pem,
            private_key_from_pem,
            public_key_pem,
        )

        adir = _write_fake_adapter(tmp_path)
        sign_adapter(str(adir), backend="ed25519", key_path=self._key(tmp_path))
        # A DIFFERENT trusted key -> authentication must fail.
        other = tmp_path / "other.pub"
        other.write_text(
            public_key_pem(private_key_from_pem(generate_ed25519_private_pem())),
            encoding="utf-8",
        )
        report = verify_adapter(str(adir), trusted_public_key=str(other))
        assert report.valid is False
        assert any("trust" in f.lower() or "untrusted" in f.lower() for f in report.findings)

    def test_sign_ed25519_without_key_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("SOUP_SIGNING_KEY", raising=False)
        from soup_cli.utils.adapter_sign import sign_adapter

        adir = _write_fake_adapter(tmp_path)
        with pytest.raises(ValueError, match="private key"):
            sign_adapter(str(adir), backend="ed25519")

    def test_sign_generate_key(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.adapter_sign import sign_adapter, verify_adapter

        adir = _write_fake_adapter(tmp_path)
        out_key = tmp_path / "fresh.pem"
        rec = sign_adapter(
            str(adir), backend="ed25519", generate_key_path=str(out_key)
        )
        assert rec.backend == "ed25519"
        assert out_key.exists()
        # The generated key must round-trip verify.
        assert verify_adapter(str(adir)).valid is True

    def test_sigstore_still_deferred(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.adapter_sign import sign_adapter

        adir = _write_fake_adapter(tmp_path)
        with pytest.raises(NotImplementedError, match="sigstore"):
            sign_adapter(str(adir), backend="sigstore")

    def test_unsigned_still_works(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.adapter_sign import sign_adapter, verify_adapter

        adir = _write_fake_adapter(tmp_path)
        rec = sign_adapter(str(adir), backend="unsigned")
        assert rec.backend == "unsigned"
        assert verify_adapter(str(adir)).valid is True


# ---------------------------------------------------------------------------
# #179 — ed25519 backend for soup attest
# ---------------------------------------------------------------------------
class TestAttestEd25519:
    def _key(self, tmp_path):
        from soup_cli.utils.signing import generate_ed25519_private_pem

        kp = tmp_path / "priv.pem"
        kp.write_text(generate_ed25519_private_pem(), encoding="utf-8")
        return str(kp)

    def test_sign_attestation_ed25519(self, tmp_path):
        from soup_cli.utils.attest import sign_attestation, verify_attestation

        payload = b'{"_type": "in-toto"}'
        sig = sign_attestation(payload, backend="ed25519", key_path=self._key(tmp_path))
        assert sig["backend"] == "ed25519"
        assert sig["signature"]
        assert "PUBLIC KEY" in sig["public_key"]
        # The returned signature must verify against the returned public key.
        assert verify_attestation(payload, sig["signature"], sig["public_key"]) is True

    def test_verify_attestation_rejects_tamper(self, tmp_path):
        from soup_cli.utils.attest import sign_attestation, verify_attestation

        payload = b"original-statement"
        sig = sign_attestation(payload, backend="ed25519", key_path=self._key(tmp_path))
        assert verify_attestation(b"tampered", sig["signature"], sig["public_key"]) is False

    def test_sign_attestation_ed25519_without_key_raises(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SOUP_SIGNING_KEY", raising=False)
        from soup_cli.utils.attest import sign_attestation

        with pytest.raises(ValueError, match="private key"):
            sign_attestation(b"x", backend="ed25519")

    def test_sign_attestation_unsigned_unchanged(self):
        from soup_cli.utils.attest import sign_attestation

        sig = sign_attestation(b"x", backend="unsigned")
        assert sig == {"signature": "", "backend": "unsigned"}

    def test_sign_attestation_sigstore_deferred(self):
        from soup_cli.utils.attest import sign_attestation

        with pytest.raises(NotImplementedError):
            sign_attestation(b"x", backend="sigstore")


# ---------------------------------------------------------------------------
# #191 — NamespacePinStore WAL + cross-process lock
# ---------------------------------------------------------------------------
class TestNamespacePinConcurrency:
    def test_wal_mode_enabled(self, tmp_path):
        from soup_cli.utils.namespace_pin import NamespacePinStore

        db = tmp_path / "pins.db"
        with NamespacePinStore(str(db)) as store:
            mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
            assert str(mode).lower() == "wal"

    def test_busy_timeout_set(self, tmp_path):
        from soup_cli.utils.namespace_pin import NamespacePinStore

        db = tmp_path / "pins.db"
        with NamespacePinStore(str(db)) as store:
            timeout = store._conn.execute("PRAGMA busy_timeout").fetchone()[0]
            assert int(timeout) > 0

    def test_concurrent_puts_preserve_first_seen(self, tmp_path):
        import threading

        from soup_cli.utils.namespace_pin import (
            NamespacePin,
            NamespacePinStore,
            record_repo_first_seen,
        )

        db = str(tmp_path / "pins.db")
        # Establish the trust anchor first.
        with NamespacePinStore(db) as store:
            first = record_repo_first_seen(
                store,
                repo_id="owner/repo",
                author="alice",
                created_at="2024-01-01T00:00:00+00:00",
            )
        anchor_first_seen = first.first_seen

        errors: list[Exception] = []

        def worker(i: int) -> None:
            try:
                # Each thread opens its OWN connection (the realistic
                # cross-process pattern) and overwrites author/created_at.
                with NamespacePinStore(db) as s:
                    s.put(
                        NamespacePin(
                            repo_id="owner/repo",
                            author=f"alice-{i}",
                            created_at=f"2024-02-0{i}T00:00:00+00:00",
                            # A bogus first_seen that MUST be ignored — the
                            # original anchor must survive.
                            first_seen="2099-12-31T00:00:00+00:00",
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(1, 9)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"concurrent puts raised: {errors}"
        with NamespacePinStore(db) as store:
            pin = store.get("owner/repo")
        assert pin is not None
        # first_seen preserved through every concurrent overwrite (no lost
        # anchor, no "database is locked" crash).
        assert pin.first_seen == anchor_first_seen
        assert pin.first_seen != "2099-12-31T00:00:00+00:00"

    def test_concurrent_first_insert_single_anchor(self, tmp_path):
        # N threads race to FIRST-insert the same unknown repo. The cross-process
        # lock must serialise the get+insert so they converge on one anchor.
        import threading

        from soup_cli.utils.namespace_pin import (
            NamespacePinStore,
            record_repo_first_seen,
        )

        db = str(tmp_path / "race.db")
        results: list[str] = []
        lock = threading.Lock()

        def worker() -> None:
            with NamespacePinStore(db) as s:
                pin = record_repo_first_seen(
                    s,
                    repo_id="owner/repo",
                    author="alice",
                    created_at="2024-01-01T00:00:00+00:00",
                )
            with lock:
                results.append(pin.first_seen)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # One canonical anchor lands in the DB; reading it back must be stable.
        with NamespacePinStore(db) as store:
            stored = store.get("owner/repo")
        assert stored is not None
        assert results  # all workers returned something


# ---------------------------------------------------------------------------
# #186 — Wire NamespacePinStore into download_repo (anti-AI-Jacking)
# ---------------------------------------------------------------------------
class TestDownloadRepoNamespacePin:
    def _setup(self, tmp_path, monkeypatch):
        """Point the pin DB at a tmp path and chdir so local_dir stays in cwd."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(
            "SOUP_NAMESPACE_PIN_DB", str(tmp_path / "pins.db")
        )

    def test_first_use_records_and_downloads(self, tmp_path, monkeypatch):
        from unittest.mock import patch

        self._setup(tmp_path, monkeypatch)
        from soup_cli.utils.hubs import download_repo

        meta = lambda rid: ("alice", "2024-01-01T00:00:00+00:00")  # noqa: E731
        with patch("huggingface_hub.snapshot_download", return_value="/snap") as m:
            result = download_repo(
                "hf", "alice/model", local_dir="./snap_pin",
                _metadata_fn=meta,
            )
        assert result == "/snap"
        m.assert_called_once()

    def test_second_use_same_author_passes(self, tmp_path, monkeypatch):
        from unittest.mock import patch

        self._setup(tmp_path, monkeypatch)
        from soup_cli.utils.hubs import download_repo

        meta = lambda rid: ("alice", "2024-01-01T00:00:00+00:00")  # noqa: E731
        with patch("huggingface_hub.snapshot_download", return_value="/snap"):
            download_repo("hf", "alice/model", local_dir="./s1", _metadata_fn=meta)
            # Second fetch — same fingerprint, must pass.
            download_repo("hf", "alice/model", local_dir="./s2", _metadata_fn=meta)

    def test_author_change_refused(self, tmp_path, monkeypatch):
        from unittest.mock import patch

        self._setup(tmp_path, monkeypatch)
        from soup_cli.utils.hubs import download_repo

        with patch("huggingface_hub.snapshot_download", return_value="/snap"):
            download_repo(
                "hf", "alice/model", local_dir="./s1",
                _metadata_fn=lambda rid: ("alice", "2024-01-01T00:00:00+00:00"),
            )
            # Namespace re-creation: same repo id, DIFFERENT author.
            with pytest.raises(ValueError, match="author changed"):
                download_repo(
                    "hf", "alice/model", local_dir="./s2",
                    _metadata_fn=lambda rid: ("mallory", "2025-06-01T00:00:00+00:00"),
                )

    def test_allow_namespace_shift_accepts(self, tmp_path, monkeypatch):
        from unittest.mock import patch

        self._setup(tmp_path, monkeypatch)
        from soup_cli.utils.hubs import download_repo

        with patch("huggingface_hub.snapshot_download", return_value="/snap") as m:
            download_repo(
                "hf", "alice/model", local_dir="./s1",
                _metadata_fn=lambda rid: ("alice", "2024-01-01T00:00:00+00:00"),
            )
            download_repo(
                "hf", "alice/model", local_dir="./s2",
                _metadata_fn=lambda rid: ("mallory", "2025-06-01T00:00:00+00:00"),
                allow_namespace_shift="mallory",
            )
        assert m.call_count == 2

    def test_metadata_unavailable_fails_open(self, tmp_path, monkeypatch):
        # When metadata can't be fetched (offline / rate-limited), the gate
        # must fail OPEN — refusing every download would break legit use.
        from unittest.mock import patch

        self._setup(tmp_path, monkeypatch)
        from soup_cli.utils.hubs import download_repo

        with patch("huggingface_hub.snapshot_download", return_value="/snap") as m:
            result = download_repo(
                "hf", "alice/model", local_dir="./s",
                _metadata_fn=lambda rid: None,
            )
        assert result == "/snap"
        m.assert_called_once()

    def test_namespace_check_disabled(self, tmp_path, monkeypatch):
        from unittest.mock import patch

        self._setup(tmp_path, monkeypatch)
        from soup_cli.utils.hubs import download_repo

        # _metadata_fn that would refuse is never consulted when disabled.
        def boom(rid):  # pragma: no cover - must not be called
            raise AssertionError("metadata_fn should not be called")

        with patch("huggingface_hub.snapshot_download", return_value="/snap") as m:
            download_repo(
                "hf", "alice/model", local_dir="./s",
                namespace_check=False, _metadata_fn=boom,
            )
        m.assert_called_once()

    def test_hf_repo_metadata_is_exception_safe(self, tmp_path, monkeypatch):
        from unittest.mock import patch

        self._setup(tmp_path, monkeypatch)
        from soup_cli.utils import hubs

        # A real HfApi failure (network / 404) must be swallowed -> None.
        with patch("huggingface_hub.HfApi") as fake_api:
            fake_api.return_value.repo_info.side_effect = RuntimeError("no net")
            assert hubs._hf_repo_metadata("alice/model") is None


# ---------------------------------------------------------------------------
# #187 — Auto-extract license from adapter_config.json / model card
# ---------------------------------------------------------------------------
class TestLicenseExtraction:
    def test_from_adapter_config(self, tmp_path):
        from soup_cli.utils.license_matrix import extract_license_from_adapter

        adir = tmp_path / "a"
        adir.mkdir()
        (adir / "adapter_config.json").write_text(
            json.dumps({"peft_type": "LORA", "license": "apache-2.0"}),
            encoding="utf-8",
        )
        assert extract_license_from_adapter(str(adir)) == "apache-2.0"

    def test_from_config_json(self, tmp_path):
        from soup_cli.utils.license_matrix import extract_license_from_adapter

        adir = tmp_path / "a"
        adir.mkdir()
        (adir / "config.json").write_text(
            json.dumps({"license": "mit"}), encoding="utf-8"
        )
        assert extract_license_from_adapter(str(adir)) == "mit"

    def test_from_model_card_frontmatter(self, tmp_path):
        from soup_cli.utils.license_matrix import extract_license_from_adapter

        adir = tmp_path / "a"
        adir.mkdir()
        (adir / "README.md").write_text(
            "---\nlicense: apache-2.0\ntags:\n  - lora\n---\n# My adapter\n",
            encoding="utf-8",
        )
        assert extract_license_from_adapter(str(adir)) == "apache-2.0"

    def test_hf_alias_llama3_1(self, tmp_path):
        # HF model cards spell it `llama3.1`; we canonicalise to `llama-3.1`.
        from soup_cli.utils.license_matrix import extract_license_from_adapter

        adir = tmp_path / "a"
        adir.mkdir()
        (adir / "README.md").write_text(
            "---\nlicense: llama3.1\n---\n", encoding="utf-8"
        )
        assert extract_license_from_adapter(str(adir)) == "llama-3.1"

    def test_license_as_list(self, tmp_path):
        from soup_cli.utils.license_matrix import extract_license_from_adapter

        adir = tmp_path / "a"
        adir.mkdir()
        (adir / "README.md").write_text(
            "---\nlicense:\n  - mit\n---\n", encoding="utf-8"
        )
        assert extract_license_from_adapter(str(adir)) == "mit"

    def test_unknown_license_returns_none(self, tmp_path):
        from soup_cli.utils.license_matrix import extract_license_from_adapter

        adir = tmp_path / "a"
        adir.mkdir()
        (adir / "config.json").write_text(
            json.dumps({"license": "some-bespoke-eula-2099"}), encoding="utf-8"
        )
        # Unknown ids don't trip the gate spuriously — return None ("undetermined").
        assert extract_license_from_adapter(str(adir)) is None

    def test_missing_metadata_returns_none(self, tmp_path):
        from soup_cli.utils.license_matrix import extract_license_from_adapter

        adir = tmp_path / "a"
        adir.mkdir()
        assert extract_license_from_adapter(str(adir)) is None

    def test_non_string_returns_none(self):
        from soup_cli.utils.license_matrix import extract_license_from_adapter

        assert extract_license_from_adapter(None) is None  # type: ignore[arg-type]
        assert extract_license_from_adapter(123) is None  # type: ignore[arg-type]

    def test_adapter_config_precedence_over_readme(self, tmp_path):
        from soup_cli.utils.license_matrix import extract_license_from_adapter

        adir = tmp_path / "a"
        adir.mkdir()
        (adir / "adapter_config.json").write_text(
            json.dumps({"license": "mit"}), encoding="utf-8"
        )
        (adir / "README.md").write_text(
            "---\nlicense: apache-2.0\n---\n", encoding="utf-8"
        )
        assert extract_license_from_adapter(str(adir)) == "mit"

    @POSIX_ONLY
    def test_symlinked_config_rejected(self, tmp_path):
        from soup_cli.utils.license_matrix import extract_license_from_adapter

        adir = tmp_path / "a"
        adir.mkdir()
        real = tmp_path / "real.json"
        real.write_text(json.dumps({"license": "mit"}), encoding="utf-8")
        os.symlink(real, adir / "config.json")
        # Symlinked config is skipped (not followed) -> undetermined.
        assert extract_license_from_adapter(str(adir)) is None


# ---------------------------------------------------------------------------
# #190 — license-override reason -> audit log
# ---------------------------------------------------------------------------
class TestLicenseOverrideAudit:
    def test_record_license_override_writes_audit(self, tmp_path, monkeypatch):
        from soup_cli.utils.audit_log import read_audit_tail
        from soup_cli.utils.license_matrix import record_license_override

        log = tmp_path / "audit.jsonl"
        record_license_override(
            ["apache-2.0", "gpl-3.0"],
            "legal team cleared this combination on 2026-06-01",
            path=str(log),
        )
        rows = read_audit_tail(str(log))
        assert len(rows) == 1
        ev = rows[0]
        assert ev["command"] == "adapters merge"
        assert any("apache-2.0" in a for a in ev["args"])
        assert any("legal team cleared" in a for a in ev["args"])
        assert ev["exit_code"] == 0

    def test_record_license_override_redacts_secrets(self, tmp_path):
        from soup_cli.utils.audit_log import read_audit_tail
        from soup_cli.utils.license_matrix import record_license_override

        log = tmp_path / "audit.jsonl"
        record_license_override(
            ["apache-2.0", "gpl-3.0"],
            "cleared by hf_abcdefgh12345678 token holder for the audit",
            path=str(log),
        )
        rows = read_audit_tail(str(log))
        blob = json.dumps(rows[0])
        assert "hf_abcdefgh12345678" not in blob
        assert "<redacted>" in blob

    def test_record_license_override_truncates_long_reason(self, tmp_path):
        from soup_cli.utils.audit_log import read_audit_tail
        from soup_cli.utils.license_matrix import record_license_override

        log = tmp_path / "audit.jsonl"
        record_license_override(
            ["apache-2.0", "gpl-3.0"], "x" * 5000, path=str(log)
        )
        rows = read_audit_tail(str(log))
        # Each arg must respect the AuditEvent 1024-char cap.
        assert all(len(a) <= 1024 for a in rows[0]["args"])


# ---------------------------------------------------------------------------
# #192 / #190 / #187 — `adapters merge` governance gates (CLI level)
# ---------------------------------------------------------------------------
def _make_adapter(tmp_path, name, license_id=None):
    import types as _t  # noqa: F401  (kept local)

    adir = tmp_path / name
    adir.mkdir()
    cfg = {"peft_type": "LORA", "r": 8}
    if license_id is not None:
        cfg["license"] = license_id
    (adir / "adapter_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (adir / "adapter_model.safetensors").write_bytes(b"fake")
    return str(adir)


def _fake_scan(overall):
    from types import SimpleNamespace

    def _scan(path):
        return SimpleNamespace(
            adapter=os.path.basename(path),
            overall=overall,
            summary=f"{overall} (stubbed)",
            findings=(),
        )

    return _scan


def _fake_merge_report():
    from types import SimpleNamespace

    return SimpleNamespace(
        strategy="linear",
        adapters=("a", "b"),
        merged_layers=1,
        skipped_layers=(),
        verdict="UNKNOWN",
        output_dir="merged_out",
    )


class TestMergeScanGate:
    def _run(self, monkeypatch, tmp_path, args, *, scan_overall="OK", scan_raises=None):
        from typer.testing import CliRunner

        from soup_cli.commands.adapters import app

        if scan_raises is not None:
            def _scan(path):
                raise scan_raises

            monkeypatch.setattr("soup_cli.utils.adapter_scan.scan_adapter", _scan)
        else:
            monkeypatch.setattr(
                "soup_cli.utils.adapter_scan.scan_adapter", _fake_scan(scan_overall)
            )
        monkeypatch.setattr(
            "soup_cli.utils.adapter_merge.merge_adapters",
            lambda *a, **k: _fake_merge_report(),
        )
        monkeypatch.chdir(tmp_path)
        return CliRunner().invoke(app, args)

    def test_scan_fail_refuses_merge(self, tmp_path, monkeypatch):
        a = _make_adapter(tmp_path, "a")
        b = _make_adapter(tmp_path, "b")
        result = self._run(
            monkeypatch, tmp_path,
            ["merge", a, b, "-o", "out"],
            scan_overall="FAIL",
        )
        assert result.exit_code == 3, result.output
        assert "scan" in result.output.lower()
        assert "allow-unscanned" in result.output

    def test_scan_fail_bypassed_with_allow_unscanned(self, tmp_path, monkeypatch):
        a = _make_adapter(tmp_path, "a")
        b = _make_adapter(tmp_path, "b")
        result = self._run(
            monkeypatch, tmp_path,
            ["merge", a, b, "-o", "out", "--allow-unscanned"],
            scan_overall="FAIL",
        )
        assert result.exit_code == 0, result.output

    def test_scan_warn_proceeds(self, tmp_path, monkeypatch):
        a = _make_adapter(tmp_path, "a")
        b = _make_adapter(tmp_path, "b")
        result = self._run(
            monkeypatch, tmp_path,
            ["merge", a, b, "-o", "out"],
            scan_overall="WARN",
        )
        assert result.exit_code == 0, result.output

    def test_scan_ok_proceeds(self, tmp_path, monkeypatch):
        a = _make_adapter(tmp_path, "a")
        b = _make_adapter(tmp_path, "b")
        result = self._run(
            monkeypatch, tmp_path,
            ["merge", a, b, "-o", "out"],
            scan_overall="OK",
        )
        assert result.exit_code == 0, result.output

    def test_unscannable_refuses_without_flag(self, tmp_path, monkeypatch):
        a = _make_adapter(tmp_path, "a")
        b = _make_adapter(tmp_path, "b")
        result = self._run(
            monkeypatch, tmp_path,
            ["merge", a, b, "-o", "out"],
            scan_raises=ValueError("cannot load safetensors"),
        )
        assert result.exit_code == 3, result.output
        assert "allow-unscanned" in result.output

    def test_unscannable_bypassed_with_flag(self, tmp_path, monkeypatch):
        a = _make_adapter(tmp_path, "a")
        b = _make_adapter(tmp_path, "b")
        result = self._run(
            monkeypatch, tmp_path,
            ["merge", a, b, "-o", "out", "--allow-unscanned"],
            scan_raises=ValueError("cannot load safetensors"),
        )
        assert result.exit_code == 0, result.output


class TestMergeLicenseAutoExtract:
    def _run(self, monkeypatch, tmp_path, args, audit_log=None):
        from typer.testing import CliRunner

        from soup_cli.commands.adapters import app

        # Scan always OK so we isolate the license gate.
        monkeypatch.setattr(
            "soup_cli.utils.adapter_scan.scan_adapter", _fake_scan("OK")
        )
        monkeypatch.setattr(
            "soup_cli.utils.adapter_merge.merge_adapters",
            lambda *a, **k: _fake_merge_report(),
        )
        if audit_log is not None:
            monkeypatch.setenv("SOUP_AUDIT_LOG_PATH", str(audit_log))
        monkeypatch.chdir(tmp_path)
        return CliRunner().invoke(app, args)

    def test_auto_extract_compatible_proceeds(self, tmp_path, monkeypatch):
        a = _make_adapter(tmp_path, "a", "apache-2.0")
        b = _make_adapter(tmp_path, "b", "mit")
        result = self._run(monkeypatch, tmp_path, ["merge", a, b, "-o", "out"])
        assert result.exit_code == 0, result.output
        assert "auto-detected" in result.output.lower()

    def test_auto_extract_conflict_refused(self, tmp_path, monkeypatch):
        a = _make_adapter(tmp_path, "a", "apache-2.0")
        b = _make_adapter(tmp_path, "b", "gpl-3.0")
        result = self._run(monkeypatch, tmp_path, ["merge", a, b, "-o", "out"])
        assert result.exit_code == 3, result.output
        assert "license" in result.output.lower()

    def test_auto_extract_undetermined_skips_gate(self, tmp_path, monkeypatch):
        # One adapter has no license metadata -> gate skipped, merge proceeds.
        a = _make_adapter(tmp_path, "a", "apache-2.0")
        b = _make_adapter(tmp_path, "b")  # no license
        result = self._run(monkeypatch, tmp_path, ["merge", a, b, "-o", "out"])
        assert result.exit_code == 0, result.output

    def test_explicit_conflict_override_logs_audit(self, tmp_path, monkeypatch):
        a = _make_adapter(tmp_path, "a")
        b = _make_adapter(tmp_path, "b")
        log = tmp_path / "audit.jsonl"
        result = self._run(
            monkeypatch, tmp_path,
            [
                "merge", a, b, "-o", "out",
                "--license", "apache-2.0", "--license", "gpl-3.0",
                "--license-override", "legal cleared this combo for v0.71.2 demo",
            ],
            audit_log=log,
        )
        assert result.exit_code == 0, result.output
        from soup_cli.utils.audit_log import read_audit_tail

        rows = read_audit_tail(str(log))
        assert any(r["command"] == "adapters merge" for r in rows)
        assert any(
            "legal cleared" in a for r in rows for a in r["args"]
        )


# ---------------------------------------------------------------------------
# CLI smoke — the new flags / commands end-to-end
# ---------------------------------------------------------------------------
class TestCli:
    def test_adapters_sign_help_lists_ed25519(self):
        from typer.testing import CliRunner

        from soup_cli.commands.adapters import app

        out = _strip_ansi(CliRunner().invoke(app, ["sign", "--help"]).output)
        assert "--key" in out
        assert "--generate-key" in out
        assert "ed25519" in out

    def test_adapters_verify_help_lists_public_key(self):
        from typer.testing import CliRunner

        from soup_cli.commands.adapters import app

        out = _strip_ansi(CliRunner().invoke(app, ["verify", "--help"]).output)
        assert "--public-key" in out

    def test_adapters_merge_help_lists_allow_unscanned(self):
        from typer.testing import CliRunner

        from soup_cli.commands.adapters import app

        out = _strip_ansi(CliRunner().invoke(app, ["merge", "--help"]).output)
        assert "--allow-unscanned" in out

    def test_sign_verify_ed25519_end_to_end(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.commands.adapters import app

        monkeypatch.chdir(tmp_path)
        adir = _write_fake_adapter(tmp_path)
        runner = CliRunner()
        # Generate a key + sign in one shot.
        r1 = runner.invoke(
            app,
            ["sign", str(adir), "--backend", "ed25519",
             "--generate-key", "signing.pem"],
        )
        assert r1.exit_code == 0, r1.output
        assert (tmp_path / "signing.pem").exists()
        # Verify passes on the clean adapter.
        r2 = runner.invoke(app, ["verify", str(adir)])
        assert r2.exit_code == 0, r2.output

    def test_sigstore_backend_exits_nonzero(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.commands.adapters import app

        monkeypatch.chdir(tmp_path)
        adir = _write_fake_adapter(tmp_path)
        r = CliRunner().invoke(app, ["sign", str(adir), "--backend", "sigstore"])
        # L5 — pin to the exact handled exit code (NotImplementedError -> 2),
        # so a regression changing the failure mode is distinguishable.
        assert r.exit_code == 2, r.output
        assert "infra-blocked" in r.output.lower() or "sigstore" in r.output.lower()


# ---------------------------------------------------------------------------
# TDD-review follow-ups — negative / boundary / fail-closed branch coverage
# ---------------------------------------------------------------------------
class TestReviewFollowups:
    def _key(self, tmp_path, name="priv.pem"):
        from soup_cli.utils.signing import generate_ed25519_private_pem

        kp = tmp_path / name
        kp.write_text(generate_ed25519_private_pem(), encoding="utf-8")
        return str(kp)

    # H1 — attest.verify_attestation wrapper negative paths
    def test_verify_attestation_wrong_key_false(self, tmp_path):
        from soup_cli.utils.attest import sign_attestation, verify_attestation
        from soup_cli.utils.signing import (
            generate_ed25519_private_pem,
            private_key_from_pem,
            public_key_pem,
        )

        sig = sign_attestation(b"payload", backend="ed25519", key_path=self._key(tmp_path))
        other_pub = public_key_pem(private_key_from_pem(generate_ed25519_private_pem()))
        assert verify_attestation(b"payload", sig["signature"], other_pub) is False

    def test_verify_attestation_bad_hex_false(self, tmp_path):
        from soup_cli.utils.attest import sign_attestation, verify_attestation

        sig = sign_attestation(b"x", backend="ed25519", key_path=self._key(tmp_path))
        assert verify_attestation(b"x", "zz", sig["public_key"]) is False

    def test_verify_attestation_non_bytes_typeerror(self):
        from soup_cli.utils.attest import verify_attestation

        with pytest.raises(TypeError):
            verify_attestation("not bytes", "00", "pem")  # type: ignore[arg-type]

    # H2 — attest verify CLI fail-closed branches
    def _emit(self, tmp_path, runner, app):
        key = self._key(tmp_path)
        r = runner.invoke(
            app,
            ["emit", "--stage", "train", "--subject", "m", "--sha", "a" * 64,
             "--sign", "ed25519", "--key", key, "-o", "stmt.json"],
        )
        assert r.exit_code == 0, r.output
        return key

    def test_attest_verify_non_ed25519_sidecar_exit3(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.commands.attest import app

        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        self._emit(tmp_path, runner, app)
        (tmp_path / "stmt.json.sig").write_text(
            json.dumps({"backend": "unsigned", "signature": "", "public_key": ""}),
            encoding="utf-8",
        )
        r = runner.invoke(app, ["verify", "stmt.json", "--signature", "stmt.json.sig"])
        assert r.exit_code == 3, r.output
        assert "not ed25519" in r.output.lower()

    def test_attest_verify_no_pubkey_exit3(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.commands.attest import app

        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        self._emit(tmp_path, runner, app)
        sig = json.loads((tmp_path / "stmt.json.sig").read_text())
        sig["public_key"] = ""
        (tmp_path / "stmt.json.sig").write_text(json.dumps(sig), encoding="utf-8")
        r = runner.invoke(app, ["verify", "stmt.json", "--signature", "stmt.json.sig"])
        assert r.exit_code == 3, r.output
        assert "no public key" in r.output.lower()

    def test_attest_verify_untrusted_key_exit3(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.commands.attest import app
        from soup_cli.utils.signing import (
            generate_ed25519_private_pem,
            private_key_from_pem,
            public_key_pem,
        )

        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        self._emit(tmp_path, runner, app)
        other = tmp_path / "other.pub"
        other.write_text(
            public_key_pem(private_key_from_pem(generate_ed25519_private_pem())),
            encoding="utf-8",
        )
        r = runner.invoke(
            app, ["verify", "stmt.json", "--signature", "stmt.json.sig",
                  "--public-key", str(other)]
        )
        assert r.exit_code == 3, r.output
        assert "untrusted key" in r.output.lower()

    # H3 — verify_adapter strict raise paths
    def test_verify_adapter_strict_raises_on_tamper(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.adapter_sign import sign_adapter, verify_adapter

        adir = _write_fake_adapter(tmp_path)
        sign_adapter(str(adir), backend="ed25519", key_path=self._key(tmp_path))
        (adir / "adapter_model.safetensors").write_bytes(b"different")
        with pytest.raises(ValueError, match="verification failed|signature"):
            verify_adapter(str(adir), strict=True)

    def test_verify_adapter_strict_raises_on_unsigned(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.adapter_sign import verify_adapter

        adir = _write_fake_adapter(tmp_path)
        with pytest.raises(ValueError, match="not signed"):
            verify_adapter(str(adir), strict=True)

    # M1 — empty embedded public key finding
    def test_verify_adapter_empty_pubkey_finding(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.adapter_sign import sign_adapter, verify_adapter

        adir = _write_fake_adapter(tmp_path)
        sign_adapter(str(adir), backend="ed25519", key_path=self._key(tmp_path))
        sig_path = adir / ".soup-signature.json"
        payload = json.loads(sig_path.read_text())
        payload["public_key"] = ""
        sig_path.write_text(json.dumps(payload), encoding="utf-8")
        report = verify_adapter(str(adir))
        assert report.valid is False
        assert any("no public key recorded" in f for f in report.findings)

    # M2 — trusted public key unreadable
    def test_verify_adapter_trusted_unreadable(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.adapter_sign import sign_adapter, verify_adapter

        adir = _write_fake_adapter(tmp_path)
        sign_adapter(str(adir), backend="ed25519", key_path=self._key(tmp_path))
        report = verify_adapter(
            str(adir), trusted_public_key=str(tmp_path / "missing.pub")
        )
        assert report.valid is False
        assert any("unreadable" in f for f in report.findings)

    # M3 — unknown backend rejection
    def test_sign_attestation_unknown_backend(self):
        from soup_cli.utils.attest import sign_attestation

        with pytest.raises(ValueError, match="unknown signature backend"):
            sign_attestation(b"x", backend="bogus")

    def test_sign_adapter_unknown_backend(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.adapter_sign import sign_adapter

        adir = _write_fake_adapter(tmp_path)
        with pytest.raises(ValueError, match="unknown backend"):
            sign_adapter(str(adir), backend="bogus")

    def test_sign_adapter_non_string_backend(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.adapter_sign import sign_adapter

        adir = _write_fake_adapter(tmp_path)
        with pytest.raises(TypeError):
            sign_adapter(str(adir), backend=123)  # type: ignore[arg-type]

    # M4 — sign/verify payload type boundaries
    def test_sign_payload_accepts_bytearray(self, tmp_path):
        from soup_cli.utils.signing import (
            generate_ed25519_private_pem,
            private_key_from_pem,
            sign_payload,
        )

        key = private_key_from_pem(generate_ed25519_private_pem())
        sig = sign_payload(key, bytearray(b"data"))
        assert isinstance(sig, str) and len(sig) == 128

    def test_verify_payload_non_str_signature_typeerror(self, tmp_path):
        from soup_cli.utils.signing import (
            generate_ed25519_private_pem,
            private_key_from_pem,
            public_key_pem,
            verify_payload,
        )

        pub = public_key_pem(private_key_from_pem(generate_ed25519_private_pem()))
        with pytest.raises(TypeError):
            verify_payload(pub, b"x", 123)  # type: ignore[arg-type]

    # M5 — verify_namespace backward-created_at jump + bool override
    def test_verify_namespace_backward_jump_refused(self, tmp_path):
        from soup_cli.utils.namespace_pin import (
            NamespacePinStore,
            record_repo_first_seen,
            verify_namespace,
        )

        db = str(tmp_path / "pins.db")
        with NamespacePinStore(db) as store:
            record_repo_first_seen(
                store, repo_id="o/r", author="alice",
                created_at="2024-06-01T00:00:00+00:00",
            )
            report = verify_namespace(
                store, repo_id="o/r", current_author="alice",
                current_created_at="2024-01-01T00:00:00+00:00",  # earlier!
            )
        assert report.ok is False
        assert "backward" in report.reason.lower()

    def test_verify_namespace_bool_override_typeerror(self, tmp_path):
        from soup_cli.utils.namespace_pin import (
            NamespacePinStore,
            record_repo_first_seen,
            verify_namespace,
        )

        db = str(tmp_path / "pins.db")
        with NamespacePinStore(db) as store:
            record_repo_first_seen(
                store, repo_id="o/r", author="alice",
                created_at="2024-01-01T00:00:00+00:00",
            )
            with pytest.raises(TypeError):
                verify_namespace(
                    store, repo_id="o/r", current_author="mallory",
                    current_created_at="2025-01-01T00:00:00+00:00",
                    allow_namespace_shift=True,  # type: ignore[arg-type]
                )

    # M6 — NamespacePinStore constructor guards
    def test_pin_store_rejects_outside_cwd(self):
        from soup_cli.utils.namespace_pin import NamespacePinStore

        with pytest.raises(ValueError, match="must stay under"):
            NamespacePinStore("/etc/soup_evil_pin.db")

    def test_pin_store_rejects_null_byte(self):
        from soup_cli.utils.namespace_pin import NamespacePinStore

        with pytest.raises(ValueError, match="null"):
            NamespacePinStore("a\x00b.db")

    @POSIX_ONLY
    def test_pin_store_rejects_symlink_db(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.namespace_pin import NamespacePinStore

        real = tmp_path / "real.db"
        real.write_text("", encoding="utf-8")
        link = tmp_path / "link.db"
        os.symlink(real, link)
        with pytest.raises(ValueError, match="symlink"):
            NamespacePinStore(str(link))

    # M7 — license extraction precedence
    def test_license_config_json_over_readme(self, tmp_path):
        from soup_cli.utils.license_matrix import extract_license_from_adapter

        adir = tmp_path / "a"
        adir.mkdir()
        (adir / "config.json").write_text(
            json.dumps({"license": "mit"}), encoding="utf-8"
        )
        (adir / "README.md").write_text(
            "---\nlicense: apache-2.0\n---\n", encoding="utf-8"
        )
        assert extract_license_from_adapter(str(adir)) == "mit"

    def test_license_adapter_config_over_config_json(self, tmp_path):
        from soup_cli.utils.license_matrix import extract_license_from_adapter

        adir = tmp_path / "a"
        adir.mkdir()
        (adir / "adapter_config.json").write_text(
            json.dumps({"license": "mit"}), encoding="utf-8"
        )
        (adir / "config.json").write_text(
            json.dumps({"license": "apache-2.0"}), encoding="utf-8"
        )
        assert extract_license_from_adapter(str(adir)) == "mit"

    # M8 — license-override-without-auto-detect advisory branch
    def test_merge_override_without_autodetect_advisory(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.commands.adapters import app

        monkeypatch.setattr(
            "soup_cli.utils.adapter_scan.scan_adapter", _fake_scan("OK")
        )
        monkeypatch.setattr(
            "soup_cli.utils.adapter_merge.merge_adapters",
            lambda *a, **k: _fake_merge_report(),
        )
        monkeypatch.chdir(tmp_path)
        a = _make_adapter(tmp_path, "a", "apache-2.0")
        b = _make_adapter(tmp_path, "b")  # undetermined
        r = CliRunner().invoke(
            app,
            ["merge", a, b, "-o", "out",
             "--license-override", "no gate but cleared anyway 2026"],
        )
        assert r.exit_code == 0, r.output
        assert "no conflict gate triggered" in r.output.lower()

    # L1 — cryptography-unavailable raise
    def test_require_cryptography_raises_when_unavailable(self, monkeypatch):
        from soup_cli.utils import signing

        monkeypatch.setattr(signing, "is_signing_available", lambda: False)
        with pytest.raises(ValueError, match="soup-cli\\[sign\\]"):
            signing.private_key_from_pem("anything")

    # L2 — generate-key null byte
    def test_generate_key_null_byte(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("SOUP_SIGNING_KEY", raising=False)
        from soup_cli.utils.adapter_sign import sign_adapter

        adir = _write_fake_adapter(tmp_path)
        with pytest.raises(ValueError, match="null"):
            sign_adapter(str(adir), backend="ed25519", generate_key_path="a\x00b")

    # L3 — check_license_compat input guards + CLI count mismatch
    def test_check_license_compat_empty_and_nonlist(self):
        from soup_cli.utils.license_matrix import check_license_compat

        with pytest.raises(ValueError):
            check_license_compat([])
        with pytest.raises(TypeError):
            check_license_compat("apache-2.0")  # type: ignore[arg-type]

    def test_merge_license_count_mismatch(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.commands.adapters import app

        monkeypatch.chdir(tmp_path)
        a = _make_adapter(tmp_path, "a")
        b = _make_adapter(tmp_path, "b")
        r = CliRunner().invoke(
            app,
            ["merge", a, b, "-o", "out", "--allow-unscanned",
             "--license", "apache-2.0"],  # one license, two adapters
        )
        assert r.exit_code == 2, r.output
        assert "must match" in r.output.lower()

    # L4 — _load_signature symlink rejection
    @POSIX_ONLY
    def test_load_signature_symlink_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.adapter_sign import sign_adapter, verify_adapter

        adir = _write_fake_adapter(tmp_path)
        sign_adapter(str(adir), backend="unsigned")
        sig = adir / ".soup-signature.json"
        real = tmp_path / "real-sig.json"
        os.replace(sig, real)
        os.symlink(real, sig)
        with pytest.raises(ValueError, match="symlink"):
            verify_adapter(str(adir))

    # L6 — audit row populates host/operator
    def test_record_license_override_populates_identity(self, tmp_path):
        from soup_cli.utils.audit_log import read_audit_tail
        from soup_cli.utils.license_matrix import record_license_override

        log = tmp_path / "audit.jsonl"
        record_license_override(
            ["apache-2.0", "gpl-3.0"], "cleared by legal 2026-06-01",
            path=str(log),
        )
        row = read_audit_tail(str(log))[0]
        assert row["host_id"]
        assert row["operator_id"]

    def test_attest_emit_help_lists_key(self):
        from typer.testing import CliRunner

        from soup_cli.commands.attest import app

        out = _strip_ansi(CliRunner().invoke(app, ["emit", "--help"]).output)
        assert "--key" in out
        assert "ed25519" in out

    def test_attest_verify_end_to_end(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.commands.attest import app
        from soup_cli.utils.signing import generate_ed25519_private_pem

        monkeypatch.chdir(tmp_path)
        key = tmp_path / "k.pem"
        key.write_text(generate_ed25519_private_pem(), encoding="utf-8")
        runner = CliRunner()
        sha = "a" * 64
        emit = runner.invoke(
            app,
            ["emit", "--stage", "train", "--subject", "model",
             "--sha", sha, "--sign", "ed25519", "--key", str(key),
             "-o", "stmt.json"],
        )
        assert emit.exit_code == 0, emit.output
        assert (tmp_path / "stmt.json").exists()
        assert (tmp_path / "stmt.json.sig").exists()
        # Verify passes.
        ok = runner.invoke(
            app, ["verify", "stmt.json", "--signature", "stmt.json.sig"]
        )
        assert ok.exit_code == 0, ok.output
        # Tamper the statement -> verify fails (exit 3).
        (tmp_path / "stmt.json").write_text("tampered", encoding="utf-8")
        bad = runner.invoke(
            app, ["verify", "stmt.json", "--signature", "stmt.json.sig"]
        )
        assert bad.exit_code == 3, bad.output


# ---------------------------------------------------------------------------
# Security-review fix regressions (M1 / L1 / L3)
# ---------------------------------------------------------------------------
class TestSecurityFixes:
    def test_generate_key_refuses_existing_path(self, tmp_path, monkeypatch):
        # L1 — --generate-key must refuse a pre-existing target (closes the
        # Windows TOCTOU window + avoids clobbering an existing key).
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.adapter_sign import sign_adapter

        adir = _write_fake_adapter(tmp_path)
        existing = tmp_path / "exists.pem"
        existing.write_text("placeholder", encoding="utf-8")
        with pytest.raises(ValueError, match="overwrite|existing"):
            sign_adapter(
                str(adir), backend="ed25519", generate_key_path=str(existing)
            )

    def test_read_public_key_file_size_cap_and_symlink(self, tmp_path):
        # M1 — the shared public-key reader rejects oversize + (POSIX) symlinks.
        from soup_cli.utils.signing import read_public_key_file

        big = tmp_path / "big.pub"
        big.write_text("x" * (70 * 1024), encoding="utf-8")  # > 64 KiB cap
        with pytest.raises(ValueError, match="exceeds"):
            read_public_key_file(str(big))
        with pytest.raises(ValueError, match="null"):
            read_public_key_file("a\x00b")

    @POSIX_ONLY
    def test_read_public_key_file_symlink_rejected(self, tmp_path):
        from soup_cli.utils.signing import read_public_key_file

        real = tmp_path / "real.pub"
        real.write_text("k", encoding="utf-8")
        link = tmp_path / "link.pub"
        os.symlink(real, link)
        with pytest.raises(ValueError, match="symlink"):
            read_public_key_file(str(link))

    def test_download_repo_fails_open_on_store_error(self, tmp_path, monkeypatch):
        # L3 — a sqlite/infra error opening the pin store must NOT crash the
        # download (best-effort gate fails open). The intentional refusal is
        # raised outside the try, so it is unaffected.
        import sqlite3
        from unittest.mock import patch

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("SOUP_NAMESPACE_PIN_DB", str(tmp_path / "pins.db"))
        from soup_cli.utils import hubs

        def boom(*a, **k):
            raise sqlite3.OperationalError("database is locked")

        # _run_namespace_check imports NamespacePinStore from namespace_pin at
        # call time, so patch it there (not on the hubs module).
        with patch(
            "soup_cli.utils.namespace_pin.NamespacePinStore", boom
        ), patch("huggingface_hub.snapshot_download", return_value="/snap") as m:
            result = hubs.download_repo(
                "hf", "alice/model", local_dir="./s",
                _metadata_fn=lambda rid: ("alice", "2024-01-01T00:00:00+00:00"),
            )
        assert result == "/snap"
        m.assert_called_once()
