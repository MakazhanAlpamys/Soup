"""#185: Sigstore signing must never open an implicit browser on headless runs."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest


def _install_fake_sigstore(monkeypatch, *, credential):
    events = []

    root = ModuleType("sigstore")
    root.__path__ = []

    models = ModuleType("sigstore.models")

    class ClientTrustConfig:
        @classmethod
        def production(cls):
            events.append("trust")
            return SimpleNamespace(
                signing_config=SimpleNamespace(
                    get_oidc_url=lambda: "https://issuer.example"
                )
            )

    models.ClientTrustConfig = ClientTrustConfig

    oidc = ModuleType("sigstore.oidc")

    class IdentityToken:
        def __init__(self, raw):
            self.raw = raw
            events.append(("identity-token", raw))

    class Issuer:
        def __init__(self, url):
            events.append(("issuer", url))

        def identity_token(self):
            events.append("interactive-token")
            return "browser-token"

    def detect_credential():
        events.append("detect")
        return credential

    oidc.IdentityToken = IdentityToken
    oidc.Issuer = Issuer
    oidc.detect_credential = detect_credential

    sign = ModuleType("sigstore.sign")

    class Bundle:
        def to_json(self):
            return '{"bundle":"ok"}'

    class Signer:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def sign_artifact(self, payload):
            events.append(("sign", payload))
            return Bundle()

    class SigningContext:
        @classmethod
        def from_trust_config(cls, _trust):
            events.append("context")
            return cls()

        def signer(self, token, cache=False):
            events.append(("signer", token, cache))
            return Signer()

    sign.SigningContext = SigningContext

    monkeypatch.setitem(sys.modules, "sigstore", root)
    monkeypatch.setitem(sys.modules, "sigstore.models", models)
    monkeypatch.setitem(sys.modules, "sigstore.oidc", oidc)
    monkeypatch.setitem(sys.modules, "sigstore.sign", sign)
    return events


def test_default_refuses_before_browser_issuer_when_no_ambient_oidc(monkeypatch):
    from soup_cli.utils.sigstore_signing import sign_payload_sigstore

    events = _install_fake_sigstore(monkeypatch, credential=None)

    with pytest.raises(ValueError, match="--interactive-oidc"):
        sign_payload_sigstore(b"payload")

    assert "detect" in events
    assert not any(
        isinstance(event, tuple) and event[0] == "issuer" for event in events
    )


def test_interactive_oidc_is_explicit_opt_in(monkeypatch):
    from soup_cli.utils.sigstore_signing import sign_payload_sigstore

    events = _install_fake_sigstore(monkeypatch, credential=None)
    bundle = sign_payload_sigstore(b"payload", interactive=True)

    assert bundle == '{"bundle":"ok"}'
    assert ("issuer", "https://issuer.example") in events
    assert "interactive-token" in events
    assert ("sign", b"payload") in events


def test_ambient_oidc_never_needs_browser_opt_in(monkeypatch):
    from soup_cli.utils.sigstore_signing import sign_payload_sigstore

    events = _install_fake_sigstore(monkeypatch, credential="ambient-token")
    bundle = sign_payload_sigstore(b"payload")

    assert bundle == '{"bundle":"ok"}'
    assert ("identity-token", "ambient-token") in events
    assert not any(
        isinstance(event, tuple) and event[0] == "issuer" for event in events
    )


def _install_fake_verify_sigstore(monkeypatch):
    seen = {}

    root = ModuleType("sigstore")
    root.__path__ = []

    models = ModuleType("sigstore.models")

    class Bundle:
        @classmethod
        def from_json(cls, value):
            seen["bundle_json"] = value
            return "bundle-object"

    models.Bundle = Bundle

    verify = ModuleType("sigstore.verify")

    class FakeVerifier:
        def verify_artifact(self, payload, bundle, policy):
            seen.update(payload=payload, bundle=bundle, policy=policy)

    class Verifier:
        @classmethod
        def production(cls):
            seen["verifier_production"] = True
            return FakeVerifier()

    verify.Verifier = Verifier

    policy_mod = ModuleType("sigstore.verify.policy")

    class Identity:
        def __init__(self, *, identity, issuer):
            seen["identity"] = identity
            seen["issuer"] = issuer
            self.identity = identity
            self.issuer = issuer

    policy_mod.Identity = Identity

    monkeypatch.setitem(sys.modules, "sigstore", root)
    monkeypatch.setitem(sys.modules, "sigstore.models", models)
    monkeypatch.setitem(sys.modules, "sigstore.verify", verify)
    monkeypatch.setitem(sys.modules, "sigstore.verify.policy", policy_mod)
    return seen


def test_verify_policy_receives_identity_and_issuer(monkeypatch):
    from soup_cli.utils.sigstore_signing import verify_payload_sigstore

    seen = _install_fake_verify_sigstore(monkeypatch)
    verify_payload_sigstore(
        b"payload",
        '{"bundle":"ok"}',
        identity="trusted@example.com",
        issuer="https://issuer.example",
    )

    assert seen["identity"] == "trusted@example.com"
    assert seen["issuer"] == "https://issuer.example"
    assert seen["payload"] == b"payload"
    assert seen["bundle"] == "bundle-object"


@pytest.mark.parametrize("issuer", ["", "   "])
def test_verify_helper_rejects_empty_issuer_before_policy(monkeypatch, issuer):
    from soup_cli.utils.sigstore_signing import verify_payload_sigstore

    seen = _install_fake_verify_sigstore(monkeypatch)
    with pytest.raises(ValueError, match="non-empty trusted OIDC issuer"):
        verify_payload_sigstore(
            b"payload",
            '{"bundle":"ok"}',
            identity="trusted@example.com",
            issuer=issuer,
        )

    assert "issuer" not in seen


def test_sigstore_internal_value_error_is_runtime_failure(monkeypatch):
    from soup_cli.utils.sigstore_signing import sign_payload_sigstore

    _install_fake_sigstore(monkeypatch, credential="ambient-token")

    class BrokenContext:
        @classmethod
        def from_trust_config(cls, _trust):
            raise ValueError("internal validation failed")

    sys.modules["sigstore.sign"].SigningContext = BrokenContext

    with pytest.raises(RuntimeError, match="Sigstore keyless signing failed"):
        sign_payload_sigstore(b"payload")


@pytest.mark.parametrize("identity", ["", "   "])
def test_verify_helper_rejects_empty_identity_before_policy(monkeypatch, identity):
    from soup_cli.utils.sigstore_signing import verify_payload_sigstore

    seen = _install_fake_verify_sigstore(monkeypatch)
    with pytest.raises(ValueError, match="non-empty trusted identity"):
        verify_payload_sigstore(
            b"payload",
            '{"bundle":"ok"}',
            identity=identity,
            issuer="https://issuer.example",
        )

    assert "identity" not in seen


def test_verify_helper_wraps_sigstore_verification_error(monkeypatch):
    from soup_cli.utils.sigstore_signing import verify_payload_sigstore

    _install_fake_verify_sigstore(monkeypatch)

    class VerificationError(Exception):
        pass

    class RejectingVerifier:
        @classmethod
        def production(cls):
            return cls()

        def verify_artifact(self, payload, bundle, policy):
            raise VerificationError(
                "Certificate's SANs do not match trusted@example.com"
            )

    monkeypatch.setattr(
        sys.modules["sigstore.verify"], "Verifier", RejectingVerifier
    )

    with pytest.raises(
        ValueError, match="Sigstore verification failed: Certificate's SANs"
    ):
        verify_payload_sigstore(
            b"payload",
            '{"bundle":"ok"}',
            identity="trusted@example.com",
            issuer="https://issuer.example",
        )
