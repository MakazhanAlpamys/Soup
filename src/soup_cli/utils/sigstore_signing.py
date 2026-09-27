"""Lazy wrapper around the public sigstore-python 4.x API.

Sigstore is imported only inside calls, so normal CLI startup stays light.
"""

from __future__ import annotations


def _missing_sigstore(exc: Exception) -> ValueError:
    return ValueError(
        "Sigstore support requires the optional Sigstore extra: "
        "pip install soup-cli[sigstore] (sigstore>=4.4,<5)."
    )


def sign_payload_sigstore(payload: bytes, *, interactive: bool = False) -> str:
    """Keylessly sign opaque bytes and return a complete bundle JSON."""
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if not isinstance(interactive, bool):
        raise TypeError("interactive must be bool")
    try:
        from sigstore.models import ClientTrustConfig
        from sigstore.oidc import IdentityToken, Issuer, detect_credential
        from sigstore.sign import SigningContext
    except ImportError as exc:
        raise _missing_sigstore(exc) from exc

    # Keep Soup's deliberate "no ambient credential" refusal distinct from
    # sigstore-internal ValueErrors/ValidationErrors. The latter are runtime
    # signing failures, not CLI-usage errors.
    try:
        trust = ClientTrustConfig.production()
        raw_token = detect_credential()
    except Exception as exc:
        raise RuntimeError(f"Sigstore keyless signing failed: {exc}") from exc

    if raw_token is None and not interactive:
        raise ValueError(
            "No ambient Sigstore OIDC credential was detected. "
            "Use an OIDC-enabled environment, or opt into the browser flow "
            "explicitly with --interactive-oidc."
        )

    try:
        if raw_token is None:
            oidc_issuer = Issuer(trust.signing_config.get_oidc_url())
            token = oidc_issuer.identity_token()
        else:
            token = IdentityToken(raw_token)
        context = SigningContext.from_trust_config(trust)
        with context.signer(token, cache=False) as signer:
            bundle = signer.sign_artifact(payload)
        return bundle.to_json()
    except Exception as exc:
        raise RuntimeError(f"Sigstore keyless signing failed: {exc}") from exc


def verify_payload_sigstore(
    payload: bytes,
    bundle_json: str,
    *,
    identity: str,
    issuer: str,
) -> None:
    """Verify a bundle and require an out-of-band certificate identity."""
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if not isinstance(bundle_json, str) or not bundle_json:
        raise ValueError("sigstore bundle must be a non-empty JSON string")
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError("sigstore verification requires a non-empty trusted identity")
    if not isinstance(issuer, str) or not issuer.strip():
        raise ValueError(
            "sigstore verification requires a non-empty trusted OIDC issuer"
        )
    try:
        from sigstore.models import Bundle
        from sigstore.verify import Verifier
        from sigstore.verify.policy import Identity
    except ImportError as exc:
        raise RuntimeError(str(_missing_sigstore(exc))) from exc

    try:
        bundle = Bundle.from_json(bundle_json)
        policy = Identity(identity=identity, issuer=issuer)
    except Exception as exc:
        raise ValueError(f"Sigstore verification failed: {exc}") from exc

    try:
        verifier = Verifier.production()
    except Exception as exc:
        raise RuntimeError(f"Sigstore verifier unavailable: {exc}") from exc

    try:
        verifier.verify_artifact(payload, bundle, policy)
    except Exception as exc:
        raise ValueError(f"Sigstore verification failed: {exc}") from exc
