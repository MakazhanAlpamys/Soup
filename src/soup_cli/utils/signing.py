"""Ed25519 detached signing primitives (v0.71.2 #179 / #185).

Shared by ``soup attest`` (#179) and ``soup adapters sign`` (#185). Pure
``cryptography`` — no network, fully offline, validatable on any box. This is
the half of the "Sigstore + ed25519" issues that the maintainer can honestly
smoke-test on a Windows/offline machine; the Sigstore keyless path (needs an
OIDC identity provider + Fulcio/Rekor network) stays infra-blocked and is NOT
implemented here.

``cryptography`` ships in the ``[sign]`` extra (``pip install soup-cli[sign]``)
and is imported lazily so the core CLI stays light (v0.71.0 deps-split policy).

Threat model: the detached signature provides *authentication* — proof the
signer holds the private key — on top of the Merkle-root tamper-evidence the
adapter manifest already gives. For verification to mean anything, the verifier
must compare against a trusted public key supplied out of band; the embedded
public key only gives a self-consistency check (the recorded content was signed
by *some* key whose public half is recorded).

Public surface:
- ``is_signing_available()`` -> bool (cryptography importable).
- ``generate_ed25519_private_pem()`` -> PEM str (helper + ``--generate-key``).
- ``private_key_from_pem(pem)`` -> private key object (ed25519-only).
- ``public_key_pem(private_key)`` -> PEM str.
- ``sign_payload(private_key, payload: bytes)`` -> hex signature str.
- ``verify_payload(public_key_pem, payload: bytes, signature_hex)`` -> bool.
- ``load_private_key_file(path)`` -> private key (symlink-rejected, size-capped).
- ``resolve_signing_key(key_path, *, env=None)`` -> private key (``--key`` /
  ``SOUP_SIGNING_KEY``).
"""

from __future__ import annotations

import os
import stat
from typing import TYPE_CHECKING, Any, Mapping, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime import cost
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

# Private keys live OUTSIDE the project tree (secrets are not cwd-contained),
# so we do NOT apply the project's cwd-containment policy to key paths — we
# only symlink-reject + size-cap. A 64 KiB cap is generous for any PEM.
_MAX_KEY_BYTES = 64 * 1024
_SIGNING_KEY_ENV = "SOUP_SIGNING_KEY"


def is_signing_available() -> bool:
    """Return True when the ``cryptography`` ed25519 backend is importable."""
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # noqa: F401
            Ed25519PrivateKey,
        )
    except ImportError:
        return False
    return True


def _require_cryptography() -> None:
    if not is_signing_available():
        raise ValueError(
            "ed25519 signing requires the 'cryptography' package. "
            "Install with: pip install soup-cli[sign]"
        )


def generate_ed25519_private_pem() -> str:
    """Generate a fresh ed25519 private key and return its PKCS8 PEM string."""
    _require_cryptography()
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.generate()
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")


def private_key_from_pem(pem: Any) -> Ed25519PrivateKey:
    """Parse a PEM string/bytes into an ed25519 private key.

    Rejects non-ed25519 keys (RSA / EC / etc.) with a clear ``ValueError`` so
    operators can't accidentally sign with the wrong algorithm. Raises
    ``ValueError`` on un-parseable input.
    """
    _require_cryptography()
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    if isinstance(pem, str):
        data = pem.encode("utf-8")
    elif isinstance(pem, (bytes, bytearray)):
        data = bytes(pem)
    else:
        raise TypeError(f"pem must be str or bytes, got {type(pem).__name__}")
    try:
        key = serialization.load_pem_private_key(data, password=None)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"could not parse PEM private key: {exc}") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError(
            f"key must be ed25519, got {type(key).__name__} "
            "(only ed25519 signing is supported)"
        )
    return key


def public_key_pem(private_key: Ed25519PrivateKey) -> str:
    """Return the PEM-encoded public key for an ed25519 private key."""
    _require_cryptography()
    from cryptography.hazmat.primitives import serialization

    return (
        private_key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("utf-8")
    )


def sign_payload(private_key: Ed25519PrivateKey, payload: bytes) -> str:
    """Sign ``payload`` with an ed25519 private key; return a hex signature."""
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError("payload must be bytes")
    sig = private_key.sign(bytes(payload))
    return sig.hex()


def verify_payload(public_key_pem_str: Any, payload: bytes, signature_hex: Any) -> bool:
    """Verify a detached ed25519 signature.

    Returns ``False`` on any verification failure (bad signature, wrong key,
    malformed hex). Raises ``TypeError`` only on genuinely wrong argument types
    (so callers don't mistake "invalid signature" for "wrong call").
    """
    _require_cryptography()
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError("payload must be bytes")
    if isinstance(public_key_pem_str, str):
        pub_bytes = public_key_pem_str.encode("utf-8")
    elif isinstance(public_key_pem_str, (bytes, bytearray)):
        pub_bytes = bytes(public_key_pem_str)
    else:
        raise TypeError("public_key_pem must be str or bytes")
    if not isinstance(signature_hex, str):
        raise TypeError("signature_hex must be str")
    try:
        sig = bytes.fromhex(signature_hex)
    except ValueError:
        return False
    try:
        pub = serialization.load_pem_public_key(pub_bytes)
    except (ValueError, TypeError):
        return False
    if not isinstance(pub, Ed25519PublicKey):
        return False
    try:
        pub.verify(sig, bytes(payload))
    except InvalidSignature:
        return False
    return True


def load_private_key_file(path: Any) -> Ed25519PrivateKey:
    """Load + parse an ed25519 private key from a PEM file.

    The key is a secret stored OUTSIDE the project tree, so this does NOT apply
    cwd-containment (unlike data/output paths). It DOES reject symlinks
    (TOCTOU defence — refuse to follow a planted link), require a regular file,
    and cap the file size.
    """
    if not isinstance(path, str) or not path:
        raise ValueError("key path must be a non-empty str")
    if "\x00" in path:
        raise ValueError("key path must not contain null bytes")
    try:
        st = os.lstat(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"signing key not found: {path}") from exc
    if stat.S_ISLNK(st.st_mode):
        raise ValueError(f"signing key {os.path.basename(path)!r} must not be a symlink")
    if not stat.S_ISREG(st.st_mode):
        raise ValueError(f"signing key {os.path.basename(path)!r} must be a regular file")
    if st.st_size > _MAX_KEY_BYTES:
        raise ValueError(f"signing key exceeds {_MAX_KEY_BYTES} bytes")
    with open(path, "rb") as fh:
        data = fh.read()
    return private_key_from_pem(data)


def read_public_key_file(path: Any) -> str:
    """Read a trusted public-key PEM file (symlink-rejected, size-capped).

    Public keys are not secrets and may live anywhere the operator points at,
    so this does NOT enforce cwd-containment — but it DOES refuse to follow a
    planted symlink (TOCTOU) and caps the read so a ``/dev/zero`` symlink can't
    OOM the verifier. Shared by ``soup attest verify`` and
    ``adapter_sign.verify_adapter`` for consistent hardening.
    """
    if not isinstance(path, str) or not path:
        raise ValueError("public key path must be a non-empty str")
    if "\x00" in path:
        raise ValueError("public key path must not contain null bytes")
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode):
        raise ValueError(
            f"public key {os.path.basename(path)!r} must not be a symlink"
        )
    if not stat.S_ISREG(st.st_mode):
        raise ValueError(
            f"public key {os.path.basename(path)!r} must be a regular file"
        )
    if st.st_size > _MAX_KEY_BYTES:
        raise ValueError(f"public key exceeds {_MAX_KEY_BYTES} bytes")
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def resolve_signing_key(
    key_path: Optional[str], *, env: Optional[Mapping[str, str]] = None
) -> Ed25519PrivateKey:
    """Resolve the ed25519 private key from ``--key`` or ``SOUP_SIGNING_KEY``.

    Precedence: explicit ``key_path`` wins; else the ``SOUP_SIGNING_KEY`` env
    var; else a clear ``ValueError`` (no silent ephemeral-key generation —
    persisting a freshly-minted private key into the repo would be a footgun).
    """
    if env is None:
        env = os.environ
    if key_path:
        return load_private_key_file(key_path)
    env_path = env.get(_SIGNING_KEY_ENV)
    if env_path:
        return load_private_key_file(env_path)
    raise ValueError(
        "ed25519 signing requires a private key: pass --key <pem> "
        f"or set {_SIGNING_KEY_ENV} (generate one with "
        "`soup adapters sign --backend ed25519 --generate-key`)"
    )
