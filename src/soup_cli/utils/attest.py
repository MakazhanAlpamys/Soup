"""in-toto + SLSA-3 attestation builder (v0.59.0 Part B).

The statement builder itself is pure stdlib. Optional signing is layered on top:
ed25519 uses the shared `cryptography` helper, while Sigstore uses the lazy
sigstore-python 4.x wrapper for keyless OIDC/Fulcio/Rekor bundles. The statement
wire format remains independent of either signing backend.

Schema shapes:
- ``_type``: ``https://in-toto.io/Statement/v1``
- ``predicateType``: ``https://slsa.dev/provenance/v1``
- ``subject``: ``[{name, digest: {sha256: ...}}]``
- ``predicate``: SLSA-3 provenance v1 (``buildDefinition`` + ``runDetails``).

Stage allowlist (mirrors the v0.26.0 Soup-Can lifecycle):
``extract`` / ``train`` / ``eval`` / ``export`` / ``publish``.
"""

from __future__ import annotations

import enum
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from soup_cli.utils.paths import atomic_write_text

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_STAGES = frozenset({"extract", "train", "eval", "export", "publish"})
_MAX_BUILDER_ID = 256
_MAX_NAME = 256


class SignatureBackend(str, enum.Enum):
    """Signing backend selector for unsigned, ed25519, and Sigstore."""

    UNSIGNED = "unsigned"
    ED25519 = "ed25519"
    SIGSTORE = "sigstore"


@dataclass(frozen=True)
class AttestationStatement:
    """Per-stage attestation input."""

    stage: str
    subject_name: str
    subject_sha256: str
    builder_id: str
    invocation: Mapping[str, Any]
    materials: Tuple[Mapping[str, Any], ...]
    created_at: str

    def __post_init__(self) -> None:
        if not isinstance(self.stage, str) or "\x00" in self.stage:
            raise ValueError("stage must be a non-null-byte str")
        if self.stage not in _STAGES:
            raise ValueError(
                f"stage must be one of {sorted(_STAGES)}, got {self.stage!r}"
            )
        if not isinstance(self.subject_name, str) or not self.subject_name:
            raise ValueError("subject_name must be a non-empty str")
        if "\x00" in self.subject_name or len(self.subject_name) > _MAX_NAME:
            raise ValueError("subject_name invalid (null byte or > 256 chars)")
        if not isinstance(self.subject_sha256, str) or not _SHA256_RE.match(self.subject_sha256):
            raise ValueError("subject_sha256 must be 64 hex chars")
        if not isinstance(self.builder_id, str) or not self.builder_id:
            raise ValueError("builder_id must be a non-empty str")
        if "\x00" in self.builder_id or len(self.builder_id) > _MAX_BUILDER_ID:
            raise ValueError("builder_id invalid (null byte or > 256 chars)")
        if not isinstance(self.invocation, Mapping):
            raise ValueError("invocation must be a mapping")
        if not isinstance(self.materials, tuple):
            raise ValueError("materials must be a tuple")
        for mat in self.materials:
            if not isinstance(mat, Mapping):
                raise ValueError("materials entries must be mappings")
        if not isinstance(self.created_at, str) or not self.created_at:
            raise ValueError("created_at must be a non-empty str")


_MAX_INVOCATION_ID_LEN = 256


def build_slsa_provenance(s: AttestationStatement) -> dict[str, Any]:
    """Render the SLSA-3 provenance v1 predicate body."""
    if not isinstance(s, AttestationStatement):
        raise TypeError(f"s must be AttestationStatement, got {type(s).__name__}")
    materials_resolved: list[dict] = []
    for mat in s.materials:
        uri = str(mat.get("uri", ""))
        digest = str(mat.get("digest", ""))
        item: dict[str, Any] = {"uri": uri}
        if _SHA256_RE.match(digest):
            item["digest"] = {"sha256": digest}
        materials_resolved.append(item)
    invocation_id = str(s.invocation.get("invocation_id", ""))[:_MAX_INVOCATION_ID_LEN]
    started_on = str(s.invocation.get("started_on", s.created_at))[:64]
    finished_on = str(s.invocation.get("finished_on", s.created_at))[:64]
    return {
        "buildDefinition": {
            "buildType": "https://soup.local/build/v1",
            "externalParameters": {"stage": s.stage},
            "internalParameters": {},
            "resolvedDependencies": materials_resolved,
        },
        "runDetails": {
            "builder": {"id": s.builder_id},
            "metadata": {
                "invocationId": invocation_id,
                "startedOn": started_on,
                "finishedOn": finished_on,
            },
            "byproducts": [],
        },
    }


def build_in_toto_statement(s: AttestationStatement) -> dict[str, Any]:
    """Wrap the SLSA provenance in an in-toto v1 Statement."""
    return {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": [
            {
                "name": s.subject_name,
                "digest": {"sha256": s.subject_sha256},
            }
        ],
        "predicateType": "https://slsa.dev/provenance/v1",
        "predicate": build_slsa_provenance(s),
    }


def render_attestation(s: AttestationStatement) -> str:
    return json.dumps(build_in_toto_statement(s), indent=2, sort_keys=True)


def write_attestation(s: AttestationStatement, output_path: str) -> str:
    """Atomic write of the in-toto Statement to ``output_path`` (cwd-contained)."""
    text = render_attestation(s)
    return atomic_write_text(
        text, output_path, prefix=".attest.", suffix=".json.tmp",
    )


def sign_attestation(
    payload: bytes,
    *,
    backend: SignatureBackend | str = SignatureBackend.UNSIGNED,
    key_path: str | None = None,
    sigstore_interactive: bool = False,
) -> dict:
    """Sign a payload (in-toto JSON bytes) with the chosen backend.

    ``ed25519`` (v0.71.2 #179) produces a real detached signature over
    ``payload`` using a private key resolved from ``key_path`` or the
    ``SOUP_SIGNING_KEY`` env var. ``sigstore`` produces a keyless
    Fulcio/Rekor bundle through the shared Sigstore 4.x wrapper.

    Args:
        payload: in-toto Statement bytes (typically ``render_attestation(...).encode()``).
        backend: ``"unsigned"``, ``"ed25519"``, or ``"sigstore"``.
        key_path: ed25519 private-key PEM path (``ed25519`` backend only).
        sigstore_interactive: explicitly permit browser OIDC for Sigstore.
            False by default so headless runners fail instead of hanging.

    Returns:
        - ``{"signature": "", "backend": "unsigned"}`` for the unsigned path
          (the empty signature lets verifiers refuse in strict mode).
        - ``{"signature": <hex>, "backend": "ed25519", "public_key": <pem>}``
          for the ed25519 path.
        - ``{"signature": "", "backend": "sigstore", "sigstore_bundle": <json>}``
          for the keyless Sigstore path.
    """
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError("payload must be bytes")
    if not isinstance(sigstore_interactive, bool):
        raise TypeError("sigstore_interactive must be bool")
    if isinstance(backend, str):
        try:
            backend = SignatureBackend(backend.lower())
        except ValueError as exc:
            raise ValueError(
                f"unknown signature backend: {backend!r} "
                f"(use one of {[b.value for b in SignatureBackend]})"
            ) from exc
    if sigstore_interactive and backend != SignatureBackend.SIGSTORE:
        raise ValueError("--interactive-oidc requires --sign sigstore")
    if backend == SignatureBackend.UNSIGNED:
        return {"signature": "", "backend": "unsigned"}
    if backend == SignatureBackend.ED25519:
        from soup_cli.utils.signing import (
            public_key_pem,
            resolve_signing_key,
            sign_payload,
        )

        private_key = resolve_signing_key(key_path)
        return {
            "signature": sign_payload(private_key, bytes(payload)),
            "backend": "ed25519",
            "public_key": public_key_pem(private_key),
        }
    if backend == SignatureBackend.SIGSTORE:
        if key_path is not None:
            raise ValueError("key_path applies only to the ed25519 backend")
        from soup_cli.utils.sigstore_signing import sign_payload_sigstore

        bundle = sign_payload_sigstore(
            bytes(payload),
            interactive=sigstore_interactive,
        )
        if not isinstance(bundle, str) or not bundle.strip():
            raise RuntimeError("Sigstore signing returned an empty bundle")
        return {
            "signature": "",
            "backend": "sigstore",
            "sigstore_bundle": bundle,
        }
    raise AssertionError(f"unhandled signature backend: {backend!r}")


def verify_sigstore_attestation(
    payload: bytes,
    bundle_json: str,
    *,
    identity: str,
    issuer: str,
) -> bool:
    """Verify a Sigstore attestation bundle against identity + issuer policy."""
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError("payload must be bytes")
    from soup_cli.utils.sigstore_signing import verify_payload_sigstore

    verify_payload_sigstore(
        bytes(payload),
        bundle_json,
        identity=identity,
        issuer=issuer,
    )
    return True


def verify_attestation(
    payload: bytes, signature_hex: str, public_key_pem_str: str
) -> bool:
    """Verify an ed25519 attestation signature over ``payload``.

    Returns ``False`` on any verification failure (bad signature, wrong key,
    malformed hex). Thin wrapper over :func:`soup_cli.utils.signing.verify_payload`.
    """
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError("payload must be bytes")
    from soup_cli.utils.signing import verify_payload

    return verify_payload(public_key_pem_str, bytes(payload), signature_hex)
