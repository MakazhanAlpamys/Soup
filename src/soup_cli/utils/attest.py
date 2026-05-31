"""in-toto + SLSA-3 attestation builder (v0.59.0 Part B).

Pure-stdlib. Sigstore + ed25519 live signing is **deferred to v0.59.1**
(mirrors v0.27.0 MII / v0.37.0 multipack / v0.50.0 GRPO Plus stub-then-live
pattern). The schema + atomic write surface ships now so callers can lock
the wire format.

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
    """Signing backend selector. ``sigstore`` + ``ed25519`` deferred to v0.59.1."""

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


def build_slsa_provenance(s: AttestationStatement) -> dict:
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


def build_in_toto_statement(s: AttestationStatement) -> dict:
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
    payload: bytes, *, backend: SignatureBackend | str = SignatureBackend.UNSIGNED,
) -> dict:
    """Sign a payload (in-toto JSON bytes) with the chosen backend.

    Sigstore + ed25519 are **deferred to v0.59.1**. The schema lives now so
    pipelines can be tested; the live signer lands in v0.59.1.

    Args:
        payload: in-toto Statement bytes (typically ``render_attestation(...).encode()``).
        backend: ``"unsigned"`` is the only live backend in v0.59.0; the others
                 raise NotImplementedError.

    Returns:
        ``{"signature": "", "backend": "unsigned"}`` for the unsigned path. The
        signature field is intentionally empty so downstream verifiers can detect
        the missing signature and refuse in strict mode.
    """
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError("payload must be bytes")
    if isinstance(backend, str):
        try:
            backend = SignatureBackend(backend.lower())
        except ValueError as exc:
            raise ValueError(
                f"unknown signature backend: {backend!r} "
                f"(use one of {[b.value for b in SignatureBackend]})"
            ) from exc
    if backend == SignatureBackend.UNSIGNED:
        return {"signature": "", "backend": "unsigned"}
    raise NotImplementedError(
        f"signing backend {backend.value!r} is deferred to v0.59.1"
    )
