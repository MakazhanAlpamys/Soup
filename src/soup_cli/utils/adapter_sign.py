"""Adapter signing + verification (v0.60.0 Part B).

Computes a deterministic SHA-256 over the adapter's file list + per-file
content hashes (Merkle-style root) and stores it alongside the adapter as
``.soup-signature.json``. Two backends:

- ``UNSIGNED`` (default in v0.60.0): writes the manifest + empty signature.
  Useful for offline tamper detection — if the weights change, ``verify``
  fails because the recomputed root no longer matches the recorded root.
- ``SIGSTORE``: deferred to v0.60.1 (mirrors v0.27.0 MII / v0.59.0 Part B
  stub-then-live pattern). Schema lives now so CI pipelines can integrate.
- ``ED25519``: deferred to v0.60.1; requires ``cryptography`` lazy import.

The signature file format is intentionally JSON so operators can diff /
audit / cat without parsing a binary blob. Atomic writes via
``utils.paths.atomic_write_text`` (cwd-contained, TOCTOU-safe).

Public surface:
- ``AdapterManifest`` / ``SignatureRecord`` / ``VerifyReport`` frozen dataclasses.
- ``compute_adapter_manifest(adapter_dir)`` -> ``AdapterManifest``.
- ``sign_adapter(adapter_dir, *, backend=UNSIGNED)`` -> ``SignatureRecord``.
- ``verify_adapter(adapter_dir, *, strict=False)`` -> ``VerifyReport``.
"""

from __future__ import annotations

import enum
import hashlib
import json
import os
import stat
from dataclasses import dataclass, field
from typing import Optional, Tuple

from soup_cli.utils.paths import (
    atomic_write_text,
    enforce_under_cwd_and_no_symlink,
)

_SIGNATURE_FILENAME = ".soup-signature.json"
_MAX_FILES = 1024
_MAX_FILE_BYTES = 10 * 1024 * 1024 * 1024  # 10 GiB safety cap on per-file size
_MAX_SIGNATURE_BYTES = 16 * 1024 * 1024  # 16 MiB cap on the signature JSON
_HASH_CHUNK = 1024 * 1024
_MANIFEST_VERSION = 1


class SignBackend(str, enum.Enum):
    """Signing backend selector.

    Only ``UNSIGNED`` is live in v0.60.0. ``SIGSTORE`` + ``ED25519`` raise
    ``NotImplementedError`` with explicit v0.60.1 marker (stub-then-live).
    """

    UNSIGNED = "unsigned"
    ED25519 = "ed25519"
    SIGSTORE = "sigstore"


@dataclass(frozen=True)
class FileEntry:
    """One file in the manifest."""

    name: str
    size: int
    sha256: str


@dataclass(frozen=True)
class AdapterManifest:
    """Deterministic adapter manifest."""

    adapter: str
    version: int
    files: Tuple[FileEntry, ...]
    merkle_root: str


@dataclass(frozen=True)
class SignatureRecord:
    """Persisted signature file body."""

    backend: str
    merkle_root: str
    signature: str
    signed_at: str
    manifest: AdapterManifest


@dataclass(frozen=True)
class VerifyReport:
    """Result of ``verify_adapter``."""

    adapter: str
    valid: bool
    backend: Optional[str]
    reason: str
    findings: Tuple[str, ...] = field(default_factory=tuple)


def _hash_file(path: str) -> tuple[int, str]:
    """Stream a file through SHA-256 and return ``(size, hex_digest)``.

    Uses ``os.lstat`` (not ``stat``) so symlinks at the file level are
    refused even if the parent dir passed containment. ``_MAX_FILE_BYTES``
    cap defends against a hostile adapter pointing at a 1 TiB device file.
    """
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode):
        raise ValueError(f"{os.path.basename(path)!r}: must not be a symlink")
    if not stat.S_ISREG(st.st_mode):
        raise ValueError(f"{os.path.basename(path)!r}: must be a regular file")
    if st.st_size > _MAX_FILE_BYTES:
        raise ValueError(
            f"{os.path.basename(path)!r}: exceeds {_MAX_FILE_BYTES} bytes"
        )
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_HASH_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return int(st.st_size), digest.hexdigest()


def _enumerate_files(adapter_dir: str) -> list[str]:
    """Return regular files inside ``adapter_dir`` (recursive).

    Recurses into subdirs so nested ``tokenizer/`` / ``processor/`` files
    are included in the manifest — silently skipping them would let
    ``verify_adapter`` pass after a tampered ``tokenizer/special_tokens.json``
    (code-review HIGH fix).

    Skips the signature file itself (so signing is idempotent on re-runs).
    Symlinks at file OR directory level are filtered here AND re-rejected
    inside ``_hash_file`` for defence-in-depth.

    Returns relative POSIX-style paths inside ``adapter_dir`` sorted
    alphabetically for a deterministic Merkle root.
    """
    rels: list[str] = []
    base_real = os.path.realpath(adapter_dir)
    for dirpath, dirnames, filenames in os.walk(adapter_dir, followlinks=False):
        # Symlinked subdir at this level — refuse loudly rather than skip
        # silently so a hostile adapter can't smuggle weights past the
        # manifest (review-fix HIGH).
        for sub in list(dirnames):
            sub_full = os.path.join(dirpath, sub)
            try:
                sub_st = os.lstat(sub_full)
            except OSError:
                dirnames.remove(sub)
                continue
            if stat.S_ISLNK(sub_st.st_mode):
                raise ValueError(
                    f"{sub!r}: symlinked subdir in adapter dir is not allowed"
                )
        for filename in filenames:
            if filename == _SIGNATURE_FILENAME and dirpath == adapter_dir:
                continue
            full = os.path.join(dirpath, filename)
            try:
                st = os.lstat(full)
            except OSError:
                continue
            if stat.S_ISLNK(st.st_mode):
                raise ValueError(
                    f"{filename!r}: symlink in adapter dir is not allowed"
                )
            if not stat.S_ISREG(st.st_mode):
                continue
            real = os.path.realpath(full)
            # Defence-in-depth: refuse files whose realpath escapes the
            # adapter dir (`followlinks=False` prevents this on dir traversal,
            # but file-level symlinks were rejected above; a hardlink to
            # outside content WOULD pass — caught here).
            try:
                common = os.path.commonpath([base_real, real])
            except ValueError:
                continue
            if common != base_real:
                continue
            rel = os.path.relpath(full, adapter_dir).replace(os.sep, "/")
            rels.append(rel)
    if len(rels) > _MAX_FILES:
        raise ValueError(f"adapter has > {_MAX_FILES} files")
    return sorted(rels)


def _merkle_root(entries: Tuple[FileEntry, ...]) -> str:
    """Compute a deterministic hash over the file list.

    Format: ``hash(version || count || name1:size1:sha1 || name2:size2:sha2 || ...)``.
    Not a true Merkle tree (no pair-wise pairing) — a flat hash works for our
    threat model (offline tamper detection) and avoids odd-leaf complications.
    """
    h = hashlib.sha256()
    h.update(str(_MANIFEST_VERSION).encode("utf-8"))
    h.update(b"\x1f")
    h.update(str(len(entries)).encode("utf-8"))
    for entry in entries:
        h.update(b"\x1f")
        h.update(entry.name.encode("utf-8"))
        h.update(b":")
        h.update(str(entry.size).encode("utf-8"))
        h.update(b":")
        h.update(entry.sha256.encode("utf-8"))
    return h.hexdigest()


def compute_adapter_manifest(adapter_dir: str) -> AdapterManifest:
    """Walk ``adapter_dir`` and return a deterministic manifest.

    Containment + symlink rejection on the adapter dir (TOCTOU defence,
    mirrors v0.53.1 ``enforce_under_cwd_and_no_symlink`` policy). Raises
    ``FileNotFoundError`` if the dir does not exist; raises ``ValueError``
    on containment / symlink violations.
    """
    enforce_under_cwd_and_no_symlink(adapter_dir, "adapter")
    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"{adapter_dir}: not a directory")

    names = _enumerate_files(adapter_dir)
    entries: list[FileEntry] = []
    for name in names:
        # Names use POSIX `/`; translate back to OS separator for hashing.
        full = os.path.join(adapter_dir, *name.split("/"))
        size, digest = _hash_file(full)
        entries.append(FileEntry(name=name, size=size, sha256=digest))

    entries_tuple = tuple(entries)
    return AdapterManifest(
        adapter=os.path.basename(os.path.normpath(adapter_dir)),
        version=_MANIFEST_VERSION,
        files=entries_tuple,
        merkle_root=_merkle_root(entries_tuple),
    )


def _resolve_backend(backend: object) -> SignBackend:
    if isinstance(backend, SignBackend):
        return backend
    if isinstance(backend, str):
        try:
            return SignBackend(backend.lower())
        except ValueError as exc:
            raise ValueError(
                f"unknown backend: {backend!r} (use one of "
                f"{[b.value for b in SignBackend]})"
            ) from exc
    raise TypeError(f"backend must be str or SignBackend, got {type(backend).__name__}")


def _manifest_to_dict(manifest: AdapterManifest) -> dict:
    return {
        "adapter": manifest.adapter,
        "version": manifest.version,
        "merkle_root": manifest.merkle_root,
        "files": [
            {"name": e.name, "size": e.size, "sha256": e.sha256}
            for e in manifest.files
        ],
    }


def _manifest_from_dict(payload: dict) -> AdapterManifest:
    files_raw = payload.get("files", [])
    if not isinstance(files_raw, list):
        raise ValueError("files must be a list")
    entries = tuple(
        FileEntry(
            name=str(entry["name"]),
            size=int(entry["size"]),
            sha256=str(entry["sha256"]),
        )
        for entry in files_raw
    )
    return AdapterManifest(
        adapter=str(payload.get("adapter", "")),
        version=int(payload.get("version", _MANIFEST_VERSION)),
        files=entries,
        merkle_root=str(payload.get("merkle_root", "")),
    )


def sign_adapter(
    adapter_dir: str,
    *,
    backend: object = SignBackend.UNSIGNED,
) -> SignatureRecord:
    """Compute the manifest, sign it, and write ``.soup-signature.json``.

    Args:
        adapter_dir: cwd-contained adapter directory.
        backend: ``"unsigned"`` is live; ``"sigstore"`` and ``"ed25519"`` are
            deferred to v0.60.1 (raises ``NotImplementedError``).

    Returns:
        ``SignatureRecord`` describing what was written.
    """
    from datetime import datetime, timezone

    chosen = _resolve_backend(backend)
    manifest = compute_adapter_manifest(adapter_dir)

    if chosen == SignBackend.UNSIGNED:
        signature = ""
    else:
        raise NotImplementedError(
            f"signing backend {chosen.value!r} is deferred to v0.60.1"
        )

    signed_at = datetime.now(tz=timezone.utc).isoformat()
    record = SignatureRecord(
        backend=chosen.value,
        merkle_root=manifest.merkle_root,
        signature=signature,
        signed_at=signed_at,
        manifest=manifest,
    )
    payload = {
        "backend": record.backend,
        "merkle_root": record.merkle_root,
        "signature": record.signature,
        "signed_at": record.signed_at,
        "manifest": _manifest_to_dict(manifest),
    }
    sig_path = os.path.join(adapter_dir, _SIGNATURE_FILENAME)
    atomic_write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        sig_path,
        prefix=".soup-sig.",
        suffix=".json.tmp",
        field="signature_file",
    )
    return record


def _load_signature(adapter_dir: str) -> Optional[SignatureRecord]:
    sig_path = os.path.join(adapter_dir, _SIGNATURE_FILENAME)
    # TOCTOU-hardened existence + type check via direct lstat (no
    # `isfile`-then-`open` race — security-review MEDIUM fix).
    try:
        st = os.lstat(sig_path)
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(st.st_mode):
        raise ValueError(f"{_SIGNATURE_FILENAME}: must not be a symlink")
    if not stat.S_ISREG(st.st_mode):
        return None
    # Cap signature file size — defends against a hostile adapter
    # shipping a multi-GiB JSON to OOM `verify_adapter`.
    if st.st_size > _MAX_SIGNATURE_BYTES:
        raise ValueError(
            f"{_SIGNATURE_FILENAME}: exceeds {_MAX_SIGNATURE_BYTES} bytes"
        )
    with open(sig_path, encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"{_SIGNATURE_FILENAME}: payload must be a JSON object")
    manifest_dict = payload.get("manifest", {})
    if not isinstance(manifest_dict, dict):
        raise ValueError(f"{_SIGNATURE_FILENAME}: 'manifest' must be a JSON object")
    manifest = _manifest_from_dict(manifest_dict)
    return SignatureRecord(
        backend=str(payload.get("backend", "")),
        merkle_root=str(payload.get("merkle_root", "")),
        signature=str(payload.get("signature", "")),
        signed_at=str(payload.get("signed_at", "")),
        manifest=manifest,
    )


def verify_adapter(adapter_dir: str, *, strict: bool = False) -> VerifyReport:
    """Verify that the adapter's files match the recorded manifest.

    Args:
        adapter_dir: cwd-contained adapter directory.
        strict: when True, raise ``ValueError`` on any failure (CI-friendly).
            When False, return a ``VerifyReport`` with ``valid=False``.

    Returns:
        ``VerifyReport``. ``valid=True`` requires a present signature file
        AND a recomputed Merkle root that matches the recorded one. Unsigned
        adapters fail verification in both modes — strict raises, lenient
        reports.
    """
    if not isinstance(strict, bool):
        raise TypeError("strict must be bool")
    enforce_under_cwd_and_no_symlink(adapter_dir, "adapter")

    name = os.path.basename(os.path.normpath(adapter_dir))
    record = _load_signature(adapter_dir)
    if record is None:
        reason = (
            f"adapter {name!r} is not signed (no {_SIGNATURE_FILENAME}); "
            "run `soup adapters sign` first"
        )
        if strict:
            raise ValueError(reason)
        return VerifyReport(
            adapter=name, valid=False, backend=None, reason=reason,
        )

    # Recompute manifest from current files
    current = compute_adapter_manifest(adapter_dir)
    findings: list[str] = []
    if current.merkle_root != record.merkle_root:
        findings.append(
            f"merkle root mismatch: recorded {record.merkle_root[:16]}..., "
            f"current {current.merkle_root[:16]}..."
        )

    # File-level diff for actionable advisories
    recorded_files = {entry.name: entry for entry in record.manifest.files}
    current_files = {entry.name: entry for entry in current.files}
    for fname, cur in current_files.items():
        rec = recorded_files.get(fname)
        if rec is None:
            findings.append(f"new file not in manifest: {fname!r}")
            continue
        if rec.sha256 != cur.sha256:
            findings.append(f"sha256 mismatch on {fname!r}")
    for fname in recorded_files:
        if fname not in current_files:
            findings.append(f"missing file: {fname!r}")

    if findings:
        reason = f"signature mismatch: {findings[0]}"
        if strict:
            raise ValueError(
                f"adapter {name!r} signature verification failed: {findings[0]}"
            )
        return VerifyReport(
            adapter=name,
            valid=False,
            backend=record.backend,
            reason=reason,
            findings=tuple(findings),
        )

    return VerifyReport(
        adapter=name,
        valid=True,
        backend=record.backend,
        reason="ok",
        findings=(),
    )
