"""Local byte verification for inputs bound by an AutoDistill plan."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from soup_cli.autodistill.contract import (
    ArtifactCorruptionError,
    AutoDistillPlan,
    FileDigest,
    canonicalize_jsonl_bytes,
)

_HASH_CHUNK_BYTES = 8 * 1024 * 1024


def _stream_file(path: Path, *, retain_bytes: bool) -> tuple[int, str, bytes]:
    digest = hashlib.sha256()
    size = 0
    retained: list[bytes] = []
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            size += len(chunk)
            digest.update(chunk)
            if retain_bytes:
                retained.append(chunk)
    return size, digest.hexdigest(), b"".join(retained)


def _safe_file(root: Path, relative_path: str) -> Path:
    root_absolute = os.path.abspath(root)
    root_real = os.path.realpath(root_absolute)
    candidate = os.path.abspath(os.path.join(root_absolute, relative_path))
    candidate_real = os.path.realpath(candidate)
    try:
        contained = os.path.commonpath((root_real, candidate_real)) == root_real
    except ValueError as exc:
        raise ArtifactCorruptionError("fingerprinted file escapes its root") from exc
    if not contained:
        raise ArtifactCorruptionError("fingerprinted file escapes its root")
    path = Path(candidate)
    relative_parts = Path(relative_path.replace("\\", "/")).parts
    lexical = Path(root_absolute)
    has_symlink = lexical.is_symlink()
    for part in relative_parts:
        lexical /= part
        has_symlink = has_symlink or lexical.is_symlink()
    if has_symlink or not path.is_file():
        raise ArtifactCorruptionError(f"fingerprinted file {relative_path!r} is missing")
    return path


def _verify_file(
    root: Path,
    expected: FileDigest,
    *,
    retain_bytes: bool = True,
) -> bytes:
    size, digest, data = _stream_file(
        _safe_file(root, expected.path),
        retain_bytes=retain_bytes,
    )
    if size != expected.bytes:
        raise ArtifactCorruptionError(f"fingerprinted file {expected.path!r} byte count mismatch")
    if digest != expected.sha256:
        raise ArtifactCorruptionError(f"fingerprinted file {expected.path!r} sha256 mismatch")
    return data


def verify_teacher_fingerprint(
    plan: AutoDistillPlan,
    *,
    teacher_root: str | os.PathLike[str],
) -> None:
    """Verify only the teacher files needed by capture; no student path is accepted."""

    root = Path(teacher_root)
    _, config_digest, _ = _stream_file(
        _safe_file(root, "config.json"),
        retain_bytes=False,
    )
    if config_digest != plan.teacher.config_sha256:
        raise ArtifactCorruptionError("teacher config.json sha256 mismatch")
    for expected in plan.teacher.weights:
        _verify_file(root, expected, retain_bytes=False)


def verify_tokenizer_fingerprint(
    plan: AutoDistillPlan,
    *,
    tokenizer_root: str | os.PathLike[str],
    chat_template: str,
    renderer: str,
) -> None:
    """Verify shared tokenizer bytes and the exact runtime rendering identity."""

    if renderer != plan.tokenizer.renderer:
        raise ArtifactCorruptionError("tokenizer renderer does not match plan")
    template_digest = hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
    if template_digest != plan.tokenizer.chat_template_sha256:
        raise ArtifactCorruptionError("tokenizer chat template sha256 mismatch")
    verify_tokenizer_file_fingerprint(plan, tokenizer_root=tokenizer_root)


def verify_tokenizer_file_fingerprint(
    plan: AutoDistillPlan,
    *,
    tokenizer_root: str | os.PathLike[str],
) -> None:
    """Verify bound tokenizer bytes before a backend instantiates the tokenizer."""

    root = Path(tokenizer_root)
    for expected in plan.tokenizer.files:
        _verify_file(root, expected, retain_bytes=False)


def verify_dataset_fingerprint(
    plan: AutoDistillPlan,
    *,
    dataset_root: str | os.PathLike[str],
) -> None:
    """Verify ordered source bytes and their combined canonical JSONL identity."""

    verified_dataset_bytes(plan, dataset_root=dataset_root)


def verified_dataset_bytes(
    plan: AutoDistillPlan,
    *,
    dataset_root: str | os.PathLike[str],
) -> bytes:
    """Return bound canonical JSONL bytes after all source and logical checks pass."""

    root = Path(dataset_root)
    normalized_parts: list[bytes] = []
    row_count = 0
    for expected in plan.dataset.source_files:
        source = _verify_file(root, expected)
        try:
            normalized = canonicalize_jsonl_bytes(source)
        except ValueError as exc:
            raise ArtifactCorruptionError(
                f"dataset file {expected.path!r} is not valid canonicalizable JSONL"
            ) from exc
        normalized_parts.append(normalized)
        row_count += len(normalized.splitlines())
    normalized_bytes = b"".join(normalized_parts)
    if hashlib.sha256(normalized_bytes).hexdigest() != plan.dataset.normalized_sha256:
        raise ArtifactCorruptionError("dataset normalized sha256 mismatch")
    if row_count != plan.dataset.rows:
        raise ArtifactCorruptionError("dataset normalized row count mismatch")
    return normalized_bytes
