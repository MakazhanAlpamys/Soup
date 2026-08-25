"""Durable checkpoint journal for Best-of-N generation."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

from soup_cli.utils.paths import atomic_write_text, enforce_under_cwd_and_no_symlink

_CHECKPOINT_VERSION = 1


def run_digest(prompts: list[str], config: dict[str, Any]) -> str:
    """Bind reusable work to the exact prompt sequence and generation config."""
    payload = {"prompts": prompts, "config": config}
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _record_digest(record: dict[str, Any]) -> str:
    encoded = json.dumps(
        record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def initialise_checkpoint(path: str, *, digest: str, total: int) -> None:
    """Create a new private journal containing its versioned run header."""
    if os.path.lexists(path):
        raise ValueError("checkpoint already exists; pass --resume to reuse it")
    header = {
        "_best_of_n_checkpoint": {
            "version": _CHECKPOINT_VERSION,
            "run_digest": digest,
            "total": total,
        }
    }
    text = json.dumps(header, sort_keys=True, separators=(",", ":")) + "\n"
    written = atomic_write_text(text, path, field="--checkpoint path")
    if os.name != "nt":
        os.chmod(written, 0o600)


def _open_checkpoint(path: str):
    enforce_under_cwd_and_no_symlink(path, "--checkpoint path")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("checkpoint must be a regular file")
        return os.fdopen(fd, "r", encoding="utf-8")
    except Exception:
        os.close(fd)
        raise


def load_checkpoint(
    path: str, *, digest: str, total: int
) -> list[tuple[dict, dict | None]]:
    """Validate a journal and return its exactly-once completed prompt prefix."""
    entries: list[tuple[dict, dict | None]] = []
    with _open_checkpoint(path) as handle:
        for line_number, raw in enumerate(handle, 1):
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"checkpoint has invalid JSON on line {line_number}") from exc
            if line_number == 1:
                header = record.get("_best_of_n_checkpoint") if isinstance(record, dict) else None
                if not isinstance(header, dict):
                    raise ValueError("checkpoint header is missing")
                if header.get("version") != _CHECKPOINT_VERSION:
                    raise ValueError("checkpoint version is not supported")
                if header.get("run_digest") != digest or header.get("total") != total:
                    raise ValueError("checkpoint does not match prompts or run configuration")
                continue
            if not isinstance(record, dict) or record.get("index") != len(entries):
                raise ValueError("checkpoint prompt indexes must be sequential and exactly once")
            core = {key: value for key, value in record.items() if key != "entry_digest"}
            if record.get("entry_digest") != _record_digest(core):
                raise ValueError(f"checkpoint entry {len(entries)} has a digest mismatch")
            sft = record.get("sft")
            dpo = record.get("dpo")
            if not isinstance(sft, dict) or (dpo is not None and not isinstance(dpo, dict)):
                raise ValueError(f"checkpoint entry {len(entries)} is malformed")
            entries.append((sft, dpo))
            if len(entries) > total:
                raise ValueError("checkpoint contains more prompt groups than the input")
    if not entries and not os.path.getsize(path):
        raise ValueError("checkpoint is empty")
    return entries


def append_checkpoint(path: str, *, index: int, sft: dict, dpo: dict | None) -> None:
    """Append and fsync one completed prompt group before moving to the next."""
    core = {"index": index, "sft": sft, "dpo": dpo}
    record = {**core, "entry_digest": _record_digest(core)}
    payload = (
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + os.linesep
    ).encode("utf-8")
    enforce_under_cwd_and_no_symlink(path, "--checkpoint path")
    flags = (
        os.O_WRONLY
        | os.O_APPEND
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
    )
    fd = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("checkpoint must be a regular file")
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("checkpoint append made no progress")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def dataset_text(rows: list[dict]) -> str:
    """Serialize rows deterministically for final atomic publication."""
    if not rows:
        return ""
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )


def manifest_text(
    *,
    digest: str,
    sft_path: str,
    sft_text: str,
    sft_count: int,
    pair_path: str,
    pair_text: str,
    pair_count: int,
) -> str:
    """Describe one consistent final generation; this file is published last."""
    manifest: dict[str, Any] = {
        "schema": "soup.best_of_n.manifest.v1",
        "run_digest": digest,
        "sft": {
            "file": Path(sft_path).name,
            "rows": sft_count,
            "sha256": hashlib.sha256(sft_text.encode("utf-8")).hexdigest(),
        },
        "dpo": None,
    }
    if pair_path:
        manifest["dpo"] = {
            "file": Path(pair_path).name,
            "rows": pair_count,
            "sha256": hashlib.sha256(pair_text.encode("utf-8")).hexdigest(),
        }
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"
