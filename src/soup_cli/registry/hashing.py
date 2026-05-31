"""Deterministic hashing for registry entries.

- ``hash_config`` produces a stable SHA-256 of a config dict (order-insensitive).
- ``hash_file`` streams a file's bytes through SHA-256 so large data files
  don't need to fit in memory.
- ``hash_entry`` combines config + data + base model into a single digest
  that identifies the exact training recipe used.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

_HASH_CHUNK_BYTES = 1024 * 1024  # 1 MB


def _canonical_json(value: Any) -> str:
    """Serialise to JSON with sorted keys — stable across insertion order."""
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def hash_config(config: dict) -> str:
    """SHA-256 of the config dict after canonical JSON serialisation."""
    payload = _canonical_json(config).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def hash_file(path: str) -> str:
    """SHA-256 of a file's contents, streamed in 1 MB chunks."""
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as fh:
        while True:
            chunk = fh.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def hash_entry(
    *,
    config: dict,
    data_path: Optional[str],
    base_model: str,
) -> str:
    """SHA-256 combining config, data file, and base model name.

    The data file is hashed by content (not path) so moving files doesn't
    invalidate hashes. If ``data_path`` is ``None`` the field is omitted
    from the hash input. An empty string is NOT treated as ``None`` — it
    is passed through to :func:`hash_file`, which will raise.
    """
    parts: dict[str, Any] = {
        "config": _canonical_json(config),
        "base_model": base_model,
    }
    if data_path is not None:
        parts["data_sha256"] = hash_file(data_path)
    return hash_config(parts)
