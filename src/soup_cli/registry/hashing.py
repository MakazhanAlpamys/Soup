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
    data_sha256: Optional[str] = None,
    base_revision: Optional[str] = None,
    base_revision_source: Optional[str] = None,
) -> str:
    """SHA-256 combining config, data file, and base model name.

    The data file is hashed by content (not path) so moving files doesn't
    invalidate hashes. If neither a path nor a historical digest is given,
    the data field is omitted. An empty string is NOT treated as ``None`` — it
    is passed through to :func:`hash_file`, which will raise.
    ``data_sha256`` transfers a historical digest without re-reading its path;
    an explicit ``data_path`` takes precedence. A known revision and its source
    are included, while unknown identities preserve the legacy fingerprint.
    """
    from soup_cli.utils.run_provenance import validate_identity, validate_training_hash

    validate_training_hash(data_sha256)
    validate_identity(base_revision, base_revision_source)
    parts: dict[str, Any] = {
        "config": _canonical_json(config),
        "base_model": base_model,
    }
    if data_path is not None:
        parts["data_sha256"] = hash_file(data_path)
    elif data_sha256 is not None:
        parts["data_sha256"] = data_sha256
    if base_revision is not None:
        parts["base_revision"] = base_revision
        parts["base_revision_source"] = base_revision_source
    return hash_config(parts)
