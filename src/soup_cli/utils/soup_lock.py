"""``soup.lock`` shared run lockfile (v0.67.0 Part E).

A ``soup.lock`` is the closure of three SHA-256 hashes:

    closure = sha256(base_model_sha || dataset_sha || env_hash)

Committed to git alongside ``soup.yaml``, teams coordinate on
"reproducible training run" by checking the closure on every
``soup train`` and refusing to start when the lock drifts. Composes
with v0.64 Part C ``soup env`` — operators run ``soup env lock``
to get ``env_hash``, then ``soup lock write`` to write the file.

Public surface:

- ``SoupLock`` frozen dataclass
- ``LockDrift`` frozen dataclass (ok / changes tuple)
- ``compute_lock_closure(*, base_model_sha, dataset_sha, env_hash)`` -> hex
- ``write_lock(lock, path)`` / ``read_lock(path)`` atomic JSON I/O
- ``check_lock_drift(expected, actual)`` -> ``LockDrift``
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import asdict, dataclass
from typing import Tuple

from soup_cli.utils.paths import (
    atomic_write_text,
    enforce_under_cwd_and_no_symlink,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_FILE_BYTES = 1 * 1024 * 1024  # 1 MiB cap on lock file size
_MAX_VERSION_LEN = 64
_MAX_BASE_MODEL_LEN = 512
_MAX_CREATED_AT_LEN = 64


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def _check_sha(value: object, field: str) -> str:
    if isinstance(value, bool):
        raise TypeError(f"{field} must not be bool")
    if not isinstance(value, str):
        raise TypeError(f"{field} must be str")
    if not _SHA256_RE.match(value):
        raise ValueError(f"{field} must be 64 hex chars (got {value!r})")
    return value


def _check_str(value: object, field: str, max_len: int) -> str:
    if isinstance(value, bool):
        raise TypeError(f"{field} must not be bool")
    if not isinstance(value, str):
        raise TypeError(f"{field} must be str")
    if not value:
        raise ValueError(f"{field} must be non-empty")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain null bytes")
    if len(value) > max_len:
        raise ValueError(
            f"{field} length {len(value)} > {max_len}"
        )
    return value


# ---------------------------------------------------------------------------
# compute_lock_closure
# ---------------------------------------------------------------------------


def compute_lock_closure(
    *,
    base_model_sha: str,
    dataset_sha: str,
    env_hash: str,
) -> str:
    """Return the 64-hex closure SHA over the three input hashes.

    Each input must be a 64-hex SHA-256 itself. The closure is
    ``sha256(base_model_sha || dataset_sha || env_hash)``.
    """
    _check_sha(base_model_sha, "base_model_sha")
    _check_sha(dataset_sha, "dataset_sha")
    _check_sha(env_hash, "env_hash")
    h = hashlib.sha256()
    h.update(base_model_sha.encode("ascii"))
    h.update(dataset_sha.encode("ascii"))
    h.update(env_hash.encode("ascii"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SoupLock:
    """The on-disk lockfile shape.

    Stored as JSON at ``soup.lock`` (or operator-named path); committed
    to git so the team coordinates on "reproducible training run".
    """

    soup_version: str
    base_model: str
    base_model_sha: str
    dataset_sha: str
    env_hash: str
    closure_sha: str
    created_at: str

    def __post_init__(self) -> None:
        _check_str(self.soup_version, "soup_version", _MAX_VERSION_LEN)
        _check_str(self.base_model, "base_model", _MAX_BASE_MODEL_LEN)
        _check_sha(self.base_model_sha, "base_model_sha")
        _check_sha(self.dataset_sha, "dataset_sha")
        _check_sha(self.env_hash, "env_hash")
        _check_sha(self.closure_sha, "closure_sha")
        _check_str(self.created_at, "created_at", _MAX_CREATED_AT_LEN)


@dataclass(frozen=True)
class LockDrift:
    """Result of comparing two locks. ``ok`` is True when no field changed."""

    ok: bool
    changes: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.ok, bool):
            raise TypeError("ok must be bool")
        if not isinstance(self.changes, tuple):
            raise TypeError("changes must be tuple")
        for c in self.changes:
            if not isinstance(c, str):
                raise TypeError("changes entries must be str")


# ---------------------------------------------------------------------------
# Atomic I/O
# ---------------------------------------------------------------------------


def write_lock(lock: SoupLock, path: str) -> str:
    """Atomically write a lock to JSON under cwd containment."""
    if not isinstance(lock, SoupLock):
        raise TypeError("lock must be SoupLock")
    text = json.dumps(asdict(lock), indent=2, sort_keys=True)
    return atomic_write_text(text, path, field="lock path")


def read_lock(path: str) -> SoupLock:
    """Read + validate a soup.lock from JSON. cwd-contained, symlink-rejected."""
    if not isinstance(path, str):
        raise TypeError("path must be str")
    if not path:
        raise ValueError("path must be non-empty")
    if "\x00" in path:
        raise ValueError("path must not contain null bytes")
    enforce_under_cwd_and_no_symlink(path, field="lock path")
    real = os.path.realpath(path)
    if not os.path.exists(real):
        raise FileNotFoundError(f"lock file not found: {path!r}")
    st = os.lstat(real)
    if stat.S_ISLNK(st.st_mode):
        raise ValueError("lock path must not be a symlink (TOCTOU defence)")
    if st.st_size > _MAX_FILE_BYTES:
        raise ValueError(
            f"lock file size {st.st_size} > {_MAX_FILE_BYTES}"
        )
    with open(real, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("lock file root must be JSON object")
    # SoupLock __post_init__ does the rest of the validation
    return SoupLock(
        soup_version=data.get("soup_version", ""),
        base_model=data.get("base_model", ""),
        base_model_sha=data.get("base_model_sha", ""),
        dataset_sha=data.get("dataset_sha", ""),
        env_hash=data.get("env_hash", ""),
        closure_sha=data.get("closure_sha", ""),
        created_at=data.get("created_at", ""),
    )


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------


def check_lock_drift(expected: SoupLock, actual: SoupLock) -> LockDrift:
    """Compare two locks field-by-field. Returns a ``LockDrift`` report.

    Only the four content-bearing SHA fields + base_model name affect
    drift detection. ``soup_version`` / ``created_at`` differences are
    tracked separately as advisory-only (operators upgrading Soup will
    legitimately see them change).
    """
    if not isinstance(expected, SoupLock):
        raise TypeError("expected must be SoupLock")
    if not isinstance(actual, SoupLock):
        raise TypeError("actual must be SoupLock")
    changes: list[str] = []
    for field in ("base_model", "base_model_sha", "dataset_sha", "env_hash", "closure_sha"):
        if getattr(expected, field) != getattr(actual, field):
            changes.append(f"{field} drifted")
    return LockDrift(ok=not changes, changes=tuple(changes))
