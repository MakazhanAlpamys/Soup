"""Lightweight snapshot pointers for adapter training (v0.57.0 Part D).

A `branch` is a frozen pointer to {config, dataset SHA, base model, created_at}
so two trainings can be compared cleanly. Pointers live under
``~/.soup/branches/<name>.json`` (or ``SOUP_BRANCHES_DIR``-overridden) with
``0o600`` perms.

Public surface:

- ``create_branch(name, *, config_path, base_model, dataset_path=None)`` -> ``Branch``
- ``list_branches()`` -> tuple[str, ...]
- ``load_branch(name)`` -> ``Branch``
- ``delete_branch(name)`` -> bool
- ``write_checkout(branch, target_path)`` -> snapshot config back to cwd

Validation policy mirrors the project standard: alphanumeric + ``._-``,
≤128 chars, null-byte rejected, branches dir is containment-checked under
``$HOME`` / ``$CWD`` / ``$TMPDIR`` per v0.36.0 ``SOUP_BATCH_CACHE_PATH`` policy.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Optional, Tuple

from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink, is_under

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,127}$")
_MAX_BRANCHES = 1024
_HASH_CHUNK = 65536
_MAX_CONFIG_BYTES = 1_048_576  # 1 MiB cap on snapshot config payload


@dataclass(frozen=True)
class Branch:
    name: str
    config_path: str
    config_sha256: str
    dataset_sha256: Optional[str]
    base_model: str
    created_at: float
    soup_version: str
    # v0.71.4 #173 — optional link to a v0.26 Registry entry so a branch
    # pointer participates in the lineage DAG. ``None`` preserves back-compat
    # with v0.57.0 pointers that have no Registry edge.
    registry_entry_id: Optional[str] = None


def _validate_name(name: object) -> str:
    if isinstance(name, bool) or not isinstance(name, str):
        raise TypeError("name must be str")
    if not name:
        raise ValueError("name must be non-empty")
    if "\x00" in name:
        raise ValueError("name must not contain null bytes")
    if not _NAME_RE.match(name):
        raise ValueError(
            "name must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}"
        )
    return name


def _branches_dir() -> Path:
    """Resolve the branches directory with env-var override + containment."""
    override = os.environ.get("SOUP_BRANCHES_DIR")
    if override:
        # Reject null + every C0 control char (CRLF, tabs, etc.) — mirrors
        # v0.51.0 validate_hub_endpoint policy. Silently fall through to
        # the default rather than raise; an env var is operator-supplied
        # configuration, not API input.
        if any(ord(ch) < 0x20 for ch in override):
            override = None
    if override:
        candidate = Path(os.path.realpath(override))
        bounds = [
            Path.home(),
            Path.cwd(),
            Path(tempfile.gettempdir()),
        ]
        if any(is_under(str(candidate), str(b)) for b in bounds):
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
    default = Path.home() / ".soup" / "branches"
    default.mkdir(parents=True, exist_ok=True)
    return default


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_HASH_CHUNK)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _validate_str_field(value: object, field: str, max_len: int = 512) -> str:
    if isinstance(value, bool) or not isinstance(value, str):
        raise TypeError(f"{field} must be str")
    if not value:
        raise ValueError(f"{field} must be non-empty")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain null bytes")
    if len(value) > max_len:
        raise ValueError(f"{field} must be ≤{max_len} chars")
    return value


def create_branch(
    name: str,
    *,
    config_path: str,
    base_model: str,
    dataset_path: Optional[str] = None,
    registry_entry_id: Optional[str] = None,
    dataset_sha256: Optional[str] = None,
) -> Branch:
    """Snapshot a training environment as an immutable branch pointer.

    The config file is hashed (SHA-256) and the dataset file (if provided)
    is hashed too so callers can detect "same config, different data" drift.

    ``registry_entry_id`` (v0.71.4 #173) optionally links the pointer to a
    v0.26 Registry entry. ``dataset_sha256`` lets a caller (e.g.
    ``branch_from_registry``) record a known dataset hash directly instead
    of re-hashing a file — mutually exclusive with ``dataset_path``.
    """
    from soup_cli import __version__

    name = _validate_name(name)
    enforce_under_cwd_and_no_symlink(config_path, "config_path")
    _validate_str_field(base_model, "base_model", max_len=512)
    if registry_entry_id is not None:
        registry_entry_id = _validate_str_field(
            registry_entry_id, "registry_entry_id", max_len=128
        )
    if dataset_path is not None and dataset_sha256 is not None:
        raise ValueError("pass dataset_path OR dataset_sha256, not both")

    config_full = Path(config_path)
    if not config_full.is_file():
        raise FileNotFoundError(f"config not found: {config_full.name}")
    if config_full.stat().st_size > _MAX_CONFIG_BYTES:
        raise ValueError(
            f"config exceeds {_MAX_CONFIG_BYTES} byte cap"
        )

    config_sha = _hash_file(config_full)

    dataset_sha: Optional[str] = None
    if dataset_path is not None:
        enforce_under_cwd_and_no_symlink(dataset_path, "dataset_path")
        ds_full = Path(dataset_path)
        if not ds_full.is_file():
            raise FileNotFoundError(f"dataset not found: {ds_full.name}")
        dataset_sha = _hash_file(ds_full)
    elif dataset_sha256 is not None:
        dataset_sha = _validate_str_field(
            dataset_sha256, "dataset_sha256", max_len=128
        )

    branches_dir = _branches_dir()
    existing = sorted(p for p in branches_dir.glob("*.json"))
    if len(existing) >= _MAX_BRANCHES:
        raise RuntimeError(f"branches dir has ≥{_MAX_BRANCHES} entries")

    branch = Branch(
        name=name,
        config_path=str(config_full),
        config_sha256=config_sha,
        dataset_sha256=dataset_sha,
        base_model=base_model,
        created_at=time.time(),
        soup_version=__version__,
        registry_entry_id=registry_entry_id,
    )
    _atomic_write_branch(branches_dir, branch)
    return branch


def _atomic_write_branch(branches_dir: Path, branch: Branch) -> None:
    target = branches_dir / f"{branch.name}.json"
    payload = json.dumps(asdict(branch), indent=2, sort_keys=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(branches_dir), prefix=".tmp_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp_path, str(target))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    if os.name != "nt":
        try:
            os.chmod(str(target), 0o600)
        except OSError:
            pass


def list_branches() -> Tuple[str, ...]:
    branches_dir = _branches_dir()
    return tuple(sorted(p.stem for p in branches_dir.glob("*.json")))


def load_branch(name: str) -> Branch:
    name = _validate_name(name)
    branches_dir = _branches_dir()
    target = branches_dir / f"{name}.json"
    if not target.is_file():
        raise FileNotFoundError(f"branch not found: {name}")
    # TOCTOU defence: reject symlinks at the branch pointer path before read
    # so a planted symlink to /etc/passwd cannot leak file content via the
    # JSON-parse error path (review fix HIGH).
    st = os.lstat(str(target))
    if stat.S_ISLNK(st.st_mode):
        raise ValueError(f"branch pointer must not be a symlink: {name}")
    text = target.read_text(encoding="utf-8")
    if len(text) > _MAX_CONFIG_BYTES:
        raise ValueError("branch pointer exceeds size cap")
    raw = json.loads(text)
    return Branch(
        name=_validate_name(raw["name"]),
        config_path=_validate_str_field(raw["config_path"], "config_path", max_len=4096),
        config_sha256=_validate_str_field(raw["config_sha256"], "config_sha256", max_len=128),
        dataset_sha256=(
            _validate_str_field(raw["dataset_sha256"], "dataset_sha256", max_len=128)
            if raw.get("dataset_sha256") is not None
            else None
        ),
        base_model=_validate_str_field(raw["base_model"], "base_model"),
        created_at=float(raw["created_at"]),
        soup_version=_validate_str_field(raw["soup_version"], "soup_version", max_len=64),
        # ``.get`` so v0.57.0 pointers (no registry_entry_id key) load cleanly.
        registry_entry_id=(
            _validate_str_field(
                raw["registry_entry_id"], "registry_entry_id", max_len=128
            )
            if raw.get("registry_entry_id") is not None
            else None
        ),
    )


def delete_branch(name: str) -> bool:
    name = _validate_name(name)
    branches_dir = _branches_dir()
    target = branches_dir / f"{name}.json"
    if not target.is_file():
        return False
    # TOCTOU defence: reject symlinks before unlink so a planted symlink
    # at <branches>/<name>.json -> /important/file cannot be silently deleted.
    st = os.lstat(str(target))
    if stat.S_ISLNK(st.st_mode):
        raise ValueError(f"branch pointer must not be a symlink: {name}")
    target.unlink()
    return True


def branch_pointer_path(name: str) -> Path:
    """Return the on-disk JSON pointer path for a branch (must exist)."""
    name = _validate_name(name)
    target = _branches_dir() / f"{name}.json"
    if not target.is_file():
        raise FileNotFoundError(f"branch not found: {name}")
    return target


def attach_branch_to_registry(name: str, registry_ref: str) -> int:
    """Attach a branch pointer to a v0.26 Registry entry as ``branch_ref``.

    Resolves ``registry_ref`` (id / prefix / name:tag), attaches the branch
    JSON file as a ``branch_ref`` artifact (``enforce_cwd=False`` since the
    pointer lives in ``~/.soup/branches`` outside cwd — a library-produced
    path), then re-writes the pointer with ``registry_entry_id`` set so the
    edge is recorded on both sides. Returns the artifact rowid.
    """
    from soup_cli.registry.store import RegistryStore

    branch = load_branch(name)
    pointer = branch_pointer_path(name)
    branches_dir = _branches_dir()
    with RegistryStore() as store:
        resolved = store.resolve(registry_ref)
        if resolved is None:
            raise ValueError(f"registry entry not found: {registry_ref}")
        rowid = store.add_artifact(
            entry_id=resolved,
            kind="branch_ref",
            path=str(pointer),
            enforce_cwd=False,
        )
    # Re-write the pointer with the resolved id so the edge is bidirectional.
    _atomic_write_branch(branches_dir, replace(branch, registry_entry_id=resolved))
    return rowid


def branch_from_registry(
    name: str,
    registry_ref: str,
    *,
    config_out: Optional[str] = None,
) -> Branch:
    """Create a branch by deriving config + base_model + dataset from a Registry entry.

    Materialises the entry's stored ``config_json`` to a cwd file (default
    ``<name>.from-registry.yaml``) so the snapshot points at a real config,
    derives ``base_model`` from the entry, and records the entry's
    ``data_hash`` as the branch's ``dataset_sha256`` (the registry keeps the
    file hash, not a path). The branch is linked back via ``registry_entry_id``.
    """
    import yaml

    from soup_cli.registry.store import RegistryStore

    name = _validate_name(name)
    with RegistryStore() as store:
        resolved = store.resolve(registry_ref)
        if resolved is None:
            raise ValueError(f"registry entry not found: {registry_ref}")
        entry = store.get(resolved)
        if entry is None:
            raise ValueError(f"registry entry not found: {registry_ref}")
        base_model = entry.get("base_model") or "unknown"
        data_hash = entry.get("data_hash")
        try:
            config = json.loads(entry.get("config_json") or "{}")
        except (TypeError, ValueError):
            config = {}

    out_name = config_out if config_out is not None else f"{name}.from-registry.yaml"
    enforce_under_cwd_and_no_symlink(out_name, "config_out")
    out_path = Path(out_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    return create_branch(
        name,
        config_path=str(out_path),
        base_model=base_model,
        registry_entry_id=resolved,
        dataset_sha256=data_hash if data_hash else None,
    )


def write_checkout(branch: Branch, target_path: str) -> Path:
    """Copy the snapshotted config into cwd so the user can re-run."""
    if not isinstance(branch, Branch):
        raise TypeError("branch must be Branch")
    enforce_under_cwd_and_no_symlink(target_path, "target_path")

    source = Path(branch.config_path)
    if not source.is_file():
        raise FileNotFoundError(
            f"branch config no longer exists: {source.name}"
        )

    actual_sha = _hash_file(source)
    if actual_sha != branch.config_sha256:
        raise ValueError(
            f"branch config drifted from snapshot SHA "
            f"({actual_sha[:8]} vs {branch.config_sha256[:8]})"
        )

    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(target.parent), prefix=".tmp_")
    try:
        with os.fdopen(fd, "wb") as fh:
            with open(source, "rb") as src:
                while True:
                    chunk = src.read(_HASH_CHUNK)
                    if not chunk:
                        break
                    fh.write(chunk)
        os.replace(tmp_path, str(target))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return target
