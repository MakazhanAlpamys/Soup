"""Pack registry entries into ``.can`` artifacts (v0.26.0 Part E)."""

from __future__ import annotations

import io
import json
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from soup_cli.cans.schema import CAN_FORMAT_VERSION, Manifest
from soup_cli.registry.store import RegistryStore
from soup_cli.utils.paths import is_under_cwd

_MAX_CAN_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB


def _add_text(tar: tarfile.TarFile, name: str, content: str) -> None:
    data = content.encode("utf-8")
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mtime = int(datetime.now().timestamp())
    tar.addfile(info, io.BytesIO(data))


def pack_entry(
    *, entry_id: str, out_path: str, author: str = "unknown",
    description: Optional[str] = None,
) -> Path:
    """Pack a registry entry as a ``.can`` tarball.

    Args:
        entry_id: registry entry id (full id or prefix).
        out_path: destination ``.can`` file. Must stay under the cwd captured
            at call time.
        author: author handle baked into the manifest.
        description: optional free-form description.

    Returns:
        The absolute path to the written file.
    """
    out = Path(out_path)
    if not is_under_cwd(out):
        raise ValueError(f"out path '{out_path}' is outside cwd - refusing")

    with RegistryStore() as store:
        resolved = store.resolve(entry_id)
        if resolved is None:
            raise ValueError(f"registry entry not found: {entry_id}")
        entry = store.get(resolved)
        if entry is None:
            raise ValueError(f"registry entry not found: {entry_id}")
        try:
            config = json.loads(entry.get("config_json") or "{}")
        except (TypeError, ValueError):
            config = {}

    manifest = Manifest(
        can_format_version=CAN_FORMAT_VERSION,
        name=entry["name"],
        author=author,
        created_at=datetime.now().isoformat(timespec="seconds"),
        base_hash=entry.get("entry_hash", ""),
        description=description or entry.get("notes"),
        tags=list(entry.get("tags", [])),
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, mode="w:gz") as tar:
        _add_text(tar, "manifest.yaml", yaml.safe_dump(
            manifest.model_dump(), sort_keys=False,
        ))
        _add_text(tar, "config.yaml", yaml.safe_dump(config, sort_keys=False))
        _add_text(tar, "data_ref.yaml", yaml.safe_dump(
            {"kind": "local", "note": "user must supply data locally"},
            sort_keys=False,
        ))
        _add_text(tar, "recipe.md",
                  f"# {manifest.name}\n\nBase: {entry['base_model']}\n"
                  f"Task: {entry['task']}\n\n"
                  f"{manifest.description or ''}\n")

    # Enforce size cap — reject anything larger than 100 MB
    if out.stat().st_size > _MAX_CAN_SIZE_BYTES:
        out.unlink(missing_ok=True)
        raise ValueError(
            f"can exceeds max size ({_MAX_CAN_SIZE_BYTES // 1024 // 1024} MB) - "
            "trim metadata or exclude large fields"
        )
    return out


_FORBIDDEN_MOD_KEYS = frozenset({
    "__class__", "__init__", "__dict__", "__globals__", "__builtins__",
    "__import__", "__bases__", "__mro__",
})


def _apply_modification(config: dict, mod: str) -> None:
    """Apply a single ``dotted.path=value`` modification in-place."""
    if "=" not in mod:
        raise ValueError(
            f"modification '{mod}' must be 'dotted.path=value'"
        )
    path, raw_value = mod.split("=", 1)
    if "\x00" in path:
        raise ValueError(f"modification key contains null byte: {path!r}")
    keys = path.split(".")
    if any(key in _FORBIDDEN_MOD_KEYS or key.startswith("__") for key in keys):
        raise ValueError(
            f"modification key '{path}' references a dunder / forbidden name"
        )
    # Try parsing as JSON (number, bool, null, string) for proper typing.
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    target = config
    for key in keys[:-1]:
        target = target.setdefault(key, {})
        if not isinstance(target, dict):
            raise ValueError(
                f"cannot modify '{path}': '{key}' is not a dict"
            )
    target[keys[-1]] = value


def fork_can(
    *, source: str, out_path: str, modifications: list[str],
    author: str = "unknown",
) -> Path:
    """Apply ``modifications`` to a can's config and re-pack."""
    from soup_cli.cans.unpack import inspect_can, read_config

    src = Path(source)
    if not is_under_cwd(src):
        raise ValueError(f"source '{source}' is outside cwd - refusing")
    if not src.exists():
        raise FileNotFoundError(f"source can not found: {source}")

    out = Path(out_path)
    if not is_under_cwd(out):
        raise ValueError(f"out path '{out_path}' is outside cwd - refusing")

    manifest = inspect_can(source)
    config = read_config(source)
    for mod in modifications:
        _apply_modification(config, mod)

    with tarfile.open(out, mode="w:gz") as tar:
        _add_text(tar, "manifest.yaml", yaml.safe_dump({
            **manifest.model_dump(),
            "author": author,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "name": f"{manifest.name}-fork",
        }, sort_keys=False))
        _add_text(tar, "config.yaml", yaml.safe_dump(config, sort_keys=False))
        _add_text(tar, "data_ref.yaml", yaml.safe_dump(
            {"kind": "local", "note": "forked"},
            sort_keys=False,
        ))

    if out.stat().st_size > _MAX_CAN_SIZE_BYTES:
        out.unlink(missing_ok=True)
        raise ValueError(
            f"forked can exceeds max size "
            f"({_MAX_CAN_SIZE_BYTES // 1024 // 1024} MB)"
        )
    return out
