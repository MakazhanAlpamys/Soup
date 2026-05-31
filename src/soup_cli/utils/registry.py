"""Dataset info registry — local name → path + format mapping.

Stores registry at ~/.soup/datasets.json so datasets can be referenced
by name in soup.yaml instead of by path.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional


def _default_registry_path() -> Path:
    """Default registry path: ~/.soup/datasets.json."""
    return Path.home() / ".soup" / "datasets.json"


def _validate_name(name: str) -> None:
    """Validate dataset name — no path separators, null bytes, or empty."""
    if not name:
        raise ValueError("Dataset name must not be empty")
    if re.search(r'[/\\:\x00]', name):
        raise ValueError(
            f"Dataset name '{name}' must not contain "
            "path separators (/ \\ :) or null bytes"
        )


def load_registry(registry_path: Optional[Path] = None) -> dict[str, dict]:
    """Load the dataset registry. Returns empty dict if file doesn't exist."""
    path = registry_path or _default_registry_path()
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Registry file is corrupted ({path}): {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Registry file has unexpected format in {path}")
    return data


def _save_registry(registry: dict, registry_path: Optional[Path] = None) -> None:
    """Save the dataset registry to disk."""
    path = registry_path or _default_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=2, ensure_ascii=False)


def register_dataset(
    name: str,
    data_path: str,
    data_format: str,
    registry_path: Optional[Path] = None,
) -> None:
    """Register a dataset by name."""
    _validate_name(name)
    registry = load_registry(registry_path)
    registry[name] = {
        "path": data_path,
        "format": data_format,
    }
    _save_registry(registry, registry_path)


def unregister_dataset(
    name: str,
    registry_path: Optional[Path] = None,
) -> bool:
    """Unregister a dataset. Returns True if removed, False if not found."""
    registry = load_registry(registry_path)
    if name not in registry:
        return False
    del registry[name]
    _save_registry(registry, registry_path)
    return True


def resolve_dataset(
    name: str,
    registry_path: Optional[Path] = None,
) -> Optional[dict]:
    """Resolve a dataset name to its entry (path + format). None if not found.

    Callers must apply resolve + relative_to(cwd) before trusting the path.
    The returned path is validated for null bytes.
    """
    registry = load_registry(registry_path)
    entry = registry.get(name)
    if entry is not None:
        stored_path = entry.get("path", "")
        if "\x00" in stored_path:
            raise ValueError(
                f"Registry entry '{name}' contains null bytes in path"
            )
    return entry
