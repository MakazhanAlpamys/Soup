"""Soup template registry (v0.39.0 Part E).

Templates live as YAML files alongside this module. ``manifest.json``
declares which files exist; the loader resolves a template name to its
file content. Falls back to the inline ``TEMPLATES`` dict in
``soup_cli.config.schema`` for back-compat.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

# Cap to prevent symlink-based escapes or pathologically huge templates.
_MAX_TEMPLATE_BYTES = 256 * 1024


def _templates_dir() -> Path:
    return Path(__file__).resolve().parent


def _load_manifest() -> dict:
    path = _templates_dir() / "manifest.json"
    if not path.is_file():
        return {"templates": {}, "version": 1}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"templates": {}, "version": 1}
    if not isinstance(data, dict):
        return {"templates": {}, "version": 1}
    templates = data.get("templates")
    if not isinstance(templates, dict):
        templates = {}
    return {"templates": templates, "version": int(data.get("version", 1))}


def _validate_name(name: str) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("Template name must be a non-empty string")
    if "\x00" in name or "/" in name or "\\" in name or ".." in name:
        raise ValueError(f"Invalid template name: {name!r}")


def list_templates() -> list[str]:
    """Return all known template names (YAML manifest + inline fallback)."""
    names: set[str] = set()
    manifest = _load_manifest()
    names.update(manifest["templates"].keys())
    # Fold in inline templates for back-compat.
    try:
        from soup_cli.config.schema import TEMPLATES as _INLINE
        names.update(_INLINE.keys())
    except ImportError:
        pass
    return sorted(names)


def load_template(name: str) -> Optional[str]:
    """Resolve a template name to its YAML body. Returns None if unknown.

    Order:
      1. YAML file in this directory (per ``manifest.json``).
      2. Inline ``TEMPLATES`` dict in ``soup_cli.config.schema``.
    """
    _validate_name(name)
    manifest = _load_manifest()
    filename = manifest["templates"].get(name)
    if isinstance(filename, str) and filename:
        # Re-validate filename to defeat manifest tampering. A tampered
        # manifest pointing at "../secret.yaml" must not crash the loader —
        # silently fall back to inline so callers see a degraded but safe
        # result instead of a propagating ValueError.
        try:
            _validate_name(filename.replace(".yaml", ""))
        except ValueError:
            return _fallback_inline(name)
        if not filename.endswith(".yaml"):
            return _fallback_inline(name)
        templates_dir_str = os.path.realpath(str(_templates_dir()))
        path = _templates_dir() / filename
        # Containment: defeat crafted manifest entries with extra dots / mixed
        # separators that slip past _validate_name on Windows.
        try:
            real_path = os.path.realpath(str(path))
            if os.path.commonpath([real_path, templates_dir_str]) != templates_dir_str:
                return _fallback_inline(name)
        except (OSError, ValueError):
            return _fallback_inline(name)
        try:
            if path.is_file():
                size = path.stat().st_size
                if size > _MAX_TEMPLATE_BYTES:
                    return _fallback_inline(name)
                with path.open("r", encoding="utf-8") as f:
                    return f.read()
        except OSError:
            pass
    return _fallback_inline(name)


def _fallback_inline(name: str) -> Optional[str]:
    try:
        from soup_cli.config.schema import TEMPLATES as _INLINE
    except ImportError:
        return None
    return _INLINE.get(name)
