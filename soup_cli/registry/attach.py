"""Attach eval/export artifacts to existing registry entries (v0.33.0 #35).

Thin wrappers around ``RegistryStore.add_artifact`` so the eval and export CLI
commands have a single, consistent entry point for post-hoc artifact attachment.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional


def attach_artifact(
    entry_id: str, *, path: str, kind: str, enforce_cwd: bool = True,
) -> Optional[int]:
    """Attach a file at ``path`` to a registry entry as ``kind``.

    Returns the inserted artifact rowid, or None on lookup failure. Raises
    ``ValueError`` / ``FileNotFoundError`` for explicit user-actionable errors
    so the CLI layer can render them.
    """
    from soup_cli.registry.store import RegistryStore

    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"artifact not found: {path}")

    with RegistryStore() as store:
        resolved_id = store.resolve(entry_id)
        if resolved_id is None:
            raise ValueError(f"registry entry not found: {entry_id}")
        return store.add_artifact(
            entry_id=resolved_id,
            kind=kind,
            path=str(artifact_path),
            enforce_cwd=enforce_cwd,
        )


def write_eval_json(
    output_path: str, *, payload: dict[str, Any],
) -> Path:
    """Write an eval payload as JSON, returning the resolved path.

    Writes are confined to cwd via realpath + commonpath check.
    """
    cwd_real = os.path.realpath(os.getcwd())
    out_real = os.path.realpath(output_path)
    try:
        common = os.path.commonpath([cwd_real, out_real])
    except ValueError as exc:
        raise ValueError(
            f"output path '{output_path}' is outside cwd"
        ) from exc
    if common != cwd_real:
        raise ValueError(f"output path '{output_path}' is outside cwd")

    out = Path(out_real)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out


def lookup_entry_by_output_dir(output_dir: str) -> Optional[str]:
    """Find a registry entry whose stored ``output`` directory matches.

    Used by ``soup export`` to auto-attach artifacts when the user did not
    pass ``--registry-id`` explicitly. Returns None if no match.
    """
    import warnings

    from soup_cli.registry.store import RegistryStore

    lookup_limit = 1000
    target_real = os.path.realpath(output_dir)
    with RegistryStore() as store:
        rows = store.list(limit=lookup_limit)
        if len(rows) >= lookup_limit:
            warnings.warn(
                f"lookup_entry_by_output_dir scanned only the most recent "
                f"{lookup_limit} registry entries; older entries with a "
                "matching output dir will be missed. Pass --registry-id "
                "explicitly to disambiguate.",
                ResourceWarning,
                stacklevel=2,
            )
        for entry in rows:
            stored = entry.get("output") or ""
            if not stored:
                continue
            try:
                if os.path.realpath(stored) == target_real:
                    return entry.get("id")
            except (OSError, ValueError):
                continue
    return None
