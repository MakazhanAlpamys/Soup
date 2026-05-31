"""Verify the integrity of a ``.can`` file (v0.26.0 Part E)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from soup_cli.cans.unpack import inspect_can, read_config


@dataclass(frozen=True)
class VerifyReport:
    manifest_ok: bool
    config_ok: bool
    message: str = ""


def verify_can(path: str) -> VerifyReport:
    """Run basic validation — manifest schema + config parseability."""
    can_path = Path(path)
    if not can_path.exists():
        return VerifyReport(manifest_ok=False, config_ok=False,
                            message=f"can not found: {path}")
    try:
        manifest = inspect_can(path)
    except Exception as exc:  # pydantic / yaml errors
        return VerifyReport(manifest_ok=False, config_ok=False,
                            message=f"manifest invalid: {exc}")
    try:
        read_config(path)
    except Exception as exc:
        return VerifyReport(manifest_ok=True, config_ok=False,
                            message=f"config invalid: {exc}")
    return VerifyReport(
        manifest_ok=True, config_ok=True,
        message=f"OK (format v{manifest.can_format_version}, name={manifest.name})",
    )
