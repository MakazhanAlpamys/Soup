"""Cross-platform path containment helpers.

Uses ``os.path.realpath`` + ``os.path.commonpath`` rather than
``Path.resolve() + relative_to()`` to survive Windows 8.3 short names
(e.g. ``C:\\Users\\RUNNER~1``) that can appear in one of the two paths
but not the other on older Python.

Why this lives in one place: the same helper is needed by autopilot,
registry, cans, eval-gate, quant-check, and trace harvesting. Keeping it
in a single module guarantees a single behaviour across the CLI.
"""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path
from typing import Union


def is_under(path: Union[str, Path], base: Union[str, Path]) -> bool:
    """Return True when ``path`` resolves inside ``base``."""
    try:
        resolved_path = os.path.realpath(str(path))
        resolved_base = os.path.realpath(str(base))
    except (OSError, ValueError):
        return False
    if os.name == "nt":
        resolved_path = resolved_path.lower()
        resolved_base = resolved_base.lower()
    try:
        return os.path.commonpath([resolved_path, resolved_base]) == resolved_base
    except ValueError:
        return False


def is_under_cwd(path: Union[str, Path]) -> bool:
    """Whether ``path`` is inside the current working directory."""
    return is_under(path, Path.cwd())


def enforce_under_cwd_and_no_symlink(path: str, field: str) -> str:
    """Apply cwd containment + ``os.lstat + S_ISLNK`` rejection (TOCTOU defence).

    Shared helper for v0.53.1 export / merge / advanced-GGUF dispatch.
    Mirrors v0.33.0 #22 / v0.43.0 Part C / v0.46.0 Part A / v0.47.0 TOCTOU
    policy: rejects symlinks at the target path before any open/write so a
    pre-placed symlink cannot redirect a write to ``/etc/cron.d``.
    """
    if not isinstance(path, str):
        raise TypeError(f"{field} must be str, got {type(path).__name__}")
    if not path:
        raise ValueError(f"{field} must be non-empty")
    if "\x00" in path:
        raise ValueError(f"{field} must not contain null bytes")
    if not is_under_cwd(path):
        raise ValueError(
            f"{field} {os.path.basename(path)!r} must stay under cwd"
        )
    if os.path.lexists(path):
        try:
            st = os.lstat(path)
        except OSError as exc:
            raise ValueError(
                f"{field} unreadable: {type(exc).__name__}"
            ) from exc
        if stat.S_ISLNK(st.st_mode):
            raise ValueError(
                f"{field} must not be a symlink (TOCTOU defence)"
            )
    return path


def atomic_write_text(
    text: str,
    output_path: str,
    *,
    prefix: str = ".soup.",
    suffix: str = ".tmp",
    field: str = "output",
) -> str:
    """Atomically write ``text`` to ``output_path`` under cwd containment.

    Pipeline: ``enforce_under_cwd_and_no_symlink`` -> ``mkstemp`` in the
    parent dir -> write -> ``os.replace`` -> best-effort cleanup of the
    tmp file on failure. Returns the realpath of the written file.

    Centralised in v0.59.0 from four separate copies in
    ``bom.py`` / ``attest.py`` / ``annex_xi.py`` / ``repro_receipt.py``
    so the TOCTOU defence stays single-source-of-truth (code-review
    HIGH fix mirrors v0.40.6 / v0.53.5 peft_wiring centralisation policy).
    """
    enforce_under_cwd_and_no_symlink(output_path, field)
    parent = os.path.dirname(os.path.abspath(output_path)) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp_path, output_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return os.path.realpath(output_path)
