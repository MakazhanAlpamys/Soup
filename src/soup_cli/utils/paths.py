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
        # Windows junctions / mount-point reparse points do NOT report
        # S_ISLNK (they carry IO_REPARSE_TAG_MOUNT_POINT, not _SYMLINK), so
        # the check above misses them — yet shutil.rmtree would happily delete
        # the junction TARGET's contents. Refuse any reparse point on Windows.
        if os.name == "nt":
            reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            if getattr(st, "st_file_attributes", 0) & reparse:
                raise ValueError(
                    f"{field} must not be a reparse point / junction "
                    "(TOCTOU defence)"
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


def atomic_write_bytes(
    data: bytes,
    output_path: str,
    *,
    prefix: str = ".soup.",
    suffix: str = ".tmp",
    field: str = "output",
) -> str:
    """Atomically write ``data`` (bytes) to ``output_path`` under cwd containment.

    Binary sibling of :func:`atomic_write_text` (v0.71.3 #181 — used by the
    Annex XI/XII PDF renderer). Same TOCTOU-safe pipeline.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be bytes")
    enforce_under_cwd_and_no_symlink(output_path, field)
    parent = os.path.dirname(os.path.abspath(output_path)) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(bytes(data))
        os.replace(tmp_path, output_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return os.path.realpath(output_path)


def atomic_write_bytes_group(
    outputs: list[tuple[bytes, str, str]],
    *,
    removals: list[tuple[str, str]] | None = None,
) -> list[str]:
    """Publish a group of byte outputs atomically as one logical generation.

    Every payload is staged before an existing target is moved aside. If any
    replacement fails, newly published targets are removed and every previous
    target is restored. ``removals`` participate in the same transaction: they
    disappear only after every replacement succeeds and are restored on
    failure. This gives multi-file commands an all-new-or-all-old result
    instead of exposing a partial generation.
    """
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("outputs must be a non-empty list")

    prepared: list[tuple[bytes, str, str]] = []
    identities: set[str] = set()
    for item in outputs:
        if not isinstance(item, tuple) or len(item) != 3:
            raise TypeError("each output must be a (data, path, field) tuple")
        data, output_path, field = item
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("output data must be bytes")
        enforce_under_cwd_and_no_symlink(output_path, field)
        identity = os.path.normcase(os.path.realpath(output_path))
        if identity in identities:
            raise ValueError("output paths must be distinct")
        identities.add(identity)
        prepared.append((bytes(data), output_path, field))

    prepared_removals: list[tuple[str, str]] = []
    if removals is not None and not isinstance(removals, list):
        raise TypeError("removals must be a list")
    for item in removals or []:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("each removal must be a (path, field) tuple")
        removal_path, field = item
        enforce_under_cwd_and_no_symlink(removal_path, field)
        identity = os.path.normcase(os.path.realpath(removal_path))
        if identity in identities:
            raise ValueError("output and removal paths must be distinct")
        identities.add(identity)
        prepared_removals.append((removal_path, field))

    staged: dict[str, str] = {}
    backups: dict[str, str] = {}
    committed: set[str] = set()
    try:
        for data, output_path, _field in prepared:
            parent = os.path.dirname(os.path.abspath(output_path)) or "."
            os.makedirs(parent, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                prefix=".soup.group.", suffix=".tmp", dir=parent
            )
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(data)
            except Exception:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise
            staged[output_path] = tmp_path

        existing = [
            (output_path, field) for _data, output_path, field in prepared
        ] + prepared_removals
        for output_path, field in existing:
            if not os.path.lexists(output_path):
                continue
            st = os.lstat(output_path)
            if not stat.S_ISREG(st.st_mode):
                raise ValueError(f"{field} must be a regular file")
            parent = os.path.dirname(os.path.abspath(output_path)) or "."
            fd, backup_path = tempfile.mkstemp(
                prefix=".soup.backup.", suffix=".tmp", dir=parent
            )
            os.close(fd)
            try:
                os.replace(output_path, backup_path)
            except Exception:
                os.unlink(backup_path)
                raise
            backups[output_path] = backup_path

        for _data, output_path, _field in prepared:
            os.replace(staged[output_path], output_path)
            committed.add(output_path)
            del staged[output_path]
    except Exception as exc:
        rollback_failed = False
        for output_path in committed:
            if output_path in backups:
                continue
            try:
                os.unlink(output_path)
            except FileNotFoundError:
                pass
            except OSError:
                rollback_failed = True
        for output_path, backup_path in backups.items():
            try:
                os.replace(backup_path, output_path)
            except OSError:
                rollback_failed = True
        if rollback_failed:
            raise OSError("failed to restore a previous output generation") from exc
        raise
    finally:
        for tmp_path in staged.values():
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass

    for backup_path in backups.values():
        try:
            os.unlink(backup_path)
        except FileNotFoundError:
            pass

    return [os.path.realpath(output_path) for _data, output_path, _field in prepared]
