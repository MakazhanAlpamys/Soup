"""Stale layer-shard cache slots (#2196).

A slot is ``<model_slug>@<commit>`` (or the bare slug for a legacy cache) under the cache root.
A base update leaves the old slot behind. This module lists the other slots of one base and
removes the unreferenced ones — only when the caller passes ``confirm=True``. Training never
calls :func:`prune_slots`. Stripe folders live outside the cache root: reported, never deleted.
"""

from __future__ import annotations

import os
import shutil
import stat
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class StripeFolder:
    path: str
    bytes: int


@dataclass(frozen=True)
class CacheSlot:
    name: str
    path: str
    bytes: int
    referenced: bool
    reason: str = ""
    stripes: tuple[StripeFolder, ...] = field(default_factory=tuple)


def _bytes(path: str) -> int:
    """Bytes of regular files. Never follows links; unreadable entries are skipped."""
    total = 0
    for dirpath, _dirs, files in os.walk(path, followlinks=False):
        for name in files:
            try:
                info = os.lstat(os.path.join(dirpath, name))
            except OSError:
                continue
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
    return total


def _writing(path: str) -> bool:
    """A sharder temp file (``.soup.*.tmp``) is in the slot: a write may be in progress."""
    try:
        return any(n.startswith(".soup.") and n.endswith(".tmp") for n in os.listdir(path))
    except OSError:
        return True


def _stripes(slot_path: str, index) -> tuple[StripeFolder, ...]:
    from soup_cli.utils.layer_shard import stripe_dirs
    from soup_cli.utils.stripe_roots import stripe_folder_problem

    roots = tuple(index.stripe_roots)
    out = []
    for root, folder in zip(roots, stripe_dirs(slot_path, roots)[1:]):
        if stripe_folder_problem(folder, root) is None and os.path.isdir(folder):
            out.append(StripeFolder(folder, _bytes(folder)))
    return tuple(out)


def _same_base(name: str, slug: str) -> bool:
    return name == slug or name.split("@", 1)[0] == slug and "@" in name


def list_base_slots(
    model: str,
    cache_dir: Optional[str] = None,
    current_slot: Optional[str] = None,
) -> list[CacheSlot]:
    """Every slot of ``model``'s base under the cache root, sorted by name.

    ``current_slot`` is the slot this invocation resolves to (path or name); it is kept.
    So is the newest complete slot (the ``index.json`` with the latest mtime), because
    the commit a later run pins is not known here and re-sharding a large base takes hours.
    """
    from soup_cli.utils.layer_shard import (
        _RESHARDING_INDEX_NAME,
        model_slug,
        read_shard_index,
        resolve_cache_root,
    )

    root = resolve_cache_root(cache_dir)
    slug = model_slug(model)
    current = os.path.basename(os.path.normpath(current_slot)) if current_slot else None
    mtimes: dict[str, float] = {}
    slots: list[CacheSlot] = []
    try:
        entries = sorted(os.scandir(root), key=lambda e: e.name)
    except OSError:
        return []
    for entry in entries:
        try:
            if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                continue
            if not _same_base(entry.name, slug):
                continue
            path = os.path.realpath(entry.path)
            if os.path.commonpath([root, path]) != root or os.path.dirname(path) != root:
                continue
            size = _bytes(path)
            stripes: tuple[StripeFolder, ...] = ()
            reason = ""
            if entry.name == current:
                reason = "current slot"
            elif os.path.exists(os.path.join(path, _RESHARDING_INDEX_NAME)):
                reason = "resharding"
            elif _writing(path):
                reason = "write in progress, kept"
            else:
                try:
                    index = read_shard_index(path)
                except Exception:  # noqa: BLE001 - any unreadable index means "may be mid-shard"
                    index = None
                    reason = "incomplete, kept"
                if index is not None:
                    stripes = _stripes(path, index)
                    mtimes[entry.name] = os.stat(os.path.join(path, "index.json")).st_mtime
            slots.append(CacheSlot(entry.name, path, size, bool(reason), reason, stripes))
        except OSError:
            continue
    if mtimes:
        newest = max(mtimes, key=lambda n: (mtimes[n], n))
        slots = [
            CacheSlot(x.name, x.path, x.bytes, True, "newest complete slot", x.stripes)
            if x.name == newest and not x.referenced else x
            for x in slots
        ]
    return slots


def prune_slots(
    model: str,
    cache_dir: Optional[str] = None,
    current_slot: Optional[str] = None,
    *,
    confirm: bool,
) -> list[tuple[CacheSlot, Optional[str]]]:
    """Unreferenced slots and what happened to each: ``None`` = removed (or, with
    ``confirm=False``, would be removed), else the error. Re-scans, so a slot that became
    referenced since the caller last looked is kept."""
    results: list[tuple[CacheSlot, Optional[str]]] = []
    for slot in list_base_slots(model, cache_dir, current_slot):
        if slot.referenced:
            continue
        if not confirm:
            results.append((slot, None))
            continue
        try:
            if os.path.islink(slot.path):
                raise OSError("is a link")
            shutil.rmtree(slot.path)
            results.append((slot, None))
        except OSError as exc:
            results.append((slot, str(exc)))
    return results
