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
import time
from dataclasses import dataclass, field
from typing import Optional

#: A slot with a file modified this recently may belong to a concurrent run.
RECENT_SECONDS = 3600


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


def _walk(path: str) -> tuple[int, float]:
    """(bytes of regular files, newest mtime). Never follows links; unreadable entries skip."""
    total, newest = 0, 0.0
    for dirpath, _dirs, files in os.walk(path, followlinks=False):
        for name in files:
            try:
                info = os.lstat(os.path.join(dirpath, name))
            except OSError:
                continue
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
                newest = max(newest, info.st_mtime)
    return total, newest


def _stripes(slot_path: str, index) -> tuple[StripeFolder, ...]:
    from soup_cli.utils.layer_shard import stripe_dirs
    from soup_cli.utils.stripe_roots import stripe_folder_problem

    roots = tuple(index.stripe_roots)
    out = []
    for root, folder in zip(roots, stripe_dirs(slot_path, roots)[1:]):
        if stripe_folder_problem(folder, root) is None and os.path.isdir(folder):
            out.append(StripeFolder(folder, _walk(folder)[0]))
    return tuple(out)


def _same_base(name: str, slug: str) -> bool:
    return name == slug or name.split("@", 1)[0] == slug and "@" in name


def list_base_slots(
    model: str,
    cache_dir: Optional[str] = None,
    current_slot: Optional[str] = None,
    *,
    now: Optional[float] = None,
) -> list[CacheSlot]:
    """Every slot of ``model``'s base under the cache root, sorted by name.

    ``current_slot`` is the slot this invocation resolves to (path or name); it is kept.
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
    clock = time.time() if now is None else now
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
            size, newest = _walk(path)
            stripes: tuple[StripeFolder, ...] = ()
            reason = ""
            if entry.name == current:
                reason = "current slot"
            elif os.path.exists(os.path.join(path, _RESHARDING_INDEX_NAME)):
                reason = "resharding"
            else:
                try:
                    index = read_shard_index(path)
                except Exception:  # noqa: BLE001 - any unreadable index means "may be mid-shard"
                    index = None
                    reason = "incomplete, kept"
                if index is not None:
                    stripes = _stripes(path, index)
                    if clock - newest < RECENT_SECONDS:
                        reason = "modified within the last hour"
            slots.append(CacheSlot(entry.name, path, size, bool(reason), reason, stripes))
        except OSError:
            continue
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
