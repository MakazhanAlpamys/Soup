"""Read standard Linux cgroup memory bounds without importing the training stack."""

from __future__ import annotations

import os
from pathlib import Path


def _number(path: Path) -> int | None:
    try:
        value = int(path.read_text().strip())
        return value if value >= 0 else None
    except (OSError, ValueError):
        return None


def _directories() -> list[tuple[Path, int]]:
    """Include namespace roots and any visible containing cgroups."""
    roots = [(Path("/sys/fs/cgroup"), 2), (Path("/sys/fs/cgroup/memory"), 1)]
    result = list(roots)
    try:
        memberships = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        memberships = []
    for line in memberships:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        if parts[0] == "0" and not parts[1]:
            root, version = roots[0]
        elif "memory" in parts[1].split(","):
            root, version = roots[1]
        else:
            continue
        relative = Path(parts[2].lstrip("/"))
        if ".." in relative.parts:
            continue
        anchor = os.path.realpath(root)
        candidate = os.path.realpath(root / relative)
        try:
            if os.path.commonpath([anchor, candidate]) != anchor:
                continue
        except ValueError:
            continue
        current = Path(candidate)
        while True:
            result.append((current, version))
            if current == Path(anchor):
                break
            current = current.parent
    return list(dict.fromkeys(result))


def cgroup_memory_bounds(*, use_protection_budget: bool = False) -> tuple[int | None, int | None]:
    """Return visible hard capacity bounds and reclaim-aware headroom.

    Reclaim protection (v2 memory.low/min, v1 soft limits) is not capacity and
    does not constrain the default probes. An integration whose provider
    explicitly defines that protection as its baseline allocation may opt into
    ``use_protection_budget`` before passing the budget to the planner. This
    policy still honors smaller hard/ancestor limits; it is not a kernel cap.
    Clean file-LRU pages are reclaimable even after recent checkpoint hashing;
    shared, dirty/writeback and anonymous memory are not added back. Missing
    usage under a known limit supplies zero headroom,
    rather than treating the host's free memory as available to this container.
    An absent/unlimited controller leaves the host measurement unchanged.
    """
    limits, available = [], []
    for directory, version in _directories():
        hard = _number(directory / ("memory.max" if version == 2 else "memory.limit_in_bytes"))
        if version == 1 and hard is not None and hard >= 1 << 60:
            # v1 represents an unlimited controller as LONG_MAX rounded to pages.
            hard = None
        protected = 0
        if use_protection_budget:
            if version == 2:
                protected = max(
                    _number(directory / "memory.low") or 0,
                    _number(directory / "memory.min") or 0,
                )
            else:
                protected = _number(directory / "memory.soft_limit_in_bytes") or 0
                if protected >= 1 << 60:
                    protected = 0
        budgets = ([hard] if hard is not None else []) + ([protected] if protected else [])
        if not budgets:
            continue
        limit = min(budgets)
        usage = _number(directory / ("memory.current" if version == 2 else "memory.usage_in_bytes"))
        limits.append(limit)
        if usage is None:
            available.append(0)
            continue
        try:
            lines = (directory / "memory.stat").read_text().splitlines()
            stat = dict(line.split() for line in lines)
            prefix = "" if version == 2 else "total_"
            inactive = max(0, int(stat.get(prefix + "inactive_file", "0")))
            active = max(0, int(stat.get(prefix + "active_file", "0")))
            file_key = "file" if version == 2 else "total_cache"
            if file_key in stat:
                file_bytes = max(0, int(stat[file_key]))
                unavailable = sum(max(0, int(stat.get(name, "0"))) for name in (
                    ("shmem", "file_dirty", "file_writeback") if version == 2 else
                    ("total_shmem", "total_dirty", "total_writeback")))
                reclaimable = min(inactive + active, max(0, file_bytes - unavailable))
            else:
                reclaimable = inactive
        except (OSError, ValueError):
            reclaimable = 0
        charged = max(0, usage - min(usage, reclaimable))
        available.append(max(0, limit - charged))
    return (min(limits), min(available)) if limits else (None, None)
