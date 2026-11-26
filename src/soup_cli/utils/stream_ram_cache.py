"""Conservative host-memory admission and a fixed, byte-preserving SSD hotset."""

import os
import posixpath
import re
import sys
from pathlib import Path

_UNLIMITED = 2**60
_RUNTIME_RESERVE = 512 * 2**20


def _unescape(value):
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)


def _cgroup_paths(cgroup_file, mountinfo_file):
    memberships = []
    for line in cgroup_file.read_text().splitlines():
        _hierarchy, controllers, member = line.split(":", 2)
        if not controllers or "memory" in controllers.split(","):
            memberships.append((2 if not controllers else 1, posixpath.normpath(member)))
    found = []
    for line in mountinfo_file.read_text().splitlines():
        before, after = line.split(" - ", 1)
        fields, filesystem = before.split(), after.split()
        if filesystem[0] not in {"cgroup", "cgroup2"}:
            continue
        version = 2 if filesystem[0] == "cgroup2" else 1
        if version == 1 and "memory" not in filesystem[2].split(","):
            continue
        mount_root = posixpath.normpath(_unescape(fields[3]))
        root = os.path.realpath(_unescape(fields[4]))
        for kind, member in memberships:
            if kind != version:
                continue
            relative = posixpath.relpath(member, mount_root)
            if relative == ".." or relative.startswith("../"):
                continue
            own = os.path.realpath(os.path.join(root, *relative.split("/")))
            if os.path.commonpath([root, own]) != root or not os.path.isdir(own):
                raise ValueError("memory cgroup path escapes its visible mount")
            current = own
            while True:
                found.append((version, Path(current)))
                if current == root:
                    break
                current = os.path.dirname(current)
    if memberships and not found:
        raise ValueError("cannot locate the process memory cgroup")
    return list(dict.fromkeys(found))


def available_memory(
    *,
    host_available=None,
    cgroup_file=Path("/proc/self/cgroup"),
    mountinfo_file=Path("/proc/self/mountinfo"),
):
    """Host available RAM capped by visible cgroup hard/high/positive soft limits.

    Clean file cache can be reclaimed, including active file pages. Dirty,
    writeback, shared and unevictable pages are excluded conservatively. No swap
    capacity is admitted. A finite v1 soft limit is a planning hint, not a hard cap.
    """
    if host_available is None:
        from soup_cli.utils.layer_stream import free_ram_bytes

        host_available = free_ram_bytes()
    available = max(0, int(host_available or 0))
    report = {"host_available_bytes": available, "available_bytes": available, "constraints": []}
    cgroup_file, mountinfo_file = Path(cgroup_file), Path(mountinfo_file)
    if not cgroup_file.exists():
        if sys.platform == "linux":
            report.update(available_bytes=0, error="process cgroup information unavailable")
        return report
    try:
        for version, folder in _cgroup_paths(cgroup_file, mountinfo_file):
            names = (
                ("memory.max", "memory.high")
                if version == 2
                else ("memory.limit_in_bytes", "memory.soft_limit_in_bytes")
            )
            limits = []
            for name in names:
                path = folder / name
                if not path.exists():
                    continue
                text = path.read_text().strip()
                limit = _UNLIMITED if text == "max" else int(text)
                if name == "memory.soft_limit_in_bytes" and limit == 0:
                    continue
                if 0 <= limit < _UNLIMITED:
                    limits.append((name, limit))
            if not limits:
                continue
            usage_name = "memory.current" if version == 2 else "memory.usage_in_bytes"
            usage = max(0, int((folder / usage_name).read_text().strip()))
            stats_path = folder / "memory.stat"
            stats = (
                dict(
                    (name, int(value))
                    for name, value in (row.split() for row in stats_path.read_text().splitlines())
                )
                if stats_path.exists()
                else {}
            )
            if version == 1:

                def stat(name):
                    return max(0, stats.get("total_" + name, stats.get(name, 0)))

                clean = stat("cache") - sum(
                    stat(key) for key in ("shmem", "dirty", "writeback", "unevictable")
                )
            else:
                clean = stats.get("file", 0) - sum(
                    max(0, stats.get(key, 0))
                    for key in ("shmem", "file_dirty", "file_writeback", "unevictable")
                )
            reclaimable = min(usage, max(0, clean))
            for name, limit in limits:
                remaining = max(0, limit - usage + reclaimable)
                report["constraints"].append(
                    {
                        "path": str(folder / name),
                        "limit_bytes": limit,
                        "usage_bytes": usage,
                        "reclaimable_file_bytes": reclaimable,
                        "available_bytes": remaining,
                        "kind": "soft_hint" if "soft_limit" in name else name,
                    }
                )
                available = min(available, remaining)
        report["available_bytes"] = available
    except (OSError, ValueError, IndexError) as error:
        report.update(available_bytes=0, error=f"{type(error).__name__}: {error}")
    return report


def cache_budget(requested, available):
    """The requested allocation ceiling, with 30% slack and future runtime RAM."""
    return min(requested, max(0, available * 7 // 10 - _RUNTIME_RESERVE))


def pinned_allocation_bytes(plan, sizes, align):
    """Page-locked bytes the disk source's staging allocator will take for ``plan``.

    That allocator shifts each arena's base onto a sector boundary and, when the
    worst-case shift does not fit behind the arena's fill, takes an arena twice the
    size. Charge the worst case up front so an admitted hotset never exceeds the
    budget it was admitted under.
    """
    fills = [0] * len(plan.arena_sizes)
    for (arena_index, offset), size in zip(plan.placements, sizes):
        fills[arena_index] = max(fills[arena_index], offset + size)
    return sum(
        arena if fill + align - 1 <= arena else 2 * arena
        for arena, fill in zip(plan.arena_sizes, fills)
    )


def plan_hotset(spans, groups, depth, budget, *, pinned, excluded=()):
    """Center repeated groups; charge actual arena/alignment allocation, not payload.

    Singleton/small groups already fit entirely in ordinary staging. The fixed
    hotset is loaded once and never evicted or modified during training.
    """
    from soup_cli.utils.safetensors_reader import SECTOR_BYTES

    selected, allocated = [], 0
    for group in sorted(set(groups)):
        members = [
            idx for idx, value in enumerate(groups) if value == group and idx not in excluded
        ]
        if len(members) <= depth:
            continue
        center = (members[0] + members[-1]) / 2
        for idx in sorted(members, key=lambda value: (abs(value - center), value)):
            # The cache allocates the selected layers in sorted order.
            sizes = [spans[value] for value in sorted([*selected, idx])]
            if pinned:
                from soup_cli.utils.layer_stream_runtime import plan_pinned_arenas

                plan = plan_pinned_arenas(sizes, align=SECTOR_BYTES)
                cost = pinned_allocation_bytes(plan, sizes, SECTOR_BYTES)
            else:
                cost = sum(sizes) + len(sizes) * SECTOR_BYTES
            if cost <= budget:
                selected.append(idx)
                allocated = cost
    return {"indices": sorted(selected), "allocated_bytes": allocated}
