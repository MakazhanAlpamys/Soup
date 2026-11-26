"""``training.l2l`` activation spill: the RAM budget, the plan, and the spill file.

L2L keeps ``n_layers x k`` boundary activations between the forward and the backward
walk (``l2l_activations.ActivationStore``). When they do not fit the host-RAM budget,
the lowest layers go to an unbuffered file instead, as many as needed and no more.

* **Budget.** ``budget_of(available)``: the smaller of ``RAM_TIER_HEADROOM`` of the available
  memory and the available memory less ``RAM_RESERVE_BYTES``, where available is the smaller
  of psutil's available memory and the smallest remainder over the process's cgroup chain
  (own group up to the root of its namespace). A level's remainder is ``limit - (used -
  inactive_file)``: inactive file cache is reclaimable and does not count, page-locked host
  memory is shmem on the anon LRU and does. A level without a limit is skipped; a level whose
  limit cannot be read is reported and the other levels decide.
* **Plan.** Each resident chunk keeps its micro-batch position's capacity. A pinned chunk
  is its own allocation and torch rounds it up to the next power of two. Spill slots hold
  the largest chunk; the two staging rings (only when something spills) hold ``2 x ring``
  buffers of one sector-rounded slot plus the sector ``aligned_buffer`` over-allocates
  (#1531). The fewest layers spill that make the rest fit; when even every layer spilled
  does not fit, the run stops.
* **Spill directory.** ``training.l2l_spill_dir``, default ``<output>/_l2l_spill``.
  Refused: RAM-backed, stacked and network filesystems, a drive short of the file plus
  a margin, a volume where an unbuffered write and read do not round-trip, and (disk
  tier) the default directory on the weights' disk or on a disk that cannot be told
  apart from it.
* **Cleanup** is best-effort: the trainer closes the store on every exit it sees, and
  the next run in the same directory removes files of processes that are gone.

Standard library at the top level; torch is imported inside functions.
"""

from __future__ import annotations

import errno
import logging
import os
import re
import secrets
import shutil
import stat
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, Iterator, List, Optional, Sequence, Tuple

from rich.console import Console

from soup_cli.utils.l2l_activations import SECTOR_BYTES, sector_round_up

logger = logging.getLogger(__name__)
console = Console()

#: Staging buffers per direction for the trainer's store (the depth it was validated with).
#: ``l2l_activations.DEFAULT_RING`` stays for the other callers of the store.
TRAINER_RING = 4

#: The default spill directory, under the run's output directory.
SPILL_DIR_NAME = "_l2l_spill"
SPILL_FILE_PREFIX = "soup-l2l-spill-"
PROBE_FILE_PREFIX = "soup-l2l-probe-"
SPILL_FILE_GLOB = SPILL_FILE_PREFIX + "*.bin"
PROBE_FILE_GLOB = PROBE_FILE_PREFIX + "*.bin"
#: Our own file names, and only them: ``soup-l2l-{spill|probe}-<pid>-<hex>.bin``
#: (32 hex digits; 8 are still recognised so older stale files are swept).
_OWN_FILE = re.compile(r"^soup-l2l-(?:spill|probe)-(\d{1,10})-([0-9a-f]{8}(?:[0-9a-f]{24})?)\.bin$")

#: Roots the RAM budget and the volume checks read; module constants so tests can point
#: them at fake trees.
PROC_ROOT = "/proc"
CGROUP_V2_ROOT = "/sys/fs/cgroup"
CGROUP_V1_ROOT = "/sys/fs/cgroup/memory"
SYS_ROOT = "/sys"

#: A cgroup v1 limit at or above this is "no limit" (PAGE_COUNTER_MAX reads
#: 9223372036854771712 on x86-64).
UNLIMITED_BYTES = 1 << 60

#: What the rest of a training step may still allocate after the store is planned: the
#: budget never leaves less than this free. Chosen, not measured. One GPU run
#: (7B NF4, LoRA r 8, k 4, seq 512) saw about 183 MB of growth outside the store after
#: the budget was read; a 0.3 headroom alone left 130 MB there and the run was OOM-killed.
#: Above ``RAM_RESERVE_BYTES / (1 - RAM_TIER_HEADROOM)`` (1.79 GB) the 0.7 share decides.
RAM_RESERVE_BYTES = 512 << 20

#: The unbuffered round trip the volume check makes.
PROBE_BYTES = 1 << 20
#: Free space the spill file must leave: the larger of these two.
SPACE_MARGIN_MIN_BYTES = 1 << 30
SPACE_MARGIN_FRACTION = 0.10

REFUSED_FS = {
    "tmpfs": "it is tmpfs, which is RAM: a spill there frees no memory",
    "ramfs": "it is ramfs, which is RAM: a spill there frees no memory",
    "devtmpfs": "it is devtmpfs, which is RAM",
    "overlay": (
        "it is an overlay filesystem: the drive below and its unbuffered I/O path belong to "
        "the upper layer; spill to a bind-mounted local volume"
    ),
    "aufs": "it is aufs, a stacked filesystem; spill to a bind-mounted local volume",
    "squashfs": "it is squashfs, which is read-only",
}
NETWORK_FS_PREFIXES = (
    "nfs", "cifs", "smb", "smb3", "9p", "virtiofs", "fuse", "ceph", "glusterfs", "lustre",
    "beegfs", "gpfs", "afs", "davfs", "sshfs",
)
#: GetDriveTypeW results.
_DRIVE_REMOTE = 4
_DRIVE_RAMDISK = 6

_DISK_FULL_ERRNOS = {errno.ENOSPC} | ({errno.EDQUOT} if hasattr(errno, "EDQUOT") else set())
#: ERROR_HANDLE_DISK_FULL and ERROR_DISK_FULL; CPython maps both to ENOSPC as well.
_DISK_FULL_WINERRORS = {39, 112}

_FIX = (
    "Set training.l2l_spill_dir to a directory on a local drive with room, or lower "
    "gradient_accumulation_steps, data.max_length or batch_size, or free RAM so that "
    "fewer layers have to spill."
)


class SpillRefusedError(ValueError):
    """The activations cannot be placed: the message says what is wrong and what to change."""


class SpillDiskFullError(RuntimeError):
    """The spill drive ran out of space. Raised after the spill file is removed."""


def say(line: str, notify: Optional[Callable[[str], Any]] = None) -> None:
    """One console line, never wrapped (a path in it stays copyable)."""
    if notify is not None:
        notify(line)
    else:
        console.print(line, soft_wrap=True)


def _size(nbytes: Optional[float]) -> str:
    if nbytes is None:
        return "unknown"
    if nbytes >= 1e8:
        return f"{nbytes / 1e9:.2f} GB"
    if nbytes >= 1e5:
        return f"{nbytes / 1e6:.1f} MB"
    return f"{int(nbytes):,} B"


# ---------------------------------------------------------------------------
# the RAM budget
# ---------------------------------------------------------------------------
@dataclass
class CgroupLevel:
    """One limited cgroup level and the remainder it leaves."""

    path: str
    version: int
    limit: int
    limit_file: str
    used: int
    inactive_file: Optional[int]
    remainder: int
    stat_problem: Optional[str] = None

    def numbers(self) -> Dict[str, Any]:
        return {
            "limit": self.limit,
            "limit_file": self.limit_file,
            "used": self.used,
            "inactive_file": self.inactive_file,
            "remainder": self.remainder,
        }

    def describe(self) -> str:
        cache = (
            f"inactive_file {_size(self.inactive_file)}"
            if self.inactive_file is not None
            else f"memory.stat unreadable ({self.stat_problem}), so limit - used"
        )
        return (
            f"cgroup {self.path}: {self.limit_file} {_size(self.limit)}, "
            f"used {_size(self.used)}, {cache}"
        )


@dataclass
class RamBudget:
    """What the activations may hold in host RAM, and which level decided it."""

    available_bytes: Optional[int]
    budget_bytes: Optional[int]
    limiting_level: Optional[str]
    limiting_numbers: Dict[str, Any]
    psutil_available: Optional[int]
    levels: List[CgroupLevel] = field(default_factory=list)
    failed_levels: List[Dict[str, str]] = field(default_factory=list)

    def describe(self) -> str:
        if self.budget_bytes is None:
            text = "no RAM budget is known (psutil is missing and no cgroup sets a limit)"
        else:
            if self.limiting_level == "psutil":
                source = f"psutil available {_size(self.psutil_available)}"
            else:
                level = next(lv for lv in self.levels if lv.path == self.limiting_level)
                source = level.describe()
            text = (
                f"RAM budget {_size(self.budget_bytes)} = "
                f"{budget_rule(int(self.available_bytes or 0))}, limited by {source}"
            )
        if self.failed_levels:
            failed = "; ".join(f"{row['level']} ({row['reason']})" for row in self.failed_levels)
            text += f"; cgroup level(s) not read: {failed}"
        return text


def budget_of(available: int) -> int:
    """The activations' share of ``available`` host bytes: the smaller of
    ``RAM_TIER_HEADROOM`` of it and what is left after ``RAM_RESERVE_BYTES``, never below 0."""
    from soup_cli.utils.layer_stream import RAM_TIER_HEADROOM

    return max(0, min(int(available * RAM_TIER_HEADROOM), int(available) - RAM_RESERVE_BYTES))


def budget_rule(available: int) -> str:
    """How ``budget_of`` reached its number at ``available``, in the console's words."""
    from soup_cli.utils.layer_stream import RAM_TIER_HEADROOM

    if int(available * RAM_TIER_HEADROOM) <= int(available) - RAM_RESERVE_BYTES:
        return f"{RAM_TIER_HEADROOM:.0%} of {_size(available)} available"
    floor = ", floored at 0" if available < RAM_RESERVE_BYTES else ""
    return (
        f"{_size(available)} available less the {RAM_RESERVE_BYTES >> 20} MiB reserve kept for "
        f"the rest of the training step{floor}"
    )


def _read(path: str) -> Tuple[Optional[str], Optional[str]]:
    """``(text, None)``, ``(None, "absent")`` or ``(None, <why it cannot be read>)``."""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read().strip(), None
    except FileNotFoundError:
        return None, "absent"
    except OSError as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _hierarchy(proc_self_cgroup: str, *, v2: bool) -> Optional[List[str]]:
    """This process's cgroup path components: the unified line (``0::/path``) or the v1
    line whose controllers include ``memory``. None when there is no such line."""
    for line in proc_self_cgroup.splitlines():
        hierarchy, _, rest = line.partition(":")
        controllers, _, path = rest.partition(":")
        if v2 and hierarchy == "0" and controllers == "":
            return [part for part in path.split("/") if part]
        if not v2 and "memory" in controllers.split(","):
            return [part for part in path.split("/") if part]
    return None


def _chain(parts: List[str], root: str) -> List[str]:
    """Directories from the own group up to the root, innermost first."""
    return [
        root + "".join("/" + part for part in parts[:depth])
        for depth in range(len(parts), -1, -1)
    ]


class _LevelUnreadableError(Exception):
    """A cgroup level's limit or usage file exists but cannot be read or parsed."""


def _limit_value(directory: str, name: str) -> Optional[int]:
    """A limit in bytes; None when the file is absent or says "no limit"."""
    text, problem = _read(os.path.join(directory, name))
    if problem == "absent":
        return None
    if text is None:
        raise _LevelUnreadableError(f"{name} cannot be read: {problem}")
    if text == "max":
        return None
    if not text.isdigit():
        raise _LevelUnreadableError(f"{name} holds {text[:40]!r}, not a byte count")
    value = int(text)
    return None if value >= UNLIMITED_BYTES else value


def _counter(directory: str, name: str) -> int:
    text, problem = _read(os.path.join(directory, name))
    if text is None or not text.isdigit():
        raise _LevelUnreadableError(f"{name} cannot be read: {problem or repr(text[:40])}")
    return int(text)


def _stat_value(directory: str, key: str) -> Tuple[Optional[int], Optional[str]]:
    text, problem = _read(os.path.join(directory, "memory.stat"))
    if text is None:
        return None, f"memory.stat {problem}"
    for line in text.splitlines():
        name, _, value = line.partition(" ")
        if name == key and value.strip().isdigit():
            return int(value.strip()), None
    return None, f"memory.stat has no {key}"


def _level(directory: str, version: int) -> Optional[CgroupLevel]:
    """The level's limit and remainder; None when it sets no limit. Raises
    ``_LevelUnreadableError`` when a limit or usage file exists but cannot be read."""
    if version == 2:
        found = [
            (value, name)
            for name in ("memory.max", "memory.high")
            for value in [_limit_value(directory, name)]
            if value is not None
        ]
        if not found:
            return None
        limit, limit_file = min(found)
        used = _counter(directory, "memory.current")
        stat_key = "inactive_file"
    else:
        limit = _limit_value(directory, "memory.limit_in_bytes")
        if limit is None:
            return None
        limit_file = "memory.limit_in_bytes"
        used = _counter(directory, "memory.usage_in_bytes")
        stat_key = "total_inactive_file"
    inactive, stat_problem = _stat_value(directory, stat_key)
    if stat_problem is not None:
        logger.warning(
            "cgroup level %s: %s; its remainder is counted as limit - used", directory,
            stat_problem,
        )
    charged = max(0, used - (inactive or 0))
    return CgroupLevel(
        path=directory,
        version=version,
        limit=limit,
        limit_file=limit_file,
        used=used,
        inactive_file=inactive,
        remainder=max(0, limit - charged),
        stat_problem=stat_problem,
    )


def cgroup_levels(
    *, proc_root: Optional[str] = None, v2_root: Optional[str] = None,
    v1_root: Optional[str] = None,
) -> Tuple[List[CgroupLevel], List[Dict[str, str]]]:
    """Every limited accessible ancestor of this process's memory cgroups.

    Mountinfo maps hierarchy membership to its visible mount, including controller
    co-mounts and subtree roots. Explicit roots keep the fixture convention: membership
    is relative to that root. The conventional roots also work without mountinfo.
    """
    proc_root = PROC_ROOT if proc_root is None else proc_root
    text, _problem = _read(os.path.join(proc_root, "self", "cgroup"))
    mount_text, _problem = _read(os.path.join(proc_root, "self", "mountinfo"))
    mounts = _mountinfo_entries(mount_text or "")
    mount_points, stacked_points = set(), set()
    for mount in mounts:
        point = mount["mount_point"]
        if point in mount_points:
            stacked_points.add(point)
        mount_points.add(point)
    levels: List[CgroupLevel] = []
    failed: List[Dict[str, str]] = []
    roots = ((2, v2_root, CGROUP_V2_ROOT), (1, v1_root, CGROUP_V1_ROOT))
    for version, explicit_root, conventional_root in roots:
        parts = _hierarchy(text or "", v2=version == 2)
        if parts is None:
            continue
        memory_mounts = [
            mount for mount in mounts
            if (version == 2 and mount["fstype"] == "cgroup2")
            or (version == 1 and mount["fstype"] == "cgroup"
                and "memory" in mount["super_options"].split(","))
        ]
        if explicit_root is not None or not memory_mounts:
            directories = _chain(parts, explicit_root or conventional_root)
        else:
            directories = []
            for mount in memory_mounts:
                mount_parts = [part for part in mount["root"].split("/") if part]
                if mount_parts and all(part == ".." for part in mount_parts):
                    # An inherited mount above the cgroup namespace hides the names
                    # leading to its root. The mount itself is a proven ancestor;
                    # joining the membership suffix here would guess a sibling group.
                    candidates = [mount["mount_point"]]
                    logger.warning(
                        "cgroup membership /%s cannot be fully mapped through inherited "
                        "mount %s (root %s); only its known ancestor is read, so a "
                        "tighter nested memory limit may be unknown",
                        "/".join(parts), mount["mount_point"], mount["root"],
                    )
                elif parts[:len(mount_parts)] != mount_parts:
                    continue  # A bind mount of another group does not contain this process.
                else:
                    candidates = _chain(parts[len(mount_parts):], mount["mount_point"])
                for directory in candidates:
                    if any(
                        directory == point or directory.startswith(point.rstrip("/") + "/")
                        for point in stacked_points
                    ):
                        logger.warning(
                            "cgroup ancestor %s is ambiguous under stacked mounts; its "
                            "memory limit is unknown through this path", directory,
                        )
                        continue
                    visible = _mount_at(mounts, directory)
                    if visible is mount:
                        directories.append(directory)
                    else:
                        logger.warning(
                            "cgroup ancestor %s is hidden by another mount; its memory "
                            "limit cannot be read through this path", directory,
                        )
        for directory in dict.fromkeys(directories):
            if not os.path.isdir(directory):
                continue
            try:
                level = _level(directory, version)
            except _LevelUnreadableError as exc:
                logger.warning(
                    "cgroup level %s is limited but unreadable (%s); the budget comes "
                    "from the other levels", directory, exc,
                )
                failed.append({"level": directory, "reason": str(exc)})
                continue
            if level is not None:
                levels.append(level)
    return levels, failed


def ram_budget(
    *, proc_root: Optional[str] = None, v2_root: Optional[str] = None,
    v1_root: Optional[str] = None,
) -> RamBudget:
    """The activations' host-RAM budget: ``budget_of(min(psutil available, smallest cgroup
    remainder))``. Unknown (None) without psutil and without a limit."""
    from soup_cli.utils import layer_stream

    psutil_available = layer_stream.free_ram_bytes()
    levels, failed = cgroup_levels(proc_root=proc_root, v2_root=v2_root, v1_root=v1_root)
    available, level_name, numbers = psutil_available, None, {}
    if psutil_available is not None:
        level_name, numbers = "psutil", {"psutil_available": int(psutil_available)}
    tightest = min(levels, key=lambda lv: lv.remainder, default=None)
    if tightest is not None and (available is None or tightest.remainder < available):
        available, level_name, numbers = tightest.remainder, tightest.path, tightest.numbers()
    budget = None if available is None else budget_of(int(available))
    return RamBudget(
        available_bytes=None if available is None else int(available),
        budget_bytes=budget,
        limiting_level=level_name,
        limiting_numbers=numbers,
        psutil_available=psutil_available,
        levels=levels,
        failed_levels=failed,
    )


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------
def allocation_cost(request: int, *, pin: bool) -> int:
    """Host bytes one allocation of ``request`` bytes holds: a pinned block is rounded up
    to the next power of two by torch's caching host allocator; pageable memory is not."""
    from soup_cli.utils.layer_stream_runtime import _next_power_of_two

    if request <= 0:
        return 0
    return _next_power_of_two(int(request)) if pin else int(request)


@dataclass(frozen=True)
class SpillPlan:
    """Which layers stay resident, which spill, and what each side costs in real bytes."""

    n_layers: int
    k: int
    chunk_bytes: int
    ring: int
    pin: bool
    budget_bytes: Optional[int]
    spill_layers: int
    mb_bytes: Optional[Tuple[int, ...]] = None

    @property
    def slot_bytes(self) -> int:
        return sector_round_up(self.chunk_bytes)

    @property
    def chunk_cost(self) -> int:
        return allocation_cost(self.chunk_bytes, pin=self.pin)

    @property
    def resident_layer_cost(self) -> int:
        """One resident layer: the sum of its per-position allocation costs."""
        if self.mb_bytes is None:
            return self.k * self.chunk_cost
        return sum(allocation_cost(size, pin=self.pin) for size in self.mb_bytes)

    @property
    def ring_buffer_cost(self) -> int:
        """One staging buffer: a slot plus the sector ``aligned_buffer`` over-allocates."""
        return allocation_cost(self.slot_bytes + SECTOR_BYTES, pin=self.pin)

    @property
    def minimum_bytes(self) -> int:
        """Every layer spilled: the two staging rings only."""
        return 2 * self.ring * self.ring_buffer_cost

    @property
    def all_resident_bytes(self) -> int:
        return self.n_layers * self.resident_layer_cost

    @property
    def resident_layers(self) -> int:
        return self.n_layers - self.spill_layers

    @property
    def host_bytes(self) -> int:
        rings = self.minimum_bytes if self.spill_layers else 0
        return self.resident_layers * self.resident_layer_cost + rings

    @property
    def spill_file_bytes(self) -> int:
        return self.spill_layers * self.k * self.slot_bytes

    def as_dict(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "spill_layers": self.spill_layers,
            "n_layers": self.n_layers,
            "k": self.k,
            "chunk_bytes": self.chunk_bytes,
            "slot_bytes": self.slot_bytes,
            "ring": self.ring,
            "pin": self.pin,
            "host_bytes": self.host_bytes,
            "all_resident_bytes": self.all_resident_bytes,
            "minimum_bytes": self.minimum_bytes,
            "budget_bytes": self.budget_bytes,
            "spill_file_bytes": self.spill_file_bytes,
        }
        if self.mb_bytes is not None:
            record["mb_bytes"] = list(self.mb_bytes)
        return record


def plan_spill(
    budget_bytes: Optional[int],
    *,
    n_layers: int,
    k: int,
    chunk_bytes: int,
    mb_bytes: Optional[Sequence[int]] = None,
    ring: int = TRAINER_RING,
    pin: bool,
    budget_source: str = "",
) -> SpillPlan:
    """The fewest spilled layers whose plan fits ``budget_bytes`` of real host bytes.

    Nothing spills when everything fits or when no budget is known (None). Raises
    ``SpillRefusedError`` when even every layer spilled does not fit. ``budget_source``
    says where the budget came from, for that message. ``mb_bytes`` gives each micro-batch
    position's resident capacity; its pinned allocation is rounded independently. Slots
    and rings still use ``chunk_bytes``, the largest capacity.
    """
    sizes = tuple(int(size) for size in mb_bytes) if mb_bytes is not None else None
    if sizes is not None and (
        len(sizes) != int(k) or any(size < 0 or size > int(chunk_bytes) for size in sizes)
    ):
        raise ValueError("mb_bytes must contain k capacities between 0 and chunk_bytes")
    plan = SpillPlan(
        n_layers=int(n_layers), k=int(k), chunk_bytes=int(chunk_bytes), ring=int(ring),
        pin=bool(pin), budget_bytes=budget_bytes, spill_layers=0, mb_bytes=sizes,
    )
    if budget_bytes is None or plan.all_resident_bytes <= budget_bytes:
        return plan
    minimum = plan.minimum_bytes
    if budget_bytes < minimum:
        rounding = (
            f", the largest a {_size(plan.chunk_bytes)} request holding a {_size(plan.chunk_cost)} "
            "pinned block"
            if plan.chunk_cost != plan.chunk_bytes
            else ""
        )
        source = f" ({budget_source})" if budget_source else ""
        raise SpillRefusedError(
            f"training.l2l keeps {n_layers} layers x {k} micro-batches of activations in host "
            f"RAM ({_size(plan.all_resident_bytes)}{rounding}), and even with every layer "
            f"spilled to disk the 2 staging rings of {ring} buffers x "
            f"{_size(plan.ring_buffer_cost)} need {_size(minimum)}, but the RAM budget is "
            f"{_size(budget_bytes)}{source}. Lower gradient_accumulation_steps, "
            f"data.max_length or batch_size, or free RAM."
        )
    resident = min(n_layers - 1, (budget_bytes - minimum) // plan.resident_layer_cost)
    return SpillPlan(
        n_layers=plan.n_layers, k=plan.k, chunk_bytes=plan.chunk_bytes, ring=plan.ring,
        pin=plan.pin, budget_bytes=budget_bytes, spill_layers=n_layers - resident,
        mb_bytes=plan.mb_bytes,
    )


# ---------------------------------------------------------------------------
# file names, the stale sweep
# ---------------------------------------------------------------------------
def spill_file_name(prefix: str = SPILL_FILE_PREFIX) -> str:
    """A new file name of this process: ``<prefix><pid>-<32 hex>.bin`` (128 random bits)."""
    return f"{prefix}{os.getpid()}-{secrets.token_hex(16)}.bin"


def _plain_entry(info: os.stat_result, *, directory: bool = False) -> bool:
    """Regular entries only, excluding every Windows reparse point (not just symlinks)."""
    kind = stat.S_ISDIR if directory else stat.S_ISREG
    return kind(info.st_mode) and not (getattr(info, "st_file_attributes", 0) & 0x400)


@contextmanager
def _windows_directory_guard(root: str) -> Iterator[None]:
    """Keep every resolved path component from being renamed or replaced while sweeping."""
    import ctypes
    from ctypes import wintypes

    class AttributeTagInfo(ctypes.Structure):
        _fields_ = [("attributes", wintypes.DWORD), ("reparse_tag", wintypes.DWORD)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    ]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.GetFileInformationByHandleEx.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
    ]
    kernel32.GetFileInformationByHandleEx.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    ancestors = [root]
    while os.path.dirname(ancestors[-1]) != ancestors[-1]:
        ancestors.append(os.path.dirname(ancestors[-1]))
    handles = []
    try:
        for path in reversed(ancestors):
            # FILE_LIST_DIRECTORY participates in sharing checks; metadata-only access
            # does not. Share reads/writes but not delete/rename for every ancestor.
            # BACKUP_SEMANTICS opens directories, OPEN_REPARSE_POINT never follows them.
            handle = kernel32.CreateFileW(path, 0x1, 3, None, 3, 0x02200000, None)
            if handle == wintypes.HANDLE(-1).value:
                raise ctypes.WinError(ctypes.get_last_error())
            handles.append(handle)
            info = AttributeTagInfo()
            if not kernel32.GetFileInformationByHandleEx(
                handle, 9, ctypes.byref(info), ctypes.sizeof(info),
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            if info.attributes & 0x400 or not info.attributes & 0x10:
                raise OSError(
                    f"spill directory component is a reparse point or not a directory: {path}",
                )
        yield
    finally:
        for handle in reversed(handles):
            kernel32.CloseHandle(handle)


@contextmanager
def _spill_directory(directory: str) -> Iterator[Tuple[str, Optional[int], os.stat_result]]:
    """Anchor the actual directory; a pathname replacement must not redirect deletion."""
    root = os.path.realpath(directory)
    identity = os.lstat(root)
    if not _plain_entry(identity, directory=True):
        raise OSError(f"spill directory is not a plain directory: {root}")
    if os.name == "nt":
        with _windows_directory_guard(root):
            if not os.path.samestat(identity, os.lstat(root)):
                raise OSError(f"spill directory changed before it could be anchored: {root}")
            yield root, None, identity
    else:
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            if not os.path.samestat(identity, os.fstat(descriptor)):
                raise OSError(f"spill directory changed before it could be anchored: {root}")
            yield root, descriptor, identity
        finally:
            os.close(descriptor)


def _own_entries(
    root: str, descriptor: Optional[int],
) -> List[Tuple[str, int, os.stat_result]]:
    """Original basenames, PIDs and no-follow identities inside the anchored directory."""
    found = []
    with os.scandir(root if descriptor is None else descriptor) as entries:
        for entry in entries:
            match = _OWN_FILE.match(entry.name)
            if match is None:
                continue
            try:
                # DirEntry.stat on Windows omits st_ino/st_dev; lstat supplies the
                # actual identity. Still inspect the entry without following links.
                identity = os.lstat(os.path.join(root, entry.name)) if descriptor is None else (
                    entry.stat(follow_symlinks=False)
                )
                if descriptor is None and not _plain_entry(entry.stat(follow_symlinks=False)):
                    continue
            except OSError:
                continue
            if _plain_entry(identity):
                found.append((entry.name, int(match.group(1)), identity))
    return sorted(found)


def spill_files(directory: str) -> List[str]:
    """Absolute original paths of this module's regular spill and probe files, sorted."""
    try:
        with _spill_directory(directory) as (root, descriptor, _identity):
            return [
                os.path.join(root, name) for name, _pid, _stat in _own_entries(root, descriptor)
            ]
    except OSError:
        return []


def process_alive(pid: int) -> bool:
    """Whether ``pid`` is a running process. Unknown counts as alive (the file is kept).

    POSIX asks with signal 0; Windows opens the process and reads its exit code, since
    ``os.kill`` there terminates it.
    """
    if pid <= 0:
        return True
    if sys.platform == "win32":
        return _windows_process_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True  # PermissionError: alive, owned by someone else
    return True


def _windows_process_alive(pid: int) -> bool:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    query_limited_information, still_active, invalid_parameter = 0x1000, 259, 87
    handle = kernel32.OpenProcess(query_limited_information, False, int(pid))
    if not handle:
        return ctypes.get_last_error() != invalid_parameter
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return True
        return code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def _posix_remove_stale(
    root: str, descriptor: int, name: str, identity: os.stat_result,
) -> bool:
    """Claim by atomic rename, then dispose only a verified object in a private directory."""
    quarantine = f".soup-l2l-sweep-{os.getpid()}-{secrets.token_hex(8)}"
    os.mkdir(quarantine, 0o700, dir_fd=descriptor)  # Exclusive; never reuse a directory.
    quarantine_fd = None
    captured = False
    captured_path = os.path.join(root, quarantine, name)
    try:
        created = os.stat(quarantine, dir_fd=descriptor, follow_symlinks=False)
        quarantine_fd = os.open(
            quarantine, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor,
        )
        private = os.fstat(quarantine_fd)
        if (not stat.S_ISDIR(private.st_mode) or private.st_uid != os.geteuid()
                or stat.S_IMODE(private.st_mode) != 0o700
                or not os.path.samestat(created, private)
                or not os.path.samestat(
                    private, os.stat(quarantine, dir_fd=descriptor, follow_symlinks=False),
                )):
            raise OSError("the stale-file quarantine is not a private anchored directory")
        captured = True  # Set before the syscall: interruption may arrive as it returns.
        os.rename(name, name, src_dir_fd=descriptor, dst_dir_fd=quarantine_fd)
        current = os.stat(name, dir_fd=quarantine_fd, follow_symlinks=False)
        if _plain_entry(current) and os.path.samestat(identity, current):
            os.unlink(name, dir_fd=quarantine_fd)
            captured = False
            return True
        # A replacement won the claim race. link() restores without overwriting a
        # concurrent replacement and never follows a captured symlink. Unsupported
        # links, directories or occupied names remain preserved in quarantine.
        try:
            os.link(
                name, name, src_dir_fd=quarantine_fd, dst_dir_fd=descriptor,
                follow_symlinks=False,
            )
        except OSError:
            return False
        os.unlink(name, dir_fd=quarantine_fd)
        captured = False
        return False
    finally:
        if quarantine_fd is not None:
            os.close(quarantine_fd)
        try:
            os.rmdir(quarantine, dir_fd=descriptor)  # Empty only, even if capture failed.
        except OSError as exc:
            if captured:
                logger.warning(
                    "stale-file cleanup preserved an unverified or unremoved file at %s",
                    captured_path,
                )
            else:
                logger.warning("could not remove empty spill quarantine %s: %s",
                               os.path.join(root, quarantine), exc)


def _windows_dispose_file(kernel32: Any, handle: Any) -> None:
    """Mark the already-verified no-follow handle for deletion; never look up its path."""
    import ctypes
    from ctypes import wintypes

    class DispositionInfo(ctypes.Structure):
        _fields_ = [("delete_file", ctypes.c_ubyte)]

    kernel32.SetFileInformationByHandle.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
    ]
    kernel32.SetFileInformationByHandle.restype = wintypes.BOOL
    disposition = DispositionInfo(1)
    if not kernel32.SetFileInformationByHandle(
        handle, 4, ctypes.byref(disposition), ctypes.sizeof(disposition),
    ):
        raise ctypes.WinError(ctypes.get_last_error())


def _windows_remove_stale(path: str, identity: os.stat_result) -> bool:
    """Open the original basename without following reparses, verify, delete by handle."""
    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    ]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    # DELETE | FILE_READ_ATTRIBUTES, share reads/writes but not replacement. Reparse
    # points are opened themselves, never traversed (including directory junctions).
    handle = kernel32.CreateFileW(path, 0x10080, 3, None, 3, 0x02200000, None)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    descriptor = None
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY)
        current = os.fstat(descriptor)
        if not _plain_entry(current) or not os.path.samestat(identity, current):
            return False
        _windows_dispose_file(kernel32, handle)
        return True
    finally:
        if descriptor is None:
            kernel32.CloseHandle(handle)
        else:
            os.close(descriptor)  # The CRT owns the native handle after open_osfhandle.


def sweep_stale(
    directory: Optional[str], *, notify: Optional[Callable[[str], Any]] = None
) -> List[str]:
    """Remove spill and probe files of processes that are gone from ``directory``.

    Only our own names, only directly inside the directory, never through a link, never
    a file whose process still runs. Returns the removed paths and names them in one line.
    """
    removed = []
    if not directory:
        return removed
    try:
        with _spill_directory(directory) as (root, descriptor, directory_identity):
            for name, pid, identity in _own_entries(root, descriptor):
                if process_alive(pid):
                    continue
                path = os.path.join(root, name)
                try:
                    current_root = os.lstat(root)
                    if not _plain_entry(current_root, directory=True) or not os.path.samestat(
                        directory_identity, current_root,
                    ):
                        break
                    current = (
                        os.lstat(path) if descriptor is None
                        else os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                    )
                    if not _plain_entry(current) or not os.path.samestat(identity, current):
                        continue
                    if descriptor is None:
                        deleted = _windows_remove_stale(path, identity)
                    else:
                        deleted = _posix_remove_stale(root, descriptor, name, identity)
                    if not deleted:
                        continue
                except OSError as exc:
                    logger.warning("could not remove the stale spill file %s: %s", path, exc)
                    continue
                removed.append(path)
    except OSError:
        return removed
    if removed:
        from soup_cli.utils.terminal import for_terminal

        say(
            f"[dim]L2L: removed {len(removed)} spill file(s) left by runs that are no "
            f"longer alive: {for_terminal(', '.join(removed))}[/]",
            notify,
        )
    return removed


# ---------------------------------------------------------------------------
# the disk-full error
# ---------------------------------------------------------------------------
def is_disk_full(exc: Optional[BaseException]) -> Optional[OSError]:
    """The OSError in ``exc``'s cause chain that means "no space left", or None."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, OSError) and (
            exc.errno in _DISK_FULL_ERRNOS
            or getattr(exc, "winerror", None) in _DISK_FULL_WINERRORS
        ):
            return exc
        exc = exc.__cause__ or exc.__context__
    return None


def disk_full_message(
    path: str, *, file_bytes: int, during: str, written: int, left_behind: Optional[str],
    error: BaseException,
) -> str:
    gone = (
        f"the spill file could NOT be removed and is left at {left_behind}"
        if left_behind
        else "the spill file was removed"
    )
    return (
        f"training.l2l: the drive holding the activation spill directory "
        f"{os.path.dirname(path)} ran out of space while {during} the {_size(file_bytes)} "
        f"spill file ({file_bytes:,} bytes; {written:,} bytes written before the drive "
        f"refused); {gone}. Free space on that drive, or set training.l2l_spill_dir to a "
        f"local drive with room. OS error: {error}"
    )


def open_spill_file(path: str, size: int) -> Any:
    """The unbuffered spill file of ``size`` bytes at ``path``, zero-filled.

    A full drive raises ``SpillDiskFullError`` with the file already removed. The name is
    kept while the file is open: kill -9, an OOM kill or a power loss leave it for the
    next run's sweep, which is the documented contract.
    """
    from soup_cli.utils.l2l_activations import DirectSpillFile

    try:
        return DirectSpillFile(path, size)
    except OSError as exc:
        full = is_disk_full(exc)
        left = None
        # Remove only a file this attempt created: a pre-existing entry (FileExistsError
        # from the exclusive create) belongs to someone else and is never touched.
        if getattr(exc, "spill_file_created", False) and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                left = path
        if full is None:
            raise
        raise SpillDiskFullError(
            disk_full_message(
                path, file_bytes=size, during="creating", written=0, left_behind=left,
                error=full,
            )
        ) from exc


# ---------------------------------------------------------------------------
# the spill volume
# ---------------------------------------------------------------------------
def _mountinfo_entries(text: str) -> List[Dict[str, str]]:
    """Mount records shared by the cgroup and filesystem checks."""
    def unescape(value: str) -> str:
        return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), value)

    mounts = []
    for line in text.splitlines():
        left, sep, right = line.partition(" - ")
        fields, tail = left.split(), right.split()
        if not sep or len(fields) < 6 or len(tail) < 3:
            continue
        mounts.append({
            "root": unescape(fields[3]), "mount_point": unescape(fields[4]),
            "fstype": tail[0], "source": unescape(tail[1]), "super_options": tail[2],
        })
    return mounts


def _mount_at(mounts: List[Dict[str, str]], path: str) -> Optional[Dict[str, str]]:
    """The visible longest-prefix record, retaining its filesystem root for cgroups."""
    best: Optional[Dict[str, str]] = None
    for mount in mounts:
        mount_point = mount["mount_point"]
        inside = path == mount_point or path.startswith(mount_point.rstrip("/") + "/")
        if inside and (best is None or len(mount_point) >= len(best["mount_point"])):
            best = mount
    return best


def parse_mountinfo(text: str, path: str) -> Optional[Dict[str, str]]:
    """The mount holding ``path`` (resolved, ``/``-separated), from mountinfo text: the
    longest mount point that is a prefix of it."""
    best = _mount_at(_mountinfo_entries(text), path)
    return None if best is None else {
        key: best[key] for key in ("mount_point", "fstype", "source")
    }


def classify_fs(fstype: str) -> Optional[str]:
    """Why a filesystem type cannot hold a spill, or None."""
    name = fstype.lower()
    if name in REFUSED_FS:
        return REFUSED_FS[name]
    if name.startswith(NETWORK_FS_PREFIXES):
        return f"it is {fstype}, a network filesystem, not a local drive"
    return None


def _windows_drive_type(path: str) -> Optional[int]:
    import ctypes
    from ctypes import wintypes

    root = os.path.splitdrive(path)[0] + "\\"
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetDriveTypeW.argtypes = [wintypes.LPCWSTR]
    kernel32.GetDriveTypeW.restype = wintypes.UINT
    try:
        return int(kernel32.GetDriveTypeW(root))
    except OSError:
        return None


def _darwin_mount_info(path: str) -> Dict[str, Any]:
    """Native statfs64 locality, type, exact mount point and source device."""
    import ctypes

    class StatFs(ctypes.Structure):
        _fields_ = [
            ("bsize", ctypes.c_uint32), ("iosize", ctypes.c_int32),
            ("blocks", ctypes.c_uint64), ("bfree", ctypes.c_uint64),
            ("bavail", ctypes.c_uint64), ("files", ctypes.c_uint64),
            ("ffree", ctypes.c_uint64), ("fsid", ctypes.c_int32 * 2),
            ("owner", ctypes.c_uint32), ("type", ctypes.c_uint32),
            ("flags", ctypes.c_uint32), ("subtype", ctypes.c_uint32),
            ("fstype", ctypes.c_char * 16), ("mount_point", ctypes.c_char * 1024),
            ("source", ctypes.c_char * 1024), ("reserved", ctypes.c_uint32 * 8),
        ]

    libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    # Intel's legacy statfs symbol uses the old layout; arm64 is inode64-only.
    native_statfs = getattr(libc, "statfs$INODE64", None) or libc.statfs
    native_statfs.argtypes = [ctypes.c_char_p, ctypes.POINTER(StatFs)]
    native_statfs.restype = ctypes.c_int
    result = StatFs()
    if native_statfs(os.fsencode(path), ctypes.byref(result)):
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number), path)
    return {
        "flags": int(result.flags), "fstype": os.fsdecode(result.fstype),
        "mount_point": os.fsdecode(result.mount_point), "source": os.fsdecode(result.source),
    }


def _darwin_filesystem_refusal(path: str) -> Optional[str]:
    """Filesystem locality is distinct from the optional physical-disk identity check."""
    import plistlib
    import subprocess

    from soup_cli.utils.layer_stream import (
        _darwin_apfs_physical_stores,
        _darwin_bsd_whole_disk,
        _resolve_tool,
    )

    try:
        mount = _darwin_mount_info(path)
    except (OSError, AttributeError, ValueError) as exc:
        return f"its local filesystem and disk backing could not be verified ({exc})"
    reason = classify_fs(mount["fstype"])
    if reason is not None:
        return reason
    if not mount["flags"] & 0x1000:  # MNT_LOCAL, not a physical-disk identity heuristic.
        return f"it is {mount['fstype']}, a network filesystem, not a local drive"
    tool = _resolve_tool("diskutil", "/usr/sbin/diskutil")
    if tool is None:
        return "its disk backing could not be verified (diskutil is unavailable)"
    target = mount["source"]
    if _darwin_bsd_whole_disk(target) is None:
        target = mount["mount_point"]  # diskutil accepts exact mount points, not child paths.
    pending, seen = [target], set()
    while pending:
        target = pending.pop()
        if target in seen:
            continue
        seen.add(target)
        try:
            result = subprocess.run(
                [tool, "info", "-plist", target], capture_output=True, timeout=60,
                check=False, shell=False,
            )
            info = plistlib.loads(result.stdout) if result.returncode == 0 else None
        except (OSError, ValueError, TypeError, plistlib.InvalidFileException,
                subprocess.SubprocessError) as exc:
            return f"its disk backing could not be verified ({exc})"
        if not isinstance(info, dict):
            return "its disk backing could not be verified (diskutil returned no volume info)"
        reason = classify_fs(str(info.get("FilesystemType", "")))
        if reason is not None:
            return reason
        hints = " ".join(str(info.get(key, "")) for key in (
            "BusProtocol", "MediaName", "MediaType", "DeviceModel",
        )).lower()
        if info.get("DiskImage") is True or "disk image" in hints or "ram disk" in hints:
            return "it is a RAM disk or disk image whose disk backing cannot be verified"
        stores = _darwin_apfs_physical_stores(info)
        if not stores and not str(info.get("BusProtocol", "")).strip():
            return "its disk backing could not be verified (diskutil reports no backing protocol)"
        # A synthesized APFS volume may be disk-backed or live on a RAM disk. Follow
        # only the stores diskutil reports; missing whole-disk IDs alone prove neither.
        pending.extend("/dev/" + disk for disk in stores)
    return None


def filesystem_refusal(resolved: str) -> Optional[str]:
    """Why the volume cannot hold a spill: Linux mountinfo, Windows drive type, or
    Darwin's native locality flag and diskutil backing information."""
    text, _problem = _read(os.path.join(PROC_ROOT, "self", "mountinfo"))
    if text:
        mount = parse_mountinfo(text, resolved.replace(os.sep, "/"))
        if mount is not None:
            reason = classify_fs(mount["fstype"])
            if reason is not None:
                return f"{reason} (mounted at {mount['mount_point']} from {mount['source']})"
    if sys.platform == "win32":
        kind = _windows_drive_type(resolved)
        if kind == _DRIVE_REMOTE:
            return "it is a network drive, not a local drive"
        if kind == _DRIVE_RAMDISK:
            return "it is a RAM disk: a spill there frees no memory"
    if sys.platform == "darwin":
        return _darwin_filesystem_refusal(resolved)
    return None


def _round_trip(directory: str) -> Optional[str]:
    """None when a 1 MiB aligned unbuffered write and read return the same bytes; else why."""
    import torch

    from soup_cli.utils.l2l_activations import DirectSpillFile, aligned_buffer

    path = os.path.join(directory, spill_file_name(PROBE_FILE_PREFIX))
    try:
        probe = DirectSpillFile(path, PROBE_BYTES)
    except (OSError, ValueError) as exc:
        return f"an unbuffered file cannot be created there ({type(exc).__name__}: {exc})"
    try:
        sent = aligned_buffer(PROBE_BYTES, pin=False)
        sent.copy_(torch.arange(PROBE_BYTES, dtype=torch.int64).remainder(251).to(torch.uint8))
        got = aligned_buffer(PROBE_BYTES, pin=False)
        probe.write(0, sent)
        probe.read(0, got)
        if not torch.equal(sent, got):
            return "an unbuffered write and read there returned other bytes"
        return None
    except (OSError, ValueError) as exc:
        return f"an unbuffered write or read failed there ({type(exc).__name__}: {exc})"
    finally:
        probe.close(remove=True)


def space_margin_bytes(file_bytes: int) -> int:
    """Free space a spill file must leave on its drive."""
    return max(SPACE_MARGIN_MIN_BYTES, int(file_bytes * SPACE_MARGIN_FRACTION))


def _existing(path: str) -> str:
    probed = os.path.realpath(path)
    while not os.path.exists(probed) and os.path.dirname(probed) != probed:
        probed = os.path.dirname(probed)
    return probed


def _whole_disks(sys_path: str) -> List[str]:
    """Whole disks under a /sys/devices block entry: a partition's parent, or the disks
    under a device-mapper entry's slaves."""
    if os.path.exists(os.path.join(sys_path, "partition")):
        sys_path = os.path.dirname(sys_path)
    slaves = os.path.join(sys_path, "slaves")
    names = sorted(os.listdir(slaves)) if os.path.isdir(slaves) else []
    if not names:
        return [os.path.basename(sys_path)]
    disks: List[str] = []
    for name in names:
        disks += _whole_disks(os.path.realpath(os.path.join(slaves, name)))
    return sorted(set(disks))


def physical_disks(path: str) -> Tuple[Optional[FrozenSet[str]], str]:
    """``(whole disks, description)`` holding ``path``, or ``(None, why unknown)``.

    Linux: ``st_dev`` -> ``/sys/dev/block/<major>:<minor>`` -> the whole disk (a
    partition's parent, or the slaves of a device-mapper entry). Elsewhere unknown.
    """
    if not sys.platform.startswith("linux"):
        return None, f"the physical disk is not read on {sys.platform}"
    probed = _existing(path)
    try:
        st_dev = os.stat(probed).st_dev
    except OSError as exc:
        return None, f"cannot stat {probed}: {exc}"
    entry = os.path.join(SYS_ROOT, "dev", "block", f"{os.major(st_dev)}:{os.minor(st_dev)}")
    if not os.path.exists(entry):
        return None, f"no block device behind {probed}"
    try:
        disks = frozenset(_whole_disks(os.path.realpath(entry)))
    except OSError as exc:
        return None, f"cannot read {entry}: {exc}"
    return disks, ", ".join(sorted(disks))


def check_weights_disk(
    spill_dir: str, weight_dirs: Sequence[str], *, explicit: bool,
    notify: Optional[Callable[[str], Any]] = None,
) -> None:
    """Disk tier: the spill must not share a physical disk with the layer cache.

    Same disk or unknown with the default directory: refused. With an explicitly set
    ``training.l2l_spill_dir``: allowed with a warning.
    """
    from soup_cli.utils.terminal import for_terminal

    spill_disks, spill_why = physical_disks(spill_dir)
    same, unknown = [], []
    if spill_disks is None:
        unknown.append(f"{spill_dir}: {spill_why}")
    if not weight_dirs:
        unknown.append("the layer cache directories were not passed")
    for directory in weight_dirs:
        disks, why = physical_disks(directory)
        if disks is None:
            unknown.append(f"{directory}: {why}")
        elif spill_disks is not None and disks & spill_disks:
            same.append(f"{directory} ({why})")
    if not same and not unknown:
        return
    if same:
        problem = f"it is on the same physical disk as the layer cache: {'; '.join(same)}"
    else:
        problem = (
            "it could not be verified to be on another physical disk than the layer cache: "
            + "; ".join(unknown)
        )
    if not explicit:
        raise SpillRefusedError(
            f"training.l2l has to spill activations while the weights stream from disk, and "
            f"the default spill directory {spill_dir} cannot be used: {problem}. Set "
            f"training.l2l_spill_dir to a directory on another disk."
        )
    if same:
        note = (
            f"the spill directory {spill_dir} shares a physical disk with the layer cache "
            f"({'; '.join(same)}). Spill writes and weight reads then compete for one drive, "
            "which adds time to every step; how much depends on the drive."
        )
    else:
        note = (
            f"could not verify that the spill directory {spill_dir} is on another physical "
            f"disk than the layer cache ({'; '.join(unknown)}). If it shares that disk, "
            "spill and weight reads compete for it."
        )
    say(f"[yellow]L2L: {for_terminal(note)}[/]", notify)


def _allowed_roots() -> Tuple[str, ...]:
    """$HOME, $CWD and $TMPDIR, the roots a spill directory may live under."""
    import tempfile

    return (os.path.expanduser("~"), os.getcwd(), tempfile.gettempdir())


def _outside_allowed_roots(directory: str) -> Optional[str]:
    """The reason ``directory`` is refused, or None when it resolves under $HOME, $CWD or
    $TMPDIR (the bound ``layer_shard`` applies to the shard directory). ``realpath`` +
    ``commonpath`` through ``paths.is_under``, so symlinks and Windows short names resolve."""
    from soup_cli.utils.paths import is_under
    from soup_cli.utils.terminal import for_terminal

    if "\x00" in directory:
        return "the path contains a null byte"
    if any(is_under(directory, bound) for bound in _allowed_roots()):
        return None
    return (
        f"{for_terminal(directory)} resolves to {for_terminal(os.path.realpath(directory))}, "
        "which is outside $HOME, $CWD and $TMPDIR, the only places the spill may be written"
    )


def prepare_spill_dir(
    directory: str,
    *,
    explicit: bool,
    spill_layers: int,
    file_bytes: int,
    tier: str = "ram",
    weight_dirs: Sequence[str] = (),
    notify: Optional[Callable[[str], Any]] = None,
) -> str:
    """Create and check the spill directory; return its realpath. Every refusal names the
    directory, the reason and the fix."""

    def refuse(reason: str) -> SpillRefusedError:
        from soup_cli.utils.terminal import for_terminal

        return SpillRefusedError(
            f"training.l2l has to spill {spill_layers} layers of activations "
            f"({_size(file_bytes)}) to {for_terminal(directory)}, but {reason}. {_FIX}"
        )

    outside = _outside_allowed_roots(directory)
    if outside is not None:
        raise refuse(outside)
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as exc:
        raise refuse(f"the directory cannot be created ({type(exc).__name__}: {exc})") from exc
    resolved = os.path.realpath(directory)
    if not os.path.isdir(resolved):
        raise refuse("it is not a directory")
    reason = filesystem_refusal(resolved)
    if reason is not None:
        raise refuse(reason)
    if tier == "disk":
        check_weights_disk(resolved, weight_dirs, explicit=explicit, notify=notify)
    margin = space_margin_bytes(file_bytes)
    try:
        free = int(shutil.disk_usage(resolved).free)
    except OSError as exc:
        raise refuse(f"its free space cannot be read ({exc})") from exc
    if free < file_bytes + margin:
        raise refuse(
            f"its drive has {_size(free)} free and the file needs {_size(file_bytes)} plus a "
            f"margin of {_size(margin)} (the larger of 1 GiB and 10% of the file)"
        )
    problem = _round_trip(resolved)
    if problem is not None:
        raise refuse(problem)
    return resolved
