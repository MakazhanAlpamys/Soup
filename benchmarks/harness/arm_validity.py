#!/usr/bin/env python3
"""Validity checks for one benchmark arm (F1).

Wrap an arm in ``ArmWatch`` and read ``watch.record`` afterwards: the arm's
identity, one entry per check under ``checks`` and the arm's overall
``outcome``. Each check ends in ok, void (the sensor answered and the arm is
contaminated) or unknown (the sensor gave no answer). A sensor that raises or
returns None makes its check unknown, never ok (ADR 0001).

The record also holds a box stamp from before and after the arm and a GPU clock
record. Both are for the reader and never change the arm's outcome.

The module also gives the A-B-B-A run order of a block (``block_order``) and the
repeats rule (ADR 0003): ``repeats`` gives sigma, the mean of A and the
blocks N a 10% effect needs, ``blocks_needed`` gives N alone, and
``blocks_to_add`` says how many blocks to run next: the 5-block pilot, then up
to N, then one top-up if N recomputed from all blocks is larger.

Sensors are callables passed in by the caller, with real defaults per OS. The
Linux defaults need no root and no third-party packages. The Windows defaults
port the reference ``l2l_box.py`` (upstream ``benchmarks/harness``) and read
per-process counters through psutil. On macOS every check is unknown.

Benchmark harness code, not shipped. Standard library only, except psutil on
Windows, imported only when the sensor runs.
"""

from __future__ import annotations

import ctypes
import math
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
)

_T = TypeVar("_T")

OK = "ok"
VOID = "void"
UNKNOWN = "unknown"

LINUX = "Linux"
WINDOWS = "Windows"

#: Boot-time minus monotonic time grows by the time spent suspended. Growth past
#: this many seconds during an arm means the machine slept.
SLEEP_OFFSET_LIMIT_S = 1.0

SUSPEND_SUCCESS_PATH = "/sys/power/suspend_stats/success"

POWER_SUPPLY_DIR = "/sys/class/power_supply"

#: Seconds between power samples during an arm.
POWER_INTERVAL_S = 2.0

#: Windows: a wall-clock gap between two samples longer than this many seconds
#: means the machine slept (rule V5 of the reference ``l2l_box.SuspendWatch``).
WALL_CLOCK_GAP_LIMIT_S = 30.0

#: Windows: seconds between wall-clock samples during an arm.
WALL_CLOCK_INTERVAL_S = 2.0

#: Windows ``SYSTEM_POWER_STATUS`` values: the AC line states, and the battery
#: flag bit for "no system battery" and the flag value for "status unknown".
AC_LINE_OFFLINE = 0
AC_LINE_ONLINE = 1
NO_SYSTEM_BATTERY = 128
BATTERY_FLAG_UNKNOWN = 255

#: Foreign reads of this many bytes or more from the disk under test void an arm
#: (validity row V6).
FOREIGN_READ_LIMIT_BYTES = 1_000_000_000

PROC_DIR = "/proc"

DISKSTATS_PATH = "/proc/diskstats"
SYS_BLOCK_DIR = "/sys/block"
NOT_PHYSICAL_DISK_PREFIXES = ("loop", "ram", "zram")

#: /proc/diskstats counts 512-byte sectors whatever the hardware sector size.
DISKSTATS_SECTOR_BYTES = 512

MEMINFO_PATH = "/proc/meminfo"

#: Seconds between GPU clock samples during an arm.
GPU_CLOCK_INTERVAL_S = 2.0

#: The GPU the nvidia-smi sensors read when CUDA_VISIBLE_DEVICES is unset.
DEFAULT_NVIDIA_GPU_ID = "0"

#: The variable that maps the workload's cuda:0 to a physical GPU.
CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"

#: Seconds before an nvidia-smi call counts as failed.
NVIDIA_SMI_TIMEOUT_S = 20.0

#: Bits of nvidia-smi's ``clocks_event_reasons.active`` mask, from nvml.h.
CLOCK_EVENT_REASONS = {
    0x1: "gpu_idle",
    0x2: "applications_clocks_setting",
    0x4: "sw_power_cap",
    0x8: "hw_slowdown",
    0x10: "sync_boost",
    0x20: "sw_thermal_slowdown",
    0x40: "hw_thermal_slowdown",
    0x80: "hw_power_brake_slowdown",
    0x100: "display_clock_setting",
}

#: Every check an arm record carries, in record order.
CHECKS = ("suspend", "power", "foreign_reads")

#: The arms of one block in run order. Linear drift over the block cancels.
BLOCK_ORDER = ("A", "B", "B", "A")

#: (z for alpha 0.05 two-sided + z for power 0.8) squared, as in the NIST
#: e-Handbook sample size formula (ADR 0003).
SAMPLE_SIZE_FACTOR = 7.85

#: The effect a comparison must detect, as a fraction of the mean of A.
EFFECT_FRACTION = 0.1

#: The repeats rule never asks for fewer blocks than this.
MIN_BLOCKS = 2

#: Blocks in the pilot. They estimate sigma and count toward N.
PILOT_BLOCKS = 5


class SensorUnavailableError(Exception):
    """A sensor this OS or environment cannot provide. Its message is the whole reason."""


def _not_implemented_on(system: str) -> Callable[[], None]:
    """A sensor for an OS the module has no checks for."""

    def sensor() -> None:
        raise SensorUnavailableError(f"not implemented on {system}")

    return sensor


def system_name(platform: str) -> str:
    """The OS a ``sys.platform`` value names: Linux, Windows, macOS, or the value itself."""
    if platform.startswith("linux"):
        return LINUX
    return {"win32": WINDOWS, "darwin": "macOS"}.get(platform, platform)


def sleep_offset() -> Optional[float]:
    """Boot-time minus monotonic time in seconds; None where the OS has no boot-time clock."""
    boottime = getattr(time, "CLOCK_BOOTTIME", None)
    if boottime is None:
        return None
    return time.clock_gettime(boottime) - time.clock_gettime(time.CLOCK_MONOTONIC)


def suspend_count() -> Optional[int]:
    """The kernel's count of successful suspends since boot."""
    with open(SUSPEND_SUCCESS_PATH, encoding="ascii") as handle:
        return int(handle.read().strip())


class PowerReading(TypedDict):
    """One read of the power supplies: whether a battery is present, and which
    non-battery supplies are online."""

    battery: bool
    online: Dict[str, bool]


def _supply_attribute(supply: str, name: str) -> Optional[str]:
    """One sysfs attribute of a power supply, or None if the supply lacks it."""
    try:
        with open(os.path.join(supply, name), encoding="ascii") as handle:
            return handle.read().strip()
    except FileNotFoundError:
        return None


def power_supplies() -> PowerReading:
    """A ``PowerReading`` from the kernel's power supplies.

    Follows the kernel's ``power_supply_is_system_supplied()``: supplies with
    scope "Device" (a mouse battery, say) are skipped, and ``online`` covers
    every remaining supply that is not a battery and has an ``online`` value.
    ``battery`` says whether any remaining supply is a battery.
    """
    battery = False
    online: Dict[str, bool] = {}
    for name in sorted(os.listdir(POWER_SUPPLY_DIR)):
        supply = os.path.join(POWER_SUPPLY_DIR, name)
        if _supply_attribute(supply, "scope") == "Device":
            continue
        kind = _supply_attribute(supply, "type")
        if kind == "Battery":
            battery = True
            continue
        state = _supply_attribute(supply, "online")
        if kind is not None and state is not None:
            online[name] = int(state) != 0
    return {"battery": battery, "online": online}


def _whole_disks(sys_path: str) -> List[str]:
    """Whole disks under a /sys/devices block entry: a partition's parent, or the
    disks under a device-mapper entry's slaves."""
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


def disk_device(path: str) -> Optional[str]:
    """The whole disk that holds ``path``, as named in /proc/diskstats.

    None if the file system has no block device (tmpfs, btrfs subvolumes,
    network mounts). Raises ValueError if the device spans several disks.
    """
    st_dev = os.stat(path).st_dev
    entry = f"/sys/dev/block/{os.major(st_dev)}:{os.minor(st_dev)}"
    if not os.path.exists(entry):
        return None
    disks = _whole_disks(os.path.realpath(entry))
    if len(disks) != 1:
        raise ValueError(f"{path} spans several disks: {', '.join(disks)}")
    return disks[0]


def disk_reads(device: str) -> Optional[int]:
    """Bytes read from ``device`` since boot, from /proc/diskstats; None if it is not listed."""
    with open(DISKSTATS_PATH, encoding="ascii") as handle:
        for line in handle:
            fields = line.split()
            if fields[2] == device:
                return int(fields[5]) * DISKSTATS_SECTOR_BYTES
    return None


def _stacks_on_other_devices(name: str) -> bool:
    slaves = os.path.join(SYS_BLOCK_DIR, name, "slaves")
    return os.path.isdir(slaves) and bool(os.listdir(slaves))


def other_disk_reads(device: str) -> Optional[int]:
    """Bytes read since boot from every physical disk except ``device``; 0 if it is the only one.

    A physical disk is a /sys/block entry that stacks on no other device, so a
    device-mapper or RAID device is not counted on top of the disks under it.
    Loop, RAM and zram devices are left out.
    """
    disks = {
        name
        for name in os.listdir(SYS_BLOCK_DIR)
        if name != device
        and not name.startswith(NOT_PHYSICAL_DISK_PREFIXES)
        and not _stacks_on_other_devices(name)
    }
    total = 0
    with open(DISKSTATS_PATH, encoding="ascii") as handle:
        for line in handle:
            fields = line.split()
            if fields[2] in disks:
                total += int(fields[5]) * DISKSTATS_SECTOR_BYTES
    return total


class ProcessRead(TypedDict):
    name: str
    read_bytes: int


class ProcessReads(TypedDict):
    """One scan of /proc.

    ``own`` is the storage bytes read by this process (its waited-for children
    included) and by its live descendants. ``others`` holds every other process
    of this user whose /proc/<pid>/io is readable, by pid. ``uninspectable``
    counts the processes whose counters were denied or belong to other users.
    """

    own: int
    others: Dict[int, ProcessRead]
    uninspectable: int


def _io_read_bytes(pid: int) -> int:
    with open(os.path.join(PROC_DIR, str(pid), "io"), encoding="ascii") as handle:
        for line in handle:
            if line.startswith("read_bytes:"):
                return int(line.split()[1])
    raise ValueError(f"{PROC_DIR}/{pid}/io has no read_bytes")


def _name_and_parent(pid: int) -> Tuple[str, int]:
    path = os.path.join(PROC_DIR, str(pid), "stat")
    with open(path, encoding="utf-8", errors="replace") as handle:
        stat = handle.read()
    name_end = stat.rindex(")")
    return stat[stat.index("(") + 1 : name_end], int(stat[name_end + 2 :].split()[1])


def process_reads() -> ProcessReads:
    """A ``ProcessReads`` scan of /proc. Needs no root; denied counters are counted, not skipped.

    Raises PermissionError if a process of our own tree is unreadable: Linux
    denies an exited child's counter until it is reaped, and its reads would
    otherwise count as foreign.
    """
    names: Dict[int, str] = {}
    parents: Dict[int, int] = {}
    for entry in os.listdir(PROC_DIR):
        if entry.isdigit():
            try:
                names[int(entry)], parents[int(entry)] = _name_and_parent(int(entry))
            except (FileNotFoundError, ProcessLookupError):
                continue  # exited during the scan
    own_tree = {os.getpid()}
    grew = True
    while grew:
        children = {pid for pid, parent in parents.items() if parent in own_tree}
        grew = not children <= own_tree
        own_tree |= children
    own = 0
    others: Dict[int, ProcessRead] = {}
    uninspectable = 0
    uid = os.getuid()
    for pid, name in names.items():
        try:
            if pid not in own_tree and os.stat(os.path.join(PROC_DIR, str(pid))).st_uid != uid:
                uninspectable += 1
                continue
            read_bytes = _io_read_bytes(pid)
        except PermissionError:
            if pid in own_tree:
                raise PermissionError(
                    f"cannot read the counter of own child {pid} ({name}); "
                    "reap child processes before the arm stops"
                ) from None
            uninspectable += 1
            continue
        except (FileNotFoundError, ProcessLookupError):
            continue
        if pid in own_tree:
            own += read_bytes
        else:
            others[pid] = {"name": name, "read_bytes": read_bytes}
    return {"own": own, "others": others, "uninspectable": uninspectable}


def psutil_process_reads() -> ProcessReads:
    """A ``ProcessReads`` from psutil's per-process I/O counters, for Windows.

    The port of the reference ``l2l_box.process_read_bytes``, except that it
    counts a process whose counters Windows denies in ``uninspectable`` instead
    of skipping it. The import of psutil happens here, so the module loads
    without it.
    """
    try:
        import psutil
    except ImportError:
        raise SensorUnavailableError("psutil not installed") from None
    me = psutil.Process()
    own_tree = {me.pid} | {child.pid for child in me.children(recursive=True)}
    own = 0
    others: Dict[int, ProcessRead] = {}
    uninspectable = 0
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            read_bytes = int(proc.io_counters().read_bytes)
        except psutil.NoSuchProcess:
            continue  # exited during the scan
        except (psutil.Error, OSError):
            uninspectable += 1
            continue
        pid = int(proc.info["pid"])
        if pid in own_tree:
            own += read_bytes
        else:
            others[pid] = {"name": str(proc.info["name"]), "read_bytes": read_bytes}
    return {"own": own, "others": others, "uninspectable": uninspectable}


def ram_available() -> Optional[int]:
    """MemAvailable from /proc/meminfo in bytes: the RAM new work can get without swapping."""
    with open(MEMINFO_PATH, encoding="ascii") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return None


def workload_gpu() -> Dict[str, Optional[str]]:
    """The physical GPU the workload sees as ``cuda:0``, read from the environment.

    ``id`` is an index or a UUID for ``nvidia-smi --id``, taken from the first
    entry of CUDA_VISIBLE_DEVICES, or GPU 0 when it is unset. It is None when
    the variable leaves the workload no GPU. ``source`` names where it came from.
    With CUDA_DEVICE_ORDER unset, CUDA and nvidia-smi can number the GPUs of a
    mixed machine differently, so name the GPU by UUID there.
    """
    visible = os.environ.get(CUDA_VISIBLE_DEVICES)
    if visible is None:
        return {"id": DEFAULT_NVIDIA_GPU_ID, "source": "default"}
    first = visible.split(",")[0].strip()
    if first in ("", "NoDevFiles") or first.startswith("-"):
        return {"id": None, "source": CUDA_VISIBLE_DEVICES}
    return {"id": first, "source": CUDA_VISIBLE_DEVICES}


class CommitReading(TypedDict):
    """Memory promised to processes, in bytes, against the most the OS will promise."""

    used_bytes: int
    limit_bytes: int


def commit_charge() -> CommitReading:
    """Committed_AS and CommitLimit from /proc/meminfo.

    Raises ValueError if either line is missing, as on a kernel built without
    memory accounting.
    """
    found: Dict[str, int] = {}
    with open(MEMINFO_PATH, encoding="ascii") as handle:
        for line in handle:
            name, _, rest = line.partition(":")
            if name in ("Committed_AS", "CommitLimit"):
                found[name] = int(rest.split()[0]) * 1024
    for name in ("Committed_AS", "CommitLimit"):
        if name not in found:
            raise ValueError(f"{MEMINFO_PATH} has no {name} line")
    return {"used_bytes": found["Committed_AS"], "limit_bytes": found["CommitLimit"]}


def _nvidia_smi(*args: str) -> List[str]:
    """The non-empty lines nvidia-smi prints about the workload's GPU for ``args``.

    Raises RuntimeError if CUDA_VISIBLE_DEVICES leaves the workload no GPU or if
    nvidia-smi exits non-zero, and FileNotFoundError if it is not on PATH.
    """
    gpu = workload_gpu()
    if gpu["id"] is None:
        raise RuntimeError(
            f"{CUDA_VISIBLE_DEVICES}={os.environ.get(CUDA_VISIBLE_DEVICES)!r} leaves "
            "the workload no GPU"
        )
    tool = shutil.which("nvidia-smi")
    if tool is None:
        raise FileNotFoundError("nvidia-smi is not on PATH")
    completed = subprocess.run(
        [tool, f"--id={gpu['id']}", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=NVIDIA_SMI_TIMEOUT_S,
        check=False,
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"nvidia-smi exited {completed.returncode}: {message}")
    return [line for line in completed.stdout.splitlines() if line.strip()]


def _query_gpu(*fields: str) -> List[str]:
    """The values of ``fields`` from one ``--query-gpu`` call."""
    lines = _nvidia_smi(f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits")
    values = [value.strip() for value in lines[0].split(",")] if len(lines) == 1 else []
    if len(values) != len(fields):
        raise ValueError(f"expected one nvidia-smi line of {len(fields)} values, got {lines}")
    return values


def gpu_memory_used() -> Optional[int]:
    """MiB of memory in use on the GPU."""
    (used,) = _query_gpu("memory.used")
    return int(used)


class GpuProcess(TypedDict):
    """A process on the GPU. ``type`` is C (compute), G (graphics) or C+G;
    ``used_mib`` is None where the driver does not report it."""

    pid: int
    type: str
    name: str
    used_mib: Optional[int]


def gpu_processes() -> Optional[List[GpuProcess]]:
    """Every compute and graphics process on the GPU, from ``nvidia-smi -q -d PIDS``.

    The CSV query lists compute processes only, so it would miss a desktop or a
    browser that renders on the GPU.
    """
    blocks: List[Dict[str, str]] = []
    for line in _nvidia_smi("-q", "-d", "PIDS"):
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "Process ID":
            blocks.append({})
        if blocks and key in ("Process ID", "Type", "Name", "Used GPU Memory"):
            blocks[-1][key] = value
    processes: List[GpuProcess] = []
    for block in blocks:
        if not {"Process ID", "Type", "Name"} <= block.keys():
            raise ValueError(f"nvidia-smi listed a process without pid, type or name: {block}")
        used = block.get("Used GPU Memory", "").split()
        processes.append({
            "pid": int(block["Process ID"]),
            "type": block["Type"],
            "name": block["Name"],
            "used_mib": int(used[0]) if used and used[0].isdigit() else None,
        })
    return processes


class ClockReading(TypedDict):
    """One GPU clock sample: the SM clock and the clock event reasons active at that moment."""

    sm_clock_mhz: int
    reasons: List[str]


def clock_event_reasons(mask: int) -> List[str]:
    """Names of the bits set in an nvidia-smi ``clocks_event_reasons.active`` mask."""
    names = [name for bit, name in CLOCK_EVENT_REASONS.items() if mask & bit]
    unnamed = mask & ~sum(CLOCK_EVENT_REASONS)
    if unnamed:
        names.append(f"unnamed_bits_{unnamed:#x}")
    return names


def gpu_clock() -> Optional[ClockReading]:
    """The GPU's SM clock and active clock event reasons."""
    clock, mask = _query_gpu("clocks.sm", "clocks_event_reasons.active")
    return {"sm_clock_mhz": int(clock), "reasons": clock_event_reasons(int(mask, 16))}


class _SystemPowerStatus(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", ctypes.c_ubyte),
        ("BatteryFlag", ctypes.c_ubyte),
        ("BatteryLifePercent", ctypes.c_ubyte),
        ("SystemStatusFlag", ctypes.c_ubyte),
        ("BatteryLifeTime", ctypes.c_ulong),
        ("BatteryFullLifeTime", ctypes.c_ulong),
    ]


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _kernel32() -> Any:
    """Windows' kernel32.dll. Tests replace this function with a fake."""
    return getattr(ctypes, "windll").kernel32


def windows_power_supplies() -> PowerReading:
    """A ``PowerReading`` from ``GetSystemPowerStatus``, the call of the reference
    ``l2l_box.on_ac_power``.

    The AC line is the one non-battery supply, named ``ac_line``. Raises
    ValueError if the AC line status is unknown on a machine that may have a
    battery; with no system battery the machine is on mains whatever the line says.
    """
    status = _SystemPowerStatus()
    if not _kernel32().GetSystemPowerStatus(ctypes.byref(status)):
        raise OSError("GetSystemPowerStatus failed")
    ac_line, flag = int(status.ACLineStatus), int(status.BatteryFlag)
    no_battery = flag != BATTERY_FLAG_UNKNOWN and bool(flag & NO_SYSTEM_BATTERY)
    if ac_line not in (AC_LINE_OFFLINE, AC_LINE_ONLINE):
        if no_battery:
            return {"battery": False, "online": {}}
        raise ValueError(f"AC line status {ac_line} is unknown (battery flag {flag})")
    return {
        "battery": ac_line == AC_LINE_OFFLINE or not no_battery,
        "online": {"ac_line": ac_line == AC_LINE_ONLINE},
    }


def _memory_status() -> _MemoryStatusEx:
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not _kernel32().GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return status


def windows_ram_available() -> int:
    """Available physical memory in bytes from ``GlobalMemoryStatusEx``, as in the
    reference ``l2l_box.memory_status``."""
    return int(_memory_status().ullAvailPhys)


def windows_commit_charge() -> CommitReading:
    """The commit charge from ``GlobalMemoryStatusEx``: the commit limit
    (``ullTotalPageFile``) minus the commit still available (``ullAvailPageFile``)."""
    status = _memory_status()
    limit, available = int(status.ullTotalPageFile), int(status.ullAvailPageFile)
    return {"used_bytes": limit - available, "limit_bytes": limit}


def _read(
    name: str, sensor: Callable[[], Optional[_T]], none_problem: Optional[str] = None
) -> Tuple[Optional[_T], Optional[str]]:
    """``(value, problem)``: the reading, or None and why there is none."""
    try:
        value = sensor()
    except SensorUnavailableError as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001 - any sensor failure is an unknown
        return None, f"{name} sensor raised {type(exc).__name__}: {exc}"
    if value is None:
        return None, none_problem or f"{name} sensor returned None"
    return value, None


def _check_result(outcome: str, reason: str, evidence: Dict[str, object]) -> Dict[str, object]:
    return {"outcome": outcome, "reason": reason, "evidence": evidence}


@dataclass
class SleepReadings:
    """One reading of both sleep sensors, and why either gave nothing."""

    offset: Optional[float]
    count: Optional[int]
    problems: List[str]

    @classmethod
    def take(
        cls,
        offset_sensor: Callable[[], Optional[float]],
        count_sensor: Callable[[], Optional[int]],
    ) -> "SleepReadings":
        offset, offset_problem = _read("boot-time offset", offset_sensor)
        count, count_problem = _read("suspend count", count_sensor)
        return cls(offset, count, [p for p in (offset_problem, count_problem) if p])


def suspend_check_result(start: SleepReadings, end: SleepReadings) -> Dict[str, object]:
    """Void if boot-time offset grew past the limit or the suspend count rose.

    Either sensor seeing sleep is enough to void. Otherwise a silent sensor
    makes the check unknown. There is no wall-clock gap rule on Linux.
    """
    evidence: Dict[str, object] = {
        "sleep_offset_start_s": start.offset,
        "sleep_offset_end_s": end.offset,
        "suspend_count_start": start.count,
        "suspend_count_end": end.count,
    }
    slept = []
    growth = None
    if start.offset is not None and end.offset is not None:
        growth = end.offset - start.offset
        if growth > SLEEP_OFFSET_LIMIT_S:
            slept.append(f"boot-time offset grew {growth:.3f} s (limit {SLEEP_OFFSET_LIMIT_S} s)")
    if start.count is not None and end.count is not None and end.count > start.count:
        slept.append(f"suspend count rose from {start.count} to {end.count}")
    if slept:
        return _check_result(VOID, "; ".join(slept), evidence)
    problems = start.problems + end.problems
    if problems:
        return _check_result(UNKNOWN, "; ".join(problems), evidence)
    return _check_result(
        OK,
        f"boot-time offset grew {growth:.3f} s (limit {SLEEP_OFFSET_LIMIT_S} s); "
        f"suspend count stayed at {end.count}",
        evidence,
    )


@dataclass
class Sample(Generic[_T]):
    """One sample of a sensor. ``problem`` says why ``reading`` is None."""

    at_s: float
    reading: Optional[_T]
    problem: Optional[str]


class Sampler(Generic[_T]):
    """Reads a sensor at start, every ``interval_s`` seconds on a thread, and at stop."""

    def __init__(
        self, name: str, sensor: Callable[[], Optional[_T]], interval_s: float
    ) -> None:
        self._name = name
        self._sensor = sensor
        self._interval_s = interval_s
        self._stopped = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0 = 0.0
        self.samples: List[Sample[_T]] = []

    def _sample(self) -> None:
        reading, problem = _read(self._name, self._sensor)
        self.samples.append(Sample(time.monotonic() - self._t0, reading, problem))

    def _loop(self) -> None:
        while not self._stopped.wait(self._interval_s):
            self._sample()

    def start(self) -> None:
        self._t0 = time.monotonic()
        self._sample()
        self._thread = threading.Thread(
            target=self._loop, name=f"arm-{self._name}-sampler", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stopped.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()


def wall_clock_check_result(samples: List[Sample[float]]) -> Dict[str, object]:
    """Windows: void if the wall clock moved more than the limit between two samples.

    The check measures each gap between consecutive samples that read. A gap
    over the limit voids the arm even when some samples gave no reading. With no
    such gap, a sample with no reading makes the check unknown.
    """
    clocks = [sample.reading for sample in samples if sample.reading is not None]
    problems = [sample.problem for sample in samples if sample.problem]
    max_gap = max((after - before for before, after in zip(clocks, clocks[1:])), default=0.0)
    evidence: Dict[str, object] = {
        "samples": len(samples),
        "max_gap_s": max_gap,
        "unknown_samples": len(problems),
    }
    limit = f"(limit {WALL_CLOCK_GAP_LIMIT_S} s)"
    if max_gap > WALL_CLOCK_GAP_LIMIT_S:
        return _check_result(
            VOID, f"wall clock jumped {max_gap:.1f} s between two samples {limit}", evidence
        )
    if problems:
        return _check_result(
            UNKNOWN,
            f"{len(problems)} of {len(samples)} wall-clock samples gave no reading; "
            f"first: {problems[0]}",
            evidence,
        )
    return _check_result(
        OK, f"largest wall-clock gap between samples {max_gap:.1f} s {limit}", evidence
    )


def _on_mains(reading: PowerReading) -> bool:
    """A machine with no battery runs on mains; otherwise some non-battery supply must be online."""
    return not reading["battery"] or any(reading["online"].values())


def power_check_result(samples: List[Sample[PowerReading]]) -> Dict[str, object]:
    """Void if any sample shows the machine off mains.

    Otherwise a sample with no reading makes the check unknown.
    """
    off_mains = [
        {"sample": index, "at_s": round(sample.at_s, 3), "online": sample.reading["online"]}
        for index, sample in enumerate(samples)
        if sample.reading is not None and not _on_mains(sample.reading)
    ]
    readings = [sample.reading for sample in samples if sample.reading is not None]
    problems = [sample.problem for sample in samples if sample.problem]
    evidence: Dict[str, object] = {
        "samples": len(samples),
        "off_mains_samples": off_mains,
        "battery_present": any(reading["battery"] for reading in readings),
        "unknown_samples": len(problems),
        "last_reading": readings[-1] if readings else None,
    }
    if off_mains:
        return _check_result(
            VOID, f"{len(off_mains)} of {len(samples)} power samples off mains", evidence
        )
    if problems:
        return _check_result(
            UNKNOWN,
            f"{len(problems)} of {len(samples)} power samples gave no reading; "
            f"first: {problems[0]}",
            evidence,
        )
    if not evidence["battery_present"]:
        return _check_result(
            OK, f"no battery in any of {len(samples)} power samples, so on mains", evidence
        )
    return _check_result(OK, f"on mains in all {len(samples)} power samples", evidence)


@dataclass
class DiskReadings:
    """One reading of the disk counter and the process scan, and why either gave nothing."""

    disk: Optional[int]
    other_disks: Optional[int]
    processes: Optional[ProcessReads]
    problems: List[str]

    @classmethod
    def take(
        cls,
        device: Optional[str],
        disk_sensor: Callable[[str], Optional[int]],
        other_disks_sensor: Callable[[str], Optional[int]],
        process_sensor: Callable[[], Optional[ProcessReads]],
    ) -> "DiskReadings":
        if device is None:
            disk, disk_problem, others, others_problem = None, None, None, None
        else:
            disk, disk_problem = _read("disk read", lambda: disk_sensor(device))
            others, others_problem = _read("other disks read", lambda: other_disks_sensor(device))
        processes, process_problem = _read("process read", process_sensor)
        problems = [disk_problem, others_problem, process_problem]
        return cls(disk, others, processes, [p for p in problems if p])


class Reader(TypedDict):
    pid: int
    name: str
    read_bytes: int


def _readers(start: ProcessReads, end: ProcessReads) -> List[Reader]:
    """Other processes that read during the arm, most bytes first. A pid that is
    new, or now has another name, counts its whole counter."""
    readers: List[Reader] = []
    for pid, now in end["others"].items():
        before = start["others"].get(pid)
        base = before["read_bytes"] if before and before["name"] == now["name"] else 0
        if now["read_bytes"] > base:
            read = now["read_bytes"] - base
            readers.append({"pid": pid, "name": now["name"], "read_bytes": read})
    return sorted(readers, key=lambda reader: (-reader["read_bytes"], reader["pid"]))


def _foreign_result(
    outcome: str,
    reason: str,
    evidence: Dict[str, object],
    readers: List[Reader],
    unattributed_bytes: Optional[int],
) -> Dict[str, object]:
    result = _check_result(outcome, reason, evidence)
    result.update({"readers": readers, "unattributed_bytes": unattributed_bytes})
    return result


def foreign_reads_check_result(
    model_path: Optional[str],
    device: Optional[str],
    device_problem: Optional[str],
    start: DiskReadings,
    end: DiskReadings,
) -> Dict[str, object]:
    """Void if the disk under test read 1 GB or more beyond this process tree's reads (ADR 0002).

    The tree's counter covers every disk, so how much of it hit ``device`` is
    unknown. The reads of the other disks bound it: at most all of the tree's
    reads hit ``device``, and at least the part the other disks did not take.
    That bounds the foreign reads between ``foreign_read_bytes_min`` and
    ``foreign_read_bytes_max``. The check is void if the minimum reaches the
    limit, ok if the maximum stays under it, and unknown between the two.
    ``foreign_read_bytes`` is the exact figure when the bounds meet, else None.

    ``readers`` names the other processes this user may inspect that read during
    the arm; ``unattributed_bytes`` is the maximum they do not explain, floored at 0.
    """
    evidence: Dict[str, object] = {
        "model_path": model_path,
        "device": device,
        "disk_read_bytes_start": start.disk,
        "disk_read_bytes_end": end.disk,
        "other_disk_read_bytes_start": start.other_disks,
        "other_disk_read_bytes_end": end.other_disks,
        "own_read_bytes": None,
        "foreign_read_bytes": None,
        "foreign_read_bytes_min": None,
        "foreign_read_bytes_max": None,
        "uninspectable_processes": None,
    }
    problems = ([device_problem] if device_problem else []) + start.problems + end.problems
    if problems:
        return _foreign_result(UNKNOWN, "; ".join(problems), evidence, [], None)
    assert start.disk is not None and end.disk is not None
    assert start.other_disks is not None and end.other_disks is not None
    assert start.processes is not None and end.processes is not None
    own = end.processes["own"] - start.processes["own"]
    on_device = end.disk - start.disk
    elsewhere = end.other_disks - start.other_disks
    least = max(on_device - own, 0)
    most = max(on_device - max(own - elsewhere, 0), 0)
    readers = _readers(start.processes, end.processes)
    evidence.update(
        own_read_bytes=own,
        foreign_read_bytes=least if least == most else None,
        foreign_read_bytes_min=least,
        foreign_read_bytes_max=most,
        uninspectable_processes=end.processes["uninspectable"],
    )
    named_bytes = sum(reader["read_bytes"] for reader in readers)
    limit = f"(limit {FOREIGN_READ_LIMIT_BYTES})"
    if least >= FOREIGN_READ_LIMIT_BYTES:
        outcome, reason = VOID, f"at least {least} bytes of foreign reads on {device} {limit}"
    elif most < FOREIGN_READ_LIMIT_BYTES:
        outcome, reason = OK, f"at most {most} bytes of foreign reads on {device} {limit}"
    else:
        outcome = UNKNOWN
        reason = (
            f"{own} bytes of own reads cannot be placed on {device}: {elsewhere} bytes were "
            f"read from the other disks, so foreign reads on {device} are between {least} "
            f"and {most} bytes {limit}"
        )
    return _foreign_result(
        outcome,
        reason,
        evidence,
        readers,
        None if outcome == UNKNOWN else max(most - named_bytes, 0),
    )


def per_process_reads_check_result(
    start: Optional[ProcessReads], end: Optional[ProcessReads], problems: List[str]
) -> Dict[str, object]:
    """Windows: void if one other process read 1 GB or more during the arm, the rule
    of the reference ``l2l_box.foreign_readers``.

    There is no disk-wide counter here, so a process this user may not inspect
    goes unseen; ``uninspectable_processes`` counts them and ``unattributed_bytes``
    is None. Windows counts every read a process makes, from any disk or device.
    """
    evidence: Dict[str, object] = {
        "own_read_bytes": None,
        "foreign_read_bytes": None,
        "uninspectable_processes": None,
    }
    if problems:
        return _foreign_result(UNKNOWN, "; ".join(dict.fromkeys(problems)), evidence, [], None)
    assert start is not None and end is not None
    readers = _readers(start, end)
    evidence.update(
        own_read_bytes=end["own"] - start["own"],
        foreign_read_bytes=sum(reader["read_bytes"] for reader in readers),
        uninspectable_processes=end["uninspectable"],
    )
    limit = f"(limit {FOREIGN_READ_LIMIT_BYTES} per process)"
    uninspectable = f"{end['uninspectable']} processes could not be inspected"
    heavy = [reader for reader in readers if reader["read_bytes"] >= FOREIGN_READ_LIMIT_BYTES]
    if heavy:
        return _foreign_result(
            VOID,
            f"{heavy[0]['name']} (pid {heavy[0]['pid']}) read {heavy[0]['read_bytes']} bytes "
            f"{limit}; {len(heavy)} processes over the limit; {uninspectable}",
            evidence,
            readers,
            None,
        )
    return _foreign_result(
        OK,
        f"no other process read {FOREIGN_READ_LIMIT_BYTES} bytes or more; {uninspectable}",
        evidence,
        readers,
        None,
    )


def box_stamp(
    power: Sample[PowerReading],
    ram_available: Callable[[], Optional[int]],
    commit_charge: Callable[[], Optional[CommitReading]],
    gpu_memory_used: Callable[[], Optional[int]],
    gpu_processes: Callable[[], Optional[List[GpuProcess]]],
) -> Dict[str, object]:
    """The machine's state at one moment. A field whose sensor gave nothing is
    None, with the reason under ``unknown``. ``power`` is a power sample taken at
    the same moment, so the power check and the stamp read the sensor once."""
    unknown: Dict[str, str] = {}

    def field(name: str, sensor: Callable[[], Optional[_T]]) -> Optional[_T]:
        value, problem = _read(name, sensor)
        if problem:
            unknown[name] = problem
        return value

    stamp: Dict[str, object] = {
        "unix_s": time.time(),
        "ram_available_bytes": field("ram_available_bytes", ram_available),
        "commit_charge": field("commit_charge", commit_charge),
        "power_source": None,
        "gpu_processes": field("gpu_processes", gpu_processes),
        "gpu_memory_used_mib": field("gpu_memory_used_mib", gpu_memory_used),
    }
    if power.reading is None:
        unknown["power_source"] = power.problem or "power sensor gave no reading"
    else:
        stamp["power_source"] = "mains" if _on_mains(power.reading) else "battery"
    stamp["unknown"] = unknown
    return stamp


def gpu_clock_record(samples: List[Sample[ClockReading]]) -> Dict[str, object]:
    """The SM clock's min, median and max over the arm, and how many samples saw
    each clock event reason. Ok or unknown, never void. Throttling goes on record
    for the reader and does not decide the arm.

    A sample with no reading makes the record unknown; the statistics then cover
    the samples that did read, and are None if none did.
    """
    readings = [sample.reading for sample in samples if sample.reading is not None]
    problems = [sample.problem for sample in samples if sample.problem]
    clocks = [reading["sm_clock_mhz"] for reading in readings]
    reasons_seen: Dict[str, int] = {}
    for reading in readings:
        for reason in reading["reasons"]:
            reasons_seen[reason] = reasons_seen.get(reason, 0) + 1
    record: Dict[str, object] = {
        "outcome": UNKNOWN if problems else OK,
        "reason": "",
        "samples": len(samples),
        "unknown_samples": len(problems),
        "sm_clock_mhz": {
            "min": min(clocks) if clocks else None,
            "median": statistics.median(clocks) if clocks else None,
            "max": max(clocks) if clocks else None,
        },
        "reasons_seen": reasons_seen,
    }
    if problems:
        record["reason"] = (
            f"{len(problems)} of {len(samples)} GPU clock samples gave no reading; "
            f"first: {problems[0]}"
        )
    else:
        seen = ", ".join(sorted(reasons_seen)) or "none"
        record["reason"] = (
            f"SM clock {min(clocks)} to {max(clocks)} MHz over {len(samples)} samples; "
            f"clock event reasons seen: {seen}"
        )
    return record


def arm_outcome(checks: Dict[str, Dict[str, object]], required: Collection[str]) -> str:
    """Void if any check is void, else unknown if any required check is unknown, else ok."""
    if any(check["outcome"] == VOID for check in checks.values()):
        return VOID
    if any(checks[name]["outcome"] == UNKNOWN for name in required):
        return UNKNOWN
    return OK


class _LinuxChecks:
    """The Linux checks: boot-time offset and suspend count for sleep, the kernel's
    power supplies for power, and the disk-wide read counter for foreign reads."""

    def __init__(
        self,
        sleep_offset: Callable[[], Optional[float]],
        suspend_count: Callable[[], Optional[int]],
        model_path: Optional[str],
        disk_device: Callable[[str], Optional[str]],
        disk_reads: Callable[[str], Optional[int]],
        other_disk_reads: Callable[[str], Optional[int]],
        process_reads: Callable[[], Optional[ProcessReads]],
    ) -> None:
        self._sleep_offset = sleep_offset
        self._suspend_count = suspend_count
        self._model_path = model_path
        self._disk_device = disk_device
        self._disk_reads = disk_reads
        self._other_disk_reads = other_disk_reads
        self._process_reads = process_reads
        self._device: Optional[str] = None
        self._device_problem: Optional[str] = None
        self._sleep: List[SleepReadings] = []
        self._disk: List[DiskReadings] = []

    def _find_device(self) -> None:
        if self._model_path is None:
            self._device_problem = "no model path given"
            return
        path = self._model_path
        self._device, self._device_problem = _read(
            "disk device", lambda: self._disk_device(path), f"{path} maps to no block device"
        )

    def _read_disk(self) -> None:
        self._disk.append(
            DiskReadings.take(
                self._device, self._disk_reads, self._other_disk_reads, self._process_reads
            )
        )

    def _read_sleep(self) -> None:
        self._sleep.append(SleepReadings.take(self._sleep_offset, self._suspend_count))

    def start(self) -> None:
        self._find_device()
        self._read_disk()
        self._read_sleep()

    def end(self) -> None:
        self._read_sleep()
        self._read_disk()

    def results(self, power: List[Sample[PowerReading]]) -> Dict[str, Dict[str, object]]:
        return {
            "suspend": suspend_check_result(self._sleep[0], self._sleep[-1]),
            "power": power_check_result(power),
            "foreign_reads": foreign_reads_check_result(
                self._model_path,
                self._device,
                self._device_problem,
                self._disk[0],
                self._disk[-1],
            ),
        }


class _WindowsChecks:
    """The Windows checks, ported from the reference ``l2l_box.py``: a wall-clock
    gap for sleep, the system power status for power, and per-process read
    counters for foreign reads."""

    def __init__(
        self,
        wall_clock: Callable[[], Optional[float]],
        interval_s: float,
        process_reads: Callable[[], Optional[ProcessReads]],
    ) -> None:
        self._wall_clock = Sampler("wall clock", wall_clock, interval_s)
        self._process_reads = process_reads
        self._processes: List[Optional[ProcessReads]] = []
        self._problems: List[str] = []

    def _read_processes(self) -> None:
        processes, problem = _read("process read", self._process_reads)
        self._processes.append(processes)
        if problem:
            self._problems.append(problem)

    def start(self) -> None:
        self._read_processes()
        self._wall_clock.start()

    def end(self) -> None:
        self._wall_clock.stop()
        self._read_processes()

    def results(self, power: List[Sample[PowerReading]]) -> Dict[str, Dict[str, object]]:
        return {
            "suspend": wall_clock_check_result(self._wall_clock.samples),
            "power": power_check_result(power),
            "foreign_reads": per_process_reads_check_result(
                self._processes[0], self._processes[-1], self._problems
            ),
        }


class _UnimplementedChecks:
    """Every check unknown, for an OS the module has no checks for."""

    def __init__(self, system: str) -> None:
        self._reason = f"not implemented on {system}"

    def start(self) -> None:
        pass

    def end(self) -> None:
        pass

    def results(self, power: List[Sample[PowerReading]]) -> Dict[str, Dict[str, object]]:
        checks = {name: _check_result(UNKNOWN, self._reason, {}) for name in CHECKS}
        checks["foreign_reads"] = _foreign_result(UNKNOWN, self._reason, {}, [], None)
        return checks


#: The power, process reads, RAM and commit charge sensors ``ArmWatch`` uses by default, per OS.
_DEFAULT_SENSORS: Dict[
    str,
    Tuple[
        Callable[[], Optional[PowerReading]],
        Callable[[], Optional[ProcessReads]],
        Callable[[], Optional[int]],
        Callable[[], Optional[CommitReading]],
    ],
] = {
    LINUX: (power_supplies, process_reads, ram_available, commit_charge),
    WINDOWS: (
        windows_power_supplies,
        psutil_process_reads,
        windows_ram_available,
        windows_commit_charge,
    ),
}


class ArmWatch:
    """Watches one arm. Use as a context manager, then read ``record``.

    ``required`` names the checks whose unknown makes the arm unknown; it
    defaults to every check. A void check voids the arm whether required or not.
    ``model_path`` is the path to the model files. On Linux the foreign reads
    check watches the disk that holds it, and is unknown without it. Windows
    has no disk-wide counter, so its check ignores ``model_path``.

    ``platform`` picks the checks, as a ``sys.platform`` value. On an OS without
    checks every check is unknown. ``power_supplies``, ``process_reads``,
    ``ram_available`` and ``commit_charge`` default to that OS's sensor. ``sleep_offset``,
    ``suspend_count``, ``disk_device``, ``disk_reads`` and ``other_disk_reads``
    serve Linux only, and ``wall_clock`` serves Windows only.
    """

    def __init__(
        self,
        *,
        arm: str,
        round: int,
        run: int,
        label: Optional[str] = None,
        required: Optional[Collection[str]] = None,
        platform: str = sys.platform,
        sleep_offset: Callable[[], Optional[float]] = sleep_offset,
        suspend_count: Callable[[], Optional[int]] = suspend_count,
        power_supplies: Optional[Callable[[], Optional[PowerReading]]] = None,
        power_interval_s: float = POWER_INTERVAL_S,
        wall_clock: Callable[[], Optional[float]] = time.time,
        wall_clock_interval_s: float = WALL_CLOCK_INTERVAL_S,
        model_path: Optional[str] = None,
        disk_device: Callable[[str], Optional[str]] = disk_device,
        disk_reads: Callable[[str], Optional[int]] = disk_reads,
        other_disk_reads: Callable[[str], Optional[int]] = other_disk_reads,
        process_reads: Optional[Callable[[], Optional[ProcessReads]]] = None,
        ram_available: Optional[Callable[[], Optional[int]]] = None,
        commit_charge: Optional[Callable[[], Optional[CommitReading]]] = None,
        gpu_memory_used: Callable[[], Optional[int]] = gpu_memory_used,
        gpu_processes: Callable[[], Optional[List[GpuProcess]]] = gpu_processes,
        gpu_clock: Callable[[], Optional[ClockReading]] = gpu_clock,
        gpu_clock_interval_s: float = GPU_CLOCK_INTERVAL_S,
    ):
        self.arm = arm
        self.round = round
        self.run = run
        self.label = label
        self.required = list(CHECKS if required is None else required)
        unknown_names = sorted(set(self.required) - set(CHECKS))
        if unknown_names:
            raise ValueError(f"no such checks: {', '.join(unknown_names)}")
        self.system = system_name(platform)
        self.model_path = model_path
        unavailable = _not_implemented_on(self.system)
        default_power, default_processes, default_ram, default_commit = _DEFAULT_SENSORS.get(
            self.system, (unavailable, unavailable, unavailable, unavailable)
        )
        self._checks: Union[_LinuxChecks, _WindowsChecks, _UnimplementedChecks]
        if self.system == LINUX:
            self._checks = _LinuxChecks(
                sleep_offset,
                suspend_count,
                model_path,
                disk_device,
                disk_reads,
                other_disk_reads,
                process_reads or default_processes,
            )
        elif self.system == WINDOWS:
            self._checks = _WindowsChecks(
                wall_clock, wall_clock_interval_s, process_reads or default_processes
            )
        else:
            self._checks = _UnimplementedChecks(self.system)
        self._power = Sampler("power", power_supplies or default_power, power_interval_s)
        self._ram_available = ram_available or default_ram
        self._commit_charge = commit_charge or default_commit
        self._gpu_memory_used = gpu_memory_used
        self._gpu_processes = gpu_processes
        self._box_before: Optional[Dict[str, object]] = None
        self._gpu: Dict[str, Optional[str]] = {}
        self._clock = Sampler("gpu_clock", gpu_clock, gpu_clock_interval_s)
        self.record: Optional[Dict[str, object]] = None

    def _box_stamp(self, power: Sample[PowerReading]) -> Dict[str, object]:
        return box_stamp(
            power,
            self._ram_available,
            self._commit_charge,
            self._gpu_memory_used,
            self._gpu_processes,
        )

    def start(self) -> None:
        self._gpu = workload_gpu()
        self._power.start()
        self._box_before = self._box_stamp(self._power.samples[0])
        self._checks.start()
        self._clock.start()

    def stop(self) -> Dict[str, object]:
        if self._box_before is None:
            raise RuntimeError("stop() called before start()")
        # The clock sampler runs nvidia-smi, so it stops before the process scan
        # can find one of its children exited but not yet reaped.
        self._clock.stop()
        self._checks.end()
        self._power.stop()
        box_after = self._box_stamp(self._power.samples[-1])
        checks = self._checks.results(self._power.samples)
        self.record = {
            "arm": self.arm,
            "round": self.round,
            "run": self.run,
            "label": self.label,
            "gpu": self._gpu,
            "required": self.required,
            "outcome": arm_outcome(checks, self.required),
            "box_stamp_before": self._box_before,
            "box_stamp_after": box_after,
            "gpu_clock": gpu_clock_record(self._clock.samples),
            "checks": checks,
        }
        return self.record

    def __enter__(self) -> "ArmWatch":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()


def block_order(count: int) -> List[Tuple[str, ...]]:
    """The arms of each of ``count`` blocks in run order: A, B, B, A."""
    return [BLOCK_ORDER] * count


@dataclass(frozen=True)
class Run:
    """One run: the median of its timed steps and its arm record's outcome."""

    value: float
    outcome: str


def _arm_values(block: Sequence[Run], arm: str) -> List[float]:
    return [run.value for run, name in zip(block, BLOCK_ORDER) if name == arm]


def valid_blocks(blocks: Sequence[Sequence[Run]]) -> List[Sequence[Run]]:
    """The blocks with no void run. A block with a void run is replaced, not counted.

    A run whose outcome is unknown still counts. The rule committed in METHOD.md
    decides whether its gate gets a verdict.
    """
    for block in blocks:
        if len(block) != len(BLOCK_ORDER):
            raise ValueError(f"a block holds {len(BLOCK_ORDER)} runs, got {len(block)}")
    return [block for block in blocks if all(run.outcome != VOID for run in block)]


@dataclass(frozen=True)
class Repeats:
    """What the repeats rule reads from a set of blocks."""

    sigma: float
    mean_a: float
    blocks_needed: int


def repeats(blocks: Sequence[Sequence[Run]]) -> Repeats:
    """Sigma, the mean of A and the blocks a 10% effect needs: ceil(7.85 * (sigma / delta)^2).

    Sigma is the run-to-run spread within an arm, pooled over A and B:
    sqrt((variance of the A runs + variance of the B runs) / 2). Delta is 10% of
    the mean of A. The block count is never below ``MIN_BLOCKS``. Blocks with a
    void run do not count.
    """
    blocks = valid_blocks(blocks)
    if len(blocks) < MIN_BLOCKS:
        raise ValueError(f"sigma needs at least {MIN_BLOCKS} blocks without a void run")
    a_runs = [value for block in blocks for value in _arm_values(block, "A")]
    b_runs = [value for block in blocks for value in _arm_values(block, "B")]
    sigma = math.sqrt((statistics.variance(a_runs) + statistics.variance(b_runs)) / 2)
    mean_a = statistics.fmean(a_runs)
    if mean_a <= 0:
        raise ValueError(f"the mean of A must be positive, got {mean_a}")
    delta = EFFECT_FRACTION * mean_a
    needed = max(MIN_BLOCKS, math.ceil(SAMPLE_SIZE_FACTOR * (sigma / delta) ** 2))
    return Repeats(sigma, mean_a, needed)


def blocks_needed(blocks: Sequence[Sequence[Run]]) -> int:
    """Blocks a comparison needs to detect a 10% effect, from ``repeats``."""
    return repeats(blocks).blocks_needed


def blocks_to_add(blocks: Sequence[Sequence[Run]]) -> int:
    """How many more blocks to run: the pilot, then up to N, then one top-up.

    N comes from the pilot. On reaching it, N is recomputed once from every
    block so far, and a larger N is topped up to. Blocks after that never change
    N, so the rule never asks for a second top-up. Blocks with a void run do not
    count, so each one adds a block to run.
    """
    blocks = valid_blocks(blocks)
    if len(blocks) < PILOT_BLOCKS:
        return PILOT_BLOCKS - len(blocks)
    first_n = max(PILOT_BLOCKS, blocks_needed(blocks[:PILOT_BLOCKS]))
    if len(blocks) < first_n:
        return first_n - len(blocks)
    final_n = max(first_n, blocks_needed(blocks[:first_n]))
    return max(0, final_n - len(blocks))
