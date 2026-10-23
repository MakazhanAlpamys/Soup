#!/usr/bin/env python3
"""Box stamps for the L2L probe, and a watch for sleep and battery.

The probe runs beside other Claude sessions on a laptop. Every block records the
memory, power and GPU state it ran in (rule V3), and every arm runs under a
``SuspendWatch`` that samples the wall clock and the AC line every 2 s (rule
V5): a gap over 30 s means the box slept, and a battery sample means the clocks
were not the ones the gates assume. On 2026-10-01 a peer's probe died exactly
this way when the laptop was unplugged.

Rule V6 is the foreign reader: R2's attempt 1 lost an arm to SearchIndexer.exe
reading 4.94 GB from the drive under test. ``process_read_bytes`` snapshots every
process's read counter (psutil; ``None`` without it) and ``foreign_readers``
names the other processes that read ``threshold`` bytes or more between two
snapshots.

Benchmark harness code, not shipped. Standard library only.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
import threading
import time
from typing import Callable, Collection, Dict, List, Optional

_GB = 1e9
FOREIGN_READ_BYTES = 1_000_000_000


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


class _SystemPowerStatus(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", ctypes.c_ubyte),
        ("BatteryFlag", ctypes.c_ubyte),
        ("BatteryLifePercent", ctypes.c_ubyte),
        ("SystemStatusFlag", ctypes.c_ubyte),
        ("BatteryLifeTime", ctypes.c_ulong),
        ("BatteryFullLifeTime", ctypes.c_ulong),
    ]


def memory_status() -> Dict[str, Optional[float]]:
    keys = ("total_phys_gb", "avail_phys_gb", "commit_limit_gb", "commit_avail_gb")
    if sys.platform != "win32":
        return dict.fromkeys(keys)
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return dict.fromkeys(keys)
    return {
        "total_phys_gb": status.ullTotalPhys / _GB,
        "avail_phys_gb": status.ullAvailPhys / _GB,
        "commit_limit_gb": status.ullTotalPageFile / _GB,
        "commit_avail_gb": status.ullAvailPageFile / _GB,
    }


def on_ac_power() -> Optional[bool]:
    if sys.platform != "win32":
        return None
    status = _SystemPowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        return None
    return {0: False, 1: True}.get(int(status.ACLineStatus))


def _nvidia_smi(query: List[str]) -> Optional[str]:
    tool = shutil.which("nvidia-smi")
    if tool is None:
        return None
    try:
        out = subprocess.run(
            [tool, *query],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout


def gpu_compute_pids() -> Optional[List[int]]:
    text = _nvidia_smi(["--query-compute-apps=pid", "--format=csv,noheader"])
    if text is None:
        return None
    return [int(line.strip()) for line in text.splitlines() if line.strip().isdigit()]


def gpu_memory_used_mib() -> Optional[int]:
    text = _nvidia_smi(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    if not text or not text.strip().splitlines()[0].strip().isdigit():
        return None
    return int(text.strip().splitlines()[0].strip())


def python_process_count() -> Optional[int]:
    if sys.platform != "win32":
        return None
    try:
        out = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",  # tasklist prints in the console's OEM code page
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return sum(1 for line in out.stdout.splitlines() if line.startswith('"python.exe"'))


def box_stamp() -> Dict[str, object]:
    return {
        "unix": time.time(),
        **memory_status(),
        "ac": on_ac_power(),
        "gpu_pids": gpu_compute_pids(),
        "gpu_mem_used_mib": gpu_memory_used_mib(),
        "python_processes": python_process_count(),
    }


class SuspendWatch:
    """Samples (wall clock, AC line) every ``interval`` s on a daemon thread."""

    def __init__(
        self,
        interval: float = 2.0,
        gap_limit: float = 30.0,
        clock: Callable[[], float] = time.time,
        power: Callable[[], Optional[bool]] = on_ac_power,
    ):
        self.interval = float(interval)
        self.gap_limit = float(gap_limit)
        self._clock = clock
        self._power = power
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last: Optional[float] = None
        self.samples = 0
        self.max_gap_s = 0.0
        self.battery_samples = 0
        self.unknown_power_samples = 0

    def sample(self) -> None:
        now = self._clock()
        ac = self._power()
        with self._lock:
            if self._last is not None:
                self.max_gap_s = max(self.max_gap_s, now - self._last)
            self._last = now
            self.samples += 1
            if ac is False:
                self.battery_samples += 1
            elif ac is None:
                self.unknown_power_samples += 1

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.sample()

    def start(self) -> None:
        self.sample()
        self._thread = threading.Thread(target=self._loop, name="l2l-suspend-watch", daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, object]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        self.sample()
        return self.report()

    def report(self) -> Dict[str, object]:
        with self._lock:
            return {
                "samples": self.samples,
                "max_gap_s": self.max_gap_s,
                "battery_samples": self.battery_samples,
                "unknown_power_samples": self.unknown_power_samples,
                "void": self.max_gap_s > self.gap_limit or self.battery_samples > 0,
            }


def process_read_bytes() -> Optional[Dict[int, Dict[str, object]]]:
    """``{pid: {"name", "read"}}`` for every process whose counters can be read."""
    try:
        import psutil
    except ImportError:
        return None
    out: Dict[int, Dict[str, object]] = {}
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            counters = proc.io_counters()
        except (psutil.Error, OSError):
            continue
        out[int(proc.info["pid"])] = {"name": proc.info["name"], "read": int(counters.read_bytes)}
    return out


def foreign_readers(
    before: Optional[Dict[int, Dict[str, object]]],
    after: Optional[Dict[int, Dict[str, object]]],
    *,
    own_pids: Collection[int],
    threshold: int = FOREIGN_READ_BYTES,
) -> Optional[List[Dict[str, object]]]:
    """Other processes that read ``threshold`` bytes or more between two snapshots,
    heaviest first. A process absent from ``before`` counts its whole counter.
    ``None`` when either snapshot is missing: unknown, not clean."""
    if before is None or after is None:
        return None
    heavy = []
    for pid, row in after.items():
        if pid in own_pids:
            continue
        read = int(row["read"]) - int((before.get(pid) or {"read": 0})["read"])
        if read >= threshold:
            heavy.append({"pid": pid, "name": row["name"], "read_bytes": read})
    return sorted(heavy, key=lambda item: item["read_bytes"], reverse=True)
