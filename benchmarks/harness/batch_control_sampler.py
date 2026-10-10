#!/usr/bin/env python3
"""The R2 batch control's GPU samplers: ``nvidia-smi`` and the Windows GPU memory counters.

Split out of ``batch_control_probe.py`` so the driver stays readable. ``BatchSampler`` is the
two-drive gate's power and PDH sampler (``two_drive_stripe_gate.Sampler``, imported unchanged)
plus two threads: ``nvidia-smi`` every NVSMI_POLL_S (memory in use, SM clock, temperature,
power, utilisation, throttle reasons) and the Windows GPU memory counters every
GPU_MEMORY_POLL_S (dedicated and shared usage, per adapter and for the arm's own process).
Windows spills an over-allocation into shared host memory without an error, and these are
where a spill shows. Recorded, never judged, except through the rule's row 4 and the edge
label (``batch_control_rule.py``). No torch import. Windows only.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import os
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

import two_drive_stripe_gate as gate

NVSMI_POLL_S = 2.0
GPU_MEMORY_POLL_S = 1.0
GPU_QUERY_REOPEN_S = 5.0  # reopen the GPU memory query this often until the arm's pid shows
NVSMI_FIELDS = (
    "memory.used,memory.total,clocks.sm,temperature.gpu,power.draw,utilization.gpu,"
    "clocks_throttle_reasons.active"
)
GPU_MEMORY_COUNTERS = {
    "adapter_dedicated": r"\GPU Adapter Memory(*)\Dedicated Usage",
    "adapter_shared": r"\GPU Adapter Memory(*)\Shared Usage",
    "process_dedicated": r"\GPU Process Memory(*)\Dedicated Usage",
    "process_shared": r"\GPU Process Memory(*)\Shared Usage",
}
_PDH_MORE_DATA = 0x800007D2
_PDH_CSTATUS_OK = (0, 1)  # PDH_CSTATUS_VALID_DATA, PDH_CSTATUS_NEW_DATA


# ==========================================================================
# sampling
# ==========================================================================
def _number(text: str) -> Optional[float]:
    try:
        return float(text)
    except ValueError:
        return None


def parse_nvsmi(text: Optional[str]) -> List[Any]:
    """``[used_mib, total_mib, sm_mhz, temp_c, power_w, util_pct, throttle_reasons]``."""
    lines = (text or "").strip().splitlines()
    parts = [part.strip() for part in lines[0].split(",")] if lines else []
    if len(parts) != 7:
        return [None] * 7
    return [_number(part) for part in parts[:6]] + [parts[6]]


class _PdhItem(ctypes.Structure):
    """PDH_FMT_COUNTERVALUE_ITEM_W: an instance name and its formatted value."""

    _fields_ = [("szName", ctypes.c_wchar_p), ("FmtValue", gate._PdhValue)]


class BatchSampler(gate.Sampler):
    """The gate's power and PDH sampler, plus two threads that watch the GPU's memory.

    ``nvidia-smi`` every NVSMI_POLL_S, and the Windows GPU memory counters (wildcard instances,
    in a query of their own) every GPU_MEMORY_POLL_S: dedicated and shared usage per adapter,
    and per process for the arm's own pid once ``pid`` is set. Recorded, never judged.
    """

    def __init__(self) -> None:
        super().__init__()
        self.pid: Optional[int] = None
        self.gpu_rows: List[List[Any]] = []
        self.gpu_memory_rows: List[List[Any]] = []
        self.gpu_query_opens = 0
        self._gpu_query = ctypes.c_void_p()
        self._gpu_handles: Dict[str, ctypes.c_void_p] = {}
        self._gpu_opened_at = 0.0
        self._pid_seen = False
        pdh = self._pdh
        pdh.PdhGetFormattedCounterArrayW.restype = ctypes.c_long
        pdh.PdhGetFormattedCounterArrayW.argtypes = [
            ctypes.c_void_p,
            wt.DWORD,
            ctypes.POINTER(wt.DWORD),
            ctypes.POINTER(wt.DWORD),
            ctypes.c_void_p,
        ]
        self._open_gpu_query()

    def _open_gpu_query(self) -> None:
        """(Re)open the GPU memory query and add the wildcard counters.

        Reopened every GPU_QUERY_REOPEN_S until the arm's own process shows up, so the per-process
        rows do not depend on whether PDH re-expands a wildcard for a process that started after
        the query was opened: the arm's CUDA context appears seconds after its process does.
        """
        pdh = self._pdh
        if self._gpu_query:
            pdh.PdhCloseQuery(self._gpu_query)
        self._gpu_query = ctypes.c_void_p()
        self._gpu_handles = {}
        self._gpu_opened_at = time.monotonic()
        self.gpu_query_opens += 1
        if pdh.PdhOpenQueryW(None, 0, ctypes.byref(self._gpu_query)) != 0:
            self.errors.append("PdhOpenQueryW (GPU memory) failed")
            self._gpu_query = ctypes.c_void_p()
            return
        for key, path in GPU_MEMORY_COUNTERS.items():
            handle = ctypes.c_void_p()
            status = pdh.PdhAddEnglishCounterW(self._gpu_query, path, 0, ctypes.byref(handle))
            if status != 0:
                self.errors.append(f"{path}: PDH status {status & 0xFFFFFFFF:#010x}")
                continue
            self._gpu_handles[key] = handle

    def start(self) -> None:
        self._threads.append(threading.Thread(target=self._nvsmi_loop, daemon=True))
        if self._gpu_handles:
            self._threads.append(threading.Thread(target=self._gpu_memory_loop, daemon=True))
        super().start()  # starts every thread in self._threads, these two included

    def stop(self) -> None:
        super().stop()
        if self._gpu_query:
            self._pdh.PdhCloseQuery(self._gpu_query)
            self._gpu_query = ctypes.c_void_p()

    def _nvsmi_loop(self) -> None:
        query = [f"--query-gpu={NVSMI_FIELDS}", "--format=csv,noheader,nounits"]
        while True:
            row = parse_nvsmi(gate._run_quiet(["nvidia-smi", *query]))
            self.gpu_rows.append([time.time(), *row])
            if self._stop.wait(NVSMI_POLL_S):
                return

    def counter_array(self, handle: ctypes.c_void_p) -> Dict[str, float]:
        """Every instance of one wildcard counter at the last collection: ``{name: value}``."""
        fmt = gate._PDH_FMT_DOUBLE | gate._PDH_FMT_NOCAP100
        size, count = wt.DWORD(0), wt.DWORD(0)
        status = self._pdh.PdhGetFormattedCounterArrayW(
            handle, fmt, ctypes.byref(size), ctypes.byref(count), None
        )
        if (status & 0xFFFFFFFF) != _PDH_MORE_DATA or size.value == 0:
            return {}
        buffer = ctypes.create_string_buffer(size.value)
        status = self._pdh.PdhGetFormattedCounterArrayW(
            handle, fmt, ctypes.byref(size), ctypes.byref(count), buffer
        )
        if status != 0 or count.value == 0:
            return {}
        items = ctypes.cast(buffer, ctypes.POINTER(_PdhItem * count.value)).contents
        return {
            str(item.szName): float(item.FmtValue.doubleValue)
            for item in items
            if item.FmtValue.CStatus in _PDH_CSTATUS_OK
        }

    def _gpu_memory_loop(self) -> None:
        while not self._stop.wait(GPU_MEMORY_POLL_S):
            try:
                if (
                    self.pid
                    and not self._pid_seen
                    and time.monotonic() - self._gpu_opened_at >= GPU_QUERY_REOPEN_S
                ):
                    self._open_gpu_query()
                if not self._gpu_query or self._pdh.PdhCollectQueryData(self._gpu_query) != 0:
                    continue
                prefix = f"pid_{self.pid}_" if self.pid else None
                row: Dict[str, Dict[str, float]] = {}
                for key, handle in self._gpu_handles.items():
                    values = self.counter_array(handle)
                    if key.startswith("process_"):
                        values = {
                            name[len(prefix) :]: value
                            for name, value in values.items()
                            if prefix and name.startswith(prefix)
                        }
                        self._pid_seen = self._pid_seen or bool(values)
                    row[key] = values
                self.gpu_memory_rows.append([time.time(), row])
            except Exception as exc:  # a diagnostic must never stop the arm
                self.errors.append(f"GPU memory sample: {type(exc).__name__}: {exc}")

    def summary(self) -> Dict[str, Any]:
        out = super().summary()

        def column(index: int) -> List[float]:
            return [row[index] for row in self.gpu_rows if row[index] is not None]

        clocks = column(3)
        out.update(
            {
                "nvsmi_samples": len(self.gpu_rows),
                "nvsmi_used_mib_max": max(column(1), default=None),
                "nvsmi_total_mib": max(column(2), default=None),
                "sm_mhz_min_max": [min(clocks), max(clocks)] if clocks else None,
                "temp_c_max": max(column(4), default=None),
                "power_w_max": max(column(5), default=None),
                "throttle_reasons_seen": sorted({str(row[7]) for row in self.gpu_rows}),
                "gpu_memory_samples": len(self.gpu_memory_rows),
                "gpu_query_opens": self.gpu_query_opens,
                "gpu_pid_seen": self._pid_seen,
                **gpu_memory_summary(self.gpu_memory_rows),
            }
        )
        return out


def gpu_memory_summary(rows: Sequence[Sequence[Any]]) -> Dict[str, Any]:
    """Peak dedicated and shared usage of the busiest adapter, and of the arm's own process.

    The adapter is the one whose dedicated usage peaked highest during the arm: the integrated
    GPU has no dedicated memory, so on this box that is the RTX 5070 whenever an arm ran.
    """
    peaks: Dict[str, Dict[str, float]] = {}
    for _stamp, row in rows:
        for key, values in row.items():
            bucket = peaks.setdefault(key, {})
            for name, value in values.items():
                bucket[name] = max(bucket.get(name, value), value)
    dedicated = peaks.get("adapter_dedicated", {})
    adapter = max(dedicated, key=dedicated.get) if dedicated else None
    shared_first = next(
        (
            row["adapter_shared"][adapter]
            for _stamp, row in rows
            if adapter in row.get("adapter_shared", {})
        ),
        None,
    )
    process_dedicated = peaks.get("process_dedicated", {})
    process_shared = peaks.get("process_shared", {})

    def gigabytes(value: Optional[float]) -> Optional[float]:
        return value / 1e9 if value is not None else None

    return {
        "gpu_adapter": adapter,
        "adapter_dedicated_gb_max": gigabytes(dedicated.get(adapter) if adapter else None),
        "adapter_shared_gb_first": gigabytes(shared_first),
        "adapter_shared_gb_max": gigabytes(
            peaks.get("adapter_shared", {}).get(adapter) if adapter else None
        ),
        "process_dedicated_gb_max": gigabytes(max(process_dedicated.values(), default=None)),
        "process_shared_gb_max": gigabytes(max(process_shared.values(), default=None)),
    }


def self_test_sampler(seconds: float = 4.0) -> Dict[str, Any]:
    """Dry run only: run the sampler for a few seconds with no arm and report what it read."""
    sampler = BatchSampler()
    sampler.pid = os.getpid()
    sampler.start()
    time.sleep(seconds)
    sampler.stop()
    summary = sampler.summary()
    adapters = sorted(
        {name for _t, row in sampler.gpu_memory_rows for name in row.get("adapter_dedicated", {})}
    )
    return {
        "errors": sampler.errors,
        "pdh_samples": summary["pdh_samples"],
        "power_samples": summary["power_samples"],
        "nvsmi_samples": summary["nvsmi_samples"],
        "nvsmi_last": sampler.gpu_rows[-1][1:] if sampler.gpu_rows else None,
        "gpu_memory_samples": summary["gpu_memory_samples"],
        "gpu_adapters_seen": adapters,
        "adapter_shared_gb_first": summary["adapter_shared_gb_first"],
    }
