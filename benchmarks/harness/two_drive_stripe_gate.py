#!/usr/bin/env python3
"""Driver for the two-drive striping gate (R4): one ``stream_probe.py`` process per arm.

The record is ``benchmarks/gate-two-drive-striping.md``; its decision rule (§2) and this
protocol (§4) were committed before any arm ran. This script runs the record's §1 command
once per arm, never two at once, in the order the rule fixes:

    round 0: SINGLE, STRIPED   round 1: STRIPED, SINGLE   round 2: SINGLE, STRIPED

Around every arm it takes the §4 box stamp (before and after), and while the arm runs it
samples AC power and commit every 2 s and Windows performance counters (PDH, English names)
every 1 s: per-volume read/write bytes for C: and D:, paging, commit, CPU. The harness only
READS the two drives, so a write on either during an arm is someone else's I/O. Since
sequence 3, each stamp also records every process's cumulative read/write bytes
(``Win32_Process``, one CIM query) and whether the battery is charging, and the after stamp
names the processes that read most during the arm; diagnostic only, it gates nothing.

It decides nothing about shipping. It applies only the two rows of the rule a single arm
can settle (a SINGLE arm outside 15.0-21.5 s; ``direct_io`` or ``pinned`` not true), and
stops when one fires, because no later arm can turn a no-verdict outcome into a verdict.
An arm that sees battery power, that the box was suspended in (two consecutive 2-s power
samples more than 30 s apart: a closed lid sleeps this laptop, and a suspended step would
read as a slow one), or that writes no ``step_plain`` point, is void: kept under a ``_void``
name and re-run once in the same position.

``--plan build`` runs one untimed arm (``--steps 1 --warmup 0 --step --skip-events``) to
build a cache before the rounds. No torch import: the driver must not charge commit or
touch the GPU itself. Windows only (GetSystemPowerStatus, PDH, powercfg).

usage: two_drive_stripe_gate.py [--plan rounds|build] [--build-arm STRIPED|SINGLE]
                                [--settle-s 60] [--dry-run]
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

WORKTREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(WORKTREE, "src").replace("\\", "/")
OUT_REL = "benchmarks/results/probe-rtx5070/two-drive"
OUT_DIR = os.path.join(WORKTREE, *OUT_REL.split("/"))
BOX_LOG = os.path.join(OUT_DIR, "gate_box_state.log")
DRIVER_JSON = os.path.join(OUT_DIR, "gate_driver.json")
WEIGHTS = "D:/synth/llama-70b-shape-v32k-f4"

ARM_FLAGS: Dict[str, List[str]] = {
    "SINGLE": ["--read-ahead", "2"],
    "STRIPED": [
        "--shards",
        "C:/Users/user/.soup/layer-stream-striped/D___synth__llama-70b-shape-v32k-f4",
        "--stripe-dir",
        "D:/soup-stripe",
        "--read-ahead",
        "3",
    ],
}
#: The order the rule fixes: (round, arm).
ROUND_PLAN: Tuple[Tuple[int, str], ...] = (
    (0, "SINGLE"),
    (0, "STRIPED"),
    (1, "STRIPED"),
    (1, "SINGLE"),
    (2, "SINGLE"),
    (2, "STRIPED"),
)
SINGLE_VALID = (15.0, 21.5)  # the rule's first row, inclusive
HEADROOM_MIN = 8 * 2**30  # commit headroom before an arm, bytes
GPU_MIB_MAX = 1024  # GPU memory in use before an arm
PRECHECK_WAIT_S = 15 * 60
PRECHECK_POLL_S = 30
AC_POLL_S = 2.0
PDH_POLL_S = 1.0
SUSPEND_GAP_S = 30.0  # a gap this long between 2-s power samples means the box slept
_AC_INDEX = re.compile(r"Current AC Power Setting Index:\s*(0x[0-9a-fA-F]+)")
_BATTERY_FLAG_UNKNOWN = 255  # GetSystemPowerStatus: battery status unknown
_BATTERY_CHARGING = 0x08  # GetSystemPowerStatus BatteryFlag bit: the battery is charging
PROC_IO_TOP = 8  # processes listed in an after stamp, by read bytes gained during the arm
# One CIM query; fields joined by a tab ([char]9) so the command needs no quoting.
_PROC_IO_QUERY = (
    "Get-CimInstance Win32_Process -Property ProcessId,Name,ReadTransferCount,"
    "WriteTransferCount | ForEach-Object { ($_.ProcessId,$_.ReadTransferCount,"
    "$_.WriteTransferCount,$_.Name) -join [char]9 }"
)


# ==========================================================================
# box state
# ==========================================================================
class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", wt.DWORD),
        ("dwMemoryLoad", wt.DWORD),
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
        ("BatteryLifeTime", wt.DWORD),
        ("BatteryFullLifeTime", wt.DWORD),
    ]


def power() -> Tuple[int, int]:
    """(ACLineStatus, battery %): 1 = on AC, 0 = on battery, 255 = unknown."""
    status = _SystemPowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        return 255, 255
    return int(status.ACLineStatus), int(status.BatteryLifePercent)


def battery_charging() -> Tuple[int, Optional[bool]]:
    """(BatteryFlag, charging); charging is None when Windows reports the status unknown."""
    status = _SystemPowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        return _BATTERY_FLAG_UNKNOWN, None
    flag = int(status.BatteryFlag)
    if flag == _BATTERY_FLAG_UNKNOWN:
        return flag, None
    return flag, bool(flag & _BATTERY_CHARGING)


def memory() -> Dict[str, int]:
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return {
        "avail_phys": int(status.ullAvailPhys),
        "commit_limit": int(status.ullTotalPageFile),
        "commit_used": int(status.ullTotalPageFile - status.ullAvailPageFile),
        "commit_headroom": int(status.ullAvailPageFile),
    }


def _run_quiet(cmd: Sequence[str], env: Optional[Dict[str, str]] = None) -> Optional[str]:
    # Console tools print in the OEM code page (cp866 on this box); decode leniently.
    try:
        done = subprocess.run(
            list(cmd),
            capture_output=True,
            encoding="oem",
            errors="replace",
            timeout=30,
            check=False,
            env=env,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def gpu_state() -> Tuple[Optional[int], Optional[str]]:
    used = _run_quiet(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    apps = _run_quiet(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ]
    )
    try:
        used_mib = int(used.strip().splitlines()[0]) if used else None
    except (ValueError, IndexError):
        used_mib = None
    return used_mib, (apps.strip().replace("\n", " | ") if apps is not None else None)


def aspm_ac_index() -> Optional[int]:
    out = _run_quiet(["powercfg", "/q", "SCHEME_CURRENT", "SUB_PCIEXPRESS", "ASPM"])
    match = _AC_INDEX.search(out or "")
    return int(match.group(1), 16) if match else None


def python_processes() -> Optional[int]:
    tasks = _run_quiet(["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV", "/NH"])
    if tasks is None:
        return None
    return sum(1 for line in tasks.splitlines() if line.startswith('"python'))


def arm_env() -> Dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def soup_cli_file(env: Dict[str, str]) -> Optional[str]:
    out = _run_quiet([sys.executable, "-c", "import soup_cli; print(soup_cli.__file__)"], env)
    return out.strip() if out else None


def process_io() -> Tuple[Optional[List[List[Any]]], Optional[str]]:
    """Every process's cumulative I/O bytes: ``[[pid, name, read, write], ...]`` or an error.

    Diagnostic only (added after sequence 2, before sequence 3): it names who else read a
    drive during a disturbed arm, and it never fails the arm; any failure comes back as the
    error string. Counters are ``Win32_Process`` ``ReadTransferCount``/``WriteTransferCount``
    (all I/O the process issued, any device), from one CIM query.
    """
    try:
        done = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", _PROC_IO_QUERY],
            capture_output=True,
            encoding="oem",
            errors="replace",
            timeout=60,
            check=False,
        )
        if done.returncode != 0:
            # One line, so the box-state log keeps one line per stamp.
            return None, f"exit {done.returncode}: {' '.join(done.stderr.split())[:300]}"
        rows: List[List[Any]] = []
        for text in done.stdout.splitlines():
            parts = text.split("\t", 3)
            if len(parts) != 4:
                continue
            try:
                rows.append([int(parts[0]), parts[3], int(parts[1] or 0), int(parts[2] or 0)])
            except ValueError:
                continue
        return (rows, None) if rows else (None, "no process rows parsed")
    except Exception as exc:  # a diagnostic must never fail the arm
        return None, f"{type(exc).__name__}: {' '.join(str(exc).split())[:300]}"


def top_readers(before: List[List[Any]], after: List[List[Any]]) -> List[Dict[str, Any]]:
    """The PROC_IO_TOP processes whose read bytes grew most between two stamps of one arm.

    Keyed by pid and name, so a reused pid under another name reads as a new process. A
    process absent from ``before``, or whose counter went down, started during the arm and
    counts from 0. The arm's own process has exited by the after stamp, and a process that
    started and exited within the arm is in neither stamp: the list names the survivors.
    """
    earlier = {(row[0], row[1]): row for row in before}
    ranked: List[Dict[str, Any]] = []
    for pid, name, read, write in after:
        prev = earlier.get((pid, name))
        new = prev is None or read < prev[2] or write < prev[3]
        ranked.append(
            {
                "pid": pid,
                "name": name,
                "read_delta": read if new else read - prev[2],
                "write_delta": write if new else write - prev[3],
                "new": new,
            }
        )
    ranked.sort(key=lambda row: row["read_delta"], reverse=True)
    return ranked[:PROC_IO_TOP]


def stamp(
    run: str, when: str, env: Dict[str, str], before: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    ac_line, battery = power()
    battery_flag, charging = battery_charging()
    mem = memory()
    gpu_used, apps = gpu_state()
    io_rows, io_error = process_io()
    state: Dict[str, Any] = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run": run,
        "when": when,
        "ac_line_status": ac_line,
        "battery_pct": battery,
        "battery_flag": battery_flag,
        "battery_charging": charging,
        **mem,
        "python_processes": python_processes(),
        "gpu_mem_used_mib": gpu_used,
        "compute_apps": apps,
        "aspm_ac_index": aspm_ac_index(),
        "soup_cli_file": soup_cli_file(env),
        "process_io": io_rows,
        "process_io_error": io_error,
    }
    top_text = ""
    if before is not None:
        try:
            if io_rows and before.get("process_io"):
                state["top_readers"] = top_readers(before["process_io"], io_rows)
                top_text = (
                    " top_read=["
                    + ", ".join(
                        f"{row['name']}({row['pid']}{',new' if row['new'] else ''}) "
                        f"{row['read_delta'] / 1e6:.1f}MB"
                        for row in state["top_readers"]
                    )
                    + "]"
                )
            else:
                state["top_readers"] = None
        except Exception as exc:  # a diagnostic must never fail the arm
            state["top_readers"] = None
            state["process_io_error"] = f"top_readers {type(exc).__name__}: {exc}"
    error_text = (
        f" process_io_error={state['process_io_error']}" if state["process_io_error"] else ""
    )
    line = (
        f"{state['time']} run={run} when={when} AC={ac_line} battery={battery}% "
        f"charging={charging} "
        f"avail_phys={mem['avail_phys'] / 1e9:.2f}GB "
        f"commit={mem['commit_used'] / 1e9:.2f}/{mem['commit_limit'] / 1e9:.2f}GB "
        f"headroom={mem['commit_headroom'] / 2**30:.2f}GiB "
        f"python_processes={state['python_processes']} gpu_used={gpu_used}MiB "
        f"compute_apps=[{apps}] aspm_ac={state['aspm_ac_index']} "
        f"soup_cli={state['soup_cli_file']}"
        f"{top_text}{error_text}"
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(BOX_LOG, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)
    return state


def precheck_ok() -> Tuple[bool, str]:
    ac_line, battery = power()
    headroom = memory()["commit_headroom"]
    gpu_used, _apps = gpu_state()
    reasons = []
    if ac_line != 1:
        reasons.append(f"AC={ac_line} battery={battery}%")
    if gpu_used is None or gpu_used > GPU_MIB_MAX:
        reasons.append(f"gpu_used={gpu_used}MiB")
    if headroom < HEADROOM_MIN:
        reasons.append(f"headroom={headroom / 2**30:.2f}GiB")
    return not reasons, "; ".join(reasons) or "ok"


def wait_for_precheck() -> Tuple[bool, List[str]]:
    deadline = time.monotonic() + PRECHECK_WAIT_S
    notes: List[str] = []
    while True:
        ok, why = precheck_ok()
        if ok:
            return True, notes
        note = f"{time.strftime('%H:%M:%S')} pre-arm check not met: {why}"
        notes.append(note)
        print(note, flush=True)
        if time.monotonic() >= deadline:
            return False, notes
        time.sleep(PRECHECK_POLL_S)


# ==========================================================================
# sampling while an arm runs
# ==========================================================================
class _PdhValue(ctypes.Structure):
    _fields_ = [("CStatus", wt.DWORD), ("doubleValue", ctypes.c_double)]


_PDH_FMT_DOUBLE = 0x00000200
_PDH_FMT_NOCAP100 = 0x00008000

COUNTERS = {
    "read_c": r"\LogicalDisk(C:)\Disk Read Bytes/sec",
    "write_c": r"\LogicalDisk(C:)\Disk Write Bytes/sec",
    "read_d": r"\LogicalDisk(D:)\Disk Read Bytes/sec",
    "write_d": r"\LogicalDisk(D:)\Disk Write Bytes/sec",
    "pages_s": r"\Memory\Pages/sec",
    "committed_bytes": r"\Memory\Committed Bytes",
    "cpu_pct": r"\Processor(_Total)\% Processor Time",
    "cpu_perf_pct": r"\Processor Information(_Total)\% Processor Performance",
}


class Sampler:
    """PDH counters every PDH_POLL_S and power/commit every AC_POLL_S, on two threads."""

    def __init__(self) -> None:
        self.pdh_rows: List[Tuple[float, Dict[str, float]]] = []
        self.power_rows: List[Tuple[float, int, int, int]] = []
        self.errors: List[str] = []
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self._query = ctypes.c_void_p()
        self._handles: Dict[str, ctypes.c_void_p] = {}
        pdh = ctypes.windll.pdh
        pdh.PdhOpenQueryW.restype = ctypes.c_long
        pdh.PdhOpenQueryW.argtypes = [ctypes.c_wchar_p, ctypes.c_size_t, ctypes.c_void_p]
        pdh.PdhAddEnglishCounterW.restype = ctypes.c_long
        pdh.PdhAddEnglishCounterW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_wchar_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        pdh.PdhCollectQueryData.restype = ctypes.c_long
        pdh.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
        pdh.PdhGetFormattedCounterValue.restype = ctypes.c_long
        pdh.PdhGetFormattedCounterValue.argtypes = [
            ctypes.c_void_p,
            wt.DWORD,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        pdh.PdhCloseQuery.restype = ctypes.c_long
        pdh.PdhCloseQuery.argtypes = [ctypes.c_void_p]
        self._pdh = pdh
        if pdh.PdhOpenQueryW(None, 0, ctypes.byref(self._query)) != 0:
            self.errors.append("PdhOpenQueryW failed")
            return
        for key, path in COUNTERS.items():
            handle = ctypes.c_void_p()
            status = pdh.PdhAddEnglishCounterW(self._query, path, 0, ctypes.byref(handle))
            if status != 0:
                self.errors.append(f"{path}: PDH status {status & 0xFFFFFFFF:#010x}")
                continue
            self._handles[key] = handle

    def start(self) -> None:
        if self._handles:
            self._pdh.PdhCollectQueryData(self._query)  # a rate counter needs a first sample
            self._threads.append(threading.Thread(target=self._pdh_loop, daemon=True))
        self._threads.append(threading.Thread(target=self._power_loop, daemon=True))
        for thread in self._threads:
            thread.start()

    def _pdh_loop(self) -> None:
        while not self._stop.wait(PDH_POLL_S):
            if self._pdh.PdhCollectQueryData(self._query) != 0:
                continue
            row: Dict[str, float] = {}
            for key, handle in self._handles.items():
                value = _PdhValue()
                status = self._pdh.PdhGetFormattedCounterValue(
                    handle, _PDH_FMT_DOUBLE | _PDH_FMT_NOCAP100, None, ctypes.byref(value)
                )
                if status == 0:
                    row[key] = value.doubleValue
            self.pdh_rows.append((time.time(), row))

    def _power_loop(self) -> None:
        while True:
            ac_line, battery = power()
            self.power_rows.append((time.time(), ac_line, battery, memory()["commit_headroom"]))
            if self._stop.wait(AC_POLL_S):
                return

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join()
        if self._query:
            self._pdh.PdhCloseQuery(self._query)

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "pdh_samples": len(self.pdh_rows),
            "pdh_errors": list(self.errors),
            "power_samples": len(self.power_rows),
            "ac_min": min((row[1] for row in self.power_rows), default=None),
            "ac_values_seen": sorted({row[1] for row in self.power_rows}),
            "battery_pct_first_last": (
                [self.power_rows[0][2], self.power_rows[-1][2]] if self.power_rows else None
            ),
            "commit_headroom_min_gib": (
                min(row[3] for row in self.power_rows) / 2**30 if self.power_rows else None
            ),
        }
        for key in COUNTERS:
            values = [row[key] for _stamp, row in self.pdh_rows if key in row]
            if values:
                out[f"{key}_mean"] = statistics.fmean(values)
                out[f"{key}_max"] = max(values)
        return out


# ==========================================================================
# one arm
# ==========================================================================
def harness_command(arm: str, label: str, out_rel: str, build: bool) -> List[str]:
    if build:
        steps = ["--steps", "1", "--warmup", "0", "--step", "--skip-events"]
    else:
        steps = ["--steps", "3", "--warmup", "1", "--step"]
    return [
        sys.executable,
        "benchmarks/harness/stream_probe.py",
        "--weights",
        WEIGHTS,
        "--quant",
        "nf4",
        "--tier",
        "disk",
        "--seq",
        "512",
        "--batch",
        "1",
        *steps,
        *ARM_FLAGS[arm],
        "--label",
        label,
        "--out",
        out_rel,
    ]


def step_plain(json_path: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    try:
        with open(json_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None, {}
    meta = payload.get("meta", {})
    for record in payload.get("records", []):
        if record.get("kind") == "step" and record.get("label") == "step_plain":
            return record, meta
    return None, meta


def _void_name(path: str) -> str:
    stem, ext = os.path.splitext(path)
    candidate = f"{stem}_void{ext}"
    count = 2
    while os.path.exists(candidate):
        candidate = f"{stem}_void{count}{ext}"
        count += 1
    return candidate


def run_arm(name: str, arm: str, label: str, build: bool, dry_run: bool) -> Dict[str, Any]:
    env = arm_env()
    out_rel = f"{OUT_REL}/{name}.json"
    json_path = os.path.join(OUT_DIR, f"{name}.json")
    log_path = os.path.join(OUT_DIR, f"{name}.log")
    cmd = harness_command(arm, label, out_rel, build)
    entry: Dict[str, Any] = {"run": name, "arm": arm, "label": label, "command": cmd}
    entry["before"] = stamp(name, "before", env)
    if dry_run:
        print("DRY RUN:", subprocess.list2cmdline(cmd), flush=True)
        entry["dry_run"] = True
        return entry
    sampler = Sampler()
    sampler.start()
    started = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"# {time.strftime('%Y-%m-%dT%H:%M:%S')} {subprocess.list2cmdline(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=WORKTREE, env=env, stdout=log, stderr=subprocess.STDOUT)
        entry["pid"] = proc.pid
        exit_code = proc.wait()
    ended = time.time()
    sampler.stop()
    entry["exit_code"] = exit_code
    entry["started_unix"] = started
    entry["ended_unix"] = ended
    entry["wall_s"] = ended - started
    entry["during"] = sampler.summary()
    entry["during_samples"] = {
        "power": [list(row) for row in sampler.power_rows],
        "pdh": [[stamp_s, row] for stamp_s, row in sampler.pdh_rows],
    }
    entry["after"] = stamp(name, "after", env, before=entry["before"])
    plain, meta = step_plain(json_path)
    entry["step_plain_mean_s"] = plain["step_s_mean"] if plain else None
    entry["direct_io"] = meta.get("direct_io")
    entry["pinned"] = meta.get("pinned")
    entry["shard_seconds"] = meta.get("shard_seconds")
    entry["layer_roots_head"] = (meta.get("layer_roots") or [])[:8]
    entry["stripe_roots"] = meta.get("stripe_roots")
    # Gaps between consecutive power samples, and from the last one to the arm's end: the
    # sampler thread freezes with the rest of the box, so a suspend shows up as one long gap.
    sample_times = [row[0] for row in sampler.power_rows] + [ended]
    max_gap = max((b - a for a, b in zip(sample_times, sample_times[1:])), default=None)
    entry["during"]["max_power_sample_gap_s"] = max_gap
    void_reasons = []
    if entry["during"]["ac_min"] != 1 or entry["after"]["ac_line_status"] != 1:
        void_reasons.append(f"AC values seen {entry['during']['ac_values_seen']}")
    if max_gap is None or max_gap > SUSPEND_GAP_S:
        void_reasons.append(f"suspended: {max_gap} s between power samples")
    if plain is None:
        void_reasons.append(f"no step_plain point (exit code {exit_code})")
    entry["void_reasons"] = void_reasons
    if void_reasons:
        for path in (json_path, log_path):
            if os.path.exists(path):
                target = _void_name(path)
                os.replace(path, target)
                entry.setdefault("void_files", []).append(os.path.basename(target))
    print(
        f"{name}: exit {exit_code}, wall {entry['wall_s']:.1f} s, step_plain "
        f"{entry['step_plain_mean_s']}, direct_io {entry['direct_io']}, pinned "
        f"{entry['pinned']}, void {void_reasons or 'no'}",
        flush=True,
    )
    return entry


def stop_row(entry: Dict[str, Any]) -> Optional[str]:
    """The two rows of the rule one arm can settle; None when neither fires."""
    if entry.get("direct_io") is not True or entry.get("pinned") is not True:
        return (
            f"no verdict: {entry['run']} has direct_io {entry.get('direct_io')} and pinned "
            f"{entry.get('pinned')}"
        )
    mean = entry.get("step_plain_mean_s")
    low, high = SINGLE_VALID
    if entry["arm"] == "SINGLE" and mean is not None and not low <= mean <= high:
        return f"no verdict: SINGLE arm {entry['run']} at {mean!r} s is outside {low}-{high} s"
    return None


# ==========================================================================
# main
# ==========================================================================
def load_driver_json() -> Dict[str, Any]:
    try:
        with open(DRIVER_JSON, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {"runs": [], "invocations": []}


def save_driver_json(payload: Dict[str, Any]) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = DRIVER_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
    os.replace(tmp, DRIVER_JSON)


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--plan", choices=("rounds", "build"), default="rounds")
    parser.add_argument("--build-arm", choices=tuple(ARM_FLAGS), default="STRIPED")
    parser.add_argument(
        "--settle-s",
        type=float,
        default=0.0,
        help="seconds to wait before the first arm (the record asks 60 after the build run)",
    )
    parser.add_argument("--dry-run", action="store_true", help="stamp and print, run nothing")
    parser.add_argument(
        "--run-prefix",
        default="gate",
        help="file-name prefix of the round arms, so a complete re-run does not overwrite an "
        "earlier sequence's committed files (default: gate)",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"gate[a-z0-9]{0,8}", args.run_prefix):
        parser.error("--run-prefix must be 'gate' plus at most 8 lowercase letters or digits")
    return args


def main() -> int:
    args = parse()
    if sys.platform != "win32":
        print("SKIP: this driver is Windows-only (GetSystemPowerStatus, PDH, powercfg)")
        return 0
    payload = load_driver_json()
    invocation = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "plan": args.plan,
        "build_arm": args.build_arm if args.plan == "build" else None,
        "settle_s": args.settle_s,
        "python": sys.executable,
        "pythonpath": SRC,
        "dry_run": args.dry_run,
        "run_prefix": args.run_prefix,
    }
    payload["invocations"].append(invocation)
    save_driver_json(payload)
    if args.plan == "build":
        plan = [("gate_build_" + args.build_arm.lower(), args.build_arm, None)]
    else:
        plan = [(f"{args.run_prefix}_r{rnd}_{arm.lower()}", arm, rnd) for rnd, arm in ROUND_PLAN]
    if args.settle_s > 0:
        print(f"settling {args.settle_s:.0f} s before the first arm", flush=True)
        time.sleep(args.settle_s)
    outcome = "complete"
    for position, (name, arm, rnd) in enumerate(plan):
        label = f"R4 gate build arm {arm}" if rnd is None else f"R4 gate round {rnd} arm {arm}"
        entry: Optional[Dict[str, Any]] = None
        for attempt in (1, 2):
            ok, notes = wait_for_precheck()
            if not ok:
                outcome = f"stopped: pre-arm checks not met within 15 min before {name}"
                payload["runs"].append({"run": name, "arm": arm, "precheck_notes": notes})
                break
            entry = run_arm(name, arm, label, build=rnd is None, dry_run=args.dry_run)
            entry["attempt"] = attempt
            entry["precheck_notes"] = notes
            entry["position"] = position
            payload["runs"].append(entry)
            save_driver_json(payload)
            if not entry.get("void_reasons"):
                break
            entry = None
        if outcome != "complete":
            break
        if entry is None:
            outcome = f"stopped: {name} void twice — no verdict"
            break
        if rnd is not None and not args.dry_run:
            row = stop_row(entry)
            if row is not None:
                outcome = f"stopped: {row}"
                break
    ran = {run.get("run") for run in payload["runs"] if run.get("exit_code") is not None}
    not_run = [name for name, _arm, _rnd in plan if name not in ran]
    invocation["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    invocation["outcome"] = outcome
    invocation["not_run"] = not_run if not args.dry_run else []
    save_driver_json(payload)
    print(f"OUTCOME: {outcome}; not run: {invocation['not_run']}", flush=True)
    if outcome == "complete":
        return 0
    return 5 if "pre-arm" in outcome else 4


if __name__ == "__main__":
    sys.exit(main())
