#!/usr/bin/env python3
"""D2 of the two-drive probe: does PCIe link power management slow one drive's reads?

The dev box's active power scheme has PCIe Link State Power Management at "Maximum power
savings" on AC (index 2), and D1 found the CPU package's idle state is not what makes each
drive read faster beside the other. This reads drive A alone with the scheme's AC ASPM index
at the box's own value and at 0 (Off), interleaved arm by arm, then — with ``--r1-out`` —
runs ``two_drive_read.py`` (R1) once more with ASPM Off.

THIS CHANGES A SYSTEM POWER SETTING for the duration, so it is run only with the owner's
consent. The original index is read first and restored in a ``finally`` block whatever
happens, then read back; the script exits non-zero if the read-back disagrees. It refuses to
run on battery, where the AC index is not the one in force.

usage: two_drive_aspm_probe.py --drive-a DIR [DIR ...] --out JSON
                               [--drive-b DIR [DIR ...] --r1-out JSON]
                               [--threads 4] [--spans-per-arm 24] [--rounds 3]
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
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import torch
from issue974_unbuffered import SECTOR
from two_drive_read import (
    CounterSampler,
    Span,
    alone_rate,
    box_state,
    check_bytes,
    collect_spans,
    read_spans,
)

#: Arm orders per round; each state runs first at least once over three rounds.
ARM_ORDERS = (("on", "off"), ("off", "on"), ("on", "off"))

#: ASPM "Off" in the power scheme's SUB_PCIEXPRESS / ASPM setting.
ASPM_OFF = 0

#: Seconds after a policy change before the timed read starts.
SETTLE_S = 1.0

_AC_INDEX = re.compile(r"Current AC Power Setting Index:\s*(0x[0-9a-fA-F]+)")


class _SystemPowerStatus(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", ctypes.c_ubyte),
        ("BatteryFlag", ctypes.c_ubyte),
        ("BatteryLifePercent", ctypes.c_ubyte),
        ("SystemStatusFlag", ctypes.c_ubyte),
        ("BatteryLifeTime", wt.DWORD),
        ("BatteryFullLifeTime", wt.DWORD),
    ]


def on_ac_power() -> bool:
    status = _SystemPowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        return False
    return status.ACLineStatus == 1


def _powercfg(*args: str) -> str:
    done = subprocess.run(
        ["powercfg", *args],
        capture_output=True,
        encoding="oem",
        errors="replace",
        timeout=30,
        check=False,
    )
    if done.returncode != 0:
        raise RuntimeError(f"powercfg {' '.join(args)} failed ({done.returncode}): {done.stderr}")
    return done.stdout


def read_aspm_ac() -> int:
    match = _AC_INDEX.search(_powercfg("/q", "SCHEME_CURRENT", "SUB_PCIEXPRESS", "ASPM"))
    if match is None:
        raise RuntimeError("could not read the active scheme's AC ASPM index")
    return int(match.group(1), 16)


def set_aspm_ac(index: int) -> None:
    _powercfg("/setacvalueindex", "SCHEME_CURRENT", "SUB_PCIEXPRESS", "ASPM", str(index))
    _powercfg("/setactive", "SCHEME_CURRENT")
    now = read_aspm_ac()
    if now != index:
        raise RuntimeError(f"ASPM AC index reads {now} after setting {index}")


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--drive-a", nargs="+", required=True, help="directories on drive A")
    parser.add_argument("--drive-b", nargs="+", help="drive B, only for the R1 re-run")
    parser.add_argument("--r1-out", help="also run R1 with ASPM Off, writing this JSON")
    parser.add_argument("--threads", type=int, default=4, help="parallel ranges per span (K)")
    parser.add_argument("--spans-per-arm", type=int, default=24)
    parser.add_argument("--span-mib", type=int, default=400)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--label", default="")
    parser.add_argument("--out", required=True, help="JSON results, rewritten after each round")
    return parser.parse_args()


def run_r1(args: argparse.Namespace) -> int:
    harness = os.path.join(os.path.dirname(os.path.abspath(__file__)), "two_drive_read.py")
    cmd = [
        sys.executable,
        harness,
        "--drive-a",
        *args.drive_a,
        "--drive-b",
        *args.drive_b,
        "--threads",
        str(args.threads),
        "--spans-per-arm",
        str(args.spans_per_arm),
        "--span-mib",
        str(args.span_mib),
        "--rounds",
        str(args.rounds),
        "--label",
        "R1 run 3, ASPM Off (AC index 0) set by two_drive_aspm_probe.py",
        "--out",
        args.r1_out,
    ]
    return subprocess.run(cmd, check=False).returncode


def main() -> int:
    if sys.platform != "win32":
        raise SystemExit("two_drive_aspm_probe.py changes a Windows power setting; Windows only")
    args = parse()
    if args.r1_out and not args.drive_b:
        raise SystemExit("--r1-out needs --drive-b")
    if not on_ac_power():
        raise SystemExit("on battery: the AC ASPM index is not the one in force; refusing")
    span_bytes = args.span_mib * 1024 * 1024
    torch.cuda.init()
    torch.zeros(1, device="cuda")

    spans = collect_spans(args.drive_a, span_bytes)
    needed = 1 + args.rounds * 2 * args.spans_per_arm
    if len(spans) < needed:
        raise SystemExit(f"{len(spans)} spans available, {needed} needed — none read twice")
    cursor = 0

    def take(count: int) -> List[Span]:
        nonlocal cursor
        cursor += count
        return spans[cursor - count : cursor]

    arena = torch.empty(span_bytes, dtype=torch.uint8, device="cpu", pin_memory=True)
    if arena.data_ptr() % SECTOR:
        raise SystemExit(f"arena is not sector-aligned: {arena.data_ptr()}")
    letter = os.path.splitdrive(os.path.abspath(args.drive_a[0]))[0].upper()
    original = read_aspm_ac()
    states = {"on": original, "off": ASPM_OFF}
    record: Dict[str, object] = {
        "label": args.label,
        "args": vars(args),
        "volume": letter,
        "span_bytes": span_bytes,
        "aspm_ac_index_original": original,
        "aspm_states": states,
        "box_start": box_state(),
        "rounds": [],
    }

    def write() -> None:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=1)
            handle.write("\n")

    sampler = CounterSampler(
        {
            "read": rf"\LogicalDisk({letter})\Disk Read Bytes/sec",
            "write": rf"\LogicalDisk({letter})\Disk Write Bytes/sec",
            "cpu_pct": r"\Processor(_Total)\% Processor Time",
            "cpu_perf_pct": r"\Processor Information(_Total)\% Processor Performance",
        }
    )
    record["counter_errors"] = sampler.errors
    executor = ThreadPoolExecutor(max_workers=args.threads)
    restored: Optional[int] = None
    r1_exit: Optional[int] = None
    sampler.start()
    try:
        read_spans(take(1), arena.data_ptr(), args.threads, executor)  # untimed warm-up
        for round_index in range(args.rounds):
            entry: Dict[str, object] = {"round": round_index, "box": box_state(), "wall": {}}
            for arm in ARM_ORDERS[round_index % len(ARM_ORDERS)]:
                set_aspm_ac(states[arm])
                time.sleep(SETTLE_S)
                chunk = take(args.spans_per_arm)
                wall_start = time.time()
                log = read_spans(chunk, arena.data_ptr(), args.threads, executor)
                entry["wall"][arm] = [wall_start, time.time()]
                entry[f"{arm}_gb_s"] = alone_rate(log)
                entry[f"{arm}_span_s"] = [round(end - start, 4) for start, end, _n in log]
                entry[f"check_{arm}"] = check_bytes(chunk[-1], arena)
                print(f"round {round_index}  ASPM {arm:<3}  {entry[f'{arm}_gb_s']:6.2f} GB/s")
            entry["off_over_on"] = entry["off_gb_s"] / entry["on_gb_s"]
            print(f"round {round_index}  off/on {entry['off_over_on']:.3f}")
            record["rounds"].append(entry)
            write()
        executor.shutdown(wait=True)
        if args.r1_out:
            set_aspm_ac(ASPM_OFF)
            time.sleep(SETTLE_S)
            print("R1 run 3 with ASPM Off:")
            r1_exit = run_r1(args)
    finally:
        executor.shutdown(wait=True)
        set_aspm_ac(original)
        restored = read_aspm_ac()
        print(f"ASPM AC index restored to {restored} (original {original})")
        time.sleep(0.6)  # let the last arm's closing counter interval land
        sampler.stop()

    rounds = record["rounds"]
    for entry in rounds:
        entry["counters"] = {arm: sampler.mean_over(*wall) for arm, wall in entry["wall"].items()}
    record["median"] = {
        key: statistics.median(entry[key] for entry in rounds)
        for key in ("on_gb_s", "off_gb_s", "off_over_on")
    }
    checks = [value for entry in rounds for key, value in entry.items() if key.startswith("check_")]
    record["all_byte_checks_ok"] = all(checks)
    record["aspm_ac_index_restored"] = restored
    record["r1_exit"] = r1_exit
    record["counter_samples"] = sampler.samples
    record["box_end"] = box_state()
    write()
    print("median " + "  ".join(f"{k} {v:.3f}" for k, v in record["median"].items()))
    print(f"byte checks: {'all OK' if record['all_byte_checks_ok'] else 'MISMATCH'}")
    ok = record["all_byte_checks_ok"] and restored == original and r1_exit in (None, 0)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
