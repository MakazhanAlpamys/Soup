#!/usr/bin/env python3
"""D1 of the two-drive probe: is one drive's read held back by the host going idle?

In both R1 runs each drive read FASTER while the other drive was also reading, which a
shared bottleneck cannot produce. Two host-side mechanisms fit: the CPU package dropping
into a deep idle state between I/O completions, and PCIe link power management (ASPM is at
"Maximum power savings" on AC on the dev box). A busy loop on another core keeps the package
awake and cannot reach a link's ASPM, so reading one drive alone, with and without spinner
processes, separates the two.

Arms per round, in an order that rotates: the drive alone; the drive alone while N spinner
processes run ``while True: pass``. Same read primitive, spans, byte check and counters as
``two_drive_read.py`` (imported from it). The decision rule is in
``benchmarks/probe-rtx5070-two-drive-read.md`` (section D1), committed before this ran.

usage: two_drive_host_probe.py --drive DIR [DIR ...] --out JSON
                               [--spinners 2] [--threads 4] [--spans-per-arm 24] [--rounds 3]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

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

#: Arm orders per round; each arm runs first at least once over three rounds.
ARM_ORDERS = (("alone", "spin"), ("spin", "alone"), ("alone", "spin"))

#: Seconds the spinners run before the timed read starts, so the arm does not
#: time the processes starting.
SPIN_RAMP_S = 0.5


def start_spinners(count: int) -> List[subprocess.Popen]:
    return [subprocess.Popen([sys.executable, "-c", "while True: pass"]) for _ in range(count)]


def stop_spinners(procs: List[subprocess.Popen]) -> None:
    for proc in procs:
        proc.kill()
    for proc in procs:
        proc.wait(timeout=10)


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--drive", nargs="+", required=True, help="directories on the drive")
    parser.add_argument("--spinners", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4, help="parallel ranges per span (K)")
    parser.add_argument("--spans-per-arm", type=int, default=24)
    parser.add_argument("--span-mib", type=int, default=400)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--label", default="")
    parser.add_argument("--out", required=True, help="JSON results, rewritten after each round")
    return parser.parse_args()


def main() -> int:
    if sys.platform != "win32":
        raise SystemExit("two_drive_host_probe.py measures the Win32 #974 primitive; Windows only")
    args = parse()
    if args.spinners < 1:
        raise SystemExit("--spinners must be at least 1")
    span_bytes = args.span_mib * 1024 * 1024
    torch.cuda.init()
    torch.zeros(1, device="cuda")

    spans = collect_spans(args.drive, span_bytes)
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
    letter = os.path.splitdrive(os.path.abspath(args.drive[0]))[0].upper()
    sampler = CounterSampler(
        {
            "read": rf"\LogicalDisk({letter})\Disk Read Bytes/sec",
            "write": rf"\LogicalDisk({letter})\Disk Write Bytes/sec",
            "pages_s": r"\Memory\Pages/sec",
            "cpu_pct": r"\Processor(_Total)\% Processor Time",
            "cpu_perf_pct": r"\Processor Information(_Total)\% Processor Performance",
        }
    )
    sampler.start()
    record: Dict[str, object] = {
        "label": args.label,
        "args": vars(args),
        "volume": letter,
        "span_bytes": span_bytes,
        "counter_errors": sampler.errors,
        "box_start": box_state(),
        "rounds": [],
    }

    def write() -> None:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=1)
            handle.write("\n")

    executor = ThreadPoolExecutor(max_workers=args.threads)
    try:
        read_spans(take(1), arena.data_ptr(), args.threads, executor)  # untimed warm-up
        for round_index in range(args.rounds):
            entry: Dict[str, object] = {"round": round_index, "box": box_state(), "wall": {}}
            for arm in ARM_ORDERS[round_index % len(ARM_ORDERS)]:
                chunk = take(args.spans_per_arm)
                procs = start_spinners(args.spinners) if arm == "spin" else []
                try:
                    if procs:
                        time.sleep(SPIN_RAMP_S)
                    wall_start = time.time()
                    log = read_spans(chunk, arena.data_ptr(), args.threads, executor)
                    entry["wall"][arm] = [wall_start, time.time()]
                finally:
                    stop_spinners(procs)
                entry[f"{arm}_gb_s"] = alone_rate(log)
                entry[f"{arm}_span_s"] = [round(end - start, 4) for start, end, _n in log]
                entry[f"check_{arm}"] = check_bytes(chunk[-1], arena)
                print(f"round {round_index}  {arm:<5}  {entry[f'{arm}_gb_s']:6.2f} GB/s")
            entry["spin_over_alone"] = entry["spin_gb_s"] / entry["alone_gb_s"]
            print(f"round {round_index}  spin/alone {entry['spin_over_alone']:.3f}")
            record["rounds"].append(entry)
            write()
    finally:
        executor.shutdown(wait=True)
        time.sleep(0.6)  # let the last arm's closing counter interval land
        sampler.stop()

    rounds = record["rounds"]
    for entry in rounds:
        entry["counters"] = {arm: sampler.mean_over(*wall) for arm, wall in entry["wall"].items()}
        for arm, means in entry["counters"].items():
            print(
                f"round {entry['round']}  {arm:<5} counters: read {means.get('read', 0) / 1e6:.0f}"
                f" MB/s  write {means.get('write', 0) / 1e6:.1f} MB/s"
                f"  cpu {means.get('cpu_pct', float('nan')):.0f}%"
                f"  perf {means.get('cpu_perf_pct', float('nan')):.0f}%"
                f"  ({means['samples']:.0f} samples)"
            )
    record["median"] = {
        key: statistics.median(entry[key] for entry in rounds)
        for key in ("alone_gb_s", "spin_gb_s", "spin_over_alone")
    }
    checks = [value for entry in rounds for key, value in entry.items() if key.startswith("check_")]
    record["all_byte_checks_ok"] = all(checks)
    record["counter_samples"] = sampler.samples
    record["box_end"] = box_state()
    write()
    print("median " + "  ".join(f"{k} {v:.3f}" for k, v in record["median"].items()))
    print(f"byte checks: {'all OK' if record['all_byte_checks_ok'] else 'MISMATCH'}")
    return 0 if record["all_byte_checks_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
