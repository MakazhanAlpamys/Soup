#!/usr/bin/env python3
"""R3' of the read-rate track: does the reader's per-layer barrier cost one drive throughput?

``AsyncDiskSource._read_layer`` (before R4) handed a layer's K ranges to K workers and blocked
until every range had landed before the next layer was dispatched, so exactly one layer was in
flight and the workers that finished early idled on the slowest range's tail. This reads one
drive with the SAME K workers and the same ranges two ways, interleaved: depth 1 (the reader
before R4: dispatch a span, wait for all of it) and depth 2 (the next span's ranges are queued
behind the current one's, so a worker that frees up starts on it at once). The number of
requests outstanding never exceeds K in either arm; only the barrier differs. Both arms run
through one function, ``read_pipelined``, so the code path differs by the depth argument alone.

The decision rule is in ``benchmarks/probe-rtx5070-two-drive-read.md`` (section R3'),
committed before this ran. Refuses to run on battery, like every other arm of that record.

usage: two_drive_depth_probe.py --drive DIR [DIR ...] --out JSON
                                [--threads 4] [--spans-per-arm 24] [--rounds 3]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Deque, Dict, List, Sequence, Tuple

import torch
from issue974_unbuffered import SECTOR
from two_drive_aspm_probe import on_ac_power
from two_drive_read import (
    CounterSampler,
    Span,
    SpanLog,
    _read_range,
    alone_rate,
    box_state,
    check_bytes,
    collect_spans,
    span_ranges,
)

#: Arm orders per round; each depth runs first at least once over three rounds.
ARM_ORDERS = ((1, 2), (2, 1), (1, 2))

_InFlight = Tuple[Span, List[Future], float, int]


def read_pipelined(
    spans: Sequence[Span],
    arenas: Sequence[torch.Tensor],
    threads: int,
    pool: ThreadPoolExecutor,
    depth: int,
) -> Tuple[SpanLog, int]:
    """Keep up to ``depth`` spans queued on the same ``threads`` workers.

    Span ``i`` lands in ``arenas[i % depth]``, which span ``i + depth`` reuses only after span
    ``i`` has completed. Returns the per-span log and the index of the last span's arena.
    """
    log: SpanLog = []
    inflight: Deque[_InFlight] = collections.deque()

    def submit(index: int, span: Span) -> _InFlight:
        base = arenas[index % depth].data_ptr()
        started = time.perf_counter()
        futures = [
            pool.submit(_read_range, span.path, start, base + (start - span.offset), end - start)
            for start, end in span_ranges(span, threads)
        ]
        return span, futures, started, index % depth

    def complete() -> int:
        span, futures, started, slot = inflight.popleft()
        for future in futures:
            future.result()
        log.append((started, time.perf_counter(), span.nbytes))
        return slot

    last_slot = 0
    for index, span in enumerate(spans):
        inflight.append(submit(index, span))
        if len(inflight) == depth:
            last_slot = complete()
    while inflight:
        last_slot = complete()
    return log, last_slot


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--drive", nargs="+", required=True, help="directories on the drive")
    parser.add_argument("--threads", type=int, default=4, help="workers = ranges per span (K)")
    parser.add_argument("--spans-per-arm", type=int, default=24)
    parser.add_argument("--span-mib", type=int, default=400)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--label", default="")
    parser.add_argument("--out", required=True, help="JSON results, rewritten after each round")
    return parser.parse_args()


def main() -> int:
    if sys.platform != "win32":
        raise SystemExit("two_drive_depth_probe.py measures the Win32 #974 primitive; Windows only")
    args = parse()
    if not on_ac_power():
        raise SystemExit("on battery: every arm of this record runs on AC; refusing")
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

    arenas = [
        torch.empty(span_bytes, dtype=torch.uint8, device="cpu", pin_memory=True) for _ in range(2)
    ]
    for arena in arenas:
        if arena.data_ptr() % SECTOR:
            raise SystemExit(f"arena is not sector-aligned: {arena.data_ptr()}")
    letter = os.path.splitdrive(os.path.abspath(args.drive[0]))[0].upper()
    sampler = CounterSampler(
        {
            "read": rf"\LogicalDisk({letter})\Disk Read Bytes/sec",
            "write": rf"\LogicalDisk({letter})\Disk Write Bytes/sec",
            "cpu_pct": r"\Processor(_Total)\% Processor Time",
            "cpu_perf_pct": r"\Processor Information(_Total)\% Processor Performance",
        }
    )
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

    pool = ThreadPoolExecutor(max_workers=args.threads)
    sampler.start()
    try:
        read_pipelined(take(1), arenas, args.threads, pool, 1)  # untimed warm-up
        for round_index in range(args.rounds):
            entry: Dict[str, object] = {"round": round_index, "box": box_state(), "wall": {}}
            for depth in ARM_ORDERS[round_index % len(ARM_ORDERS)]:
                arm = f"d{depth}"
                chunk = take(args.spans_per_arm)
                wall_start = time.time()
                log, last_slot = read_pipelined(chunk, arenas, args.threads, pool, depth)
                entry["wall"][arm] = [wall_start, time.time()]
                entry[f"{arm}_gb_s"] = alone_rate(log)
                entry[f"{arm}_span_s"] = [round(end - start, 4) for start, end, _n in log]
                entry[f"check_{arm}"] = check_bytes(chunk[-1], arenas[last_slot])
                print(f"round {round_index}  depth {depth}  {entry[f'{arm}_gb_s']:6.2f} GB/s")
            entry["d2_over_d1"] = entry["d2_gb_s"] / entry["d1_gb_s"]
            print(f"round {round_index}  d2/d1 {entry['d2_over_d1']:.3f}")
            record["rounds"].append(entry)
            write()
    finally:
        pool.shutdown(wait=True)
        time.sleep(0.6)  # let the last arm's closing counter interval land
        sampler.stop()

    rounds = record["rounds"]
    for entry in rounds:
        entry["counters"] = {arm: sampler.mean_over(*wall) for arm, wall in entry["wall"].items()}
    record["median"] = {
        key: statistics.median(entry[key] for entry in rounds)
        for key in ("d1_gb_s", "d2_gb_s", "d2_over_d1")
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
