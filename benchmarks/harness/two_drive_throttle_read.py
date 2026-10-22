#!/usr/bin/env python3
"""One arm of the per-drive throttle diagnostic: read C:, D: or both, flat out, for a fixed time.

The record is ``benchmarks/probe-rtx5070-drive-throttle.md``. This is the R1 read harness
(``two_drive_read.py``: unbuffered ``ReadFile`` through the #974 primitive, one handle per
range, K = 4 parallel ranges per span, a pinned arena per drive, one span in flight per drive)
turned from "three short arms" into "one long arm". It differs from R1 in exactly two ways:

- it reads for a fixed number of seconds, wrapping round the drive's span list, so the same
  files are read again and again exactly as a training run re-reads them every step (R1 read no
  span twice; an unbuffered read never touches the page cache, so a re-read is as cold as the
  first);
- it keeps every span's start and duration, not a rate, so the driver can bin the arm at any
  width afterwards.

With ``--arm both`` each drive has its own thread and its own arena and they share only a start
barrier: unlike the striped training reader they are NOT locked to one layer at a time, so a
slow drive slows only itself and the per-drive rates stay separable. Windows only, CUDA for the
pinned arenas; no GPU compute. The torch import is inside ``main`` so ``read_cyclic`` can be
tested without it.

usage: two_drive_throttle_read.py --arm c|d|both --seconds 300 --c-dir DIR --d-dir DIR --out JSON
                                  [--threads 4] [--span-mib 400] [--label TEXT]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

#: (path, byte offset, byte count) of one sector-aligned span lying inside one file.
SpanT = Tuple[str, int, int]
ARM_DRIVES = {"c": ("c",), "d": ("d",), "both": ("c", "d")}


def read_cyclic(
    spans: Sequence[SpanT],
    read_span: Callable[[SpanT], None],
    seconds: float,
    barrier: Optional[threading.Barrier] = None,
    clock: Callable[[], float] = time.perf_counter,
) -> Tuple[float, List[Tuple[float, float, int]]]:
    """Read ``spans`` in order, wrapping, one at a time, until ``seconds`` have passed.

    Returns ``(begin, log)``: the clock value when reading began (after the barrier) and one
    ``(start, end, nbytes)`` per span. The span in flight at the deadline is finished and kept.
    """
    if not spans:
        raise ValueError("no spans to read")
    if barrier is not None:
        barrier.wait()
    begin = clock()
    deadline = begin + seconds
    log: List[Tuple[float, float, int]] = []
    index = 0
    while True:
        started = clock()
        if started >= deadline:
            return begin, log
        span = spans[index % len(spans)]
        read_span(span)
        log.append((started, clock(), span[2]))
        index += 1


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--arm", choices=sorted(ARM_DRIVES), required=True)
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--c-dir", nargs="+", required=True, help="directories read on C:")
    parser.add_argument("--d-dir", nargs="+", required=True, help="directories read on D:")
    parser.add_argument("--threads", type=int, default=4, help="parallel ranges per span (K)")
    parser.add_argument("--span-mib", type=int, default=400)
    parser.add_argument("--pageable", action="store_true", help="pageable arenas (control only)")
    parser.add_argument("--label", default="")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> int:
    if sys.platform != "win32":
        raise SystemExit("two_drive_throttle_read.py reads with Win32 unbuffered I/O; Windows only")
    args = parse()
    perf0, unix0 = time.perf_counter(), time.time()

    import torch
    from two_drive_read import Span, _read_range, box_state, check_bytes, collect_spans, span_ranges

    span_bytes = args.span_mib * 1024 * 1024
    pin = not args.pageable
    if pin:
        torch.cuda.init()
        torch.zeros(1, device="cuda")
    dirs = {"c": args.c_dir, "d": args.d_dir}
    letters = {
        d: os.path.splitdrive(os.path.abspath(dirs[d][0]))[0].upper() for d in ARM_DRIVES[args.arm]
    }
    if len(set(letters.values())) != len(letters):
        raise SystemExit(f"the drives resolve to one volume {letters}; nothing to separate")
    pools = {d: collect_spans(dirs[d], span_bytes) for d in ARM_DRIVES[args.arm]}
    for drive, pool in pools.items():
        if not pool:
            raise SystemExit(
                f"drive {drive}: no file under {dirs[drive]} holds a {args.span_mib} MiB span"
            )
    arenas = {
        d: torch.empty(span_bytes, dtype=torch.uint8, device="cpu", pin_memory=pin) for d in pools
    }
    executors = {d: ThreadPoolExecutor(max_workers=args.threads) for d in pools}

    def reader(drive: str) -> Callable[[SpanT], None]:
        base = arenas[drive].data_ptr()

        def read_span(span: SpanT) -> None:
            path, offset, nbytes = span
            futures = [
                executors[drive].submit(
                    _read_range, path, start, base + (start - offset), end - start
                )
                for start, end in span_ranges(Span(path, offset, nbytes), args.threads)
            ]
            for future in futures:
                future.result()

        return read_span

    tuples = {d: [(s.path, s.offset, s.nbytes) for s in pool] for d, pool in pools.items()}
    for drive in pools:  # one untimed span per drive, so no arm pays for the pool's start-up
        reader(drive)(tuples[drive][0])

    barrier = threading.Barrier(len(pools))
    results: Dict[str, Tuple[float, List[Tuple[float, float, int]]]] = {}
    errors: List[BaseException] = []

    def run(drive: str) -> None:
        try:
            results[drive] = read_cyclic(
                tuples[drive], reader(drive), args.seconds, barrier, time.perf_counter
            )
        except BaseException as exc:  # re-raised on the main thread below
            errors.append(exc)
            barrier.abort()

    record: Dict[str, Any] = {
        "kind": "throttle_arm",
        "label": args.label,
        "arm": args.arm,
        "args": vars(args),
        "span_bytes": span_bytes,
        "pinned": pin,
        "python": sys.version,
        "torch": torch.__version__,
        "box_start": box_state(),
    }
    try:
        import soup_cli

        record["soup_cli_file"] = soup_cli.__file__
    except ImportError:
        record["soup_cli_file"] = None
    workers = [threading.Thread(target=run, args=(d,)) for d in pools]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    for executor in executors.values():
        executor.shutdown(wait=True)
    if errors:
        raise errors[0]

    begin = min(b for b, _log in results.values())
    t0_unix = unix0 + (begin - perf0)
    record["t0_unix"] = t0_unix
    record["read_s_planned"] = args.seconds
    record["ended_unix"] = time.time()
    ok = True
    record["drives"] = {}
    for drive, (_b, log) in results.items():
        last = tuples[drive][(len(log) - 1) % len(tuples[drive])]
        check_ok = check_bytes(Span(*last), arenas[drive])
        ok = ok and check_ok
        total = sum(n for _s, _e, n in log)
        record["drives"][drive] = {
            "dir": dirs[drive],
            "volume": letters[drive],
            "pool_spans": len(tuples[drive]),
            "pool_bytes": len(tuples[drive]) * span_bytes,
            "passes": len(log) / len(tuples[drive]),
            "bytes": total,
            "n_spans": len(log),
            "check_ok": check_ok,
            "spans": {
                "start_s": [round(s - begin, 4) for s, _e, _n in log],
                "dur_s": [round(e - s, 4) for s, e, _n in log],
                "nbytes": span_bytes,
            },
        }
        print(
            f"{drive.upper()}: {len(log)} spans, {total / 1e9:.1f} GB, "
            f"{total / (log[-1][1] - log[0][0]) / 1e9:.3f} GB/s, "
            f"{len(log) / len(tuples[drive]):.1f} passes, byte check "
            f"{'OK' if check_ok else 'MISMATCH'}",
            flush=True,
        )
    record["box_end"] = box_state()
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(record, handle, separators=(",", ":"))
        handle.write("\n")
    print(f"wrote {args.out}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
