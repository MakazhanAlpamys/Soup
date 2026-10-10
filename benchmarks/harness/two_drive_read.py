#!/usr/bin/env python3
"""R1 of the disk-tier read-rate track: does the second NVMe add read bandwidth on this box?

The cold 70B-shaped step is read-bound (probe-rtx5070 §21: 70.38 GB per step at 4.1 GB/s)
and the dev box carries TWO NVMe drives, while the disk tier reads from one. Before any code
stripes a store across both, this measures whether the drives deliver more together than
apart, with the exact read primitive the #974 reader is built on — imported from
``issue974_unbuffered.py``, not re-typed: FILE_FLAG_NO_BUFFERING, one handle per range, one
ReadFile per range, into a pinned arena. Unbuffered reads never touch the page cache, so
nothing has to be evicted: every read here is a cold read.

Each drive reads fixed-size, sector-aligned spans (one span ~ one 70B NF4 layer) one at a
time, each span as K parallel ranges — the reader's shape. No span is read twice in a run.
Per round, in an order that rotates so every arm runs first once: drive A alone, drive B
alone, both at once.

Derived per round, from the both-at-once arm. Each drive's rate there is taken over the
window in which BOTH drives were reading (a span straddling the window's end is pro-rated),
so the drive that finishes first does not flatter the other's tail:

    r_sum = (rate_A + rate_B) / max(alone_A, alone_B)       # weighted striping's ceiling
    r_alt = 2 * min(rate_A, rate_B) / max(alone_A, alone_B)  # plain alternate-layer striping

The decision rule lives in ``benchmarks/probe-rtx5070-two-drive-read.md`` §2 and was
committed before this ran. Windows only (it is the #974 primitive, which is Win32).

usage: two_drive_read.py --drive-a DIR [DIR ...] --drive-b DIR [DIR ...] --out JSON
                         [--threads 4] [--spans-per-arm 24] [--span-mib 400] [--rounds 3]
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from issue974_unbuffered import SECTOR, kernel32, open_unbuffered, read_unbuffered

#: Arm orders per round, rotated so each arm runs first exactly once over three rounds.
ARM_ORDERS = (("a", "b", "ab"), ("b", "ab", "a"), ("ab", "a", "b"))

#: Bytes compared against a buffered read after every arm, at each end of the last span.
CHECK_BYTES = 1 << 20

SpanLog = List[Tuple[float, float, int]]


@dataclass(frozen=True)
class Span:
    """A sector-aligned byte range that lies entirely inside one file."""

    path: str
    offset: int
    nbytes: int


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


def _run_quiet(cmd: Sequence[str]) -> Optional[str]:
    # Console tools print in the OEM code page (cp866 on this box, whose NBSP is 0xFF in
    # tasklist's memory column), so decoding them as UTF-8 raises on the reader thread.
    try:
        done = subprocess.run(
            cmd, capture_output=True, encoding="oem", errors="replace", timeout=20, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


class _PdhValue(ctypes.Structure):
    _fields_ = [("CStatus", wt.DWORD), ("doubleValue", ctypes.c_double)]


_PDH_FMT_DOUBLE = 0x00000200
_PDH_FMT_NOCAP100 = 0x00008000


class CounterSampler:
    """Windows performance counters sampled on a background thread (PDH, English names).

    The harness only READS, so any write a drive's counter shows during an arm is someone
    else's I/O — the pagefile, a peer's test suite — and that is the contamination the
    validity row in the record exists to catch. Rate counters report the interval between
    two collections; a sample is attributed to the arm its interval ENDS in.
    """

    def __init__(self, paths: Dict[str, str], interval: float = 0.25) -> None:
        self.samples: List[Tuple[float, Dict[str, float]]] = []
        self.errors: List[str] = []
        self._interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._counters: Dict[str, ctypes.c_void_p] = {}
        self._query = ctypes.c_void_p()
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
        for key, path in paths.items():
            handle = ctypes.c_void_p()
            status = pdh.PdhAddEnglishCounterW(self._query, path, 0, ctypes.byref(handle))
            if status != 0:
                self.errors.append(f"{path}: PDH status {status & 0xFFFFFFFF:#010x}")
                continue
            self._counters[key] = handle

    def start(self) -> None:
        if not self._counters:
            return
        self._pdh.PdhCollectQueryData(self._query)  # a rate counter needs a first sample
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            if self._pdh.PdhCollectQueryData(self._query) != 0:
                continue
            now = time.time()
            row: Dict[str, float] = {}
            for key, handle in self._counters.items():
                value = _PdhValue()
                status = self._pdh.PdhGetFormattedCounterValue(
                    handle, _PDH_FMT_DOUBLE | _PDH_FMT_NOCAP100, None, ctypes.byref(value)
                )
                if status == 0:
                    row[key] = value.doubleValue
            self.samples.append((now, row))

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self._query:
            self._pdh.PdhCloseQuery(self._query)

    def mean_over(self, start: float, end: float) -> Dict[str, float]:
        rows = [row for stamp, row in self.samples if start < stamp <= end]
        out: Dict[str, float] = {"samples": float(len(rows))}
        for key in sorted({k for row in rows for k in row}):
            out[key] = statistics.fmean(row[key] for row in rows if key in row)
        return out


def box_state() -> Dict[str, object]:
    """What else the box was doing: memory, commit, Python processes, the GPU's occupants."""
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    tasks = _run_quiet(["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV", "/NH"])
    gpu = _run_quiet(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "avail_phys_gb": status.ullAvailPhys / 1e9,
        "commit_used_gb": (status.ullTotalPageFile - status.ullAvailPageFile) / 1e9,
        "commit_limit_gb": status.ullTotalPageFile / 1e9,
        "python_processes": None
        if tasks is None
        else sum(1 for line in tasks.splitlines() if line.startswith('"python')),
        "gpu_mem_used_mib_util_pct_temp_c": None if gpu is None else gpu.strip(),
    }


def collect_spans(dirs: Sequence[str], span_bytes: int) -> List[Span]:
    """Every whole span that fits inside a file under ``dirs``, in sorted file order."""
    spans: List[Span] = []
    for root in dirs:
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if not os.path.isfile(path):
                continue
            size = os.path.getsize(path)
            for offset in range(0, size - span_bytes + 1, span_bytes):
                spans.append(Span(path, offset, span_bytes))
    return spans


def span_ranges(span: Span, parts: int) -> List[Tuple[int, int]]:
    """``parts`` sector-aligned [start, end) file ranges covering the span exactly."""
    cuts = [span.offset + ((span.nbytes * i // parts) & ~(SECTOR - 1)) for i in range(parts)]
    cuts.append(span.offset + span.nbytes)
    return [(cuts[i], cuts[i + 1]) for i in range(parts)]


def _read_range(path: str, start: int, ptr: int, length: int) -> None:
    handle = open_unbuffered(path)
    try:
        read_unbuffered(handle, start, ptr, length, length, 0)
    finally:
        kernel32.CloseHandle(handle)


def read_spans(
    spans: Sequence[Span],
    arena_ptr: int,
    threads: int,
    pool: ThreadPoolExecutor,
    barrier: Optional[threading.Barrier] = None,
) -> SpanLog:
    """Read each span as ``threads`` parallel ranges; one span at a time, like the reader."""
    log: SpanLog = []
    if barrier is not None:
        barrier.wait()
    for span in spans:
        started = time.perf_counter()
        futures = [
            pool.submit(
                _read_range, span.path, start, arena_ptr + (start - span.offset), end - start
            )
            for start, end in span_ranges(span, threads)
        ]
        for future in futures:
            future.result()
        log.append((started, time.perf_counter(), span.nbytes))
    return log


def alone_rate(log: SpanLog) -> float:
    return sum(n for _s, _e, n in log) / (log[-1][1] - log[0][0]) / 1e9


def windowed_rates(log_a: SpanLog, log_b: SpanLog) -> Tuple[float, float, float]:
    """Each drive's GB/s over the window in which both were reading; spans pro-rated."""
    start = max(log_a[0][0], log_b[0][0])
    end = min(log_a[-1][1], log_b[-1][1])
    width = end - start

    def bytes_in(log: SpanLog) -> float:
        total = 0.0
        for span_start, span_end, nbytes in log:
            overlap = min(span_end, end) - max(span_start, start)
            if overlap > 0:
                total += nbytes * overlap / (span_end - span_start)
        return total

    return bytes_in(log_a) / width / 1e9, bytes_in(log_b) / width / 1e9, width


def check_bytes(span: Span, arena: torch.Tensor) -> bool:
    """The arena's two ends must equal the file's bytes at the span's two ends."""
    with open(span.path, "rb") as handle:
        handle.seek(span.offset)
        head = handle.read(CHECK_BYTES)
        handle.seek(span.offset + span.nbytes - CHECK_BYTES)
        tail = handle.read(CHECK_BYTES)
    got_head = bytes(arena[:CHECK_BYTES].numpy())
    got_tail = bytes(arena[span.nbytes - CHECK_BYTES : span.nbytes].numpy())
    return got_head == head and got_tail == tail


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--drive-a", nargs="+", required=True, help="directories on drive A")
    parser.add_argument("--drive-b", nargs="+", required=True, help="directories on drive B")
    parser.add_argument("--threads", type=int, default=4, help="parallel ranges per span (K)")
    parser.add_argument("--spans-per-arm", type=int, default=24)
    parser.add_argument("--span-mib", type=int, default=400)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--pageable", action="store_true", help="pageable arenas (control)")
    parser.add_argument("--label", default="")
    parser.add_argument("--out", required=True, help="JSON results, rewritten after each round")
    return parser.parse_args()


def main() -> int:
    if sys.platform != "win32":
        raise SystemExit("two_drive_read.py measures the Win32 #974 primitive; Windows only")
    args = parse()
    span_bytes = args.span_mib * 1024 * 1024
    if span_bytes % SECTOR or span_bytes < 2 * CHECK_BYTES:
        raise SystemExit(f"--span-mib must give a sector multiple >= {2 * CHECK_BYTES} bytes")
    pin = not args.pageable
    if pin:
        torch.cuda.init()
        torch.zeros(1, device="cuda")

    pools = {
        "a": collect_spans(args.drive_a, span_bytes),
        "b": collect_spans(args.drive_b, span_bytes),
    }
    needed = 1 + args.rounds * 2 * args.spans_per_arm  # warm-up span + two arms per round
    for drive, spans in pools.items():
        if len(spans) < needed:
            raise SystemExit(
                f"drive {drive}: {len(spans)} spans of {args.span_mib} MiB available, "
                f"{needed} needed — no span may be read twice in a run"
            )
    cursors = {"a": 0, "b": 0}

    def take(drive: str, count: int) -> List[Span]:
        start = cursors[drive]
        cursors[drive] = start + count
        return pools[drive][start : start + count]

    arenas = {
        drive: torch.empty(span_bytes, dtype=torch.uint8, device="cpu", pin_memory=pin)
        for drive in ("a", "b")
    }
    for drive, arena in arenas.items():
        if arena.data_ptr() % SECTOR:
            raise SystemExit(f"arena {drive} is not sector-aligned: {arena.data_ptr()}")
    executors = {drive: ThreadPoolExecutor(max_workers=args.threads) for drive in ("a", "b")}

    letters = {
        "a": os.path.splitdrive(os.path.abspath(args.drive_a[0]))[0].upper(),
        "b": os.path.splitdrive(os.path.abspath(args.drive_b[0]))[0].upper(),
    }
    if letters["a"] == letters["b"]:
        raise SystemExit(f"both drives resolve to volume {letters['a']}; nothing to compare")
    counter_paths = {
        f"{kind}_{drive}": rf"\LogicalDisk({letter})\Disk {label} Bytes/sec"
        for drive, letter in letters.items()
        for kind, label in (("read", "Read"), ("write", "Write"))
    }
    counter_paths.update(
        {
            "pages_s": r"\Memory\Pages/sec",
            "committed_bytes": r"\Memory\Committed Bytes",
            "cpu_pct": r"\Processor(_Total)\% Processor Time",
            "cpu_perf_pct": r"\Processor Information(_Total)\% Processor Performance",
        }
    )
    sampler = CounterSampler(counter_paths)
    sampler.start()

    try:
        import soup_cli

        soup_cli_file = soup_cli.__file__
    except ImportError:
        soup_cli_file = None
    record: Dict[str, object] = {
        "label": args.label,
        "args": vars(args),
        "span_bytes": span_bytes,
        "spans_available": {drive: len(spans) for drive, spans in pools.items()},
        "pinned": pin,
        "python": sys.version,
        "torch": torch.__version__,
        "soup_cli_file": soup_cli_file,
        "volumes": letters,
        "counter_paths": counter_paths,
        "counter_errors": sampler.errors,
        "box_start": box_state(),
        "rounds": [],
    }

    def write() -> None:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=1)
            handle.write("\n")

    # Warm-up: one span per drive, untimed, so no arm pays for spinning up its pool.
    for drive in ("a", "b"):
        read_spans(take(drive, 1), arenas[drive].data_ptr(), args.threads, executors[drive])

    print(
        f"span {args.span_mib} MiB, K={args.threads}, {args.spans_per_arm} spans per drive per "
        f"arm, {args.rounds} rounds, arenas {'pinned' if pin else 'pageable'}"
    )
    try:
        for round_index in range(args.rounds):
            order = ARM_ORDERS[round_index % len(ARM_ORDERS)]
            entry: Dict[str, object] = {"round": round_index, "order": order, "box": box_state()}
            walls: Dict[str, List[float]] = {}
            entry["wall"] = walls
            for arm in order:
                if arm in ("a", "b"):
                    spans = take(arm, args.spans_per_arm)
                    wall_start = time.time()
                    log = read_spans(spans, arenas[arm].data_ptr(), args.threads, executors[arm])
                    walls[arm] = [wall_start, time.time()]
                    entry[f"alone_{arm}_gb_s"] = alone_rate(log)
                    entry[f"alone_{arm}_span_s"] = [round(e - s, 4) for s, e, _n in log]
                    entry[f"check_alone_{arm}"] = check_bytes(spans[-1], arenas[arm])
                    print(
                        f"round {round_index}  {arm} alone   {entry[f'alone_{arm}_gb_s']:6.2f} GB/s"
                    )
                    continue
                spans_ab = {drive: take(drive, args.spans_per_arm) for drive in ("a", "b")}
                barrier = threading.Barrier(2)
                logs: Dict[str, SpanLog] = {}
                errors: List[BaseException] = []

                def run(drive: str) -> None:
                    try:
                        logs[drive] = read_spans(
                            spans_ab[drive],
                            arenas[drive].data_ptr(),
                            args.threads,
                            executors[drive],
                            barrier,
                        )
                    except BaseException as exc:  # re-raised on the main thread below
                        errors.append(exc)
                        barrier.abort()

                workers = [threading.Thread(target=run, args=(d,)) for d in ("a", "b")]
                wall_start = time.time()
                for worker in workers:
                    worker.start()
                for worker in workers:
                    worker.join()
                walls["ab"] = [wall_start, time.time()]
                if errors:
                    raise errors[0]
                rate_a, rate_b, window = windowed_rates(logs["a"], logs["b"])
                first = min(logs["a"][0][0], logs["b"][0][0])
                last = max(logs["a"][-1][1], logs["b"][-1][1])
                total = sum(n for log in logs.values() for _s, _e, n in log)
                entry.update(
                    {
                        "both_a_gb_s": rate_a,
                        "both_b_gb_s": rate_b,
                        "both_window_s": window,
                        "both_aggregate_gb_s": total / (last - first) / 1e9,
                        "both_a_alone_rate_gb_s": alone_rate(logs["a"]),
                        "both_b_alone_rate_gb_s": alone_rate(logs["b"]),
                        "both_a_span_s": [round(e - s, 4) for s, e, _n in logs["a"]],
                        "both_b_span_s": [round(e - s, 4) for s, e, _n in logs["b"]],
                        "check_both_a": check_bytes(spans_ab["a"][-1], arenas["a"]),
                        "check_both_b": check_bytes(spans_ab["b"][-1], arenas["b"]),
                    }
                )
                print(
                    f"round {round_index}  both      a {rate_a:6.2f} + b {rate_b:6.2f} GB/s over "
                    f"{window:.2f} s  (aggregate {entry['both_aggregate_gb_s']:.2f})"
                )
            best_alone = max(entry["alone_a_gb_s"], entry["alone_b_gb_s"])
            entry["r_sum"] = (entry["both_a_gb_s"] + entry["both_b_gb_s"]) / best_alone
            entry["r_alt"] = 2 * min(entry["both_a_gb_s"], entry["both_b_gb_s"]) / best_alone
            print(f"round {round_index}  r_sum {entry['r_sum']:.3f}  r_alt {entry['r_alt']:.3f}")
            record["rounds"].append(entry)
            write()
    finally:
        for executor in executors.values():
            executor.shutdown(wait=True)
        time.sleep(0.6)  # let the last arm's closing counter interval land
        sampler.stop()

    rounds = record["rounds"]
    for entry in rounds:
        entry["counters"] = {arm: sampler.mean_over(*wall) for arm, wall in entry["wall"].items()}
        for arm, means in entry["counters"].items():
            print(
                f"round {entry['round']}  {arm:<5} counters: "
                + "  ".join(
                    f"{key} {means[key] / 1e6:.1f} MB/s"
                    for key in sorted(means)
                    if key.startswith(("read_", "write_"))
                )
                + f"  pages/s {means.get('pages_s', float('nan')):.0f}"
                + f"  cpu {means.get('cpu_pct', float('nan')):.0f}%"
                + f"  perf {means.get('cpu_perf_pct', float('nan')):.0f}%"
                + f"  ({means['samples']:.0f} samples)"
            )
    record["counter_samples"] = sampler.samples
    record["median"] = {
        key: statistics.median(entry[key] for entry in rounds)
        for key in ("alone_a_gb_s", "alone_b_gb_s", "both_a_gb_s", "both_b_gb_s", "r_sum", "r_alt")
    }
    checks = [value for entry in rounds for key, value in entry.items() if key.startswith("check_")]
    record["all_byte_checks_ok"] = all(checks)
    record["box_end"] = box_state()
    write()
    print("median " + "  ".join(f"{k} {v:.3f}" for k, v in record["median"].items()))
    print(f"byte checks: {'all OK' if record['all_byte_checks_ok'] else 'MISMATCH'}")
    print(f"wrote {args.out}")
    return 0 if record["all_byte_checks_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
