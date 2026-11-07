"""Read one layer file from each cache with Soup's direct-I/O opener, alternating A/B.

Same shape as the shipped reader: the layer's data span widened to 4 KiB sectors, split by
``plan_ranges`` into 4 ranges, each read by its own thread through its own ``open_direct``
handle into a sector-aligned buffer. Read-only. No torch import.

Section 6 of ``benchmarks/probe-rtx5070-moonlight-fp8-source.md`` (the T6 diagnosis, run after
the verdict), run from cwd ``D:/soup-bench`` with ``CUDA_VISIBLE_DEVICES=-1`` and ``PYTHONPATH``
at the tree under test: once as committed, then once more in a fresh process with
``ORDER = [("B", 10), ("A", 10)]``. The first read in each process pays the buffer's first
touch and is not a disk figure. Output: ``results/probe-rtx5070/fp8-loss-check/t6-readbench.txt``.
"""
import json
import mmap
import os
import struct
import sys
import threading
import time

import psutil

from soup_cli.utils.safetensors_reader import (
    SECTOR_BYTES,
    aligned_span,
    open_direct,
    plan_ranges,
)

A = "D:/soup-bench/soup-cache/D___soup-bench__models__Moonlight-16B-A3B"
B = "D:/soup-bench/fp8-loss-check/soup-cache/D___soup-bench__models__Moonlight-16B-A3B-fp8block"
RANGES = 4
ORDER = [("A", 10), ("B", 10), ("A", 10), ("B", 10), ("A", 10), ("B", 10), ("B", 20), ("A", 20)]


def span_of(path):
    with open(path, "rb") as handle:
        (n,) = struct.unpack("<Q", handle.read(8))
        meta = json.loads(handle.read(n))
    meta.pop("__metadata__", None)
    lo = 8 + n + min(v["data_offsets"][0] for v in meta.values())
    hi = 8 + n + max(v["data_offsets"][1] for v in meta.values())
    return aligned_span(lo, hi), os.path.getsize(path)


def read_once(path, buffer):
    (start, end), size = span_of(path)
    ranges = plan_ranges(start, end, RANGES)
    view = memoryview(buffer)
    errors = []

    def work(lo, hi):
        try:
            want = min(hi, size) - lo
            with open_direct(path) as handle:
                handle.seek(lo)
                done = 0
                target = view[lo - start: hi - start]
                while done < want:
                    got = handle.readinto(target[done:])
                    if not got:
                        raise OSError(f"short read {done}/{want}")
                    done += got
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work, args=r) for r in ranges]
    began = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    seconds = time.perf_counter() - began
    if errors:
        raise errors[0]
    nbytes = min(end, size) - start
    return nbytes, seconds


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        print("refusing: CUDA_VISIBLE_DEVICES must be -1")
        return 2
    free = psutil.virtual_memory().available / 1e9
    if free < 10:
        print(f"refusing: {free:.2f} GB available < 10")
        return 2
    spans = [
        span_of(os.path.join(root, f"layer_{i:03d}.safetensors"))[0]
        for root in (A, B) for i in (10, 20)
    ]
    biggest = max(end - start for start, end in spans)
    buffer = mmap.mmap(-1, biggest + SECTOR_BYTES)  # page-aligned anonymous memory
    results = []
    total = 0
    for arm, layer in ORDER:
        path = os.path.join(A if arm == "A" else B, f"layer_{layer:03d}.safetensors")
        nbytes, seconds = read_once(path, buffer)
        total += nbytes
        row = {"arm": arm, "layer": layer, "bytes": nbytes, "seconds": round(seconds, 3),
               "GBps": round(nbytes / seconds / 1e9, 2),
               "free_ram_gb": round(psutil.virtual_memory().available / 1e9, 2)}
        results.append(row)
        print(json.dumps(row), flush=True)
        if total > 14e9:
            break
    buffer.close()
    print("total GB read:", round(total / 1e9, 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
