#!/usr/bin/env python3
"""#974 — does UNBUFFERED I/O (FILE_FLAG_NO_BUFFERING) reach the drive's rated 3.5 GB/s cold?

`bench_cold_ranges.py` measured every buffered primitive at 1.25-2.4 GB/s cold on the C:
NVMe, whichever way the bytes were asked for — so the cache manager, not the request
pattern, looks like the ceiling. This reads the same fresh layers of the 70B-shaped NF4
store straight from the drive into the pinned arena, bypassing the page cache, at K
parallel ranges (one handle per thread, synchronous ReadFile each), and optionally in
fixed-size chunks per thread.

Alignment rules for NO_BUFFERING: file offset, byte count and buffer address all multiples
of the sector size (4096 here). The layer's data section starts at 8 + header length, so
the read covers the 4 KiB-aligned superset [lo & ~4095, roundup(hi)] and the tensors are
views at (start - aligned_lo) — which is exactly how a reader built on this would place them.

  unbuf        K aligned ranges, one ReadFile each
  unbuf-chunk  K aligned ranges, each read in --chunk-mib pieces (queue of shorter requests)
  buffered     K ranges through open(path,"rb").readinto, same layers policy — the control

usage: bench_unbuffered.py --shard-dir <dir> [--per-config 6] [--threads 1,2,4,8]
                           [--modes unbuf,buffered] [--chunk-mib 4] [--start 0]
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

import torch

from soup_cli.utils.layer_shard import layer_shard_path
from soup_cli.utils.layer_stream_runtime import RamSource, _torch_dtype
from soup_cli.utils.safetensors_reader import read_header

SECTOR = 4096
GENERIC_READ = 0x80000000
FILE_SHARE_READ = 0x00000001
OPEN_EXISTING = 3
FILE_FLAG_NO_BUFFERING = 0x20000000
FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

kernel32 = ctypes.windll.kernel32
kernel32.CreateFileW.restype = wt.HANDLE
kernel32.CreateFileW.argtypes = [
    wt.LPCWSTR,
    wt.DWORD,
    wt.DWORD,
    ctypes.c_void_p,
    wt.DWORD,
    wt.DWORD,
    wt.HANDLE,
]
kernel32.ReadFile.restype = wt.BOOL
kernel32.ReadFile.argtypes = [
    wt.HANDLE,
    ctypes.c_void_p,
    wt.DWORD,
    ctypes.POINTER(wt.DWORD),
    ctypes.c_void_p,
]
kernel32.SetFilePointerEx.restype = wt.BOOL
kernel32.SetFilePointerEx.argtypes = [
    wt.HANDLE,
    ctypes.c_longlong,
    ctypes.POINTER(ctypes.c_longlong),
    wt.DWORD,
]
kernel32.CloseHandle.argtypes = [wt.HANDLE]


def open_unbuffered(path: str) -> int:
    handle = kernel32.CreateFileW(
        path,
        GENERIC_READ,
        FILE_SHARE_READ,
        None,
        OPEN_EXISTING,
        FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN,
        None,
    )
    if handle == INVALID_HANDLE_VALUE or handle is None:
        raise OSError(f"CreateFileW failed for {path}: {ctypes.get_last_error()}")
    return handle


def read_unbuffered(
    handle: int, offset: int, ptr: int, length: int, expected: int, chunk: int
) -> None:
    """ReadFile at `offset` into `ptr` until `expected` bytes arrived, `chunk` bytes a request.

    `length` is the sector-aligned request span (it may run past EOF, where the last request
    returns short); every request length is therefore a sector multiple, which NO_BUFFERING
    demands, and the loop stops on `expected`, the bytes the file actually holds in the span.
    """
    if not kernel32.SetFilePointerEx(handle, offset, None, 0):
        raise OSError(f"SetFilePointerEx failed: {ctypes.GetLastError()}")
    done = 0
    got = wt.DWORD(0)
    while done < expected:
        want = length - done if chunk <= 0 else min(chunk, length - done)
        if not kernel32.ReadFile(handle, ptr + done, want, ctypes.byref(got), None):
            raise OSError(f"ReadFile failed at {offset + done}: {ctypes.GetLastError()}")
        if got.value == 0:
            raise OSError(f"ReadFile returned 0 bytes at {offset + done}")
        done += got.value
    if done != expected:
        raise OSError(f"read {done} bytes at {offset}, expected {expected}")


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--per-config", type=int, default=6)
    parser.add_argument("--threads", default="1,2,4,8")
    parser.add_argument("--modes", default="unbuf,buffered")
    parser.add_argument("--chunk-mib", type=int, default=4)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n-layers", type=int, default=80)
    parser.add_argument("--pageable", action="store_true")
    parser.add_argument("--out", default="")
    return parser.parse_args()


def main() -> int:
    args = parse()
    torch.cuda.init()
    torch.zeros(1, device="cuda")
    pin = not args.pageable
    spec0 = RamSource.spec_from_shard(args.shard_dir, args.start)
    names = list(spec0)
    layer_bytes = sum(
        int(torch.empty(0, dtype=_torch_dtype(dt)).element_size()) * int(torch.Size(shape).numel())
        for _n, (shape, dt) in spec0.items()
    )
    threads_list = [int(t) for t in args.threads.split(",")]
    modes = [m.strip() for m in args.modes.split(",")]
    configs = [(mode, k) for mode in modes for k in threads_list]
    needed = len(configs) * args.per_config
    if args.start + needed > args.n_layers:
        raise SystemExit(
            f"need {needed} distinct layers from {args.start}, store has {args.n_layers}"
        )
    layer_ids = list(range(args.start, args.start + needed))
    paths = {i: layer_shard_path(args.shard_dir, i) for i in layer_ids}
    headers = {i: read_header(paths[i]) for i in layer_ids}
    spans = {}
    for i in layer_ids:
        lo = min(headers[i][n].start for n in names)
        hi = max(headers[i][n].end for n in names)
        spans[i] = (lo & ~(SECTOR - 1), (hi + SECTOR - 1) & ~(SECTOR - 1), lo, hi)
    span = max(hi_a - lo_a for lo_a, hi_a, _lo, _hi in spans.values())
    arena = torch.empty(span, dtype=torch.uint8, device="cpu", pin_memory=pin)
    base_ptr = arena.data_ptr()
    if base_ptr % SECTOR:
        raise SystemExit(f"arena is not sector-aligned: {base_ptr}")
    print(
        f"layer bytes {layer_bytes / 1e6:.1f} MB  aligned span {span / 1e6:.1f} MB  "
        f"tensors {len(names)}  staging {'pinned' if pin else 'pageable'}  "
        f"{args.per_config} fresh layers per config  chunk {args.chunk_mib} MiB"
    )
    chunk_bytes = args.chunk_mib * 1024 * 1024

    def ranges(idx: int, parts: int) -> List[tuple]:
        lo_a, hi_a, _lo, _hi = spans[idx]
        size = hi_a - lo_a
        out = []
        for part in range(parts):
            a = lo_a + ((size * part // parts) & ~(SECTOR - 1))
            b = hi_a if part == parts - 1 else lo_a + ((size * (part + 1) // parts) & ~(SECTOR - 1))
            out.append((a, b, a - lo_a))
        return out

    sizes = {i: os.path.getsize(paths[i]) for i in layer_ids}

    def run_unbuf(idx: int, rng: tuple, chunk: int) -> None:
        a, b, off = rng
        handle = open_unbuffered(paths[idx])
        try:
            read_unbuffered(handle, a, base_ptr + off, b - a, min(b, sizes[idx]) - a, chunk)
        finally:
            kernel32.CloseHandle(handle)

    def run_buffered(idx: int, rng: tuple) -> None:
        a, b, off = rng
        expected = min(b, sizes[idx]) - a
        view = arena[off : off + expected]
        with open(paths[idx], "rb") as fh:
            fh.seek(a)
            got = fh.readinto(memoryview(view.numpy()))
            assert got == expected, (got, expected)

    results = []
    cursor = args.start
    for mode, k in configs:
        layers = list(range(cursor, cursor + args.per_config))
        cursor += args.per_config
        per_layer = []
        started_all = time.perf_counter()
        with ThreadPoolExecutor(max_workers=k) as pool:
            for idx in layers:
                started = time.perf_counter()
                if mode == "unbuf":
                    futs = [pool.submit(run_unbuf, idx, rng, 0) for rng in ranges(idx, k)]
                elif mode == "unbuf-chunk":
                    futs = [pool.submit(run_unbuf, idx, rng, chunk_bytes) for rng in ranges(idx, k)]
                elif mode == "buffered":
                    futs = [pool.submit(run_buffered, idx, rng) for rng in ranges(idx, k)]
                else:
                    raise SystemExit(f"unknown mode {mode}")
                for f in futs:
                    f.result()
                per_layer.append(time.perf_counter() - started)
        elapsed = time.perf_counter() - started_all
        rate = len(layers) * layer_bytes / elapsed / 1e9
        results.append(
            {
                "mode": mode,
                "threads": k,
                "layers": layers,
                "gb_per_s": rate,
                "seconds": elapsed,
                "per_layer_s": per_layer,
            }
        )
        print(
            f"{mode:<12} threads {k:<2} layers {layers[0]:>2}-{layers[-1]:>2}  {rate:6.2f} GB/s  "
            f"{elapsed:6.2f} s  per layer " + " ".join(f"{s:.2f}" for s in per_layer)
        )
    # Byte check on the LAST layer read: the arena must hold the file's bytes at the aligned offset.
    last = layers[-1]
    lo_a, _hi_a, lo, hi = spans[last]
    with open(paths[last], "rb") as fh:
        fh.seek(lo)
        expect = fh.read(min(1 << 20, hi - lo))
    got = bytes(arena[lo - lo_a : lo - lo_a + len(expect)].numpy())
    print(
        f"byte check on layer {last}: {'OK' if got == expect else 'MISMATCH'} ({len(expect)} bytes)"
    )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "shard_dir": args.shard_dir,
                    "layer_bytes": layer_bytes,
                    "span": span,
                    "pinned": pin,
                    "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "results": results,
                    "args": vars(args),
                },
                fh,
                indent=1,
            )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
