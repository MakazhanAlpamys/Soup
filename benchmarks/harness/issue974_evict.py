#!/usr/bin/env python3
"""Evict a shard store from the Windows page cache without admin rights.

Two stages, because `cache_populate.py` showed that pressure alone leaves ~2.5 GB
of the standby list — the most recently used pages, i.e. exactly the store under
test — resident: (1) commit and touch --pressure-gb of pageable RAM (stream_probe's
heuristic), (2) read --displace-gb of OTHER files through buffered readinto so the
survivors are displaced by unrelated content (readinto populates the cache; measured
2026-09-15, cache_populate_run1.json).

usage: evict.py [--pressure-gb 17] [--displace-dir <dir of big files>] [--displace-gb 8]
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time


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


def host_memory() -> str:
    if not sys.platform.startswith("win"):
        return "n/a"
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    avail = status.ullAvailPhys / 1e9
    commit = (status.ullTotalPageFile - status.ullAvailPageFile) / 1e9
    return (
        f"avail phys {avail:.2f} GB, commit {commit:.2f} of {status.ullTotalPageFile / 1e9:.2f} GB"
    )


def avail_phys_gb() -> float:
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return status.ullAvailPhys / 1e9


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pressure-gb", type=float, default=17.0)
    parser.add_argument(
        "--displace-dir", default=r"C:\Users\user\.soup\layer-stream\D___synth__qwen2.5-14b-shape"
    )
    parser.add_argument("--displace-gb", type=float, default=8.0)
    parser.add_argument("--keep-free-gb", type=float, default=1.5)
    args = parser.parse_args()
    import torch

    print(f"before   {host_memory()}")
    gigabytes = min(args.pressure_gb, max(0.0, avail_phys_gb() - args.keep_free_gb))
    started = time.perf_counter()
    buffer = torch.empty(int(gigabytes * 1e9), dtype=torch.uint8, device="cpu")
    buffer.fill_(1)
    print(
        f"pressure {gigabytes:.1f} GB touched in {time.perf_counter() - started:.1f} s; "
        f"during: {host_memory()}"
    )
    del buffer
    # Stage 2: displace the survivors with other content.
    chunk = bytearray(64 * 1024 * 1024)
    view = memoryview(chunk)
    read = 0
    target = int(args.displace_gb * 1e9)
    started = time.perf_counter()
    files = sorted(
        os.path.join(args.displace_dir, name)
        for name in os.listdir(args.displace_dir)
        if name.endswith(".safetensors")
    )
    for path in files:
        if read >= target:
            break
        with open(path, "rb") as fh:
            while read < target:
                got = fh.readinto(view)
                if not got:
                    break
                read += got
    print(
        f"displace {read / 1e9:.2f} GB read in {time.perf_counter() - started:.1f} s "
        f"from {args.displace_dir}"
    )
    print(f"after    {host_memory()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
