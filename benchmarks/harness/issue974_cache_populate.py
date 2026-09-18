#!/usr/bin/env python3
"""#974 — does the async source's buffered ``readinto`` populate the Windows page cache?

The #974 pilot (HANDOFF-stream.md §2) saw the async arm stay at 1.7 GB/s for a whole block
(3 warm-up + 8 timed steps, 86 GB read) when it ran first after a page-cache eviction, while
the mmap control in the same slot ran warm, and the async arm was fast again AFTER the control
had run. Hypothesis: the shipped read path (``open(path, "rb")`` + ``handle.readinto`` into a
pinned tensor, one call per tensor — the exact code in ``AsyncDiskSource._run``) reads through
the cache manager but leaves nothing resident, while the mmap path (``safe_open().get_tensor``)
does. This script isolates the file-API question from the streaming machinery: it runs the two
read primitives over the same 32 decoder shards in a chosen ORDER, with an eviction where the
sequence says, and reports GB/s per pass. Same destination for both primitives (a per-tensor
pinned slot, as the shipped source stages) so the copy side is identical.

Modes (each one pass over --layers shards of --shard-dir):
  readinto          open(path,"rb") + read_into per tensor  (the shipped AsyncDiskSource path)
  readinto_random   the same through os.open(..., O_RANDOM)     (FILE_FLAG_RANDOM_ACCESS)
  readinto_seq      the same through os.open(..., O_SEQUENTIAL) (FILE_FLAG_SEQUENTIAL_SCAN)
  readinto_raw      open(path,"rb",buffering=0) — no Python BufferedReader in between
  mmap              safe_open(path).get_tensor(name) then slot.copy_ (the shipped DiskSource path)
  evict             commit + touch --evict-gb of pageable RAM and free it (stream_probe's heuristic)

Reports the host baseline (available physical, commit charge) around every eviction, from
GlobalMemoryStatusEx, so the record can say what the box looked like.

usage: cache_populate.py --shard-dir <dir> [--layers 32]
                         [--seq readinto,evict,readinto,readinto,mmap,readinto] [--evict-gb 17]
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from typing import Dict, List

import torch

from soup_cli.utils.layer_shard import layer_shard_path
from soup_cli.utils.layer_stream_runtime import RamSource, _torch_dtype
from soup_cli.utils.safetensors_reader import read_header, read_into


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


def host_memory() -> Dict[str, float]:
    if not sys.platform.startswith("win"):
        return {}
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return {
        "avail_phys_gb": status.ullAvailPhys / 1e9,
        "total_phys_gb": status.ullTotalPhys / 1e9,
        "commit_gb": (status.ullTotalPageFile - status.ullAvailPageFile) / 1e9,
        "commit_limit_gb": status.ullTotalPageFile / 1e9,
    }


def fmt_mem(mem: Dict[str, float]) -> str:
    if not mem:
        return "n/a"
    return (
        f"avail phys {mem['avail_phys_gb']:.2f} GB, commit {mem['commit_gb']:.2f} of "
        f"{mem['commit_limit_gb']:.2f} GB"
    )


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--seq", default="readinto,evict,readinto,readinto,mmap,readinto")
    parser.add_argument("--evict-gb", type=float, default=17.0)
    parser.add_argument("--pageable", action="store_true", help="pageable staging, not pinned")
    parser.add_argument("--out", default="")
    return parser.parse_args()


def evict(gigabytes: float) -> Dict[str, float]:
    """stream_probe.evict_page_cache, with the box's available memory respected."""
    before = host_memory()
    if before:
        # Leave 2 GB for the other sessions on this box; the standby list is
        # repurposed first, which is the point.
        gigabytes = min(gigabytes, max(0.0, before["avail_phys_gb"] - 2.0))
    started = time.perf_counter()
    buffer = torch.empty(int(gigabytes * 1e9), dtype=torch.uint8, device="cpu")
    buffer.fill_(1)
    during = host_memory()
    del buffer
    elapsed = time.perf_counter() - started
    after = host_memory()
    print(
        f"evict        {gigabytes:.1f} GB touched in {elapsed:.1f} s | before: {fmt_mem(before)} | "
        f"during: {fmt_mem(during)} | after: {fmt_mem(after)}"
    )
    return {"gb": gigabytes, "seconds": elapsed, "before": before, "during": during, "after": after}


def main() -> int:
    args = parse()
    torch.cuda.init()
    torch.zeros(1, device="cuda")
    n_layers = args.layers
    specs = RamSource.layer_specs_from_shards(args.shard_dir, n_layers)
    paths = [layer_shard_path(args.shard_dir, i) for i in range(n_layers)]
    headers = [read_header(p) for p in paths]
    pin = not args.pageable
    slot: Dict[str, torch.Tensor] = {
        name: torch.empty(tuple(shape), dtype=_torch_dtype(dt), device="cpu", pin_memory=pin)
        for name, (shape, dt) in specs[0].items()
    }
    names = list(specs[0])
    layer_bytes = sum(t.numel() * t.element_size() for t in slot.values())
    total = n_layers * layer_bytes
    print(
        f"layers {n_layers}  tensors/layer {len(names)}  layer bytes {layer_bytes / 1e6:.1f} MB  "
        f"pass bytes {total / 1e9:.3f} GB  staging {'pinned' if pin else 'pageable'}  "
        f"host: {fmt_mem(host_memory())}"
    )

    def pass_readinto(opener) -> None:
        for idx in range(n_layers):
            with opener(paths[idx]) as fh:
                for name in names:
                    read_into(fh, headers[idx][name], slot[name])

    def open_plain(path: str):
        return open(path, "rb")

    def open_raw(path: str):
        return open(path, "rb", buffering=0)

    def open_flag(flag: int):
        def opener(path: str):
            fd = os.open(path, os.O_RDONLY | os.O_BINARY | flag)
            return os.fdopen(fd, "rb")

        return opener

    def pass_mmap() -> None:
        from safetensors import safe_open

        for idx in range(n_layers):
            with safe_open(paths[idx], framework="pt") as fh:
                for name in names:
                    slot[name].copy_(fh.get_tensor(name))

    modes = {
        "readinto": lambda: pass_readinto(open_plain),
        "readinto_raw": lambda: pass_readinto(open_raw),
        "readinto_random": lambda: pass_readinto(open_flag(os.O_RANDOM)),
        "readinto_seq": lambda: pass_readinto(open_flag(os.O_SEQUENTIAL)),
        "mmap": pass_mmap,
    }
    results: List[Dict[str, object]] = []
    for step, mode in enumerate([m.strip() for m in args.seq.split(",") if m.strip()]):
        if mode == "evict":
            results.append({"step": step, "mode": "evict", **evict(args.evict_gb)})
            continue
        if mode not in modes:
            raise SystemExit(f"unknown mode {mode!r}; known: evict, {', '.join(modes)}")
        started = time.perf_counter()
        modes[mode]()
        elapsed = time.perf_counter() - started
        rate = total / elapsed / 1e9
        print(
            f"{step:>2} {mode:<16} {rate:6.2f} GB/s  {elapsed:6.2f} s   "
            f"host: {fmt_mem(host_memory())}"
        )
        results.append(
            {
                "step": step,
                "mode": mode,
                "gb_per_s": rate,
                "seconds": elapsed,
                "host": host_memory(),
            }
        )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "shard_dir": args.shard_dir,
                    "layers": n_layers,
                    "pass_bytes": total,
                    "pinned": pin,
                    "seq": args.seq,
                    "evict_gb": args.evict_gb,
                    "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "results": results,
                },
                fh,
                indent=1,
            )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
