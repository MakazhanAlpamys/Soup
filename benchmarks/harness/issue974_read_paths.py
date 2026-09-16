"""#974 — how fast can this box move a shard from the page cache into pinned host memory?

Compares the read primitives the disk tier could use, warm (store in the page cache):

  readinto      one `handle.readinto(pinned view)` per tensor (the shipped AsyncDiskSource path)
  mmap          `safe_open(...).get_tensor(name)` (an mmap view) then `pinned.copy_(view)`
  readinto-1    ONE readinto for the whole data section of the layer into an arena (tensors are
                contiguous byte ranges in a safetensors file), then per-tensor views

each split across K threads (tensors of one layer divided among the threads; every thread owns
its own file handle). Reports GB/s per (mode, threads) over --passes passes of --layers layers.

usage: bench_read_paths.py --shard-dir <dir> [--layers 16] [--threads 1,2,4,8] [--passes 2]
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import torch

from soup_cli.utils.layer_shard import layer_shard_path
from soup_cli.utils.layer_stream_runtime import RamSource, _torch_dtype
from soup_cli.utils.safetensors_reader import read_header, read_into


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--shard-dir", required=True)
    p.add_argument("--layers", type=int, default=16)
    p.add_argument("--threads", default="1,2,4,8")
    p.add_argument("--modes", default="readinto,mmap,readinto-1")
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--pin", action="store_true", default=True)
    p.add_argument("--pageable", action="store_true", help="pageable staging instead of pinned")
    return p.parse_args()


def main() -> int:
    args = parse()
    torch.cuda.init()
    torch.zeros(1, device="cuda")
    n_layers = args.layers
    specs = RamSource.layer_specs_from_shards(args.shard_dir, n_layers)
    paths = [layer_shard_path(args.shard_dir, i) for i in range(n_layers)]
    headers = [read_header(p) for p in paths]
    pin = not args.pageable
    # one staging slot (per-tensor pinned tensors), reused for every layer
    slot: Dict[str, torch.Tensor] = {
        name: torch.empty(tuple(shape), dtype=_torch_dtype(dt), device="cpu", pin_memory=pin)
        for name, (shape, dt) in specs[0].items()
    }
    names = list(specs[0])
    layer_bytes = sum(t.numel() * t.element_size() for t in slot.values())
    # the data section span for readinto-1
    starts = [min(h[n].start for n in names) for h in headers]
    ends = [max(h[n].start + h[n].nbytes for n in names) for h in headers]
    span = max(e - s for s, e in zip(starts, ends))
    arena = torch.empty(span, dtype=torch.uint8, device="cpu", pin_memory=pin)
    print(
        f"layers {n_layers}  tensors/layer {len(names)}  layer bytes {layer_bytes / 1e6:.1f} MB  "
        f"data span {span / 1e6:.1f} MB  staging {'pinned' if pin else 'pageable'}"
    )
    threads_list = [int(t) for t in args.threads.split(",")]
    modes = [m.strip() for m in args.modes.split(",")]

    def chunk(items: List[str], k: int) -> List[List[str]]:
        return [items[i::k] for i in range(k)]

    def run_readinto(idx: int, part: List[str]) -> None:
        with open(paths[idx], "rb") as fh:
            for name in part:
                read_into(fh, headers[idx][name], slot[name])

    def run_mmap(idx: int, part: List[str]) -> None:
        from safetensors import safe_open

        with safe_open(paths[idx], framework="pt") as fh:
            for name in part:
                slot[name].copy_(fh.get_tensor(name))

    def run_readinto_1(idx: int, part_index: int, parts: int) -> None:
        # split the contiguous data span into `parts` byte ranges
        lo, hi = starts[idx], ends[idx]
        size = hi - lo
        a = lo + size * part_index // parts
        b = lo + size * (part_index + 1) // parts
        view = arena[a - lo : b - lo]
        with open(paths[idx], "rb") as fh:
            fh.seek(a)
            got = fh.readinto(memoryview(view.numpy()))
            assert got == b - a, (got, b - a)

    results = []
    for mode in modes:
        for k in threads_list:
            per_pass = []
            for _ in range(args.passes):
                started = time.perf_counter()
                with ThreadPoolExecutor(max_workers=k) as pool:
                    for idx in range(n_layers):
                        if mode == "readinto":
                            futs = [
                                pool.submit(run_readinto, idx, part) for part in chunk(names, k)
                            ]
                        elif mode == "mmap":
                            futs = [pool.submit(run_mmap, idx, part) for part in chunk(names, k)]
                        elif mode == "readinto-1":
                            futs = [pool.submit(run_readinto_1, idx, i, k) for i in range(k)]
                        else:
                            raise SystemExit(f"unknown mode {mode}")
                        for f in futs:
                            f.result()
                elapsed = time.perf_counter() - started
                per_pass.append(n_layers * layer_bytes / elapsed / 1e9)
            best = max(per_pass)
            results.append((mode, k, per_pass))
            print(
                f"{mode:<12} threads {k:<2} "
                + "  ".join(f"{r:6.2f} GB/s" for r in per_pass)
                + f"   best {best:.2f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
