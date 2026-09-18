#!/usr/bin/env python3
"""#974 — how fast can the disk tier read a layer that is NOT in the page cache?

`bench_read_paths.py` measured the read primitives WARM (store in the page cache). The cold
70B step is read-bound (gate-971 §10: 84.7% of the step in load brackets at 1.46-2.32 GB/s
from an NVMe that reads 3.5+ GB/s), so the number that decides the fix is the COLD one:
what each primitive delivers when every byte comes off the drive.

Every (mode, threads) configuration reads its OWN set of --per-config layers of a store that
is larger than RAM (the 70B-shaped NF4 store, 34 GB against 31.7 GB), never reading a layer
twice in the process, so no configuration benefits from a previous one's pages. Run
`evict.py` first so the earliest layers are not left over from an earlier session either.

  readinto     one `handle.readinto(pinned view)` per tensor, tensors of a layer split across
               K threads (each thread its own handle) — the shipped AsyncDiskSource path at K=1
  readinto-1   the layer's contiguous data section as K byte ranges, one readinto each, into
               one pinned arena (the #974 fix's read plan)

usage: bench_cold_ranges.py --shard-dir <dir> [--per-config 6] [--threads 1,2,4,8]
                            [--modes readinto,readinto-1] [--start 0]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import torch

from soup_cli.utils.layer_shard import layer_shard_path
from soup_cli.utils.layer_stream_runtime import RamSource, _torch_dtype
from soup_cli.utils.safetensors_reader import read_header, read_into


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--per-config", type=int, default=6)
    parser.add_argument("--threads", default="1,2,4,8")
    parser.add_argument("--modes", default="readinto,readinto-1")
    parser.add_argument("--start", type=int, default=0, help="first layer index to use")
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
    slot: Dict[str, torch.Tensor] = {
        name: torch.empty(tuple(shape), dtype=_torch_dtype(dt), device="cpu", pin_memory=pin)
        for name, (shape, dt) in spec0.items()
    }
    layer_bytes = sum(t.numel() * t.element_size() for t in slot.values())
    threads_list = [int(t) for t in args.threads.split(",")]
    modes = [m.strip() for m in args.modes.split(",")]
    configs = [(mode, k) for mode in modes for k in threads_list]
    needed = len(configs) * args.per_config
    if args.start + needed > args.n_layers:
        raise SystemExit(
            f"need {needed} distinct layers from {args.start}, store has {args.n_layers}"
        )
    # Headers for every layer this run will touch (header reads are a few hundred KB each).
    layer_ids = list(range(args.start, args.start + needed))
    paths = {i: layer_shard_path(args.shard_dir, i) for i in layer_ids}
    headers = {i: read_header(paths[i]) for i in layer_ids}
    spans = {}
    for i in layer_ids:
        lo = min(headers[i][n].start for n in names)
        hi = max(headers[i][n].end for n in names)
        spans[i] = (lo, hi)
    span = max(hi - lo for lo, hi in spans.values())
    arena = torch.empty(span, dtype=torch.uint8, device="cpu", pin_memory=pin)
    print(
        f"layer bytes {layer_bytes / 1e6:.1f} MB  data span {span / 1e6:.1f} MB  "
        f"tensors {len(names)}  staging {'pinned' if pin else 'pageable'}  "
        f"{args.per_config} fresh layers per config"
    )

    def chunk(items: List[str], k: int) -> List[List[str]]:
        return [items[i::k] for i in range(k)]

    def run_readinto(idx: int, part: List[str]) -> None:
        with open(paths[idx], "rb") as fh:
            for name in part:
                read_into(fh, headers[idx][name], slot[name])

    def run_readinto_1(idx: int, part_index: int, parts: int) -> None:
        lo, hi = spans[idx]
        size = hi - lo
        a = lo + size * part_index // parts
        b = lo + size * (part_index + 1) // parts
        view = arena[a - lo : b - lo]
        with open(paths[idx], "rb") as fh:
            fh.seek(a)
            got = fh.readinto(memoryview(view.numpy()))
            assert got == b - a, (got, b - a)

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
                if mode == "readinto":
                    futs = [pool.submit(run_readinto, idx, part) for part in chunk(names, k)]
                elif mode == "readinto-1":
                    futs = [pool.submit(run_readinto_1, idx, i, k) for i in range(k)]
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
