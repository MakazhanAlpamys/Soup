#!/usr/bin/env python3
"""Reproduce the Windows safetensors mapping limit without Soup, CUDA or bitsandbytes.

``shard_checkpoint`` enters every source shard into one ``ExitStack`` and keeps
all the memory maps alive for the whole of pass 2. On the 2026-09-12 dev box
(Windows 11, 31.7 GB RAM, commit limit 36.7 GB) that pattern dies with
``Windows fatal exception: access violation`` once roughly 96-106 GB of
safetensors mappings are live in one process, whatever the file count
(62 x 1.71 GB and 14 x 6.85 GB both crash; per-file open/read/close passes).
See ``benchmarks/probe-rtx5070-what-bounds-streaming.md`` §12-§14.

Two modes over the same directory of ``*.safetensors`` files::

    python benchmarks/harness/mmap_limit_repro.py --dir D:/synth/llama-70b-shape-v32k all
    python benchmarks/harness/mmap_limit_repro.py --dir D:/synth/llama-70b-shape-v32k perfile

``all`` holds every handle open (the shipped sharder's pattern) and is expected
to crash on an affected box; ``perfile`` opens, reads and closes each file in
turn and is expected to pass. ``get_tensor`` is a zero-copy view on this stack,
so neither mode reads the bytes unless ``--touch`` is given.
"""

from __future__ import annotations

import argparse
import contextlib
import faulthandler
import glob
import os
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mode", choices=("all", "perfile"))
    parser.add_argument("--dir", required=True, help="directory holding *.safetensors")
    parser.add_argument(
        "--touch",
        action="store_true",
        help="also sum every tensor so the pages are actually read, not only mapped",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    faulthandler.enable(all_threads=True)
    from safetensors import safe_open

    paths = sorted(glob.glob(os.path.join(args.dir, "*.safetensors")))
    if not paths:
        print(f"ERROR: no *.safetensors under {args.dir}")
        return 2
    started = time.perf_counter()
    total = 0

    def visit(handle: object, label: str) -> None:
        nonlocal total
        for key in handle.keys():  # type: ignore[attr-defined]
            tensor = handle.get_tensor(key)  # type: ignore[attr-defined]
            if args.touch:
                tensor.float().sum().item()
            total += tensor.numel() * tensor.element_size()
            del tensor
        elapsed = time.perf_counter() - started
        print(f"{label}  {total / 1e9:7.1f} GB mapped  {elapsed:6.0f} s", flush=True)

    if args.mode == "all":
        with contextlib.ExitStack() as stack:
            handles = [stack.enter_context(safe_open(path, framework="pt")) for path in paths]
            print(f"opened {len(handles)} handles", flush=True)
            for index, handle in enumerate(handles):
                visit(handle, f"file {index:3d} done")
    else:
        for index, path in enumerate(paths):
            with safe_open(path, framework="pt") as handle:
                visit(handle, f"file {index:3d} done")
    print(f"OK {args.mode}: {total / 1e9:.1f} GB in {time.perf_counter() - started:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
