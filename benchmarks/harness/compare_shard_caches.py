#!/usr/bin/env python3
"""Compare a striped layer-stream cache with a single-root one, tensor by tensor.

The two-drive striping gate (``benchmarks/gate-two-drive-striping.md`` §2) says the two
arms read "the same bytes" and checks it directly, after the six timed arms and never
between them: every safetensors file of the single-root cache is matched with its
counterpart in the striped cache (decoder layer ``i`` at the root the striped
``index.json`` names in ``layer_roots``), and every tensor is compared by name, dtype,
shape and a SHA-256 of its data bytes.

The safetensors header is parsed here directly (8-byte little-endian length, then JSON),
and data is read with plain buffered reads in 64 MiB chunks, so the check needs no torch
and maps no file (a mapped file charges commit for its whole size on Windows). It reads
both caches in full (~34 GB each), so it populates the page cache; that is why it runs
only after the timed arms.

usage: compare_shard_caches.py --single DIR --striped DIR --out JSON
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
from typing import Any, Dict, List, Tuple

CHUNK = 64 * 2**20


def read_header(path: str) -> Tuple[int, Dict[str, Any]]:
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length).decode("utf-8"))
    return 8 + length, header


def tensor_digests(path: str) -> Dict[str, Dict[str, Any]]:
    base, header = read_header(path)
    header.pop("__metadata__", None)
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, "rb") as handle:
        for name, spec in sorted(header.items(), key=lambda item: item[1]["data_offsets"][0]):
            start, end = spec["data_offsets"]
            handle.seek(base + start)
            digest = hashlib.sha256()
            remaining = end - start
            while remaining:
                chunk = handle.read(min(CHUNK, remaining))
                if not chunk:
                    raise OSError(f"{path}: short read in {name}")
                digest.update(chunk)
                remaining -= len(chunk)
            out[name] = {
                "dtype": spec["dtype"],
                "shape": spec["shape"],
                "nbytes": end - start,
                "sha256": digest.hexdigest(),
            }
    return out


def file_pairs(single: str, striped: str) -> List[Tuple[str, str, str]]:
    with open(os.path.join(striped, "index.json"), encoding="utf-8") as handle:
        index = json.load(handle)
    # Root 0 is the --shards folder; stripe root k holds this cache's own folder, named by
    # soup_cli.utils.stripe_roots.stripe_folder_name (slug + hash of the primary cache's
    # realpath, R4 fix wave). A cache sharded before that change used the bare slug — the
    # gate's caches did — so that spelling is the fallback when the new one is absent.
    from soup_cli.utils.stripe_roots import stripe_folder_name

    slug = os.path.basename(os.path.normpath(striped))
    roots = [striped]
    for root in index.get("stripe_roots") or []:
        folder = os.path.join(root, stripe_folder_name(striped))
        roots.append(folder if os.path.isdir(folder) else os.path.join(root, slug))
    layer_roots = list(index.get("layer_roots") or [])
    pairs: List[Tuple[str, str, str]] = []
    for name in sorted(os.listdir(single)):
        if not name.endswith(".safetensors"):
            continue
        if name.startswith("layer_"):
            layer = int(name[len("layer_") : -len(".safetensors")])
            root_index = layer_roots[layer] if layer_roots else 0
            root = roots[root_index]
        else:
            root = striped
        pairs.append((name, os.path.join(single, name), os.path.join(root, name)))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--single", required=True)
    parser.add_argument("--striped", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    started = time.time()
    pairs = file_pairs(args.single, args.striped)
    files: List[Dict[str, Any]] = []
    mismatches: List[str] = []
    tensors = 0
    nbytes = 0
    for name, single_path, striped_path in pairs:
        entry: Dict[str, Any] = {"file": name, "single": single_path, "striped": striped_path}
        if not os.path.exists(striped_path):
            entry["result"] = "missing in striped cache"
            mismatches.append(name)
            files.append(entry)
            continue
        left = tensor_digests(single_path)
        right = tensor_digests(striped_path)
        differing = sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))
        entry["tensors"] = len(left)
        entry["bytes"] = sum(spec["nbytes"] for spec in left.values())
        entry["result"] = "equal" if not differing else "differ"
        if differing:
            entry["differing_tensors"] = differing[:50]
            mismatches.append(name)
        tensors += len(left)
        nbytes += entry["bytes"]
        files.append(entry)
        print(f"{name}: {entry['result']} ({len(left)} tensors)", flush=True)
    extra_striped = sorted(
        name
        for root in [args.striped]
        for name in os.listdir(root)
        if name.endswith(".safetensors") and name not in {pair[0] for pair in pairs}
    )
    payload = {
        "single": args.single,
        "striped": args.striped,
        "files_compared": len(pairs),
        "tensors_compared": tensors,
        "bytes_compared": nbytes,
        "mismatched_files": mismatches,
        "safetensors_only_in_striped_root0": extra_striped,
        "all_equal": not mismatches and not extra_striped,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
        "seconds": time.time() - started,
        "files": files,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
    print(
        f"ALL EQUAL: {payload['all_equal']}; files {len(pairs)}, tensors {tensors}, "
        f"{nbytes / 1e9:.3f} GB, {payload['seconds']:.0f} s",
        flush=True,
    )
    return 0 if payload["all_equal"] else 1


if __name__ == "__main__":
    sys.exit(main())
