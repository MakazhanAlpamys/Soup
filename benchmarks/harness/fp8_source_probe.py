"""B3 probe: shard a REAL block-scaled FP8 checkpoint to NF4, time it, gate G1/G3.

Rule: benchmarks/gate-b3-fp8-source.md section 1, committed before the first run.

    python benchmarks/harness/fp8_source_probe.py --source DIR --out DIR \
        --double-quant on --json result.json

``DIR`` holds ``config.json`` (with the fp8 ``quantization_config``) and one or
more ``*.safetensors`` source files. The sharder is called directly, so the
arch check that ``soup train`` runs first is not involved (see the record's
B2 section).

The G1 reference decoder below is deliberately independent of
``soup_cli.utils.fp8_source``: it reads e4m3 from its bit fields through a
table and expands scales with ``repeat_interleave``, and never calls a float8
cast or any sharder code.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

NF4_CODEC_BLOCK = 64
G3_BOUND = 0.17
_REF_CHUNK_ROWS = 1024


def _peak_rss_bytes() -> int | None:
    """Peak RSS of this process in bytes, None where ``resource`` does not exist.

    ``resource`` is Unix-only, so it is imported here and not at module top:
    importing the probe must work on Windows. ``ru_maxrss`` is KiB on Linux and
    already bytes on macOS.
    """
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def _e4m3_table_f32():
    import torch

    values = []
    for byte in range(256):
        sign = -1.0 if byte & 0x80 else 1.0
        exponent = (byte >> 3) & 0xF
        mantissa = byte & 0x7
        if exponent == 0xF and mantissa == 0x7:
            values.append(float("nan"))
        elif exponent == 0:
            values.append(sign * (mantissa / 8.0) * 2.0**-6)
        else:
            values.append(sign * (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7))
    return torch.tensor(values, dtype=torch.float32)  # every e4m3 value is exact in fp32


def _independent_decode(q, scale, block, table):
    """Row-chunked so the int64 index tensor stays small on big weights."""
    import torch

    rows, cols = q.shape
    out = torch.empty((rows, cols), dtype=torch.bfloat16)
    as_bytes = q.view(torch.uint8)
    scale32 = scale.to(torch.float32)
    for start in range(0, rows, _REF_CHUNK_ROWS):
        stop = min(start + _REF_CHUNK_ROWS, rows)
        values = table[as_bytes[start:stop].long()]
        rows_idx = torch.arange(start, stop) // block[0]
        expanded = scale32[rows_idx].repeat_interleave(block[1], dim=1)[:, :cols]
        out[start:stop] = (values * expanded).to(torch.bfloat16)
    return out


def _stage_without(source: Path, staged: Path, exclude: set) -> dict:
    """Copy ``source`` to ``staged`` with the ``exclude`` keys left out.

    Tensor bytes are copied verbatim by offset (no decode, no re-encode); only
    the header and the file layout change. Needed because no prefix of the
    DeepSeek-V3 files is closed: each one leaves a weight whose scale is in the
    next file, which the sharder rightly refuses.
    """
    import shutil
    import struct

    staged.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "config.json", staged / "config.json")
    dropped = []
    for path in sorted(source.glob("*.safetensors")):
        with open(path, "rb") as src:
            header_len = struct.unpack("<Q", src.read(8))[0]
            header = json.loads(src.read(header_len))
            base = 8 + header_len
            metadata = header.pop("__metadata__", None)
            kept = [(k, v) for k, v in header.items() if k not in exclude]
            dropped += [k for k in header if k in exclude]
            new_header, offset = {}, 0
            for key, entry in kept:
                size = entry["data_offsets"][1] - entry["data_offsets"][0]
                new_header[key] = {**entry, "data_offsets": [offset, offset + size]}
                offset += size
            if metadata is not None:
                new_header["__metadata__"] = metadata
            blob = json.dumps(new_header, separators=(",", ":")).encode()
            blob += b" " * (-len(blob) % 8)
            with open(staged / path.name, "wb") as dst:
                dst.write(struct.pack("<Q", len(blob)))
                dst.write(blob)
                for _key, entry in kept:
                    start, stop = entry["data_offsets"]
                    src.seek(base + start)
                    remaining = stop - start
                    while remaining:
                        chunk = src.read(min(remaining, 64 * 2**20))
                        dst.write(chunk)
                        remaining -= len(chunk)
    return {"excluded": sorted(dropped), "staged_dir": str(staged)}


def _box(repo: Path, out_dir: Path) -> dict:
    import bitsandbytes
    import safetensors
    import torch

    cpu = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    mem_total = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                mem_total = int(line.split()[1]) * 1024
    except OSError:
        pass
    # A missing ``git`` raises FileNotFoundError even with ``check=False``.
    try:
        tree = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--", "src"],
            capture_output=True, text=True, check=False,
        ).stdout.strip())
    except OSError:
        tree, dirty = "unknown", None
    return {
        "when_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "mem_total_bytes": mem_total,
        # shutil.disk_usage, not os.statvfs: statvfs does not exist on Windows.
        "disk_free_bytes": shutil.disk_usage(out_dir.parent).free,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "safetensors": safetensors.__version__,
        "tree": tree,
        "src_dirty": dirty,
        "torch_threads": torch.get_num_threads(),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--double-quant",
        choices=("on", "off", "default"),
        default="on",
        help="default leaves it unset, the path `soup train` takes when the config does not set it",
    )
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument(
        "--exclude-key",
        action="append",
        default=[],
        help="stage the source without this tensor (bytes of every other tensor kept)",
    )
    args = parser.parse_args(argv)

    import torch
    from bitsandbytes.functional import dequantize_4bit
    from safetensors import safe_open
    from safetensors.torch import load_file

    from soup_cli.utils import layer_shard
    from soup_cli.utils.fp8_source import (
        dequantize_fp8_blockwise,
        is_float8_dtype,
        is_scale_key,
        load_fp8_source_config,
        scale_key_for,
    )
    from soup_cli.utils.layer_stream_runtime import rebuild_quant_state

    repo = Path(__file__).resolve().parents[2]
    result = {"rule": "benchmarks/gate-b3-fp8-source.md section 1", "label": "REAL"}
    result["box"] = _box(repo, args.out)
    if args.exclude_key:
        staged = args.out.parent / (args.out.name + "-staged-source")
        result["staging"] = _stage_without(args.source, staged, set(args.exclude_key))
        missing = set(args.exclude_key) - set(result["staging"]["excluded"])
        if missing:
            print(f"--exclude-key not found in source: {sorted(missing)}", file=sys.stderr)
            return 2
        args.source = staged
    config = load_fp8_source_config(str(args.source))
    if config is None:
        print("source has no fp8 quantization_config", file=sys.stderr)
        return 2
    block = config.block

    # ---- inventory -------------------------------------------------------
    files = sorted(args.source.glob("*.safetensors"))
    result["source_files"] = [{"name": f.name, "bytes": f.stat().st_size} for f in files]
    source_bytes = sum(f.stat().st_size for f in files)
    fp8_keys, layer_of = [], {}
    # Every tensor's own file: a weight's ``weight_scale_inv`` may sit in another
    # file (DeepSeek-V3 has 155 such pairs), so the G1/G3 loop reads each scale
    # from where it actually is, as the sharder does.
    location = {}
    for path in files:
        with safe_open(str(path), framework="pt") as handle:
            for key in handle.keys():
                location[key] = path
                if is_scale_key(key):
                    continue
                if is_float8_dtype(handle.get_slice(key).get_dtype()):
                    fp8_keys.append((path, key))
                    if key.startswith("model.layers."):
                        layer_of[key] = int(key.split(".")[2])
    suffixes = sorted({k.split(".", 3)[3] for _, k in fp8_keys if k in layer_of})
    per_layer = defaultdict(int)
    for key in layer_of:
        per_layer[layer_of[key]] += 1
    result["inventory"] = {
        "source_bytes": source_bytes,
        "fp8_weights": len(fp8_keys),
        "fp8_weights_per_layer": dict(sorted(per_layer.items())),
        "nf4_suffixes": len(suffixes),
    }

    # ---- G6 + G5: shard, timed and split ---------------------------------
    spent = defaultdict(float)

    def timed(name, fn):
        def wrapper(*a, **kw):
            t0 = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                spent[name] += time.perf_counter() - t0
        return wrapper

    layer_shard._read_fp8_tensor = timed("fp8_dequant_s", layer_shard._read_fp8_tensor)
    layer_shard._quantize_nf4 = timed("nf4_quantise_s", layer_shard._quantize_nf4)
    double_quant = {"on": True, "off": False, "default": None}[args.double_quant]
    t0 = time.perf_counter()
    index = layer_shard.shard_checkpoint(
        str(args.source), str(args.out), dtype="bfloat16", arch="deepseek_v3",
        quant=layer_shard.QUANT_NF4, quant_suffixes=suffixes,
        double_quant=double_quant, quant_device="cpu", force=True,
    )
    wall = time.perf_counter() - t0
    peak_rss = _peak_rss_bytes()
    out_bytes = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
    result["g6"] = {
        "double_quant_requested": args.double_quant,
        "double_quant": index.double_quant,
        "wall_s": round(wall, 2),
        "fp8_dequant_s": round(spent["fp8_dequant_s"], 2),
        "nf4_quantise_s": round(spent["nf4_quantise_s"], 2),
        "other_s": round(wall - spent["fp8_dequant_s"] - spent["nf4_quantise_s"], 2),
        "source_gb": round(source_bytes / 1e9, 3),
        "s_per_source_gb": round(wall / (source_bytes / 1e9), 2),
        "cache_bytes": out_bytes,
        "n_layers_in_index": index.n_layers,
    }
    result["g5"] = {"peak_rss_bytes_through_sharding": peak_rss}
    if peak_rss is None:
        result["g5"]["peak_rss_note"] = "resource module unavailable on this platform"
    json_tmp = args.json.with_suffix(".partial.json")
    json_tmp.write_text(json.dumps(result, indent=2))

    # ---- G1 + G3 ------------------------------------------------------------
    table = _e4m3_table_f32()
    codes = load_file(layer_shard.extras_shard_path(str(args.out)))
    rows = []
    by_layer = defaultdict(list)
    for path, key in fp8_keys:
        by_layer[layer_of.get(key, -1)].append((path, key))
    for layer in sorted(by_layer):
        blob = None
        if layer >= 0:
            blob = load_file(layer_shard.layer_shard_path(str(args.out), layer))
        for path, key in by_layer[layer]:
            scale_key = scale_key_for(key)
            with safe_open(str(location[scale_key]), framework="pt") as scale_handle:
                scale = scale_handle.get_tensor(scale_key)
            with safe_open(str(path), framework="pt") as handle:
                q = handle.get_tensor(key)
                reference = _independent_decode(q, scale, block, table)
                del q
                mine = dequantize_fp8_blockwise(
                    handle.get_slice(key), scale, block=block, dtype="bfloat16", key=key
                )
            row = {
                "key": key,
                "shape": list(reference.shape),
                "g1_equal": torch.equal(mine, reference),
            }
            del mine
            if blob is not None:
                short = key.split(".", 3)[3]
                state = rebuild_quant_state(short, blob, index.quant_specs[short], codes)
                back = dequantize_4bit(blob[short], state)
                got = back.float().reshape(-1, NF4_CODEC_BLOCK)
                want = reference.float().reshape(-1, NF4_CODEC_BLOCK)
                err = (got - want).abs().amax(dim=1)
                absmax = want.abs().amax(dim=1)
                zero = absmax == 0
                ratio = torch.where(zero, torch.zeros_like(err), err / absmax.clamp_min(1e-38))
                row.update({
                    "g3_worst_ratio": round(ratio.max().item(), 6),
                    "g3_pass": bool((err <= G3_BOUND * absmax).all())
                    and bool(torch.isfinite(back).all()),
                    "zero_blocks": int(zero.sum()),
                    "zero_blocks_exact": bool((err[zero] == 0).all()) if zero.any() else None,
                    "absmax_spread": [
                        float(f"{absmax[~zero].min().item():.3e}"),
                        float(f"{absmax.max().item():.3e}"),
                    ]
                    if (~zero).any()
                    else None,
                })
                del back, got, want, err, absmax
            del reference
            rows.append(row)
        del blob
    result["tensors"] = rows
    g3_rows = [r for r in rows if "g3_pass" in r]
    result["summary"] = {
        "g1_pass": all(r["g1_equal"] for r in rows),
        "g1_tensors": len(rows),
        "g3_pass": all(r["g3_pass"] for r in g3_rows),
        "g3_tensors": len(g3_rows),
        "g3_worst_ratio": max((r["g3_worst_ratio"] for r in g3_rows), default=None),
        "g3_failures": [r["key"] for r in g3_rows if not r["g3_pass"]],
        "zero_blocks_total": sum(r["zero_blocks"] for r in g3_rows),
    }
    args.json.write_text(json.dumps(result, indent=2))
    json_tmp.unlink(missing_ok=True)
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
