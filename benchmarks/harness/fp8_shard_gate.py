"""G3 of ``benchmarks/gate-fp8-shard.md``: an FP8 block copy of a real checkpoint, sharded.

CPU only (``CUDA_VISIBLE_DEVICES=-1``): header and tensor reads, no GPU.

    # 1. the FP8 block copy (``fp8_block_quantize.py``), under the RAM watchdog
    python benchmarks/harness/fp8_shard_gate.py convert --src <bf16 dir> --dst <new dir> \
        --report <convert.json>
    # 2. shard it with Soup's sharder into a NEW cache folder, and check the cache
    python benchmarks/harness/fp8_shard_gate.py shard --weights <fp8 dir> --out <new dir> \
        --quant nf4 --layers 27 --check-layers 1 13 26 --report <shard.json>

``shard`` sizes the cache with ``estimate_shard_cache_bytes`` first and refuses when the
cache volume would keep less than ``--keep-free-gb`` free; it then holds the estimate
against the files written (payload exact per decoder layer and outside the layers, total
within [1.00, 1.05] of the bytes on disk) and checks the real data: every stored tensor of
``--check-layers`` that came from an FP8 weight equals the harness's own per-block reference
dequant of that weight, cast to bf16, and every NF4 tensor there equals bitsandbytes' CPU
quantisation of that reference (packed bytes and every sidecar).

Both commands refuse to start below ``--min-free-ram-gb`` of available RAM and stop the
process (exit 3, a partial report written) if it falls below ``--abort-ram-gb``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import struct
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

GB = 1e9
GIB = 2**30


class Watchdog:
    """Samples RSS and available RAM every ``period`` s; kills the process below a floor."""

    def __init__(self, *, abort_gib: float, report_path: str, period: float = 2.0):
        import psutil

        self._psutil = psutil
        self._process = psutil.Process()
        self.abort_bytes = abort_gib * GIB
        self.report_path = report_path
        self.period = period
        self.peak_rss = 0
        self.min_available = None
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.partial: dict = {}

    def _run(self) -> None:
        while not self._stop.is_set():
            rss = self._process.memory_info().rss
            available = self._psutil.virtual_memory().available
            self.peak_rss = max(self.peak_rss, rss)
            if self.min_available is None or available < self.min_available:
                self.min_available = available
            self.samples += 1
            if available < self.abort_bytes:
                self.partial.update(
                    aborted=f"available RAM {available / GIB:.2f} GiB fell below "
                    f"{self.abort_bytes / GIB:.2f} GiB",
                    peak_rss_bytes=self.peak_rss,
                )
                with open(self.report_path, "w", encoding="utf-8") as handle:
                    json.dump(self.partial, handle, indent=2)
                print(self.partial["aborted"], flush=True)
                os._exit(3)
            self._stop.wait(self.period)

    def __enter__(self) -> "Watchdog":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        self._thread.join()

    def summary(self) -> dict:
        return {
            "peak_rss_bytes": self.peak_rss,
            "min_available_ram_bytes": self.min_available,
            "ram_samples": self.samples,
            "sample_period_s": self.period,
        }


def _box(paths) -> dict:
    import psutil
    import torch

    return {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": os.cpu_count(),
        "available_ram_bytes": psutil.virtual_memory().available,
        "total_ram_bytes": psutil.virtual_memory().total,
        "disk_free_bytes": {path: shutil.disk_usage(path).free for path in paths},
    }


def _require_ram(min_gib: float) -> None:
    import psutil

    available = psutil.virtual_memory().available
    if available < min_gib * GIB:
        raise SystemExit(
            f"refusing to start: {available / GIB:.2f} GiB of RAM available, "
            f"{min_gib:.1f} GiB required"
        )


def cmd_convert(args) -> int:
    import fp8_block_quantize

    _require_ram(args.min_free_ram_gb)
    if os.path.exists(args.dst):
        print(f"refusing: {args.dst} already exists")
        return 2
    disk = os.path.dirname(os.path.abspath(args.dst))
    box = _box([disk])
    with Watchdog(abort_gib=args.abort_ram_gb, report_path=args.report) as watch:
        watch.partial.update(box=box, step="convert")
        report = fp8_block_quantize.convert_checkpoint(
            args.src, args.dst, block=tuple(args.block)
        )
    report["box"] = box
    report["memory"] = watch.summary()
    report["disk_free_after_bytes"] = shutil.disk_usage(disk).free
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    printable = {key: report[key] for key in report if key != "per_tensor"}
    print(json.dumps(printable, indent=2))
    return 0


def _payload(path: str) -> int:
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
    return os.path.getsize(path) - 8 - length


def _weight_locations(weights_dir: str) -> dict:
    """key -> file name, from the index or, for a one-file checkpoint, that file's keys."""
    index = os.path.join(weights_dir, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index, encoding="utf-8") as handle:
            return json.load(handle)["weight_map"]
    from safetensors import safe_open

    with safe_open(os.path.join(weights_dir, "model.safetensors"), framework="pt") as handle:
        return {key: "model.safetensors" for key in handle.keys()}


class _Source:
    """Reads source tensors by key, one safe_open handle at a time."""

    def __init__(self, weights_dir: str):
        self.dir = weights_dir
        self.where = _weight_locations(weights_dir)
        self._name = None
        self._handle = None

    def get(self, key: str):
        from safetensors import safe_open

        name = self.where[key]
        if name != self._name:
            self.close()
            self._handle = safe_open(os.path.join(self.dir, name), framework="pt").__enter__()
            self._name = name
        return self._handle.get_tensor(key).clone()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.__exit__(None, None, None)
        self._handle, self._name = None, None


def _reference_bf16(source: _Source, key: str, block):
    """The source tensor as the cache should hold it: FP8 ones reference-dequantised."""
    import torch
    from fp8_block_quantize import reference_dequantize

    tensor = source.get(key)
    if tensor.dtype != torch.float8_e4m3fn:
        return tensor.to(torch.bfloat16)
    return reference_dequantize(tensor, source.get(key + "_scale_inv"), block).to(torch.bfloat16)


def _check_layer(source: _Source, layer_file: str, layer: int, block, *, nf4_keys) -> dict:
    """Every FP8-derived tensor of one cached layer against the reference."""
    import torch
    from bitsandbytes.functional import quantize_4bit
    from safetensors.torch import load_file

    stored = load_file(layer_file)
    prefix = f"model.layers.{layer}."
    checked, mismatches, nf4_checked = 0, [], 0
    for short in sorted(stored):
        if "::" in short or short.startswith("mlp.experts."):
            continue
        key = prefix + short
        if key + "_scale_inv" not in source.where:
            continue  # not an FP8 weight
        reference = _reference_bf16(source, key, block)
        if short in nf4_keys:
            packed, state = quantize_4bit(
                reference, blocksize=64, compress_statistics=True, quant_type="nf4"
            )
            want = {
                "": packed,
                "::absmax": state.absmax,
                "::nested_absmax": state.state2.absmax,
                "::nested_offset": state.offset,
            }
            for suffix, value in want.items():
                got = stored[short + suffix]
                if not torch.equal(got, value.reshape(got.shape)):
                    mismatches.append(short + suffix)
            nf4_checked += 1
        elif not torch.equal(stored[short], reference):
            mismatches.append(short)
        checked += 1
    if "mlp.experts.gate_up_proj" in stored:
        gate_up, down = stored["mlp.experts.gate_up_proj"], stored["mlp.experts.down_proj"]
        for expert in range(gate_up.shape[0]):
            base = f"{prefix}mlp.experts.{expert}."
            want_gate_up = torch.cat(
                (
                    _reference_bf16(source, base + "gate_proj.weight", block),
                    _reference_bf16(source, base + "up_proj.weight", block),
                )
            )
            if not torch.equal(gate_up[expert], want_gate_up):
                mismatches.append(f"mlp.experts.gate_up_proj[{expert}]")
            want_down = _reference_bf16(source, base + "down_proj.weight", block)
            if not torch.equal(down[expert], want_down):
                mismatches.append(f"mlp.experts.down_proj[{expert}]")
            checked += 2
    return {
        "layer": layer,
        "fp8_derived_tensors_checked": checked,
        "nf4_tensors_checked": nf4_checked,
        "mismatches": mismatches,
    }


def cmd_shard(args) -> int:

    from soup_cli.utils.fp8_block import read_checkpoint_quantization_config
    from soup_cli.utils.layer_shard import (
        _discover_safetensors,
        estimate_shard_cache_bytes,
        layer_paths,
        shard_cache_disk_bytes,
        shard_checkpoint,
    )

    _require_ram(args.min_free_ram_gb)
    if os.path.exists(args.out):
        print(f"refusing: {args.out} already exists")
        return 2
    disk = os.path.dirname(os.path.abspath(args.out))
    box = _box([disk])
    quant_config = read_checkpoint_quantization_config(args.weights)
    block = tuple(quant_config["weight_block_size"])
    suffixes = ()
    if args.quant == "nf4":
        from soup_cli.utils.layer_stream_runtime import (
            build_meta_skeleton,
            quantised_layer_suffixes,
        )

        probe = build_meta_skeleton(args.weights, dtype=args.dtype, quant="nf4")
        suffixes = tuple(sorted(quantised_layer_suffixes(probe)))
        del probe
    common = dict(
        dtype=args.dtype, arch="deepseek_v3", quant=args.quant, quant_suffixes=suffixes,
        double_quant=True, expert_layout="", num_hidden_layers=args.layers,
    )
    estimate = estimate_shard_cache_bytes(
        _discover_safetensors(args.weights), quantization_config=quant_config, **common
    )
    free = shutil.disk_usage(disk).free
    if free - estimate.total_bytes < args.keep_free_gb * GB:
        print(
            f"refusing: the cache needs {estimate.total_bytes / GB:.2f} GB and {disk} has "
            f"{free / GB:.2f} GB free; {args.keep_free_gb} GB must stay free"
        )
        return 2
    notes = []
    with Watchdog(abort_gib=args.abort_ram_gb, report_path=args.report) as watch:
        watch.partial.update(box=box, step="shard", estimate_total_bytes=estimate.total_bytes)
        started = time.perf_counter()
        index = shard_checkpoint(
            args.weights, args.out, quant_device="cpu", notify=notes.append, **common
        )
        shard_seconds = time.perf_counter() - started
    memory = watch.summary()
    layer_files = layer_paths(args.out, index)
    payload = [_payload(path) for path in layer_files]
    other = sum(
        _payload(os.path.join(args.out, name)) for name in os.listdir(args.out)
        if name == "extras.safetensors" or name.startswith("large_")
    )
    on_disk = shard_cache_disk_bytes(args.out, index)
    size = {
        "estimate_total_bytes": estimate.total_bytes,
        "cache_bytes_on_disk": on_disk,
        "ratio_estimate_to_disk": estimate.total_bytes / on_disk,
        "layer_payload_exact": list(estimate.layer_bytes) == payload,
        "layer_payload_mismatches": [
            idx for idx, (want, got) in enumerate(zip(estimate.layer_bytes, payload))
            if want != got
        ],
        "other_payload_exact": estimate.other_bytes == other,
        "estimate_other_bytes": estimate.other_bytes,
        "cache_other_payload_bytes": other,
    }
    size["pass"] = (
        size["layer_payload_exact"]
        and size["other_payload_exact"]
        and on_disk <= estimate.total_bytes <= 1.05 * on_disk
    )
    source = _Source(args.weights)
    checks = []
    started = time.perf_counter()
    try:
        for layer in args.check_layers:
            checks.append(
                _check_layer(source, layer_files[layer], layer, block, nf4_keys=set(suffixes))
            )
    finally:
        source.close()
    check_seconds = time.perf_counter() - started
    exact = all(not check["mismatches"] for check in checks) and all(
        check["fp8_derived_tensors_checked"] for check in checks
    )
    report = {
        "weights": args.weights,
        "out": args.out,
        "arguments": {**common, "quant_suffixes": list(suffixes), "quant_device": "cpu"},
        "box": box,
        "shard_wall_seconds": shard_seconds,
        "memory": memory,
        "n_layers": index.n_layers,
        "total_params": index.total_params,
        "notes": notes,
        "size": size,
        "real_data_checks": checks,
        "real_data_check_seconds": check_seconds,
        "real_data_exact": exact,
        "disk_free_after_bytes": shutil.disk_usage(disk).free,
        "verdict": "PASS" if size["pass"] and exact else "FAIL",
    }
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({key: report[key] for key in report if key != "notes"}, indent=2))
    return 0 if report["verdict"] == "PASS" else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("convert", "shard"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--report", required=True)
        cmd.add_argument("--min-free-ram-gb", type=float, default=10.0)
        cmd.add_argument("--abort-ram-gb", type=float, default=3.0)
    convert = sub.choices["convert"]
    convert.add_argument("--src", required=True)
    convert.add_argument("--dst", required=True)
    convert.add_argument("--block", type=int, nargs=2, default=(128, 128))
    shard = sub.choices["shard"]
    shard.add_argument("--weights", required=True)
    shard.add_argument("--out", required=True)
    shard.add_argument("--quant", choices=("nf4", "none"), default="nf4")
    shard.add_argument("--dtype", default="bfloat16")
    shard.add_argument("--layers", type=int, required=True)
    shard.add_argument("--check-layers", type=int, nargs="+", default=[1])
    shard.add_argument("--keep-free-gb", type=float, default=100.0)
    args = parser.parse_args(argv)
    return cmd_convert(args) if args.command == "convert" else cmd_shard(args)


if __name__ == "__main__":
    sys.exit(main())
