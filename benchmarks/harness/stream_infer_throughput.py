#!/usr/bin/env python3
"""Forward-only throughput of the layer-streaming runtime, for a grid of batch x seq_len.

Builds the inference model (``soup_cli.utils.stream_inference``) from an EXISTING layer-stream
shard cache, then times one full forward (``logits``) per grid point: ``--warmup`` untimed runs,
then ``--repeats`` timed runs, ``torch.cuda.synchronize()`` around each, the MEDIAN reported.
The forward reads every decoder layer once, so tokens/s should rise with ``batch * seq_len`` until
the compute, not the read, bounds it: that crossover is what the grid is for.

Per point (``points[]`` in the JSON):

* ``median_s`` / ``runs_s`` / ``tokens_per_s`` (= ``batch * seq_len / median_s``)
* ``peak_vram_bytes``: ``torch.cuda.max_memory_allocated`` over that point's warmup and timed
  runs, reset per point; ``null`` on CPU
* ``layer_loads_per_forward`` / ``large_loads_per_forward``: the runtime's own load counters.
  A forward reloads every layer unless the pool still holds it from the previous forward: with
  ``--buffers`` close to the layer count the later forwards read little or nothing, which
  inflates tokens/s; keep ``--buffers`` far below the layer count when measuring the stream
* ``bytes_loaded_est_per_forward``: those loads times the pool's slot size. The runtime keeps a
  load COUNTER, not a byte counter, so this is an upper bound that is exact for a decoder whose
  layers all hold the same weights; it is the bytes copied into the pool, not the bytes read from
  disk (a warm page cache reads none)
* an out-of-memory error at a point is recorded as ``{"batch", "seq_len", "error": "oom"}`` and
  the run goes on with the next point; the exit code is then 3

The JSON is rewritten after every point, so an interrupted run keeps what it measured. A point is
a measurement, not a verdict: no threshold here, and nothing in CI runs this on a GPU.

Typical invocation::

    python benchmarks/harness/stream_infer_throughput.py \
        --shard-dir ~/.soup/layer_stream/<model-slug> --model Qwen/Qwen2.5-7B \
        --batches 1,2,4,8 --seq-lens 128,512,2048 --repeats 5 --out stream_infer.json

Exit codes: 0 ok, 2 bad arguments, 3 at least one point ran out of memory.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

SCHEMA = 1
DEFAULT_BATCHES = "1,2,4"
DEFAULT_SEQ_LENS = "128,512"
DEFAULT_WARMUP = 2
DEFAULT_REPEATS = 5
INPUT_SEED = 17


def parse_int_list(text: str, flag: str) -> List[int]:
    """A comma-separated list of distinct positive integers, order kept."""
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if not parts:
        raise ValueError(f"{flag} needs at least one value, e.g. 1,2,4")
    values: List[int] = []
    for part in parts:
        try:
            value = int(part)
        except ValueError:
            raise ValueError(f"{flag} takes integers, got {part!r}") from None
        if value <= 0:
            raise ValueError(f"{flag} values must be positive, got {value}")
        if value in values:
            raise ValueError(f"{flag} has a duplicate value: {value}")
        values.append(value)
    return values


def median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("no runs to take a median of")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Forward-only throughput of the layer-streaming runtime (batch x seq_len)."
    )
    parser.add_argument(
        "--shard-dir", required=True, help="an existing layer-stream shard cache directory"
    )
    parser.add_argument(
        "--model", required=True,
        help="the model the cache was sharded from (Hub id or local directory, for config.json)",
    )
    parser.add_argument("--device", default="cuda", help="cuda, cuda:N or cpu (default cuda)")
    parser.add_argument(
        "--dtype", default=None, help="compute dtype (default: the shard index's dtype)"
    )
    parser.add_argument("--tier", choices=("ram", "disk"), default="ram")
    parser.add_argument("--buffers", type=int, default=2, help="VRAM layer slots, 2-8")
    parser.add_argument("--read-ahead", type=int, default=None, help="disk-tier reader depth")
    parser.add_argument(
        "--pin", choices=("auto", "on", "off"), default="auto",
        help="page-locked host memory (auto: on for CUDA)",
    )
    parser.add_argument("--weights-dir", default=None, help="only for a Qwen4 external-PLE cache")
    parser.add_argument("--batches", default=DEFAULT_BATCHES, help="e.g. 1,2,4")
    parser.add_argument("--seq-lens", default=DEFAULT_SEQ_LENS, help="e.g. 128,512,2048")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="untimed runs")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help="timed runs")
    parser.add_argument("--out", default="stream_infer_throughput.json", help="JSON path")
    return parser


def _device_info(device: str) -> Dict[str, Any]:
    import torch

    info: Dict[str, Any] = {
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if str(device).startswith("cuda") and torch.cuda.is_available():
        index = torch.device(device).index
        index = torch.cuda.current_device() if index is None else index
        info["gpu"] = torch.cuda.get_device_name(index)
        info["gpu_total_bytes"] = int(torch.cuda.get_device_properties(index).total_memory)
    return info


def _sync(device: str) -> None:
    import torch

    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def make_input_ids(batch: int, seq_len: int, vocab: int, device: str) -> Any:
    """Deterministic random token ids in ``[0, vocab)``."""
    import torch

    generator = torch.Generator().manual_seed(INPUT_SEED)
    return torch.randint(0, vocab, (batch, seq_len), generator=generator).to(device)


def _slot_bytes(runtime: Any) -> Dict[str, int]:
    pool = runtime.pool
    large = runtime.large_pool
    return {
        "layer": int(pool.nbytes // max(1, pool.n)),
        "large": int(getattr(large, "nbytes", 0) or 0),
    }


def measure_point(
    stream: Any, batch: int, seq_len: int, *, vocab: int, device: str, warmup: int, repeats: int
) -> Dict[str, Any]:
    """Time ``repeats`` forwards of one ``(batch, seq_len)`` after ``warmup`` untimed ones."""
    import torch

    input_ids = make_input_ids(batch, seq_len, vocab, device)
    on_cuda = str(device).startswith("cuda")
    if on_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup):
        stream.logits(input_ids)
    _sync(device)
    before = stream.stats()
    runs: List[float] = []
    for _ in range(repeats):
        _sync(device)
        start = time.perf_counter()
        stream.logits(input_ids)
        _sync(device)
        runs.append(time.perf_counter() - start)
    after = stream.stats()
    layer_loads = (after["layer_loads"] - before["layer_loads"]) / repeats
    large_loads = (after["large_loads"] - before["large_loads"]) / repeats
    slots = _slot_bytes(stream.runtime)
    middle = median(runs)
    tokens = batch * seq_len
    return {
        "batch": batch,
        "seq_len": seq_len,
        "tokens": tokens,
        "runs_s": runs,
        "median_s": middle,
        "tokens_per_s": tokens / middle,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if on_cuda else None,
        "layer_loads_per_forward": layer_loads,
        "large_loads_per_forward": large_loads,
        "bytes_loaded_est_per_forward": int(
            layer_loads * slots["layer"] + large_loads * slots["large"]
        ),
    }


def _write(path: str, record: Dict[str, Any]) -> None:
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    os.replace(temporary, path)


def _is_oom(exc: BaseException) -> bool:
    from soup_cli.utils.layer_stream_runtime import _is_out_of_memory

    return _is_out_of_memory(exc)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        batches = parse_int_list(args.batches, "--batches")
        seq_lens = parse_int_list(args.seq_lens, "--seq-lens")
    except ValueError as exc:
        parser.error(str(exc))
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    if not os.path.isdir(args.shard_dir):
        parser.error(f"--shard-dir {args.shard_dir!r} is not a directory")

    import torch  # noqa: F401  (fail early, and once, if torch is missing)

    from soup_cli.utils.layer_shard import read_shard_index
    from soup_cli.utils.stream_inference import build_streamed_inference_model

    try:
        index = read_shard_index(args.shard_dir)
    except FileNotFoundError:
        parser.error(f"--shard-dir {args.shard_dir!r} holds no layer-stream shard index")
    options: Dict[str, Any] = {}
    if args.read_ahead is not None:
        options["read_ahead"] = args.read_ahead
    if args.pin != "auto":
        options["pin"] = args.pin == "on"
    record: Dict[str, Any] = {
        "schema": SCHEMA,
        "kind": "stream_infer_throughput",
        "started_unix": time.time(),
        **_device_info(args.device),
        "model": args.model,
        "shard_dir": os.path.abspath(args.shard_dir),
        "index": {
            "n_layers": int(index.n_layers),
            "dtype": str(index.dtype),
            "quant": str(index.quant),
            "total_params": int(index.total_params),
        },
        "settings": {
            "tier": args.tier,
            "buffers": args.buffers,
            "read_ahead": args.read_ahead,
            "pin": args.pin,
            "dtype": args.dtype,
            "batches": batches,
            "seq_lens": seq_lens,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "points": [],
    }
    exit_code = 0
    with build_streamed_inference_model(
        model_id=args.model,
        shard_dir=args.shard_dir,
        index=index,
        device=args.device,
        dtype=args.dtype,
        buffers=args.buffers,
        tier=args.tier,
        weights_dir=args.weights_dir,
        **options,
    ) as stream:
        vocab = int(stream.model.config.vocab_size)
        record["settings"]["vocab_size"] = vocab
        max_positions = getattr(stream.model.config, "max_position_embeddings", None)
        for batch in batches:
            for seq_len in seq_lens:
                if max_positions is not None and seq_len > int(max_positions):
                    point = {"batch": batch, "seq_len": seq_len, "error": "seq_len_over_context"}
                else:
                    try:
                        point = measure_point(
                            stream, batch, seq_len, vocab=vocab, device=args.device,
                            warmup=args.warmup, repeats=args.repeats,
                        )
                    except Exception as exc:  # noqa: BLE001 - an OOM is a result, not a crash
                        if not _is_oom(exc):
                            raise
                        point = {"batch": batch, "seq_len": seq_len, "error": "oom"}
                        exit_code = 3
                        if str(args.device).startswith("cuda"):
                            torch.cuda.empty_cache()
                record["points"].append(point)
                _write(args.out, record)
                if "error" in point:
                    print(f"batch {batch} x seq {seq_len}: {point['error']}")
                else:
                    print(
                        f"batch {batch} x seq {seq_len}: {point['tokens_per_s']:.1f} tok/s "
                        f"(median {point['median_s'] * 1000:.1f} ms)"
                    )
    record["finished_unix"] = time.time()
    _write(args.out, record)
    print(f"wrote {args.out}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
