#!/usr/bin/env python3
"""#974 — where does a WARM disk-tier step spend its time, stage by stage?

`stream_probe.py` brackets the copy stream and the compute-stream stall with
CUDA events; those two numbers cannot separate the host-side stages the async
source added, which is exactly what the warm regression (1.03-1.20x vs the
synchronous ``DiskSource`` with the store in the page cache) needs. This driver
builds through the shipped path, keeps the probe's CUDA instruments, and adds
host wall-clock timers around each stage, per step:

  async arm   consumer_wait  time the compute thread spends blocked in
                             ``AsyncDiskSource.get`` (a hit costs microseconds,
                             so this is the miss time)
              reader_read    time the reader thread spends inside ``_read_layer``
                             (one call per layer: since #974 the data section
                             read as K direct-I/O ranges into the slot's region;
                             before it, the per-tensor ``read_into`` loop)
              reader_drain   time the reader spends in ``draining.synchronize()``
                             before it may refill a slot (the H2D copy out of
                             that slot draining)
  control arm consumer_read  time the compute thread spends in ``DiskSource.get``
                             (``safe_open(...).get_tensor``, an mmap page-cache
                             copy on the compute thread)
  ram arm     (baseline; the store is already pinned)
  every arm   copy_s / stall_s from the probe's CUDA-event replicas, wall, tok/s

One arm per process, so run order is explicit and the page cache carries
across arms exactly as it does for `stream_probe.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--arm", required=True, choices=("async", "control", "ram"))
    parser.add_argument("--quant", default="nf4", choices=("none", "nf4"))
    parser.add_argument("--read-ahead", type=int, default=2)
    parser.add_argument("--buffers", type=int, default=2)
    parser.add_argument("--no-pin", action="store_true")
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--input-seed", type=int, default=17)
    parser.add_argument("--label", default="")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


class StageClock:
    """Thread-safe accumulators, reset per step."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.seconds: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def add(self, stage: str, seconds: float) -> None:
        with self._lock:
            self.seconds[stage] = self.seconds.get(stage, 0.0) + seconds
            self.counts[stage] = self.counts.get(stage, 0) + 1

    def take(self) -> Dict[str, Any]:
        with self._lock:
            out = {"seconds": dict(self.seconds), "counts": dict(self.counts)}
            self.seconds = {}
            self.counts = {}
            return out


def _install_async_timers(source: Any, pools: List[Any], clock: StageClock) -> None:
    """Wrap the three async stages without touching the scheduler logic."""
    import torch

    import soup_cli.utils.async_disk_source as ads

    real_get = source.get

    def get(idx: int, name: str):
        started = time.perf_counter()
        try:
            return real_get(idx, name)
        finally:
            clock.add("consumer_wait", time.perf_counter() - started)

    source.get = get

    # One bracket per LAYER read, on the instance: `_run` looks `_read_layer` up
    # on `self`, so this sees the whole K-range read as one number rather than
    # K overlapping per-range times summed to more than the wall clock.
    real_read_layer = source._read_layer

    def read_layer(idx, region):
        started = time.perf_counter()
        try:
            return real_read_layer(idx, region)
        finally:
            clock.add("reader_read", time.perf_counter() - started)

    source._read_layer = read_layer
    del ads  # the module is imported for the reader's constants only

    class TimedEvent(torch.cuda.Event):
        def synchronize(self) -> None:  # the reader's drain wait
            started = time.perf_counter()
            try:
                super().synchronize()
            finally:
                clock.add("reader_drain", time.perf_counter() - started)

    for pool in pools:
        if pool is None:
            continue
        if hasattr(pool, "events"):
            pool.events = [TimedEvent() for _ in pool.events]
        elif getattr(pool, "event", None) is not None:
            pool.event = TimedEvent()


def _install_control_timer(source: Any, clock: StageClock) -> None:
    real_get = source.get

    def get(idx: int, name: str):
        started = time.perf_counter()
        try:
            return real_get(idx, name)
        finally:
            clock.add("consumer_read", time.perf_counter() - started)

    source.get = get


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(HERE))
    import soup_cli

    print(f"soup_cli      {soup_cli.__file__}")
    import stream_probe
    import torch

    from soup_cli.utils.layer_stream import resolve_stream_dtype

    if not torch.cuda.is_available():
        print("SKIP: CUDA is required")
        return 0
    device = "cuda"
    dtype = resolve_stream_dtype(device)
    build_args = argparse.Namespace(
        weights=args.weights,
        shards=None,
        quant=args.quant,
        tier="ram" if args.arm == "ram" else "disk",
        buffers=args.buffers,
        read_ahead=args.read_ahead,
        no_pin=args.no_pin,
        seed=args.seed,
        lora_r=args.lora_r,
        lora_targets=args.lora_targets,
        lazy_shard_handles=False,
        control_sync_source=(args.arm == "control"),
    )
    facts = stream_probe.gpu_facts(device)
    model, runtime, config, index, _wd, shard_dir, shard_s, build_s = stream_probe.build(
        build_args, device, dtype
    )
    stats = runtime.stats()
    source_class = type(runtime.source).__name__
    print(
        f"arm {args.arm}  source {source_class}  tier {stats['tier']}  "
        f"store {stats['store_bytes'] / 1e9:.3f} GB "
        f"{'pinned' if stats['pinned'] else 'pageable'}  read_ahead {stats['read_ahead']}  "
        f"shard {shard_s:.1f} s  build {build_s:.1f} s"
    )
    expected = {
        "async": "AsyncDiskSource",
        "control": stream_probe.CONTROL_SOURCE_NAME,
        "ram": "RamSource",
    }
    if source_class != expected[args.arm]:
        print(f"ERROR: arm {args.arm} built a {source_class}, expected {expected[args.arm]}")
        return 2

    inst = stream_probe.Instruments(runtime, model, args.quant)
    inst.events_on = True
    clock = StageClock()
    if args.arm == "async":
        _install_async_timers(runtime.source, [runtime.pool, runtime.large_pool], clock)
    elif args.arm == "control":
        _install_control_timer(runtime.source, clock)
    # Host time spent inside the pool's load_async (AFTER Instruments installed its
    # replica): microseconds for a pinned source (an enqueue), but for the pageable
    # control it is the driver's staged memcpy, blocking the compute thread.
    for pool_obj, tag in (
        (runtime.pool, "consumer_load_async"),
        (runtime.large_pool, "consumer_load_large"),
    ):
        if pool_obj is None:
            continue
        real_load = pool_obj.load_async

        def timed_load(*a, _real=real_load, _tag=tag, **kw):
            started = time.perf_counter()
            try:
                return _real(*a, **kw)
            finally:
                clock.add(_tag, time.perf_counter() - started)

        pool_obj.load_async = timed_load

    import bitsandbytes as bnb

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = bnb.optim.PagedAdamW8bit(trainable, lr=1e-4)
    shapes = stream_probe.model_shapes(config)
    generator = torch.Generator(device=device).manual_seed(args.input_seed)
    ids = torch.randint(
        0, shapes["vocab"], (args.batch, args.seq), generator=generator, device=device
    )
    tokens = int(ids.numel())

    records: List[Dict[str, Any]] = []
    for step in range(args.warmup + args.steps):
        inst.reset()
        clock.take()
        torch.cuda.synchronize()
        if step == args.warmup:
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        wall = time.perf_counter() - started
        copy_s, stall_s, copies = inst.consume()
        stages = clock.take()
        if step >= args.warmup:
            records.append(
                {
                    "step_s": wall,
                    "copy_s": copy_s,
                    "stall_s": stall_s,
                    "copy_events": copies,
                    "stages": stages,
                    "loss": float(out.loss.detach()),
                }
            )
    keys = sorted({k for rec in records for k in rec["stages"]["seconds"]})
    mean = lambda vals: sum(vals) / len(vals)  # noqa: E731
    summary = {
        "arm": args.arm,
        "label": args.label,
        "source_class": source_class,
        "read_ahead": stats["read_ahead"],
        "pinned": stats["pinned"],
        "store_gb": stats["store_bytes"] / 1e9,
        "disk_gb": stats["disk_bytes"] / 1e9,
        "tokens_per_step": tokens,
        "steps": args.steps,
        "warmup": args.warmup,
        "step_s_mean": mean([r["step_s"] for r in records]),
        "step_s_min": min(r["step_s"] for r in records),
        "step_s_max": max(r["step_s"] for r in records),
        "tok_per_s": tokens / mean([r["step_s"] for r in records]),
        "copy_s_mean": mean([r["copy_s"] for r in records]),
        "stall_s_mean": mean([r["stall_s"] for r in records]),
        "stage_s_mean": {
            k: mean([r["stages"]["seconds"].get(k, 0.0) for r in records]) for k in keys
        },
        "stage_count_mean": {
            k: mean([r["stages"]["counts"].get(k, 0) for r in records]) for k in keys
        },
        "peak_alloc_gb": torch.cuda.max_memory_allocated() / 1e9,
        "records": records,
        "facts": facts,
        "shard_dir": shard_dir,
        "driver": os.path.relpath(__file__, HERE.parents[1]).replace(os.sep, "/")
        if HERE.parents[1] in Path(__file__).resolve().parents
        else str(Path(__file__).resolve()),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "args": vars(args),
    }
    line = (
        f"{args.arm:<8} step {summary['step_s_mean']:.3f} s ({summary['step_s_min']:.3f}-"
        f"{summary['step_s_max']:.3f})  {summary['tok_per_s']:.1f} tok/s  "
        f"copy {summary['copy_s_mean']:.3f} s  "
        f"stall {summary['stall_s_mean'] * 1e3:.1f} ms"
    )
    for k in keys:
        line += f"  {k} {summary['stage_s_mean'][k]:.3f} s/{summary['stage_count_mean'][k]:.0f}"
    print(line)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, default=str)
    runtime.close()
    print(f"wrote         {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
