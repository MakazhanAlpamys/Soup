#!/usr/bin/env python3
"""Per-layer read brackets: what layer 0 waits for at the head of a step.

``AsyncDiskSource._plan_queue`` walks one index out of a single-member group,
so a vocabulary-sized read is planned ahead of decoder layer 0 at the moment
the compute thread blocks on layer 0. ``stream_probe.py`` already brackets
every load with CUDA events — and that bracket includes the CPU-side
``source.get()`` that runs before the copy is enqueued — but it SUMS them per
step, so it cannot say which layer paid.

This driver changes nothing in ``stream_probe.py`` and nothing in ``src/``. It
imports the probe, builds through it, installs the probe's own ``Instruments``,
and then wraps the already-installed ``load_async`` / ``wait`` once more to
remember WHICH layer each appended event pair belongs to. Everything timed is
still the shipped scheduler, seen through the probe's own replicas.

It instruments from the first step and takes no warm-up, so its ABSOLUTE step
times are not throughput evidence — the ratio between layer 0 and the rest is
what it measures. Record: ``benchmarks/gate-971-async-nvme-source.md`` §8.

Typical invocation::

    python benchmarks/harness/layer0_wait.py \
        --weights D:/synth/llama-70b-shape-v32k-f4 --tier disk --quant nf4 \
        --seq 512 --batch 1 --steps 3 --read-ahead 2 --out layer0.json

A machine without CUDA is an intentional skip and exits 0.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def _probe() -> Any:
    """Import the sibling probe, whichever directory this was invoked from."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import stream_probe

    return stream_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--tier", choices=("ram", "disk"), default="disk")
    parser.add_argument("--quant", choices=("none", "nf4"), default="nf4")
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3, help="timed steps; there is no warm-up")
    parser.add_argument("--read-ahead", type=int, default=2)
    parser.add_argument("--buffers", type=int, default=2)
    parser.add_argument("--no-pin", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--input-seed", type=int, default=17)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", default="")
    return parser.parse_args()


def _label_wrappers(inst: Any, copy_labels: List[str], stall_labels: List[str]) -> None:
    """Wrap the probe's installed hooks so each event pair carries a name."""
    pool = inst.pool
    pool_load, pool_wait = pool.load_async, pool.wait

    def load_async(idx: int, source: Any, stream: Any = None) -> Any:
        before = len(inst._copy_pairs)
        out = pool_load(idx, source, stream)
        copy_labels.extend([f"layer{idx:03d}"] * (len(inst._copy_pairs) - before))
        return out

    def wait(idx: int) -> Any:
        before = len(inst._stall_pairs)
        out = pool_wait(idx)
        stall_labels.extend([f"layer{idx:03d}"] * (len(inst._stall_pairs) - before))
        return out

    pool.load_async, pool.wait = load_async, wait

    large = inst.large_pool
    if large is None:
        return
    large_load, large_wait = large.load_async, large.wait

    def large_load_async(key: str, source: Any, stream: Any = None) -> Any:
        before = len(inst._copy_pairs)
        out = large_load(key, source, stream)
        copy_labels.extend([f"large:{key}"] * (len(inst._copy_pairs) - before))
        return out

    def large_wait_fn(key: str) -> Any:
        before = len(inst._stall_pairs)
        out = large_wait(key)
        stall_labels.extend([f"large:{key}"] * (len(inst._stall_pairs) - before))
        return out

    large.load_async, large.wait = large_load_async, large_wait_fn


def main() -> int:
    cli = parse_args()
    stream_probe = _probe()
    if not stream_probe.cuda_available():
        print("SKIP: CUDA is required for layer0_wait.py")
        return 0

    import torch

    from soup_cli.utils.layer_stream import resolve_stream_dtype

    device = "cuda"
    dtype = resolve_stream_dtype(device)
    # stream_probe.build() reads a full probe Namespace; spell every field it
    # touches rather than reusing its parser, which carries modes this has not.
    args = argparse.Namespace(
        weights=cli.weights,
        shards=None,
        quant=cli.quant,
        tier=cli.tier,
        no_pin=cli.no_pin,
        buffers=cli.buffers,
        read_ahead=cli.read_ahead,
        seq=cli.seq,
        batch=cli.batch,
        lora_r=cli.lora_r,
        lora_targets=cli.lora_targets,
        seed=cli.seed,
        input_seed=cli.input_seed,
        lazy_shard_handles=False,
        control_sync_source=False,
    )

    model, runtime, config, _index, _weights, shard_dir, shard_s, build_s = stream_probe.build(
        args, device, dtype
    )
    stats = runtime.stats()
    source_class = type(runtime.source).__name__
    print(
        f"source {source_class}  read_ahead {stats['read_ahead']}  "
        f"staging {stats['store_bytes'] / 1e9:.3f} GB "
        f"{'pinned' if stats['pinned'] else 'pageable'}  "
        f"(shard {shard_s:.1f} s, build {build_s:.1f} s)",
        flush=True,
    )

    import bitsandbytes as bnb

    optimizer = bnb.optim.PagedAdamW8bit(
        [param for param in model.parameters() if param.requires_grad], lr=1e-4
    )
    inst = stream_probe.Instruments(runtime, model, args.quant)
    inst.events_on = True
    copy_labels: List[str] = []
    stall_labels: List[str] = []
    _label_wrappers(inst, copy_labels, stall_labels)

    generator = torch.Generator(device=device).manual_seed(cli.input_seed)
    ids = torch.randint(
        0, int(config.vocab_size), (cli.batch, cli.seq), generator=generator, device=device
    )

    steps: List[Dict[str, Any]] = []
    payload = {
        "driver": "benchmarks/harness/layer0_wait.py",
        "label": cli.label,
        "weights": cli.weights,
        "shard_dir": shard_dir,
        "source_class": source_class,
        "tier": stats["tier"],
        "pinned": stats["pinned"],
        "read_ahead": stats["read_ahead"],
        "store_gb": stats["store_bytes"] / 1e9,
        "seq": cli.seq,
        "batch": cli.batch,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "steps": steps,
    }

    for index in range(cli.steps):
        copy_labels.clear()
        stall_labels.clear()
        inst.reset()
        torch.cuda.synchronize()
        started = time.perf_counter()
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        wall = time.perf_counter() - started
        copies = [
            {"what": name, "ms": start.elapsed_time(end)}
            for name, (start, end) in zip(copy_labels, inst._copy_pairs)
        ]
        stalls = [
            {"what": name, "ms": before.elapsed_time(after)}
            for name, (before, after) in zip(stall_labels, inst._stall_pairs)
        ]
        inst.reset()
        steps.append({"step": index, "step_s": wall, "copies": copies, "stalls": stalls})
        head = ", ".join(f"{rec['what']} {rec['ms']:.0f} ms" for rec in copies[:6])
        print(f"step {index}  {wall:.2f} s  first loads: {head}", flush=True)
        # After every step, like stream_probe: a long run that dies must not
        # take its earlier points with it.
        Path(cli.out).write_text(json.dumps(payload, indent=1), encoding="utf-8")

    runtime.close()
    print(f"wrote {cli.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
