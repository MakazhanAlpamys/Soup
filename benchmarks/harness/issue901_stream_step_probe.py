#!/usr/bin/env python3
"""Build a streamed model through the shipped path and run ONE forward+backward, logging
the allocator along the way (#901).

Not a benchmark. It answers "does this shape fit, and where does the memory go": it
prints ``memory_allocated`` / ``max_memory_allocated`` at the embedding, at a few
decoder layers (forward AND the start of their backward), at ``lm_head`` and at the
loss, records the allocator history so an out-of-memory can be attributed to live
blocks rather than guessed at, and writes a JSON summary (plus a ``.pickle`` snapshot
readable with ``torch.cuda._memory_viz``).

This is the instrument that reproduced #901 on a synthetic Qwen2.5-14B shape — see
``synth_checkpoint.py --shape qwen2.5-14b --arch qwen2`` — and then showed the
shape fitting in 2.944 GB once the fallback was repaired. Every step it takes is
the shipped runtime's (``stream_probe.build`` -> ``build_streamed_model``).

Typical invocation::

    python benchmarks/harness/issue901_stream_step_probe.py \
        --weights D:/synth/qwen2.5-14b-shape --seq 384 --lora-r 16 \
        --lora-targets q_proj,v_proj --out issue901_14b_seq384.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", required=True, help="model id or checkpoint path")
    parser.add_argument("--quant", default="nf4", choices=("none", "nf4"))
    parser.add_argument("--tier", default="ram", choices=("ram", "disk"))
    parser.add_argument("--buffers", type=int, default=2)
    parser.add_argument("--read-ahead", type=int, default=2)
    parser.add_argument("--no-pin", action="store_true", help="pageable host memory")
    parser.add_argument("--seed", type=int, default=3, help="adapter init seed")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-targets", default="q_proj,v_proj")
    parser.add_argument("--seq", type=int, default=384)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--layers", default="0,12,24,36,47", help="decoder layers to mark (comma list)"
    )
    parser.add_argument("--no-history", action="store_true", help="skip the allocator history")
    parser.add_argument(
        "--out", required=True, help="JSON summary; a .pickle snapshot lands beside it"
    )
    return parser.parse_args()


def gb(value: float) -> str:
    return f"{value / 1e9:.3f} GB"


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(HERE))
    import soup_cli

    print(f"soup_cli      {soup_cli.__file__}")
    import stream_probe
    import torch

    from soup_cli.utils.layer_stream import resolve_stream_dtype
    from soup_cli.utils.layer_stream_runtime import decoder_owner

    if not torch.cuda.is_available():
        print("SKIP: CUDA is required")
        return 0
    device = "cuda"
    dtype = resolve_stream_dtype(device)
    free0, total = torch.cuda.mem_get_info()
    print(f"torch         {torch.__version__}  free VRAM {gb(free0)} of {gb(total)}")
    print(f"dtype {dtype}  quant {args.quant}  tier {args.tier}  seq {args.seq} batch {args.batch}")

    build_args = argparse.Namespace(
        weights=args.weights,
        shards=None,
        quant=args.quant,
        tier=args.tier,
        buffers=args.buffers,
        read_ahead=args.read_ahead,
        no_pin=args.no_pin,
        seed=args.seed,
        lora_r=args.lora_r,
        lora_targets=args.lora_targets,
        lazy_shard_handles=False,
        control_sync_source=False,
    )
    started = time.perf_counter()
    model, runtime, config, index, _weights_dir, shard_dir, shard_s, build_s = stream_probe.build(
        build_args, device, dtype
    )
    stats = runtime.stats()
    print(f"shards        {shard_dir} ({shard_s:.1f} s); build {build_s:.1f} s")
    large = stats["large_buffer_bytes"]
    pinned_line = f"{'pinned' if stats['pinned'] else 'pageable'}"
    if stats.get("pinned_bytes"):
        pinned_line += f" ({gb(stats['pinned_bytes'])} page-locked)"
    print(
        f"store         {gb(stats['store_bytes'])} {pinned_line} tier {stats['tier']}; "
        f"buffers {stats['buffers']} x "
        f"{(stats['buffer_bytes'] - large) / stats['buffers'] / 1e6:.0f} MB + large slot "
        f"{large / 1e6:.0f} MB; large keys {list(index.large_keys)}"
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable     {trainable / 1e6:.2f}M")
    torch.cuda.synchronize()
    print(
        f"after build   allocated {gb(torch.cuda.memory_allocated())}  reserved "
        f"{gb(torch.cuda.memory_reserved())}  free {gb(torch.cuda.mem_get_info()[0])}"
    )
    resident = {}
    for name, param in model.named_parameters():
        if param.device.type == "cuda":
            resident[name] = param.numel() * param.element_size()
    for name, buf in model.named_buffers():
        if buf is not None and buf.device.type == "cuda":
            resident["buf:" + name] = buf.numel() * buf.element_size()
    top = sorted(resident.items(), key=lambda item: -item[1])[:12]
    print(f"resident params+buffers on cuda: {gb(sum(resident.values()))} over {len(resident)}")
    for name, size in top:
        print(f"    {size / 1e6:9.1f} MB  {name}")

    log = []

    def mark(tag: str) -> None:
        torch.cuda.synchronize()
        rec = {
            "tag": tag,
            "allocated": torch.cuda.memory_allocated(),
            "max_allocated": torch.cuda.max_memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "t": time.perf_counter() - started,
        }
        log.append(rec)
        print(
            f"  [{tag:<24}] alloc {gb(rec['allocated'])}  peak {gb(rec['max_allocated'])}  "
            f"reserved {gb(rec['reserved'])}"
        )

    layers = decoder_owner(model).layers
    wanted = {int(part) for part in args.layers.split(",") if part.strip()}
    handles = []
    for idx in sorted(wanted):
        if idx >= len(layers):
            continue

        def forward_hook(_module, _inputs, output, idx=idx):
            mark(f"L{idx} fwd out")
            first = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(first) and first.requires_grad:
                first.register_hook(lambda _grad, idx=idx: mark(f"L{idx} bwd start"))

        handles.append(layers[idx].register_forward_hook(forward_hook))
    handles.append(
        model.get_input_embeddings().register_forward_hook(lambda *_a: mark("embed fwd out"))
    )

    def head_hook(_module, _inputs, output):
        mark("lm_head fwd out")
        if torch.is_tensor(output) and output.requires_grad:
            output.register_hook(lambda _grad: mark("lm_head bwd start"))

    handles.append(model.get_output_embeddings().register_forward_hook(head_hook))

    if not args.no_history:
        torch.cuda.memory._record_memory_history(max_entries=200000)

    ids = torch.randint(0, int(config.vocab_size), (args.batch, args.seq), device=device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    mark("before step")
    outcome = "ok"
    error = ""
    try:
        out = model(input_ids=ids, labels=ids)
        mark("after forward (loss)")
        out.loss.backward()
        mark("after backward")
        del out
    except Exception as exc:  # catching the OOM is the point
        outcome = type(exc).__name__
        error = str(exc).splitlines()[0][:300]
        print(f"STEP RAISED {outcome}: {error}")
        traceback.print_exc(limit=12)
    memory = torch.cuda.memory_stats()
    summary = {
        "outcome": outcome,
        "error": error,
        "free_vram_start": free0,
        "total_vram": total,
        "peak_allocated": memory.get("allocated_bytes.all.peak"),
        "peak_reserved": memory.get("reserved_bytes.all.peak"),
        "num_alloc_retries": memory.get("num_alloc_retries"),
        "num_ooms": memory.get("num_ooms"),
        "store": stats,
        "large_keys": list(index.large_keys),
        "resident_top": top,
        "resident_total": sum(resident.values()),
        "log": log,
        "args": vars(args),
        "driver": str(Path(__file__).resolve().relative_to(REPO)).replace(os.sep, "/"),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    print(
        f"RESULT {outcome}: peak allocated {gb(memory.get('allocated_bytes.all.peak', 0))}  "
        f"peak reserved {gb(memory.get('reserved_bytes.all.peak', 0))}  "
        f"alloc retries {memory.get('num_alloc_retries')}  ooms {memory.get('num_ooms')}"
    )
    if not args.no_history:
        snapshot = torch.cuda.memory._snapshot()
        with open(args.out + ".pickle", "wb") as handle:
            pickle.dump(snapshot, handle)
        live = []
        for segment in snapshot.get("segments", []):
            for block in segment.get("blocks", []):
                if block.get("state") != "active_allocated":
                    continue
                frames = block.get("frames") or []
                where = [
                    f"{frame.get('filename', '?').split(os.sep)[-1]}:{frame.get('line')} "
                    f"{frame.get('name')}"
                    for frame in frames
                    if "torch" not in frame.get("filename", "")
                    or "bitsandbytes" in frame.get("filename", "")
                ][:4]
                live.append((block.get("size", 0), where))
        live.sort(key=lambda item: -item[0])
        print(f"live blocks at the end: {len(live)}, {gb(sum(size for size, _ in live))}")
        for size, where in live[:20]:
            origin = " <- ".join(where) if where else "(torch internal)"
            print(f"    {size / 1e6:9.1f} MB  {origin}")
        summary["live_top"] = live[:40]
        torch.cuda.memory._record_memory_history(enabled=None)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1, default=str)
    for hook in handles:
        hook.remove()
    runtime.close()
    print(f"wrote         {args.out}")
    return 0 if outcome == "ok" else 3


if __name__ == "__main__":
    sys.exit(main())
