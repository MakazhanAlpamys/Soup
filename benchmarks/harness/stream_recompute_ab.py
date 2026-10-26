#!/usr/bin/env python3
"""Interleaved A/B of training.stream_recompute: ON for every layer vs OFF for the last K.

One process, one streamed model (NF4, the scale cache on, as in the record), over several
``batch x seq`` shapes. Per shape, K is the largest number of last layers whose retained
activations keep the allocated peak under ``--cap-gb``: one layer's retained bytes are
measured as what stays allocated after the FORWARD with K=2 minus the same with every layer
recomputing, halved. Not a difference of peaks: peaks are not additive, and a peak
difference under-estimated a Mistral-7B layer 3x at 1024 tokens, picked K=32 and spilled
the card (benchmarks/gate-stream-recompute.md). ``--rounds`` interleave ON and OFF blocks;
each block is ``--warmup`` discarded and ``--steps`` timed steps of forward, backward and a
PagedAdamW8bit step, synchronised, so a step time is a wall clock of one full step.

This is the method behind the record's shape table. ``soup train`` itself chooses K with
``plan_stream_recompute_auto`` (a synthetic step at the configured shape and a reserved-peak
budget); this harness does not use that planner.

Typical invocation::

    python benchmarks/harness/stream_recompute_ab.py \
        --weights unsloth/mistral-7b-instruct-v0.3 --cap-gb 7 --shapes 1x512,1x1024,1x2048

A machine without CUDA is an intentional skip and exits 0.
"""

import argparse
import contextlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]


def _stream_probe() -> Any:
    """Import the sibling probe, whichever directory this ran from."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import stream_probe

    return stream_probe


def _assert_this_tree() -> str:
    """The measured code must be this checkout's, not a same-version copy elsewhere."""
    import soup_cli

    origin = os.path.normcase(os.path.realpath(soup_cli.__file__))
    src = os.path.normcase(os.path.realpath(REPO / "src"))
    assert origin.startswith(src), (
        f"soup_cli is imported from {soup_cli.__file__}, not from {src}; put this checkout's "
        "src first on PYTHONPATH"
    )
    return soup_cli.__file__


def _shapes(text: str) -> List[Tuple[int, int]]:
    shapes = []
    for item in text.split(","):
        batch, seq = item.strip().lower().split("x")
        shapes.append((int(batch), int(seq)))
    return shapes


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", default="unsloth/mistral-7b-instruct-v0.3")
    parser.add_argument("--shards", default=None, help="an already-sharded directory")
    parser.add_argument("--tier", choices=("ram", "disk"), default="ram")
    parser.add_argument("--cap-gb", type=float, default=7.0, help="allocated-peak cap for K")
    parser.add_argument("--shapes", type=_shapes, default=_shapes("1x1024,2x512,1x2048,4x512"))
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--input-seed", type=int, default=17)
    parser.add_argument("--out", default=None, help="JSON results file (default: stdout only)")
    return parser.parse_args(argv)


@contextlib.contextmanager
def _nf4_scale_cache() -> Iterator[None]:
    """stream_probe.build() has no switch for training.stream_nf4_scale_cache; the record's
    rows were measured with it on, so the builder is wrapped for the build only."""
    import soup_cli.utils.layer_stream_runtime as runtime_module

    original = runtime_module.build_streamed_model

    def with_cache(*args: Any, **kwargs: Any) -> Any:
        kwargs["nf4_scale_cache"] = True
        return original(*args, **kwargs)

    runtime_module.build_streamed_model = with_cache
    try:
        yield
    finally:
        runtime_module.build_streamed_model = original


def main(argv: Optional[List[str]] = None) -> int:
    cli = parse_args(argv)
    stream_probe = _stream_probe()
    if not stream_probe.cuda_available():
        print("SKIP: no CUDA device")
        return 0
    import bitsandbytes as bnb
    import torch

    from soup_cli.utils.layer_stream import resolve_stream_dtype
    from soup_cli.utils.layer_stream_runtime import set_stream_recompute

    print("soup_cli from", _assert_this_tree())
    device = "cuda"
    args = argparse.Namespace(
        weights=cli.weights,
        shards=cli.shards,
        quant="nf4",
        tier=cli.tier,
        no_pin=False,
        buffers=2,
        read_ahead=2,
        lora_r=cli.lora_r,
        lora_targets=cli.lora_targets,
        seed=cli.seed,
        lazy_shard_handles=False,
        control_sync_source=False,
    )
    with _nf4_scale_cache():
        model, _runtime, config, *_ = stream_probe.build(
            args, device, resolve_stream_dtype(device)
        )
    model.train()
    n_layers = int(config.num_hidden_layers)
    trainable = [param for param in model.parameters() if param.requires_grad]
    optimizer = bnb.optim.PagedAdamW8bit(trainable, lr=1e-4)
    generator = torch.Generator(device=device).manual_seed(cli.input_seed)

    def block(ids: Any, retain: Optional[int], warmup: int, steps: int) -> Dict[str, float]:
        if retain is None:
            assert set_stream_recompute(model, True) == n_layers
        else:
            assert set_stream_recompute(model, False, retain=retain) == n_layers
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        kept = 0.0
        for index in range(warmup + steps):
            torch.cuda.synchronize()
            started = time.perf_counter()
            base = torch.cuda.memory_allocated()
            out = model(input_ids=ids, labels=ids)
            kept = (torch.cuda.memory_allocated() - base) / 1e9
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            if index >= warmup:
                times.append(time.perf_counter() - started)
        return {
            "seconds": statistics.median(times),
            "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
            "kept_after_forward_gb": kept,
        }

    results: Dict[str, Any] = {
        "cap_gb": cli.cap_gb,
        "card_total_gb": torch.cuda.mem_get_info()[1] / 1e9,
        "gpu": torch.cuda.get_device_name(),
        "weights": cli.weights,
        "tier": cli.tier,
        "shapes": [],
    }
    for batch, seq in cli.shapes:
        ids = torch.randint(
            0, int(config.vocab_size), (batch, seq), generator=generator, device=device
        )
        on = block(ids, None, warmup=1, steps=2)
        two = block(ids, 2, warmup=1, steps=1)
        per_layer = max((two["kept_after_forward_gb"] - on["kept_after_forward_gb"]) / 2, 1e-3)
        k = int(max(0, min(n_layers, (cli.cap_gb - on["peak_gb"]) // per_layer)))
        row: Dict[str, Any] = {
            "batch": batch,
            "seq": seq,
            "per_layer_gb": per_layer,
            "K": k,
            "n_layers": n_layers,
            "rounds": [],
        }
        for _ in range(cli.rounds):
            row["rounds"].append(
                {
                    "on": block(ids, None, cli.warmup, cli.steps),
                    "off": block(ids, k, cli.warmup, cli.steps),
                }
            )
        on_s = statistics.median(r["on"]["seconds"] for r in row["rounds"])
        off_s = statistics.median(r["off"]["seconds"] for r in row["rounds"])
        row.update(
            {
                "on_s": on_s,
                "off_s": off_s,
                "faster_pct": 100 * (1 - off_s / on_s),
                "on_peak_gb": max(r["on"]["peak_gb"] for r in row["rounds"]),
                "off_peak_gb": max(r["off"]["peak_gb"] for r in row["rounds"]),
                "off_reserved_gb": max(r["off"]["reserved_gb"] for r in row["rounds"]),
            }
        )
        results["shapes"].append(row)
        summary = {key: value for key, value in row.items() if key != "rounds"}
        print(json.dumps(summary), flush=True)
    text = json.dumps(results, indent=1)
    if cli.out:
        Path(cli.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
