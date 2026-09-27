#!/usr/bin/env python3
"""Interleaved A/B of where the head's load is issued: the step's total wait.

``StreamPrefetcher.head_prefetch_layer`` picks the decoder layer whose forward
issues the head's copy into the shared large-layer slot. ``None`` is the last
layer (the original timing); ``0`` is right after the embedding lookup. It is a
plain attribute, so this driver flips it between blocks and runs every arm in ONE
process on ONE model, the way #1178 ran ``backward_tail_prefetch=None`` as arm A.

What it reports is the compute stream's ``wait()`` brackets from the committed
``stream_probe.Instruments`` (the same brackets as gate-971 section 8), labelled
per load by ``layer0_wait``'s wrappers. The number to judge an arm by is the
step's TOTAL wait, every entry, not the head's own: the head's entry can improve
while a decoder entry behind it in the copy stream gets worse. The per-group
medians (embedding, head, layer 0, other decoder layers) say where a change went.

Schedule: ``--warmup`` steps on the first arm, discarded, then ``--blocks`` rounds
over the arms; each arm's block is ``--transition`` discarded steps followed by
``--measure`` recorded steps. The arm order rotates each round, so no arm always
follows the same neighbour. Every step is synchronised, so step time is a wall
clock of one full step; there is no no-sync mode.

Step shapes: ``sft`` is forward, backward, optimiser. ``preference`` runs a no-grad
forward first, like a preference loss's reference pass, then the same step. It is
the shape in which a head ``get()`` at layer 0 can replan the reader's queue on
the disk tier at ``--read-ahead 3``. It is not a TRL loss.

This needs a tree that has ``head_prefetch_layer`` (the #1258 branch). On any other
tree it exits with a message instead of measuring nothing.

I build the shards once and discard that process's numbers before recording
anything: the sharding itself runs on the same card as the arms, so a process
that also shards measures both arms slower than one handed an already-sharded
directory (I saw a last-layer head wait of 27.2 ms in the sharding process
against 10.9-11.5 ms in five later ones). Point ``--shards`` at an existing
directory to skip sharding in the measured process entirely; it also keeps the
shard directory's name off the weights path, which matters on Windows, where
``resolve_shard_dir`` names it after that path and a long one trips ``WinError
206`` (MAX_PATH).

Typical invocations::

    python benchmarks/harness/head_prefetch_ab.py --weights D:/synth/untied-8l \
        --tier ram --arms last,0 --out ab_ram.json
    python benchmarks/harness/head_prefetch_ab.py --weights D:/synth/untied-8l \
        --shards D:/synth/untied-8l-shards \
        --tier disk --read-ahead 3 --step-shape preference --out ab_disk_pref.json

A machine without CUDA is an intentional skip and exits 0.
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _harness() -> Any:
    """Import the sibling probe and labeller, whichever directory this ran from."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import layer0_wait
    import stream_probe

    return stream_probe, layer0_wait


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", required=True)
    parser.add_argument(
        "--shards",
        default=None,
        help="an already-sharded directory, so stream_probe.build() does not shard in "
        "the measured process (sharding there slows every arm)",
    )
    parser.add_argument("--tier", choices=("ram", "disk"), default="ram")
    parser.add_argument("--quant", choices=("none", "nf4"), default="nf4")
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--read-ahead", type=int, default=2)
    parser.add_argument("--buffers", type=int, default=2)
    parser.add_argument("--no-pin", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--input-seed", type=int, default=17)
    parser.add_argument(
        "--arms",
        default="last,0",
        help="comma list of head_prefetch_layer values; 'last' is None (the original timing)",
    )
    parser.add_argument("--step-shape", choices=("sft", "preference"), default="sft")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--transition", type=int, default=1)
    parser.add_argument("--measure", type=int, default=2)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", default="")
    return parser.parse_args()


def parse_arms(text: str) -> List[Optional[int]]:
    arms: List[Optional[int]] = []
    for part in text.split(","):
        part = part.strip().lower()
        if not part:
            continue
        arms.append(None if part in ("last", "none") else int(part))
    if len(arms) < 2 or len(set(arms)) != len(arms):
        raise SystemExit("--arms needs at least two distinct values, e.g. last,0")
    return arms


def arm_name(arm: Optional[int]) -> str:
    return "last" if arm is None else str(arm)


def _source_sha() -> str:
    """The commit of the tree that ``soup_cli`` got imported from, or ``unknown``.

    I resolve this from ``soup_cli.__file__``, not from this driver's own file:
    the driver and the ``soup_cli`` it measures can come from different
    checkouts (a ``PYTHONPATH`` override, an editable install elsewhere), and I
    want the tree that was actually measured. ``-dirty`` covers an uncommitted
    change on top of that commit. Written the way ``variant2_gate.py``'s
    ``_source_sha`` is.
    """
    import soup_cli

    package_file = getattr(soup_cli, "__file__", None)
    if package_file is None:
        return "unknown"
    try:
        package_dir = Path(package_file).resolve().parent
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=package_dir,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        if not root:
            return "unknown"
        commit = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(root),
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            .stdout.strip()
            .lower()
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=Path(root),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        return "unknown"
    return f"{commit}-dirty" if dirty else commit


def group_of(name: str) -> str:
    """Bucket one labelled wait: embedding, head, layer 0, or another decoder layer."""
    if name.startswith("large:"):
        return "head" if "lm_head" in name else "embed"
    return "layer000" if name == "layer000" else "decoder_other"


def run_one_step(model: Any, optimizer: Any, ids: Any, shape: str) -> None:
    import torch

    if shape == "preference":
        with torch.no_grad():
            model(input_ids=ids)
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ["total", "embed", "head", "layer000", "decoder_other"]
    out: Dict[str, Any] = {"n": len(rows)}
    for key in keys:
        values = [row["wait_ms"][key] for row in rows]
        out[key] = {
            "median_ms": statistics.median(values),
            "mean_ms": statistics.fmean(values),
            "min_ms": min(values),
            "max_ms": max(values),
        }
    steps = [row["step_ms"] for row in rows]
    out["step"] = {"median_ms": statistics.median(steps), "mean_ms": statistics.fmean(steps)}
    return out


def print_table(summary: Dict[str, Dict[str, Any]], arms: List[Optional[int]]) -> None:
    base = arm_name(arms[0])
    columns = ("arm", "n", "total", "embed", "head", "layer0", "decoder", "step")
    widths = (6, 3, 9, 8, 8, 8, 8, 9)
    header = " ".join(f"{name:>{width}}" for name, width in zip(columns, widths))
    print(header + "   (medians, ms)")
    for arm in arms:
        row = summary[arm_name(arm)]
        print(
            f"{arm_name(arm):>6} {row['n']:>3} {row['total']['median_ms']:>9.2f} "
            f"{row['embed']['median_ms']:>8.2f} {row['head']['median_ms']:>8.2f} "
            f"{row['layer000']['median_ms']:>8.2f} {row['decoder_other']['median_ms']:>8.2f} "
            f"{row['step']['median_ms']:>9.1f}",
            flush=True,
        )
    for arm in arms[1:]:
        row, ref = summary[arm_name(arm)], summary[base]
        print(
            f"  {arm_name(arm)} vs {base}: total wait "
            f"{row['total']['median_ms'] - ref['total']['median_ms']:+.2f} ms, step "
            f"{row['step']['median_ms'] - ref['step']['median_ms']:+.1f} ms",
            flush=True,
        )


def main() -> int:
    cli = parse_args()
    arms = parse_arms(cli.arms)
    stream_probe, layer0_wait = _harness()
    if not stream_probe.cuda_available():
        print("SKIP: CUDA is required for head_prefetch_ab.py")
        return 0

    import torch

    from soup_cli.utils.layer_stream import resolve_stream_dtype

    device = "cuda"
    dtype = resolve_stream_dtype(device)
    args = argparse.Namespace(
        weights=cli.weights,
        shards=cli.shards,
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
    prefetcher, pool = runtime.prefetcher, runtime.large_pool
    if pool is None:
        print("SKIP: a tied checkpoint has no large pool, so there is no head load to move")
        runtime.close()
        return 0
    if not hasattr(type(prefetcher), "head_prefetch_layer"):
        print(
            "ERROR: this tree has no StreamPrefetcher.head_prefetch_layer (it is on the "
            "#1258 branch); measuring would compare an arm with itself."
        )
        runtime.close()
        return 2
    for arm in arms:
        if arm is not None and not 0 <= arm < runtime.n_layers:
            print(f"ERROR: arm {arm} is outside [0, {runtime.n_layers})")
            runtime.close()
            return 2

    stats = runtime.stats()
    source_class = type(runtime.source).__name__
    print(
        f"source {source_class}  tier {stats['tier']}  read_ahead {stats['read_ahead']}  "
        f"layers {runtime.n_layers}  shared slot {pool.nbytes / 2**20:.1f} MiB  "
        f"(shard {shard_s:.1f} s, build {build_s:.1f} s)",
        flush=True,
    )

    import bitsandbytes as bnb

    optimizer = bnb.optim.PagedAdamW8bit(
        [param for param in model.parameters() if param.requires_grad], lr=1e-4
    )
    inst = stream_probe.Instruments(runtime, model, cli.quant)
    inst.events_on = True
    copy_labels: List[str] = []
    stall_labels: List[str] = []
    layer0_wait._label_wrappers(inst, copy_labels, stall_labels)

    generator = torch.Generator(device=device).manual_seed(cli.input_seed)
    ids = torch.randint(
        0, int(config.vocab_size), (cli.batch, cli.seq), generator=generator, device=device
    )

    def one_step() -> Dict[str, Any]:
        copy_labels.clear()
        stall_labels.clear()
        inst.reset()
        torch.cuda.synchronize()
        started = time.perf_counter()
        run_one_step(model, optimizer, ids, cli.step_shape)
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - started) * 1000.0
        waits = {"total": 0.0, "embed": 0.0, "head": 0.0, "layer000": 0.0, "decoder_other": 0.0}
        for name, (before, after) in zip(stall_labels, inst._stall_pairs):
            ms = before.elapsed_time(after)
            waits["total"] += ms
            waits[group_of(name)] += ms
        inst.reset()
        return {"step_ms": wall_ms, "wait_ms": waits}

    def set_arm(arm: Optional[int]) -> None:
        prefetcher.head_prefetch_layer = arm

    set_arm(arms[0])
    for _ in range(cli.warmup):
        one_step()

    records: Dict[str, List[Dict[str, Any]]] = {arm_name(arm): [] for arm in arms}
    payload = {
        "driver": "benchmarks/harness/head_prefetch_ab.py",
        # the card, its NVIDIA driver and PCIe link, torch, and soup_cli_file
        "gpu": stream_probe.gpu_facts(device),
        "git_sha": _source_sha(),
        "label": cli.label,
        "weights": cli.weights,
        "shard_dir": shard_dir,
        "source_class": source_class,
        "tier": stats["tier"],
        "pinned": stats["pinned"],
        "read_ahead": stats["read_ahead"],
        "n_layers": runtime.n_layers,
        "shared_slot_bytes": pool.nbytes,
        "seq": cli.seq,
        "batch": cli.batch,
        "step_shape": cli.step_shape,
        "arms": [arm_name(arm) for arm in arms],
        "schedule": {
            "warmup": cli.warmup,
            "blocks": cli.blocks,
            "transition": cli.transition,
            "measure": cli.measure,
        },
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "records": records,
    }

    for block in range(cli.blocks):
        order = arms[block % len(arms) :] + arms[: block % len(arms)]
        for arm in order:
            set_arm(arm)
            for _ in range(cli.transition):
                one_step()
            for _ in range(cli.measure):
                row = one_step()
                row["block"] = block
                records[arm_name(arm)].append(row)
        print(
            f"block {block + 1}/{cli.blocks} done (order {[arm_name(a) for a in order]})",
            flush=True,
        )
        # After every block, so a run that dies keeps its earlier points.
        Path(cli.out).write_text(json.dumps(payload, indent=1), encoding="utf-8")

    summary = {name: summarise(rows) for name, rows in records.items()}
    payload["summary"] = summary
    Path(cli.out).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print_table(summary, arms)
    runtime.close()
    print(f"wrote {cli.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
