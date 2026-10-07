#!/usr/bin/env python3
"""L2L step 0: layer-major micro-batching on the SHIPPED streamed runtime.

The rule this runs against is §2 of benchmarks/probe-rtx5070-l2l-step0.md,
committed before the first run. Modes, one process each:

  --mode correctness  GA, GA again, then L2L over the same k micro-batches;
                      LoRA grads and per-micro-batch losses compared with
                      torch.equal (G1). --deterministic is the fallback.
  --mode timing       plain batch-1 A/B, L2L A/B at --ks, L2L C/D at --cd-k,
                      two interleaved rounds (G2-G4, V2, I2, I3, I5).
  --mode spill        L2L A at k=16, every activation pinned vs the lowest
                      layers through an unbuffered file on --spill-dir (I4).

Every arm restores the adapter snapshot and builds a fresh optimizer, runs
under a SuspendWatch, and is appended to --out the moment it exists. A machine
without CUDA is an intentional skip and exits 0.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import l2l_activations  # noqa: E402
import l2l_box  # noqa: E402
import l2l_schedule  # noqa: E402
import stream_probe  # noqa: E402

V3_MARGIN_GB = 4.0
V3_GPU_IDLE_MIB = 500


# ==========================================================================
# planning (pure, CPU-tested)
# ==========================================================================
def plan_timing_arms(ks: Sequence[int], rounds: int, cd_k: int) -> List[Tuple[int, str, str, int]]:
    plan: List[Tuple[int, str, str, int]] = []
    for rnd in range(rounds):
        forward = rnd % 2 == 0
        pair = ("A", "B") if forward else ("B", "A")
        plan += [(rnd, "plain", arm, 1) for arm in pair]
        for k in (sorted(ks) if forward else sorted(ks, reverse=True)):
            plan += [(rnd, "l2l", arm, k) for arm in pair]
        plan += [(rnd, "l2l", arm, cd_k) for arm in (("C", "D") if forward else ("D", "C"))]
    return plan


def plan_spill_arms(rounds: int) -> List[Tuple[int, str]]:
    plan: List[Tuple[int, str]] = []
    for rnd in range(rounds):
        order = ("pinned", "spill") if rnd % 2 == 0 else ("spill", "pinned")
        plan += [(rnd, variant) for variant in order]
    return plan


def spill_layers_for(n_layers: int, k: int, budget_k: int) -> int:
    """Layers whose activations do not fit a pinned budget of ``budget_k`` micro-batches."""
    if budget_k < 1:
        raise ValueError(f"the pinned budget must hold at least one micro-batch; got {budget_k}")
    resident_layers = min(n_layers, (n_layers * budget_k) // k)
    return n_layers - resident_layers


def fits_in_ram(need_bytes: int, free_gb: Optional[float],
                commit_gb: Optional[float] = None) -> bool:
    """V3's RAM row: pinning ``need_bytes`` must leave ``V3_MARGIN_GB`` of physical memory
    AND of Windows commit (page-locked memory is charged to both). Unknown free memory
    never fits: pinning blind can take the whole box, and its peers, down."""
    need_gb = need_bytes / 1e9 + V3_MARGIN_GB
    if free_gb is None or need_gb > free_gb:
        return False
    return commit_gb is None or need_gb <= commit_gb


def _recover() -> None:
    """The shipped #901 recovery: drain the stale CUDA error AND return the page-locked
    blocks a partial store left in torch's host cache, so a failed k=16 does not keep
    ~8 GB pinned on a shared box for the rest of the block."""
    from soup_cli.utils.layer_stream_runtime import recover_from_failed_page_lock

    recover_from_failed_page_lock()


def start_refusal(mode: str, v3: Optional[Dict[str, Any]], direct_io: Any,
                  tier: str) -> Optional[str]:
    """Why a timing or spill block cannot reach a verdict before its first arm (V3, V1)."""
    if mode not in ("timing", "spill"):
        return None
    if not (v3 or {}).get("ok"):
        return f"V3 fails at the block's start: {v3}"
    if tier == "disk" and direct_io is not True:
        return f"V1 fails: the disk source reads through the page cache (direct_io={direct_io})"
    return None


def release_arm_state(inst: Any, arm: str) -> None:
    """The nodequant arms (C, D) cache one zero weight per shape on the GPU (~1.1 GB on
    the 70B shape). Drop it, so every later arm sees the VRAM the earlier ones saw."""
    if arm not in ("C", "D"):
        return
    inst._zero_cache.clear()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def allocate_store(**kwargs: Any) -> Tuple[Optional[Any], Optional[str]]:
    """The store, or (None, reason) when pinning it fails (#901/#1003)."""
    try:
        return l2l_activations.ActivationStore(**kwargs), None
    except (RuntimeError, OSError) as exc:
        _recover()
        return None, f"activation store allocation failed: {exc}"


# ==========================================================================
# arguments and build
# ==========================================================================
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mode", choices=("correctness", "timing", "spill"), required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--shards", default=None)
    parser.add_argument("--quant", choices=("none", "nf4"), default="nf4")
    parser.add_argument("--tier", choices=("ram", "disk"), default="disk")
    parser.add_argument("--buffers", type=int, default=2)
    parser.add_argument("--read-ahead", type=int, default=2)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--input-seed", type=int, default=17)
    parser.add_argument("--k", type=int, default=2, help="micro-batches (correctness)")
    parser.add_argument("--ks", default="1,2,3,4,8,16", help="L2L micro-batch counts (timing)")
    parser.add_argument("--cd-k", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--spill-k", type=int, default=16)
    parser.add_argument("--spill-budget-k", type=int, default=8)
    parser.add_argument("--spill-dir", default="D:/soup-l2l-spill")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-invalid",
        action="store_true",
        help="run a timing/spill block even when V1/V3 already fail at its start "
             "(its gates will have no verdict); the default exits 3",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", default="")
    args = parser.parse_args(argv)
    # The fields stream_probe.build() reads and this probe never varies.
    args.lazy_shard_handles = False
    args.control_sync_source = False
    args.no_pin = False
    return args


def chunk_shape(config: Any, seq: int) -> Tuple[int, int, int]:
    return (1, int(seq), int(config.hidden_size))


def v3_check(need_bytes: int, gpu_used_before_mib: Optional[int]) -> Dict[str, Any]:
    mem = l2l_box.memory_status()
    pids = l2l_box.gpu_compute_pids() or []
    foreign = [pid for pid in pids if pid != os.getpid()]
    free = mem["avail_phys_gb"]
    commit = mem["commit_avail_gb"]
    need_gb = need_bytes / 1e9
    ram_ok = fits_in_ram(need_bytes, free, commit)
    gpu_ok = gpu_used_before_mib is not None and gpu_used_before_mib <= V3_GPU_IDLE_MIB
    return {
        "free_gb": free, "commit_avail_gb": commit, "need_gb": need_gb,
        "margin_gb": V3_MARGIN_GB,
        "gpu_used_before_build_mib": gpu_used_before_mib, "gpu_pids": pids,
        "foreign_gpu_pids": foreign, "ram_ok": ram_ok, "gpu_ok": gpu_ok and not foreign,
        "ok": ram_ok and gpu_ok and not foreign,
    }


# ==========================================================================
# one L2L arm
# ==========================================================================
def run_l2l_steps(step: Any, optimizer: Any, inst: Any, micro_batches: Sequence[Any],
                  store: Any, *, steps: int, warmup: int, label: str) -> Dict[str, Any]:
    import torch

    from soup_cli.utils.layer_stream_runtime import sm_clock_mhz

    pool, large = inst.pool, inst.large_pool
    tokens = sum(int(ids.numel()) for ids in micro_batches)
    records: List[Dict[str, Any]] = []
    clock_start = sm_clock_mhz()
    for index in range(warmup + steps):
        stats = store.stats
        before = (pool.loads, large.loads if large is not None else 0, stats.bytes_written,
                  stats.bytes_read, stats.put_wait_s, stats.get_wait_s)
        torch.cuda.synchronize()
        if index == warmup:
            torch.cuda.reset_peak_memory_stats()
        started_unix = time.time()
        wall_start = time.perf_counter()
        losses = step.run(micro_batches, store)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        wall = time.perf_counter() - wall_start
        if index >= warmup:
            records.append({
                "step_s": wall, "started_unix": started_unix,
                "loss": stream_probe.finite(float(sum(float(loss) for loss in losses))),
                "layer_loads": pool.loads - before[0],
                "large_loads": (large.loads - before[1]) if large is not None else 0,
                "spill_written": stats.bytes_written - before[2],
                "spill_read": stats.bytes_read - before[3],
                "put_wait_s": stats.put_wait_s - before[4],
                "get_wait_s": stats.get_wait_s - before[5],
            })
    clock_end = sm_clock_mhz()
    n = len(records)
    step_mean = sum(rec["step_s"] for rec in records) / n
    per_layer = pool.nbytes // pool.n
    large_bytes = large.nbytes if large is not None else 0
    loads = sum(rec["layer_loads"] for rec in records) / n
    large_loads = sum(rec["large_loads"] for rec in records) / n
    moved = loads * per_layer + large_loads * large_bytes
    return {
        "label": label, "tokens_per_step": tokens, "steps": steps, "warmup": warmup,
        "nocopy": inst.nocopy, "nodequant": inst.nodequant,
        "step_s_mean": step_mean,
        "step_s_min": min(rec["step_s"] for rec in records),
        "step_s_max": max(rec["step_s"] for rec in records),
        "tok_per_s": tokens / step_mean,
        "layer_loads_per_step": loads, "large_loads_per_step": large_loads,
        "bytes_moved_per_step": moved, "implied_h2d_gb_per_s": moved / step_mean / 1e9,
        "peak_alloc_gb": torch.cuda.max_memory_allocated() / 1e9,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
        "sm_clock_mhz_start": clock_start, "sm_clock_mhz_end": clock_end,
        "records": records,
    }


def _store_kwargs(config: Any, args: argparse.Namespace, k: int, dtype: Any,
                  spill_layers: int = 0) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = dict(
        n_layers=int(config.num_hidden_layers), k=k, chunk_shape=chunk_shape(config, args.seq),
        dtype=dtype, device="cuda", pin=True, spill_layers=spill_layers,
    )
    if spill_layers:
        Path(args.spill_dir).mkdir(parents=True, exist_ok=True)
        path = str(Path(args.spill_dir) / f"l2l-spill-{os.getpid()}.bin")
        kwargs["spill_factory"] = lambda size: l2l_activations.DirectSpillFile(path, size)
    return kwargs


def _store_bytes(config: Any, args: argparse.Namespace, k: int, element: int) -> int:
    return int(config.num_hidden_layers) * k * args.seq * int(config.hidden_size) * element


# ==========================================================================
# modes
# ==========================================================================
def mode_correctness(args, model, runtime, config, sink, torch_dtype) -> None:
    import torch

    l2l_schedule.make_non_vacuous(model)
    gen = torch.Generator(device="cuda").manual_seed(args.input_seed)
    vocab = int(config.vocab_size)
    mbs = [torch.randint(0, vocab, (1, args.seq), generator=gen, device="cuda")
           for _ in range(args.k)]
    record: Dict[str, Any] = {"kind": "correctness", "fixture": args.label, "k": args.k,
                              "deterministic": bool(args.deterministic)}
    try:
        l2l_schedule.zero_grads(model)
        ga1 = l2l_schedule.ga_step(model, mbs)
        g1 = l2l_schedule.collect_grads(model)
        l2l_schedule.zero_grads(model)
        ga2 = l2l_schedule.ga_step(model, mbs)
        g2 = l2l_schedule.collect_grads(model)
        call = l2l_schedule.capture_layer_call(model, mbs[0])
        step = l2l_schedule.L2LStep(model, runtime, call)
        store, reason = allocate_store(**_store_kwargs(config, args, args.k, torch_dtype))
        if store is None:
            record["skipped"] = reason
            sink.add("correctness", record)
            return
        l2l_schedule.zero_grads(model)
        losses = step.run(mbs, store)
        g3 = l2l_schedule.collect_grads(model)
        store.close()
    except RuntimeError as exc:
        if not args.deterministic:
            raise
        record["deterministic_error"] = str(exc)
        sink.add("correctness", record)
        return
    record.update({
        "a_vs_a": l2l_schedule.compare(g1, g2),
        "a_vs_a_losses_equal": l2l_schedule.losses_equal(ga1, ga2),
        "l2l": l2l_schedule.compare(g1, g3),
        "l2l_losses_equal": l2l_schedule.losses_equal(ga1, losses),
        "losses_ga": [float(x) for x in ga1], "losses_l2l": [float(x) for x in losses],
    })
    sink.add("correctness", record)
    print(f"correctness {args.label} k={args.k}: A-vs-A {record['a_vs_a']['equal']}, "
          f"L2L {record['l2l']['equal']} (max |d| {record['l2l']['max_abs']:.3e}), "
          f"losses {record['l2l_losses_equal']}")


def observe(body: Any, *, interval: float = 2.0) -> Dict[str, Any]:
    """Run ``body()`` under a SuspendWatch (V5) and a foreign-reader snapshot pair (V6),
    and attach the box stamps from before and after."""
    watch = l2l_box.SuspendWatch(interval=interval)
    box_before = l2l_box.box_stamp()
    reads_before = l2l_box.process_read_bytes()
    watch.start()
    try:
        result = body()
    finally:
        suspend = watch.stop()
    reads_after = l2l_box.process_read_bytes()
    result.update({
        "suspend": suspend,
        "box_before": box_before,
        "box_after": l2l_box.box_stamp(),
        "foreign_readers": l2l_box.foreign_readers(
            reads_before, reads_after, own_pids={os.getpid()}
        ),
    })
    return result


def _arm(args, model, inst, snapshot, label, body) -> Dict[str, Any]:
    import bitsandbytes as bnb

    l2l_schedule.restore_adapters(model, snapshot)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = bnb.optim.PagedAdamW8bit(trainable, lr=1e-4)
    result = observe(lambda: body(optimizer))
    result["label"] = label
    return result


def mode_timing(args, model, runtime, config, sink, torch_dtype, meta) -> None:
    import torch

    inst = stream_probe.Instruments(runtime, model, args.quant)
    snapshot = l2l_schedule.snapshot_adapters(model)
    ks = sorted({int(part) for part in args.ks.split(",") if part.strip()})
    gen = torch.Generator(device="cuda").manual_seed(args.input_seed)
    vocab = int(config.vocab_size)
    pool_mbs = [torch.randint(0, vocab, (1, args.seq), generator=gen, device="cuda")
                for _ in range(max(ks + [args.cd_k]))]
    call = l2l_schedule.capture_layer_call(model, pool_mbs[0])
    step = l2l_schedule.L2LStep(model, runtime, call)
    element = torch.empty((), dtype=torch_dtype).element_size()
    free_gb = meta["v3"]["free_gb"]
    commit_gb = meta["v3"]["commit_avail_gb"]
    fits = {k: fits_in_ram(_store_bytes(config, args, k, element), free_gb, commit_gb)
            for k in ks + [args.cd_k]}
    for rnd, mode, arm, k in plan_timing_arms(ks, args.rounds, args.cd_k):
        inst.nocopy = arm in ("B", "D")
        inst.nodequant = arm in ("C", "D")
        label = f"{mode}_{arm}_k{k}_r{rnd}"
        base = {"kind": "arm", "mode": mode, "arm": arm, "k": k, "round": rnd, "variant": None,
                "skipped": None, "store": None}
        if mode == "plain":
            result = _arm(args, model, inst, snapshot, label, lambda opt: stream_probe.run_steps(
                model, opt, inst, pool_mbs[0], steps=args.steps, warmup=args.warmup, label=label))
        elif not fits[k]:
            need_gb = _store_bytes(config, args, k, element) / 1e9
            result = {"skipped": f"V3: k={k} needs {need_gb:.1f} GB + {V3_MARGIN_GB} GB, "
                                 f"{free_gb} GB free",
                      "step_s_mean": None}
        else:
            store, reason = allocate_store(**_store_kwargs(config, args, k, torch_dtype))
            if store is None:
                result = {"skipped": reason, "step_s_mean": None}
            else:
                try:
                    result = _arm(args, model, inst, snapshot, label, lambda opt: run_l2l_steps(
                        step, opt, inst, pool_mbs[:k], store, steps=args.steps,
                        warmup=args.warmup, label=label))
                    result["store"] = vars(store.stats).copy()
                finally:
                    store.close()
                    del store
        sink.add("arm", {**base, **result})
        release_arm_state(inst, arm)
        print(f"{label:<22} " + (f"SKIPPED {result['skipped']}" if result.get("skipped") else
              f"{result['tok_per_s']:7.1f} tok/s  step {result['step_s_mean']:.3f} s  "
              f"peak {result['peak_alloc_gb']:.3f} GB  void {result['suspend']['void']}"))
    inst.nocopy = inst.nodequant = False


def mode_spill(args, model, runtime, config, sink, torch_dtype, meta) -> None:
    import torch

    inst = stream_probe.Instruments(runtime, model, args.quant)
    snapshot = l2l_schedule.snapshot_adapters(model)
    gen = torch.Generator(device="cuda").manual_seed(args.input_seed)
    vocab = int(config.vocab_size)
    mbs = [torch.randint(0, vocab, (1, args.seq), generator=gen, device="cuda")
           for _ in range(args.spill_k)]
    step = l2l_schedule.L2LStep(model, runtime, l2l_schedule.capture_layer_call(model, mbs[0]))
    spill_layers = spill_layers_for(int(config.num_hidden_layers), args.spill_k,
                                    args.spill_budget_k)
    meta["spill_layers"] = spill_layers
    for rnd, variant in plan_spill_arms(args.rounds):
        label = f"spill_{variant}_k{args.spill_k}_r{rnd}"
        base = {"kind": "arm", "mode": "l2l", "arm": "A", "k": args.spill_k, "round": rnd,
                "variant": variant, "skipped": None, "store": None}
        layers = spill_layers if variant == "spill" else 0
        element = torch.empty((), dtype=torch_dtype).element_size()
        resident = _store_bytes(config, args, args.spill_k, element)
        resident = resident * (int(config.num_hidden_layers) - layers) // int(
            config.num_hidden_layers)
        store, reason = None, None
        if not fits_in_ram(resident, meta["v3"]["free_gb"], meta["v3"]["commit_avail_gb"]):
            reason = (f"V3: {resident / 1e9:.1f} GB pinned + {V3_MARGIN_GB} GB margin, "
                      f"{meta['v3']['free_gb']} GB free")
        else:
            store, reason = allocate_store(**_store_kwargs(
                config, args, args.spill_k, torch_dtype, spill_layers=layers))
        if store is None:
            result = {"skipped": reason, "step_s_mean": None}
        elif layers and not store.stats.direct:
            store.close()
            result = {"skipped": "the spill backend is not unbuffered", "step_s_mean": None}
        else:
            try:
                result = _arm(args, model, inst, snapshot, label, lambda opt: run_l2l_steps(
                    step, opt, inst, mbs, store, steps=args.steps, warmup=args.warmup,
                    label=label))
                result["store"] = vars(store.stats).copy()
                if layers:
                    meta["spill_direct"] = bool(store.stats.direct)
            finally:
                store.close()
        sink.add("arm", {**base, **result})
        print(f"{label:<24} " + (f"SKIPPED {result['skipped']}" if result.get("skipped") else
              f"{result['tok_per_s']:7.1f} tok/s  step {result['step_s_mean']:.3f} s"))
    sink.flush()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if not stream_probe.cuda_available():
        print("SKIP: CUDA is required for l2l_probe.py")
        return 0
    import torch

    from soup_cli.utils.layer_stream import resolve_stream_dtype

    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    gpu_before = l2l_box.gpu_memory_used_mib()
    device = "cuda"
    dtype = resolve_stream_dtype(device)
    torch_dtype = getattr(torch, dtype)
    facts = stream_probe.gpu_facts(device)
    print("RUN: L2L step-0 probe, mode", args.mode)
    for key, value in facts.items():
        print(f"{key:<16}{value}")
    model, runtime, config, index, weights_dir, shard_dir, shard_s, build_s = stream_probe.build(
        args, device, dtype)
    stats = runtime.stats()
    direct_io = getattr(runtime.source, "direct_io", None)
    element = torch.empty((), dtype=torch_dtype).element_size()
    if args.mode == "timing":
        biggest = max([int(p) for p in args.ks.split(",") if p.strip()] + [args.cd_k])
    elif args.mode == "spill":
        biggest = args.spill_k
    else:
        biggest = args.k
    need = _store_bytes(config, args, biggest, element)
    meta = {**facts, **vars(args), "weights_dir": weights_dir, "shard_dir": shard_dir,
            "dtype": dtype, "arch": str(config.model_type), "n_layers": stats["n_layers"],
            "hidden": int(config.hidden_size), "tier": stats["tier"], "pinned": stats["pinned"],
            "source_class": type(runtime.source).__name__, "direct_io": direct_io,
            "read_ahead": stats["read_ahead"], "store_gb": stats["store_bytes"] / 1e9,
            "shard_seconds": shard_s, "build_seconds": build_s,
            "chunk_bytes": args.seq * int(config.hidden_size) * element,
            "v3": v3_check(need, gpu_before), "box": l2l_box.box_stamp(),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    print(f"{'shard_dir':<16}{shard_dir}\n{'direct_io':<16}{direct_io}\n"
          f"{'v3':<16}{meta['v3']}")
    sink = stream_probe.Sink(args.out, meta)
    refusal = start_refusal(args.mode, meta["v3"], direct_io, stats["tier"])
    if args.dry_run:
        print(f"dry run: a real {args.mode} block would "
              + (f"be REFUSED: {refusal}" if refusal else "start"))
        if fits_in_ram(need, meta["v3"]["free_gb"], meta["v3"]["commit_avail_gb"]):
            store, reason = allocate_store(**_store_kwargs(config, args, biggest, torch_dtype))
            print(f"dry run: pinned store for k={biggest} "
                  f"{'allocated' if store is not None else 'FAILED: ' + str(reason)}")
            if store is not None:
                store.close()
        else:
            print(f"dry run: k={biggest} needs {need / 1e9:.1f} GB pinned + {V3_MARGIN_GB} GB; "
                  f"{meta['v3']['free_gb']} GB free — NOT allocated (V3 would skip it)")
        runtime.close()
        return 0
    if refusal and not args.allow_invalid:
        meta["refused"] = refusal
        sink.payload["meta"] = meta
        sink.flush()
        runtime.close()
        print(f"REFUSED (exit 3): {refusal}. --allow-invalid runs it anyway, without a verdict.")
        return 3
    if args.mode == "correctness":
        mode_correctness(args, model, runtime, config, sink, torch_dtype)
    elif args.mode == "timing":
        mode_timing(args, model, runtime, config, sink, torch_dtype, meta)
    else:
        mode_spill(args, model, runtime, config, sink, torch_dtype, meta)
    sink.payload["meta"] = meta
    sink.flush()
    runtime.close()
    print(f"wrote         {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
