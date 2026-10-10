#!/usr/bin/env python3
"""Speed gate for the largest-first pinned-arena plan (#1702): in-order vs packed, A B B A.

The record is ``benchmarks/gate-1702-arena-packing.md``. Its plan and decision rule were
written in issue #1702 before any run; this script runs that plan and applies that rule,
and decides nothing else.

Two arms, each a fresh process through the shipped streaming path (``stream_probe.build``):

    in-order  the plan before #1702. Made inside the process by pointing the planner's
              largest-first walk at its in-order walk, so the candidate can never win.
              No config field exists for this and none is added.
    packed    the shipped planner.

Two modes: the RAM tier is the subject; the disk tier (direct I/O) is the negative control,
because its plan is the same in both arms and any difference there is noise or run order.

Per seed and mode the four runs go A B B A, and the next seed starts with the other arm.
Every result is appended to ``series.jsonl`` the moment it exists.

    --mode plan     print the schedule and exit (the default; touches nothing)
    --mode series   run the schedule, then print the verdict
    --mode strict   3 deterministic optimizer steps per arm and mode, SHA-256 of every
                    step's logits, loss, gradients and updated adapter tensors, compared
    --mode verdict  re-apply the rule to an existing ``series.jsonl``
    --mode trial    one run (what ``series`` and ``strict`` spawn)

The driver modes import no torch: the driver must not hold VRAM or page-locked memory
while a trial runs. A machine without CUDA is an intentional skip and exits 0.

Typical invocation::

    python benchmarks/harness/issue1702_arena_pack_gate.py --mode series \
        --weights D:/models/Qwen3-8B --shards D:/shards/qwen3-8b-nf4 \
        --out benchmarks/results/probe-rtx5070/arena-pack
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
ARMS = ("in-order", "packed")
SUBJECT, CONTROL = "ram", "disk"
SEEDS = (3, 17, 29, 43, 59)
#: The rule's floor for the noise band, and the bound on the control's own mean.
NOISE_FLOOR = 0.03
#: A mode with fewer differences than this (of five seeds) is inconclusive.
MIN_BLOCKS = 4
#: The allocator cap the 2026-09-28 experiment ran under, kept for comparability.
ALLOCATOR_CAP_BYTES = 4_000_000_000
#: Page-locked bytes each arm must show on the RAM tier for Qwen3-8B NF4 (#1702).
EXPECTED_SUBJECT_PINNED = {"in-order": 8 * 2**30, "packed": 6 * 2**30}


# ==========================================================================
# The schedule and the rule: pure, tested on CPU (tests/test_issue1702_arena_pack_gate.py)
# ==========================================================================
def schedule(
    seeds: Sequence[int] = SEEDS, modes: Sequence[str] = (SUBJECT, CONTROL)
) -> List[Dict[str, Any]]:
    """A B B A per seed and mode; the first arm and the first mode alternate by seed."""
    runs: List[Dict[str, Any]] = []
    for position, seed in enumerate(seeds):
        first, second = ARMS if position % 2 == 0 else ARMS[::-1]
        ordered = list(modes) if position % 2 == 0 else list(modes)[::-1]
        for mode in ordered:
            for repeat, arm in enumerate((first, second, second, first)):
                runs.append(
                    {
                        "name": f"seed{seed:02d}-{mode}-{repeat}-{arm}",
                        "seed": seed,
                        "mode": mode,
                        "arm": arm,
                        "repeat": repeat,
                    }
                )
    return runs


def block_difference(rows: Sequence[Dict[str, Any]]) -> Optional[float]:
    """``mean(packed) / mean(in-order) - 1`` for one seed and mode.

    None unless both arms have their two completed runs: a failed run costs its block,
    it is never averaged around.
    """
    by_arm = {
        arm: [float(r["tok_per_s"]) for r in rows if r["arm"] == arm and r.get("status") == "ok"]
        for arm in ARMS
    }
    if any(len(values) != 2 for values in by_arm.values()):
        return None
    return statistics.fmean(by_arm["packed"]) / statistics.fmean(by_arm["in-order"]) - 1.0


def mode_differences(rows: Sequence[Dict[str, Any]], mode: str) -> Dict[int, Optional[float]]:
    seeds = sorted({row["seed"] for row in rows if row["mode"] == mode})
    return {
        seed: block_difference([r for r in rows if r["mode"] == mode and r["seed"] == seed])
        for seed in seeds
    }


def arena_faults(
    rows: Sequence[Dict[str, Any]], expected: Dict[str, int] = EXPECTED_SUBJECT_PINNED
) -> List[str]:
    """Runs whose page-locked bytes show the arm switch did not do what the arm says."""
    faults = []
    control: Dict[int, set] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        pinned = row.get("pinned_bytes")
        if row["mode"] == SUBJECT and pinned != expected[row["arm"]]:
            faults.append(
                f"{row['name']}: {pinned} bytes page-locked, the {row['arm']} arm "
                f"must show {expected[row['arm']]}"
            )
        if row["mode"] == CONTROL:
            control.setdefault(row["seed"], set()).add(pinned)
    for seed, seen in sorted(control.items()):
        if len(seen) > 1:
            faults.append(f"seed {seed} control: the arms page-locked {sorted(seen)}, not one size")
    return faults


def runs_on_battery(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """Runs whose stamp before or after says the box was not on AC power.

    Measured on this laptop: a Qwen2.5-1.5B step takes 0.31 s on AC and 0.83 s on battery.
    A stamp that could not be read (None, or no stamp at all) accuses nobody.
    """
    stamps = ("box_before", "box_after")
    return [
        row["name"]
        for row in rows
        if any((row.get(key) or {}).get("on_ac_power") is False for key in stamps)
    ]


def decide(
    rows: Sequence[Dict[str, Any]],
    *,
    expected: Dict[str, int] = EXPECTED_SUBJECT_PINNED,
    floor: float = NOISE_FLOOR,
    min_blocks: int = MIN_BLOCKS,
) -> Dict[str, Any]:
    """The rule of issue #1702, in the order it is written there."""
    subject = mode_differences(rows, SUBJECT)
    control = mode_differences(rows, CONTROL)
    failures = {
        arm: [r["name"] for r in rows if r["arm"] == arm and r.get("status") != "ok"]
        for arm in ARMS
    }
    out: Dict[str, Any] = {
        "subject_differences": subject,
        "control_differences": control,
        "failed_runs": failures,
        "floor": floor,
    }
    faults = arena_faults(rows, expected)
    if faults:
        return {**out, "verdict": "VOID", "reason": "harness fault: " + "; ".join(faults)}
    on_battery = runs_on_battery(rows)
    if on_battery:
        return {
            **out,
            "verdict": "INCONCLUSIVE",
            "reason": (
                f"{len(on_battery)} run(s) started or ended on battery power, first "
                f"{on_battery[0]}: the GPU is power-limited there, so the box was not quiet"
            ),
        }
    subject_d = [value for value in subject.values() if value is not None]
    control_d = [value for value in control.values() if value is not None]
    for label, values in (("control", control_d), ("RAM tier", subject_d)):
        if len(values) < min_blocks:
            return {
                **out,
                "verdict": "INCONCLUSIVE",
                "reason": f"the {label} has {len(values)} differences, fewer than {min_blocks}",
            }
    control_mean = statistics.fmean(control_d)
    band = max(floor, max(abs(value) for value in control_d))
    mean = statistics.fmean(subject_d)
    out.update(
        control_mean=control_mean,
        control_sd=statistics.stdev(control_d),
        band=band,
        subject_mean=mean,
        subject_sd=statistics.stdev(subject_d),
    )
    if abs(control_mean) > floor:
        return {
            **out,
            "verdict": "INCONCLUSIVE",
            "reason": (
                f"the control's mean difference is {control_mean:+.2%}, outside +/-{floor:.0%}: "
                "the box was not quiet or the order did not cancel"
            ),
        }
    if mean < -band:
        return {**out, "verdict": "SLOWER", "reason": f"{mean:+.2%} is below -{band:.2%}"}
    if mean > band:
        return {**out, "verdict": "FASTER", "reason": f"{mean:+.2%} is above +{band:.2%}"}
    return {
        **out,
        "verdict": "NO DIFFERENCE",
        "reason": f"{mean:+.2%} is inside +/-{band:.2%}",
    }


def compare_strict(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Per mode: are the two arms' per-step digests equal, and how many were compared."""
    out: Dict[str, Any] = {}
    for mode in (SUBJECT, CONTROL):
        by_arm = {r["arm"]: r for r in results if r["mode"] == mode}
        if set(by_arm) != set(ARMS) or any(r.get("status") != "ok" for r in by_arm.values()):
            out[mode] = {"exact": False, "reason": "an arm is missing or failed"}
            continue
        first, second = (by_arm[arm]["digests"] for arm in ARMS)
        compared = sum(len(step) for step in first)
        out[mode] = {
            "exact": bool(first) and first == second,
            "steps": len(first),
            "digests_per_step": compared // max(1, len(first)),
        }
    return out


# ==========================================================================
# The box, read without torch
# ==========================================================================
def box_stamp() -> Dict[str, Any]:
    """RAM in use, commit charge and AC power. Windows; elsewhere whatever psutil gives."""
    stamp: Dict[str, Any] = {"unix": time.time()}
    try:
        import psutil

        memory = psutil.virtual_memory()
        stamp.update(ram_used_percent=memory.percent, ram_available_bytes=memory.available)
    except Exception as exc:  # the stamp is context, never a reason to stop a series
        stamp["psutil_error"] = repr(exc)
    if sys.platform == "win32":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong)
                for name in (
                    "ullTotalPhys",
                    "ullAvailPhys",
                    "ullTotalPageFile",
                    "ullAvailPageFile",
                    "ullTotalVirtual",
                    "ullAvailVirtual",
                    "ullAvailExtendedVirtual",
                )
            ]

        class PowerStatus(ctypes.Structure):
            _fields_ = [
                ("ACLineStatus", ctypes.c_ubyte),
                ("BatteryFlag", ctypes.c_ubyte),
                ("BatteryLifePercent", ctypes.c_ubyte),
                ("SystemStatusFlag", ctypes.c_ubyte),
                ("BatteryLifeTime", ctypes.c_ulong),
                ("BatteryFullLifeTime", ctypes.c_ulong),
            ]

        memory_status = MemoryStatus()
        memory_status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status)):
            limit = memory_status.ullTotalPageFile
            stamp["commit_percent"] = 100.0 * (limit - memory_status.ullAvailPageFile) / limit
        power = PowerStatus()
        if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(power)):
            stamp["on_ac_power"] = power.ACLineStatus == 1
    return stamp


# ==========================================================================
# One run
# ==========================================================================
def use_in_order_plan() -> None:
    """Make the planner return the plan it returned before #1702, for this process."""
    import soup_cli.utils.layer_stream_runtime as runtime

    runtime._plan_largest_first = runtime._plan_in_order


def _digest(tensor: Any) -> str:
    import torch

    flat = tensor.detach().to("cpu").contiguous().reshape(-1)
    return hashlib.sha256(flat.view(torch.uint8).numpy().tobytes()).hexdigest()


def _load_blocks(args: argparse.Namespace, vocab_size: int, device: str) -> Tuple[List[Any], str]:
    import torch

    count = args.warmup + args.steps
    if args.batches:
        raw = Path(args.batches).read_bytes()
        rows = json.loads(raw)["train"]
        blocks = [torch.tensor(row, dtype=torch.long, device=device).unsqueeze(0) for row in rows]
        if any(tuple(block.shape) != (1, args.seq) for block in blocks):
            raise ValueError(f"{args.batches}: every training block must hold {args.seq} tokens")
        return blocks, "file sha256 " + hashlib.sha256(raw).hexdigest()
    generator = torch.Generator().manual_seed(args.input_seed)
    ids = torch.randint(0, vocab_size, (count, 1, args.seq), generator=generator)
    return [row.to(device) for row in ids], f"random ids, input seed {args.input_seed}"


def run_trial(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result: Dict[str, Any] = {
        "status": "started",
        "name": out.stem,
        "seed": args.seed,
        "mode": args.tier,
        "arm": args.arm,
        "strict": args.strict,
        "pid": os.getpid(),
        "started_unix": time.time(),
    }

    def save() -> None:
        out.write_text(json.dumps(result, indent=1, default=str) + "\n", encoding="utf-8")

    save()
    started = time.perf_counter()
    runtime_handle = None
    try:
        import torch

        if not torch.cuda.is_available():
            print("SKIP: CUDA is required")
            result["status"] = "skipped"
            return 0
        sys.path.insert(0, str(HERE))
        import bitsandbytes as bnb
        import stream_probe

        import soup_cli
        from soup_cli.utils.layer_stream import resolve_stream_dtype

        result["soup_cli"] = soup_cli.__file__
        result["torch"] = torch.__version__
        if args.arm == "in-order":
            use_in_order_plan()
        if args.strict:
            if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
                raise RuntimeError(
                    "--strict needs CUBLAS_WORKSPACE_CONFIG=:4096:8 in the environment"
                )
            torch.use_deterministic_algorithms(True)
        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
        device = "cuda"
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.cap_bytes / total))
        try:
            oversized = torch.empty(args.cap_bytes + 256_000_000, dtype=torch.uint8, device=device)
        except torch.OutOfMemoryError:
            pass
        else:
            del oversized
            if args.cap_bytes + 256_000_000 <= total:
                raise RuntimeError("the allocator cap did not refuse an over-budget allocation")
        torch.cuda.empty_cache()
        result["cap_bytes"] = args.cap_bytes
        result["import_and_cuda_s"] = time.perf_counter() - started

        build_args = argparse.Namespace(
            weights=args.weights,
            shards=args.shards,
            quant="nf4",
            tier=args.tier,
            buffers=2,
            read_ahead=2,
            no_pin=False,
            seed=args.seed,
            lora_r=16,
            lora_targets="q_proj,v_proj",
            lazy_shard_handles=False,
            control_sync_source=False,
        )
        dtype = resolve_stream_dtype(device)
        model, runtime_handle, config, _index, weights_dir, shard_dir, shard_s, build_s = (
            stream_probe.build(build_args, device, dtype)
        )
        torch.cuda.synchronize()
        stats = runtime_handle.stats()
        source = runtime_handle.source
        result.update(
            dtype=dtype,
            weights_dir=str(weights_dir),
            shard_dir=str(shard_dir),
            shard_s=shard_s,
            build_s=build_s,
            pinned=bool(stats["pinned"]),
            pinned_bytes=stats.get("pinned_bytes"),
            store_bytes=stats.get("store_bytes"),
            arena_sizes=list(getattr(source, "arena_sizes", ()) or ()),
            source_class=type(source).__name__,
            direct_io=getattr(source, "direct_io", None),
        )
        save()
        if not stats["pinned"]:
            raise RuntimeError("the host staging is pageable: the page-lock was refused")
        if args.tier == "disk" and result["direct_io"] is not True:
            raise RuntimeError("the disk tier is not reading with direct I/O: not the control")

        trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
        result["trainable_tensors"] = len(trainable)
        optimizer = bnb.optim.PagedAdamW8bit(list(trainable.values()), lr=2e-4)
        blocks, result["data"] = _load_blocks(args, int(config.vocab_size), device)
        model.train()

        def one_step(ids: Any) -> Dict[str, Any]:
            torch.cuda.synchronize()
            tick = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            output = model(input_ids=ids, labels=ids, use_cache=False)
            output.loss.backward()
            digests = None
            if args.strict:
                digests = [_digest(output.logits), _digest(output.loss)]
                digests += [_digest(trainable[name].grad) for name in sorted(trainable)]
            optimizer.step()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - tick
            if digests is not None:
                digests += [_digest(trainable[name]) for name in sorted(trainable)]
            return {"seconds": seconds, "loss": float(output.loss.detach()), "digests": digests}

        warmup = [one_step(blocks[index % len(blocks)]) for index in range(args.warmup)]
        torch.cuda.reset_peak_memory_stats()
        steps = [
            one_step(blocks[(args.warmup + index) % len(blocks)]) for index in range(args.steps)
        ]
        seconds = [step["seconds"] for step in steps]
        result.update(
            warmup_s=[step["seconds"] for step in warmup],
            step_s=seconds,
            losses=[step["loss"] for step in warmup + steps],
            median_step_s=statistics.median(seconds),
            tok_per_s=args.steps * args.seq / sum(seconds),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        if args.strict:
            result["digests"] = [step["digests"] for step in warmup + steps]
        try:
            import psutil

            info = psutil.Process().memory_info()
            result["peak_rss_bytes"] = getattr(info, "peak_wset", None) or info.rss
        except Exception as exc:
            result["peak_rss_error"] = repr(exc)
        result["status"] = "ok"
        return 0
    except Exception as exc:
        result.update(
            status="failed",
            error_type=type(exc).__name__,
            error=str(exc).splitlines()[0][:400] if str(exc) else "",
            traceback=traceback.format_exc(),
        )
        print(result["traceback"])
        return 1
    finally:
        if runtime_handle is not None:
            try:
                runtime_handle.close()
            except Exception as exc:
                result["close_error"] = repr(exc)
        result["total_wall_s"] = time.perf_counter() - started
        save()


# ==========================================================================
# The driver
# ==========================================================================
def _spawn(args: argparse.Namespace, run: Dict[str, Any], *, strict: bool) -> Dict[str, Any]:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{'strict-' if strict else ''}{run['name']}.json"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode",
        "trial",
        "--weights",
        args.weights,
        "--tier",
        run["mode"],
        "--arm",
        run["arm"],
        "--seed",
        str(run["seed"]),
        "--warmup",
        str(0 if strict else args.warmup),
        "--steps",
        str(3 if strict else args.steps),
        "--seq",
        str(args.seq),
        "--input-seed",
        str(args.input_seed),
        "--cap-bytes",
        str(args.cap_bytes),
        "--out",
        str(target),
    ]
    if args.shards:
        command += ["--shards", args.shards]
    if args.batches:
        command += ["--batches", args.batches]
    environment = dict(os.environ)
    if strict:
        command.append("--strict")
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    before = box_stamp()
    # A result left by an earlier run of the same name must never be read as this one's.
    target.unlink(missing_ok=True)
    code: Optional[int] = None
    timed_out = False
    with open(target.with_suffix(".log"), "w", encoding="utf-8") as log:
        try:
            code = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                timeout=args.run_timeout_s,
            ).returncode
        except subprocess.TimeoutExpired:
            # subprocess.run has already killed the child; the run is a failed run.
            timed_out = True
    try:
        row = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        row = {"status": "failed", "error": f"no result file: {exc!r}"}
    if timed_out:
        row.update(
            status="failed", error=f"stopped after {args.run_timeout_s:.0f} s without a result"
        )
    elif row.get("status") == "started":
        row["status"] = "failed"
        row.setdefault("error", "the process ended without a result")
    row.update(run, returncode=code, box_before=before, box_after=box_stamp())
    return row


def run_series(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    journal = out_dir / "series.jsonl"
    if journal.exists():
        print(f"{journal} exists: a series is never appended to or re-run in place")
        return 2
    rows = []
    runs = schedule()
    for position, run in enumerate(runs, start=1):
        if box_stamp().get("on_ac_power") is False:
            # Power-limited steps are not the steps the plan measures: do not time them.
            print(f"the box is on battery power before run {position}: the series stops here")
            if not rows:
                return 2
            break
        row = _spawn(args, run, strict=False)
        rows.append(row)
        with open(journal, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str) + "\n")
        figure = f"{row['tok_per_s']:.1f} tok/s" if row.get("status") == "ok" else row.get("error")
        print(f"[{position:2d}/{len(runs)}] {run['name']:<28} {row.get('status'):<7} {figure}")
        if row.get("status") == "skipped":
            print("SKIP: CUDA is required")
            return 0
        time.sleep(args.settle_s)
    return print_verdict(rows, out_dir, expected_pinned(args))


def expected_pinned(args: argparse.Namespace) -> Dict[str, int]:
    return {"in-order": args.expect_in_order_bytes, "packed": args.expect_packed_bytes}


def print_verdict(rows: Sequence[Dict[str, Any]], out_dir: Path, expected: Dict[str, int]) -> int:
    verdict = decide(rows, expected=expected)
    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=1) + "\n", encoding="utf-8")
    for label, key in (("RAM tier", "subject_differences"), ("control", "control_differences")):
        shown = ", ".join(
            f"seed {seed}: {'n/a' if value is None else format(value, '+.2%')}"
            for seed, value in verdict[key].items()
        )
        print(f"{label:<9} {shown}")
    print(f"VERDICT   {verdict['verdict']}: {verdict['reason']}")
    return 0


def run_strict(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    results = []
    for mode in (SUBJECT, CONTROL):
        for arm in ARMS:
            run = {"name": f"seed{args.seed:02d}-{mode}-{arm}", "seed": args.seed, "mode": mode}
            row = _spawn(args, {**run, "arm": arm}, strict=True)
            results.append(row)
            print(f"strict {run['name']:<24} {row.get('status')}")
            if row.get("status") == "skipped":
                print("SKIP: CUDA is required")
                return 0
    outcome = compare_strict(results)
    # Equal digests mean something only if the RAM tier's two arms ran two different plans.
    faults = arena_faults(results, expected_pinned(args))
    summary = {
        "faults": faults,
        **{
            mode: {
                **outcome[mode],
                "pinned_bytes": {
                    row["arm"]: row.get("pinned_bytes") for row in results if row["mode"] == mode
                },
                # One line per arm for the record; the per-step digests stay in the run files.
                "sha256_of_digests": {
                    row["arm"]: hashlib.sha256(
                        "".join(d for step in row.get("digests") or [] for d in step).encode()
                    ).hexdigest()
                    for row in results
                    if row["mode"] == mode
                },
            }
            for mode in outcome
        },
    }
    (out_dir / "strict.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    for mode, entry in outcome.items():
        pinned = summary[mode]["pinned_bytes"]
        print(f"STRICT    {mode}: {'byte-identical' if entry['exact'] else 'DIFFERENT'} {pinned}")
    for fault in faults:
        print(f"FAULT     {fault}")
    return 0 if not faults and all(entry["exact"] for entry in outcome.values()) else 3


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--mode", choices=("plan", "series", "strict", "verdict", "trial"), default="plan"
    )
    parser.add_argument("--weights", help="model id or checkpoint path")
    parser.add_argument("--shards", default=None, help="shard cache dir (default: soup's own)")
    parser.add_argument(
        "--batches",
        default=None,
        help='JSON with a "train" list of token-id blocks; default: seeded random ids',
    )
    parser.add_argument("--out", default="arena-pack-gate", help="result folder (or file, trial)")
    parser.add_argument("--tier", choices=(SUBJECT, CONTROL), default=SUBJECT)
    parser.add_argument("--arm", choices=ARMS, default="packed")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--input-seed", type=int, default=17)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--cap-bytes", type=int, default=ALLOCATOR_CAP_BYTES)
    parser.add_argument("--settle-s", type=float, default=5.0, help="pause between runs")
    parser.add_argument(
        "--run-timeout-s",
        type=float,
        default=900.0,
        help="a run with no result after this long is stopped and counted as failed",
    )
    parser.add_argument(
        "--expect-in-order-bytes",
        type=int,
        default=EXPECTED_SUBJECT_PINNED["in-order"],
        help="page-locked bytes the RAM tier must show in the in-order arm (Qwen3-8B NF4: 8 GiB)",
    )
    parser.add_argument(
        "--expect-packed-bytes",
        type=int,
        default=EXPECTED_SUBJECT_PINNED["packed"],
        help="page-locked bytes the RAM tier must show in the packed arm (Qwen3-8B NF4: 6 GiB)",
    )
    parser.add_argument("--strict", action="store_true", help="trial only: deterministic + digests")
    args = parser.parse_args(argv)
    if args.mode in ("series", "strict", "trial") and not args.weights:
        parser.error(f"--mode {args.mode} needs --weights")
    if args.steps < 1 or args.warmup < 0 or args.seq < 2:
        parser.error("--steps must be >= 1, --warmup >= 0 and --seq >= 2")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.mode == "trial":
        return run_trial(args)
    if args.mode == "series":
        return run_series(args)
    if args.mode == "strict":
        return run_strict(args)
    if args.mode == "verdict":
        journal = Path(args.out) / "series.jsonl"
        lines = journal.read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line]
        return print_verdict(rows, Path(args.out), expected_pinned(args))
    for position, run in enumerate(schedule(), start=1):
        print(f"{position:2d}  {run['name']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
