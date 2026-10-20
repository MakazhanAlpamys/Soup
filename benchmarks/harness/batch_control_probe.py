#!/usr/bin/env python3
"""Driver for the R2 batch-control probe: one ``stream_probe.py`` process per arm.

The record is ``benchmarks/probe-rtx5070-batch-control.md``; its decision rule (§2) and this
protocol (§4) were committed before any arm ran. The question: on the cold 70B-shaped NF4 step
at seq 512, does tok/s scale with ``--batch`` 1 -> 2 -> 3 as the read-bound model of
``probe-rtx5070-what-bounds-streaming.md`` §21 predicts, on one drive (SINGLE) and on two
(STRIPED), and what peak VRAM does each batch need on this 8 GB card?

Two rounds, the second the exact reverse of the first (``batch_control_rule.ROUND_PLAN``):

    round 0: SINGLE b1, STRIPED b1, SINGLE b2, STRIPED b2, SINGLE b3, STRIPED b3
    round 1: STRIPED b3, SINGLE b3, STRIPED b2, SINGLE b2, STRIPED b1, SINGLE b1

Every arm runs the step block (3 timed steps after 1 warm-up, uninstrumented). A SINGLE arm
then runs the read-free compute floor (``--ablate --arms B_nocopy``) at its own batch, after
the step block; it is recorded and never judged.

Imported unchanged from ``two_drive_stripe_gate.py``, not copied: the two layouts' flags, the
box stamp before and after every arm, the pre-arm checks and their 15-minute wait, the 2-s
power and 1-s PDH sampler, and the void rule for battery power and a suspend. The rule itself
lives in ``batch_control_rule.py`` and the GPU samplers in ``batch_control_sampler.py``. Added:

- a foreign-reader void (another process reading 1 GB or more during an arm);
- an out-of-memory error, or a round arm still running after ``ARM_WALL_CAP_S``, is a RESULT
  (row 4: does not fit), not a void: it is recorded and the sequence goes on without the
  larger batches of that layout; so is a spill by row 4's test (the card as full as the
  spill of what-bounds §18, and the step slower than the read and the compute in series);
- ``nvidia-smi`` every 2 s (memory in use, SM clock, temperature, power, throttle reasons) and
  the Windows GPU memory counters every 1 s (dedicated and shared usage, per adapter and for
  the arm's own process): Windows spills an over-allocation into shared host memory without
  an error, and these are where a spill shows;
- the pre-arm GPU check at 0 MiB in use (the gate's bar was 1024);
- a CPU-only cache check (stat and index reads, no torch) that decides the build arms;
- its own box log and driver JSON (``batch_*``) under ``results/probe-rtx5070/batch-control``.

No torch import: the driver must not charge commit or touch the GPU itself. Windows only.

usage: batch_control_probe.py [--plan all|build|rounds] [--settle-s S] [--dry-run]
                              [--run-prefix batch] [--out-dir DIR]
       batch_control_probe.py --summarize [--run-prefix batch] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import batch_control_rule as rule
import two_drive_stripe_gate as gate
from batch_control_sampler import BatchSampler, self_test_sampler

# -- protocol (§4 of the record) ----------------------------------------------------------------
#: GPU memory in use before an arm, MiB. The gate allowed 1024; every gate stamp read 0 (gate
#: §5.6, §5.12, §5.21), and a peer's CUDA process holding 752 MiB passed the gate's bar at this
#: probe's dry run. A fit measurement needs the card empty.
GPU_MIB_MAX = 0
BUILD_WALL_CAP_S = 1800.0  # a build arm may re-shard (the gate's took 275.9 s)
SETTLE_S = 300.0  # between the build arms and the first round arm
DEFAULT_OUT_DIR = os.path.join(
    gate.WORKTREE, "benchmarks", "results", "probe-rtx5070", "batch-control"
)

# -- the printed wall-time estimate only --------------------------------------------------------
EST_READ_S = {"SINGLE": 17.44, "STRIPED": 9.97}  # gate sequence-4 medians (gate §5.18)
EST_OVERHEAD_S = 15.0  # gate arms' wall time minus their steps: 13-15 s
EST_BUILD_HIT_S = 40.0  # process start, build and one cold step
EST_BUILD_MISS_S = 276.0  # gate_build_striped, 275.9 s (gate §5.1)


# ==========================================================================
# files, the tree, the caches
# ==========================================================================
def use_out_dir(out_dir: str) -> None:
    """Point the gate's stamp and driver-JSON helpers at this probe's own files, and its
    pre-arm GPU check at this probe's bar (the gate's helpers read these globals when called)."""
    gate.OUT_DIR = out_dir
    gate.BOX_LOG = os.path.join(out_dir, "batch_box_state.log")
    gate.DRIVER_JSON = os.path.join(out_dir, "batch_driver.json")
    gate.GPU_MIB_MAX = GPU_MIB_MAX


def harness_out(out_dir: str, name: str) -> Tuple[str, str]:
    """(the ``--out`` argument, relative to the worktree when it can be; the absolute path)."""
    path = os.path.join(out_dir, f"{name}.json")
    try:
        rel = os.path.relpath(path, gate.WORKTREE)
    except ValueError:  # another drive
        return path, path
    return (path if rel.startswith("..") else rel.replace("\\", "/")), path


def harness_command(layout: str, batch: int, label: str, out_arg: str, build: bool) -> List[str]:
    if build:
        steps = ["--steps", "1", "--warmup", "0"]
        compute: List[str] = []
    else:
        steps = ["--steps", str(rule.TIMED_STEPS), "--warmup", str(rule.WARMUP)]
        compute = []
        if layout == "SINGLE":
            compute = ["--ablate", "--arms", rule.COMPUTE_ARM, "--rounds", "1"]
    return [
        sys.executable,
        "benchmarks/harness/stream_probe.py",
        "--weights",
        gate.WEIGHTS,
        "--quant",
        "nf4",
        "--tier",
        "disk",
        "--seq",
        str(rule.SEQ),
        "--batch",
        str(batch),
        *steps,
        "--step",
        "--skip-events",
        *compute,
        *gate.ARM_FLAGS[layout],
        "--label",
        label,
        "--out",
        out_arg,
    ]


def worktree_state() -> Dict[str, Any]:
    head = gate._run_quiet(["git", "-C", gate.WORKTREE, "rev-parse", "HEAD"])
    dirty = gate._run_quiet(
        ["git", "-C", gate.WORKTREE, "status", "--porcelain", "--", "src", "benchmarks/harness"]
    )
    return {
        "head": head.strip() if head else None,
        "src_or_harness_dirty": bool(dirty.strip()) if dirty is not None else None,
    }


def stripe_folder(env: Dict[str, str]) -> Optional[str]:
    """Root 1's folder for the STRIPED cache, as the code under test names it (slug + hash)."""
    code = (
        "from soup_cli.utils.stripe_roots import stripe_folder_name; "
        f"print(stripe_folder_name({rule.STRIPED_SHARDS!r}))"
    )
    out = gate._run_quiet([sys.executable, "-c", code], env)
    return os.path.join(os.path.normpath(rule.STRIPE_ROOT), out.strip()) if out else None


#: The sharder's own cache check with the arguments stream_probe.py passes (bfloat16, nf4,
#: double quant, quantised on cuda, no external tensors). Stat and index reads only: no torch,
#: no shard is opened, nothing is written.
_CACHE_CHECK = (
    "import json, os, sys\n"
    "from soup_cli.utils import layer_shard as ls\n"
    "from soup_cli.utils.stripe_roots import validate_stripe_root\n"
    "weights, shard_dir, roots = sys.argv[1], sys.argv[2], sys.argv[3:]\n"
    "shard_dir = shard_dir if shard_dir != '-' else ls.resolve_shard_dir(weights)\n"
    "out = ls._validate_out_dir(shard_dir)\n"
    "stripe = []\n"
    "for root in roots:\n"
    "    stripe.append(validate_stripe_root(root, primary_root=os.path.dirname(out),"
    " accepted=stripe))\n"
    "shards = ls._discover_safetensors(weights)\n"
    "comps = ls.checkpoint_source_components(weights, ls._source_file_components(shards),"
    " include_config=False)\n"
    "index, reason = ls.inspect_shard_cache(out, 'bfloat16', ls._fingerprint_components(comps),"
    " comps, 'nf4', True, 'cuda', '', stripe_roots=tuple(stripe))\n"
    "print(json.dumps({'hit': index is not None, 'reason': reason, 'shard_dir': out}))\n"
)


def cache_state(layout: str, env: Dict[str, str]) -> Dict[str, Any]:
    """``{"hit": bool, "reason": str, "shard_dir": str}``, or ``{"hit": None, "error": ...}``."""
    shards = rule.STRIPED_SHARDS if layout == "STRIPED" else "-"
    roots = [rule.STRIPE_ROOT] if layout == "STRIPED" else []
    try:
        done = subprocess.run(
            [sys.executable, "-c", _CACHE_CHECK, gate.WEIGHTS, shards, *roots],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=False,
            env=env,
            cwd=gate.WORKTREE,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"hit": None, "error": f"{type(exc).__name__}: {exc}"}
    lines = done.stdout.strip().splitlines()
    if done.returncode != 0 or not lines:
        return {"hit": None, "error": " ".join(done.stderr.split())[-400:] or "no output"}
    try:
        return json.loads(lines[-1])
    except ValueError:
        return {"hit": None, "error": f"unparsable: {lines[-1][:200]}"}


# ==========================================================================
# one arm
# ==========================================================================
def foreign_readers(after: Dict[str, Any], driver_pid: int) -> Optional[List[Dict[str, Any]]]:
    """Processes other than the driver that read >= FOREIGN_READ_MAX during the arm.

    ``None`` when the per-process capture did not run (recorded, not void). The arm's own
    process has exited by the after stamp and is not in the list.
    """
    top = after.get("top_readers")
    if top is None:
        return None
    return [
        row
        for row in top
        if row["pid"] != driver_pid and row["read_delta"] >= rule.FOREIGN_READ_MAX
    ]


def void_reasons_of(
    entry: Dict[str, Any], sampler: BatchSampler, ended: float, build: bool
) -> List[str]:
    """The gate's void rule (battery, a suspend), the foreign reader, and an arm that stopped
    without a step_plain point for any reason but out of memory or the wall cap."""
    sample_times = [row[0] for row in sampler.power_rows] + [ended]
    max_gap = max((b - a for a, b in zip(sample_times, sample_times[1:])), default=None)
    entry["during"]["max_power_sample_gap_s"] = max_gap
    reasons = []
    if entry["during"]["ac_min"] != 1 or entry["after"]["ac_line_status"] != 1:
        reasons.append(f"AC values seen {entry['during']['ac_values_seen']}")
    if max_gap is None or max_gap > gate.SUSPEND_GAP_S:
        reasons.append(f"suspended: {max_gap} s between power samples")
    heavy = foreign_readers(entry["after"], os.getpid())
    entry["foreign_readers"] = heavy
    if heavy:
        reasons.append(
            "foreign reader: "
            + ", ".join(
                f"{row['name']}({row['pid']}) {row['read_delta'] / 1e9:.2f} GB" for row in heavy
            )
        )
    facts = entry["facts"]
    a_result = facts.get("oom") or facts.get("oom_in_log") or facts.get("timed_out")
    if not facts.get("has_step_plain") and (build or not a_result):
        reasons.append(f"no step_plain point (exit code {entry['exit_code']})")
    return reasons


def run_arm(
    name: str,
    layout: str,
    batch: int,
    label: str,
    build: bool,
    dry_run: bool,
    out_dir: str,
) -> Dict[str, Any]:
    env = gate.arm_env()
    out_arg, json_path = harness_out(out_dir, name)
    log_path = os.path.join(out_dir, f"{name}.log")
    cmd = harness_command(layout, batch, label, out_arg, build)
    entry: Dict[str, Any] = {
        "run": name,
        "layout": layout,
        "batch": batch,
        "label": label,
        "build": build,
        "command": cmd,
    }
    entry["before"] = gate.stamp(name, "before", env)
    if dry_run:
        print("DRY RUN:", subprocess.list2cmdline(cmd), flush=True)
        entry["dry_run"] = True
        return entry
    cap = BUILD_WALL_CAP_S if build else rule.ARM_WALL_CAP_S
    sampler = BatchSampler()
    sampler.start()
    started = time.time()
    timed_out = False
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"# {time.strftime('%Y-%m-%dT%H:%M:%S')} {subprocess.list2cmdline(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(
            cmd, cwd=gate.WORKTREE, env=env, stdout=log, stderr=subprocess.STDOUT
        )
        sampler.pid = proc.pid
        entry["pid"] = proc.pid
        try:
            exit_code = proc.wait(timeout=cap)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            exit_code = proc.wait()
    ended = time.time()
    sampler.stop()
    entry.update(
        {
            "exit_code": exit_code,
            "timed_out": timed_out,
            "started_unix": started,
            "ended_unix": ended,
            "wall_s": ended - started,
            "during": sampler.summary(),
            "during_samples": {
                "power": [list(row) for row in sampler.power_rows],
                "pdh": [[stamp_s, row] for stamp_s, row in sampler.pdh_rows],
                "gpu": sampler.gpu_rows,
                "gpu_memory": sampler.gpu_memory_rows,
            },
        }
    )
    entry["after"] = gate.stamp(name, "after", env, before=entry["before"])
    facts = rule.arm_facts(json_path, log_path, layout, batch, timed_out, build=build)
    entry["facts"] = facts
    entry["problems"] = facts["problems"]
    entry["fits"], entry["fit_reason"] = rule.fits(facts)
    window = rule.timed_window_gpu(facts, sampler.gpu_rows)
    entry["window"] = window
    used = window["used_mib_max"]
    if used is None:
        used = entry["during"].get("nvsmi_used_mib_max")
    entry["used_mib"] = used
    entry["edge"] = rule.at_edge(used)
    entry["void_reasons"] = void_reasons_of(entry, sampler, ended, build)
    if entry["void_reasons"]:
        for path in (json_path, log_path):
            if os.path.exists(path):
                target = gate._void_name(path)
                os.replace(path, target)
                entry.setdefault("void_files", []).append(os.path.basename(target))
    print(
        f"{name}: exit {exit_code}{' (killed at the wall cap)' if timed_out else ''}, wall "
        f"{entry['wall_s']:.1f} s, step_plain {facts.get('step_s_mean')}, peak "
        f"{facts.get('peak_alloc_gb')} GB, in use {used} MiB, power "
        f"{window['power_w_median']} W, SM {window['sm_mhz_median']} MHz, fits {entry['fits']}"
        f"{' (' + entry['fit_reason'] + ')' if entry['fit_reason'] else ''}, problems "
        f"{entry['problems'] or 'none'}, void {entry['void_reasons'] or 'no'}",
        flush=True,
    )
    return entry


def stop_row(entry: Dict[str, Any]) -> Optional[str]:
    """Rows 1 and 2 of §2, the two one arm can settle; None when neither fires."""
    mean = entry["facts"].get("step_s_mean")
    low, high = rule.SINGLE_VALID
    if entry["layout"] == "SINGLE" and entry["batch"] == 1:
        if mean is None or not low <= mean <= high:
            return f"no verdict (row 1): SINGLE b1 {entry['run']} at {mean!r} s, not {low}-{high}"
    if entry.get("problems"):
        return f"no verdict (row 2): {entry['run']}: {'; '.join(entry['problems'])}"
    return None


# ==========================================================================
# the sequence
# ==========================================================================
def run_position(
    name: str, layout: str, batch: int, label: str, build: bool, out_dir: str, payload: Dict
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """One position: up to two attempts. Returns (the valid entry, or None; a stop outcome)."""
    for attempt in (1, 2):
        ok, notes = gate.wait_for_precheck()
        if not ok:
            payload["runs"].append(
                {"run": name, "layout": layout, "batch": batch, "precheck_notes": notes}
            )
            gate.save_driver_json(payload)
            return None, f"stopped: pre-arm checks not met within 15 min before {name}"
        entry = run_arm(name, layout, batch, label, build=build, dry_run=False, out_dir=out_dir)
        entry.update({"attempt": attempt, "precheck_notes": notes})
        payload["runs"].append(entry)
        gate.save_driver_json(payload)
        if not entry["void_reasons"]:
            return entry, None
    return None, f"stopped: {name} void twice; no verdict"


def run_rounds(prefix: str, dry_run: bool, payload: Dict[str, Any], out_dir: str) -> str:
    """The round arms in §4's order; returns the outcome string.

    Row 4 is applied live only to decide skips: an arm that does not fit (out of memory, the
    cap, allocated above the card, or the spill test against the latest batch-1 step of its
    layout) stops that layout's later positions at that batch or above. The verdict recomputes
    row 4 from the files with each round's own batch-1 step.
    """
    no_fit: Dict[str, int] = {}  # layout -> smallest batch that did not fit (row 4)
    last_b1: Dict[str, float] = {}  # layout -> the latest batch-1 step_s_mean
    for position, (rnd, layout, batch) in enumerate(rule.ROUND_PLAN):
        name = rule.arm_name(prefix, rnd, layout, batch)
        label = f"R2 batch control round {rnd} arm {layout} batch {batch}"
        if layout in no_fit and batch >= no_fit[layout]:
            note = f"not run: {layout} b{no_fit[layout]} did not fit (row 4)"
            print(f"{name}: {note}", flush=True)
            payload["runs"].append({"run": name, "layout": layout, "batch": batch, "skipped": note})
            gate.save_driver_json(payload)
            continue
        if dry_run:
            ok, why = gate.precheck_ok()
            print(f"DRY RUN: pre-arm check {'met' if ok else 'NOT met: ' + why}", flush=True)
            dry = run_arm(name, layout, batch, label, build=False, dry_run=True, out_dir=out_dir)
            dry["position"] = position
            payload["runs"].append(dry)
            gate.save_driver_json(payload)
            continue
        entry, stopped = run_position(name, layout, batch, label, False, out_dir, payload)
        if stopped:
            return stopped
        entry["position"] = position
        gate.save_driver_json(payload)
        row = stop_row(entry)
        if row is not None:
            return f"stopped: {row}"
        step = entry["facts"].get("step_s_mean")
        if batch == 1 and step is not None:
            last_b1[layout] = step
        spill = rule.spill_reason(step, last_b1.get(layout), batch, entry["used_mib"])
        if batch > 1 and spill:
            entry["spill_live"] = spill
            gate.save_driver_json(payload)
            print(f"{name}: {spill}", flush=True)
        if not entry["fits"] or (batch > 1 and spill):
            no_fit[layout] = min(batch, no_fit.get(layout, batch))
    return "complete"


def run_builds(
    prefix: str, builds: Sequence[str], dry_run: bool, payload: Dict[str, Any], out_dir: str
) -> str:
    """The build arms: one untimed step each, not rounds, never judged."""
    for layout in builds:
        name = rule.arm_name(prefix, None, layout, 1)
        label = f"R2 batch control build arm {layout}"
        if dry_run:
            ok, why = gate.precheck_ok()
            print(f"DRY RUN: pre-arm check {'met' if ok else 'NOT met: ' + why}", flush=True)
            payload["runs"].append(
                run_arm(name, layout, 1, label, build=True, dry_run=True, out_dir=out_dir)
            )
            gate.save_driver_json(payload)
            continue
        _entry, stopped = run_position(name, layout, 1, label, True, out_dir, payload)
        if stopped:
            return stopped
    return "complete"


def estimate_s(builds: Sequence[Tuple[str, bool]], settle_s: float, rounds: bool) -> float:
    """Wall time from the gate's step times and §21's model, without voids or waits."""
    total = settle_s if rounds else 0.0
    for _layout, hit in builds:
        total += EST_BUILD_HIT_S if hit else EST_BUILD_MISS_S
    if rounds:
        steps = rule.TIMED_STEPS + rule.WARMUP
        for _rnd, layout, batch in rule.ROUND_PLAN:
            floor = rule.model_step_s(rule.SEQ * batch)
            total += EST_OVERHEAD_S + steps * max(EST_READ_S[layout], floor)
            if layout == "SINGLE":
                total += steps * floor
    return total


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--plan", choices=("all", "build", "rounds"), default="all")
    parser.add_argument(
        "--settle-s",
        type=float,
        default=None,
        help=f"seconds before the first round arm (default {SETTLE_S:.0f} for --plan all, "
        "0 for --plan rounds)",
    )
    parser.add_argument("--dry-run", action="store_true", help="stamp and print, run no arm")
    parser.add_argument(
        "--summarize", action="store_true", help="apply §2 to the committed files, run nothing"
    )
    parser.add_argument(
        "--run-prefix",
        default="batch",
        help="file-name prefix, so a complete re-run keeps an earlier sequence's files",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="where the arms' files, the box log and the driver JSON go (default: the record's "
        "results folder; point a test dry run elsewhere)",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"batch[a-z0-9]{0,8}", args.run_prefix):
        parser.error("--run-prefix must be 'batch' plus at most 8 lowercase letters or digits")
    return args


def main() -> int:
    args = parse()
    if sys.platform != "win32":
        print("SKIP: this driver is Windows-only (GetSystemPowerStatus, PDH, powercfg)")
        return 0
    out_dir = os.path.abspath(args.out_dir)
    if args.summarize:
        return rule.summarize(out_dir, args.run_prefix)
    use_out_dir(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    env = gate.arm_env()
    payload = gate.load_driver_json()
    caches: Dict[str, Any] = {}
    builds: List[str] = []
    if args.plan in ("all", "build"):
        # STRIPED always (a hit costs one cold step); SINGLE only when not a confirmed hit.
        caches = {layout: cache_state(layout, env) for layout in rule.LAYOUTS}
        builds = ["STRIPED"] + ([] if caches["SINGLE"].get("hit") is True else ["SINGLE"])
    do_rounds = args.plan in ("all", "rounds")
    settle = args.settle_s
    if settle is None:
        settle = SETTLE_S if args.plan == "all" else 0.0
    invocation: Dict[str, Any] = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "plan": args.plan,
        "settle_s": settle,
        "python": sys.executable,
        "pythonpath": gate.SRC,
        "dry_run": args.dry_run,
        "run_prefix": args.run_prefix,
        "worktree": worktree_state(),
        "soup_cli_file": gate.soup_cli_file(env),
        "striped_shards": rule.STRIPED_SHARDS,
        "stripe_folder": stripe_folder(env),
        "caches": caches,
        "builds": builds,
        "round_plan": [list(position) for position in rule.ROUND_PLAN] if do_rounds else [],
        "estimated_s": estimate_s(
            [(layout, caches.get(layout, {}).get("hit") is True) for layout in builds],
            settle,
            do_rounds,
        ),
    }
    if args.dry_run:
        invocation["sampler_self_test"] = self_test_sampler()
    payload["invocations"].append(invocation)
    gate.save_driver_json(payload)
    print(
        f"plan {args.plan}: builds {builds}; caches {caches}; worktree {invocation['worktree']}; "
        f"soup_cli {invocation['soup_cli_file']}; STRIPED root 1 folder "
        f"{invocation['stripe_folder']}; estimated {invocation['estimated_s'] / 60:.1f} min "
        "without voids or waits",
        flush=True,
    )
    if args.dry_run:
        print(f"DRY RUN: sampler self-test {invocation['sampler_self_test']}", flush=True)
    outcome = run_builds(args.run_prefix, builds, args.dry_run, payload, out_dir)
    if outcome == "complete" and do_rounds:
        if args.dry_run:
            print(f"DRY RUN: would settle {settle:.0f} s before the first round arm", flush=True)
        elif settle > 0:
            print(f"settling {settle:.0f} s before the first round arm", flush=True)
            time.sleep(settle)
        outcome = run_rounds(args.run_prefix, args.dry_run, payload, out_dir)
    planned = [rule.arm_name(args.run_prefix, *position) for position in rule.ROUND_PLAN]
    ran = {
        run.get("run")
        for run in payload["runs"]
        if run.get("exit_code") is not None and not run.get("void_reasons")
    }
    skipped = {run.get("run") for run in payload["runs"] if run.get("skipped")}
    invocation["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    invocation["outcome"] = outcome if not args.dry_run else "dry run"
    invocation["not_run"] = (
        []
        if args.dry_run or not do_rounds
        else [name for name in planned if name not in ran and name not in skipped]
    )
    invocation["skipped"] = sorted(skipped & set(planned))
    gate.save_driver_json(payload)
    print(
        f"OUTCOME: {invocation['outcome']}; not run: {invocation['not_run']}; skipped: "
        f"{invocation['skipped']}",
        flush=True,
    )
    if do_rounds and not args.dry_run:
        rule.summarize(out_dir, args.run_prefix)
    if outcome == "complete":
        return 0
    return 5 if "pre-arm" in outcome else 4


if __name__ == "__main__":
    sys.exit(main())
