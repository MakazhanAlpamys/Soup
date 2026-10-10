#!/usr/bin/env python3
"""Driver for the per-drive throttle diagnostic: one long reader process per arm.

The record is ``benchmarks/probe-rtx5070-drive-throttle.md``; its decision rule (section 2,
in code as ``two_drive_throttle_rule.py``) and this protocol (section 4) were committed before
any arm ran. The sustained striping probe saw recurring slow stretches with both drives at about
2.5 GB/s instead of about 4. This asks which drive does it, alone or only when both read. Six
arms, one process each, never two at once, the second round in the reverse order of the first:

    round 0: C-ALONE, D-ALONE, BOTH     round 1: BOTH, D-ALONE, C-ALONE

Imported unchanged from ``two_drive_stripe_gate.py`` and ``two_drive_sustained.py``, not
copied: the box stamp before and after every arm (power, commit, GPU, ASPM,
``soup_cli.__file__``, every process's I/O bytes), the pre-arm checks and their 15-minute wait,
the 2-s power and 1-s PDH sampler (plus the 10-s GPU rows), and the void rule (battery power,
a suspend, no result, a foreign reader of 1 GB or more). The reader is
``two_drive_throttle_read.py``. Added: a rest between arms, a metadata-only listing of the two
cache folders before the first arm (``--check-files`` does only that), the rule's per-arm
analysis, and ``--summarize``, which recomputes the verdict from the committed files.

No torch import here: the driver must not charge commit or touch the GPU itself. Windows only.

usage: two_drive_throttle.py [--arm-s 300] [--rest-s 30] [--dry-run] [--run-prefix throttle]
                             [--out-dir DIR] [--c-dir DIR] [--d-dir DIR]
       two_drive_throttle.py --summarize [--run-prefix throttle] [--out-dir DIR]
       two_drive_throttle.py --check-files | --selftest
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import two_drive_stripe_gate as gate
import two_drive_sustained as sus
import two_drive_throttle_rule as rule

#: The files each arm cycles over: the one-root 70B cache on C: and the striped cache's odd
#: layers on D: (the folder the sustained probe built; its name is a hash of the C: primary).
C_DIR = "C:/Users/user/.soup/layer-stream/D___synth__llama-70b-shape-v32k-f4"
D_DIR = "D:/soup-stripe/D___synth__llama-70b-shape-v32k-f4-cba3360a9c80"
SPAN_MIB = 400
REST_S = 30.0
EST_START_S = 25.0  # reader start-up: torch, CUDA context, listing, the untimed span
EST_STAMP_S = 20.0  # the after stamp plus the before stamp of the next arm
ARM_LABEL = {"c": "C-ALONE", "d": "D-ALONE", "both": "BOTH"}


# ==========================================================================
# files
# ==========================================================================
def use_out_dir(out_dir: str) -> None:
    """Point the gate's stamp and driver-JSON helpers at this probe's own files."""
    gate.OUT_DIR = out_dir
    gate.BOX_LOG = os.path.join(out_dir, "throttle_box_state.log")
    gate.DRIVER_JSON = os.path.join(out_dir, "throttle_driver.json")


def count_spans(directory: str, span_bytes: int) -> Tuple[int, int]:
    """(whole spans, files) under ``directory``: metadata only, no file is opened."""
    spans = files = 0
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            files += 1
            spans += os.path.getsize(path) // span_bytes
    return spans, files


def preflight_files(c_dir: str, d_dir: str) -> Tuple[bool, List[str]]:
    """List both cache folders (names and sizes only) and check each holds enough bytes."""
    span_bytes = SPAN_MIB * 1024 * 1024
    lines, ok = [], True
    for drive, directory in (("C", c_dir), ("D", d_dir)):
        try:
            spans, files = count_spans(directory, span_bytes)
        except OSError as exc:
            lines.append(f"{drive}: {directory}: {exc}")
            ok = False
            continue
        enough = spans * span_bytes >= rule.POOL_MIN_BYTES
        ok = ok and enough
        lines.append(
            f"{drive}: {directory}: {files} files, {spans} spans of {SPAN_MIB} MiB "
            f"({spans * span_bytes / 1e9:.1f} GB) - {'ok' if enough else 'TOO SMALL'}"
        )
    return ok, lines


def worker_command(arm: str, label: str, out_arg: str, args: argparse.Namespace) -> List[str]:
    return [
        sys.executable,
        "benchmarks/harness/two_drive_throttle_read.py",
        "--arm",
        arm,
        "--seconds",
        str(args.arm_s),
        "--c-dir",
        args.c_dir,
        "--d-dir",
        args.d_dir,
        "--span-mib",
        str(SPAN_MIB),
        "--label",
        label,
        "--out",
        out_arg,
    ]


def load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


# ==========================================================================
# one arm
# ==========================================================================
def run_arm(
    name: str, arm: str, label: str, dry_run: bool, out_dir: str, args: argparse.Namespace
) -> Dict[str, Any]:
    env = gate.arm_env()
    out_arg, json_path = sus.harness_out(out_dir, name)
    log_path = os.path.join(out_dir, f"{name}.log")
    cmd = worker_command(arm, label, out_arg, args)
    entry: Dict[str, Any] = {"run": name, "arm": arm, "label": label, "command": cmd}
    entry["before"] = gate.stamp(name, "before", env)
    if dry_run:
        print("DRY RUN:", subprocess.list2cmdline(cmd), flush=True)
        entry["dry_run"] = True
        return entry
    sampler = sus.SustainSampler()
    sampler.start()
    started = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"# {time.strftime('%Y-%m-%dT%H:%M:%S')} {subprocess.list2cmdline(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(
            cmd, cwd=gate.WORKTREE, env=env, stdout=log, stderr=subprocess.STDOUT
        )
        entry["pid"] = proc.pid
        exit_code = proc.wait()
    ended = time.time()
    sampler.stop()
    entry.update(
        exit_code=exit_code, started_unix=started, ended_unix=ended, wall_s=ended - started
    )
    entry["during"] = sampler.summary()
    entry["during_samples"] = {
        "power": [list(row) for row in sampler.power_rows],
        "pdh": [[stamp_s, row] for stamp_s, row in sampler.pdh_rows],
        "gpu": sampler.gpu_rows,
    }
    entry["after"] = gate.stamp(name, "after", env, before=entry["before"])
    worker = load_json(json_path)
    entry["problems"] = []
    if worker is not None:
        entry["analysis"] = rule.analyze_arm(arm, worker, entry["during_samples"]["pdh"])
        entry["problems"] = entry["analysis"]["problems"]
        if exit_code != 0 and not entry["problems"]:
            entry["problems"] = [f"reader exit code {exit_code}"]
    sample_times = [row[0] for row in sampler.power_rows] + [ended]
    max_gap = max((b - a for a, b in zip(sample_times, sample_times[1:])), default=None)
    entry["during"]["max_power_sample_gap_s"] = max_gap
    void_reasons = []
    if entry["during"]["ac_min"] != 1 or entry["after"]["ac_line_status"] != 1:
        void_reasons.append(f"AC values seen {entry['during']['ac_values_seen']}")
    if max_gap is None or max_gap > gate.SUSPEND_GAP_S:
        void_reasons.append(f"suspended: {max_gap} s between power samples")
    headroom = entry["during"].get("commit_headroom_min_gib")
    if headroom is None or headroom * 2**30 < gate.HEADROOM_MIN:
        void_reasons.append(
            f"commit headroom {headroom} GiB, below {gate.HEADROOM_MIN / 2**30:.0f}"
        )
    if worker is None:
        void_reasons.append(f"no reader result (exit code {exit_code})")
    heavy = sus.foreign_readers(entry["after"], os.getpid())
    entry["foreign_readers"] = heavy
    if heavy:
        void_reasons.append(
            "foreign reader: "
            + ", ".join(f"{r['name']}({r['pid']}) {r['read_delta'] / 1e9:.2f} GB" for r in heavy)
        )
    entry["void_reasons"] = void_reasons
    if void_reasons:
        for path in (json_path, log_path):
            if os.path.exists(path):
                target = gate._void_name(path)
                os.replace(path, target)
                entry.setdefault("void_files", []).append(os.path.basename(target))
    print(
        f"{name}: exit {exit_code}, wall {entry['wall_s']:.1f} s, "
        f"{describe(entry.get('analysis'))}, "
        f"problems {entry['problems'] or 'none'}, void {void_reasons or 'no'}",
        flush=True,
    )
    return entry


def describe(analysis: Optional[Dict[str, Any]]) -> str:
    if not analysis or not analysis.get("drives"):
        return "no analysis"
    return "; ".join(
        f"{drive.upper()}: mean {info['mean_own_gbs']} GB/s, best bin {info['best_bin_gbs']}, "
        f"{len(info['episodes'])} episode(s)"
        for drive, info in analysis["drives"].items()
    )


def stop_row(entry: Dict[str, Any]) -> Optional[str]:
    """Row 1, the one row a single arm can settle: no later arm can undo an invalid arm."""
    if entry.get("problems"):
        return f"no verdict (row 1): {entry['run']}: {'; '.join(entry['problems'])}"
    return None


# ==========================================================================
# the rule, from the committed files
# ==========================================================================
def collect(out_dir: str, prefix: str) -> Tuple[Dict[Tuple[int, str], Dict[str, Any]], str]:
    """The valid arm of every position, re-analysed from its JSON and its PDH samples."""
    use_out_dir(out_dir)
    payload = gate.load_driver_json()
    invocations = [
        inv
        for inv in payload.get("invocations", [])
        if inv.get("run_prefix") == prefix and not inv.get("dry_run")
    ]
    outcome = invocations[-1].get("outcome", "not finished") if invocations else "no invocation"
    arms: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for rnd, arm in rule.ROUND_PLAN:
        name = f"{prefix}_r{rnd}_{arm}"
        valid = [
            run
            for run in payload.get("runs", [])
            if run.get("run") == name
            and run.get("exit_code") is not None
            and not run.get("void_reasons")
            and not run.get("dry_run")
        ]
        worker = load_json(os.path.join(out_dir, f"{name}.json")) if valid else None
        if worker is None:
            continue
        pdh = valid[-1].get("during_samples", {}).get("pdh", [])
        arms[(rnd, arm)] = rule.analyze_arm(arm, worker, pdh)
    rule.annotate_continuity(arms)
    return arms, outcome


def summarize(out_dir: str, prefix: str) -> int:
    arms, outcome = collect(out_dir, prefix)
    print(f"sequence '{prefix}' in {out_dir}: outcome {outcome}")
    if not arms:
        print("no valid arm on record")
    for rnd, arm in rule.ROUND_PLAN:
        if (rnd, arm) not in arms:
            continue
        rec = arms[(rnd, arm)]
        print(
            f"  r{rnd} {ARM_LABEL[arm]:<8} read {rec.get('read_s', 0):.1f} s  "
            f"{describe(rec)}  problems {rec['problems'] or 'none'}"
        )
        for drive, info in rec["drives"].items():
            period = info["period_s"]
            for ep in info["episodes"]:
                print(
                    f"      {drive.upper()} episode at {ep['start_s']:.0f} s "
                    f"(drive read continuously {ep.get('continuous_read_s')} s), "
                    f"{ep['length_s']:.0f} s long, min {ep['min_gbs']} / mean "
                    f"{ep['mean_gbs']} GB/s, "
                    f"writes {ep['write_mbs']} MB/s, cpu {ep['cpu_pct']}%, other drive in a dip "
                    f"{ep.get('other_drive_dip_share')}"
                )
            if period is not None:
                print(f"      {drive.upper()} spacing of episode starts: median {period:.0f} s")
    rows, labels = rule.verdict(arms, outcome)
    for row, result in rows:
        print(f"  row {row}: {result}")
    print(f"  VERDICT  C: {labels['c']}   D: {labels['d']}")
    return 0


# ==========================================================================
# main
# ==========================================================================
def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--arm-s", type=float, default=rule.ARM_S, help="seconds each arm reads")
    parser.add_argument("--rest-s", type=float, default=REST_S, help="rest between arms")
    parser.add_argument("--settle-s", type=float, default=0.0, help="wait before the first arm")
    parser.add_argument("--dry-run", action="store_true", help="stamp and print, read nothing")
    parser.add_argument("--summarize", action="store_true", help="apply the rule to the files")
    parser.add_argument("--check-files", action="store_true", help="list the cache folders only")
    parser.add_argument("--selftest", action="store_true", help="the rule's synthetic tests")
    parser.add_argument("--run-prefix", default="throttle")
    parser.add_argument(
        "--out-dir", default=None, help="default: the results folder; a dry run: a scratch folder"
    )
    parser.add_argument("--c-dir", default=C_DIR)
    parser.add_argument("--d-dir", default=D_DIR)
    args = parser.parse_args()
    if not re.fullmatch(r"throttle[a-z0-9]{0,8}", args.run_prefix):
        parser.error("--run-prefix must be 'throttle' plus at most 8 lowercase letters or digits")
    return args


def estimate_s(n_arms: int, args: argparse.Namespace) -> float:
    return (
        args.settle_s
        + n_arms * (args.arm_s + EST_START_S)
        + (n_arms - 1) * (args.rest_s + EST_STAMP_S)
    )


def selftest() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        for name, size in (("a.bin", 10_000), ("b.bin", 25_000), ("tiny.bin", 100)):
            with open(os.path.join(tmp, name), "wb") as handle:
                handle.write(b"\0" * size)
        assert count_spans(tmp, 4096) == (2 + 6, 3), count_spans(tmp, 4096)
        ok, lines = preflight_files(tmp, tmp)
        assert not ok and "TOO SMALL" in lines[0], lines
        assert not preflight_files(os.path.join(tmp, "missing"), tmp)[0]
    assert re.fullmatch(r"throttle[a-z0-9]{0,8}", "throttle2")
    with tempfile.TemporaryDirectory() as tmp:  # collect() and summarize() on written files
        payload: Dict[str, Any] = {
            "runs": [{"run": "throttle_r0_c", "exit_code": 0, "void_reasons": ["AC"]}],
            "invocations": [{"run_prefix": "throttle", "outcome": "complete"}],
        }
        for position, (rnd, arm) in enumerate(rule.ROUND_PLAN):
            rates = {
                drive: rule._dipping(100.0, 30.0) if drive == "d" and arm != "c" else rule._quiet
                for drive in rule.ARM_DRIVES[arm]
            }
            worker, pdh = rule._synth_inputs(rates, t0=1e9 + 360.0 * position)
            name = f"throttle_r{rnd}_{arm}"
            with open(os.path.join(tmp, f"{name}.json"), "w", encoding="utf-8") as handle:
                json.dump(worker, handle)
            payload["runs"].append(
                {"run": name, "exit_code": 0, "void_reasons": [], "during_samples": {"pdh": pdh}}
            )
        with open(os.path.join(tmp, "throttle_driver.json"), "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        arms, outcome = collect(tmp, "throttle")
        assert sorted(arms) == sorted(rule.ROUND_PLAN) and outcome == "complete", (arms, outcome)
        assert rule.verdict(arms, outcome)[1] == {"c": "STEADY", "d": "THROTTLES"}
        assert arms[(1, "both")]["drives"]["d"]["episodes"][0]["continuous_read_s"] == 820.0
        assert summarize(tmp, "throttle") == 0
        os.remove(os.path.join(tmp, "throttle_r1_d.json"))  # a missing arm is no verdict
        arms, outcome = collect(tmp, "throttle")
        assert rule.verdict(arms, outcome)[1]["d"] == "NO VERDICT"
    return rule.selftest()


def main() -> int:
    args = parse()
    if args.selftest:
        return selftest()
    if sys.platform != "win32":
        print("SKIP: this driver is Windows-only (GetSystemPowerStatus, PDH, powercfg)")
        return 0
    if args.check_files:
        ok, lines = preflight_files(args.c_dir, args.d_dir)
        print("\n".join(lines))
        return 0 if ok else 2
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="throttle_dry_") if args.dry_run else gate.OUT_DIR
    out_dir = os.path.abspath(out_dir)
    if args.summarize:
        return summarize(out_dir, args.run_prefix)
    use_out_dir(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    payload = gate.load_driver_json()
    plan = [(f"{args.run_prefix}_r{rnd}_{arm}", arm, rnd) for rnd, arm in rule.ROUND_PLAN]
    invocation = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dry_run": args.dry_run,
        "run_prefix": args.run_prefix,
        "python": sys.executable,
        "pythonpath": gate.SRC,
        "worktree": sus.worktree_state(),
        "arm_s": args.arm_s,
        "rest_s": args.rest_s,
        "settle_s": args.settle_s,
        "c_dir": args.c_dir,
        "d_dir": args.d_dir,
        "span_mib": SPAN_MIB,
        "plan": [name for name, _arm, _rnd in plan],
        "estimated_s": estimate_s(len(plan), args),
    }
    payload["invocations"].append(invocation)
    gate.save_driver_json(payload)
    print(
        f"plan {invocation['plan']}; worktree {invocation['worktree']}; arms {args.arm_s:.0f} s, "
        f"rest {args.rest_s:.0f} s; estimated {invocation['estimated_s'] / 60:.1f} min without "
        f"voids or waits; scratch/output folder {out_dir}",
        flush=True,
    )
    if args.dry_run:
        print("DRY RUN: the cache folders are not listed and no reader is started", flush=True)
        ok, why = gate.precheck_ok()
        print(f"DRY RUN: pre-arm check {'met' if ok else 'NOT met: ' + why}", flush=True)
        for name, arm, rnd in plan[:1]:  # one stamp is the whole dry-run stamp
            entry = run_arm(
                name, arm, f"throttle round {rnd} {ARM_LABEL[arm]}", True, out_dir, args
            )
            payload["runs"].append(entry)
        for name, arm, rnd in plan[1:]:
            print(
                "DRY RUN:",
                subprocess.list2cmdline(
                    worker_command(
                        arm, f"throttle round {rnd} {ARM_LABEL[arm]}", f"{name}.json", args
                    )
                ),
                flush=True,
            )
        invocation["outcome"] = "dry run"
        gate.save_driver_json(payload)
        return 0
    stale = [
        name
        for name in os.listdir(out_dir)
        if re.fullmatch(rf"{args.run_prefix}_r[01]_(?:c|d|both)\.(?:json|log)", name)
    ]
    if stale:
        invocation["outcome"] = f"stopped: {stale} already exist; use a new --run-prefix"
        gate.save_driver_json(payload)
        print(invocation["outcome"], flush=True)
        return 2
    ok, lines = preflight_files(args.c_dir, args.d_dir)
    invocation["files"] = lines
    print("\n".join(lines), flush=True)
    if not ok:
        invocation["outcome"] = "stopped: cache folders missing or too small"
        gate.save_driver_json(payload)
        return 2
    if args.settle_s > 0:
        print(f"settling {args.settle_s:.0f} s before the first arm", flush=True)
        time.sleep(args.settle_s)
    outcome = "complete"
    for position, (name, arm, rnd) in enumerate(plan):
        label = f"throttle round {rnd} {ARM_LABEL[arm]}"
        if position > 0:
            print(f"resting {args.rest_s:.0f} s before {name}", flush=True)
            time.sleep(args.rest_s)
        entry: Optional[Dict[str, Any]] = None
        for attempt in (1, 2):
            ok, notes = gate.wait_for_precheck()
            if not ok:
                outcome = f"stopped: pre-arm checks not met within 15 min before {name}"
                payload["runs"].append({"run": name, "arm": arm, "precheck_notes": notes})
                break
            entry = run_arm(name, arm, label, False, out_dir, args)
            entry["attempt"], entry["precheck_notes"], entry["position"] = attempt, notes, position
            payload["runs"].append(entry)
            gate.save_driver_json(payload)
            if not entry.get("void_reasons"):
                break
            entry = None
        if outcome != "complete":
            break
        if entry is None:
            outcome = f"stopped: {name} void twice; no verdict"
            break
        row = stop_row(entry)
        if row is not None:
            outcome = f"stopped: {row}"
            break
    ran = {
        run.get("run")
        for run in payload["runs"]
        if run.get("exit_code") is not None and not run.get("void_reasons")
    }
    invocation["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    invocation["outcome"] = outcome
    invocation["not_run"] = [n for n, _a, _r in plan if n not in ran]
    gate.save_driver_json(payload)
    print(f"OUTCOME: {outcome}; not run: {invocation['not_run']}", flush=True)
    summarize(out_dir, args.run_prefix)
    if outcome == "complete":
        return 0
    return 5 if "pre-arm" in outcome else 4


if __name__ == "__main__":
    sys.exit(main())
