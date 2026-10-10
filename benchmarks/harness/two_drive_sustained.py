#!/usr/bin/env python3
"""Driver for the sustained two-drive striping probe (R4): one long ``stream_probe.py`` per arm.

The record is ``benchmarks/probe-rtx5070-two-drive-sustained.md``; its decision rule (§2) and
this protocol (§4) were committed before any arm ran. The two-drive gate
(``benchmarks/gate-two-drive-striping.md``) timed three-step bursts. This asks whether its
speed-up holds when each arm reads for about six minutes without a break. One process per arm,
never two at once, in the gate's order:

    round 0: SINGLE, STRIPED   round 1: STRIPED, SINGLE   round 2: SINGLE, STRIPED

Imported unchanged from ``two_drive_stripe_gate.py``, not copied: the arms' flags and order,
the box stamp before and after every arm (power, commit, GPU, ASPM, ``soup_cli.__file__``,
every process's I/O bytes), the pre-arm checks and their 15-minute wait, the 2-s power and
1-s PDH sampler, and the void rule (battery power, a suspend, no ``step_plain`` point). Added:

- step counts per arm: SINGLE 21 and STRIPED 36 timed steps after one warm-up, with
  ``--skip-events``, so both arms read for about six minutes (the record's §1 says why);
- one more void condition: a process other than this driver whose read bytes grew by at least
  1 GB during the arm, from the after stamp's per-process capture;
- a 10-s GPU sampler (SM clock, temperature, power, throttle reasons), recorded, never judged;
- the inputs the rule reads, per arm: every timed step's wall time, the early (timed steps
  1-5) and late (the last 10) medians, each drive's mean read rate over those two windows,
  and 10-s bins of the drive counters;
- its own box log and driver JSON (``sustain_*``), so the gate's committed logs are untouched;
- ``--summarize``: recompute the rule's inputs and rows from the committed files, run nothing.

The two rows a single arm can settle stop the sequence (§2 rows 1 and 2), because no later arm
can turn a no-verdict outcome into a verdict. No torch import: the driver must not charge
commit or touch the GPU itself. Windows only (GetSystemPowerStatus, PDH, powercfg).

usage: two_drive_sustained.py [--plan rounds|build] [--settle-s 300] [--dry-run]
                              [--run-prefix sustain] [--out-dir DIR]
       two_drive_sustained.py --summarize [--run-prefix sustain] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import two_drive_stripe_gate as gate

# -- the arms (the gate's flags and order; only the step counts differ) -----------------------
#: Timed steps after WARMUP. About six minutes of reading each at the gate's sequence-4 step
#: times (gate §5.18): 21 x 17.44 s = 366 s, 36 x 9.97 s = 359 s.
TIMED_STEPS = {"SINGLE": 21, "STRIPED": 36}
WARMUP = 1
EARLY_STEPS = 5  # timed steps 1-5, i.e. process steps 2-6
LATE_STEPS = 10  # the last 10 timed steps
STRIPED_SHARDS = gate.ARM_FLAGS["STRIPED"][gate.ARM_FLAGS["STRIPED"].index("--shards") + 1]
STRIPE_ROOT = gate.ARM_FLAGS["STRIPED"][gate.ARM_FLAGS["STRIPED"].index("--stripe-dir") + 1]

# -- §2 of the record; each value's source line is quoted there ------------------------------
SINGLE_VALID = gate.SINGLE_VALID  # (15.0, 21.5) s, gate §2 row 1
SUSTAINED_MIN = 1.595  # lowest valid in-round burst speed-up on record (gate §5.9, round 0)
FLOOR_MIN = 1.20  # the gate's 14.0 s line, "1.2x the ... single-drive step" (gate §0)
ARM_DRIFT_FLAG = 1.080  # largest max/min step ratio in a valid gate arm (gate2_r0_striped)
DRIVE_DRIFT_FLAG = 1.10  # 10-s bin spread per drive over sequence 4's arms (gate §5.21)
FOREIGN_READ_MAX = 10**9  # bytes another process may read during one arm (gate max: 58.2 MB)
RESHARD_S = 30.0  # shard_seconds above this means the arm re-sharded (cache hits: <= 0.158 s)
READ_AHEAD = {"SINGLE": 2, "STRIPED": 3}
N_LAYERS = 80
LOADS_PER_STEP = (157, 2)
ARM_DRIVES = {"SINGLE": ("read_c",), "STRIPED": ("read_c", "read_d")}

# -- sampling and the wall-time estimate -------------------------------------------------------
GPU_POLL_S = 10.0
BIN_S = 10.0
GPU_FIELDS = "clocks.sm,temperature.gpu,power.draw,clocks_throttle_reasons.active"
#: For the printed estimate only: gate sequence-4 step medians and per-process overhead
#: (gate_driver.json wall_s minus 8 steps: 13-15 s).
EST_STEP_S = {"SINGLE": 17.44, "STRIPED": 9.97}
EST_OVERHEAD_S = 15.0
EST_BUILD_S = 276.0  # gate_build_striped wall_s, 275.9 s (gate §5.1)


# ==========================================================================
# files
# ==========================================================================
def use_out_dir(out_dir: str) -> None:
    """Point the gate's stamp and driver-JSON helpers at this probe's own files.

    ``gate.stamp`` and ``gate.load_driver_json`` / ``save_driver_json`` read these module
    globals when called, so the probe never appends to the gate's committed logs.
    """
    gate.OUT_DIR = out_dir
    gate.BOX_LOG = os.path.join(out_dir, "sustain_box_state.log")
    gate.DRIVER_JSON = os.path.join(out_dir, "sustain_driver.json")


def harness_out(out_dir: str, name: str) -> Tuple[str, str]:
    """(the ``--out`` argument, relative to the worktree when it can be; the absolute path)."""
    path = os.path.join(out_dir, f"{name}.json")
    try:
        rel = os.path.relpath(path, gate.WORKTREE)
    except ValueError:  # another drive
        return path, path
    return (path if rel.startswith("..") else rel.replace("\\", "/")), path


def harness_command(arm: str, label: str, out_arg: str, build: bool) -> List[str]:
    if build:
        steps = ["--steps", "1", "--warmup", "0"]
    else:
        steps = ["--steps", str(TIMED_STEPS[arm]), "--warmup", str(WARMUP)]
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
        "512",
        "--batch",
        "1",
        *steps,
        "--step",
        "--skip-events",
        *gate.ARM_FLAGS[arm],
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
    """Root 1's folder for the STRIPED cache, named by the code under test (slug + hash)."""
    code = (
        "from soup_cli.utils.stripe_roots import stripe_folder_name; "
        f"print(stripe_folder_name({STRIPED_SHARDS!r}))"
    )
    out = gate._run_quiet([sys.executable, "-c", code], env)
    return os.path.join(os.path.normpath(STRIPE_ROOT), out.strip()) if out else None


# ==========================================================================
# sampling
# ==========================================================================
def parse_gpu(text: Optional[str]) -> List[Any]:
    """``[sm_mhz, temp_c, power_w, throttle_reasons]``; ``None`` fields when unreadable."""
    try:
        parts = [part.strip() for part in (text or "").strip().splitlines()[0].split(",")]
        return [int(parts[0]), int(parts[1]), float(parts[2]), parts[3]]
    except (IndexError, ValueError):
        return [None, None, None, None]


class SustainSampler(gate.Sampler):
    """The gate's power and PDH sampler, plus ``nvidia-smi`` every GPU_POLL_S on a third thread.

    Recorded, never used to set a verdict: it is what tells a compute-side slowdown (the SM
    clock falling, a thermal throttle reason) from a drive-side one.
    """

    def __init__(self) -> None:
        super().__init__()
        self.gpu_rows: List[List[Any]] = []

    def start(self) -> None:
        self._threads.append(threading.Thread(target=self._gpu_loop, daemon=True))
        super().start()  # starts every thread in self._threads, this one included

    def _gpu_loop(self) -> None:
        query = [f"--query-gpu={GPU_FIELDS}", "--format=csv,noheader,nounits"]
        while True:
            self.gpu_rows.append([time.time(), *parse_gpu(gate._run_quiet(["nvidia-smi", *query]))])
            if self._stop.wait(GPU_POLL_S):
                return


# ==========================================================================
# the rule's inputs, per arm
# ==========================================================================
def _mean_between(pdh: List[List[Any]], key: str, start: float, end: float) -> Optional[float]:
    """Mean of the 1-s PDH samples of ``key`` stamped in ``(start, end]``."""
    values = [row[key] for stamp_s, row in pdh if start < stamp_s <= end and key in row]
    return statistics.fmean(values) if values else None


def bins_10s(pdh: List[List[Any]], started_unix: float) -> Dict[str, List[Any]]:
    """10-s bins of the drive counters, from the arm's process start: reads GB/s, writes MB/s."""
    grouped: Dict[int, List[Dict[str, float]]] = {}
    for stamp_s, row in pdh:
        grouped.setdefault(int((stamp_s - started_unix) // BIN_S), []).append(row)
    fields = (
        ("read_c", "read_c_gbs", 1e9),
        ("read_d", "read_d_gbs", 1e9),
        ("write_c", "write_c_mbs", 1e6),
        ("write_d", "write_d_mbs", 1e6),
    )
    out: Dict[str, List[Any]] = {"t0_s": []}
    for _key, name, _scale in fields:
        out[name] = []
    for index in sorted(grouped):
        out["t0_s"].append(index * BIN_S)
        for key, name, scale in fields:
            values = [row[key] for row in grouped[index] if key in row]
            out[name].append(round(statistics.fmean(values) / scale, 4) if values else None)
    return out


def arm_summary(
    arm: str, plain: Dict[str, Any], pdh: List[List[Any]], started_unix: float
) -> Dict[str, Any]:
    """Early and late medians, drift, each drive's window rates, per-step rates and 10-s bins."""
    records = plain["records"]
    times = [float(rec["step_s"]) for rec in records]
    out: Dict[str, Any] = {"step_s": times, "losses": [rec.get("loss") for rec in records]}
    if len(times) < EARLY_STEPS + LATE_STEPS:
        out["error"] = f"{len(times)} timed steps; the windows need {EARLY_STEPS + LATE_STEPS}"
        return out
    out["early_median_s"] = statistics.median(times[:EARLY_STEPS])
    out["late_median_s"] = statistics.median(times[-LATE_STEPS:])
    out["drift"] = out["late_median_s"] / out["early_median_s"]
    starts = [rec.get("started_unix") for rec in records]
    if not all(isinstance(value, (int, float)) for value in starts):
        out["windows_unix"] = None  # a harness without per-step start times
        return out
    late_first = len(records) - LATE_STEPS
    windows = {
        "early": (starts[0], starts[EARLY_STEPS - 1] + times[EARLY_STEPS - 1]),
        "late": (starts[late_first], starts[-1] + times[-1]),
    }
    out["windows_unix"] = windows
    for window, (start, end) in windows.items():
        for key in ("read_c", "read_d"):
            mean = _mean_between(pdh, key, start, end)
            out[f"{window}_{key}_gbs"] = mean / 1e9 if mean is not None else None
    out["drive_drift"] = {}
    for key in ARM_DRIVES[arm]:
        early, late = out[f"early_{key}_gbs"], out[f"late_{key}_gbs"]
        out["drive_drift"][key] = early / late if early and late else None
    out["per_step_read_gbs"] = [
        [_round(_mean_between(pdh, key, start, start + step), 1e9) for key in ("read_c", "read_d")]
        for start, step in zip(starts, times)
    ]
    out["bins_10s"] = bins_10s(pdh, started_unix)
    return out


def _round(value: Optional[float], scale: float) -> Optional[float]:
    return round(value / scale, 4) if value is not None else None


def arm_problems(
    arm: str, meta: Dict[str, Any], plain: Optional[Dict[str, Any]], build: bool
) -> List[str]:
    """Row 2's per-arm inputs: everything that says the arm did not run the path under test."""
    problems: List[str] = []
    for key in ("direct_io", "pinned"):
        if meta.get(key) is not True:
            problems.append(f"{key} {meta.get(key)!r}")
    if meta.get("source_class") != "AsyncDiskSource":
        problems.append(f"source {meta.get('source_class')!r}")
    if meta.get("read_ahead") != READ_AHEAD[arm]:
        problems.append(f"read_ahead {meta.get('read_ahead')!r}, not {READ_AHEAD[arm]}")
    roots = [os.path.normcase(os.path.normpath(root)) for root in meta.get("stripe_roots") or []]
    placement = list(meta.get("layer_roots") or [])
    if arm == "STRIPED":
        if roots != [os.path.normcase(os.path.normpath(STRIPE_ROOT))]:
            problems.append(f"stripe roots {meta.get('stripe_roots')!r}")
        if placement != [idx % 2 for idx in range(N_LAYERS)]:
            problems.append("layer placement is not i mod 2 over 80 layers")
    elif roots or placement:
        problems.append(f"SINGLE arm with stripe roots {meta.get('stripe_roots')!r}")
    if build:
        return problems
    shard_s = meta.get("shard_seconds")
    if not isinstance(shard_s, (int, float)) or shard_s > RESHARD_S:
        problems.append(f"shard_seconds {shard_s!r} (re-sharded instead of reading the cache)")
    if plain is not None:
        records = plain.get("records", [])
        if len(records) != TIMED_STEPS[arm]:
            problems.append(f"{len(records)} timed steps, not {TIMED_STEPS[arm]}")
        bad = [
            idx + 1
            for idx, rec in enumerate(records)
            if (rec.get("layer_loads"), rec.get("large_loads")) != LOADS_PER_STEP
        ]
        if bad:
            problems.append(f"timed steps {bad} did not load 157 + 2")
    return problems


def foreign_readers(after: Dict[str, Any], driver_pid: int) -> Optional[List[Dict[str, Any]]]:
    """Processes other than the driver that read >= FOREIGN_READ_MAX during the arm.

    ``None`` when the per-process capture did not run, so the check has no input (recorded,
    not void). The arm's own process has exited by the after stamp and is not in the list.
    """
    top = after.get("top_readers")
    if top is None:
        return None
    return [
        row for row in top if row["pid"] != driver_pid and row["read_delta"] >= FOREIGN_READ_MAX
    ]


# ==========================================================================
# one arm
# ==========================================================================
def run_arm(
    name: str, arm: str, label: str, build: bool, dry_run: bool, out_dir: str
) -> Dict[str, Any]:
    env = gate.arm_env()
    out_arg, json_path = harness_out(out_dir, name)
    log_path = os.path.join(out_dir, f"{name}.log")
    cmd = harness_command(arm, label, out_arg, build)
    entry: Dict[str, Any] = {"run": name, "arm": arm, "label": label, "command": cmd}
    entry["before"] = gate.stamp(name, "before", env)
    if dry_run:
        print("DRY RUN:", subprocess.list2cmdline(cmd), flush=True)
        entry["dry_run"] = True
        return entry
    sampler = SustainSampler()
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
    entry["exit_code"] = exit_code
    entry["started_unix"] = started
    entry["ended_unix"] = ended
    entry["wall_s"] = ended - started
    entry["during"] = sampler.summary()
    entry["during_samples"] = {
        "power": [list(row) for row in sampler.power_rows],
        "pdh": [[stamp_s, row] for stamp_s, row in sampler.pdh_rows],
        "gpu": sampler.gpu_rows,
    }
    entry["after"] = gate.stamp(name, "after", env, before=entry["before"])
    plain, meta = gate.step_plain(json_path)
    entry["step_plain_mean_s"] = plain["step_s_mean"] if plain else None
    for key in ("direct_io", "pinned", "read_ahead", "source_class", "shard_seconds"):
        entry[key] = meta.get(key)
    entry["stripe_roots"] = meta.get("stripe_roots")
    entry["layer_roots_head"] = (meta.get("layer_roots") or [])[:8]
    entry["problems"] = arm_problems(arm, meta, plain, build) if plain is not None else []
    if plain is not None and not build:
        entry["summary"] = arm_summary(arm, plain, entry["during_samples"]["pdh"], started)
    # The gate's void rule, then the foreign reader.
    sample_times = [row[0] for row in sampler.power_rows] + [ended]
    max_gap = max((b - a for a, b in zip(sample_times, sample_times[1:])), default=None)
    entry["during"]["max_power_sample_gap_s"] = max_gap
    void_reasons = []
    if entry["during"]["ac_min"] != 1 or entry["after"]["ac_line_status"] != 1:
        void_reasons.append(f"AC values seen {entry['during']['ac_values_seen']}")
    if max_gap is None or max_gap > gate.SUSPEND_GAP_S:
        void_reasons.append(f"suspended: {max_gap} s between power samples")
    if plain is None:
        void_reasons.append(f"no step_plain point (exit code {exit_code})")
    heavy = foreign_readers(entry["after"], os.getpid())
    entry["foreign_readers"] = heavy
    if heavy:
        void_reasons.append(
            "foreign reader: "
            + ", ".join(
                f"{row['name']}({row['pid']}) {row['read_delta'] / 1e9:.2f} GB" for row in heavy
            )
        )
    entry["void_reasons"] = void_reasons
    if void_reasons:
        for path in (json_path, log_path):
            if os.path.exists(path):
                target = gate._void_name(path)
                os.replace(path, target)
                entry.setdefault("void_files", []).append(os.path.basename(target))
    summary = entry.get("summary") or {}
    print(
        f"{name}: exit {exit_code}, wall {entry['wall_s']:.1f} s, early median "
        f"{summary.get('early_median_s')}, late median {summary.get('late_median_s')}, "
        f"problems {entry['problems'] or 'none'}, void {void_reasons or 'no'}",
        flush=True,
    )
    return entry


def stop_row(entry: Dict[str, Any]) -> Optional[str]:
    """The two rows of §2 one arm can settle, in the rule's order; None when neither fires."""
    summary = entry.get("summary") or {}
    low, high = SINGLE_VALID
    early = summary.get("early_median_s")
    if entry["arm"] == "SINGLE" and early is not None and not low <= early <= high:
        return (
            f"no verdict (row 1): SINGLE arm {entry['run']} early median {early!r} s is "
            f"outside {low}-{high} s"
        )
    if entry.get("problems") or summary.get("error"):
        reasons = list(entry.get("problems") or []) + (
            [summary["error"]] if "error" in summary else []
        )
        return f"no verdict (row 2): {entry['run']}: {'; '.join(reasons)}"
    return None


# ==========================================================================
# the rule, from the committed files
# ==========================================================================
def _ratio(top: Optional[float], bottom: Optional[float]) -> Optional[float]:
    return top / bottom if top is not None and bottom else None


def _fmt(value: Optional[float], digits: int = 3) -> str:
    return "none" if value is None else f"{value:.{digits}f}"


def collect(out_dir: str, prefix: str) -> Tuple[Dict[Tuple[int, str], Dict[str, Any]], str]:
    """The valid arm of every round position, recomputed from its JSON and its PDH samples.

    Returns ``({(round, arm): arm_record}, outcome)``. The outcome is that of the last
    ``rounds`` invocation with this prefix, or ``"no rounds invocation"``.
    """
    use_out_dir(out_dir)
    payload = gate.load_driver_json()
    invocations = [
        inv
        for inv in payload.get("invocations", [])
        if inv.get("plan") == "rounds"
        and inv.get("run_prefix") == prefix
        and not inv.get("dry_run")
    ]
    outcome = (
        invocations[-1].get("outcome", "not finished") if invocations else "no rounds invocation"
    )
    arms: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for rnd, arm in gate.ROUND_PLAN:
        name = f"{prefix}_r{rnd}_{arm.lower()}"
        valid = [
            run
            for run in payload.get("runs", [])
            if run.get("run") == name
            and run.get("exit_code") is not None
            and not run.get("void_reasons")
            and not run.get("dry_run")
        ]
        if not valid:
            continue
        run = valid[-1]
        plain, meta = gate.step_plain(os.path.join(out_dir, f"{name}.json"))
        record: Dict[str, Any] = {"run": name, "problems": [], "summary": None}
        if plain is None:
            record["problems"] = [f"{name}.json has no step_plain point"]
        else:
            record["problems"] = arm_problems(arm, meta, plain, build=False)
            pdh = run.get("during_samples", {}).get("pdh", [])
            record["summary"] = arm_summary(arm, plain, pdh, run["started_unix"])
            if "error" in record["summary"]:
                record["problems"].append(record["summary"]["error"])
        arms[(rnd, arm)] = record
    return arms, outcome


def verdict(arms: Dict[Tuple[int, str], Dict[str, Any]], outcome: str) -> List[Tuple[str, str]]:
    """§2 applied row by row: ``[(row, result), ...]``, the last entry the verdict."""
    rounds = sorted({rnd for rnd, _arm in gate.ROUND_PLAN})
    low, high = SINGLE_VALID
    out: List[Tuple[str, str]] = []
    early = {
        rnd: arms[(rnd, "SINGLE")]["summary"].get("early_median_s")
        for rnd in rounds
        if (rnd, "SINGLE") in arms and arms[(rnd, "SINGLE")]["summary"]
    }
    outside = {rnd: v for rnd, v in early.items() if v is not None and not low <= v <= high}
    out.append(
        (
            "1 SINGLE early median outside 15.0-21.5 s",
            f"fires: {outside}" if outside else "does not fire",
        )
    )
    if outside:
        return out + [
            ("VERDICT", "NO VERDICT (row 1): the session does not reproduce the known step")
        ]
    problems = {record["run"]: record["problems"] for record in arms.values() if record["problems"]}
    missing = [f"r{rnd} {arm}" for rnd, arm in gate.ROUND_PLAN if (rnd, arm) not in arms]
    row2 = []
    if problems:
        row2.append(f"off the path under test: {problems}")
    if missing:
        row2.append(f"no valid arm: {missing}")
    if outcome != "complete":
        row2.append(f"protocol: {outcome}")
    out.append(
        (
            "2 off the path under test, or the protocol stopped",
            "fires: " + "; ".join(row2) if row2 else "does not fire",
        )
    )
    if row2:
        return out + [
            ("VERDICT", "NO VERDICT (row 2): the arms did not measure what the rule reads")
        ]
    late = {
        rnd: _ratio(
            arms[(rnd, "SINGLE")]["summary"]["late_median_s"],
            arms[(rnd, "STRIPED")]["summary"]["late_median_s"],
        )
        for rnd in rounds
    }
    text = ", ".join(f"r{rnd} {_fmt(value)}" for rnd, value in late.items())
    if all(value is not None and value >= SUSTAINED_MIN for value in late.values()):
        out.append((f"3 u_late >= {SUSTAINED_MIN} in every round", f"met: {text}"))
        return out + [("VERDICT", "SUSTAINED")]
    out.append((f"3 u_late >= {SUSTAINED_MIN} in every round", f"not met: {text}"))
    if all(value is not None and value >= FLOOR_MIN for value in late.values()):
        out.append((f"4 u_late >= {FLOOR_MIN} in every round", f"met: {text}"))
        return out + [("VERDICT", "DEGRADES")]
    out.append((f"4 u_late >= {FLOOR_MIN} in every round", f"not met: {text}"))
    return out + [("VERDICT", "DEGRADES BELOW THE GATE'S FLOOR")]


def summarize(out_dir: str, prefix: str) -> int:
    arms, outcome = collect(out_dir, prefix)
    print(f"sequence '{prefix}' in {out_dir}: outcome {outcome}")
    if not arms:
        print("no valid round arm on record")
    for (rnd, arm), record in sorted(arms.items()):
        summary = record["summary"] or {}
        flags = []
        if (summary.get("drift") or 0) > ARM_DRIFT_FLAG:
            flags.append(f"arm drift > {ARM_DRIFT_FLAG}")
        for key, ratio in (summary.get("drive_drift") or {}).items():
            if ratio is not None and ratio > DRIVE_DRIFT_FLAG:
                flags.append(f"{key[-1].upper()}: slowed > {DRIVE_DRIFT_FLAG}x")
        print(
            f"  r{rnd} {arm:<7} early {_fmt(summary.get('early_median_s'))} s  late "
            f"{_fmt(summary.get('late_median_s'))} s  d {_fmt(summary.get('drift'))}  "
            f"C: {_fmt(summary.get('early_read_c_gbs'))}->{_fmt(summary.get('late_read_c_gbs'))} "
            f"D: {_fmt(summary.get('early_read_d_gbs'))}->{_fmt(summary.get('late_read_d_gbs'))} "
            f"GB/s  problems {record['problems'] or 'none'}  {', '.join(flags)}"
        )
    for rnd in sorted({rnd for rnd, _arm in gate.ROUND_PLAN}):
        single, striped = arms.get((rnd, "SINGLE")), arms.get((rnd, "STRIPED"))
        if not (single and striped and single["summary"] and striped["summary"]):
            continue
        u_early = _ratio(
            single["summary"].get("early_median_s"), striped["summary"].get("early_median_s")
        )
        u_late = _ratio(
            single["summary"].get("late_median_s"), striped["summary"].get("late_median_s")
        )
        print(f"  round {rnd}: u_early {_fmt(u_early)}  u_late {_fmt(u_late)}")
    for row, result in verdict(arms, outcome):
        print(f"  row {row}: {result}")
    return 0


# ==========================================================================
# main
# ==========================================================================
def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--plan", choices=("rounds", "build"), default="rounds")
    parser.add_argument(
        "--settle-s",
        type=float,
        default=0.0,
        help="seconds to wait before the first arm (the record asks 300 after the build run)",
    )
    parser.add_argument("--dry-run", action="store_true", help="stamp and print, run nothing")
    parser.add_argument(
        "--summarize", action="store_true", help="apply §2 to the committed files, run nothing"
    )
    parser.add_argument(
        "--run-prefix",
        default="sustain",
        help="file-name prefix of the arms, so a complete re-run keeps an earlier sequence's "
        "files (default: sustain)",
    )
    parser.add_argument(
        "--out-dir",
        default=gate.OUT_DIR,
        help="where the arms' files, the box log and the driver JSON go (default: the record's "
        "results folder; point a test dry run elsewhere)",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"sustain[a-z0-9]{0,8}", args.run_prefix):
        parser.error("--run-prefix must be 'sustain' plus at most 8 lowercase letters or digits")
    return args


def estimate_s(plan: List[Tuple[str, str, Optional[int]]], settle_s: float) -> float:
    total = settle_s
    for _name, arm, rnd in plan:
        if rnd is None:
            total += EST_BUILD_S
        else:
            total += (TIMED_STEPS[arm] + WARMUP) * EST_STEP_S[arm] + EST_OVERHEAD_S
    return total


def main() -> int:
    args = parse()
    if sys.platform != "win32":
        print("SKIP: this driver is Windows-only (GetSystemPowerStatus, PDH, powercfg)")
        return 0
    out_dir = os.path.abspath(args.out_dir)
    if args.summarize:
        return summarize(out_dir, args.run_prefix)
    use_out_dir(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    payload = gate.load_driver_json()
    if args.plan == "build":
        plan: List[Tuple[str, str, Optional[int]]] = [
            (f"{args.run_prefix}_build_striped", "STRIPED", None)
        ]
    else:
        plan = [
            (f"{args.run_prefix}_r{rnd}_{arm.lower()}", arm, rnd) for rnd, arm in gate.ROUND_PLAN
        ]
    invocation = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "plan": args.plan,
        "settle_s": args.settle_s,
        "python": sys.executable,
        "pythonpath": gate.SRC,
        "dry_run": args.dry_run,
        "run_prefix": args.run_prefix,
        "worktree": worktree_state(),
        "striped_shards": STRIPED_SHARDS,
        "stripe_folder": stripe_folder(gate.arm_env()),
        "timed_steps": TIMED_STEPS,
        "warmup": WARMUP,
        "estimated_s": estimate_s(plan, args.settle_s),
    }
    payload["invocations"].append(invocation)
    gate.save_driver_json(payload)
    print(
        f"plan {args.plan}: {[name for name, _arm, _rnd in plan]}; worktree "
        f"{invocation['worktree']}; STRIPED root 1 folder {invocation['stripe_folder']}; "
        f"estimated {invocation['estimated_s'] / 60:.1f} min without voids or waits",
        flush=True,
    )
    if args.settle_s > 0:
        if args.dry_run:
            print(f"DRY RUN: would settle {args.settle_s:.0f} s before the first arm", flush=True)
        else:
            print(f"settling {args.settle_s:.0f} s before the first arm", flush=True)
            time.sleep(args.settle_s)
    outcome = "complete"
    for position, (name, arm, rnd) in enumerate(plan):
        label = (
            f"R4 sustained build arm {arm}"
            if rnd is None
            else f"R4 sustained round {rnd} arm {arm}"
        )
        if args.dry_run:
            ok, why = gate.precheck_ok()
            print(f"DRY RUN: pre-arm check {'met' if ok else 'NOT met: ' + why}", flush=True)
            dry = run_arm(name, arm, label, build=rnd is None, dry_run=True, out_dir=out_dir)
            dry["position"] = position
            payload["runs"].append(dry)
            gate.save_driver_json(payload)
            continue
        entry: Optional[Dict[str, Any]] = None
        for attempt in (1, 2):
            ok, notes = gate.wait_for_precheck()
            if not ok:
                outcome = f"stopped: pre-arm checks not met within 15 min before {name}"
                payload["runs"].append({"run": name, "arm": arm, "precheck_notes": notes})
                break
            entry = run_arm(name, arm, label, build=rnd is None, dry_run=False, out_dir=out_dir)
            entry["attempt"] = attempt
            entry["precheck_notes"] = notes
            entry["position"] = position
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
        if rnd is not None:
            row = stop_row(entry)
            if row is not None:
                outcome = f"stopped: {row}"
                break
    # A position counts as run only when a non-void attempt exists (the gate's list counted a
    # void attempt as run; gate §5.7 item 1).
    ran = {
        run.get("run")
        for run in payload["runs"]
        if run.get("exit_code") is not None and not run.get("void_reasons")
    }
    invocation["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    invocation["outcome"] = outcome if not args.dry_run else "dry run"
    invocation["not_run"] = [] if args.dry_run else [n for n, _a, _r in plan if n not in ran]
    gate.save_driver_json(payload)
    print(f"OUTCOME: {invocation['outcome']}; not run: {invocation['not_run']}", flush=True)
    if args.plan == "rounds" and not args.dry_run:
        summarize(out_dir, args.run_prefix)
    if outcome == "complete":
        return 0
    return 5 if "pre-arm" in outcome else 4


if __name__ == "__main__":
    sys.exit(main())
