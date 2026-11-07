#!/usr/bin/env python3
"""Plan and run the whole ds_compare series (protocol sections 2-6 and 8). Use ``run_all.sh``.

One command repeats the series on an H100: ``bash run_all.sh --all``. ``--dry-run`` prints
every command, writes every soup.yaml and runs nothing. ``--only S1,D1`` restricts the arms.

Order of execution (value first; the budget rule drops from the other end):

    warm-up (untimed S1, D1)
    Shape A, 80 GB:  3 repeats, interleaved by repeat (every arm repeat 1, then 2, then 3):
                     S0 S1 S2 S3 S4 S5 S6 S7 D1 D2 D3 D4 R1 R2  (reversed on even repeats)
    SB derivation    from the Shape A medians (rule below) -> sb_overrides.json -> SB x 3
    variants         D1PF / D2PF x 3 when that arm's GPU was idle more than 50% (rule, not
                     result) -> d_best.json (fastest DeepSpeed arm of Shape A)
    Shapes B and C   rep 1: SB S1 D-best D1 R1
    24 GB cap        SB D1 D2 D3 R1, 2 repeats
    Shapes B and C   rep 2
    16 GB cap        1 repeat
    S8               informational, 3 repeats

SB derivation: every single change S2..S7 that is faster than S1 by more than S1's spread
(median tok/s, Shape A, 80 GB) is combined; S2 and S3 both set ``stream_recompute`` and cannot
be combined, the faster one is taken. No qualifying change means SB equals S1.

Budget rule (protocol section 8, driven by ``DEADLINE_EPOCH`` and optionally
``BUDGET_USD`` with ``RENTAL_RATE_USD_PER_HOUR``, alias ``VAST_RATE_USD_PER_HOUR``): before
every run the estimated time of all pending runs is compared with the time left; when it does
not fit, whole groups are dropped in this order: S8, the prefetch variants, the 16 GB cap,
Shape B repeat 2, Shape C repeat 2, the 24 GB cap. Shape A's S1 and D1, and repeat 1 of
Shapes B and C, are never dropped.

Void rules (section 5) are applied per run; a void run is repeated once. Every attempt writes
its json at once: ``results/<shape>/<budget>/<arm>/rep<k>.json`` (+ ``.gpu.csv``,
``.rss.csv``, ``.log``); ``results/DONE`` is written last.

Parts. ``--part part1|part2|part3`` runs the series in three rented instances, each with its own
deadline. The plan of a part is the slots of the full plan whose phase belongs to it (the three
sets are disjoint and their union is the full plan):

    part1  warmup, A, SB, variants, S8   Shape A at 80 GB: S0..S8, SB, D1..D4, R1, R2, the
                                         prefetch variants (rule), 3 repeats
    part2  B1, C1, B2, C2                Shapes B and C at 80 GB (SB, S1, D-best, D1, R1)
    part3  cap24, cap16                  Shape A under the 24 and 16 GB caps

Part 2 and part 3 read the run jsons of part 1 from the results directory they are given and
derive SB and D-best from them with the same functions the full series uses; they refuse to
start without part 1's results. The budget rule of a part drops only the groups inside the part,
in the protocol's order; ``core`` is never dropped. Files that every part writes carry the part
in their name (``session_part1.json``, ``dropped_part1.json``, ``DONE_part1``) so that a part
never overwrites an earlier part's record.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gates as gates_module  # noqa: E402
import soup_arm  # noqa: E402
from ds_common import (  # noqa: E402
    HERE,
    MODEL_ID,
    MODEL_REVISION,
    SHAPES,
    atomic_write_json,
    median,
    read_json,
    rows_path,
)
from sampler import RunSampler, summarize  # noqa: E402

# ---------------------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------------------

HF_ARMS: dict[str, dict[str, Any]] = {
    "D1": {"ds": "ds_d1_z3_param_offload.json", "gc": False},
    "D2": {"ds": "ds_d2_z3_param_opt_offload.json", "gc": False},
    "D3": {"ds": "ds_d3_z2_opt_offload.json", "gc": False},
    "D4": {"ds": "ds_d4_z3_no_offload.json", "gc": False},
    "D1PF": {"ds": "ds_d1_prefetch5e8.json", "gc": False},
    "D2PF": {"ds": "ds_d2_prefetch5e8.json", "gc": False},
    "R1": {"ds": None, "gc": False},
    "R2": {"ds": None, "gc": True},
}
DS_PLAIN = ("D1", "D2", "D3", "D4")
SHAPE_A_ARMS = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "D1", "D2", "D3", "D4", "R1", "R2"]
CAP_ARMS = ["SB", "D1", "D2", "D3", "R1"]
BC_ARMS = ["SB", "S1", "DBEST", "D1", "R1"]
SB_CANDIDATES = ("S2", "S3", "S4", "S5", "S6", "S7")

#: Drop order of protocol section 8. Group "core" is never dropped.
DROP_ORDER = ("S8", "prefetch", "cap16", "B_rep2", "C_rep2", "cap24")

#: The series in parts (one rented instance each): the phases of the full plan that belong to a
#: part. Every phase is in exactly one part.
PART_PHASES: dict[str, tuple[str, ...]] = {
    "part1": ("warmup", "A", "SB", "variants", "S8"),
    "part2": ("B1", "C1", "B2", "C2"),
    "part3": ("cap24", "cap16"),
}
PARTS = tuple(PART_PHASES)
_PHASE_PART = {phase: part for part, phases in PART_PHASES.items() for phase in phases}
#: DeepSpeed arms that D-best is chosen from (Shape A, 80 GB).
D_BEST_CANDIDATES = DS_PLAIN + ("D1PF", "D2PF")

# Minutes per run, ESTIMATED from the STEP 3 / STEP 4 records (H100:1051-1056, 1105-1111):
# bf16 streaming 129 s and DeepSpeed ZeRO-3 param offload 378 s for the 256-step shape A run.
# Shapes B and C scale with the tokens per step. Everything else is a guess and is replaced by
# the measured median wall time of that arm once a run has finished.
EST_TRAIN_MIN: dict[tuple[str, str], float] = {
    ("stream", "A"): 2.2,
    ("stream", "B"): 1.9,
    ("stream", "C"): 2.7,
    ("D1", "A"): 6.5,
    ("D1", "B"): 3.8,
    ("D1", "C"): 4.3,
    ("D2", "A"): 7.0,
    ("D2", "B"): 4.0,
    ("D2", "C"): 4.5,
    ("D3", "A"): 1.5,
    ("D3", "B"): 1.0,
    ("D3", "C"): 0.9,
    ("D4", "A"): 2.5,
    ("D4", "B"): 1.4,
    ("D4", "C"): 1.6,
    ("R1", "A"): 1.0,
    ("R1", "B"): 0.7,
    ("R1", "C"): 0.8,
    ("R2", "A"): 1.3,
    ("R2", "B"): 0.9,
    ("R2", "C"): 1.0,
}
EST_STARTUP_MIN = {"stream": 1.0, "ds": 1.5, "resident": 0.7}
SETUP_HOURS = 1.0
HOURS_LIMIT = 8.0
FINISH_MARGIN_S = 480.0
SAFETY_FACTOR = 1.05


def arm_family(arm: str) -> str:
    """Key of the time-estimate table: streaming arms share one, DeepSpeed variants follow D1/D2."""
    if arm in HF_ARMS:
        return {"D1PF": "D1", "D2PF": "D2"}.get(arm, arm)
    return "D1" if arm == "DBEST" else "stream"


def startup_class(arm: str) -> str:
    if arm_family(arm) == "stream":
        return "stream"
    return "resident" if arm in ("R1", "R2") else "ds"


def estimate_minutes(arm: str, shape: str) -> float:
    return EST_TRAIN_MIN[(arm_family(arm), shape)] + EST_STARTUP_MIN[startup_class(arm)]


# ---------------------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------------------


@dataclass
class Slot:
    index: int
    phase: str
    group: str
    arm: str
    shape: str
    budget: int
    rep: int
    warmup: bool = False
    est_min: float = 0.0
    conditional: str | None = None
    done: bool = False
    skipped: str | None = None
    resolved_arm: str | None = None
    note: str = field(default="")
    part: str = field(default="")

    @property
    def label(self) -> str:
        arm = self.resolved_arm or self.arm
        tag = "warmup" if self.warmup else f"rep{self.rep}"
        return f"{self.shape}/{self.budget}GB/{arm}/{tag}"


def _alternate(arms: list[str], rep: int) -> list[str]:
    return list(arms) if rep % 2 else list(reversed(arms))


def build_plan(
    only: set[str] | None = None,
    phases: set[str] | None = None,
    part: str | None = None,
) -> list[Slot]:
    """The ordered slots of the series, restricted by arm, by phase and/or to one part.

    ``Slot.index`` is the position in the FULL plan whatever the restriction, so a slot has the
    same number in every dry run.
    """
    if part is not None and part not in PART_PHASES:
        raise ValueError(f"unknown part {part!r}: choose from {', '.join(PARTS)}")
    slots: list[Slot] = []
    position = 0

    def add(phase: str, group: str, arm: str, shape: str, budget: int, rep: int, **kw: Any) -> None:
        nonlocal position
        index = position
        position += 1
        if only is not None and arm not in only:
            return
        if phases is not None and phase not in phases:
            return
        if part is not None and phase not in PART_PHASES[part]:
            return
        if arm == "SB":
            kw["conditional"] = "sb_derivation"
        elif arm == "DBEST":
            kw["conditional"] = "d_best"
        slots.append(
            Slot(
                index,
                phase,
                group,
                arm,
                shape,
                budget,
                rep,
                est_min=estimate_minutes(arm, shape),
                part=_PHASE_PART[phase],
                **kw,
            )
        )

    for arm in ("S1", "D1"):
        add("warmup", "core", arm, "A", 80, 0, warmup=True)
    for rep in (1, 2, 3):
        for arm in _alternate(SHAPE_A_ARMS, rep):
            add("A", "core", arm, "A", 80, rep)
    for rep in (1, 2, 3):
        add("SB", "core", "SB", "A", 80, rep)
    for arm in ("D1PF", "D2PF"):
        for rep in (1, 2, 3):
            add("variants", "prefetch", arm, "A", 80, rep, conditional="idle_rule")
    for shape in ("B", "C"):
        for arm in _alternate(BC_ARMS, 1):
            add(f"{shape}1", "core", arm, shape, 80, 1)
    for rep in (1, 2):
        for arm in _alternate(CAP_ARMS, rep):
            add("cap24", "cap24", arm, "A", 24, rep)
    for shape in ("B", "C"):
        for arm in _alternate(BC_ARMS, 2):
            add(f"{shape}2", f"{shape}_rep2", arm, shape, 80, 2)
    for arm in CAP_ARMS:
        add("cap16", "cap16", arm, "A", 16, 1)
    for rep in (1, 2, 3):
        add("S8", "S8", "S8", "A", 80, rep)
    return slots


# ---------------------------------------------------------------------------------------
# Parts: the estimate of a plan, and what a part needs from the earlier ones
# ---------------------------------------------------------------------------------------

#: A part's own set-up (model download, DeepSpeed build with cpu_adam, dataset check) and the
#: time kept free at the end of its rental. Used only for the suggested ``--max-hours``.
SETUP_MIN_PER_PART = 35.0
MARGIN_MIN_PER_PART = 15.0


def measured_wall_minutes(runs: list[dict[str, Any]]) -> dict[tuple[str, str], float]:
    """Median wall minutes of the finished runs of each (arm, shape)."""
    walls: dict[tuple[str, str], list[float]] = {}
    for run in runs:
        if run.get("wall_seconds"):
            walls.setdefault((run["arm"], run["shape"]), []).append(run["wall_seconds"] / 60.0)
    return {key: median(values) for key, values in walls.items()}


def estimate_plan(
    plan: list[Slot], table: dict[tuple[str, str], float] | None = None
) -> dict[str, Any]:
    """Run count and estimated GPU minutes of a plan, with the estimator of the budget rule.

    ``table`` (arm, shape) -> measured median wall minutes replaces the built-in estimate of
    that arm and shape, as it does while a series runs.
    """

    def minutes(slot: Slot) -> float:
        measured = (table or {}).get((slot.resolved_arm or slot.arm, slot.shape))
        return measured if measured is not None else slot.est_min

    conditional = [slot for slot in plan if slot.conditional in ("idle_rule", "d_best")]
    by_group: dict[str, float] = {}
    for slot in plan:
        by_group[slot.group] = by_group.get(slot.group, 0.0) + minutes(slot)
    total = sum(by_group.values())
    conditional_minutes = sum(minutes(slot) for slot in conditional)
    return {
        "runs": len(plan),
        "warmups": sum(1 for slot in plan if slot.warmup),
        "conditional_runs": len(conditional),
        "minutes": total,
        "minutes_conditional": conditional_minutes,
        "minutes_unconditional": total - conditional_minutes,
        "by_group": by_group,
    }


def suggested_max_hours(gpu_minutes: float) -> float:
    """``--max-hours`` for a part: set-up + runs + margin, rounded up to the next 0.25 h."""
    hours = (SETUP_MIN_PER_PART + gpu_minutes + MARGIN_MIN_PER_PART) / 60.0
    return math.ceil(hours * 4.0 - 1e-9) / 4.0


def part_prerequisites(
    part: str, results: str | Path, only: set[str] | None = None
) -> list[str]:
    """What ``part`` still needs from the earlier parts: a list of problems (empty: ready).

    Part 1 needs nothing. Part 2 and part 3 derive SB from part 1's Shape A runs at 80 GB (a
    valid S1 run is the base of the rule) and part 2 also derives D-best (a valid run of at
    least one DeepSpeed arm). The evidence is the run jsons; a stray ``sb_overrides.json``
    without them is not accepted.
    """
    if part not in PART_PHASES:
        raise ValueError(f"unknown part {part!r}: choose from {', '.join(PARTS)}")
    if part == "part1":
        return []
    plan = build_plan(only, None, part)
    needs_sb = any(slot.arm == "SB" for slot in plan)
    needs_d_best = any(slot.arm == "DBEST" for slot in plan)
    if not (needs_sb or needs_d_best):
        return []
    runs = gates_module.load_runs(results)
    fix = (
        f"ship part 1's results folder with this part's payload "
        f"(make_payload.sh {part} <out_dir> <part 1 results dir>)"
    )
    if not runs:
        return [
            f"{part} needs the results of part1 and found none: no <shape>/<budget>/<arm>/"
            f"rep<k>.json under {results}; {fix}"
        ]
    problems: list[str] = []
    if needs_sb and gates_module.arm_stats(runs, "A", 80, "S1")["status"] != "measured":
        problems.append(
            f"{part} needs part1's Shape A runs at 80 GB: arm S1 has no valid run under "
            f"{results} (SB is derived from it); {fix}"
        )
    if (
        needs_d_best
        and gates_module.best_deepspeed(runs, "A", 80, D_BEST_CANDIDATES) is None
    ):
        problems.append(
            f"{part} needs part1's Shape A runs at 80 GB: none of {', '.join(D_BEST_CANDIDATES)} "
            f"has a valid run under {results} (D-best is derived from them); {fix}"
        )
    return problems


# ---------------------------------------------------------------------------------------
# Derivations (pure functions of results; tested without a GPU)
# ---------------------------------------------------------------------------------------


def derive_sb_overrides(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """The SB rule of protocol section 2."""
    s1 = gates_module.arm_stats(runs, "A", 80, "S1")
    out: dict[str, Any] = {
        "rule": "every single change S2..S7 faster than S1 by more than S1's spread, combined",
        "selected": [],
        "training": {},
        "env": {},
        "arm_changes_pct": {},
    }
    if s1["status"] != "measured":
        out["note"] = "S1 not measured: SB equals S1"
        return out
    base, s1_spread = s1["tok_per_s_median"], s1["spread"]
    out.update({"s1_median_tok_per_s": base, "s1_spread": s1_spread})
    qualifying: list[tuple[str, float]] = []
    for arm in SB_CANDIDATES:
        stats = gates_module.arm_stats(runs, "A", 80, arm)
        if stats["status"] != "measured":
            out["arm_changes_pct"][arm] = None
            continue
        change = (stats["tok_per_s_median"] - base) / base
        out["arm_changes_pct"][arm] = change * 100.0
        if change > s1_spread:
            qualifying.append((arm, stats["tok_per_s_median"]))
    recompute = [item for item in qualifying if item[0] in ("S2", "S3")]
    if len(recompute) == 2:
        loser = min(recompute, key=lambda item: item[1])[0]
        qualifying = [item for item in qualifying if item[0] != loser]
        out["recompute_conflict_dropped"] = loser
    for arm, _ in qualifying:
        change = soup_arm.ARM_CHANGES[arm]
        out["selected"].append(arm)
        out["training"].update(change.get("training", {}))
        out["env"].update(change.get("env", {}))
    return out


def idle_rule(runs: list[dict[str, Any]], arm: str) -> bool:
    """True when the arm's GPU was idle more than 50% (mean utilization below 50%)."""
    values = []
    for run in runs:
        if (
            run["shape"] == "A"
            and run["budget_gb"] == 80
            and run["arm"] == arm
            and run.get("tok_per_s")
        ):
            sampler = run.get("sampler") or {}
            window = sampler.get("train_window") or sampler.get("whole_run") or {}
            if window.get("gpu_util_mean_pct") is not None:
                values.append(window["gpu_util_mean_pct"])
    return bool(values) and median(values) < 50.0


def void_reasons(
    run: dict[str, Any],
    budget: int,
    prior_losses: list[float],
    session_clocks: list[float],
    preflight_busy: bool,
) -> list[str]:
    """Protocol section 5, plus two operational checks (cap not applied, trace incomplete)."""
    reasons: list[str] = []
    status = run.get("status")
    sampler = run.get("sampler") or {}
    foreign = (sampler.get("foreign_gpu_processes") or {}).get("foreign_pids")
    if preflight_busy:
        reasons.append("another process was on the GPU before the run")
    if foreign:
        reasons.append(f"another process on the GPU during the run: {foreign}")
    window = (sampler.get("train_window") or {}).get("sm_clock_busy_median_mhz")
    if window and len(session_clocks) >= 3:
        reference = statistics.median(session_clocks)
        if window < reference * (1 - gates_module.CLOCK_TOLERANCE):
            reasons.append(
                f"busy SM clock {window:.0f} MHz is more than 5% below the session median "
                f"{reference:.0f} MHz"
            )
    if status == "failed":
        reasons.append(f"non-zero exit ({run.get('exit_code')}) that is not an out-of-memory")
    if status == "oom" and budget >= 80:
        reasons.append("out of memory without a cap")
    if status == "ok" and run.get("trace_complete") is False:
        reasons.append("the recorder saw no train-end event")
    if budget < 80 and not run.get("cap_applied"):
        reasons.append("the VRAM cap was not applied (no cap line in the log)")
    reserved = run.get("max_memory_reserved_bytes")
    if budget < 80 and reserved and reserved > budget * 1024**3 * 1.02:
        reasons.append(
            f"peak reserved memory {reserved / 1024**3:.1f} GiB exceeds the {budget} GB cap: "
            "the cap was not effective"
        )
    final_loss = run.get("final_loss")
    if status == "ok" and final_loss is not None and prior_losses:
        reference_loss = statistics.median(prior_losses)
        if abs(final_loss - reference_loss) > 0.20 * max(abs(reference_loss), 1e-12):
            reasons.append(
                f"final loss {final_loss:.5g} differs from the arm's other repeats "
                f"({reference_loss:.5g}) by more than 20%"
            )
    return reasons


# ---------------------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------------------


def env_float(name: str) -> float | None:
    value = os.environ.get(name)
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


class Session:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.results = Path(args.results).resolve()
        self.model = args.model
        self.deadline = env_float("DEADLINE_EPOCH")
        self.rate = env_float("RENTAL_RATE_USD_PER_HOUR") or env_float("VAST_RATE_USD_PER_HOUR")
        self.budget_usd = env_float("BUDGET_USD")
        self.setup_seconds = env_float("SETUP_SECONDS") or 0.0
        self.started = time.time()
        self.part: str | None = getattr(args, "part", None)
        #: suffix of the files every part writes (a part never overwrites an earlier part's)
        self.tag = f"_{self.part}" if self.part else ""
        self.plan = build_plan(
            set(args.only.split(",")) if args.only else None,
            set(args.phases.split(",")) if args.phases else None,
            self.part,
        )
        self.dropped: list[str] = []
        self.sb_overrides: dict[str, Any] | None = None
        self.d_best: str | None = None
        self.run_counter = 0
        self.session_clocks: list[float] = []
        self.deepspeed_bin = (
            os.environ.get("DEEPSPEED_BIN") or shutil.which("deepspeed") or "deepspeed"
        )
        self.s0_bin = os.environ.get("SOUP_S0_BIN") or str(HERE / "venv_s0" / "bin" / "soup")
        self.soup_bin = os.environ.get("SOUP_BIN") or "soup"
        self.gpu = os.environ.get("BENCH_GPU", "0")
        self.nvidia_smi = os.environ.get("DS_COMPARE_NVIDIA_SMI", "nvidia-smi")

    # -- paths ---------------------------------------------------------------------------
    def prefix(self, slot: Slot) -> Path:
        arm = slot.resolved_arm or slot.arm
        if slot.warmup:
            return self.results / "warmup" / arm / "warmup"
        return self.results / slot.shape / str(slot.budget) / arm / f"rep{slot.rep}"

    def runs(self) -> list[dict[str, Any]]:
        return gates_module.load_runs(self.results)

    # -- time and cost -------------------------------------------------------------------
    def time_left(self) -> float | None:
        return None if self.deadline is None else self.deadline - time.time()

    def spent_usd(self) -> float | None:
        if self.rate is None:
            return None
        return (self.setup_seconds + time.time() - self.started) / 3600.0 * self.rate

    def measured_table(self) -> dict[tuple[str, str], float]:
        """Median wall minutes of the finished runs of each (arm, shape)."""
        return measured_wall_minutes(self.runs())

    def estimate_slot(self, slot: Slot, table: dict[tuple[str, str], float] | None = None) -> float:
        arm = slot.resolved_arm or slot.arm
        measured = (table if table is not None else self.measured_table()).get((arm, slot.shape))
        return measured if measured is not None else slot.est_min

    def pending(self) -> list[Slot]:
        return [
            slot
            for slot in self.plan
            if not slot.done and slot.skipped is None and slot.group not in self.dropped
        ]

    def estimate_remaining_minutes(self) -> float:
        table = self.measured_table()
        return sum(self.estimate_slot(slot, table) for slot in self.pending())

    def enforce_budget(self) -> None:
        """Drop whole groups, in protocol order, until what is left fits.

        Only groups that still have pending slots can be dropped, so a part (whose plan holds
        only its own slots) drops only groups inside the part, in the protocol's order; ``core``
        is not in the drop order and is never dropped.
        """
        while True:
            remaining_s = self.estimate_remaining_minutes() * 60.0 * SAFETY_FACTOR
            fits = True
            left = self.time_left()
            if left is not None and left - FINISH_MARGIN_S < remaining_s:
                fits = False
            spent = self.spent_usd()
            if spent is not None and self.budget_usd is not None and self.rate:
                if spent + remaining_s / 3600.0 * self.rate > self.budget_usd:
                    fits = False
            if fits:
                return
            candidates = [
                group
                for group in DROP_ORDER
                if group not in self.dropped and any(s.group == group for s in self.pending())
            ]
            if not candidates:
                return
            group = candidates[0]
            self.dropped.append(group)
            self.log(
                f"BUDGET: dropping group {group!r} (time left {left}, "
                f"remaining estimate {remaining_s / 60:.0f} min)"
            )
            atomic_write_json(
                self.results / f"dropped{self.tag}.json", {"dropped_in_order": self.dropped}
            )

    # -- output --------------------------------------------------------------------------
    def log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        self.results.mkdir(parents=True, exist_ok=True)
        with open(self.results / "orchestrate.log", "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    # -- commands ------------------------------------------------------------------------
    def command_for(self, slot: Slot, overrides_path: str | None) -> list[str]:
        """The argv of one run (environment: ``child_env``).

        ``DBEST`` is resolved to a real arm before a run; the dry run shows D1's command.
        """
        arm = slot.resolved_arm or slot.arm
        if arm == "DBEST":
            arm = "D1"
        prefix = self.prefix(slot)
        base_out = f"{prefix}.raw.json"
        if arm in soup_arm.STREAM_ARMS:
            command = [
                sys.executable,
                str(HERE / "soup_arm.py"),
                "--arm",
                arm,
                "--shape",
                slot.shape,
                "--workdir",
                f"{prefix}.work",
                "--out",
                base_out,
                "--model",
                self.model,
                "--soup-bin",
                self.s0_bin if arm == "S0" else self.soup_bin,
                "--rep",
                str(slot.rep),
            ]
            if overrides_path:
                command += ["--overrides-json", overrides_path]
            if slot.budget < 80:
                command += ["--cap-gb", str(slot.budget)]
            return command
        spec = HF_ARMS[arm]
        spec_args = [
            "--model",
            self.model,
            "--rows",
            str(rows_path(slot.shape)),
            "--max-length",
            str(SHAPES[slot.shape]["max_length"]),
            "--epochs",
            str(SHAPES[slot.shape]["epochs"]),
            "--arm",
            arm,
            "--shape",
            slot.shape,
            "--rep",
            str(slot.rep),
            "--out",
            base_out,
            "--deepspeed",
            str(HERE / spec["ds"]) if spec["ds"] else "none",
        ]
        if spec["gc"]:
            spec_args.append("--gradient-checkpointing")
        if os.environ.get("DS_COMPARE_DTYPE"):  # CPU rehearsals only; the series is bf16
            spec_args += ["--dtype", os.environ["DS_COMPARE_DTYPE"]]
        if slot.budget < 80:
            spec_args += ["--cap-gb", str(slot.budget)]
        script = str(HERE / "hf_baseline.py")
        if spec["ds"]:
            port = 29500 + (self.run_counter % 200)
            command = [self.deepspeed_bin, "--num_gpus", "1", "--master_port", str(port), script]
        else:
            command = [sys.executable, script]
        return command + spec_args

    def child_env(self, slot: Slot) -> dict[str, str]:
        env = dict(os.environ)
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        env.pop("SOUP_BENCH_VRAM_CAP_GB", None)
        env.pop("SOUP_BENCH_TRACE_OUT", None)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(HERE)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )
        env["CUDA_VISIBLE_DEVICES"] = self.gpu
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("HF_HUB_OFFLINE", "1")
        if slot.budget < 80:
            env["SOUP_BENCH_VRAM_CAP_GB"] = str(slot.budget)
        return env

    # -- one attempt ---------------------------------------------------------------------
    def gpu_busy_before(self) -> bool:
        if shutil.which(self.nvidia_smi) is None and not os.path.exists(self.nvidia_smi):
            return False
        for _ in range(4):
            try:
                out = subprocess.run(
                    [
                        self.nvidia_smi,
                        f"--id={self.gpu}",
                        "--query-compute-apps=pid",
                        "--format=csv,noheader",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                ).stdout.strip()
            except (OSError, subprocess.TimeoutExpired):
                return False
            if not out:
                return False
            time.sleep(20)
        return True

    def attempt(
        self, slot: Slot, overrides: dict[str, Any] | None, attempt_no: int
    ) -> dict[str, Any]:
        prefix = self.prefix(slot)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        overrides_path = None
        if overrides:
            overrides_path = f"{prefix}.overrides.json"
            atomic_write_json(overrides_path, overrides)
        command = self.command_for(slot, overrides_path)
        env = self.child_env(slot)
        for stale in (".raw.json", ".raw.trace.json", ".raw.log"):  # a crashed earlier session
            Path(f"{prefix}{stale}").unlink(missing_ok=True)
        shutil.rmtree(f"{prefix}.work", ignore_errors=True)
        self.run_counter += 1
        preflight_busy = self.gpu_busy_before()
        self.log(f"RUN {slot.label} attempt {attempt_no}: {' '.join(command)}")
        started = time.time()
        timeout = max(1200.0, 4.0 * self.estimate_slot(slot) * 60.0)
        with open(f"{prefix}.log", "w", encoding="utf-8", errors="replace") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            sampler = RunSampler(
                str(prefix),
                process.pid,
                int(self.gpu) if self.gpu.isdigit() else 0,
                self.nvidia_smi,
            )
            sampler.start()
            timed_out = False
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                kill_tree(process.pid)
                process.wait()
            finally:
                sampler.stop()
        wall = time.time() - started

        raw_path = Path(f"{prefix}.raw.json")
        run: dict[str, Any]
        if raw_path.exists():
            run = read_json(raw_path)
        else:
            run = {
                "schema": "ds_compare.run.v1",
                "arm": slot.resolved_arm or slot.arm,
                "status": "failed",
                "exit_code": process.returncode,
                "error": "the run wrote no result json",
            }
            log_text = Path(f"{prefix}.log").read_text(encoding="utf-8", errors="replace")
            run["error"] += "; log tail: " + log_text[-1500:]
            if soup_arm.parse_soup_log(log_text)["oom_in_log"]:
                run["status"] = "oom"
        if timed_out:
            run["status"] = "failed"
            run["error"] = f"killed after {timeout:.0f} s (timeout)"
        log_text = Path(f"{prefix}.log").read_text(encoding="utf-8", errors="replace")
        child_log = Path(f"{prefix}.raw.log")  # soup train's own output (streaming arms)
        if child_log.exists():
            log_text += child_log.read_text(encoding="utf-8", errors="replace")
        run["cap_applied"] = "VRAM cap applied" in log_text

        # Training window: from the trainer's train-begin event to the end of its last step.
        # (train_runtime also covers engine set-up that precedes the event.)
        window = None
        begin = run.get("trace_begin_wall")
        span = run.get("trace_last_step_seconds") or run.get("train_runtime_s")
        if begin and span:
            window = (float(begin), float(begin) + float(span))
        sampler_summary = summarize(str(prefix), window)
        sampler_summary["foreign_gpu_processes"] = sampler.foreign_gpu_processes()
        sampler_summary["preflight_busy"] = preflight_busy
        run["sampler"] = sampler_summary
        run["peak_vram_mib"] = (sampler_summary.get("whole_run") or {}).get("vram_peak_mib")
        run.update(
            {
                "arm": slot.resolved_arm or slot.arm,
                "shape": slot.shape,
                "budget_gb": slot.budget,
                "cap_gb": slot.budget if slot.budget < 80 else None,
                "rep": slot.rep,
                "warmup": slot.warmup,
                "attempt": attempt_no,
                "wall_seconds": wall,
                "child_exit_code": process.returncode,
                "started_at": started,
                "cost_usd": (wall / 3600.0 * self.rate) if self.rate else None,
                "rerun_recompute_on": bool(overrides and overrides.get("_rerun_recompute_on")),
            }
        )
        return run

    def prior_valid(self, slot: Slot) -> list[dict[str, Any]]:
        arm = slot.resolved_arm or slot.arm
        return [
            run
            for run in self.runs()
            if run["arm"] == arm
            and run["shape"] == slot.shape
            and run["budget_gb"] == slot.budget
            and run.get("status") == "ok"
            and not run.get("void")
            and run["rep"] != slot.rep
        ]

    def cleanup_work(self, slot: Slot) -> None:
        work = Path(f"{self.prefix(slot)}.work")
        if work.exists():
            output = work / "output"
            if output.exists():
                keep = output / "adapter_config.json"
                saved = keep.read_bytes() if keep.exists() else None
                shutil.rmtree(output, ignore_errors=True)
                if saved is not None:
                    output.mkdir()
                    (output / "adapter_config.json").write_bytes(saved)
            (work / "rows.jsonl").unlink(missing_ok=True)

    def run_slot(self, slot: Slot) -> None:
        arm = slot.resolved_arm or slot.arm
        final_path = Path(f"{self.prefix(slot)}.json")
        if final_path.exists():
            try:
                existing = read_json(final_path)
                if existing.get("status") in ("ok", "oom") or existing.get("void_attempts", 0) >= 2:
                    slot.done = True
                    self.log(f"SKIP {slot.label}: result already on disk")
                    return
            except (OSError, ValueError):
                pass
        overrides: dict[str, Any] | None = None
        if arm == "SB":
            sb = self.ensure_sb()
            overrides = {"training": dict(sb.get("training", {})), "env": dict(sb.get("env", {}))}
        voids = 0
        rerun_done = False
        for attempt_no in (1, 2, 3):
            run = self.attempt(slot, overrides, attempt_no)
            self.cleanup_work(slot)
            # Shape C: a streaming arm out of memory is rerun once with stream_recompute: on.
            if (
                run["status"] == "oom"
                and slot.shape == "C"
                and arm in soup_arm.STREAM_ARMS
                and not rerun_done
                and (overrides or {}).get("training", {}).get("stream_recompute", "on") != "on"
            ):
                rerun_done = True
                atomic_write_json(f"{self.prefix(slot)}.oom.json", run)
                Path(f"{self.prefix(slot)}.raw.json").unlink(missing_ok=True)
                overrides = {
                    "training": {**(overrides or {}).get("training", {}), "stream_recompute": "on"},
                    "env": dict((overrides or {}).get("env", {})),
                    "_rerun_recompute_on": True,
                }
                self.log(f"OOM at shape C: rerunning {slot.label} with stream_recompute: on")
                continue
            prior_losses = [r["final_loss"] for r in self.prior_valid(slot) if r.get("final_loss")]
            reasons = void_reasons(
                run,
                slot.budget,
                prior_losses,
                self.session_clocks,
                bool((run.get("sampler") or {}).get("preflight_busy")),
            )
            run["void_reasons"] = reasons
            if reasons and not (run["status"] == "oom" and slot.budget < 80):
                voids += 1
                run["void"] = True
                self.log(f"VOID {slot.label} attempt {attempt_no}: {'; '.join(reasons)}")
                if slot.warmup or voids >= 2:
                    run["void_attempts"] = voids
                    break
                atomic_write_json(f"{self.prefix(slot)}.void{voids}.json", run)
                Path(f"{self.prefix(slot)}.raw.json").unlink(missing_ok=True)
                continue
            run["void"] = False
            break
        run["void_attempts"] = voids
        atomic_write_json(final_path, run)
        slot.done = True
        if not slot.warmup:
            clock = ((run.get("sampler") or {}).get("train_window") or {}).get(
                "sm_clock_busy_median_mhz"
            )
            if clock and run["status"] == "ok":
                self.session_clocks.append(clock)
        self.record(slot, run)

    def record(self, slot: Slot, run: dict[str, Any]) -> None:
        line = {
            "label": slot.label,
            "status": run.get("status"),
            "void": run.get("void"),
            "tok_per_s": run.get("tok_per_s"),
            "steady_tok_per_s": run.get("steady_tok_per_s"),
            "peak_vram_mib": run.get("peak_vram_mib"),
            "final_loss": run.get("final_loss"),
            "wall_seconds": run.get("wall_seconds"),
            "cost_usd": run.get("cost_usd"),
        }
        with open(self.results / "run_log.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(line) + "\n")
        self.log(
            f"DONE {slot.label}: {run.get('status')} tok/s {run.get('tok_per_s')} "
            f"peak {run.get('peak_vram_mib')} MiB wall {run.get('wall_seconds', 0):.0f} s"
        )
        atomic_write_json(
            self.results / f"progress{self.tag}.json",
            {
                "updated": time.time(),
                "dropped": self.dropped,
                "pending": len(self.pending()),
                "last": slot.label,
            },
        )

    # -- derivations at run time ---------------------------------------------------------
    def resolve(self, slot: Slot) -> bool:
        """False to skip the slot (with ``slot.skipped`` set)."""
        runs = self.runs()
        if slot.conditional == "sb_derivation":
            self.ensure_sb()
        elif slot.conditional == "idle_rule":
            if not idle_rule(runs, slot.arm[:2]):
                slot.skipped = "GPU not idle more than 50%"
        elif slot.conditional == "d_best":
            self.resolve_d_best(runs)
            if self.d_best is None:
                slot.skipped = "no DeepSpeed arm measured at Shape A"
            elif self.d_best == "D1":
                slot.skipped = "D-best is D1 (already in the arm list)"
            else:
                slot.resolved_arm = self.d_best
        return slot.skipped is None

    def ensure_sb(self) -> dict[str, Any]:
        """sb_overrides.json: read when a previous session wrote it, else derive and write it."""
        if self.sb_overrides is None:
            path = self.results / "sb_overrides.json"
            if path.exists():
                self.sb_overrides = read_json(path)
            else:
                self.sb_overrides = derive_sb_overrides(self.runs())
                atomic_write_json(path, self.sb_overrides)
                self.log(
                    f"SB derived: selected {self.sb_overrides['selected'] or 'nothing (SB = S1)'}"
                )
        return self.sb_overrides

    def resolve_d_best(self, runs: list[dict[str, Any]]) -> None:
        if self.d_best is not None:
            return
        path = self.results / "d_best.json"
        if path.exists():
            self.d_best = read_json(path).get("arm")
            return
        best = gates_module.best_deepspeed(runs, "A", 80, D_BEST_CANDIDATES)
        if best is not None:
            self.d_best = best["arm"]
            atomic_write_json(self.results / "d_best.json", {"arm": self.d_best, "stats": best})
            self.log(f"D-best (fastest DeepSpeed arm, Shape A, 80 GB): {self.d_best}")

    # -- main loop ----------------------------------------------------------------------
    def refusals(self) -> list[str]:
        """Reasons not to start at all (checked before anything is written or run)."""
        problems: list[str] = []
        if self.part:
            only = set(self.args.only.split(",")) if self.args.only else None
            problems += part_prerequisites(self.part, self.results, only)
        if any(slot.arm == "S0" for slot in self.plan) and not Path(self.s0_bin).exists():
            problems.append(
                f"arm S0 is in the plan but its executable {self.s0_bin} does not exist "
                "(setup.sh was run with SKIP_S0=1, or not at all): run setup.sh without "
                "SKIP_S0, or leave S0 out with --only"
            )
        return problems

    def execute(self) -> int:
        problems = self.refusals()
        if problems:
            for problem in problems:
                print(f"REFUSED: {problem}", file=sys.stderr, flush=True)
            return 2
        self.results.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            self.results / f"session{self.tag}.json",
            {
                "part": self.part,
                "started": self.started,
                "model": self.model,
                "model_revision": MODEL_REVISION,
                "deadline_epoch": self.deadline,
                "rate_usd_per_hour": self.rate,
                "budget_usd": self.budget_usd,
                "plan_slots": len(self.plan),
                "args": vars(self.args),
            },
        )
        for slot in self.plan:
            if slot.done or slot.skipped:
                continue
            if slot.group in self.dropped:
                continue
            left = self.time_left()
            if left is not None and left <= FINISH_MARGIN_S and slot.group != "core":
                continue
            if left is not None and left <= 0:
                self.log("deadline reached: no new runs")
                break
            if not self.resolve(slot):
                self.log(f"SKIP {slot.label}: {slot.skipped}")
                continue
            self.enforce_budget()
            if slot.group in self.dropped:
                continue
            self.run_slot(slot)
        return self.finish()

    def finish(self) -> int:
        self.log("computing gates")
        try:
            result = gates_module.compute_all(self.results)
            atomic_write_json(self.results / "gates.json", result)
            summary = gates_module.render_summary(result)
            (self.results / "gates.txt").write_text(summary + "\n", encoding="utf-8")
            self.log("\n" + summary)
        except Exception as exc:  # noqa: BLE001 - the DONE marker must still be written
            self.log(f"gates failed: {type(exc).__name__}: {exc}")
        self.write_runs_csv()
        atomic_write_json(
            self.results / f"DONE{self.tag}",
            {
                "part": self.part,
                "finished": time.time(),
                "dropped_groups": self.dropped,
                "skipped": [f"{s.label}: {s.skipped}" for s in self.plan if s.skipped],
                "elapsed_seconds": time.time() - self.started,
                **self.derivations_for_the_record(),
            },
        )
        return 0

    def derivations_for_the_record(self) -> dict[str, Any]:
        """What part 1 hands to the later parts. Logged and recorded, never written as a file
        here: ``sb_overrides.json`` is written when SB is first needed and ``d_best.json`` when
        D-best is first needed (part 2), so an interrupted part 1 cannot freeze a choice made
        from half of its data."""
        if self.part != "part1":
            return {}
        try:
            sb_path = self.results / "sb_overrides.json"
            selected = read_json(sb_path).get("selected") if sb_path.exists() else None
            best = gates_module.best_deepspeed(self.runs(), "A", 80, D_BEST_CANDIDATES)
        except Exception as exc:  # noqa: BLE001 - the DONE marker must still be written
            return {"derivation_error": f"{type(exc).__name__}: {exc}"}
        d_best = best["arm"] if best else None
        self.log(
            f"part1 hands over: SB selected {selected}; D-best from these results: {d_best} "
            "(part2 and part3 derive both again from the run jsons)"
        )
        return {"sb_selected": selected, "d_best_now": d_best}

    def write_runs_csv(self) -> None:
        fields = [
            "shape",
            "budget_gb",
            "arm",
            "rep",
            "status",
            "void",
            "attempt",
            "tok_per_s",
            "steady_tok_per_s",
            "train_runtime_s",
            "tokens_trained",
            "peak_vram_mib",
            "max_memory_reserved_bytes",
            "final_loss",
            "wall_seconds",
            "cost_usd",
        ]
        with open(self.results / "runs.csv", "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for run in sorted(
                self.runs(), key=lambda r: (r["shape"], r["budget_gb"], r["arm"], r["rep"])
            ):
                writer.writerow(run)


def kill_tree(pid: int) -> None:
    try:
        import psutil

        root = psutil.Process(pid)
        for child in root.children(recursive=True):
            child.kill()
        root.kill()
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------------------


def dry_run(session: Session) -> int:
    results = session.results
    configs = results / "dry_run_configs"
    for shape in SHAPES:
        for arm in soup_arm.STREAM_ARMS:
            config, env, notes = soup_arm.build_config(
                arm, shape, session.model, "rows.jsonl", "./output"
            )
            target = configs / shape / arm
            target.mkdir(parents=True, exist_ok=True)
            (target / "soup.yaml").write_text(soup_arm.dump_yaml(config), encoding="utf-8")
            atomic_write_json(target / "env_and_notes.json", {"env": env, "notes": notes})
    part_note = f", {session.part}" if session.part else ""
    print(f"plan: {len(session.plan)} slots (results {results}, model {session.model}{part_note})")
    if session.part:
        only = set(session.args.only.split(",")) if session.args.only else None
        for problem in part_prerequisites(session.part, results, only):
            print(f"NOTE: a real run would be REFUSED: {problem}")
    print(f"soup.yaml written for every streaming arm and shape under {configs}")
    if session.part in ("part2", "part3"):
        print("SB: derived from part1's Shape A results (the dry run writes S1's yaml)")
    else:
        print("SB: derived after Shape A repeat 3 (the dry run writes S1's yaml as a stand-in)")
    if session.part in (None, "part2"):
        print("D-best (Shapes B and C): the fastest DeepSpeed arm of Shape A, decided at run time")
    print()
    totals: dict[str, float] = {}
    conditional_total = 0.0
    for slot in session.plan:
        command = session.command_for(slot, None)
        print(
            f"[{slot.index:03d}] {slot.phase:8s} {slot.label:24s} group={slot.group:9s} "
            f"est {slot.est_min:4.1f} min"
            + (f"  (conditional: {slot.conditional})" if slot.conditional else "")
        )
        print("      " + " ".join(command))
        extra = {
            "S5": "env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False",
        }.get(slot.arm)
        if extra:
            print("      " + extra)
        if slot.budget < 80:
            print(f"      env SOUP_BENCH_VRAM_CAP_GB={slot.budget}")
        totals[slot.group] = totals.get(slot.group, 0.0) + slot.est_min
        if slot.conditional in ("idle_rule", "d_best"):
            conditional_total += slot.est_min
    grand = sum(totals.values())
    print()
    warmups = sum(1 for s in session.plan if s.warmup)
    print(f"runs planned: {len(session.plan)} (incl. {warmups} warm-ups)")
    by_phase: dict[str, int] = {}
    for slot in session.plan:
        by_phase[slot.phase] = by_phase.get(slot.phase, 0) + 1
    print("runs per phase: " + ", ".join(f"{k}={v}" for k, v in by_phase.items()))
    for group, minutes in sorted(totals.items(), key=lambda item: -item[1]):
        print(f"  group {group:9s} {minutes:6.1f} min")
    print(
        f"estimated GPU time, everything: {grand:.0f} min = {grand / 60:.2f} h "
        f"(conditional runs included: {conditional_total:.0f} min)"
    )
    print(
        f"without the conditional runs (variants, D-best when it differs from D1): "
        f"{(grand - conditional_total):.0f} min = {(grand - conditional_total) / 60:.2f} h"
    )
    total_h = grand / 60 + SETUP_HOURS
    verdict = "FITS" if total_h <= HOURS_LIMIT else "DOES NOT FIT (the budget rule drops groups)"
    if session.part:
        print(
            f"suggested --max-hours for {session.part}: {suggested_max_hours(grand):.2f} "
            f"(setup {SETUP_MIN_PER_PART:.0f} min + runs {grand:.0f} min + margin "
            f"{MARGIN_MIN_PER_PART:.0f} min, rounded up to 0.25 h)"
        )
    else:
        print(
            f"with {SETUP_HOURS:.0f} h of setup: {total_h:.2f} h against the {HOURS_LIMIT:.0f} h "
            f"limit -> {verdict}"
        )
    print(
        "estimates: train time from the STEP 3 / STEP 4 records, plus a startup per run; "
        "replaced by measured wall times as runs finish"
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--all", action="store_true", help="run the whole series")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, write configs")
    parser.add_argument("--only", default=None, help="comma separated arm ids (S1,D1,SB,...)")
    parser.add_argument(
        "--phases",
        default=None,
        help="comma separated phases: warmup,A,SB,variants,B1,C1,cap24,B2,C2,cap16,S8",
    )
    parser.add_argument(
        "--part",
        choices=PARTS,
        default=None,
        help="run only this part of the series (one rented instance each): "
        + "; ".join(f"{part} = {', '.join(phases)}" for part, phases in PART_PHASES.items()),
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="with --part: check that the earlier parts' results are in --results and exit "
        "(0 ready, 2 refused); runs nothing, writes nothing",
    )
    parser.add_argument(
        "--results", default=os.environ.get("DS_COMPARE_RESULTS") or str(HERE / "results")
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("DS_COMPARE_MODEL_DIR") or str(HERE / "model"),
        help=f"local directory of {MODEL_ID} at {MODEL_REVISION[:12]}",
    )
    args = parser.parse_args(argv)
    if args.preflight and not args.part:
        parser.error("--preflight needs --part")
    if not (args.all or args.only or args.phases or args.dry_run or args.part):
        parser.error("give --all, --part <part>, --only <arms>, --phases <names> or --dry-run")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.preflight:
        only = set(args.only.split(",")) if args.only else None
        problems = part_prerequisites(args.part, args.results, only)
        for problem in problems:
            print(f"REFUSED: {problem}", file=sys.stderr, flush=True)
        if not problems:
            print(f"preflight {args.part}: ready (results {args.results})")
        return 2 if problems else 0
    session = Session(args)
    if args.dry_run:
        return dry_run(session)
    return session.execute()


if __name__ == "__main__":
    raise SystemExit(main())
