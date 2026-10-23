#!/usr/bin/env python3
"""The R2 batch-control rule (§2 of ``benchmarks/probe-rtx5070-batch-control.md``), from files.

Everything the rule reads, recomputed from an arm's committed JSON and log and from the
driver's ``batch_driver.json``: no GPU, no sampling, nothing is run. The driver
(``batch_control_probe.py``) imports this module for the per-arm checks it applies while the
sequence runs (rows 1, 2 and 4); ``--summarize`` there applies every row here.

Each threshold below is quoted with its source line in the record's §2.
"""

from __future__ import annotations

import json
import os
import re
import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple

import two_drive_stripe_gate as gate

# -- the arms -----------------------------------------------------------------------------------
LAYOUTS = ("SINGLE", "STRIPED")
BATCHES = (1, 2, 3)
SEQ = 512
TIMED_STEPS = 3
WARMUP = 1
COMPUTE_ARM = "B_nocopy"  # the read-free compute floor, run in SINGLE arms after the step block
_ROUND_0: Tuple[Tuple[str, int], ...] = (
    ("SINGLE", 1),
    ("STRIPED", 1),
    ("SINGLE", 2),
    ("STRIPED", 2),
    ("SINGLE", 3),
    ("STRIPED", 3),
)
#: (round, layout, batch) in the order the rule fixes: round 1 is round 0 reversed, so every
#: arm sits at the same mean position and a linear drift cancels in the two-round mean.
ROUND_PLAN: Tuple[Tuple[int, str, int], ...] = tuple(
    (0, layout, batch) for layout, batch in _ROUND_0
) + tuple((1, layout, batch) for layout, batch in reversed(_ROUND_0))
ROUNDS = (0, 1)
READ_AHEAD = {"SINGLE": 2, "STRIPED": 3}
STRIPED_SHARDS = gate.ARM_FLAGS["STRIPED"][gate.ARM_FLAGS["STRIPED"].index("--shards") + 1]
STRIPE_ROOT = gate.ARM_FLAGS["STRIPED"][gate.ARM_FLAGS["STRIPED"].index("--stripe-dir") + 1]
#: The cache each layout must read. SINGLE: the default one-root cache stream_probe.py resolves
#: with no --shards (never the ``.refit`` copy beside it, nor a SOUP_LAYER_STREAM_CACHE_DIR one).
SHARD_DIRS = {
    "SINGLE": "C:/Users/user/.soup/layer-stream/D___synth__llama-70b-shape-v32k-f4",
    "STRIPED": STRIPED_SHARDS,
}
N_LAYERS = 80
LOADS_PER_STEP = (157, 2)

# -- §2 of the record ---------------------------------------------------------------------------
SINGLE_VALID = gate.SINGLE_VALID  # (15.0, 21.5) s, gate §2 row 1
#: 70,378,258,732 bytes per step over 9.15 GB/s, the best two-drive read on record (gate §0),
#: and the gate's 12.0 s SHIP line (gate §2 row 3).
STRIPED_VALID = (7.69, 12.0)
NEAR_FREE_MIN = 0.95  # e = step(b1) / step(b): the step grew by at most 1/0.95 - 1 = 5.3%
NONE_MAX = 1.05  # r = b * e, the tok/s ratio over batch 1: tok/s rose by at most 5%
EDGE_MIB = 7620  # nvidia-smi memory in use at the WDDM spill what-bounds §18 describes
#: B(S) = MODEL_INTERCEPT_S + MODEL_SLOPE_S * S, through the two arm-B means of what-bounds §21
#: (i974_ablate_cold70b_seq{256,512}.json): 3.833720 s at 256 tokens, 6.425469 s at 512.
MODEL_INTERCEPT_S = 1.241971
MODEL_SLOPE_S = 0.0101240
RESHARD_S = 30.0  # shard_seconds above this in a round arm means it re-sharded
FOREIGN_READ_MAX = 10**9  # bytes another process may read during one arm
ARM_WALL_CAP_S = 600.0  # a round arm still running after this is killed: a result, not a void
_OOM_TEXT = re.compile(r"OutOfMemoryError|out of memory|ALLOC_FAILED", re.IGNORECASE)
#: Columns of the driver's 2-s nvidia-smi rows: [unix time, used MiB, total MiB, SM MHz, temp C,
#: power W, utilisation %, throttle reasons].
_GPU_USED, _GPU_SM, _GPU_POWER, _GPU_UTIL = 1, 3, 5, 6


def model_step_s(tokens: int) -> float:
    """The read-free compute floor B(S) of what-bounds §21, extrapolated as a line."""
    return MODEL_INTERCEPT_S + MODEL_SLOPE_S * tokens


def arm_name(prefix: str, rnd: Optional[int], layout: str, batch: int) -> str:
    if rnd is None:
        return f"{prefix}_build_{layout.lower()}"
    return f"{prefix}_r{rnd}_{layout.lower()}_b{batch}"


def _same_path(first: Any, second: str) -> bool:
    if not isinstance(first, str):
        return False
    return os.path.normcase(os.path.normpath(first)) == os.path.normcase(os.path.normpath(second))


# ==========================================================================
# one arm, from its files
# ==========================================================================
def read_points(json_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    try:
        with open(json_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return {}, []
    return payload.get("meta", {}), list(payload.get("records", []))


def point(records: Sequence[Dict[str, Any]], kind: str, label: str) -> Optional[Dict[str, Any]]:
    for record in records:
        if record.get("kind") == kind and record.get("label") == label:
            return record
    return None


def log_mentions_oom(log_path: str) -> Optional[str]:
    """The last out-of-memory line of an arm's log, for an error outside a guarded block."""
    try:
        with open(log_path, encoding="utf-8", errors="replace") as handle:
            lines = [line.strip() for line in handle if _OOM_TEXT.search(line)]
    except OSError:
        return None
    return lines[-1][:500] if lines else None


def arm_problems(
    layout: str,
    batch: int,
    meta: Dict[str, Any],
    plain: Dict[str, Any],
    build: bool,
) -> List[str]:
    """Row 2's per-arm inputs: everything that says the arm did not run the path under test."""
    problems: List[str] = []
    for key in ("direct_io", "pinned"):
        if meta.get(key) is not True:
            problems.append(f"{key} {meta.get(key)!r}")
    if meta.get("source_class") != "AsyncDiskSource":
        problems.append(f"source {meta.get('source_class')!r}")
    if meta.get("read_ahead") != READ_AHEAD[layout]:
        problems.append(f"read_ahead {meta.get('read_ahead')!r}, not {READ_AHEAD[layout]}")
    if not _same_path(meta.get("shard_dir"), SHARD_DIRS[layout]):
        problems.append(f"shard_dir {meta.get('shard_dir')!r}, not {SHARD_DIRS[layout]}")
    roots = [os.path.normcase(os.path.normpath(root)) for root in meta.get("stripe_roots") or []]
    placement = list(meta.get("layer_roots") or [])
    if layout == "STRIPED":
        if roots != [os.path.normcase(os.path.normpath(STRIPE_ROOT))]:
            problems.append(f"stripe roots {meta.get('stripe_roots')!r}")
        if placement != [idx % 2 for idx in range(N_LAYERS)]:
            problems.append("layer placement is not i mod 2 over 80 layers")
    elif roots or placement:
        problems.append(f"SINGLE arm with stripe roots {meta.get('stripe_roots')!r}")
    if meta.get("batch") != batch or meta.get("seq") != SEQ:
        problems.append(f"batch x seq {meta.get('batch')!r} x {meta.get('seq')!r}")
    if build:
        return problems
    shard_s = meta.get("shard_seconds")
    if not isinstance(shard_s, (int, float)) or shard_s > RESHARD_S:
        problems.append(f"shard_seconds {shard_s!r} (re-sharded instead of reading the cache)")
    if plain.get("tokens_per_step") != batch * SEQ:
        problems.append(f"tokens_per_step {plain.get('tokens_per_step')!r}")
    records = plain.get("records", [])
    if len(records) != TIMED_STEPS:
        problems.append(f"{len(records)} timed steps, not {TIMED_STEPS}")
    bad = [
        idx + 1
        for idx, rec in enumerate(records)
        if (rec.get("layer_loads"), rec.get("large_loads")) != LOADS_PER_STEP
    ]
    if bad:
        problems.append(f"timed steps {bad} did not load 157 + 2")
    rows = (plain.get("ids") or {}).get("row_sha256")
    if not isinstance(rows, list) or len(rows) != batch:
        problems.append(f"ids row hashes {rows!r}, not {batch} rows")
    return problems


def arm_facts(
    json_path: str, log_path: str, layout: str, batch: int, timed_out: bool, build: bool = False
) -> Dict[str, Any]:
    """Everything the rule reads from one arm's files, recomputable from the committed JSON.

    ``layout`` and ``batch`` come from the arm's position in the plan, never from its JSON: the
    checks exist to catch a JSON that disagrees with the position.
    """
    meta, records = read_points(json_path)
    plain = point(records, "step", "step_plain")
    compute = point(records, "ablate", f"{COMPUTE_ARM}_r0")
    oom = next((record for record in records if record.get("kind") == "oom"), None)
    oom_keys = ("label", "error_type", "error", "peak_alloc_gb", "peak_reserved_gb")
    facts: Dict[str, Any] = {
        "has_step_plain": plain is not None,
        "timed_out": timed_out,
        "vram_total_gb": meta.get("vram_total_gb"),
        "shard_dir": meta.get("shard_dir"),
        "shard_seconds": meta.get("shard_seconds"),
        "direct_io": meta.get("direct_io"),
        "pinned": meta.get("pinned"),
        "read_ahead": meta.get("read_ahead"),
        "source_class": meta.get("source_class"),
        "stripe_roots": meta.get("stripe_roots"),
        "layer_roots_head": (meta.get("layer_roots") or [])[:8],
        "oom": None if oom is None else {key: oom.get(key) for key in oom_keys},
        "oom_in_log": None if oom or plain else log_mentions_oom(log_path),
        "problems": arm_problems(layout, batch, meta, plain, build) if plain else [],
    }
    if plain is not None:
        steps = plain.get("records", [])
        facts.update(
            {
                "step_s_mean": plain["step_s_mean"],
                "step_s": [rec["step_s"] for rec in steps],
                "started_unix": [rec.get("started_unix") for rec in steps],
                "tok_per_s": plain["tok_per_s"],
                "losses": [rec.get("loss") for rec in steps],
                "peak_alloc_gb": plain["peak_alloc_gb"],
                "peak_reserved_gb": plain["peak_reserved_gb"],
                "implied_h2d_gb_per_s": plain.get("implied_h2d_gb_per_s"),
                "memory": plain.get("memory"),
                "ids_row_sha256": (plain.get("ids") or {}).get("row_sha256"),
            }
        )
    if compute is not None:
        facts["compute_floor"] = {
            "step_s_mean": compute["step_s_mean"],
            "step_s": [rec["step_s"] for rec in compute.get("records", [])],
            "peak_alloc_gb": compute["peak_alloc_gb"],
        }
    return facts


def timed_window_gpu(facts: Dict[str, Any], gpu_rows: Sequence[Sequence[Any]]) -> Dict[str, Any]:
    """The driver's 2-s nvidia-smi samples that fall inside the timed steps of the step block.

    Each timed step runs from its ``started_unix`` for ``step_s``. Returns the peak memory in
    use, and the median and minimum power and SM clock over those samples (``None`` when the
    arm has no timed steps or no sample fell inside them).
    """
    starts, times = facts.get("started_unix") or [], facts.get("step_s") or []
    windows = [
        (start, start + step)
        for start, step in zip(starts, times)
        if isinstance(start, (int, float)) and isinstance(step, (int, float))
    ]
    inside = [row for row in gpu_rows if any(low <= row[0] <= high for low, high in windows)]

    def column(index: int) -> List[float]:
        return [row[index] for row in inside if isinstance(row[index], (int, float))]

    used, power, clock, util = (
        column(_GPU_USED),
        column(_GPU_POWER),
        column(_GPU_SM),
        column(_GPU_UTIL),
    )
    return {
        "samples": len(inside),
        "used_mib_max": max(used, default=None),
        "power_w_median": statistics.median(power) if power else None,
        "power_w_min": min(power, default=None),
        "sm_mhz_median": statistics.median(clock) if clock else None,
        "sm_mhz_min": min(clock, default=None),
        "util_pct_median": statistics.median(util) if util else None,
    }


def fits(facts: Dict[str, Any]) -> Tuple[bool, str]:
    """Row 4's per-arm tests (out of memory, the cap, allocated above the card)."""
    if not facts.get("has_step_plain"):
        if facts.get("oom"):
            oom = facts["oom"]
            return False, f"out of memory in {oom['label']}: {oom['error_type']}"
        if facts.get("oom_in_log"):
            return False, f"out of memory outside a measured block: {facts['oom_in_log']}"
        if facts.get("timed_out"):
            return False, f"no step_plain point within {ARM_WALL_CAP_S:.0f} s"
        return False, "no step_plain point"
    peak, total = facts.get("peak_alloc_gb"), facts.get("vram_total_gb")
    if isinstance(peak, (int, float)) and isinstance(total, (int, float)) and peak > total:
        return False, f"allocated {peak:.3f} GB on a {total:.3f} GB card (spilled)"
    return True, ""


def spill_reason(
    step_s: Optional[float], read_s: Optional[float], batch: int, used_mib: Any
) -> str:
    """Row 4's spill test: the card as full as §18's spill AND the step slower than the read and
    the compute one after the other (the model's no-overlap bound). Empty when it does not fire.
    """
    if step_s is None or read_s is None or not at_edge(used_mib):
        return ""
    bound = read_s + model_step_s(SEQ * batch)
    if step_s <= bound:
        return ""
    return (
        f"spill: step {step_s:.3f} s above read + compute in series {bound:.3f} s with "
        f"{used_mib:.0f} MiB in use"
    )


def at_edge(nvsmi_used_mib_max: Any) -> bool:
    return isinstance(nvsmi_used_mib_max, (int, float)) and nvsmi_used_mib_max >= EDGE_MIB


# ==========================================================================
# the rule, over a sequence
# ==========================================================================
def classify(e_value: float, batch: int) -> str:
    """Row 5's bands, applied to a measured or a predicted e = step(b1) / step(b)."""
    if e_value >= NEAR_FREE_MIN:
        return "NEAR-FREE"
    if batch * e_value <= NONE_MAX:
        return "NONE"
    return "PARTIAL"


def _fmt(value: Optional[float], digits: int = 3) -> str:
    return "none" if value is None else f"{value:.{digits}f}"


def load_driver_json(out_dir: str) -> Dict[str, Any]:
    try:
        with open(os.path.join(out_dir, "batch_driver.json"), encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {"runs": [], "invocations": []}


def collect(
    out_dir: str, prefix: str
) -> Tuple[Dict[Tuple[int, str, int], Dict[str, Any]], List[str], str]:
    """The valid attempt of every round position, its facts recomputed from its JSON.

    Returns ``({(round, layout, batch): record}, [positions never run], outcome)``. A position
    the driver skipped (row 4: a smaller batch of that layout did not fit) is not "never run".
    """
    payload = load_driver_json(out_dir)
    invocations = [
        inv
        for inv in payload.get("invocations", [])
        if inv.get("plan") in ("rounds", "all")
        and inv.get("run_prefix") == prefix
        and not inv.get("dry_run")
    ]
    outcome = (
        invocations[-1].get("outcome", "not finished") if invocations else "no rounds invocation"
    )
    arms: Dict[Tuple[int, str, int], Dict[str, Any]] = {}
    missing: List[str] = []
    for rnd, layout, batch in ROUND_PLAN:
        name = arm_name(prefix, rnd, layout, batch)
        runs = [run for run in payload.get("runs", []) if run.get("run") == name]
        valid = [
            run
            for run in runs
            if run.get("exit_code") is not None
            and not run.get("void_reasons")
            and not run.get("dry_run")
        ]
        if not valid:
            if not any(run.get("skipped") for run in runs):
                missing.append(name)
            continue
        run = valid[-1]
        facts = arm_facts(
            os.path.join(out_dir, f"{name}.json"),
            os.path.join(out_dir, f"{name}.log"),
            layout,
            batch,
            bool(run.get("timed_out")),
        )
        fit, why = fits(facts)
        gpu_rows = (run.get("during_samples") or {}).get("gpu") or []
        window = timed_window_gpu(facts, gpu_rows)
        arm_max = (run.get("during") or {}).get("nvsmi_used_mib_max")
        used = window["used_mib_max"] if window["used_mib_max"] is not None else arm_max
        arms[(rnd, layout, batch)] = {
            "run": name,
            "facts": facts,
            "problems": facts["problems"],
            "fits": fit,
            "fit_reason": why,
            "window": window,
            "used_mib": used,
            "edge": at_edge(used),
            "during": run.get("during") or {},
        }
    # Row 4 (iv) needs the same round's batch-1 step of the same layout.
    for (rnd, layout, batch), rec in arms.items():
        base = arms.get((rnd, layout, 1))
        rec["spill"] = (
            ""
            if batch == 1 or base is None
            else spill_reason(
                rec["facts"].get("step_s_mean"),
                base["facts"].get("step_s_mean"),
                batch,
                rec["used_mib"],
            )
        )
    return arms, missing, outcome


def ids_problems(arms: Dict[Tuple[int, str, int], Dict[str, Any]]) -> List[str]:
    """Row 2: every arm of one batch must train on the same rows (identical row hashes)."""
    problems = []
    for batch in BATCHES:
        seen = {
            tuple(rec["facts"]["ids_row_sha256"])
            for (_rnd, _layout, arm_batch), rec in arms.items()
            if arm_batch == batch and rec["facts"].get("ids_row_sha256")
        }
        if len(seen) > 1:
            problems.append(f"batch {batch}: {len(seen)} different id sets across its arms")
    return problems


def layout_report(arms: Dict[Tuple[int, str, int], Dict[str, Any]], layout: str) -> Dict[str, Any]:
    """Rows 4-6 for one layout, and what the record reports beside them."""
    step = {
        (rnd, batch): arms[(rnd, layout, batch)]["facts"].get("step_s_mean")
        for rnd, lay, batch in ROUND_PLAN
        if lay == layout and (rnd, layout, batch) in arms
    }
    base = [step[(rnd, 1)] for rnd in ROUNDS if step.get((rnd, 1)) is not None]
    read_s = statistics.fmean(base) if base else None
    report: Dict[str, Any] = {"R": read_s, "batches": {}}
    if read_s is None:
        return report
    report["crossover_model_tokens"] = (read_s - MODEL_INTERCEPT_S) / MODEL_SLOPE_S
    stopped = False
    for batch in BATCHES[1:]:
        present = [
            (rnd, arms[(rnd, layout, batch)]) for rnd in ROUNDS if (rnd, layout, batch) in arms
        ]
        e_pred = read_s / max(read_s, model_step_s(SEQ * batch))
        row: Dict[str, Any] = {"e_pred": e_pred, "class_pred": classify(e_pred, batch)}
        report["batches"][batch] = row
        if stopped or not present:
            row["result"] = "NOT RUN (a smaller batch did not fit)" if stopped else "NOT RUN"
            stopped = True
            continue
        row["peak_alloc_gb"] = [cell["facts"].get("peak_alloc_gb") for _rnd, cell in present]
        row["peak_reserved_gb"] = [cell["facts"].get("peak_reserved_gb") for _rnd, cell in present]
        row["used_mib"] = [cell["used_mib"] for _rnd, cell in present]
        row["window"] = {rnd: cell["window"] for rnd, cell in present}
        misfits = [cell["fit_reason"] for _rnd, cell in present if not cell["fits"]]
        misfits += [cell["spill"] for _rnd, cell in present if cell.get("spill")]
        if misfits:
            row["result"] = "DOES NOT FIT"
            row["why"] = misfits
            stopped = True
            continue
        per_round = {
            rnd: step[(rnd, 1)] / step[(rnd, batch)]
            for rnd in ROUNDS
            if step.get((rnd, 1)) and step.get((rnd, batch))
        }
        if len(per_round) != len(ROUNDS):
            row["result"] = "NO INPUT (a round lacks this batch or its batch-1 arm)"
            continue
        e_mean = statistics.fmean(per_round.values())
        row.update(
            {
                "e_per_round": per_round,
                "e": e_mean,
                "r": batch * e_mean,
                "tok_per_s": statistics.fmean(cell["facts"]["tok_per_s"] for _rnd, cell in present),
                "class": classify(e_mean, batch),
                "edge": any(cell["edge"] for _rnd, cell in present),
                # Slower than reading and computing one after the other, with the card NOT as
                # full as a spill: outside the model, and named so it is not read as compute.
                "beyond_series": any(
                    cell["facts"]["step_s_mean"] > step[(rnd, 1)] + model_step_s(SEQ * batch)
                    for rnd, cell in present
                ),
            }
        )
        row["labels"] = [
            text
            for text, on in (
                ("AT THE CARD'S EDGE", row["edge"]),
                ("SLOWER THAN READ + COMPUTE IN SERIES", row["beyond_series"]),
            )
            if on
        ]
        row["result"] = row["class"] + "".join(f" ({text})" for text in row["labels"])
        row["agrees"] = row["class"] == row["class_pred"]
    # The measured bracket: past the largest batch judged NEAR-FREE (batch 1 by definition),
    # up to the first batch judged PARTIAL or NONE; open when neither happens below 3.
    low, high = SEQ, None
    for batch in BATCHES[1:]:
        row = report["batches"][batch]
        if row.get("class") == "NEAR-FREE":
            low = SEQ * batch
            continue
        if row.get("class") in ("PARTIAL", "NONE"):
            high = SEQ * batch
        break
    report["crossover_bracket_tokens"] = [low, high]
    return report


def compute_floor(arms: Dict[Tuple[int, str, int], Dict[str, Any]]) -> Dict[str, Any]:
    """The in-session read-free floor at each batch (SINGLE arms) and the line through it."""
    means: Dict[int, float] = {}
    for batch in BATCHES:
        values = [
            arms[(rnd, "SINGLE", batch)]["facts"]["compute_floor"]["step_s_mean"]
            for rnd in ROUNDS
            if (rnd, "SINGLE", batch) in arms
            and arms[(rnd, "SINGLE", batch)]["facts"].get("compute_floor")
        ]
        if values:
            means[SEQ * batch] = statistics.fmean(values)
    out: Dict[str, Any] = {"means": means}
    if len(means) >= 2:
        xs, ys = list(means), list(means.values())
        x_bar, y_bar = statistics.fmean(xs), statistics.fmean(ys)
        slope = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys)) / sum(
            (x - x_bar) ** 2 for x in xs
        )
        intercept = y_bar - slope * x_bar
        out["line"] = {"intercept_s": intercept, "slope_s_per_token": slope}
        out["max_residual_s"] = max(abs(y - (intercept + slope * x)) for x, y in zip(xs, ys))
    return out


def verdict(
    arms: Dict[Tuple[int, str, int], Dict[str, Any]],
    missing: List[str],
    outcome: str,
) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    """§2 applied row by row: ``([(row, result), ...], {layout: report})``."""
    out: List[Tuple[str, str]] = []
    low, high = SINGLE_VALID
    single_b1 = {
        rnd: arms[(rnd, "SINGLE", 1)]["facts"].get("step_s_mean")
        for rnd in ROUNDS
        if (rnd, "SINGLE", 1) in arms
    }
    outside = {rnd: v for rnd, v in single_b1.items() if v is None or not low <= v <= high}
    out.append(
        (f"1 SINGLE b1 outside {low}-{high} s", f"fires: {outside}" if outside else "no fire")
    )
    if outside:
        return out + [("VERDICT", "NO VERDICT (row 1): the session does not reproduce it")], {}
    problems = {rec["run"]: rec["problems"] for rec in arms.values() if rec["problems"]}
    row2 = []
    if problems:
        row2.append(f"off the path under test: {problems}")
    row2.extend(ids_problems(arms))
    if missing:
        row2.append(f"no valid arm: {missing}")
    if outcome != "complete":
        row2.append(f"protocol: {outcome}")
    text = "fires: " + "; ".join(row2) if row2 else "no fire"
    out.append(("2 off the path, or the protocol stopped", text))
    if row2:
        return out + [("VERDICT", "NO VERDICT (row 2): the arms did not measure the rule")], {}
    s_low, s_high = STRIPED_VALID
    striped_b1 = {
        rnd: arms[(rnd, "STRIPED", 1)]["facts"].get("step_s_mean")
        for rnd in ROUNDS
        if (rnd, "STRIPED", 1) in arms
    }
    striped_out = {rnd: v for rnd, v in striped_b1.items() if v is None or not s_low <= v <= s_high}
    out.append(
        (
            f"3 STRIPED b1 outside {s_low}-{s_high} s",
            f"fires: {striped_out}: NO VERDICT for STRIPED" if striped_out else "no fire",
        )
    )
    layouts = ["SINGLE"] + ([] if striped_out else ["STRIPED"])
    reports = {layout: layout_report(arms, layout) for layout in layouts}
    for layout in layouts:
        out.extend(_layout_rows(layout, reports[layout]))
    return out + [("VERDICT", "per layout: rows 4-6 above")], reports


def _layout_rows(layout: str, rep: Dict[str, Any]) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for batch, row in rep["batches"].items():
        result = row["result"] if "class" not in row else "fits"
        why = f": {row['why']}" if row.get("why") else ""
        rows.append((f"4 {layout} b{batch}", result + why))
    for batch, row in rep["batches"].items():
        if "class" not in row:
            continue
        per_round = ", ".join(f"r{rnd} {_fmt(value)}" for rnd, value in row["e_per_round"].items())
        rows.append(
            (
                f"5 {layout} b{batch}",
                f"{row['result']}: e {_fmt(row['e'])} ({per_round}), tok/s ratio {_fmt(row['r'])}",
            )
        )
    for batch, row in rep["batches"].items():
        if "agrees" not in row:
            continue
        text = "AGREES" if row["agrees"] else f"DISAGREES AT b{batch}"
        label = "".join(f" ({text})" for text in row["labels"])
        rows.append(
            (
                f"6 {layout} b{batch}",
                f"{text}{label}: measured {row['class']}, model {row['class_pred']} "
                f"(e_pred {_fmt(row['e_pred'])})",
            )
        )
    low, high = rep.get("crossover_bracket_tokens") or [None, None]
    rows.append(
        (
            f"6 {layout} crossover",
            f"measured ({low}, {high if high else 'not reached'}] tokens; model "
            f"{_fmt(rep.get('crossover_model_tokens'), 0)} tokens at R {_fmt(rep['R'])} s",
        )
    )
    return rows


def summarize(out_dir: str, prefix: str) -> int:
    arms, missing, outcome = collect(out_dir, prefix)
    print(f"sequence '{prefix}' in {out_dir}: outcome {outcome}")
    if not arms:
        print("no valid round arm on record")
    for (rnd, layout, batch), rec in sorted(arms.items()):
        facts, window, during = rec["facts"], rec["window"], rec["during"]
        floor = facts.get("compute_floor") or {}
        memory = facts.get("memory") or {}
        print(
            f"  r{rnd} {layout:<7} b{batch}  step {_fmt(facts.get('step_s_mean'))} s  tok/s "
            f"{_fmt(facts.get('tok_per_s'), 2)}  peak {_fmt(facts.get('peak_alloc_gb'))} / "
            f"{_fmt(facts.get('peak_reserved_gb'))} GB of {_fmt(facts.get('vram_total_gb'))}  "
            f"retries {memory.get('num_alloc_retries')}  in use {rec['used_mib']} MiB  "
            f"B {_fmt(floor.get('step_s_mean'))} s  fits {rec['fits'] and not rec['spill']} "
            f"{rec['fit_reason'] or rec['spill']}  problems {rec['problems'] or 'none'}"
        )
        print(
            f"      timed steps: power median/min {_fmt(window['power_w_median'], 1)} / "
            f"{_fmt(window['power_w_min'], 1)} W, SM median/min "
            f"{_fmt(window['sm_mhz_median'], 0)} / {_fmt(window['sm_mhz_min'], 0)} MHz, "
            f"utilisation {_fmt(window['util_pct_median'], 0)}%, {window['samples']} samples; "
            f"arm: process dedicated/shared {_fmt(during.get('process_dedicated_gb_max'))} / "
            f"{_fmt(during.get('process_shared_gb_max'))} GB, adapter shared first/max "
            f"{_fmt(during.get('adapter_shared_gb_first'))} / "
            f"{_fmt(during.get('adapter_shared_gb_max'))} GB"
        )
    floor = compute_floor(arms)
    for tokens, value in floor["means"].items():
        print(
            f"  read-free floor at {tokens} tokens: {value:.3f} s "
            f"(model {model_step_s(tokens):.3f} s)"
        )
    if "line" in floor:
        line = floor["line"]
        print(
            f"  in-session floor line: {line['intercept_s']:.3f} s + "
            f"{line['slope_s_per_token']:.6f} s x tokens (max residual "
            f"{floor['max_residual_s']:.3f} s)"
        )
    ids_b1 = (arms.get((0, "SINGLE", 1)) or {}).get("facts", {}).get("ids_row_sha256") or []
    for batch in BATCHES[1:]:
        rows = (arms.get((0, "SINGLE", batch)) or {}).get("facts", {}).get("ids_row_sha256")
        if rows and ids_b1:
            print(f"  ids: row 0 of the b{batch} draw equals the b1 draw: {rows[0] == ids_b1[0]}")
    rows, reports = verdict(arms, missing, outcome)
    for layout, rep in reports.items():
        if "line" in floor and rep.get("R") is not None:
            line = floor["line"]
            tokens = (rep["R"] - line["intercept_s"]) / line["slope_s_per_token"]
            print(f"  {layout}: crossover from the in-session floor line, {tokens:.0f} tokens")
    for row, result in rows:
        print(f"  row {row}: {result}")
    return 0
