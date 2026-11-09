#!/usr/bin/env python3
"""Gate values G1..G4, the ablation table and the spreads (protocol section 6), as written.

Reads ``<results>/<shape>/<budget_gb>/<arm>/rep<k>.json`` (written by ``orchestrate.py``) and
writes ``<results>/gates.json`` plus a plain-text summary (``gates.txt``, also printed).

Definitions used, all from the protocol:

* the value of an arm = the median tok/s of its valid repeats; spread = (max - min) / median;
* a run counts when its status is "ok" and it is not void; an arm whose runs all ran out of
  memory is "does not fit" at that budget; an arm with two void attempts of one repeat is
  "not measured";
* G1: median tok/s(S1) / median tok/s(D1), Shape A, 80 GB. CONFIRMED if >= 2.64, else LOWER.
  Also S0 / D1;
* G2: median tok/s(SB) / max median tok/s over the DeepSpeed arms D1..D4 that fit, per budget.
  No pass mark. The prefetch variants (D1PF, D2PF: D1 / D2 with a 5e8 prefetch bucket) are
  reported next to it as ``ratio_with_prefetch_variants`` because they are the same arms
  tuned; the strict D1..D4 value is the protocol's;
* G3: R1 / SB and R2 / SB at 80 GB, Shape A;
* G4: for Shapes B and C at 80 GB: SB / D-best, SB / D1 and R1 / SB with both peak VRAM
  values. HOLDS if SB / D-best >= 2.0, SMALLER if 1.0 <= SB / D-best < 2.0, REVERSED below
  1.0. D-best is the fastest DeepSpeed arm of Shape A at 80 GB (``d_best.json`` when the
  orchestrator wrote it, else recomputed here);
* ablation: change of median tok/s against S1 in percent, with S1's spread; a change inside
  S1's spread is "no measurable effect";
* near-miss: a gate value within 2% of its threshold is flagged and not rounded either way.

Usage: ``python gates.py <results dir>``
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ds_common import atomic_write_json, median, read_json, spread  # noqa: E402

G1_THRESHOLD = 2.64
G4_HOLDS = 2.0
G4_SMALLER_FLOOR = 1.0
NEAR_MISS = 0.02
CLOCK_TOLERANCE = 0.05

DS_ARMS = ("D1", "D2", "D3", "D4")
DS_VARIANTS = ("D1PF", "D2PF")
ABLATION_ARMS = ("S2", "S3", "S4", "S5", "S6", "S7")

_RUN_FILE = re.compile(r"^rep(\d+)\.json$")


def load_runs(results_dir: str | Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    root = Path(results_dir)
    for path in sorted(root.glob("*/*/*/rep*.json")):
        if not _RUN_FILE.match(path.name):
            continue
        try:
            run = read_json(path)
        except (OSError, ValueError):
            continue
        run.setdefault("shape", path.parts[-4])
        run.setdefault("budget_gb", int(path.parts[-3]))
        run.setdefault("arm", path.parts[-2])
        run["_path"] = str(path)
        runs.append(run)
    return runs


def _peak_mib(run: dict[str, Any]) -> float | None:
    sampler = run.get("sampler") or {}
    return (sampler.get("whole_run") or {}).get("vram_peak_mib")


def _valid(run: dict[str, Any]) -> bool:
    return run.get("status") == "ok" and not run.get("void") and run.get("tok_per_s") is not None


def arm_stats(runs: list[dict[str, Any]], shape: str, budget: int, arm: str) -> dict[str, Any]:
    group = [
        run
        for run in runs
        if run["shape"] == shape and run["budget_gb"] == budget and run["arm"] == arm
    ]
    valid = [run for run in group if _valid(run)]
    stats: dict[str, Any] = {
        "arm": arm,
        "shape": shape,
        "budget_gb": budget,
        "runs_found": len(group),
        "runs_valid": len(valid),
    }
    if valid:
        values = [run["tok_per_s"] for run in valid]
        peaks = [peak for peak in (_peak_mib(run) for run in valid) if peak is not None]
        reserved = [
            run["max_memory_reserved_bytes"]
            for run in valid
            if run.get("max_memory_reserved_bytes") is not None
        ]
        steady = [run["steady_tok_per_s"] for run in valid if run.get("steady_tok_per_s")]
        stats.update(
            {
                "status": "measured",
                "tok_per_s_values": values,
                "tok_per_s_median": median(values),
                "tok_per_s_min": min(values),
                "tok_per_s_max": max(values),
                "spread": spread(values),
                "steady_tok_per_s_median": median(steady) if steady else None,
                "peak_vram_mib_median": median(peaks) if peaks else None,
                "max_memory_reserved_gib_median": (
                    median(reserved) / 1024**3 if reserved else None
                ),
                "final_loss_median": median(
                    [run["final_loss"] for run in valid if run.get("final_loss") is not None]
                )
                if any(run.get("final_loss") is not None for run in valid)
                else None,
            }
        )
    elif any(run.get("status") == "oom" for run in group):
        stats["status"] = "does not fit"
    elif any(run.get("void") for run in group):
        stats["status"] = "not measured"
    else:
        stats["status"] = "not run" if not group else "failed"
    return stats


def _median(stats: dict[str, Any]) -> float | None:
    return stats.get("tok_per_s_median") if stats.get("status") == "measured" else None


def _ratio(numerator: dict[str, Any], denominator: dict[str, Any]) -> float | None:
    top, bottom = _median(numerator), _median(denominator)
    if top is None or not bottom:
        return None
    return top / bottom


def _near(value: float | None, threshold: float) -> bool:
    return value is not None and abs(value - threshold) <= NEAR_MISS * threshold


def _brief(stats: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "arm",
        "status",
        "runs_valid",
        "tok_per_s_median",
        "spread",
        "peak_vram_mib_median",
        "max_memory_reserved_gib_median",
    )
    return {key: stats.get(key) for key in keys}


def best_deepspeed(
    runs: list[dict[str, Any]], shape: str, budget: int, arms: tuple[str, ...]
) -> dict[str, Any] | None:
    measured = [
        stats
        for stats in (arm_stats(runs, shape, budget, arm) for arm in arms)
        if stats["status"] == "measured"
    ]
    if not measured:
        return None
    return max(measured, key=lambda stats: stats["tok_per_s_median"])


def compute_g1(runs: list[dict[str, Any]]) -> dict[str, Any]:
    s1 = arm_stats(runs, "A", 80, "S1")
    s0 = arm_stats(runs, "A", 80, "S0")
    d1 = arm_stats(runs, "A", 80, "D1")
    ratio = _ratio(s1, d1)
    verdict = None if ratio is None else ("CONFIRMED" if ratio >= G1_THRESHOLD else "LOWER")
    return {
        "definition": "median tok/s(S1) / median tok/s(D1), Shape A, 80 GB",
        "threshold": G1_THRESHOLD,
        "ratio": ratio,
        "verdict": verdict,
        "near_miss": _near(ratio, G1_THRESHOLD),
        "S1": _brief(s1),
        "D1": _brief(d1),
        "S0_over_D1": _ratio(s0, d1),
        "S0": _brief(s0),
    }


def compute_g2(runs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for budget in (80, 24, 16):
        sb = arm_stats(runs, "A", budget, "SB")
        strict = best_deepspeed(runs, "A", budget, DS_ARMS)
        with_variants = best_deepspeed(runs, "A", budget, DS_ARMS + DS_VARIANTS)
        attempted = [
            arm
            for arm in DS_ARMS + DS_VARIANTS
            if arm_stats(runs, "A", budget, arm)["runs_found"] > 0
        ]
        entry: dict[str, Any] = {
            "budget_gb": budget,
            "SB": _brief(sb),
            "ds_arms_attempted": attempted,
        }
        if strict is None:
            entry["ratio"] = None
            entry["note"] = (
                f"DeepSpeed does not fit at {budget} GB"
                if attempted
                else "no DeepSpeed arm was run at this budget"
            )
        else:
            entry["best_deepspeed"] = _brief(strict)
            entry["ratio"] = _ratio(sb, strict)
        if with_variants is not None:
            entry["best_deepspeed_with_prefetch_variants"] = _brief(with_variants)
            entry["ratio_with_prefetch_variants"] = _ratio(sb, with_variants)
        out[str(budget)] = entry
    return out


def compute_g3(runs: list[dict[str, Any]]) -> dict[str, Any]:
    sb = arm_stats(runs, "A", 80, "SB")
    out: dict[str, Any] = {"SB": _brief(sb)}
    for arm in ("R1", "R2"):
        resident = arm_stats(runs, "A", 80, arm)
        ratio = _ratio(resident, sb)
        out[arm] = {
            "stats": _brief(resident),
            "ratio_over_SB": ratio,
            "reading": None
            if ratio is None
            else ("resident faster than SB" if ratio > 1 else "SB at least as fast as resident"),
        }
    return out


def d_best_arm(results_dir: str | Path, runs: list[dict[str, Any]]) -> str | None:
    path = Path(results_dir) / "d_best.json"
    if path.exists():
        try:
            return read_json(path).get("arm")
        except (OSError, ValueError):
            pass
    best = best_deepspeed(runs, "A", 80, DS_ARMS + DS_VARIANTS)
    return best["arm"] if best else None


def compute_g4(runs: list[dict[str, Any]], d_best: str | None) -> dict[str, Any]:
    out: dict[str, Any] = {"d_best_arm": d_best}
    for shape in ("B", "C"):
        sb = arm_stats(runs, shape, 80, "SB")
        s1 = arm_stats(runs, shape, 80, "S1")
        d1 = arm_stats(runs, shape, 80, "D1")
        r1 = arm_stats(runs, shape, 80, "R1")
        best = arm_stats(runs, shape, 80, d_best) if d_best else None
        ratio_best = _ratio(sb, best) if best else None
        if ratio_best is None:
            verdict = None
        elif ratio_best >= G4_HOLDS:
            verdict = "HOLDS"
        elif ratio_best >= G4_SMALLER_FLOOR:
            verdict = "SMALLER"
        else:
            verdict = "REVERSED"
        out[shape] = {
            "SB_over_Dbest": ratio_best,
            "SB_over_D1": _ratio(sb, d1),
            "R1_over_SB": _ratio(r1, sb),
            "S1_over_D1": _ratio(s1, d1),
            "verdict": verdict,
            "near_miss": _near(ratio_best, G4_HOLDS) or _near(ratio_best, G4_SMALLER_FLOOR),
            "SB": _brief(sb),
            "S1": _brief(s1),
            "D_best": _brief(best) if best else None,
            "D1": _brief(d1),
            "R1": _brief(r1),
            "note": (
                "SB rerun with stream_recompute on after an out-of-memory first attempt"
                if any(
                    run.get("rerun_recompute_on")
                    for run in runs
                    if run["shape"] == shape and run["arm"] == "SB"
                )
                else None
            ),
        }
    return out


def compute_ablation(runs: list[dict[str, Any]]) -> dict[str, Any]:
    s1 = arm_stats(runs, "A", 80, "S1")
    table: dict[str, Any] = {"S1": _brief(s1), "arms": {}}
    base = _median(s1)
    s1_spread = s1.get("spread")
    for arm in ABLATION_ARMS + ("S8",):
        stats = arm_stats(runs, "A", 80, arm)
        median_value = _median(stats)
        if base is None or median_value is None or s1_spread is None:
            change, reading = None, "not measured"
        else:
            change = (median_value - base) / base * 100.0
            reading = (
                "no measurable effect"
                if abs(change) <= s1_spread * 100.0
                else ("faster" if change > 0 else "slower")
            )
        table["arms"][arm] = {
            "stats": _brief(stats),
            "change_pct_vs_S1": change,
            "S1_spread_pct": None if s1_spread is None else s1_spread * 100.0,
            "reading": reading,
            "informational_only": arm == "S8",
        }
    return table


def clock_report(runs: list[dict[str, Any]]) -> dict[str, Any]:
    clocks = []
    for run in runs:
        window = (run.get("sampler") or {}).get("train_window") or {}
        value = window.get("sm_clock_busy_median_mhz")
        if value:
            clocks.append((run["_path"], value))
    if not clocks:
        return {"session_median_mhz": None, "flagged_valid_runs": []}
    session = median([value for _, value in clocks])
    flagged = [
        path
        for path, value in clocks
        if value < session * (1 - CLOCK_TOLERANCE)
        and _valid(next(run for run in runs if run["_path"] == path))
    ]
    return {"session_median_mhz": session, "flagged_valid_runs": flagged}


def compute_all(results_dir: str | Path) -> dict[str, Any]:
    runs = load_runs(results_dir)
    d_best = d_best_arm(results_dir, runs)
    return {
        "runs_loaded": len(runs),
        "G1": compute_g1(runs),
        "G2": compute_g2(runs),
        "G3": compute_g3(runs),
        "G4": compute_g4(runs, d_best),
        "ablation": compute_ablation(runs),
        "clock": clock_report(runs),
    }


def _fmt(value: Any, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_summary(gates: dict[str, Any]) -> str:
    lines: list[str] = []
    g1 = gates["G1"]
    flag = " (NEAR MISS: within 2% of the threshold)" if g1["near_miss"] else ""
    lines.append(
        f"G1  S1/D1 = {_fmt(g1['ratio'])}  threshold {g1['threshold']}  "
        f"{g1['verdict'] or 'NOT COMPUTED'}{flag}   S0/D1 = {_fmt(g1['S0_over_D1'])}"
    )
    lines.append(
        f"    S1 {_fmt(g1['S1'].get('tok_per_s_median'))} tok/s  "
        f"D1 {_fmt(g1['D1'].get('tok_per_s_median'))} tok/s"
    )
    for budget, entry in gates["G2"].items():
        text = entry.get("note") or f"SB/DS-best = {_fmt(entry['ratio'])}"
        best = entry.get("best_deepspeed") or {}
        lines.append(
            f"G2  {budget} GB  {text}  (SB {_fmt(entry['SB'].get('tok_per_s_median'))} tok/s, "
            f"best DeepSpeed {best.get('arm')} {_fmt(best.get('tok_per_s_median'))} tok/s, "
            f"peak VRAM MiB {_fmt(entry['SB'].get('peak_vram_mib_median'), 0)} / "
            f"{_fmt(best.get('peak_vram_mib_median'), 0)}; with prefetch variants "
            f"{_fmt(entry.get('ratio_with_prefetch_variants'))})"
        )
    g3 = gates["G3"]
    for arm in ("R1", "R2"):
        lines.append(
            f"G3  {arm}/SB = {_fmt(g3[arm]['ratio_over_SB'])}  (peak VRAM MiB "
            f"{_fmt(g3[arm]['stats'].get('peak_vram_mib_median'), 0)} / "
            f"{_fmt(g3['SB'].get('peak_vram_mib_median'), 0)})"
        )
    g4 = gates["G4"]
    for shape in ("B", "C"):
        entry = g4[shape]
        flag = " (NEAR MISS)" if entry["near_miss"] else ""
        lines.append(
            f"G4  shape {shape}  SB/D-best({g4['d_best_arm']}) = {_fmt(entry['SB_over_Dbest'])}  "
            f"SB/D1 = {_fmt(entry['SB_over_D1'])}  R1/SB = {_fmt(entry['R1_over_SB'])}  "
            f"{entry['verdict'] or 'NOT COMPUTED'}{flag}"
        )
        peaks = {
            name: _fmt((entry[name] or {}).get("peak_vram_mib_median"), 0)
            for name in ("SB", "D_best", "D1", "R1")
        }
        lines.append(f"    peak VRAM MiB: {peaks}")
    ablation = gates["ablation"]
    lines.append(
        f"Ablation vs S1 ({_fmt(ablation['S1'].get('tok_per_s_median'))} tok/s, spread "
        f"{_fmt((ablation['S1'].get('spread') or 0) * 100, 1)}%)"
    )
    for arm, entry in ablation["arms"].items():
        suffix = " [informational]" if entry["informational_only"] else ""
        lines.append(
            f"    {arm}: {_fmt(entry['change_pct_vs_S1'], 1)}%  {entry['reading']}{suffix}"
        )
    clock = gates["clock"]
    lines.append(
        f"SM clock session median {_fmt(clock['session_median_mhz'], 0)} MHz; valid runs more "
        f"than 5% below it: {len(clock['flagged_valid_runs'])}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(__doc__)
        return 2
    results = Path(args[0])
    gates = compute_all(results)
    atomic_write_json(results / "gates.json", gates)
    summary = render_summary(gates)
    (results / "gates.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)
    print(f"\nwrote {results / 'gates.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
