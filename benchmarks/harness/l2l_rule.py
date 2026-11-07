#!/usr/bin/env python3
"""The L2L step-0 verdict, recomputed from the committed JSON alone.

Implements §2 of benchmarks/probe-rtx5070-l2l-step0.md: gates G1-G4, validity
rows V1-V6 and the informational rows. A gate whose inputs fail a validity row
reports "no verdict", never a pass. Standard library only.

Usage: python benchmarks/harness/l2l_rule.py <results dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

V2_LOADS = 0.5
V2_BYTES = 1e6
V4_TOL = 0.05
G2_RATIO = 0.8
G3_RATIO = 1.3
G4_GB = 0.5
I4_FREE = 0.95
PASS, FAIL, NONE = "pass", "fail", "no verdict"


def _load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(records: Sequence[Dict[str, Any]], mode: str, arm: str, k: int,
          variant: Optional[str] = None) -> List[Dict[str, Any]]:
    return [
        rec for rec in records
        if rec.get("kind") == "arm" and rec["mode"] == mode and rec["arm"] == arm
        and rec["k"] == k and rec.get("variant") == variant
    ]


def arm_value(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean of the two rounds, and why it is invalid if it is (V3 skip, V4, V5, V6)."""
    why: List[str] = []
    measured = [rec for rec in rows if not rec.get("skipped") and rec.get("step_s_mean")]
    for rec in rows:
        if rec.get("skipped"):
            why.append(f"skipped: {rec['skipped']}")
        elif (rec.get("suspend") or {}).get("void"):
            why.append(f"V5: round {rec['round']} void (sleep or battery)")
        elif rec.get("foreign_readers"):
            heavy = ", ".join(
                f"{row['name']} {row['read_bytes'] / 1e9:.2f} GB" for row in rec["foreign_readers"]
            )
            why.append(f"V6: round {rec['round']} foreign reader {heavy}")
    if len(measured) != 2:
        why.append(f"{len(measured)} measured rounds, need 2")
        return {"valid": False, "why": why, "value": None, "tok_s": None, "peak": None}
    first, second = (rec["step_s_mean"] for rec in measured)
    value = (first + second) / 2
    spread = abs(first - second) / value
    if spread > V4_TOL:
        why.append(f"V4: round spread {100 * spread:.1f}% > {100 * V4_TOL:.0f}%")
    return {
        "valid": not why,
        "why": why,
        "value": value,
        "spread": spread,
        "tok_s": measured[0]["tokens_per_step"] / value,
        "peak": max(rec["peak_alloc_gb"] for rec in measured),
    }


def _gate(status: str, detail: str) -> Dict[str, str]:
    return {"status": status, "detail": detail}


def _first(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    records = (payload or {}).get("records") or []
    return records[0] if records else None


def _not_exact(rec: Optional[Dict[str, Any]]) -> Optional[str]:
    """Why a correctness record cannot be judged (None when it can)."""
    if rec is None:
        return "no correctness record (a missing file, or a block that crashed)"
    if rec.get("skipped"):
        return f"skipped: {rec['skipped']}"
    if rec.get("deterministic_error"):
        return f"deterministic mode raised: {rec['deterministic_error']}"
    a_vs_a = rec.get("a_vs_a") or {}
    if not (a_vs_a.get("equal") and rec.get("a_vs_a_losses_equal")):
        return f"GA vs GA not exact (max |d| {float(a_vs_a.get('max_abs', float('nan'))):.3e})"
    if not rec.get("l2l"):
        return "no L2L comparison in the record"
    return None


def _g1_fixture(default: Optional[Dict[str, Any]], det: Optional[Dict[str, Any]],
                name: str) -> Dict[str, str]:
    chosen = _first(default)
    why = _not_exact(chosen)
    if why is not None:
        det_rec = _first(det)
        det_why = _not_exact(det_rec) if det is not None else "missing"
        if det_why is not None:
            return _gate(NONE, f"{name}: {why}; deterministic re-run: {det_why}")
        chosen = det_rec
    zero = chosen["a_vs_a"]["all_zero"]
    if zero:
        return _gate(FAIL, f"{name}: all-zero reference gradients {zero[:3]}")
    mode = "deterministic" if chosen.get("deterministic") else "default kernels"
    if chosen["l2l"]["equal"] and chosen["l2l_losses_equal"]:
        return _gate(PASS, f"{name}: {chosen['l2l']['tensors']} tensors + losses equal ({mode})")
    return _gate(FAIL, f"{name}: {len(chosen['l2l']['unequal'])} tensors differ, max |d| "
                       f"{chosen['l2l']['max_abs']:.3e}, losses equal "
                       f"{chosen['l2l_losses_equal']} ({mode})")


def _combine(parts: Sequence[Dict[str, str]]) -> Dict[str, str]:
    statuses = [part["status"] for part in parts]
    status = FAIL if FAIL in statuses else (NONE if NONE in statuses else PASS)
    return _gate(status, "; ".join(part["detail"] for part in parts))


def evaluate(results_dir: str) -> Dict[str, Any]:
    root = Path(results_dir)
    g1 = _combine([
        _g1_fixture(_load(root / f"correctness_{tag}.json"),
                    _load(root / f"correctness_{tag}_det.json"), tag.upper())
        for tag in ("f1", "f2")
    ])

    timing = _load(root / "timing.json") or {"meta": {}, "records": []}
    recs = timing["records"]
    meta = timing.get("meta", {})
    arms = {key: arm_value(_rows(recs, *key)) for key in [
        ("plain", "A", 1), ("plain", "B", 1), ("l2l", "A", 1), ("l2l", "B", 1),
        ("l2l", "A", 16), ("l2l", "B", 16), ("l2l", "A", 4), ("l2l", "B", 4),
        ("l2l", "C", 4), ("l2l", "D", 4), ("l2l", "A", 2), ("l2l", "A", 3),
    ]}
    v1 = {"ok": meta.get("direct_io") is True, "detail": f"direct_io={meta.get('direct_io')}"}
    v3 = {"ok": bool((meta.get("v3") or {}).get("ok")), "detail": str(meta.get("v3"))}

    def work(mode: str, arm: str) -> Optional[Tuple[float, float]]:
        """Mean loads and bytes per step over the rounds that ran. Timing validity (V4,
        V5, V6) does not enter: V2 is about the work, and a gate that reads an arm's
        time already carries that arm's own validity."""
        rows = [rec for rec in _rows(recs, mode, arm, 1)
                if not rec.get("skipped") and rec.get("step_s_mean")]
        if not rows:
            return None
        loads = sum(rec.get("layer_loads_per_step", 0.0) + rec.get("large_loads_per_step", 0.0)
                    for rec in rows) / len(rows)
        moved = sum(rec.get("bytes_moved_per_step", 0.0) for rec in rows) / len(rows)
        return loads, moved

    def same_work(arm: str) -> Tuple[bool, str]:
        """V2: L2L at k=1 moves exactly the shipped batch-1 step's loads and bytes."""
        l2l, plain = work("l2l", arm), work("plain", arm)
        if l2l is None or plain is None:
            return False, f"{arm}: L2L at k=1 or the shipped step has no measured round"
        ok = abs(l2l[0] - plain[0]) <= V2_LOADS and abs(l2l[1] - plain[1]) <= V2_BYTES
        return ok, (f"{arm}: loads {l2l[0]:.1f} vs {plain[0]:.1f}, "
                    f"bytes {l2l[1] / 1e9:.3f} vs {plain[1] / 1e9:.3f} GB")

    (ok_a, detail_a), (ok_b, detail_b) = same_work("A"), same_work("B")
    v2 = {"ok": ok_a and ok_b, "detail": f"{detail_a}; {detail_b}"}
    global_ok = v1["ok"] and v2["ok"] and v3["ok"]
    global_why = [name for name, row in (("V1", v1), ("V2", v2), ("V3", v3)) if not row["ok"]]

    def timing_gate(needed: Sequence[Tuple[str, str, int]], test, detail) -> Dict[str, str]:
        bad = [f"{key}: {', '.join(arms[key]['why'])}" for key in needed if not arms[key]["valid"]]
        if not global_ok or bad:
            return _gate(NONE, "; ".join(global_why + bad))
        return _gate(PASS if test() else FAIL, detail())

    a16, b16, a1 = ("l2l", "A", 16), ("l2l", "B", 16), ("l2l", "A", 1)
    g2 = timing_gate([a16, b16], lambda: arms[a16]["tok_s"] >= G2_RATIO * arms[b16]["tok_s"],
                     lambda: f"T_A(16) {arms[a16]['tok_s']:.1f} vs 0.8 x T_B(16) "
                             f"{G2_RATIO * arms[b16]['tok_s']:.1f} tok/s")
    g3 = timing_gate([a16, a1], lambda: arms[a16]["tok_s"] >= G3_RATIO * arms[a1]["tok_s"],
                     lambda: f"T_A(16) {arms[a16]['tok_s']:.1f} vs 1.3 x T_A(1) "
                             f"{G3_RATIO * arms[a1]['tok_s']:.1f} tok/s")
    g4 = timing_gate([a16, a1], lambda: arms[a16]["peak"] <= arms[a1]["peak"] + G4_GB,
                     lambda: f"peak A16 {arms[a16]['peak']:.3f} vs A1 {arms[a1]['peak']:.3f} "
                             f"+ {G4_GB} GB")
    gates = {"G1": g1, "G2": g2, "G3": g3, "G4": g4}

    info: Dict[str, Any] = {}
    ratios = {}
    for arm in ("A", "B"):
        l2l, plain = arms[("l2l", arm, 1)], arms[("plain", arm, 1)]
        if l2l["valid"] and plain["valid"]:
            ratios[arm.lower()] = l2l["value"] / plain["value"]
    if ratios:
        info["I6"] = ratios
    b4, d4, c4, a4 = (arms[("l2l", x, 4)] for x in ("B", "D", "C", "A"))
    if b4["valid"] and d4["valid"]:
        c_over_a = c4["value"] / a4["value"] if c4["valid"] and a4["valid"] else None
        info["I2"] = {"dequant_share": (b4["value"] - d4["value"]) / b4["value"],
                      "c4_over_a4": c_over_a}
    p1b = arms[("plain", "B", 1)]
    if arms[b16]["valid"] and p1b["valid"]:
        info["I3"] = {"overhead": arms[b16]["value"] / (16 * p1b["value"])}
    spill = _load(root / "spill.json")
    if spill:
        pinned = arm_value(_rows(spill["records"], "l2l", "A", 16, "pinned"))
        spilled = arm_value(_rows(spill["records"], "l2l", "A", 16, "spill"))
        if pinned["valid"] and spilled["valid"]:
            ratio = spilled["tok_s"] / pinned["tok_s"]
            info["I4"] = {"ratio": ratio, "free": ratio >= I4_FREE,
                          "direct": spill.get("meta", {}).get("spill_direct")}
    info["arms"] = {f"{m}/{a}/k{k}": row for (m, a, k), row in arms.items()}

    statuses = {name: gate["status"] for name, gate in gates.items()}
    if all(status == PASS for status in statuses.values()):
        verdict = "BUILD STEP 1"
    elif FAIL in statuses.values():
        verdict = "STOP: " + ", ".join(n for n, s in statuses.items() if s == FAIL) + " failed"
    else:
        verdict = "NO VERDICT: " + ", ".join(n for n, s in statuses.items() if s == NONE)
    return {"gates": gates, "validity": {"V1": v1, "V2": v2, "V3": v3}, "info": info,
            "verdict": verdict}


def main(argv: Sequence[str]) -> int:
    if len(argv) != 2:
        print("usage: l2l_rule.py <results dir>")
        return 2
    out = evaluate(argv[1])
    print("| gate | status | detail |\n|---|---|---|")
    for name, gate in out["gates"].items():
        print(f"| {name} | {gate['status']} | {gate['detail']} |")
    print("\n| row | ok | detail |\n|---|---|---|")
    for name, row in out["validity"].items():
        print(f"| {name} | {row['ok']} | {row['detail']} |")
    print("\n| arm | valid | step s | tok/s | peak GB | why |\n|---|---|---|---|---|---|")
    for name, row in out["info"]["arms"].items():
        value = f"{row['value']:.3f}" if row["value"] else "-"
        tok = f"{row['tok_s']:.1f}" if row["tok_s"] else "-"
        peak = f"{row['peak']:.3f}" if row["peak"] else "-"
        print(f"| {name} | {row['valid']} | {value} | {tok} | {peak} | {'; '.join(row['why'])} |")
    for key in ("I2", "I3", "I4", "I6"):
        if key in out["info"]:
            print(f"\n{key}: {out['info'][key]}")
    print(f"\n**Verdict: {out['verdict']}**")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
