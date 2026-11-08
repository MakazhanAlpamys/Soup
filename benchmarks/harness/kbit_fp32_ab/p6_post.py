#!/usr/bin/env python3
"""Post-processing of the pulled p6 results (issue #2609 step 1).  Runs on the laptop; reads only the pulled tree.

    python -I p6_post.py --runs <pulled out/runs dir> --analysis <analysis.json> \
        --out <results dir>

Cells are named <ARM>-r<round>-<P|T>.  T cells give timing, clocks, peak memory and loss; P cells give the category table, the
per-phase memory peaks, the parameter dtypes and the allocator-snapshot blocks (analysis.json comes from case1-profile/analyze.py).
Writes results/timing.json, loss.json, profile.json, tables.md.  Every number printed comes from a file under --runs / --analysis.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics

WARM = 5
WIN = 100
TOK_PER_STEP = 8 * 512
ARMS = ["A", "B", "C", "D"]
TOL_STEP1 = 0.005
TOL_LAST16 = 0.01
ISSUE = {"ew_soup_ms": 915.6, "ew_ref_ms": 97.8, "ew_gap_ms": 817.8, "ew_fp32_in_ms": 837.0, "peak_mib": 17425,
         "step_s": 2.270, "gap_mib": 10264, "logits3_mib": 7128}


def load_json(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def read_jsonl(path):
    rows = []
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        break
    except OSError:
        pass
    return rows


def pct(values, q):
    vals = sorted(values)
    if not vals:
        return None
    idx = (len(vals) - 1) * q
    lo, hi = int(idx), min(int(idx) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (idx - lo)


def cell_name_parts(name):
    arm, rnd, kind = name.split("-")
    return arm, int(rnd[1:]), kind


def timing_cell(cdir):
    steps = read_jsonl(os.path.join(cdir, "steps.jsonl"))
    logs = read_jsonl(os.path.join(cdir, "log.jsonl"))
    begin = load_json(os.path.join(cdir, "train_begin.json")) or {}
    samples = (load_json(os.path.join(cdir, "gpu_samples.json")) or {}).get("rows", [])
    result = load_json(os.path.join(cdir, "result.json")) or {}
    out = {"n_steps": len(steps), "status": result.get("status"), "reason": result.get("reason"),
           "seed": result.get("seed")}
    if len(steps) < WARM + WIN:
        out["complete"] = False
        return out, logs
    meas = steps[WARM:WARM + WIN]
    dts = [s["t_end"] - s["t_begin"] for s in meas]
    med = statistics.median(dts)
    out.update({"complete": True, "median_step_s": med, "mean_step_s": statistics.mean(dts), "min_step_s": min(dts),
                "max_step_s": max(dts), "p10_step_s": pct(dts, 0.1), "p90_step_s": pct(dts, 0.9),
                "tok_s_median_step": TOK_PER_STEP / med,
                "tok_s_wall": (result.get("throughput") or {}).get("tok_s_wall"),
                "first_half_median_s": statistics.median(dts[:WIN // 2]),
                "second_half_median_s": statistics.median(dts[WIN // 2:]),
                "tokens_per_step": sorted({s["tokens"] for s in meas}),
                "peak_alloc_mib": max(s.get("alloc_peak_mib") or 0 for s in steps),
                "peak_reserved_mib": max(s.get("reserved_peak_mib") or 0 for s in steps),
                "fp": [s.get("fp") for s in steps]})
    if begin.get("unix") is not None and samples:
        to_unix = lambda perf: begin["unix"] + (perf - begin["perf"])  # noqa: E731
        lo, hi = to_unix(meas[0]["t_begin"]), to_unix(meas[-1]["t_end"])
        window = [row for row in samples if lo <= row[0] <= hi]
        clocks = [row[2] for row in window if row[2] is not None]
        power = [row[3] for row in window if row[3] is not None]
        temp = [row[4] for row in window if row[4] is not None]
        util = [row[5] for row in window if row[5] is not None]
        out["window_s"] = hi - lo
        out["gpu_samples_in_window"] = len(window)
        if clocks:
            out.update({"sm_clock_median_mhz": statistics.median(clocks), "sm_clock_min_mhz": min(clocks),
                        "sm_clock_p10_mhz": pct(clocks, 0.1), "sm_clock_max_mhz": max(clocks)})
        if power:
            out.update({"power_median_w": statistics.median(power), "power_max_w": max(power)})
        if temp:
            out["temp_max_c"] = max(temp)
        if util:
            out["util_median_pct"] = statistics.median(util)
    curve = [(int(r["step"]), float(r["loss"])) for r in logs if r.get("loss") is not None]
    out["loss_n"] = len(curve)
    if curve:
        out["loss_step1"] = curve[0][1]
        out["loss_last16"] = statistics.mean(v for _, v in curve[-16:])
        out["loss_first9"] = [v for _, v in curve[:9]]
        out["loss_curve"] = [v for _, v in curve]
    return out, logs


def build_timing(runs_dir):
    cells = {}
    for cdir in sorted(glob.glob(os.path.join(runs_dir, "*-r*-T"))):
        name = os.path.basename(cdir)
        arm, rnd, _ = cell_name_parts(name)
        info, _ = timing_cell(cdir)
        info.update({"arm": arm, "round": rnd})
        cells[name] = info
    return cells


def rel(x, ref):
    return None if (x is None or not ref) else (x - ref) / ref


def build_loss(cells):
    rows = []
    for name, c in sorted(cells.items()):
        if not c.get("complete"):
            continue
        base = cells.get(f"A-r{c['round']}-T")
        if not base or not base.get("complete"):
            continue
        same_batches = c["fp"] == base["fp"]
        rows.append({"cell": name, "arm": c["arm"], "round": c["round"], "seed": c.get("seed"),
                     "step1": c["loss_step1"], "A_step1": base["loss_step1"],
                     "step1_rel": rel(c["loss_step1"], base["loss_step1"]),
                     "last16": c["loss_last16"], "A_last16": base["loss_last16"],
                     "last16_rel": rel(c["loss_last16"], base["loss_last16"]),
                     "same_batches_as_A": same_batches})
    verdict = {}
    for arm in ARMS:
        mine = [r for r in rows if r["arm"] == arm]
        if not mine or arm == "A":
            continue
        s1 = [abs(r["step1_rel"]) for r in mine]
        l16 = [r["last16_rel"] for r in mine]
        verdict[arm] = {
            "rounds": [r["round"] for r in mine],
            "max_abs_step1_rel": max(s1), "step1_ok": all(v <= TOL_STEP1 for v in s1),
            "per_seed_last16_rel": l16, "mean_last16_rel": statistics.mean(l16),
            "per_seed_last16_ok": all(abs(v) <= TOL_LAST16 for v in l16),
            "mean_last16_ok": abs(statistics.mean(l16)) <= TOL_LAST16,
            "all_batches_identical_to_A": all(r["same_batches_as_A"] for r in mine)}
        verdict[arm]["inside_tolerance"] = (verdict[arm]["step1_ok"] and verdict[arm]["per_seed_last16_ok"]
                                            and verdict[arm]["mean_last16_ok"])
    return rows, verdict


def noise_floor(runs_dir):
    """Same seed, same arm: step-1..9 loss of the P cell against the T cell (non-determinism floor)."""
    out = []
    for pdir in sorted(glob.glob(os.path.join(runs_dir, "*-r*-P"))):
        name = os.path.basename(pdir)
        tdir = pdir[:-1] + "T"
        plog, tlog = read_jsonl(os.path.join(pdir, "log.jsonl")), read_jsonl(os.path.join(tdir, "log.jsonl"))
        pl = [float(r["loss"]) for r in plog if r.get("loss") is not None][:9]
        tl = [float(r["loss"]) for r in tlog if r.get("loss") is not None][:9]
        if len(pl) >= 1 and len(tl) >= 1:
            n = min(len(pl), len(tl))
            diffs = [abs(a - b) / ((abs(a) + abs(b)) / 2) for a, b in zip(pl[:n], tl[:n])]
            out.append({"cell": name, "n": n, "step1_rel": diffs[0], "max_rel_first9": max(diffs)})
    return out


def agg_timing(cells):
    per_arm = {}
    for arm in ARMS:
        mine = [c for c in cells.values() if c["arm"] == arm and c.get("complete")]
        if not mine:
            continue
        meds = [c["median_step_s"] for c in mine]
        per_arm[arm] = {
            "rounds": sorted(c["round"] for c in mine),
            "median_step_s_by_round": {c["round"]: c["median_step_s"] for c in sorted(mine, key=lambda x: x["round"])},
            "median_of_medians_s": statistics.median(meds),
            "range_rel": (max(meds) - min(meds)) / statistics.median(meds),
            "tok_s": TOK_PER_STEP / statistics.median(meds),
            "tok_s_wall_by_round": {c["round"]: c["tok_s_wall"] for c in sorted(mine, key=lambda x: x["round"])},
            "sm_clock_median_mhz_by_round": {c["round"]: c.get("sm_clock_median_mhz") for c in sorted(mine, key=lambda x: x["round"])},
            "power_median_w_by_round": {c["round"]: c.get("power_median_w") for c in sorted(mine, key=lambda x: x["round"])},
            "peak_alloc_mib": max(c["peak_alloc_mib"] for c in mine),
            "peak_alloc_mib_by_round": {c["round"]: c["peak_alloc_mib"] for c in sorted(mine, key=lambda x: x["round"])},
            "peak_reserved_mib_by_round": {c["round"]: c["peak_reserved_mib"] for c in sorted(mine, key=lambda x: x["round"])},
        }
    return per_arm


# ---------------------------------------------------------------- profile cells
def tot(d):
    return sum(d.values()) if d else 0.0


def profile_cell(pdir, analysis):
    name = os.path.basename(pdir)
    meta = load_json(os.path.join(pdir, "prof", "prof_meta.json")) or {}
    an = analysis.get(name)
    out = {"cell": name}
    if an:
        cats = an["categories_ms_per_step"]
        out["categories_ms"] = {k: tot(v) for k, v in cats.items()}
        out["gpu_busy_ms"] = an["gpu_busy_ms_per_step"]
        out["step_wall_ms"] = an["wall_ms_per_step"]
        out["idle_ms"] = an["idle_ms_per_step"]
        ew = an.get("elementwise_split", [])
        out["elementwise_fp32_in_ms"] = sum(r["ms_per_step"] for r in ew if r["dtype"] == "fp32 in")
        out["elementwise_other_in_ms"] = sum(r["ms_per_step"] for r in ew if r["dtype"] != "fp32 in")
        out["elementwise_split"] = ew
        out["ce_split"] = an.get("ce_split", [])
        out["rewind_recorder_gpu_ms"] = an.get("rewind_recorder_gpu_ms_per_step")
        out["top_kernels"] = an["kernels"][:12]
        out["memcpy"] = an.get("memcpy")
        out["recompute_over_forward"] = (sum(d.get("recompute", 0.0) for d in cats.values())
                                         / max(sum(d.get("fwd_layers", 0.0) for d in cats.values()), 1e-9))
    steps = meta.get("steps", {})
    phases = {}
    for key in ["pre_step", "pre_fwd", "fwd_pre_body", "fwd_body", "head_ce", "backward", "clip", "optimizer", "post"]:
        peaks = [st["mem"][key]["peak_mib"] for sid, st in steps.items() if int(sid) >= 3 and "mem" in st and key in st["mem"]]
        if peaks:
            phases[key] = max(peaks)
    out["phase_peaks_mib"] = phases
    out["step_peak_mib_phase_max"] = max(phases.values()) if phases else None
    out["after_load"] = meta.get("after_load")
    out["param_groups"] = (meta.get("params") or {}).get("groups")
    snap = meta.get("snapshot_summary") or {}
    out["snapshot"] = {"peak_total_mib_estimate": snap.get("peak_total_mib_estimate"),
                       "base_alloc_before_step_mib": snap.get("base_alloc_before_step_mib"),
                       "largest_blocks": snap.get("largest_blocks", [])[:12],
                       "by_stack_top3": snap.get("by_stack_top3", [])[:12]}
    out["unprofiled_wall_s_steps_2_5"] = [steps[k]["wall_s"] for k in ("2", "3", "4", "5") if k in steps and "wall_s" in steps[k]]
    out["profiled_wall_s_steps_6_8"] = [steps[k]["wall_s"] for k in ("6", "7", "8") if k in steps and "wall_s" in steps[k]]
    arm_info = load_json(os.path.join(pdir, "arm_info.json")) or {}
    out["arm_info"] = {"prepare_calls": arm_info.get("prepare_calls"),
                       "events": [e["msg"] for e in arm_info.get("events", [])],
                       "dtypes_at_construction": (arm_info.get("census_at_trainer_construction") or {}).get("dtypes"),
                       "dtypes_at_train_end": (arm_info.get("census_at_train_end") or {}).get("dtypes"),
                       "module_classes": (arm_info.get("census_at_train_end") or {}).get("module_classes"),
                       "causal_lm_forward": (arm_info.get("census_at_train_end") or {}).get("causal_lm_forward"),
                       "use_liger_kernel_arg": arm_info.get("use_liger_kernel_arg")}
    return out


def build_profile(runs_dir, analysis):
    return {os.path.basename(p): profile_cell(p, analysis) for p in sorted(glob.glob(os.path.join(runs_dir, "*-r*-P")))}


def arm_mean(profile, arm, getter):
    vals = [getter(c) for name, c in profile.items() if name.startswith(arm + "-") and getter(c) is not None]
    return (statistics.mean(vals), vals) if vals else (None, [])


# ---------------------------------------------------------------- markdown
def f1(x, nd=1):
    return "n/a" if x is None else f"{x:,.{nd}f}"


def fp(x, nd=2):
    return "n/a" if x is None else f"{100 * x:+.{nd}f}%"


def tables(timing_cells, per_arm, loss_rows, loss_verdict, noise, profile):
    L = []
    w = L.append
    w("### T1. Unprofiled timing: median step over steps 6-105 of each T cell (one cell per round; a round = one seed)\n")
    w("| arm | round (seed) | median step s | tok/s (4096 / median) | tok/s wall (harness) | p10 / p90 step s | 2nd half / 1st half | SM clock median / min MHz | power median / max W | temp max C | peak alloc MiB | peak reserved MiB |")
    w("|---|---|---:|---:|---:|---|---:|---|---|---:|---:|---:|")
    for name, c in sorted(timing_cells.items(), key=lambda kv: (kv[1]["round"], kv[1]["arm"])):
        if not c.get("complete"):
            w(f"| {c['arm']} | r{c['round']} | not complete: {c.get('status')} {c.get('reason')} ({c['n_steps']} steps) | | | | | | | | | |")
            continue
        w(f"| {c['arm']} | r{c['round']} ({c.get('seed')}) | {c['median_step_s']:.4f} | {c['tok_s_median_step']:,.0f} | {f1(c['tok_s_wall'], 0)} | "
          f"{c['p10_step_s']:.3f} / {c['p90_step_s']:.3f} | {c['second_half_median_s'] / c['first_half_median_s']:.3f} | "
          f"{f1(c.get('sm_clock_median_mhz'), 0)} / {f1(c.get('sm_clock_min_mhz'), 0)} | {f1(c.get('power_median_w'), 0)} / {f1(c.get('power_max_w'), 0)} | "
          f"{f1(c.get('temp_max_c'), 0)} | {c['peak_alloc_mib']:,.0f} | {c['peak_reserved_mib']:,.0f} |")
    w("\n### T2. Per arm over rounds (median of the per-cell medians; range = (max-min)/median across rounds)\n")
    w("| arm | rounds | median step s | tok/s | range across rounds | vs A step time | SM clock median MHz (by round) | peak allocated MiB | peak reserved MiB (by round) |")
    w("|---|---|---:|---:|---:|---:|---|---:|---|")
    a_med = per_arm.get("A", {}).get("median_of_medians_s")
    for arm in ARMS:
        a = per_arm.get(arm)
        if not a:
            w(f"| {arm} | none | not run / no complete cell | | | | | | |")
            continue
        clocks = ", ".join(f1(v, 0) for v in a["sm_clock_median_mhz_by_round"].values())
        reserved = ", ".join(f"{v:,.0f}" for v in a["peak_reserved_mib_by_round"].values())
        w(f"| {arm} | {','.join(map(str, a['rounds']))} | {a['median_of_medians_s']:.4f} | {a['tok_s']:,.0f} | {100 * a['range_rel']:.1f}% | "
          f"{fp(rel(a['median_of_medians_s'], a_med))} | {clocks} | {a['peak_alloc_mib']:,.0f} | {reserved} |")
    w("\n### L1. Loss against arm A of the same seed (tolerance: step-1 <= 0.5%, last-16 mean <= 1.0% per seed and in the mean)\n")
    w("| arm | round (seed) | step-1 loss | A step-1 | delta | last-16 mean | A last-16 | delta | same batches as A |")
    w("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for r in sorted(loss_rows, key=lambda x: (x["arm"], x["round"])):
        w(f"| {r['arm']} | r{r['round']} ({r['seed']}) | {r['step1']:.4f} | {r['A_step1']:.4f} | {fp(r['step1_rel'], 3)} | "
          f"{r['last16']:.4f} | {r['A_last16']:.4f} | {fp(r['last16_rel'], 3)} | {'yes' if r['same_batches_as_A'] else 'NO'} |")
    w("\n| arm | max abs step-1 delta | per-seed last-16 deltas | mean last-16 delta | inside tolerance |")
    w("|---|---:|---|---:|---|")
    for arm, v in loss_verdict.items():
        w(f"| {arm} | {100 * v['max_abs_step1_rel']:.3f}% | {', '.join(fp(x, 3) for x in v['per_seed_last16_rel'])} | "
          f"{fp(v['mean_last16_rel'], 3)} | {'YES' if v['inside_tolerance'] else 'NO'} |")
    if noise:
        w("\nNon-determinism floor (same arm, same seed, P cell against T cell, steps 1-9): "
          + "; ".join(f"{n['cell'][:-2]}: step-1 {100 * n['step1_rel']:.3f}%, max over 9 steps {100 * n['max_rel_first9']:.3f}%" for n in noise))
    if profile:
        cat_order = [("base-weight GEMM (q,k,v,o,gate,up,down)", "base-weight GEMM"), ("LoRA GEMM (rank 16)", "LoRA GEMM"),
                     ("LM head GEMM", "lm_head GEMM"), ("NF4 dequant (bitsandbytes)", "NF4 dequant"), ("attention", "attention"),
                     ("cross-entropy / logits kernels", "cross-entropy + logits-side"),
                     ("RMSNorm / RoPE / SwiGLU / elementwise", "RMSNorm/RoPE/SwiGLU/elementwise/casts"),
                     ("optimizer + grad clip", "optimizer + clip"), ("memcpy / memset", "memcpy / memset (D2D)")]
        w("\n### P1. GPU time per step by category (ms; mean of the 3 profiled steps; mean over the P cells of the arm)\n")
        w("| category | " + " | ".join(f"{a} (n={len(arm_mean(profile, a, lambda c: c.get('step_wall_ms'))[1])})" for a in ARMS) + " |")
        w("|---|" + "---:|" * len(ARMS))
        for key, label in cat_order:
            row = []
            for arm in ARMS:
                m, _ = arm_mean(profile, arm, lambda c, k=key: (c.get("categories_ms") or {}).get(k))
                row.append(f1(m))
            w(f"| {label} | " + " | ".join(row) + " |")
        for label, getter in (("GPU busy", lambda c: c.get("gpu_busy_ms")), ("step wall (profiled)", lambda c: c.get("step_wall_ms")),
                              ("  of which elementwise ops with an fp32 input", lambda c: c.get("elementwise_fp32_in_ms")),
                              ("  of which elementwise ops with no fp32 input", lambda c: c.get("elementwise_other_in_ms")),
                              ("Soup rewind recorder (row_losses)", lambda c: c.get("rewind_recorder_gpu_ms"))):
            w(f"| {label} | " + " | ".join(f1(arm_mean(profile, a, getter)[0]) for a in ARMS) + " |")
        w("\n### P2. What each arm removed (profile cells, ms per step), against the issue's 818 ms elementwise gap\n")
        a_ew, _ = arm_mean(profile, "A", lambda c: (c.get("categories_ms") or {}).get("RMSNorm / RoPE / SwiGLU / elementwise"))
        a_ce, _ = arm_mean(profile, "A", lambda c: (c.get("categories_ms") or {}).get("cross-entropy / logits kernels"))
        a_busy, _ = arm_mean(profile, "A", lambda c: c.get("gpu_busy_ms"))
        a_fp32, _ = arm_mean(profile, "A", lambda c: c.get("elementwise_fp32_in_ms"))
        w("| arm | elementwise bucket ms | removed vs A ms | share of the issue's 818 ms gap | fp32-input elementwise ms | removed vs A | CE + logits ms | removed vs A | GPU busy ms | removed vs A |")
        w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for arm in ARMS:
            ew, _ = arm_mean(profile, arm, lambda c: (c.get("categories_ms") or {}).get("RMSNorm / RoPE / SwiGLU / elementwise"))
            if ew is None:
                continue
            fp32, _ = arm_mean(profile, arm, lambda c: c.get("elementwise_fp32_in_ms"))
            ce, _ = arm_mean(profile, arm, lambda c: (c.get("categories_ms") or {}).get("cross-entropy / logits kernels"))
            busy, _ = arm_mean(profile, arm, lambda c: c.get("gpu_busy_ms"))
            w(f"| {arm} | {f1(ew)} | {f1(a_ew - ew)} | {100 * (a_ew - ew) / ISSUE['ew_gap_ms']:.0f}% | {f1(fp32)} | {f1(a_fp32 - fp32)} | "
              f"{f1(ce)} | {f1(a_ce - ce)} | {f1(busy)} | {f1(a_busy - busy)} |")
        w("\n### P3. Memory: per-phase allocator peaks of the P cells (MiB, max over steps 3-9) and parameter dtypes at trainer construction\n")
        phases = ["pre_step", "pre_fwd", "fwd_body", "head_ce", "backward", "optimizer"]
        w("| arm (cell) | after load | " + " | ".join(phases) + " | step peak (max phase) | snapshot peak estimate | blocks >= 1,000 MiB at the snapshot peak |")
        w("|---|---:|" + "---:|" * len(phases) + "---:|---:|---|")
        for name, c in sorted(profile.items(), key=lambda kv: kv[0]):
            big = ", ".join(f"{b['mib']:,.0f}" for b in c["snapshot"]["largest_blocks"] if b["mib"] >= 1000)
            ph = c["phase_peaks_mib"]
            w(f"| {name} | {f1((c.get('after_load') or {}).get('peak_mib_since_process_start'), 0)} | "
              + " | ".join(f1(ph.get(p), 0) for p in phases)
              + f" | {f1(c.get('step_peak_mib_phase_max'), 0)} | {f1(c['snapshot'].get('peak_total_mib_estimate'), 0)} | {big or 'none'} |")
        w("\n### P4. Profiler overhead and agreement with the T cells (seconds per step)\n")
        w("| P cell | unprofiled steps 2-5 mean | profiled steps 6-8 mean | overhead | T cell median (same arm and round) | steps 2-5 mean vs T median |")
        w("|---|---:|---:|---:|---:|---:|")
        for name, c in sorted(profile.items()):
            unprof, prof = c.get("unprofiled_wall_s_steps_2_5"), c.get("profiled_wall_s_steps_6_8")
            tmed = timing_cells.get(name[:-1] + "T", {}).get("median_step_s")
            if unprof and prof:
                um, pm = statistics.mean(unprof), statistics.mean(prof)
                w(f"| {name} | {um:.4f} | {pm:.4f} | {fp(rel(pm, um))} | {f1(tmed, 4)} | {fp(rel(um, tmed)) if tmed else 'n/a'} |")
        w("\nParameter bytes by group and dtype (MiB) at trainer construction, first P cell of each arm:\n")
        for arm in ARMS:
            first = next((c for n, c in sorted(profile.items()) if n.startswith(arm + "-") and c.get("param_groups")), None)
            if first:
                groups = [g for g in first["param_groups"] if any(t in g["group"] for t in ("embed_tokens", "lm_head", "norm", "bias", "lora"))]
                w(f"* {arm} ({first['cell']}): " + "; ".join(f"{g['group']} {g['mib']:,.1f}" for g in groups))
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="the pulled .../out/runs directory")
    ap.add_argument("--analysis", help="analysis.json written by case1-profile/analyze.py for the P cells")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cells = build_timing(args.runs)
    per_arm = agg_timing(cells)
    loss_rows, loss_verdict = build_loss(cells)
    noise = noise_floor(args.runs)
    analysis = load_json(args.analysis) if args.analysis and os.path.exists(args.analysis) else {}
    profile = build_profile(args.runs, analysis or {})
    slim_cells = {k: {kk: vv for kk, vv in v.items() if kk not in ("fp", "loss_curve")} for k, v in cells.items()}
    for name, path, obj in (("timing.json", None, {"cells": slim_cells, "per_arm": per_arm}),
                            ("loss.json", None, {"rows": loss_rows, "verdict": loss_verdict, "noise_floor": noise,
                                                 "curves": {k: v.get("loss_curve") for k, v in cells.items()}}),
                            ("profile.json", None, profile)):
        with open(os.path.join(args.out, name), "w", encoding="utf-8") as handle:
            json.dump(obj, handle, indent=1)
    with open(os.path.join(args.out, "tables.md"), "w", encoding="utf-8") as handle:
        handle.write(tables(cells, per_arm, loss_rows, loss_verdict, noise, profile))
    print(open(os.path.join(args.out, "tables.md"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
