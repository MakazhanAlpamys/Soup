#!/usr/bin/env python3
"""Offline analysis of the case-1 profiler traces (soup-8a-prof).  Reads trace.json.gz written by tco_profile.py.

    python analyze.py --arm soup=<run dir>/prof --arm unsloth=<run dir>/prof --out analysis.json

For every GPU event (kernel / memcpy / memset) it finds
  * the CPU launch time (the cuda_runtime event with the same correlation id),
  * the TCO:: record_function ranges that contain that launch time  -> step, phase, layer, recompute flag,
  * the aten op that launched it (External id) and that op's input dims,
and classifies it into a category by kernel name, parent op and dims.  Everything is divided by the number of
profiled steps, so all times are ms per step.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import gzip
import json
import os
import re

VOCABS = (152064, 151936)   # Qwen2.5-7B padded vocab (152,064; the small Qwen2.5 sizes use 151,936): any op with this dim touches the logits / lm_head
VOCAB = 152064
RANK = 16        # LoRA rank

RE_DEQUANT = re.compile(r"dequant", re.I)
RE_BNB_OTHER = re.compile(r"kquantize|quantize_blockwise|kdequant|kgemm_4bit", re.I)
RE_ATTN = re.compile(r"flash|fmha|sdpa|scaled_dot|mem_efficient|attention|_attn|cutlass.*fmha|splitkv|xformers|"
                     r"softmax_warp|cudnn_generated_oss", re.I)
RE_GEMM = re.compile(r"gemm|cutlass|cublas|xmma|ampere_|gemv|nvjet|splitk|s16816|s1688|wgmma|sm\d+_|implicit_convolve|"
                     r"magma|kernel2<", re.I)
RE_CE = re.compile(r"cross_entropy|nll_loss|log_softmax|softmax|logsumexp|cunn_softmax", re.I)
RE_OPT = re.compile(r"adam|foreach|multi_tensor|apply_|lerp|clip_grad|norm_|l2_norm|reduce_", re.I)
RE_RMS = re.compile(r"rms_layernorm|rmsnorm|rms_norm|layernorm|layer_norm|_rms_", re.I)
RE_ROPE = re.compile(r"rope|rotary", re.I)
RE_SWIGLU = re.compile(r"_fg_kernel|swiglu|_dwf_dw_dfg|_geglu|silu|_dw_df", re.I)
RE_MEMCPY = re.compile(r"memcpy|memset", re.I)

CAST_OPS = {"aten::to", "aten::_to_copy", "aten::copy_", "aten::contiguous", "aten::clone", "aten::type_as",
            "aten::_to_copy_", "aten::float", "aten::half", "aten::bfloat16"}
WHERE_ORDER = ["fwd_embed_norm", "fwd_layers", "head_fwd", "recompute", "bwd", "clip", "optimizer", "post", "prep"]


def short_name(name: str) -> str:
    """Kernel name without the template/argument noise, for set comparison across arms."""
    name = re.sub(r"\(.*$", "", name)
    name = re.sub(r"<.*$", "", name)
    name = name.replace("void ", "").strip()
    return name[:120]


def load_trace(path: str) -> dict:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def flat_dims(dims) -> set:
    out = set()
    for item in dims or []:
        if isinstance(item, list):
            for val in item:
                if isinstance(val, int):
                    out.add(val)
    return out


class Ranges:
    """Sorted, mostly disjoint [start, end) ranges with a label; find(t) -> label or None."""

    def __init__(self, items):
        items = sorted(items)
        self.starts = [i[0] for i in items]
        self.ends = [i[1] for i in items]
        self.labels = [i[2] for i in items]

    def find(self, t):
        i = bisect.bisect_right(self.starts, t) - 1
        for j in range(i, max(i - 3, -1), -1):   # tolerate slight nesting
            if j >= 0 and self.starts[j] <= t <= self.ends[j]:
                return self.labels[j]
        return None


def merge_len(intervals, lo, hi) -> float:
    """Length of the union of intervals clipped to [lo, hi]."""
    clipped = sorted((max(a, lo), min(b, hi)) for a, b in intervals if b > lo and a < hi)
    total, cur_s, cur_e = 0.0, None, None
    for a, b in clipped:
        if cur_e is None or a > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = a, b
        else:
            cur_e = max(cur_e, b)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def elementwise_split(kname: str, opname: str, dimset: set, types) -> str:
    """Sub-label for the elementwise bucket: by Triton kernel name (Unsloth) or by tensor shape (eager PyTorch)."""
    low = kname.lower()
    if RE_RMS.search(low):
        return "RMSNorm"
    if RE_ROPE.search(low):
        return "RoPE"
    if RE_SWIGLU.search(low) and "silu" not in low:
        return "SwiGLU (fused)"
    if opname in CAST_OPS:
        if set(VOCABS) & dimset:
            return "casts / copies [..,vocab]"
        for width in (18944, 3584, 512, 128):
            if width in dimset:
                return f"casts / copies [..,{width}]"
        return "casts / copies (other shapes)"
    if 18944 in dimset:
        return "MLP act (silu*up) [..,18944]"
    if 128 in dimset and (28 in dimset or 4 in dimset):
        return "RoPE / q,k,v layout [..,H,128]"
    if 3584 in dimset:
        return "hidden-state elementwise (norm, residual) [..,3584]"
    return "other elementwise"


def classify(kname: str, cat: str, opname: str, dimset: set, where: str) -> str:
    low = kname.lower()
    if cat in ("gpu_memcpy", "gpu_memset") or RE_MEMCPY.search(low) and cat != "kernel":
        return "memcpy / memset"
    if RE_DEQUANT.search(low):
        return "NF4 dequant (bitsandbytes)"
    if RE_BNB_OTHER.search(low):
        return "bitsandbytes other"
    if where in ("clip", "optimizer"):
        return "optimizer + grad clip"
    in_layer = where in ("fwd_layers", "recompute", "bwd") and not (set(VOCABS) & dimset)
    if RE_ATTN.search(low) and not (RE_CE.search(low) and bool(set(VOCABS) & dimset)) and (in_layer or "softmax" not in low):
        return "attention"
    if opname in ("aten::_scaled_dot_product_flash_attention", "aten::_scaled_dot_product_efficient_attention",
                  "aten::_scaled_dot_product_cudnn_attention", "aten::_scaled_dot_product_flash_attention_backward",
                  "aten::_scaled_dot_product_efficient_attention_backward", "aten::_flash_attention_forward",
                  "aten::_flash_attention_backward", "aten::_efficient_attention_forward",
                  "aten::_efficient_attention_backward"):
        return "attention"
    if RE_GEMM.search(low) and not RE_CE.search(low):
        if bool(set(VOCABS) & dimset) or where == "head_fwd":    # Unsloth's fused linear+CE works on vocab chunks: no 152,064 dim
            return "LM head GEMM"
        if RANK in dimset:
            return "LoRA GEMM (rank 16)"
        return "base-weight GEMM (q,k,v,o,gate,up,down)"
    if RE_CE.search(low):
        return "cross-entropy / logits kernels"
    if bool(set(VOCABS) & dimset):
        return "cross-entropy / logits kernels"      # fp32 upcast, entropy, argmax, ... of the [8,512,151936] logits
    if RE_RMS.search(low):
        return "RMSNorm / RoPE / SwiGLU / elementwise"
    if RE_ROPE.search(low) or RE_SWIGLU.search(low):
        return "RMSNorm / RoPE / SwiGLU / elementwise"
    return "RMSNorm / RoPE / SwiGLU / elementwise" if where != "unknown" else "other"


def analyse(trace_path: str, n_steps: int) -> dict:
    data = load_trace(trace_path)
    events = data["traceEvents"]
    ann = collections.defaultdict(list)       # name -> [(ts, end, tid)]
    ops = {}                                  # external id -> (name, dims)
    runtime = {}                              # correlation -> (ts, tid)
    gpu = []                                  # raw gpu events
    for ev in events:
        if ev.get("ph") != "X":
            continue
        cat, name, ts, dur = ev.get("cat"), ev.get("name", ""), ev.get("ts"), ev.get("dur", 0)
        args = ev.get("args") or {}
        if cat == "user_annotation" and name.startswith("TCO::"):
            ann[name].append((ts, ts + dur, ev.get("tid")))
        elif cat == "cpu_op":
            ext = args.get("External id")
            if ext is not None:
                ops[ext] = (name, args.get("Input Dims"), args.get("Input type"))
        elif cat in ("cuda_runtime", "cuda_driver"):
            corr = args.get("correlation")
            if corr is not None and ("Launch" in name or "launch" in name or "Memcpy" in name or "Memset" in name):
                runtime[corr] = (ts, ev.get("tid"), name)
        elif cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            gpu.append((cat, name, ts, dur, args.get("correlation"), args.get("External id"), args))
    steps = sorted(ann.get("TCO::step", []))
    if len(steps) != n_steps:
        print(f"  note: found {len(steps)} TCO::step ranges, expected {n_steps}")
    n_steps = max(len(steps), 1)
    step_rng = Ranges([(s, e, i) for i, (s, e, _) in enumerate(steps)])
    phase_names = ["prep", "fwd_body", "head_ce", "backward", "clip", "optimizer", "post"]
    phase_rng = Ranges([(s, e, p) for p in phase_names for (s, e, _) in ann.get("TCO::" + p, [])])
    fwd_layers = Ranges([(s, e, int(n.rsplit("_", 1)[1])) for n, lst in ann.items() if n.startswith("TCO::fwd_layer_")
                         for (s, e, _) in lst])
    rec_layers = Ranges([(s, e, int(n.rsplit("_", 1)[1])) for n, lst in ann.items() if n.startswith("TCO::recompute_layer_")
                         for (s, e, _) in lst])
    rewind_rng = Ranges([(s, e, "rewind") for (s, e, _) in ann.get("TCO::soup_rewind_row_losses", [])])

    def where_of(t):
        if rec_layers.find(t) is not None:
            return "recompute"
        if fwd_layers.find(t) is not None:
            return "fwd_layers"
        ph = phase_rng.find(t)
        return {"fwd_body": "fwd_embed_norm", "head_ce": "head_fwd", "backward": "bwd", "clip": "clip",
                "optimizer": "optimizer", "post": "post", "prep": "prep"}.get(ph, "unknown")

    cat_where = collections.defaultdict(lambda: collections.defaultdict(float))     # category -> where -> us
    kern = collections.defaultdict(lambda: [0, 0.0, ""])    # name -> [calls, us, category]
    sub = collections.defaultdict(float)
    ew_split = collections.defaultdict(float)
    ce_split = collections.defaultdict(float)
    memcpy = collections.defaultdict(lambda: [0, 0.0, 0.0])   # name -> calls, us, bytes
    intervals = collections.defaultdict(list)               # step idx -> [(ts, end)]
    missing_launch = 0
    no_ext = 0
    per_step_busy = collections.defaultdict(float)
    rewind_us = 0.0
    n_by_where = collections.defaultdict(lambda: collections.defaultdict(int))
    for cat, name, ts, dur, corr, ext, args in gpu:
        launch = runtime.get(corr)
        t_launch = launch[0] if launch else None
        if t_launch is None:
            missing_launch += 1
            t_launch = ts
        sidx = step_rng.find(t_launch)
        if sidx is None:
            sidx = step_rng.find(ts)
        if sidx is None:
            continue
        intervals[sidx].append((ts, ts + dur))
        where = where_of(t_launch)
        opname, dims, types = ops.get(ext, ("", None, None))
        if ext is None:
            no_ext += 1
        dimset = flat_dims(dims)
        category = classify(name, cat, opname, dimset, where)
        cat_where[category][where] += dur
        n_by_where[where][category] += 1
        key = short_name(name) if cat == "kernel" else name
        rec = kern[name if cat == "kernel" else "[" + cat + "] " + name]
        rec[0] += 1
        rec[1] += dur
        rec[2] = category
        if cat != "kernel":
            mrec = memcpy[name]
            mrec[0] += 1
            mrec[1] += dur
            mrec[2] += float(args.get("bytes", 0) or 0)
        if rewind_rng.find(t_launch) is not None:
            rewind_us += dur
        sub[(category, opname)] += dur
        if category == "RMSNorm / RoPE / SwiGLU / elementwise":
            label = elementwise_split(name, opname, dimset, types)
            fp32 = bool(types) and any(t == "float" for t in types if isinstance(t, str))
            ew_split[(label, "fp32 in" if fp32 else "bf16/other in")] += dur
        if category == "cross-entropy / logits kernels":
            ce_split[(short_name(name), opname)] += dur

    wall_us = sum(e - s for s, e, _ in steps)
    busy_us = sum(merge_len(intervals[i], steps[i][0], steps[i][1]) for i in range(len(steps)))
    kernel_sum_us = sum(v[1] for v in kern.values())
    # CPU time of the phases (profiled; includes profiler overhead)
    phase_cpu_us = {p: sum(e - s for s, e, _ in ann.get("TCO::" + p, [])) for p in phase_names}
    # GPU busy per phase (union of GPU intervals clipped to the phase's CPU range is wrong when the CPU runs ahead;
    # kernel time by launch phase is reported instead, via cat_where)
    per_n = 1.0 / n_steps
    result = {
        "n_steps": n_steps,
        "step_wall_ms": [round((e - s) / 1000, 2) for s, e, _ in steps],
        "wall_ms_per_step": wall_us * per_n / 1000,
        "gpu_busy_ms_per_step": busy_us * per_n / 1000,
        "gpu_kernel_sum_ms_per_step": kernel_sum_us * per_n / 1000,
        "idle_ms_per_step": (wall_us - busy_us) * per_n / 1000,
        "phase_cpu_ms_per_step": {p: v * per_n / 1000 for p, v in phase_cpu_us.items()},
        "ranges": {n: len(v) * per_n for n, v in ann.items() if not n.startswith("TCO::fwd_layer_") and not n.startswith("TCO::recompute_layer_")},
        "n_fwd_layer_ranges_per_step": sum(len(v) for n, v in ann.items() if n.startswith("TCO::fwd_layer_")) * per_n,
        "n_recompute_layer_ranges_per_step": sum(len(v) for n, v in ann.items() if n.startswith("TCO::recompute_layer_")) * per_n,
        "gpu_events": len(gpu), "missing_launch_link": missing_launch, "no_external_id": no_ext,
        "rewind_recorder_gpu_ms_per_step": rewind_us * per_n / 1000,
        "categories_ms_per_step": {c: {w: v * per_n / 1000 for w, v in d.items()} for c, d in cat_where.items()},
        "kernels": sorted(({"name": k, "calls_per_step": v[0] * per_n, "ms_per_step": v[1] * per_n / 1000, "category": v[2]}
                           for k, v in kern.items()), key=lambda r: -r["ms_per_step"]),
        "memcpy": {k: {"calls_per_step": v[0] * per_n, "ms_per_step": v[1] * per_n / 1000, "mb_per_step": v[2] * per_n / 1e6}
                   for k, v in memcpy.items()},
        "elementwise_split": sorted(({"label": k[0], "dtype": k[1], "ms_per_step": us * per_n / 1000}
                                     for k, us in ew_split.items()), key=lambda r: -r["ms_per_step"]),
        "ce_split": sorted(({"kernel": k[0], "op": k[1], "ms_per_step": us * per_n / 1000}
                            for k, us in ce_split.items()), key=lambda r: -r["ms_per_step"])[:20],
        "category_by_parent_op": sorted(({"category": c, "op": o, "ms_per_step": us * per_n / 1000} for (c, o), us in sub.items()),
                                        key=lambda r: -r["ms_per_step"])[:80],
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="name=<prof dir containing trace.json.gz>")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = {}
    for spec in args.arm:
        name, path = spec.split("=", 1)
        trace = os.path.join(path, "trace.json.gz")
        if not os.path.exists(trace):
            trace = os.path.join(path, "trace.json")
        print("analysing", name, trace)
        out[name] = analyse(trace, args.steps)
        r = out[name]
        print(f"  wall {r['wall_ms_per_step']:.1f} ms/step, gpu busy {r['gpu_busy_ms_per_step']:.1f}, "
              f"kernel sum {r['gpu_kernel_sum_ms_per_step']:.1f}, idle {r['idle_ms_per_step']:.1f}; "
              f"gpu events {r['gpu_events']} (no launch link {r['missing_launch_link']}, no ext id {r['no_external_id']})")
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1)


if __name__ == "__main__":
    main()
