"""F4 item 4 - split weight restore, routing and the multiply itself.

One OLMoE MoE layer forward, 512 tokens, 64 experts, top-8.
Four arms, each adding one term. Differences give the split.

  A gemm      64 experts x 2 GEMMs, weights already bf16, contiguous rows
  B +gather   adds row gather and index_add_ scatter
  C +router   adds the router GEMM, softmax and topk
  D +dequant  adds NF4 -> bf16 dequantisation per expert

  A      = the multiply
  B - A  = indexing
  C - B  = routing
  D - C  = weight restore
"""

import json
import sys
import time

import torch
import torch.nn.functional as functional

if not torch.cuda.is_available():
    print("SKIP: no CUDA")
    sys.exit(0)

ROUND = sys.argv[1] if len(sys.argv) > 1 else "0"
NATIVE_BF16 = torch.cuda.is_bf16_supported(including_emulation=False)
DTYPE = torch.bfloat16 if NATIVE_BF16 else torch.float16

# OLMoE-1B-7B-0924, from config.json
HIDDEN, INTER, EXPERTS, TOPK = 2048, 1024, 64, 8
TOKENS = 512
M_EXPERT = TOKENS * TOPK // EXPERTS

REPEATS, ITERS, WARMUP = 5, 20, 5

try:
    import bitsandbytes.functional as bnb
    HAVE_BNB = True
except Exception as exc:
    HAVE_BNB = False
    BNB_ERR = repr(exc)

print("round       ", ROUND)
print("torch       ", torch.__version__)
print("GPU         ", torch.cuda.get_device_name(0))
print("capability  ", torch.cuda.get_device_capability(0))
print("dtype       ", DTYPE)
print("bitsandbytes", "yes" if HAVE_BNB else f"NO - arm D skipped ({BNB_ERR})")
print(f"shape       hidden={HIDDEN} inter={INTER} experts={EXPERTS} "
      f"top_k={TOPK} tokens={TOKENS} -> M={M_EXPERT}")


def timed(fn, iters=ITERS, warmup=WARMUP, repeats=REPEATS):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(repeats):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / iters)
    return best


def settle(seconds=60):
    a = torch.randn(4096, 4096, device="cuda", dtype=DTYPE)
    b = torch.randn(4096, 4096, device="cuda", dtype=DTYPE)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        for _ in range(50):
            a @ b
        torch.cuda.synchronize()
    del a, b
    torch.cuda.empty_cache()
    print(f"warmed up for {seconds}s")


x = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=DTYPE)
gate_up = torch.randn(EXPERTS, 2 * INTER, HIDDEN, device="cuda", dtype=DTYPE) * 0.02
down = torch.randn(EXPERTS, HIDDEN, INTER, device="cuda", dtype=DTYPE) * 0.02
router_w = torch.randn(EXPERTS, HIDDEN, device="cuda", dtype=DTYPE) * 0.02

fixed_idx = torch.arange(TOKENS * TOPK, device="cuda") % EXPERTS
fixed_rows = [torch.nonzero(fixed_idx == e).squeeze(-1) % TOKENS for e in range(EXPERTS)]


def arm_a():
    out = torch.zeros_like(x)
    for e in range(EXPERTS):
        cur = x[:M_EXPERT]
        g, u = functional.linear(cur, gate_up[e]).chunk(2, dim=-1)
        h = functional.silu(g) * u
        out[:M_EXPERT] += functional.linear(h, down[e])
    return out


def arm_b():
    out = torch.zeros_like(x)
    for e in range(EXPERTS):
        rows = fixed_rows[e]
        cur = x[rows]
        g, u = functional.linear(cur, gate_up[e]).chunk(2, dim=-1)
        h = functional.silu(g) * u
        out.index_add_(0, rows, functional.linear(h, down[e]).to(out.dtype))
    return out


def route():
    logits = functional.linear(x, router_w)
    probs = torch.softmax(logits, dtype=torch.float, dim=-1)
    w, idx = torch.topk(probs, TOPK, dim=-1)
    mask = torch.nn.functional.one_hot(idx, num_classes=EXPERTS).permute(2, 1, 0)
    return w, mask


def arm_c():
    _, mask = route()
    out = torch.zeros_like(x)
    for e in range(EXPERTS):
        _, rows = torch.where(mask[e])
        if rows.numel() == 0:
            continue
        cur = x[rows]
        g, u = functional.linear(cur, gate_up[e]).chunk(2, dim=-1)
        h = functional.silu(g) * u
        out.index_add_(0, rows, functional.linear(h, down[e]).to(out.dtype))
    return out


if HAVE_BNB:
    q_gate_up = [bnb.quantize_4bit(gate_up[e].contiguous(), quant_type="nf4")
                 for e in range(EXPERTS)]
    q_down = [bnb.quantize_4bit(down[e].contiguous(), quant_type="nf4")
              for e in range(EXPERTS)]

    def arm_d():
        _, mask = route()
        out = torch.zeros_like(x)
        for e in range(EXPERTS):
            w_gu = bnb.dequantize_4bit(q_gate_up[e][0], q_gate_up[e][1])
            w_dn = bnb.dequantize_4bit(q_down[e][0], q_down[e][1])
            _, rows = torch.where(mask[e])
            if rows.numel() == 0:
                continue
            cur = x[rows]
            g, u = functional.linear(cur, w_gu.to(DTYPE)).chunk(2, dim=-1)
            h = functional.silu(g) * u
            out.index_add_(0, rows, functional.linear(h, w_dn.to(DTYPE)).to(out.dtype))
        return out


def ref(m):
    a = torch.randn(m, HIDDEN, device="cuda", dtype=DTYPE)
    b = torch.randn(HIDDEN, 2 * INTER, device="cuda", dtype=DTYPE)
    ops = 2 * m * HIDDEN * (2 * INTER)
    iters = max(50, min(20000, int(2e12 / ops)))
    ms = timed(lambda: a @ b, iters=iters, warmup=5, repeats=3)
    return ops / (ms * 1e-3) / 1e12


settle()

print("\n=== reference point, olmoe_gate_up (K=2048, N=2048) ===")
t512 = ref(512)
t64 = ref(M_EXPERT)
print(f"M=512 {t512:7.3f} TFLOPS   M={M_EXPERT} {t64:7.3f} TFLOPS   "
      f"factor {t64 / t512:.3f}")

print(f"\n=== one MoE layer forward, {TOKENS} tokens, {EXPERTS} experts ===")
arms = [("A gemm", arm_a), ("B +gather", arm_b), ("C +router", arm_c)]
if HAVE_BNB:
    arms.append(("D +dequant", arm_d))

ms = {}
for name, fn in arms:
    ms[name] = timed(fn)
    print(f"{name:<12} {ms[name]:8.3f} ms")

print("\n=== split ===")
total = ms[arms[-1][0]]
parts = [("the multiply", ms["A gemm"])]
parts.append(("indexing (gather/scatter)", ms["B +gather"] - ms["A gemm"]))
parts.append(("routing", ms["C +router"] - ms["B +gather"]))
if HAVE_BNB:
    parts.append(("weight restore (NF4)", ms["D +dequant"] - ms["C +router"]))
for label, v in parts:
    print(f"{label:<28} {v:8.3f} ms   {v / total * 100:5.1f}%")
print(f"{'total':<28} {total:8.3f} ms")

json.dump(
    {"round": ROUND, "gpu": torch.cuda.get_device_name(0),
     "torch": torch.__version__, "dtype": str(DTYPE), "bnb": HAVE_BNB,
     "shape": {"hidden": HIDDEN, "inter": INTER, "experts": EXPERTS,
               "top_k": TOPK, "tokens": TOKENS, "M": M_EXPERT},
     "reference": {"tflops_512": t512, "tflops_M": t64, "factor": t64 / t512},
     "arms_ms": ms,
     "split_ms": dict(parts)},
    open(f"f4_item4_round_{ROUND}.json", "w"), indent=2)
print(f"\nraw -> f4_item4_round_{ROUND}.json")

# --- extra: router and mask, but precomputed rows (no torch.where) ---
def arm_c2():
    _, mask = route()
    out = torch.zeros_like(x)
    for e in range(EXPERTS):
        rows = fixed_rows[e]
        cur = x[rows]
        g, u = functional.linear(cur, gate_up[e]).chunk(2, dim=-1)
        h = functional.silu(g) * u
        out.index_add_(0, rows, functional.linear(h, down[e]).to(out.dtype))
    return out

c2 = timed(arm_c2)
print(f"\nC2 (router+mask, no torch.where) {c2:8.3f} ms")
print(f"  router + one_hot   {c2 - ms['B +gather']:7.3f} ms")
print(f"  64x torch.where    {ms['C +router'] - c2:7.3f} ms")
