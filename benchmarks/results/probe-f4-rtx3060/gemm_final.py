import json
import statistics
import sys
import time

import torch

if not torch.cuda.is_available():
    print("SKIP: no CUDA")
    sys.exit(0)

ROUND = sys.argv[1] if len(sys.argv) > 1 else "0"

NATIVE_BF16 = torch.cuda.is_bf16_supported(including_emulation=False)
DTYPE = torch.bfloat16 if NATIVE_BF16 else torch.float16

WARMUP_SECONDS = 60
FLOP_BUDGET = 2e12


def clock():
    try:
        return int(torch.cuda.clock_rate())
    except Exception:
        return 0


print("round       ", ROUND)
print("torch       ", torch.__version__)
print("GPU         ", torch.cuda.get_device_name(0))
print("capability  ", torch.cuda.get_device_capability(0))
print("bf16 native ", NATIVE_BF16)
print("dtype       ", DTYPE)
print("idle clock  ", clock(), "MHz")


def settle(seconds=WARMUP_SECONDS):
    a = torch.randn(4096, 4096, device="cuda", dtype=DTYPE)
    b = torch.randn(4096, 4096, device="cuda", dtype=DTYPE)
    t0 = time.perf_counter()
    samples = []
    while time.perf_counter() - t0 < seconds:
        for _ in range(50):
            a @ b
        torch.cuda.synchronize()
        samples.append(clock())
    tail = [s for s in samples[-10:] if s]
    if tail:
        med = statistics.median(tail)
        spread = (max(tail) - min(tail)) / med if med else 0
        print(f"after {seconds}s warm-up: {med:.0f} MHz, spread {spread * 100:.1f}%")
        if spread > 0.05:
            print("!!! clock did NOT settle - this round is not valid")
    del a, b
    torch.cuda.empty_cache()


def tflops(M, K, N, repeats=3):
    a = torch.randn(M, K, device="cuda", dtype=DTYPE)
    b = torch.randn(K, N, device="cuda", dtype=DTYPE)
    assert (a @ b).shape == (M, N), f"bad shape: {(a @ b).shape} != {(M, N)}"

    ops = 2 * M * K * N
    iters = max(50, min(20000, int(FLOP_BUDGET / ops)))

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)

    best, clocks = 0.0, []
    for _ in range(repeats):
        for _ in range(5):
            a @ b
        torch.cuda.synchronize()
        s.record()
        for _ in range(iters):
            a @ b
        e.record()
        torch.cuda.synchronize()
        clocks.append(clock())
        ms = s.elapsed_time(e) / iters
        best = max(best, ops / (ms * 1e-3) / 1e12)
    return best, statistics.median(clocks), iters


def measure_pair(M, K, N, m_denom=512):
    a1, c1, _ = tflops(m_denom, K, N)
    b1, c2, it = tflops(M, K, N)
    b2, c3, _ = tflops(M, K, N)
    a2, c4, _ = tflops(m_denom, K, N)
    cl = [c for c in (c1, c2, c3, c4) if c]
    med = statistics.median(cl) if cl else 0
    spread = (max(cl) - min(cl)) / med if med else 0
    return {
        "M": M, "point": (b1 + b2) / 2, "denom": (a1 + a2) / 2,
        "ratio": ((b1 + b2) / 2) / ((a1 + a2) / 2),
        "a2_over_a1": a2 / a1, "clock_mhz": med, "clock_spread": spread,
        "iters": it,
    }


SHAPES = {
    "olmoe_gate_up":   (2048, 2048),
    "olmoe_down":      (1024, 2048),
    "granite_gate_up": (1024, 1024),
    "granite_down":    (512, 1024),
}
M_SWEEP = [1, 16, 32, 64, 128, 256, 512, 1024, 2048, 8192]

settle()

results = {
    "round": ROUND,
    "gpu": torch.cuda.get_device_name(0),
    "capability": list(torch.cuda.get_device_capability(0)),
    "torch": torch.__version__,
    "dtype": str(DTYPE),
    "bf16_native": NATIVE_BF16,
    "shapes": {},
}

for name, (K, N) in SHAPES.items():
    print(f"\n=== {name}:  K={K}  N={N} ===")
    print(f"{'M':>6} {'TFLOPS':>10} {'denom':>10} {'ratio':>8} "
          f"{'a2/a1':>8} {'MHz':>6} {'dMHz%':>7} {'iters':>7}")
    rows = []
    for M in M_SWEEP:
        r = measure_pair(M, K, N)
        rows.append(r)
        flag = " <-" if r["clock_spread"] > 0.10 else ""
        print(f"{r['M']:>6} {r['point']:>10.3f} {r['denom']:>10.3f} "
              f"{r['ratio']:>8.3f} {r['a2_over_a1']:>8.3f} "
              f"{r['clock_mhz']:>6.0f} {r['clock_spread'] * 100:>6.1f}% {r['iters']:>7}{flag}")
    results["shapes"][name] = {"K": K, "N": N, "rows": rows}

out = f"f4_round_{ROUND}.json"
with open(out, "w") as fh:
    json.dump(results, fh, indent=2)
print(f"\nraw results -> {out}")
