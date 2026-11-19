"""Balanced AB/BA pairs of benchmarks/harness/issue361_nf4_throughput.py, one fresh
process per arm, with admission guards checked before EVERY arm (no waiting: a failed
guard stops the run and keeps what was measured).

Rule (written before measuring, also copied into <out>/RULE.md):
  metric   = measured-window seconds of an arm (sum of the harness's measured step times)
  per pair = reduction % = (A - B) / A * 100, A = control arm, B = candidate arm
  verdict  = CONFIRMED if median >= 2.0 and the 95% whole-pair bootstrap lower bound > 0
             (10,000 draws, seed 42); NO GAIN if the upper bound < 0; else INCONCLUSIVE.
  No exclusion on timing magnitude; only an operational failure (non-zero exit) voids a pair.

Usage: (Windows only, the memory and power guards use kernel32)
  python benchmarks/harness/stream_ab_pairs.py --tree <worktree> --weights <hf snapshot dir>
      --shards <nf4 shard dir> --out <dir> --pairs 5 --label scale-cache
      --b-args=--nf4-scale-cache
  python benchmarks/harness/stream_ab_pairs.py report <dir>
"""

import argparse
import ctypes
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

MIN_FREE_GIB = 10.0


class _Mem(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
        (n, ctypes.c_ulonglong) for n in ("tp", "ap", "tpf", "apf", "tv", "av", "aev")
    ]


class _Power(ctypes.Structure):
    _fields_ = [
        ("ac", ctypes.c_ubyte),
        ("flag", ctypes.c_ubyte),
        ("pct", ctypes.c_ubyte),
        ("saver", ctypes.c_ubyte),
        ("life", ctypes.c_ulong),
        ("full", ctypes.c_ulong),
    ]


def free_gib():
    mem = _Mem()
    mem.dwLength = ctypes.sizeof(_Mem)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
    return mem.ap / 2**30


def on_ac():
    power = _Power()
    ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(power))
    return power.ac == 1


def foreign_pytest():
    out = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
            "Where-Object { $_.CommandLine -match 'pytest' } | "
            "ForEach-Object { $_.ProcessId }",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout.split()
    return [pid for pid in out if pid.strip()]


def guards():
    problems = []
    free = free_gib()
    if free < MIN_FREE_GIB:
        problems.append(f"free RAM {free:.2f} GiB < {MIN_FREE_GIB}")
    if not on_ac():
        problems.append("not on AC power")
    pids = foreign_pytest()
    if pids:
        problems.append(f"pytest running: pids {pids}")
    return problems, round(free, 2)


def run_arm(args, pair, arm, extra):
    out, common = Path(args.out), args.common
    harness = os.path.realpath(os.path.join(args.tree, "benchmarks", "harness", args.harness))
    harness_root = os.path.realpath(os.path.join(args.tree, "benchmarks", "harness"))
    if os.path.commonpath([harness_root, harness]) != harness_root:
        raise ValueError("harness must stay inside benchmarks/harness")
    path = out / f"p{pair}-{arm}.json"
    cmd = [
        sys.executable,
        harness,
        "--weights",
        args.weights,
        "--shards",
        args.shards,
        "--json",
        str(path),
        *common.split(),
        *extra.split(),
    ]
    env = dict(os.environ, PYTHONPATH=os.path.join(args.tree, "src"))
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=os.path.dirname(harness),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdin=subprocess.DEVNULL,
        timeout=1800,
    )
    (out / f"p{pair}-{arm}.log").write_text(
        proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8"
    )
    return proc.returncode, time.time() - started


def measure(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rule = __doc__.split("Usage:")[0]
    rule_path = out / "RULE.md"
    if not rule_path.exists():
        rule_path.write_text(
            f"label: {args.label}\nA args: '{args.a_args}'\nB args: '{args.b_args}'\n"
            f"common: '{args.common}'\npairs: {args.pairs}\nwritten: "
            f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n\n{rule}",
            encoding="utf-8",
        )
    ledger_path = out / "ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else []
    if args.prime:
        problems, free = guards()
        if problems:
            print(f"STOP before prime: {problems}")
            return 2
        rc, secs = run_arm(args, "prime", "A", args.a_args)
        ledger.append(
            {
                "pair": "prime",
                "arm": "A",
                "rc": rc,
                "wall_s": round(secs, 1),
                "free_gib_before": free,
            }
        )
        ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
        print(f"prime arm (not counted): rc={rc} wall={secs:.0f}s free_before={free}GiB")
        return 0 if rc == 0 else 3
    last = args.pairs if args.to_pair is None else args.to_pair + 1
    for pair in range(args.from_pair, last):
        order = ("A", "B") if pair % 2 == 0 else ("B", "A")
        for arm in order:
            problems, free = guards()
            if problems:
                ledger.append({"pair": pair, "arm": arm, "status": "refused", "why": problems})
                (out / "ledger.json").write_text(json.dumps(ledger, indent=2), encoding="utf-8")
                print(f"STOP before pair {pair} arm {arm}: {problems}")
                return 2
            extra = args.a_args if arm == "A" else args.b_args
            rc, secs = run_arm(args, pair, arm, extra)
            ledger.append(
                {
                    "pair": pair,
                    "arm": arm,
                    "rc": rc,
                    "wall_s": round(secs, 1),
                    "free_gib_before": free,
                }
            )
            (out / "ledger.json").write_text(json.dumps(ledger, indent=2), encoding="utf-8")
            print(f"pair {pair} arm {arm}: rc={rc} wall={secs:.0f}s free_before={free}GiB")
            sys.stdout.flush()
            if rc != 0:
                print("arm failed: stopping (operational failure voids the pair)")
                return 3
    if last >= args.pairs:
        return report(argparse.Namespace(out=args.out))
    return 0


def report(args):
    out = Path(args.out)
    rows = {}
    for path in sorted(out.glob("p[0-9]*-*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") != "ok":
            continue
        pair, arm = path.stem[1:].split("-")
        rows.setdefault(int(pair), {})[arm] = data
    pairs = sorted(p for p, arms in rows.items() if {"A", "B"} <= set(arms))
    reductions, lines = [], []
    for pair in pairs:
        a, b = rows[pair]["A"], rows[pair]["B"]
        ta, tb = sum(a["step_times_s"]), sum(b["step_times_s"])
        red = (ta - tb) / ta * 100
        reductions.append(red)
        lines.append(
            f"pair {pair} ({'AB' if pair % 2 == 0 else 'BA'}): A {ta:.3f}s "
            f"B {tb:.3f}s -> {red:+.3f}% | tok/s {a['tokens_per_s']:.1f} -> "
            f"{b['tokens_per_s']:.1f} | peak alloc {a['peak_vram_gb']['allocated']} -> "
            f"{b['peak_vram_gb']['allocated']} GB | cache B {b.get('nf4_scale_cache_bytes')}"
        )
    print("\n".join(lines))
    if not reductions:
        print("no complete pair")
        return 1
    rng = random.Random(42)
    boots = sorted(
        statistics.median(rng.choice(reductions) for _ in reductions) for _ in range(10_000)
    )
    low, high = boots[249], boots[9749]
    med = statistics.median(reductions)
    if med >= 2.0 and low > 0:
        verdict = "CONFIRMED"
    elif high < 0:
        verdict = "NO GAIN"
    else:
        verdict = "INCONCLUSIVE"
    summary = {
        "pairs": len(reductions),
        "reductions_pct": [round(r, 3) for r in reductions],
        "median_pct": round(med, 3),
        "ci95": [round(low, 3), round(high, 3)],
        "verdict": verdict,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    return 0


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    rep = sub.add_parser("report")
    rep.add_argument("out")
    parser.add_argument("--out")
    parser.add_argument("--tree", help="Soup tree holding the harness and src/")
    parser.add_argument("--weights", help="HF snapshot directory of the model")
    parser.add_argument("--shards", help="NF4 shard directory")
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--label", default="")
    parser.add_argument(
        "--harness",
        default="issue361_nf4_throughput.py",
        help="benchmark script under benchmarks/harness (default unchanged)",
    )
    parser.add_argument("--a-args", default="")
    parser.add_argument("--b-args", default="")
    parser.add_argument("--common", default="--warmup 10 --steps 50 --no-ceiling")
    parser.add_argument("--from-pair", type=int, default=0)
    parser.add_argument("--to-pair", type=int, default=None, help="inclusive")
    parser.add_argument("--prime", action="store_true", help="one uncounted A arm, then exit")
    args = parser.parse_args()
    if args.cmd == "report":
        return report(args)
    missing = [n for n in ("out", "tree", "weights", "shards") if not getattr(args, n)]
    if missing:
        parser.error("required: " + ", ".join("--" + n for n in missing))
    return measure(args)


if __name__ == "__main__":
    sys.exit(main())
