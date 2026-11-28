"""CPU selection benchmark for #3250 (no model download).

Run: python benchmarks/bench_semdedup.py --output results.json
Optional peak process memory (psutil required on Windows):
    python benchmarks/bench_semdedup.py --memory candidate --rows 50000
    python benchmarks/bench_semdedup.py --memory reference --rows 50000
Each memory invocation must use a fresh process; peaks include fixture creation.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import io
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
from rich.console import Console

from soup_cli.utils.semdedup import DedupReport, greedy_semdedup


def reference(vectors: np.ndarray, *, threshold: float) -> DedupReport:
    """The loop on main before this change (76f9b207), with #3254's comparison."""
    arr = np.asarray(vectors, dtype=np.float32)
    kept, dropped, pairs = [], [], []
    for idx in range(len(arr)):
        if kept:
            sims = arr[kept] @ arr[idx]
            best = int(np.argmax(sims))
            score = float(sims[best])
            matches = score >= threshold if threshold == 0 else (
                score > 0 and score >= threshold - 1e-6
            )
            if matches:
                dropped.append(idx)
                pairs.append((idx, kept[best], score))
                continue
        kept.append(idx)
    return DedupReport(tuple(kept), tuple(dropped), tuple(pairs), threshold)


def fixture(rows: int, dimension: int, duplicate_fraction: float) -> np.ndarray:
    """Seeded unit vectors, with a precisely sized tail of exact duplicates."""
    rng = np.random.default_rng(3250)
    vectors = rng.standard_normal((rows, dimension), dtype=np.float32)
    # Normalise in chunks: a whole-matrix norm allocates a temporary as large as
    # the fixture, which would set the process peak before the measured call.
    for start in range(0, rows, 1000):
        chunk = vectors[start : start + 1000]
        chunk /= np.linalg.norm(chunk, axis=1, keepdims=True)
    duplicates = int(rows * duplicate_fraction)
    if duplicates:
        vectors[-duplicates:] = vectors[rng.integers(0, rows - duplicates, duplicates)]
    return vectors


def peak_bytes() -> int:
    if platform.system() == "Windows":
        import psutil

        return psutil.Process().memory_info().peak_wset
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if platform.system() == "Darwin" else peak * 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[2000, 5000, 10000])
    parser.add_argument("--dimension", type=int, default=384)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--memory", choices=["candidate", "reference"])
    args = parser.parse_args()
    if min(args.rows) < 1 or args.dimension < 1 or args.repeats < 1:
        parser.error("rows, dimension and repeats must be positive")
    config = io.StringIO()
    with contextlib.redirect_stdout(config):
        np.show_config()
    results = {
        "base_commit": "76f9b2071d2d16451be13f4c15c29d0af3228916",
        "checkout_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "candidate_source_sha256": hashlib.sha256(
            Path(inspect.getfile(greedy_semdedup)).read_bytes()
        ).hexdigest(),
        "python": platform.python_version(), "numpy": np.__version__,
        "platform": platform.platform(), "cpu": platform.processor(),
        "numpy_config": config.getvalue(), "seed": 3250, "threshold": 0.9,
        "dimension": args.dimension, "repeats": 1 if args.memory else args.repeats,
        "measurements": [],
    }
    console = Console(stderr=True)
    calls = {"reference": reference, "candidate": greedy_semdedup}
    for rows in args.rows:
        for fraction in ([0.0] if args.memory else [0.0, 0.8]):
            vectors = fixture(rows, args.dimension, fraction)
            if args.memory:
                before = peak_bytes()
                start = time.perf_counter()
                report = calls[args.memory](vectors, threshold=0.9)
                seconds = time.perf_counter() - start
                after = peak_bytes()
                results["measurements"].append({
                    "rows": rows, "implementation": args.memory,
                    "seconds": seconds, "kept": len(report.kept),
                    "peak_process_bytes": after, "peak_before_call_bytes": before,
                    "peak_increase_bytes": after - before,
                    "buffer_allocation_bytes": rows * args.dimension * 4,
                })
                continue
            expected = calls["reference"](vectors, threshold=0.9)
            assert vars(greedy_semdedup(vectors, threshold=0.9)) == vars(expected)
            timings = {"reference": [], "candidate": []}
            for repeat in range(args.repeats):
                order = list(calls) if repeat % 2 == 0 else list(reversed(calls))
                for name in order:
                    start = time.perf_counter()
                    actual = calls[name](vectors, threshold=0.9)
                    timings[name].append(time.perf_counter() - start)
                    assert vars(actual) == vars(expected), (rows, fraction, name)
                    assert np.asarray([p[2] for p in actual.pairs], dtype="f8").tobytes() == (
                        np.asarray([p[2] for p in expected.pairs], dtype="f8").tobytes()
                    )
            medians = {name: statistics.median(values) for name, values in timings.items()}
            results["measurements"].append({
                "rows": rows, "duplicate_fraction": fraction, "kept": len(expected.kept),
                "dropped": len(expected.dropped), "exact_report": True,
                "seconds": timings, "median_seconds": medians,
                "candidate_over_reference": medians["candidate"] / medians["reference"],
            })
            console.print(f"Completed rows={rows}, duplicate_fraction={fraction}")
    encoded = json.dumps(results, indent=2)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        Console().print(encoded, markup=False, highlight=False, soft_wrap=True)


if __name__ == "__main__":
    main()
