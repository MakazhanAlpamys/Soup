"""NIG mixture sweep for `soup ab` (#1265): Type-I error and power under peeking.

Since #1265 `soup ab` decides with the normal-inverse-gamma Bayes factor (right-Haar limit)
instead of the two-point mixture with a plug-in variance and a burn-in (#1227). The first
`PRIOR_SCALE_ROWS` rows of each arm are held out to put `effect_size` in standard deviations;
the Bayes factor runs on the rows after them and rejects H0 at 1/alpha. This script measures
that design against the burn-in design it replaces, and it is how `PRIOR_SCALE_ROWS` was
chosen.

Procedure, per effect_size / sigma ratio (effect_size 0.1, sigma = effect_size / ratio,
beta 0.20):
- H0 (`--mode h0`): both arms N(0, sigma^2). H1 (`--mode h1`): the treatment is shifted by
  +effect_size, the smallest difference the operator asked to detect.
- The operator re-runs the test after every new (control, treatment) pair and stops at the
  first terminal verdict, reject_h0 or accept_h0. A run that reaches the horizon (rows per
  arm) undecided stops there. Horizons 200 and 1000.
- NIG with k held-out rows per arm, k in HELD_OUT: verdicts from pair k + 2 on.
- Burn-in: the #1227 statistic (`ab_burn_in_sweep.point_llrs`), no verdict before 30 rows
  per arm at alpha >= 0.05 and 40 below, reject at log((1 - beta) / alpha).
- Reported per design: the share of runs whose first verdict is reject_h0 (under H1, split
  by the reported direction), and the mean pairs at stop.

`run` first checks the vectorised Bayes factor against `_nig_log_bayes_factor`, and whole
simulated runs against `msprt_step`, before recording anything.

Run it in parts (each part writes one JSON file), then merge:
  python benchmarks/harness/ab_nig_sweep.py run --mode h0 --ratios 0.1,0.2 --out h0-1.json
  python benchmarks/harness/ab_nig_sweep.py report h0-1.json h0-2.json h1-1.json ...

CPU only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ab_burn_in_sweep import (  # noqa: E402
    DEFAULT_RATIOS,
    next_true,
    point_llrs,
    summaries,
    tolerance,
)

EFFECT_SIZE = 0.1
BETA = 0.20
ALPHAS = (0.10, 0.05, 0.025, 0.01, 0.005)
HELD_OUT = (3, 5, 8)
HORIZONS = (200, 1000)
BURN_IN = {0.10: 30, 0.05: 30, 0.025: 40, 0.01: 40, 0.005: 40}
CHUNK = 2000


def nig_log_bf(control: np.ndarray, treatment: np.ndarray, prior_variance) -> tuple:
    """The Bayes factor after 2..n rows per arm (column j <-> j + 2), and the mean difference.

    `prior_variance` is a scalar or a (runs, 1) column. Vectorised `_nig_log_bayes_factor`.
    """
    rows, mean_c, mean_t, pooled = summaries(control, treatment)
    nu = 2.0 * rows - 2.0
    n_eff = rows / 2.0
    diff = mean_t - mean_c
    t_squared = diff * diff * n_eff / pooled
    spread = 1.0 + n_eff * prior_variance
    log_bf = -0.5 * np.log(spread) - 0.5 * (nu + 1.0) * (
        np.log1p(t_squared / (nu * spread)) - np.log1p(t_squared / nu)
    )
    return log_bf, diff


def split_nig(control: np.ndarray, treatment: np.ndarray, held: int) -> tuple:
    """NIG with `held` rows per arm held out, padded so column j <-> pair j + 2."""
    head_c, head_t = control[:, :held], treatment[:, :held]
    scale_variance = (head_c.var(axis=1, ddof=1) + head_t.var(axis=1, ddof=1)) / 2.0
    prior = (EFFECT_SIZE * EFFECT_SIZE / scale_variance)[:, None]
    log_bf, diff = nig_log_bf(control[:, held:], treatment[:, held:], prior)
    pad = np.zeros((control.shape[0], held))
    return np.concatenate([pad, log_bf], axis=1), np.concatenate([pad, diff], axis=1)


def first_verdicts(stat, diff, upper, lower, start, horizons) -> dict:
    """Per horizon: (rejections, rejections with diff > 0, sum of pairs at stop)."""
    next_up = next_true(stat >= upper)[:, start - 2]
    next_lo = next_true(stat <= lower)[:, start - 2]
    out = {}
    for horizon in horizons:
        last = horizon - 2
        reject = (next_up < next_lo) & (next_up <= last)
        col = np.where(reject, next_up, 0)
        positive = reject & (np.take_along_axis(diff, col[:, None], axis=1)[:, 0] > 0)
        stop = np.minimum(np.minimum(next_up, next_lo), last) + 2
        out[horizon] = (int(reject.sum()), int(positive.sum()), int(stop.sum()))
    return out


def self_check(rng: np.random.Generator) -> str:
    """The vectorised statistic equals the shipped code, point by point and run by run."""
    try:
        from soup_cli.utils.ab_test import (
            PRIOR_SCALE_ROWS,
            MsprtConfig,
            _nig_log_bayes_factor,
            msprt_step,
        )
    except ImportError as exc:  # pragma: no cover - reported, not fatal
        return f"self-check skipped ({exc})"
    worst = 0.0
    points = 0
    for sigma in (0.05, 0.2, 1.0):
        control = rng.normal(0.0, sigma, size=(10, 400))
        treatment = rng.normal(0.02, sigma, size=(10, 400))
        log_bf, diff = nig_log_bf(control, treatment, 0.7)
        rows, _, _, pooled = summaries(control, treatment)
        for run in range(10):
            for col in (0, 1, 18, 98, 398):
                ref = _nig_log_bayes_factor(
                    n_control=int(rows[col]),
                    n_treatment=int(rows[col]),
                    mean_difference=float(diff[run, col]),
                    pooled_variance=float(pooled[run, col]),
                    prior_variance=0.7,
                )
                worst = max(worst, abs(ref - log_bf[run, col]) / max(1.0, abs(ref)))
                points += 1
    mismatched = 0
    runs = 0
    config = MsprtConfig(metric="judge_score", alpha=0.05, beta=BETA, effect_size=EFFECT_SIZE)
    for ratio in (0.35, 1.0):
        sigma = EFFECT_SIZE / ratio
        control = rng.normal(0.0, sigma, size=(8, 120))
        treatment = rng.normal(EFFECT_SIZE, sigma, size=(8, 120))
        stat, _ = split_nig(control, treatment, PRIOR_SCALE_ROWS)
        for run in range(8):
            for pairs in (PRIOR_SCALE_ROWS + 2, 20, 60, 120):
                shipped = msprt_step(
                    config,
                    control=control[run, :pairs].tolist(),
                    treatment=treatment[run, :pairs].tolist(),
                ).log_likelihood_ratio
                mismatched += not math.isclose(shipped, stat[run, pairs - 2], rel_tol=1e-9,
                                               abs_tol=1e-9)
                runs += 1
    return (f"self-check: max rel diff {worst:.1e} over {points} points vs "
            f"_nig_log_bayes_factor; {mismatched} of {runs} msprt_step peeks differ")


def sweep_ratio(ratio: float, mode: str, runs: int) -> dict:
    rng = np.random.default_rng([1265, 0 if mode == "h0" else 1, int(round(ratio * 10_000))])
    sigma = EFFECT_SIZE / ratio
    shift = 0.0 if mode == "h0" else EFFECT_SIZE
    n_max = max(HORIZONS)
    counts: dict[str, list[int]] = {}

    def add(key, found):
        for horizon, values in found.items():
            name = f"{key}|{horizon}"
            old = counts.get(name, [0, 0, 0])
            counts[name] = [a + b for a, b in zip(old, values)]

    for _ in range(runs // CHUNK):
        control = rng.normal(0.0, sigma, size=(CHUNK, n_max))
        treatment = rng.normal(shift, sigma, size=(CHUNK, n_max))
        rows, mean_c, mean_t, pooled = summaries(control, treatment)
        plus, minus = point_llrs(rows, mean_c, mean_t, pooled)
        mixture = np.logaddexp(plus, minus) - math.log(2.0)
        burn_diff = mean_t - mean_c
        del rows, mean_c, mean_t, pooled, plus, minus
        for alpha in ALPHAS:
            upper = math.log((1.0 - BETA) / alpha)
            lower = math.log(BETA / (1.0 - alpha))
            add(f"burn-in|{alpha}",
                first_verdicts(mixture, burn_diff, upper, lower, BURN_IN[alpha], HORIZONS))
        del mixture, burn_diff
        for held in HELD_OUT:
            stat, diff = split_nig(control, treatment, held)
            for alpha in ALPHAS:
                upper = math.log(1.0 / alpha)
                lower = math.log(BETA / (1.0 - alpha))
                add(f"nig-k{held}|{alpha}",
                    first_verdicts(stat, diff, upper, lower, held + 2, HORIZONS))
    return {"ratio": ratio, "mode": mode, "runs": (runs // CHUNK) * CHUNK, "counts": counts}


def cmd_run(args: argparse.Namespace) -> None:
    ratios = [float(r) for r in args.ratios.split(",")] if args.ratios else list(DEFAULT_RATIOS)
    print(self_check(np.random.default_rng(99)), flush=True)
    results = []
    for ratio in ratios:
        start = time.perf_counter()
        results.append(sweep_ratio(ratio, args.mode, args.runs))
        print(f"{args.mode} ratio {ratio}: {time.perf_counter() - start:.0f} s", flush=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"beta": BETA, "effect_size": EFFECT_SIZE, "results": results}, fh, indent=1)
        fh.write(chr(10))
    print(f"wrote {args.out}")


def _load(parts: list[str]) -> dict:
    merged: dict[tuple[str, float], dict] = {}
    for path in parts:
        with open(path, encoding="utf-8") as fh:
            for item in json.load(fh)["results"]:
                merged[(item["mode"], item["ratio"])] = item
    return merged


def cmd_report(args: argparse.Namespace) -> None:
    merged = _load(args.parts)
    designs = ["burn-in"] + [f"nig-k{k}" for k in HELD_OUT]
    h0 = {r: item for (mode, r), item in sorted(merged.items()) if mode == "h0"}
    h1 = {r: item for (mode, r), item in sorted(merged.items()) if mode == "h1"}
    if h0:
        runs = min(item["runs"] for item in h0.values())
        print(f"H0: Type-I rate (first verdict reject_h0), {runs} runs per ratio; "
              "bound = alpha + 3 SE, * = above it")
        for horizon in sorted(HORIZONS, reverse=True):
            for alpha in ALPHAS:
                bound = alpha + tolerance(alpha, runs)
                print(f"\nalpha {alpha}, horizon {horizon} (bound {bound:.5f})")
                print("  ratio " + "".join(f"{d:>12s}" for d in designs))
                worst = {d: (0.0, None) for d in designs}
                for ratio, item in h0.items():
                    cells = []
                    for design in designs:
                        rate = item["counts"][f"{design}|{alpha}|{horizon}"][0] / item["runs"]
                        if rate >= worst[design][0]:
                            worst[design] = (rate, ratio)
                        cells.append(f"{rate:11.5f}{'*' if rate > bound else ' '}")
                    print((f"  {ratio:5.3f} " + "".join(cells)).rstrip())
                print("  worst " + "".join(f"{w:11.5f} " for w, _ in worst.values())
                      + " at " + ", ".join(f"{r:g}" for _, r in worst.values()))
    if h1:
        runs = min(item["runs"] for item in h1.values())
        print(f"\nH1 (true difference = +effect_size): power (reject_h0, direction right) / "
              f"wrong-direction rejections / mean pairs at stop, {runs} runs per ratio")
        for horizon in sorted(HORIZONS, reverse=True):
            for alpha in (0.05, 0.01):
                print(f"\nalpha {alpha}, horizon {horizon}")
                print("  ratio " + "".join(f"{d:>24s}" for d in designs))
                for ratio, item in h1.items():
                    cells = []
                    for design in designs:
                        reject, right, stop = item["counts"][f"{design}|{alpha}|{horizon}"]
                        n = item["runs"]
                        cells.append(f"{right / n:8.4f} {(reject - right) / n:7.5f} "
                                     f"{stop / n:7.1f}")
                    print(f"  {ratio:5.3f} " + "".join(cells))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="sweep some ratios, write one JSON part")
    run.add_argument("--mode", choices=("h0", "h1"), required=True)
    run.add_argument("--ratios", default="", help="comma-separated effect_size / sigma values")
    run.add_argument("--runs", type=int, default=100_000, help="runs per ratio")
    run.add_argument("--out", required=True)
    run.set_defaults(func=cmd_run)
    report = sub.add_parser("report", help="merge JSON parts and print the tables")
    report.add_argument("parts", nargs="+")
    report.set_defaults(func=cmd_report)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
