"""Burn-in sweep for `soup ab` (#1227): Type-I error of the two-sided mSPRT under peeking.

`soup ab` estimates the variance from the rows. From a handful of rows that estimate is too
noisy for the likelihood ratio to be a martingale, so the command gives no verdict until each
arm has `min_rows_per_arm(alpha)` rows (the burn-in). This script measures the false-positive
rate that burn-in leaves, and it is how the values in `soup_cli.utils.ab_test.
BURN_IN_ROWS_BY_ALPHA` were chosen.

Procedure, per effect_size / sigma ratio:
- H0: control and treatment rows both N(0, sigma^2), sigma = effect_size / ratio,
  effect_size 0.1 (only the ratio matters), beta 0.20.
- The operator re-runs the test after every new (control, treatment) pair and stops at the
  first terminal verdict, reject_h0 or accept_h0. Below the burn-in there is no verdict.
- A run that reaches the horizon (rows per arm) undecided stops there. Horizons 200 and 1000.
- Reported: the share of runs whose first verdict is reject_h0, for the shipped statistic
  (the two-point mixture) and, for comparison, the one-sided statistic it replaced and the
  |z| shortcut. The variance is estimated from the rows exactly as `soup ab` does (Bessel,
  pooled); `--known-variance` adds the same with the true variance.
- Criterion: rate <= alpha + 3 * sqrt(alpha * (1 - alpha) / runs) at every ratio.

`run` also checks that the vectorised statistic equals the shipped decision code's
log-likelihood ratio (`_verdict_from_summary`) at sampled points before recording anything.

Run it in parts (each part writes one JSON file), then merge:
  python benchmarks/harness/ab_burn_in_sweep.py run --ratios 0.1,0.15,0.2 --out part1.json
  python benchmarks/harness/ab_burn_in_sweep.py report part1.json part2.json ...

CPU only; about 30 s per ratio at 100,000 runs and a 1000-row horizon (about 55 s with
`--known-variance`).
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

EFFECT_SIZE = 0.1
BETA = 0.20
ALPHAS = (0.10, 0.05, 0.025, 0.01, 0.005)
BURN_INS = (2, 20, 30, 40, 50, 60)  # 2 = no burn-in: the variance needs 2 rows anyway
CANDIDATES = (20, 30, 40, 50, 60)
HORIZONS = (200, 1000)
# Each tier is calibrated at every alpha listed for it; alpha < 0.01 uses the second tier's
# value and is reported, not calibrated.
TIERS = (
    ("alpha >= 0.05", (0.05, 0.10)),
    ("0.01 <= alpha < 0.05", (0.01, 0.025)),
)
UNCALIBRATED_ALPHA = 0.005
DEFAULT_RATIOS = (
    0.1, 0.15, 0.2, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.5, 0.55, 0.6,
    0.7, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0,
)
CHUNK = 2000
BIG = 1 << 30


def bounds(alpha: float) -> tuple[float, float]:
    return math.log((1.0 - BETA) / alpha), math.log(BETA / (1.0 - alpha))


def tolerance(alpha: float, runs: int) -> float:
    return 3.0 * math.sqrt(alpha * (1.0 - alpha) / runs)


def summaries(control: np.ndarray, treatment: np.ndarray):
    """Running means and the Bessel-pooled variance for rows 2..n_max (column j <-> j + 2)."""
    rows = np.arange(2, control.shape[1] + 1, dtype=float)
    mean_c = np.cumsum(control, axis=1)[:, 1:] / rows
    mean_t = np.cumsum(treatment, axis=1)[:, 1:] / rows
    var_c = (np.cumsum(control * control, axis=1)[:, 1:] - rows * mean_c * mean_c) / (rows - 1)
    var_t = (np.cumsum(treatment * treatment, axis=1)[:, 1:] - rows * mean_t * mean_t) / (
        rows - 1
    )
    return rows, mean_c, mean_t, (var_c + var_t) / 2.0


def point_llrs(rows, mean_c, mean_t, pooled):
    """Wald's log-likelihood ratios for +effect_size and -effect_size (the shipped formula)."""
    se = np.sqrt(pooled * (2.0 / rows))
    z = (mean_t - mean_c) / se
    mu = EFFECT_SIZE / se
    n_eff = rows / 2.0
    n_ratio = n_eff / (n_eff + 1.0)
    shift = z * mu * np.sqrt(n_ratio)
    drift = 0.5 * mu * mu * n_ratio
    return shift - drift, -shift - drift


def next_true(mask: np.ndarray) -> np.ndarray:
    """For every column, the index of the first True at or after it (BIG if none)."""
    idx = np.where(mask, np.arange(mask.shape[1]), BIG)
    return np.minimum.accumulate(idx[:, ::-1], axis=1)[:, ::-1]


def count_rejections(llr: np.ndarray, alpha: float, burn_ins, horizons) -> dict:
    """Runs whose first verdict at or after the burn-in, within the horizon, is reject_h0."""
    upper, lower = bounds(alpha)
    next_up = next_true(llr >= upper)
    next_lo = next_true(llr <= lower)
    out = {}
    for burn_in in burn_ins:
        first_up = next_up[:, burn_in - 2]
        first_lo = next_lo[:, burn_in - 2]
        for horizon in horizons:
            last = horizon - 2
            out[(burn_in, horizon)] = int(((first_up < first_lo) & (first_up <= last)).sum())
    return out


def self_check(rng: np.random.Generator) -> str:
    """The vectorised mixture equals `_verdict_from_summary`'s log-likelihood ratio."""
    try:
        from soup_cli.utils.ab_test import MsprtConfig, _verdict_from_summary
    except ImportError as exc:  # pragma: no cover - reported, not fatal
        return f"self-check skipped ({exc})"
    config = MsprtConfig(metric="latency", alpha=0.05, beta=BETA, effect_size=EFFECT_SIZE)
    worst = 0.0
    points = 0
    for sigma in (0.05, 0.2, 1.0):
        control = rng.normal(0.0, sigma, size=(20, 1000))
        treatment = rng.normal(0.0, sigma, size=(20, 1000))
        rows, mean_c, mean_t, pooled = summaries(control, treatment)
        plus, minus = point_llrs(rows, mean_c, mean_t, pooled)
        mixture = np.logaddexp(plus, minus) - math.log(2.0)
        for run in range(20):
            for col in (0, 18, 98, 498, 998):
                shipped = _verdict_from_summary(
                    config,
                    n_control=int(rows[col]),
                    n_treatment=int(rows[col]),
                    mean_control=float(mean_c[run, col]),
                    mean_treatment=float(mean_t[run, col]),
                    pooled_variance=float(pooled[run, col]),
                ).log_likelihood_ratio
                ref = float(mixture[run, col])
                worst = max(worst, abs(shipped - ref) / max(1.0, abs(ref)))
                points += 1
    return f"self-check vs _verdict_from_summary: max rel diff {worst:.1e} over {points} points"


def sweep_ratio(ratio: float, runs: int, known_variance: bool) -> dict:
    rng = np.random.default_rng([1227, int(round(ratio * 10_000))])
    sigma = EFFECT_SIZE / ratio
    n_max = max(HORIZONS)
    counts: dict[str, int] = {}

    def add(key, found):
        for (burn_in, horizon), value in found.items():
            name = f"{key}|{burn_in}|{horizon}"
            counts[name] = counts.get(name, 0) + value

    for _ in range(runs // CHUNK):
        control = rng.normal(0.0, sigma, size=(CHUNK, n_max))
        treatment = rng.normal(0.0, sigma, size=(CHUNK, n_max))
        rows, mean_c, mean_t, pooled = summaries(control, treatment)
        del control, treatment
        plus, minus = point_llrs(rows, mean_c, mean_t, pooled)
        mixture = np.logaddexp(plus, minus) - math.log(2.0)
        for alpha in ALPHAS:
            add(f"estimated|mixture|{alpha}", count_rejections(mixture, alpha, BURN_INS, HORIZONS))
        for alpha in (0.05, 0.01):
            add(f"estimated|one-sided|{alpha}", count_rejections(plus, alpha, BURN_INS, HORIZONS))
            add(
                f"estimated|abs(z)|{alpha}",
                count_rejections(np.maximum(plus, minus), alpha, BURN_INS, HORIZONS),
            )
        if known_variance:
            plus_k, minus_k = point_llrs(rows, mean_c, mean_t, np.full_like(pooled, sigma**2))
            mixture_k = np.logaddexp(plus_k, minus_k) - math.log(2.0)
            for alpha in (0.05, 0.01):
                add(f"known|mixture|{alpha}", count_rejections(mixture_k, alpha, BURN_INS,
                                                               HORIZONS))
                add(
                    f"known|abs(z)|{alpha}",
                    count_rejections(np.maximum(plus_k, minus_k), alpha, BURN_INS, HORIZONS),
                )
    return {"ratio": ratio, "runs": (runs // CHUNK) * CHUNK, "counts": counts}


def cmd_run(args: argparse.Namespace) -> None:
    ratios = [float(r) for r in args.ratios.split(",")] if args.ratios else list(DEFAULT_RATIOS)
    print(self_check(np.random.default_rng(99)), flush=True)
    results = []
    for ratio in ratios:
        start = time.perf_counter()
        results.append(sweep_ratio(ratio, args.runs, args.known_variance))
        print(f"ratio {ratio}: {time.perf_counter() - start:.0f} s", flush=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"beta": BETA, "effect_size": EFFECT_SIZE, "results": results}, fh, indent=1)
        fh.write(chr(10))  # end the file with a newline
    print(f"wrote {args.out}")


def _rates(parts: list[str]) -> tuple[dict, int]:
    merged: dict[float, dict] = {}
    runs = None
    for path in parts:
        with open(path, encoding="utf-8") as fh:
            for item in json.load(fh)["results"]:
                merged[item["ratio"]] = item
                runs = item["runs"] if runs is None else min(runs, item["runs"])
    rates = {
        ratio: {k: v / item["runs"] for k, v in item["counts"].items()}
        for ratio, item in sorted(merged.items())
    }
    return rates, runs


def _worst(rates: dict, key: str) -> tuple[float, float]:
    ratio = max(rates, key=lambda r: rates[r].get(key, -1.0))
    return rates[ratio][key], ratio


def cmd_report(args: argparse.Namespace) -> None:
    rates, runs = _rates(args.parts)
    ratios = list(rates)
    print(f"runs per ratio: {runs}; ratios: {', '.join(f'{r:g}' for r in ratios)}")
    print(f"beta {BETA}; bound = alpha + 3 * sqrt(alpha * (1 - alpha) / {runs}); * = above it")
    for horizon in sorted(HORIZONS, reverse=True):
        for alpha in ALPHAS:
            bound = alpha + tolerance(alpha, runs)
            print(f"\nestimated variance, mixture, alpha {alpha}, horizon {horizon} rows per arm "
                  f"(bound {bound:.5f}); rejection rate by burn-in N (N=2: none)")
            print("  ratio " + " ".join(f"{'N=' + str(b):>9s}" for b in BURN_INS))
            for ratio in ratios:
                cells = []
                for burn_in in BURN_INS:
                    rate = rates[ratio][f"estimated|mixture|{alpha}|{burn_in}|{horizon}"]
                    cells.append(f"{rate:8.5f}{'*' if rate > bound else ' '}")
                print((f"  {ratio:5.3f} " + " ".join(cells)).rstrip())
            print("  worst: " + "  ".join(
                "N={}: {:.5f} at {:g}".format(
                    b, *_worst(rates, f"estimated|mixture|{alpha}|{b}|{horizon}"))
                for b in BURN_INS))
    print("\nDECISION: per tier, the smallest N in "
          f"{CANDIDATES} within the bound at every ratio, at every alpha of the tier, at both "
          f"horizons {HORIZONS}")
    chosen = {}
    for tier, alphas in TIERS:
        for burn_in in CANDIDATES:
            ok = True
            notes = []
            for alpha in alphas:
                bound = alpha + tolerance(alpha, runs)
                for horizon in HORIZONS:
                    worst, at = _worst(rates, f"estimated|mixture|{alpha}|{burn_in}|{horizon}")
                    notes.append(f"a{alpha}@{horizon}: {worst:.5f} at {at:g} (bound {bound:.5f})")
                    ok = ok and worst <= bound
            print(f"  {tier:22s} N={burn_in}: {'PASS' if ok else 'fail'}; " + "; ".join(notes))
            if ok:
                chosen[tier] = burn_in
                break
    print("  chosen: " + "; ".join(f"{tier} -> {n}" for tier, n in chosen.items()))
    if "0.01 <= alpha < 0.05" in chosen:
        n_low = chosen["0.01 <= alpha < 0.05"]
        bound = UNCALIBRATED_ALPHA + tolerance(UNCALIBRATED_ALPHA, runs)
        for horizon in HORIZONS:
            worst, at = _worst(
                rates, f"estimated|mixture|{UNCALIBRATED_ALPHA}|{n_low}|{horizon}"
            )
            print(f"  alpha {UNCALIBRATED_ALPHA} (below the calibrated range) with N={n_low}, "
                  f"horizon {horizon}: worst {worst:.5f} at {at:g} (bound {bound:.5f})")
    print("\nfor comparison, estimated variance, worst over ratios (horizon 1000 / 200):")
    for label in ("one-sided", "abs(z)"):
        for alpha in (0.05, 0.01):
            line = []
            for burn_in in BURN_INS:
                w1, r1 = _worst(rates, f"estimated|{label}|{alpha}|{burn_in}|1000")
                w2, r2 = _worst(rates, f"estimated|{label}|{alpha}|{burn_in}|200")
                line.append(f"N={burn_in}: {w1:.4f} ({r1:g}) / {w2:.4f} ({r2:g})")
            print(f"  {label:9s} alpha {alpha}: " + "; ".join(line))
    known = [r for r in ratios if any(k.startswith("known|") for k in rates[r])]
    if known:
        print("\nknown variance (the martingale argument without the estimate), worst over "
              "ratios (horizon 1000 / 200):")
        for label in ("mixture", "abs(z)"):
            for alpha in (0.05, 0.01):
                line = []
                for burn_in in BURN_INS:
                    w1, r1 = _worst({r: rates[r] for r in known},
                                    f"known|{label}|{alpha}|{burn_in}|1000")
                    w2, r2 = _worst({r: rates[r] for r in known},
                                    f"known|{label}|{alpha}|{burn_in}|200")
                    line.append(f"N={burn_in}: {w1:.4f} ({r1:g}) / {w2:.4f} ({r2:g})")
                print(f"  {label:7s} alpha {alpha}: " + "; ".join(line))
    if args.ratios_detail:
        wanted = [float(r) for r in args.ratios_detail.split(",")]
        print("\nper-ratio detail (estimated variance), horizon 200 / 1000:")
        for ratio in wanted:
            if ratio not in rates:
                continue
            for key in sorted(k for k in rates[ratio] if k.startswith("estimated|")):
                label, alpha, burn_in, horizon = key.split("|")[1:]
                if horizon == "200":
                    other = rates[ratio][key.rsplit("|", 1)[0] + "|1000"]
                    print(f"  ratio {ratio:g} {label:9s} alpha {alpha:5s} N={burn_in:>2s}: "
                          f"{rates[ratio][key]:.5f} / {other:.5f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="sweep some ratios, write one JSON part")
    run.add_argument("--ratios", default="", help="comma-separated effect_size / sigma values")
    run.add_argument("--runs", type=int, default=100_000, help="runs per ratio")
    run.add_argument("--known-variance", action="store_true", help="also plug in sigma^2")
    run.add_argument("--out", required=True)
    run.set_defaults(func=cmd_run)
    report = sub.add_parser("report", help="merge JSON parts and print the tables")
    report.add_argument("parts", nargs="+")
    report.add_argument("--ratios-detail", default="", help="ratios to print in full")
    report.set_defaults(func=cmd_report)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
