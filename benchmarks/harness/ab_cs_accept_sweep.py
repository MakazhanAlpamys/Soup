"""Confidence-sequence accept rule for `soup ab` (#1418): time to accept_h0 and wrong accepts.

#1265 accepts H0 once the normal-inverse-gamma Bayes factor falls to log(beta / (1 - alpha)).
#1418 keeps the Bayes factor for reject_h0 and accepts once the always-valid confidence
sequence for the difference, the same mixture inverted, lies inside (-effect_size,
+effect_size). This script measures the two against each other, at the two levels the
sequence could run at; #1418 ships level alpha and retires `--beta`.

Procedure, as in `ab_nig_sweep.py` (effect_size 0.1, sigma = effect_size / ratio, k = 5
held-out rows per arm, a look after every pair from pair 7 on, the first terminal verdict
stops the run, horizons 200 and 1000 rows per arm). H0: both arms N(0, sigma^2). H1: the
treatment shifted by +effect_size, the smallest difference the operator asked to detect, so a
run that ends in accept_h0 is a wrong accept. The seeds and the order of the draws are
`ab_nig_sweep.py`'s, so the same runs are replayed.

Designs (`|alpha|beta` in the keys; reject_h0 at log(1/alpha) in all of them, tested first):
- `gate-1265`: accept at log(beta / (1 - alpha)), no accept hold: the design
  `gate-1265-ab-nig.md` recorded. `run` checks its counts against that record's JSON.
- `nig`: the same with the accept hold #1265 shipped (tested pooled sd > 3 x held-out sd).
  The baseline: today's rule.
- `cs`: accept once the confidence sequence at coverage 1 - beta lies inside +-effect_size,
  with the hold (`--beta` as the sequence's level: measured in the record, not shipped).
- `cs-alpha`: the same at coverage 1 - alpha, with the hold. What #1418 ships (`--beta`
  retired). Keyed under every beta, though beta does not enter it.
- `cs-nohold`: `cs` without the hold.

Recorded per design: runs whose first verdict is reject_h0, of them with diff > 0, runs whose
first verdict is accept_h0, and the sum of pairs at stop.

`check` runs only the self-check, so the shipped rule can be re-checked without a sweep.
Run it in parts (each part writes one JSON file), then merge:
  python benchmarks/harness/ab_cs_accept_sweep.py run --mode h0 --ratios 0.1,0.2 --out h0-1.json
  python benchmarks/harness/ab_cs_accept_sweep.py report h0-1.json h1-1.json ...

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
from ab_burn_in_sweep import DEFAULT_RATIOS, next_true, summaries, tolerance  # noqa: E402
from ab_nig_sweep import (  # noqa: E402
    ALPHAS,
    CHUNK,
    EFFECT_SIZE,
    HORIZONS,
    split_nig,
)

HELD = 5
BETAS = (0.20, 0.10)
HOLD_RATIO = 3.0
DESIGNS = ("gate-1265", "nig", "cs", "cs-alpha", "cs-nohold")
GATE_1265 = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results", "gate-1265-ab-nig"
)


def half_width(rows, pooled, prior_variance, level: float) -> np.ndarray:
    """Vectorised `_nig_confidence_half_width`, rows per arm equal (column j <-> j + 2)."""
    nu = 2.0 * rows - 2.0
    n_eff = rows / 2.0
    spread = 1.0 + n_eff * prior_variance
    c = np.exp(-(2.0 * math.log(1.0 / level) + np.log(spread)) / (nu + 1.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_squared = np.where(c > 1.0 / spread, nu * (1.0 - c) / (c - 1.0 / spread), np.inf)
    return np.sqrt(t_squared * pooled / n_eff)


def tested(control: np.ndarray, treatment: np.ndarray):
    """Bayes factor, diff, half-width inputs and the hold, padded so column j <-> pair j + 2."""
    stat, diff = split_nig(control, treatment, HELD)
    head_c, head_t = control[:, :HELD], treatment[:, :HELD]
    scale_variance = (head_c.var(axis=1, ddof=1) + head_t.var(axis=1, ddof=1)) / 2.0
    prior = (EFFECT_SIZE * EFFECT_SIZE / scale_variance)[:, None]
    rows, _, _, pooled = summaries(control[:, HELD:], treatment[:, HELD:])
    hold = np.sqrt(pooled) > HOLD_RATIO * np.sqrt(scale_variance)[:, None]

    def pad(values, fill):
        return np.concatenate([np.full((control.shape[0], HELD), fill), values], axis=1)

    return stat, diff, (rows, pooled, prior), pad(hold, True)


def first_verdicts(reject, accept, diff, horizons) -> dict:
    """Per horizon: (rejections, rejections with diff > 0, acceptances, sum of pairs at stop).

    Both masks are padded; the first HELD columns can be neither. reject is tested first.
    """
    next_up = next_true(reject)[:, HELD]
    next_lo = next_true(accept)[:, HELD]
    out = {}
    for horizon in horizons:
        last = horizon - 2
        rejected = (next_up <= next_lo) & (next_up <= last)
        accepted = (next_lo < next_up) & (next_lo <= last)
        col = np.where(rejected, next_up, 0)
        positive = rejected & (np.take_along_axis(diff, col[:, None], axis=1)[:, 0] > 0)
        stop = np.minimum(np.minimum(next_up, next_lo), last) + 2
        out[horizon] = (int(rejected.sum()), int(positive.sum()), int(accepted.sum()),
                        int(stop.sum()))
    return out


def self_check(rng: np.random.Generator) -> str:
    """The vectorised half-width and the cs-alpha design equal the shipped code."""
    try:
        from soup_cli.utils.ab_test import (
            PRIOR_SCALE_ROWS,
            MsprtConfig,
            _nig_confidence_half_width,
            msprt_step,
        )
    except ImportError as exc:  # pragma: no cover - reported, not fatal
        return f"self-check skipped ({exc})"
    assert PRIOR_SCALE_ROWS == HELD
    worst = 0.0
    points = 0
    for rows_per_arm in (2, 3, 10, 57, 400):
        for prior in (0.01, 0.7, 40.0):
            for level in (0.2, 0.05, 0.005):
                ref = _nig_confidence_half_width(
                    n_control=rows_per_arm, n_treatment=rows_per_arm, pooled_variance=0.3,
                    prior_variance=prior, level=level,
                )
                got = float(half_width(np.array([float(rows_per_arm)]), 0.3, prior, level)[0])
                if math.isinf(ref) or math.isinf(got):
                    worst = max(worst, 0.0 if ref == got else math.inf)
                else:
                    worst = max(worst, abs(ref - got) / ref)
                points += 1
    mismatched = 0
    runs = 0
    for ratio, shift in ((0.5, 0.0), (1.0, 0.0), (1.0, EFFECT_SIZE), (3.0, 0.5 * EFFECT_SIZE)):
        sigma = EFFECT_SIZE / ratio
        control = rng.normal(0.0, sigma, size=(8, 150))
        treatment = rng.normal(shift, sigma, size=(8, 150))
        stat, diff, (rows, pooled, prior), hold = tested(control, treatment)
        width = half_width(rows, pooled, prior, 0.05)
        accept = np.concatenate([np.zeros((8, HELD), bool), np.abs(diff[:, HELD:]) + width
                                 < EFFECT_SIZE], axis=1) & ~hold
        reject = stat >= math.log(1.0 / 0.05)
        config = MsprtConfig(metric="judge_score", alpha=0.05, effect_size=EFFECT_SIZE)
        for run in range(8):
            hits = np.flatnonzero(reject[run] | accept[run])
            want = None
            if hits.size:
                col = int(hits[0])
                want = ("reject_h0" if reject[run, col] else "accept_h0", col + 2)
            got = None
            for pairs in range(HELD + 2, 151):
                verdict = msprt_step(config, control=control[run, :pairs].tolist(),
                                     treatment=treatment[run, :pairs].tolist())
                if verdict.decision != "continue":
                    got = (verdict.decision, pairs)
                    break
            mismatched += got != want
            runs += 1
    return (f"self-check: max rel diff {worst:.1e} over {points} points vs "
            f"_nig_confidence_half_width; {mismatched} of {runs} msprt_step runs differ")


def sweep_ratio(ratio: float, mode: str, runs: int) -> dict:
    # ab_nig_sweep.py's seed, so these are the runs gate-1265-ab-nig.md recorded.
    rng = np.random.default_rng([1265, 0 if mode == "h0" else 1, int(round(ratio * 10_000))])
    sigma = EFFECT_SIZE / ratio
    shift = 0.0 if mode == "h0" else EFFECT_SIZE
    n_max = max(HORIZONS)
    counts: dict[str, list[int]] = {}

    def add(key, found):
        for horizon, values in found.items():
            name = f"{key}|{horizon}"
            old = counts.get(name, [0, 0, 0, 0])
            counts[name] = [a + b for a, b in zip(old, values)]

    for _ in range(runs // CHUNK):
        control = rng.normal(0.0, sigma, size=(CHUNK, n_max))
        treatment = rng.normal(shift, sigma, size=(CHUNK, n_max))
        stat, diff, (rows, pooled, prior), hold = tested(control, treatment)
        del control, treatment
        size = np.abs(diff)
        inside = {}
        for level in sorted(set(ALPHAS) | set(BETAS)):
            width = half_width(rows, pooled, prior, level)
            inside[level] = np.concatenate(
                [np.zeros((CHUNK, HELD), bool), size[:, HELD:] + width < EFFECT_SIZE], axis=1
            )
        for alpha in ALPHAS:
            reject = stat >= math.log(1.0 / alpha)
            for beta in BETAS:
                bf_low = stat <= math.log(beta / (1.0 - alpha))
                bf_low[:, :HELD] = False
                masks = {
                    "gate-1265": bf_low,
                    "nig": bf_low & ~hold,
                    "cs": inside[beta] & ~hold,
                    "cs-alpha": inside[alpha] & ~hold,
                    "cs-nohold": inside[beta],
                }
                for design, accept in masks.items():
                    add(f"{design}|{alpha}|{beta}",
                        first_verdicts(reject, accept, diff, HORIZONS))
    return {"ratio": ratio, "mode": mode, "runs": (runs // CHUNK) * CHUNK, "counts": counts}


def check_against_gate_1265(results: list[dict]) -> str:
    """The `gate-1265` design must give gate-1265-ab-nig.md's nig-k5 counts exactly."""
    recorded = {}
    for name in sorted(os.listdir(GATE_1265)):
        if name.endswith(".json"):
            with open(os.path.join(GATE_1265, name), encoding="utf-8") as fh:
                for item in json.load(fh)["results"]:
                    recorded[(item["mode"], item["ratio"])] = item
    compared = differ = 0
    for item in results:
        old = recorded.get((item["mode"], item["ratio"]))
        if old is None or old["runs"] != item["runs"]:
            continue
        for alpha in ALPHAS:
            for horizon in HORIZONS:
                reject, positive, _, stop = item["counts"][f"gate-1265|{alpha}|0.2|{horizon}"]
                compared += 1
                differ += [reject, positive, stop] != old["counts"][f"nig-k5|{alpha}|{horizon}"]
    return f"gate-1265 replay: {differ} of {compared} cells differ from the recorded counts"


def cmd_run(args: argparse.Namespace) -> None:
    ratios = [float(r) for r in args.ratios.split(",")] if args.ratios else list(DEFAULT_RATIOS)
    print(self_check(np.random.default_rng(1418)), flush=True)
    results = []
    for ratio in ratios:
        start = time.perf_counter()
        results.append(sweep_ratio(ratio, args.mode, args.runs))
        print(f"{args.mode} ratio {ratio}: {time.perf_counter() - start:.0f} s", flush=True)
    print(check_against_gate_1265(results), flush=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"effect_size": EFFECT_SIZE, "results": results}, fh, indent=1)
        fh.write(chr(10))
    print(f"wrote {args.out}")


def _load(parts: list[str]) -> dict:
    merged: dict[tuple[str, float], dict] = {}
    for path in parts:
        with open(path, encoding="utf-8") as fh:
            for item in json.load(fh)["results"]:
                merged[(item["mode"], item["ratio"])] = item
    return merged


def _cell(item, design, alpha, beta, horizon):
    reject, positive, accept, stop = item["counts"][f"{design}|{alpha}|{beta}|{horizon}"]
    n = item["runs"]
    return reject / n, positive / n, accept / n, stop / n


def cmd_report(args: argparse.Namespace) -> None:
    merged = _load(args.parts)
    shown = ("nig", "cs", "cs-alpha", "cs-nohold")
    h0 = {r: item for (mode, r), item in sorted(merged.items()) if mode == "h0"}
    h1 = {r: item for (mode, r), item in sorted(merged.items()) if mode == "h1"}
    if h0:
        runs = min(item["runs"] for item in h0.values())
        print(f"H0: time to a verdict (mean pairs at stop) and share ending in accept_h0, "
              f"{runs} runs per ratio")
        for horizon in sorted(HORIZONS, reverse=True):
            for alpha, beta in ((0.05, 0.2), (0.01, 0.2), (0.05, 0.1), (0.01, 0.1)):
                print(f"\nalpha {alpha}, beta {beta}, horizon {horizon}")
                print("  ratio " + "".join(f"{d:>18s}" for d in shown))
                for ratio, item in h0.items():
                    cells = []
                    for design in shown:
                        _, _, accept, stop = _cell(item, design, alpha, beta, horizon)
                        cells.append(f"{stop:8.1f} / {accept:6.4f}")
                    print(f"  {ratio:5.3f} " + "".join(f"{c:>18s}" for c in cells))
        print(f"\nH0: Type-I rate (first verdict reject_h0), beta 0.2, {runs} runs per ratio; "
              "bound = alpha + 3 SE, * = above it")
        for horizon in sorted(HORIZONS, reverse=True):
            for alpha in ALPHAS:
                bound = alpha + tolerance(alpha, runs)
                worst = {}
                for design in DESIGNS:
                    rates = [(_cell(item, design, alpha, 0.2, horizon)[0], ratio)
                             for ratio, item in h0.items()]
                    worst[design] = max(rates)
                print(f"  alpha {alpha:<6} horizon {horizon:<5} bound {bound:.5f}  " + "  ".join(
                    f"{d} {w:.5f}{'*' if w > bound else ''} @{r:g}"
                    for d, (w, r) in worst.items()))
    if h1:
        runs = min(item["runs"] for item in h1.values())
        print(f"\nH1 (true difference = +effect_size): power (reject_h0, direction right) / "
              f"wrong accept_h0 / mean pairs at stop, {runs} runs per ratio")
        for horizon in sorted(HORIZONS, reverse=True):
            for alpha, beta in ((0.05, 0.2), (0.01, 0.2), (0.05, 0.1), (0.01, 0.1)):
                print(f"\nalpha {alpha}, beta {beta}, horizon {horizon}")
                print("  ratio " + "".join(f"{d:>26s}" for d in shown))
                for ratio, item in h1.items():
                    cells = []
                    for design in shown:
                        _, right, accept, stop = _cell(item, design, alpha, beta, horizon)
                        cells.append(f"{right:7.4f} {accept:7.4f} {stop:7.1f}")
                    print(f"  {ratio:5.3f} " + "".join(f"{c:>26s}" for c in cells))
        print("\nH1: worst wrong accept_h0 over the ratios (the guarantee: at most beta)")
        for horizon in sorted(HORIZONS, reverse=True):
            for alpha in ALPHAS:
                for beta in BETAS:
                    worst = {d: max((_cell(item, d, alpha, beta, horizon)[2], ratio)
                                    for ratio, item in h1.items()) for d in DESIGNS}
                    print(f"  alpha {alpha:<6} beta {beta:<5} horizon {horizon:<5} " + "  ".join(
                        f"{d} {w:.4f} @{r:g}" for d, (w, r) in worst.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="sweep some ratios, write one JSON part")
    run.add_argument("--mode", choices=("h0", "h1"), required=True)
    run.add_argument("--ratios", default="", help="comma-separated effect_size / sigma values")
    run.add_argument("--runs", type=int, default=100_000, help="runs per ratio")
    run.add_argument("--out", required=True)
    run.set_defaults(func=cmd_run)
    check = sub.add_parser("check", help="only the self-check against the shipped code")
    check.set_defaults(func=lambda _: print(self_check(np.random.default_rng(1418))))
    report = sub.add_parser("report", help="merge JSON parts and print the tables")
    report.add_argument("parts", nargs="+")
    report.set_defaults(func=cmd_report)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
