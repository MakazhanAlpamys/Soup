# `soup ab` burn-in calibration (#1227)

Measured on 2026-09-25 for [#1227](https://github.com/MakazhanAlpamys/Soup/issues/1227), on
CPU. This is the record behind `BURN_IN_ROWS_BY_ALPHA` and `CALIBRATED_HORIZON_ROWS` in
[`src/soup_cli/utils/ab_test.py`](../src/soup_cli/utils/ab_test.py).

## What it gates

`soup ab` runs a two-sided mSPRT. It averages Wald's likelihood ratios for a
treatment-minus-control difference of `+effect_size` and of `-effect_size`, and it estimates
the variance from the rows. From a handful of rows that estimate can come out far too small.
An operator who re-runs the command after every new pair of rows would then reject a true H0
far more often than `--alpha`. The simulation below puts that at up to **0.164** at alpha
0.05 without a burn-in; the one-sided test the two-sided one replaced reached **0.109**.

So `soup ab` gives no verdict, only `continue`, until each arm has `min_rows_per_arm(alpha)`
rows. This record is how those values were chosen, and how far they hold.

## Headline

| `--alpha` | burn-in (rows per arm) | worst Type-I, runs to 1000 rows | worst, runs to 200 rows | bound (alpha + 3 SE) |
|---|---|---|---|---|
| 0.05 and above | **30** | alpha 0.05: **0.0512** (ratio 0.35) · alpha 0.10: 0.0994 | 0.0492 · 0.0934 | 0.0521 · 0.1029 |
| 0.01 to below 0.05 | **40** | alpha 0.01: **0.0109** (0.01088, ratio 0.4) · alpha 0.025: 0.0261 | 0.0105 · 0.0244 | 0.0109 (0.01094) · 0.0265 |
| below 0.01 | 40, **not calibrated** | alpha 0.005: 0.0058, above its bound | 0.0056 | 0.0057 |

"Ratio" is effect_size / sigma, the only thing the decision depends on. The calibration holds
up to **1000 rows per arm**; past that, `soup ab` prints a warning. Below alpha 0.01 it prints
another.

## Procedure

- **Data.** H0: control and treatment rows both N(0, sigma²), with effect_size 0.1 and
  sigma = 0.1 / ratio. There are 22 ratios: 0.1, 0.15, 0.2, 0.25, 0.275, 0.3, 0.325, 0.35,
  0.375, 0.4, 0.425, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 1, 1.5, 2, 3 and 5.
- **Procedure.** A look after every new pair. The first terminal verdict (`reject_h0` or
  `accept_h0`) stops the run, and there is no verdict below the burn-in. A run still undecided
  at the horizon stops there. Horizons are 1000 and 200 rows per arm.
- **Statistic.** It is computed exactly as `soup ab` computes it: the variance is estimated
  from the rows (Bessel-corrected, pooled), with beta 0.20.
  - Every run of the harness first checks its vectorised statistic against the shipped
    decision code, `_verdict_from_summary`: maximum relative difference 0 over 300 points,
    in all four parts.
- **Runs.** 100,000 per ratio, seeded per ratio.
- **Criterion.** For each alpha tier, the smallest burn-in in {20, 30, 40, 50, 60} whose rate
  stays at or below alpha + 3·sqrt(alpha(1−alpha)/100000) at **every ratio**, at **both
  horizons**, and at **every alpha measured in the tier**: 0.05 and 0.10 for the first tier,
  0.01 and 0.025 for the second.

## Decision

Worst rate over the 22 ratios. A ✗ marks a value above the bound.

| tier | alpha | horizon | N = 20 | N = 30 | N = 40 | N = 50 | N = 60 |
|---|---|---|---|---|---|---|---|
| ≥ 0.05 | 0.05 | 1000 | 0.05342 ✗ | **0.05119** | 0.04977 | 0.04965 | 0.04937 |
| ≥ 0.05 | 0.05 | 200 | 0.05206 | 0.04919 | 0.04672 | 0.04409 | 0.04210 |
| ≥ 0.05 | 0.10 | 1000 | 0.10073 | 0.09935 | 0.09817 | 0.09757 | 0.09674 |
| ≥ 0.05 | 0.10 | 200 | 0.09948 | 0.09335 | 0.08784 | 0.08419 | 0.08060 |
| 0.01–0.05 | 0.01 | 1000 | 0.01224 ✗ | 0.01135 ✗ | **0.01088** | 0.01077 | 0.01075 |
| 0.01–0.05 | 0.01 | 200 | 0.01224 ✗ | 0.01111 ✗ | 0.01045 | 0.00988 | 0.00934 |
| 0.01–0.05 | 0.025 | 1000 | 0.02764 ✗ | 0.02686 ✗ | 0.02611 | 0.02511 | 0.02496 |
| 0.01–0.05 | 0.025 | 200 | 0.02719 ✗ | 0.02550 | 0.02440 | 0.02305 | 0.02185 |
| < 0.01 | 0.005 | 1000 | 0.00721 ✗ | 0.00601 ✗ | 0.00577 ✗ | 0.00563 | 0.00560 |
| < 0.01 | 0.005 | 200 | 0.00721 ✗ | 0.00601 ✗ | 0.00558 | 0.00539 | 0.00501 |

The bounds are 0.05207 at alpha 0.05, 0.10285 at 0.10, 0.01094 at 0.01, 0.02648 at 0.025 and
0.00567 at 0.005.

**Chosen:** 30 rows at alpha ≥ 0.05, and 40 rows at 0.01 ≤ alpha < 0.05. Below 0.01 the value
stays 40 with a warning, because 40 rows do not pass at alpha 0.005 over 1000 rows.

## What this shows

- **The horizon matters.** An earlier calibration stopped every run at 200 rows. It picked 20
  rows at alpha 0.05, and here 20 still passes at 200 rows (0.05206). Followed to 1000 rows,
  runs that are still undecided at 200 keep producing false rejections, and 20 rows reach
  0.05342. So the burn-in is calibrated for runs of up to 1000 rows per arm.
- **The small-ratio tail does not depend on the burn-in.** At ratio 0.1, 30 rows give 0.0003
  at 200 rows and 0.0278 at 1000 (alpha 0.05), because those rejections come late, long after
  any burn-in. The same holds at alpha 0.01: past 40 rows the worst case moves only from
  0.01088 to 0.01075 (N = 60), a floor set around ratio 0.25.
- **The alpha 0.01 tier passes narrowly:** 0.01088 against 0.01094, about 0.2 standard errors.
  An independent review of the earlier code measured the same cells at 0.0101–0.0116 in other
  seed sets. Treat the true worst case as roughly alpha + 10%, which is within the tolerance
  the criterion allows, not below alpha.
- **With the true variance plugged in**, the same statistic stays below alpha everywhere: worst
  0.0468 at alpha 0.05 and 0.0095 at alpha 0.01, over 1000 rows. What the burn-in leaves is
  the variance estimate's effect, not the mixture's.
- **The alternatives, estimated variance, worst over ratios at 1000 rows:**

  | statistic | no burn-in, alpha 0.05 | 30 rows, alpha 0.05 | no burn-in, alpha 0.01 | 40 rows, alpha 0.01 |
  |---|---|---|---|---|
  | two-point mixture (shipped) | 0.1637 | 0.0512 | 0.1009 | 0.0109 |
  | one-sided (the replaced test) | 0.1091 | 0.0512 | 0.0603 | 0.0109 |
  | the `\|z\|` shortcut | 0.2232 | 0.1074 | 0.1216 | 0.0221 |

  `|z|` in the one-sided formula roughly doubles the rate at any burn-in. This is why the
  statistic averages the two ratios instead.

## What this does not show

- **beta 0.20 only.** An independent check on the earlier code, with the same runs at beta
  0.05, 0.10 and 0.30, found a lower beta more conservative and 0.30 within noise of 0.20.
- **Gaussian rows only.** `soup ab` documents that binary metrics should be pre-aggregated per
  prompt.
- **Nothing past 1000 rows per arm.** Small ratios keep accumulating there, which is why
  `soup ab` warns.
- **Not a variance-robust statistic.** A statistic that would need no burn-in is
  [#1265](https://github.com/MakazhanAlpamys/Soup/issues/1265).

## Environment

Windows 11, Python 3.12.10, numpy 2.5.3, CPU only. About 55 s per ratio at 100,000 runs over
1000 rows, with `--known-variance`.

## Reproducing

The harness is [`harness/ab_burn_in_sweep.py`](harness/ab_burn_in_sweep.py). Run it in parts,
then merge; the parts are split to fit a 10-minute session:

```bash
python benchmarks/harness/ab_burn_in_sweep.py run --runs 100000 --known-variance \
    --ratios 0.1,0.15,0.2,0.25,0.275,0.3,0.325 --out part1.json
python benchmarks/harness/ab_burn_in_sweep.py run --runs 100000 --known-variance \
    --ratios 0.35,0.375,0.4,0.425,0.45,0.5,0.55 --out part2.json
python benchmarks/harness/ab_burn_in_sweep.py run --runs 100000 --known-variance \
    --ratios 0.6,0.7,0.8,1.0 --out part3.json
python benchmarks/harness/ab_burn_in_sweep.py run --runs 100000 --known-variance \
    --ratios 1.5,2.0,3.0,5.0 --out part4.json
python benchmarks/harness/ab_burn_in_sweep.py report part1.json part2.json part3.json part4.json \
    --ratios-detail 0.25,0.4,0.425,0.55,1.0,2.0
```

The parts and the merged report from this run are in
[`results/gate-1227-ab-burn-in/`](results/gate-1227-ab-burn-in/): `part1.json` to `part4.json`
and `sweep.log`. The seeds are fixed per ratio, so the same numpy reproduces the numbers above.
