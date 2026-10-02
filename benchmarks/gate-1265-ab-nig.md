# `soup ab` normal-inverse-gamma mixture (#1265)

Measured on 2026-09-27 for [#1265](https://github.com/MakazhanAlpamys/Soup/issues/1265), on
CPU. This is the record behind the statistic and `PRIOR_SCALE_ROWS` in
[`src/soup_cli/utils/ab_test.py`](../src/soup_cli/utils/ab_test.py). It replaces the burn-in
of [#1227](gate-1227-ab-burn-in.md).

**Accept rule superseded by [#1418](gate-1418-ab-cs-accept.md):** `soup ab` now accepts H0
once a confidence sequence from the same mixture lies inside +-effect_size, not at
`log(beta/(1-alpha))`. The statistic, `PRIOR_SCALE_ROWS` and the reject rule recorded here are
unchanged, and this record's runs are the baseline that sweep replays.

## What it gates

The #1227 statistic plugged the estimated variance into a likelihood ratio built for a known
one. From a handful of rows that estimate is too noisy, so `soup ab` held every verdict back
until 30 rows per arm (40 below alpha 0.05). Even so, at alpha 0.01 the Type-I rate under
peeking stayed near 0.011, longer burn-ins did not lower it, and below alpha 0.01 or past
1000 rows per arm it was not calibrated at all.

The statistic is now the normal-inverse-gamma Bayes factor in its right-Haar limit: a flat
prior on the common mean, 1/sigma on sigma, and N(0, g) on the standardised difference
d = delta / sigma. It has a closed form in the pooled two-sample t statistic (Gönen, Johnson,
Lu and Westfall, 2005), with nu = n_c + n_t - 2 and n_eff = n_c n_t / (n_c + n_t):

```text
log BF = -0.5 log(1 + n_eff g) - (nu + 1)/2 [log1p(t² / (nu (1 + n_eff g))) - log1p(t² / nu)]
```

It is scale-invariant, so under H0 its law does not depend on sigma, and as a ratio of
marginal likelihoods it is a martingale under H0. Ville's inequality bounds the chance that it
ever reaches 1/alpha by alpha: from the first rows on, at any horizon, at any alpha. `soup ab`
rejects at `log(1/alpha)` and accepts at `log(beta/(1-alpha))`.

`--effect-size` is in the metric's units and g needs it in standard deviations. The first
`PRIOR_SCALE_ROWS` rows of each arm are held out; their pooled standard deviation s0 sets
g = (effect_size / s0)², and the Bayes factor runs on the rows after them. Because the prior
is fixed before those rows are seen, the guarantee stays exact. This record measures the
design against the burn-in design and picks `PRIOR_SCALE_ROWS`.

## Headline

Worst Type-I rate over 22 values of effect_size / sigma, 100,000 runs each (the bound is
alpha + 3 SE; * = above it):

| `--alpha` | NIG, k = 5, runs to 1000 rows | runs to 200 rows | burn-in (#1227), runs to 1000 rows | bound |
|---|---|---|---|---|
| 0.10 | **0.0676** | 0.0576 | 0.0999 | 0.1029 |
| 0.05 | **0.0328** | 0.0287 | 0.0511 | 0.0521 |
| 0.025 | **0.0164** | 0.0139 | 0.0257 | 0.0265 |
| 0.01 | **0.0066** | 0.0055 | 0.0110 * | 0.0109 |
| 0.005 | **0.0034** | 0.0027 | 0.0058 * | 0.0057 |

The mixture stays well below alpha everywhere, with no burn-in and no alpha or horizon limit,
so the 0.01 floor and both `soup ab` warnings go. **`PRIOR_SCALE_ROWS = 5`**: verdicts can
come from 7 rows per arm.

## Procedure

- **Data.** effect_size 0.1, sigma = 0.1 / ratio, 22 ratios: 0.1, 0.15, 0.2, 0.25, 0.275,
  0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 1, 1.5, 2, 3 and 5.
  H0: both arms N(0, sigma²), 100,000 runs per ratio. H1: the treatment shifted by
  +effect_size, the smallest difference the operator asked to detect, 20,000 runs per ratio
  (17 of the ratios).
- **Procedure.** A look after every new pair; the first terminal verdict stops the run; a run
  still undecided at the horizon stops there. Horizons 1000 and 200 rows per arm. beta 0.20.
- **Designs.** NIG with k = 3, 5 and 8 held-out rows per arm (verdicts from pair k + 2), and
  the #1227 burn-in design (30 rows at alpha >= 0.05, 40 below, reject at
  `log((1-beta)/alpha)`), computed by the #1227 harness's own functions.
- **Self-check.** Every harness run first compares its vectorised Bayes factor with the
  shipped `_nig_log_bayes_factor` (maximum relative difference 0 over 150 points) and replays
  simulated runs through `msprt_step` itself (0 of 64 peeks differ), in all 11 parts.
- **Closed form.** `tests/test_issue1265_ab_nig.py` also checks the closed form against a
  direct quadrature of the two marginal likelihoods (to 2e-4, including unequal arms and
  nu = 2).

## Power against the asked-for effect

Share of H1 runs that end in `reject_h0` with the right direction, and mean pairs at stop,
runs to 1000 rows per arm. Wrong-direction rejections were at most 0.0002 in the designs shown.

| ratio | burn-in, alpha 0.05 | NIG k=5, alpha 0.05 | burn-in, alpha 0.01 | NIG k=5, alpha 0.01 |
|---|---|---|---|---|
| 0.1 | 0.446 / 800 | 0.311 / 885 | 0.176 / 922 | 0.130 / 960 |
| 0.15 | 0.768 / 450 | 0.724 / 652 | 0.637 / 628 | 0.513 / 805 |
| 0.2 | 0.811 / 261 | **0.946** / 431 | 0.790 / 389 | **0.859** / 583 |
| 0.3 | 0.817 / 119 | **0.996** / 201 | 0.818 / 177 | **0.997** / 279 |
| 0.5 | 0.840 / 49 | **0.996** / 79 | 0.864 / 71 | **0.997** / 107 |
| 1 | 0.977 / 31 | **0.997** / 26 | 0.989 / 41 | **0.997** / 33 |
| 2 | 1.000 / 30 | 0.998 / 12 | 1.000 / 40 | 0.999 / 15 |
| 5 | 1.000 / 30 | 1.000 / 8 | 1.000 / 40 | 1.000 / 9 |

From ratio 0.2 up, the mixture detects the effect more often, and from ratio 1 up it also
decides sooner. The burn-in design's power stalls near 0.82 between ratios 0.2 and 0.5
because it wrongly ends about 18% of those runs in `accept_h0`. That is also why it stops
sooner there. Below ratio 0.2 neither design is reliable within 1000 rows (a fixed-n test
needs about 1570 per arm at ratio 0.1), and the burn-in design detects more, while sitting at
its Type-I limit. At a 200-row horizon the same pattern holds, shifted: the burn-in design
detects more up to ratio 0.35 at alpha 0.05 (0.4 at alpha 0.01) and the mixture above that
(full tables in `report.txt`).

## What it costs: time to `accept_h0`

Under H0 the mixture needs more rows to accept. Mean pairs at stop, runs to 1000 rows (the
"H0: time to accept_h0" section of `report.txt`):

| ratio | 0.2 | 0.3 | 0.4 | 0.5 | 0.7 | 1 | 1.5 | 2 | 5 |
|---|---|---|---|---|---|---|---|---|---|
| burn-in, alpha 0.05 | 235 | 107 | 62 | 43 | 34 | 31 | 30 | 30 | 30 |
| NIG k=5, alpha 0.05 | 889 | 621 | 406 | 281 | 155 | 82 | 41 | 26 | 10 |

A mixture Bayes factor gathers evidence for H0 only like -0.5 log(n_eff g), so reaching
`log(beta/(1-alpha))` takes rows. The burn-in design's early acceptances come from the same
plug-in variance that costs it power under H1. `continue` remains the honest answer until
then: `soup ab` never claims `accept_h0` early.

## Choosing `PRIOR_SCALE_ROWS`

Type-I is within its bound for every k, as it must be. Power at ratio >= 0.3, alpha 0.05,
1000 rows: about 0.980 for k = 3, 0.996 for k = 5, 0.999 for k = 8. Each held-out row adds
about one pair to every decision, and k = 3 loses 1.5 points of power to a noisier prior
scale. k = 5 keeps almost all of k = 8's power for 3 fewer rows.

## Limits

- **Gaussian rows.** As before, binary metrics should be pre-aggregated per prompt.
- **beta 0.20 only** in the H1 runs; beta sets only the accept boundary now.
- **Row order matters for the held-out rows.** The first k rows of each arm, in file order,
  set the scale. If an arm's first rows are all identical, more rows are held out until one
  differs; this depends only on the held-out rows, so the guarantee is unchanged.

## Environment

Windows 11, Python 3.12.13, numpy 2.5.2, CPU only. About 350 s per ratio for an H0 part at
100,000 runs over 1000 rows (8 parts in parallel), about 70 s per ratio for H1 at 20,000.

## Reproducing

The harness is [`harness/ab_nig_sweep.py`](harness/ab_nig_sweep.py); it imports the burn-in
design from [`harness/ab_burn_in_sweep.py`](harness/ab_burn_in_sweep.py). Run it in parts,
then merge:

```bash
python benchmarks/harness/ab_nig_sweep.py run --mode h0 --ratios 0.1,0.15,0.2 --out h0-part1.json
python benchmarks/harness/ab_nig_sweep.py run --mode h0 --ratios 0.25,0.275,0.3 --out h0-part2.json
python benchmarks/harness/ab_nig_sweep.py run --mode h0 --ratios 0.325,0.35,0.375 --out h0-part3.json
python benchmarks/harness/ab_nig_sweep.py run --mode h0 --ratios 0.4,0.425,0.45 --out h0-part4.json
python benchmarks/harness/ab_nig_sweep.py run --mode h0 --ratios 0.5,0.55,0.6 --out h0-part5.json
python benchmarks/harness/ab_nig_sweep.py run --mode h0 --ratios 0.7,0.8,1.0 --out h0-part6.json
python benchmarks/harness/ab_nig_sweep.py run --mode h0 --ratios 1.5,2.0 --out h0-part7.json
python benchmarks/harness/ab_nig_sweep.py run --mode h0 --ratios 3.0,5.0 --out h0-part8.json
python benchmarks/harness/ab_nig_sweep.py run --mode h1 --runs 20000 \
    --ratios 0.1,0.15,0.2,0.25,0.3,0.35 --out h1-part1.json
python benchmarks/harness/ab_nig_sweep.py run --mode h1 --runs 20000 \
    --ratios 0.4,0.45,0.5,0.6,0.7,0.8 --out h1-part2.json
python benchmarks/harness/ab_nig_sweep.py run --mode h1 --runs 20000 \
    --ratios 1.0,1.5,2.0,3.0,5.0 --out h1-part3.json
python benchmarks/harness/ab_nig_sweep.py report h0-part*.json h1-part*.json
```

The parts, the merged report and the run log are in
[`results/gate-1265-ab-nig/`](results/gate-1265-ab-nig/): `h0-part1.json` to `h0-part8.json`,
`h1-part1.json` to `h1-part3.json`, `report.txt` and `sweep.log`. The seeds are fixed per
mode and ratio, so the same numpy reproduces the numbers above.
