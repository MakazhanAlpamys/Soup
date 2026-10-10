# `soup ab` confidence-sequence accept rule (#1418)

Measured on 2026-10-02 for [#1418](https://github.com/MakazhanAlpamys/Soup/issues/1418), on
CPU. This is the record behind the accept rule in
[`src/soup_cli/utils/ab_test.py`](../src/soup_cli/utils/ab_test.py). It changes only how
`soup ab` reaches `accept_h0`; the statistic and the reject rule are
[#1265](gate-1265-ab-nig.md)'s. `--beta` is retired.

## What it gates

Under #1265, `soup ab` accepted H0 once the normal-inverse-gamma Bayes factor fell to
`log(beta / (1 - alpha))`. A mixture Bayes factor gathers evidence for H0 only like
-0.5 log(n_eff g), so with no difference it took 621 pairs to accept at effect_size / sigma
0.3 (alpha 0.05, beta 0.20, runs to 1000 rows), against 107 for the #1227 burn-in design.
Part of the burn-in design's speed was wrong: it ended about 18% of runs with a real
difference of effect_size in `accept_h0`. The aim was to accept sooner without giving that
back, and without moving `reject_h0`'s Type-I rate or power.

The reject rule is unchanged: `reject_h0` once the Bayes factor reaches `1/alpha`. The
accept rule now reads the same mixture the other way. Shifting the treatment by a candidate
difference delta0 leaves the pooled variance alone and moves t to
(mean difference - delta0) / se, so the Bayes factor at that t tests "the difference is
delta0", and it is a martingale when delta0 is the true difference. The differences it has
not rejected at `1/level` form an always-valid confidence sequence, mean difference +- h, and
by Ville's inequality the true difference stays inside it at every look with probability at
least 1 - level. The Bayes factor grows with t², so h has a closed form. With
spread = 1 + n_eff g and nu, n_eff as in #1265:

```text
c  = exp(-(2 log(1/level) + log(spread)) / (nu + 1))
t² = nu (1 - c) / (c - 1/spread)          (h = inf when c <= 1/spread)
h  = sqrt(t² * pooled_variance / n_eff)
```

`soup ab` accepts H0 once |mean difference| + h < effect_size: every difference the rows
still allow is smaller than the one the operator asked about. **The level is `--alpha`.** A
difference of effect_size or more then ends in `accept_h0` in at most an alpha share of
runs, however often the operator looks, so `--alpha` bounds both wrong verdicts and `--beta`
has nothing left to set. Why alpha and not beta is below, with the beta numbers kept.

## Headline

Mean pairs at stop when the two arms are the same, runs to 1000 rows per arm, 100,000 runs
per ratio:

| effect_size / sd | 0.2 | 0.3 | 0.4 | 0.5 | 0.7 | 1 | 1.5 | 2 | 5 |
|---|---|---|---|---|---|---|---|---|---|
| #1265, alpha 0.05, beta 0.20 | 890 | 621 | 407 | 281 | 155 | 83 | 42 | 27 | 10 |
| **#1418, alpha 0.05** | **661** | **307** | **177** | **117** | **64** | **35** | **20** | **15** | **9** |
| #1265, alpha 0.01, beta 0.20 | 921 | 671 | 448 | 314 | 175 | 94 | 47 | 30 | 11 |
| **#1418, alpha 0.01** | **840** | **411** | **236** | **154** | **83** | **45** | **25** | **18** | **10** |
| #1227 burn-in, alpha 0.05 | 235 | 107 | 62 | 43 | 34 | 31 | 30 | 30 | 30 |

Accepting H0 is 2.0 to 2.4 times faster at alpha 0.05 from ratio 0.3 to 1 (1.6 to 2.1 at
alpha 0.01), and from ratio 1.5 up it is faster than the burn-in design too.

Against a true difference of exactly effect_size (20,000 runs per ratio, 17 ratios from 0.1
to 5, runs to 1000 rows):

| `--alpha` | worst wrong `accept_h0`, #1265 (beta 0.20) | worst wrong `accept_h0`, #1418 | power from ratio 0.3 up, #1265 | power from ratio 0.3 up, #1418 |
|---|---|---|---|---|
| 0.10 | 0.0063 | 0.0158 | 0.99360 | 0.98420 |
| 0.05 | 0.0042 | 0.0083 | 0.99585 | 0.99165 |
| 0.025 | 0.0036 | 0.0042 | 0.99650 | 0.99585 |
| 0.01 | 0.0032 | 0.0019 | 0.99685 | 0.99805 |
| 0.005 | 0.0031 | 0.0012 | 0.99665 | 0.99815 |

Power is the smallest share of runs rejecting in the right direction over ratios 0.3 to 5
(exact at 20,000 runs, so not rounded).
The #1227 burn-in design was at about 0.18 wrong accepts and 0.82 power between ratios 0.2
and 0.5.

## Procedure

- **Data and looks.** As in #1265's record: effect_size 0.1, sigma = 0.1 / ratio, the same
  22 ratios; H0 both arms N(0, sigma²), 100,000 runs per ratio; H1 the treatment shifted by
  +effect_size, 20,000 runs per ratio at 17 of the ratios; a look after every pair from pair 7
  on, the first terminal verdict stops the run, horizons 1000 and 200 rows per arm;
  `PRIOR_SCALE_ROWS = 5`; alphas 0.1, 0.05, 0.025, 0.01 and 0.005; betas 0.20 and 0.10 for
  the designs that use one.
- **Same runs.** The seeds and the order of the draws are `ab_nig_sweep.py`'s, so every design
  sees exactly the runs #1265's record saw.
- **Designs.** `gate-1265`: #1265's rule without the accept hold, as its record ran it.
  `nig`: #1265's rule with the hold it shipped, the baseline. `cs-alpha`: the sequence at
  level alpha, with the hold (shipped). `cs`: the sequence at level beta, with the hold (the
  option not taken). `cs-nohold`: `cs` without the hold.
- **Self-check.** Every harness part first compares its vectorised half-width with the shipped
  `_nig_confidence_half_width` (maximum relative difference 0 over 45 points) and replays
  simulated runs through `msprt_step` itself (0 of 32 runs differ). The sweep ran while the
  branch shipped level beta, so the parts checked `cs`; after the switch to level alpha the
  harness checks `cs-alpha`, and `check` gives the same 0 and 0 (end of `sweep.log`). Pointed
  at the old level instead, it reports 21 of 32 runs differing, so it tells the two apart.
  No count depends on which design ships, so the sweep was not re-run.
- **Replay check.** Every part also checks its `gate-1265` counts against
  `results/gate-1265-ab-nig/`: 0 of 390 cells differ across the 11 parts, so the baseline is
  the recorded one, not a re-implementation of it.

## `reject_h0` is unchanged

The reject rule is the same function at the same boundary, so the reject decision at any one
look is identical. What can change is how often a run gets to a look that rejects, because an
accept now ends some runs sooner.

- **Type-I can only go down,** and does. Worst rate over the 22 ratios, runs to 1000 rows
  (bound alpha + 3 SE, as in #1265):

| `--alpha` | 0.10 | 0.05 | 0.025 | 0.01 | 0.005 |
|---|---|---|---|---|---|
| #1265 | 0.0676 | 0.0328 | 0.0164 | 0.0066 | 0.0034 |
| #1418 | 0.0594 | 0.0289 | 0.0145 | 0.0058 | 0.0030 |

  At a 200-row horizon the same holds (0.0287 to 0.0258 at alpha 0.05); none is above its
  bound (`report.txt`).
- **Power at exactly effect_size moves by the wrong accepts.** Share of H1 runs rejecting
  in the right direction, runs to 1000 rows:

| ratio | 0.1 | 0.15 | 0.2 | 0.3 | 0.5 | 1 | 2 | 5 |
|---|---|---|---|---|---|---|---|---|
| #1265, alpha 0.05 | 0.311 | 0.724 | 0.946 | 0.997 | 0.997 | 0.997 | 0.999 | 1.000 |
| #1418, alpha 0.05 | 0.311 | 0.724 | 0.947 | 0.993 | 0.993 | 0.992 | 0.994 | 1.000 |
| #1265, alpha 0.01 | 0.130 | 0.513 | 0.859 | 0.997 | 0.997 | 0.998 | 0.999 | 1.000 |
| #1418, alpha 0.01 | 0.130 | 0.513 | 0.860 | 0.998 | 0.998 | 0.999 | 0.999 | 1.000 |

From ratio 0.3 up, at alpha 0.05 power is 0.004 to 0.005 lower; at alpha 0.01 and below it is the same or a
little higher, because #1418's wrong accepts there are rarer than #1265's. Each run that
stops in a wrong `accept_h0` is a run that would otherwise, almost always, have gone on to
reject. Wrong-direction rejections were at most 0.0004 in every design.

## Why level alpha, and what `--beta` became

Two readings were measured and laid out in the PR for a maintainer decision:

- **Level beta (`cs`, not taken).** `--beta` would keep its name and bound wrong accepts at
  beta. At the default beta 0.20 that accepts faster (220 pairs at ratio 0.3, alpha 0.05)
  but ends up to 0.0389 of runs at exactly effect_size in a wrong `accept_h0` (0.0430 over
  every alpha), so power there falls from 0.997 to about 0.965. At beta 0.10: up to 0.0175,
  power about 0.985. Full tables in `report.txt`.
- **Level alpha (`cs-alpha`, shipped).** Power at effect_size stays within 0.005 of today at
  alpha 0.05 (0.0094 lower at 0.1) and at or above it from alpha 0.01 down, which is #1418's bar: `reject_h0` Type-I and
  power unchanged.

So `--beta` is retired. It still parses, for one release: `soup ab --beta X` prints that
`--beta` no longer does anything, that `--alpha` now bounds wrong accepts too and that a
stricter accept means a lower `--alpha`, and then runs exactly as without it.
`MsprtConfig(beta=...)` raises the same message as a `SoupConfigDeprecationWarning`. Both end
with the release that will refuse it, read from `DEPRECATED_VALUE_REJECTION_VERSION`, the same
one-release staging Soup uses for deprecated config values (#759). #1339's
`alpha + beta < 1` check guarded the old accept boundary and is gone with it.

## The accept hold

#1265 shipped a hold: while the tested rows' standard deviation is more than 3 times the
held-out rows', `accept_h0` waits as `continue`. It is unchanged here; whether it stays is
[#1524](https://github.com/MakazhanAlpamys/Soup/issues/1524)'s question. For that issue:
the sequence's coverage holds for any prior scale fixed before the tested rows, so a
saturated start changes how wide the sequence is, not whether it covers. In the sweep the
hold changes nothing measurable (`cs` against `cs-nohold`: under 1 pair and at most 0.001 in
accept share). In `tests/test_issue1265_ab_nig.py`'s saturated-start rows without the hold,
both shifts are accepted at 12 and 19 rows per arm, and both accepts are right: the shifts,
0 and -0.03, are below `--effect-size` 0.1.

## What `log_likelihood_ratio` shows on an accept

The verdict's `log_likelihood_ratio` (in the table and the webhook payload) is still the
Bayes factor against "no difference", the statistic of the reject rule. It is not what
accepts, so on an `accept_h0` it can be anything below `log(1/alpha)`, positive included: in
511 simulated accepts at alpha 0.05 (effect sizes 0.5 to 3 sd, true differences from 0 to
0.8 effect_size), 82 had a positive value, the largest 2.93 against the reject edge 3.00.
A positive value there means the rows lean toward a difference, one smaller than
effect_size.

## Limits

- **Gaussian rows,** as before. Binary metrics should be pre-aggregated per prompt.
- **H1 at exactly effect_size only.** That is the hardest case for the accept guarantee; a
  larger difference is accepted less often.
- **Short runs at small effect sizes.** Capped at 200 rows per arm and at ratio 0.3 or
  below, #1418 reaches `accept_h0` less often than #1265 did (0.017 against 0.041 of runs at
  ratio 0.3, alpha 0.05); from ratio 0.35 up it is far more often (0.426 against 0.096 at
  0.35). Neither rule accepts much there within 200 rows.
- **The guarantee is loose here.** Wrong accepts measured well under alpha.

## Environment

Windows 11, Python 3.12.13, numpy 2.5.2, CPU only. About 430 to 600 s per ratio for an H0
part at 100,000 runs over 1000 rows (8 parts in parallel), about 130 s per ratio for H1 at
20,000.

## Reproducing

The harness is [`harness/ab_cs_accept_sweep.py`](harness/ab_cs_accept_sweep.py); it imports
#1265's [`harness/ab_nig_sweep.py`](harness/ab_nig_sweep.py). Run it in parts, then merge:

```bash
python benchmarks/harness/ab_cs_accept_sweep.py run --mode h0 --ratios 0.1,0.15,0.2 --out h0-part1.json
python benchmarks/harness/ab_cs_accept_sweep.py run --mode h0 --ratios 0.25,0.275,0.3 --out h0-part2.json
python benchmarks/harness/ab_cs_accept_sweep.py run --mode h0 --ratios 0.325,0.35,0.375 --out h0-part3.json
python benchmarks/harness/ab_cs_accept_sweep.py run --mode h0 --ratios 0.4,0.425,0.45 --out h0-part4.json
python benchmarks/harness/ab_cs_accept_sweep.py run --mode h0 --ratios 0.5,0.55,0.6 --out h0-part5.json
python benchmarks/harness/ab_cs_accept_sweep.py run --mode h0 --ratios 0.7,0.8,1.0 --out h0-part6.json
python benchmarks/harness/ab_cs_accept_sweep.py run --mode h0 --ratios 1.5,2.0 --out h0-part7.json
python benchmarks/harness/ab_cs_accept_sweep.py run --mode h0 --ratios 3.0,5.0 --out h0-part8.json
python benchmarks/harness/ab_cs_accept_sweep.py run --mode h1 --runs 20000 \
    --ratios 0.1,0.15,0.2,0.25,0.3,0.35 --out h1-part1.json
python benchmarks/harness/ab_cs_accept_sweep.py run --mode h1 --runs 20000 \
    --ratios 0.4,0.45,0.5,0.6,0.7,0.8 --out h1-part2.json
python benchmarks/harness/ab_cs_accept_sweep.py run --mode h1 --runs 20000 \
    --ratios 1.0,1.5,2.0,3.0,5.0 --out h1-part3.json
python benchmarks/harness/ab_cs_accept_sweep.py report h0-part*.json h1-part*.json
python benchmarks/harness/ab_cs_accept_sweep.py check
```

The parts, the merged report and the run log are in
[`results/gate-1418-ab-cs-accept/`](results/gate-1418-ab-cs-accept/): `h0-part1.json` to
`h0-part8.json`, `h1-part1.json` to `h1-part3.json`, `report.txt` and `sweep.log`. The seeds
are fixed per mode and ratio, so the same numpy reproduces the numbers above.
