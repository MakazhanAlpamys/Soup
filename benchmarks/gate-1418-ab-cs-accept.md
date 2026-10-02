# `soup ab` confidence-sequence accept rule (#1418)

Measured on 2026-10-02 for [#1418](https://github.com/MakazhanAlpamys/Soup/issues/1418), on
CPU. This is the record behind the accept rule in
[`src/soup_cli/utils/ab_test.py`](../src/soup_cli/utils/ab_test.py). It changes only how
`soup ab` reaches `accept_h0`; the statistic and the reject rule are
[#1265](gate-1265-ab-nig.md)'s.

## What it gates

Under #1265, `soup ab` accepted H0 once the normal-inverse-gamma Bayes factor fell to
`log(beta / (1 - alpha))`. A mixture Bayes factor gathers evidence for H0 only like
-0.5 log(n_eff g), so with no difference it took 621 pairs to accept at effect_size / sigma
0.3 (alpha 0.05, beta 0.20, runs to 1000 rows), against 107 for the #1227 burn-in design.
Part of the burn-in design's speed was wrong: it ended about 18% of runs with a real
difference of effect_size in `accept_h0`. The aim was to accept sooner without giving that
back.

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
still allow is smaller than the one the operator asked about. A difference of effect_size or
more then ends in `accept_h0` in at most a `level` share of runs, however often the operator
looks. What `level` should be is the `--beta` question below; this PR ships level = beta.

## Headline

Mean pairs at stop when the two arms are the same, alpha 0.05, beta 0.20, runs to 1000 rows
per arm, 100,000 runs per ratio:

| effect_size / sd | 0.2 | 0.3 | 0.4 | 0.5 | 0.7 | 1 | 1.5 | 2 | 5 |
|---|---|---|---|---|---|---|---|---|---|
| #1265 (today) | 890 | 621 | 407 | 281 | 155 | 83 | 42 | 27 | 10 |
| **sequence at level beta (shipped)** | **479** | **220** | **128** | **85** | **47** | **27** | **16** | **12** | **8** |
| sequence at level alpha | 661 | 307 | 177 | 117 | 64 | 35 | 20 | 15 | 9 |
| #1227 burn-in | 235 | 107 | 62 | 43 | 34 | 31 | 30 | 30 | 30 |

Against a true difference of exactly effect_size (20,000 runs per ratio), the worst share of
runs ending in `accept_h0`, over 17 ratios from 0.1 to 5, alpha 0.05, runs to 1000 rows:

| | beta 0.20 | beta 0.10 |
|---|---|---|
| #1265 (today) | 0.0042 | 0.0000 |
| sequence at level beta (shipped) | 0.0389 | 0.0175 |
| sequence at level alpha | 0.0083 | 0.0083 |
| #1227 burn-in (from #1265's record) | about 0.18 | |

Accepting H0 is 2.8 to 3.3 times faster from ratio 0.3 to 1, and from ratio 1 up it is
faster than the burn-in design too. Wrong accepts stay far below beta, in every cell of the
sweep (at most 0.0430, at alpha 0.005). They are higher than today's, which is where the
speed comes from; both `--beta` options are laid out below.

## Procedure

- **Data and looks.** As in #1265's record: effect_size 0.1, sigma = 0.1 / ratio, the same
  22 ratios; H0 both arms N(0, sigma²), 100,000 runs per ratio; H1 the treatment shifted by
  +effect_size, 20,000 runs per ratio at 17 of the ratios; a look after every pair from pair 7
  on, the first terminal verdict stops the run, horizons 1000 and 200 rows per arm;
  `PRIOR_SCALE_ROWS = 5`; alphas 0.1, 0.05, 0.025, 0.01 and 0.005; betas 0.20 and 0.10.
- **Same runs.** The seeds and the order of the draws are `ab_nig_sweep.py`'s, so every design
  sees exactly the runs #1265's record saw.
- **Designs.** `gate-1265`: #1265's rule without the accept hold, as its record ran it.
  `nig`: #1265's rule with the hold it shipped, the baseline. `cs`: the sequence at level
  beta, with the hold (shipped). `cs-alpha`: the sequence at level alpha, with the hold.
  `cs-nohold`: `cs` without the hold.
- **Self-check.** Every harness part first compares its vectorised half-width with the shipped
  `_nig_confidence_half_width` (maximum relative difference 0 over 45 points) and replays
  simulated runs through `msprt_step` itself (0 of 32 runs differ), in all 11 parts.
- **Replay check.** Every part also checks its `gate-1265` counts against
  `results/gate-1265-ab-nig/`: 0 of 390 cells differ across the 11 parts, so the baseline
  is the recorded one, not a re-implementation of it.

## `reject_h0` is unchanged

The reject rule is the same function at the same boundary, so the reject decision at any one
look is identical. What can change is how often a run gets to a look that rejects, because an
accept now ends some runs sooner.

- **Type-I can only go down,** and does. Worst rate over the 22 ratios, runs to 1000 rows
  (bound alpha + 3 SE, as in #1265): alpha 0.05 0.0328 today, 0.0271 shipped, 0.0289 at
  level alpha; alpha 0.01 0.0066, 0.0053, 0.0058. Every alpha and both horizons are in
  `report.txt`; none is above its bound.
- **Power at exactly effect_size moves by the wrong accepts.** Share of H1 runs rejecting
  in the right direction, alpha 0.05, runs to 1000 rows:

| ratio | 0.1 | 0.15 | 0.2 | 0.3 | 0.5 | 1 | 2 | 5 |
|---|---|---|---|---|---|---|---|---|
| #1265 (today), beta 0.20 | 0.311 | 0.724 | 0.946 | 0.997 | 0.997 | 0.997 | 0.999 | 1.000 |
| shipped, beta 0.20 | 0.311 | 0.724 | 0.936 | 0.964 | 0.966 | 0.964 | 0.965 | 0.988 |
| shipped, beta 0.10 | 0.311 | 0.724 | 0.945 | 0.985 | 0.984 | 0.984 | 0.985 | 0.997 |
| level alpha | 0.311 | 0.724 | 0.947 | 0.993 | 0.993 | 0.992 | 0.994 | 1.000 |

Each run that stops in a wrong `accept_h0` is a run that would otherwise, almost always, have
gone on to reject. Wrong-direction rejections were at most 0.0004 in every design.

## What `--beta` means now: two options

The maintainer asked for both, with their consequences, and for the choice to be his.

**Option A, `--beta` is the sequence's level (what this PR implements).** `--beta` keeps its
name and its reading, "a real difference of `--effect-size` ends in `accept_h0` in at most
this share of runs", and that is now a guarantee under a look after every pair, which the
old boundary only approximated. Consequences:

- At the default beta 0.20, wrong accepts at exactly effect_size go from at most 0.0042 to
  at most 0.0389 (at most 0.043 over every alpha), and power there from 0.997 to about 0.965.
  At beta 0.10: at most 0.0175 and about 0.985.
- Accept is about 3 times faster at the default, and far more at beta 0.10, where today's
  boundary `log(0.1/0.95)` is so low that accept almost never comes within 1000 rows (958
  pairs at ratio 0.3; shipped 265).
- Anyone who passes `--beta` gets a test that accepts sooner and is a little more willing to
  accept a borderline difference, within the bound they set. No flag or config changes.
- `alpha + beta < 1` (#1339) stays. Its reason changes: at or past 1 a coin flip that never
  reads the rows meets both rates. The message now says that.

**Option B, the sequence at level alpha, `--beta` retired.** One line in `msprt_step`
(`level=config.alpha`), plus a deprecation: `--beta` still parses but warns that it no
longer does anything, and the `alpha + beta` check goes. Consequences:

- Wrong accepts at most 0.0083 at alpha 0.05 and 0.0019 at alpha 0.01, close to today; power
  0.993 at alpha 0.05, 0.998 at 0.01.
- Accept is 2 to 2.4 times faster than today at alpha 0.05 from ratio 0.3 to 1 (307 pairs
  at ratio 0.3), less so at alpha 0.01 and small ratios (840 against 921 at ratio 0.2), and
  one knob now sets both error rates.
- Anyone who passes `--beta` sees a warning and no effect, and a config or script that
  relies on it to tune accept speed loses that.

A middle way is also possible: keep `--beta` as in A but change its default from 0.20 to,
say, 0.10, which halves the wrong accepts for about 20% more rows.

## The accept hold

#1265 shipped a hold: while the tested rows' standard deviation is more than 3 times the
held-out rows', `accept_h0` waits as `continue`. It was there because a saturated start
makes the prior far too wide, and the Bayes factor then fell to the old accept boundary
whether or not there was a difference. The sequence does not need it: its coverage holds for
any prior scale fixed before the tested rows, so a saturated start only changes how wide the
sequence is, not whether it covers. In the sweep the hold changes nothing measurable
(`cs` against `cs-nohold`: at most 1 pair and 0.001 in accept share). In
`tests/test_issue1265_ab_nig.py`'s saturated-start rows, without the hold both shifts are
accepted at 11 and 18 rows per arm, and both are right: the shifts are below
`--effect-size`. This PR keeps the hold, as #1265 designed it, and offers to remove it here
or in a follow-up if the maintainer agrees.

## Limits

- **Gaussian rows,** as before. Binary metrics should be pre-aggregated per prompt.
- **H1 at exactly effect_size only.** That is the hardest case for the accept guarantee; a
  larger difference is accepted less often.
- **The guarantee is loose here.** Wrong accepts measured well under beta. A tighter
  sequence would accept sooner still, but would need its own argument.
- **Below ratio 0.2** neither rule accepts much within 1000 rows, as before.

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
```

The parts, the merged report and the run log are in
[`results/gate-1418-ab-cs-accept/`](results/gate-1418-ab-cs-accept/): `h0-part1.json` to
`h0-part8.json`, `h1-part1.json` to `h1-part3.json`, `report.txt` and `sweep.log`. The seeds
are fixed per mode and ratio, so the same numpy reproduces the numbers above.
