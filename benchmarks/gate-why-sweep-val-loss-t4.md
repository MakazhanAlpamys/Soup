# Gate: `soup why` and `soup sweep` read the validation loss (Kaggle T4)

Since #1223, Soup evaluates the `data.val_split` rows after every epoch and stores the
result as `val_loss`. On main nothing reads it afterwards.
- `soup why` says "No anomalies detected" on a run whose validation loss climbed 29%.
- `soup sweep` ranks the arms by final training loss, which puts the arm that trained
  longest first.

This record judges the same six training runs twice:
- **before:** Soup main at `a75ff972`;
- **after:** the same commit with the patch applied.

Four predictions and their pass bars were written down before the first run. They are
quoted unchanged below. All four passed.

## Environment and method

**Hardware and software**
- Kaggle kernel `<kaggle-user>/ew-main` v1: 2x Tesla T4, driver 580.178.04, CUDA 13.0, 2026-11-25.
- Stack pinned to what the T4 runs in this repository's earlier work used: torch
  2.11.0+cu130, transformers 5.20.2, peft 0.21.2, trl 0.29.1, bitsandbytes 0.50.2,
  lm_eval 0.4.13.

**Two environments in one kernel**
- Soup was cloned at `a75ff972` twice: `before/` as is, and `after/` with the patch applied.
  The patch's sha256 is `6a47854e`.
- Each checkout was installed `--no-deps -e` into its own venv on top of one shared
  dependency install.
- The kernel checked that each venv imports `soup_cli` from its own checkout.

**Training: one `soup sweep` per seed, in the `after` environment, one seed per T4, seeds
0 and 1**

```
soup sweep --config soup.yaml --param training.epochs=1,2,3 --name s<seed> --yes
```

- Model and method: `Qwen/Qwen2.5-1.5B-Instruct`, QLoRA (Soup's default NF4).
- LoRA: lr 2e-4, `target_modules: all-linear`, r 16 / alpha 32.
- Batch: 1 x gradient accumulation 4, `max_length` 1024, `logging_steps` 5.
- Data: 5,000 MetaMathQA rows. They were shuffled with seed 0 before the file was written,
  so `val_split: 0.1` holds out a random 500 and trains on 4,500. Both files' sha256 were
  checked inside the kernel against the manifest.
- Everything else stays at the schema default.

**Measured per run**
- From Soup's tracker: train minutes (`duration_secs`) and every `val_loss` row.
- Held-out loss: mean NLL per assistant token on 500 other MetaMathQA rows, scored on the
  fp16 merge.
- GSM8K: `lm_eval --model hf --tasks gsm8k --num_fewshot 5 --apply_chat_template
  --fewshot_as_multiturn --batch_size 32 --seed 0`, `exact_match` under
  `flexible-extract`, n = 1,319.

**Judging**
- `soup why <run_id>` ran in both environments on every run, and the order alternated by
  seed: seed 0 before then after, seed 1 after then before.
- A sweep cannot re-rank finished runs: it trains. So:
  - the "after" table is the live sweep's own output;
  - the "before" table is main's `_display_summary`, which sorts with main's
    `_loss_sort_key`, run on the same results rebuilt from the tracker;
  - the patched `_display_summary` on that rebuilt list gave the same order as the live
    sweep in both seeds.

**Compute.**
- The main kernel ran 09:12 to 13:57 UTC: 4 h 44 min of wall time on two T4s, or 9.5
  T4-hours.
- `kaggle quota` read 10.97 h used before the first smoke kernel and 15.92 h after this
  one, so 4.95 h of quota in total. A 2x T4 session counts once.

**Analysis** ran on a CPU kernel (`<kaggle-user>/ew-analyze` v1). The paired bootstrap is over the
1,319 questions: per-question scores averaged over the two seeds, 10,000 resamples, seed 0,
95% percentile interval.

## Result

| Run | Epochs | Train min | Val loss after each epoch | Rise from lowest | Held-out loss | GSM8K | `soup why` before | `soup why` after | Sweep rank before | Sweep rank after |
|---|---:|---:|---|---:|---:|---:|---|---|---:|---:|
| s0_1 | 1 | 32.6 | 0.1648 | 0.0% | 0.1705 | 64.44 | no anomalies | no anomalies | 3 | **1** |
| s0_2 | 2 | 64.5 | 0.1701, 0.1764 | +3.7% | 0.1847 | 63.15 | no anomalies | no anomalies | 2 | 2 |
| s0_3 | 3 | 98.0 | 0.1745, 0.1814, 0.2250 | +28.9% | 0.2313 | 63.08 | no anomalies | **warns** | **1** | 3 |
| s1_1 | 1 | 32.7 | 0.1652 | 0.0% | 0.1703 | 64.22 | no anomalies | no anomalies | 3 | **1** |
| s1_2 | 2 | 64.8 | 0.1697, 0.1768 | +4.2% | 0.1849 | 63.53 | no anomalies | no anomalies | 2 | 2 |
| s1_3 | 3 | 98.0 | 0.1736, 0.1787, 0.2282 | +31.4% | 0.2334 | 64.22 | no anomalies | **warns** | **1** | 3 |

All six runs and all six evaluations completed on the first attempt.

## The frozen predictions

The bars, quoted unchanged from the plan written before the first run (`PREREG` is this
repository author's earlier pre-registration, the source of the 2.6-point noise bar):

```text
Per-question GSM8K scores are averaged over the 2 seeds before any comparison.
The paired bootstrap is `analyze.paired_bootstrap`, the pre-registered code: 10,000
resamples, seed 0, 95% percentile CI, in points. The noise bar is 2.6 GSM8K points
(Miller 2024, alpha 0.05, power 0.8, as in PREREG).

- **H1. The warning fires where it should.** Take each 3-epoch run whose tracker
  `val_loss` series rises, meaning its last value is above its minimum and the minimum came
  at an earlier evaluation. On each such run:
  - the patched `soup why` prints the validation-loss warning;
  - main's `soup why` prints nothing about validation loss. Its output contains none of
    "validation", "val_loss", "val loss" or "overfit".

  H1 passes if this holds on every such run and there is at least one such run.
  A rising 3-epoch run on which the patched rule stays silent (a rise under 5%) is a
  **missed case**. H1 then fails, and the rule is reviewed under the rule-change policy.
  If no 3-epoch run rises, H1 is untestable and reported as such.

- **H2. No false alarm on one epoch.** On each 1-epoch run the patched `soup why` prints no
  validation-loss warning. Note, stated now: with one evaluation per epoch a 1-epoch run has
  one `val_loss` row, so the rule cannot fire there. H2 checks that nothing else in the
  patch fires. It is not evidence about the 5% threshold. That evidence is the secondary
  check S1.

- **H3. The two versions rank differently, each by its own metric.** In each seed:
  - main's ranking puts first the run with the lowest final training loss;
  - the patched ranking puts first the run with the lowest last validation loss.

  H3 passes if both hold in both seeds. My prediction, recorded but not part of the bar:
  main picks `epochs=3` and the patch picks `epochs=1` in both seeds.

- **H4. The patched pick costs much less and is no worse.**
  - Let M_s and P_s be main's and the patch's first-ranked run in seed s.
  - Quality: the paired bootstrap of GSM8K (P − M), seeds averaged per question, has a CI
    lower bound above −2.6 points.
  - Cost: mean train minutes of P_0 and P_1, from Soup's tracker, is at most 0.5 times the
    mean train minutes of M_0 and M_1.

  H4 passes if both parts hold. If P_s = M_s in both seeds, H4 is reported as "no
  difference to judge" and counts as a fail of the PR's claim.
```


| | Prediction | Result |
|---|---|---|
| H1 | On a 3-epoch run whose validation loss rises, the patch warns and main says nothing about it | **Pass.** It warned on both (+28.9%, +31.4%). Main's output has none of "validation", "val_loss", "val loss" or "overfit". |
| H2 | On a 1-epoch run the patch stays silent | **Pass**, on both. A 1-epoch run has one evaluation, so the rule cannot fire there. This checks that nothing else in the patch fires. |
| H3 | Main ranks the lowest training loss first, the patch the lowest validation loss | **Pass** in both seeds. Main picks `epochs=3`, the patch picks `epochs=1`. |
| H4 | The patch's pick is no worse on GSM8K (95% interval lower bound above -2.6 points) and trains in at most half the time | **Pass.** GSM8K +0.68 points [-1.29, +2.73]. Train time 32.6 vs 98.0 min, ratio 0.33. |

## Is the advice right?

`soup why` told both 3-epoch runs to use `training.epochs: 1`. The 1-epoch run of the same
seed is that advice, already trained.

| Seed | Held-out loss, 3 epochs | Held-out loss, 1 epoch | GSM8K, 1 epoch minus 3 epochs |
|---|---:|---:|---|
| 0 | 0.2313 | 0.1705 | +1.36 [-1.21, +3.94] |
| 1 | 0.2334 | 0.1703 | 0.00 [-2.58, +2.58] |

The advice lowers held-out loss in both seeds and costs nothing on GSM8K within the noise.
It saves 65 minutes of training per run.

## What the rule misses

The 2-epoch runs' validation loss rose 3.7% and 4.2%, under the 5% rule, so `soup why`
stays silent on them. Their held-out loss (0.185) is still worse than the 1-epoch run's
(0.170), and they cost 32 more minutes.

The rule compares evaluations inside one run. It cannot see that a shorter run, which
anneals its learning rate sooner, would end lower. The 5% threshold was fixed before the
runs and is not tuned on them here. A lower threshold would have caught these two runs,
but no run in this record shows where it would start to raise false alarms.

## What this does not show

- **No accuracy claim.** The change prints a warning and reorders a table. It does not stop
  training, keep a checkpoint, or change a default. The adapter a run saves is still the one
  from its last step.
- **One task, one model, one dataset, one lr.** That is `task: sft` on Qwen2.5-1.5B with
  math word problems at lr 2e-4. How large a rise a run shows depends on all of these.
  Validation losses from other tasks (DPO, KTO) or from `eval_steps` schedules were not
  tested.
- **GSM8K differences are inside the noise.** The noise bar is 2.6 points (alpha 0.05,
  power 0.8). The claim is "no worse, a third of the time", not "better".

## Code version

- **What ran on the GPUs:** patch `6a47854e` (src and docs).
- **What the final branch adds:** one guard and one widened `except`.
  - A sweep arm with no finite training loss (diverged, or none observed) is never ranked
    on either metric, and does not count when the sweep decides which metric to rank by.
    This keeps the documented rule that such an arm cannot be the best run.
  - The timestamp parser also catches `TypeError`.
- None of the six runs touches either path. Re-running the judging with the final branch
  on the same tracker databases (a CPU kernel) gave byte-identical output for all 20 files:
  12 `soup why` transcripts, 4 sweep tables and 4 rankings.
- The plan described the sweep rule as "when every completed run has a finite last
  `val_loss`". With the guard, the rule counts ranked arms only. Every completed run in this
  record is ranked, so the two wordings agree here. The plan file itself is unchanged.

## Files

- [`evidence/why-sweep-val-loss-t4.json`](evidence/why-sweep-val-loss-t4.json) holds:
  - per run: the tracker values, the `val_loss` series with timestamps, held-out loss, and
    per-question GSM8K scores;
  - the analysis output;
  - the exact terminal output of `soup why` and of the sweep tables, before and after;
  - provenance: the Soup commit, the patch sha256, the harness file hashes, and the data
    sha256 check.
- `tests/fixtures/val_loss_t4_sweep.json`: seed 0's tracker rows, used by
  `tests/test_why_sweep_val_loss.py`.
