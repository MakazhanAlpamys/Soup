<!--
Working measurement record, published verbatim. The two hypotheses that were
refuted by their own covariate, the shape that completed while allocating more
memory than the card has, and the four refusals are kept in the order they
happened.

Hardware: RTX 5070 Laptop GPU (Blackwell sm_120, 8151 MiB by nvidia-smi,
driver 616.92, CUDA 13.4), Intel i9-14900HX, Windows 11 Pro 26200.
THE MACHINE WAS ON BATTERY for every run below (PowerOnline False, 79%,
discharging ~57 W): the GPU's enforced power limit was 50.00 W against a
115.00 W maximum, so no throughput figure here is a statement about what this
card can do on mains.
Stack: Python 3.12.10, torch 2.14.0+cu130, transformers 5.17.0, peft 0.20.0,
trl 0.29.1, bitsandbytes 0.50.2, accelerate 1.15.0, datasets 5.0.1.
torchao is NOT installed. Every older number in benchmarks/ up to the
2026-09-10 box change was taken on an RTX 3050 Laptop 4 GB and is NOT
comparable to anything here.
Soup: detached worktree at origin/main 71ecbbb3 (a6d52a7d = PR #1146, the
commit this record gates), C:\Users\user\projects\Soup-b836, run as
PYTHONPATH=<worktree>/src python -m soup_cli.cli, with
soup_cli.__file__ checked under the worktree before the first run.
Harness: none. `soup bench train` IS the harness; the fixture that drives it
is quoted in section 2 and committed under benchmarks/results/gate-836/
together with every JSON report quoted below.
-->

# Gate record — #836: what `soup bench train` reports, on a real card

**Status: the contract holds; the number it reports does not repeat.**
2026-09-22. Acceptance item 7 of [#836](https://github.com/MakazhanAlpamys/Soup/issues/836)
asked for a published reference run of `soup bench train`, which shipped in
[#1146](https://github.com/MakazhanAlpamys/Soup/pull/1146). This is it, and the
useful half is not the reference number.

Every field the command promises was measured and every one of them held.
Across **thirteen runs of a byte-identical config** the config hash, the
supervised and total token counts, `max_memory_allocated`, `max_memory_reserved`
and all three checks were **identical to the byte** — and the reported
throughput ranged from **376.4 to 915.3 supervised tokens/s, a 2.43x spread**
(coefficient of variation 35.3%). The spread is **additive, not
multiplicative**: across four shapes whose fast-mode step time spans 26x
(0.197 s to 5.26 s) the fastest and slowest repeat of each shape differ by a
roughly constant **0.17–0.28 s per step**. That is the number a future
comparison has to clear, and on a 0.2 s step it is 1.9x while on a 5.3 s step
it is 4.6%.

**This is not a throughput claim.** One card, on battery, at a 50 W cap; two
small models; one sequence shape each; no quality claim of any kind. What is
being gated here is the *contract* — that the command reports what it says it
reports, refuses when the measurement cannot mean anything, and does not
fabricate an absent value.

Unit convention: decimal MB/GB unless a figure is a power of two, which is
written in MiB/GiB.

---

## 0. What this record is for

`soup bench train` exists so a throughput number carries evidence that the
model was training while it was measured. So the two questions here are:

1. Does it report what it promises, on a real card, end to end?
2. What is the noise floor, i.e. what size of change could a future
   comparison of two `soup bench train` numbers actually resolve?

Question 2 turned out to be the whole record. Question 1 has a short answer:
yes, every field, including the four refusals in section 9.

## 1. The box, and why nothing older is comparable

| | value | how read |
|---|---|---|
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU | `nvidia-smi`; report `provenance.card` |
| VRAM | 8151 MiB (8547.04 MB) | `nvidia-smi` |
| compute capability | 12.0 (sm_120, Blackwell) | report `provenance.compute_capability` |
| driver / CUDA runtime | 616.92 / 13.0 | report `provenance.driver`, `provenance.cuda_runtime` |
| max SM clock | 3090 MHz; observed to 2797 MHz under load | `nvidia-smi --query-gpu=clocks.max.sm` |
| **power** | **current and default limit 50.00 W, max 115.00 W** | `nvidia-smi -q -d POWER` |
| **AC** | **absent — `PowerOnline: False`, battery 79%, discharging 57.2 W** | `root\WMI` `BatteryStatus` |
| OS / Python | Windows-11-10.0.26200-SP0 / 3.12.10 | report `provenance` |
| dtype / optimizer | `torch.bfloat16` / `OptimizerNames.ADAMW_TORCH` | report `provenance` |
| seed / data_seed | 1234 / 1234 | set in the config, echoed in `provenance` |

The battery line is not colour. A laptop GPU capped at 50 W of a possible 115 W
is running at 43% of its power budget, and shape B below drew 47.7 W peak — i.e.
it was against the cap. Re-running any of this on mains is a **new baseline**,
not a comparison.

## 2. The fixture, and the four shapes

Two fully-cached public models, a synthetic fixture generated once and hashed,
LoRA in every arm, `quantization: none`, `transformers` backend, `task: sft`.

| file | sha256 | what |
|---|---|---|
| `bench_fixture.jsonl` | `5fcb5e40f8fd16f8ad6ca63e4da169ec5be1b482a98e7fd3196652a4dd9265fd` | 64 alpaca rows, short (~85 tokens each) |
| `bench_fixture_long.jsonl` | `c9d7362c41807ab1023434b17fcc48d400dfa761acf51ba443a6c1b85a7b14cc` | 64 alpaca rows, ~482 output tokens each |
| `bench.yaml` | `6f7c3edbbf4436a6c5c0cfc156b98da2445669987aafbefe6d98fcc878a7d510` | shape A |
| `bench_b.yaml` | `7275d80c1e0d73c1be5fcdb6c39055579ef613abddf5b28653d49af37db79bfc` | shapes B |
| `bench_c.yaml` | `3528b1aee16348daa076595f7cee4db0e191967fbef6da83e9f1f5aeb2f5e678` | shapes C and D |
| `bench_frozen.yaml` | (section 9) | zero-trainable refusal |
| `bench_dpo.yaml` | (section 9) | out-of-scope-task refusal |

All seven files above, and all 25 JSON reports, are committed under
[`results/gate-836/`](results/gate-836/).

| shape | model | trainable | batch x tokens | steps/warmup |
|---|---|---|---:|---|
| **A** | `HuggingFaceTB/SmolLM2-135M-Instruct` | 921,600 / 135,436,608 (0.68%), 120 tensors | 4 x ~85 | 12/4 and 60/20 |
| **B** | `Qwen/Qwen2.5-0.5B-Instruct` | 1,081,344 / 495,114,112 (0.22%), 96 tensors | 8 x 512 | 12/4 |
| **C** | `Qwen/Qwen2.5-0.5B-Instruct` | as B | 2 x 512 | 12/4 |
| **D** | `Qwen/Qwen2.5-0.5B-Instruct` | as B | 2 x 512 | 200/40 |

LoRA is `r: 16, alpha: 32, dropout: 0.0` on `q_proj`/`v_proj` throughout.
`data.val_split` is the schema default, so the loader splits 64 rows into
57 train / 7 eval; `bench train` sets `eval_strategy = "no"`, so the eval rows
are loaded and never used.

## 3. The reference run — shape A, `--steps 12 --warmup 4`

The command, verbatim:

```
cd <scratch>
PYTHONPATH=C:/Users/user/projects/Soup-b836/src \
  python -m soup_cli.cli bench train --config bench.yaml --steps 12 --warmup 4 -o run1.json
```

and its last four lines, verbatim:

```
{'train_runtime': '21.45', 'train_samples_per_second': '2.238', 'train_steps_per_second': '0.559', 'train_loss': '4.004', 'epoch': '0.8'}
12 steps (4 warm-up discarded): median 0.3053s, p95 0.3654s, 1448 supervised of
2724 tokens, 594.7 supervised tok/s
Report: run1.json
Valid: the model trained while it was measured.
```

Exit code 0. Note the two throughput figures on adjacent lines and what
separates them: the HF Trainer's own `train_steps_per_second` of 0.559
(1.79 s/step) folds in dataloader construction and the whole first step;
`bench train`'s median of 0.3053 s does not. That gap — 5.9x — is the reason the
warm-up discard exists, and it is visible in the shipped output without
instrumenting anything.

The report (`results/gate-836/shapeA_run1.json`):

| field | value |
|---|---|
| `valid` / `failures` | `true` / `[]` |
| `checks.trainable_parameters` | 120 |
| `checks.grad_norm` | `"all reported steps finite and non-zero"` |
| `checks.parameters_changed` | `true` |
| `timing` | median 0.3053 s, p95 0.3654 s, total 2.435 s, counted 8, warm-up discarded 4 |
| `tokens` | useful 1448, total 2724, utilisation 0.5316 |
| `throughput` | 594.70 useful tok/s, 1118.75 total tok/s |
| `memory` | allocated 796,314,624 B (796.31 MB) / reserved 1,140,850,688 B (1140.85 MB) |
| `steps_requested` / `steps_measured` | 12 / 12 |
| `config_hash` | `268ca2462c1a7b852e4f77c8d98ed4fb89d89e481b202ded21035ab641326cbc` |

Shape A fits with room to spare: 796.31 MB allocated against 8547.04 MB of
VRAM. `provenance.sm_clock_mhz_after_run` read **285** — see section 10.3 for
why that number is not usable.

## 4. Thirteen repeats of a byte-identical config

`--steps 12 --warmup 4`, same config file, same fixture, same session, same
box. Runs 1–3 were unsampled; 7–10 ran with `nvidia-smi -lms 100` alongside;
e1–e6 ran with a process/CPU-frequency sampler alongside (section 7).

| run | median (s) | p95 (s) | total (s) | useful tok/s | sampler |
|---|---:|---:|---:|---:|---|
| run1 | 0.3053 | 0.3654 | 2.435 | 594.7 | none |
| run2 | 0.2252 | 0.3157 | 1.877 | 771.5 | none |
| run3 | 0.4378 | 0.4678 | 3.452 | 419.5 | none |
| run7 | 0.4346 | 0.4694 | 3.473 | 417.0 | nvidia-smi |
| run8 | 0.4355 | 0.4754 | 3.539 | 409.2 | nvidia-smi |
| run9 | 0.4746 | 0.5976 | 3.847 | 376.4 | nvidia-smi |
| run10 | 0.4289 | 0.4566 | 3.443 | 420.6 | nvidia-smi |
| e1 | 0.4706 | 0.5152 | 3.748 | 386.4 | process + CPU |
| e2 | 0.1972 | 0.2659 | 1.655 | 874.7 | process + CPU |
| e3 | 0.1982 | 0.2049 | 1.582 | 915.3 | process + CPU |
| e4 | 0.4330 | 0.4617 | 3.464 | 418.0 | process + CPU |
| e5 | 0.3431 | 0.3714 | 2.763 | 524.1 | process + CPU |
| e6 | 0.3559 | 0.3770 | 2.852 | 507.7 | process + CPU |

- median step time: **0.1972 – 0.4746 s**, ratio **2.41x**, difference **0.2774 s**, median-of-medians 0.4289 s
- useful tok/s: **376.4 – 915.3**, ratio **2.43x**, median 420.6, mean 541.2, stdev 190.8, **CV 35.3%**

The distribution is not clean bimodality — e5 and e6 sit between the clusters —
but the fast and slow ends are well separated and both are populated by
unsampled runs (run2 fast, run3 slow), so the samplers are not the cause.

**If this record published one number, it would be wrong by up to 2.4x.** That
is the finding.

## 5. Everything that is not timing is identical to the byte

Over the same thirteen runs:

| field | distinct values |
|---|---|
| `config_hash` | 1 (`268ca246…`) |
| `tokens.useful` / `tokens.total` | 1 (`1448` / `2724`) |
| `memory.max_memory_allocated_bytes` | 1 (`796314624`) |
| `memory.max_memory_reserved_bytes` | 1 (`1140850688`) |
| `checks` | 1 (`{trainable_parameters: 120, grad_norm: "all reported steps finite and non-zero", parameters_changed: true}`) |

So the deterministic half of the contract is genuinely deterministic on this
box, and a regression in tokens, memory or the training checks would be visible
at n=1. Only the timing half needs repeats.

One exception is worth keeping: `max_memory_reserved` is **not** invariant
across shapes with the same allocation. Shape C read 4,982,833,152 B in two
runs and 4,980,736,000 B in the third, and shape D — the same config at a
different `--steps` — read 5,073,010,688 B, while `max_memory_allocated` was
`4052621824` in all six. That is the caching allocator, and it is exactly why
the contract reports the two separately rather than picking one.

## 6. The spread is additive, not multiplicative

The same command over four shapes, fastest and slowest repeat of each:

| shape | steps/warmup | fastest median | slowest median | ratio | **difference** |
|---|---|---:|---:|---:|---:|
| A (n=13) | 12/4 | 0.1972 | 0.4746 | 2.41x | **0.2774 s** |
| A (n=3) | 60/20 | 0.1965 | 0.4401 | 2.24x | **0.2436 s** |
| B (n=3) | 12/4 | 5.2596 | 5.5039 | 1.046x | **0.2443 s** |
| C (n=3) | 12/4 | 0.2045 | 0.3750 | 1.83x | **0.1705 s** |
| D (n=3) | 200/40 | 0.2036 | 0.3854 | 1.89x | **0.1818 s** |

The ratio column spans 1.05x to 2.41x. The difference column spans 0.17 s to
0.28 s across shapes whose fast-mode step time differs by 26x. Read the
difference column, not the ratio column: **something adds about a fifth of a
second to a step, occasionally, and whether that looks catastrophic or
invisible depends entirely on how long the step already was.**

Two consequences, both practical:

- Shape B's 4.6% agreement is **not** evidence that the tool is precise. It is
  the same 0.24 s of noise divided by a 5.3 s step. A record that had only run
  shape B would have published "repeats agree to within 5%" and been wrong.
- Running longer does not fix it. Shape D counted 160 steps over a 33–63 s
  window and still spread 1.89x; shape A at 40 counted steps spread 1.93x. The
  additive cost is not a start-up transient that a longer window averages away.

The mechanism is **not established**. Two candidate explanations were tested
and both failed (section 7). What is established is the size.

## 7. Two hypotheses, each refuted by its own covariate

**Hypothesis 1 — another session on the box.** The dev box is shared; while
shape D's third run was in flight, `Win32_Process` showed a peer session's two
`pytest` processes starting 33 s and 41 s after it (23:06:25 and 23:06:33
against d3's 23:05:52). That run is the one genuinely contaminated reading in
this record and it is reported as such: **d3 median 0.2092 s (fast mode), p95
0.3726 s (slow mode), total 40.890 s**, throughput 3676.6 tok/s landing between
the two modes — a run that crossed from one mode to the other while it was
being measured, which is also the clearest case of the report's
median-and-p95 pair openly disagreeing about which mode the run was in. Good
behaviour by the tool.

But the hypothesis does not survive the e-series, which sampled the peer
process count for each run's whole lifetime (the loop asked for 250 ms;
`Get-CimInstance Win32_Process` is slow enough that 18–29 samples landed per
run, i.e. roughly one a second):

| run | useful tok/s | peer `python.exe` (median/max/samples) |
|---|---:|---|
| e1 | 386.4 (slow) | 0.0 / 2 / 29 |
| e2 | 874.7 (fast) | 0.0 / 1 / 24 |
| e3 | 915.3 (fast) | 0.5 / 1 / 18 |
| e4 | 418.0 (slow) | 0.0 / 1 / 26 |
| e5 | 524.1 (mid) | 0.0 / 1 / 27 |
| e6 | 507.7 (mid) | 0.0 / 0 / 23 |

The median was 0 in all six and the maximum never exceeded 2. The fastest run
and the slowest run had the same peer count. **Refuted.**

**Hypothesis 2 — CPU frequency scaling on battery.** Shape A's GPU utilisation
was sampled at 11–15% *maximum* over each run's whole lifetime (mean 0.8–1.5%,
mean while non-zero 4.0–5.7%), so shape A is a host-bound step and CPU
frequency was the obvious suspect. `\Processor Information(_Total)\% Processor
Performance` was sampled alongside:

| run | useful tok/s | % Processor Performance (median/min/max) |
|---|---:|---|
| e1 | 386.4 (slow) | 114 / 92 / 123 |
| e2 | 874.7 (fast) | 117 / 102 / 126 |
| e3 | 915.3 (fast) | 114 / 97 / 121 |
| e4 | 418.0 (slow) | 112 / 94 / 157 |
| e5 | 524.1 (mid) | 116 / 63 / 153 |
| e6 | 507.7 (mid) | **150** / 137 / 155 |

The slowest three and the fastest two sit within 5 points of each other, and
the run with by far the *highest* sustained CPU frequency (e6, 150%) is in the
slow half. **Refuted.**

Both are published as negative results. The GPU-side numbers that were
collected while testing them stand on their own and are worth keeping: during
shape A the card's SM clock *while busy* had a per-run median of 2370–2557 MHz
and peaked at 2610–2782 MHz, and power had a per-run median of 13.8–21.3 W
peaking at 28.3 W against a 50 W cap — so shape A was neither clock-starved nor
power-starved. It simply was not GPU work.

## 8. Shape B completed while allocating 1.51x the card's VRAM

Shape B (batch 8 x 512 on a 151,936-row vocabulary) is the one arm that
saturated the GPU: 100% utilisation whenever busy, 71–78% of samples busy,
2782 MHz median clock while busy (2797 peak), 32.7–33.3 W median and 47.7 W
peak against the 50 W cap.

| run | median (s) | p95 (s) | total (s) | useful tok/s | allocated | reserved |
|---|---:|---:|---:|---:|---:|---:|
| b1 | 5.3080 | 5.4830 | 39.131 | 695.2 | 12,879.61 MB | 16,944.99 MB |
| b2 | 5.5039 | 5.5611 | 39.973 | 680.6 | 12,879.61 MB | 16,944.99 MB |
| b3 | 5.2596 | 5.3270 | 38.268 | 710.9 | 12,879.61 MB | 16,944.99 MB |

**12,879.61 MB allocated on a card with 8547.04 MB.** All three runs completed,
exit 0, `valid: true`. This is the WDDM behaviour this folder's README already
warns about — Windows spills into shared host memory rather than raising
`CUDA out of memory` — and it lands on `bench train` in a specific way: the
`memory` fields come from `torch.cuda.max_memory_allocated` / `_reserved`, which
are the allocator's counters and are **correct**; they are simply not a
statement that the configuration fit in VRAM. On this platform, a `memory`
figure above the card's capacity is the signal that it did not.

The cost of not fitting is measurable against shape C, the same model and
sequence length at batch 2 instead of 8:

| | shape B (batch 8) | shape C (batch 2) |
|---|---:|---:|
| allocated | 12,879.61 MB (1.51x VRAM) | 4052.62 MB (0.47x VRAM) |
| median step | 5.2596–5.5039 s | 0.2045–0.3750 s |
| useful tok/s | 680.6–710.9 | 2553.5–4666.6 |

The larger batch is **3.8x to 6.6x slower per supervised token**, depending on
which repeat of each you pair — which is section 6's noise showing up again.
"It completed" is not "it fit", and the report says which one happened — if you
read `memory` against the card rather than on its own.

## 9. Four refusals, exercised for real

All four verbatim, all exit code 1.

**(a) Nothing left to measure.**

```
$ python -m soup_cli.cli bench train --config bench.yaml --steps 2 --warmup 2 -o refusal_a.json
--steps 2 leaves nothing after --warmup 2: raise --steps or lower --warmup.
EXIT=1
```
No model was loaded and **no report file was written** (`ls refusal_a.json` →
no such file).

**(b) Zero trainable parameters.** `bench_frozen.yaml` drops LoRA and sets
`training.unfrozen_parameters: ["^no_module_is_named_this$"]`, a Spectrum
pattern that matches nothing:

```
$ python -m soup_cli.cli bench train --config bench_frozen.yaml --steps 6 --warmup 2 -o refusal_b.json
apply_unfrozen_parameters: no parameter matched any of 1 pattern(s) — the model has NO trainable parameters. Check your training.unfrozen_parameters against the model's parameter names (run `soup spectrum scan` to regenerate).
Spectrum targeted FT: 0 parameter tensor(s) unfrozen (LoRA off)
Spectrum targeted FT: 0 trainable / 134,515,008 total (0.00%)
...
0 trainable parameter tensors with real storage.
EXIT=1
```
The model and dataset were loaded, the trainer was built, and the pre-flight
`check_trainable` fired before `trainer.train()` — so the Trainer never got to
die inside autograd, which is the point of checking early. Again **no report
file was written**.

**(c) Out of scope by task.** Instant, before any load:

```
$ python -m soup_cli.cli bench train --config bench_dpo.yaml --steps 12 --warmup 4 -o refusal_c.json
Cannot bench this config: task: dpo -- bench train measures the SFT trainer only
EXIT=1
```

**(d) Missing config.**

```
$ python -m soup_cli.cli bench train --config nope.yaml -o refusal_d.json
Config not found: nope.yaml
EXIT=1
```

Each names the actual reason. None guessed.

Two further promises were checked and held: the config's own `output:`
directory (`./output_bench836/`) was **never created** by any of the 25 runs,
and the `soup-bench-*` scratch directories under `%TEMP%` were **all cleaned
up** — zero left behind after the session.

## 10. Four observations on the contract itself. None is a defect.

**10.1 `config_hash` covers the tool's own overrides, so it changes with
`--steps`.** Shape C at `--steps 12` hashes `7ec538f1f605…`; the identical file
at `--steps 200` hashes `c2cd23cc9807…`. Diffing the two `resolved_config`
blocks gives exactly one difference: `training.save_steps` 13 vs 61 — the
`steps + 1` that `run_bench_train` sets so nothing is saved. That is arguably
the correct behaviour (the hash describes the run that happened, not the file),
but a reader comparing two reports of the same `soup.yaml` at different step
counts will see different hashes and should not conclude the config drifted.
Same for shape A: `268ca246…` at 12 steps, `70e143f2…` at 60.

**10.2 "the report is still written" applies to the post-run checks, not to the
two pre-flight refusals.** `docs/serving-and-export.md` says the command "exits
1 when any check fails. The report is still written, with the failures in it",
and then lists `trainable_parameters` as one of those checks. Measured: when
`trainable_parameters` fires — which in ordinary use is always the pre-flight
copy, since the Trainer would otherwise die in autograd — it raises before any
report exists and **nothing is written** (section 9b). The sentence and the
table are each true about different code paths; a reader who plans to parse the
report after a failure should know which.

**10.3 `provenance.sm_clock_mhz_after_run` is an idle reading on this card and
cannot serve as the covariate it was added to be.** Its own docstring calls it
"the clock the measured steps ended at". Measured against a 100 ms sampler: the
card's busy-sample clock median was 2370–2557 MHz *during* the steps, and the
field recorded, over all 25 runs in this record, **180 MHz fourteen times,
1417 four times, 225 three times, and 172 / 232 / 277 / 285 once each**. The
1417 readings are a catch mid-ramp-down; the rest are idle. Decisively, shape
C's three runs — 4645.6, 4666.6 and 2553.5 tok/s — recorded 225, 232 and 225 MHz,
so the field does not discriminate a 1.83x difference in the thing it is meant
to explain. Sampling during the counted window instead of after would fix it;
that is a follow-up, not a release blocker, and the field is honest about being
a single post-run reading.

**10.4 The report carries no per-step distribution.** `timing` is
`median / p95 / counted_steps / warmup_steps_discarded / total_seconds`. That is
enough to *notice* a run that changed mode mid-flight — d3's median 0.2092 s
against its p95 0.3726 s and its total of 40.890 s over 160 steps (mean
0.2556 s) is exactly that signature — but not enough to localise it. Given
section 6, a future revision that emitted the per-step times (or a couple more
quantiles) would make this class of noise diagnosable from the report alone.

## 11. What these numbers do NOT mean

- **Not a card-capability figure.** RTX 5070 Laptop on **battery**, GPU capped
  at 50 W of 115 W. Mains would be a different baseline.
- **Not a cross-model claim.** Two models, 135M and 0.5B, both LoRA, both
  `quantization: none`. Nothing here transfers to a 7B, to NF4, or to the
  streaming path — which this release does not touch at all.
- **Not a cross-shape claim.** Four shapes, each at one sequence length and one
  batch size. Section 8 shows how far two shapes of the *same model* diverge.
- **Not a quality claim.** `parameters_changed: true` proves something moved. It
  is a floor, not a gate. Shape D's loss fell to `0.07103` over 200 steps on 57
  rows, which is memorisation of a synthetic fixture and means nothing.
- **Not a comparison baseline at n=1.** Any future use of `soup bench train` to
  show a change made training faster needs repeats and needs to state its
  spread. On this box a change that saves less than ~0.2 s per step cannot be
  resolved by single runs.
- **Not comparable to any `gate-v0.72.*` record**, all of which were taken on
  an RTX 3050 Laptop 4 GB.

## 12. Reproducing

No harness script is added: the shipped command is the harness, and the two
fixtures are generated deterministically and committed.

```bash
pip install "soup-cli[train]"          # or an editable checkout at >= the #1146 merge
# fixtures + configs: benchmarks/results/gate-836/
soup bench train --config bench.yaml   --steps 12  --warmup 4  -o run1.json    # shape A
soup bench train --config bench_b.yaml --steps 12  --warmup 4  -o b1.json      # shape B
soup bench train --config bench_c.yaml --steps 12  --warmup 4  -o c1.json      # shape C
soup bench train --config bench_c.yaml --steps 200 --warmup 40 -o d1.json      # shape D
```

Repeat each at least three times and read the spread before reading the number.
The refusals in section 9 reproduce in seconds and need no GPU for (c) and (d).

All 25 JSON reports from this session are committed verbatim under
[`results/gate-836/`](results/gate-836/), including the contaminated `d3.json`.

## Addendum, 2026-09-23 — the instrument changed after this record (#1166)

Everything above is unchanged and describes the command as it was measured. Three
of the section 10 observations have since been acted on:

- **10.3:** `provenance.sm_clock_mhz_after_run` is gone. It is replaced by
  `provenance.sm_clock_mhz_busy`, which gives `min` / `median` / `max` /
  `sample_count` over `nvidia-smi` samples kept only from the counted
  (post-warm-up) steps, plus `unavailable_reason` when there are no numbers.
  Queries start on a 100 ms schedule, and a slow query skips ticks. Each
  sample is timestamped at the midpoint of its query. The
  `sm_clock_mhz_after_run` values in this record (and in the
  committed `results/gate-836/` JSON) are post-run idle reads. Do not compare them
  with a later report's `sm_clock_mhz_busy`.
- **10.4:** `timing.step_seconds` now lists every counted step in run order.
- **10.2:** `docs/serving-and-export.md` now separates pre-flight refusals, which
  write no report, from post-run check failures, which write a `valid: false`
  report. No code changed there, because the behaviour measured in section 9 was
  already correct.

Timing and token counting are unchanged, but the sampler is new load: while the
timed steps run, a thread spawns `nvidia-smi` about every 100 ms. Whether any
throughput figure above moves is therefore open, not settled. **Follow-up:** re-run
section 12's commands on the section 1 box twice, once with the sampler and once
without it (for example with a no-op `clock_sampler`). Compare the two against
this record, and record busy-window clocks for shapes A–D. No GPU was available
when this change was made.
