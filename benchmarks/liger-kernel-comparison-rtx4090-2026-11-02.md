# Liger cross-entropy versus Transformers on a Linux RTX 4090

## Question

Does replacing standard Transformers cross-entropy with Liger cross-entropy
change useful training throughput or peak GPU memory under the same QLoRA
recipe?

## Environment

The timing series ran on 2026-11-02 on an exclusive NVIDIA GeForce RTX 4090
(24 GB class, 24,564 MiB reported), driver 580.173.02, CUDA runtime 13.0.
The training process reported Linux 5.15.0-191-generic x86_64, glibc 2.39 and
Python 3.11.16. The first counted run recorded 64,476,618,752 bytes of available
RAM before launch. The timing packet does not identify the CPU model or report
installed RAM capacity. The runtime used torch 2.14.1, Transformers 5.17.0,
TRL 0.29.1, PEFT 0.21.2, bitsandbytes 0.50.2, Accelerate 1.15.0 and
liger-kernel 0.8.5.

Both arms used `unsloth/Meta-Llama-3.1-8B` at
`e9a141a2091ea561b96483212645a2a05e6f99fc`, an NF4 base and BF16 autocast,
with gradient checkpointing. Both used LoRA rank 16, alpha 32 and dropout 0
on the seven attention and MLP projections, AdamW at `2e-4`, batch size 1
and gradient accumulation 4. Model and data seeds were 42. Only the
cross-entropy kernel was substituted in the Liger arm.

## Harness and method

The timing harness starts a fresh training process per run with the frozen
`perf_baseline.json` or `perf_liger_ce.json` configuration and
`perf_workload.jsonl`. The workload has 96 synthetic rows, with 404–956
tokens per formatted row and the same ordered input IDs and target masks
in both arms. A separate update check ran 16 microbatches and 4 optimizer
updates per arm. The corrected audit found that 448/448 adapter tensors changed
in each arm while frozen model and quantization state were preserved. Kernel
tracing and patch bindings attributed cross-entropy to Liger only in that arm.

Each timed run made 20 optimizer updates, discarded the first 4 as warm-up,
and counted 32,105 useful targets out of 44,084 tokens over the remaining 16
updates. Useful throughput divides the useful targets by the sum of
CUDA-synchronized step times. Each block ran baseline, Liger, Liger, baseline.
The predeclared repeat rule required 5 blocks, so the 5 pilot blocks were the
full series. All 20 processes passed the frozen per-step token counts, update
checks and GPU exclusivity. The suspend, mains power and foreign-read checks
passed in every run. Peak allocated and reserved bytes are the medians of the
per-run CUDA memory peaks. The throughput test uses the paired block differences
with a two-sided t interval.

The recorded series launcher used the invocation below with the measured
checkout and local model path. The second command recomputes the verdict from
the published series.

```sh
python -B benchmarks/harness/h2_perf_abba.py \
  series "$OUTPUT" A,L "$MODEL" "$DEADLINE_UNIX_S" "$BLOCK_ESTIMATE_S" "$LABEL"
python benchmarks/harness/h2_perf_abba.py review \
  benchmarks/h2/reports/20261102_h2_close/evidence/stage_p_series.json
```

## Results

| Metric | Transformers baseline | Liger cross-entropy only |
|---|---:|---:|
| Mean useful tokens/s | 934.4481187304984 | 946.742080090599 |
| Median peak allocated bytes | 11,203,267,584 | 10,956,698,112 |
| Median peak reserved bytes | 16,819,159,040 | 11,226,054,656 |

| Paired throughput metric | Observed value |
|---|---:|
| Valid blocks | 5 |
| Void blocks | 0 |
| Mean difference (Liger minus baseline), useful tokens/s | 12.293961360100548 |
| 95% t interval for the difference | [-2.1189974334504633, 26.70692015365156] |
| Liger/baseline ratio | 1.0131563872982083 |
| 95% interval for the ratio | [0.9977323541125759, 1.0285804204838407] |
| p-value | 0.07696742343738142 |

**Verdict: NEGATIVE.** No detectable speed difference on this GPU and recipe.

## Limitations

The interval crosses zero for the difference and leaves small effects
unresolved. Liger reserved less CUDA memory in this series. The study held
batch size fixed, so performance at a larger batch is unknown. One setup
attempt failed before any timing was collected. Unsloth 2026.10.7 loaded the
model but its trainer required making the special-token embedding and output
head trainable, so it supplied no timing under the fixed recipe. These results
cover one GPU, one configuration and one substituted operator on synthetic
data. Model quality was outside the experiment.

Measured by @nurseit14

Evidence: https://github.com/SoupTeam/soup-rnd/tree/5ff09d88a017b6d771928e7009ce4b961b124dc3/benchmarks/h2/reports/20261102_h2_close
