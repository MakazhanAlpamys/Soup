# Measured cache placement for short streamed SFT jobs

**Verdict (2026-11-04): positive.** Measured placement reduced end-to-end job time in the two tested settings. The dense 32B job improved by avoiding cache setup, and a later synthetic run improved by retaining activations.

Measured by @kok-o.

## Question and setup

Can profiling select a placement that minimizes setup plus training time within the available memory?

- Dense cohort, November 4. Qwen3-32B NF4 with BF16 compute and rank-4 LoRA. RTX 4060 Ti with 8,188 MiB VRAM, host RAM limit 31,950,110,720 bytes, no swap. Fixed caches use a 14 GiB RAM ceiling and seven resident decoder layers. The paired jobs use the same checkpoint, fixed text, 512-token microbatches, accumulation of two, deterministic settings and dropout disabled. The source report does not provide an OS release, driver, CUDA or package inventory for this cohort.
- Synthetic cohort, November 7. Llama with 329,303,040 parameters and 24 layers, NF4 with FP16 compute and FP32 AdamW states. Physical Tesla T4 16 GB with an 8 GiB allocator cap and a 32,212,254,720-byte RAM limit. Python 3.13, CUDA 13.0, torch 2.11.0+cu130, Transformers 5.20.2, PEFT 0.21.2, Accelerate 1.15.0 and bitsandbytes 0.50.2. The fixed arm uses eight resident layers and a 96 MiB RAM-cache ceiling, fitting nine host-cached layers. The selected arm also retains activations for all 24 layers.
- `training.stream_cache_profile` with `training.stream_cache_profile_mode`, `training.stream_resident_layers` and `training.stream_ram_cache_mb` was pending in the CLI at the publication base.

## Method and results

Dense timing used four balanced fresh-process pairs, each with two warmup and 16 timed updates. The profile chose ordinary disk streaming. Before timing, four eight-update trajectories yielded 179,904 exact comparisons covering loss, logits, gradients, LoRA parameters and AdamW state. The synthetic follow-up used two balanced fresh-process pairs per job length with 26,912 exact comparisons in the four-update correctness gate. Both timing windows exclude profile recording and checkpoint preparation.

| Dense 32B median | Fixed caches | Profile selection |
|---|---:|---:|
| Whole 18-update job | 188.30 s | 180.16 s |
| Setup | 30.17 s | 11.83 s |
| Optimizer update | 8.59 s | 9.12 s |
| Tokens/s | 119.24 | 112.32 |
| Peak CUDA allocated | 5.68 GiB | 4.04 GiB |
| Peak CUDA reserved | 5.95 GiB | 4.32 GiB |
| Peak process RSS | 19.50 GiB | 5.42 GiB |

The paired whole-job reduction was 4.47%, with a 95% bootstrap interval of 4.30–4.78%. The selected arm took 6.17% longer per update.

| Synthetic median | Fixed caches | Profile selection |
|---|---:|---:|
| Eight updates with setup | 12.328 s | 9.878 s |
| 64 updates with setup | 78.380 s | 59.302 s |
| Step after first two, 64 updates | 1.180 s | 0.889 s |
| Peak CUDA allocated | 219.5 MiB | 576.9 MiB |
| Peak process RSS | 1.535 GiB | 1.538 GiB |

## Reproduction

The synthetic runner, probe and exact source patch live in the frozen source packet at the evidence commit. The dense report identifies the measured revision as `0e585bdca3b37887f3f84adec846b1282e973379` but does not preserve its original launcher alongside the report.

```text
runner.py
probe.py
```

## Limits and superseded runs

The synthetic pilot stopped before timings because its shard output was outside the allowed roots. The completed run used a valid output root. Reuse accounting now subtracts 1.754 s of recording-only planning work for the measured combined synthetic configuration. The correction leaves both measured selections intact. Its changed prediction at three updates is unmeasured.

Settings, models and precision differ across cohorts. The results cover short controlled jobs. End-to-end returns on longer dense jobs, training quality and generalization to other hardware remain open.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/96df4325d422397384d21bb21d492843c67fb84e/benchmarks
