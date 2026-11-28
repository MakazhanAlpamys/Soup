# Qwen2.5-14B streaming capacity on physical 8 GiB GPUs

**Verdict (2026-10-20): positive.** Real 14B LoRA training completed with pinned NF4 weights and with automatic BF16 overflow to NVMe. The same BF16 RAM-only recipe refused at admission.

Measured by @kok-o.

## Question and setup

Can Qwen2.5-14B LoRA update and save an adapter on a physical 8 GiB card, and will streaming select NVMe or refuse under a real host-memory constraint?

- Forced-tier cohort, October 19–20. Linux 6.8.0-147-generic, RTX 3060 Ti with 8,589,934,592 physical VRAM bytes, driver 610.57.04, 33,364,897,792 bytes of host RAM and a 22,648,193,024-byte container limit. The cache mapped to an SK hynix HFS256GD9TNG-62A0A NVMe device.
- Capacity cohort, October 20. Linux 6.8.0-138-generic, RTX 5060 Ti with 8151 MiB physical VRAM, driver 610.57.04 and a Samsung MZVL21T0HCLR-00BH1 NVMe device. The host had about 101.15 GB RAM. The container's protected baseline was 32,028,753,920 B and its burst ceiling was 56,060,018,688 B. The baseline was memory protection and the planner's conservative budget, while the burst ceiling was the hard limit.
- The measurement preflights used Python 3.11.13, torch 2.8.0+cu128, Transformers 5.17.0, TRL 0.29.1, PEFT 0.20.0 and bitsandbytes 0.50.2. The capacity preflight also records Soup CLI 0.75.0, Accelerate 1.15.0, datasets 5.1.0 and psutil 7.0.0.
- Both used `Qwen/Qwen2.5-14B-Instruct` at `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`, batch and accumulation 1, LoRA r=16/alpha=16 on q/v, dropout 0, AdamW at 2e-5 and weight decay 0.01. The same checkpoint was stored as NF4 for the RAM baseline and as BF16 for the capacity test. The forced-tier cohort used NF4 for both working tiers.

## Method and results

Each working launch ran 12 steps with four excluded as warmup. Ordinary `soup train` checked that the adapter updated and saved. Rates are medians of valid native `soup bench train` reports. The harness verified the physical cache mapping with full-file direct reads and calibrated its foreign-read detector before timing. The forced-tier campaign used an explicit HDD policy override as its refusal control. The capacity follow-up let `training.stream_source: auto` choose the verified NVMe volume and tested `training.stream_source: ram` with the same BF16 store. The capacity follow-up used cgroup-aware budgeting for `training.stream_source`, pending in the CLI at the publication base.

| GPU and condition | Token cap | Train admitted | Bench admitted | Step s | Useful tok/s | Total tok/s |
|---|---:|---:|---:|---:|---:|---:|
| RTX 3060 Ti, RAM NF4 | 128 | 10/10 | 10/10 | 1.149 | 79.16 | 109.24 |
| RTX 3060 Ti, RAM NF4 | 384 | 10/10 | 10/10 | 1.542 | 221.27 | 243.64 |
| RTX 3060 Ti, RAM NF4 | 1024 | 10/10 | 10/10 | 3.211 | 304.21 | 315.06 |
| RTX 3060 Ti, forced NVMe NF4 | 128 | 9/10 | 10/10 | 17.527 | 5.10 | 7.04 |
| RTX 3060 Ti, forced NVMe NF4 | 384 | 10/10 | 10/10 | 17.634 | 19.48 | 21.45 |
| RTX 3060 Ti, forced NVMe NF4 | 1024 | 9/10 | 10/10 | 18.459 | 52.08 | 53.94 |
| RTX 5060 Ti, RAM NF4 | 128 | 12/12 | 12/12 | 1.186 | 78.17 | 107.88 |
| RTX 5060 Ti, RAM NF4 | 384 | 12/12 | 12/12 | 1.287 | 270.76 | 298.13 |
| RTX 5060 Ti, RAM NF4 | 1024 | 12/12 | 12/12 | 2.356 | 419.65 | 434.61 |
| RTX 5060 Ti, auto NVMe BF16 | 128 | 12/12 | 10/12 | 8.628 | 10.75 | 14.83 |
| RTX 5060 Ti, auto NVMe BF16 | 384 | 12/12 | 12/12 | 8.646 | 40.32 | 44.40 |
| RTX 5060 Ti, auto NVMe BF16 | 1024 | 12/12 | 12/12 | 8.706 | 113.60 | 117.65 |

| Refusal control | Token cap | Train refusals | Bench refusals |
|---|---:|---:|---:|
| RTX 3060 Ti, HDD policy override | 128 | 5/5 | 5/5 |
| RTX 3060 Ti, HDD policy override | 384 | 5/5 | 5/5 |
| RTX 3060 Ti, HDD policy override | 1024 | 5/5 | 5/5 |
| RTX 5060 Ti, RAM-only BF16 | 128 | 6/6 | 5/6 |
| RTX 5060 Ti, RAM-only BF16 | 384 | 6/6 | 6/6 |
| RTX 5060 Ti, RAM-only BF16 | 1024 | 6/6 | 6/6 |

The forced-tier cohort logged 118 passes, 2 exclusions and 30 expected policy refusals in 150 commands. The capacity cohort logged 142 passes, 35 expected capacity refusals and 3 foreign-read exclusions in 180 commands. The six saved adapters from the capacity cohort each held 192 finite tensors with 96 nonzero LoRA-B tensors. The 29,540,125,622 B BF16 cache triggered automatic fallback and RAM-only refusal under the measured admission policy.

## Reproduction

The following launchers and the frozen recipe live at the evidence commit. The archives include the command receipts, package inventories, native reports, sampled traces and adapter fingerprints. Each campaign needs a fresh output directory and an on-host calibration.

```text
benchmarks/harness/f2_reference.py
benchmarks/harness/f2_capacity_reference.py
benchmarks/harness/f2_calibrate.py
benchmarks/harness/f2_capacity_audit.py
benchmarks/harness/f2_capacity_verify_adapters.py
```

## Limits and superseded runs

The earlier forced-tier result remains valid for its own configuration. Its initial partial or incorrectly mapped cohorts and its superseded process sampler do not enter the table. Its BF16 overflow preparation failed the mapping criterion before any auxiliary CLI measurements. The later capacity cohort superseded an interrupted run with false RAM refusals after the memory probe was corrected and the inputs refrozen.

The capacity table uses different formats on RAM and NVMe. It characterizes fallback and refusal and leaves the causal tier-speed comparison open. Both campaigns used short fixed text fixtures. Model quality, numerical exactness and behavior at physical host-memory exhaustion remain unmeasured.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/96df4325d422397384d21bb21d492843c67fb84e/benchmarks/results
