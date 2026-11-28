# Gate record - L2L readiness for streamed Mistral SFT

**Verdict (2026-10-29) - Positive with limits.** The RTX 5070 Ti campaign passed the loss and allocated-VRAM checks, and the mixed-length control passed on its own two cards. The RTX 5070 Ti timing interval included zero.

Measured by @M1kado0.

## 1. Setup

- RTX 5070 Ti 16 GB, AMD EPYC 7V13, 540792193024 bytes of RAM reported by `free -b`, local XFS NVMe volume, Ubuntu 24.04 CUDA 13.0 development container, driver 595.104.02, driver-reported CUDA 13.2. The comparison used that same host and volume for both stages.
- torch 2.14.1+cu130, transformers 5.19.0, peft 0.21.2, bitsandbytes 0.50.2, accelerate 1.15.0, trl 0.29.1.
- `mistralai/Mistral-7B-v0.1`, NF4 with BF16 compute, LoRA rank 8, batch 1, four microbatches, sequence cap 256, RAM-tier weights. Both arms used the CLI SFT training path. The pending `training.l2l` field selected the candidate. That implementation refused `training.stream_recompute`; the campaign contains no combined-mode measurement.
- The mixed-length check used RTX 3070 8 GB (driver 610.43.02, cgroup cap 80060874752 bytes) and RTX 3090 24 GB (driver 615.71.09, cgroup cap 105858990080 bytes), with the same recorded torch, transformers, peft and bitsandbytes versions.

## 2. Method and results

Readiness required loss within the gradient-accumulation (GA) tolerance at every step, a passing mixed-length check, and allocated VRAM at or below GA. The campaign used 64 Alpaca rows, fresh A-B-B-A processes, deterministic MATH SDPA, and a second SDPA backend to set the BF16 tolerance. Timing excluded step zero. The broader rule was written after earlier speed and loss data were available. The local-volume rerun rules and the larger repeat cap were written before those reruns.

| Measure | Ordinary streamed GA | L2L |
|---|---:|---:|
| Throughput, tok/s | 197.2 | 200.7 |
| Peak allocated VRAM, GB | 0.916 | 0.891 |
| Peak process RSS, GB | 6.79 | 7.10 |
| Layer loads per step | 244 | 64.3 |
| Recorded step read counter | 0 | 0 |

| Readiness or validity check | Result |
|---|---|
| Loss | 22 L2L curves across 11 valid blocks, 16 steps each, all within tolerance |
| Mixed lengths without padding | Four passing records, three seeds on RTX 3070 and one on RTX 3090 |
| Local-volume run validity | 52 arms, 50 OK, 2 void, 0 unknown. Two whole blocks excluded for unrelated reads of 7,095,701,504 and 4,689,460,736 bytes |
| Repeats and throughput difference | 11 valid blocks, 9 required, none left to add. +3.6 tok/s L2L minus GA, 95% interval [-11.6, +18.7] |
| Earlier CPU baseline on macOS | 281 passed, 15 GPU cases skipped |

## 3. Commands

The campaign launcher, configurations, comparator and per-run commands live at the evidence commit. The remote launcher expects the model cache and working data on the mounted local volume. The decision command reads saved results on CPU.

```bash
CARD=5070ti-16gb MAXBLOCKS04=16 SKIP_T11=1 bash benchmarks/harness/l2l_ticket06_valid/remote.sh
python benchmarks/harness/l2l_ticket06_decision.py
benchmarks/harness/l2l_fwdbwd.py
PYTHONPATH=$PWD/src pytest tests/test_l2l_*.py
```

## 4. Limits and earlier runs

The final campaign replaced the seven-block local-volume pilot for readiness. It also superseded the earlier readiness conclusion drawn from hosts with unknown foreign-read telemetry. Those results retain no verdict. The mixed-length records used here replaced an earlier run whose reference backward ran inside autocast. That correctness check was conducted on the RTX 3070 and RTX 3090.

The loss gate qualifies this one real model on one measured timing host. Cold weight reads and trainer spill were outside this campaign. The RAM over-limit probe refused before allocation. This is adapter training on a frozen NF4 base. The study did not measure downstream quality or larger real models.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/96df4325d422397384d21bb21d492843c67fb84e/benchmarks
