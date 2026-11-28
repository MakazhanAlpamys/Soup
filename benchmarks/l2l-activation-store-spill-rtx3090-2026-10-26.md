# Gate record - RAM-bounded L2L activation storage and drive placement

**Verdict (2026-10-26) - Mixed.** The store passed equality, RAM accounting, cleanup and RAM-tier timing checks. On the shared virtual disk, weight-read throughput failed the placement threshold.

Measured by @Batyrkhantzy111.

## 1. Setup

- RTX 3090 24 GB in a Linux KVM guest, 49.3 GiB host RAM, driver 580.95.05, CUDA 13.0 torch build, one virtio disk for weights and spill. The session ran in a cgroup limited to 24 GiB with swap disabled. The archived record identifies a CUDA 12.9.1 guest image.
- torch 2.14.1+cu130, transformers 5.19.0, peft 0.21.2, bitsandbytes 0.50.2.
- `unsloth/mistral-7b-instruct-v0.3` revision `60b99dcbc7f853b7cd18472a607e8d045b96f03e`, NF4 with BF16 compute. The harness exercised the activation store and compared pinned and direct-I/O activations, with RAM-tier and disk-tier weight placement.
- The product fields for this work are `training.l2l` and `training.l2l_spill_dir`, both pending. This run used the store harness.

## 2. Method and results

The planned store bytes include pinned chunks, rounding and I/O staging rings. The amended rule compares memory outside the activation store with the no-spill arms in the same block. The harness checks losses and nonzero LoRA gradients against both pinned L2L and gradient accumulation (GA), with a corrupted-read control. Timing uses fresh A-B-B-A processes, a 95% interval and an upper-bound threshold below 10% for RAM-tier step overhead. Same-drive reads require the lower bound to reach 0.90 of the no-spill read rate.

| Check | Result |
|---|---|
| Deterministic equality at 2, 3 and 4 microbatches, 20–32 spilled layers | Forward losses and backward gradients bit-equal to both references. The corrupted-read control differed |
| Store RAM accounting | 140 arms met their plan and budget. No more excess than the no-spill arms, which held 4 B of excess in each block |
| Cleanup | Ctrl-C during backward, exception during forward, and injected and real ENOSPC passed. The real test used a 64 MiB ext4 loop volume |
| RAM-tier overhead, 2 microbatches, half budget | +2.51%, 95% interval [-1.04%, +6.07%], 5 blocks |
| RAM-tier overhead, 4 microbatches, half budget | +1.69%, 95% interval [-2.13%, +5.50%], 5 blocks |
| Same-drive weight-read rate, 4 microbatches | 0.830x, 95% interval [0.822, 0.837], 5 blocks |

## 3. Commands

The harness lives at the evidence commit. Set the spill and weight-drive directories and prepare the pinned model and NF4 cache using the study runbook. The measured child commands and arguments are in the archived protocol.

```bash
A0_PYTHON=python3.12 bash benchmarks/a0-bootstrap.sh
.venv/bin/python benchmarks/harness/a4_admission.py --spill-dir "$SPILL" \
  --out .soup/admission.json --log .soup/admission-per-second.csv \
  --cycles .soup/admission-cycles.csv
systemd-run --user --scope -p MemoryMax=24G -p MemorySwapMax=0 -- \
  .venv/bin/python benchmarks/harness/a4_session.py \
  --weights .soup/models/mistral-7b --shards .soup/f1-nf4 --spill-dir "$SPILL" \
  --weights-spill-dir "$WSPILL" --enospc-dir /mnt/a4-small \
  --out .soup/results/a4-01 --execute
.venv/bin/python benchmarks/harness/a4_rule.py .soup/results/a4-01 --out verdict.json
```

## 4. Limits and earlier runs

The first scoring rule failed RAM on 280 arms because it charged a 4 B allocation that both controls and spill shared. The amended rule applied prospectively. A later sustained-write admission screen refused three machines before scoring, so the final screen exercised the run's actual I/O pattern. The earlier verdicts stand.

The same-drive result favors separating weight reads and spill writes for this setup. Separate-drive throughput, whole-trainer behavior, hard-kill recovery and model quality were outside this experiment.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/96df4325d422397384d21bb21d492843c67fb84e/benchmarks/results/a4-v3-rtx3090-20261026
