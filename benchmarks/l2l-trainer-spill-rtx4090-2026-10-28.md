# Gate record - automatic activation spill in the L2L trainer

**Verdict (2026-10-28) - Positive for spill.** Spill matched no spill bit for bit, stayed inside the cgroup cap, met cleanup requirements, and cleared the step-overhead threshold. The separate comparison with ordinary gradient accumulation failed its original tolerance.

Measured by @Batyrkhantzy111.

## 1. Setup

- RTX 4090 24 GB, PCIe Gen 4 x16, driver 580.95.05, 49.35 GiB RAM, Linux KVM kernel 6.8.0-59-generic, systemd 249, cgroup v2 with swap disabled, local ext4 on one virtio disk. An earlier RTX 4090 run used a different guest with 47.4 GiB RAM and ext4.
- Python 3.12.16, torch 2.14.1+cu130, transformers 5.19.0, peft 0.21.2, bitsandbytes 0.50.2 and accelerate 1.15.0.
- `unsloth/mistral-7b-instruct-v0.3` at `60b99dcbc7f853b7cd18472a607e8d045b96f03e`, RAM-tier NF4 with BF16 compute, LoRA rank 8 on q/k/v/o, four microbatches, sequence 512. `training.l2l` and `training.l2l_spill_dir` are pending.

## 2. Method and results

The trainer selected the activation layout under a cgroup limit. Two generous no-spill runs checked repeatability. The tight run used a limit calibrated after introducing a 512 MiB reserve. Deterministic SDPA MATH arms compared every LoRA gradient hash, logged losses, learning rates and optimizer step counts. Timing used the median of five timed steps after two warm-ups, five valid A-B-B-A blocks, and a 95% interval whose upper bound had to be below 10%.

| Check | Final measurement |
|---|---|
| Equality over 3 optimizer steps | 13 of 32 layers spilled. The spill and both no-spill runs matched |
| Cgroup RAM | Spill peaked at 6,154,133,504 B under 6,409,945,088 B. No-spill peaks 6,354,161,664 and 6,374,264,832 B under 24 GiB. No OOM kill, allocator excess 36,877 B in all three |
| Normal end, SIGINT and SIGKILL | Normal end removed the file. SIGINT in the backward-recompute hold exited 130 and removed the file. SIGKILL left the file for the next run, which removed it and trained 3 steps |
| Step overhead | +2.39%, 95% interval [-1.80%, +6.57%], with 12–18 layers spilled in timing arms |
| Separate BF16 comparison with ordinary gradient accumulation (GA) | Gradient relative errors 0.0181, 0.0213, 0.1772 versus limits 0.0165, 0.0206, 0.0767. Step-0 loss error 0.000222 versus 0.0000103 |

### Diagnostic follow-up (2026-10-29)

On the earlier run's RTX 4090 hardware, PCIe Gen 4 x16 and driver 580.95.05, the follow-up reproduced the GA failure with the same software stack. Removing the native GA mask preserved losses and all 256 gradients. Padding the resulting matched-mask GA inputs gave the same first loss as L2L. Matching the inputs still left a backward discrepancy. The diagnostic disabled ambient CUDA autocast at the backward call, preserving the original forward and leaving the gradient accumulation schedule in place.

| Diagnostic comparison | Result |
|---|---|
| First loss, native GA versus original L2L | 12.834332466125488 versus 12.831486701965332 |
| Padded GA versus original L2L | 144 forward tensors matched, gradient relative L2 0.007499608746411462 |
| Padded GA versus L2L with aligned backward entry | All 256 gradients matched in the initial comparison, with exact gradients across the 3-step confirmation |
| Confirmation loss at step 1 | 12.808198928833008 versus 12.808199882507324, within the comparator's matched-input tolerance |

## 3. Commands

The harness scripts live at the evidence commit. The spill protocol records every child argv with sanitized path placeholders. The follow-up command repeats the original diagnostic phase. The adaptive arms require the wrapper behavior recorded with the follow-up.

```bash
benchmarks/harness/a3_signal_preflight.py
benchmarks/harness/a3_admission.py
benchmarks/harness/a3_spill_stand.py
python benchmarks/harness/a3_spill_rule.py <out> --out verdict.json
```

```bash
python benchmarks/harness/a3_1b_diagnose.py session \
  --baseline-path /work/baseline-ec795ef1 --weights /work/mistral-7b \
  --shards /work/nf4-shards --out /work/new-diagnostic --execute
python benchmarks/harness/a3_1b_analyze.py /work/new-diagnostic \
  --baseline /work/baseline-ec795ef1 --out /work/new-analysis.json
```

## 4. Limits and earlier runs

The first protocol stopped at an in-training calibration OOM and left the scored spill gate without a verdict. The next protocol passed equality, RAM and timing, with cleanup unscored because the SIGINT scenario was missed. The final protocol restored the signal handler and synchronized delivery with a backward hold. Both earlier attempts retain their verdicts.

Cleanup after SIGKILL depends on a subsequent start using the same directory. The chosen reserve and overhead estimate describe this fixture and host. The diagnostic leaves the internal cast or kernel and earliest backward divergence unresolved. Its matched-input pass answers a different comparison from the original GA gate. The public diagnostic contains derived tensor comparisons and hashes. Full tensor replay requires the retained private files.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/96df4325d422397384d21bb21d492843c67fb84e/benchmarks/results
