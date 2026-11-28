# Ordinary versus complete streaming on Kanana

Measured by @M1kado0 on 2026-11-15.

## 1. Question and setup

Does the combined expert-streaming recipe shorten optimizer steps for conventional
gradient accumulation or layer-to-layer backward while preserving numerical behavior?

The Linux host had a physical RTX 5070 (12 GB). Arm receipts record a cgroup-v2
limit of 91161100288 bytes and a zero cgroup swap limit. A separate memory-ballast
process held 55.14 GiB inside the cgroup to target 29.756 GiB of effective headroom.
This emulates the smaller RAM envelope. The setup log points to Ubuntu 24.04
package sources. Driver and CUDA runtime versions are unspecified in the published
arm metadata. The stack was Python 3.12.3, Torch 2.15.0, Transformers 5.23.0,
PEFT 0.21.3, bitsandbytes 0.50.2, safetensors 0.8.0 and Soup CLI 0.75.0.

Kanana-2-30B-A3B-base-2601 was pinned at `0d31365acf656e5f9a593cacc4913e9c0d36cb1c`.
The recipe used disk streaming, double-quantized NF4, BF16 compute and FP32 LoRA
on `q_proj`, `kv_b_proj` and `o_proj` (rank 4, alpha 8), microbatch one, and
synthetic 512-token sequences. Complete streaming combined grouped routes,
packed transfer and an 8 GiB expert host cache. For conventional accumulation
it also used two reads per expert hit. The historical complete layer-to-layer arm
used three.

## 2. Method

The two schedules each compared ordinary and complete streaming within schedule.
Here `k` is the number of microbatches in an optimizer update. Before timing,
two-update AdamW captures at k=2 and k=8 were judged against 1.5 times the same-run
Torch SDPA math versus efficient-attention numerical distance. The saved comparisons
passed and recorded bitwise-equal gradients and losses. These numerical captures
are distinct from the timing validity checks.

Each timing arm used the median of six measured updates after two warmups. Fresh
ABBA blocks used a two-sided paired t-test on block mean differences, with
alpha 0.05 and a 10% practical reduction threshold. The rule began at five clean
blocks with a single variance-based update to the required count. Suspend,
power and foreign reads were mandatory. Void blocks remained in the ledger and
were eligible for replacement. Clock telemetry was reported separately.

## 3. Results

The reduction is the stored fraction, `(ordinary - complete) / ordinary`.

| Schedule | k | Clean / run blocks | Ordinary step, s | Complete step, s | Reduction | p | Outcome |
|---|---:|---:|---:|---:|---:|---:|---|
| Conventional accumulation | 1 | 5 / 5 | 29.039791595668067 | 13.911824478360359 | 0.5209392452928066 | 1.605052576240636e-07 | Win |
| Layer-to-layer backward | 1 | 5 / 9 | 28.254467169230338 | 19.927124927507247 | 0.29472657161950405 | 4.324325713230376e-05 | Win |
| Conventional accumulation | 2 | 5 / 7 | 57.095565909694415 | 28.10973705392098 | 0.5076721527135586 | 9.337866109881405e-08 | Win |
| Layer-to-layer backward | 2 | 3 / 9 | — | — | — | — | No verdict |
| Both schedules | 4 and 8 | 0 / 0 each | — | — | — | — | No verdict |

**Verdict — mixed.** Both conventional comparisons and the k=1 layer-to-layer comparison met the win rule. The k=2 layer-to-layer comparison had too few valid blocks.

## 4. Commands

The frozen rule, arm runner and ABBA driver live at the evidence commit. The
original study used the driver's capture, block and verdict commands. The
runner's recorded invocation form for a single complete arm is below. Supply
the pinned snapshot, per-expert shard cache and run output path.

```bash
benchmarks/harness/oc_stage1_drive.py
python benchmarks/harness/oc_arm_run.py run --arm ga_complete --k 2 --source SRC --cache CACHE --out OUT --device cuda
python benchmarks/harness/oc_arm_run.py judge --runs RUNS --k 2
```

## 5. Limits and run history

The saved timing blocks exclude void observations according to the frozen rule.
All original timing verdicts stand and the later two-read series uses fresh blocks.
The compared recipe bundles multiple placement and routing changes, and the
recorded wins apply to this synthetic workload and its RAM emulation. Timing
evidence for k=4 and k=8 is absent. `training.l2l`, `training.l2l_grouped_expert_routes`,
`training.l2l_two_read_experts` and `training.stream_experts` are pending.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/96df4325d422397384d21bb21d492843c67fb84e/benchmarks/results/ordinary-vs-complete-l2l-stage1-20261115
