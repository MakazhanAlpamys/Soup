# Two expert reads per hit in layer-to-layer backward

Measured by @M1kado0 on 2026-11-16.

## 1. Question and setup

With complete expert streaming already enabled, does removing the recompute
read preserve the numerical result and shorten layer-to-layer optimizer steps?

The separate Linux run used a physical RTX 5070 (12 GB). Arm receipts show the same
91161100288-byte cgroup-v2 limit and zero cgroup swap limit. The protocol describes
an emulated 29.756 GiB RAM envelope. No new ballast receipt, Linux release,
driver version or CUDA runtime version was verified for this series. The arm
stack was Python 3.12.3, Torch 2.15.0, Transformers 5.23.0, PEFT 0.21.3,
bitsandbytes 0.50.2, safetensors 0.8.0 and Soup CLI 0.75.0.

Kanana-2-30B-A3B-base-2601 was pinned at `0d31365acf656e5f9a593cacc4913e9c0d36cb1c`.
The recipe used disk streaming, double-quantized NF4, BF16 compute and FP32 LoRA
on `q_proj`, `kv_b_proj` and `o_proj` (rank 4, alpha 8), microbatch one and
synthetic 512-token sequences. Both complete variants grouped routes, transferred
packed weights and used an 8 GiB expert host cache. The two-read variant records
routing on the first forward and replays it during recompute, skipping the
recompute expert read.

## 2. Method

The frozen amendment required CUDA accuracy at k=2 only. Here k is the number of
microbatches per optimizer update. The saved recapture passed all four numerical
pairs and recorded bitwise equality for checked losses and gradients. The read
probe recorded three reads per hit for complete streaming and two for its candidate.
Timing used fresh ABBA processes, two warmup updates and six measured updates.
Five clean blocks per comparison supported the paired two-sided t-test at
alpha 0.05, with a 10% minimum reduction for a win.

Suspend, power and foreign reads were mandatory. This series admitted maximum
possible foreign reads up to `max(1,000,000,000 bytes, 0.05 * own reads)`. The driver
also recomputed every outcome under the original strict limit of less than
1,000,000,000 bytes, using the saved disk bounds and the same block-count rule.
The protocol stopped the series after two recorded runs whose minimum foreign
reads exceeded 5% of their own.

## 3. Results

Reductions are stored fractions. Each time is the mean of arm medians for the
relative-threshold view.

| Comparison | k | Control step, s | Candidate step, s | Reduction | p | Relative threshold | Original threshold |
|---|---:|---:|---:|---:|---:|---|---|
| Ordinary → complete | 2 | 57.6032578573795 | 37.78643690176541 | 0.34402257255446844 | 4.014047071783687e-06 | Win, 5 blocks | No verdict, 1 block |
| Complete → two-read complete | 1 | 19.725985202821903 | 13.907170520606451 | 0.2949822086139982 | 1.0181868261453474e-05 | Win, 5 blocks | No verdict, 4 blocks |
| Complete → two-read complete | 2 | 37.67353969811229 | 26.333204492356163 | 0.30101591983734827 | 3.9781810846552795e-07 | Win, 5 blocks | No verdict, 3 blocks |

**Verdict — mixed.** All three comparisons won under the frozen relative threshold. All three had insufficient clean blocks under the original threshold.

## 4. Commands

The protocol and benchmark driver live at the evidence commit. These are the
frozen command profiles. Supply the pinned snapshot with `--source`, the
per-expert shard cache with `--cache`, and the study directory with `--root`.

```bash
python benchmarks/harness/oc_stage1_drive.py stage1 --two-read --capture-ks 2 --source SRC --cache CACHE --root ROOT --revision 0d31365acf656e5f9a593cacc4913e9c0d36cb1c
python benchmarks/harness/oc_stage1_drive.py verdict --two-read --capture-ks 2 --ks 1 2 --source SRC --cache CACHE --root ROOT
```

## 5. Limits and run history

The first capture is preserved, including a void foreign-read observation.
The committed recapture supplies the final numerical gate. The timing ledger
records 20 eligible arms per comparison under the relative threshold. The
prior ordinary-versus-complete series at k=2 retains its original no-verdict
result. This study leaves k=4 and k=8 performance and k=8 accuracy unmeasured.
The relative and original threshold outcomes are sensitive to foreign I/O
and should travel together. `training.l2l`, `training.l2l_grouped_expert_routes`,
`training.l2l_two_read_experts` and `training.stream_experts` are pending.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/96df4325d422397384d21bb21d492843c67fb84e/benchmarks/results/two-read-l2l-kanana-20261116
