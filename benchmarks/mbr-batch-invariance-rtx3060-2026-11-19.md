# Moonlight MBR batch invariance and segmented GEMM on RTX 3060

Measured by @M1kado0 on 2026-11-19.

**Question.** Can MBR reproduce the loss and LoRA gradient bits of native per-microbatch L2L, and does doing so fit a 10% overhead allowance against native grouped MBR?

## Setup and commands

One RTX 3060 12 GB, an AMD Ryzen Threadripper PRO 3945WX, driver 615.71.09, CUDA 13.0 and PyTorch 2.14.1+cu130. The run used a Linux container and a warm 12 GiB expert host cache for the timed segmented comparisons. Physical host RAM, Linux release and other library versions are unreported in the run environment receipt. Moonlight-16B-A3B ran in BF16 with per-expert NF4 weights, a fixed 8 × 64-token window and a group of eight.

The launchers live at the evidence commit. The second launcher drives balanced fresh-process comparisons with the phase-profile bridge.

```bash
bash benchmarks/results/mbr-batch-invariance-gpu-20261119/remote-series.sh
bash benchmarks/results/mbr-batch-invariance-f1-20261119/remote-seg.sh A0d B0s B0s2
bash benchmarks/results/mbr-batch-invariance-f1-20261119/remote-f1-b0s.sh
```

## Method

Native L2L uses one expert GEMM per microbatch. The segmented MBR harness retained those GEMM shapes, grouped expert reads, then backpropagated through the remainder of each layer in L2L order. That last technique corresponds to pending `training.l2l_mbr_split_backward` under pending `training.l2l_route_preserving_mbr`. A separate row-wise reference tested whether agreement was possible when both arms changed the GEMM arithmetic. The pending Triton option `training.stream_expert_gemm: batch_invariant` is a different implementation from the row-wise and segmented harness paths measured here.

The gradient series compared loss and raw-gradient digests. After exploratory results, the timing protocol added two segmented comparisons. Each used five control–candidate–candidate–control blocks, fresh processes, the median of three timed steps per process, and a two-sided paired t-test at 5%.

## Results

| Numerical comparison | L2L loss | MBR loss | Raw LoRA gradients |
|---|---:|---:|---|
| Native per-microbatch L2L versus native grouped MBR | 5.668856 | 5.667779 | Digests differ |
| Native per-microbatch L2L versus segmented MBR with split backward | 5.668856 | 5.668856 | Digests match |
| Row-wise L2L versus corrected row-wise MBR | 5.671227931976318 | 5.671227931976318 | Digests match |

| Segmented MBR compared with | Mean change in step time | Paired t | Outcome |
|---|---:|---:|---|
| Native per-microbatch L2L | −62.2% | −92.5 | Faster with matching gradient bits |
| Native grouped MBR | +43.1% | 28.8 | Exceeded the 10% overhead allowance |

**Verdict — mixed.** Segmented MBR preserved native L2L bits on this window and missed the overhead allowance against native grouped MBR.

## Limitations and superseded runs

Row-wise arithmetic established a different reference from native L2L. The corrected route-weight reduction superseded the initial row-wise attempt, which matched loss and differed in gradients. The segmented comparisons were added after exploration. The row-wise timing pilot lacked required sensors and padding was only screened. The study measures a fixed window on one GPU and does not establish long-run quality or parity across library versions.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/96df4325d422397384d21bb21d492843c67fb84e/benchmarks/results/mbr-batch-invariance-f1-20261119
