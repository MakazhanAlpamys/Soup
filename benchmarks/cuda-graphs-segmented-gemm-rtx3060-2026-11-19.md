# CUDA graphs for segmented expert GEMM on RTX 3060

Measured by @M1kado0 on 2026-11-19.

**Question.** Are exact segment shapes reused often enough in a Moonlight step to justify one CUDA graph per expert-call shape?

## Setup and commands

One RTX 3060 12 GB, driver 615.71.09, CUDA 13.0 and PyTorch 2.14.1+cu130 on the Linux batch-invariance test host with a Threadripper PRO 3945WX. Physical host RAM, Linux release and additional library versions are unreported. The shape trace used Moonlight-16B-A3B, BF16 compute and per-expert NF4 weights. It exercised the pending `training.l2l_route_preserving_mbr` setting with segmented native matmul in the harness. The separate capture probe used one synthetic BF16 expert.

The scripts live at the evidence commit.

```bash
bash benchmarks/results/mbr-batch-invariance-f1-20261119/micro/run-plans-stats.sh
python benchmarks/results/mbr-batch-invariance-f1-20261119/micro/micro_cuda_graph.py
```

## Method and results

The study reviewed CUDA graph capture, fixed shapes and numerical reproducibility. On hardware it counted ordered tuples of segment sizes from one segmented MBR step. The isolated probe compared eager and replayed outputs and input and weight gradients for fixed shapes. The trace is retained. The probe's measured output is unavailable.

| Shape trace metric | Count |
|---|---:|
| Expert calls | 1,572 |
| Distinct ordered segment-size tuples | 1,357 |
| Tuples seen once | 1,313 |
| Distinct individual segment sizes | 64 |

**Verdict — no verdict.** Whole-call graph reuse was sparse in the trace. End-to-end speed and eager-versus-replay exactness remain unverified here.

## Limitations and superseded runs

The result covers one step. It does not estimate reuse across steps, graph-pool memory or the gain from capturing individual segments. No earlier trace was superseded and no full-model capture timing was retained.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/96df4325d422397384d21bb21d492843c67fb84e/benchmarks/results/mbr-batch-invariance-f1-20261119/micro
