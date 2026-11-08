# Checkpointed NF4 projections on a Kaggle T4 (#842)

The [rule posted before timing](https://github.com/MakazhanAlpamys/Soup/issues/842#issuecomment-6723852052)
sets the arm order, noise criterion and stopping rule. The unit is one projection at the
specified shape. Peak means peak allocated bytes on the CUDA device. Projection tokens/s
counts input rows per forward and backward pair. It excludes model layers, data loading,
host transfers and the optimizer.

Run from the repository root with Soup installed. The GPU probes use bitsandbytes 0.50.2 and
CUDA fp16. The CPU self-test uses PyTorch and exercises the counters and decision rules.

```bash
python benchmarks/harness/nf4_stream_bench.py --self-test
python benchmarks/harness/nf4_stream_bench.py --ablation --json ablation.json
python benchmarks/harness/nf4_stream_bench.py --aliasing --json aliasing.json
python benchmarks/harness/nf4_stream_bench.py --grid --json grid.json
```

The ablation compares the shipped repair (A) with cached dense weights (B) at M=512,
N=K=4096. Both arms hold the cached matrix throughout. The grid compares the
checkpoint-visible Function (A) with explicit dequantisation and linear (B).
The native reference uses the same quantised bytes, input and upstream gradient.
Backward timing includes any checkpoint recomputation. Each child samples the SM clock and
memory-controller utilisation during the timed loop. Its same-shape fp16 GEMM ceiling
comes from 20 warm-ups and 100 forward matrix multiplications into preallocated output.
A negative time reduction means B took longer.

The aliasing control chains four synthetic layers with two slots. Both the packed nibbles
and nested quantisation statistics are overwritten on reuse. Each layer has a trainable
input scale so the loss gradient depends on the weight read in backward. The native
private-buffer arm supplies the reference. Every layer output must match. The unsafe arm
must produce a finite wrong gradient, and bypassing the pool must restore exact gradients.

## Kaggle Tesla T4, native fp16

Kaggle ran commit `714cb9f7` on device 0, sm_75, driver 580.178.04, CUDA 13.0,
PyTorch 2.11.0+cu130, bitsandbytes 0.50.2, peft 0.21.2, transformers 5.20.2.
The input was synthetic. Scales were double-quantised with 64-element weight blocks.

| Fresh process | Shipped / cached ms | Reduction | Projection tokens/s, shipped / cached | Peak MiB, shipped / cached | Timed SM clock min/median/max MHz | Median memory-controller use | Same-shape dense GEMM TFLOP/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.025 / 1.451 | 28.34% | 252,850 / 352,858 | 104.51 / 72.51 | 585/1110/1200 | 37% | 22.84 |
| 2 | 2.064 / 1.491 | 27.77% | 248,029 / 343,388 | 104.51 / 72.51 | 630/960/1125 | 30% | 24.31 |
| 3 | 2.269 / 1.470 | 35.22% | 225,603 / 348,284 | 104.51 / 72.51 | 765/1110/1170 | 41% | 26.05 |

All three paired reductions exceeded the 2% stopping threshold. The gap between
the arm medians was 0.594 ms and the larger within-arm range was 0.245 ms.
Across all 300 timed pairs the shipped arm dequantised once in forward and once
in recomputation. The cached arm used the same 32 MiB matrix each time. The
forward and input gradient matched resident NF4 bit for bit in all three checks.
[Raw ablation rows and clock summaries](evidence/issue842/t4-ablation.json.gz).

In the four-layer aliasing probe, every forward matched the private reference.
The unsafe arm's input gradient differed by up to 0.07002, and its scale gradients
for layers 0 and 1 differed by 0.16223 and 0.16455. Layers 2 and 3 were exact.
Bypassing the pool and enabling the shipped repair each restored exact input
and scale gradients. All four projections took the fused backend in every
forward. The repair recomputed four fused forwards in backward. Each synthetic
layer had 171.125 MiB of packed weight before quantisation metadata.
[Aliasing results and executed paths](evidence/issue842/t4-aliasing.json).
