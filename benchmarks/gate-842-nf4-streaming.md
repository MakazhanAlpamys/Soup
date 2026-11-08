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

Hardware measurements pending.
