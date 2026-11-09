# Checkpointed NF4 projections on a Kaggle T4 (#842)

The [rule posted before timing](https://github.com/MakazhanAlpamys/Soup/issues/842#issuecomment-6723852052)
sets the arm order, noise criterion and stopping rule. The unit is one projection at the
specified shape. Peak means peak allocated bytes on the CUDA device. Projection tokens/s
counts input rows per forward and backward pair. It excludes model layers, data loading,
host transfers and the optimizer.

All measurements apply on this card, at these shapes, and make no end-to-end claim
about a full model training step. They form a new baseline on the Tesla T4 (sm_75)
that cannot be combined with the earlier RTX 3050 or RTX 5070 figures in #842.

Run from the repository root with Soup installed. The GPU probes use bitsandbytes 0.50.2 and
CUDA fp16. The CPU self-test uses PyTorch and exercises the counters and decision rules.

```bash
python benchmarks/harness/nf4_stream_bench.py --self-test
python benchmarks/harness/nf4_stream_bench.py --ablation --json ablation.json
python benchmarks/harness/nf4_stream_bench.py --aliasing --json aliasing.json
python benchmarks/harness/nf4_stream_bench.py --grid --json grid.json
python benchmarks/harness/nf4_stream_bench.py --model --json model.json
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
Commit `714cb9f7` produced `t4-ablation.json.gz` and `t4-aliasing.json`. Because later
commits added `--model` and tracing to `nf4_stream_bench.py`, running `--ablation` from
the branch head produces a different script SHA-256 hash than the record.

| Fresh process | Shipped / cached ms | Reduction | Projection tokens/s, shipped / cached | Peak MiB, shipped / cached | Timed SM clock min/median/max MHz (5 samples) | Median memory-controller use (5 samples) | Same-shape dense GEMM TFLOP/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.025 / 1.451 | 28.34% | 252,850 / 352,858 | 104.51 / 72.51 | 585/1110/1200 | 37% | 22.84 |
| 2 | 2.064 / 1.491 | 27.77% | 248,029 / 343,388 | 104.51 / 72.51 | 630/960/1125 | 30% | 24.31 |
| 3 | 2.269 / 1.470 | 35.22% | 225,603 / 348,284 | 104.51 / 72.51 | 765/1110/1170 | 41% | 26.05 |

All three paired reductions exceeded the 2% stopping threshold. The gap between
the arm medians was 0.594 ms and the larger within-arm range was 0.245 ms.
The timed SM clock and memory-controller metrics rest on 5 in-window samples per process
(`sample_count: 5`), with the SM clock moving between 585 and 1200 MHz under load.
Across all 300 timed pairs the shipped arm dequantised once in forward and once
in recomputation. The cached arm used the same 32 MiB matrix each time. The reduction includes dispatch
and launch overhead avoided by reusing that matrix. It is an ablation of one projection. The
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

## ZeroGPU correctness check

The same four-layer fixture from commit `714cb9f7` ran on an NVIDIA RTX PRO 6000 Blackwell Server Edition (producing `blackwell-aliasing.json`),
MIG 2g.48gb, sm_120, driver 580.178.04, CUDA 13.0, PyTorch 2.11.0 and
bitsandbytes 0.50.2. All forward outputs matched. With the unsafe arm, the first
two scale gradients differed by up to 0.16992 and 0.15448, and the input gradient
by 0.07291. The bypass and repaired arms matched the private reference bit for bit
in all five gradients. Each forward used four fused kernels. The repaired backward
used four fused recomputations and four dense dequantisations. This pass used native
fp16 and no timing loop. PEFT and transformers were absent from this worker.
The first request returned a generic Gradio error. The second exposed an integer slot ID
that Gradio could not serialise. The successful request returned JSON text.
[Blackwell result](evidence/issue842/blackwell-aliasing.json) ·
[Space revision](https://huggingface.co/spaces/none02007/stored1/tree/27a5ec647b23d2a78d22ded053d5bdb1169794f5).

## Projection shapes on the T4

Kaggle ran commit `a47f3f76` on the same T4 stack. A calls the checkpoint-visible operation, B calls explicit dequantisation and linear. At shapes labelled fallback, bitsandbytes also dequantised in A. The delta columns compare B with native resident NF4 on the same inputs. A must match the resident forward and input gradient exactly to publish a timing. Times below are medians of three process medians. The forward/backward verdict describes B relative to A under the posted noise rule.

| N × K | M | A path | Max Δ forward / dX | Forward ms A / B | Backward ms A / B | Forward / backward verdict for B | Projection ktok/s A / B | Peak MiB A / B |
|---|---:|---|---:|---:|---:|---|---:|---:|
| 4096 × 4096 | 3 | kernel | 0.0019531 / 0 | 0.627 / 0.820 | 0.943 / 0.543 | slower / faster | 1.9 / 2.2 | 57.58 / 57.58 |
| 4096 × 4096 | 8 | kernel | 0.00048828 / 0 | 0.632 / 0.798 | 0.980 / 0.552 | slower / faster | 4.8 / 6.0 | 57.69 / 57.69 |
| 4096 × 4096 | 32 | kernel | 0.0019531 / 0 | 0.656 / 0.803 | 0.959 / 0.510 | slower / faster | 19.4 / 24.6 | 58.26 / 58.26 |
| 4096 × 4096 | 128 | fallback | 0 / 0 | 0.832 / 0.776 | 1.012 / 0.511 | noise / faster | 70.0 / 100.1 | 62.51 / 60.51 |
| 4096 × 4096 | 512 | fallback | 0 / 0 | 1.253 / 1.225 | 1.774 / 0.885 | noise / faster | 167.5 / 241.2 | 74.51 / 72.51 |
| 1024 × 4096 | 3 | kernel | 0.0019531 / 0 | 0.423 / 0.484 | 0.847 / 0.692 | slower / faster | 2.3 / 2.6 | 26.60 / 26.60 |
| 1024 × 4096 | 8 | kernel | 0.0019531 / 0 | 0.504 / 0.496 | 0.761 / 0.690 | noise / faster | 6.3 / 6.8 | 26.66 / 26.66 |
| 1024 × 4096 | 32 | kernel | 0.00097656 / 0 | 0.543 / 0.498 | 0.789 / 0.684 | noise / noise | 25.5 / 27.7 | 26.94 / 26.94 |
| 1024 × 4096 | 128 | fallback | 0 / 0 | 0.540 / 0.522 | 0.906 / 0.613 | noise / faster | 88.5 / 113.4 | 28.82 / 28.82 |
| 1024 × 4096 | 512 | fallback | 0 / 0 | 0.690 / 0.673 | 0.926 / 0.596 | noise / faster | 314.3 / 394.9 | 36.32 / 36.32 |
| 14336 × 4096 | 3 | kernel | 0.0019531 / 0 | 0.934 / 1.589 | 1.842 / 1.253 | slower / faster | 1.1 / 1.1 | 160.83 / 160.83 |
| 14336 × 4096 | 8 | kernel | 0.0019531 / 0 | 0.960 / 1.560 | 1.904 / 1.263 | slower / faster | 2.8 / 2.8 | 161.14 / 161.14 |
| 14336 × 4096 | 32 | kernel | 0.0019531 / 0 | 1.077 / 1.699 | 1.956 / 1.222 | slower / faster | 10.4 / 10.9 | 162.64 / 162.64 |
| 14336 × 4096 | 128 | kernel | 0.0019531 / 0 | 2.284 / 1.724 | 3.357 / 1.441 | faster / faster | 22.7 / 40.2 | 168.64 / 168.64 |
| 14336 × 4096 | 512 | fallback | 0 / 0 | 4.492 / 4.434 | 8.696 / 4.565 | noise / faster | 39.0 / 56.8 | 210.14 / 193.14 |
| 4096 × 14336 | 3 | kernel | 0.0019531 / 0 | 0.988 / 1.630 | 1.918 / 1.251 | slower / faster | 1.0 / 1.0 | 160.77 / 160.77 |
| 4096 × 14336 | 8 | kernel | 0.0019531 / 0 | 1.191 / 1.632 | 2.118 / 1.265 | slower / faster | 2.4 / 2.8 | 160.99 / 160.99 |
| 4096 × 14336 | 32 | kernel | 0.0019531 / 0 | 1.362 / 1.818 | 2.377 / 1.386 | slower / faster | 8.4 / 9.9 | 162.02 / 162.02 |
| 4096 × 14336 | 128 | fallback | 0 / 0 | 2.741 / 2.675 | 4.704 / 2.300 | faster / faster | 17.3 / 25.7 | 170.64 / 166.14 |
| 4096 × 14336 | 512 | fallback | 0 / 0 | 6.242 / 6.274 | 11.142 / 5.352 | noise / faster | 29.2 / 44.0 | 193.14 / 193.14 |

“Noise” means no measurable difference under the posted rule. Peak is the maximum across processes. Each process also records its dense GEMM ceiling and its in-window SM clock and memory-controller samples. [All timings, path counts and telemetry summaries](evidence/issue842/t4-grid.json.gz).

Across 20 cells, 13 used the fused kernel and seven used the backend fallback. In the
13 fused cells, the Function won ten forward comparisons, lost one and had two within
noise. Explicit dequantisation won twelve backward comparisons, with one within noise.
The supplied gradient was fixed in both arms. The largest fused/dense forward difference
was 0.001953125. The input gradient matched in every cell.

## Streamed model against resident NF4 on the T4

`--model` builds a four-decoder Llama fixture, hidden size 4096, intermediate size
26624, vocab 64, M=3. Each NF4 layer shard occupied 181.60 MiB. Both streamed arms
read the same pinned RAM source through two slots. The resident model used native
bitsandbytes. The q/v projections in all four layers had rank-16 LoRA adapters with
nonzero B matrices. Each comparison used the same adapters, input and loss.

| Arm | Forward logits | Adapter gradients | Fused calls in forward | Backward reads of native pooled weights |
|---|---|---|---:|---|
| Shipped repair | bit-exact on 5/5 | 16/16 bit-exact on each backward | 28 per pass | 0 |
| Repair disabled | bit-exact on 5/5 | 16/16 bit-exact on each backward | 28 per pass | 25 per pass |

**The full-model negative control did not fire.** The first run at `a47f3f76`
already had exact logits and gradients in both arms. The diagnostic at `5c0aef61`
repeated the same backward five times per arm, clearing gradients between passes.
It recorded the packed pointer's slot, owner and generation at native forward
and checked them when native backward consumed the pointer. On each backward,
14 reads had the original generation and 11 had a later generation with the same
layer as owner. All 25 pointers mapped to the pool. The counters describe the
scheduler state at dispatch. The separate recycled-slot fixture directly
demonstrated wrong gradients when the saved weight belonged to another layer.

The full-model command exits 1 on this result. The observed forward and backward
parity stands for this configuration. Its two arms do not distinguish the repair.
Layer size alone did not make the historical failure appear here.
[First pass](evidence/issue842/t4-model-first-pass.json) ·
[Five repeats, per-parameter gradients and slot ownership](evidence/issue842/t4-model-repeated.json).
