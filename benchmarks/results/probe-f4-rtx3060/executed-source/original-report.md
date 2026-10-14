# F4: MoE small-GEMM factor

Measured on two rented RTX 3060s, bfloat16, three rounds each: the GEMM curve
2026-10-11, the layer breakdown 2026-10-12. Rule committed 2026-10-09, before
any run.

The per-expert GEMM alone runs at 0.59-0.72 of the same GEMM at 512 rows, which
is what the plan assumed. A whole MoE layer, routing and NF4 dequantisation
included, runs at 0.148 of a dense layer doing the same arithmetic.

```
factor(M) = TFLOPS at M rows / TFLOPS at 512 rows, same K and N, same session
tokens per expert = batch x seq x top_k / experts
```

Only M changes. The denominator is re-measured inside every row, so a card that
moves during the sweep is visible.

## Shapes

N = 2 * intermediate (gate_up is one fused matrix in transformers).
From config.json.

| model | hidden | inter | experts | top-k | gate_up K,N | down K,N |
|---|---|---|---|---|---|---|
| OLMoE-1B-7B-0924 | 2048 | 1024 | 64 | 8 | 2048, 2048 | 1024, 2048 |
| granite-3.0-1b-a400m | 1024 | 512 | 32 | 8 | 1024, 1024 | 512, 1024 |

## Answer, the GEMM alone

| tokens/step | model | tokens/expert | gate_up | down |
|---|---|---|---|---|
| 512 | OLMoE | 64 | 0.601 | 0.592 |
| 512 | granite | 128 | 0.627 | 0.721 |
| 2048 | OLMoE | 256 | 0.821 | 0.938 |
| 2048 | granite | 512 | 0.999 | 1.000 |

The plan assumed 0.5-0.7. Holds for a 512-token step; at 2048 tokens most of
the loss is gone. The factor is a function of tokens per expert — step size,
expert count and top-k together — not a property of MoE. 0.5-0.7 is not a
general number.

## Curves

Median of three rounds.

| M | olmoe_gate_up | olmoe_down | granite_gate_up | granite_down |
|---|---|---|---|---|
| 1 | 0.009 | 0.010 | 0.009 | 0.006 |
| 16 | 0.119 | 0.142 | 0.066 | 0.065 |
| 32 | 0.299 | 0.297 | 0.275 | 0.215 |
| 64 | 0.601 | 0.592 | 0.293 | 0.430 |
| 128 | 0.511 | 0.648 | 0.627 | 0.721 |
| 256 | 0.821 | 0.938 | 0.655 | 0.758 |
| 512 | 0.998 | 0.998 | 0.999 | 1.000 |
| 1024 | 0.986 | 1.001 | 0.976 | 1.179 |
| 2048 | 1.079 | 1.197 | 1.134 | 1.385 |
| 8192 | 1.126 | 1.228 | 1.205 | 1.417 |

TFLOPS at M=512, the denominator: 23.51, 20.83, 20.75, 16.11.

Plotted: `benchmarks/results/probe-f4-rtx3060/f4_curve.svg`

## The whole layer

Checklist item 4. One OLMoE MoE layer forward, 512 tokens, 64 experts, top-8.
Four arms, each adding one term; differences give the split. Median of three
rounds.

| part | ms | share |
|---|---|---|
| the multiply, 128 GEMMs | 4.883 | 32.9% |
| 64 x torch.where | 4.881 | 32.8% |
| weight restore, NF4 | 4.174 | 28.1% |
| gather + index_add_ | 0.639 | 4.3% |
| router GEMM + softmax + top-k | 0.160 | 1.1% |
| total | 14.859 | |

Parts sum to 99.2% of the total; the rest is slop in the differences.

Routing was first measured as one term of 5.163 ms, then split with an extra
arm running the router but using precomputed row indices: the router is
0.160 ms, the 64 `torch.where` calls are 4.881 ms — 94.5% of it.

`torch.where(mask[e])` returns a tensor whose size depends on the data, so the
CPU must wait for the GPU to learn it. One forced synchronisation per expert,
64 per layer. `modeling_olmoe.py:306`.

This is a property of the reference implementation, not of MoE. Sorting tokens
by expert once and working with offsets removes the syncs, so roughly a third
of the layer is avoidable overhead. Input for D3.

### Cascade

One denominator throughout: a dense layer doing the same arithmetic. The layer
is 51.5 GFLOP, which at the measured 23.48 TFLOPS takes 2.20 ms.

| what | factor |
|---|---|
| isolated GEMM at M=64 | 0.593 |
| 64 experts in a Python loop | 0.450 |
| whole layer, routing and NF4 included | 0.148 |

The layer is 6.8x slower than the dense equivalent, not 1.7x. For a whole layer
the plan's 0.5-0.7 is about four times too optimistic.

Expectation recorded before the run: NF4 dequantisation comparable to the
multiply. Held — 4.174 against 4.883 ms. The indexing cost was not predicted.

## What is not in the number

The backward pass, attention, a whole training step. One MoE layer forward only.

Tensors are random: real shapes, synthetic data, no weights loaded. Arms A and
B use a fixed balanced token-to-expert assignment; arms C and D use the real
router on random hidden states, so per-expert counts are not those a trained
router gives on real text.

## Checks

| check | result | bar |
|---|---|---|
| denominator spread inside a shape | 1.0003-1.0074x | - |
| a2/a1 over all 120 arms | 0.994-1.013 | - |
| round to round, the four answer rows | 1.002-1.023x | - |
| sanity check factor(1) | 0.006-0.010 | under 0.25 |
| SM clock under load | 1897-2002 MHz, median 1987 | V3: 10% |
| samples under 90% of median clock | 0 of 401 | - |
| layer arms, round to round | 1.008-1.015x | - |

`nvidia-smi --lock-gpu-clocks` is not permitted in the container, so the clock
was sampled once a second into gpu_log.csv: 406 s, card busy 98.8%, power peak
170.1 W of 170, temperature 41-65 C.

The layer run is on a second physical card and re-measures the reference point
in its own session: 0.593 against 0.601 on the first card, 1.013x apart. Every
term in the cascade comes from that one session.

## Commands and versions

```bash
# GEMM curve, 2026-10-11
for i in 0 1 2; do python gemm_final.py $i | tee run_$i.txt; done

# layer breakdown, 2026-10-12
for i in 0 1 2; do python moe_breakdown.py $i | tee item4_$i.txt; done

# clock sampled once a second during both
nvidia-smi --query-gpu=timestamp,clocks.sm,temperature.gpu,power.draw,\
utilization.gpu --format=csv -l 1 -f gpu_log.csv
```

Both cards: RTX 3060 12 GB (sm_86), driver 570.195.03, CUDA 12.8, power limit
170 W, torch 2.11.0+cu128, bfloat16, no other process on the GPU.
Curve: Vast.ai instance 55384207, machine 47491, Ryzen 5 3500, 32 GB RAM.
Layer: instance 55531564, machine 47487, same host, bitsandbytes 0.50.2.

Raw: `benchmarks/results/probe-f4-rtx3060/` — run_0..2.txt, f4_round_0..2.json,
gpu_log.csv, gemm_final.py, item4_0..2.txt, f4_item4_round_0..3.json,
gpu_log_item4.csv, moe_breakdown.py.
