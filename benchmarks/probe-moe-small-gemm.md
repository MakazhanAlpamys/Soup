# F4: MoE small-GEMM factor

I measured this on 2026-10-11 on a rented RTX 3060, in bfloat16, three rounds.
The rule is in git from 2026-10-09, before I ran anything.

What was measured:

```
factor(M) = TFLOPS at M rows / TFLOPS at 512 rows, same K and N, same session
tokens per expert = batch x seq x top_k / experts
```

Only M changes. The denominator is measured again inside every row so it can
seen if the card moved.

## Shapes

N = 2 * intermediate (gate_up is one fused matrix in transformers)
| model | hidden | inter | experts | top-k | gate_up K,N | down K,N |
|---|---|---|---|---|---|---|
| OLMoE-1B-7B-0924 | 2048 | 1024 | 64 | 8 | 2048, 2048 | 1024, 2048 |
| granite-3.0-1b-a400m | 1024 | 512 | 32 | 8 | 1024, 1024 | 512, 1024 |

Data taken from config.json
## Answer

| tokens/step | model | tokens/expert | gate_up | down |
|---|---|---|---|---|
| 512 | OLMoE | 64 | 0.601 | 0.592 |
| 512 | granite | 128 | 0.627 | 0.721 |
| 2048 | OLMoE | 256 | 0.821 | 0.938 |
| 2048 | granite | 512 | 0.999 | 1.000 |

The plan assumed 0.5-0.7. It appropriate for 512-token, but at 2048 tokens most of the loss is gone.

So the factor is not a property of MoE in general. It depends on how many
tokens one expert gets (comes from the step size, the expert count and
top-k). 0.5–0.7 cannot be reused as a general number.

## What is not in the number

Matrix multiply only. Not included: NF4 dequantisation on every weight visit,
the router, row gather and scatter, 64 kernel launches per layer, the backward
pass. Splitting those is checklist item 4, which I did not do.

Tensors are random: real shapes, synthetic data, no weights loaded.
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

TFLOPS at M=512, used as the denominator: 23.51, 20.83, 20.75, 16.11.

Plotted: `benchmarks/results/probe-f4-rtx3060/f4_curve.svg`

## Checks I ran

| check | result | bar |
|---|---|---|
| denominator spread inside a shape | 1.0003–1.0074x | - |
| a2/a1 over all 120 arms | 0.994–1.013 | - |
| round to round, the four answer rows | 1.002–1.023x | - |
| sanity check factor(1) | 0.006–0.010 | under 0.25 |
| SM clock under load | 1897–2002 MHz, median 1987 | V3: 10% |
| samples under 90% of median clock | 0 of 401 | - |

I could not use `nvidia-smi --lock-gpu-clocks` (no permission in the
container), so I sampled the clock once a second into gpu_log.csv.
406 s total, the card was busy 98.8% of it, power peaked at 170.1 W of 170,
temperature 41-65 C.

## Commands and versions

```bash
for i in 0 1 2; do python gemm_final.py $i | tee run_$i.txt; done

nvidia-smi --query-gpu=timestamp,clocks.sm,temperature.gpu,power.draw,\
utilization.gpu --format=csv -l 1 -f gpu_log.csv
```

Vast.ai instance 55384207, machine 47491. RTX 3060 12 GB (sm_86), driver
570.195.03, CUDA 12.8, power limit 170 W. Ryzen 5 3500, 6 cores, 32 GB RAM.
torch 2.11.0+cu128, bfloat16. Nothing else was running on the GPU.

Raw files are in `benchmarks/results/probe-f4-rtx3060/`: run_0..2.txt,
f4_round_0..2.json, gpu_log.csv, gemm_final.py.
