# F4: MoE small-GEMM factor

**Corrected descriptive report, 2026-10-15. No formal validity verdict.**
GEMM curves were recorded on an RTX 3060 on 2026-10-11; the synthetic layer
extension used a second RTX 3060 on 2026-10-12. No new GPU run was made for this
correction. The original [pre-run rule](results/probe-f4-rtx3060/executed-source/pre-run-rule.md)
at [11bf21ae](https://github.com/SoupTeam/soup-rnd/commit/11bf21ae76674c8fad6b331b323976076d0b3f43)
and the [withdrawn original report](results/probe-f4-rtx3060/executed-source/original-report.md)
are preserved byte-for-byte.

The isolated GEMM ratios at the stated 512-token recipe are 0.592–0.721 relative
to the same shape at 512 rows. These are observations for these shapes and cards.
The synthetic, unweighted expert loop's 0.148 is a **modeled normalization**
against layer FLOPs divided by a measured gate-up GEMM throughput; an equivalent
dense layer was not measured. It is not a measured training factor for giant MoE.

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

The historical plan assumed 0.5-0.7. Three rows at the 512-token recipe fall
in that interval; granite down is 0.721, above 0.70. Larger-M ratios increase
for these shapes, but this does not establish a formal validity verdict. The factor is a function of tokens per expert — step size,
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

## Synthetic layer extension: observed arm times

One OLMoE-shaped expert loop forward, 512 tokens, 64 experts, top-8. This
extension is outside the original GEMM rule. Median arm times from rounds
0–2 (round 3 is also retained, but was not used for the published medians):

| arm | measured ms | workload |
|---|---|---|
| A | 4.883 | 64 experts, contiguous 64-row inputs, bf16 weights |
| B | 5.522 | fixed row gather and scatter, bf16 weights |
| C | 10.685 | random router, actual mask rows, bf16 weights |
| D | 14.859 | random router, actual mask rows, NF4 weight restore |

A and B do not have equivalent outputs. The fixed-row modulo fixture gives
64 rows per expert but only eight unique token indices, each repeated eight
times. C and D use the router's actual rows and discard routing weights.
The arms are synthetic workloads, not faithful weighted MoE inference or training.

The historical B−A (0.639 ms), C−B (5.163 ms) and D−C (4.174 ms) are descriptive
median latency differences. B and C change row counts, input data and scatter
contention as well as routing. D changes weights through quantization. No output
or gradient equivalence control was saved, so these differences do not establish
an additive causal breakdown or the cost of isolated components.

### Withdrawn torch.where attribution

The previous 4.881 ms / 32.8% attribution and claim that sorting removes a third
of the layer are withdrawn. C2 computes a mask but discards it and uses
`fixed_rows`; C uses the actual mask. The difference changes workloads as well
as synchronization. The saved JSON/logs do not include C2 timings, so even the
reported C2 difference cannot be independently replayed from the supplied records.
Those numbers remain in the original report solely as withdrawn claims.

Dynamic-size `torch.where` is a hypothesis for future profiling. A controlled
comparison must reuse identical routing rows and weights, verify equivalent
outputs, and commit its timing/validity rule before any new arms. No removable
fraction of a real layer has been established here.

### Modeled normalization, not a dense-layer measurement

The arithmetic estimate is 51.5 GFLOP / 23.48 TFLOPS = about 2.20 ms. The
23.48 TFLOPS comes from a gate-up GEMM at 512 rows in the layer session; down
projection has different throughput. There is no timed dense layer denominator.

| observation or model | ratio | interpretation |
|---|---|---|
| isolated gate-up GEMM at M=64 | 0.593 | measured same-shape GEMM ratio |
| modeled 2.20 ms / A time | 0.450 | estimated normalization of the synthetic loop |
| modeled 2.20 ms / D time | 0.148 | estimated normalization including routing/NF4 |

The corresponding 6.8x is also a modeled ratio. The earlier assertion that the
whole-layer assumption is four times too optimistic is withdrawn. D3 may use
the GEMM curves as descriptive inputs and the arm times as profiling leads;
none of these modeled ratios replaces a measured giant-training factor.

## Protocol deviations and validity limits (2026-10-15)

The exact rule is archived above; its numerical gates have not been changed.
Execution differs from it:

- Original point protocol: three warmups, 20 timed iterations, three repeats.
  Executed GEMM script: five warmups, 50–20000 iterations from a 2e12 FLOP budget,
  three repeats retaining the best, plus a 60-second settling phase.
- M=8192 observations are present; the original report omitted that normalization.
  The CPU replay now records both denominators in `replay-summary.json`.
- The originally specified equal-arithmetic A/B/C controls are absent from the
  submitted runs. The later A/B/C/D/C2 extension is a different experiment.
- GEMM points use 512 → M → M → 512 inside a process. Layer arms use fixed
  A/B/C/D/C2 order; this is not interleaving in fresh processes.
- GPU-peer, suspend and foreign-reader conditions are not independently recorded
  per arm. Rented hosts make laptop AC checks inapplicable; other unrecorded
  conditions remain unknown. CSV clock samples are useful but do not prove all
  validity conditions. Zero in JSON `clock_mhz` denotes an unavailable reading.

Result: **descriptive observations / no formal verdict**. Missing controls
cannot be reconstructed into a validity pass. No void arm files were supplied;
this does not prove that no void events occurred. Backward, attention, full
training, real weights/text and routing-weighted equivalence were not measured.

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
in its own session: 0.593 against 0.601 on the first card, 1.013x apart. The modeled layer normalization uses its own session; the GEMM curve is
reported separately. The cross-card 1.013x agreement is descriptive only.

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
170 W, torch 2.11.0+cu128, bfloat16. The author reported no other GPU process; per-arm evidence is missing.
Curve: Vast.ai instance 55384207, machine 47491, Ryzen 5 3500, 32 GB RAM.
Layer: instance 55531564, machine 47487, same host, bitsandbytes 0.50.2.

Raw: `benchmarks/results/probe-f4-rtx3060/` — run_0..2.txt, f4_round_0..2.json,
gpu_log.csv, gemm_final.py, item4_0..2.txt, f4_item4_round_0..3.json,
gpu_log_item4.csv, moe_breakdown.py.


## Source preservation and replay

`executed-source/*.py.txt` preserve the exact submitted script bytes. The `.py`
files are packaging copies with identifier-only lint fixes; they were not rerun
on a GPU. `provenance.json` pins both versions and the original rule. These are
hashes of submitted sources, not retrospective proof of remote execution bytes.
The scripts retain their historical print labels; interpret them using this
corrected report, not as an endorsed causal analysis.

Run `python benchmarks/harness/replay_f4_small_gemm.py` on CPU to check hashes,
recompute both GEMM normalizations and observed layer medians. The JSON output
is a replay of saved evidence, not a new performance measurement.
