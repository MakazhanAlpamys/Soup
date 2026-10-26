<!--
Measurement record, published as measured.

Hardware: RTX 5070 Laptop GPU 8 GB (Blackwell sm_120) — a NEW baseline, never compared
with RTX 3050 numbers. Intel i9-14900HX, Windows 11, AC power.
Stack: torch 2.14.0+cu130, bitsandbytes 0.50.2, transformers 5.17.0.
Model: unsloth/mistral-7b-instruct-v0.3, NF4 with double quantisation, layer-streamed,
RAM tier unless a row says otherwise; one synthetic Qwen2.5-14B row on the disk tier.
Soup: the measurements in sections 1-4 ran on the local experiment branch
exp/1c-nf4-dequant (base 1cff0ae45, 2026-10-21/22); section 5 on the feature as ported.
Harness: benchmarks/harness/stream_recompute_exactness.py (section 3) and
benchmarks/harness/stream_recompute_ab.py (section 4), ported here from the experiment
branch's exp1c/exact_e2e.py and exp1c/ab_shapes.py; the micro-benchmarks behind sections 1
and 2 (micro1.py, micro2_zero.py, micro3b_recompute.py) and the e2e A/B scripts behind the
section 3 speed table (ab_e2e.py, ab_e2e_sc.py, ab_e2e_tail.py) stay on that branch.
-->

# Gate record — `training.stream_recompute`: streamed layers without the checkpoint recompute

**Status: MEASURED on one laptop (RTX 5070 Laptop 8 GB).** On the RAM tier, skipping the
checkpoint recompute takes **22 % / 17 % / 8.5 %** off a Mistral-7B NF4 step at 512 /
1024 / 2048 tokens per step, as far as the activations fit (all 32 layers at 512, 24 at
1024, 12 at 2048). Loss and all 256 LoRA gradients are **bit-identical** to the recompute
under deterministic algorithms. On a read-bound disk tier there is **no gain**, and at a
high VRAM fill it is slower. `soup train` with `stream_recompute: auto` chose 32/32 layers
at max_length 512 and 10/32 at 2048, with the same train_loss as `on` per shape.

Unit convention: **decimal GB**, matching the other records.

---

## 1. Where the dequantisation ceiling is (micro, synthetic Mistral-7B shapes, M = 512)

| | per decoder layer |
|---|---|
| copy roofline (device) | 333 GB/s (fill 357 GB/s) |
| bnb `dequantize_4bit`, nested scales | 2.29 ms |
| same, expanded fp32 scales (#1892) | 2.00 ms (gate/up/down at 290-294 GB/s = ~88 % of copy roofline) |
| 7 forward GEMMs (cuBLAS, bf16) | 5.39 ms (40-43 TFLOPS) |
| dequant on a side stream overlapped with the GEMMs | no gain (6.90 vs 6.65 ms serial) |

- bitsandbytes' dequantisation is already near the bandwidth roofline: a faster dequant
  kernel cannot buy much.
- 64 visits x 2.29 ms = 147 ms, about 14 % of a 1.05 s step.
- An earlier "-23 % without dequantisation" figure multiplied by an ALL-ZERO cached weight.
  On this power-limited card a zero-weight GEMM runs 7.6 % faster than a real one (5.01 vs
  5.42 ms per layer, 4 interleaved rounds, consistent), so part of that figure was a
  power/clock artefact; the dequantisation's real share is about 14-18 %.

## 2. The bigger lever: the recompute

A streamed layer runs its forward twice: in the forward, and again as the checkpoint
recompute in the backward. When the activations fit in VRAM, the recompute can be dropped.
Pool-slot tensors must still never be saved as live views (#331): the layer runs under
`torch.autograd.graph.saved_tensors_hooks`, saves pool tensors as tokens, and at its first
backward unpack does the same `pool.wait` + `prefetcher.advance` the recompute did. NF4
linears save only the packed bytes (`packed_dequant_nf4_linear`, `save_for_backward`) and
dequantise again for `grad_x`, so no dense weight is kept.

One layer, 1 x 512: checkpoint 23.1 ms; no recompute with the packed save 18.6 ms (-19.5 %,
keeps 76 MB per layer); no recompute with a dense save 16.9 ms (-26.5 %, keeps 492 MB per
layer). All three bit-identical under deterministic mode and the MATH sdpa backend. The
packed save is what ships: the dense one costs 6.5x the VRAM per layer.

## 3. Real streamed Mistral-7B NF4, RAM tier, 1 x 512

**Exactness** (`stream_recompute_exactness.py`: deterministic algorithms, MATH sdpa, LoRA B
given non-zero values so every gradient is non-zero): ON vs ON, OFF vs ON and OFF vs OFF —
loss equal, **0 of 256 LoRA gradients differ**; the negative control (one input token
changed) differs in 256 of 256. A first run reported "equal" vacuously because the switch
found 0 layers (the wrapper is transparent to `named_modules()`); the harness now refuses
unless every layer was switched.

**Speed** (default sdpa, forward + backward + PagedAdamW8bit, arms interleaved in ONE
process, 3 warm-up + 8 timed steps per block, 2 rounds):

| arm | step | tok/s | vs recompute | peak VRAM |
|---|---|---|---|---|
| recompute (`on`) | 0.764-0.770 s | 665-670 | — | 1.34 GB |
| `on` + `stream_nf4_scale_cache` | 0.761 s | 673 | -1.1 % | — |
| no recompute (`off`) | 0.608-0.610 s | 839-842 | **-20.5 / -21.0 %** | 4.20 GB |
| `off` + scale cache | 0.598-0.600 s | 853-856 | **-21.8 / -22.1 %** | 4.65 GB |
| `off` + dense weight kept for the last 2 / 4 layers | 0.594 / 0.597 s | 862 / 858 | -22.3 / -21.9 % (noise) | 5.5 / 6.4 GB |

Round-to-round spread within an arm: under 0.5 %. The dense-weight option measured no gain
over the packed save and is not part of the feature.

## 4. Wider shapes, the disk tier, a model that does not fit

`stream_recompute_ab.py`: one process per model and tier, scale cache on, arms ON / OFF
interleaved, 2 rounds. OFF skips the recompute only for the LAST K layers
(`set_stream_recompute(..., retain=K)`), K the largest that keeps the allocated peak under a
cap. One layer's retained bytes are what stays allocated after the forward (OFF with K = 2
minus ON). An earlier version used the difference of PEAKS, which under-estimated a layer
3x at 1024 tokens, picked K = 32, overflowed the card (8.26 GB) and WDDM spilled into host
RAM (OFF 2.3x slower); those two points are discarded.

| model, tier | tokens/step | retained bytes/layer | K of L | ON step | OFF step | gain |
|---|---|---|---|---|---|---|
| Mistral-7B, RAM | 1x512 (section 3) | 0.10 GB | 32/32 | 0.770 s | 0.600 s | 22 % |
| Mistral-7B, RAM | 1x1024 | 0.20 GB | 24/32 | 1.405 s | 1.167 s | 16.9 % |
| Mistral-7B, RAM | 2x512 | 0.20 GB | 24/32 | 1.382 s | 1.139 s | 17.6 % |
| Mistral-7B, RAM | 1x2048 | 0.34 GB* | 12/32 | 3.455 s | 3.163 s | 8.5 % |
| Mistral-7B, RAM | 4x512 | 0.34 GB* | 12/32 | 3.158 s | 2.890 s | 8.5 % |
| Mistral-7B, disk (warm page cache) | 1x512 | 0.10 GB | 32/32 | 1.649 s | 1.503 s | 8.9 % |
| Mistral-7B, disk (warm page cache) | 1x1024 | 0.20 GB | 24/32 | 1.702 s | 1.550 s | 9.0 % |
| synthetic Qwen2.5-14B, disk | 1x512 | 0.11 GB | 24/48 (cap 7 GB) | 3.73 s | 4.37 s | -17 % |
| synthetic Qwen2.5-14B, disk | 1x512 | 0.11 GB | 6/48 (cap 5 GB) | 5.22 s | 5.32 s | -1.8 % (noise) |

\* peak-difference estimate (the discarded method), but K = 12 kept the peak at 7.4 GB, no
spill.

- Compute-bound (RAM tier): the gain is about 22 % x K/L, so it shrinks with tokens per step
  as fewer layers fit.
- Read-bound (the 14B, whose 9.3 GB store does not stay in the page cache): no gain — every
  layer is still read twice per step, since the backward visit re-reads it — and at a high
  VRAM fill it hurts. Disk-tier ON times varied 3.7 vs 5.2 s between runs (other processes
  on C:), so the disk numbers are rough.
- What follows for the feature: `auto` engages only on the RAM tier and budgets K from
  measured retained bytes and the RESERVED peak, with headroom below the card size; `off`
  on the disk tier warns.

## 5. The feature as shipped: `soup train` smoke

`stream_recompute: auto` decides at setup, before step 1: one synthetic forward+backward at
the configured `batch_size` x `data.max_length` (the worst case any batch can pad up to)
with the last layer retained measures that layer's allocated bytes and the step's reserved
peak. A layer is charged at its bytes times the step's reserved/allocated ratio when that
is above 1; the budget is the smallest of 85 % of the card, free + reserved VRAM and
`stream_vram_override`, minus that peak; K = budget // charge. This differs from section
4's method (which capped the allocated peak); the GPU check below is of the shipped planner.

`soup train`, 24 rows, Mistral-7B NF4 streamed, RAM tier:

| | max_length 512 | max_length 2048 |
|---|---|---|
| `auto` K | 32/32 | 10/32 |
| charged per layer | 99.4 MB | 422.8 MB |
| budget | 5.83 GB | 4.47 GB |
| train_loss `auto` vs `on` | equal | equal |

Train runtime: `on` 27.3 s (a first, cold run read 38.3 s), `auto` 21.0 s, `off` 21.2 s.
These three agree with max_length 512, where `auto` chose all 32 layers, the same as `off`.

## 6. Limits

- One laptop, one model family measured end to end (Mistral-7B NF4); task `sft` only, which
  is what the config allows.
- K errs low by up to one layer: the synthetic step's peak already holds the measured
  layer.
- `auto` costs one extra step at setup.
- The CPU suite (`tests/test_stream_recompute.py`) proves the bit-exactness and the #331
  negative controls on a tiny 4-layer / 2-buffer model; speed and VRAM are measured only
  here.
