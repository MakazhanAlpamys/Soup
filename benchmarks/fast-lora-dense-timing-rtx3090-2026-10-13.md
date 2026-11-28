# Gate record - isolated dense Fast-LoRA projection timing

**Verdict (2026-10-13) — Mixed.** QKV and MLP improved with input gradients. The other measured cases had lower PEFT-to-Fast-LoRA ratios than 1.

Measured by @kok-o.

## 1. Setup

Does Fast-LoRA reduce synchronized forward and backward time for isolated dense projections compared with PEFT?

- NVIDIA GeForce RTX 3090, 24 GB, sm86, driver 595.71.05. Linux, Python 3.12.3, torch 2.11.0+cu126, PEFT 0.20.0, Transformers 5.17.0, bitsandbytes 0.50.1 and Triton 3.6.0. The report records native BF16 CUDA support and leaves the Linux release and host RAM unspecified.
- Synthetic Llama-3.1-8B projection shapes with hidden size 4096, KV size 1024, intermediate size 14336, rank 16, batch 1, 128 tokens and seed 792. Base and adapters both use the selected FP16 or BF16 dtype. The three fixtures exercise a single projection, QKV or SwiGLU MLP, with and without an input gradient.

## 2. Method and results

Each case ran five ABBA cycles in fresh GPU processes with identical frozen tensors. Each arm warmed up for at least 20 iterations and two seconds, then ran at least 100 synchronized forward/backward event pairs and one second. A CPU-only NVML supervisor checked CUDA context ownership, clocks, power and throttling. All 60 cycles and 240 arms passed, with 42,542 recorded sample pairs, timed SM clocks of 1740 MHz and zero within-cycle clock spread.

Times are medians of cycle implementation medians. The total column is the median of the paired PEFT/Fast-LoRA ratios, where a ratio above 1 favors Fast-LoRA. Each memory reading is peak allocated memory for a single arm after warmup.

| Dtype | Projection | Input gradient | Forward ms PEFT / Fast-LoRA | Backward ms PEFT / Fast-LoRA | Total ratio | Cycle range | Allocated MiB PEFT / Fast-LoRA |
|---|---|---|---:|---:|---:|---:|---:|
| BF16 | MLP | No | 3.3823 / 3.1734 | 4.0970 / 4.3754 | 0.991x | 0.988–0.993x | 426.3 / 437.6 |
| BF16 | MLP | Yes | 3.4121 / 3.1846 | 5.2429 / 5.1343 | 1.040x | 1.039–1.043x | 426.3 / 438.6 |
| BF16 | QKV | No | 3.2512 / 3.8216 | 3.3987 / 3.1903 | 0.948x | 0.947–0.951x | 118.1 / 118.9 |
| BF16 | QKV | Yes | 3.2903 / 3.8288 | 5.1333 / 4.2691 | 1.039x | 1.035–1.042x | 119.7 / 121.5 |
| BF16 | Single | No | 1.1899 / 1.2324 | 1.4776 / 1.5711 | 0.950x | 0.943–0.954x | 101.3 / 101.3 |
| BF16 | Single | Yes | 1.2073 / 1.2329 | 2.0582 / 2.0695 | 0.988x | 0.985–0.991x | 102.5 / 102.5 |
| FP16 | MLP | No | 3.4018 / 3.1867 | 4.0612 / 4.3540 | 0.989x | 0.988–0.990x | 426.3 / 437.6 |
| FP16 | MLP | Yes | 3.4304 / 3.1846 | 5.1927 / 5.1121 | 1.040x | 1.037–1.040x | 426.3 / 438.6 |
| FP16 | QKV | No | 3.2752 / 3.8380 | 3.4181 / 3.2205 | 0.947x | 0.946–0.954x | 118.1 / 118.9 |
| FP16 | QKV | Yes | 3.3179 / 3.8380 | 5.1236 / 4.3204 | 1.035x | 1.034–1.036x | 119.7 / 121.5 |
| FP16 | Single | No | 1.1980 / 1.2339 | 1.4725 / 1.5759 | 0.950x | 0.884–0.954x | 101.3 / 101.3 |
| FP16 | Single | Yes | 1.2114 / 1.2380 | 2.0438 / 2.0675 | 0.983x | 0.977–0.986x | 102.5 / 102.5 |

All 12 pre-timing gates passed. Independent CPU replay of final outputs and requested gradients checked 1,640 comparisons, 1,600 exact and 40 within tolerance, with zero failures.

## 3. Commands

The harness lives at the evidence commit. Its recorded GPU launcher is `benchmarks/harness/d2_prefork_timing.py`, with workers in `benchmarks/harness/d2_isolated_timing.py`. The handoff contains the original launch receipts. Reassemble and extract the 50-part evidence ZIP before running the recorded replay commands.

```bash
python benchmarks/harness/d2_materialize_timing.py /absolute/path/extracted
python benchmarks/harness/d2_audit_timing.py /absolute/path/extracted/run3-timing-v5 --output replay.json
```

## 4. Superseded runs and limits

The accepted fresh-process run superseded the short-warmup and interrupted slow-import pilots and the CPU template that initialized CUDA before forking. An earlier host failed to start the CUDA workload. The previous same-process timings and peer-guard failures retain their original no-verdict status.

Dropout was zero and optimization was outside the benchmark. Only the final output and gradients of each timed arm were checked. The allocation figures exclude driver overhead, and the fixtures do not measure full-model training, NF4 timing or task quality.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/b2923981b7cb938852999459b19c968e5fbeb37f/benchmarks/evidence/d2-rtx3090-20261013
