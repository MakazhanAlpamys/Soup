# Bounded-RAM disk streaming with Qwen3-32B NF4

Measured by @kok-o on 2026-10-24.

## 1. Question and setup

Can a 32B NF4 model train on a physical 8 GB card with at most 32 GiB host RAM,
and which weight placement improves matched optimizer steps?

The Linux container ran on a physical RTX 4060 Ti (8188 MiB reported).
Physical host RAM was 30.9970 GiB, with an enforced native cgroup bound of
29.7559 GiB. The workload audit observed zero container swap. The frozen stack
used Python 3.12, Torch 2.8.0+cu128, Transformers 5.17.0, PEFT 0.20.0,
TRL 0.29.1, Accelerate 1.14.0 and bitsandbytes 0.50.2. The accessible setup and
hardware summaries leave the driver and Linux release unspecified. The backing
disk appeared inside the container as a loop device.

The model was `Qwen/Qwen3-32B` at `9216db5781bf21249d130ec9da846c4624c16137`,
with fixed double-quantized NF4 weights, BF16 decoder compute and identical nonzero native LoRA
initialization across arms. Rank/alpha was 4 on seven decoder projections, with
batch one, accumulation two and context 512. The frozen harness used deterministic
arithmetic and controlled answer-only batches.

## 2. Method

The disk-streaming control was compared with a 14336 MiB RAM hotset, seven
persistent GPU layers, and both. The combined configuration held 56 decoder
layers in RAM and seven different layers on the GPU, so 63/64 avoided recurring
disk-source reads. The vocabulary matrices still streamed. Eight captures
qualified before timing and matched bitwise under the unchanged reference-first
`rtol=.01`, `atol=1e-5`, finite-value and three-decimal loss gates.

Four balanced blocks ran in 16 fresh processes with two warmup and four measured
updates each. All 96 updates, final adapters, losses, minibatch identities and
memory-scope receipts were retained or audited. The practical reduction gate
was at least 2%. Whole-block bootstrap intervals describe the sampled blocks.

## 3. Results

| Metric | Ordinary disk | RAM cache | GPU cache | Both |
|---|---:|---:|---:|---:|
| Optimizer step, s | 9.2608 | 8.6317 | 9.1622 | 8.5955 |
| Native setup/preload, s | 21.3162 | 28.4580 | 35.6367 | 39.8500 |
| Whole fresh six-update entry, s | 82.5211 | 84.8784 | 93.7806 | 94.1461 |
| Peak allocated VRAM, GiB | 3.6444 | 3.6444 | 5.4083 | 5.4083 |
| Peak reserved VRAM, GiB | 3.8184 | 3.8184 | 5.7812 | 5.7812 |
| Peak process RSS, GiB | 5.8270 | 19.8284 | 5.9035 | 19.8955 |

| Comparison | Paired step reduction | 95% interval | Practical gate |
|---|---:|---|---|
| Disk → RAM | +6.794% | [+6.476%, +8.102%] | Pass |
| Disk → GPU | +1.085% | [+0.710%, +2.467%] | Fail |
| Disk → both | +7.153% | [+6.827%, +8.593%] | Pass |
| RAM → both | +0.386% | [+0.375%, +0.534%] | Fail |

**Verdict — mixed.** The 32B workload fit and RAM caching passed the step-time gate. GPU residency fell short as an incremental gain, and every cache arm took longer for the full six-update entry.

## 4. Commands

The executed setup and worker live in the numbered review archive at the evidence commit.
Reassemble and check the parts against its evidence manifest. The frozen setup runs
the following worker from the Soup checkout. It delegates the four-arm study to the
archived joint worker and benchmark.

```bash
/root/liger/venv/bin/python /root/liger/pin_entry.py /root/liger/study_worker.py
```

## 5. Limits and run history

An earlier full-RAM policy refusal was a placement admission result. A promotion
ownership error and then a CUDA fixture precondition stopped earlier disk attempts.
The completed four-arm retry supplies the measured comparison. Physical NVMe
backing remains unverified. Step timing excludes setup and cannot establish
time to convergence. The fixed cache size and deterministic settings leave other
placements, production-setting performance and task quality unmeasured.
The measured overlay predates pending `training.stream_ram_cache_mb` and
`training.stream_resident_layers`.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/96df4325d422397384d21bb21d492843c67fb84e/benchmarks/nf4-ssd-32b-20261024
