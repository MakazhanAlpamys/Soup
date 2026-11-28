# Measurement record - Frozen-prefix caching and top-layer LoRA

**Measured 2026-10-15 (OLMoE) and 2026-11-04 to 2026-11-09 (Mistral).**

## 1. Questions

Can a frozen-prefix cache reproduce uncached training with LoRA on the same
top layers? How much validation loss comes from freezing the bottom layers
from the first step? Can more adapter capacity or a full first epoch of LoRA
close that gap? How long does the hybrid schedule with a cache take relative
to all-layer LoRA, including the epoch that fills the cache?

## 2. Setup

| Series | Model and dtype | Measurement host |
|---|---|---|
| OLMoE fixed top layers | `allenai/OLMoE-1B-7B-0924` at `6d84c48581ece794365f2b8e9cfb043c68ade9c5`, FP16 base and autocast, FP32 LoRA, GradScaler | The study's setup lists an RTX 2080 Ti with 22 GiB VRAM. The retained full-matrix summary omits per-run host and software inventories. |
| Mistral recipe and hybrid runs | `mistralai/Mistral-7B-v0.3` at `caa1feb0e54d415e2df31207e5f4e273e33509b1`, FP16 base without quantization, split across both GPUs | Shared Linux notebook VMs, two NVIDIA Tesla T4 GPUs, 15360 MiB per GPU, driver 580.178.04. The recorded stack includes Python 3.13, CUDA 13.0 build, torch 2.11.0+cu130, Transformers 5.20.2, PEFT 0.21.2, tokenizers 0.23.3, Accelerate 1.15.0 and Soup CLI 0.75.0. |

CPU models and host RAM are unrecorded in the retained summaries and run
metadata. The notebook records do not give a Linux distribution release.
The OLMoE smoke records have software pins of their own. The full OLMoE
matrix lacks a driver, CUDA runtime, torch and library inventory.

## 3. OLMoE with fixed top layers

The October full matrix used 2,000 Alpaca training rows, 200 validation rows,
ChatML and sequence length 128. Runs used seed 42, deterministic CUDA, two
epochs (4,000 updates), batch 1, AdamW at 1e-4 with linear decay, no warm-up and
zero weight decay. LoRA used rank 4, alpha 8, on `q_proj` and `v_proj` in the
top K of 16 layers. The study paired cached and uncached runs at each K
below 16 and used an all-layer, uncached run as the quality reference.

The full-matrix summary and the three matching parity reports provide the
following values.

| Trainable top layers | Validation task loss | Increase vs all layers (%) | Same-K cache parity |
|---:|---:|---:|---|
| 4 | 1.1888684494256974 | 17.4518390612308 | Exact |
| 8 | 1.0687360456705093 | 5.5836027069822824 | Exact |
| 12 | 1.0242358920812606 | 1.1873005929096125 | Exact |
| 16 | 1.012217823857069 | 0.0 | Uncached reference |

Each pair recorded a maximum absolute training- and validation-loss
difference of 0.0, identical initial and final adapters and identical
adapter files. Each cached run logged 2,000 misses, 2,000 hits and one full
frozen-prefix fingerprint recomputation.

This full matrix supersedes the eight-row quality smoke for task loss.
The earlier synthetic-token speed runs remain exploratory. There was no
approved quality or safety margin for this matrix, and independent
equivalence to the Hugging Face Trainer sampler was unverified. The
follow-up protocol for hybrid OLMoE with a cache specifies top 4 of 16,
two T4s and router load-balancing loss disabled for all arms. It has no
completed result at the evidence commit.

## 4. Mistral recipe check and hybrid quality

Mistral used 2,000 Alpaca training rows and 200 validation rows, sequence
length 128, batch 1 with accumulation 1, linear learning rate 1e-4, no
warm-up, zero weight decay and dropout, and deterministic algorithms.
Both quality checks used two epochs (4,000 updates), cache off, and
eight bundled gate suites with 274 greedy outputs per gated run, a 256
new-token cap and scorer revision 5. The analysis re-scored the saved
outputs.

The recipe check used seed 42. Each recipe compared all-layer LoRA to
training the top 24 layers from step 1. The wider recipe used rank 16,
alpha 32, on `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
`up_proj` and `down_proj`. The smaller recipe used rank 4, alpha 8, on
`q_proj` and `v_proj`. The rule called the gap persistent if the wider
recipe retained at least 75% of the smaller recipe's CE gap. Closing
the gap required at most +0.5% CE, every general-suite delta at least
-0.05 and over-refusal at least 0.95.

| LoRA recipe | All-layer validation CE | Top-24 validation CE | Top-24 CE gap | MMLU-mini delta | Common-sense delta | Instruction delta | Arithmetic delta | Tool-call delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Rank 4, query/value | 0.8758 | 0.8893 | +1.54% | -0.23 | -0.17 | -0.08 | -0.03 | -0.08 |
| Rank 16, seven projections | 0.9828 | 1.0055 | +2.31% | -0.08 | 0.00 | -0.04 | -0.03 | -0.13 |

The recipe report found the CE gap persisted. The wider recipe had
higher validation CE even in the all-layer control. Two initial starts
failed in adapter setup and were restarted after removal of the
incompatible preinstalled torchao 0.10.0. The optional wider top-16
run was stopped before completion.

The hybrid trained all 32 layers for epoch 1, then froze the bottom
layers at step 2,000. It used the rank-4 recipe. Frozen tensors had
changed in epoch 1 and then remained bit-identical. The reference kept
all layers trainable for both epochs. The three-seed report paired
seeds 42, 901 and 1234 with the same-seed controls and held the
validation rows fixed. An all-layer anchor through the extended
runner reproduced seed 42's validation CE bit for bit.

| Trainable top layers after freeze | Mean CE gap | CE gap by seed, % (42 / 901 / 1234) | Mean MMLU-mini delta | Mean tool-call delta | Decision |
|---:|---:|---|---:|---:|---|
| 24 | -0.09% | -0.21 / -0.08 / +0.02 | +0.01 | +0.05 | Within margin |
| 16 | -0.05% | -0.22 / +0.01 / +0.08 | -0.01 | +0.03 | Within margin |
| 8 | +0.00% | -0.21 / +0.03 / +0.18 | -0.04 | +0.03 | Within margin |

The hybrid rule required mean CE gap at most +0.5%, each seed's gap
at most +1.0%, mean delta at least -0.05 on each of the six general
suites and over-refusal at least 0.95 on every seed. The safety suite
was scored outside the rule. The three-seed result supersedes the
one-seed quality decision, which passed top 24 and missed the
MMLU-mini margin for top 16. It retains those original seed-42
measurements and selects top 8 for the speed check.

## 5. Mistral hybrid cache parity and timing

The timing study used seed 42 and the same rank-4 recipe extended to
three epochs (6,000 updates). It ran 13 fresh processes in one
session. An uncached hybrid ran first for exactness. Three cycles
then alternated all-layer LoRA and the cached top-8 hybrid in
control, candidate, candidate, control order.

The hybrid froze layers 0–23 after step 2,000. Cache installation and
fingerprinting were inside the first timed step of epoch 2, with
disk writes in epoch 2 and warm reads in epoch 3. Each cached run
recorded 2,000 misses and zero hits in epoch 2, followed by 2,000
hits and zero misses in epoch 3. All six cached hybrids matched the
uncached hybrid's final LoRA fingerprint, training-batch hash and
validation CE (0.8775755167007446) bit for bit. The six all-layer
controls matched one another (validation CE 0.8966562151908875).

The speed report divides the mean all-layer time by the mean
cached-hybrid time within each cycle. The primary measure sums step
times in epoch 3. The secondary whole-training measure sums all
three epochs, including cache setup and writes. Each measure
required a median ratio of at least 1.10 and every cycle at least
1.00.

| Cycle | Warm epoch ratio | Whole-training ratio | Cache-fill epoch ratio |
|---|---:|---:|---:|
| 1 | 3.20x | 1.46x | 1.42x |
| 2 | 3.27x | 1.52x | 1.49x |
| 3 | 3.09x | 1.46x | 1.41x |
| Median | 3.20x | 1.46x | 1.42x |

The median full subprocess ratio, including model load, was 1.44x.
The retained run supersedes the 2026-11-08 attempt, whose missing
archive left only process times. Its results were excluded. Earlier
one-cycle cache diagnostics and two-epoch preflights did not enter
this decision.

**Verdict — Mixed.** Same-K cache parity held on OLMoE, with higher validation loss at every smaller K. On Mistral, widening LoRA left the fixed top-layer gap, while the hybrid met its measured quality margins and the cached top-8 hybrid passed both timing rules.

## 6. Reproduction and limits

The measurement harnesses and notebooks live at the evidence commit.
The first command pair shows the OLMoE run and comparison for one
cached pair, with `$out` set to a fresh output directory. Repeat for
K of 8 and 12 and run the uncached reference at K of 16.

```bash
root=benchmarks/e2
scripts=$root/moe/olmoe_research/scripts
for mode in baseline cached; do
  python "$scripts/olmoe_pretrained_quality_v2.py" --k 4 --mode "$mode" \
    --train-rows 2000 --eval-rows 200 --epochs 2 --max-length 128 \
    --probe "$scripts/olmoe_disk_cache_research_v1.py" \
    --guard-module "$scripts/olmoe_guarded_cache_v2.py" --output-dir "$out/$mode"
done
python "$scripts/olmoe_compare_quality_v2.py" --baseline "$out/baseline/report.json" \
  --cached "$out/cached/report.json" --output "$out/parity.json"
```

The Mistral notebooks carry the complete arm order and configuration.
The first dispatches the recipe training and gate runners. The later
notebooks use the hybrid runner, with the cache directory supplied
only for the cached timing arms.

```text
benchmarks/e2/recipe_check/e2_recipe_check.ipynb
benchmarks/e2/hybrid/e2_hybrid.ipynb
benchmarks/e2/hybrid_seeds/e2_hybrid_seeds.ipynb
benchmarks/e2/hybrid_cache/e2_hybrid_cache.ipynb
```

The published tables and parity summaries are retained at these paths.

```text
benchmarks/e2/moe/olmoe_research/reports/olmoe_v2_full_matrix_summary.json
benchmarks/e2/moe/olmoe_research/reports/olmoe_v2_full_k{4,8,12}_parity.json
benchmarks/e2/recipe_check/reports/20261104/REPORT.md
benchmarks/e2/hybrid_seeds/reports/20261106/REPORT.md
benchmarks/e2/hybrid_cache/reports/20261109/REPORT.md
```

The two Mistral quality decisions use small gate suites, one dataset
and one model. The recipe check and timing study use a single seed
each. The cached speed comparison combines the freeze schedule and
cache against all-layer LoRA. The single uncached hybrid provides a
parity control and leaves the isolated cache speed effect unmeasured
by the balanced timing design. The speed study uses a shared host and
its local disk. GPU state was recorded before and after each process.
The measured timing ratios apply to this Mistral setup.

Measured by @nurseit14

Evidence: benchmarks/evidence/frozen-prefix-cache-t4-2026-11-09/ (source commit ee4f4bed3d29)
