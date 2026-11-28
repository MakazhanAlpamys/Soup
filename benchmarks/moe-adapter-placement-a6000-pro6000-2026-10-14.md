# Measurement record - LoRA placement on MoE models

**Measured 2026-10-11 and confirmed 2026-10-13 to 2026-10-14.**

## 1. Question

How much task quality and general behavior does LoRA lose when it skips
routed experts? Does adapting the busiest quarter recover it at the same
rank, on the same data and training schedule?

## 2. Setup

Both models ran resident with BF16 base weights and compute, FP32 adapter
weights, eager attention and the `grouped_mm` expert path.

- `allenai/OLMoE-1B-7B-0125-Instruct` at
  `b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e` on NVIDIA RTX A6000 48 GB
  (compute capability 8.6). The original hosts logged 12 logical CPUs and
  67,522,531,328 / 67,522,543,616 bytes of host RAM.
- `Qwen/Qwen3.5-35B-A3B` at
  `59d61f3ce65a6d9863b86d2e96597125219dc754` (text tower) on NVIDIA RTX PRO
  6000 Blackwell Server Edition 96 GB (compute capability 12.0). The original
  hosts logged 16 logical CPUs and 151,876,038,656 / 152,077,185,024 bytes of
  host RAM. All counted Qwen runs used
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Original stack: Linux kernel 6.8.0-90-generic, glibc 2.35, Python 3.12.15,
  CUDA 13.0, torch 2.14.1+cu130, Transformers 5.19.0, huggingface-hub 1.33.0,
  safetensors 0.8.0, tokenizers 0.23.2, Jinja2 3.1.6, Rich 15.0.0,
  psutil 7.2.2 and NumPy 2.5.3. The harness supplied its own LoRA modules.
- Confirmation used the same resident CUDA and library pins with Python
  3.12.3, glibc 2.39 and Linux kernels 6.8.0-138-generic,
  6.8.0-142-generic and 7.0.0-1012-aws. The RTX A6000 hosts logged 28 or
  48 logical CPUs and 304,005,406,720 or 405,668,175,872 bytes of host RAM. Qwen used
  Workstation and Server editions of the RTX PRO 6000. Two hosts
  logged 128 logical CPUs and 540,847,951,872 or 607,927,001,088 bytes
  of host RAM. The third logged 96 logical CPUs and 1,073,242,869,760 bytes.
  The retained host metadata give CPU counts. CPU models, Linux distribution
  releases and NVIDIA driver versions are unrecorded.

## 3. Method

- The pinned `meta-math/MetaMathQA` snapshot
  `aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18` supplied 512 calibration,
  4,096 training and 256 tuning questions in seed-1234 shuffle order. The
  split admits one row per source question. Both tokenizers had to fit the
  rendered prompt and response in 512 tokens. Training uses response-only
  labels and disables thinking for Qwen. Confirmation adds 256 unused
  source questions from the same shuffle, with near-duplicates excluded.
- The attention placement covers the four Q/K/V/O projections on OLMoE and
  Qwen's full-attention layers, and all five linear-attention projections on
  Qwen. The shared-expert placement adds the shared gate, up and down
  projections. The two routed placements also adapt either the busiest
  quarter or all of the routed experts, plus the shared expert where
  present. Calibration counted top-k assignments per layer and broke ties
  by lower expert index. The selected quarters were 16 of 64 experts
  for OLMoE and 64 of 256 for Qwen. Routers, the shared-expert gate, norms,
  embeddings and output head stayed frozen.
- LoRA used rank 16, alpha 32, dropout 0, Kaiming-uniform input factors and
  zero output factors. Routed adapters targeted the fused gate/up and down
  projections. AdamW used betas 0.9 and 0.999, epsilon 1e-8 and weight decay
  0, with gradient clipping at 1.0 and gradient checkpointing. Each run
  trained for 256 updates at batch 16, with 16 linear learning-rate warm-up
  steps followed by a constant rate. The reported step time is training
  time per step.
- On OLMoE, seed 0 swept 1e-4, 2e-4 and 4e-4, with a one-point extension
  when the best rate was on an edge. Seeds 1 and 2 reran the winning rate.
  Qwen reused the OLMoE rates for seeds 0 and 1, including the attention-only
  rate for the shared-expert placement. The confirmation retrained the same
  configurations from scratch at the frozen rates and repeated the OLMoE sweep.
  Base scores were checked again on other hosts.
- Validity required finite loss, a zero-step adapter within 1e-3 nat/token
  of the base, a settled rate and at least 0.05 nat/token improvement over
  base on every seed of the fullest fitting placement. The task gap against
  that placement was material only above both 0.02 nat/token and twice the
  larger sample standard deviation. The general-behavior gate of `soup ship`
  required a drop of at most 0.05 from the base on each suite and seed,
  using greedy generation for 256 new tokens and scorer revision 4. Task
  accuracy used greedy generation for 384 new tokens and was descriptive.

## 4. Results

The first report supplies tuning-set means, sample standard deviations, and
accuracy differences paired by question across seeds. The confirmation report
supplies the confirmation-set NLL and gap columns. The fully adapted placement
is the reference for gaps on each model. Both evaluations passed validity
checks and gave the same placement verdicts.

| Model | Placement | LR | Tuning NLL (nat/token) | Tuning gap | Confirmation NLL (nat/token) | Confirmation gap | Tuning accuracy difference | Gate |
|---|---|---:|---:|---:|---:|---:|---|---|
| OLMoE | Base | — | 0.9360 | — | 0.9028 | — | — | Reference |
| OLMoE | Attention | 2e-4 | 0.2508 ± 0.0009 | +0.0083 | 0.2396 | +0.0072 | -3.1 pp, p 0.058 | Regresses |
| OLMoE | Attention + busiest quarter | 1e-4 | 0.2442 ± 0.0007 | +0.0016 | 0.2333 | +0.0009 | +0.3 pp, p 0.92 | Regresses |
| OLMoE | Attention + all routed experts | 1e-4 | 0.2426 ± 0.0009 | 0 | 0.2324 | +0.0000 | — | Regresses |
| Qwen | Base | — | 0.6341 | — | 0.6143 | — | — | Reference |
| Qwen | Attention | 2e-4 | 0.1348 ± 0.0003 | -0.0038 | 0.1275 | -0.0029 | +2.3 pp, p 0.081 | Acceptable |
| Qwen | Attention + shared expert | 2e-4 | 0.1355 ± 0.0002 | -0.0031 | 0.1279 | -0.0024 | +2.1 pp, p 0.14 | Acceptable |
| Qwen | Attention + shared + busiest quarter | 1e-4 | 0.1366 ± 0.0002 | -0.0020 | 0.1286 | -0.0017 | -0.8 pp, p 0.54 | Regresses |
| Qwen | Attention + shared + all routed experts | 1e-4 | 0.1386 ± 0.0003 | 0 | 0.1303 | +0.0000 | — | Regresses |

The original report gives these suite drops from the base. The table shows
every suite that crossed the 0.05 limit on any seed for that placement.

| Model | Placement | Suite | Drop by seed (0 / 1 / 2 where applicable) |
|---|---|---|---|
| OLMoE | Attention | Common sense | 0.083 / 0.083 / 0.042 |
| OLMoE | Attention + busiest quarter | Common sense | 0 / 0.083 / 0 |
| OLMoE | Attention + busiest quarter | Instruction | 0.042 / 0.083 / 0.083 |
| OLMoE | Attention + all routed experts | Common sense | 0.042 / 0.125 / 0 |
| Qwen | Attention + shared + busiest quarter | Safety | 0.075 / 0.025 |
| Qwen | Attention + shared + all routed experts | Safety | 0 / 0.075 |
| Qwen | Attention + shared + all routed experts | Over-refusal | 0.075 / 0.025 |

The first report gives the resident costs below. Peak CUDA is allocated memory,
using the maximum over seeds for each placement.

| Model | Placement | Trainable parameters | FP32 adapter state | Peak CUDA | s/step |
|---|---|---:|---:|---:|---:|
| OLMoE | Attention | 4.2M | 0.06 GiB | 14.25 GiB | 1.49 |
| OLMoE | Attention + busiest quarter | 33.6M | 0.50 GiB | 14.70 GiB | 2.11 |
| OLMoE | Attention + all routed experts | 121.6M | 1.81 GiB | 16.01 GiB | 2.93 |
| Qwen | Attention | 16.3M | 0.24 GiB | 70.87 GiB | 4.03 |
| Qwen | Attention + shared expert | 21.2M | 0.32 GiB | 70.94 GiB | 4.04 |
| Qwen | Attention + shared + busiest quarter | 251.9M | 3.75 GiB | 74.38 GiB | 5.29 |
| Qwen | Attention + shared + all routed experts | 943.9M | 14.07 GiB | 84.69 GiB | 8.46 |

Within each model, peak CUDA above attention-only tracked the additional FP32
adapter weights, gradients and AdamW moments (16 bytes per trainable parameter).
The largest residual was 12.8 MiB using per-placement maxima and 13.2 MiB
when paired by seed.

**Verdict — Mixed.** Attention-only and attention plus the shared expert passed on resident Qwen3.5-35B-A3B. No tested OLMoE placement passed.

## 5. Reproduction and limits

The harness and full run plans live at the evidence commit. The original
record's reproduction sections document the dataset and split preparation,
archived harness bytes, hot-set files, commands for each seed and control,
confirmation and RAM-tier probes, and report regeneration. The following
commands show one original and one confirmation run, with the dataset,
counted split and output directory supplied as `$data`, `$split` and `$out`.

```bash
r=benchmarks/results/probe-moe-adapter-placement
export PYTHONPATH="$PWD/src" TOKENIZERS_PARALLELISM=false
cp "$r/harness-348f6a48.py.txt" benchmarks/harness/e1-harness.py
python benchmarks/harness/e1-harness.py run --model olmoe --arm A --lr 2e-4 --seed 0 \
  --data "$data" --split "$split" --out "$out/run-olmoe-A-lr2e-4-s0.json"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python benchmarks/harness/moe_adapter_confirm.py run --model qwen35 --arm AS \
  --lr 2e-4 --seed 0 --gen-batch 16 --data "$data" --split "$split" \
  --confirm-split "$r/confirm-split.json" --out "$out/confirm-qwen35-AS-lr2e-4-s0.json"
```

- The first OLMoE feasibility pilot used a withdrawn split that allowed
  rephrasings across training and evaluation. It was replaced before counted
  runs. The gate rule was written before the first run and approved after the
  initial results without changes. The confirmation then reproduced all 19
  counted configurations and preserved every verdict and task-gap sign.
- The quality result covers one recipe, with small general-behavior suites.
  Qwen reused the OLMoE learning rates. In a separate seed-0 check, the
  all-expert placement at 5e-5 stayed within the general-behavior drop
  limit, with a largest drop of 0.050 on safety. The counted attention-only
  NLL was 0.0008 nat below that run and attention plus the shared expert
  was 0.0001 nat above it. The default Qwen gate template opens a thinking
  block and the base left 151 of 274 generations inside that block at the
  token limit. Task scoring uses thinking disabled. The step timings are
  indicative because GPU peers and clocks were uncontrolled.
- The separate Qwen RAM-streaming pilot used an NF4 configuration on a
  96 GB RTX PRO 6000 Blackwell Server Edition with 96 logical CPUs and
  1,073,242,869,760 bytes of host RAM. It logged Linux 7.0.0-1012-aws,
  glibc 2.39, Python 3.12.3, the same torch/CUDA/Transformers versions and
  PEFT 0.21.2, bitsandbytes 0.50.2, Accelerate 1.15.0, TRL 0.29.1 and
  Datasets 5.1.0. Its global relative Euclidean gradient error was 0.010718087008025574
  against a limit of 0.01. Later losses also exceeded the 0.001 relative
  limit. The study stopped with no RAM-tier verdict. Seed 1 and the noise
  diagnostic were unrun, and the seed-0 training results were descriptive.
  There is no measured placement verdict for the intended 8 GB GPU and
  32 GB host RAM configuration.

Measured by @Batyrkhantzy111

Evidence: benchmarks/evidence/moe-adapter-placement-a6000-pro6000-2026-10-14/ (source commit 40b0e5748859)
