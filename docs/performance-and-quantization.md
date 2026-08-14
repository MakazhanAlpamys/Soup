# Performance & Quantization

[← Back to the Soup README](../README.md)

> QAT, FP8, the Quant Menu (I + II), KV-cache, NVFP4, save formats, Cut Cross-Entropy, gradient checkpointing, kernel auto-composition, activation offloading, and multi-GPU / DeepSpeed / FSDP.

**Contents:**

- [Quantization-Aware Training (QAT)](#quantization-aware-training-qat)
- [FP8 Training (Hopper+)](#fp8-training-hopper)
- [Cut Cross-Entropy (Large-Vocab Models)](#cut-cross-entropy-large-vocab-models)
- [Gradient Checkpointing Tiers](#gradient-checkpointing-tiers)
- [Kernel Auto-Composition](#kernel-auto-composition)
- [Cross-Document Attention Masking](#cross-document-attention-masking)
- [Quant Menu — 9 Quantization Formats](#quant-menu--9-quantization-formats)
- [Activation Offloading (Small-VRAM Large-Batch)](#activation-offloading-small-vram-large-batch)
- [Layer Streaming (BETA, v0.72.0; NF4 v0.72.2; disk + wider archs v0.72.3)](#layer-streaming-beta-v0720-nf4-v0722-disk--wider-archs-v0723-preference-losses-v0724)
- [Correctness First (v0.36.0)](#correctness-first-v0360)
- [Multi-GPU / DeepSpeed / FSDP](#multi-gpu--deepspeed--fsdp)
- [Performance + Long-Context](#performance--long-context)
- [Live CUDA Batch-Size Probe](#live-cuda-batch-size-probe)
- [FSDP Shard Consolidation](#fsdp-shard-consolidation)
- [BitNet 1.58-Bit Fine-Tuning (BETA, live in v0.71.20)](#bitnet-158-bit-fine-tuning-beta-live-in-v07120)
- [MoE Expert Quantization + Router-Only Training (live in v0.71.20)](#moe-expert-quantization--router-only-training-live-in-v07120)
- [Unsloth Dynamic 2.0 GGUF Ladder (v0.53.0)](#unsloth-dynamic-20-gguf-ladder-v0530)
- [KV Cache Types (v0.53.0)](#kv-cache-types-v0530)
- [FP8 Attention + NVFP4 + Native `unsloth_bnb_4bit` (v0.53.0)](#fp8-attention--nvfp4--native-unsloth_bnb_4bit)
- [LF / Axolotl Quant Parity (v0.53.0)](#lf--axolotl-quant-parity-v0530)
- [Advanced Save Formats (v0.53.0)](#advanced-save-formats-v0530)
- [Quant Menu II + Export Pipeline (v0.53.1)](#quant-menu-ii--export-pipeline-v0531)

---

## Quantization-Aware Training (QAT)

Train with simulated quantization for significantly better post-quantization quality compared to standard QLoRA:

```bash
# Install QAT support
pip install "soup-cli[qat]"
```

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: sft

data:
  train: ./data/train.jsonl
  format: alpaca

training:
  epochs: 3
  lr: 2e-5
  quantization: 4bit
  quantization_aware: true  # Enable QAT
  lora:
    r: 64
    alpha: 16

output: ./output
```

**When to use QAT vs post-training quantization:**
- **QAT** (`quantization_aware: true`): Better quality when you plan to deploy with aggressive quantization (int8/int4). ~5-10% slower training, but the model learns to compensate for quantization noise.
- **Post-training quantization** (default): Faster training, good enough for most use cases. Quantize after training with `soup export --quant q4_k_m`.

QAT works with all training tasks (SFT, DPO, GRPO, PPO, KTO, ORPO, SimPO, IPO, Pretrain) and vision modality. Not compatible with the unsloth backend. After QAT training, export to GGUF normally with `soup export`.


## FP8 Training (Hopper+)

For H100 / H200 / B100 / B200 GPUs, train with float8 matmuls for ~2x speedup vs bf16 at comparable quality. This extends QAT infrastructure via `torchao.float8`:

```bash
pip install "soup-cli[qat]"   # torchao >= 0.5.0 includes torchao.float8
```

```yaml
training:
  quantization_aware: fp8   # ← string 'fp8', not bool true
  quantization: none        # FP8 converts linears directly; no bnb 4bit needed
```

### FP8 Scaling Recipes (v0.28.1)

Choose a scaling recipe to trade off speed vs accuracy:

```yaml
training:
  quantization_aware: fp8
  fp8_recipe: rowwise      # tensorwise | rowwise | rowwise_with_gw_hp
```

| Recipe | Kernel | Scaling | Trade-off |
|---|---|---|---|
| `tensorwise` (default) | cuBLAS | Single scale per tensor | Fastest, good accuracy |
| `rowwise` | CUTLASS | Per-row scale, e4m3, power-of-2 scales | Slower, more accurate |
| `rowwise_with_gw_hp` | CUTLASS | Rowwise + grad_weight in high precision | Slowest, most accurate |

Omitting `fp8_recipe` defaults to `tensorwise` (identical to v0.28.0 behavior).

Bool `true` stays on the int8 QAT path for backward compatibility. FP8 requires CUDA + Hopper+ (compute capability ≥ 9.0) and is rejected on unsloth/mlx backends. Wired across every transformer-backend trainer (SFT, DPO, GRPO, KTO, ORPO, SimPO, IPO, PPO, Reward-Model, Embedding, Pretrain).


## Cut Cross-Entropy (Large-Vocab Models)

Models with 128k+ vocabularies (Llama 3.1, Qwen2) materialise a huge `(batch, seq, vocab)` logits tensor that dominates VRAM. Cut Cross-Entropy computes the loss in chunks instead:

```bash
pip install "soup-cli[cce]"    # or: pip install cut-cross-entropy
```

```yaml
training:
  use_cut_ce: true   # Patches the CE kernel before model load
```

Architecture detection matches on the model name's last path component (`meta-llama/Llama-3.1-8B` → llama patcher) so org prefixes don't trigger the wrong recipe. Saves 8-24 GB VRAM at common batch × seq shapes. Not compatible with unsloth (own CE kernel) or mlx. Wired across every transformer-backend trainer (SFT, DPO, GRPO, KTO, ORPO, SimPO, IPO, PPO, Reward-Model, Embedding, Pretrain) — note that PPO has its own forward loop so cut_ce no-ops gracefully there.


## Gradient Checkpointing Tiers

Instead of a boolean, `gradient_checkpointing` now accepts a tier that trades compute for memory more precisely:

```yaml
training:
  # One of: false | true | "selective" | "medium" | "full" | "auto"
  gradient_checkpointing: auto
```

- **`full`** / `true` — every transformer block (~30% slowdown, biggest save).
- **`medium`** — every other block (balance).
- **`selective`** — attention only (~10% slowdown, modest save).
- **`auto`** — pick based on detected VRAM: < 24 GB → full, 24-80 GB → medium, > 80 GB → selective.

Legacy boolean configs continue to work unchanged.


## Kernel Auto-Composition

Let Soup benchmark available kernel combinations and pick the fastest for your GPU on the first training steps:

```yaml
training:
  kernel_auto_compose: true
```

Enumerates baseline / Liger / FlashAttention / Cut-Cross-Entropy combos, benchmarks each briefly on the trainer's actual model (forward-only under `torch.no_grad()` so live gradients aren't polluted), and adopts the fastest. Falls back to baseline on CPU and backs off for unsloth/mlx backends (both manage kernels internally). Wired across every transformer-backend trainer (SFT, DPO, GRPO, KTO, ORPO, SimPO, IPO, PPO, Reward-Model, Embedding, Pretrain).


## Cross-Document Attention Masking

When `packing: true` packs multiple short documents into one sequence, the default causal mask allows attention to bleed across doc boundaries. Enable block-diagonal masking to prevent this:

```yaml
training:
  packing: true
  packing_cross_doc_attn_mask: true
```

The mask builder is numpy-vectorised (`np.tril` per block) to stay fast at large `max_length`. Misconfiguring it without `packing: true` is rejected at config-load time.


## Quant Menu — 9 Quantization Formats

Pick the right quantization format for your base model and hardware. Soup
loads the appropriate `quantization_config` and trains LoRA on top:

```yaml
# Train LoRA on top of a pre-quantized GPTQ checkpoint:
base: TheBloke/Llama-2-7B-Chat-GPTQ
training:
  quantization: gptq        # or: awq, hqq:4bit, aqlm, eetq, mxfp4, fp8

# FSDP + QLoRA — set quant_storage:
training:
  quantization: 4bit
  bnb_4bit_quant_storage: bfloat16
```

| Format | Bits | Use case | Optional dep |
|---|---|---|---|
| `4bit` | 4 | Default. Best general LoRA training. | bitsandbytes |
| `8bit` | 8 | Larger memory budget, more accurate gradients. | bitsandbytes |
| `none` | 16/32 | Full fine-tuning or DPO/PPO without quant. | — |
| `gptq` | 2/3/4/8 | Train LoRA on top of an existing GPTQ checkpoint. | gptqmodel |
| `awq` | 4 | Train LoRA on top of an existing AWQ checkpoint. | autoawq |
| `hqq:Nbit` | 1, 2, 3, 4, 5, 6, 8 | Wide bit range; compose with LoRA. | hqq |
| `aqlm` | 2 | Extreme compression. | aqlm |
| `eetq` | 8 | Fast 8-bit kernel for SM75+. | eetq |
| `mxfp4` | 4 | Newer 4-bit type with better activation distribution. | bitsandbytes ≥ 0.45 |
| `fp8` | — | Train fp16/bf16 on top of FP8-released checkpoints. | transformers ≥ 4.45 |

**Compatibility matrix.** `soup train` runs `check_quant_distributed_compat()` at
startup. HQQ / EETQ / AQLM hard-fail with FSDP and ZeRO-3 (sourced from
LlamaFactory's matrix at `quantization.py:199/211`); BNB 4-bit + FSDP without
`bnb_4bit_quant_storage` emits a yellow warning.

**Pre-quantized + QAT.** `gptq` / `awq` / `hqq:*` / `aqlm` / `eetq` / `mxfp4` /
`fp8` all carry their own scale; combining with `quantization_aware` (int8 QAT or
`'fp8'`) is rejected at config-load.

**Multi-trainer support.** Quant Menu is wired across all 12 transformer-backend
trainers (SFT / DPO / GRPO / KTO / ORPO / SimPO / IPO / PPO / RewardModel /
Pretrain / Embedding / BCO). PPO's reward model also loads with the same Quant
Menu config as the policy when `tcfg` is passed in, so a GPTQ-policy + GPTQ-reward
run does not silently OOM in fp16. MLX backend is rejected with a distinct error
message; vision and audio modality now thread the same unified Quant Menu loader
(the `modality: text` gate was dropped in v0.71.19), so the full menu —
`gptq` / `awq` / `hqq:*` / `aqlm` / `eetq` / `mxfp4` / `fp8` — applies to
multi-modal SFT too (a given vision/audio checkpoint still needs a class + kernel
that supports the chosen format, e.g. `autoawq` for awq).


## Activation Offloading (Small-VRAM Large-Batch)

Offload saved activations to RAM or disk during the backward pass to fit bigger effective batch sizes on smaller GPUs:

```yaml
training:
  activation_offloading: cpu    # or "disk"
```

`cpu` moves saved tensors to RAM (fast, bounded by system RAM); `disk` writes them to a scratch dir under the training output directory (slower, bounded by free disk). Scratch paths are containment-checked vs the current working directory, `torch.load(weights_only=True)` prevents arbitrary Python deserialization on reload, and the context manager best-effort cleans up scratch files on normal exit **and** on crash.

Not compatible with unsloth (own memory manager) or mlx. Wired across every transformer-backend trainer (SFT, DPO, GRPO, KTO, ORPO, SimPO, IPO, PPO, Reward-Model, Embedding, Pretrain).


## Layer Streaming (BETA, v0.72.0; NF4 v0.72.2; disk + wider archs v0.72.3; preference losses v0.72.4)

Stream frozen base-model decoder layers ONE at a time from CPU RAM into small VRAM buffers instead of keeping the whole base resident. Peak VRAM is bounded by the size of a single layer, not the entire model — so models that don't fit resident on your GPU can now train at all.

```yaml
training:
  stream_layers: true          # Enable layer streaming
  stream_source: auto          # 'auto' (same-host RAM), 'ram', 'disk' (v0.72.3)
  stream_buffers: 2            # Double-buffering; range [2, 8]
  # stream_vram_override: 4_000_000_000   # Bytes to assume free (v0.73.x); see below
  # stream_vram_probe: true    # Decide the fit by MEASURING one step (sft only); see below
```

```bash
# Layer streaming is a CONFIG key, not a CLI flag — just train normally:
soup train --config soup.yaml
```

**How it works.** LoRA adapters + their gradients + optimizer state stay resident in VRAM (they are small). The frozen base lives in CPU RAM, page-locked when the machine allows it, and is streamed: each decoder layer is copied into one of two pre-allocated VRAM buffers on a dedicated CUDA stream while the previous layer is still computing, so the load overlaps the compute. Each layer is read **twice** per step — once in the forward pass and once when the backward pass recomputes it — because `dL/dx = Wᵀ · dL/dy` needs the weights to reach the layers below. That is physics, not an implementation detail, and it is why streaming costs time.

The tradeoff: **1.43× slower than resident training**, measured at 0.5B — the only apples-to-apples comparison available on the reference box, because 1.5B and above cannot run resident there at all.

### NF4 streaming (`quantization: 4bit`)

Quantising the streamed base to NF4 makes the RAM store ~4× smaller. That matters for two reasons, and the second is the bigger one:

1. A bigger model fits in host RAM at all — an 8B base is ~3.6 GB of NF4 instead of ~16 GB of bf16.
2. **The store fits under the machine's page-locked memory ceiling.** Pinned host memory is what lets `copy_(non_blocking=True)` actually overlap with compute. The reference box tops out at ~7.1 GB of page-locked memory, so a 5.55 GB bf16 3B base fell back to pageable and lost overlap; the 1.43 GB NF4 store pins, and utilisation goes from 79.3% to 100%.

The base is quantised **once, offline**, one tensor at a time, and cached. The shard cache is keyed to the quantisation, the dtype, the quantisation device and a fingerprint of the source checkpoint, so switching `none` ⇄ `4bit` — or retraining a base in place — re-shards rather than silently streaming the wrong bytes.

Correctness is not a tradeoff here either: a streamed NF4 run is **bit-exact** against a *resident* NF4 run (the same quantised bytes through the same bitsandbytes kernels), and that is a regression test, not a one-off measurement.

**Measured numbers (RTX 3050 Laptop 4 GB, Windows 11, LoRA, batch 1, 50 steps after 10 warmup):**

| Model | Quant | Seq | Throughput | GPU Util | Peak VRAM | RAM store |
|---|---|---|---|---|---|---|
| **Llama-3.1-8B-Instruct** | **NF4** | 512 | **119.6 tok/s** | 100% | **3.32 GB** | 3.60 GB pinned |
| Qwen2.5-3B | NF4 | 512 | 264.2 tok/s | 100% | 1.76 GB | 1.43 GB pinned |
| Qwen2.5-3B | bf16 | 512 | 143.1 tok/s | 79.3% | 2.15 GB | 5.55 GB pageable |
| Qwen2.5-1.5B | bf16 | 512 | 525.0 tok/s | 96.8% | 1.82 GB | pinned |
| Qwen2.5-1.5B | bf16 | 1024 | 487.6 tok/s | 96.7% | 2.96 GB | pinned |
| Qwen2.5-0.5B | bf16 | 512 | 978.6 tok/s | 91.4% | 1.47 GB | pinned |

**Headline:** **Llama-3.1-8B fine-tunes on a 4 GB card at 119.6 tok/s in 3.32 GB.** For scale, 1M training tokens is ~2.3 h at 8B (arithmetic from the measured rate, not a separate measurement).

The 3B NF4-vs-bf16 rows differ by 1.85×, but attribute that to **pinning, not arithmetic** — see point 2 above. The two rows also come from different sessions, and this card's boost clock varies ~13% between sessions, so treat the factor as indicative and the mechanism as the claim.

Untied `embed_tokens` + `lm_head` stay resident and unquantised (2.10 GB of the 8B row's 3.32 GB), which is why 8B sits close to this card's ceiling; treating them as streamed large layers is deferred beyond v0.72.3.

**Honest scope:**
- **RAM tier + disk overflow (v0.72.3).** `stream_source: auto` picks RAM when it fits, falls back to NVMe disk when not; SATA/HDD rejected. Correctness verified; disk performance unmeasured on the reference box.
- **Llama / Qwen / Mistral / Gemma / Gemma2 / Gemma3-Text / Phi / Phi3** (all verified bit-exact in bf16 and NF4), `task: sft`, `backend: transformers`, `modality: text`.
- **Batch sizes, gradient accumulation, `--resume` / `--hf-resume`** all now work (v0.72.3).
- **Pre-Ampere cards (T4, P100, V100, GTX 16xx, RTX 20xx) now stream in fp16 instead of bf16.** Until this fix the store dtype was hardcoded to bf16 on every CUDA device, so the entire free notebook tier was streaming a dtype its GPU has no units for, and nothing said so — it could not fail on the Ampere card every number above was measured on. fp16 is bit-exact against a resident reference of matching numerics, `0.000000e+00` in both quantisations, exactly as bf16 is.
  **The capability question is asked as `torch.cuda.is_bf16_supported(including_emulation=False)`, and the keyword is load-bearing.** The bare call defaults to including emulation: when its compute-capability fast path fails it falls through to constructing a bf16 tensor, which software emulation satisfies, so **a T4 answers True**. The first version of this fix asked the bare question and was therefore a no-op on exactly the hardware it targeted — found by running the [proof notebook](../notebooks/proof-4gb.ipynb) on a real T4, not by reasoning. `get_compute_dtype` was a second copy of the same question and now delegates to the same helper.
  **Still not measured on a pre-Ampere card**: the fp16 exactness above was measured *using* fp16 on Ampere, so it establishes the plumbing, not the Turing/Pascal kernels — bitsandbytes NF4 on sm_75 in particular.
- **A streamed 8B run now completes on a Turing card — free-tier Colab, Tesla T4 (sm_75) — and that is all it shows.** `NousResearch/Meta-Llama-3.1-8B-Instruct`, NF4, `stream_buffers: 2`, batch 1, `max_length: 256`, LoRA r=8, fp16: 7 steps, exit 0, adapter written with 128 of 128 tensors non-zero, **measured peak 2.91 GB** against a predicted ~3.02 GB (the pre-flight over-predicts by 3.8%, the safe direction it was fitted for). The T4 has 15.6 GB, so the process was capped to **4.00 GB** with `torch.cuda.set_per_process_memory_fraction`, and the cap was shown to bite — a 4.29 GiB allocation was refused. **No throughput is quoted from this run**: a card under an artificial cap is not a benchmark, and the [notebook](../notebooks/proof-4gb.ipynb) deliberately quotes none either. **What it does not establish**: backward/gradient exactness at 8B on Turing (a non-zero adapter shows gradients flowed, not that they were correct), and the notebook's streamed-vs-resident comparison produced no captured output, so it is recorded as unrun rather than as a pass. Note also that the pre-flight read **free VRAM 15.10 GB** — the device, not the per-process cap — so on capped hardware it is `training.stream_vram_override` and not the fit decision that enforces the real budget. Record: [`benchmarks/run-t4-colab-free-tier.md`](../benchmarks/run-t4-colab-free-tier.md).
- **The bf16 3B throughput above is a LOWER BOUND.** The reference box could not page-lock the 5.55 GB base (its measured page-locked ceiling is 7.65 GB, and a CUDA context plus the model skeleton did not leave room), so that run fell back to a pageable store. Pageable memory makes the host-to-device copy synchronous, which costs overlap — visible as the GPU-utilisation drop from 96.8% (1.5B, pinned) to 79.3% (3B, pageable). Soup does this fallback automatically **and prints the cost** rather than absorbing it silently. NF4 lifts this at 3B: the store drops under the ceiling and pins.
- Numbers are Windows/WDDM and therefore systematically pessimistic versus Linux. `expandable_segments:True` is silently ignored on Windows; Soup detects that and does not claim it is active.

### Sizing a streaming run (v0.72.3)

Streaming bounds the **weights**. It does nothing for activations or for the logits
tensor, and both scale with `batch × seq`. On a large-vocabulary model that second term
dominates everything else: measured on Qwen2.5-0.5B (vocab 151 936) at batch 8, S=512,
the logits alone are **8.71 GB — 146× the entire layer-buffer pool (0.060 GB)**. A
pre-flight that budgeted only weights and buffers would wave that configuration through.

So `soup train` predicts peak VRAM before building the model, and **refuses a run it
expects not to fit**:

```
peak VRAM    ~0.48 GB at batch 2 x seq 256 (logits 0.35 GB)
free VRAM    3.46 GB
forecast     5685-8361 tok/s — a compute-bound bound, not a promise
             (from 6.75 TFLOPS measured on this card now @ 862 MHz)
```

The prediction was fitted to ten real runs across two models, a 3.1× vocabulary contrast,
batch 1–8 and two sequence lengths: **worst error 0.85%, and it never under-predicts** —
the only safe direction for a number allowed to stop a run. The refusal names the two
knobs that actually scale it (`training.batch_size`, `data.max_length`).

Refusing rather than warning is deliberate. On Linux an over-budget step is a hard OOM.
On Windows it is worse: WDDM silently spills to host memory and the run merely becomes an
order of magnitude slower — measured here as a 9.27 GB peak on a 4.29 GB card with **no
exception raised at all**. Read as "streaming is slow", that would be exactly the wrong
conclusion.

The throughput line is a **bound, not a promise**. It comes from a bf16 GEMM benchmarked
on your card in that session and is printed with the SM clock it was taken at, because
this card alone produced 3.5 and 7.6 TFLOPS in two sessions at the same reported clock. A
per-card constant compiled into Soup would be a fabrication. Real streamed runs landed at
68–100% of their measured ceiling.

### Batch size vs gradient accumulation

Both work from v0.72.3, and they are not interchangeable. Measured on Qwen2.5-0.5B bf16,
S=256, pinned store, 50 steps after 10 warm-up:

| batch | accum | effective batch | throughput | peak VRAM |
|---|---|---|---|---|
| 1 | 1 | 1 | 556.6 tok/s | 0.842 GB |
| 1 | 4 | 4 | 540.1 tok/s | 0.846 GB |
| 4 | 1 | 4 | **1378.0 tok/s** | 2.28 GB |

Accumulation is **per-token I/O-neutral** — layer reads per 1000 tokens held constant
across accum 1, 2 and 4, because `accum=N` re-reads the base N times *and* processes N
times the tokens. Its cost is opportunity cost: at the **same effective batch of 4**,
raising `batch_size` instead was **2.52× faster**, because one weight read is amortised
over four times the tokens.

What accumulation buys is effective batch at **constant VRAM** (0.842 → 0.846 GB across
accum 1→4, where raising batch cost 0.842 → 2.28 GB). So the rule is: **raise
`batch_size` until the VRAM pre-flight refuses, then accumulate for the rest.** Soup
prints this advice when it sees you accumulating.

**Rejected at config load (each names the release that lifts it):**
- `batch_size: "auto"` → OOM-probes a resident model that streaming never loads; explicit batch sizes allowed (v0.72.3)
- `quantization` other than `none` or `4bit` → other formats cannot be streamed into a pooled buffer
- `backend: unsloth` / `backend: mlx` → streaming replaces the model-load path those backends own
- `task` other than `sft` / `dpo` / `orpo` / `simpo` / `kto` → named explicitly. `grpo` and `ppo` are refused **permanently**, not pending: generation rollouts re-read every layer once per generated token, which destroys the amortisation streaming depends on
- `task: kto` with `batch_size: 1` → TRL's KL term is degenerate at batch 1; refused when the config is read rather than minutes later after sharding
- `lora.use_dora` / `lora.use_vera` / `lora.init_strategy` other than `random` → these initialise from the real base weight, which is on the meta device under streaming
- `unfrozen_parameters`, `lisa_enabled`, `packing`, `multipack`, `use_fsdp2_compile`, `train_router_only`, `expand_layers` → each independently rewrites or re-freezes the same layers
- `stream_source` / `stream_buffers` / `stream_vram_override` / `stream_vram_probe` set while `stream_layers: false` → a footgun, refused
- `stream_vram_probe` on any task other than `sft` → the probe runs a plain causal-LM step, which *is* the SFT step but is not a preference loss. Measured at one matching shape it is conservative there too (6.02 GB against a real DPO step's 5.30 GB, +13.5%), but one shape is not a validation, so it is not offered for `dpo`/`orpo`/`simpo`/`kto` yet
- an architecture outside the supported list (llama / qwen2 / qwen3 / mistral / gemma / gemma2 / gemma3_text / phi / phi3) → named explicitly

**Config example:**

```yaml
base: Qwen/Qwen2.5-3B
task: sft
backend: transformers

data:
  train: ./data.jsonl
  format: alpaca
  max_length: 512
  val_split: 0.1

training:
  epochs: 3
  lr: 2e-5
  batch_size: 1           # explicit sizes allowed; "auto" rejected
  gradient_accumulation_steps: 1   # values > 1 now allowed (v0.72.3)
  quantization: 4bit      # NF4 — ~4x smaller RAM store than bf16 (or `none`)
  gradient_checkpointing: true     # handled per-layer by the streamer
  stream_layers: true     # Enable layer streaming
  stream_source: auto     # RAM with auto-fallback to NVMe disk (v0.72.3)
  stream_buffers: 2       # double-buffering
  lora:
    r: 64
    alpha: 16

output: ./output
```

**Performance notes:**
- 1.43× slower than resident training, measured at 0.5B (the only size on the reference box where a resident baseline genuinely fits in 4 GB and is therefore a fair comparison).
- The 1.5B runs sit at ~97% GPU utilisation, i.e. compute-bound: with a page-locked store the layer loads hide almost completely behind compute. The 3B run's 79.3% is **not** a model-size effect — it is the cost of the pageable-store fallback on that particular box.
- Correctness is not a tradeoff: streamed and resident forward passes were verified **bit-exact**, and a 100-step streamed loss curve matched resident exactly. Streaming substitutes the same weight bytes into the same kernels.

> **v0.72.0 adapters are unloadable — re-run them on v0.72.1.** In v0.72.0 a streamed run saved every adapter tensor under a key carrying an extra `.inner.` segment, so `soup merge`, `soup serve`, `soup chat` and `PeftModel.from_pretrained` loaded **zero** tensors and silently returned the untuned base (PEFT emitted only a `UserWarning`). The training itself was correct — only the saved file was affected. Check with:
>
> ```bash
> python -c "from safetensors.torch import load_file; \
> print([k for k in load_file('adapter_model.safetensors') if '.inner.' in k][:3])"
> ```
>
> If that prints anything, the adapter is affected. From v0.72.1 a streamed adapter is byte-for-byte in the same layout as an ordinary LoRA run.

**Troubleshooting:**
- **"layer streaming needs the base to fit in RAM"** — the base is larger than free RAM. Set `stream_source: auto` to fall back to the NVMe disk tier, free RAM, or pick a smaller base.
- **"could not page-lock the base … falling back to a PAGEABLE RAM store"** — expected on a busy machine. Training continues, more slowly. Close other applications to keep the pinned store.
- **"layer streaming does not support model_type=…"** — the supported list is llama / qwen2 / qwen3 / mistral / gemma / gemma2 / gemma3_text / phi / phi3. Multimodal `gemma3` is excluded on purpose; use `gemma3_text`.
- **"predicted peak … exceeds free VRAM" and you believe it is wrong** — lower `batch_size` or `data.max_length` first. Otherwise there are two escape hatches and they are not interchangeable. `training.stream_vram_probe: true` (`sft` only) **measures** one real forward+backward at your configured shape and decides on that, printing the prediction beside it; it costs one step (1–5 s measured) and it can also refuse a run the formula accepted. `training.stream_vram_override: <bytes>` instead **replaces** the free-VRAM figure the check runs against — that is an assertion you are making, not a measurement, so raising it past a real limit is an OOM on Linux and a silent spill on Windows. Prefer the probe when you want to be told the truth; use the override when you know something the driver cannot report.
- **The prediction is not equally trustworthy at every sequence length.** Measured on a 4 GB RTX 3050 with SmolLM2-135M streamed in bf16 at batch 1, the formula over-predicts by 8% at seq 4352 (safe) and then **under-predicts — 0.934x the real peak at seq 5120 and 0.787x at 6144**. The grid it was fitted on only ever varied batch size, at seq 256 and 512, so long-context streaming is exactly where it has the least evidence behind it. If you are streaming at multi-thousand-token sequences, turn on `stream_vram_probe`. Record: [`benchmarks/gate-v0.73.1-measured-vram-fit.md`](../benchmarks/gate-v0.73.1-measured-vram-fit.md).
- **The pre-flight reports the whole card on a capped or shared GPU** — `torch.cuda.mem_get_info()` is a device-level driver query and cannot see `set_per_process_memory_fraction`, a MIG slice, or another process on the same card. Set `training.stream_vram_override` to what your process may actually use; the check then refuses configurations that would exceed *that*, which is also how you rehearse a 4 GB card on a 16 GB one.
- **Slower than you expected** — layer streaming trades time for memory. If the model already fits resident on your card, do not enable it.

### Preference losses over streaming (v0.72.4)

`dpo`, `orpo`, `simpo` and `kto` stream exactly like `sft` — same config keys, same
pre-flight, same refusals. The interesting part is DPO's reference model.

**DPO compares the model being trained against a frozen reference.** Implemented as a
second model instance that doubles memory and there is no point streaming at all. Soup
instead uses *the same streamed base with its LoRA adapters switched off*, so the
reference costs no extra weights. Measured on an RTX 3050 4 GB with a 730 MB model:

| arm | peak VRAM | vs SFT |
|---|---|---|
| streamed SFT | 89.53 MB | — |
| **streamed DPO** | **81.87 MB** | **0.914×** |
| the same run forced to build a real second model | 812.32 MB | 9.92× |

The third row is the control: a second instance costs **+730.44 MB against 730.44 MB of
weights**, i.e. exactly one copy. The RAM store and the VRAM buffer pool are
byte-identical between the SFT and DPO arms.

**KTO is not reference-free**, however it is usually described — it selects a reference
the same way DPO does, so it gets the same treatment. ORPO and SimPO genuinely are
reference-free. All four are verified **bit-exact** against a resident run of the same
loss.

**The cost is time, not memory.** DPO runs the layer stack three times per step (policy
forward, reference forward, checkpoint recompute) against SFT's two — measured **1.52×**
the layer reads on a 24-layer model. Streaming makes the reference free in memory; it
does not make it free.

**Two things to know before you configure it:**

- **`kto` needs `batch_size: 2` or more.** TRL's KL term is degenerate at batch 1, so
  the run cannot work; Soup refuses it when your config is read rather than after
  sharding the checkpoint. (KTO is streamable at all only because v0.72.3 lifted
  streaming's own batch-1 restriction.)
- **The VRAM pre-flight is deliberately conservative for paired losses.** DPO, ORPO and
  SimPO send chosen and rejected through the model as one tensor, so the budget charges
  twice the rows — correct, and it never under-predicts. But it charges them at the
  *supervised* loss's measured per-element rate, and TRL's preference losses use a
  cheaper path, so the estimate is an upper bound rather than a tight one. Concretely,
  on a 4 GB card with a 128k-vocab 1B model: DPO at `max_length: 512` is allowed, and
  from `max_length: 768` up it is refused even though it would probably fit. Lower
  `max_length` if you hit that. (Tracked as a follow-up; under-predicting would be the
  strictly worse failure, because on Windows it is not an error but a silent spill to
  host memory.)

**Roadmap:**
- A published 14B-on-8 GB reference benchmark — hardware-blocked; it needs an 8 GB card and 32 GB of RAM, which the development box does not have
- GRPO and PPO are explicitly **not** planned: rollouts need generation, which re-reads the model per token

**Shard cache.** The first streaming run rewrites the checkpoint into one safetensors shard per decoder layer under `~/.soup/layer-stream/` (override with `SOUP_LAYER_STREAM_CACHE_DIR`). That costs disk space roughly equal to the base. The cache is keyed to a fingerprint of the source checkpoint, so a base retrained in place re-shards instead of silently training against stale weights.


## Correctness First (v0.36.0)

Four silent-failure modes Soup had → loud failures.

### Assistant-only loss masking

By default, Soup masks every non-assistant token with `-100` so the SFT loss reflects only what the model should *generate*. Toggle via `data.train_on_responses_only` (default `true`):

```yaml
data:
  train: data.jsonl
  train_on_responses_only: true   # default
  # OR per-message control:
  # train_on_messages_with_train_field: true
```

When the tokenizer ships a chat template with `{% generation %}` markers, the mask is exact. Without those markers, Soup falls back to an incremental tokenize-delta walk and documents the looseness.

### `--trust-remote-code` opt-in (every command, every trainer)

Every command that loads a model now requires `--trust-remote-code` to execute custom Python from a model repo (`auto_map` in `config.json`). First-party orgs (Meta, Mistral, Qwen, Google, etc.) suppress the warning panel; everything else prints a `REMOTE CODE WARNING` panel before loading. Unknown-org local checkpoints with `auto_map` raise a friendly `ValueError` at construction time instead of silently exec'ing inside `from_pretrained`.

Coverage:
- `soup train` (every task — SFT, DPO, GRPO, KTO, ORPO, SimPO, IPO, PPO, Reward Model, Pretrain, Embedding, BCO, and the unified Preference dispatcher)
- `soup chat`, `soup serve`, `soup data download`, `soup eval auto`
- `soup diff`, `soup export`, `soup merge`, `soup infer`, `soup data generate`

```bash
soup train --config soup.yaml --trust-remote-code
soup infer --model my-org/custom-arch-model --input prompts.jsonl --trust-remote-code
soup export --model ./adapter --format gguf --trust-remote-code
```

### Chat-template hardening

Tokenizers without a chat template now raise a `ValueError` with a fix suggestion instead of silently building garbage `f"{role}: {content}"` strings.

```yaml
data:
  train: data.jsonl
  chat_template: chatml   # or: llama3, qwen2.5, mistral, gemma3, phi4, deepseek-r1, or a raw Jinja string
```

Raw Jinja strings are validated: null bytes / >64KB / filesystem-touching directives (`{% include %}`, `{% import %}`, `{% from %}`, `{% macro %}`, `{% extends %}`) are rejected at config-load.

### OOM-probe auto batch size

```yaml
training:
  batch_size: auto                  # unchanged
  auto_batch_size_strategy: probe   # NEW: 'static' | 'probe' | 'auto' (default)
```

Replaces the static memory formula with a real try-halve-then-double-to-ceiling loop. Picked size is cached at `~/.soup/batch_cache.json` keyed on `(model, max_length, quantization, lora_r, gpu_name, gpu_memory_gb)` so repeat runs short-circuit.


## Multi-GPU / DeepSpeed / FSDP

Train on multiple GPUs with DeepSpeed or PyTorch FSDP2:

```bash
# DeepSpeed ZeRO Stage 2 (recommended for most cases)
soup train --config soup.yaml --deepspeed zero2

# DeepSpeed ZeRO Stage 3 (for very large models)
soup train --config soup.yaml --deepspeed zero3

# DeepSpeed ZeRO Stage 2 with CPU offload (optimizer states -> CPU)
soup train --config soup.yaml --deepspeed zero2_offload

# DeepSpeed ZeRO Stage 3 with CPU offload (parameters -> CPU; not enough VRAM)
soup train --config soup.yaml --deepspeed zero3_offload

# DeepSpeed ZeRO++ — quantized weights + gradients, hierarchical partitioning
soup train --config soup.yaml --deepspeed zero++

# FSDP2 Full Shard (native PyTorch, like ZeRO-3)
soup train --config soup.yaml --fsdp full_shard

# FSDP2 Shard Grad Op (like ZeRO-2)
soup train --config soup.yaml --fsdp shard_grad

# FSDP2 Full Shard with CPU offload
soup train --config soup.yaml --fsdp full_offload
```

`zero3_offload` keeps `offload_optimizer: none`: offloading the optimizer makes DeepSpeed JIT-build its `cpu_adam` op, which requires a matching CUDA toolkit (`nvcc`) on the box. Copy the emitted JSON and flip it if you have one — or start from the bundled `soup fetch deepspeed_configs zero3-cpu-offload`, which is the optimizer-offloading variant and therefore needs that toolkit. Measured on one H100 with Llama-3.1-8B (bf16, LoRA r=8, 256 steps): 21.65 tok/s at a 38,135 MiB peak — see [benchmarks/gate-h100-validation.md](../benchmarks/gate-h100-validation.md), STEP 3, which also compares it against layer streaming on the same box, data and model.

### `--gpus` flag — topology-aware launch

```bash
# Auto-detect GPU count; print the exact accelerate command
soup train --config soup.yaml --gpus auto

# Explicit GPU count
soup train --config soup.yaml --gpus 4
```

`soup` detects NVLink / PCIe interconnect and prints the correct
`accelerate launch` command. Copy-paste to start distributed training
(auto-reexec ships in v0.27.1).

### FSDP2 + `torch.compile`

Stack `torch.compile` on top of any FSDP preset for +20-30% throughput:

```yaml
# soup.yaml
training:
  use_fsdp2_compile: true
```

Requires `--fsdp`, CUDA, and `backend: transformers`.

### Pipeline parallelism config (wiring only in v0.27.0)

```yaml
training:
  parallelism: pipeline
  pipeline_stages: 4
```

Config validation ships in v0.27.0; live execution ships in v0.27.1. See
`recipes/deepseek-v3-pipeline` for a full scaffold.


## Performance + Long-Context

Optimize training throughput and extend context windows:

```yaml
# soup.yaml — performance options
training:
  use_liger: true            # Liger Kernel fused ops (measured 12.9% memory, 5.1% throughput)
  use_flash_attn: true       # FlashAttention v2/v3 auto-detection
  gradient_checkpointing: true  # Required for long sequences

  # Long-context (128k+ tokens)
  rope_scaling_type: dynamic  # RoPE scaling: linear, dynamic, yarn, longrope
  # use_ring_attention: true  # Sequence parallelism across GPUs

data:
  max_length: 131072          # Up to 1M tokens supported
```

Install optional performance packages:

```bash
pip install "soup-cli[liger]"     # Liger Kernel fused operations
pip install flash-attn --no-build-isolation  # FlashAttention
pip install "soup-cli[ring-attn]" # Ring FlashAttention (sequence parallelism)
```


## Live CUDA Batch-Size Probe

Set `auto_batch_size_strategy: probe` in `training:` and Soup will run a real OOM-probe before training:

```yaml
training:
  batch_size: auto
  auto_batch_size_strategy: probe
```

For each candidate size `B`, the probe runs ONE forward + backward + step on a synthetic batch of `B` sequences of length `max_length`. On `torch.cuda.OutOfMemoryError` it halves; otherwise it doubles up to `4 × static_estimate`. The picked size is cached per `(model, max_length, quantization, lora_r, gpu)` tuple in `~/.soup/batch_cache.json` so subsequent runs skip the probe.

CPU sessions and `auto_batch_size_strategy: static` skip the probe. Synthetic batch tensors are freed before the backward pass so peak VRAM reflects the realistic training step. SFT-only this release — non-SFT trainers fall back to the static estimate.


## FSDP Shard Consolidation

```bash
# Preview the plan (which shards, total size) without writing
soup merge-sharded-fsdp-weights ./fsdp-checkpoint -o ./merged.safetensors --plan-only

# Consolidate for real
soup merge-sharded-fsdp-weights ./fsdp-checkpoint -o ./merged.safetensors
```

Consolidates `pytorch_model_fsdp_*.bin` shard files into a single `.safetensors`. Each shard is loaded one at a time (streaming, not all-at-once) with `torch.load(weights_only=True)`, tensor shapes validated (a duplicate key with a conflicting shape is rejected; a same-shape duplicate keeps the first and warns), and the merged dict written atomically. cwd-containment + symlink rejection apply to the output path and every shard; per-shard 16 GiB cap; `_MAX_SHARDS=1024`. `--plan-only` prints the plan and exits 0. Live torch-side consolidation shipped in v0.71.14.


## BitNet 1.58-Bit Fine-Tuning (BETA, live in v0.71.20)

`training.quantization: bitnet_1.58` routes to a live `BitNetTrainerWrapper`
(an SFT subclass) for ternary-weight training. It is gated on the upstream
`onebitllms` package — when absent, training fails fast with a friendly
`RuntimeError` naming it (`onebitllms` is CUDA/Linux-only). The export targets
run a **real llama.cpp TQ1_0 ternary GGUF** export (reusing the v0.53.1
convert→quantize pipeline) instead of a stub:

```bash
soup export --model ./output --format bitnet   # → TQ1_0 ternary GGUF
soup export --model ./output --format tq1_0     # same flavour, explicit name
```

The export requires a built llama.cpp toolchain (the convert/quantize binaries
raise a friendly `FileNotFoundError` when missing). A ready-made
`falcon-e-bitnet-sft` recipe is shipped:

```bash
soup recipes use falcon-e-bitnet-sft
soup train --config soup.yaml
```

Restricted to `task ∈ {sft, pretrain, dpo}` on `backend ∈ {transformers, unsloth}` with text modality; the cross-validator rejects MLX and vision/audio configurations loudly at config load.


## MoE Expert Quantization + Router-Only Training (live in v0.71.20)

For fused-MoE models trained with `moe_lora: true`, two live toggles:

- `training.moe_expert_quant: nf4 | int8_rowwise` — quantizes **just the
  fused-MoE expert `nn.Linear` layers** with bitsandbytes (`Linear4bit` for
  `nf4`, `Linear8bitLt` for `int8_rowwise`), leaving attention + the gating
  router in full precision. The swap runs **before** `get_peft_model`
  (QLoRA-on-experts), so PEFT attaches its adapters to the quantized base. The
  source weights are genuinely carried into the quantized layer (validated
  dequant error 0.0155 vs source on an RTX 3050). CUDA + bitsandbytes are
  required — a friendly `RuntimeError` fires on CPU / without bnb.
- `training.train_router_only: true` — freeze every expert parameter and train
  only the gating router (applied after LoRA, on the final parameter set).

Both reject silently-no-op combinations: setting either flag without `moe_lora=true` fails at config load with an actionable message.


## Unsloth Dynamic 2.0 GGUF Ladder (v0.53.0)

`soup export --format gguf-ud --calibration-data <calib.jsonl>` is the planned dispatch surface for the 14-entry UD ladder (`UD-Q8_K_XL` … `UD-IQ1_M`). v0.53.0 ships the closed-allowlist validators, `MappingProxyType`-wrapped metadata, and a calibration-data path shape check; live llama.cpp `imatrix` invocation lands in v0.53.1. The IQ + Apple/ARM-friendly GGUF flavours (`IQ4_NL`, `Q4_0_4_4`, `Q5_K_M`, etc.) ship as separate frozensets so future export-CLI dispatch can pick by family.


## KV Cache Types (v0.53.0)

`training.kv_cache_type: q8_0 | bf16 | f16 | fp8` controls the inference-time KV cache element type. `fp8` is Hopper-only; the MLX backend is rejected at config load.

The **live serve runtime shipped in v0.71.14** for the transformers backend:

```bash
soup serve --model ./output --kv-cache-type bf16     # cache stored in the model compute dtype
soup serve --model ./output --kv-cache-type q8_0     # 8-bit quantized KV cache (needs `hqq`)
```

- `bf16` / `f16` resolve the model compute dtype for the default `DynamicCache` (no extra dependency).
- `q8_0` wires the transformers quantized KV cache (`cache_implementation="quantized"`, hqq backend). If no quant backend (`hqq` / `optimum-quanto`) is installed, the CLI exits 2 with an install hint rather than crashing.
- `fp8` is rejected on pre-Hopper GPUs (compute capability < 9.0) with a friendly runtime error naming vLLM as the path on Ampere/Ada.
- vLLM / SGLang serve wiring is still tracked under [#140](https://github.com/MakazhanAlpamys/Soup/issues/140) (`infra-blocked`).


## FP8 Attention + NVFP4 + Native `unsloth_bnb_4bit`

Three TrainingConfig bools extend the v0.28.0 FP8 menu. `fp8_attention` and `nvfp4` are LIVE
torchao converters as of v0.71.21 (hardware-gated):

- `fp8_attention: true` — requires `quantization_aware: fp8` AND a non-MLX backend. Converts the attention projections (q/k/v/o and fused variants) to torchao float8 training on Hopper+ GPUs. Missing torchao or a pre-Hopper GPU degrades to a clear advisory; a conversion-phase failure raises an honest "model may be PARTIALLY converted" error instead of training on a half-converted model.
- `nvfp4: true` — Blackwell-only FP4 training via torchao `NVFP4Config` + `quantize_`. Gated to non-MLX + `modality: text`; the SM ≥ 10 runtime check fires at trainer construction.
- `unsloth_bnb_4bit: true` — promotes "Unsloth Dynamic 4-bit" from an implicit `backend=unsloth + quantization=4bit` combo to a named flag. Mutual rejection of inconsistent combos at config load.

Cross-validator ordering picks the most actionable error: `quantization_aware='fp8'` prerequisite fires before the MLX rejection on `fp8_attention`, so a YAML missing both surfaces the deeper issue first.


## LF / Axolotl Quant Parity (v0.53.0)

- `bnb_4bit_use_double_quant: true` — requires `quantization: 4bit`. Activates BNB's double-quantization. Combinations with the Quant Menu formats (gptq / awq / hqq:Nbit / aqlm / eetq / mxfp4 / fp8) are rejected at config load.
- `llm_int8: true` — an explicit 8-bit assertion. Unlike v0.41.0 `load_in_8bit` (which **rewrites** `quantization` to `8bit`), `llm_int8` enforces that the user has ALSO set `quantization: 8bit`. Mismatch raises with an actionable message.
- `quantize_ref_model: true` / `quantize_reward_model: true` — extend the v0.40.5 Quant Menu wiring to the reference / reward models inside preference and RLHF training. `quantize_ref_model` accepts any task with a reference policy (`dpo / ipo / simpo / orpo / bco / kto / preference / grpo / ppo`); `quantize_reward_model` accepts `ppo / reward_model`.


## Advanced Save Formats (v0.53.0)

`soup merge --save-format 4bit` and `--save-format 4bit_forced` will write a single BNB-4bit-quantized merged checkpoint without the wasteful dequant → merge → requant cycle (unsloth `merged_4bit` recipe). v0.53.0 ships the closed allowlist + spec metadata; the live writer lands in v0.53.1.

`soup export --format torchao --quant-config <yaml>` is the planned PTQ export surface for `torchao.quantize_` + `save_pretrained`. Four schemes are allowlisted: `Int4WeightOnly`, `Int8DynActInt4`, `Float8DynActFloat8`, `NVFP4`. CASE-SENSITIVE — these are PyTorch class names and `torchao.quantize_` looks them up by exact name. Diverges from `--save-format` (lowercase-normalised) on purpose; documented at both validators.


## Quant Menu II + Export Pipeline (v0.53.1)

v0.53.1 lifts the v0.53.0 schema-only stubs to live wiring:

```bash
# Single-stage BNB-4bit merged checkpoint (no dequant/merge/requant)
soup merge -a ./adapter -o ./merged_4bit --save-format 4bit

# TorchAO PTQ export — closed per-scheme kwarg allowlist
cat > q.yaml <<EOF
scheme: Int4WeightOnly
group_size: 32
EOF
soup export --model ./merged --format torchao --quant-config ./q.yaml --output ./out

# Unsloth Dynamic 2.0 / IQ / Apple-ARM GGUF via llama.cpp imatrix
soup export --model ./merged --format gguf-ud \
    --gguf-flavour UD-Q4_K_XL \
    --calibration-data ./calib.jsonl \
    --output ./out/model.UD-Q4_K_XL.gguf

# Deploy autopilot with live Quant-Lobotomy measurement
soup deploy autopilot --target rtx-4090-24gb \
    --base meta-llama/Llama-3.2-1B \
    --measure --tasks ./eval_tasks.jsonl \
    --measure-candidates 4bit,gptq,awq
```

Autopilot also detects pre-quantized bases automatically — `TheBloke/Llama-2-7B-Chat-GPTQ` is recommended `gptq` instead of stacking 4-bit on top. Detection runs against the base-model name regex AND any local `config.json`'s `quantization_config.quant_method`. Out-of-cwd model paths are silently skipped (soft-probe semantics).

The advanced GGUF pipeline uses POSIX `O_NOFOLLOW` to defeat the TOCTOU race between the dispatch-time symlink check and the actual open of the calibration data — a crafted environment cannot race-swap the calibration file between validate and read.

`soup deploy autopilot --measure` caches results at `~/.soup/deploy_autopilot_cache.json` keyed on `(base, profile, eval-tasks)`. Repeat invocations short-circuit; pass `SOUP_DEPLOY_AUTOPILOT_CACHE=<path>` to redirect (constrained to home / cwd / tempdir). The recommended candidate uses soft-fallback: first `OK` by insertion order, else the candidate with the smallest delta (least drop relative to its own baseline).
