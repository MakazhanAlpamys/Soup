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
- [Correctness First (v0.36.0)](#correctness-first-v0360)
- [Multi-GPU / DeepSpeed / FSDP](#multi-gpu--deepspeed--fsdp)
- [Performance + Long-Context](#performance--long-context)
- [Live CUDA Batch-Size Probe](#live-cuda-batch-size-probe)
- [FSDP Shard Consolidation](#fsdp-shard-consolidation)
- [BitNet 1.58-Bit Fine-Tuning (BETA, v0.52.0)](#bitnet-158-bit-fine-tuning-beta-v0520)
- [MoE Expert Quantization + Router-Only Training (v0.52.0)](#moe-expert-quantization--router-only-training-v0520)
- [Unsloth Dynamic 2.0 GGUF Ladder (v0.53.0)](#unsloth-dynamic-20-gguf-ladder-v0530)
- [KV Cache Types (v0.53.0)](#kv-cache-types-v0530)
- [FP8 Attention + NVFP4 + Native `unsloth_bnb_4bit` (v0.53.0)](#fp8-attention--nvfp4--native-unsloth_bnb_4bit-v0530)
- [LF / Axolotl Quant Parity (v0.53.0)](#lf--axolotl-quant-parity-v0530)
- [Advanced Save Formats (v0.53.0)](#advanced-save-formats-v0530)
- [Quant Menu II + Export Pipeline (v0.53.1)](#quant-menu-ii--export-pipeline-v0531)

---

## Quantization-Aware Training (QAT)

Train with simulated quantization for significantly better post-quantization quality compared to standard QLoRA:

```bash
# Install QAT support
pip install 'soup-cli[qat]'
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
pip install 'soup-cli[qat]'   # torchao >= 0.5.0 includes torchao.float8
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
pip install 'soup-cli[cce]'    # or: pip install cut-cross-entropy
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
message; vision / audio modality is still SFT-only inline-BNB (multi-modal
Quant Menu wiring tracked as a follow-up).


## Activation Offloading (Small-VRAM Large-Batch)

Offload saved activations to RAM or disk during the backward pass to fit bigger effective batch sizes on smaller GPUs:

```yaml
training:
  activation_offloading: cpu    # or "disk"
```

`cpu` moves saved tensors to RAM (fast, bounded by system RAM); `disk` writes them to a scratch dir under the training output directory (slower, bounded by free disk). Scratch paths are containment-checked vs the current working directory, `torch.load(weights_only=True)` prevents arbitrary Python deserialization on reload, and the context manager best-effort cleans up scratch files on normal exit **and** on crash.

Not compatible with unsloth (own memory manager) or mlx. Wired across every transformer-backend trainer (SFT, DPO, GRPO, KTO, ORPO, SimPO, IPO, PPO, Reward-Model, Embedding, Pretrain).


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

# DeepSpeed ZeRO Stage 2 with CPU offload (memory-constrained)
soup train --config soup.yaml --deepspeed zero2_offload

# DeepSpeed ZeRO++ — quantized weights + gradients, hierarchical partitioning
soup train --config soup.yaml --deepspeed zero++

# FSDP2 Full Shard (native PyTorch, like ZeRO-3)
soup train --config soup.yaml --fsdp full_shard

# FSDP2 Shard Grad Op (like ZeRO-2)
soup train --config soup.yaml --fsdp shard_grad

# FSDP2 Full Shard with CPU offload
soup train --config soup.yaml --fsdp full_offload
```

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
  use_liger: true            # Liger Kernel fused ops (20-60% memory savings)
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
pip install 'soup-cli[liger]'     # Liger Kernel fused operations
pip install flash-attn --no-build-isolation  # FlashAttention
pip install 'soup-cli[ring-attn]' # Ring FlashAttention (sequence parallelism)
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


## BitNet 1.58-Bit Fine-Tuning (BETA, v0.52.0)

New `training.quantization: bitnet_1.58` for ternary-weight training (axolotl + onebitllms wrapping). Schema-only on the trainer side; the new export targets are wired as CLI stubs:

```bash
# Schema-locked; live export lands in v0.52.1.
soup export --model ./output --format bitnet
soup export --model ./output --format tq1_0
```

A ready-made `falcon-e-bitnet-sft` recipe is shipped:

```bash
soup recipes use falcon-e-bitnet-sft
soup train --config soup.yaml
```

Restricted to `task ∈ {sft, pretrain, dpo}` on `backend ∈ {transformers, unsloth}` with text modality; the cross-validator rejects MLX and vision/audio configurations loudly at config load.


## MoE Expert Quantization + Router-Only Training (v0.52.0)

For fused-MoE models trained with `moe_lora: true`, two new toggles ship:

- `training.moe_expert_quant: nf4 | int8_rowwise` — per-expert weight quantization (axolotl).
- `training.train_router_only: true` — freeze every expert and train only the gating router (unsloth pattern).

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


## FP8 Attention + NVFP4 + Native `unsloth_bnb_4bit` (v0.53.0)

Three new TrainingConfig bools extend the v0.28.0 FP8 menu:

- `fp8_attention: true` — requires `quantization_aware: fp8` AND a non-MLX backend. Targets axolotl parity for FP8 attention on Hopper+ GPUs.
- `nvfp4: true` — Blackwell-only FP4 training. Gated to non-MLX + `modality: text`; the SM ≥ 12.0 runtime check fires at trainer construction.
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


