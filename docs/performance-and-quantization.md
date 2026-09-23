# Performance & Quantization

[← Back to the Soup README](../README.md)

> QAT, experimental QuEST, FP8, the Quant Menu (I + II), KV-cache, NVFP4, save formats, Cut Cross-Entropy, gradient checkpointing, kernel auto-composition, activation offloading, and multi-GPU / DeepSpeed / FSDP.

**Contents:**

- [Quantization-Aware Training (QAT)](#quantization-aware-training-qat)
- [Experimental QuEST mixed W4/A4+A16 route](#experimental-quest-mixed-w4a4a16-route)
- [FP8 Training (Ada+)](#fp8-training-ada)
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
- [BitNet 1.58-Bit Export](#bitnet-158-bit-export)
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


## Experimental QuEST mixed W4/A4+A16 route

`quantization_aware: quest` enables the exact experimental route retained for
[#674](https://github.com/MakazhanAlpamys/Soup/issues/674): all 168 transformer
linear weights use group-128 fake W4; 161 activations use group-128 fake A4;
and the seven attention/MLP linears in decoder block 23 keep A16 activations.
Both operands use a full-width normalized Hadamard transform. Activation clips
are selected from five fixed candidates on the first 32 tokenized **training**
rows only.

```yaml
base: ahxt/LiteLlama-460M-1T
task: sft
backend: transformers
modality: text

data:
  train: ./data/train.jsonl

training:
  quantization_aware: quest
  quantization: none
  batch_size: 2
  lora:
    r: 0

output: ./output
```

This first slice fails closed unless the loaded model has the measured 24-block
Llama topology with exactly those 168 linears, compatible power-of-two input
widths, and FP32 master weights. **That gate is topological only:** model
identity is not checked, so any matching 24-block / 168-linear Llama receives
the route even though the retained quality measurement used only
`ahxt/LiteLlama-460M-1T`. It also requires one visible Ampere-or-newer CUDA GPU.
DDP, DataParallel, DeepSpeed, FSDP, layer streaming, LoRA, activation offloading,
NVFP4, other backends/tasks/modalities, and pre-quantized loading are not
accepted. On a multi-GPU host, expose one card to the process, for example:

```bash
CUDA_VISIBLE_DEVICES=0 soup train --config soup.yaml
```

`use_cut_ce: true` remains supported: Cut Cross-Entropy is patched before model
loading and does not depend on the v0.28 post-load path that QuEST bypasses.

The final artifact and each periodic checkpoint contain
`quest_mixed_precision.json`. The closed, versioned sidecar records every A4
and A16 route, clipping scale, calibration-row digest, transform, grid,
surrogate, and route provenance. The provenance explains why this topology was
selected; it makes no training-quality claim about the artifact beside it.
Resume is refused if the executable route metadata differs.
Because generic Transformers cannot infer fake-quant execution from the master
weights, load the executable route explicitly:

```python
from soup_cli.utils.quest import load_mixed_quest_artifact

model = load_mixed_quest_artifact("./output")
```

### Evidence boundary

This is an engineering integration of an **evaluation-only** result, not a
validated training recipe. The [mixed-route record](../benchmarks/gate-674-quest-mixed-route.md)
used one model and no backward passes or optimizer updates. Its seven-A16 route
measured a 0.086344 nat/target gap to fixed FP, with a paired 95% interval of
[0.080241, 0.093190], over 704 examples / 25,017 targets. It is therefore:

- not pure W4A4;
- not evidence of mixed-route training quality or cross-model generality;
- not upstream QuEST numerical parity;
- not packed INT4, and not a speed or memory-efficiency claim.

The implementation keeps FP32 masters and performs dense fake quantization and
Hadamard arithmetic during execution. Treat it as a reproducible research path,
not as a cheaper deployment format.


## FP8 Training (Ada+)

For Ada, Hopper and Blackwell GPUs (RTX 40/50-series, L4, L40S, RTX 6000 Ada, H100 / H200, B100 / B200), train with float8 matmuls for ~2x speedup vs bf16 at comparable quality. This extends QAT infrastructure via `torchao.float8`:

```bash
pip install "soup-cli[qat]"   # torchao (floor: TORCHAO_MIN_VERSION in utils/torchao_compat.py)
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

Bool `true` stays on the int8 QAT path for backward compatibility. FP8 requires CUDA + an Ada or newer GPU (compute capability ≥ 8.9) and is rejected on unsloth/mlx backends. The `rowwise` and `rowwise_with_gw_hp` recipes run a separate torch kernel with its own limits: it needs a torch that dispatches it on the card (Ada 8.9: torch ≥ 2.7; Hopper 9.x and Blackwell datacenter 10.x: any supported torch; RTX 50-series 12.x: torch ≥ 2.8; 11.x: torch ≥ 2.10; no release through 2.14 runs it on 13.x), it is **never built on Windows**, and before torch 2.11 it is only built against CUDA 12 or newer. `tensorwise` has none of these limits. When FP8 is requested and this card, OS or torch build cannot run it, **the run stops at setup** with the reason (`FP8HardwareUnsupportedError`), before any layer is converted, on every trainer that reaches the converter; it never trains on without FP8. (SFT's audio, layer-streaming and unsloth setup branches do not call the converter at all, so they still accept the flag without applying it — pre-existing, tracked separately.) `fp8_attention` asks the same gate (#835). Wired across every transformer-backend trainer (SFT, DPO, GRPO, KTO, ORPO, SimPO, IPO, PPO, Reward-Model, Embedding, Pretrain).


## Cut Cross-Entropy (Large-Vocab Models)

Models with 128k+ vocabularies (Llama 3.1, Qwen2) materialise a huge `(batch, seq, vocab)` logits tensor that dominates VRAM. Cut Cross-Entropy computes the loss in chunks instead:

```bash
pip install "soup-cli[cce]"    # or: pip install cut-cross-entropy
```

```yaml
training:
  use_cut_ce: true   # Patches the CE kernel before model load
```

Architecture detection reads `config.model_type` first, so a local checkpoint directory (`soup merge`/`soup shrink`/`soup draft distill` output) is patched the same as a hub id; a name-based match on the last path component (`meta-llama/Llama-3.1-8B` → llama patcher) is the fallback when config resolution is unavailable. Saves 8-24 GB VRAM at common batch × seq shapes. Not compatible with unsloth (own CE kernel) or mlx. Wired across every transformer-backend trainer (SFT, DPO, GRPO, KTO, ORPO, SimPO, IPO, PPO, Reward-Model, Embedding, Pretrain) — note that PPO has its own forward loop so cut_ce no-ops gracefully there.


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

On SFT, `medium` uses Transformers' native `every_n_layers=2` path, while
`selective` wraps each decoder block's direct attention module and leaves HF's
full-model checkpointing off to avoid double recomputation. If an architecture
does not expose a direct attention child, Soup reports and applies a `full`
fallback instead of claiming an inactive selective tier. Other task wrappers
currently treat any enabled tier as full checkpointing. Legacy boolean configs
continue to work unchanged.


## Kernel Auto-Composition

`training.kernel_auto_compose: true` is rejected at config load. The original implementation timed the same already-loaded model once per candidate, so it neither compared different kernel configurations nor applied the name it reported.

Choose the supported optimization explicitly instead:

```yaml
training:
  use_liger: true
  # or, on a compatible CUDA setup:
  use_flash_attn: true
```

On Apple Silicon, keep both flags disabled and use `backend: mlx`; MLX manages its own kernels.


## Cross-Document Attention Masking

`training.packing_cross_doc_attn_mask` is rejected at config load. Soup used to set TRL `packing_strategy="attention_free"`, which has never been a valid strategy on any released trl (allowlist is `bfd` / `bfd-requeue` / `wrapped`), so the flag has always been a `TypeError` at setup rather than a working mask.

Packed-document isolation is TRL's default `bfd` strategy when FlashAttention is the `attn_implementation`. Use:

```yaml
training:
  packing: true
```

Do not set `packing_cross_doc_attn_mask`.


## Quant Menu — 9 Quantization Formats

Pick the right quantization format for your base model and hardware. Soup
loads the appropriate `quantization_config` and trains LoRA on top:

```yaml
# Train LoRA on top of a pre-quantized GPTQ checkpoint:
base: TheBloke/Llama-2-7B-Chat-GPTQ
training:
  quantization: gptq        # or: awq, hqq:4bit, aqlm, eetq, mxfp4, fp8

# FSDP + QLoRA — Soup resolves this to the compute dtype automatically;
# pin it explicitly when the recipe targets known bf16 hardware:
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
LlamaFactory's matrix at `quantization.py:199/211`). BNB 4-bit + FSDP resolves
`bnb_4bit_quant_storage` to the effective floating compute dtype before model
load, then aligns trainable adapter parameters to that dtype before FSDP wraps
the model. This avoids both FSDP failure modes: integer storage and mixed
adapter/storage dtypes.

**Pre-quantized + QAT.** `gptq` / `awq` / `hqq:*` / `aqlm` / `eetq` / `mxfp4` /
`fp8` all carry their own scale; combining with `quantization_aware` (int8 QAT,
`'fp8'`, or the experimental `'quest'` route) is rejected at config-load.

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

**Non-quantized module dtype (#339/#471/#492).** `from_pretrained`'s own `torch_dtype` kwarg
— set to `"auto"` (or, on a pre-Ampere CUDA card, an explicit `torch.float16` override; see
the "Load dtype" note in `docs/training.md`'s Full fine-tuning section) — now also applies to a
`4bit`/`8bit` QLoRA load, governing the modules `quantization_config` doesn't quantize
(`embed_tokens`, norms, `lm_head`). This matches `bnb_4bit_compute_dtype`, which already
resolves the same card-aware `get_compute_dtype()`, rather than leaving those modules at
whatever `from_pretrained`'s bare default happened to pick.


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
  stream_ngram_source: auto    # Qwen4 PLE only: 'auto', 'ram', or read-only 'disk'
  stream_buffers: 2            # Double-buffering; range [2, 8]
  stream_read_ahead: 2         # Disk tier only; range [1, 8]. N stages N-1 layers ahead, and each level costs one layer of pinned host RAM
  # stream_pin: false          # Force the pinned RAM store off (escape hatch) or on; unset = automatic. See below
  # stream_vram_override: 4_000_000_000   # Bytes to assume free (v0.73.x); see below
  # stream_vram_probe: true    # Decide the fit by MEASURING one step (sft only); see below
  # stream_disk_kind: nvme     # Override auto disk-kind detection: nvme/ssd/hdd; see below
```

```bash
# Layer streaming is a CONFIG key, not a CLI flag — just train normally:
soup train --config soup.yaml
```

**How it works.** LoRA adapters + their gradients + optimizer state stay resident in VRAM (they are small). The frozen base lives in CPU RAM, page-locked when the machine allows it, and is streamed: each decoder layer is copied into one of two pre-allocated VRAM buffers on a dedicated CUDA stream while the previous layer is still computing, so the load overlaps the compute. Vocabulary-sized `embed_tokens` and an untied `lm_head` use one additional shared slot: the embedding is loaded for the model input, then the same allocation is reused for the output head after the last decoder layer. Each decoder layer is read **twice** per step — once in the forward pass and once when the backward pass recomputes it — because `dL/dx = Wᵀ · dL/dy` needs the weights to reach the layers below. That is physics, not an implementation detail, and it is why streaming costs time.

**Qwen3.8-Flash-Next / Qwen4-Exp PLE.** The frozen PLE N-gram table is not a
decoder-layer weight for storage purposes: putting it in the PLE layer's shard
would make the shared layer buffer as large as the whole table. Soup keeps the
checkpoint's row-contiguous `ngram_embedding.shard_*` tensors in place.
`stream_ngram_source: disk` opens those original safetensors read-only and
gathers only the rows requested by the current tokens; it never rewrites or
copies the table into Soup's shard cache. For dense Transformers checkpoints,
`ram` loads the same parts into CPU RAM without concatenating a second full
table, and `auto` selects RAM only when the measured table plus the selected
base tier fits the host-memory headroom.

oMLX/oQ affine Qwen4 bundles are supported by this narrow text-only path. Soup
dequantizes each frozen decoder layer once into the reusable stream cache and
maps the fused Switch-MLP expert weights to the Transformers decoder. The much
larger packed PLE table remains in the original checkpoint: only requested rows
are dequantized, so oQ requires `stream_ngram_source: disk` (or `auto`). The
vision tower and MTP component are ignored because the instantiated model is
`AutoModelForCausalLM`, not the multimodal or speculative-decoding wrapper.
The CPU parity gate covers exact rows, logits, loss, LoRA gradients, source-file
non-mutation, and mapping cleanup. The same float32 tiny-checkpoint gate passes
on M4 Max within its published MPS tolerance; CPU remains the bit-exact oracle.
Production-checkpoint throughput and peak memory remain unmeasured. The initial
gate is deliberately narrow: `task: sft` and `quantization: none` (the latter is
Soup's optional NF4 transform, not the accepted oQ source encoding).
Preference-loss and streamed-NF4 parity are pending and those Qwen4 combinations
fail before sharding. Resident-versus-streamed Qwen4 parity has only been run in
float32; CUDA selects BF16, but a BF16 CUDA parity gate has not been measured and
production readiness on that path remains pending. The production 176.9B oQ
checkpoint completed cache construction and training setup on M4 Max, but its
one-step smoke was stopped without completing an optimizer step because the
workload destabilized the host. It is not validated as trainable on a 128 GiB
Mac; see the [M4 Max gate record](../benchmarks/gate-qwen4-ple-m4-max.md).

**Apple Silicon is experimental.** With `backend: transformers`, MPS uses a pageable CPU
source and MPS layer buffers; host pinning is disabled. PyTorch 2.7+ may otherwise turn
`torch.empty(device="cpu", pin_memory=True)` into an MPS tensor, charging the entire base
to the MPS allocator while `is_pinned()` is still false (#434). Soup refuses that state at
the source boundary and also disables pinning before allocation. Apple Silicon has unified
physical memory, so the CUDA capacity and throughput numbers below do not transfer: only
the MPS allocator's streamed decoder and vocabulary weights are bounded by their buffer
pools, while the CPU source still consumes unified memory. On macOS 14+ the store and compute
dtype are bfloat16 after a live one-element MPS capability probe; an older runtime falls back
to float32 explicitly. An untied float32 toy decoder is bit-exact against its resident MPS
control on Apple Silicon, including two boundary-weight loads through one slot. No claim is
made yet that streaming fits a larger model or runs faster than resident MPS training.
`backend: mlx` remains a separate, incompatible model-loading path and is rejected with
`stream_layers`.

Resident Transformers training shares the same live MPS BF16 capability probe.
SFT, DPO, GRPO/RLVR, reward modelling, and PRM use BF16 autocast on a capable
runtime; the remaining trainers retain the conservative FP32 policy until they
have task-specific Apple Silicon validation. GRPO uses local Transformers
generation rather than vLLM on MPS. PRM keeps FP32 master weights even when
autocast is BF16 because a BF16 trainable base currently triggers a fatal Metal
optimizer dtype mismatch.

The tradeoff: **1.43× slower than resident training**, measured at 0.5B — the only apples-to-apples comparison available on the reference box, because 1.5B and above cannot run resident there at all.

### NF4 streaming (`quantization: 4bit`)

Quantising the streamed base to NF4 makes the RAM store ~4× smaller. That matters for two reasons, and the second is the bigger one:

1. A bigger model fits in host RAM at all — an 8B base is ~3.6 GB of NF4 instead of ~16 GB of bf16.
2. **The store fits under the machine's page-locked memory ceiling.** Pinned host memory is what lets `copy_(non_blocking=True)` actually overlap with compute. The reference box topped out at ~7.1 GB of page-locked memory, so a 5.55 GB bf16 3B base fell back to pageable and lost overlap; the 1.43 GB NF4 store pins, and utilisation goes from 79.3% to 100%. **That ceiling was mostly Soup's own accounting, not the box's** (#901): torch's caching host allocator rounds every pinned request *up to the next power of two*, and the store used to be pinned one tensor at a time, so a store cost 1.7–1.9× its size in page-locked memory (measured: 6.82 GB of Qwen2.5-14B NF4 decoder tensors cost 11.83 GB). The store is now pinned in a few power-of-two arenas with every tensor a view — the 9.93 GB 14B NF4 store page-locks as exactly 10 GiB — and the ready line prints the real figure (`9.93 GB pinned RAM store (10.74 GB page-locked)`).

The base is quantised **once, offline**, one tensor at a time, and cached. The shard cache is keyed to the quantisation, the dtype, the quantisation device and a fingerprint of the source checkpoint, so switching `none` ⇄ `4bit` — or retraining a base in place — re-shards rather than silently streaming the wrong bytes.

Correctness is not a tradeoff here either: a streamed NF4 run is **bit-exact** against a *resident* NF4 run (the same quantised bytes through the same bitsandbytes kernels), and that is a regression test, not a one-off measurement.

**Measured numbers (RTX 3050 Laptop 4 GB, Windows 11, LoRA, batch 1, 50 steps after 10 warmup):**

| Model | Quant | Seq | Throughput | GPU Util | Peak VRAM | RAM store |
|---|---|---|---|---|---|---|
| **Llama-3.1-8B-Instruct** | **NF4** | 512 | **119.6 tok/s (pre-repair, [#361](https://github.com/MakazhanAlpamys/Soup/issues/361))** | 100% | **3.32 GB** | 3.60 GB pinned |
| Qwen2.5-3B | NF4 | 512 | 264.2 tok/s | 100% | 1.76 GB | 1.43 GB pinned |
| Qwen2.5-3B | bf16 | 512 | 143.1 tok/s | 79.3% | 2.15 GB | 5.55 GB pageable |
| Qwen2.5-1.5B | bf16 | 512 | 525.0 tok/s | 96.8% | 1.82 GB | pinned |
| Qwen2.5-1.5B | bf16 | 1024 | 487.6 tok/s | 96.7% | 2.96 GB | pinned |
| Qwen2.5-0.5B | bf16 | 512 | 978.6 tok/s | 91.4% | 1.47 GB | pinned |

**Headline:** **Llama-3.1-8B fine-tunes on a 4 GB card at 119.6 tok/s in 3.32 GB (pre-repair; re-measurement pending in [#361](https://github.com/MakazhanAlpamys/Soup/issues/361)).** For scale, 1M training tokens is ~2.3 h at 8B (arithmetic from the measured rate, not a separate measurement).

The 3B NF4-vs-bf16 rows differ by 1.85×, but attribute that to **pinning, not arithmetic** — see point 2 above. The two rows also come from different sessions, and this card's boost clock varies ~13% between sessions, so treat the factor as indicative and the mechanism as the claim.

The 3.32 GB 8B row above predates large-layer streaming: its untied, unquantised `embed_tokens` + `lm_head` both stayed resident and occupied 2.10 GB. Current code writes them as separate large-layer shards and reuses one device slot sized to the larger matrix, so an equally shaped untied pair should reclaim one matrix while a tied model keeps the same one-matrix requirement. CPU CI pins bit-exact logits for both controls. The updated CUDA peak remains to be measured on the reference RTX 3050; the historical 3.32 GB figure is not relabelled as a new measurement.

**Honest scope:**
- **RAM tier + disk overflow (v0.72.3).** `stream_source: auto` picks RAM when the store fits both dynamic free-RAM headroom and a physical-host ceiling, falls back to NVMe disk when not; SATA/HDD rejected. Correctness verified. **The read is off the compute thread**: a background reader parses each shard's header itself (no memory map) and stages `training.stream_read_ahead` layers in host RAM (page-locked where the box allows — see `stream_pin` below), so the GPU is fed while the next layer is still arriving. **Measured cold and warm against a same-day control of the source it replaces** (RTX 5070 Laptop, 2026-09-14, [record](../benchmarks/gate-971-async-nvme-source.md)).
  - **Cold — a store larger than RAM, which is what this tier is for — is 2.1–3.1x faster.** On a 36 GB 70B-shaped NF4 store at batch 1 x seq 512 a step went from 92–100 s to **30–48 s**, each pair position-matched (5.1–5.6 -> **10.6–16.9 tok/s**; 0.70–0.76 -> **1.46–2.32 GB/s** at the source), against 124.5 s / 4.1 tok/s / 0.57 GB/s for the old synchronous path measured 2026-09-12 ([earlier record](../benchmarks/probe-rtx5070-what-bounds-streaming.md)). The range is position in the run, not depth — see below. Peak VRAM is unchanged at 4.38 GB.
  - **Since #974 the reader bypasses the page cache.** Each layer's data section is read as sector-aligned byte ranges through direct I/O (`FILE_FLAG_NO_BUFFERING` on Windows, `O_DIRECT` on Linux, `F_NOCACHE` on macOS; a buffered `open` where a filesystem refuses, e.g. tmpfs — the pre-flight log says which), four ranges in parallel, into **one staging region per slot** packed into the same power-of-two pinned arenas as the RAM store. The reason is a measurement, not a preference: on that same 36 GB store, cold, every *buffered* read primitive — per-tensor `readinto` (the path this replaces), one `readinto` per byte range, any thread count — topped out at **1.2–2.9 GB/s** on the dev box's NVMe, while unbuffered reads of the same fresh layers reached **3.5–5.65 GB/s**, best at 2–4 ranges ([record](../benchmarks/gate-974-disk-tier-direct-io.md)). The earlier 1.46–2.32 GB/s at the source was that buffered ceiling. Two consequences worth knowing: the staging no longer pays the per-tensor power-of-two rounding (#901's mechanism, 1.7–1.9x measured — the 70B store's four slots page-lock as one 2 GiB arena for 1.96 GB, and the ready line prints the figure on this tier too), and a disk-tier run no longer warms the page cache for anything that runs after it. Through the same cold protocol as the bullet above (36 GB 70B-shaped NF4 store, batch 1 x seq 512), the step is now **16.2–17.3 s at both run positions** — 6.4x the synchronous path measured in the same slot (110.5 s), 2.0–2.8x the reader this replaces (34.8–45.6 s) — with the read at 62–64% of the step at 6.0–7.0 GB/s; the new reader's two positions differ by 1.07x where the old one's differed by 1.31x, because a reader that bypasses the cache gives the same number wherever it runs ([record](../benchmarks/gate-974-disk-tier-direct-io.md) §8).
  - **Warm — the whole store in the page cache — was reported as a 1.03–1.20x REGRESSION** against the synchronous source (gate-971 §5). Position-matched in both run orders it is **0.91–0.97x**, i.e. not one: the bracket was the run-order effect that record had already named for the cold fixture (#974). After the change, in one session against the reader it replaces, the new reader is 0.76x of it warm (1.58 vs 2.09 s) and 2.25x faster right after a page-cache eviction (1.79 vs 4.02 s). **If the store fits RAM, use the RAM tier** — 1.1 s against 1.8–2.0 s for either disk arm in the same session; the disk tier exists for a store that does not.
  - **It is still bound by the read, less so since #974**: per-layer read brackets were 82.6–84.9% of the cold step in gate-971 and are 62–64% after the direct-I/O reader. There is no measured read-free floor at this sequence to compare that against — the ~14 s figure is at a shorter one — so no headroom ratio is quoted.
  - **`stream_read_ahead` did not change throughput** in that record. Depths 1, 2 and 4 were indistinguishable once run order was controlled for — the same configuration measured 48.09 s run first and 30.28 s run last, as the page cache warmed across blocks, which is larger than the whole spread across depths. Treat it as a **host-memory knob**: 1.5 GB of pinned staging at depth 1, 2.0 GB at 2, 2.8 GB at 4 on a 70B. The pre-flight prints that figure on the disk tier (`host staging read_ahead N -> X MB`) and **refuses the run** when it plus the resident extras will not fit the free-RAM headroom, naming `stream_read_ahead` as the knob to lower — the embedding and an untied `lm_head` take one slot each at any depth, because they are one layer each.
  - **Disk-kind detection.** A paravirtual (virtio) disk reports `rotational=1` with no media hint, so a genuinely NVMe-backed cloud disk was misread as an HDD and refused (#365); detection now measures a bounded O_DIRECT sequential read when the rotational flag is unreliable and admits NVMe-class throughput (>= 1 GB/s), while a genuinely slow disk stays rejected. Set `training.stream_disk_kind: nvme` (or `ssd`/`hdd`) to override when detection is still wrong — the resolved value is printed beside what was detected.
- **Apple APFS disk detection.** On macOS, an APFS volume may report `Apple Fabric`
  even when its physical store is Apple's internal NVMe. Soup resolves the target
  volume to its APFS physical store and admits it only when that exact device is
  listed by `SPNVMeDataType`; an unmatched solid-state device remains `ssd`, and
  unknown hardware remains refused. `training.stream_disk_kind` still has final
  authority when explicitly set.
- **Llama / Qwen / Qwen3.5 dense and MoE text / Qwen4-Exp text / Mistral / Gemma / Gemma2 / Gemma3-Text / Phi / Phi3** (`qwen3_5`, `qwen3_5_text`, `qwen3_5_moe`, and `qwen3_5_moe_text` route through the qwen3 streamer; `qwen4_exp_text` routes through the Qwen4-Exp streamer), `task: sft`, `backend: transformers`, `modality: text`. The original list is verified bit-exact in bf16 and NF4. Qwen3.5's heterogeneous dense and MoE decoder paths are verified bit-exact against resident controls on CPU; the MoE path also has live streamed-training validation on `Qwen/Qwen3.5-35B-A3B`, whose real 35B run has no resident control because the available hardware could not load it resident. Qwen4-Exp currently has an exact float32 tiny-model parity gate, including its external PLE table; real-checkpoint and NF4 validation are still pending.
- **Heterogeneous layer keys are allowed only at the presence/absence level.** The sharder reads every layer's safetensors header and the runtime builds the RAM/disk source from those per-layer specs, then merges them into one VRAM buffer pool. A key that appears in multiple layers must keep the same stored shape and dtype everywhere; NF4 weights also keep one `NF4WeightSpec` per short key, validate every packed sidecar against the shard header, and share only the small code tables after proving they are equal.
- **Batch sizes, gradient accumulation, `--resume` / `--hf-resume`** all now work (v0.72.3).
- **Pre-Ampere cards (T4, P100, V100, GTX 16xx, RTX 20xx) now stream in fp16 instead of bf16.** Until this fix the store dtype was hardcoded to bf16 on every CUDA device, so the entire free notebook tier was streaming a dtype its GPU has no units for, and nothing said so — it could not fail on the Ampere card every number above was measured on. fp16 is bit-exact against a resident reference of matching numerics, `0.000000e+00` in both quantisations, exactly as bf16 is.
  **The capability question is asked as `torch.cuda.is_bf16_supported(including_emulation=False)`, and the keyword is load-bearing.** The bare call defaults to including emulation: when its compute-capability fast path fails it falls through to constructing a bf16 tensor, which software emulation satisfies, so **a T4 answers True**. The first version of this fix asked the bare question and was therefore a no-op on exactly the hardware it targeted — found by running the [proof notebook](../notebooks/proof-4gb.ipynb) on a real T4, not by reasoning. `get_compute_dtype` was a second copy of the same question and now delegates to the same helper.
  **Still not measured on a pre-Ampere card**: the fp16 exactness above was measured *using* fp16 on Ampere, so it establishes the plumbing, not the Turing/Pascal kernels — bitsandbytes NF4 on sm_75 in particular.
- **LoRA adapters are cast to fp32 when training streams in fp16.** peft creates the adapter weights in the base checkpoint's dtype (bf16 for Llama-3.1); on a pre-Ampere card that dtype has no bf16 units and the fp16 GradScaler raises `_amp_foreach_non_finite_check_and_unscale_cuda not implemented for 'BFloat16'` (#425). `align_trainable_dtype_for_fp16` casts the trainable `*lora_*` params to fp32 before the optimizer is created, and every trainer wrapper calls it — enforced by a scanner test rather than a hand-written list.
- **A streamed 8B run now completes on a Turing card — free-tier Colab, Tesla T4 (sm_75) — and that is all it shows.** `NousResearch/Meta-Llama-3.1-8B-Instruct`, NF4, `stream_buffers: 2`, batch 1, `max_length: 256`, LoRA r=8, fp16: 7 steps, adapter written with 128 of 128 tensors non-zero in both runs (the second ran in-process, per the notebook, so it has no exit code to quote). The first run measured peak **2.91 GB** against a predicted ~3.02 GB (the pre-flight over-predicts by 3.8%, the safe direction it was fitted for), with the pre-flight reading **free VRAM 15.10 GB** — the device, not the per-process cap. The T4 has 15.6 GB, so the process was capped to **4.00 GB** with `torch.cuda.set_per_process_memory_fraction`, and the cap was shown to bite — a 4.29 GiB allocation was refused. The committed [notebook](../notebooks/proof-4gb.ipynb) now sets `training.stream_vram_override` to that same 4.00 GB, which replaces the yardstick the fit decision is taken against — the pre-flight then reports **free VRAM 4.00 GB** instead of the driver's whole-card figure, and it is a harder gate than the first run faced. Against it: predicted ~1.97 GB, **measured peak 1.83 GB** (a +7.7% over-prediction, still the safe direction). The drop from 2.91 GB is self-consistent, not a contradiction: the embeddings and `lm_head` moved out of the resident allocation into a streamed large-layer slot between the two runs. **No throughput is quoted from either run**: a card under an artificial cap is not a benchmark, and the notebook deliberately quotes none either. **What neither establishes**: backward/gradient exactness at 8B on Turing (a non-zero adapter shows gradients flowed, not that they were correct). The notebook's streamed-vs-resident comparison is now recorded rather than unrun — bit-exact, max `|streamed - resident|` = 0.0, `torch.equal` = True — but on `SmolLM2-135M`, fp16, unquantized, not the 8B NF4 configuration this bullet is about. Record: [`benchmarks/run-t4-colab-free-tier.md`](../benchmarks/run-t4-colab-free-tier.md).
- **The bf16 3B throughput above is a LOWER BOUND.** The reference box could not page-lock the 5.55 GB base (its measured page-locked ceiling was 7.65 GB — which #901 later traced to per-tensor pinning being rounded up to powers of two, so that store was really asking for ~10 GB; it would pin today), so that run fell back to a pageable store. Pageable memory makes the host-to-device copy synchronous, which costs overlap — visible as the GPU-utilisation drop from 96.8% (1.5B, pinned) to 79.3% (3B, pageable). Soup does this fallback automatically **and prints the cost** rather than absorbing it silently. NF4 lifts this at 3B: the store drops under the ceiling and pins.
- Numbers are Windows/WDDM and therefore systematically pessimistic versus Linux. `expandable_segments:True` is silently ignored on Windows; Soup detects that and does not claim it is active.

### Forcing the pin (`training.stream_pin`)

Pinning is chosen automatically; `stream_pin` is how a config overrides that choice. **Since
#971 the flag covers both tiers**: on the RAM tier it decides whether the base store is
page-locked, and on the disk tier whether the async reader's host *staging* is.

- **Unset (the default)** attempts page-locked host memory on a CUDA target and **falls back
  loudly** if the box cannot provide it — the base store on the RAM tier, the reader's
  staging on the disk tier. The fallback names what it costs: host→device copies become
  synchronous, and measured GPU utilisation drops from **~97% to ~79%**. On the disk tier it
  also names `stream_read_ahead`, because the depth is what decides how much gets page-locked.
  Since #974 that staging is packed into the same power-of-two arenas as the RAM store (one
  region per slot, tensors as views), so it page-locks at close to its own size instead of
  the per-tensor 1.7–1.9x, and the ready line prints the figure on this tier too.
- **`stream_pin: false`** forces pageable host memory on either tier — the base store on the
  RAM tier, the reader's staging on the disk tier. The pre-flight states the throughput this
  costs rather than absorbing it silently: up to **6.56×** measured (Qwen2.5-32B NF4),
  **7.41×** on a synthetic. Those two figures were measured on the **RAM store**
  ([record](../benchmarks/gate-h100-validation.md)); the same mechanism applies to the
  reader's staging, but its magnitude there is not measured. This is also the escape hatch:
  it was the only known mitigation while #331 was live.
- **`stream_pin: true`** forces page-locked host memory and **REFUSES the run on a CUDA
  target** if the box cannot provide it, instead of degrading to pageable and spending the
  whole margin pinning exists to provide. The refusal names the store size on the RAM tier,
  and `stream_read_ahead` on the disk tier.

**Where `true` announces instead of refusing.** Page-locking needs a CUDA device to copy to.
On a non-CUDA target there is nothing to force, so the request is **inapplicable rather than
unsatisfiable** and the run *proceeds with an announcement*:

| Tier / device | `stream_pin: true` does |
|---|---|
| RAM tier on CUDA | pins the base store, or **refuses** naming the store size |
| Disk tier on CUDA (base does not fit in RAM, weights stream from NVMe) | pins the reader's staging, or **refuses** naming `stream_read_ahead`, the depth that decides how much is page-locked |
| Non-CUDA target (CPU or MPS) | announces that CUDA host pinning does not apply, proceeds with a pageable CPU source |

Refusing on a non-CUDA target would make `stream_pin: true` uncommittable to a `soup.yaml`
shared between a GPU box and a non-CUDA box. **The disk-tier row changed in #971**: that tier
used to announce that pinning did not apply and proceed, because it held nothing to
page-lock. The async reader stages whole layers in host RAM, so the flag has real semantics
there now — a `soup.yaml` carrying `stream_pin: true` that used to warn on the disk tier can
refuse instead.

Set while `stream_layers: false` the key is rejected as a footgun, like the other
`stream_*` keys.

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
             (from 6.75 TFLOPS measured on this card now using bfloat16 @ 862 MHz)
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

The throughput line is a **bound, not a promise**. It comes from a GEMM benchmarked
with the card's resolved stream dtype in that session — `bfloat16` on cards with native
BF16 support and `float16` otherwise — and is printed with the SM clock it was taken at,
because this card alone produced 3.5 and 7.6 TFLOPS in two sessions at the same reported
clock. A per-card constant compiled into Soup would be a fabrication. Real streamed runs landed at
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
- `moe_expert_quant` → expert quantization runs only in the resident model-construction path and would otherwise be silently ignored
- `unfrozen_parameters`, `lisa_enabled`, `packing`, `multipack`, `use_fsdp2_compile`, `train_router_only`, `expand_layers` → each independently rewrites or re-freezes the same layers
- `stream_source` / `stream_ngram_source` / `stream_buffers` / `stream_read_ahead` / `stream_vram_override` / `stream_vram_probe` / `stream_disk_kind` / `stream_pin` set while `stream_layers: false` → a footgun, refused
- a non-default `stream_read_ahead` beside `stream_source: ram` → refused. The read-ahead reader belongs to the NVMe disk tier and `ram` never falls back to it, so the setting would validate, be documented, and reach nothing. The default is accepted, because a default is not a decision
- a disk-tier run whose host staging (`stream_read_ahead` layers, plus one slot each for the embedding and an untied `lm_head`) plus the resident extras will not fit the free-RAM headroom → refused by name, naming the depth to lower. Refused rather than clamped: a depth you set is a decision, and silently lowering it would hand back a slower run than the one you configured
- more than 8 distinct layer shapes in one shard index → refused. Staging is allocated per shape and a shape with one member takes a full slot at any depth, so an index whose every layer differed would hold most of the model in staging at once — page-locked when the box allows it and `stream_pin` is not false, a loud pageable fallback otherwise, and a refusal only under `stream_pin: true` on a CUDA target — which is what the disk tier exists to avoid. A real model has one to three
- `stream_vram_probe` on any task other than `sft` → the probe runs a plain causal-LM step, which *is* the SFT step but is not a preference loss. Measured at one matching shape it is conservative there too (6.02 GB against a real DPO step's 5.30 GB, +13.5%), but one shape is not a validation, so it is not offered for `dpo`/`orpo`/`simpo`/`kto` yet
- an architecture outside the supported list (llama / qwen2 / qwen3, including qwen3_5_moe text aliases / qwen4_exp text / mistral / gemma / gemma2 / gemma3_text / phi / phi3) → named explicitly

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

> **peft 0.21.0 (released 2026-09-15) saved a streamed adapter EMPTY on every Soup before the #1005 fix — re-run those trainings.** peft 0.21 selects the adapter tensors to save by the prefixes it reads off `model.named_modules()`; the streaming wrapper's module names carried an `.inner.` segment its saved keys did not, so `trainer.save_model()`, every `save_steps` checkpoint and `PeftModel.save_pretrained()` wrote a 40-byte `adapter_model.safetensors` holding **zero** tensors, training reported success, and `--resume` loaded nothing. The training itself was correct — only the file is empty, and nothing in it can be recovered. Check with:
>
> ```bash
> python -c "from safetensors.torch import load_file; \
> print(len(load_file('adapter_model.safetensors')))"
> ```
>
> `0` means the adapter is lost. Fixed on `main` by PR #1010 (the wrapper now reports canonical names, so peft 0.20 and 0.21 both save every tensor; no `peft<0.21` pin); the next release carries it. Until you run a Soup with that fix, `pip install "peft<0.21"` is the workaround.
>
> Since #1011, a streamed run also checks every checkpoint and the final save: if `adapter_model.safetensors` is missing, carries the wrapper's `.inner.` keys, or holds a different number of LoRA tensors than the model trained, the run stops with a `RuntimeError` instead of reporting success. The manual check above is only needed for adapters saved before that.

**Troubleshooting:**
- **"trainable LoRA parameters remain on the meta device"** — PEFT attached an
  adapter without real storage and Soup refused the run before installing the
  streaming runtime. This guard is deliberately based on the final parameter
  state rather than on how many tensors Soup materialised: some PEFT versions
  create real adapters themselves. Include the PEFT version and the named
  parameter from the error when reporting this.
- **A streamed adapter asks for `--base` when opened** — check
  `adapter_config.json`. A healthy artifact records the exact configured model
  reference in `base_model_name_or_path`; an empty value means the adapter was
  produced by an older streaming path that lost the meta skeleton's origin.
  Passing `--base` remains a valid workaround for that existing artifact.
- **"layer streaming needs the base to fit in RAM"** — the base is larger than free RAM. Set `stream_source: auto` to fall back to the NVMe disk tier, free RAM, or pick a smaller base.
- **"base exceeds the physical RAM safety ceiling"** — `stream_source: auto` fell back to the NVMe disk tier because the RAM tier would keep the store plus resident extras above Soup's physical-host ceiling. Set `stream_source: ram` only when you want that case to refuse instead of falling back.
- **"layer streaming needs NVMe or more RAM … the detected disk is 'hdd'"** on a fast cloud disk — a virtio device reports `rotational=1` with no media hint. Detection now measures the disk when the flag is unreliable; if it still misreads yours, set `training.stream_disk_kind: nvme` to force the tier on (`ssd`/`hdd` force it off).
- **"could not page-lock the base … falling back to a PAGEABLE RAM store"** — expected on a busy machine. Training continues, more slowly. Close other applications to keep the pinned store. Since #901 the fallback also prints `after the failed page-lock: cleared the stale CUDA out-of-memory error … and released N GB of page-locked memory it left cached` — a failed page-lock leaves the CUDA runtime holding a stale "out of memory" that the run's first kernel launch would otherwise report as a real one (the report's `AcceleratorError: CUDA error: out of memory` inside `SFTTrainer.__init__`, with gigabytes of VRAM free), and the blocks pinned before the failure stay in torch's host cache until released. Both are handled before the pageable store is built. The store also pins in power-of-two arenas now, so the fallback itself is rarer: a 9.93 GB 14B NF4 store page-locks as 10 GiB where per-tensor pinning asked for ~17 GB.
- **"layer streaming does not support model_type=…"** — the supported list is llama / qwen2 / qwen3, including `qwen3_5_moe` text aliases / qwen4_exp text / mistral / gemma / gemma2 / gemma3_text / phi / phi3. Multimodal `gemma3` is excluded on purpose; use `gemma3_text`.
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
- A published 14B-on-8 GB reference benchmark — the **memory** half is done: a Qwen2.5-14B-shaped NF4 run at batch 1 x seq 384 trains end to end on an RTX 5070 Laptop 8 GB / 32 GB box with the store page-locked, measured peak 2.94 GB against a 3.39 GB prediction ([record](../benchmarks/gate-901-14b-on-8gb.md), #901). It is a synthetic *shape* with random weights, so no throughput or quality figure is quoted; a real Qwen2.5-14B-Instruct run on an 8 GB card is still wanted
- GRPO and PPO are explicitly **not** planned: rollouts need generation, which re-reads the model per token

**Disk pre-flight and shard cache.** Before Soup materialises or shards a checkpoint, it
reports the complete projected footprint: the HF/local source, any regular-file copy Soup
still needs, and the per-layer shard cache. Required writes are grouped by target volume and
the run refuses before either write when that volume lacks free space. Override the two cache
roots with `SOUP_SPECTRUM_CACHE_DIR` and `SOUP_LAYER_STREAM_CACHE_DIR`; both retain Soup's
home/cwd/tmp containment policy.

Hugging Face snapshots normally expose symlinks into their blob cache, which the sharder
deliberately does not follow. Soup materialises those weights under its Spectrum cache. If the
HF cache already exposes real files, Soup now reads them in place instead of creating a second
copy. The layer shards remain under `~/.soup/layer-stream/`. Their index records each source
filename, size, and `mtime_ns`, so a necessary re-shard says which component changed instead
of silently spending minutes rebuilding the cache.

This materialisation also works with `HF_HUB_OFFLINE=1` when the standard Hugging Face
snapshot is complete. Soup pins the commit resolved by the initial cache lookup and copies
only verified snapshot files from that commit's blob store; it does not perform a second Hub
metadata request for the regular-file directory. A missing blob or an escaping symlink aborts
before the destination is published, rather than leaving a partial checkpoint that the sharder
could consume.


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

**On the MLX backend the masking differs, and the two backends are not comparable**
([#683](https://github.com/MakazhanAlpamys/Soup/issues/683)). MLX supervises every
assistant turn through a per-token mask and **excludes the assistant header**, where
the transformers path above includes it. MLX also **refuses** a chat template whose
partial renderings are not prefixes of the full one, rather than emitting a mask that
looks plausible — the refusal happens at dataset construction, before the training
loop, and names `data.train_on_responses_only: false` as the remedy.

In practice this affects thinking-style templates: `Qwen3` injects its empty thinking
block only for the *last* assistant message, so multi-turn Qwen3 rows are refused
(single-turn rows are fine). Llama 3.1 and Gemma 3 mask correctly on both shapes.
`data.train_on_messages_with_train_field` and `data.train_on_prompt` have no MLX
equivalent and are reported in the `MLX backend ignores:` line rather than silently
dropped.

After tokenization and truncation, every response-only row must retain at least one shifted
causal-loss target. Soup rejects the row by split and row number when
`data.max_length` truncates the complete assistant response, rather than training on an
all-masked sequence or saving a non-finite adapter.

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

`soup serve` resolves this gate **once** and hands the result to whichever backend
runs. That was true of the transformers and vLLM backends from the start, and is
true of the SGLang backend as of #619 — before that its runtime and tokenizer
loaded with `trust_remote_code` hardcoded on, so the sentence above did not hold
for `--backend sglang`. A custom-code model on that backend now fails to load
without the flag where it previously loaded and ran silently.

### Chat-template hardening

Tokenizers without a chat template now raise a `ValueError` with a fix suggestion instead of silently building garbage `f"{role}: {content}"` strings.

```yaml
data:
  train: data.jsonl
  chat_template: chatml   # or: llama3, qwen2.5, mistral, gemma3, phi4, deepseek-r1, or a raw Jinja string
```

The override is installed before conversational SFT, preference, reward-model,
PPO, GRPO, or Online DPO data is rendered. The saved tokenizer keeps the same template
for inference. Tasks that do not render chat (`pretrain`, `embedding`, `classifier`,
`reranker`, `cross_encoder`, `prm`, `asr`, `moe_lora_routing`, and `unlearn`) reject
`data.chat_template` instead of silently ignoring it.

Raw Jinja strings are validated: null bytes / >64KB / filesystem-touching directives (`{% include %}`, `{% import %}`, `{% from %}`, `{% macro %}`, `{% extends %}`) are rejected at config-load.

### OOM-probe auto batch size

```yaml
training:
  batch_size: auto                  # unchanged
  auto_batch_size_strategy: probe   # NEW: 'static' | 'probe' | 'auto' (default)
```

Replaces the static memory formula with a real try-halve-then-double-to-ceiling loop. Picked size is cached at `~/.soup/batch_cache.json` keyed on `(model, max_length, quantization, lora_r, gpu_name, gpu_memory_gb)` so repeat runs short-circuit.

A candidate is refused when the step raises an out-of-memory error **or** when it completes with a measured peak (`max_memory_allocated`) above the VRAM this process can reach. The second check exists for the WDDM driver (native Windows and WSL2), where the allocator spills to host memory instead of raising: the step finishes, an order of magnitude slower, and without the measurement the probe would approve a batch that does not fit and cache it (#649). Cache entries written before that check carry a different key and are ignored.


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

### `--deepspeed <file>` — your own JSON

`--deepspeed` also takes a path to a JSON config instead of a preset name. That
file is yours: it reaches DeepSpeed **byte-identical, by the same path**, unless
it carries a key that is invalid for the run it is about to start.

Two keys are rewritten, and both are errors rather than preferences (#359):

| key | why it is repaired |
|---|---|
| `zero_hpz_partition_size` | DeepSpeed refuses a value the world size is not divisible by, so the ZeRO++ preset's placeholder `8` is invalid on any box that is not a multiple of 8 |
| `zero_quantized_weights` / `zero_quantized_gradients` | the fp16 CUDA quantiser against the `bf16` the same file enables makes the dequantised all-gather come back `c10::Half` and meet a `c10::BFloat16` activation, raising `expected mat1 and mat2 to have the same dtype` |

The documented way to customise ZeRO++ is to copy the preset JSON — which copies
both defects — so an unresolved user file would inherit a crash the presets are
already protected from. When a rewrite happens it is **printed**, the repaired
config goes to a temp copy, and **your file on disk is never modified**. A
config that uses none of those keys is not touched at all.

A malformed JSON is passed straight through: DeepSpeed reports a bad config
better than Soup can, and refusing here would reject files DeepSpeed accepts.

### DeepSpeed + LoRA

Every trainer that can be launched with `--deepspeed` prunes HF's empty no-decay
optimizer group before the LR scheduler is built (#336, extended to all wrappers
in #359). Without it, LoRA runs die at the first `lr_scheduler.step()`: every
trainable LoRA tensor is 2-D, so the no-decay group comes out empty, DeepSpeed
drops it while the scheduler keeps two `base_lrs`, and torch's strict `zip`
raises. Full fine-tuning populates both groups, so nothing is pruned there.

### `--gpus` flag: topology-aware launch

```bash
# Auto-detect local GPU count and launch under Accelerate
soup train --config soup.yaml --gpus auto

# Explicit local GPU count
soup train --config soup.yaml --gpus 4

# Print the launch command without starting training
soup train --config soup.yaml --gpus 4 --no-reexec
```

With more than one GPU, Soup detects the local NVLink / PCIe topology and
replaces itself with `accelerate launch`. Use `--no-reexec` to print a command
for manual execution instead; this advisory mode exits with status 1.
`--dry-run` validates the configuration and data without launching training.

### Multi-node launch

Run Soup on **each node**, with a different `--node-rank`. For two machines
with eight GPUs each:

```bash
# On rank 0 (replace 10.0.0.10 with its reachable private hostname or IP)
soup train --config soup.yaml --gpus 8 --nodes 2 \
  --node-rank 0 --master-addr 10.0.0.10 --master-port 29500

# On rank 1
soup train --config soup.yaml --gpus 8 --nodes 2 \
  --node-rank 1 --master-addr 10.0.0.10 --master-port 29500
```

`--gpus` is the number of GPUs **per node**. Both commands pass
`--num_processes 16 --num_machines 2` to Accelerate, which starts eight workers
on each machine. One GPU per node is also supported. `--gpus auto` detects the
local count, so use it only when every node has the same number of visible GPUs.

`--nodes` defaults to 1. With multiple nodes, `--master-addr` is required;
`--node-rank` defaults to 0 and `--master-port` to 29500. Ranks must be distinct
and between 0 and `nodes - 1`. Use the same node count, coordinator address,
and port on every machine. Soup uses static rendezvous; it does not provision
machines, copy files, or run the command on other nodes.

`--master-addr` takes an address only: an IPv4 or IPv6 literal, or a hostname.
The port belongs in `--master-port`, so `--master-addr head:29500` is rejected
rather than exported as an unreachable `MASTER_ADDR` on every node.

Install the same Soup and training dependencies on all nodes, and make the
model, configuration, and data available at the paths used by each command.
All nodes need network access to rank 0's coordinator port and to the peer
connections used by NCCL. Configure the cluster's private network and firewall
accordingly; opening only the coordinator port may not be sufficient.

Multi-node launches default to `NCCL_IB_DISABLE=0` to allow InfiniBand. On a
TCP-only cluster, set `NCCL_IB_DISABLE=1` before launching on every node.
Soup preserves existing NCCL environment settings. `--no-reexec` prints the
launch command and this advice without setting environment variables.
`--dry-run` also prints the multi-node command, then validates locally without
starting workers or changing NCCL settings. Neither mode tests peer connectivity.

Multi-node options cannot be combined with `--cloud` or `--find-lr`.

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

`use_liger` and `use_flash_attn` are read only by the SFT-family trainer
(`task: sft` or `task: tts`); setting either on any other task is rejected
at config load.

```yaml
# soup.yaml — performance options
training:
  use_liger: true            # Liger Kernel fused ops (measured 12.9% memory, 5.1% throughput); sft/tts only
  use_flash_attn: true       # FlashAttention v2/v3 auto-detection; sft/tts only
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

For each candidate size `B`, the probe runs ONE forward + backward + step on a synthetic batch of `B` sequences of length `max_length`. On an out-of-memory error, or on a measured peak above what this process can reach on the device (the WDDM spill case, #649), it halves; otherwise it doubles up to `4 × static_estimate`. The picked size is cached per `(model, max_length, quantization, lora_r, gpu)` tuple in `~/.soup/batch_cache.json` so subsequent runs skip the probe.

CPU sessions and `auto_batch_size_strategy: static` skip the probe. Synthetic batch tensors are freed before the backward pass so peak VRAM reflects the realistic training step. SFT-only this release — non-SFT trainers fall back to the static estimate.


## FSDP Shard Consolidation

```bash
# Preview the plan (which shards, total size) without writing
soup merge-sharded-fsdp-weights ./fsdp-checkpoint -o ./merged.safetensors --plan-only

# Consolidate for real
soup merge-sharded-fsdp-weights ./fsdp-checkpoint -o ./merged.safetensors
```

Consolidates `pytorch_model_fsdp_*.bin` shard files into a single `.safetensors`. Each shard is loaded one at a time (streaming, not all-at-once) with `torch.load(weights_only=True)`, tensor shapes validated (a duplicate key with a conflicting shape is rejected; a same-shape duplicate keeps the first and warns), and the merged dict written atomically. cwd-containment + symlink rejection apply to the output path and every shard; per-shard 16 GiB cap; `_MAX_SHARDS=1024`. `--plan-only` prints the plan and exits 0. Live torch-side consolidation shipped in v0.71.14.


## BitNet 1.58-Bit Export

BitNet 1.58 training is not implemented yet. Setting
`training.quantization: bitnet_1.58` is rejected at config load with an
actionable error instead of falling through to the ordinary SFT path.

Export of an existing BitNet checkpoint remains live through llama.cpp's
TQ1_0 ternary GGUF pipeline:

```bash
soup export --model ./output --format bitnet   # → TQ1_0 ternary GGUF
soup export --model ./output --format tq1_0     # same flavour, explicit name
```

The export requires a built llama.cpp toolchain; the convert/quantize binaries
raise a friendly `FileNotFoundError` when missing.


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

**Which tasks read which flag (#798).** These were accepted on every task and applied by only some, so the stored config claimed a run that never happened:

| flag | applied by | elsewhere |
|---|---|---|
| `moe_lora` | `sft`, `pretrain`, `tts`, (since #798) `dpo`, `kto`, `orpo`, `simpo`, `grpo`, (since #1099) `ipo`, `bco`, `reward_model`, `ppo`, `embedding`, `online_dpo`, and (since #1151) `distill`, `unlearn`, and `classifier`, `reranker`, `cross_encoder` with `classifier_lora: true` | refused at config load, naming the task, on `asr`, `moe_lora_routing`, and the classifier family without `classifier_lora: true` and `lora.r > 0`; still loads unread on `prm` |
| `moe_expert_quant`, `train_router_only` | `sft`, `tts` | refused at config load, naming the task |
| `moe_aux_loss_coeff` | `sft`, `tts`, `pretrain` | a **non-default** value is refused; the default `0.01` still loads, because every stored config and eleven shipped recipes write it |

**`moe_lora` on the remaining LoRA tasks (#1099).** #798 left it loading but unread on `ipo`, `bco`, `reward_model`, `ppo` and `embedding`, and `online_dpo` had the same gap. All six build their adapter through the same `build_lora_config` path, so they were wired to the same helper rather than refused. On `embedding` it applies only with `lora.r >= 1`; at `r: 0` that trainer full-fine-tunes and builds no adapter for the flag to select. #1151 closed the remainder: `distill` and `unlearn` build their adapter the same way and are wired to the same helper, and so is `classifier` / `reranker` / `cross_encoder` (one trainer) on its opt-in adapter path. Without `classifier_lora: true` and `lora.r > 0` that trainer full-fine-tunes and builds no adapter, so there the flag is refused at config load; `asr`, which trains only Whisper (no experts), and `moe_lora_routing`, which builds no LoRA adapter, refuse the flag at config load. `prm` builds no adapter either and still accepts the flag unread, pending a ruling on #1151. `preference` is covered through the trainers it dispatches to, and `tts` through the SFT trainer it subclasses.

**`moe_lora` requires `lora.dropout: 0.0` on a fused-expert MoE.** transformers 5.x keeps a Qwen3-MoE's experts as fused 3-D parameters (`mlp.experts.gate_up_proj`), which peft adapts through `lora.ParamWrapper`, and that wrapper raises `lora.ParamWrapper does not work with lora_dropout != 0.` With the schema default of `0.05` the LoRA attach failed outright, so `moe_lora` did not work on any task — including `sft`. Soup now stops at the attach with a message naming the flag, instead of letting peft's reach the user, and all 31 shipped MoE recipes pin `lora.dropout: 0.0`. The check is made against the loaded model, not at config load: whether the experts are fused depends on the checkpoint and the transformers version, and a model with one module per expert takes dropout normally. A dense base is untouched — there the flag is a no-op.

**`moe_lora` does not reach every MoE family (measured, v0.75.0).** `get_moe_target_modules` picks module names, and whether peft turns those into adapters on the fused expert parameters depends on the architecture. On tiny stand-ins with transformers 5.16.1 / peft 0.20.0:

| family | expert adapters attach | recipes |
|---|---|---|
| `qwen3_moe` | yes | 11 |
| `deepseek_v3` | yes | 6 |
| `glm4_moe` | yes | 3 |
| `minimax` | **no — attention-only** | 2 (`minimax-m3-sft`, `minimax-m3-dpo`) |
| `mixtral` | **no — attention-only** | — |
| `kimi_k2`, `mistral-large-3` | **not measured** (no stand-in builds here) | 9 |

So `minimax-m3-sft` and `minimax-m3-dpo` still train attention-only LoRA: peft has no v4→v5 conversion mapping for those model types, so their experts are never targeted and the attach succeeds quietly. Extending target resolution per architecture is #1070. The nine `kimi-k2.x` and `mistral-large-3` recipes are untested rather than known-good — no tiny stand-in for those configs exists in the installed transformers.

**`target_modules: auto` on a MoE base.** Until #1070 `resolve_lora_target_modules` had no mapping for any MoE architecture Soup ships, so `auto` resolved to `None` and peft refused with `No target_modules passed but also no target_parameters found`. Those architectures now resolve to their attention projections (see `docs/peft-and-efficiency.md`); a MoE architecture neither Soup nor peft maps is refused at setup, naming it. With `moe_lora: true` the targets come from the model scan instead, and that is applied *before* the refusal is decided, so `moe_lora` still works on an unmapped MoE such as `qwen2_moe`.


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

- `fp8_attention: true` — requires `quantization_aware: fp8` AND a non-MLX backend. Converts the attention projections (q/k/v/o and fused variants) to torchao float8 training on Ada or newer GPUs (the same gate as `quantization_aware: fp8`). A card, OS or torch build the gate refuses stops the run at setup; missing torchao degrades to a clear advisory; a conversion-phase failure raises an honest "model may be PARTIALLY converted" error instead of training on a half-converted model.
- `nvfp4: true` — Blackwell-only FP4 training via torchao `NVFP4TrainingConfig` + `quantize_`, which replaces `nn.Linear` with `NVFP4Linear` and quantises the forward **and backward** GEMMs. (Through v0.75.0 Soup asked for `NVFP4Config`, a name torchao has never exported, so the flag degraded to a yellow advisory and a bf16 run — #826.) Gated to supported tasks (rejected on `task: distill` and tasks without v0.28 speed/memory wiring) + non-MLX + `modality: text`; the SM ≥ 10 runtime check fires at trainer construction. The fix makes the flag **reach** NVFP4 training — the linears really are replaced by `NVFP4Linear`. A training step on top of that has torchao's own requirements, which Soup does not check for you: the kernels are triton, and `nvfp4_mm_triton` raises `ValueError: requires M, K, N all divisible by 128` for a shape it cannot serve (`torchao/prototype/moe_training/nvfp4_training/nvfp4_linear.py`), so a model whose hidden/intermediate sizes are not multiples of 128 will fail there rather than train slowly.
- `unsloth_bnb_4bit: true` — promotes "Unsloth Dynamic 4-bit" from an implicit `backend=unsloth + quantization=4bit` combo to a named flag. Mutual rejection of inconsistent combos at config load.

Cross-validator ordering picks the most actionable error: `quantization_aware='fp8'` prerequisite fires before the MLX rejection on `fp8_attention`, and task compatibility checks fire at config load before runtime trainer construction.


## LF / Axolotl Quant Parity (v0.53.0)

- `bnb_4bit_use_double_quant` — controls BNB's double-quantization. **Defaults to `true`** (matching every 4-bit load path — resident, layer-streaming, and the `soup merge` 4bit save formats), and is now honoured everywhere (#321): set `false` to disable it and it actually reaches BNB. Explicitly setting it requires `quantization: 4bit`; combinations with the Quant Menu formats (gptq / awq / hqq:Nbit / aqlm / eetq / mxfp4 / fp8) are rejected at config load.
- `llm_int8: true` — an explicit 8-bit assertion. Unlike v0.41.0 `load_in_8bit` (which **rewrites** `quantization` to `8bit`), `llm_int8` enforces that the user has ALSO set `quantization: 8bit`. Mismatch raises with an actionable message.
- `quantize_ref_model: true` / `quantize_reward_model: true` — extend the v0.40.5 Quant Menu wiring to the reference / reward models inside preference and RLHF training. `quantize_ref_model` accepts any task with a reference policy (`dpo / ipo / simpo / orpo / bco / kto / preference / grpo / ppo`); `quantize_reward_model` accepts `ppo / reward_model`.


## Advanced Save Formats (v0.53.0)

`soup merge --save-format 4bit` and `--save-format 4bit_forced` will write a single BNB-4bit-quantized merged checkpoint without the wasteful dequant → merge → requant cycle (unsloth `merged_4bit` recipe). v0.53.0 ships the closed allowlist + spec metadata; the live writer lands in v0.53.1.

`soup export --format torchao --quant-config <yaml>` is the planned PTQ export surface for `torchao.quantize_` + `save_pretrained`. Four schemes are allowlisted: `Int4WeightOnly`, `Int8DynActInt4`, `Float8DynActFloat8`, `NVFP4`. CASE-SENSITIVE — these are Soup's scheme names, mapped to the torchao class that implements each one in `utils/torchao_compat.py` (three of the four names Soup used to look up by `hasattr` do not exist in torchao; #826). `Int4WeightOnly` accepts `group_size`; `inner_k_tiles` was removed, because `Int4WeightOnlyConfig` raises `TypeError` for it. Diverges from `--save-format` (lowercase-normalised) on purpose; documented at both validators.


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

# AWQ/GPTQ export — an explicit local calibration set is required
soup export --model ./merged --format awq \
    --calibration-data ./calib.jsonl --output ./out-awq
soup export --model ./merged --format gptq \
    --calibration-data ./calib.jsonl --output ./out-gptq

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

Direct AWQ and GPTQ exports require `--calibration-data`. Soup refuses a missing or unusable JSONL before importing the quantizer or loading the model. This keeps calibration inputs explicit and prevents AutoAWQ from silently downloading its large default dataset. Use `--calibration-samples` to cap the number of usable JSONL rows (default: 128).

The advanced GGUF pipeline uses POSIX `O_NOFOLLOW` to defeat the TOCTOU race between the dispatch-time symlink check and the actual open of the calibration data — a crafted environment cannot race-swap the calibration file between validate and read.

`soup deploy autopilot --measure` caches results at `~/.soup/deploy_autopilot_cache.json` keyed on `(base, profile, eval-tasks)`. Repeat invocations short-circuit; pass `SOUP_DEPLOY_AUTOPILOT_CACHE=<path>` to redirect (constrained to home / cwd / tempdir). The recommended candidate uses soft-fallback: first `OK` by insertion order, else the candidate with the smallest delta (least drop relative to its own baseline).
