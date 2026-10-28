# Performance & Quantization

[← Back to the Soup README](../README.md)

> Quantization-aware training (FP8 and the experimental QuEST route), the Quant Menu (I + II), KV-cache, NVFP4, save formats, Cut Cross-Entropy, gradient checkpointing, kernel auto-composition, activation offloading, and multi-GPU / DeepSpeed / FSDP.

**Contents:**

- [Experimental CUDA graph decoding](#experimental-cuda-graph-decoding)
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

## Experimental CUDA graph decoding

`soup infer` and `soup bench infer` accept `--cuda-graphs`. On a resident Qwen2 or Llama model,
optionally with one LoRA adapter from `soup train`, generation replays CUDA graphs instead of
launching every kernel from Python. The default path is unchanged.

```bash
soup infer --model ./output --input prompts.jsonl --output answers.jsonl --cuda-graphs
soup bench infer ./model --backend transformers --max-tokens 128 --cuda-graphs
```

Measured on an RTX 5070 Laptop (Windows, unpinned, fresh processes, A-B-B-A-B-A) with
Qwen2.5-1.5B-Instruct in FP16, 128 greedy tokens per request: the slowest run with the flag was
**2.9x** the fastest run without it (60.1 vs 20.7 tok/s), and **2.8x** under an unmerged LoRA
(49.2 vs 17.8 tok/s), with identical token IDs on every request and +42 MiB of reserved VRAM.
`soup infer` over 24 prompts of varied length, model load and compile included, took 96-102 s with
the flag against 131-282 s without. Without the flag the run-to-run spread is large (13.5-20.7 tok/s):
the decode is bound by the host launching ~1,200 kernels per token, and Windows moves that thread
between the hybrid CPU's fast and slow cores; with the flag the spread is 4%. The record is
[`benchmarks/gate-v0.76.0-cuda-graph-decode.md`](../benchmarks/gate-v0.76.0-cuda-graph-decode.md).

- **Cold start is slower.** Before the first row, Soup runs one greedy warm-up on your longest
  prompt, which compiles and captures the graph (about 8-10 s on that card) and sizes the static
  cache once for the whole batch. A one-prompt run can take longer overall. The summary panel's
  `Duration` and `Throughput` start after the warm-up.
- **Outputs are the same.** The flag changes no precision, sampling or stopping rule; sampled rows
  use the temperature you asked for, and the warm-up is greedy so it consumes no sampling RNG. A
  LoRA adapter stays unmerged, exactly as without the flag.
- **Supported:** stock Qwen2/Llama classes, one CUDA GPU, unquantized weights, and at most one
  active plain LoRA adapter. Anything else (quantization, CPU/disk offload, layer streaming, DoRA
  or other LoRA variants, several adapters, flash attention, ASR) is refused before any output is
  written, and every refusal ends with `Omit --cuda-graphs`.
- **Needs PyTorch >= 2.14.** It uses private PyTorch compiler APIs validated on 2.14.0 (it fixes an
  accounting defect in the stock `cudagraphs` backend that otherwise clones every weight into the
  graph pool). Older versions, or a changed API, are refused with a one-line message.
- **Not in `soup chat` or `soup serve`.** A chat history grows every turn, and a static cache that
  grows recompiles every turn.
- **Unmeasured:** `--max-tokens` far above 128 (each K/V update costs an extra copy per step, growing
  with prompt + max_tokens), Linux/macOS, batch sizes above 1, and architectures other than Qwen2.
  Over a 1024-prompt batch the graph tree re-records about once per three rows without any eager
  fallback or memory growth.

If generation fails with the flag on, Soup stops with `Generation failed with --cuda-graphs: ...`,
keeps the rows already written, and names the way out. Run the same command without
`--cuda-graphs`.

## Quantization-Aware Training (QAT)

`training.quantization_aware` selects one of two quantization-aware training
routes, each described in its own section:

- `quantization_aware: fp8`: [FP8 training](#fp8-training-ada) on Ada or newer GPUs.
- `quantization_aware: quest`: the [experimental QuEST mixed W4/A4+A16 route](#experimental-quest-mixed-w4a4a16-route).

**`quantization_aware: true` (int8 QAT) is refused at config load, on every
task and backend ([#1222](https://github.com/MakazhanAlpamys/Soup/issues/1222)).**
It never ran quantization-aware training:

- Where it was applied (the transformers model setup of the SFT, DPO, KTO,
  ORPO, IPO, BCO, SimPO, GRPO, PPO and pretraining trainers, which
  `task: tts` and `task: preference` also run), it ran torchao's
  post-training int8 weight-only quantization over every Linear layer after
  LoRA, the adapter's own layers included, and froze them all. A LoRA run had
  zero trainable parameters, and every run tried (SFT, SFT with gradient
  checkpointing, full fine-tuning, DPO) exited with an error.
- Elsewhere it was not applied at all. `backend: mlx` ignored it with a warning
  and trained without it. `backend: unsloth` never applied it: `soup train`
  refused the combination, and `soup sweep` and `soup bench train` trained
  without it. The other tasks' trainers never read it, and neither did layer
  streaming or SFT's audio path, so those runs trained without it too.

A config that sets it now stops at load, whichever group it was in, with a
message naming `training.quantization_aware`; remove the key or set it to
`false`. The `fp8` and `quest` routes are unaffected, because they use their
own code paths. Real int8 QAT (fake quantization during training, plus an
export that quantizes the base the same way) is a separate feature and is not
available yet.

To ship a smaller model today, use post-training quantization, the default:
train normally, then quantize at export, for example with
`soup export --quant q4_k_m`.


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
For Hub models, `base_model` remains the repo ID. For a local base, new
format-v2 metadata records a `local-sha256:` identity derived from the
`config.json`, optional `generation_config.json`, standard tokenizer assets
(including `additional_chat_templates/*.jinja`), and the weight file that
Transformers selects (or its index and referenced shards). The identity uses
relative file names and content, so moving the same base keeps resume valid;
changing a selected file refuses resume. Soup checks the identity before and
after loading the model. A first resume from a format-v1 checkpoint reuses the
identity the run already computed instead of reading the base a third time
(#1199).
A local base whose `config.json` declares another
layer count or an incompatible width is refused before any file is hashed; a
value that is missing or not an integer is left to the post-load gate. The
sidecar and `config.json["soup_quest"]` contain the identity, not the source
directory.
Local models or tokenizers using custom `auto_map` code are refused by this
fingerprint route because code loaded from elsewhere would not be covered.
Local tokenizers declaring `fast_tokenizer_files` are also refused because
their versioned tokenizer JSON files are not covered by this fingerprint.

To continue training from a local QuEST artifact, set `base` to its directory
and keep `training.quantization_aware: quest`. Soup restores the artifact's
clipping table and route instead of recalibrating on the new training rows.
The sidecar must match the declaration in `config.json`; missing or inconsistent
metadata is refused. This preserves the inherited metadata format and provenance.

Format-v1 sidecars remain readable. Since they did not record a content hash,
resuming from the original base still compares its path string; that route
writes v2 metadata for subsequent checkpoints. A v1 artifact cannot
prove that a relocated base is the same one, and existing v1 sidecars may
already disclose the old local path.
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

The [post-fit development record](../benchmarks/quest-674-postfit-development-results.md)
preserves a selected research continuation with a Q/strongest-FP response-NLL
ratio of 1.038367 and paired gap of +0.078519 nat/target on the same, now
selection-spent, 704-row panel. It uses a research CE/KL adapter around native
SFT; this objective is not exposed as a public QuEST CLI recipe. The gap's paired
95% interval remains above zero, so `training_quality_validated` stays false.
The record also retains unsuccessful follow-ups and checkpoint hash bindings.

The implementation keeps FP32 masters. The Hadamard and fake-quant grid arithmetic
run under the trainer's CUDA autocast: BF16 by default on the Ampere-or-newer GPUs
this route requires. `training.auto_mixed_precision: true` is refused for QuEST
because it can select FP16, which the #674 gate did not measure. Activation
calibration always runs under BF16. Treat this as a reproducible research path,
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

Bool `true` is not a spelling of FP8: it selected the int8 path, which is refused at config load ([#1222](https://github.com/MakazhanAlpamys/Soup/issues/1222)). FP8 requires CUDA + an Ada or newer GPU (compute capability ≥ 8.9) and is rejected at config load on the mlx and unsloth backends. `soup train --dry-run` reports this machine's card as a note and exits 0, since FP8 configs are often written for another machine; a missing torchao still fails it. The `rowwise` and `rowwise_with_gw_hp` recipes run a separate torch kernel with its own limits: it needs a torch that dispatches it on the card (Ada 8.9: torch ≥ 2.7; Hopper 9.x and Blackwell datacenter 10.x: any supported torch; RTX 50-series 12.x: torch ≥ 2.8; 11.x: torch ≥ 2.10; no release through 2.14 runs it on 13.x), it is **never built on Windows**, and before torch 2.11 it is only built against CUDA 12 or newer. `tensorwise` has none of these limits. When FP8 is requested and this card, OS or torch build cannot run it, **the run stops at setup** with the reason (`FP8HardwareUnsupportedError`), before any layer is converted, on every trainer that reaches the converter; it never trains on without FP8. **A missing torchao stops the run the same way** (`FP8DependencyMissingError`, naming `pip install "soup-cli[qat]"`), on a card that could run FP8 too. So does a float8 conversion that fails: it quotes torchao's error and stops, instead of training in bf16 (#1152). Until this release the trainers printed a yellow advisory and trained in bf16 under a config that said FP8 (#835). `soup train` asks both questions, card first, before it loads the model, so it stops with the same reason and no download. SFT's text, vision and audio setups all convert the model after the LoRA adapter is attached. `stream_layers: true` refuses `quantization_aware: fp8`, `fp8_attention` and `nvfp4` at config load on every task, and so does `backend: unsloth`: neither path calls the converters (#1124). `fp8_attention` asks the same gate (#835). Wired across every transformer-backend trainer (SFT, DPO, GRPO, KTO, ORPO, SimPO, IPO, PPO, Reward-Model, Embedding, Pretrain).


## Cut Cross-Entropy (Large-Vocab Models)

Models with 128k+ vocabularies (Llama 3.1, Qwen2) materialise a huge `(batch, seq, vocab)` logits tensor that dominates VRAM. Cut Cross-Entropy computes the loss in chunks instead:

```bash
pip install "soup-cli[cce]"    # or: pip install cut-cross-entropy
```

```yaml
training:
  use_cut_ce: true   # Patches the CE kernel before model load
```

Architecture detection reads `config.model_type` first, so a local checkpoint directory (`soup merge`/`soup shrink`/`soup draft distill` output) is patched the same as a hub id; a name-based match on the last path component (`meta-llama/Llama-3.1-8B` → llama patcher) is the fallback when config resolution is unavailable. Saves 8-24 GB VRAM at common batch × seq shapes. Not compatible with unsloth (own CE kernel), mlx or layer streaming. Wired across every transformer-backend trainer (SFT, DPO, GRPO, KTO, ORPO, SimPO, IPO, PPO, Reward-Model, Embedding, Pretrain) — note that PPO has its own forward loop so cut_ce no-ops gracefully there.


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

GRPO, IPO, BCO, reward-model, online-DPO, pretrain and embedding explicitly
forward this setting to their TRL/HF training arguments: `false` disables it
and `true` enables it, including unquantized runs (`quantization: none`). These
wrappers use full checkpointing for any enabled tier.

Distillation (`task: distill`), the classifier wrapper (`classifier`,
`reranker`, `cross_encoder`) and Whisper ASR (`asr`) forward it to their HF
training arguments the same way, and unlearning (`task: unlearn`), which trains
in its own loop rather than through a `Trainer`, turns checkpointing on for its
LoRA policy model when the setting is enabled. Before, these four ignored the
setting and always trained with checkpointing off. Every enabled tier is full
checkpointing on them, and the default `false` keeps it off.


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
| `mxfp4` | 4 (stored) | Train LoRA on top of an MXFP4 pre-quantized checkpoint (for example GPT-OSS). Loaded with `Mxfp4Config(dequantize=True)`, so it trains in bf16: transformers does not train MXFP4 weights as loaded. Budget memory for the bf16 model, an estimated 2 bytes per parameter (roughly 42 GB for a 20B base). The `soup train` VRAM pre-flight checks an `mxfp4` run against that estimate, as it does a `quantization: none` run, and `--allow-oom-attempt` launches one it refuses. | — |
| `fp8` | — | Train LoRA on top of an FP8-released checkpoint. Loaded with `FineGrainedFP8Config(dequantize=True)`, so it trains in bf16: budget memory for the bf16 model, an estimated 2 bytes per parameter, not for the FP8 file. | transformers ≥ 5.17.0 |

**VRAM pre-flight.** Before the model loads, `soup train` estimates the run's
peak, weights included, and refuses a run that does not fit the card
(`--allow-oom-attempt` launches it anyway; the terms are in [Hardware-Fit
Calculator](backends-and-ops.md#hardware-fit-calculator)). The weights are priced
in bytes per parameter: `none` 2.0, `gptq` and `awq` 0.55, `aqlm` 0.5, `eetq` 1.0,
and 2.0 for `mxfp4` and `fp8`, which are dequantized on load. Under `4bit` and
`8bit` only the linear layers are quantized (0.53 and 1.02 bytes a parameter); the
embeddings, output head and norms are fp32 for training, so a small model with a
large vocabulary costs more than a flat 0.5 or 1.0 suggests. Without the model's
`config.json` the flat figures (0.5 and 1.0) are used. `hqq:*` has no figure yet, so the run starts and the pre-flight prints
`Pre-flight skipped: no memory model for quantization: hqq:4bit` (with the
format you set). No refusal there does not mean the run fits.

**Compatibility matrix.** `soup train` runs `check_quant_distributed_compat()` at
startup. HQQ / EETQ / AQLM hard-fail with FSDP and ZeRO-3 (sourced from
LlamaFactory's matrix at `quantization.py:199/211`). BNB 4-bit + FSDP resolves
`bnb_4bit_quant_storage` to the effective floating compute dtype before model
load, then aligns trainable adapter parameters to that dtype before FSDP wraps
the model. This avoids both FSDP failure modes: integer storage and mixed
adapter/storage dtypes.

**Pre-quantized + QAT.** `gptq` / `awq` / `hqq:*` / `aqlm` / `eetq` / `mxfp4` /
`fp8` all carry their own scale; combining with `quantization_aware` (`'fp8'`
or the experimental `'quest'` route) is rejected at config-load. `true` is
refused on its own, whatever the format
([#1222](https://github.com/MakazhanAlpamys/Soup/issues/1222)).

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
  # stream_nf4_scale_cache: true  # Keep expanded NF4 scales resident (4bit + double quant); see below
  # stream_recompute: auto     # Keep activations instead of the checkpoint recompute: on/off/auto; see below
  # stream_loss_chunk: 128     # Output head and loss N tokens at a time, for long contexts (sft only)
  # stream_disk_kind: nvme     # Override auto disk-kind detection: nvme/ssd/hdd; see below
```

```bash
# Layer streaming is a CONFIG key, not a CLI flag — just train normally:
soup train --config soup.yaml
```

**Model and tokenizer code.** With `training.stream_layers: true`, Soup uses
Transformers' own implementation for the model config, the meta skeleton used by
setup probes, and the streamed model. `--trust-remote-code` still reaches the
tokenizer, so a supported model can use a tokenizer supplied as custom repository
code. For example, Moonlight-16B-A3B needs that flag for its tokenizer while its
`deepseek_v3` model uses the native implementation that matches Soup's stacked
expert shards. Streaming requires a supported native model implementation; the
flag does not enable custom model code on this path. The dated Moonlight hardware
runs are recorded in the [probe report](../benchmarks/probe-rtx5070-moonlight-16b.md).
A Kimi K2 base is read the same way, through a guarded rewrite of its `config.json`
(see "Kimi K2" below).

**How it works.** LoRA adapters + their gradients + optimizer state stay resident in VRAM (they are small). The frozen base lives in CPU RAM, page-locked when the machine allows it, and is streamed: each decoder layer is copied into one of two pre-allocated VRAM buffers on a dedicated CUDA stream while the previous layer is still computing, so the load overlaps the compute. Vocabulary-sized `embed_tokens` and an untied `lm_head` use one additional shared slot: the embedding is loaded for the model input, then the same allocation is reused for the output head, whose copy is issued right after the embedding lookup (`StreamPrefetcher.head_prefetch_layer`, default `0`) rather than at the last decoder layer, and the embedding is reloaded into it at the tail of that step's backward pass so the copy overlaps the remaining compute instead of blocking the next step's first lookup. Each decoder layer is read **twice** per step — once in the forward pass and once when the backward pass recomputes it — because `dL/dx = Wᵀ · dL/dy` needs the weights to reach the layers below. That is physics, not an implementation detail, and it is why streaming costs time.

For streamed NF4 on CUDA, Soup follows bitsandbytes' own 4-bit dispatch. When
bitsandbytes selects its custom fused GEMM, Soup executes that GEMM through a
checkpoint-visible autograd Function so pooled packed weights cannot outlive their
slot ownership. When bitsandbytes selects dequantisation plus `F.linear` at larger
training shapes, Soup keeps that path and avoids adding a third dequantisation to a
checkpointed step. The fused arm reduces saved-weight VRAM but its backward still
dequantises because bitsandbytes exposes no transposed 4-bit GEMM; it is therefore a
bounded memory/time trade, not a claim that #842's full kernel ceiling is complete.

**Expert-granularity streaming (`training.stream_expert_layout: per_expert`, DeepSeek-V3 family).**

What it does:
- The sharder stores each MoE layer's routed experts in their own file, one pair of matrices per expert, NF4 per matrix under `quantization: 4bit`.
- The runtime reads an expert only when the router picks it, and again in the backward, so the layer buffers hold just the dense part (attention, norms, router, shared expert).
- The VRAM bound no longer grows with the expert count.

Measured (all in the [Moonlight-16B record](../benchmarks/probe-rtx5070-moonlight-16b.md) unless linked):
- **Numerics.** It matches transformers' eager experts bit for bit, forward and backward, on the same weights ([gate](../benchmarks/gate-expert-streaming.md)).
- **Harness steps** (seq 512): peak allocated VRAM 1,882 MiB against 3,993 MiB fused, at 17.5-18.0 s against 8.9-9.8 s per step.
- **`soup train`** (9 short rows): Soup reported a GPU peak of 1.0 GB against 3.1 GB fused, at about 42 s against 16 s per step.

It buys capacity, not speed:
- Each routed expert is read from the cache volume about three times per step (forward, recompute, backward), one at a time. Expect slower steps, and keep the cache on NVMe: Soup warns when the volume is not.
- The default stays `fused`.
- Under `quantization: 4bit` the routed experts are NF4 here but stay at the stream dtype with `fused`, so the two layouts train against numerically different bases.

What the VRAM pre-flight counts:
- the layer buffers, measured from the dense-only layer files;
- plus one routed expert in flight, shown as an `experts` line.

What the disk pre-flight counts: the cache the sharder will write in the layout you chose, sized from the source safetensors headers (see "Disk pre-flight and shard cache" below). Under `quantization: 4bit` that is the experts at NF4 here and at the stream dtype with `fused`: for Moonlight it charges 9.24 GB here and 30.60 GB fused, within 0.01% of the caches the sharder then wrote. (Before, it charged the NF4 rate to every parameter, 8.23 GB in both layouts, so a fused run could pass the check and fill the disk mid-shard.)

Limits and refusals:
- **Architecture.** Only `model_type: deepseek_v3` is admitted, which includes a Kimi K2 base that passes the guard in "Kimi K2" below (it streams as `deepseek_v3`). Any other base is refused before weights are resolved.
- **Adapters on routed experts.** `moe_lora` is refused with this layout, because a streamed expert has nowhere to keep an adapter between its visits. Adapt attention or the shared expert instead.
- **Older Soup versions.** A Soup from before this layout accepts a per-expert cache as reusable and then fails at install ("shard is missing decoder weights"). Delete the cache directory, or re-shard with `fused`, before going back to an older version.

**FP8 block-quantised bases (DeepSeek-V3 family).** Layer streaming keeps a frozen base off the card and streams it in layer by layer (layer-to-layer streaming, L2L: Pudipeddi et al. 2020). DeepSeek-V3 is published as an FP8 block-quantised checkpoint, and Kimi K2 uses the same format. Soup now shards such a checkpoint by dequantising it as it writes the layer cache, instead of refusing it. There is no config key for this: the format is detected from the checkpoint.

What the format is:
- each decoder linear weight is `float8_e4m3fn`;
- beside it, `<name>.weight_scale_inv` holds a float32 scale per 128 x 128 block;
- `config.json` declares it in `quantization_config` (`quant_method: fp8`, `weight_block_size: [128, 128]`);
- embeddings, the head, the norms and the router stay bf16.

What the sharder does:
- multiplies each FP8 weight by its block scales in fp32, cropping the last block at a ragged edge (DeepSeek's own dequantisation rule), and casts the result to the stream dtype;
- stores it by the usual rules, so under `quantization: 4bit` the same weights become NF4 that would become NF4 from a bf16 checkpoint;
- dequantises routed experts before they are stacked (`fused`) or written to their experts file (`per_expert`);
- keeps a linear that the checkpoint stores in bf16 without a scale, beside FP8 siblings, as it is stored (DeepSeek-V3's checkpoints mix the two: only the FP8 tensors carry a scale);
- works one band of rows at a time into an output already at the stream dtype, so a tensor costs its output plus a small fp32 band, and a block size past the matrix edge costs nothing extra;
- never stores the scales. The streamed model is built at the stream dtype: the config's FP8 `quantization_config` is dropped from the streamed skeleton only, so `config.json` and every resident load are unchanged.

Measured:
- **Numerics.** On a tiny DeepSeek-V3, the model streamed from the FP8 checkpoint matches a resident model holding the same dequantised weights bit for bit, in its logits, loss and LoRA gradients, in both expert layouts. Its layer cache holds the same tensors, bit for bit, as the one sharded from those dequantised weights, NF4 included ([gate](../benchmarks/gate-fp8-shard.md)).
- **Disk and RAM.** The disk pre-flight charges each FP8 weight at the rate its dequantised tensor is stored at, and nothing for the scales. So does the RAM check that runs before sharding under `stream_source: ram`, so a base whose dequantised weights do not fit in RAM is refused before anything is written.
- **A real checkpoint, CPU only.** Moonlight-16B-A3B, converted to this format, sharded in 93 s (27 layers, NF4, `fused`). The 30.60 GB cache came within 0.0002% of the pre-flight's figure (0.000115%: 35,336 bytes), and every FP8-derived tensor checked in three layers equals an independent dequantisation of the source. A GPU `soup train` comparison against the bf16 source is recorded in [`probe-rtx5070-moonlight-fp8-source.md`](../benchmarks/probe-rtx5070-moonlight-fp8-source.md) (pre-registered verdict FAIL on per-step loss and step time; see the record).

Limits and refusals:
- **Architecture.** Only bases that stream as `model_type: deepseek_v3` are admitted: DeepSeek-V3 itself, and a Kimi K2 base (`model_type: kimi_k2`) whose config passes the guard in "Kimi K2" below.
- **Other FP8 formats.** Each of these is refused, and the message names the supported format: per-tensor scales, `e5m2` weights, `compressed-tensors`, ue8m0 (power-of-two) scales, static activation scales, a missing or misshapen scale, an FP8 tensor without a scale, and a scale without its tensor. A `weight_block_size` axis outside [1, 1048576] is refused too. A spelling the config already gives away is refused at setup, before any weight is resolved or downloaded; the rest before the first byte of the cache is written.
- **Integer and packed formats.** Kimi-K2-Thinking's compressed-tensors `pack-quantized` INT4 (int32 `weight_packed`, `weight_scale`, int64 `weight_shape`) is not dequantised, and nor is any other integer quantiser; see the refusal list below.
- **Damaged values.** A weight that is NaN after dequantisation (the e4m3fn bytes 0x7F and 0xFF), a negative scale, a non-finite scale, and a weight past float16's range under a float16 stream (pre-Ampere cards; stream it in bfloat16) are refused, naming the tensor, when that tensor is read: before the cache index is written, though files of earlier layers may already be on disk (no index names them, so they are never used, and the next run shards again). A zero scale, as an all-zero block has, is kept.
- **Cache identity.** For `deepseek_v3`, `config.json` is part of the cache's source identity, as are the weight files, so a cache sharded from the bf16 checkpoint is never reused for its FP8 copy at the same path, nor the reverse.
- **After training.** The adapter is trained against the dequantised weights. Merging it or serving it loads the base through transformers' own FP8 path, which is a different numerical path from the one the adapter was trained on.

**Kimi K2 (`model_type: kimi_k2`).** Kimi-K2-Base and Kimi-K2-Instruct are DeepSeek-V3's decoder (MLA attention, 384 routed experts plus a shared one, 61 layers, about 1T parameters) in DeepSeek-V3's FP8 block format, but their `config.json` declares `model_type: kimi_k2` and an `auto_map` onto the repository's own code. transformers registers no `kimi_k2` config, and layer streaming never runs a repository's model code, so such a base used to fail with transformers' "contains custom code" error before the architecture list was even checked. Layer streaming (Soup's layer-to-layer streaming of a frozen base, L2L: Pudipeddi et al. 2020) now reads that `config.json` as plain data into transformers' own `DeepseekV3Config`, with `model_type` set to `deepseek_v3` and `auto_map` dropped (transformers does the same for the text tower of Kimi K2.5), and streams the base as `deepseek_v3`: its FP8 weights are dequantised as described above, both expert layouts are available, and `target_modules: auto` adapts the same MLA projections as on DeepSeek-V3. There is no config key; the rewrite is chosen from `config.json`.

A guard runs first, before the tokenizer is loaded. It admits the config only where transformers' class computes what K2's own `modeling_deepseek.py` computes, and otherwise refuses before any weight is resolved, naming every field that differs and its value:
- `architectures` exactly `["DeepseekV3ForCausalLM"]`, `topk_method: noaux_tc` and `scoring_func: sigmoid` (the only router K2's code implements, and the one transformers always runs);
- `n_group: 1` and `topk_group: 1`: K2 masks the expert groups it does not pick with 0.0 and transformers with -inf, which can pick different experts; with one group nothing is masked;
- `num_experts_per_tok` of at least 2 while `norm_topk_prob` is on: K2 renormalises the top-k weights only above one expert;
- `moe_layer_freq` 1, integer `n_routed_experts` and `n_shared_experts`, and `num_key_value_heads` equal to `num_attention_heads`: K2's attention ignores it, while transformers divides by it and defaults it to 128;
- every top-level field of the published Base / Instruct `config.json` present, except `auto_map` and `quantization_config`, which may be left out (`torch_dtype` may be spelled `dtype`). A missing field is not read by any row above, and transformers' class would compute with its own default for it: `rope_theta` 10000.0 where K2 publishes 50000.0, `routed_scaling_factor` 2.5 where K2 publishes 2.827;
- no top-level field outside the published Base / Instruct `config.json` except three metadata keys (`_commit_hash`, which transformers' reader adds; `_name_or_path`; `dtype`). transformers' class can read a field K2's code ignores, such as `head_dim`, `partial_rotary_factor`, a transformers-5 `rope_parameters`, `ep_size` or `rope_interleave`, and then compute differently or crash, so any such field is refused by name;
- `rope_scaling` null, or yarn holding only the keys K2's yarn reads, with `factor` and `original_max_position_embeddings` set, no zero `beta_fast` or `beta_slow`, and an `mscale` consistent with `mscale_all_dim`: otherwise the two compute different rotary frequencies or attention factors;
- `quantization_config` absent, or DeepSeek-V3's block FP8 as above.

Kimi-K2-Base and -Instruct as published pass it. Other K2 releases are out of scope for now, and the refusal says so before it lists the fields. Kimi-K2-Thinking does not pass: its weights are compressed-tensors INT4, refused as `quantization_config.quant_method='compressed-tensors'`, and its file carries 50 more fields the published one does not, each named. Kimi-K2-Instruct-0905 has the same decoder, but its file adds `ep_size`, `pad_token_id` and the serialised `PretrainedConfig` defaults, so it is refused too. Kimi K2.5 and K2.6 (`model_type: kimi_k25`, a vision-language wrapper) are not admitted either.

**The K2 tokenizer.** It exists only as code in the repository (`tokenization_kimi.py`), so a K2 run needs `--trust-remote-code`, and Soup prints its REMOTE CODE WARNING panel for it. Under layer streaming that flag reaches the tokenizer only; the config and the model are transformers' own classes. The tokenizer load is handed the config Soup already loaded, so `AutoTokenizer` does not read the config again with the flag (it would import and run the repository's `configuration_deepseek.py`; Moonlight-16B's tokenizer load did that before this change). The tokenizer also imports `tiktoken`, which Soup does not install: `pip install tiktoken`. When the tokenizer fails to load on a K2 base, Soup names both requirements before the error.

Measured ([gate](../benchmarks/gate-kimi-k2-admission.md), CPU only):
- **Config.** The published Base and Instruct `config.json` load into `DeepseekV3Config`. The gate's field-by-field check fails as registered on one of 43 fields, `rope_scaling`: transformers adds `rope_theta` and `rope_type` to it, as it does for any DeepSeek-V3 config, and every value K2 declares is kept. The result equals what `AutoConfig` builds from the same file spelled `deepseek_v3`.
- **Numerics.** On a tiny K2-like checkpoint (its `config.json` the published one with tiny dimensions: `kimi_k2`, `auto_map`, one expert group, yarn), the streamed model matches a resident `DeepseekV3ForCausalLM` on the same weights bit for bit, in its logits, loss and LoRA gradients, in both expert layouts, from a bf16 checkpoint and from its FP8 block copy.
- **The real index.** A header map rebuilt from the published Base index (no weights downloaded) has its 139,644 tensors and its exact byte total, 1,029,173,256,720 B. Over it, the disk pre-flight sizes each layer of the cache exactly as an independent count from the shapes does: under `quantization: 4bit` (bf16), **533.2 GB** with `per_expert` (533,246,633,930 B, of which the RAM tier holds 9.8 GB) and **2.04 TB** `fused` (2,039,152,160,406 B), 3.8x larger because the fused routed experts stay bf16.

Not measured:
- No K2 weights were downloaded, nothing was trained on K2, and no GPU ran. The cache sizes above are the pre-flight's arithmetic, not a cache written to disk.
- **A brief 22.5 GB CPU allocation per MoE layer.** On a full-size K2, transformers' `DeepseekV3NaiveMoe` allocates its fused experts with `torch.empty(384, 4096, 7168)` in bf16 on the CPU, about 22.5 GB of commit, before accelerate moves them to meta. This happens in the setup's NF4 / `moe_lora` probe and in `build_streamed_model`. RSS barely moves, but the commit charge needs that headroom, and a box short of it fails with a raw `DefaultCPUAllocator` error. DeepSeek-V3 had the same before this change, at about 15 GB.
- Parity with K2's own code: Soup never executes it. Read as text, it would not import under transformers 5.17: it imports `is_torch_fx_available`, which 5.17 does not define. That is an inference from reading the code, not an observed error. Its router asserts it is not training and its MoE block fails in training mode, so it is inference-only anyway; transformers' class is the training path. The guard was derived by reading it, and its numeric rows were checked against a hand transcription of its router and yarn.
- Known numeric differences from K2's own code that remain: the MoE combine (K2 sums the weighted expert outputs in fp32 and rounds once; transformers rounds each weighted output to the model dtype before accumulating, which is bf16 rounding noise), eager versus sdpa attention, and the router bias kept as a float32 buffer rather than a Parameter (float32 in the checkpoint either way).
- The checkpoint carries 61 stale `self_attn.rotary_emb.inv_freq` tensors. The sharder writes them into the layer files as stored, and nothing streams or reads them: K2's code keeps that buffer non-persistent, and transformers computes the rope from the config.

**Experimental two-read replay (API only, off by default).**
`build_streamed_model(..., two_read_experts=True)` keeps the decoder checkpoint but
does not fetch routed experts during its replay. The original forward and the expert's
own backward each fetch once. The router reuses the forward's top-k decisions while
recomputing the differentiable scores, preserving the routing-weight contribution to
the input gradient. No expert weights are retained between visits.

This requires the `per_expert` layout and still refuses routed-expert adapters. There is
no training-config switch or default change. It reaches only DeepSeek-V3-family bases:
`per_expert` admits only `deepseek_v3`, and a direct
`install_expert_streaming(..., two_reads=True)` call refuses every decoder class other
than `DeepseekV3DecoderLayer`. A layer that
`stream_recompute` switches off runs without the checkpoint and so without the replay:
it reads each expert in the forward and the backward, as the default path does there.

The replay runs the layer with the routed experts' output at zero, so the outputs of the
decoder layer, its MoE block and its routed experts differ from the forward's there. A
forward hook on one of those three modules, or a global forward hook, is therefore refused
with a `RuntimeError` when the layer's forward or recompute starts, including a hook
registered after `build_streamed_model` or between forward and backward. No hook is judged
safe by what it does, and nothing falls back to three reads. Forward pre-hooks, and hooks
on the router, attention, norms or the streamed layer wrapper, see the forward's values and
are allowed. A patched or replaced `forward` inside a two-read layer is not detected: if it
uses the MoE output's values in the backward, its gradients are wrong without an error.

On CPU, against the resident eager model and the default path, losses, `dX`, the router
gradients and every LoRA gradient are bit-identical with exactly two reads per expert visit,
including gradient accumulation, two forwards before one backward, `retain_graph`, a
no-grad forward and an exception mid-step. Hardware measurements are not part of this
change and are discussed with its design (#2170).

**`stream_nf4_scale_cache` (opt-in).** On the dequantise + `F.linear` path, a
double-quantised NF4 weight expands its 8-bit block scales to fp32 on every visit,
forward and recompute alike: two small kernels and a temporary per projection.
The base is frozen, so with `training.stream_nf4_scale_cache: true` each
(layer, weight) is expanded once and kept on the compute device; results are
bit-identical. It costs 4 bytes per 64-element block for the whole run (416 MiB on
Mistral-7B, one eighth of the packed 4-bit weights), which the pre-flight charges
up front and prints as a `scale cache` line. The fused GEMM arm expands the scales
inside its kernel and is unchanged, so on a shape bitsandbytes sends there the
cache stays empty. Needs `quantization: 4bit` with double quantization (the
default). It trades VRAM for time, and the time saved has not been measured
across machines yet, so it is off by default.

**`stream_recompute` (`on` by default).** A streamed layer normally runs its
forward twice: once in the forward and again as the checkpoint recompute in the
backward. `off` keeps every layer's activations instead. The backward still
re-reads each layer's weights into the buffer pool before it uses them, so pool
slots are never read stale (#331), and the gradients are the recompute's
(bit-identical under deterministic algorithms, measured on Mistral-7B NF4 and in
the CPU tests). On the RAM tier it measured 22% / 17% / 8.5% off the step at 512 /
1024 / 2048 tokens per step on an 8 GB card, as far as the activations fit; it
costs every layer's activations in VRAM, which the formula pre-flight does not
count (`stream_vram_probe: true` measures it, and runs after the switch). `auto`
decides at setup, before step 1: one synthetic forward+backward at the configured
`batch_size` x `data.max_length` (the worst case any batch can pad up to) measures
what one layer keeps allocated and the step's reserved peak. The last K layers then
keep their activations and the rest recompute, with K = (the smallest of 85% of
the card, free + reserved VRAM and `stream_vram_override`, minus that peak) divided
by the layer's bytes times the step's reserved/allocated ratio. It prints K and the
figures it used, and keeps the recompute (saying why) when it cannot measure. On
the disk tier `auto` keeps the recompute and `off` warns: the step is read-bound
there and skipping the recompute does not skip the backward read (no gain measured
on a read-bound 14B, worse at a high VRAM fill). `off` and `auto` are `task: sft`
only and refused beside `activation_offloading`, whose hooks would never see the
activations a retained layer keeps.

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
2. **The store fits under the machine's page-locked memory ceiling.** Pinned host memory is what lets `copy_(non_blocking=True)` actually overlap with compute. The reference box topped out at ~7.1 GB of page-locked memory, so a 5.55 GB bf16 3B base fell back to pageable and lost overlap; the 1.43 GB NF4 store pins, and utilisation goes from 79.3% to 100%. **That ceiling was mostly Soup's own accounting, not the box's** (#901): torch's caching host allocator rounds every pinned request *up to the next power of two*, and the store used to be pinned one tensor at a time, so a store cost 1.7–1.9× its size in page-locked memory (measured: 6.82 GB of Qwen2.5-14B NF4 decoder tensors cost 11.83 GB). The store is now pinned in a few power-of-two arenas with every tensor a view — the 9.93 GB 14B NF4 store page-locks as exactly 10 GiB — and the ready line prints the real figure (`9.93 GB pinned RAM store (10.74 GB page-locked)`). Since #1702 the arenas are filled largest tensor first whenever that page-locks fewer bytes: a 6.07 GB Qwen3-8B NF4 store, whose two 1.24 GB vocabulary matrices each used to open a 2 GiB arena of their own behind the decoder, page-locks 6 GiB where it took 8 GiB. A store that gains nothing is laid out exactly as before.

The base is quantised **once, offline**, one tensor at a time, and cached. The shard cache is keyed to the quantisation, the dtype, the quantisation device and a fingerprint of the source checkpoint, so switching `none` ⇄ `4bit` — or retraining a base in place — re-shards rather than silently streaming the wrong bytes.

Correctness is not a tradeoff here either: a streamed NF4 run is **bit-exact** against a *resident* NF4 run (the same quantised bytes through the same bitsandbytes kernels), and that is a regression test, not a one-off measurement.

**Scope of that comparison.** The resident control is a plain NF4 model *without* PEFT's `prepare_model_for_kbit_training`. The default resident 4-bit/8-bit SFT path calls it (`trainer/sft.py`), and it casts every non-quantised parameter to float32 — `embed_tokens`, the norms and an untied `lm_head` (checked on `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5`: all bfloat16 before, all float32 after). Streaming keeps those tensors in the store dtype (bf16 on Ampere and newer, fp16 before), so a default resident run and a streamed run start from a different numerical setup even with identical quantised blocks. One contributor-reported case (`Qwen/Qwen3-8B` at `b968826d`, RTX 4070 SUPER, bitsandbytes 0.50.2, transformers 5.17.0, peft 0.21.1, torch 2.14.0+cu126; not part of the benchmark record), with an NF4 Qwen3-8B, one 73-token text and a forward pass at step 0: the resident control without the upcast equals streaming bit for bit, while the default resident path differs (max |Δlogit| 0.52, mean 0.054). It is one forward on one text: it says nothing about backward, resume or the whole training trajectory, and the LoRA `A` initialisation also differs by construction (streaming materialises the adapters from a seeded CPU generator, `materialize_meta_adapters`; the resident path uses PEFT's own initialisation from the global RNG).

**Measured numbers (RTX 3050 Laptop 4 GB, Windows 11, LoRA, batch 1, 50 steps after 10 warmup):**

| Model | Quant | Seq | Throughput | GPU Util | Peak VRAM | RAM store |
|---|---|---|---|---|---|---|
| **Llama-3.1-8B-Instruct** | **NF4** | 512 | **119.6 tok/s at 952 MHz (pre-repair)** | 100% | **3.32 GB** | 3.60 GB pinned |
| **Llama-3.1-8B-Instruct**, post-#331 (@umran666, #1103) | **NF4** | 512 | **208.6 tok/s at 1935 MHz** (read with its clock, not directly comparable) | 100% | **2.40 GB** | 5.70 GB pinned |
| Qwen2.5-3B | NF4 | 512 | 264.2 tok/s | 100% | 1.76 GB | 1.43 GB pinned |
| Qwen2.5-3B | bf16 | 512 | 143.1 tok/s | 79.3% | 2.15 GB | 5.55 GB pageable |
| Qwen2.5-1.5B | bf16 | 512 | 525.0 tok/s | 96.8% | 1.82 GB | pinned |
| Qwen2.5-1.5B | bf16 | 1024 | 487.6 tok/s | 96.7% | 2.96 GB | pinned |
| Qwen2.5-0.5B | bf16 | 512 | 978.6 tok/s | 91.4% | 1.47 GB | pinned |

**Headline:** **Llama-3.1-8B fine-tunes on a 4 GB card at 119.6 tok/s at 952 MHz in 3.32 GB (pre-repair), and at 208.6 tok/s at 1935–1957 MHz in 2.40 GB after the #331 repair (@umran666, #1103).** The two rates are not directly comparable: the later run's clock is about twice the earlier one's, and per unit clock it is about 14% slower (59% against 68% of the same-session GEMM ceiling; large-layer streaming changed the memory pattern between the rows, so even that fraction is not like-for-like, and the shortfall is not attributed to any one cause). The row, its clock and the caveat are in [the gate record](../benchmarks/gate-v0.72.2-nf4.md). For scale, 1M training tokens is ~2.3 h at the pre-repair 119.6 tok/s (arithmetic from the measured rate, not a separate measurement).

The 3B NF4-vs-bf16 rows differ by 1.85×, but attribute that to **pinning, not arithmetic** — see point 2 above. The two rows also come from different sessions, and this card's boost clock varies ~13% between sessions, so treat the factor as indicative and the mechanism as the claim.

The 3.32 GB 8B row above predates large-layer streaming: its untied, unquantised `embed_tokens` + `lm_head` both stayed resident and occupied 2.10 GB. Current code writes them as separate large-layer shards and reuses one device slot sized to the larger matrix, so an equally shaped untied pair should reclaim one matrix while a tied model keeps the same one-matrix requirement. Under `dpo` and `kto` that saving is spent again: the output head's forward keeps a private copy of the matrix for its backward every step (about 1.05 GB for Llama-3.1-8B in bf16, charged by the pre-flight, #1049). CPU CI pins bit-exact logits for both controls. The post-#331 row above is that updated peak, measured on another RTX 3050 Laptop 4 GB: 2.40 GB allocated (2.66 GB reserved), while the pinned store grew from 3.60 GB to 5.70 GB, exactly the two 0.525B bf16 tensors (2.10 GB) that moved into it; the historical 3.32 GB figure is not relabelled as a new measurement.

**Honest scope:**
- **RAM tier + disk overflow (v0.72.3).** `stream_source: auto` picks RAM when the store fits both dynamic free-RAM headroom and a physical-host ceiling, falls back to NVMe disk when not; SATA/HDD rejected. Correctness verified. **The read is off the compute thread**: a background reader parses each shard's header itself (no memory map) and stages `training.stream_read_ahead` layers in host RAM (page-locked where the box allows — see `stream_pin` below), so the GPU is fed while the next layer is still arriving. **Measured cold and warm against a same-day control of the source it replaces** (RTX 5070 Laptop, 2026-09-14, [record](../benchmarks/gate-971-async-nvme-source.md)).
  - **Cold — a store larger than RAM, which is what this tier is for — is 2.1–3.1x faster.** On a 36 GB 70B-shaped NF4 store at batch 1 x seq 512 a step went from 92–100 s to **30–48 s**, each pair position-matched (5.1–5.6 -> **10.6–16.9 tok/s**; 0.70–0.76 -> **1.46–2.32 GB/s** at the source), against 124.5 s / 4.1 tok/s / 0.57 GB/s for the old synchronous path measured 2026-09-12 ([earlier record](../benchmarks/probe-rtx5070-what-bounds-streaming.md)). The range is position in the run, not depth — see below. Peak VRAM is unchanged at 4.38 GB.
  - **Since #974 the reader bypasses the page cache.** Each layer's data section is read as sector-aligned byte ranges through direct I/O (`FILE_FLAG_NO_BUFFERING` on Windows, `O_DIRECT` on Linux, `F_NOCACHE` on macOS; a buffered `open` where a filesystem refuses, e.g. tmpfs — the pre-flight log says which), four ranges in parallel, into **one staging region per slot** packed into the same power-of-two pinned arenas as the RAM store. The reason is a measurement, not a preference: on that same 36 GB store, cold, every *buffered* read primitive — per-tensor `readinto` (the path this replaces), one `readinto` per byte range, any thread count — topped out at **1.2–2.9 GB/s** on the dev box's NVMe, while unbuffered reads of the same fresh layers reached **3.5–5.65 GB/s**, best at 2–4 ranges ([record](../benchmarks/gate-974-disk-tier-direct-io.md)). The earlier 1.46–2.32 GB/s at the source was that buffered ceiling. Two consequences worth knowing: the staging no longer pays the per-tensor power-of-two rounding (#901's mechanism, 1.7–1.9x measured — the 70B store's four slots page-lock as one 2 GiB arena for 1.96 GB, and the ready line prints the figure on this tier too), and a disk-tier run no longer warms the page cache for anything that runs after it. Through the same cold protocol as the bullet above (36 GB 70B-shaped NF4 store, batch 1 x seq 512), the step is now **16.2–17.3 s at both run positions** — 6.4x the synchronous path measured in the same slot (110.5 s), 2.0–2.8x the reader this replaces (34.8–45.6 s) — with the read at 62–64% of the step at 6.0–7.0 GB/s; the new reader's two positions differ by 1.07x where the old one's differed by 1.31x, because a reader that bypasses the cache gives the same number wherever it runs ([record](../benchmarks/gate-974-disk-tier-direct-io.md) §8).
  - **Warm — the whole store in the page cache — was reported as a 1.03–1.20x REGRESSION** against the synchronous source (gate-971 §5). Position-matched in both run orders it is **0.91–0.97x**, i.e. not one: the bracket was the run-order effect that record had already named for the cold fixture (#974). After the change, in one session against the reader it replaces, the new reader is 0.76x of it warm (1.58 vs 2.09 s) and 2.25x faster right after a page-cache eviction (1.79 vs 4.02 s). **If the store fits RAM, use the RAM tier** — 1.1 s against 1.8–2.0 s for either disk arm in the same session; the disk tier exists for a store that does not.
  - **It is still bound by the read, less so since #974**: per-layer read brackets were 82.6–84.9% of the cold step in gate-971 and are 62–64% after the direct-I/O reader. There is no measured read-free floor at this sequence to compare that against — the ~14 s figure is at a shorter one — so no headroom ratio is quoted.
  - **`stream_read_ahead` did not change throughput** in that record. Depths 1, 2 and 4 were indistinguishable once run order was controlled for — the same configuration measured 48.09 s run first and 30.28 s run last, as the page cache warmed across blocks, which is larger than the whole spread across depths. Treat it as a **host-memory knob**: 1.5 GB of pinned staging at depth 1, 2.0 GB at 2, 2.8 GB at 4 on a 70B. The pre-flight prints that figure on the disk tier (`host staging read_ahead N -> X MB`) and **refuses the run** when it plus the resident extras will not fit the free-RAM headroom, naming `stream_read_ahead` as the knob to lower — the embedding and an untied `lm_head` take one slot each at any depth, because they are one layer each.
  - **Disk-kind detection.** On Linux, detection follows the requested path's device through `/sys/dev/block`, including partitions and device-mapper/RAID members; a mixed volume uses its slowest member. An unrelated NVMe disk does not qualify the cache volume. A paravirtual (virtio) disk reports `rotational=1` with no media hint, so a genuinely NVMe-backed cloud disk was misread as an HDD and refused (#365); detection measures a bounded O_DIRECT sequential read on the requested path when the rotational flag is unreliable or the device cannot be attributed, and admits NVMe-class throughput (>= 1 GB/s), while a genuinely slow disk stays rejected. Set `training.stream_disk_kind: nvme` (or `ssd`/`hdd`) to override when detection is still wrong — the resolved value is printed beside what was detected.
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
- Numbers are Windows/WDDM and therefore systematically pessimistic versus Linux. `expandable_segments:True` is silently ignored on Windows; Soup detects that and does not claim it is active. On Linux (WSL2 included) `soup train` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for a streamed run before it detects the device, so the hint takes effect; a value you exported is kept, and when that value is why the hint is off, the `expandable_segments allocator hint not enabled` line says `PYTORCH_CUDA_ALLOC_CONF is set to something else` (#2156).

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

The same term decides what an 8B base needs on a 4 GB card. Llama-3.1-8B NF4 (vocab
128 256), streamed, LoRA r=8: **3.448 GB at batch 1, S=512**, then **1.063 GB for every
further 512 tokens per step** — 4.512 GB at 1024, already above the 4.294 GB card
([probe v0.73.0, section 9](../benchmarks/probe-v0.73.0-what-bounds-streaming.md#9-peak-vram-and-the-microbatch-sweep)).
`data.max_length` defaults to 2048, so the "8B on 4 GB" setup needs `data.max_length: 512`.
`training.stream_loss_chunk` (below) is built to stop the logits term growing with the length; no chunked
run of that setup has been measured.

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

**Long contexts: the head and the loss in chunks (`training.stream_loss_chunk`, #1693).**
The logits term is `vocab × seq × batch`, so it is what caps the context length on a small
card. `training.stream_loss_chunk: 128` runs the output head and the loss 128 tokens at a
time: each chunk runs the head, takes its cross-entropy and the gradient on its own slice of
the hidden states, and drops its logits before the next chunk starts. The logits term the
pre-flight charges becomes `vocab × min(seq, chunk) × batch`. By the pre-flight's own
arithmetic (the shipped 14 bytes per element, the 128 256-token vocabulary above, batch 1)
that is an estimated 0.23 GB at a chunk of 128 whatever `data.max_length` is, against
0.92 GB at 512 tokens, 1.84 GB at 1024 and 3.68 GB at 2048 unchunked. These are estimates,
not measurements: no run with the option has been measured on a GPU (see the end of this
section). The option takes an explicit number of tokens from 16 to 8192; there is no default
and no `true`, and unset is off.

**What it costs.** Two `[batch, seq, hidden]` tensors in the stream dtype, which the
pre-flight adds to its estimate: the gradient on the hidden states, filled chunk by chunk in
the forward and handed to the backward, and its product with the incoming gradient. At
2048 tokens, hidden 4096 and bf16 that charge is 34 MB for the two. They are not live at the
same time as a chunk's logits, so charging both beside the logits term errs high. And a loop
over the head, one matmul per chunk instead of one over the sequence; what that does to speed
has not been measured.

**What it gives up.**

- `outputs.logits` on a labelled forward: the output carries the loss and no logits. The
  trainer's `entropy`, `mean_token_accuracy` and `num_tokens` columns and the rewind log's
  per-row losses (`rewind.jsonl`) are written as before, under the same names, from what the
  chunk loop collects. A forward without labels (generation, `predict`) is unchanged; a
  labelled forward without gradients (the eval loss) is chunked too.
- Equality to the last bit with an unchunked step. The per-token terms are the ones
  `ForCausalLMLoss` computes; the order of the final sum and the shape of each matmul are
  not. `tests/test_stream_loss_chunk.py` holds the loss, the gradients, the logged columns
  and the rewind log's losses to a relative `1e-5` of the same run unchunked, on a tiny
  Llama. The bit-exactness the streamed path is published with does not extend to a run with
  this option.
- A trainable output head: the gradient is taken on the hidden states only.

**The pre-flight** charges the logits on `min(seq, chunk)` tokens at the shipped bytes per
element, plus the two tensors above; the panel line ends
`(logits 0.23 GB in chunks of 128 tokens)` at batch 1 with the vocabulary above. That charge
is arithmetic. Of the 14 bytes per element, 12 are the loss arithmetic measured on
unchunked steps and 2 are charged for a retained copy of the logits whose cause is not
identified (#327); whether a chunked step still pays those 2 is not known. The hook is on the model before `stream_vram_probe` runs, so with
`stream_vram_probe: true` the step that decides the fit is the chunked one.

**Refused.** At config load, as listed under "Rejected at config load" below: without
`stream_layers`, for a task other than `sft`, with `data.format: raft`, and beside
`use_cut_ce`. When the model is built, with the reason: a trainable output head, a model
without one, and a model that changes its logits after the head: Gemma 2's
`final_logit_softcapping` is read from its config, and any other such model is caught at its
first labelled forward.

**Not measured yet.** #1693 asks for peak VRAM at 512, 1024 and 2048 tokens with and without
the option, on one model and stack, and for a pre-flight that does not under-predict on those
rows. That table needs a CUDA card and is left for whoever has one. To produce it:

```bash
python benchmarks/harness/stream_loss_chunk_vram.py --base <id-or-path> --quantization 4bit
```

The harness runs the setup `soup train` uses and one step per arm at batch 1, each arm in a
process of its own as each `soup train` is (`--seqs` and `--chunk` change the lengths and the
chunk, 512,1024,2048 and 128 by default), and prints one row per length and arm: the
pre-flight's prediction for that run, the free VRAM it compared against, and
`torch.cuda.max_memory_allocated()` over the step. An arm that runs out of memory is a row,
not a crash. It exits 1 when a row under-predicts, including an arm that ran out of memory
inside the predicted fit. `--self-test` runs the same protocol on a tiny local Llama with no
GPU and no download, and reads no peak.

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

**Saving and resuming boundary adapters (#1048 item 1).** LoRA targets on the
streamed `embed_tokens` or `lm_head` of an untied checkpoint support names, regexes
and dotted paths, default PEFT saves, Trainer checkpoints and resume. Adapter-only
`save_embedding_layers=False` and explicit `True` exports also work. A tied checkpoint
keeps both weights resident and follows the normal PEFT path. This completes the path
protected by the build-time guard from #2082 by @ybai08.

Saving a streamed model's state exports an independent CPU snapshot only for an input
embedding or output head with a LoRA target, read directly from its shard cache. An
untargeted boundary stays on meta without opening its shard. The shared streaming slot
and the meta parameters remain unchanged, so this adds no persistent GPU base allocation.
PEFT obtains the state before deciding which embedding weights to save; even an
adapter-only save therefore temporarily needs CPU memory for each adapted boundary matrix.
Keep the run's matching shard cache available through saving and resume. Loading an
adapter through `load_adapter` defaults to the existing runtime's device and device
map, so a CPU runtime stays on CPU even when another accelerator is available.
PEFT placement overrides must stay consistent with the runtime's own buffers;
automatic redistribution cannot relocate those buffers.
Loading an included frozen boundary weight that differs from that cache is refused with a base/cache
diagnostic rather than silently ignoring it. The lower-level streaming builder also
refuses `modules_to_save` on these frozen boundaries; Soup does not expose that PEFT field.
CPU tests with PEFT 0.20.0 and 0.21.0 cover exact default-adapter reloads and SFT/DPO
checkpoint continuation, including DPO's frozen `ref` adapter. Creating that reference
uses a model-scoped injection bridge that restores streaming wrappers even if PEFT fails.
Loading weights into an arbitrary new named adapter remains unsupported. These checks
establish CPU correctness, not CUDA performance or memory measurements.

**Rejected at config load (each names the release that lifts it):**
- `batch_size: "auto"` → OOM-probes a resident model that streaming never loads; explicit batch sizes allowed (v0.72.3)
- `quantization` other than `none` or `4bit` → other formats cannot be streamed into a pooled buffer
- `backend: unsloth` / `backend: mlx` → streaming replaces the model-load path those backends own
- `task` other than `sft` / `dpo` / `orpo` / `simpo` / `kto` → named explicitly. `grpo` and `ppo` are refused **permanently**, not pending: generation rollouts re-read every layer once per generated token, which destroys the amortisation streaming depends on
- `task: kto` with `batch_size: 1` → TRL's KL term is degenerate at batch 1; refused when the config is read (this applies to every KTO run, not only to streaming) rather than minutes later after sharding
- `lora.use_dora` / `lora.use_vera` / `lora.init_strategy` other than `random` → these initialise from the real base weight, which is on the meta device under streaming
- `moe_expert_quant` → expert quantization runs only in the resident model-construction path and would otherwise be silently ignored
- `unfrozen_parameters`, `lisa_enabled`, `packing`, `multipack`, `use_fsdp2_compile`, `train_router_only`, `expand_layers` → each independently rewrites or re-freezes the same layers
- `use_cut_ce` is refused with `stream_layers` (#1206): streamed setup does not install the resident CCE patch. Disable CCE to stream with the conservative full-logits budget, or disable streaming to use resident CCE. A tiny local streamed Llama was exercised through sharding, forward/loss and backward on CPU; this is no CUDA memory measurement.
- `use_liger` / `use_flash_attn` → **not reached at all** under streaming (#1692). Both are read by the *resident* setup: Liger patches the MLP forward there, and flash-attn is handed to `from_pretrained` as `attn_implementation`. A streamed model is built by `from_config` with no attention argument and no Liger patch, and `_liger_applied` is never set, so `use_liger_kernel` misses `TrainingArguments` too. These are the switches you reach for when a streamed run will not fit, so they are refused rather than dropped; each is offered again once it is wired and shown to engage on a streamed model. On every task outside `{sft, tts}` they are already refused at config load (#806)
- `stream_source` / `stream_ngram_source` / `stream_buffers` / `stream_read_ahead` / `stream_vram_override` / `stream_vram_probe` / `stream_nf4_scale_cache` / `stream_recompute` / `stream_expert_layout` / `stream_disk_kind` / `stream_pin` / `stream_loss_chunk` set while `stream_layers: false` → a footgun, refused
- `stream_expert_layout: per_expert` with `moe_lora` → refused: a routed expert streamed one at a time has nowhere to keep an adapter. On a base that is not `model_type: deepseek_v3` → refused at setup, before the weights are resolved
- an FP8 checkpoint in any format other than DeepSeek-V3's block FP8, or an FP8 base that is not `model_type: deepseek_v3` → refused naming the supported format: at setup when `config.json` gives it away, otherwise before the cache is written. The block format itself is dequantised at shard time (see "FP8 block-quantised bases" above)
- any other quantised checkpoint → refused by name, before a byte of the cache is written, and in the same words by the sharder, the disk pre-flight and the setup. The sharder checks the source first, from its `config.json` and headers, so a refusal creates no cache folder, leaves an existing cache and its `index.json` untouched, and never reuses a cache an older Soup sharded from such a source. Layer streaming reads plain floating-point weights, DeepSeek-V3's block FP8 and, on qwen4_exp only, oMLX/oQ affine bundles; the message says which format it found, that layer streaming cannot dequantise it, and to point `training.base` at a bf16 or FP8-block copy. Refused:
  - a `config.json` `quantization_config` whose `quant_method` is not `fp8`: `compressed-tensors` in any format (Kimi-K2-Thinking's `pack-quantized` INT4 among them), `gptq`, `awq`, `bitsandbytes` (or `load_in_4bit` / `load_in_8bit` with no method), `hqq`, `aqlm`, `eetq`, `torchao`, `quanto`, `fbgemm_fp8`, `mxfp4`, `mxfp8`, `nvfp4`, `gguf` and every other quantiser transformers registers, or a name it does not. Also read from `text_config` when the top level has none, and a value that is not a mapping (a string, a list, a number) is refused as malformed. `quant_method` is matched exactly, as transformers matches it: `FP8` is not `fp8`. Refused at setup, before any weight is resolved or downloaded. A config with no `quant_method` (the oMLX/oQ spelling, `bits` and `group_size` only) is decided by its tensors;
  - a tensor stored as an integer or packed type (I8, I16, I32, I64, U8, U16, U32, U64, BOOL, F4), which a plain read would cast to floats and train on: GPTQ/AWQ `qweight`, compressed-tensors `weight_packed`, bitsandbytes uint8 and int8 weights, oMLX/oQ uint32 words on an architecture other than qwen4_exp;
  - a floating-point tensor named like a quantiser's companion: `weight_scale` (compressed-tensors; DeepSeek-V3's block FP8 spells its scale `weight_scale_inv`), `weight_global_scale`, `weight_zero_point`, `input_scale`, `qzeros`, `g_idx`, `SCB`, `absmax`, `quant_map`, `W_q`, `codebooks` and the like.

  Not refused: the integer buffers a supported model really carries, which are qwen4_exp's PLE n-gram hashing tables (`layer_multipliers`, `ngram_heads_vocab_sizes`, `ngram_heads_offsets`, int64; the runtime keeps the skeleton's own values) and `position_ids`. The supported families' own checkpoints carry no other integer tensor (pinned by a tiny checkpoint of each in `tests/test_stream_refuses_packed_quant.py`). Dequantise any other source to bf16 first
- `stream_nf4_scale_cache` without `quantization: 4bit` and double quantization → refused: there are no nested NF4 scales to cache
- a non-default `stream_read_ahead` beside `stream_source: ram` → refused. The read-ahead reader belongs to the NVMe disk tier and `ram` never falls back to it, so the setting would validate, be documented, and reach nothing. The default is accepted, because a default is not a decision
- a disk-tier run whose host staging (`stream_read_ahead` layers, plus one slot each for the embedding and an untied `lm_head`) plus the resident extras will not fit the free-RAM headroom → refused by name, naming the depth to lower. Refused rather than clamped: a depth you set is a decision, and silently lowering it would hand back a slower run than the one you configured
- more than 8 distinct layer shapes in one shard index → refused. Staging is allocated per shape and a shape with one member takes a full slot at any depth, so an index whose every layer differed would hold most of the model in staging at once — page-locked when the box allows it and `stream_pin` is not false, a loud pageable fallback otherwise, and a refusal only under `stream_pin: true` on a CUDA target — which is what the disk tier exists to avoid. A real model has one to three
- `stream_vram_probe` on any task other than `sft` → the probe runs a plain causal-LM step, which *is* the SFT step but is not a preference loss. Measured at one matching shape it is conservative there too (6.02 GB against a real DPO step's 5.30 GB, +13.5%), but one shape is not a validation, so it is not offered for `dpo`/`orpo`/`simpo`/`kto` yet
- `stream_loss_chunk` on any task other than `sft`, or with `data.format: raft` → the chunked step rebuilds the causal-LM cross-entropy and returns no logits. A preference loss reduces the logits to per-token log-probs instead, and RAFT's weighted loss reads all of them (#1693)
- `stream_loss_chunk` beside `use_cut_ce` → each replaces how the loss over the logits is computed. Refused before the `use_cut_ce` refusal above, with a message that names both fields (#1693)
- an architecture outside the supported list (llama / qwen2 / qwen3, including qwen3_5_moe text aliases / qwen4_exp text / mistral / gemma / gemma2 / gemma3_text / phi / phi3 / deepseek_v3) → named explicitly. `deepseek_v3` is the DeepSeek-V3 family (MLA attention, routed experts plus a shared expert); a Kimi K2 checkpoint uses the same decoder under `model_type: kimi_k2` and is admitted through the guard in "Kimi K2" above. Under `quantization: 4bit` its attention, shared-expert and dense-MLP projections are NF4 while the routed experts stay at the stream dtype: they are 3-D parameters, not `nn.Linear` modules, so a DeepSeek-V3 layer streams most of its bytes unquantised
- a Kimi K2 config (`model_type: kimi_k2`) that transformers' DeepSeek-V3 class would compute differently from K2's own code → refused before any weight is resolved, naming every such field and its value (Kimi-K2-Thinking: `quantization_config.quant_method='compressed-tensors'`). The fields, and why each matters, are under "Kimi K2" above

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
- Correctness is not a tradeoff: streamed and resident forward passes were verified **bit-exact**, and a 100-step streamed loss curve matched resident exactly. Streaming substitutes the same weight bytes into the same kernels. (The resident control is the plain NF4 one described in the scope note under *Correctness* above.)

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
> `0` means the adapter is lost. Fixed on `main` by PR #1010 (the wrapper now reports canonical names, so peft 0.20 and 0.21 both save every tensor; no `peft<0.21` pin); v0.75.1 and every later release carry it. Until you run a Soup with that fix, `pip install "peft<0.21"` is the workaround.
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
- **"layer streaming does not support model_type=…"** — the supported list is llama / qwen2 / qwen3, including `qwen3_5_moe` text aliases / qwen4_exp text / mistral / gemma / gemma2 / gemma3_text / phi / phi3 / deepseek_v3. Multimodal `gemma3` is excluded on purpose; use `gemma3_text`.
- **"Kimi K2 (model_type 'kimi_k2') streams through transformers' own DeepseekV3 class …"** — your K2 `config.json` sets a field that transformers' class would compute differently from K2's own code, leaves out a published field, or adds one; the message names each one and its value. Kimi-K2-Base and -Instruct as published pass. Other K2 releases (Kimi-K2-Instruct-0905, Kimi-K2-Thinking) are out of scope for now, and Kimi-K2-Thinking's compressed-tensors INT4 is not dequantised by layer streaming.
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
reference costs no extra weights. On an untied checkpoint, `dpo` and `kto` also hold
one copy of the output head per step (about 1.05 GB for Llama-3.1-8B in bf16), which
the VRAM pre-flight charges (#1049); tied checkpoints and `sft`/`orpo`/`simpo` pay
nothing. Measured on an RTX 3050 4 GB with a 730 MB model:

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
loss on a tied checkpoint; on an untied one, that check covers `dpo`
(`tests/test_issue1049_untied_streamed_preference.py`). Same resident-control scope as the note under *Correctness* above.

**The cost is time, not memory.** DPO runs the layer stack three times per step (policy
forward, reference forward, checkpoint recompute) against SFT's two — measured **1.52×**
the layer reads on a 24-layer model. Streaming adds no second copy of the weights for the
reference (on an untied checkpoint `dpo` and `kto` still hold one output-head copy, see
above); it does not make the reference free.

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
- A published 14B-on-8 GB reference benchmark — the **memory** half is done: a Qwen2.5-14B-shaped NF4 run at batch 1 x seq 384 trains end to end on an RTX 5070 Laptop 8 GB / 32 GB box with the store page-locked, measured peak 2.94 GB against a 3.39 GB prediction ([record](../benchmarks/gate-901-14b-on-8gb.md), #901). It is a synthetic *shape* with random weights, so no throughput or quality figure is quoted; a real Qwen2.5-14B-Instruct run on an RTX 3070 8 GB, reported by @hasheng on #901, pinned the 9.93 GB store (10.74 GB page-locked), measured a 2.92 GB peak against a 3.39 GB prediction, and trained 18 of 18 steps
- GRPO and PPO are explicitly **not** planned: rollouts need generation, which re-reads the model per token

**Disk pre-flight and shard cache.** Before Soup materialises or shards a checkpoint, it
reports the complete projected footprint: the HF/local source, any regular-file copy Soup
still needs, and the per-layer shard cache. Required writes are grouped by target volume and
the run refuses before either write when that volume lacks free space. With
`SOUP_LAYER_STREAM_STRIPE_DIRS` set, each root is charged the bytes it will receive
(the primary root also holds the embedding, head and extras); when a header cannot be
read or the model is qwen4_exp the split is even and the panel says approximately.
Override the two cache roots with `SOUP_SPECTRUM_CACHE_DIR` and
`SOUP_LAYER_STREAM_CACHE_DIR`; both retain Soup's home/cwd/tmp containment policy.
A shard cache Soup will reuse is reported at the size of its own files, on every drive it is
striped over. One it has to write is sized from the source safetensors headers (shapes and
dtypes only: no weight is loaded), tensor by tensor, by the rules the sharder writes with:
NF4 for the decoder weights the meta probe converts to 4-bit under `quantization: 4bit`,
the stream dtype for norms, embeddings, the head, an MoE router and fused routed experts,
float32 for the layer buffers the runtime loads back, plus an allowance for the file headers
and the index. A snapshot that still has to be copied is read where the copy step reads it.
A checkpoint the sharder would refuse (more than 512 decoder layers, a missing layer, an
incomplete expert, one tensor spelled twice, an FP8 spelling it does not dequantise) is
refused here, in the sharder's words, before anything is copied. An FP8 block-quantised
source is charged at the rate its dequantised tensors are stored at, and nothing for its
block scales, which the sharder consumes. When the files cannot be listed or a header cannot be read, the cache is
charged the NF4 rate for every parameter and the panel says approximately; so does a
qwen4_exp base that is not an oQ checkpoint, which keeps its source-size figure.

This raises the figure for a dense base under `quantization: 4bit`, because the embeddings,
an untied head and the norms stay at the stream dtype rather than NF4: Llama-3.1-8B goes from
4.14 GB to 5.70 GB and Llama-3.2-1B from 0.64 GB to 1.03 GB (derived from their shapes), and
Mistral-7B-Instruct-v0.3 from 3.74 GB to 4.14 GB, against a 4.14 GB cache measured on disk.
A run that passed the pre-flight before by less than that margin is now refused: the old
figure was low, and the sharder would have written the larger amount. The early RAM check
(`stream_source: ram`, before sharding) charges the same estimate, read once: the decoder-layer
files, the embedding and head and the resident extras, which is what the check after sharding
measures on the real store (the `per_expert` experts files stay on disk on every tier and are
not charged). It refuses by that check's rule, 70% of free RAM or 55% of physical RAM, so a
base the exact check would refuse is refused before anything is sharded. When the headers
cannot be listed or read, it charges the flat NF4 rate on the size on disk and its refusal
says approximately. `stream_source: auto` is not checked before sharding: it picks its tier
after sharding, from the real store.

**Two or more NVMe drives.** On a cold store larger than RAM the disk-tier step can wait on the
read (measured for a 70B-shaped store), and then one drive is the ceiling. Set
`SOUP_LAYER_STREAM_STRIPE_DIRS` to extra folders on OTHER NVMe drives (`os.pathsep`-separated:
`;` on Windows, `:` elsewhere) and the layer cache is striped over N roots, where N is the primary
cache root plus the folders listed: decoder layer `i` lives on root `i mod N`, the index, the
extras and the embedding/head files stay on the primary root, and the reader keeps one layer in
flight per drive. The cache is laid out this way whichever tier the run picks; only the disk tier
reads the drives in parallel. Every entry is checked, and any failure refuses the run and names
the entry and the rule: it must be an absolute path (on Windows, with a drive letter or UNC share;
no `\\?\` or `\\.\` spelling, no control characters), already exist as a folder (a drive that is
not mounted is never mistaken for an empty one), not be a symlink or a directory junction (a
genuine Windows volume mount point is accepted — a junction and a mount point carry the same
reparse tag, but only a junction's target names a folder, under a drive letter, a UNC share or
even a volume GUID path, that can be repointed at any time; a mount point's target names a
whole volume rather than a folder), not lie inside or contain the
primary cache root or another entry, sit on a different volume from the primary root and from
every other entry, and classify as NVMe (`training.stream_disk_kind` overrides a wrong probe); the
list holds at most 6 folders. The checks that need no disk probe run when `soup train` or
`--dry-run` starts, and `soup doctor` lists each entry, with `--disk` for the NVMe rule. The bytes
are exactly the ones the single-drive cache holds, so striping changes the speed, not the result.

Stripe folders are outside the home/cwd/tmp containment the primary cache has, by design — a
second drive is never under `$HOME`. What that containment gave is enforced directly instead.
Inside each entry Soup writes only its own folder, named after the model plus a hash of the
primary cache's path, so two primary caches (two users, or two `SOUP_LAYER_STREAM_CACHE_DIR`
values) sharing one stripe drive never touch each other's files. That folder is made private to
the account running Soup (mode 0700 on POSIX; on Windows a protected ACL for that account and
SYSTEM, replacing the inherited one, which on a data drive typically lets every signed-in
account modify files), checked again on every reuse and again before every per-layer write and
the marker write of a shard, and never followed if it is a link or junction; a folder Soup
cannot make private refuses the run by name. A root renamed out from under a running shard and
replaced with a link is refused at the next write, not just at the start — but that only
narrows the window to a single write, it does not close it, which would need writes relative
to an open directory handle (the standard library has none on Windows). Soup never deletes anything
inside a stripe folder. Soup makes only its own folder owner-only, so also make the stripe ROOT
itself writable only by your own account (on a default Windows data drive other accounts can
modify it).

It costs one more layer of host staging per extra drive: `stream_read_ahead` defaults to N + 1.
A configured value of 2 counts as the default and is raised the same way; any other value is
kept as set (with a warning if it leaves drives idle), and a refusal that asks for a lower depth
names a value that is not raised back up. Changing the list re-shards the cache; unsetting it
re-shards to one root and names the folder left behind. Measured on two PM9B1 drives, with no
formal verdict: 7.65-9.15 GB/s together against ~4 for one
([record](../benchmarks/probe-rtx5070-two-drive-read.md)). Gated: on a cold 70B-shaped NF4 store
at seq 512 the training step went from 17.4 s to 9.97 s, 1.75x
([gate](../benchmarks/gate-two-drive-striping.md)). That is a burst from a rested box. Under
sustained back-to-back reading both drives of this laptop throttle to about 2.5 GB/s each, and
the late-window speed-up measured 1.28x and 1.41x; a single drive throttles later and less
([sustained probe](../benchmarks/probe-rtx5070-two-drive-sustained.md),
[per-drive check](../benchmarks/probe-rtx5070-drive-throttle.md); neither has a formal verdict
yet). Plan a long run on roughly 1.3-1.4x, and on drives with good cooling.

Hugging Face snapshots normally expose symlinks into their blob cache, which the sharder
deliberately does not follow. Soup materialises those weights under its Spectrum cache. If the
HF cache already exposes real files, Soup now reads them in place instead of creating a second
copy. The layer shards remain under `~/.soup/layer-stream/`. Their index records each source
filename, size, and `mtime_ns`, so a necessary re-shard says which component changed instead
of silently spending minutes rebuilding the cache.

`base:` may also be a local path to a Hugging Face cache snapshot
(`.../models--org--name/snapshots/<commit>`). Soup copies it to regular files and reuses that
copy while the commit and blob ids match. When the path is the folder the Hub id `org/name`
resolves to in the active Hugging Face cache (`HF_HUB_CACHE` / `HF_HOME`; a cache that was
moved and linked is the same folder), the copy lives in the Spectrum cache slot of that Hub
id, `weights/org__name`, so the two ways of naming the model share one copy. A snapshot
folder anywhere else gets a slot of its own, `weights/<folder name>@<digest of its resolved
location>`: a copy taken from one folder is never used for another folder or for the Hub id,
and such a folder costs one more copy on disk. A directory whose `.safetensors` files are symlinks but which is not such a
snapshot (including a regular directory that also holds an alias symlink to a shard) is
refused with a message naming the accepted layouts: pass the Hub id, the snapshot directory,
or a directory of regular files.

This materialisation also works with `HF_HUB_OFFLINE=1` when the standard Hugging Face
snapshot is complete. Soup pins the commit resolved by the initial cache lookup and copies
only verified snapshot files from that commit's blob store; it does not perform a second Hub
metadata request for the regular-file directory. A missing blob or an escaping symlink aborts
before the destination is published, rather than leaving a partial checkpoint that the sharder
could consume. With huggingface_hub 1.32 or later, links into its marked cache-wide store
(`<cache>/blobs`) are followed too, and any other target is still refused.
The repo's own `blobs` directory is treated like that store: it has to be a real directory.
When it is itself a link (a symlink, or a junction on Windows), Soup stops with a message
naming it instead of following it, so a link in the cache that points outside the cache is
never followed. To keep a model's files on another disk, move the whole cache or the whole
repo folder and link that: the snapshot path is resolved first. A junction inside a snapshot
directory is refused like a directory symlink. A copy made earlier is reused only while the
snapshot still passes these checks: when its layout changes into a refused one afterwards,
the next run stops with the same message and leaves the copy in place.


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

For [layer streaming](#layer-streaming-beta-v0720-nf4-v0722-disk--wider-archs-v0723-preference-losses-v0724),
the flag authorizes tokenizer code only; the model config, setup probes and model
always use the supported native Transformers implementation.

Coverage:
- `soup train` (every task - SFT, DPO, GRPO, KTO, ORPO, SimPO, IPO, PPO, Reward Model, Pretrain, Embedding, BCO, and the unified Preference dispatcher; automatic evaluation after training follows the run's flag, see `docs/evaluation.md`)
- `soup chat`, `soup serve`, `soup data download`, `soup eval auto`
- `soup diff`, `soup export`, `soup merge`, `soup infer`, `soup data generate`
- `soup eval benchmark`, `soup eval custom`, `soup train --find-lr`, `soup draft distill`, `soup shrink`

`soup data download --trust-remote-code` is refused when the installed `datasets` is 4 or newer, because those versions no longer support remote dataset code and would silently ignore the flag.

```bash
soup train --config soup.yaml --trust-remote-code
soup infer --model my-org/custom-arch-model --input prompts.jsonl --output out.jsonl --trust-remote-code
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
An unregistered template name is refused by `soup train` and `soup data preprocess`
before the model loads, and the message lists the known names.

Raw Jinja strings are validated: null bytes / >64KB / filesystem-touching directives (`{% include %}`, `{% import %}`, `{% from %}`, `{% macro %}`, `{% extends %}`) are rejected at config-load.

`backend: mlx` also refuses `data.chat_template` at config-load — the MLX trainer renders the model's shipped template only, so the override would be silently dropped; `backend: transformers` applies it. The exception is `task: grpo`, whose MLX loop applies the override (and the transformers path's fallback template), so it is accepted there.

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
at config load. Liger is reported as enabled only when the model's
`config.model_type` is in liger's own kernel table; for a type it has no kernels
for (`gpt_neox`, `phi`, `gpt2`, ...) the run prints "no matching architecture
found" and trains without it, instead of announcing kernels that were never
applied (#1811).

```yaml
# soup.yaml — performance options
training:
  use_liger: true            # Liger Kernel fused ops (measured 12.9% memory, 5.1% throughput); sft/tts only
  use_flash_attn: true       # FlashAttention v2/v3 auto-detection; sft/tts only
  gradient_checkpointing: true  # Required for long sequences

  # Long-context (128k+ tokens)
  rope_scaling_type: dynamic  # RoPE scaling: linear, dynamic, yarn, llama3; sft/tts/pretrain on backend: transformers only
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
  required — a friendly `RuntimeError` fires on CPU / without bnb. It needs the
  one-`nn.Linear`-per-expert layout: on a transformers-5 model whose routed
  experts are fused 3-D parameters (`mlp.experts.gate_up_proj`, which is every
  in-library MoE family on the declared stack) there is no expert `nn.Linear` to
  wrap, and the run is refused at setup naming the architecture and the fused
  parameters, before the CUDA check (#1421; it used to print `applied to 0
  expert Linear block(s)` and train in full precision). Matching no expert at all
  is refused the same way.
- `training.train_router_only: true` — freeze every expert parameter and train
  only the gating router (applied after LoRA, on the final parameter set).

Both reject silently-no-op combinations: setting either flag without `moe_lora=true` fails at config load with an actionable message.

**Which tasks read which flag (#798).** These were accepted on every task and applied by only some, so the stored config claimed a run that never happened:

| flag | applied by | elsewhere |
|---|---|---|
| `moe_lora` | `sft`, `pretrain`, `tts`, (since #798) `dpo`, `kto`, `orpo`, `simpo`, `grpo`, (since #1099) `ipo`, `bco`, `reward_model`, `ppo`, `embedding`, `online_dpo`, and (since #1151) `distill`, `unlearn`, and `classifier`, `reranker`, `cross_encoder` with `classifier_lora: true` | refused at config load, naming the reason, on `asr`, `moe_lora_routing`, `prm`, the classifier family without `classifier_lora: true` and `lora.r > 0`, `backend: unsloth` (any task), and `task: sft` with `modality: vision` or `audio` |
| `moe_expert_quant`, `train_router_only` | `sft`, `tts` | refused at config load, naming the task |
| `moe_aux_loss_coeff` | `sft` (text), `tts`, `pretrain` | a **non-default** value is refused, including on `task: sft` with `modality: vision` or `audio` (#1394), on `backend: unsloth`, and with `training.stream_layers: true` (#1412), whose setups never apply the auxiliary loss; the default `0.01` still loads, because every stored config and the shipped MoE recipes on those tasks write it |

**`moe_lora` on the remaining LoRA tasks (#1099).** #798 left it loading but unread on `ipo`, `bco`, `reward_model`, `ppo` and `embedding`, and `online_dpo` had the same gap. All six build their adapter through the same `build_lora_config` path, so they were wired to the same helper rather than refused. On `embedding` it applies only with `lora.r >= 1`; at `r: 0` that trainer full-fine-tunes and builds no adapter for the flag to select. #1151 closed the remainder: `distill` and `unlearn` build their adapter the same way and are wired to the same helper, and so is `classifier` / `reranker` / `cross_encoder` (one trainer) on its opt-in adapter path. Without `classifier_lora: true` and `lora.r > 0` that trainer full-fine-tunes and builds no adapter, so there the flag is refused at config load; `asr`, which trains only Whisper (no experts), `moe_lora_routing`, which builds no LoRA adapter, and `prm`, which fine-tunes every base parameter, refuse the flag at config load. Two paths refuse it whatever the task: `task: sft` with `modality: vision` or `audio`, whose setup builds its adapter without the MoE step, and `backend: unsloth`. On unsloth the reason depends on the task (#1264): the thirteen tasks with an unsloth setup attach its fixed attention list and never read the flag, while the rest have no unsloth setup at all, so there it is the backend that goes unapplied. Both messages point to `backend: transformers`, where the flag is read. The other `moe_lora` refusals in the same check come first, so `asr`, `prm`, `moe_lora_routing` and SFT vision/audio get one refusal whatever the backend. Every path now either reads `moe_lora` or refuses it, except `backend: mlx`, where it loads and `soup doctor --config` reports it as ignored; a source ratchet keeps a new adapter-building trainer from missing it. `train_router_only` and `moe_expert_quant` require `moe_lora: true`, so on `backend: unsloth` they now fail to load with the same message. `preference` is covered through the trainers it dispatches to, and `tts` through the SFT trainer it subclasses.

**`moe_lora` requires `lora.dropout: 0.0` on a fused-expert MoE.** transformers 5.x keeps a Qwen3-MoE's experts as fused 3-D parameters (`mlp.experts.gate_up_proj`), which peft adapts through `lora.ParamWrapper`, and that wrapper raises `lora.ParamWrapper does not work with lora_dropout != 0.` With the schema default of `0.05` the LoRA attach failed outright, so `moe_lora` did not work on any task - including `sft`. Soup now stops at the attach with a message naming the flag, instead of letting peft's reach the user, and all 29 shipped recipes that set `moe_lora` pin `lora.dropout: 0.0`. The check is made against the loaded model, not at config load: whether the experts are fused depends on the checkpoint and the transformers version, and a model with one module per expert takes dropout normally. A dense base is untouched - there the flag is a no-op. Since #1421 the refusal covers every fused-expert family, Mixtral and MiniMax included: their experts are now targeted (next paragraph), so peft's wrapper refuses dropout on them too, where before it attached quietly and adapted no experts.

**`moe_lora` reaches the fused experts on every family (#1421).** On transformers 5 an MoE layer keeps all its routed experts as two 3-D parameters on one module (`mlp.experts.gate_up_proj` and `down_proj`), not as one `nn.Linear` per expert, so module names cannot address them. Until #1421 `moe_lora` named modules only, and whether those names reached the experts depended on peft's own v4→v5 name conversion, which exists for some architectures and not others. Now `resolve_moe_lora_targets` finds the fused parameters on the loaded model and hands them to peft as `target_parameters`, beside the module targets it already resolved (list targets are kept: what `auto` resolves to, so Qwen3.5's `in_proj_qkv` / `out_proj` stay adapted, and an explicit `lora.target_modules` list; a string target such as qwen4_exp's `"all-linear"` or a regex keeps its meaning, without the MoE routers (see below); `stream_layers` does the same, resolving the targets before its meta-skeleton probe and handing them to the same helper). Measured on tiny stand-ins with transformers 5.16.1 / peft 0.20.0, routed-expert adapters per family, by module names alone and through the trainer path:

| family | module names alone (before #1421) | trainer path (now) |
|---|---|---|
| `qwen3_moe` | yes | yes |
| `deepseek_v3` | yes | yes |
| `glm4_moe` | yes | yes |
| `olmoe` | yes | yes |
| `qwen2_moe` | **no — attention + shared expert** | yes |
| `qwen3_5_moe_text` | **no — shared expert only** | yes |
| `minimax` | **no — attention-only** | yes |
| `mixtral` | **no — attention-only** | yes |
| `kimi_k2` | **not measured** (no stand-in builds here) | not measured |

`minimax-m3-sft` still trains attention-only LoRA: it sets no `moe_lora`, since it loads through SFT's vision path, which never runs the MoE step, and `minimax-m3-dpo` was removed because DPO cannot build the `minimax_m3_vl` wrapper (#1145). The seven `kimi-k2.x` recipes that set `moe_lora` are untested rather than known-good — no tiny stand-in for those configs exists in the installed transformers. A checkpoint that still keeps one `nn.Linear` per expert (the transformers-4 layout) is handled as before: the projections are named as modules and no `target_parameters` are added. `moe_lora` together with `lora.use_vera: true` is refused on a fused-expert model (#1197): VeRA wraps `nn.Linear` modules only and has no `target_parameters` path, so the flag would adapt attention (and a shared expert) exactly as VeRA without it; set `use_vera: false` to adapt the experts with LoRA, or drop `moe_lora`. The per-expert-Linear layout is not refused, VeRA wraps those.

**A string `target_modules` keeps its meaning under `moe_lora`.** A list target is extended with the MoE names (above). A string is not a list to extend, and it used to be replaced by the MoE list. peft reads a string in one of two ways, and each is now kept:

- `"all-linear"` (in any casing; it is what `target_modules: auto` resolves to on qwen4_exp) means every `nn.Linear` and `Conv1D` except the output embedding, which already includes every module the MoE names reach, so the fused expert parameters are added as `target_parameters`. On qwen4_exp that keeps the Gated DeltaNet, QSA indexer, hyper-connection and PLE projections the MoE list does not name.
- Any other string is a regex that peft full-matches against each module name. It becomes one regex, `(?:<your regex>)|(?:(?:.*\.)?(?:q_proj|k_proj|…))`, whose second half is peft's list rule for the MoE names, so peft adapts exactly what either half matches, plus the fused expert parameters. A leading global flag such as `(?i)` or `(?x)` stays at the front and applies to your half only. A regex that does not compile is refused at setup, naming `training.lora.target_modules`. A bare module name written as a string (`target_modules: in_proj_qkv`) is still a full-match regex, so it matches no nested module: the setup prints a warning that the regex matches no module and only the `moe_lora` targets attach. Write a list instead.

**Under `moe_lora` a string never adapts a router.** No module whose output is routing logits gets an adapter from a string target. That covers the router (one logit per expert: `mlp.gate` on qwen3_moe, deepseek_v3 and qwen4_exp, `ffn.router` on DBRX, `mlp.router` and the attention's `experts.router` on JetMoE) and the shared-expert gate (the single logit that scales the shared expert, `shared_expert_gate`). They are found from the model's structure:

- Under a module that holds `experts`, a router is a child whose output dimension is the expert count, and a shared-expert gate is a child whose output dimension is 1. The count comes from the experts' 3-D tensors or per-expert modules, or from the config where the experts are stacked 2-D tensors (DBRX).
- Where there is no `experts` child (JetMoE), a router is a child named `router` or `gate` that sits beside a 3-D expert tensor with as many experts as it has outputs.
- A router that wraps one `nn.Linear` (DBRX, JetMoE) is kept out through that Linear.

A dense `gate_proj` is never taken for one, and neither is an attention `gate`. They stay out even when your regex names them. Router LoRA is a training-stability risk that a string would otherwise switch on without saying so. To adapt a router anyway, name it in an explicit `target_modules` list, which `moe_lora` extends and never filters.

How the gates are kept out is decided per string, by asking peft's own handling of it:

- The string is kept, and the gates it would reach go to peft's `exclude_modules`. This applies to `all-linear` where peft adapts it as written (qwen4_exp, DBRX, JetMoE) and to a regex on any family. On the families peft converts from the transformers-4 layout (`qwen3_moe`, `deepseek_v3`, `glm4_moe`, `olmoe`, `mixtral`, `minimax` and the rest of transformers' conversion mapping), peft resolves a regex to the last names of the modules it matches, skipping excluded ones. So a regex naming one layer's `q_a_proj` reaches every layer's, with or without `moe_lora`.
- `all-linear` on those converted families becomes an explicit list. There peft adds the router's old name `gate` to `all-linear` whatever is excluded, and adapts the router weight `mlp.gate.weight` as a parameter, which `exclude_modules` does not reach. The list is what the string means without the gates, plus the MoE names: a leaf name (`q_proj`) where it reaches nothing outside that meaning, the full module path otherwise. A regex on those families becomes such a list only if it reaches a non-router whose last name is `gate`, which peft would rewrite onto every router.
- With `lora.use_vera: true` a string becomes such a list whenever it would reach a gate, because peft's `VeraConfig` has no `exclude_modules`.

On the converted families peft also renames every target ending in an old expert projection name (`gate_proj`, `up_proj`, `down_proj`) to the fused expert tensor. A dense or shared-expert `nn.Linear` with that name (deepseek_v3's first dense layer and its shared experts) is therefore not adapted as a module, with a list target too.

Without `moe_lora` a string reaches peft unchanged, router included where peft includes it.

The `ScatterMoE LoRA:` setup line says which happened: `keeping target_modules 'all-linear'` or `targeting the target_modules regex or the N MoE module patterns (one combined regex)`, followed by `excluding K MoE router/gate module(s)` when any are excluded; or, for a string turned into a list, `targeting 'all-linear' as a list of N target names` (or `the target_modules regex as a list of N target names`) with the gates it leaves out. Layer streaming hands its resolved targets to the same helper (#2072), so all of this holds under `stream_layers` too.

**`target_modules: auto` on a MoE base.** Until #1070 `resolve_lora_target_modules` had no mapping for any MoE architecture Soup ships, so `auto` resolved to `None` and peft refused with `No target_modules passed but also no target_parameters found`. Those architectures now resolve to their attention projections (see `docs/peft-and-efficiency.md`); a MoE architecture neither Soup nor peft maps is refused at setup, naming it. With `moe_lora: true` the targets come from the model scan instead, and that is applied *before* the refusal is decided, so `moe_lora` still works on an unmapped MoE such as `qwen2_moe`.


## Unsloth Dynamic 2.0 GGUF Ladder (v0.53.0)

`soup export --format gguf-ud --gguf-flavour <flavour> --calibration-data <calib.jsonl>` is the dispatch surface for the 14-entry UD ladder (`UD-Q8_K_XL` … `UD-IQ1_M`). v0.53.0 shipped the closed-allowlist validators, `MappingProxyType`-wrapped metadata, and a calibration-data path shape check; the live llama.cpp `imatrix` + `quantize` run landed in v0.53.1. The IQ + Apple/ARM-friendly GGUF flavours (`IQ4_NL`, `Q4_0_4_4`, `Q5_K_M`, etc.) ship as separate frozensets, and `--gguf-flavour` picks from all three.


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

- `fp8_attention: true` — requires `quantization_aware: fp8` AND a non-MLX, non-Unsloth backend (incompatible with `stream_layers: true`). Converts the attention projections (q/k/v/o and fused variants) to torchao float8 training on Ada or newer GPUs (the same gate as `quantization_aware: fp8`). A card, OS or torch build the gate refuses stops the run at setup, and so does a missing torchao (#835); a conversion-phase failure raises an honest "model may be PARTIALLY converted" error instead of training on a half-converted model, and a model with no attention projections stops too, instead of training on as a no-op (#1152).
- `nvfp4: true` — Blackwell-only FP4 training via torchao `NVFP4TrainingConfig` + `quantize_`, which replaces `nn.Linear` with `NVFP4Linear` and quantises the forward **and backward** GEMMs. (Through v0.75.0 Soup asked for `NVFP4Config`, a name torchao has never exported, so the flag degraded to a yellow advisory and a bf16 run — #826.) Gated to supported tasks (rejected on `task: distill` and tasks without v0.28 speed/memory wiring) + non-MLX, non-Unsloth + `modality: text` (incompatible with `stream_layers: true`); the SM ≥ 10 runtime check fires at trainer construction. The fix makes the flag **reach** NVFP4 training — the linears really are replaced by `NVFP4Linear`. A training step on top of that has torchao's own requirements, which Soup does not check for you: the kernels are triton, and `nvfp4_mm_triton` raises `ValueError: requires M, K, N all divisible by 128` for a shape it cannot serve (`torchao/prototype/moe_training/nvfp4_training/nvfp4_linear.py`), so a model whose hidden/intermediate sizes are not multiples of 128 will fail there rather than train slowly.
- `unsloth_bnb_4bit: true` — promotes "Unsloth Dynamic 4-bit" from an implicit `backend=unsloth + quantization=4bit` combo to a named flag. Mutual rejection of inconsistent combos at config load.

Cross-validator ordering picks the most actionable error: `quantization_aware='fp8'` prerequisite fires before the MLX rejection on `fp8_attention`, and task compatibility checks fire at config load before runtime trainer construction.

On `backend: unsloth`, active `quantization_aware`, `fp8_attention` and `nvfp4`
settings are refused at config load. They do not appear as `rejected` rows in
`soup doctor --config`: a loadable config cannot reach such a row. Select a
supported transformers path for these precision features.


## LF / Axolotl Quant Parity (v0.53.0)

- `bnb_4bit_use_double_quant` — controls BNB's double-quantization. **Defaults to `true`** (matching every 4-bit load path — resident, layer-streaming, and the `soup merge` 4bit save formats), and is now honoured everywhere (#321): set `false` to disable it and it actually reaches BNB. Explicitly setting it requires `quantization: 4bit`; combinations with the Quant Menu formats (gptq / awq / hqq:Nbit / aqlm / eetq / mxfp4 / fp8) are rejected at config load.
- `quantize_reward_model: true` — live for `ppo` and `reward_model` training, extending the v0.40.5 Quant Menu wiring to reward models (`trainer/ppo.py`, `trainer/reward_model.py`).
- `quantize_ref_model: true` — accepted for reference-policy tasks (`dpo / ipo / simpo / orpo / bco / kto / preference / grpo / ppo`), but read by nothing. Staged field: warned in v0.76; refused at load as of v0.77 (#808).


## Advanced Save Formats (v0.53.0)

`soup merge --save-format 4bit` and `--save-format 4bit_forced` write a single BNB-4bit-quantized merged checkpoint without the wasteful dequant → merge → requant cycle (unsloth `merged_4bit` recipe). `4bit_forced` quantizes all linear layers including `lm_head`; models with tied embeddings (`tie_word_embeddings: true`) cannot hold a 4-bit `lm_head` sharing weights with embeddings and are explicitly refused by name with `ValueError`. v0.53.0 shipped the closed allowlist + spec metadata; the live writer landed in v0.53.1.

`soup export --format torchao --quant-config <yaml>` is the live PTQ export (v0.53.1, #142): it runs `torchao.quantize_` on the model and writes the result with `save_pretrained`, to `--output` or, by default, next to the model as `<model>.torchao.<scheme>`. `--quant-config` is required with `--format torchao` (the command exits 2 without it) and the YAML must declare a top-level `scheme:` key. Four schemes are allowlisted: `Int4WeightOnly`, `Int8DynActInt4`, `Float8DynActFloat8`, `NVFP4`. CASE-SENSITIVE — these are Soup's scheme names, mapped to the torchao class that implements each one in `utils/torchao_compat.py` (three of the four names Soup used to look up by `hasattr` do not exist in torchao; #826). `Int4WeightOnly` and `Int8DynActInt4` accept `group_size`, `Float8DynActFloat8` and `NVFP4` take no extra keys, and any other key is refused (exit 2) with a message listing the allowed ones; `inner_k_tiles` was removed, because `Int4WeightOnlyConfig` raises `TypeError` for it. Diverges from `--save-format` (lowercase-normalised) on purpose; documented at both validators.


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
    --measure-candidates 4bit,8bit,fp8
```

Autopilot also detects pre-quantized bases automatically - `TheBloke/Llama-2-7B-Chat-GPTQ` is recommended `gptq` instead of stacking 4-bit on top. Detection runs against the base-model name regex AND any local `config.json`'s `quantization_config.quant_method`. Out-of-cwd model paths are silently skipped (soft-probe semantics). A detected `mxfp4` base is budgeted as the bf16 model it loads (2 bytes per parameter), and Autopilot refuses a `--gpu-budget` below that.

The advanced GGUF pipeline uses POSIX `O_NOFOLLOW` to defeat the TOCTOU race between the dispatch-time symlink check and the actual open of the calibration data — a crafted environment cannot race-swap the calibration file between validate and read.

`soup deploy autopilot --measure` caches results at `~/.soup/deploy_autopilot_cache.json` keyed on `(base, profile, eval-tasks)`. Repeat invocations short-circuit; pass `SOUP_DEPLOY_AUTOPILOT_CACHE=<path>` to redirect (constrained to home / cwd / tempdir). The recommended candidate uses soft-fallback: first `OK` by insertion order, else the candidate with the smallest delta (least drop relative to its own baseline).
