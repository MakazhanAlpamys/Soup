# PEFT, Long Context & Training Efficiency

[← Back to the Soup README](../README.md)

> DoRA/LoRA+/rsLoRA/VeRA/OLoRA/NEFTune, PiSSA/ReLoRA, the optimizer & PEFT zoo, LLaMA Pro, GaLore, YaRN/LongLoRA long-context, packing, curriculum, freeze, loss watchdog, and auto-tuning.

**Contents:**

- [LongLoRA Forward Override](#longlora-forward-override)
- [Multipack — FFD Bin-Packing Sampler](#multipack--ffd-bin-packing-sampler)
- [Long Context — YaRN, Llama 3.1 NTK, LongLoRA](#long-context--yarn-llama-31-ntk-longlora)
- [LLaMA Pro Block Expansion](#llama-pro-block-expansion)
- [Optimizer & PEFT Zoo](#optimizer--peft-zoo)
- [LoRA Quality — PiSSA, ReLoRA, Per-Pattern Rank, Surgical Patches](#lora-quality--pissa-relora-per-pattern-rank-surgical-patches)
- [DoRA (Weight-Decomposed LoRA)](#dora-weight-decomposed-lora)
- [LoRA+ (Differentiated Learning Rates)](#lora-differentiated-learning-rates)
- [rsLoRA (Rank-Stabilized Scaling)](#rslora-rank-stabilized-scaling)
- [VeRA & OLoRA (Smaller-Footprint PEFT)](#vera--olora-smaller-footprint-peft)
- [NEFTune (Noisy Embeddings Fine-Tuning)](#neftune-noisy-embeddings-fine-tuning)
- [Sample Packing](#sample-packing)
- [Curriculum Learning](#curriculum-learning)
- [Freeze Training](#freeze-training)
- [Loss Watchdog](#loss-watchdog)
- [Training Stability & Auto-Tuning](#training-stability--auto-tuning)
- [Training Intelligence (Forgetting + Checkpoint Quality)](#training-intelligence-forgetting--checkpoint-quality)
- [GaLore (Memory-Efficient Full-Parameter Training)](#galore-memory-efficient-full-parameter-training)
- [Depth Pruning + Distill-Heal (`soup shrink`)](#depth-pruning--distill-heal-soup-shrink)

---

## Depth Pruning + Distill-Heal (`soup shrink`)

`soup shrink` makes a model smaller by dropping its least-important **contiguous
block of decoder layers**, then optionally *healing* the loss with knowledge
distillation. It implements "The Unreasonable Ineffectiveness of the Deeper
Layers" (Gromov et al., arXiv:2403.17887), with a Minitron-style distillation
heal instead of the paper's plain LoRA fine-tune.

**How it ranks layers.** For each candidate block `[L, L+n)`, one forward pass
per calibration prompt (`output_hidden_states=True`) measures the **angular
distance** of the residual stream entering (`hidden_states[L]`) vs leaving
(`hidden_states[L+n]`) the block, averaged over every non-pad token across the
calibration set. The block with the *lowest* distance transforms the residual
stream least, so it is the safest to drop. The first and last decoder layers are
always protected (they carry the most transformation).

```bash
# See the importance table + chosen block without writing anything:
soup shrink --model HuggingFaceTB/SmolLM2-135M-Instruct --drop-ratio 0.25 \
    --calib calib.jsonl --device cpu --plan-only

# Prune 25% + heal (distill the original into the pruned student, fuse to one
# dense model) + get a SHIP/DON'T-SHIP perplexity verdict:
soup shrink --model HuggingFaceTB/SmolLM2-135M-Instruct --drop-ratio 0.25 \
    --calib calib.jsonl --heal heal.jsonl --heal-steps 200 \
    --tolerance 0.10 -o shrunk --device cpu --attach-to-registry <id>
```

- **`--drop-ratio F` / `--drop-layers N`** (exactly one) — how many contiguous
  layers to drop; the *position* is chosen automatically by the importance scan.
- **`--calib <jsonl>`** — calibration prompts (`{"text": ...}` / `{"prompt":
  ...}` / chat `messages`). Must stay under cwd.
- **`--heal <jsonl> --heal-steps N`** — distill the full-depth original into the
  pruned student (LoRA logit-KD) as an isolated `soup train` subprocess, then
  fuse the adapter back into the pruned base so the output is a single dense
  model. Heal keeps the teacher resident (~2× model memory); with `--device
  cpu` the heal runs on CPU (validated ≤ 3 B on a 4 GB card).
- **`--tolerance F`** — ship if the perplexity regression stays within `F`
  (default `0.10` = 10 %). Exit code `0 = SHIP`, `2 = DON'T SHIP`, `1 = error`.
- **`-o <dir>`** — the shrunk model lands in `<dir>/model`; the verdict in
  `<dir>/shrink_report.json`.

**Arch support (v1):** Llama / Qwen / SmolLM. Others are a friendly reject.
MoE configs that place their MoE layers by layer number (a non-default `mlp_only_layers` or `decoder_sparse_step` on Qwen MoE models, `moe_layers` or `interleave_moe_layer_step` on Llama4-text) are refused before the model is loaded, because a prune cannot renumber them. Per-layer lists such as `layer_types` and `no_rope_layers` are sliced along with the layers.
The importance pass loads the model, so live-validated on ≤ 3 B; larger models
work but are unvalidated on the reference hardware. Perplexity is an unweighted
mean of per-example perplexities — valid for the before/after *ratio* the
verdict uses, not directly comparable to `soup eval` absolute numbers.

---

## LongLoRA Forward Override

`training.use_longlora: true` is **refused at config load**
([#1240](https://github.com/MakazhanAlpamys/Soup/issues/1240)). The override it
installed was not S² shifted sparse attention. It rolled the query/key
projection outputs of half the heads along the sequence with `torch.roll`,
which wraps the last `group_size // 2` tokens around to the front, while
attention stayed full causal. Earlier positions could attend to keys computed
from the last tokens of the sequence, so future tokens leaked into the training
loss, and no grouped attention ran, so it saved no memory or compute either. A
config that set it never trained correctly.

It stays refused until real S² attention exists: RoPE first, then shift q, k
and v of half the heads, attend causally within each group, and roll the
output back. Every spelling read as true (`yes`, `on`, `1` ...) is refused;
remove the key or set it to `false`. To extend the context, set
`training.rope_scaling_type` (see
[Long Context](#long-context--yarn-llama-31-ntk-longlora)) and train plain
LoRA.


## Multipack — FFD Bin-Packing Sampler

Soup's largest single throughput win on chat fine-tuning over uneven-length data. Instead of padding every sample to `max_length`, Multipack uses **First-Fit-Decreasing bin packing** to group variable-length samples into bins approaching `batch_size × max_seq_length` — eliminating padding waste.

```yaml
training:
  multipack: true
  packing: false   # mutually exclusive with multipack
```

**How it composes:**
- **Multipack** picks WHICH samples go together (FFD packing).
- Packed-document isolation is TRL's default `bfd` strategy when FlashAttention is the `attn_implementation`. `packing_cross_doc_attn_mask` is rejected at config load (it never mapped to a valid TRL `packing_strategy`).

**Architecture allowlist** — 18 supported (Llama 3.x, Qwen 2/3, Mistral, Gemma 2/3, Phi 3/4, DeepSeek V2/V3, Mixtral, Falcon, StableLM, SmolLM2). Unknown architectures **fail loudly at config-load** instead of silently no-opping (critical fix vs Axolotl's silent-miss footgun).

**Live wiring** — landed. SFT and Pretrain trainer wrappers actually instantiate the multipack subclass when `multipack: true` is set. The factory's `get_train_dataloader` override installs `MultipackBatchSampler(real_batches=False)` (yields a flat `list[int]` per packed sequence — DataLoader-compatible) as the DataLoader's `batch_sampler=`, forwarding `dataloader_drop_last`/`num_workers`/`pin_memory` from `TrainingArguments`. The `_get_train_sampler` override stays as a defensive no-op fallback that always delegates to super, so any HF eval / prediction loop bypassing `get_train_dataloader` still gets the correct `Sampler[int]` shape (no nested-list shape mismatch). Multipack is **sft / pretrain only** on the `transformers` backend; preference / RLHF trainers and MLX backend get distinct error messages naming the actual reason. Datasets must expose `input_ids` (preferred) or `length` per row; raw text triggers an all-zeros warning.

**Multi-GPU sharding (v0.71.19).** Under FSDP / DeepSpeed ZeRO / DDP (`num_processes > 1`) the `get_train_dataloader` override routes the multipack DataLoader through `accelerator.prepare`, so accelerate's `BatchSamplerShard` round-robins whole FFD-packed bins to each rank (preserving the packing; the bin seed is identical across ranks so every rank agrees on the global order before sharding). The single-GPU path returns the raw DataLoader unchanged. Multi-GPU correctness is mocked-tested — a real 2+-GPU validation run is tracked QA.

**DoS hardening** — the FFD packer caps at 1M items (a bound on retained memory; placement itself is O(N log N) since #726); the 4D mask builder caps allocations at 2³¹ cells; the chat-template Jinja analyzer caps at 128KB. Every numeric input rejects `bool` explicitly (matches v0.30.0+ project policy).

The `JinjaTemplateAnalyzer` (also v0.37.0) walks chat-template ASTs to discover non-standard `message.<field>` references (`tool_calls`, `name`, `weight`, `train`) — used by the v0.36.0 `train_on_messages_with_train_field` path so per-message training masks are aware of fields beyond `role` / `content`. The analyzer parses templates without rendering them, so a crafted `soup.yaml` cannot trigger SSRF.


## Long Context — YaRN, Llama 3.1 NTK, LongLoRA

Soup ships four RoPE-scaling strategies (`linear`, `dynamic`, `yarn`, `llama3`); LongLoRA is refused, see below:

```yaml
# soup.yaml
base: meta-llama/Meta-Llama-3-8B  # 8k, no native RoPE scaling
task: sft
data:
  train: ./data.jsonl
  max_length: 32768  # extend from 8k → 32k
training:
  rope_scaling_type: yarn      # linear | dynamic | yarn | llama3
  yarn_factor: 4.0             # 4x extension
  yarn_beta_fast: 32
  yarn_beta_slow: 1
  yarn_attn_factor: 1.0
  gradient_checkpointing: true  # required above 64k
```

**YaRN.** Best quality for 4-8x extension. Tunables (`yarn_factor`, `yarn_attn_factor`, `yarn_beta_fast`, `yarn_beta_slow`) only apply when `rope_scaling_type=yarn`; the schema rejects them otherwise. Pure-Python math kernels are exposed at `soup_cli.utils.long_context.yarn_*` for reference / config-emit. The actual RoPE rotation runs inside HF Transformers.

**Llama 3.1 NTK-aware.** Use `rope_scaling_type: llama3` for Llama 3.1-style frequency-band scaling. On a checkpoint without RoPE scaling it emits `factor = data.max_length / max_position_embeddings` over `original_max_position_embeddings = max_position_embeddings`, with `low_freq_factor` 1 and `high_freq_factor` 4, so an 8k checkpoint extended to 64k gets Llama 3.1's own factor 8 over 8192. `detect_llama3_rope_in_config` can identify the block in an HF model config dict, but `soup train` changes RoPE only when `rope_scaling_type` is explicit; omitting it preserves the checkpoint's native RoPE configuration. On a checkpoint that already ships a `llama3` block (Llama 3.1, 3.2 and 3.3 do), `rope_scaling_type: llama3` composes with it instead of replacing it: the checkpoint's `original_max_position_embeddings`, `low_freq_factor` and `high_freq_factor` are kept, and its `factor` is multiplied by `data.max_length / max_position_embeddings`. Llama-3.1-8B extended from 131072 to 262144 tokens trains with factor 16 over 8192, so no frequency pair rotates faster than it did in pretraining.

RoPE scaling is applied before model construction on `task: sft` (the text path with `backend: transformers` and no layer streaming) and on `task: tts` and `task: pretrain` with `backend: transformers`. Everywhere else `rope_scaling_type` is refused at config load ([#1697](https://github.com/MakazhanAlpamys/Soup/issues/1697)): on the other 20 tasks, on `sft` with `backend: mlx`, `modality: vision`, `modality: audio` or `training.stream_layers: true`, and on any of the three with `backend: unsloth`. Those setups build the model with the checkpoint's own RoPE; the message names each setting to change, and the `yarn_*` keys set beside it. Existing type-independent model parameters such as `rope_theta` are preserved; tunables belonging to a previous RoPE algorithm are removed when the type changes. A checkpoint whose RoPE block is already scaled (any `rope_type` other than `default`, for example `yarn`, `longrope` or `llama3`) is never replaced: apart from `llama3` on a `llama3` block, extending it is refused before the model is built, and the error names the checkpoint's `rope_type` and `factor`. A `data.max_length` at or below the checkpoint's `max_position_embeddings` extends nothing, so none of these refusals applies to it. Models such as Gemma 3 that use nested per-layer RoPE sections are refused rather than partially modified. `rope_scaling_type: longrope` is refused at config load, whatever `data.max_length` is. Its per-dimension `short_factor` and `long_factor` vectors exist only on checkpoints already scaled with LongRoPE, and extending those is refused, so it cannot extend any checkpoint. To extend a checkpoint without RoPE scaling, use `linear`, `dynamic`, `yarn` or `llama3`. To fine-tune a LongRoPE checkpoint such as Phi-3-mini-128k at its native length, leave `rope_scaling_type` unset: the checkpoint's own RoPE block is used as shipped.

**LongLoRA S².** `training.use_longlora: true` is refused at config load ([#1240](https://github.com/MakazhanAlpamys/Soup/issues/1240)): the override it installed leaked future tokens into earlier positions and applied no S² grouping (see [LongLoRA Forward Override](#longlora-forward-override)). Use one of the RoPE-scaling strategies above with plain LoRA instead.

```yaml
# Llama 3.1 ships llama3 scaling to 128k; this composes with it out to 256k
base: meta-llama/Llama-3.1-8B
training:
  rope_scaling_type: llama3
  gradient_checkpointing: full
data:
  max_length: 262144
```


## LLaMA Pro Block Expansion

Add `N` zero-initialised transformer blocks to a base model and train **only the new blocks** — keeps the original behaviour intact while adding capacity for a new domain (per the LLaMA Pro paper, `arxiv.org/abs/2401.02415`).

```yaml
# soup.yaml — LLaMA Pro continued-training on a Llama-3.1 base
base: meta-llama/Llama-3.1-8B
task: sft
data:
  train: ./domain.jsonl
training:
  quantization: none            # block expansion needs an unquantized base
  expand_layers: 4              # append 4 zero-init decoder blocks
  freeze_trainable_layers: 4    # must equal expand_layers: freeze the original model, train only the appended blocks
  lora:
    r: 0                        # train the appended blocks directly; a LoRA rank is refused with expand_layers
  lr: 5e-5
  epochs: 1
```

**What happens at trainer start.** Soup deep-copies the last `expand_layers` decoder blocks, zero-inits each clone's residual projections (`mlp.down_proj` + `self_attn.o_proj`) so the appended block initially acts as identity, appends them to `model.model.layers`, and updates `config.num_hidden_layers`. `freeze_trainable_layers` must equal `expand_layers`: it freezes every parameter except the appended blocks, the canonical LLaMA Pro "train only new blocks" recipe. It does not select the top-N or bottom-N layers, and any other value, including `0` or a negative one, is refused at config load.

**Scope.** Works on `task: sft` with `backend: transformers`, `modality: text`, `quantization: none` and `lora.r: 0`. A LoRA rank is refused together with `expand_layers` ([#1326](https://github.com/MakazhanAlpamys/Soup/issues/1326)): the expansion runs before `get_peft_model`, which freezes every non-adapter parameter again, appended blocks included, so only adapters on the original layers would train. `task: pretrain` is refused at config load whatever the rank ([#1813](https://github.com/MakazhanAlpamys/Soup/issues/1813)): a rank would freeze the appended blocks again, and the pretrain trainer has no `lora.r: 0` branch, so the recipe needs `task: sft`. Other tasks, backends and modalities, and a quantized base, are refused at config load. No other trainer applies the expansion, and the appended blocks are only supported on an unquantized base. Bounds: `expand_layers ∈ [1, 64]`. Over-expansion (more new blocks than the base has layers) silently clamps to the base layer count. Non-Llama-shaped architectures (e.g. Falcon's `dense_4h_to_h`) emit a `warnings.warn` because the residual zero-init heuristic only matches the standard `down_proj` / `o_proj` names — the appended blocks are still appended + trainable, but lose the identity-init guarantee.


## Optimizer & PEFT Zoo

Pick from a wider catalogue of optimizers and use quantization-aware LoRA initialisation:

```yaml
training:
  # HF-native, bnb-8bit and v0.41.0 additions (lomo, apollo_adamw,
  # grokadamw, schedule_free). Ten retired names (badam, adam_mini, muon,
  # dion, came_pytorch, ao_adamw_{fp8,4bit,8bit}, ...) are refused at
  # config load because transformers 5.x rejects them (#1269);
  # muon and adamw_hf still work on backend: mlx.
  optimizer: adafactor

  # Friendly aliases for users coming from LlamaFactory / Axolotl
  # load_in_8bit: true      # equivalent to quantization: 8bit
  # load_in_16bit: true     # equivalent to quantization: none

  quantization: none        # required: PEFT LoftQ quantizes the base itself

  lora:
    init_strategy: loftq    # quantization-aware LoRA init (also: pissa / olora / random)
    loftq_iter: 1
    loftq_bits: 4

  # LLaMA Pro block expansion needs lora.r: 0, so it cannot share a run with a
  # LoRA init such as LoftQ; see LLaMA Pro Block Expansion above.
```

Catch-all friendly errors: typos in `optimizer:` are rejected at config-load with the v0.41.0 additions listed in the message; `load_in_8bit` mixed with `load_in_16bit` raises rather than picking one silently.

**`lr_groups` is not applied.** The per-module learning rate it describes is parsed and validated (patterns must be compilable regexes) but no optimizer reads it, so every parameter trains at `lr`. Setting it warned in v0.76; refused at load as of v0.77 (#761).

PiSSA, OLoRA, LoftQ, and VeRA are applied through the shared PEFT constructor on
the Transformers backend. Soup refuses these variants on MLX and Unsloth rather
than silently substituting ordinary LoRA. PiSSA and LoftQ additionally require
`quantization: none`: PiSSA needs floating-point base weights for its SVD, while
LoftQ performs the low-bit conversion itself, so an already quantized base is
invalid for either initializer.

On the Transformers backend, `target_modules: auto` is resolved in this order, and
an explicit target list always wins unchanged:

1. **Architectures PEFT maps itself** (`llama`, `mistral`, `qwen2`, …) are left to
   PEFT's own default.
2. **Architectures Soup maps** are resolved from `utils/peft_wiring.py`. The Qwen3.5
   family targets `q_proj` and `v_proj` in full-attention layers plus `in_proj_qkv`
   and `out_proj` in the fused linear-attention layers, because PEFT does not map
   `qwen3_5_text`. The MoE architectures Soup ships recipes for (`qwen3_moe`,
   `deepseek_v3`, `deepseek_v4`, `glm4_moe`, `glm_moe_dsa`, `granitemoehybrid`,
   `kimi_k2`/`kimi_k25`, `gpt_oss`, `minimax_m2`, `minimax_m3_vl`, `mistral3`) target
   their attention projections; PEFT maps none of them (#1070). On the DeepSeek-V3
   multi-head latent attention bases `deepseek_v3` and `kimi_k2` the query
   projections follow the checkpoint's `q_lora_rank`: an integer gives the model
   `q_a_proj` and `q_b_proj`, while `q_lora_rank: null` (Moonlight-16B-A3B) gives it a
   single `q_proj`, which is targeted instead. A `kimi_k25` base keeps the
   `q_a_proj` / `q_b_proj` names even with `q_lora_rank: null`, so its `q_proj`
   would get no adapter. No shipped Kimi K2.5 or K2.6 config sets `null`. The
   reason is its vision tower, which has a `q_proj` of its own: the entry would have
   to be a regex scoped to the language tower, and `moe_lora: true` adds its names
   to a list of targets but discards a regex, so the key/value projections would
   lose their adapter. The same holds for any other wrapper around a `deepseek_v3`
   text config with `q_lora_rank: null` (Kimi-VL style): Soup has no row for the
   wrapper, so its text tower keeps the `q_a_proj` / `q_b_proj` names and its
   `q_proj` gets no adapter. MiniMax-M3 uses a regex
   scoped to its language tower, so a text fine-tune does not adapt the vision
   encoder. `mistral3` (the vision-language wrapper behind `mistral-medium-3-5-sft`,
   which therefore sets `modality: vision` so SFT loads the image-text class) uses the
   same kind of language-tower regex: `q_proj`, `k_proj`, `v_proj` and `o_proj` under
   `language_model...self_attn` only, so the Pixtral vision tower, the multimodal
   projector and the MLP projections stay unadapted (#1395). `glm4_moe` is GLM-4.6 and
   is *not* `glm_moe_dsa` (GLM-5 / GLM-5.1): the two have different attention shapes.
3. **Anything else fails closed.** `auto` on an architecture neither PEFT nor Soup
   maps is refused at setup, naming the `model_type`, rather than reaching PEFT's
   `No target_modules passed`. This is not new behaviour — PEFT refused those too —
   only a clearer message. It covers dense models as well as MoE ones: at the time of
   writing `phi3`, `smollm3`, `lfm2` and several vision/audio architectures in the
   catalogue land here. Give an explicit `target_modules` list, or for a MoE model set
   `training.moe_lora: true`, which supplies expert targets and is checked *before*
   the refusal. `training.lora.target_parameters` on its own also suffices.

**`granitemoehybrid` is adapted only in part, and says so at setup.** Granite 4.0
is a hybrid: on `ibm-granite/granite-4.0-h-tiny-base` only 4 of the 40
decoder layers carry a `self_attn` at all (`config.layer_types` is 36
`linear_attention` + 4 `full_attention`), so the attention-projection entry above
reaches a tenth of the decoder. The other 36 layers are Mamba-2 blocks
(`mamba.in_proj`, `mamba.out_proj`), and every layer's shared-expert projections
(`shared_mlp.input_linear`, `shared_mlp.output_linear`) and fused routed experts
are left alone as well — consistent with every other row of that table, where the
policy is attention projections only. Training prints a yellow
`Partial LoRA coverage:` line naming the model type, the counted fraction and what
was skipped, so the small adapter is not a surprise at merge time. If you want to
reach the state-space or shared-expert projections, name them explicitly:

```yaml
training:
  lora:
    target_modules: [q_proj, k_proj, v_proj, o_proj, in_proj, out_proj]
```

That list is correct as module names on this architecture — it is what
`named_modules()` reports — but Soup has not measured whether adapting a
state-space projection trains well, and it is not the default for that reason.
Treat it as the way to reach those layers, not as a recommendation to.

The MLX backend keeps its separate full-key default (`self_attn.q_proj`,
`self_attn.v_proj`).

Qwen4-Exp routed experts are raw 3-D parameters rather than `nn.Linear` modules, so
`target_modules: auto` / `all-linear` deliberately does not include them. Opt into
PEFT's parameter-targeting path for a higher-capacity resident SFT or continued-pretrain
adapter:

```yaml
training:
  lora:
    r: 16
    alpha: 32
    dropout: 0                 # required by PEFT ParamWrapper
    target_modules: auto       # every Qwen4-Exp linear family
    target_parameters: auto    # routed gate_up_proj + down_proj tensors
    rank_pattern:
      experts.gate_up_proj: 2
      experts.down_proj: 2
```

`target_parameters: auto` fails closed when an architecture has no registered mapping;
an explicit list of parameter-name suffixes is also accepted. It is currently limited to
resident text `sft` / `pretrain` on the Transformers backend and plain LoRA/rsLoRA with
random initialization. PEFT requires zero dropout for raw parameters and warns that
`torch.compile` may recompile or graph-break around parameter wrappers. Parameter-targeted
MoE adapters also materialize a contribution for every expert during inference; merge the
adapter into the base for deployment when hot-swapping is not required.

See `soup_cli.utils.optimizer_zoo.SUPPORTED_OPTIMIZERS` for the complete optimizer allowlist.


## LoRA Quality — PiSSA, ReLoRA, Per-Pattern Rank, Surgical Patches

Five PEFT-surface improvements that LlamaFactory and Axolotl maintain:

```yaml
training:
  quantization: none            # required: schema default is 4bit; ReLoRA/PiSSA need float
  lora:
    init_strategy: pissa          # 'random' (default), 'pissa', 'olora'
    rank_pattern:                 # per-target-module rank override
      q_proj: 8
      v_proj: 16
    alpha_pattern:                # per-target-module alpha override
      q_proj: 16
  relora_steps: 500               # merge+reinit restart every 500 steps
  relora_warmup_ratio: 0.1        # skip first 10% of training
  relora_prune_ratio: 0.9         # deprecated; kept for old YAML (ignored)
  relora_reset_optimizer: true    # clear optimizer state after each restart
```

**PiSSA** initializes the LoRA pair from the SVD of the base weight, giving faster
early convergence than random init at the cost of one extra SVD pass on the first
epoch. `init_strategy: olora` is also accepted; setting the legacy `use_olora: true`
auto-aligns for back-compat.

**ReLoRA is a behaviour break.** Soup's `training.quantization` **defaults to `4bit`**.
ReLoRA restarts merge the LoRA update into the base weight, so `relora_steps` now
**requires an explicit `quantization: none`** in soup.yaml. A config that previously
set only `relora_steps` is refused at parse. No shipped recipe, template, or example
sets `relora_steps`.

ReLoRA fires every N global steps, merges the LoRA update (B @ A, with PEFT
scaling) into the frozen base weight, reinitializes `lora_A` (Kaiming) and
`lora_B` (zeros), optionally clears optimizer state for those adapter parameters,
and runs a short learning-rate re-warmup. The first post-restart step uses
`1/(W+1)` of the target LR, then advances by that same increment to full LR.
Restarts accumulate faithfully only on an fp32 base; bf16/fp16 bases lose merge
delta to rounding. Embedding adapters (empty `lora_A`, typically `embed_tokens`)
are skipped; Linear LoRA is merged even though peft puts empty
`lora_embedding_A`/`lora_embedding_B` dicts on every `LoraLayer`. Useful for very
long training runs where adapter capacity saturates. `relora_prune_ratio` is
retained for backward-compatible YAML but no longer prunes adapter weights.

After training, Soup merges the active adapter into the already-accumulated
in-memory base and saves the final output as a standalone dense model. Load or
export that output directly; do not run `soup merge` on it. Intermediate
checkpoint directories remain trainer artifacts, and `--resume` / `--hf-resume`
are refused when `relora_steps` is configured because those checkpoints do not
encode the accumulated restart state.

**Per-pattern rank/alpha** map module name patterns to integer ranks. Useful in MoE
configs where expert FFNs need lower rank than attention. Caps: 256 keys × value 1024.

**Surgical patches** (Gemma 4 `ClippableLinear` swap, fused-MoE 3-D expert
`lora_dropout` strip) auto-fire when the model name and architecture match. Both are
gated and silent on unrelated models.

**Template registry** — the 21 built-in templates now live as
`src/soup_cli/templates/*.yaml` with a `manifest.json` index. `soup init --template <name>`
reads the YAML; the inline copies in `schema.py` stay as a back-compat fallback,
deprecated in favour of the YAML registry.

**Multi-trainer scope** — ReLoRA and the surgical patches are wired into every
transformer-backend trainer: `sft`, `dpo`, `grpo`, `kto`, `orpo`, `simpo`, `ipo`,
`ppo`, `reward_model`, `pretrain`, `embedding`, `bco`, plus the unified
`task: preference` dispatcher, though ReLoRA is refused for the tasks with a frozen
reference (below). Schema cross-validator rejects MLX backend,
quantization other than `none`, `stream_layers`, `lora.use_vera`, `lora.use_dora`,
and `use_fsdp2_compile` (the callback is HF Trainer-specific and restart requires
a writable float base).

**ReLoRA is refused where the loss has a frozen reference**: `task: dpo`, `ipo`,
`kto` and `bco`, `task: grpo` with `grpo_beta` above 0, and `task: preference`
whose loss or blend names `dpo`, `ipo` or `bco`. trl reads that reference from the
policy's own base weights (a `ref` adapter over them, or the model with its adapter
disabled), and every restart merges the trained adapter into those weights. From
the first restart on, the reference would be the policy as of that step instead of
the model the run started from. Measured on a fixed batch with the tiny test
model: the `dpo` reference does not move until the first restart and moves by 0.83
at it. `sft`, `orpo`, `simpo`, `reward_model`, `pretrain`, `embedding`, `grpo` with
`grpo_beta: 0` and a `simpo` / `orpo` preference run are not affected. `ppo` reads its
reference the same way as `kto` and `bco` but has not been measured.


## DoRA (Weight-Decomposed LoRA)

Enable DoRA for improved LoRA quality with magnitude decomposition:

```yaml
training:
  lora:
    r: 64
    alpha: 16
    use_dora: true  # Enable DoRA
```

Works with all training tasks on the transformers backend (see the note on
quantized bases below). The MLX backend also supports DoRA for text SFT on dense
linear targets with unquantized or pre-quantized 4-bit bases; see
[DoRA on MLX](backends-and-ops.md#dora-on-mlx) for restrictions and native adapter
loading/fusion. On `backend: unsloth`, `use_dora: true` remains refused at config
load because that path does not build a DoRA adapter.

> **Not on GPTQ / AWQ / AQLM / EETQ bases.** peft has no DoRA variant for those
> layers and raises when the adapter is attached, so `use_dora: true` with
> `quantization: gptq`, `awq`, `aqlm` or `eetq` is refused when the config is
> loaded. Use plain LoRA on those bases, or a `4bit` / `8bit` / `hqq:Nbit` /
> unquantised base to keep DoRA.


## LoRA+ (Differentiated Learning Rates)

Use different learning rates for LoRA A and B matrices:

```yaml
training:
  lr: 2e-5
  loraplus_lr_ratio: 16.0  # lr_B = lr × 16
  lora:
    r: 64
    alpha: 16
```

- **Compatibility:** Wired on every Trainer-based task that trains a LoRA adapter, including the preference and RL tasks (`dpo`, `kto`, `orpo`, `simpo`, `ipo`, `bco`, `grpo`, `online_dpo`, `ppo`, `reward_model`, `distill`). Most attach the LoRA+ optimizer once the trainer is built; `ppo` passes it to the trainer's constructor instead, because trl's PPO trainer builds its optimizer and LR scheduler eagerly, and adds the value model to it at the base learning rate, as trl's default optimizer does. `classifier`, `reranker`, `cross_encoder` and `asr` full fine-tune by default, so LoRA+ applies there only with `classifier_lora: true` / `asr_lora: true` and `lora.r > 0`. Refused at config parse on tasks with no trainable LoRA $B$ matrix: `prm` (full fine-tune), `moe_lora_routing` (only the routing gate trains), and `classifier` / `reranker` / `cross_encoder` / `asr` without their LoRA flag. Also refused on `unlearn`, which runs its own optimizer loop rather than a Trainer. Refused with `lora.use_vera` (VeRA trains scaling vectors, not $A$/$B$ matrices, so every trainable tensor would run at `lr * ratio`). Also refused on the `mlx` backend (#1324): the MLX optimizer Soup builds (`trainer/mlx_optim.py`) runs every LoRA tensor at one learning rate (no parameter groups), so the ratio would be silently ignored; remove it or use `backend: transformers`. `backend: unsloth` gets the same LoRA+ optimizer as `transformers`. Mutually exclusive with `use_lorafa`.
- **Optimizers:** LoRA+ builds its optimizer without the model, so it is refused at config load with `optimizer: apollo_adamw`, `lomo` or `adalomo` (transformers can only build those from the model) and with `use_galore`.


## LoRA-FA (Frozen-A LoRA)

Freeze random projection matrices in LoRA $A$ and update only LoRA $B$ matrices using PEFT's `create_lorafa_optimizer` ([arXiv:2308.03303](https://arxiv.org/abs/2308.03303)):

```yaml
training:
  lr: 2e-4
  use_lorafa: true
  lora:
    r: 64
    alpha: 16
```

### Operating Point & Caveats
- **Measured Adapter Operating Point:** Trains exactly 50.0% fewer parameters per adapted projection (trains $B$, freezes $A$), reducing AdamW optimizer states (`exp_avg_B`, `exp_avg_sq_B`) by half for square projections.
- **Analytic Activation Retention:** Freezing $A$ avoids storing input activations $x \in \mathbb{R}^{B \times L \times d_{in}}$ for adapter backpropagation through $A$. Only $u = A x \in \mathbb{R}^{B \times L \times r}$ is retained, yielding an analytic adapter activation ratio of $r / d_{in}$ (~64× reduction for rank 64 on hidden dim 4096; the exact ratio scales with your rank choice).
- **Scope & Limitations:** These values represent a micro-benchmark operating point and an analytic saved-tensor ratio for the adapter projections — **they are not total or peak LLM VRAM savings, an end-to-end throughput result, or a quality claim.** Peak training VRAM in full LLM fine-tuning is dominated by base model activations, KV caches, and weights; total end-to-end VRAM savings are substantially smaller. Downstream task quality and end-to-end throughput vs standard LoRA remain unmeasured. See [`benchmarks/gate-725-lorafa-operating-point.md`](../benchmarks/gate-725-lorafa-operating-point.md) for measured figures.
- **Compatibility:** Supported on the `transformers` backend for `sft`, `pretrain`, and `embedding` tasks. Mutually exclusive with `loraplus_lr_ratio` (which differentiates $A$ and $B$ rates), `use_galore`, `lora.use_vera` (VeRA trains scaling vectors, so `create_lorafa_optimizer` finds no $B$ matrices), non-AdamW optimizers, and the `mlx` backend. Requires explicit `lora.r` and `lora.alpha`. LoRA-FA has not been validated under `stream_layers: true` (layer streaming); combining them is not recommended.


## Fast-LoRA kernels (`training.fast_lora`)

Runs a LoRA's projections through Soup's own autograd functions instead of
peft's modules: q/k/v with one shared input and one accumulated input gradient,
gate/up/down around SwiGLU as one block, and every other adapted Linear (such as
`o_proj`) as a single projection. The math is LoRA's; the kernels change the
order of the work and what is kept for the backward pass.

```yaml
training:
  fast_lora: true
  lora:
    dropout: 0          # required: Soup's default is 0.05
    target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

Setup prints how many modules each kernel took over, and refuses the run if the
flag is on and none was. Off by default.

- **Scope:** `task: sft` on the `transformers` backend, `modality: text`,
  `quantization: none` or `4bit`, resident or with `stream_layers`. Refused, with
  the reason: any other task, backend, modality or quantization; `lora.r: 0`;
  `lora.use_dora`, `lora.use_vera`, `lora.target_parameters`; `lora.dropout`
  other than 0 (peft gives every projection its own dropout, so q/k/v and
  gate/up would no longer share one input); `training.use_liger`, which
  replaces the MLP's forward too; `unfrozen_parameters` (Spectrum) and
  `lisa_enabled`, which train the model's own weights and skip the LoRA setup;
  and `quantization_aware: fp8`, whose converted projections the kernels would
  read around. A model whose adapters the kernels cannot run (GPT-2's Conv1D
  layout) is refused at setup, as if nothing had been patched.
- **What is verified, on CPU:** float64 `gradcheck` of each kernel through the
  switch; float32 forward and backward parity with peft's modules; the same
  gradients with and without non-reentrant checkpointing; the base model's
  output with the adapter disabled; fifty training steps whose losses stay
  within 1e-3 of peft's; and a streamed model with more layers than
  `stream_buffers`, whose LoRA gradients on every layer equal those without the
  flag.
- **Not verified:** bf16, fp16 and NF4 parity, and streaming on a GPU. Those
  tests exist (`pytest -m gpu`) and need a CUDA card with bitsandbytes.
- **No speedup is claimed.** Nothing here measures end-to-end throughput or
  peak VRAM. Under gradient checkpointing, which layer streaming always uses, a
  block's saved tensors exist only during its recompute, so any saving lowers a
  per-layer transient, not the stored total.

## rsLoRA (Rank-Stabilized Scaling)

Use rank-stabilized LoRA scaling for better performance at high ranks:

```yaml
training:
  lora:
    r: 64
    alpha: 16
    use_rslora: true  # Enable rank-stabilized scaling
```

Works with all training tasks and backends. Recommended for LoRA rank ≥ 32.

The scale is `alpha / sqrt(r)` instead of plain LoRA's `alpha / r` (2.0 instead of 0.25
for the config above) on every backend: PEFT's `LoraConfig(use_rslora=True)` on
`transformers`, the `scale` Soup hands mlx-lm (in `adapter_config.json` and at training
time) on `mlx`, and `get_peft_model(..., use_rslora=True)` on `unsloth`.


## VeRA & OLoRA (Smaller-Footprint PEFT)

Two further LoRA variants for tighter memory budgets:

**VeRA** (Vector-based Random Adaptation) — shares random frozen projection matrices across all layers, trains only small scaling vectors. Much smaller adapter file.

```yaml
training:
  lora:
    r: 256           # VeRA typically needs higher rank (128-512)
    alpha: 1
    use_vera: true
```

**OLoRA** (Orthonormal LoRA) — initializes LoRA weights from QR-decomposed base weights, converges faster.

```yaml
training:
  lora:
    r: 64
    alpha: 16
    use_olora: true
```

> **Mutually exclusive:** `use_dora`, `use_vera`, and `use_olora` cannot be combined in one config. Soup validates this at load time.


## NEFTune (Noisy Embeddings Fine-Tuning)

Add noise to embeddings during training for better chat model quality:

```yaml
training:
  neftune_alpha: 5.0  # Noise intensity (0-50, typically 5-15)
```

Works with SFT, TTS, DPO, KTO, ORPO, SimPO, IPO, BCO, and preference tasks.


## Sample Packing

Pack multiple short samples into one sequence for faster training:

```yaml
training:
  packing: true  # Pack short samples together (faster training)
```

Works with SFT and Pretrain tasks. Packed SFT keeps the assistant-only loss mask (`train_on_responses_only`, `train_on_messages_with_train_field`, `mask_history`, or a pre-tokenised `labels` column). Warning emitted if `max_length < 256`.


## Curriculum Learning

Sort dataset by difficulty (easy → hard) for better convergence:

```yaml
training:
  curriculum: true             # Enable curriculum learning
  curriculum_metric: length    # Sort by: length, perplexity, or loss
  curriculum_buckets: 4        # Number of difficulty stages
```


## Freeze Training

Freeze bottom layers of the model — train only the top layers (like LLaMA-Factory's `finetuning_type: freeze`).
Only `task: sft` (and `tts`, which trains through the SFT trainer) applies these two fields; on any
other task the config is refused at load, because that trainer would train every layer (#1497).
Within `sft` and `tts` they are applied on the text path with `backend: transformers` only:
`backend: mlx`, `backend: unsloth`, `modality: vision`, `modality: audio` and `stream_layers: true`
are refused at load too: those setups never read the fields and always attach an adapter, so the run
would train every layer (#1581).

```yaml
training:
  freeze_layers: 24    # Freeze first 24 layers, train the rest
  # OR
  freeze_ratio: 0.75   # Freeze 75% of layers from the bottom
```

Works with and without LoRA. When used with LoRA, the adapter is attached only to the unfrozen
layers, so a frozen layer stays frozen rather than training through a fresh adapter.

`freeze_layers`/`freeze_ratio` cannot currently be combined with LoRA when
`training.lora.target_modules` is a regex string (an explicit one, or an `auto`-resolved
architecture table entry such as Mistral 3.5's), when `training.lora.target_parameters` is
set (per-parameter LoRA, e.g. fused MoE experts), or with `training.lora.use_vera`. A
`target_modules` list entry that only matches modules outside a numbered decoder layer (such as
`embed_tokens`) and a freeze plan that freezes every layer refuse too. Each of these refuses when
the model is built rather than silently drop the freeze plan (#1432). A top-level `lm_head` target
is accepted and keeps its adapter.

Known limitation: a `target_modules` entry spelled as a full module path, such as
`model.layers.0.self_attn.q_proj`, keeps its adapter even when that layer is frozen, because peft
accepts an exact module name before it reads `layers_to_transform`. Use module-name suffixes such
as `q_proj` with `freeze_layers` / `freeze_ratio`.

## LISA — Layerwise Importance Sampling (v0.71.34)

LISA (Layerwise Importance Sampled AdamW, [arXiv:2403.17919](https://arxiv.org/abs/2403.17919)) targets full-fine-tuning quality at LoRA-like memory. **Measured at 7B+, it delivers the first half and not the second** — see [what it actually costs](#what-lisa-actually-costs-measured-at-3b-and-8b) below before choosing it over LoRA. Instead of picking layers once (that's Spectrum's static `unfrozen_parameters`), LISA re-samples a small random set of decoder layers **every N steps** and freezes the rest; the input embeddings, the LM head, and the final norm stay trainable throughout by default (set `lisa_train_embeddings: false` to freeze that group too — see [the memory trade-off](#reclaiming-the-always-on-overhead-lisa_train_embeddings) below).

```yaml
task: sft                 # or `pretrain` — continued pre-training is the same
                          # full-FT-of-active-layers mechanism (#307)
backend: transformers
modality: text
training:
  quantization: none      # LISA is full-FT of the active layers
  lisa_enabled: true
  lisa_num_layers: 2       # decoder layers active per interval (clamped to model depth)
  lisa_interval_steps: 20  # re-sample cadence, in global steps
  lisa_train_embeddings: true  # default; false freezes embeddings + head + final norm
```

Because only a handful of layers train at any moment (and their optimizer state is cleared when they're re-frozen), peak optimizer memory is roughly `embeddings + head + lisa_num_layers` — far below a full fine-tune, while every layer still gets updated over the course of training. LISA is `sft` or `pretrain` + `transformers` + `text` + `quantization: none` only, and is mutually exclusive with LoRA features, `freeze_layers`/`freeze_ratio`, and Spectrum's `unfrozen_parameters` (each independently decides what trains).

### What LISA actually costs, measured at 3B and 8B

Measured on one H100 80GB, Alpaca, 200 steps, 3 interleaved repeats per arm, SM
clock pinned at 1980 MHz throughout. LISA engagement was verified rather than
assumed: the trainable-layer set rotates exactly on the interval, the trainable
parameter count matches the arithmetic to the unit, and `lisa_num_layers` set to
*all* layers reproduces the full-fine-tuning arm to three decimals.

**Llama-3.1-8B-Instruct:**

| arm | peak VRAM | held-out loss |
|---|---|---|
| full fine-tuning | **does not fit** (OOM at 73.94 GB, also at `batch_size 1`) | – |
| LISA (2 layers / 20 steps) | 52.14 GB | 1.294 |
| LoRA r=16 | **34.56 GB** | **1.275** |

**Qwen2.5-3B-Instruct, each arm at its own better learning rate:**

| arm | peak VRAM | held-out loss |
|---|---|---|
| full fine-tuning | 57.60 GB | 1.2905 |
| LISA | 19.37 GB | **1.2463** |
| LoRA r=16 | **15.93 GB** | **1.2420** |

**The quality claim holds — LISA beat full fine-tuning at both learning rates.**
The memory claim does not: LISA is **1.22x LoRA at 3B and 1.51x at 8B**, and the
gap *widens* with scale.

The reason is structural rather than a tuning miss. The input embeddings, LM head
and final norm stay trainable **every** interval, and at 8B those are **70.7%** of
everything LISA trains (66.9% at 3B). So `lisa_num_layers` only controls about
30% of the cost, and the other 70% grows with vocabulary x hidden size — an
overhead LoRA never pays at all.

### Reclaiming the always-on overhead: `lisa_train_embeddings`

Set `lisa_train_embeddings: false` to freeze the always-on group (input
embeddings, LM head, final norm) so only the sampled `lisa_num_layers` decoder
layers train. Because that group is the majority of what LISA trains, this is
the knob that actually moves LISA's memory toward the LoRA-like target the paper
promises.

It is a **real trade, not a free win**: the always-on set is presumably
load-bearing for LISA's quality result, so freezing it may move held-out loss.
The default stays `true` (LISA exactly as published) precisely because this
should be a measured choice, not a silent change — measure both ways on your
model before committing to it.

> **Pre-flight caveat.** The analytical VRAM pre-flight still classifies LISA as
> full fine-tuning regardless of `lisa_train_embeddings`, so it does **not** yet
> credit the saving from freezing the always-on group — a frozen-embeddings run
> that would fit can still be refused before launch. This is deliberate:
> over-predicting is the safe failure (under-predicting is a silent spill on
> Windows), and crediting the saving needs a measured constant on GPU hardware.
> Use `--allow-oom-attempt` to launch a run the pre-flight conservatively
> refuses.

**Choose LISA when you need full-rank updates on a model too large to
full-fine-tune** — its real win is that 8B trains on a single 80 GB card where
full fine-tuning needs about 120 GB. Otherwise prefer LoRA: at these sizes it
matched or beat LISA on held-out loss while using a third less memory.

Defaults are well chosen and need no change at 7B+. Raising `lisa_num_layers`
degrades memory, speed **and** quality monotonically (3B held-out: 1.2504 at 2,
1.2673 at 8, 1.2950 at 16), and above 8 it OOMs an 80 GB card at 8B.
`lisa_interval_steps` sits in a wide flat optimum — 1 through 50 are
indistinguishable, and only 200 (a single sample over the run) degrades.

One caveat on the numbers: held-out quality here is in-distribution loss and token
accuracy on an Alpaca validation split, not a downstream benchmark, so it cannot
see capability regressions a task benchmark would. Full measurement record:
[`benchmarks/gate-h100-validation.md`](../benchmarks/gate-h100-validation.md).

Implementation note: the model is left fully trainable at trainer-setup time so HF's optimizer (built before the first callback fires) contains every decoder parameter; the LISA callback then toggles `requires_grad` per interval — frozen parameters produce no gradient and the optimizer skips them.


## Loss Watchdog

Auto-stop training when loss spikes above a threshold (like Axolotl's `loss_watchdog_threshold`):

```yaml
training:
  loss_watchdog: true           # Enable loss spike detection
  loss_watchdog_threshold: 3.0  # Stop if loss exceeds this value
  loss_watchdog_patience: 5     # Consecutive loss reports above threshold before stopping
```

The comparison is strictly `loss > loss_watchdog_threshold`; a report at or below
the threshold resets patience. A missing/`None` loss neither advances nor resets it.

**MLX text SFT:** reports occur every `logging_steps` completed optimizer updates
(every `logging_steps × gradient_accumulation_steps` micro-batches), plus at the
end of the budget. Each loss averages the micro-batches since the preceding
report. On the triggering report, training stops immediately after its completed
update, with no later update or partial accumulation group. Soup saves the current
`adapters.safetensors` and loadable `adapter_config.json`, runs the final validation
when `eval_steps` is configured, and releases the display and rewind recorder.
This saves the last state, not a best-loss or rollback checkpoint; a high loss can
already have affected its weights. No full optimizer-state resume is claimed.

The returned summary and `adapter_config.json` record `training_outcome`
(`stopped` or `completed`), `stop_reason` (`loss_watchdog` or `budget_completed`),
`completed_updates` and `completed_iterations`. `total_steps` remains in MLX
micro-batch units and counts only executed iterations. A controlled stop is a
successful command after saving; the CLI says **Training Stopped**, while the
tracker's `completed` status means the run finalized successfully. Consult the
adapter metadata for the termination reason. Non-finite reported training loss
raises an error immediately; save failures, other errors and keyboard interruption
propagate rather than becoming successful watchdog stops.

**Difference from Transformers:** its current watchdog applies the threshold
comparison to the losses reported by Trainer, with no separate non-finite check
in that callback. On MLX, a reported NaN, +Inf or -Inf is an immediate training
error, regardless of patience. The Transformers callback is unchanged.

With the watchdog disabled, existing MLX report cadence and behavior are unchanged.
Other MLX tasks/modalities still refuse it, and MLX continues to refuse
`loss_spike_recovery` because checkpoint recovery is a separate feature.

## Training Stability & Auto-Tuning

Pre-flight tuning + in-training stability nets. All flags are opt-in.

### LR Range Finder

Run a fast.ai-style geometric LR sweep before the real training run. Soup writes a JSON report with the recommended LR, the loss curve, and divergence point so you can pick the LR with confidence.

```bash
soup train --config soup.yaml \
  --find-lr \
  --find-lr-start 1e-7 \
  --find-lr-end 1e-1 \
  --find-lr-steps 100 \
  --find-lr-output ./lr_finder.json
```

The report contains the geometric `lrs[]`, raw + EMA-smoothed `losses[]`, the recommended LR (steepest negative gradient before divergence), the LR with min loss, and the divergence point if any.

The sweep takes one training row per step. With fewer rows than `--find-lr-steps`, it runs one step per row over the same `--find-lr-start` → `--find-lr-end` range and prints a line saying so. The recommendation needs at least 4 points, so `--find-lr-steps` must be at least 4 and a training set with fewer than 4 rows is refused before the model loads. If the loss turns non-finite partway through, the report covers the steps before it; if that leaves fewer than 4, the command says where it diverged instead of writing a report. If the sweep cannot run at all (the config does not load, the model or dataset cannot be loaded, or `torch` is not installed), the command names the cause, exits 1 and writes no report.

### Auto Warmup Schedule

```yaml
training:
  warmup_auto: true       # Pick warmup_steps from dataset_size × epochs × warmup_ratio
  warmup_ratio: 0.03      # 3% of total update steps (default)
```

Clamped to `[10, 1000]` so tiny datasets get some warmup and huge datasets don't burn half a million wasted steps.

### Auto Mixed-Precision

```yaml
training:
  auto_mixed_precision: true
```

Picks `bf16` on Ampere+, `fp16` on Turing or known fp16-stable models (Qwen2 / Qwen2.5 / Phi-3 / Phi-3.5), `no` on pre-Pascal. Multi-version pairs (`qwen2.5` vs `qwen2`, `phi-3.5` vs `phi-3`) match the longest substring deterministically.

Only `task: sft` reads the field (and `task: tts`, which trains through the SFT trainer); since #1618, setting it on any other task is refused at config load, because those trainers pick bf16/fp16 on their own and would train at their default precision regardless of this key.

The experimental QuEST route (`quantization_aware: quest`) refuses this flag at
config load because its evidence covers BF16, not FP16; see the
[QuEST evidence boundary](performance-and-quantization.md#evidence-boundary).

### Loss Spike Auto-Recovery

Extends the watchdog: when the watchdog halts training on a loss spike, writes `<output>/spike_recovery.json` with decayed LR and attempt count for re-launch (e.g. `soup train --resume`). Capped at `loss_spike_recovery_max_attempts` (default 3); each launch hitting a spike evaluates prior completed attempts against the cap (`attempts < max_attempts`), writes the incremented count, and sets `should_recover: false` once the recovery budget is exhausted.

The hint stays under the working directory and is replaced atomically. A symbolic-link
target is refused with a warning; a failed replacement preserves the previous hint and
does not consume an attempt. Replacing a hard-linked hint leaves its other links unchanged.

```yaml
training:
  loss_watchdog: true                   # required
  loss_spike_recovery: true             # opt in to recovery
  loss_spike_recovery_max_attempts: 3
  loss_spike_recovery_lr_decay: 0.5     # halve LR each recovery
```

> **Backend Note:** Setting `loss_spike_recovery: true` is refused on `backend: mlx`
> at config validation: MLX does not implement checkpoint rollback or recovery hints.

### Convergence Detector

```yaml
training:
  convergence_detection: true
  convergence_window: 50      # Steps to inspect for plateau / oscillation
  convergence_rel_tol: 0.005  # Relative range below this == plateau
```

Computes `continue` / `early_stop` / `lower_lr` advice from the loss curve for
callers that invoke the detector. `convergence_detection` is reported as not
enforced by `soup train`; setting `convergence_window` or `convergence_rel_tol` away from their
defaults warned at load in v0.76 and is refused at load as of v0.77 (#808).

### VRAM Pressure Advisory

```yaml
training:
  grad_accum_auto_tune: true
  grad_accum_pressure_threshold: 0.92
```

Records peak memory each step. Pressure is peak allocated memory divided by the total memory of the CUDA device the run uses (#1620), so the threshold means the same thing on a 4 GB laptop card and on an 80 GB accelerator. When pressure crosses the threshold, recommends a new `(batch, accum)` pair preserving effective batch (capped at `accum=1024`), together with the measured peak and total.

> **Backend Note:** Setting `grad_accum_auto_tune: true` is refused on `backend: mlx` at config validation (there is no VRAM total to measure pressure against on unified memory).

> **v0.33.0:** `--find-lr` now runs an in-process LR-sweep training loop (replaces the v0.32.0 stub curve), spike-recovery writes a `spike_recovery.json` hint with the decayed LR for re-launch, and the grad-accum advisory prints a recommended `(batch, accum)` pair when VRAM pressure crosses the threshold. Live optimizer-state rewind and live DataLoader rebuild remain follow-ups (HF Trainer / TRL upstream constraints).


## Training Intelligence (Forgetting + Checkpoint Quality)

The `forgetting_detection`, `checkpoint_intelligence`, `early_stop_on_regression`, `convergence_detection`, and `forgetting_threshold` settings are
reserved for planned in-training callbacks. They are accepted by the schema but
are not enforced during training in this build. `soup train` prints an advisory note
when one is set away from its default, directing users to `--gate <suite.yaml>`.
Their tuning knobs (`forgetting_eval_steps`, `forgetting_benchmark`, `forgetting_stop`,
`checkpoint_eval_steps`, `checkpoint_eval_metric`, `checkpoint_eval_tasks`, `checkpoint_keep_top`,
`convergence_window`, and `convergence_rel_tol`) warned at load in v0.76 and are
refused at load as of v0.77 per #808, alongside `early_stop_patience` (#761): the
refusal names each one set away from its default, one line per field, and says to
delete the line.

Use the live eval gate for regression detection and automatic stopping today:

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: sft
data:
  train: ./data/chat.jsonl
training:
  epochs: 5
  eval_gate:
    enabled: true
    suite: ./evals/gate.yaml
    every_n_epochs: 1
    regression_threshold: 0.05
    baseline: registry://llama31-chat-v1
    on_regression: stop
```

The gate runs at epoch boundaries. See [Eval-Gated Training](evaluation.md#eval-gated-training)
for the suite format and post-training invocation.


## GaLore (Memory-Efficient Full-Parameter Training)

Train without LoRA using gradient low-rank projection — saves optimizer memory:

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: sft

data:
  train: ./data/train.jsonl
  format: alpaca

training:
  epochs: 3
  lr: 2e-5
  quantization: none      # Required: GaLore is incompatible with quantization
  use_galore: true
  galore_rank: 128
  galore_update_proj_gap: 200
  galore_scale: 0.25
```

> **Note:** GaLore requires `quantization: none` and `backend: transformers` (not unsloth).
