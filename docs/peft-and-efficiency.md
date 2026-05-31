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

---

## LongLoRA Forward Override

When `use_longlora: true` is set on an SFT config with a Llama / CodeLlama /
Mistral / Qwen / Phi base, the trainer wraps the model in a
`LongLoRAForwardOverride` context that monkey-patches every attention forward
to apply the S² shifted-sparse shift (paper §3.2) — half the heads are rolled
by `group_size // 2` along the sequence dim. Restoration on context exit is
idempotent and best-effort safe; FlashAttention v3 builds are rejected at the
schema gate (the custom-mask kernels conflict).


## Multipack — FFD Bin-Packing Sampler

Soup's largest single throughput win on chat fine-tuning over uneven-length data. Instead of padding every sample to `max_length`, Multipack uses **First-Fit-Decreasing bin packing** to group variable-length samples into bins approaching `batch_size × max_seq_length` — eliminating padding waste.

```yaml
training:
  multipack: true
  packing: false   # mutually exclusive with multipack
```

**How it composes:**
- **Multipack** picks WHICH samples go together (FFD packing).
- **`packing_cross_doc_attn_mask`** sets HOW the attention mask is built (block-diagonal causal — see section above).
- The two layer cleanly: enable both for FA-incompatible backends; FA varlen path is auto-selected when FlashAttention is available.

**Architecture allowlist** — 18 supported (Llama 3.x, Qwen 2/3, Mistral, Gemma 2/3, Phi 3/4, DeepSeek V2/V3, Mixtral, Falcon, StableLM, SmolLM2). Unknown architectures **fail loudly at config-load** instead of silently no-opping (critical fix vs Axolotl's silent-miss footgun).

**Live wiring** — landed. SFT and Pretrain trainer wrappers actually instantiate the multipack subclass when `multipack: true` is set. The factory's `get_train_dataloader` override installs `MultipackBatchSampler(real_batches=False)` (yields a flat `list[int]` per packed sequence — DataLoader-compatible) as the DataLoader's `batch_sampler=`, forwarding `dataloader_drop_last`/`num_workers`/`pin_memory` from `TrainingArguments`. The `_get_train_sampler` override stays as a defensive no-op fallback that always delegates to super, so any HF eval / prediction loop bypassing `get_train_dataloader` still gets the correct `Sampler[int]` shape (no nested-list shape mismatch). Multipack is **sft / pretrain only** on the `transformers` backend; preference / RLHF trainers and MLX backend get distinct error messages naming the actual reason. Datasets must expose `input_ids` (preferred) or `length` per row; raw text triggers an all-zeros warning.

**DoS hardening** — the FFD packer caps at 1M items (algorithm is O(N²) worst-case); the 4D mask builder caps allocations at 2³¹ cells; the chat-template Jinja analyzer caps at 128KB. Every numeric input rejects `bool` explicitly (matches v0.30.0+ project policy).

The `JinjaTemplateAnalyzer` (also v0.37.0) walks chat-template ASTs to discover non-standard `message.<field>` references (`tool_calls`, `name`, `weight`, `train`) — used by the v0.36.0 `train_on_messages_with_train_field` path so per-message training masks are aware of fields beyond `role` / `content`. The analyzer parses templates without rendering them, so a crafted `soup.yaml` cannot trigger SSRF.


## Long Context — YaRN, Llama 3.1 NTK, LongLoRA

Soup ships five RoPE-scaling strategies plus a LongLoRA schema gate:

```yaml
# soup.yaml
base: meta-llama/Llama-3.1-8B
task: sft
data:
  train: ./data.jsonl
  max_length: 32768  # extend from 8k → 32k
training:
  rope_scaling_type: yarn      # linear | dynamic | yarn | longrope | llama3
  yarn_factor: 4.0             # 4x extension
  yarn_beta_fast: 32
  yarn_beta_slow: 1
  yarn_attn_factor: 1.0
  gradient_checkpointing: true  # required above 64k
```

**YaRN.** Best quality for 4-8x extension. Tunables (`yarn_factor`, `yarn_attn_factor`, `yarn_beta_fast`, `yarn_beta_slow`) only apply when `rope_scaling_type=yarn`; the schema rejects them otherwise. Pure-Python math kernels are exposed at `soup_cli.utils.long_context.yarn_*` for reference / config-emit. The actual RoPE rotation runs inside HF Transformers.

**Llama 3.1 NTK-aware.** Use `rope_scaling_type: llama3` for the canonical Llama 3.1 frequency-band scaling (`scale_factor=8`, `low_freq_factor=1`, `high_freq_factor=4`, `old_context_len=8192`). `detect_llama3_rope_in_config` auto-detects the block in any HF model config dict. Omit `rope_scaling_type` from your YAML (so it stays `None`) on a Llama 3.1 base and `apply_long_context_config` will auto-pick `llama3` by reading `model.config.rope_scaling` at load time — explicit caller picks still win.

**LongLoRA S² (schema-only this release).** `training.use_longlora: true` requires `task=sft`, `backend=transformers`, a base in the architecture allowlist (Llama / CodeLlama / Mistral / Qwen / Phi — Mixtral excluded), and `use_ring_attention=false`. The schema also rejects the combo with FlashAttention v3 installed (the S² custom-mask kernel conflicts with FA-v3 native custom-mask). The schema gate fails fast at config load; live forward override mirroring LlamaFactory `model/model_utils/longlora.py` lands in a follow-up release.

```yaml
# Llama 3.1 with NTK-aware scaling out to 128k
base: meta-llama/Llama-3.1-8B
training:
  rope_scaling_type: llama3
  gradient_checkpointing: full
data:
  max_length: 131072
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
  expand_layers: 4              # append 4 zero-init decoder blocks
  freeze_trainable_layers: 4    # train only the appended blocks
  lr: 5e-5
  epochs: 1
```

**What happens at trainer start.** Soup deep-copies the last `expand_layers` decoder blocks, zero-inits each clone's residual projections (`mlp.down_proj` + `self_attn.o_proj`) so the appended block initially acts as identity, appends them to `model.model.layers`, and updates `config.num_hidden_layers`. When `freeze_trainable_layers > 0` is set, every parameter except the appended blocks is frozen — this is the canonical LLaMA Pro "train only new blocks" recipe.

**Scope.** Works on both `task: sft` and `task: pretrain` with `backend: transformers`. Bounds: `expand_layers ∈ [1, 64]`. Over-expansion (more new blocks than the base has layers) silently clamps to the base layer count. Non-Llama-shaped architectures (e.g. Falcon's `dense_4h_to_h`) emit a `warnings.warn` because the residual zero-init heuristic only matches the standard `down_proj` / `o_proj` names — the appended blocks are still appended + trainable, but lose the identity-init guarantee.


## Optimizer & PEFT Zoo

Pick from a wider catalogue of optimizers, target individual modules with their own LR, and use quantization-aware LoRA initialisation:

```yaml
training:
  # 30+ optimizers — HF-native, bnb, BAdam, APOLLO, Adam-mini, lomo,
  # grokadamw, schedule_free, muon, dion, came_pytorch, ao_adamw_{fp8,4bit,8bit}
  optimizer: badam

  # Per-module LR override (first match wins; remaining params use base lr)
  lr_groups:
    q_proj: 1e-4
    v_proj: 5e-5
    mlp:    1e-5

  # Friendly aliases for users coming from LlamaFactory / Axolotl
  load_in_8bit: true        # equivalent to quantization: 8bit
  # load_in_16bit: true     # equivalent to quantization: none

  lora:
    init_strategy: loftq    # quantization-aware LoRA init (also: pissa / olora / random)
    loftq_iter: 1
    loftq_bits: 4

  # LLaMA Pro block expansion (schema only in v0.41.0; live wiring in v0.41.1)
  expand_layers: 4
  freeze_trainable_layers: 4
```

Catch-all friendly errors: typos in `optimizer:` are rejected at config-load with the v0.41.0 additions listed in the message; `lr_groups` patterns are validated as compilable regexes (length-capped + benign-string ReDoS probe); `load_in_8bit` mixed with `load_in_16bit` raises rather than picking one silently.

See `soup_cli.utils.optimizer_zoo.SUPPORTED_OPTIMIZERS` for the complete optimizer allowlist.


## LoRA Quality — PiSSA, ReLoRA, Per-Pattern Rank, Surgical Patches

Five PEFT-surface improvements that LlamaFactory and Axolotl maintain:

```yaml
training:
  lora:
    init_strategy: pissa          # 'random' (default), 'pissa', 'olora'
    rank_pattern:                 # per-target-module rank override
      q_proj: 8
      v_proj: 16
    alpha_pattern:                # per-target-module alpha override
      q_proj: 16
  relora_steps: 500               # magnitude-prune LoRA every 500 steps
  relora_warmup_ratio: 0.1        # skip first 10% of training
  relora_prune_ratio: 0.9         # zero out smallest 90% by magnitude
  relora_reset_optimizer: true    # clear optimizer state on each fire
```

**PiSSA** initializes the LoRA pair from the SVD of the base weight, giving faster
early convergence than random init at the cost of one extra SVD pass on the first
epoch. `init_strategy: olora` is also accepted; setting the legacy `use_olora: true`
auto-aligns for back-compat.

**ReLoRA** fires every N global steps, magnitude-prunes the LoRA adapter weights
(keeping the top `1 - relora_prune_ratio` by absolute value), and optionally clears
optimizer state for the pruned parameters so momentum doesn't fight the new sparse
weights. Useful for very long training runs where the LoRA capacity saturates.

**Per-pattern rank/alpha** map module name patterns to integer ranks. Useful in MoE
configs where expert FFNs need lower rank than attention. Caps: 256 keys × value 1024.

**Surgical patches** (Gemma 4 `ClippableLinear` swap, fused-MoE 3-D expert
`lora_dropout` strip) auto-fire when the model name and architecture match. Both are
gated and silent on unrelated models.

**Template registry** — the 16 built-in templates now live as
`soup_cli/templates/*.yaml` with a `manifest.json` index. `soup init --template <name>`
reads the YAML; the inline copies in `schema.py` stay as a back-compat fallback,
deprecated in favour of the YAML registry.

**Multi-trainer scope** — ReLoRA and the surgical patches are wired into every
transformer-backend trainer: `sft`, `dpo`, `grpo`, `kto`, `orpo`, `simpo`, `ipo`,
`ppo`, `reward_model`, `pretrain`, `embedding`, `bco`, plus the unified
`task: preference` dispatcher. Schema cross-validator only rejects MLX backend
(the callback is HF Trainer-specific).


## DoRA (Weight-Decomposed LoRA)

Enable DoRA for improved LoRA quality with magnitude decomposition:

```yaml
training:
  lora:
    r: 64
    alpha: 16
    use_dora: true  # Enable DoRA
```

Works with all training tasks and backends.


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

Works with SFT, DPO, KTO, ORPO, SimPO, and IPO tasks.


## Sample Packing

Pack multiple short samples into one sequence for faster training:

```yaml
training:
  packing: true  # Pack short samples together (faster training)
```

Works with SFT and Pretrain tasks. Warning emitted if `max_length < 256`.


## Curriculum Learning

Sort dataset by difficulty (easy → hard) for better convergence:

```yaml
training:
  curriculum: true             # Enable curriculum learning
  curriculum_metric: length    # Sort by: length, perplexity, or loss
  curriculum_buckets: 4        # Number of difficulty stages
```


## Freeze Training

Freeze bottom layers of the model — train only the top layers (like LLaMA-Factory's `finetuning_type: freeze`):

```yaml
training:
  freeze_layers: 24    # Freeze first 24 layers, train the rest
  # OR
  freeze_ratio: 0.75   # Freeze 75% of layers from the bottom
```

Works with and without LoRA. When used with LoRA, LoRA is applied only to unfrozen layers.


## Loss Watchdog

Auto-stop training when loss spikes above a threshold (like Axolotl's `loss_watchdog_threshold`):

```yaml
training:
  loss_watchdog: true           # Enable loss spike detection
  loss_watchdog_threshold: 3.0  # Stop if loss exceeds this value
  loss_watchdog_patience: 5     # Consecutive steps above threshold before stopping
```


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

### Loss Spike Auto-Recovery

Extends the watchdog: instead of stopping on a spike, decay LR and resume. Capped at 3 attempts by default.

```yaml
training:
  loss_watchdog: true                   # required
  loss_spike_recovery: true             # opt in to recovery
  loss_spike_recovery_max_attempts: 3
  loss_spike_recovery_lr_decay: 0.5     # halve LR each recovery
```

### Convergence Detector

```yaml
training:
  convergence_detection: true
  convergence_window: 50      # Steps to inspect for plateau / oscillation
  convergence_rel_tol: 0.005  # Relative range below this == plateau
```

Surfaces `continue` / `early_stop` / `lower_lr` advice based on the loss curve.

### VRAM Pressure Advisory

```yaml
training:
  grad_accum_auto_tune: true
  grad_accum_pressure_threshold: 0.92
```

Records peak memory each step. When pressure crosses the threshold, recommends a new `(batch, accum)` pair preserving effective batch (capped at `accum=1024`).

> **v0.33.0:** `--find-lr` now runs an in-process LR-sweep training loop (replaces the v0.32.0 stub curve), spike-recovery writes a `spike_recovery.json` hint with the decayed LR for re-launch, and the grad-accum advisory prints a recommended `(batch, accum)` pair when VRAM pressure crosses the threshold. Live optimizer-state rewind and live DataLoader rebuild remain follow-ups (HF Trainer / TRL upstream constraints).


## Training Intelligence (Forgetting + Checkpoint Quality)

Two optional in-training evaluators that run alongside your main loss curve.

**Forgetting detection** — runs a small benchmark during training to detect catastrophic forgetting (quality regression on abilities the base model had). Can auto-stop if forgetting exceeds a threshold.

```yaml
training:
  forgetting_detection: true
  forgetting_eval_steps: 500       # How often to evaluate (10-10,000)
  forgetting_benchmark: mmlu        # Baseline benchmark to track
  forgetting_threshold: 0.10        # Regression threshold (0.01-0.50)
  forgetting_stop: true             # Halt training on breach (default: warn only)
```

**Checkpoint intelligence** — tracks a quality metric across checkpoints and keeps only the top-N by eval score (not by loss). Pairs nicely with `early_stop_on_regression`.

```yaml
training:
  checkpoint_intelligence: true
  checkpoint_eval_steps: 500
  checkpoint_eval_metric: accuracy   # or: bleu, rouge, exact_match, custom
  checkpoint_eval_tasks: ./evals/sanity.jsonl
  checkpoint_keep_top: 3             # Keep the 3 best (1-20)
  early_stop_on_regression: true
  early_stop_patience: 3             # Stop after N regressions (1-10)
```

Checkpoint pruning refuses to delete symlinks or paths outside the output directory — safe to run on any `output:` path.


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


