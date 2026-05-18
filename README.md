<p align="center">
  <img src="soup.png" alt="Soup" width="280">
</p>

<h1 align="center">Soup</h1>

<p align="center">
  <strong>Fine-tune LLMs in one command. No SSH, no config hell.</strong>
</p>

<p align="center">
  <a href="https://trysoup.dev">Website</a> &middot;
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#features">Features</a> &middot;
  <a href="#data-tools">Data Tools</a> &middot;
  <a href="#experiment-tracking">Tracking</a> &middot;
  <a href="#model-evaluation">Eval</a> &middot;
  <a href="#all-commands">Commands</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/soup-cli/"><img src="https://img.shields.io/pypi/v/soup-cli?color=blue" alt="PyPI"></a>
  <a href="https://pepy.tech/project/soup-cli"><img src="https://img.shields.io/pepy/dt/soup-cli?color=blue" alt="Downloads"></a>
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0 License">
  <a href="https://github.com/MakazhanAlpamys/Soup/actions"><img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/MakazhanAlpamys/65fdc943f85f3b2c46ecddb415c2b779/raw/soup_tests.json" alt="Tests"></a>
  <a href="https://github.com/MakazhanAlpamys/Soup/actions"><img src="https://github.com/MakazhanAlpamys/Soup/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://trysoup.dev"><img src="https://img.shields.io/badge/website-trysoup.dev-blue" alt="Website"></a>
</p>

---

Soup turns the pain of LLM fine-tuning into a simple workflow. One config, one command, done.

```bash
pip install soup-cli
soup init --template chat
soup train
```

## What's New

Latest highlights only. Full history: [GitHub Releases](https://github.com/MakazhanAlpamys/Soup/releases).

**v0.59.0 — Governance & Provenance: every Soup run is procurement-ready.** ML-BOMs, SLSA-3 attestations, EU AI Act Annex XI/XII auto-docs, HIPAA/SOC2 audit log, SR 11-7 reproducibility receipts. The compliance story no SaaS can tell because the operator is the only one who sees weights, dataset, eval, and cost together. v0.59 is pure orchestration on top of v0.26 Registry + v0.34 cost tracker + v0.56 diagnose — no new training code, just structured exporters that emit the formats procurement demands.

- **`soup bom emit --format cyclonedx|spdx|both`** — CycloneDX 1.6 ML-BOM + SPDX 2.3 + AI-profile dual emitter from any `RegistryEntry`. Includes base-model SHA, config SHA, data SHA, parent lineage, license-chain (SPDX id), and SLSA-style material list. Atomic write under cwd containment + `os.lstat` symlink rejection (TOCTOU-safe, mirrors v0.33.0 #22).
- **`soup attest emit --stage train --subject <name> --sha <64hex>`** — in-toto v1 Statement wrapping a SLSA-3 provenance v1 predicate. Stage allowlist (`extract` / `train` / `eval` / `export` / `publish`). Sigstore + ed25519 signing live in v0.59.1; the schema + atomic-write surface ships now so CI can integrate.
- **`soup train --annex-xi <out.md>`** — EU AI Act Annex XI Section 1+2 (technical documentation) + Annex XII (Article 53(1)(d) public training summary) markdown renderer. Top-10 domains, modality breakdown, FLOPs / kWh / CO₂ footprint. Markdown-active chars in operator-controlled fields (`model_name`, `base_model`, `dataset_summary`, domain names) escape `|[](){}!<>` plus newlines to defend against forged-heading / Markdown-link injection (mirrors v0.29.0 model-card v2 escape).
- **`soup audit-log tail / rotate`** — append-only JSONL audit at `~/.soup/audit.jsonl` (override via `SOUP_AUDIT_LOG_PATH`, containment-checked to `$HOME / $CWD / $TMPDIR`). Splunk/ELK ingestion-ready. PII redaction across every string field via the v0.40.3 `_SECRET_RE` policy (`hf_*` / `sk-*` / `Bearer …` → `<redacted>`). POSIX `O_NOFOLLOW` + `0o600` perms; rotation at 100 MiB with symlink rejection at the backup path.
- **`soup train --repro-receipt <out.json>`** — SR 11-7-style reproducibility receipt: seeds (torch / numpy / python), Python version, OS + arch, Soup version, kernel versions (CUDA / cuDNN / NCCL — best-effort from torch when available), GPU model + driver. Atomic write, cwd-contained.
- **CO₂ + energy schema (Part F).** `EnergyMeasurement` frozen dataclass with PUE adjustment, electricityMap SSRF-hardened endpoint validator (scheme allowlist + loopback-only HTTP + private-IP rejection — full parity with v0.51.0 `validate_hub_endpoint`). CodeCarbon hook lands in v0.59.1.
- **Shared `atomic_write_text` helper.** All four v0.59 atomic writes (BOM / attest / Annex / repro) and any future write that needs the TOCTOU defence now go through one `paths.atomic_write_text` — single-source-of-truth so a future contributor cannot accidentally drop the symlink check (mirrors v0.40.6 / v0.53.5 `peft_wiring` centralisation policy).
- **+93 new tests** (9193 → ~9286 net). Review-fix coverage: 3 HIGH (audit-log TOCTOU on rotation, `O_NOFOLLOW` on append, redaction extended to host_id/operator_id/command) + 5 MEDIUM (Annex markdown escape, env-override containment, atomic_write centralisation, bom artifact size_bytes validation, IPv6 hostname stripping cleanup) + 4 LOW.

## Why Soup?

Training LLMs is still painful. Even experienced teams spend 30-50% of their time fighting infrastructure instead of improving models. Soup fixes that.

- **Zero SSH.** Never SSH into a broken GPU box again.
- **One config.** A simple YAML file is all you need.
- **Auto everything.** Batch size, GPU detection, quantization — handled.
- **Works locally.** Train on your own GPU with QLoRA. No cloud required.

## Quick Start

### 1. Install

```bash
# From PyPI (recommended):
pip install soup-cli

# Or from GitHub (latest dev):
pip install git+https://github.com/MakazhanAlpamys/Soup.git
```

### 2. Create config

```bash
# Interactive wizard
soup init

# Or use a template
soup init --template chat       # conversational fine-tune
soup init --template code       # code generation
soup init --template medical    # domain expert
soup init --template reasoning  # GRPO reasoning training
soup init --template vision     # vision/multimodal fine-tune
soup init --template kto        # KTO unpaired preference alignment
soup init --template orpo       # ORPO (no reference model needed)
soup init --template simpo      # SimPO length-normalized preference
soup init --template ipo        # IPO regularized preference
soup init --template bco        # BCO binary classifier preference (v0.40.0)
soup init --template rlhf       # full RLHF pipeline (SFT→RM→PPO)
soup init --template pretrain   # continued pre-training on raw text
soup init --template moe        # MoE fine-tuning (ScatterMoE LoRA)
soup init --template longcontext # 128k+ context fine-tuning
soup init --template embedding  # sentence embedding fine-tuning
soup init --template audio      # audio/speech model fine-tuning
```

### 3. Train

```bash
soup train --config soup.yaml
```

That's it. Soup handles LoRA setup, quantization, batch size, monitoring, and checkpoints.

### 4. Test your model

```bash
soup chat --model ./output
```

### 5. Push to HuggingFace

```bash
soup push --model ./output --repo your-username/my-model
```

### 6. Merge & Export

```bash
# Merge LoRA adapter with base model
soup merge --adapter ./output

# Export to GGUF for Ollama / llama.cpp
soup export --model ./output --format gguf --quant q4_k_m

# Export to ONNX (pip install 'soup-cli[onnx]')
soup export --model ./output --format onnx

# Export to TensorRT-LLM (pip install 'soup-cli[tensorrt]')
soup export --model ./output --format tensorrt

# Export to AWQ quantized model (pip install 'soup-cli[awq]')
soup export --model ./output --format awq --bits 4 --group-size 128

# Export to GPTQ quantized model (pip install 'soup-cli[gptq]')
soup export --model ./output --format gptq --bits 4 --group-size 128

# BitNet 1.58-bit + TQ1_0 GGUF (schema-locked in v0.52.0; live conversion in v0.52.1)
soup export --model ./output --format bitnet
soup export --model ./output --format tq1_0
```

## Config Example

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: sft
# backend: unsloth  # 2-5x faster, pip install 'soup-cli[fast]'

data:
  train: ./data/train.jsonl
  format: alpaca
  val_split: 0.1

training:
  epochs: 3
  lr: 2e-5
  batch_size: auto
  lora:
    r: 64
    alpha: 16
  quantization: 4bit

output: ./output
```

## Data Flywheel (`soup loop`)

The full *production traces → preference pairs → Eval-Gated DPO → canary deploy → rollback* loop, driven from a single CLI. Connects v0.26 Trace-to-Preference + Eval-Gated Training + Registry lineage + Quant-Lobotomy verdicts + Soup Cans + v0.25 Autopilot + v0.54 Advise + v0.55 Eval Design + v0.56 Diagnose.

```bash
# One-time setup
soup loop init registry://abc12 --eval evals/lock.json --baseline registry://prod \
    --monthly-budget 50usd --max-runs-per-day 3

# Inspect counters + status
soup loop status

# Run the daemon (foreground)
soup loop watch --poll-interval 300

# Background subprocess (writes PID, no shell)
soup loop watch --detach

# Promote a canary at 5% traffic with auto-rollback on MAJOR verdict
soup loop canary registry://candidate --traffic 5% --autoroll-on-regress

# Pause/resume the daemon between iterations (atomic state flip)
soup loop pause
soup loop resume

# Replay any recorded iteration
soup loop replay iter-20260515T120000-abcdef01
```

State lives in `.soup/loop.yaml` (atomic write, cwd-contained, symlink-rejected). Per-iteration manifests under `.soup-loops/<iter-id>/iteration.json` are laid out so a v0.26 Soup Can can wrap them directly. The canary router is deterministic (SHA-256 hash of conversation id) and sticky-on-rollback — a flaky verdict can't ping-pong traffic between adapters.

## Pre-flight Decision (`soup advise`)

Run BEFORE you spend 8 hours on a GPU. `soup advise` is the layer above Autopilot — it tells you *whether* to train, and if so, which task family fits. Pure-Python heuristic, no GPU required for the verdict itself.

```bash
# Headline UX — one line gives you a verdict.
soup advise data.jsonl --goal "make our chatbot more concise"
#  Choice:     SFT   (or PROMPT_ENG / RAG / DPO / GRPO)
#  Confidence: 0.71
#  Why:        Task is summarization with 120 rows and healthy diversity ...
#  Flip when:  the prompt-engineering baseline already meets your target ...

# Optional 10-min ROI probe (zero/few-shot + RAG + 100-step LoRA).
soup advise data.jsonl --goal "summarize my reports" --probe

# Print the rubric / evidence trail of the last verdict.
soup advise explain

# Record this verdict to ~/.soup/advise_history.jsonl for later compare.
soup advise data.jsonl --goal "..." --record

# Show prior verdicts (newest first), with per-choice counts.
soup advise compare
```

**The rubric** (advisory, encoded explicitly so `explain` can print it):

1. Dataset rows expose paired `chosen` + `rejected` fields → **DPO**.
2. Task is `reasoning`, dataset has ≥500 rows AND carries `<think>` traces → **GRPO**.
3. Fewer than 50 rows → **PROMPT_ENG** (below the floor for meaningful fine-tuning).
4. Task is `factual_lookup` with high output variance → **RAG**.
5. Otherwise → **SFT**.

**Why this command exists.** "Choose fine-tuning vs RAG vs prompt-engineering" is the most-mis-made decision in the space. Reddit, HN, IBM, and Google Cloud all converge on the same advice (start with prompts, escalate to RAG, fine-tune as last resort) and almost everyone ignores it because nobody has the data to prove their case is the exception. Soup `autopilot` picks hyperparameters AFTER you've decided to train; `soup advise` owns the layer above. No trainer library has an incentive to tell users *not to train* — Unsloth's funnel, Axolotl's hosted business, LLaMA-Factory's Alibaba alignment all monetise the training event.

## Eval Design Pipeline (`soup eval design / discover / lock / coverage`)

Trainer libraries help you RUN evals — none help you DEFINE them. The eval-design
pipeline closes that gap with four CPU-only subcommands.

```bash
# 1. Draft a goal-conditioned suite from your training data.
soup eval design data.jsonl --goal "better at SQL" --output evals/design.json

# 2. Discover held-out canaries + memorization probes.
soup eval discover data.jsonl --num-clusters 5 --output evals/canaries.json

# 3. Freeze the design as a checksummed eval_suite artifact.
soup eval lock evals/design.json --output evals/locked.json

# 4. Heuristic gap analysis vs the task taxonomy.
soup eval coverage evals/design.json --task reasoning
```

`soup eval design` clusters training rows by TF-IDF salience, picks a scorer
per dimension (`exact_match` / `regex` / `judge` / `rlvr`) via a goal-keyword
dispatch matrix, and writes a versioned `evals/design.json` of frozen
`EvalDimension` rows.

`soup eval discover` runs farthest-first Jaccard clustering and emits a
`CanarySet` with three groups:

- `held_out` — cluster representatives that test generalisation.
- `adjacent_skills` — rare clusters that catch catastrophic forgetting.
- `memorization_probes` — 25 %-prefix truncations that catch verbatim regurgitation.

`soup eval lock` canonicalises the suite (sorted-key JSON, no whitespace),
computes a SHA-256 over the bytes that hit disk, and optionally attaches the
artifact to a Registry entry as `eval_suite`. Two designs hash identically
iff their semantic content matches.

`soup eval coverage` does heuristic gap analysis against the task taxonomy:
`reasoning` benefits from a `rlvr` dimension, `format_conversion` benefits
from both `regex` and `rlvr`, etc. Missing scorers surface as named
recommendations so operators can spot gaps before shipping the gate.

## Pre-Push Regression Gate (`soup eval gate-install`)

Install a portable pre-push git hook that blocks the push when an adapter
regresses past a tolerance. Threshold checks use paired-bootstrap 95 % CI
so a single outlier row doesn't flip the gate.

```bash
soup eval gate-install --baseline run-abc-123 --suite evals/locked.json
```

The generated `.git/hooks/pre-push` script:

- Compares against a baseline run id from the Soup registry.
- Watches four metrics: `task_accuracy`, `refusal_rate`, `format_validity`,
  `p95_latency_ms`.
- Treats `task_accuracy` / `refusal_rate` / `format_validity` as higher-is-better
  and `p95_latency_ms` as lower-is-better; regression is decided per metric on the
  paired-bootstrap CI bound (upper bound for higher-better, lower for lower-better).
- Uses `shlex.quote` on every embedded value — no shell-injection surface from a
  crafted run id or suite path.
- Refuses to overwrite an existing hook without `--force`; rejects pre-placed
  symlinks at the hook path (TOCTOU defence).

The hook is portable bash (`#!/usr/bin/env bash` shebang) and works under
Git-for-Windows' bundled bash on Windows.

## Autopilot (Zero-Config)

Skip the YAML entirely. Give Autopilot a base model, a dataset, and a goal — it analyzes your data, model, and hardware, then picks the task, quantization, LoRA rank, learning rate, epochs, and performance flags for you.

```bash
# Zero-config: pick everything automatically
soup autopilot --model meta-llama/Llama-3.1-8B-Instruct \
               --data ./data/train.jsonl \
               --goal chat

# Other goals: chat | code | reasoning | instruct | vision
soup autopilot --model Qwen/Qwen2.5-7B --data ./data/math.jsonl --goal reasoning

# Constrain to a GPU budget (1GB to 1TB)
soup autopilot --model <id> --data d.jsonl --goal chat --gpu-budget 24GB

# Preview the generated config without running
soup autopilot --model <id> --data d.jsonl --goal chat --dry-run
```

Autopilot writes a ready-to-run `soup.yaml`. Edit it by hand if needed, then `soup train`.

## Apple Silicon (MLX Backend)

Fine-tune on M1-M4 Macs via Apple's [MLX](https://github.com/ml-explore/mlx) framework — no CUDA, no emulation.

```bash
# Install MLX support
pip install 'soup-cli[mlx]'
```

```yaml
base: mlx-community/Llama-3.2-3B-Instruct-4bit
task: sft
backend: mlx  # Apple Silicon only

data:
  train: ./data/train.jsonl
  format: alpaca

training:
  epochs: 3
  lr: 2e-5
  lora:
    r: 16
    alpha: 32
```

MLX backend supports SFT, DPO, and GRPO. Use `soup recipes search --tag mlx` for ready-made Apple Silicon configs.

## Unsloth Backend (2-5x Faster Training)

Use the [Unsloth](https://github.com/unslothai/unsloth) backend for significantly faster training and up to 80% less VRAM:

```bash
# Install unsloth support
pip install 'soup-cli[fast]'
```

Then add one line to your config:

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: sft
backend: unsloth  # 2-5x faster, -80% VRAM

data:
  train: ./data/train.jsonl
  format: alpaca

training:
  epochs: 3
  lr: 2e-5
  quantization: 4bit
  lora:
    r: 64
    alpha: 16
```

Works with all training tasks: SFT, DPO, GRPO, PPO, KTO, ORPO, SimPO, IPO, and Pretrain. If unsloth is installed but not enabled, Soup will suggest it automatically.

> **Tip:** Soup auto-detects unsloth. When installed, you'll see a hint during `soup train` if you haven't enabled it yet.

## Continued Pre-training

Continue training a model on raw text for domain adaptation:

```yaml
base: meta-llama/Llama-3.1-8B
task: pretrain

data:
  train: ./data/corpus.jsonl   # {"text": "..."} or plain .txt files
  format: plaintext
  max_length: 4096

training:
  epochs: 1
  lr: 1e-5
  quantization: 4bit
```

```bash
soup init --template pretrain
soup train
```

## Knowledge Distillation

Train a small student model to match a larger teacher's output distribution.

```yaml
base: HuggingFaceTB/SmolLM2-135M
task: distill
modality: text
backend: transformers

data:
  train: ./data/chat.jsonl
  max_length: 2048
  chat_template: chatml

training:
  teacher_model: meta-llama/Llama-3.1-8B
  distill_divergence: forward_kl   # kl | forward_kl | reverse_kl | js
  distill_temperature: 2.0
  epochs: 3
  lr: 5e-5
  quantization: 4bit               # quantizes student only
```

Loss = student CE + (T**2) × KL(teacher_logits / T  ||  student_logits / T).
Teacher is loaded once, frozen via `requires_grad_(False)` + `.eval()`, and its
inputs / logits are auto-bridged across CPU / CUDA devices.

## Sequence Classification

Train a classifier head on top of any base model — supports single-label,
multi-label, and cross-encoder reranking.

```yaml
base: BAAI/bge-base-en-v1.5
task: classifier              # or `reranker`, `cross_encoder`
modality: text
backend: transformers

data:
  train: ./data/labelled.jsonl   # rows: {"text": "...", "label": "spam"} or {"text": "...", "label": [0, 1, 0]}
  max_length: 256

training:
  num_labels: 3
  classifier_kind: single_label   # or `multi_label`
  label_names: [ham, spam, promo] # required when labels are strings
  epochs: 5
  lr: 2e-5
  batch_size: 32
```

Routes `classifier` / `reranker` / `cross_encoder` through
`AutoModelForSequenceClassification`. Multi-label heads cap at 1024 entries per
row, dedup via set conversion, and reject null bytes in label strings.

## Reasoning Effort + EOT Control

gpt-oss-style reasoning-effort control for instruction tuning.

```yaml
training:
  reasoning_effort: high      # low | medium | high
  train_on_eot: true          # do NOT mask the EOT/EOS token in the loss
```

`reasoning_effort` injects `<|reasoning_effort|>high<|/reasoning_effort|>` into
the system turn (creating one if absent). `train_on_eot=True` makes the model
learn when to stop generating by training on the trailing EOS token instead of
masking it out. Both are gated to the SFT-family of tasks.

## EBFT / GDPO Loss Variants

Entropy-regularised SFT (`ebft_variant: structured | strided`) and generalised
DPO (`gdpo_variant: standard | length_normalized | margin`) — both attach
idempotently via `compute_loss` wrappers and auto-fire when the corresponding
variant field is set on `TrainingConfig`.

```yaml
# SFT with EBFT structured
training:
  ebft_variant: structured
  ebft_temperature: 1.0

# DPO with GDPO length_normalized
task: dpo
training:
  gdpo_variant: length_normalized
  dpo_beta: 0.1
```

## GRPO Objective Variants

Soup ships live math kernels for 6 GRPO objective variants in addition to the
default. Set `grpo_variant` in `training` and the trainer automatically
subclasses `trl.GRPOTrainer` to route `compute_loss` through the matching
kernel:

```yaml
task: grpo
training:
  reward_fn: accuracy
  num_generations: 4
  grpo_variant: gspo         # group-stabilised importance ratio
  # or: dapo / dr_grpo / bnpo / rft / two_sided
  # grpo_delta: 0.2          # required when grpo_variant=two_sided
```

Variants:

- **standard** — DeepSeek-R1-style baseline (delegates to TRL's `compute_loss`).
- **gspo** — group-stabilised importance ratio with per-batch control variate.
- **dapo** — decoupled asymmetric clipping (`eps_lo=0.2, eps_hi=0.28`).
- **dr_grpo** — token-sum without per-sample length normalisation.
- **bnpo** — length-normalised PPO surrogate.
- **two_sided** — symmetric clipping with operator-supplied `grpo_delta`.
- **rft** — rejection-sampling fine-tuning (only positive-advantage tokens contribute).

The stability callback (EMA ref-model update, replay buffer, TIS alert counter)
attaches automatically when any of `ref_model_ema_alpha` / `replay_buffer_size`
/ `tis_threshold` / etc. is set.

## Process Reward Model (PRM)

Train a scalar reward head over stepwise-supervised reasoning chains. Data
format is the v0.42.0 `prm` shape — one row per `{prompt, completions: [step1,
step2, ...], labels: [r1, r2, ...]}`:

```yaml
task: prm
data:
  format: prm
  train: ./prm_train.jsonl
  max_length: 2048
training:
  epochs: 1
  lr: 1.0e-5
```

The trainer loads `AutoModelForCausalLM`, attaches an `nn.Linear(hidden, 1)`
reward head, and computes MSE between predicted scalars at step-boundary tokens
and the per-step labels.

## LongLoRA Forward Override

When `use_longlora: true` is set on an SFT config with a Llama / CodeLlama /
Mistral / Qwen / Phi base, the trainer wraps the model in a
`LongLoRAForwardOverride` context that monkey-patches every attention forward
to apply the S² shifted-sparse shift (paper §3.2) — half the heads are rolled
by `group_size // 2` along the sequence dim. Restoration on context exit is
idempotent and best-effort safe; FlashAttention v3 builds are rejected at the
schema gate (the custom-mask kernels conflict).

## Weighted Multi-Objective Preference Loss

Mix DPO / SimPO / ORPO / IPO terms in one training run by setting
`preference_loss_weights` (must sum to 1.0):

```yaml
task: preference
training:
  preference_loss_weights:
    dpo: 0.6
    simpo: 0.4
```

The combine wrapper reads policy + reference summed log-probs from the inner
TRL trainer's per-batch inputs and computes a true weighted sum via the
in-tree `compute_dpo_term` / `compute_simpo_term` / `compute_orpo_term` /
`compute_ipo_term` kernels. BCO cannot be mixed with paired losses (data
format incompatible — rejected at config load).

## MoE Model Support

Fine-tune Mixture of Experts models (Mixtral, Qwen3-30B-A3B, DeepSeek V3) with ScatterMoE LoRA — applies LoRA to both attention layers and expert FFN layers:

```yaml
base: Qwen/Qwen3-30B-A3B
task: sft

training:
  moe_lora: true              # target expert + attention layers
  moe_aux_loss_coeff: 0.01    # router load-balancing loss
  quantization: 4bit
```

Soup auto-detects MoE architectures. Works with all training tasks.

```bash
soup init --template moe
soup train
```

## Vision / Multimodal Fine-tuning

Fine-tune vision-language models (LLaMA-3.2-Vision, Qwen2-VL, Pixtral) on image+text data:

```bash
# Install vision support
pip install 'soup-cli[vision]'

# Create a vision config
soup init --template vision

# Train
soup train --config soup.yaml
```

```yaml
base: meta-llama/Llama-3.2-11B-Vision-Instruct
task: sft
modality: vision

data:
  train: ./data/vision_train.jsonl
  format: llava
  image_dir: ./data/images
  val_split: 0.1

training:
  epochs: 3
  lr: 1e-5
  quantization: 4bit
  lora:
    r: 64
    alpha: 16
```

**Supported vision data formats:**

**LLaVA:**
```json
{"image": "photo.jpg", "conversations": [{"from": "human", "value": "<image>\nDescribe this image."}, {"from": "gpt", "value": "A cat on a mat."}]}
```

**ShareGPT4V:**
```json
{"image": "chart.png", "conversations": [{"from": "human", "value": "<image>\nWhat does this show?"}, {"from": "gpt", "value": "Quarterly revenue."}]}
```

`soup data inspect` automatically shows image statistics (count, formats, missing files) for vision datasets.

## Audio / Speech Fine-tuning

Fine-tune audio-language models (Qwen2-Audio, Whisper) on audio+text data:

```bash
# Install audio support
pip install 'soup-cli[audio]'

# Create an audio config
soup init --template audio

# Train
soup train --config soup.yaml
```

```yaml
base: Qwen/Qwen2-Audio-7B-Instruct
task: sft
modality: audio

data:
  train: ./data/audio_train.jsonl
  format: audio
  audio_dir: ./data/audio
  val_split: 0.1

training:
  epochs: 3
  lr: 1e-5
  quantization: 4bit
  lora:
    r: 64
    alpha: 16
```

**Audio data format:**
```json
{"audio": "recording.wav", "messages": [{"role": "user", "content": "Transcribe this audio."}, {"role": "assistant", "content": "Hello world."}]}
```

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
`bnb_4bit_quant_storage` emits a yellow warning. See [`docs/QUANTIZATION.md`](docs/QUANTIZATION.md)
for the full table.

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

## GRPO Plus — Objective Variants, Long-Context RL, Multi-Turn Agents

Soup ships seven GRPO objective variants, between-rollouts vLLM standby, four agent-rollout backends, seven stability/efficiency knobs, plus Process Reward Models and Vision-RL.

```yaml
# soup.yaml — DAPO with replay buffer and TIS truncation masking
base: meta-llama/Llama-3.1-8B-Instruct
task: grpo
data:
  train: ./prompts.jsonl
  format: chatml
training:
  reward_fn: accuracy
  num_generations: 8
  # New: GRPO objective variants
  grpo_variant: dapo                  # one of: gspo / dapo / dr_grpo / bnpo / two_sided / rft / standard
  # grpo_delta: 0.2                   # required when grpo_variant: two_sided
  grpo_fp16: true                     # FP16 RL (unsloth parity)
  # Long-context + memory-efficient RL
  long_context_grpo: true             # wires Tiled MLP when available
  vllm_sleep_mode: true               # between-rollouts vLLM standby
  # Multi-turn agent rollout
  rollout_backend: art                # one of: art / ruler / nemo_gym / openenv
  # Stability / efficiency knobs
  ref_model_ema_alpha: 0.99           # EMA sync policy → reference
  replay_buffer_size: 2048
  async_grpo_prefetch: true           # overlap rollout + train
  tis_threshold: 2.0                  # truncated importance sampling
  mask_truncated_completions: true    # paired with tis_threshold
  defer_rerolling: true
  skip_zero_advantage: true
  off_policy_mask_threshold: 0.5
```

Process Reward Models (stepwise-supervised):

```yaml
# soup.yaml
base: meta-llama/Llama-3.1-8B
task: prm                              # New: Process Reward Model
data:
  train: ./prm_dataset.jsonl
  format: prm                          # stepwise-supervised data shape
training:
  epochs: 3
  lr: 1e-5
```

Vision RL on Qwen2-VL / Pixtral / InternVL:

```yaml
# soup.yaml
base: Qwen/Qwen2-VL-7B-Instruct
task: grpo
modality: vision
data:
  train: ./vlm_prompts.jsonl
  format: llava
training:
  reward_fn: accuracy
  vision_grpo: true                    # VLM-RL opt-in
```

All flags ship as schema gates in v0.50.0; live loss kernels, vLLM sleep-mode plumbing, ART/RULER/NeMo Gym/OpenEnv launchers, and the PRM trainer wrapper land in v0.50.1 — schema accepts the values now so configs are stable.

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

## DPO Training

Train with preference data using Direct Preference Optimization:

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: dpo

data:
  train: ./data/preferences.jsonl
  format: dpo

training:
  epochs: 3
  dpo_beta: 0.1
  lora:
    r: 64
    alpha: 16
  quantization: 4bit
```

## Preference Variety — BCO + Unified Dispatcher + KL Variants

Five preference losses live behind one config knob. Pick a loss without
renaming your task, anneal β over training, and periodically refresh the
frozen reference.

### BCO (Binary Classifier Optimization)

Same input format as DPO; rows are split internally to TRL's BCO
unpaired schema (`{prompt, completion, label}`).

```yaml
task: bco
data:
  train: ./data/preferences.jsonl
  format: dpo
training:
  bco_beta: 0.1
```

### Unified preference dispatcher

Use `task: preference` + `training.preference_loss` to swap losses
without touching `task`. Hyperparameter sweeps over the loss type
itself become trivial.

```yaml
task: preference
data:
  train: ./data/preferences.jsonl
  format: dpo
training:
  preference_loss: dpo   # or simpo, orpo, ipo, bco
```

Legacy `task: dpo` / `task: simpo` / etc. remain first-class — the
unified surface is additive.

### KL-controlled DPO variants

Anneal β over training, periodically refresh the reference model:

```yaml
task: dpo   # or task: preference + preference_loss: dpo, or task: ipo
training:
  dpo_beta: 0.1
  dpo_beta_schedule: linear   # linear | cosine | exponential
  dpo_beta_end: 0.01
  dpo_ref_regen_epochs: 2     # copy student → ref model every 2 epochs
```

Both controls are gated to DPO-family tasks (`dpo`, `ipo`, or
`preference` with `preference_loss in {dpo, ipo}`); transformers
backend only.

### Multi-objective preference loss (schema-only in v0.40.0)

```yaml
task: preference
training:
  preference_loss_weights: {dpo: 0.7, bco: 0.3}
```

Schema validates 2–5 entries summing to 1. Live runtime weighted-loss
combination is wired in v0.40.1; v0.40.0 fails fast with an actionable
`NotImplementedError` if you actually try to train (same stub-then-live
pattern as v0.27.0 MII / v0.37.0 multipack / v0.38.0 quant menu /
v0.39.0 ReLoRA).

## GRPO Training (Reasoning)

Train reasoning models with Group Relative Policy Optimization (DeepSeek-R1 style):

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: grpo

data:
  train: ./data/reasoning_train.jsonl
  format: sharegpt
  max_length: 4096

training:
  epochs: 3
  lr: 1e-5
  grpo_beta: 0.1
  num_generations: 4
  reward_fn: accuracy   # or 'format', or path to custom .py
  lora:
    r: 64
    alpha: 16
  quantization: 4bit
```

```bash
# Create a reasoning config
soup init --template reasoning

# Train
soup train --config soup.yaml
```

**Built-in reward functions:**
- `accuracy` — checks if the final answer matches expected (supports `####` and `\boxed{}` formats)
- `format` — checks for structured `<think>...</think>` reasoning blocks

**Custom reward functions** — point to a Python file:
```python
# my_reward.py
def reward_fn(completions, **kwargs):
    """Score each completion. Return list of floats."""
    return [1.0 if "correct" in c[-1]["content"] else 0.0 for c in completions]
```
```yaml
training:
  reward_fn: ./my_reward.py
```

### Verifiable Rewards (RLVR)

Use `reward_fn: verifiable` with a `verifiable_domain` for deterministic, math-checkable rewards — no judge model, no heuristics. Great for GRPO on math, code, or structured-output tasks.

```yaml
training:
  reward_fn: verifiable
  verifiable_domain: math          # or: code, json_schema
  num_generations: 4
```

Three built-in domains:

| Domain | What it checks |
|---|---|
| `math` | Extracts the final numeric answer (supports `####`, `\boxed{}`) and compares via `float()` equality — no `eval()` on user output |
| `code` | Executes generated Python with a 5s timeout, 512 MB RLIMIT on POSIX, `python -I -S`, socket patch, ephemeral cwd. Output capped at 10KB. Warning panel on first use |
| `json_schema` | Validates output against a JSON Schema provided per-example in the dataset |

> **Note:** `code` domain runs untrusted generations. Soup sandboxes aggressively but never trust it for production-grade isolation — run in a VM or container for public data.

## Tool-Calling Fine-Tuning

Train models to emit structured function calls (OpenAI-style `tool_calls` with JSON arguments).

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: sft

data:
  train: ./data/tool_calls.jsonl
  format: tool-calling

training:
  epochs: 3
  lr: 2e-5
  quantization: 4bit
```

**Tool-calling data format:**
```json
{"messages": [
  {"role": "user", "content": "What's the weather in Paris?"},
  {"role": "assistant", "tool_calls": [
    {"id": "c1", "type": "function",
     "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}
  ]}
]}
```

Arguments are parsed as JSON only — never `eval()`. `soup eval custom` can score tool-call accuracy (function name + argument JSON equality).

```bash
soup init --template tool-calling
```

## PPO / Full RLHF Pipeline

Train models with the full RLHF pipeline: SFT warmup → Reward Model → PPO alignment.

```bash
# Create an RLHF config
soup init --template rlhf
```

**Step 1: SFT warmup** — fine-tune a base model on your data:
```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: sft
data:
  train: ./data/train.jsonl
  format: alpaca
output: ./output_sft
```

**Step 2: Train reward model** — learn preferences from human feedback:
```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: reward_model
data:
  train: ./data/preferences.jsonl
  format: dpo
output: ./output_rm
```

**Step 3: PPO alignment** — optimize the policy using the reward model:
```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: ppo
data:
  train: ./data/prompts.jsonl
  format: chatml
training:
  reward_model: ./output_rm
  ppo_epochs: 4
  ppo_clip_ratio: 0.2
  ppo_kl_penalty: 0.05
  lora:
    r: 64
    alpha: 16
  quantization: 4bit
output: ./output_ppo
```

PPO supports two reward sources:
- **Reward model** (`reward_model`): pre-trained reward model (from step 2)
- **Reward function** (`reward_fn`): callable function (same as GRPO — `accuracy`, `format`, or custom `.py`)

## KTO Training (Unpaired Preferences)

Train with unpaired preference data — no need for chosen+rejected pairs:

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: kto

data:
  train: ./data/kto_train.jsonl
  format: kto

training:
  epochs: 3
  kto_beta: 0.1
  lora:
    r: 64
    alpha: 16
  quantization: 4bit
```

**KTO data format:**
```json
{"prompt": "What is 2+2?", "completion": "4", "label": true}
{"prompt": "What is 2+2?", "completion": "Fish", "label": false}
```

## ORPO Training (No Reference Model)

ORPO combines SFT and alignment in one step — no reference model needed:

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: orpo

data:
  train: ./data/preferences.jsonl
  format: dpo

training:
  epochs: 3
  orpo_beta: 0.1
  lora:
    r: 64
    alpha: 16
  quantization: 4bit
```

## SimPO Training (Simple Preference)

SimPO uses length-normalized log probabilities as implicit rewards — reference-free:

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: simpo

data:
  train: ./data/preferences.jsonl
  format: dpo

training:
  epochs: 3
  simpo_gamma: 0.5
  cpo_alpha: 1.0
  lora:
    r: 64
    alpha: 16
  quantization: 4bit
```

## IPO Training (Regularized Preference)

IPO is a theoretically grounded DPO variant with stronger regularization:

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: ipo

data:
  train: ./data/preferences.jsonl
  format: dpo

training:
  epochs: 3
  ipo_tau: 0.1
  lora:
    r: 64
    alpha: 16
  quantization: 4bit
```

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

## Chat with your model

```bash
# Chat with a LoRA adapter (auto-detects base model)
soup chat --model ./output

# Specify base model explicitly
soup chat --model ./output --base meta-llama/Llama-3.1-8B-Instruct

# Adjust generation
soup chat --model ./output --temperature 0.3 --max-tokens 256
```

## Push to HuggingFace

```bash
# Upload model to HF Hub
soup push --model ./output --repo your-username/my-model

# Make it private
soup push --model ./output --repo your-username/my-model --private

# Group into a Collection
soup push --model ./output --repo your-username/my-model \
    --collection your-username/my-collection-abc123
```

## HuggingFace Hub Deep Integration

Soup treats HF Hub as a first-class artifact backend. One env var, one flag,
no token flags to plumb — all operations respect `huggingface-cli login`
credentials by default.

```bash
# Self-hosted Hub: set once, every command routes there.
export HF_ENDPOINT=https://hf.internal.example.com

# Auto-push each save_steps checkpoint to HF as a 'checkpoint-<N>' branch.
soup train -c soup.yaml --push-as your-username/my-model

# Resume from the latest branch pushed above.
soup train -c soup.yaml --push-as your-username/my-model --hf-resume

# Upload a local JSONL file as an HF dataset repo.
soup data push --input train.jsonl --hf-dataset your-username/my-dataset

# Wrap your fine-tuned model in a Gradio chat Space in one command.
soup deploy hf-space \
    --model your-username/my-model \
    --space your-username/my-chat-space \
    --template gradio-chat

# Or a Streamlit app:
soup deploy hf-space \
    --model your-username/my-model \
    --space your-username/my-chat-space \
    --template streamlit-chat
```

**Auto-resume workflow:** if training crashes, the next `soup train ... --push-as
... --hf-resume` call picks up the latest `checkpoint-<N>` branch from your HF
repo and downloads it back to `output_dir`, then resumes — no manual copy /
paste of checkpoint paths. Cwd containment and `local_dir_use_symlinks=False`
prevent filesystem escape from a crafted repo.

**Auth** follows standard HF conventions: `HF_TOKEN` env var > `HUGGINGFACE_HUB_TOKEN`
> `~/.cache/huggingface/token` (set by `huggingface-cli login`) > `~/.huggingface/token`.
No custom token flags. The deprecated `--token` on `soup push` still works but emits
a warning.

**Model card v2** is auto-generated on first push: it reads sidecar
`training_config.yaml` / `soup.yaml` to surface `task` / `base` / `lr` /
`optimizer`, and accepts an optional eval scorecard (markdown table).
Markdown-active chars in task names and scores are neutralised for safe
rendering on HF Hub.

## Merge LoRA Adapter

Merge a LoRA adapter with its base model into a standalone model:

```bash
# Auto-detect base model from adapter_config.json
soup merge --adapter ./output --output ./merged

# Specify base model and dtype
soup merge --adapter ./output --base meta-llama/Llama-3.1-8B --dtype bfloat16
```

## Export to GGUF

Export models to GGUF format for use with [Ollama](https://ollama.com/) and [llama.cpp](https://github.com/ggerganov/llama.cpp):

```bash
# Export LoRA adapter (auto-merges with base, then converts)
soup export --model ./output --format gguf --quant q4_k_m

# Export with different quantizations
soup export --model ./output --format gguf --quant q8_0
soup export --model ./output --format gguf --quant f16

# Export a full (already merged) model
soup export --model ./merged --format gguf

# Specify llama.cpp path manually
soup export --model ./output --format gguf --llama-cpp /path/to/llama.cpp
```

Supported quantizations: `q4_0`, `q4_k_m`, `q5_k_m`, `q8_0`, `f16`, `f32`

### ONNX Export

Export models to ONNX format for use with [ONNX Runtime](https://onnxruntime.ai/):

```bash
pip install 'soup-cli[onnx]'
soup export --model ./output --format onnx
soup export --model ./output --format onnx --output ./model_onnx
```

### TensorRT-LLM Export

Export models to TensorRT-LLM format for high-throughput GPU inference:

```bash
pip install 'soup-cli[tensorrt]'
soup export --model ./output --format tensorrt
soup export --model ./output --format tensorrt --output ./model_trt
```

After export, use with Ollama manually or auto-deploy:
```bash
# Manual (3-step)
echo 'FROM ./my-model.q4_k_m.gguf' > Modelfile
ollama create my-model -f Modelfile
ollama run my-model

# Auto-deploy (1-step)
soup export --model ./output --format gguf --deploy ollama --deploy-name my-model
```

### Deploy to Ollama

Deploy a GGUF model directly to your local [Ollama](https://ollama.com/) instance:

```bash
# Deploy a GGUF model
soup deploy ollama --model ./output/model.q4_k_m.gguf --name soup-my-model

# Deploy with system prompt and parameters
soup deploy ollama --model ./model.gguf --name soup-chat \
  --system "You are a helpful assistant." \
  --template chatml \
  --parameter temperature=0.7 \
  --parameter top_p=0.9

# Export + deploy in one command
soup export --model ./output --format gguf --deploy ollama

# List Soup-deployed models
soup deploy ollama --list

# Remove a model
soup deploy ollama --remove soup-my-model
```

Auto-detected chat templates: `chatml`, `llama`, `mistral`, `vicuna`, `zephyr` (or `auto` to infer from soup.yaml).

## Resume Training

Resume a training run from a checkpoint:

```bash
# Auto-detect latest checkpoint in output directory
soup train --config soup.yaml --resume auto

# Resume from a specific checkpoint
soup train --config soup.yaml --resume ./output/checkpoint-500
```

## Eval-Gated Training

Halt training automatically if a declarative eval suite regresses beyond a threshold vs a baseline. The gate runs at epoch boundaries — no wasted compute on runs that are already worse.

**Configure in `soup.yaml`:**

```yaml
training:
  epochs: 5
  eval_gate:
    enabled: true
    suite: ./evals/gate.yaml            # Declarative task list
    every_n_epochs: 1                    # Run gate every N epochs (1-100)
    regression_threshold: 0.05           # Allow 5% drop before halting (0.0-1.0)
    baseline: registry://llama31-chat-v1 # Or a file path, or omit for first run
    on_regression: stop                  # stop | warn | continue
```

**Or pass on the command line:**

```bash
soup train --config soup.yaml --gate ./evals/gate.yaml
```

**Run a gate suite post-hoc (no training):**

```bash
soup eval gate --suite ./evals/gate.yaml --model ./output \
  --baseline registry://llama31-chat-v1
```

**`evals/gate.yaml` example:**

```yaml
tasks:
  - name: math_sanity
    prompts: ./evals/math.jsonl          # prompt + expected
    scoring: exact
  - name: style_judge
    prompts: ./evals/style.jsonl
    scoring: judge
    judge_model: ollama://llama3.1        # SSRF-allowlisted scheme
```

Baselines may be a registry reference (`registry://<name-or-id>`), a file path, or omitted for the first run. Any structured exception (`ValueError`, `FileNotFoundError`, `OSError`) during the gate is treated as a regression under `on_regression: stop`.

## Run Management & Cleanup

LLM training generates massive checkpoint files. Soup automatically manages an SQLite database of your training loss and metrics, empowering you to safely reclaim disk space once training is complete.

```bash
# List all historical training runs
soup runs list

# Compare two differing experiments side-by-side
soup runs compare run_202611... run_202612...

# Intelligently clean up redundant checkpoints
# (Preserves the final model and the checkpoint with the lowest loss)
soup runs clean run_202611...

# Preview space that would be reclaimed across ALL experiments
soup runs clean --all --dry-run
```

By default, the `clean` command operates in "surgical mode" (`--keep-weights`), deleting huge optimizer state files (`optimizer.pt`) from lesser checkpoints to save gigabytes, but keeping their lightweight evaluation weights just in case you want to load them later.

## Alternative Model Hubs

Set `training.hub` in your `soup.yaml` to download from / push to a non-HuggingFace hub. Useful in regions where HF Hub is unreachable or blocked.

```yaml
training:
  hub: modelscope   # or 'modelers' (Openmind), default 'hf'
```

Override the endpoint via env var:

```bash
export MODELSCOPE_ENDPOINT=https://my-mirror.example.com
export MODELERS_ENDPOINT=https://corp-modelers.internal   # HTTPS only for non-loopback
soup train --config soup.yaml
```

The endpoint validator follows the same SSRF rules as `HF_ENDPOINT`: only `http`/`https` schemes; plain HTTP allowed only for `localhost` / `127.0.0.1` / `::1`; private and link-local IPs (RFC1918, 169.254/16, etc.) rejected on plain HTTP. `backend: mlx` is incompatible with non-HF hubs (`mlx-lm` only downloads from HF Hub).

The hub adapter is schema-only in this release; the live downloader and uploader land in v0.51.1.

## Model Registry & Lineage

Every fine-tune you ship should be reproducible. Soup's local registry (`~/.soup/registry.db`) tracks each entry by a content hash of its config + data + base model, plus lineage pointers to parent entries.

```bash
# Register a completed run
soup registry push --run-id run_202611_abc123 --name llama31-chat --tag v1

# List entries (filter by name, tag, base model, task)
soup registry list
soup registry list --name llama31-chat --tag prod

# Show full details: config, eval baseline, artifacts, ancestors
soup registry show llama31-chat-v1

# Side-by-side config diff + eval delta between two entries
soup registry diff llama31-chat-v1 llama31-chat-v2

# Full-text search across name / base model / task / notes
soup registry search "medical reasoning"

# Promote an entry (add a tag, e.g. "prod")
soup registry promote llama31-chat-v1 --tag prod

# Delete (cascades to artifacts + lineage links)
soup registry delete llama31-chat-v1 --yes
```

**Lineage DAG** — every entry can point to a parent (its ancestor run). Walk the DAG for any name with:

```bash
soup history llama31-chat
```

**Refs resolve flexibly** — you can use a registry ID, a name (latest), or `name:tag`. Ambiguous prefixes raise an error rather than silently picking the wrong entry. Registry files are stored with `600` perms on POSIX; override the path with `SOUP_REGISTRY_DB_PATH`.

## Diagnose (Post-Training Report Card)

`soup diagnose` scores six independent failure modes for a trained adapter and renders an OK / MINOR / MAJOR verdict per mode plus an overall headline — same taxonomy as Quant-Lobotomy. Useful for catching adapter regressions that a loss curve cannot distinguish from a healthy run.

```bash
# Heuristic neutral report (no model load — runs as a sanity check)
soup diagnose my-run-id

# Compute scores from a pre-built evidence JSON
soup diagnose my-run-id --evidence evidence.json --output diag.json

# Twitter-shareable SVG badge embeddable in a model card
soup diagnose my-run-id --badge diag.svg

# Attach the report to a Model Registry entry as a first-class artifact
soup diagnose my-run-id --output diag.json --attach-to-registry abc123
```

**Six failure-mode probes:**

| Mode | What it catches | Score range |
|------|-----------------|-------------|
| `forgetting` | Catastrophic forgetting on MMLU / HellaSwag / domain hold-outs | Δ accuracy vs base, tolerance band |
| `refusal` | Refusal-rate regression on harmful / benign probe sets | abs(Δ harmful) + abs(Δ benign) |
| `format` | JSON / regex / tool-call validity drift | fraction of valid outputs |
| `mode_collapse` | Diversity collapse at T=0 and T=1 | pairwise n-gram Jaccard distance |
| `memorization` | Verbatim training-prefix echo on partial prompts | 1 − echo_rate |
| `contamination` | Training data overlapping public benchmarks | 1 − contamination_rate |

**Verdict pill colours:** OK (≥ 0.85) green / MINOR (≥ 0.60) amber / MAJOR (< 0.60) red. `soup diagnose` exits 2 when the overall verdict is MAJOR — wire into CI to fail the build on regression.

**Post-training gate:** `soup train --diagnose-gate <evidence.json>` runs the same scorer after training finishes and refuses to mark the run successful when any mode comes back MAJOR. Composes with `--gate <eval-suite>` (v0.26) — the eval gate catches accuracy regressions vs a baseline; the diagnose gate catches behaviour regressions the eval suite is blind to.

## Adapter Management (git for LoRA)

`soup adapters` is the git-for-LoRA surface: weight-aware diff, four merge strategies, leave-one-out blame, and SHA-256 branch snapshots. All commands operate on `adapter_model.safetensors` directories (peft-compatible).

```bash
# Per-layer ΔW Frobenius diff + effective-rank drift + top-K changed projections
soup adapters diff ./run-v17 ./run-v18

# Machine-readable JSON for CI
soup adapters diff ./run-v17 ./run-v18 --format json --output diff.json

# Weighted merge with linear / ties / dare / svd strategies
soup adapters merge ./run-v17 ./run-v18 ./run-v19 -o ./merged --strategy ties \
  --weights 0.5,0.3,0.2 --density 0.2

# DARE merge (deterministic via --seed)
soup adapters merge ./run-v17 ./run-v18 -o ./merged --strategy dare \
  --density 0.5 --seed 42

# Leave-one-out ablation plan against a 4-hour wall-clock budget
soup adapters blame ./run-v18 --dataset train.jsonl --layer q_proj.7 \
  --budget 4h --shards 10 --plan-only

# Snapshot a training environment as a comparable branch
soup adapters branch v18 --config soup.yaml --base meta-llama/Llama-3.1-8B \
  --dataset train.jsonl

# Restore the snapshot's config (refuses if source SHA drifted)
soup adapters checkout v18 --output soup.yaml

# List all snapshotted branches
soup adapters branches
```

**Four merge strategies (pure numpy, no torch import at module level):**

| Strategy | Math | Use case |
|----------|------|----------|
| `linear` | Weighted average per layer | Baseline; tasks share a basis |
| `ties` | Trim by density → elect majority sign → disjoint average | Conflicting task adapters (Yadav et al. 2023) |
| `dare` | Random drop with `density` + rescale `1/density`, then average | Sparse-merge; reduces parameter interference (Yu et al. 2024) |
| `svd` | Linear-merge → low-rank reconstruction via SVD (`--rank`) | Constrain effective rank of the merged delta |

**Defaults & safety:**

- Output paths are containment-checked under cwd and reject pre-placed symlinks (TOCTOU defence).
- Safetensors writes are atomic via `tempfile.mkstemp` + `os.replace` — a crash mid-write never leaves a partial adapter at the target path.
- `.bin` (PyTorch pickle) adapter format is rejected with an explicit "re-save as safetensors" message.
- Branch pointers live under `~/.soup/branches/` (override via `SOUP_BRANCHES_DIR`, constrained to `$HOME` / `$CWD` / `$TMPDIR`).
- `soup adapters checkout` SHA-checks the source config — refuses to restore when the source has drifted from the snapshot, so reproducibility never silently lies.

**Limitations (v0.57.1):** Live blame ablation runner + live canary verdict on merged adapters are scheduled for v0.57.1; `soup adapters blame` emits the plan and exits clean today, and `MergeReport.verdict` is the `UNKNOWN` stub.

## Soup Cans (Shareable Recipes)

Share a reproducible recipe as a single `.can` file — a tarball of the manifest, full config, and a reference to the training data (URL or HF dataset). Not the weights, not the dataset bytes: just enough for someone else to re-run the same training.

```bash
# Pack a registry entry into a .can
soup can pack --entry-id llama31-chat-v1 --out ./llama31-chat.can

# Preview the manifest without extracting
soup can inspect ./llama31-chat.can

# Verify schema + config parseability
soup can verify ./llama31-chat.can

# Fork with modifications (dotted-path overrides) and re-pack
soup can fork ./llama31-chat.can --out ./llama31-chat-hot.can \
  --modify training.lr=5e-5 --modify training.epochs=5

# Run a .can end-to-end: extract → train (→ optional deploy)
soup can run ./llama31-chat.can --yes
soup can run ./llama31-chat.can --yes --deploy --env-capture ./env.txt

# Publish a .can to HF Hub as a dataset
soup can publish ./llama31-chat.can --hf-hub me/llama31-chat-recipe
```

**Security** — tar extraction uses `filter="data"` on Python 3.12+ with symlink/hardlink rejection fallback for older runtimes. Size cap: 100 MB. `DataRef.url` must be HTTPS. Fork overrides reject dunder keys (`__class__`, `__init__`) and null bytes. Manifest format version supports `1` and `2` (additive bump in v0.33.0 added `deploy_targets`). `soup can run` requires `--yes` (mandatory consent — auto-downloads data + auto-trains). `soup can publish` validates `repo_id` and resolves the HF token via env / cache files; commit messages are first-line + 200-char capped.


## Batch Inference

Run a model on a list of prompts and save results:

```bash
# JSONL input (each line: {"prompt": "..."})
soup infer --model ./output --input prompts.jsonl --output results.jsonl

# Plain text input (one prompt per line)
soup infer --model ./output --input prompts.txt --output results.jsonl

# Custom generation settings
soup infer --model ./output --input prompts.jsonl --output results.jsonl \
  --max-tokens 512 --temperature 0.3
```

Output is JSONL with `prompt`, `response`, and `tokens_generated` fields. Shows a progress bar and throughput summary.

## Inference Benchmarking

Quickly measure your model's generation speed and memory footprint before deployment:

```bash
# Benchmark local speed and VRAM usage on 3 automatically generated prompts
soup bench ./output

# Customizing benchmarking parameters
soup bench ./output --num-prompts 5 --max-tokens 256

# Use custom prompts from a text file (one per line) or JSONL
soup bench ./output --prompts-file my_prompts.txt
soup bench ./output --prompts-file bench_suite.jsonl
```

This acts as a built-in "speedometer," outputting Tokens-Per-Second (TPS), Total Latency, and Peak VRAM allocations into a clean status table.

## TensorBoard Integration

Log training metrics to TensorBoard for local visualization:

```bash
# Enable TensorBoard logging (requires: pip install tensorboard)
soup train --config soup.yaml --tensorboard

# View logs
tensorboard --logdir ./output/runs/
```

> **Note:** `--tensorboard` and `--wandb` cannot be used together. Pick one.

## Weights & Biases Integration

Send training metrics to [W&B](https://wandb.ai/) for cloud-based experiment tracking:

```bash
# Enable W&B logging (requires: pip install wandb)
soup train --config soup.yaml --wandb
```

Make sure `WANDB_API_KEY` is set or run `wandb login` first.

## Inference Server

Start a local OpenAI-compatible inference server:

```bash
# Install server dependencies
pip install 'soup-cli[serve]'

# Start server
soup serve --model ./output --port 8000

# With custom settings
soup serve --model ./output --port 8080 --host 127.0.0.1 --max-tokens 1024
```

Endpoints:
- `POST /v1/chat/completions` — chat completions (streaming supported)
- `GET /v1/models` — list available models
- `GET /health` — health check

Compatible with OpenAI SDK:
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="output",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### vLLM Backend (2-4x Faster Inference)

Use [vLLM](https://github.com/vllm-project/vllm) for significantly better throughput in production:

```bash
# Install vLLM support
pip install 'soup-cli[serve-fast]'

# Start with vLLM backend
soup serve --model ./output --backend vllm

# Multi-GPU with tensor parallelism
soup serve --model ./output --backend vllm --tensor-parallel 2

# Control GPU memory usage
soup serve --model ./output --backend vllm --gpu-memory 0.8
```

> **Tip:** Soup auto-detects vLLM. When installed, you'll see a hint during `soup serve` if you haven't enabled it yet.

### SGLang Backend

Use [SGLang](https://github.com/sgl-project/sglang) as an alternative high-throughput backend:

```bash
# Install SGLang support
pip install 'soup-cli[sglang]'

# Start with SGLang backend
soup serve --model ./output --backend sglang

# Multi-GPU with tensor parallelism
soup serve --model ./output --backend sglang --tensor-parallel 2
```

### Speculative Decoding

Use a smaller draft model to speed up generation (2-3x faster):

```bash
# Transformers backend — uses HF assisted generation
soup serve --model ./output --speculative-decoding small-draft-model --spec-tokens 5

# vLLM backend — uses vLLM native speculative decoding
soup serve --model ./output --backend vllm --speculative-decoding small-draft-model

# Auto-pair: Soup picks the draft for you based on the target family
soup serve --model meta-llama/Llama-3.1-70B-Instruct --backend vllm --auto-spec
# → auto-paired: meta-llama/Llama-3.2-1B-Instruct (target: Llama-3.1-70B-Instruct)
```

`--auto-spec` handles Llama 3.1/3.3/4, Qwen 2.5/3, Mistral Large, Mixtral, DeepSeek V3/R1, and Gemma 2/3. Models without a known draft pairing (e.g. 8B-or-smaller targets where draft+target overhead outweighs the gain) print a yellow "no draft" note and fall back to standard decoding.

### Prefix Caching

For RAG and agent workloads with a shared system prompt, enable vLLM's automatic prefix cache:

```bash
soup serve --model ./output --backend vllm --prefix-cache
```

The first request with a given prefix warms the cache; subsequent requests skip the shared prefix compute entirely. Big latency win when 100+ requests share the same system prompt.

### Dynamic LoRA Hot-Swap

Switch the active adapter at runtime without restarting the server:

```bash
soup serve --model base-model --adapters chat=./chat-adapter code=./code-adapter
```

```bash
# Activate an adapter
curl -X POST http://localhost:8000/v1/adapters/activate/chat
# → {"active": "chat", "status": "ok"}

# Return to base model
curl -X POST http://localhost:8000/v1/adapters/deactivate
# → {"active": null, "status": "ok"}

# List loaded adapters with active flag
curl http://localhost:8000/v1/adapters
# → {"adapters": [{"name": "chat", "active": true}, ...], "active": "chat"}
```

Names are validated against `^[a-zA-Z0-9][a-zA-Z0-9-]*$`; activate/deactivate calls are thread-safe behind a lock.

### Structured Output (JSON Schema / Regex)

Constrain model output to a valid JSON schema or regex pattern:

```bash
# JSON schema (schema file must live under your cwd)
soup serve --model ./output --structured-output json --json-schema product.json

# Regex (length-capped at 2048 chars, null bytes rejected)
soup serve --model ./output --structured-output regex --regex-pattern '\d{3}-\d{4}'
```

The `validate_json_schema` helper caps serialised size at 64KB and requires a top-level `type` field so malformed schemas fail fast at server startup, not per-request.

### Continuous-Batching Dashboard + `/metrics`

Track live server health:

```bash
soup serve --model ./output --dashboard
```

```bash
curl http://localhost:8000/metrics
# → {
#   "requests_total": 1234,
#   "tokens_generated_total": 456789,
#   "active_requests": 3,
#   "latency_p50_ms": 185.2,
#   "latency_p95_ms": 720.0,
#   "latency_samples": 1000
# }
```

Latency percentiles are computed from the last 1000 requests; counters include failure paths so the dashboard shows true reliability, not just success rate.

### OpenTelemetry Request Tracing

Emit per-request spans to your OTLP collector:

```bash
pip install opentelemetry-sdk opentelemetry-exporter-otlp

soup serve --model ./output \
  --trace \
  --trace-endpoint http://localhost:4317
```

The OTLP endpoint is SSRF-hardened: only http/https schemes, plain HTTP only for loopback (`localhost`/`127.0.0.1`/`::1`), and RFC1918 / link-local / `0.0.0.0` all rejected via `ipaddress.ip_address`. When the SDK is missing the flag is a no-op with a warning — the server starts fine without spans.

> **Note:** `max_tokens` is capped at 16,384 per request. Error details are never exposed in HTTP responses.

## Synthetic Data Generation

Generate training data using LLMs:

```bash
# Generate using OpenAI API
soup data generate --prompt "Create math word problems" --count 100 --format alpaca

# Use a different model
soup data generate --prompt "Medical Q&A pairs" --model gpt-4o --count 500

# Deduplicate against existing data
soup data generate --prompt "..." --count 200 --dedup-with existing.jsonl

# Use seed examples to guide style
soup data generate --prompt "..." --seed examples.jsonl --count 100

# Use a local OpenAI-compatible server (soup serve, Ollama, etc.)
soup data generate --prompt "..." --provider server --api-base http://localhost:11434/v1
```

### Multi-Provider Support

```bash
# Generate via local Ollama instance
soup data generate --prompt "..." --provider ollama --model llama3.1
soup data generate --prompt "..." --ollama-model llama3.1  # shorthand

# Generate via Anthropic Claude API (set ANTHROPIC_API_KEY env var)
soup data generate --prompt "..." --provider anthropic --model claude-3-haiku-20240307

# Generate via local vLLM server
soup data generate --prompt "..." --provider vllm --model meta-llama/Llama-3.1-8B-Instruct
```

### Domain Templates

```bash
# Code instruction pairs (Python, JS, Go, Rust, Java)
soup data generate --prompt "..." --template code --language Python --task-type function

# Multi-turn conversations
soup data generate --prompt "..." --template conversation --turns 6 --topic "science"

# QA from context document
soup data generate --prompt "..." --template qa --context document.txt

# Preference data (DPO/KTO/ORPO)
soup data generate --prompt "..." --template preference --pref-task dpo

# Chain-of-thought reasoning (GRPO)
soup data generate --prompt "..." --template reasoning --domain math
```

### Quality Pipeline

```bash
# Auto-validate after generation (remove malformed entries)
soup data generate --prompt "..." --validate

# Auto-filter by quality (coherence scoring)
soup data generate --prompt "..." --filter

# Auto-dedup (MinHash, requires: pip install 'soup-cli[data]')
soup data generate --prompt "..." --dedup

# Full quality pipeline: validate + filter + dedup
soup data generate --prompt "..." --quality-pipeline
```

## Data Augmentation

Augment an existing dataset using an LLM — rephrase for diversity, translate for multilingual coverage, or apply a style transform.

```bash
# Rephrase each example N times for more diversity
soup data augment ./data/train.jsonl --strategy rephrase --count 3 \
  --output ./data/train_augmented.jsonl

# Translate into multiple languages
soup data augment ./data/train.jsonl --strategy translate --lang es,fr,de \
  --output ./data/train_multilingual.jsonl

# Style transfer (formal / casual / technical / etc.)
soup data augment ./data/train.jsonl --strategy style --styles formal,casual \
  --output ./data/train_styled.jsonl
```

Works with any provider supported by `soup data generate` (OpenAI, Ollama, Anthropic, vLLM, local server). `--count` is capped at 10; `--lang` and `--styles` each capped at 10 entries × 32 chars.

## Trace-to-Preference

Harvest DPO / KTO-ready preference pairs from your production inference logs — no manual labeling.

```bash
# LangChain logs + thumbs-up signal
soup data from-traces --logs ./logs/langchain.jsonl \
  --format langchain --signal thumbs_up --output prefs.jsonl

# OpenAI API logs + regeneration signal (second response wins)
soup data from-traces --logs ./logs/openai.jsonl \
  --format openai --signal regeneration --output prefs.jsonl

# Soup-serve logs + user-edit signal (edited response wins over original)
soup data from-traces --logs ./logs/soup-serve.jsonl \
  --format soup_serve --signal user_edit --output prefs.jsonl

# Preview generated pairs before training
soup data review prefs.jsonl --sample 10
```

**Supported log formats:** `langchain`, `openai`, `soup_serve`
**Supported signals:** `thumbs_up` (rating-based), `regeneration` (latest wins), `user_edit` (edited wins)

Trace files are capped at 100,000 lines to prevent OOM on production logs. A PII warning panel appears on every run — redact sensitive fields before harvesting.

## Config Migration

Switch from other tools with one command:

```bash
# Import from LLaMA-Factory
soup migrate --from llamafactory llama3_lora_sft.yaml

# Import from Axolotl
soup migrate --from axolotl axolotl_config.yml

# Import from Unsloth notebook
soup migrate --from unsloth finetune.ipynb

# Preview without writing
soup migrate --from llamafactory config.yaml --dry-run
```

Automatically maps model, LoRA, training params, quantization, and task type. Warns about unsupported features.

## Ready-Made Recipes

80 pre-built configs for popular models — no guessing hyperparameters:

```bash
# List all recipes
soup recipes list

# Preview a recipe
soup recipes show llama3.1-8b-sft

# Use a recipe (writes soup.yaml)
soup recipes use llama3.1-8b-sft

# Search by task or keyword
soup recipes search --task grpo
soup recipes search "reasoning"
soup recipes search --size 7b
soup recipes search "medical"
soup recipes search "vision"
```

**What's covered:**

| Category | Models |
|---|---|
| **General SFT / DPO / GRPO / KTO / ORPO / SimPO / IPO / PPO / Embedding / Pretrain** | Llama 3.1 / 3.2 / 4, Qwen 2.5 / 3, Mistral, Gemma 3, Phi-4, DeepSeek R1 / V3 |
| **Vision (multimodal)** | Llama-3.2-Vision (11B + 90B), Pixtral-12B, Qwen2-VL (7B + 72B), InternVL 2.5, MiniCPM-V 2.6 |
| **Audio (speech)** | Qwen2-Audio, SeamlessM4T v2 (translation), Whisper-large-v3 (ASR) |
| **Reasoning** | All 6 DeepSeek-R1-Distill sizes (Qwen 1.5B / 7B / 14B / 32B + Llama 8B / 70B), Qwen3-Coder 30B, Qwen3-30B-A3B reasoning, Phi-4 reasoning |
| **Small / edge / mobile** | SmolLM2 (135M / 360M / 1.7B), Qwen2.5 (0.5B / 1.5B / 3B), Gemma 2 2B, Phi-3.5-mini, Llama-3.2 (1B / 3B) |
| **Domain specialists** | BioMistral 7B, Meditron 7B (medical) — CodeLlama (13B / 70B), Magicoder 6.7B (code) — Mathstral 7B (math) — Llama-2-13b-finance (FinGPT-style starter) — Nemotron-4 340B |
| **Multimodal reasoning** | Llama-3.2-Vision GRPO, Pixtral DPO |
| **Multi-GPU** | llama3-70b-fsdp2, qwen3-32b-zeropp, deepseek-v3-pipeline |
| **Apple Silicon (MLX)** | llama3.1-8b / qwen3-8b / gemma3-9b SFT-MLX |
| **Tool-calling / agentic** | qwen3-8b-tools, llama4-scout-tools |

## Hyperparameter Sweep

Search for the best hyperparameters:

```bash
# Grid search over learning rate and LoRA rank
soup sweep --config soup.yaml --param lr=1e-5,2e-5,5e-5 --param lora_r=8,16,32

# Random search with max runs
soup sweep --config soup.yaml --param lr=1e-5,2e-5,5e-5 --strategy random --max-runs 5

# Preview without running
soup sweep --config soup.yaml --param lr=1e-5,2e-5 --param epochs=2,3 --dry-run

# Early stopping: skip remaining runs if loss exceeds 1.5x best
soup sweep --config soup.yaml --param lr=1e-5,2e-5,5e-5 --early-stop 1.5
```

## Model Comparison

Compare outputs of two models side-by-side:

```bash
# Compare with inline prompts
soup diff --model-a ./model_v1 --model-b ./model_v2 --prompt "Explain gravity"

# Compare with a prompts file
soup diff --model-a ./base --model-b ./finetuned --prompts test_prompts.jsonl

# Save results
soup diff --model-a ./a --model-b ./b --prompts prompts.txt --output results.jsonl
```

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

## Quickstart Demo

Run a complete demo in one command — creates sample data, config, and trains a tiny model:

```bash
# Full demo (creates data + config + trains TinyLlama)
soup quickstart

# Just create files without training
soup quickstart --dry-run

# Skip confirmation
soup quickstart --yes
```

## Health Check

Check your environment for compatibility issues:

```bash
soup doctor [--nccl]
```

Shows: Python version, GPU availability, system resources (RAM/Disk), all dependency versions, and fix suggestions. Use `--nccl` to measure and check multi-GPU communication bandwidth against expected hardware ceilings.

## Version Info

```bash
# Basic version
soup version

# Machine-readable output
soup version --json
# -> {"version": "0.26.0", "python": "3.11.5", "platform": "linux"}

# Full system info (useful for bug reports)
soup version --full
# -> soup v0.26.0 | Python 3.11.5 | CUDA 12.1 | extras: serve, data

# Full system info in JSON
soup version --full --json
# -> {"version": "0.26.0", "python": "3.11.5", "platform": "linux", "torch": "2.2.0", ...}
```

## Web UI

Launch a local web interface to manage experiments, start training, explore data, and chat with models — all from your browser.

```bash
pip install 'soup-cli[ui]'
soup ui
# -> opens http://127.0.0.1:7860 in your browser
# -> prints auth token to console
```

**Pages:**
- **Dashboard** — view all experiment runs, loss charts, system info, multi-run comparison
- **New Training** — create configs from templates or 43 ready-made recipes, validate, start training with live SSE log streaming and progress bar
- **Data Explorer** — browse and inspect datasets (JSONL, JSON, CSV, Parquet)
- **Model Chat** — chat with streaming responses, configurable temperature/top_p/max_tokens, system prompt, adapter selection, markdown rendering, chat export

**Live monitoring + enhanced UX:**
- **Training Live Monitor** — real-time SSE log streaming, live metrics, progress bar with ETA
- **Enhanced Metrics** — 2x2 chart grid (loss, LR, grad_norm, throughput) + GPU memory chart, eval results table
- **Multi-Run Compare** — overlay loss curves from up to 5 runs side-by-side
- **Chat Upgrade** — SSE streaming via proxy, typing indicator, cancel button, markdown renderer (bold, italic, code blocks), chat export as JSON
- **Config Builder** — recipe dropdown (43 recipes), config schema API for dynamic form generation

**Security:** The Web UI generates a random auth token at startup (printed to console). All mutating endpoints (start/stop training, delete runs, inspect data, validate config) require `Authorization: Bearer <token>` header. CORS is restricted to the served origin. Data inspection is sandboxed to the working directory.

```bash
# Custom port, don't auto-open browser
soup ui --port 8080 --no-browser
```

## Error Handling

Soup shows friendly error messages by default (2-3 lines with a fix suggestion). For full tracebacks:

```bash
# Global flag goes BEFORE the command
soup --verbose train --config soup.yaml

# Works with any command
soup --verbose eval --model ./output --benchmarks mmlu
```

> **Note:** `--verbose` is a global flag — it must go **before** the command name, not after.

## Data Formats

Soup supports these formats (auto-detected). Files can be JSONL, JSON, CSV, Parquet, or TXT.

**Alpaca:**
```json
{"instruction": "Explain gravity", "input": "", "output": "Gravity is..."}
```

**ShareGPT:**
```json
{"conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello!"}]}
```

**ChatML:**
```json
{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
```

**DPO / ORPO / SimPO / IPO (preference pairs):**
```json
{"prompt": "Explain gravity", "chosen": "Gravity is a force...", "rejected": "I don't know"}
```

**KTO (unpaired preferences):**
```json
{"prompt": "Explain gravity", "completion": "Gravity is a force...", "label": true}
```

**LLaVA (vision):**
```json
{"image": "photo.jpg", "conversations": [{"from": "human", "value": "<image>\nDescribe this."}, {"from": "gpt", "value": "A cat."}]}
```

**ShareGPT4V (vision):**
```json
{"image": "chart.png", "conversations": [{"from": "human", "value": "<image>\nExplain this chart."}, {"from": "gpt", "value": "Revenue growth."}]}
```

**Plaintext (pre-training):**
```json
{"text": "Raw text document for continued pre-training..."}
```
Or use `.txt` files directly (one document per line).

**Embedding (sentence embedding pairs/triplets):**
```json
{"anchor": "What is Python?", "positive": "Python is a programming language."}
{"anchor": "What is Python?", "positive": "A programming language.", "negative": "A type of snake."}
```

**Audio (speech + conversation):**
```json
{"audio": "recording.wav", "messages": [{"role": "user", "content": "Transcribe."}, {"role": "assistant", "content": "Hello world."}]}
```

**PRM (process reward, stepwise-supervised):**
```json
{"prompt": "Solve 2+2", "completions": ["First, add", "Result is 4"], "labels": [true, true]}
```

**Pre-tokenized (skip tokenize stage):**
```json
{"input_ids": [1, 2, 3, ...], "labels": [-100, 2, 3, ...], "attention_mask": [1, 1, 1, ...]}
```
Use with `data.format: pre_tokenized` and `data.tokenized_path: ./.soup-tokenized/<key>` after running `soup data preprocess`.

**Input/Output (template-free, segment-level loss control):**
```json
{"segments": [{"text": "Q: hi", "label": false}, {"text": "A: hello", "label": true}]}
```

**Video:**
```json
{"video": "clip.mp4", "messages": [{"role": "user", "content": "Describe this clip."}]}
```

**Multimodal (typed content parts — text / image / audio / video in one message):**
```json
{"messages": [{"role": "user", "content": [{"type": "text", "text": "What's in this?"}, {"type": "image", "url": "x.png"}]}]}
```

## Data Pipeline Pro

Soup speaks the same dataset surface as Axolotl + LlamaFactory + Unsloth — remote URIs, streaming, sharding, multi-dataset interleaving, vocab expansion, and document ingestion all live in one schema.

**Remote datasets** (schema gate live; fsspec backend wiring lands in v0.42.1):

```yaml
data:
  train: s3://my-bucket/datasets/train.jsonl   # also gs:// gcs:// az:// abfs:// abfss:// oci://
  streaming: true
  buffer_size: 8192
  shards: 4
```

**Multi-dataset interleave:**

```yaml
data:
  interleave: { strategy: probs, probs: [0.7, 0.3] }   # also: concat / under / over
  eval_on_each_dataset: true
```

**Vocab expansion + advanced masking:**

```yaml
data:
  add_new_tokens: ["<reasoning>", "</reasoning>"]
  new_special_tokens: ["<|tool_call|>"]
  resize_vocab: true
  mask_history: true
  split_thinking: true            # Qwen3-style <think> reasoning-block masking
  image_min_pixels: 256
  image_max_pixels: 4096
  image_resize_algorithm: bicubic
  video_fps: 24
  video_maxlen: 32
  video_dir: ./videos
```

**AOT preprocessing:**

```bash
# Tokenize once, reuse the cache across runs.
soup data preprocess soup.yaml --output ./.soup-tokenized

# Then in soup.yaml:
#   data:
#     format: pre_tokenized
#     tokenized_path: ./.soup-tokenized/<16-char-cache-key>
```

**Document ingestion (PDF / DOCX / MD / TXT → JSONL):**

```bash
soup data ingest report.pdf --output report.jsonl
soup data ingest README.md
soup data ingest notes.docx
```

**Custom prompt strategies (schema only — runtime invocation in v0.42.1):**

```yaml
data:
  prompt_strategy: my_pkg.transforms:rephrase
```

## Data Tools

```bash
# Inspect a dataset
soup data inspect ./data/train.jsonl

# Validate format (auto-detects if --format not specified)
soup data validate ./data/train.jsonl
soup data validate ./data/train.jsonl --format alpaca

# Convert between formats
soup data convert ./data/train.jsonl --to sharegpt --output converted.jsonl

# Merge multiple datasets
soup data merge data1.jsonl data2.jsonl --output merged.jsonl --shuffle

# Remove near-duplicates (requires: pip install 'soup-cli[data]')
soup data dedup ./data/train.jsonl --threshold 0.8

# Extended statistics (length distribution, token counts, languages)
soup data stats ./data/train.jsonl

# Filter by quality (perplexity + coherence scoring)
soup data filter ./data/train.jsonl --coherence 0.3
soup data filter ./data/train.jsonl --perplexity 500 --coherence 0.3
soup data filter ./data/train.jsonl --score-only  # add scores without filtering
```

## Experiment Tracking

Every `soup train` run is automatically tracked in a local SQLite database (`~/.soup/experiments.db`).

```bash
# List all training runs
soup runs

# Show detailed info + loss curve for a run
soup runs show run_20260223_143052_a1b2

# Compare two runs side by side
soup runs compare run_1 run_2

# Delete a run
soup runs delete run_1

# Replay an old run's summary + loss curve from history
soup runs replay run_1
```

Every completed run also stores an estimated cost (`$` per run) computed from the
captured GPU device name and duration. `soup runs show` renders `—` for CPU /
MPS / unknown GPUs (no fabricated zeros).

### Tracker integrations (--tracker mlflow / swanlab / trackio)

```bash
# Stream metrics to MLflow (set MLFLOW_TRACKING_URI to your server URL)
soup train --config soup.yaml --tracker mlflow

# Or SwanLab (cloud or local)
soup train --config soup.yaml --tracker swanlab

# Or Trackio (offline-friendly batched upload)
soup train --config soup.yaml --tracker trackio
```

`--tracker` is mutually exclusive with `--wandb` and `--tensorboard`. Soup
validates the tracker name against a closed allowlist (`mlflow` / `swanlab` /
`trackio` / `wandb` / `tensorboard` / `none`); the upstream package itself is
loaded by HF Trainer at run time, so install the one you need separately:

```bash
pip install mlflow      # or: swanlab / trackio
```

### Telemetry (opt-in)

Soup ships a hardware-info-only telemetry payload (Soup version + command +
Python major.minor + OS + arch + duration). It is **off by default** and never
sends model names, dataset paths, or config contents. Enable explicitly:

```bash
SOUP_TELEMETRY=1 soup train --config soup.yaml
```

The PostHog network upload itself is deferred to v0.43.1; v0.43.0 ships the
payload schema only so you can audit it before opting in.

## NLG Evaluation Metrics (BLEU + ROUGE)

Pure-Python BLEU + ROUGE-1 / ROUGE-2 / ROUGE-L for `soup eval custom`:

```python
from soup_cli.utils.nlg_metrics import (
    bleu_score, rouge_l_score, compute_nlg_metric, NLG_METRICS,
    effective_tokens_per_second,
)

bleu_score(["the cat sat on the mat"], ["the cat sat on the mat"])
# 1.0
rouge_l_score(["the quick brown fox"], ["a quick brown dog"])
# 0.5
compute_nlg_metric("rouge_2", preds, refs)
# generic dispatch by canonical name

effective_tokens_per_second(unmasked_tokens=12_500_000, wall_clock_seconds=600.0)
# 20833.33  — None when wall_clock <= 0 (no fabrication)
```

Smoothed BLEU uses Chen & Cherry epsilon for zero-correct buckets where
`total[n] > 0`; empty buckets (e.g. predictions shorter than `max_n` tokens)
force the score to 0.0.

## Quant Calibration (KL Divergence)

Compare a quantized model to a full-precision baseline on a small fixed prompt
set. OK / MINOR / MAJOR thresholds at 0.05 / 0.20 mean KL — same scale as
`soup eval quant-check`.

```python
from soup_cli.eval.calibrate import run_calibration

# baseline_logits / quantized_logits: list[list[float]] aligned per-prompt
report = run_calibration(baseline_logits, quantized_logits)
print(report.delta_status, report.mean_kl)
# OK 0.012
```

The kernel is pure-math and capped at 10 000 prompts to defend against
accidental OOM. `CalibrationReport` is a frozen dataclass.

## Model Arena (Elo Tournament)

Local leaderboard with Elo ratings (K=32, base 1500). Bring your own pairwise
winners — Soup just keeps the books:

```python
from soup_cli.eval.arena import Tournament

t = Tournament()
t.record("llama-3.1-8b-finetune", "qwen2.5-7b-finetune", winner="a")
t.record("llama-3.1-8b-finetune", "mistral-7b-finetune", winner="draw")
for row in t.leaderboard():
    print(row)
```

Caps: 256 models per tournament, 1M matches. Model names with `[` or `]`
characters are rejected so leaderboard rows can't be markup-injected.

## Profiling Extras

CUDA memory snapshots, anomaly tracing, and an NCCL bandwidth reference table:

```python
from soup_cli.utils.profiling_v0_43 import (
    memory_snapshot_context, detect_anomaly_context, nccl_bandwidth_check,
)

with memory_snapshot_context("run-123") as path:
    train_step()
    # On CUDA, dumps profiles/run-123.snapshot.pickle on exit.

with detect_anomaly_context():
    train_step()
    # torch.autograd.set_detect_anomaly(True)

result = nccl_bandwidth_check(
    gpu="h100", link="nvlink", measured_gb_per_sec=400.0,
)
# {'expected_gb_per_sec': 450.0, 'measured_gb_per_sec': 400.0,
#  'ratio': 0.8889, 'status': 'OK'}
```

## VS Code Setup (`.vscode/launch.json`)

One-shot writer for a sane debugger config:

```python
from soup_cli.utils.vscode_setup import write_vscode_launch
write_vscode_launch(config_path="soup.yaml")
# Writes ./.vscode/launch.json with `soup train` + pytest entries.
```

Symlink-rejected at the target path regardless of `force=True` to defend
against pre-placed symlinks redirecting the write outside cwd.

## Demo Datasets (`soup data demo`)

Tiny JSONL fixtures bundled with Soup so you can warm up `soup train` without
hunting for data:

```bash
# List available bundles
soup data demo

# Copy one into the current directory
soup data demo alpaca_demo --output ./alpaca.jsonl
```

Bundles: `alpaca_demo`, `sharegpt_demo`, `dpo_demo`, `grpo_demo`. Output path
must stay under cwd; existing files are not overwritten.

## Observability & Dev UX

Tools that explain *why* a run misbehaved instead of dumping a stack trace.

### `soup why`

Heuristic explainer — reads the most recent (or named) run and surfaces
plain-English diagnoses with concrete next steps.

```bash
soup why                 # most recent run
soup why run_2026_abc    # specific run id (or prefix)
```

Detects: NaN/Inf loss, plateau (≥30 steps with <0.5% change), divergence
(loss > 3× initial), persistent high gradient norm, learning rate outside the
typical `[1e-6, 5e-3]` band. Pure rule-based — no model calls.

### `soup tui`

Full-screen Textual dashboard. Two-pane: run list (left) + selected-run detail
(right). `r` refreshes, `q` quits.

```bash
pip install 'soup-cli[tui]'
soup tui --refresh 1.0 --limit 50
```

### Auto-profiling — `soup train --profile`

Records a `torch.profiler` Chrome-trace over an early-steps window (default
`wait=1, warmup=1, active=5, repeat=1`). Output: `<output>/profiles/<run_id>.trace.json`.
Open in `chrome://tracing` or Perfetto.

### Crash bundles — `.crash` files

When training fails, Soup auto-writes a self-contained `.crash` JSON to
`./.soup-crashes/crash_<utc>_<hex>.crash` containing: redacted error trace,
classified failure kind (`oom` / `nan` / `cuda` / `dataloader` / `nccl` /
`other`), GPU state at crash time, env summary, last-50 metric rows, and the
config (recursively redacted of `hf_*` / `sk-*` / `Bearer …` tokens). The
output_dir is reduced to `os.path.basename` so `$HOME` doesn't leak.

### `--log-level quiet|normal|verbose|debug`

Global flag on the root `soup` command. Wires a Rich-formatted logger on the
`soup` namespace; `debug` enables timestamps + module paths.

```bash
soup --log-level verbose train --config soup.yaml
soup --log-level debug runs show <id>
```

## Model Evaluation

Full-featured evaluation platform with standard benchmarks, custom evals, LLM-as-a-judge, and human evaluation:

```bash
# Install eval dependencies
pip install 'soup-cli[eval]'

# Standard benchmarks (wraps lm-evaluation-harness)
soup eval benchmark --model ./output --benchmarks mmlu,gsm8k,hellaswag

# Custom eval tasks from JSONL
soup eval custom --tasks eval_tasks.jsonl --model ./output

# LLM-as-a-judge (score model outputs using GPT-4o, Ollama, etc.)
soup eval judge --target responses.jsonl --model gpt-4o-mini --provider openai
soup eval judge --target responses.jsonl --model llama3.1 --provider ollama

# Auto-eval after training (configure in soup.yaml)
soup eval auto --config soup.yaml

# Compare eval results between two training runs
soup eval compare run_20260301_143052_a1b2 run_20260315_091023_c3d4

# Local leaderboard across all evaluated models
soup eval leaderboard
soup eval leaderboard --format json
soup eval leaderboard --format csv

# Human A/B evaluation with Elo ratings
soup eval human --input prompts.jsonl --model-a ./model_a --model-b ./model_b
```

### Quant-Lobotomy Checker

Before you ship a quantized model, verify it didn't lose skills. The checker runs the same task list against the `--before` and `--after` models and renders a per-task OK / MINOR / MAJOR verdict.

```bash
# Compare a pre-quant model with its post-quant version
soup eval quant-check \
  --before ./output \
  --after  ./output/quantized.q4_k_m.gguf \
  --tasks  ./evals/sanity.jsonl

# Both sides may be registry refs
soup eval quant-check \
  --before registry://llama31-chat-v1 \
  --after  registry://llama31-chat-v1-q4 \
  --tasks  ./evals/sanity.jsonl

# Render as JSON for CI integration
soup eval quant-check --before X --after Y --tasks t.jsonl --format json
```

**Verdict thresholds (per task):**
- `OK` — score delta ≤ 2%
- `MINOR` — delta 2-10% (investigate)
- `MAJOR` — delta > 10% (do NOT ship)

Paths are containment-checked, and `registry://` refs are resolved with an optional `kinds` filter so you never pick the wrong artifact.

### Custom Eval Format

```jsonl
{"prompt": "What is 2+2?", "expected": "4", "category": "math", "scoring": "exact"}
{"prompt": "Explain gravity", "expected": "force.*attraction", "scoring": "regex"}
{"prompt": "Capital of France?", "expected": "Paris", "scoring": "contains"}
```

### Auto-Eval Config (soup.yaml)

```yaml
eval:
  auto_eval: true
  benchmarks: [mmlu, gsm8k]
  custom_tasks: eval_tasks.jsonl
  judge:
    model: gpt-4o-mini
    provider: openai
```

## All Commands

```
soup init [--template chat|code|...|audio]       Create config
soup autopilot --model <id> --data d.jsonl --goal <g>  Zero-configsoup train --config soup.yaml                 Start training
soup train --config soup.yaml --tensorboard   Train with TensorBoard logging
soup train --config soup.yaml --fsdp full_shard  Train with FSDP2
soup train --config soup.yaml --deepspeed zero++  DeepSpeed ZeRO++ (quantized comms)
soup train --config soup.yaml --gpus auto|N      Multi-GPU launch hint
soup train --config soup.yaml --gate evals/gate.yaml  Eval-gated training
soup train --config soup.yaml --push-as user/repo  Auto-push each checkpoint to HF as branch
soup train --config soup.yaml --push-as user/repo --hf-resume  Resume from latest HF checkpoint branch
soup train --config soup.yaml --find-lr        LR range finder: write recommended LR JSON
soup infer --model ./output --input p.jsonl   Batch inference
soup chat --model ./output                    Interactive chat
soup push --model ./output --repo user/name   Upload to HuggingFace
soup push --model ./output --repo user/name --collection user/coll-abc123  Add to HF Collection
soup merge --adapter ./output                 Merge LoRA with base model
soup export --model ./output --format gguf    Export to GGUF (Ollama)
soup export --model ./output --deploy ollama  Export GGUF + auto-deploy to Ollama
soup export --model ./output --format onnx    Export to ONNX
soup export --model ./output --format tensorrt Export to TensorRT-LLM
soup export --model ./output --format awq     Export to AWQ (4-bit)
soup export --model ./output --format gptq    Export to GPTQ (4-bit)
soup deploy ollama --model m.gguf --name x    Deploy GGUF to Ollama
soup deploy ollama --list                     List Soup-deployed models
soup deploy ollama --remove <name>            Remove model from Ollama
soup deploy hf-space --model user/m --space user/s --template gradio-chat|streamlit-chat  Create HF Space
soup deploy autopilot --target mac-m3|rtx-4090-24gb|...  Pick PEFT+quant+spec-decoding for a hardware target
soup deploy autopilot --list                  List all 10 deploy profiles
soup agent synth --spec api.yaml -o ds.jsonl  Parse OpenAPI/MCP/GraphQL spec into a tool-calling SFT dataset
soup agent train --spec api.yaml --base model  One-shot synth + planned soup train invocation
soup agent eval --spec api.yaml --predictions p.jsonl  Score predicted tool-calls vs spec catalog
soup eval benchmark --model ./output          Evaluate on standard benchmarks
soup eval custom --tasks eval.jsonl           Custom eval tasks from JSONL
soup eval judge --target resp.jsonl           LLM-as-a-judge evaluation
soup eval auto --config soup.yaml             Auto-eval from config
soup eval compare <run1> <run2>               Compare eval results
soup eval leaderboard                         Local model leaderboard
soup eval human --input p.jsonl               Human A/B evaluation
soup eval gate --suite gate.yaml              Run eval-gate suite standalonesoup eval quant-check --before X --after Y --tasks t.jsonl  Before/after quantsoup serve --model ./output --port 8000       OpenAI-compatible API server
soup serve --model ./output --backend vllm    vLLM backend (2-4x throughput)
soup serve --model ./output --backend sglang  SGLang backend
soup serve --model ./output --backend mii     DeepSpeed-MII backend (live)
soup serve --model ./output --speculative-decoding draft-model  Speculative decoding
soup serve --model <m> --auto-spec            Auto-pair draft model for speculative decoding
soup serve --model <m> --backend vllm --prefix-cache  vLLM prefix caching (RAG/agent)
soup serve --model <m> --structured-output json --json-schema s.json  Constrained output
soup serve --model <m> --structured-output regex --regex-pattern '...'  Regex-constrained output
soup serve --model <m> --dashboard            Live dashboard + /metrics endpoint
soup serve --model <m> --trace --trace-endpoint http://localhost:4317  OpenTelemetry tracing
soup serve --model <m> --trace-log ./serve.jsonl  Per-request JSONL log + rotation + secret redaction
POST /v1/adapters/activate/<name>             Hot-swap active LoRA adapter
soup sweep --config soup.yaml --param lr=...  Hyperparameter search
soup diff --model-a ./a --model-b ./b         Compare two models
soup data inspect <path>                      View dataset stats
soup data validate <path>                     Check format (auto-detect)
soup data convert <path> --to chatml          Convert between formats
soup data merge data1.jsonl data2.jsonl       Combine datasets
soup data dedup <path> --threshold 0.8        Remove duplicates (MinHash)
soup data stats <path>                        Extended statistics
soup data generate --prompt "..." --count 100 Generate synthetic data
soup data generate ... --provider ollama      Use local Ollama instance
soup data generate ... --provider anthropic   Use Claude API
soup data generate ... --provider vllm        Use local vLLM server
soup data generate ... --template code        Domain templates (code/conversation/qa/preference/reasoning)
soup data generate ... --quality-pipeline     Auto validate + filter + dedup
soup data augment <path> --strategy rephrase|translate|style  LLM-driven augmentationsoup data from-traces --logs l.jsonl --format langchain --signal thumbs_up --output p.jsonl  Preference pairs from traces
soup data from-traces ... --judge --min-confidence 0.7  LLM-judge confidence filter
soup data review prefs.jsonl --sample 10      Preview preference pairssoup data filter <path> --coherence 0.3       Quality filter (perplexity/coherence)
soup data sample <path> --n 1000             Random sample subset
soup data sample <path> --n 1000 --strategy diverse  Cluster-based diverse sampling
soup data sample <path> --n 1000 --strategy hard     Sample hardest examples
soup data sample <path> --pct 10             Sample by percentage
soup data split <path> --val 10 --test 10    Split into train/val/test
soup data split <path> --val 500 --absolute  Split with absolute counts
soup data split <path> --val 10 --stratify category  Stratified by field
soup data search "code instructions"         Search HuggingFace Hub for datasets
soup data search --sort likes --limit 10     Sort and paginate search results
soup data preview teknium/OpenHermes-2.5     Preview remote dataset metadata
soup data download user/dataset -o data.jsonl  Download HF dataset as JSONL
soup data download user/ds --samples 1000    Stream first 1000 samples
soup data register --name my-ds --path d.jsonl --format alpaca  Register dataset
soup data unregister --name my-ds            Remove from registry
soup data push --input d.jsonl --hf-dataset user/name  Upload local JSONL as HF dataset
soup data registry                           List all registered datasets
soup data demo                                List bundled demo JSONL fixtures
soup data demo alpaca_demo --output ./d.jsonl Copy a bundled demo JSONL fixture
soup data forge --docs ./docs --task sft --target-rows 1000  Synthetic data pipeline + provenance
soup data score --input rows.jsonl            Composite quality scorecard (PII + toxicity + lang + edu)
soup data decontaminate --input rows.jsonl --benchmarks mmlu,gsm8k  Drop benchmark-overlap rows
soup data toxicity --input rows.jsonl -o tox.jsonl  Flag toxic rows (keyword baseline)
soup data langdetect --input rows.jsonl -o tagged.jsonl  Tag each row with language code
soup data pii --input rows.jsonl -o pii.jsonl Flag rows containing email/phone/SSN/credit-card
soup data educational --input rows.jsonl -o scored.jsonl  Score educational value per row
soup train --config soup.yaml --tracker mlflow  MLflow / SwanLab / Trackio integration
soup profile --config soup.yaml              Estimate memory/speed before training
soup profile --config soup.yaml --gpu a100   Estimate for specific GPU
soup profile --config soup.yaml --json       Machine-readable output
soup cost --config soup.yaml                 Estimate training cost in USD across providers
soup cost --config soup.yaml --gpu H100      Estimate training cost for specific GPU
soup adapters list ./output/                 Scan for LoRA adapters
soup adapters info ./output/checkpoint-500/  Show adapter metadata
soup adapters compare adapter1/ adapter2/    Compare two adapters
soup loop init <model> --eval <s> --baseline <b>  Create .soup/loop.yaml (data flywheel)
soup loop status                              Counters + status (traces / pairs / runs / shipped)
soup loop watch [--detach] [--max-iter N]    Harvest → train → gate → deploy daemon
soup loop pause / soup loop resume           Atomic status flip
soup loop canary <adapter> --traffic 5%      Promote canary + auto-rollback on MAJOR
soup loop replay [<iter-id>]                 Replay a recorded iteration manifest
soup serve --model m --adapters chat=./c code=./d  Multi-adapter serving
soup migrate --from llamafactory config.yaml  Import config from LLaMA-Factory
soup migrate --from axolotl config.yml        Import config from Axolotl
soup migrate --from unsloth notebook.ipynb    Import config from Unsloth notebook
soup migrate --from llamafactory c.yaml --dry-run  Preview without writing
soup recipes list                             List all 43 ready-made recipes
soup recipes show llama3.1-8b-sft            Print recipe YAML
soup recipes use llama3.1-8b-sft             Copy recipe to soup.yaml
soup recipes search "reasoning"              Search by keyword/task/size
soup registry push --run-id <id> --name n --tag v1  Register runsoup registry list [--name n] [--tag v1]     List registry entriessoup registry show <ref>                      Entry details + artifacts + ancestors
soup registry diff <a> <b>                    Side-by-side config + eval delta
soup registry search "medical"                Search name/base/task/notes
soup registry promote <ref> --tag prod        Tag an entry (e.g. promote to prod)
soup registry delete <ref> --yes              Remove entry (cascades)
soup history <name>                           Lineage DAG tree for a namesoup can pack --entry-id <id> --out r.can     Pack registry entry as .cansoup can inspect r.can                        Preview manifest without extracting
soup can verify r.can                         Verify schema + config parseability
soup can fork r.can --out fork.can --modify training.lr=5e-5  Fork + re-pack
soup can run r.can --yes [--deploy] [--env-capture env.txt]  Run a .can end-to-end
soup can publish r.can --hf-hub user/name    Publish .can to HF Hub as dataset
soup runs                                     List training runs
soup runs show <run_id>                       Run details + loss graph + cost
soup runs compare <run_1> <run_2>             Compare two runs
soup runs replay <run_id>                     Replay summary + loss curve from history
soup why [run_id]                             Explain training anomalies (heuristic)
soup tui                                      Full-screen Textual dashboard (requires [tui] extra)
soup train --config soup.yaml --profile       Record torch.profiler trace to <output>/profiles/
soup --log-level quiet|normal|verbose|debug   Global logging tier (Rich-formatted)
soup ui [--port 7860]                         Web UI (experiments, training, data)
soup ui --public [--auth-token T]             Phone-scannable Web UI (v0.53.9)
soup tokenizer train --input c.jsonl --vocab-size N  Train BPE tokenizer (v0.53.9)
soup bench <model> --p50 --p95                Bench with tail-latency percentiles (v0.53.9)
soup bench <model> --backend auto             Auto-detect transformers/mlx backend (v0.53.9)
soup serve --reasoning-parser deepseek-r1     Strip <think> blocks from responses (v0.53.9)
soup doctor [--nccl]                          Check environment (optionally check NCCL bandwidth)
soup quickstart [--dry-run]                   Full demo
soup version [--full] [--json]                Show version (--full: system info, --json: JSON output)
soup --verbose <command>                      Full traceback on errors
```

## Supported Models

Soup works with **any** of the **340,000+** text-generation models on [HuggingFace Hub](https://huggingface.co/models?pipeline_tag=text-generation). If a model supports `AutoModelForCausalLM`, it works with Soup — zero config changes needed.

### Recommended Models

| Model Family | Models | Sizes | Best For |
|---|---|---|---|
| **Llama 4** | Llama-4-Scout-17B, Llama-4-Maverick-17B | 17B | General, multilingual |
| **Llama 3.x** | Llama-3.1-8B-Instruct, Llama-3.3-70B-Instruct | 1B–70B | Chat, instruction following |
| **Llama 3.2 Vision** | Llama-3.2-11B-Vision-Instruct, Llama-3.2-90B-Vision | 11B–90B | Image understanding |
| **Gemma 3** | Gemma-3-4B-IT, Gemma-3-9B-IT, Gemma-3-27B-IT | 4B–27B | Efficient, multilingual |
| **Qwen 3** | Qwen3-8B, Qwen3-14B, Qwen3-32B, Qwen3-235B-A22B | 0.6B–235B | Reasoning, code, MoE |
| **Qwen 2.5** | Qwen2.5-7B-Instruct, Qwen2.5-Coder-32B-Instruct | 0.5B–72B | Code, math |
| **DeepSeek** | DeepSeek-R1-Distill-Llama-8B, DeepSeek-V3-0324 | 1.5B–671B | Reasoning (GRPO), code |
| **Phi-4** | Phi-4-14B, Phi-4-mini-reasoning | 3.8B–14B | Compact reasoning |
| **Mistral** | Mistral-7B-Instruct-v0.3, Mistral-Small-24B-Instruct | 7B–24B | Fast, efficient |
| **Mixtral** | Mixtral-8x7B-Instruct-v0.1, Mixtral-8x22B | 47B–141B | MoE architecture |
| **CodeLlama** | CodeLlama-7b-Instruct-hf, CodeLlama-34b-Instruct | 7B–34B | Code generation |
| **StarCoder 2** | StarCoder2-15B, StarCoder2-7B | 3B–15B | Code completion |
| **Yi** | Yi-1.5-34B-Chat, Yi-1.5-9B-Chat | 6B–34B | Multilingual chat |
| **InternLM 3** | InternLM3-8B-Instruct | 8B | Chinese + English |
| **Falcon** | Falcon-11B, Falcon-40B-Instruct | 7B–180B | Open-weight |

### Vision Models (with `modality: vision`)

| Model | Size | Supported Formats |
|---|---|---|
| LLaMA-3.2-11B-Vision-Instruct | 11B | LLaVA, ShareGPT4V |
| Qwen2-VL-7B-Instruct | 7B | LLaVA, ShareGPT4V |
| Pixtral-12B-2409 | 12B | LLaVA, ShareGPT4V |

### Quick Size Guide

| VRAM | Max Model (QLoRA 4-bit) | Example |
|---|---|---|
| 8 GB | ~7B | Llama-3.1-8B, Mistral-7B |
| 16 GB | ~14B | Phi-4-14B, Qwen2.5-14B |
| 24 GB | ~34B | CodeLlama-34B, Yi-1.5-34B |
| 48 GB | ~70B | Llama-3.3-70B |
| 80 GB+ | 70B+ (full) or MoE | Mixtral-8x22B, DeepSeek-V3 |

> **Note:** Soup auto-detects your GPU and estimates the optimal batch size. Use `soup doctor` to check your setup.

## Docker

Run Soup without installing CUDA or PyTorch locally using the official Docker image (published to GitHub Container Registry on every release). This is the fastest way to get started and avoid dependency hell.

```bash
# Pull and run
docker pull ghcr.io/makazhanalpamys/soup:latest
docker run --gpus all -v $(pwd):/workspace ghcr.io/makazhanalpamys/soup train --config soup.yaml

# Or with compose (builds locally if image not pulled)
docker compose up
```

## Requirements

- Python 3.9+
- GPU with CUDA (recommended) or Apple Silicon (MPS) or CPU (experimental)
- 8 GB+ VRAM for 7B models with QLoRA

> **CPU note:** All training tasks (SFT, DPO, GRPO, PPO, KTO, ORPO, SimPO, IPO, Pretrain) work on CPU but will be very slow. Quantization (`4bit`/`8bit`) is auto-disabled on CPU. GRPO on CPU uses `min_new_tokens=1` to prevent empty generation errors. A default chat template is set automatically if the tokenizer lacks one. PPO datasets are tokenized before training to ensure compatibility with trl's experimental API.

### Optional Extras

| Extra | Install | What it adds |
|---|---|---|
| `vision` | `pip install 'soup-cli[vision]'` | Vision/multimodal fine-tuning (Pillow) |
| `qat` | `pip install 'soup-cli[qat]'` | Quantization-Aware Training (torchao) |
| `fast` | `pip install 'soup-cli[fast]'` | Unsloth backend (2-5x faster, -80% VRAM) |
| `ui` | `pip install 'soup-cli[ui]'` | Web UI + inference server (FastAPI + uvicorn) |
| `serve` | `pip install 'soup-cli[serve]'` | Inference server (FastAPI + uvicorn) |
| `serve-fast` | `pip install 'soup-cli[serve-fast]'` | vLLM inference backend (2-4x throughput) |
| `data` | `pip install 'soup-cli[data]'` | Deduplication (MinHash via datasketch) |
| `eval` | `pip install 'soup-cli[eval]'` | Benchmark evaluation (lm-evaluation-harness) |
| `deepspeed` | `pip install 'soup-cli[deepspeed]'` | Multi-GPU training (DeepSpeed ZeRO) |
| `liger` | `pip install 'soup-cli[liger]'` | Liger Kernel fused ops (20-60% memory savings) |
| `ring-attn` | `pip install 'soup-cli[ring-attn]'` | Ring FlashAttention (sequence parallelism) |
| `onnx` | `pip install 'soup-cli[onnx]'` | ONNX export (optimum + onnxruntime) |
| `tensorrt` | `pip install 'soup-cli[tensorrt]'` | TensorRT-LLM export (high-throughput GPU inference) |
| `dev` | `pip install 'soup-cli[dev]'` | Tests + linting (pytest, ruff) |

## Troubleshooting

### `ImportError: DLL load failed while importing _C` (Windows)

PyTorch's C extension fails to load. Common causes:

```bash
# Fix: reinstall PyTorch with the correct CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Or for CPU-only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Multiple Python versions conflict

If `pip show soup-cli` shows a different version than `soup version`, you have multiple Python installations with separate packages.

```bash
# Check which Python is active
python --version
which python    # Linux/macOS
where python    # Windows

# Fix: use a virtual environment
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
.venv\Scripts\activate       # Windows
pip install soup-cli
```

### Quick environment check

```bash
soup doctor    # Shows GPU, system resources, dependencies, and version info
```

## Development

```bash
git clone https://github.com/MakazhanAlpamys/Soup.git
cd Soup
pip install -e ".[dev]"

# Lint
ruff check soup_cli/ tests/

# Run unit tests (fast, no GPU needed)
pytest tests/ -v

# Run smoke tests (downloads tiny model, runs real training)
pytest tests/ -m smoke -v
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

## Trace-to-Preference: LLM-Judge Filter

`soup data from-traces --judge` filters harvested preference pairs through an LLM judge:

```bash
soup data from-traces \
  --logs ./prod-traces.jsonl --format langchain --signal thumbs_up \
  --output ./prefs.jsonl \
  --judge --judge-provider ollama --judge-model llama3 \
  --min-confidence 0.7
```

The judge scores `chosen` and `rejected` independently against its rubric (default helpfulness/accuracy/safety on a 1-5 scale). Pairs whose normalised `(chosen - rejected)` confidence falls below `--min-confidence` are dropped. Per-pair backend exceptions are counted (not crashed) and reported. Provider allowlist `{openai, server, ollama}` validated at the CLI boundary; SSRF protection on `--judge-api-base` carries over from `soup eval judge`.

## Inference Server Trace Log

`soup serve --trace-log <path>` writes a passive append-only JSONL log per chat completion:

```bash
soup serve --model ./out --trace-log ./serve-trace.jsonl --trace-log-cap-mb 100
```

Each line: `{"ts": ..., "prompt": ..., "response": ..., "latency_ms": ..., "tokens": ...}`. Path-containment validated, hard rotation cap (default 100 MB, one backup retained), symlink-reject on the backup path (TOCTOU defence), and `hf_*` / `sk-*` / `Bearer …` token shapes redacted to `<redacted>` before write. Failures (disk full, serialisation errors) never crash the request handler.

## GPU Live Monitor

```bash
soup monitor                # 2s refresh, Util / Mem / VRAM / Temp / Power per GPU
soup monitor --refresh 0.5  # faster polling
soup monitor --once         # single snapshot, no Live panel
```

Calls `nvidia-smi` via list-args subprocess (no shell), 5s timeout, list of `GpuSample` rows rendered into a Rich table. Apple Silicon prints a yellow advisory pointing at Activity Monitor / `powermetrics`; native Apple Silicon support lands in v0.44.1.

## Soup Fetch — Bundled Examples

```bash
soup fetch examples                          # list bundled entries
soup fetch examples llama-3.1-8b-lora        # write to ./llama-3.1-8b-lora.yaml
soup fetch examples qwen2.5-7b-dpo -o ./my-config.yaml --force
soup fetch deepspeed_configs zero3-cpu-offload
```

Closed catalog (`MappingProxyType`) of ready-to-edit YAML / JSON. Output path cwd-contained, bundled-source `os.path.commonpath` check (defends against catalog escape), `os.lstat + S_ISLNK` symlink-reject at the write target.

## Soup Quantize — Ergonomic Export Alias

```bash
soup quantize ./out --to gguf --bits 4
soup quantize ./out --to gptq --bits 4 -o ./out-gptq
```

Prints the equivalent `soup export …` invocation (escaped via `shlex.quote`) for copy-paste. Intentionally does NOT in-process call `soup export` — Typer commands aren't safe to re-enter.

## FSDP Shard Consolidation

```bash
soup merge-sharded-fsdp-weights ./fsdp-checkpoint -o ./merged.safetensors --yes
```

Plans consolidation of `pytorch_model_fsdp_*.bin` shard files into a single `.safetensors`. v0.44.0 ships the planner with cwd-containment + size-cap (`_MAX_SHARDS=1024`); live torch-side weight consolidation lands in v0.44.1.

## Llama 4 Delinearizer

```bash
soup delinearize-llama4 ./llama4-checkpoint --target ./out-delinearized --yes
```

Plans Llama 4 expert-weight reshape for export. v0.44.0 ships the planner; live runtime in v0.44.1. `is_llama4_model` uses a word-boundary regex matching the `is_gemma4_model` pattern — `ungemma-llama-4ish` is rejected.

## Llama.cpp Proxy

```bash
soup llama --help                  # list supported subcommands
soup llama cli -m model.gguf -p "Hello"
soup llama gguf-split --merge a.gguf b.gguf out.gguf
soup llama server -m model.gguf
```

Closed allowlist: `cli` / `mtmd-cli` / `gguf-split` / `server` / `quantize`. Forwards to `llama-*` binary on PATH (`shutil.which`) with **filtered child env** — `HF_TOKEN` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` and other secrets are dropped before exec; only `PATH` / `HOME` / `USER` / locale + llama.cpp-recognised `LLAMA_CPP_HOME` / `GGML_*` / `OMP_NUM_THREADS` are forwarded.

## Ctrl+C Graceful Save

First SIGINT → trainer writes a checkpoint and continues. Second SIGINT → trainer stops cleanly after the next save. No-state fallback raises `KeyboardInterrupt` so the user never gets stuck. `GracefulSaveHandler.install()` is idempotent and swallows `signal.signal` failures on non-main threads.

## Checkpoint-Now Trigger File

```bash
touch ./out/.checkpoint_now    # trainer saves on the next step, then deletes the trigger
```

Path containment via `is_under_cwd`; `os.lstat + S_ISLNK` rejection at the trigger target so a pre-placed symlink can't redirect the write.

## Onboarding Wizard Helper

```python
from soup_cli.utils.onboarding import render_onboarding_yaml

text = render_onboarding_yaml({
    "base": "meta-llama/Llama-3.2-1B",
    "dataset": "./train.jsonl",
    "task": "sft",
    "quantization": "4bit",
    "epochs": 3,
})
```

Five-question wizard input → fully-validated `soup.yaml`. Literal allowlists on `task` (`sft` / `dpo` / `kto` / `orpo` / `simpo` / `ipo` / `bco` / `preference`) and `quantization` (`4bit` / `8bit` / `none`); `epochs ∈ [1, 10]`; `output` cwd-contained; null-byte rejection on every string.

## Tail-Latency Stats + Tool-Call Timer

```python
from soup_cli.utils.tail_latency import summarise_latency
from soup_cli.utils.tool_outputs import ToolOutputsBuffer, ToolCallTimer

stats = summarise_latency([12.3, 14.1, 9.7, 18.8, 11.2])
# TailLatencySummary(count=5, mean=..., p50=..., p95=..., p99=..., ema=...)

buffer = ToolOutputsBuffer()
with ToolCallTimer(buffer, name="fetch_url") as timer:
    timer.set_output("...")
```

Pure-Python EMA + linear-interp percentiles (DoS cap: `MAX_SAMPLES=1_000_000`). `ToolOutputsBuffer` is a thread-safe `collections.deque(maxlen=1000)` ring with truncated previews; `ToolCallTimer` records duration / output / error per invocation for tool-calling SFT runs.

## Web UI Plugin Registry + Env Knobs

```python
# soup_cli/ui/plugins/my_tab.py
from soup_cli.ui.plugins import register_tab

def render_my_tab(request) -> str:
    return "<div>my tab body</div>"

register_tab(name="my-tab", title="My Tab", render=render_my_tab)
```

Drop-in plugin registry with kebab-case name allowlist, 32-tab cap, idempotent re-register. Plus `API_HOST` / `API_PORT` / `API_KEY` / `GRADIO_HOST` / `GRADIO_PORT` env knobs for FastAPI + Gradio surfaces.

## Standalone Sweep Config

```bash
soup sweep --config sweep.yaml
```

```yaml
# sweep.yaml
strategy: random
n_runs: 20
seed: 42
params:
  lr: [0.0001, 0.0005, 0.001]
  epochs: [1, 3, 5]
```

Strict scalar allowlist on values (`str` / `int` / `float` / `bool`); `_MAX_FILE_BYTES=256KB`, `_MAX_PARAM_KEYS=32`, `_MAX_VALUES_PER_KEY=64`; `SweepSpec.params` is `MappingProxyType[str, Tuple[Any, ...]]` for genuine immutability.

## Deploy Autopilot

Pick the optimal PEFT + quantisation + speculative-decoding combo for your hardware target in one command:

```bash
soup deploy autopilot --target rtx-4090-24gb --base meta-llama/Llama-3.2-1B
# Writes:
#   deploy_autopilot.yaml  — ready-to-train soup.yaml recipe
#   deploy_autopilot.sh    — planned deploy shell script
```

Profiles ship out of the box for Apple Silicon (`mac-m3`, `mac-m4-pro`), consumer NVIDIA (`rtx-3060-12gb`, `rtx-4090-24gb`), mobile (`iphone-16`, `pixel-9`), local runtimes (`ollama-local`, `lm-studio`), and cloud (`runpod-a100`, `hf-jobs-h100`). `--list` shows the full table. Every profile is a frozen dataclass with closed allowlists on runtime / quant / PEFT — bad config values fail at import time. The generated bash uses `shlex.quote` on the model path and writes are protected by cwd containment + `os.lstat + S_ISLNK` TOCTOU rejection.

## Agent Forge

Turn an OpenAPI 3.x, MCP server manifest, or GraphQL introspection JSON straight into a tool-calling SFT dataset — no manual labelling, no scaffolding:

```bash
# 1. Parse spec + synthesise a tool-calling dataset
soup agent synth --spec api.yaml --output ds.jsonl --examples-per-endpoint 4

# 2. Plan the training run (prints the soup train invocation)
soup agent train --spec api.yaml --base meta-llama/Llama-3.2-1B

# 3. Score model predictions against the spec catalog
soup agent eval --spec api.yaml --predictions preds.jsonl
```

Each row of the synthesised dataset is `{messages: [user, assistant_with_tool_call], tool: <name>, source_endpoint: <path>}`. `$ref` strings in OpenAPI are left opaque (no external resolution — defends against file-read SSRF), `yaml.safe_load` only, 5 MiB spec cap, 10 000-endpoint cap, atomic JSONL write via staged-tempfile + `os.replace`. `eval` enforces a 1 000 000-line cap on predictions and rejects symlinks at every read/write boundary.

## Synthetic Data Forge

Multi-stage synthetic data pipeline with full provenance — every synthetic row links back to the source document, the judge call, and the filter score:

```bash
# Pipeline: chunk docs → judge → active-prune → JSONL + provenance manifest
soup data forge \
    --docs ./my_docs/ \
    --task sft \
    --target-rows 1000 \
    --uncertainty-threshold 0.4 \
    --output forge_dataset.jsonl \
    --provenance forge_provenance.json
```

Three tasks supported: `sft` (Q&A pairs), `preference` (chosen/rejected), `tool` (tool-call hypotheses). Active learning prunes rows whose judge reply is too close to the source chunk (low Jaccard distance), keeping only uncertain / informative samples. The provenance manifest is a separate JSON file mapping every row id to `{source_doc, judge_id, chunk_id, filter_score}` so you have a complete audit trail for compliance.

Document discovery is one level deep over `.txt` / `.md` / `.json` / `.jsonl`; dotfiles + symlinked directories are skipped. All paths are cwd-contained, all writes are atomic via staged-tempfile + `os.replace`, and write targets are rejected if they're symlinks. **Judge providers are live**: `--judge-provider ollama` (localhost-only), `--judge-provider anthropic` (env-only API key), `--judge-provider vllm` (scheme-validated). Per-call judge exceptions logged at DEBUG.

## Data Quality Scorecard

Composite, lightweight data-quality triage — no GPU, no 200 MB Presidio model:

```bash
# Single-shot composite scorecard
soup data score --input training.jsonl

# Standalone subcommands — JSONL-in, enriched JSONL-out
soup data pii          --input training.jsonl --output pii_flagged.jsonl
soup data toxicity     --input training.jsonl --output tox_flagged.jsonl --threshold 0.1
soup data langdetect   --input training.jsonl --output tagged.jsonl
soup data educational  --input training.jsonl --output scored.jsonl
soup data decontaminate --input training.jsonl --benchmarks mmlu,gsm8k,humaneval --output clean.jsonl
```

The scorecard reports PII flagged, toxic flagged, language distribution, mean educational value, and decontamination removed. PII detection uses a narrow ReDoS-hardened regex set (email / phone / SSN / credit-card) with a 50 KB pre-cap on every input. Language detection is a stopword heuristic across six languages. Toxicity is a keyword baseline; the Llama-Guard-3-1B variant + FineWeb-Edu classifier ship behind `[data-pro]` extras. Decontamination uses n-gram containment against benchmark corpora: use `--benchmarks mmlu,gsm8k` for built-in allowlist, or `--benchmark-file custom_benchmark.jsonl` for your own corpus.

## Remote Datasets (S3 / GCS / Azure / OCI)

Point `data.train` at any object in the v0.42.0 fsspec allowlist and `soup train` will stream it through `fsspec.open` after running the URI through the same SSRF-hardened validator used everywhere else in Soup (bucket regex, no userinfo / query / fragment):

```yaml
data:
  train: s3://my-bucket/datasets/train.jsonl
  format: alpaca
  streaming: true       # opt-in HF datasets streaming with shuffle
  buffer_size: 10000    # shuffle buffer (requires streaming=true)
```

Recognised schemes: `s3://`, `gs://`, `gcs://`, `az://`, `abfs://`, `abfss://`, `oci://`. The matching backend SDK (`s3fs` / `gcsfs` / `adlfs` / `ocifs`) is lazy-imported — install only what you need or grab the convenience extra:

```bash
pip install soup-cli[remote]   # fsspec + s3fs + gcsfs + adlfs
```

Materialised rows are capped at 1M to defend against pathological remote objects; use a local split for larger jobs.

## Alternative Model Hubs (ModelScope / Modelers)

Set `training.hub` to fetch the base model from a non-HF Hub:

```yaml
base: baichuan-inc/Baichuan2-7B
task: sft
training:
  hub: modelscope     # or "modelers"
```

`soup train` pre-fetches the model into `./.soup_hub_cache/<sanitized-slug>/` via the matching SDK (`modelscope.snapshot_download` / `openmind_hub.snapshot_download`) and rewrites `cfg.base` to the local snapshot. Re-runs reuse the cached snapshot. Both `huggingface-hub`, `modelscope`, and `openmind-hub` are lazy-imported — install only what you need.

Programmatic API:

```python
from soup_cli.utils.hubs import download_repo, upload_repo

local_path = download_repo("modelscope", "baichuan-inc/Baichuan2-7B", local_dir="./snap")
upload_repo("modelers", "my-org/my-model", folder_path="./output", commit_message="Soup v0.53.8")
```

The dispatcher enforces shape validation on every input (bool / null-byte / leading-slash / `..` segments / control characters / oversize all rejected) and runs cwd-containment on `local_dir` / `folder_path`.

## Experiment Trackers (MLflow / SwanLab / Trackio)

Pick a tracker on the CLI; Soup threads it into HF Trainer's `report_to`:

```bash
soup train --tracker mlflow
soup train --tracker swanlab
soup train --tracker trackio
```

If the package is not installed, Soup now surfaces a friendly advisory before training starts instead of a mid-run ImportError:

```
--tracker mlflow requires the 'mlflow' package. Install with: pip install soup-cli[trackers] (or pip install mlflow)
```

```bash
pip install soup-cli[trackers]   # mlflow + swanlab + trackio
```

## Telemetry (opt-IN, hardware-info-only)

Soup ships an opt-IN telemetry sender that POSTs hardware-info-only payloads (`soup_version` / `command` / `python` major.minor / `os` / `arch` / optional `duration_seconds`) — no dataset paths, model names, or config contents. Enable per-shell:

```bash
SOUP_TELEMETRY=1 soup train --config soup.yaml
```

The sender uses a 1-second hard timeout, HTTPS-only with private-IP / link-local rejection (same SSRF policy as hub endpoints), and swallows every exception silently — telemetry can never crash training. Disabled by default until a public privacy policy ships.

## HF Space SDK Auto-Pick

When you deploy a custom Space template directory, Soup now picks `space_sdk="streamlit"` / `"gradio"` from the rendered `requirements.txt`:

```bash
soup deploy hf-space --space my-org/my-app --model my-org/my-model --template-dir ./my-template
```

If `requirements.txt` lists `streamlit`, the Space is created with the Streamlit SDK. Otherwise (no requirements, gradio listed, etc.), Soup falls back to the Gradio default. The HF Hub allows `docker` and `static` SDKs too, but those cannot be inferred from `requirements.txt` alone — use the built-in templates or supply a custom one with an explicit `--sdk` override.

## Plugin System

Drop a Python module under `soup_cli/plugins/` (or any package importable by Soup) and register at import time:

```python
from soup_cli.plugins import register_plugin

class MyPlugin:
    def pre_train(self, ctx):
        ...
    def post_train(self, ctx):
        ...

register_plugin(
    name="my-plugin",
    version="1.0.0",
    plugin=MyPlugin(),
    description="Hooks into pre/post-train",
    templates=["my-template"],         # optional
    model_groups=["my-arch-family"],   # optional
)
```

```bash
soup plugins              # list registered plugins
soup plugins enable foo
soup plugins disable foo
```

Plugin names are kebab-case (`^[a-z0-9][a-z0-9-]{0,39}$`); versions are semver-ish (`MAJOR.MINOR.PATCH`); registry caps `_MAX_PLUGINS=64`, `_MAX_TEMPLATES_PER_PLUGIN=32`, `_MAX_MODEL_GROUPS_PER_PLUGIN=32`. Re-registering the same `(name, version, plugin, templates, model_groups, description)` is idempotent; any field mismatch is rejected with a clear error. Trainer-callback wiring of `pre_train` / `post_train` / `pre_step` / `post_step` lands in v0.45.1.

## Anthropic Messages API Converter

Pure-Python converters between OpenAI chat-completions and Anthropic Messages payload shapes:

```python
from soup_cli.utils.anthropic_messages import to_anthropic, from_anthropic

anthropic_payload = to_anthropic({
    "model": "claude-3-5-sonnet",
    "messages": [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hi"},
    ],
    "max_tokens": 256,
})
```

Multiple `system` messages join with `\n\n`. `tool` role with structured (list) content is concatenated into a single `tool_result` text block, never silently dropped. `max_tokens` capped at 16384, `temperature` bounded `[0.0, 2.0]`. Live `/v1/messages` endpoint inside `soup serve` lands in v0.45.1.

## Server-Side Tools

```python
from soup_cli.utils.server_tools import (
    SUPPORTED_TOOLS, WebSearchConfig, is_domain_allowed, validate_web_search_config,
)

# SUPPORTED_TOOLS == frozenset({"python", "bash", "web_search"})
config = WebSearchConfig(
    domain_allowlist=("example.com", ".docs.example.com"),
    rate_limit_per_minute=30,
)
validate_web_search_config(config)
is_domain_allowed("a.docs.example.com:443", config.domain_allowlist)  # True
is_domain_allowed("[::1]", config.domain_allowlist)                   # False
```

`python` and `bash` reuse the v0.25.0 RLVR sandbox; `web_search` is gated by an explicit domain allowlist (default empty = deny all). `is_domain_allowed` strips `:port` suffixes before matching and rejects IPv6 literals so `Host: api.example.com:443` matches `api.example.com`. Live HTTP tool endpoints in v0.45.1.

## External Integrations Catalog

```python
from soup_cli.utils.integrations import list_integrations, get_integration

list_integrations()                       # 15 entries
get_integration("lm-studio").target_artifacts   # ("gguf",)
```

15 ecosystem targets covered: `lm-studio`, `comfyui`, `stable-diffusion-cpp`, `open-webui`, `ollama`, `tei`, `pgvector`, `faiss`, `weaviate`, `sentence-transformers`, `claude-code`, `cursor`, `continue`, `cline`, `sillytavern`. Auto-detect + launch wiring lands with v0.46.0 Deploy Autopilot.

## Advanced Trainer Plugins

```python
from soup_cli.utils.trainer_plugins import validate_trainer_plugin_list

validate_trainer_plugin_list(["grokfast", "spectrum"])
# returns ("grokfast", "spectrum") — canonical lowercase, dedup, ≤ 8 entries
```

6-entry allowlist (`cce_plugin`, `grokfast`, `spectrum`, `llmcompressor`, `sonicmoe`, `math_verify`) so a future `training.trainer_plugins: [...]` schema field has a stable surface. Live callbacks in v0.45.1.

## Data Recipe DAG

```bash
soup data recipe my_recipe.yaml
```

```yaml
nodes:
  - name: seed1
    kind: seed
    config: {path: prompts.jsonl}
  - name: llm1
    kind: llm_text
  - name: judge1
    kind: judge
  - name: samp1
    kind: sampler
edges:
  - [seed1, llm1]
  - [llm1, judge1]
  - [judge1, samp1]
```

Closed node-kind allowlist (`seed` / `llm_text` / `code` / `judge` / `validator` / `sampler`); Kahn's topological sort via `collections.deque` (deterministic, O(N+E)); cycle / self-loop / duplicate-edge / dangling-edge / unknown-kind rejection. `_MAX_NODES=256`, `_MAX_EDGES=1024`, `_MAX_FILE_BYTES=1MiB`. The recipe file must stay under cwd and **must not be a symlink** (`os.lstat + S_ISLNK` TOCTOU defence). Live offline runner against a local model lands in v0.45.1.

## Curriculum-Aware Training (BETA)

Layer dynamic re-weighting on top of the static `curriculum` bucketer. Every N steps the trainer aggregates per-sample loss + grad-norm into a per-bucket uncertainty signal, runs it through a softmax (temperature-controlled) with floor (water-filling so no bucket drops below `curriculum_dynamic_floor`), and re-weights the sampler. Empty buckets fall back to the median of populated buckets; degenerate inputs return uniform.

```yaml
training:
  curriculum: true                          # static bucketer (v0.23.0)
  curriculum_buckets: 4
  curriculum_dynamic: true                  # NEW — dynamic re-weighting
  curriculum_dynamic_recompute_steps: 50    # refresh every 50 global steps
  curriculum_dynamic_floor: 0.05            # min weight per bucket
  curriculum_dynamic_temperature: 1.0       # softmax temp on uncertainty
```

Visualise the recorded bucket-weight evolution with `soup runs curriculum-curve <run_id>`.

DDP / grad-accum safety: multi-rank launches must wire an `all_reduce` hook on per-bucket stats (a cross-validator rejects un-coordinated multi-rank runs upfront). Multi-trainer expansion beyond `sft` / `pretrain` is tracked for v0.48.1.

## Data Mixing Optimizer (BETA)

Search for the dataset mixture weights that minimise eval loss on a short proxy run.

```bash
soup data mix --optimize --budget 1h \
    --datasets dolma.jsonl,wikipedia.jsonl,arxiv.jsonl \
    --num-probes 8 --output mix_recipe.yaml
```

Writes a YAML recipe with a `data.interleave` block you can splice into your `soup.yaml`. `--budget` accepts `60s` / `5m` / `1h` / `24h`. Per-candidate proxy failures are isolated (DEBUG-logged, sentinel high loss recorded) so a single OOM combo does not abort the whole search; `partial=True` is surfaced in the report when the budget cap trips mid-loop.

Re-apply a previously written recipe:

```bash
soup data mix --apply mix_recipe.yaml
```

Live wiring of the proxy training loop into a short `soup train` run is the v0.48.1 deliverable; v0.48.0 ships a synthetic offline proxy (quadratic penalty around the uniform simplex) so the budget tracker, optimiser surface, and recipe writer can be exercised without GPUs. `scikit-optimize` is opt-in via `OptimizerProtocol`; the default fallback is a deterministic Dirichlet sampler.

## TTS Fine-Tuning (BETA, v0.52.0)

Schema-only this release; live trainer wiring lands in v0.52.1.

Five upstream model families are recognised: `orpheus`, `sesame_csm`, `llasa`, `spark`, `oute`. Pair `task: tts` with `modality: audio_out` and set `training.tts_family`. Orpheus + Oute support emotion conditioning via `training.tts_emotion` from a per-family allowlist (Orpheus: neutral / happy / sad / angry / excited / calm / whisper / laugh; Oute: neutral / happy / sad / angry / calm / excited).

```yaml
base: canopylabs/orpheus-3b-0.1-ft
task: tts
modality: audio_out
data:
  train: ./data/tts_train.jsonl
  format: audio
  audio_dir: ./data/audio
training:
  tts_family: orpheus
  tts_emotion: neutral
```

Five ready-made recipes ship in v0.52.0: `orpheus-tts-sft`, `sesame-csm-tts`, `llasa-tts`, `spark-tts`, `oute-tts` — copy with `soup recipes use <name>`. Cross-validators reject the `mlx` backend, `modality != audio_out`, and emotion tags outside the per-family allowlist.

## Classifier / Reranker / Cross-Encoder Training (BETA, v0.52.0)

Three new task types build on the existing embedding trainer: `task: classifier` (single-label or multi-label sequence classification), `task: reranker` (pointwise retrieval scoring), `task: cross_encoder` (paired-input scoring). Schema-only; live trainer wrapper in v0.52.1.

```yaml
base: BAAI/bge-base-en-v1.5
task: classifier
data:
  train: ./data/classification.jsonl
training:
  num_labels: 3
  classifier_kind: single_label
  label_names: [negative, neutral, positive]
```

`num_labels` is bounded `[1, 1024]` with explicit bool-before-int rejection; `label_names` (optional) must be unique, ≤128 chars each, and match `num_labels` in length when set.

## Knowledge Distillation (BETA, v0.52.0)

New `task: distill` with `training.teacher_model` (HF id or local path), `training.distill_divergence` (`kl` / `forward_kl` / `reverse_kl` / `js` — `kl` canonicalises to `forward_kl`), and `training.distill_temperature` (bounded `[0.05, 100.0]`, finite-only). Schema-only; live loop in v0.52.1.

```yaml
base: meta-llama/Llama-3.2-1B
task: distill
data:
  train: ./data/distill.jsonl
training:
  teacher_model: meta-llama/Llama-3.1-8B
  distill_divergence: forward_kl
  distill_temperature: 2.0
```

The cross-validator rejects `task='distill'` without `teacher_model`, and rejects `teacher_model` / `distill_*` fields when `task` is anything other than `distill`.

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

## EBFT + GDPO (BETA, v0.52.0)

Energy-Based Fine-Tuning (axolotl) lands as `training.ebft_variant ∈ {structured, strided}` + `training.ebft_temperature` (bounded `[1e-4, 100.0]`). Gated to `task: sft`. Generalized DPO lands as `training.gdpo_variant ∈ {standard, length_normalized, margin}` — gated to `task ∈ {dpo, preference}`. Live loss kernels in v0.52.1.

## MoE Expert Quantization + Router-Only Training (v0.52.0)

For fused-MoE models trained with `moe_lora: true`, two new toggles ship:

- `training.moe_expert_quant: nf4 | int8_rowwise` — per-expert weight quantization (axolotl).
- `training.train_router_only: true` — freeze every expert and train only the gating router (unsloth pattern).

Both reject silently-no-op combinations: setting either flag without `moe_lora=true` fails at config load with an actionable message.

## gpt-oss `reasoning_effort` + `train_on_eot` (v0.52.0)

`training.reasoning_effort: low | medium | high` injects a system-prefix token at training time for gpt-oss models; `training.train_on_eot: true` includes explicit EOT/EOS control tokens in the SFT loss (axolotl `train_on_eot`). Both are gated to the SFT-family task set (`sft` / `pretrain` / `distill` / `classifier` / `reranker` / `cross_encoder`) — setting them on DPO / GRPO / PPO / etc. fails at config load. Live formatter wiring in v0.52.1.

## Unsloth Dynamic 2.0 GGUF Ladder (v0.53.0)

`soup export --format gguf-ud --calibration-data <calib.jsonl>` is the planned dispatch surface for the 14-entry UD ladder (`UD-Q8_K_XL` … `UD-IQ1_M`). v0.53.0 ships the closed-allowlist validators, `MappingProxyType`-wrapped metadata, and a calibration-data path shape check; live llama.cpp `imatrix` invocation lands in v0.53.1. The IQ + Apple/ARM-friendly GGUF flavours (`IQ4_NL`, `Q4_0_4_4`, `Q5_K_M`, etc.) ship as separate frozensets so future export-CLI dispatch can pick by family.

## KV Cache Types (v0.53.0)

`training.kv_cache_type: q8_0 | bf16 | f16 | fp8` controls the inference-time KV cache element type. `fp8` is Hopper-only; the MLX backend is rejected at config load. The other three types pass through every backend in v0.53.0; v0.53.1 may narrow MLX further once the runtime serve path lands. The Hopper SM-capability check (compute capability ≥ 9.0) is intentionally runtime-only — `pip install -U vllm` users on a Hopper box won't trip it unless they ship a Hopper-incompatible GPU into the runtime.

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

## Soup Plugin Callbacks

Register a plugin once via the v0.45.0 registry API; v0.53.6 wires it into every
transformer-backend trainer as a real HF `TrainerCallback`:

```python
# soup_cli/plugins/my_plugin.py — auto-discovered at `soup` startup
from soup_cli.plugins import register_plugin

class MyPlugin:
    def pre_train(self, ctx):
        print("training about to start, args =", ctx["args"])

    def post_step(self, ctx):
        if ctx["state"].global_step % 100 == 0:
            print(f"step {ctx['state'].global_step}")

register_plugin(name="my-plugin", version="0.1.0", plugin=MyPlugin())
```

A misbehaving plugin hook is swallowed at WARNING — one bad plugin must never crash
a multi-hour training run. The hook snapshot is taken at callback-construction time,
so a plugin registered MID-run does not retroactively receive events.

## Anthropic Messages Endpoint

Both `soup serve --backend transformers` and `soup serve --backend vllm` now expose
a POST `/v1/messages` route that accepts Anthropic Messages-shaped payloads:

```bash
curl http://localhost:8000/v1/messages -H "Content-Type: application/json" -d '{
  "model": "my-model",
  "messages": [{"role": "user", "content": "hello"}],
  "max_tokens": 64
}'
```

Non-streaming requests return Anthropic-shaped envelopes. Streaming (`stream: true`)
returns Anthropic event-shape SSE with `message_start` → `content_block_delta` →
`message_delta` + `message_stop` events. Validation errors return a generic
`"Invalid request"` 400 body with details logged server-side at DEBUG. CORS restricted
to loopback-only (`localhost` / `127.0.0.1`) on both backends.

## N-gram Speculative Decoding

When a server is configured with an `NgramSpecConfig`, every chat completion forwards
`prompt_lookup_num_tokens=N` into `model.generate(...)` (HF Transformers ≥ 4.38
prompt-lookup decoding — no draft model required). Mutually exclusive with a real
`assistant_model`; if both are set, the real draft model wins.

## Server-Side Tool Endpoints

Three POST routes are now available on `soup serve`:

- **`POST /v1/tools/python`** — Sandboxed Python execution. Requires Bearer token auth.
  Wraps the RLVR sandbox with 5-second timeout and 64KB code cap. Returns 200 with
  `stdout` / `stderr` / `return_value`; 400 on validation error; 401 on bad auth.

- **`POST /v1/tools/web_search`** — Domain-allowlisted web search. Requires Bearer token
  auth. Uses httpx backend with hard 5-second timeout and 5-result cap. Returns results
  as `[{url, title, snippet}]` with snippets sanitized (null bytes stripped).
  Deny-by-default via `WebSearchConfig.domain_allowlist`.

- **`POST /v1/tools/bash`** — Deferred to v0.53.8. Current child-process isolation
  insufficient for `/bin/sh -c` (subprocess escapes the RLVR sandbox). Returns 501 with
  v0.53.8 marker pending container/namespace work.

## AOT Tokenization with `soup data preprocess`

Pre-tokenize your dataset once and cache Arrow shards keyed by
`(dataset, tokenizer, max_length, format)`:

```bash
soup data preprocess soup.yaml --output ./tokenized_cache
```

SFT and Pretrain trainers short-circuit at schema validation when
`format: pre_tokenized` + `tokenized_path: ./tokenized_cache` is set, eliminating
the per-epoch tokenization tax. Cache keys ensure resume safety; partial runs pick
up from the last completed shard.

## Data Recipe DAG Runner (`soup data recipe --execute`)

Execute a Data Recipe DAG end-to-end:

```bash
soup data recipe path/to/recipe.yaml --execute --output ./out
```

Six node kinds now run live: **seed** (JSONL load), **llm_text** (LLM generation via
any provider), **code** (execution via RLVR sandbox), **judge** (binary scoring),
**validator** (regex or JSON schema), **sampler** (deterministic selection). Checkpoint
written per node; resume rehydrates from per-node sidecars. Failed rows logged with
redacted reasons (paths stripped, capped at 256 chars).

## Bill of Materials (`soup bom emit`)

Emit a **CycloneDX 1.6 ML-BOM** or **SPDX 2.3 + AI profile** bill of materials from any
training run. Procurement teams and compliance auditors can ingest the BOM directly into
their existing tooling — no custom parser required.

```bash
soup bom emit \
  --name adapter-v1 --version 0.1.0 \
  --base-model meta-llama/Llama-3.1-8B \
  --base-sha aaaa...64hex \
  --config-sha bbbb...64hex \
  --task sft --license apache-2.0 \
  --format both --output bom
# writes bom.cdx.json + bom.spdx.json
```

Root component is `type=machine-learning-model` (per CycloneDX ML-BOM extension). Base
model + parent adapters + per-artifact files appear as components with SHA-256 hashes.
License chain uses SPDX identifiers. Energy + CO₂ properties (when attached via the
energy schema) ship under `metadata.properties`.

## Provenance Attestations (`soup attest emit`)

Emit an **in-toto v1 Statement** wrapping a **SLSA-3 provenance v1 predicate** for each
Soup Can lifecycle stage:

```bash
soup attest emit \
  --stage train \
  --subject adapter-v1 \
  --sha aaaa...64hex \
  --builder soup-cli@0.59.0 \
  --output att.json
```

Stages are a closed allowlist: `extract` / `train` / `eval` / `export` / `publish`.
Subject SHA must be 64-hex (sha256). The default `--sign unsigned` backend ships now;
Sigstore (OIDC-via-GitHub) and ed25519 air-gap signing arrive in v0.59.1.

## EU AI Act Annex XI/XII Auto-Doc (`soup train --annex-xi`)

Render an EU AI Act Annex XI (technical documentation, Sections 1+2) or Annex XII
(Article 53(1)(d) public training summary) directly from a training run:

```bash
soup train --config soup.yaml --annex-xi annex.md
```

Top-10 domains by share, modality breakdown, training compute / kWh / CO₂, model
description, base model, run id. Markdown body now; PDF in v0.59.1. Operator-controlled
fields are escape-neutralised (`|[](){}!<>` + newline / CR / tab) so a malicious model
name can't inject a forged heading into downstream PDF/HTML renderers.

## Audit Log (`soup audit-log`)

Tail or rotate the HIPAA/SOC2-shaped JSONL audit log at `~/.soup/audit.jsonl` (override
via `SOUP_AUDIT_LOG_PATH`, containment-checked to `$HOME / $CWD / $TMPDIR`):

```bash
soup audit-log tail --limit 50          # Rich table view
soup audit-log tail --json              # raw JSONL for SIEM ingestion
soup audit-log rotate --cap-mb 100      # force a rotation pass
```

PII redaction across **every** string field (`hf_*` / `sk-*` / `Bearer …` → `<redacted>`)
via the v0.40.3 `_SECRET_RE` policy. POSIX `O_NOFOLLOW` + `0o600` perms, atomic-append,
rotation at 100 MiB with symlink rejection at the backup path.

## Reproducibility Receipt (`soup train --repro-receipt`)

SR 11-7-style reproducibility receipt captures seeds (torch + numpy + python), kernel
versions (CUDA + cuDNN + NCCL), GPU model + driver, OS + arch:

```bash
soup train --config soup.yaml --repro-receipt repro.json
```

Bank model-risk teams and regulated-org auditors get a single JSON file that fingerprints
the exact environment the run executed in. Atomic write, cwd-contained.

## Changelog

See [GitHub Releases](https://github.com/MakazhanAlpamys/Soup/releases) for version history.

## License

Apache-2.0
