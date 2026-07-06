# Training Tasks & Methods

[← Back to the Soup README](../README.md)

> SFT, DPO/GRPO/PPO/KTO/ORPO/SimPO/IPO/BCO, tool-calling, PRM, pre-training, distillation, classification, vision/audio/TTS, unlearning, RAFT/RA-DIT, and the loop-hardening detectors.

**Contents:**

- [Loop Hardening](#loop-hardening)
- [Unlearning (`task='unlearn'`, NPO / SimNPO / RMU)](#unlearning-taskunlearn-npo--simnpo--rmu)
- [Continued Pre-training](#continued-pre-training)
- [Knowledge Distillation](#knowledge-distillation)
- [Sequence Classification](#sequence-classification)
- [Reasoning Effort + EOT Control](#reasoning-effort--eot-control)
- [EBFT / GDPO Loss Variants](#ebft--gdpo-loss-variants)
- [GRPO Objective Variants](#grpo-objective-variants)
- [Process Reward Model (PRM)](#process-reward-model-prm)
- [PRM-guided GRPO (process-supervised RL)](#prm-guided-grpo-process-supervised-rl)
- [Weighted Multi-Objective Preference Loss](#weighted-multi-objective-preference-loss)
- [MoE Model Support](#moe-model-support)
- [Vision / Multimodal Fine-tuning](#vision--multimodal-fine-tuning)
- [Audio / Speech Fine-tuning](#audio--speech-fine-tuning)
- [GRPO Plus — Objective Variants, Long-Context RL, Multi-Turn Agents](#grpo-plus--objective-variants-long-context-rl-multi-turn-agents)
- [DPO Training](#dpo-training)
- [Preference Variety — BCO + Unified Dispatcher + KL Variants](#preference-variety--bco--unified-dispatcher--kl-variants)
- [GRPO Training (Reasoning)](#grpo-training-reasoning)
- [Tool-Calling Fine-Tuning](#tool-calling-fine-tuning)
- [PPO / Full RLHF Pipeline](#ppo--full-rlhf-pipeline)
- [KTO Training (Unpaired Preferences)](#kto-training-unpaired-preferences)
- [ORPO Training (No Reference Model)](#orpo-training-no-reference-model)
- [SimPO Training (Simple Preference)](#simpo-training-simple-preference)
- [IPO Training (Regularized Preference)](#ipo-training-regularized-preference)
- [RAFT — Retrieval-Augmented Fine-Tuning](#raft--retrieval-augmented-fine-tuning)
- [RA-DIT — Retrieval-Augmented Dual Instruction Tuning](#ra-dit--retrieval-augmented-dual-instruction-tuning)
- [Curriculum-Aware Training (BETA)](#curriculum-aware-training-beta)
- [TTS Fine-Tuning (BETA, v0.52.0)](#tts-fine-tuning-beta-v0520)
- [Classifier / Reranker / Cross-Encoder Training (BETA, v0.52.0)](#classifier--reranker--cross-encoder-training-beta-v0520)
- [Knowledge Distillation (BETA, v0.52.0)](#knowledge-distillation-beta-v0520)
- [EBFT + GDPO (BETA, v0.52.0)](#ebft--gdpo-beta-v0520)
- [gpt-oss `reasoning_effort` + `train_on_eot` (v0.52.0)](#gpt-oss-reasoning_effort--train_on_eot-v0520)

---

## Loop Hardening

Six surfaces protect the training loop from the failure modes that cost a real GPU-hour. The schema + math kernels shipped in v0.70.0; the live trainer-callback wiring shipped in **v0.71.11**, validated end-to-end on SmolLM2-135M.

```bash
# Reward-hacking detector — auto-halt when the policy starts gaming the RM
# (InfoRM cluster-separation index, Wang et al. 2024 arXiv:2402.09345)
soup train --config soup.yaml \
    --reward-hack-detector info_rm --reward-hack-halt   # halt on HACK verdict

# Closed-loop reward-hacking MITIGATION (v0.71.26) — detect AND self-correct
soup train --config grpo.yaml --reward-hack-mitigation kl_control   # raise KL, recover
soup train --config grpo.yaml --reward-hack-mitigation log_only     # observe only, no action

# Cross-tokenizer distillation — Llama -> Mistral, no shared vocab needed
# (Universal Logit Distillation, Boizard et al. 2024 arXiv:2402.12030)
soup train --config soup.yaml --uld-strategy wasserstein

# Cross-tokenizer distillation for FULLY-DISJOINT tokenizers (v0.71.18) —
# aligns student/teacher token sequences over decoded character spans, so you
# can distill a GPT-2 BPE student from a Llama SentencePiece teacher.
#   training:
#     uld_strategy: wasserstein_aligned

# MiniLLM reverse-KL on-policy distillation — bundles 3 stability tricks
# (Gu et al. 2024 arXiv:2306.08543)
soup train --config soup.yaml --minillm-enabled \
    --minillm-teacher-mix-ratio 0.3 \
    --minillm-pretrain-anchor-weight 0.1 \
    --minillm-pretrain-anchor-path ./pretrain.jsonl

# MiniLLM TRUE on-policy rollout (v0.71.18, Gu et al. §3.1) — sample a fresh
# autoregressive rollout from the per-token teacher/student mixture each step,
# then length-normalised reverse-KL. training.minillm_rollout_length tunes the
# rollout (auto min(max_length, 32)).
soup train --config soup.yaml --minillm-enabled --minillm-on-policy

# Mid-epoch checkpoint for PPO/GRPO — TorchTune punts this; Soup ships it
soup train --config grpo.yaml \
    --rl-checkpoint-save-every-steps 500 \
    --rl-checkpoint-keep-last 3 \
    --rl-checkpoint-include-optimizer

# Iterative DPO loop driver — sample -> RM-score -> re-pair -> retrain
# (drop --plan-only to run the loop; --plan-only just renders the per-round plan)
soup iterative-dpo \
    --base-model meta-llama/Llama-3.1-8B \
    --reward-model ./output_rm \
    --prompts ./prompts.jsonl \
    --output-dir ./iterative_dpo_out \
    --rounds 5 \
    --pairs-per-round 1000

# RAGEN echo-trap detector — auto-halt when trajectories collapse to self-repetition
# (Zhu et al. 2025 arXiv:2504.14437)
soup train --config grpo.yaml \
    --echo-trap-enabled \
    --echo-trap-threshold 0.6 \
    --echo-trap-halt \
    --echo-trap-tokenizer-aware
```

`--echo-trap-tokenizer-aware` switches echo-trap n-grams from whitespace tokens to the active tokenizer's integer ids. This catches subword repetition that punctuation-heavy decoded text can hide, but the score becomes tokenizer-specific rather than vocabulary-agnostic.

### Closed-loop reward-hacking auto-mitigation (v0.71.26)

The detectors above *halt*; `training.reward_hack_mitigation` (or the `--reward-hack-mitigation` flag) makes the trainer *self-correct* mid-run. It requires `reward_hack_detector` on a `grpo`/`ppo` transformers run, and has four modes:

- **`log_only`** — observe only. Appends a per-step `mitigation_log.jsonl` under the run's output dir (the InfoRM/ensemble drop, the OK/WARN/HACK verdict, reward mean/std, completion-length trend, repetition) and provably never mutates β. Run this first to *see* hacking before you let a controller act on it.
- **`kl_control`** — a reversible **bang-bang + hysteresis** controller. When a multi-signal vote (`reward_hack_signals`: the detector drop + `length_trend` + `repetition`) stays above `reward_hack_trip_band` for `reward_hack_dwell_steps`, it multiplies β by `reward_hack_kl_gain` (clamped to `[reward_hack_beta_floor > 0, reward_hack_beta_ceil]`, never crossing 0); after `reward_hack_release_patience` below-band steps it relaxes β back toward the floor. Dwell + release-patience stop it flapping. β is written to **both** `trainer.beta` and `trainer.args.beta` so it takes effect on stock GRPO *and* Soup's GRPO variants (and `trainer.args.kl_coef` on PPO).
- **`pid_lagrangian`** — a **PID-Lagrangian** controller (Stooke et al. 2020) that holds the hacking signal at `reward_hack_signal_target` (Kp/Ki/Kd with integral anti-windup via `reward_hack_integral_clamp`), plus an **escalation ladder**: raise β → after `reward_hack_rollback_patience` persistent-HACK steps roll back to the last-good RL checkpoint (needs `rl_checkpoint_save_every_steps`) → after `reward_hack_max_recovery_attempts` rollbacks, early-stop with a plain-English give-up explanation.
- **Anti-gaming hardening** (any control mode): `reward_hack_signal_smoothing` (`ema`/`median` over `reward_hack_smoothing_window`), `reward_hack_conservative_on_disagreement` (when detectors disagree, keep KL high + guard against a bimodal reward-distribution collapse), and `reward_hack_reward_shaping` (subtract a bounded `reward_hack_shaping_strength` penalty on the gamed proxy — `length`/`repetition`/`sentinel` — over the reward-fn seam).

**Scope:** proof-of-mechanism only. Validated on SmolLM2-135M + a synthetic length-hacking task on a single RTX 3050 (all four modes live, including a real mid-run rollback). PPO ships **BETA** — the buffer + `kl_coef` mutation are wired and unit-tested, but the on-GPU proof is GRPO-only. Whether the loop suppresses hacking without collapsing true reward on 7B+ with a real reward model is an open, community-validatable question. `reward_hack_mitigation ∈ {kl_control, pid_lagrangian}` is mutually exclusive with `ref_model_ema_alpha` (both drive the KL/ref dynamics).

Every detector composes with v0.34 `soup why` (anomaly explainer), v0.32 spike recovery, and the v0.53.11 #127 `GRPOStabilityCallback` so a single GRPO run can have InfoRM + echo-trap + spike-recovery + in-place ref-model EMA all active simultaneously without duplicating trajectory / state collection. The reward-hack and echo-trap callbacks read the per-step rewards through a shared, thread-safe capture buffer (Soup wraps your reward functions so it never has to monkeypatch TRL); `rm_ensemble` needs ≥2 reward functions to compute a divergence. The MiniLLM teacher-mix is an offline distribution-blend analog of the paper's on-policy teacher-mixed *sampling*, and ULD compares the distributions after clamping teacher ids to the teacher vocab (correct for same-family / extended-vocab pairs; a genuinely different tokenization needs a sequence-alignment step). The reference-model EMA (`--ref-model-ema-alpha`) updates in place — no full `state_dict` round-trip — so it is cheap at 70B+ scale.


## Unlearning (`task='unlearn'`, NPO / SimNPO / RMU)

GDPR right-to-be-forgotten + CSAM/PII leak response, productized. Three method backends:

- **NPO** — Negative Preference Optimization (DPO-shaped negative-only loss; needs a reference model).
- **SimNPO** — length-normalised NPO without a ref model (faster, more stable on long sequences).
- **RMU** — Representation Misdirection Unlearning (residual-stream noise on forget inputs).

```yaml
# unlearn.yaml
base: HuggingFaceTB/SmolLM2-135M
task: unlearn
data:
  train: traces.jsonl
  forget_set: gdpr_deletion_set.jsonl   # rows to unlearn (messages / prompt+completion / text)
  retain_set: capability_anchors.jsonl  # optional — anchors general capability
training:
  unlearn_method: npo           # or simnpo / rmu
  unlearn_alpha: 0.5            # retain-set weighting [0.0, 10.0]
```

```bash
# Run the unlearn loop (validated on SmolLM2-135M — NPO/SimNPO drive forget loss down).
soup train --config unlearn.yaml --yes

# Score the run on TOFU / MUSE / WMDP (OK / MINOR / MAJOR verdict).
soup eval unlearning <run-id> --benchmark tofu --evidence evidence.json --output report.json
```

`soup train --task unlearn` is live (v0.71.9): it loads a LoRA-wrapped policy, a frozen reference copy (NPO / RMU), and the forget / retain JSONL sets, then optimises the per-method loss — NPO's `(2/β)·mean(-logσ(-β·(π_logp − ref_logp)))` drives the policy's forget-set log-prob below the reference (= forgetting), while the retain set anchors capability. Run NPO/SimNPO **with** a `retain_set` — without one the policy has no utility anchor and Soup warns loudly.

Three orthogonal axes: **Forget Quality** (pre/post forget-loss delta), **Model Utility** (retain-accuracy preserved), **PrivLeak** (membership-inference AUC distance from 0.5). Bundled mini-fixtures for all three benchmarks ship in the box (v0.71.1 added MUSE + WMDP alongside the existing TOFU set), so `--benchmark muse|wmdp` runs without supplying evidence. The WMDP forget-set probes ship **redacted** (placeholder prompts + `REFUSED` responses) — Soup never bundles verbatim hazardous-knowledge content.


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

Set `distill_mode: sequence` (default `token`) to train on the teacher's **generated
continuations** instead of per-token logit matching — a hard-label, cross-tokenizer-friendly
KD that works when student and teacher do not share a vocabulary. `sequence` mode is mutually
exclusive with the cross-tokenizer `uld_strategy` logit path (they are different objectives over
the same task; the trainer rejects the combination at setup). (v0.71.12)


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

Add a `lora:` section to train a **frozen encoder + LoRA adapter** classifier instead of the
full model — the small adapter plus the (freshly-initialised) classification head train, the
encoder backbone stays frozen:

```yaml
training:
  num_labels: 3
  lora:
    r: 16
    alpha: 32
```

(v0.71.12)


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
and the per-step labels. The reward head is saved inside the model checkpoint
(`reward_head.*` in `model.safetensors`) and the tokenizer is saved alongside it,
so the resulting directory is loadable standalone.


## PRM-guided GRPO (process-supervised RL)

Use a trained PRM as the **per-step reward** inside GRPO — the o1-era
process-supervision signal. Set `training.prm_reward` to a PRM directory (a
`task=prm` checkpoint) or HF id; the PRM splits each generated completion into
reasoning steps (newline heuristic), scores every step with its reward head, and
folds the per-step scores into one scalar reward that GRPO optimises. It
**replaces** `reward_fn` and rides the existing reward-shaping +
reward-hack-mitigation seam, so the v0.71.26 controller still observes it (TRL
logs it as `rewards/prm_reward`).

```yaml
task: grpo
backend: transformers        # required (the PRM reward runs a transformers forward)
modality: text               # required
data:
  format: chatml
  train: ./grpo_prompts.jsonl
training:
  prm_reward: ./my-prm       # a `soup train task=prm` checkpoint dir (or HF id)
  prm_aggregate: min         # weakest-link (default) | prod | last
  num_generations: 4
  grpo_beta: 0.04
```

`prm_aggregate='min'` (weakest-link, the standard PRM aggregation) is the safe
default; `prod` assumes calibrated `[0,1]` step scores (Soup's PRM head is
trained with unconstrained MSE, so `prod` can blow up on uncalibrated labels).

**Bundled rollout environments.** Three deterministic pure-Python toy
environments seed the openenv rollout path out-of-the-box — pair any of them
with `rollout_backend=openenv`:

```yaml
training:
  rollout_backend: openenv
  rollout_func: soup_cli.envs.calculator:rollout   # or retrieval_qa / guess_number
  reward_fn: verifiable
  verifiable_domain: math
```

Ready-made recipes: `grpo-env-calculator`, `grpo-env-retrieval-qa`,
`grpo-env-guess-number`. The environments are deterministic single-shot
prompt/answer *seeders* (the live openenv contract passes only the seed prompts,
not the model) — not interactive multi-turn episodes.

**Scope:** proof-of-mechanism only — validated on SmolLM2-135M with a tiny
synthetic PRM (the PRM reward scores good completions above bad and drives GRPO's
advantages). Not a production reward-model claim; scale validation is help-wanted
(#286).


## Online DPO (`task='online_dpo'`) — judge in the loop (v0.71.31)

Unlike offline DPO (static `prompt/chosen/rejected` rows), **Online DPO** generates
two completions per prompt **on-policy** each step and asks a *judge* — or a
*reward model* — which is better; the winner becomes `chosen`, the loser
`rejected`. The judge closes the loop. Wraps TRL `OnlineDPOTrainer`; data is
prompt-only (like GRPO). Transformers + text only.

```yaml
base: HuggingFaceTB/SmolLM2-135M-Instruct
task: online_dpo
data:
  train: ./data/prompts.jsonl      # prompt-only (or any format — prompts are extracted)
training:
  online_dpo_judge: "ollama://llama3.1"   # a pairwise judge (ollama://|https://|http://localhost)
  # OR: reward_model: ./my-reward-model   # exactly one of judge / reward_model
  online_dpo_loss_type: sigmoid           # sigmoid | ipo
  online_dpo_max_new_tokens: 64
  dpo_beta: 0.1
  lora: { r: 8, alpha: 16, target_modules: auto }
```

The judge is Soup's own OpenAI-compatible `JudgeEvaluator` adapted to TRL's
`BasePairwiseJudge` (swap-debiased: a winner is only recorded when both A,B and
B,A orders agree). Recipe: `online-dpo-smollm2-135m`. Proof-of-mechanism was
validated on SmolLM2-135M with a synthetic judge (not a production RLHF claim; #286).


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
  vllm_sleep_mode: true               # between-rollouts vLLM standby — LIVE (vLLM >= 0.7)
  # Multi-turn agent rollout — openenv is LIVE: your function's rows replace the prompt dataset
  rollout_backend: openenv            # one of: art / ruler / nemo_gym / openenv
  rollout_func: my_module:my_rollout  # module:function resolver (openenv; trusted operator code)
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


## RAFT — Retrieval-Augmented Fine-Tuning

When you need a model to *cite* the document it's reading instead of hallucinating, RAFT (Stanford 2024) is the canonical recipe. Each training row carries a query, a golden document, a list of distractor documents, and the answer — the model learns to attend to the relevant doc while ignoring the noise.

```yaml
# soup.yaml
data:
  train: ./data/raft.jsonl
  format: raft

training:
  citation_faithful: true        # enable citation precision/recall scoring
  citation_style: bracket        # cite as [doc-1] inline
  citation_recall_threshold: 0.8 # gate final save on recall >= 80%
```

```jsonl
# RAFT JSONL row shape
{"query": "When was Python released?", "golden_doc": "Python was released in 1991 by Guido van Rossum.", "distractor_docs": ["Ruby was released in 1995.", "Java was released in 1995."], "answer": "1991 [doc-1]"}
```

```bash
# Ready-made 8B Llama recipe
soup recipes show raft-llama3-8b
soup recipes use raft-llama3-8b
```

Citation scoring is exposed as a pure kernel for the eval gate:

```python
from soup_cli.utils.citation_faithful import score_citations

score = score_citations(
    predicted="The answer is 1991 [doc-1].",
    expected_ids=("doc-1",),
)
# CitationScore(precision=1.0, recall=1.0, f1=1.0, predicted_count=1, expected_count=1)
```

Citation-faithful FT is gated to `task in {sft, pretrain}` + `data.format='raft'` — misconfigured runs fail at config load with a named-field message.

Under the hood, a `format: raft` run trains **answer-only**: each row is composed into a prompt (golden + distractor docs, shuffled deterministically by `data.raft_shuffle_seed`, each labelled `[doc-N]`) followed by the answer; the prompt span is masked out of the loss and — when `citation_faithful: true` — the bracketed `[doc-id]` spans in the answer get a boosted per-token loss weight. Rows whose prompt fills `max_length` (answer fully truncated) are dropped with a warning rather than silently shrinking the dataset.

By default the document order is fixed for the whole run. Set `data.raft_epoch_shuffle: true` to **re-permute** the golden + distractor documents *each epoch* (a per-epoch salt folded into the shuffle seed) so the model can't latch onto a fixed citation slot — useful for multi-epoch runs. `epoch=0` reproduces the legacy single-permutation order exactly, so enabling it never changes the first epoch. (v0.71.17)

Score a trained model's citations from the CLI:

```bash
# {predicted, expected_ids} rows, OR RAFT rows scored against their own golden [doc-N]
soup eval citation preds.jsonl --style bracket
# RAFT rows: pass the train-time shuffle seed so the golden id lines up
soup eval citation raft.jsonl --shuffle-seed 0 --output citation.json
```

`soup diagnose` also gains a `citation` failure mode that flags a model that stopped citing the supporting document.


## RA-DIT — Retrieval-Augmented Dual Instruction Tuning

RA-DIT (Meta 2023) is the two-stage version of RAFT: first train a sentence-transformer retriever (contrastive), then fine-tune the generator on the RAFT-style rows. Two recipes ship paired:

```bash
# Stage 1 — train the retriever (uses Soup's v0.16 embedding trainer)
soup recipes use ra-dit-retriever
soup train

# Stage 2 — train the generator on RAFT data, pointing at the retriever
soup recipes use ra-dit-llama3-8b
soup train
```

The schema enforces stage-task pairing — `ra_dit_stage: retriever` requires `task: embedding`; `ra_dit_stage: generator` requires `task: sft`. A misconfigured recipe fails at config load with a named-field message.

Run both stages in one command with `soup ra-dit`:

```bash
soup ra-dit --retriever-config retriever.yaml --generator-config generator.yaml
# preview the plan + the resolved retriever link without training:
soup ra-dit -r retriever.yaml -g generator.yaml --plan-only
```

It trains the retriever, then **records** that trained retriever as the generator's paired retriever (writing its output dir into the generator's `training.ra_dit_retriever_model`) and trains the generator RAFT-style. The recorded retriever is the one used at deploy/serve time — stage-2 does not fuse the retriever weights. A plain `soup train` of a generator-stage config with no retriever model set **auto-links** the most-recent RA-DIT retriever run from the Registry; pass `--retriever-model <m>` to override.


## Curriculum-Aware Training (BETA)

Layer dynamic re-weighting on top of the static `curriculum` bucketer. Every N steps the trainer aggregates per-sample loss + grad-norm into a per-bucket uncertainty signal, runs it through a softmax (temperature-controlled) with floor (water-filling so no bucket drops below `curriculum_dynamic_floor`), and re-weights the sampler. Empty buckets fall back to the median of populated buckets; degenerate inputs return uniform.

```yaml
training:
  curriculum: true                          # static bucketer (v0.23.0)
  curriculum_buckets: 4
  curriculum_metric: perplexity             # length (default) | loss | perplexity
  curriculum_dynamic: true                  # NEW — dynamic re-weighting
  curriculum_dynamic_recompute_steps: 50    # refresh every 50 global steps
  curriculum_dynamic_floor: 0.05            # min weight per bucket
  curriculum_dynamic_temperature: 1.0       # softmax temp on uncertainty
```

**Bucketing by difficulty percentile (v0.71.5).** When `curriculum_metric` is `loss` or `perplexity`, the dynamic callback assigns each step's sample to a bucket by its *rank* within a rolling 512-step window of the difficulty signal (perplexity = `exp(min(loss, 50))`), instead of the round-robin fallback used for `length`. This keeps the buckets calibrated to the live loss distribution rather than a static length sort. `length` (the default) keeps the round-robin assignment.

Visualise the recorded bucket-weight evolution with `soup runs curriculum-curve <run_id>`.

DDP / grad-accum safety: multi-rank launches must wire an `all_reduce` hook on per-bucket stats (a cross-validator rejects un-coordinated multi-rank runs upfront). Multi-trainer expansion beyond `sft` / `pretrain` is tracked for v0.48.1.


## TTS Fine-Tuning (`task='tts'`, BETA, live in v0.71.20)

Live as of v0.71.20 (lifted from the v0.52.0 schema stub). The five families
(`orpheus`, `sesame_csm`, `llasa`, `spark`, `oute`) are all decoder language
models, so a TTS fine-tune is **next-token cross-entropy over interleaved
`[text][audio-codec-token]` chat sequences** — the same objective the SFT
trainer already runs. `TTSTrainerWrapper` reuses the SFT model/tokenizer/LoRA/CE
machinery and adds two TTS-specific pieces: per-family emotion-control
templating and registration of operator-supplied codec special tokens.

There are two workflows:

**Pre-encoded chat (live, validated).** Run the family's audio codec **offline**
so the assistant turn already contains the discrete codec-token string, then
train with `data.format: chat`. This is plain cross-entropy and runs on any GPU
(validated end-to-end on SmolLM2-135M-Instruct).

```yaml
base: HuggingFaceTB/SmolLM2-135M-Instruct   # or canopylabs/orpheus-3b-0.1-ft
task: tts
modality: audio_out
data:
  train: ./data/tts_pre_encoded.jsonl   # assistant turns carry codec tokens
  format: chat
  new_special_tokens: ["<|codec_0|>", "<|codec_1|>"]   # your codec vocab
training:
  tts_family: orpheus
  tts_emotion: neutral   # Orpheus + Oute only
  lora: true
```

Operator-supplied `data.new_special_tokens` are registered (deduplicated, only
tokens not already in the vocab) and the embedding matrix is resized through the
(possibly PEFT-wrapped) model so the codec-token ids have rows. Orpheus + Oute
support emotion conditioning via `training.tts_emotion` from a per-family
allowlist (Orpheus: neutral / happy / sad / angry / excited / calm / whisper /
laugh; Oute: neutral / happy / sad / angry / calm / excited) — the wrapper
prepends the family's emotion control string to the first user turn.

**Live-codec (hardware/dependency-gated).** Setting `data.format: audio` asks
the trainer to encode raw audio into codec tokens **at train time**, which needs
the family's heavyweight codec package (`snac` for Orpheus, `moshi` for
Sesame-CSM, `xcodec2` for Llasa, `sparktts` for Spark, `outetts` for Oute). The
**Orpheus** path is live — install `pip install snac` and a 24 kHz mono wav is
encoded to SNAC codec tokens end-to-end (audio is duration- and byte-capped and
read through an `O_NOFOLLOW` fd). The other four families still surface a
friendly per-family `RuntimeError` naming the required `pip install` and are not
yet validated on the maintainer's box — use the pre-encoded workflow above for a
runnable fine-tune with those.

Five ready-made recipes ship: `orpheus-tts-sft`, `sesame-csm-tts`, `llasa-tts`,
`spark-tts`, `oute-tts` — copy with `soup recipes use <name>`. Cross-validators
reject the `mlx` backend, `modality != audio_out`, and emotion tags outside the
per-family allowlist.


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


## EBFT + GDPO (BETA, v0.52.0)

Energy-Based Fine-Tuning (axolotl) lands as `training.ebft_variant ∈ {structured, strided}` + `training.ebft_temperature` (bounded `[1e-4, 100.0]`). Gated to `task: sft`. Generalized DPO lands as `training.gdpo_variant ∈ {standard, length_normalized, margin}` — gated to `task ∈ {dpo, preference}`. Live loss kernels in v0.52.1.


## gpt-oss `reasoning_effort` + `train_on_eot` (v0.52.0)

`training.reasoning_effort: low | medium | high` injects a system-prefix token at training time for gpt-oss models; `training.train_on_eot: true` includes explicit EOT/EOS control tokens in the SFT loss (axolotl `train_on_eot`). Both are gated to the SFT-family task set (`sft` / `pretrain` / `distill` / `classifier` / `reranker` / `cross_encoder`) — setting them on DPO / GRPO / PPO / etc. fails at config load. Live formatter wiring in v0.52.1.


## MoLE — Per-Token Adapter Routing (`task='moe_lora_routing'`)

Train a small **gating network** that routes each token to a weighted blend of N frozen task
LoRAs (Mixture of LoRA Experts, Wu et al. 2024). The base model and every task adapter stay
frozen — only the router learns which adapter(s) each token should use.

```yaml
base: HuggingFaceTB/SmolLM2-135M
task: moe_lora_routing
modality: text
backend: transformers

data:
  train: ./data/chat.jsonl
  max_length: 512

training:
  mole_task_adapters:        # 2-64 LoRA adapter paths (HF ids or local dirs)
    - ./adapters/math
    - ./adapters/code
    - ./adapters/chat
  mole_top_k: 2              # 1 <= top_k <= len(mole_task_adapters)
  mole_temperature: 1.0      # [1e-6, 100.0]
  epochs: 1
```

The gate is the only trainable parameter; it is saved as `mole_gate.pt` alongside the run.
`compute_loss` runs N+1 forwards per step (base + each adapter under `torch.no_grad()`, blended
by the per-token gate weights) so step time scales with the number of task adapters. Training
only — there is no serve-time MoLE path yet. (v0.71.12)


## Architecture Knobs — Mixture-of-Depths, LLaMA Pro, LongLoRA

Three architecture transforms that were schema-only are now live for SFT / Pretrain on
Llama / Qwen / Mistral (LongLoRA also covers Phi). All apply at trainer setup:

```yaml
training:
  # Mixture-of-Depths (arXiv 2404.02258): route only the top-k tokens through each
  # block. capacity_factor is the fraction of tokens that get the residual update.
  use_mod: true
  mod_capacity_factor: 0.125

  # LLaMA Pro: append zero-initialised identity decoder blocks and train only the new
  # ones (freeze_trainable_layers freezes the originals).
  expand_layers: 4
  freeze_trainable_layers: 4

  # LongLoRA S²: shifted-sparse attention on the Q/K projections for long-context tuning.
  use_longlora: true
```

`use_mod` / `expand_layers` attach AFTER `get_peft_model` so the new routers / blocks are
trainable. Unsupported architectures warn + skip (MoD, block expansion); `use_longlora` is
rejected at the schema gate for non-supported arches and for `use_ring_attention` / FlashAttention-3.
Pick one of MoD / LLaMA Pro / LongLoRA per run. (v0.71.12)


## Spectrum — Targeted Training on Layer SNR (`soup spectrum scan`, v0.71.23)

Spectrum (arXiv:2406.06623) fine-tunes only the layers with the most signal. `soup spectrum scan`
streams a model's `.safetensors` shards **one tensor at a time** — there is no model load, so it
runs on a CPU box even for very large models — and computes a singular-value signal-to-noise ratio
per weight matrix with a Marchenko-Pastur noise threshold. It ranks the layers within each
module-type group and prints the top `--top-percent` as a ready-to-paste config block:

```bash
soup spectrum scan --model HuggingFaceTB/SmolLM2-135M --top-percent 25 --modules mlp,attn -o patch.yaml
```

```yaml
# patch.yaml — paste into your soup.yaml
training:
  unfrozen_parameters:
  - model.layers.0.mlp.down_proj
  - model.layers.29.self_attn.v_proj
  # ...
```

Then train with the patch — the SFT trainer freezes **every** parameter and unfreezes only the
matched set (full fine-tuning, LoRA off):

```yaml
base: HuggingFaceTB/SmolLM2-135M
task: sft
training:
  quantization: none        # Spectrum trains float weights — quantization off
  unfrozen_parameters:
  - model.layers.0.mlp.down_proj
  - model.layers.29.self_attn.v_proj
```

`unfrozen_parameters` entries are regex patterns matched against parameter names. It requires
`task: sft`, `backend: transformers`, `modality: text`, and `quantization: none`, and is mutually
exclusive with LoRA features (`use_dora` / `use_vera` / `moe_lora` / `relora_steps` / …) and the
other freezing knobs (`freeze_layers` / `freeze_ratio` / `train_router_only` / `expand_layers`) —
a conflicting combo is rejected loudly at config load. Scans cache under `~/.soup/spectrum/`
(override with `SOUP_SPECTRUM_CACHE_DIR`); `--no-cache` skips it. `--modules mlp,attn` (vs the
`all` default) is recommended for very large models — it skips the giant embedding/lm_head matrices.
The SNR kernel is pure-numpy and transpose-invariant, so GPT-2 `Conv1D` weights score the same as
Linear weights. (v0.71.23)


