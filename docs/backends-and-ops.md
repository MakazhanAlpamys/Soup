# Backends, Platform & Ops

[← Back to the Soup README](../README.md)

> MLX/Unsloth backends, alternative hubs, HF Hub integration, autopilot, experiment tracking, plan/apply, env lockfiles, hardware-fit, shell completions, the plugin system, and the standalone utility commands.

**Contents:**

- [Autopilot (Zero-Config)](#autopilot-zero-config)
- [Apple Silicon (MLX Backend)](#apple-silicon-mlx-backend)
- [Unsloth Backend (2-5x Faster Training)](#unsloth-backend-2-5x-faster-training)
- [Chat with your model](#chat-with-your-model)
- [Push to HuggingFace](#push-to-huggingface)
- [HuggingFace Hub Deep Integration](#huggingface-hub-deep-integration)
- [Resume Training](#resume-training)
- [Run Management & Cleanup](#run-management--cleanup)
- [Alternative Model Hubs](#alternative-model-hubs)
- [TensorBoard Integration](#tensorboard-integration)
- [Weights & Biases Integration](#weights--biases-integration)
- [Ready-Made Recipes](#ready-made-recipes)
- [Hyperparameter Sweep](#hyperparameter-sweep)
- [Model Comparison](#model-comparison)
- [Quickstart Demo](#quickstart-demo)
- [Health Check](#health-check)
- [Version Info](#version-info)
- [Error Handling](#error-handling)
- [Unknown config keys](#unknown-config-keys)
- [Config size limits](#config-size-limits)
- [External programs](#external-programs)
- [Experiment Tracking](#experiment-tracking)
- [Profiling Extras](#profiling-extras)
- [VS Code Setup (`.vscode/launch.json`)](#vs-code-setup-vscodelaunchjson)
- [Observability & Dev UX](#observability--dev-ux)
- [GPU Live Monitor](#gpu-live-monitor)
- [Soup Fetch — Bundled Examples](#soup-fetch--bundled-examples)
- [Llama 4 Delinearizer](#llama-4-delinearizer)
- [Ctrl+C Graceful Save](#ctrlc-graceful-save)
- [Checkpoint-Now Trigger File](#checkpoint-now-trigger-file)
- [Onboarding Wizard Helper](#onboarding-wizard-helper)
- [Standalone Sweep Config](#standalone-sweep-config)
- [Alternative Model Hubs (ModelScope / Modelers)](#alternative-model-hubs-modelscope--modelers)
- [Experiment Trackers (MLflow / SwanLab / Trackio)](#experiment-trackers-mlflow--swanlab--trackio)
- [Telemetry (opt-in)](#telemetry-opt-in)
- [Plugin System](#plugin-system)
- [External Integrations Catalog](#external-integrations-catalog)
- [Advanced Trainer Plugins](#advanced-trainer-plugins)
- [Soup Plugin Callbacks](#soup-plugin-callbacks)
- [Terraform-Style Plan & Apply (`soup plan` / `soup apply`)](#terraform-style-plan--apply-soup-plan--soup-apply)
- [Hermetic Env Lockfile (`soup env`)](#hermetic-env-lockfile-soup-env)
- [Hardware-Fit Calculator](#hardware-fit-calculator)
- [Shell Completions (`soup completions`)](#shell-completions-soup-completions)
- [License Advisor (`soup license-advisor`)](#license-advisor-soup-license-advisor)

---

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

For the goals that train with SFT, Autopilot turns on FlashAttention and Liger only when the GPU has compute capability 8.0 or higher and the package is installed. When the card qualifies but a package is missing, the Autopilot Decisions panel names the package and the command that installs it.


## Apple Silicon (MLX Backend)

Fine-tune on M1-M4 Macs via Apple's [MLX](https://github.com/ml-explore/mlx) framework — no CUDA, no emulation.

```bash
# Install MLX support
pip install "soup-cli[mlx]"
```

For SFT with local JSONL, JSON, or CSV data, `[mlx]` is sufficient on its own.
It can also be installed with `[train]` when the same environment needs the
PyTorch/TRL Transformers backend: both extras now share the supported
`transformers>=5.16.1,<6` range. A Hugging Face `datasets` source or
streaming dataset still needs `datasets` because that data source owns the
dependency; the local file path below does not.

`detect_device()` and `get_gpu_info()` recognise Apple Silicon when
`backend: mlx` is set, preserving `training.quantization: 4bit` for
`mlx-community` pre-quantized checkpoints instead of silently downgrading to
`none` ([#423](https://github.com/MakazhanAlpamys/Soup/issues/423)). The
CUDA-shaped analytical VRAM preflight is skipped on the MLX path because Apple
unified memory is managed by Metal, not a fixed CUDA VRAM pool.

mlx-lm trains and evaluates in whole batches of `training.batch_size`
([#1851](https://github.com/MakazhanAlpamys/Soup/issues/1851)). A validation
split with fewer rows than one batch, which the default `data.val_split: 0.1`
produces on a small file, is not evaluated: the run warns, naming both numbers,
and records no `val_loss`. Lower `batch_size`, raise `val_split`, or set
`val_split: 0` to train on those rows instead. A training split with fewer rows
than one batch is refused before the model loads.

`training.eval_steps: N` on MLX SFT evaluates after every N completed optimizer
updates and after the last one, over every validation row including a short
tail, and is recorded in `adapter_config.json` as `eval_steps` / `eval_unit`
([#1990](https://github.com/MakazhanAlpamys/Soup/issues/1990)); with it set, a
split smaller than one batch is evaluated rather than skipped. The validation
loss uses the training loss: for plain-text rows that is mlx-lm's
`default_loss`, which also scores the first padding token after each row.
Soup triggers each evaluation when mlx-lm asks for the next training batch,
which mlx-lm 0.31 and 0.32 do only after the previous step; if a release reads
batches ahead, the run says once that the evaluation ran before its update.

MLX text SFT also supports the opt-in [loss watchdog](peft-and-efficiency.md#loss-watchdog)
with the existing threshold and patience fields. Its reports count completed
optimizer updates; a triggering report stops further updates and saves the last
adapter with the actual completed work and termination reason in
`adapter_config.json`. Recovery and watchdogs on preference tasks remain refused.

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

MLX backend supports SFT, DPO, ORPO, SimPO, KTO and GRPO. Any other task with `backend: mlx` (`ipo`, `preference`, ...) is refused when the config is loaded, with an error naming the task. Requires `mlx-lm >= 0.31.3`. Use `soup recipes search mlx` for ready-made Apple Silicon configs.

`training.lora.rank_pattern` and `training.lora.alpha_pattern` are refused on `backend: mlx` for SFT, DPO, ORPO, SimPO, KTO and GRPO ([#1996](https://github.com/MakazhanAlpamys/Soup/issues/1996)): the MLX trainer builds every adapter from the one scalar `lora.r` and `lora.alpha`, so per-module ranks and alphas would be dropped without a word. Remove them, or use `backend: transformers`, which applies them. An empty `{}` sets nothing and still loads.

#### DPO on MLX

`task: dpo` with `backend: mlx` runs Soup's own training loop (`mlx-lm` ships no DPO helper), with the sigmoid loss TRL uses by default and `training.dpo_beta`.

```yaml
base: mlx-community/Qwen2.5-0.5B-Instruct-4bit
task: dpo
backend: mlx

data:
  train: ./prefs.jsonl
  format: dpo
  max_length: 1024

training:
  lr: 2e-5
  dpo_beta: 0.1
  batch_size: 2
  gradient_accumulation_steps: 4
  gradient_checkpointing: true
  seed: 42
  lora:
    r: 16
    alpha: 32
```

- **No second model in memory.** LoRA `B` starts at zero, so before the first update the policy *is* the base model. Reference log-probs are computed once from it before training, which is what TRL's `precompute_ref_log_probs` does. `--resume` computes them before the checkpoint loads, so the reference is always the base model.
- **The rows `load_dataset` gives every backend.** Each row is read by the loader's own `format: dpo` converter, so MLX accepts exactly the layouts the transformers backend does: text (three strings) or conversational (a message-list prompt, or full `chosen` / `rejected` conversations that repeat the user turn, as the bundled examples do). Text rows get EOS appended; conversational rows go through the model's chat template; the prompt is capped at `max_length // 2` and completions at what remains, as on TRL 0.29. Where this differs from TRL, it does on purpose: a pair whose `chosen` equals its `rejected` is skipped with a count (TRL trains it at a constant loss and zero gradient), and a row the chat template is not prefix-stable for is skipped with a count and the first row's index (TRL warns and slices anyway). Fewer pairs than `training.batch_size` after skipping is refused, naming both.
- **Seeded.** `training.seed` seeds `mx.random` (LoRA init, dropout) and the batch order; `training.data_seed`, when set, owns the batch order instead. Two runs with the same seed reproduce each other's losses.
- **LoRA dropout applies.** `lora.dropout` (default 0.05) is active during training, as in MLX SFT; TRL's `disable_dropout` (on by default in 0.29.1) turns it off on the transformers backend. Set `lora.dropout: 0` to match it.
- **Memory.** Log-probs are computed from the first completion position on, with the vocabulary head run in checkpointed chunks, so the full-vocabulary logits of a batch are never held at once. The chunked path is checked against the full model on the first batch and dropped, with a warning, for an architecture that changes logits after the head. Set `training.gradient_checkpointing: true` for long sequences: measured once on an M4 Pro, Qwen2.5-0.5B-4bit with 792-token pairs, it took a training step from 10.05 GB to 3.63 GB.
- **Learning rate.** DPO is more sensitive than SFT. In one run on UltraFeedback pairs with Qwen2.5-0.5B-4bit, LoRA r=16 on attention, `lr: 5e-5` diverged to NaN within a dozen updates while `lr: 2e-5` trained; start at `2e-5` or below.
- **Progress.** The panel and `soup ui` show loss and validation loss. Reward accuracy and margin print per logged step when there is no live panel, and are returned in the run result; the reference precompute prints its progress before the panel starts.
- **Refused at load:** `modality: vision` / `audio` (MLX DPO trains on text and chat rows) and `training.lora.r: 0` (the reference is the untrained LoRA policy, so an adapter must exist), plus `dpo_beta_schedule` / `dpo_ref_regen_epochs`, LoRA+ and `data.chat_template` (MLX renders the model's shipped template only; use `backend: transformers` to apply an override).
- **Not applied on MLX:** per-row `tools` / `chat_template_kwargs` columns (TRL reads them; MLX DPO renders with the model's own template and nothing else), NEFTune, Liger, FlashAttention and the other CUDA-path options MLX SFT also warns about.

#### ORPO on MLX

`task: orpo` with `backend: mlx` runs the same loop as MLX DPO with ORPO's
objective, matched to TRL 0.29.1's ORPO trainer (the one the transformers
backend runs): the mean log-probability of each completion, the stable
odds-ratio term `log_odds = (a - b) - (log(1 - e^a) - log(1 - e^b))`, and
`loss = chosen NLL - orpo_beta * mean(logsigmoid(log_odds))`, where the NLL is a
token mean over the whole chosen sequence, prompt included. There is no
reference model and nothing is precomputed. Qualified on
Qwen2.5-0.5B-Instruct in bf16 and 4-bit on an M4 Pro.

```yaml
base: mlx-community/Qwen2.5-0.5B-Instruct-4bit
task: orpo
backend: mlx

data:
  train: ./prefs.jsonl
  format: dpo
  max_length: 1024

training:
  lr: 5e-5
  orpo_beta: 0.1
  batch_size: 2
  gradient_accumulation_steps: 4
  gradient_checkpointing: true
  quantization: 4bit
  lora:
    r: 16
    alpha: 32
```

- **Tokens.** Rows are read by the `format: dpo` converter and tokenized as
  TRL's ORPO trainer does: rendered to text with the chat template (keeping
  only the prefix the prompt and `prompt + answer` renderings share), the
  answer located even when the tokenizer merges the last prompt token into it,
  BOS prepended and EOS appended when missing. Over-long rows are cut as on the
  transformers backend (prompt to `max_length // 2`, keeping its end, answers to
  the remaining room); TRL's own earlier answer cut, which can leave the shorter
  answer empty, is not reproduced. A pair the template cannot split is skipped
  with a count; a row with no room for an answer is refused.
- **Ties train.** A pair whose `chosen` equals its `rejected` is kept, as in
  TRL: its odds-ratio term is the constant `log 2` and the NLL still learns.
- **Stops on a nonfinite loss** instead of training on NaN. The check reads
  the loss, not the gradient: a completion whose mean log-probability is
  exactly 0 gives a finite loss and a NaN gradient, as in TRL.
- **LoRA dropout applies**, as in MLX DPO above; set `lora.dropout: 0` to
  match TRL's `disable_dropout`.
- **Reported:** loss, NLL, log-odds ratio, mean log-odds, reward accuracy and
  margin (`orpo_beta * logp`, as TRL logs them), per logged step and in the run
  result (`val_*` for the validation split).
- **Saved:** `adapters.safetensors` and `adapter_config.json` (with `task:
  orpo`, `orpo_beta` and the base model), loadable with
  `mlx_lm.load(base, adapter_path=output)`, both written atomically and the
  metadata before the first step. A run that stops before its first save
  leaves the new `adapter_config.json` beside the previous run's adapter
  file. `--resume` refuses a checkpoint without that
  metadata, from another task, or from another base, before any tensor loads.
- **Refused at load:** `quantization: 8bit` (use `none`, or a pre-quantized
  4-bit base with `quantization: 4bit`), `training.lora.r: 0`, non-text
  `modality`, LoRA+, and `task: preference` with `preference_loss: orpo` on
  `backend: mlx` (use `task: orpo`). The other settings MLX DPO does not apply
  are not applied here either.

#### SimPO on MLX

`task: simpo` with `backend: mlx` runs the same loop as MLX DPO with SimPO's
objective, matched to TRL 0.29.1's CPO trainer with `loss_type: simpo` (the one
the transformers backend runs). With `a` and `b` the mean log-probability of the
chosen and rejected completion tokens:

`loss = mean(-logsigmoid(beta * (a - b) - simpo_gamma)) + cpo_alpha * NLL`

Qualified on Qwen2.5-0.5B-Instruct in bf16 and 4-bit on an M4 Pro: one epoch on
600 UltraFeedback pairs took about 4.5 minutes and 7.5-8.6 GB, and the mean
log-probability margin on 200 held-out pairs rose 1.9x (bf16) and 3.1x (4-bit),
while held-out pairwise accuracy barely moved (0.540 to 0.565, 0.545 to 0.550).
That shows the objective moving, not alignment quality.

- **Beta is 0.1, fixed.** Soup's transformers backend passes no beta to TRL, so
  `CPOConfig`'s default 0.1 applies; MLX uses the same value and records it as
  `simpo_beta` in the adapter metadata. `dpo_beta` and `orpo_beta` are not
  read. The margin `simpo_gamma` sits outside the temperature, as in TRL (which
  writes it `beta * (a - b - simpo_gamma / beta)`). Label smoothing is TRL's
  default 0.
- **The NLL covers the chosen completion only.** CPO's labels mask the prompt,
  so the NLL is a token mean over the chosen completion tokens of the whole
  batch (not a mean of per-row means). This differs from ORPO, whose NLL
  includes the prompt. `cpo_alpha` weights it (`> 0`, as on the transformers
  backend). With `gradient_accumulation_steps > 1` the NLL is a token mean per
  micro-batch and the micro-batches are averaged with equal weight, as the
  transformers Trainer does for TRL's CPO.
- **Tokens** are ORPO's above: TRL's CPO and ORPO trainers tokenize rows with
  the same code, so rows that fit in `data.max_length` train on the same tokens
  on either backend. Over-long rows are cut with Soup's cap (prompt to
  `max_length // 2`, keeping its end, answers to the remaining room). The
  transformers SimPO path applies no prompt cap: TRL 0.29.1 cuts only the
  answers there, and Soup refuses a row whose shorter answer that cut empties.
- **Ties train**, as in TRL: the preference term is the constant
  `softplus(simpo_gamma)` and the NLL still learns.
- **Stops on a nonfinite loss** instead of training on NaN.
- **LoRA dropout applies**, as in MLX DPO above; set `lora.dropout: 0` to
  match TRL's `disable_dropout`.
- **Reported:** the panel and `soup ui` show loss and validation loss. Without a
  live panel each logged step prints loss, NLL, reward accuracy and margin. The
  run result has, for the validation split, `val_preference_loss` (the SimPO
  term without the NLL, which is not on the per-step line), `val_nll_loss`, the
  chosen and rejected rewards (`0.1 * logp`, as TRL logs them), reward accuracy
  and margin, and the mean chosen and rejected log-probabilities. TRL's
  mean-logit metrics are not computed: the vocabulary head runs in chunks and
  the full logits are never held.
- **Saved:** `adapters.safetensors` and `adapter_config.json` (with `task:
  simpo`, `simpo_beta`, `simpo_gamma`, `cpo_alpha` and the base model),
  loadable with `mlx_lm.load(base, adapter_path=output)`, written as MLX ORPO
  writes them. `--resume` refuses a checkpoint without that metadata, from
  another task (ORPO and DPO included), or from another base, before any tensor
  loads.
- **Refused at load:** `quantization: 8bit`, `training.lora.r: 0`, non-text
  `modality`, LoRA+, `data.chat_template` (as for MLX DPO and ORPO), and
  `task: preference` with `preference_loss: simpo` on
  `backend: mlx` (use `task: simpo`). The other settings MLX DPO does not apply
  are not applied here either.

#### KTO on MLX

`task: kto` with `backend: mlx` runs the same loop and reference precompute as
MLX DPO with KTO's objective, matched to TRL 0.29.1's KTO trainer (the one the
transformers backend runs, with `training.kto_beta` as `beta` and every other
`KTOConfig` setting at its default):

- `r` is the policy minus the reference **summed** log-probability of the
  completion;
- `z = max(0, mean(policy_KL - reference_KL))` over the batch's KL rows,
  without gradient;
- a desirable row (`label: true`) loses `1 - sigmoid(kto_beta * (r - z))`, an
  undesirable one `1 - sigmoid(kto_beta * (z - r))`, each weighted 1.0
  (`desirable_weight` / `undesirable_weight` are not exposed), and the loss is
  the mean over every row of the batch. A batch with only one label trains on
  that label's rows alone.

Qualified on Qwen2.5-0.5B-Instruct in bf16 and 4-bit on an M4 Pro: one epoch on
1,200 unpaired UltraFeedback rows (`batch_size: 4`, accumulation 2) took about
9 minutes and 8.0-9.3 GB, and the KTO loss on 400 held-out rows fell from 0.500
to 0.448 (bf16) and 0.456 (4-bit). The desirable term fell while the undesirable
one grew, the usual one-epoch KTO shape; that shows the objective moving, not
alignment quality.

```yaml
base: mlx-community/Qwen2.5-0.5B-Instruct-4bit
task: kto
backend: mlx

data:
  train: ./kto.jsonl
  format: kto
  max_length: 1024

training:
  lr: 2e-5
  kto_beta: 0.1
  batch_size: 4
  gradient_accumulation_steps: 4
  gradient_checkpointing: true
  lora:
    r: 16
    alpha: 32
    dropout: 0
```

- **KL rows, as TRL builds them.** Each row's KL row is its own prompt with the
  completion of the previous row in its group of `batch_size` consecutive rows,
  the first row taking the last one's; a shorter last group rotates within
  itself, and a last group of one row is its own KL row. Batches run in
  dataset order, the last partial batch included, because TRL's KTO samples
  sequentially: each training batch is one KL group, so its `z` is estimated
  from the same completions it scores. With `gradient_accumulation_steps`
  above 1 the plan is cut to whole accumulation windows, as in MLX DPO and
  ORPO; in dataset order that always leaves the same final batches untrained
  (TRL instead updates on the partial window at each epoch's end), and the run
  warns before training, naming those batches and their rows. `training.seed` / `data_seed` do not
  reorder rows (on either backend); shuffle the file to mix labels.
- **Reference, precomputed.** The reference log-probs of every row and of its KL
  row come from the base model before the first update, as in MLX DPO
  (`--resume` included); the policy's KL log-probs are computed live, without
  gradient, at every step.
- **Tokens.** Rows are read by the `format: kto` converter and tokenized as
  TRL's KTO trainer does: rendered to text with the chat template (keeping the
  prefix the prompt and `prompt + completion` renderings share), the
  completion located even when the tokenizer merges the last prompt token
  into it, BOS prepended to the prompt and EOS appended to the completion when
  missing. A row longer than `data.max_length` loses the end of its completion;
  the prompt is never cut (TRL 0.29.1 has no prompt cap, and the transformers
  path adds none). A KL row is re-cut to its own prompt's room.
- **Labels** are read by the loader's `format: kto` converter, exactly as on the
  transformers backend (`true` / `false`, and the strings and numbers that
  converter accepts); MLX adds no reading of its own. Only a row handed to the
  trainer without the loader is checked for a real `true` / `false`.
- **Refused before any update:** a row with no `label`; an empty completion;
  and a row whose prompt leaves no room for one completion token in
  `data.max_length` (TRL keeps such a prompt whole and trains it past
  `max_length`). A refusal names the split and row index.
- **Batch size at least 2**, as on the transformers backend: `batch_size: 1` is
  refused at load and `batch_size: auto` runs at 2. A training split smaller
  than `batch_size` trains as one partial batch and one KL group, as in TRL;
  fewer than 2 rows is refused, because a row's KL completion is another
  row's.
- **Stops on a nonfinite loss** instead of training on it (TRL's `nanmean`
  drops a NaN row silently).
- **LoRA dropout applies**, as in MLX DPO above; set `lora.dropout: 0` to match
  TRL's `disable_dropout`. With dropout on, the KL forward draws its own mask.
- **Reported:** the panel and `soup ui` show loss and validation loss. Without a
  live panel each logged step also prints `kl`, `rewards/chosen` and
  `rewards/rejected` (`kto_beta * r` averaged per label, as TRL logs them;
  `n/a` for a label the step did not see). The run result has `val_kl`,
  `val_rewards_chosen`, `val_rewards_rejected`, `val_rewards_margins`,
  `val_logps_chosen` and `val_logps_rejected` for the validation split, which
  is scored in `batch_size` chunks in order. TRL's `logits/*` and `count/*`
  metrics are not reported.
- **Saved:** `adapters.safetensors` and `adapter_config.json` (with `task: kto`,
  `kto_beta`, the loss type, both weights, the KL group size and the base
  model), loadable with `mlx_lm.load(base, adapter_path=output)`, written as
  MLX ORPO writes them. `--resume` refuses a checkpoint without that metadata,
  from another task, or from another base, before any tensor loads.
- **Refused at load:** `quantization: 8bit` (use `none`, or a pre-quantized
  4-bit base with `quantization: 4bit`), `training.lora.r: 0` (the reference is
  the untrained LoRA policy), non-text `modality`, DoRA, LoRA+ and
  `data.chat_template` (as for MLX DPO and ORPO). The other settings MLX DPO
  does not apply are not applied here either.

#### GRPO on MLX

`task: grpo` with `backend: mlx` runs Soup's own loop: each optimizer update
generates completions from the current policy, scores them with a built-in
reward, and takes one policy-gradient step. This is the objective both backends
train today: the effective `GRPOConfig` of Soup's transformers path on TRL
0.29.1, whose defaults Soup does not override.

Qualified on Qwen2.5-0.5B-Instruct in bf16 on an M4 Pro, on 400 synthetic
arithmetic prompts (`a * b ± c`, gold in `\boxed{}`) with `reward_fn: accuracy`,
`num_generations: 4`, batch 4 × accumulation 2, `grpo_beta: 0.1`: 200 updates
took about 10 minutes and 6.0 GB, and greedy accuracy on 100 held-out prompts
went from 0.04 to 0.99 as the model learned to work the product out before
answering. That task is easy and fully verifiable; it shows the loop training,
not reasoning quality in general.

| setting | value |
|---|---|
| `loss_type` | `dapo`: per-token clipped surrogate, summed over every completion token of the update and divided by that token count (EOS included) |
| `scale_rewards` | `group`: advantage = (reward - group mean) / (group sample std + 1e-4) |
| `epsilon` | 0.2 on both sides |
| `num_iterations` | 1, so the old policy is the current one detached and the ratio is exactly 1 |
| `importance_sampling_level` | `token` |
| `beta` | `training.grpo_beta` (default 0.1), TRL's k3 KL; `0` runs no reference forward |
| sampling | temperature 1, top-p 1, top-k 0, no min-p, repetition penalty 1 |
| `max_completion_length` | `data.max_length` |
| `mask_truncated_completions` | false: completions with no EOS train on every token |

Exposing `loss_type` (for example the original per-sequence `grpo` loss) is not
a setting on either backend.

```yaml
base: Qwen/Qwen2.5-0.5B-Instruct
task: grpo
backend: mlx

data:
  train: ./math.jsonl        # chatml rows; the final assistant turn is the gold answer
  format: chatml
  max_length: 256            # = max_completion_length; keep it small

training:
  reward_fn: accuracy        # or format
  num_generations: 4
  # batch_size: auto (or anything below num_generations) is one group
  batch_size: 4              # completions per micro-step, as in TRL
  gradient_accumulation_steps: 2
  grpo_beta: 0.1
  lr: 1e-5
  quantization: none
  lora:
    r: 16
    alpha: 32
```

A row of `math.jsonl`:
`{"messages": [{"role": "user", "content": "What is 2+3? End with '#### <number>'."}, {"role": "assistant", "content": "#### 5"}]}`.
The loader does not auto-detect bare `{"prompt", "answer"}` rows; use `messages`
(or Alpaca `instruction` / `output`).

- **Units are TRL's.** `training.batch_size` counts completions per
  micro-step. One update uses `batch_size x gradient_accumulation_steps`
  completions, which is that number divided by `num_generations` prompts. The
  product must be a multiple of `num_generations`. A batch smaller than one
  group is raised to `num_generations`, as on the transformers path;
  `batch_size: auto` is one group on MLX (`num_generations` completions). Prompts are shuffled each
  epoch from `training.seed`. The trailing prompts that do not fill an update
  are dropped, as TRL's sampler does, so there are `prompts // prompts per
  update` updates per epoch. `data_seed` does not change the order on either
  backend.
- **Rollouts.** Completions are generated once per update from the policy at
  that update. Generation runs in train mode, so LoRA dropout is active, as in
  TRL. There is no gradient through generation. Each prompt's group is
  prefilled once and sampled through mlx-lm's KV cache. A completion stops at
  the tokenizer's `eos_token_id`, which it keeps, or at `data.max_length`
  tokens. A prompt plus `data.max_length` longer than the model's context is
  refused; the transformers path has no such check.
- **Prompts and rewards.** Prompts are read as on the transformers path. A
  `prompt` string is the user turn. With `messages`, a final assistant turn
  becomes the gold `answer`. Alpaca `instruction` / `output` rows work too. The
  chat template is applied with `add_generation_prompt`, including
  `data.chat_template` and the transformers path's fallback template.
  `accuracy` and `format` are called with TRL's arguments except
  `trainer_state`, which the built-in rewards do not read. Each completion is
  scored against its own row's gold, with every other column passed through.
  A missing or unreadable gold, or too few prompts for one update, is refused
  before the model loads. A reward that is not a finite number for each
  completion stops the run.
- **Reference.** With `grpo_beta > 0` the reference is the base model, as TRL
  uses for a PEFT model. It is the policy with every LoRA scale set to 0, run
  on the generated tokens before each update. No second copy is loaded.
- **Watchdog.** A nonfinite loss or gradient skips the update, as Soup's GRPO
  watchdog does on transformers. The weights and optimizer moments do not
  change. The skip is printed and counted in the result and in
  `adapter_config.json`. One difference: MLX has a single optimizer step that
  drives both the schedule and Adam's bias correction, so a skipped update
  also holds the learning-rate schedule. The transformers path still advances
  its scheduler.
- **Validation.** On `backend: mlx` the loader always holds out
  `data.val_split` (default 0.1) of the rows, while transformers GRPO holds
  rows out only when `training.eval_steps` is set. The run prints how many
  prompts were held out at the start. Those prompts are not trained on; they
  are scored once after training, `num_generations` completions each, at
  temperature 1 in eval mode (LoRA dropout off), seeded from `training.seed`,
  with no parameter change, and reported as `val_reward`: printed, returned
  in the run's result and logged to the run's metrics at the last update's
  step, on its own row with no training loss. MLX GRPO refuses
  `training.eval_steps`. Set `data.val_split: 0` to train on every row, as the
  transformers path does without `eval_steps`.
- **Reported per update:** loss, reward mean and sample std, per-reward mean
  and std, the fraction of groups with zero reward std, completion mean, min
  and max length, the truncated (no-EOS) fraction, KL (when `beta > 0`), clip
  ratios and skipped updates. They are printed when there is no live panel and
  returned in the run's `history`. KL and the clip ratios are token means over
  the whole update; TRL averages each micro-step's token mean, which differs
  when micro-steps hold different token counts. The truncated fraction counts
  completions without EOS; TRL also counts one that ends in the pad token as
  finished.
- **Saved:** `adapters.safetensors` and `adapter_config.json`, loadable with
  `mlx_lm.load(base, adapter_path=output)`. Both are written atomically, and the
  metadata before the first update. The metadata records every setting in the
  table above, the reward, the advantage epsilon and std, the reference and
  generation mode, and the skipped-update count. `save_steps` counts updates.
- **Cost.** Rollouts dominate: each update samples up to `data.max_length`
  tokens for `num_generations` completions of every prompt. Keep
  `data.max_length` small. With `grpo_beta > 0` each micro-step adds one
  reference forward. Validation adds up to `num_generations x data.max_length`
  sampled tokens per held-out prompt, one prompt after another.
- **Refused at load:** `quantization: 8bit`, `training.lora.r: 0`, DoRA,
  LoRA+, LoRA-FA, non-text `modality`, a `.py`, comma-ensemble or
  `verifiable` reward, `prm_reward`, `rollout_backend` / `rollout_func`,
  `grpo_variant` and the other stability fields, the reward-hack, echo-trap,
  RL-checkpoint and curriculum settings, `eval_steps`, vLLM options and layer
  streaming.
- **Refused at launch or after the load:** `--deepspeed` / `--fsdp`, and
  `--resume` / `--hf-resume`: only the adapter is saved, and resume is not
  implemented for MLX GRPO. A base that loads quantized is refused right after
  the load. 4-bit bases come in a follow-up. `quantization: 4bit` is Soup's default and, on MLX, only names a
  pre-quantized base, so a dense base trains with it.

#### DoRA on MLX

For text SFT, set `training.lora.use_dora: true` to use MLX-LM's native
weight-decomposed adapter instead of plain LoRA. Soup supports dense linear
targets on unquantized or pre-quantized 4-bit bases, with the same scalar
`lora.r`, `lora.alpha` and `target_modules` settings as MLX LoRA. The base remains
frozen; the low-rank factors and magnitude parameters train. Switch/expert layers,
embedding targets, unmatched target names and 8-bit target layers are refused
before adapter conversion. `use_rslora`, `rank_pattern` and `alpha_pattern` are
not supported with MLX DoRA. Unsloth DoRA remains refused.

The output records `fine_tune_type: dora` and is loadable with
`mlx_lm.load(base, adapter_path=output)`. `soup train --resume auto` warm-starts
the DoRA weights, including magnitudes; it refuses missing, extra or differently
shaped adapter tensors before loading any of them. This restores weights only,
not optimizer state, dataset position or the iteration counter.

To fuse the adapter, use the native MLX-LM command (this is not `soup merge`):

```bash
mlx_lm.fuse --model ./mlx-base --adapter-path ./output \
  --save-path ./fused-model --dequantize
```

Native regression tests use a local, randomly initialized one-layer Llama and
check training, frozen base weights, gradients, reload and dequantized fusion.
They require no model downloads. They establish adapter lifecycle correctness;
they do not establish pretrained-model quality or a speed advantage over LoRA.

#### Optimizers and schedules

`training.optimizer`, `training.scheduler`, `training.warmup_ratio` and
`training.weight_decay` are honoured on the MLX backend
([#686](https://github.com/MakazhanAlpamys/Soup/issues/686)). Before that they
were validated, accepted and dropped — every run built a bare AdamW at a
constant learning rate, whatever the recipe said.

Because they are honoured rather than ignored, a setting MLX cannot express is
now **refused when the optimizer is built**, rather than silently substituted:

| setting | MLX accepts | otherwise |
|---|---|---|
| `optimizer` | `adamw_torch`, `adamw_hf`, `adamw_torch_fused`, `sgd`, `adafactor`, `adagrad`, `rmsprop`, `muon` | refused by name, listing what is available |
| `scheduler` | `cosine`, `linear`, `constant`, `constant_with_warmup` | refused, naming the old constant-rate behaviour |
| `weight_decay` | any value on every optimizer above except `adagrad` and `rmsprop` | a non-zero value on `adagrad` / `rmsprop` is refused — those MLX constructors take no `weight_decay`, and dropping it silently is the defect above |

Soup's optimizer allowlist (`utils.optimizer_zoo`) is far wider than anything
MLX ships, so most valid values have no MLX equivalent. Run those recipes on
the transformers backend.

The schedule counts **optimizer updates**, not iterations: mlx-lm calls
`optimizer.update()` once every `gradient_accumulation_steps`, so a warmup of
`warmup_ratio × (iters // gradient_accumulation_steps)` is what actually runs.
The effective plan is written to `adapter_config.json` (`optimizer`,
`scheduler`, `warmup_updates`, `total_updates`, `weight_decay`, `peak_lr`), so
what ran is recoverable from the output directory.

`--resume auto` finds mlx-lm's step-numbered `NNNNNNN_adapters.safetensors` checkpoints and warm-starts the LoRA weights from them ([#634](https://github.com/MakazhanAlpamys/Soup/issues/634)). This restores adapter weights only, not training state: mlx-lm's LoRA trainer exposes no optimizer state or step count, so training restarts from step 0 regardless of how far the checkpoint got. See [Resume Training](#resume-training) below for the MLX-specific checkpoint shape.

### Transformers on MPS

The regular `backend: transformers` path can run more than MLX's SFT, DPO, ORPO,
SimPO, KTO and GRPO surface. On a live MPS runtime that accepts bfloat16, Soup enables BF16 autocast
for the hardware-validated text trainers: SFT, DPO, GRPO/RLVR, reward modelling,
and PRM.
The decision comes from a live MPS allocation probe rather than a macOS version
guess. CPU and an unavailable or older MPS runtime remain in FP32; FP16 is not
selected on MPS.

For BF16 checkpoints, resident SFT, DPO, GRPO, and reward-model runs preserve
the frozen base weights in BF16 while keeping LoRA parameters in FP32. PRM uses
BF16 autocast but deliberately retains FP32 master weights: loading its
trainable base in BF16 makes the Metal optimizer abort because its accumulator
and destination matrix dtypes differ. This policy was validated with one-step
runs on Apple Silicon for all five tasks. GRPO validation uses local
Transformers generation (`use_vllm: false`) and deterministic RLVR; vLLM remains
a CUDA-oriented optional path. Other Transformers trainers remain FP32 on MPS
until their task-specific kernels receive equivalent hardware coverage.

On Apple Silicon MPS hardware, Transformers' threaded weight loader has been reported to crash with SIGABRT in `_materialize_copy` on some setups (#2076). Soup sets `HF_DEACTIVATE_ASYNC_LOAD=1` by default when running on MPS to enforce synchronous model loading while preserving any explicit user environment setting. To opt out and re-enable threaded loading, export `HF_DEACTIVATE_ASYNC_LOAD=0` before training.


## Unsloth Backend (2-5x Faster Training)

Use the [Unsloth](https://github.com/unslothai/unsloth) backend for significantly faster training and up to 80% less VRAM:

```bash
# Install unsloth support
pip install "soup-cli[fast]"
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

Works with the tasks that have an unsloth setup: SFT (text; SFT vision and audio still run the transformers setup), DPO, GRPO, PPO, KTO, ORPO, SimPO, IPO, BCO, preference, Pretrain, Embedding and TTS. The others (`reward_model`, `prm`, `classifier`, `reranker`, `cross_encoder`, `distill`, `unlearn`, `moe_lora_routing`, `online_dpo` and `asr`) have no unsloth setup, so `backend: unsloth` is refused at config load for them (#1357). If unsloth is installed but not enabled, Soup suggests it, but only for a config that would still load with `backend: unsloth` (#1736): a config one of these refusals would stop gets no suggestion.

`quantization: 8bit` with `backend: unsloth` is refused at config load: Soup's Unsloth loader passes `load_in_4bit` only, so `8bit` used to load the model in 16-bit with no word (#1811). Use `4bit` or `none` on this backend, or `backend: transformers` for 8-bit.

`training.lora.rank_pattern` and `training.lora.alpha_pattern` are refused on `backend: unsloth` for every task that has an unsloth setup ([#2153](https://github.com/MakazhanAlpamys/Soup/issues/2153)): the unsloth loader attaches every adapter from the one scalar `lora.r` and `lora.alpha`, so per-module ranks and alphas would be dropped without a word. Remove them, or use `backend: transformers`, which applies them. An empty `{}` sets nothing and still loads.

> **Tip:** Soup auto-detects unsloth. When it is installed, `soup train` prints a hint if you haven't enabled it yet and your config would load with it.

### Gradient checkpointing on Unsloth

On `backend: unsloth`, `training.gradient_checkpointing` is honoured instead of silently ignored ([#797](https://github.com/MakazhanAlpamys/Soup/issues/797)):

- `false` (the schema default) disables checkpointing and may increase activation memory.
- `true` (or a tier: `selective`, `medium`, `full`, `auto`) enables Unsloth's optimized checkpointing; throughput and maximum context depend on the workload.
- Layer streaming (`training.stream_layers: true`) already checkpoints every layer, so the Unsloth kernel stays off there whatever the flag says.

```yaml
training:
  gradient_checkpointing: true  # keep Unsloth's optimized checkpointing
```


## Cloud GPU Training

No local GPU? `soup train --cloud modal|lambda|runpod` renders a provider-specific controller
from your `soup.yaml`. The config YAML is base64-embedded as **data**; credentials are read from
the environment only when a live submission starts.

```bash
pip install "soup-cli[modal]"   # only needed for live submit

# Plan-only (default): write the stub + print the `modal run` command.
soup train --config soup.yaml --cloud modal --gpu a100

# Submit live (authenticate once with `modal setup`, or set
# MODAL_TOKEN_ID + MODAL_TOKEN_SECRET).
soup train --config soup.yaml --cloud modal --gpu a100 --cloud-submit
```

`--gpu` accepts: `t4` / `l4` / `a10g` / `a100` / `a100-80gb` / `l40s` / `h100`. The rendered
`soup_modal_app.py` builds an image with `soup-cli[train]` pinned to your running version, writes
the embedded config inside the container, and runs `soup train` on the chosen GPU.

Run outputs are written to the `soup-outputs` Modal volume (created on first use), in a directory
named after the run (`soup-<12 hex>`), so checkpoints outlive the container. When the run ends,
including after a failed run, the app downloads every file in that directory into your local
output directory and prints a `modal volume get soup-outputs /<run name> <output dir> --force`
command that retries the download; that command places the files under `<output dir>/<run name>`.

### RunPod

`soup train --cloud runpod --gpu a100` writes `soup_runpod_app.py` without contacting
RunPod. GPUs: `a100` (80 GB SXM), `h100`, `a40`, `rtx3090`, `rtx4090`. The controller
uses Python's standard library and local OpenSSH `ssh`/`scp`; no RunPod SDK extra
is needed. For live submission, set `RUNPOD_API_KEY` and `RUNPOD_SSH_PRIVATE_KEY`
to your private key path, with its matching OpenSSH public key at `<path>.pub`.
Only ed25519 and RSA public keys are accepted:

```bash
soup train --config soup.yaml --cloud runpod --gpu rtx4090 --cloud-submit
```

The local controller creates one Secure Cloud GPU Pod with TCP SSH access, installs
the running Soup version, runs training over SSH and copies the configured relative
output directory back before deleting the Pod. Choose a fresh local output path;
existing outputs, symlink/junction paths and path components starting with `-` are
refused before allocation and rechecked before copying. SCP separates options from
its operands with `--` and writes into a new, empty directory created next to the
output path for the copy; only the expected output directory, holding directories
and regular files only, is then moved to the output path. Anything else stops the
move and leaves that directory in place for inspection. A failed or timed-out copy
removes that directory when nothing was written to it; otherwise the failure notice
names it, with the partial copy inside. `ssh` and `scp` run by the
absolute path found in an absolute `PATH` entry (the current directory and relative
entries are not searched), with the controller's environment minus `RUNPOD_API_KEY`.
When either is missing, the controller prints "OpenSSH ssh and scp must be installed
and on PATH before RunPod submission." and exits 2 before creating a Pod.
Submission starts Python in isolated mode (`-I`) and passes
only the RunPod token/key path, `PATH`, `HOME`, `USERPROFILE`, proxy settings
(`HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, `NO_PROXY` and their lowercase forms),
temporary-directory settings (`TEMP`, `TMP`, `TMPDIR`), Windows settings
(`SYSTEMROOT`, `SYSTEMDRIVE`, `WINDIR`, `PATHEXT`, `NoDefaultCurrentDirectoryInExePath`; on
Windows `soup` sets the last to 1 when it is not set, and the controller sets it to 1 for `ssh`
and `scp`) and `SSH_AUTH_SOCK` when present. Environment names are matched case-insensitively on Windows. The controller
reconfigures stdout/stderr to replace characters the pipe encoding cannot represent,
because isolated Python ignores `PYTHONIOENCODING` and `PYTHONUTF8`.
SSH/SCP use `IdentitiesOnly=yes` with the selected key and a temporary per-run known-hosts
file, removed even after failure; the user's known-hosts file is not modified.
The config is base64-embedded data; planning applies the
shared YAML size, nesting and alias-expansion limits, and refuses inline credential
fields. The API key and private SSH key remain local;
only the public key enters the Pod environment. Models and datasets must be publicly
accessible remotely: local input files and local credentials are not uploaded.

Deletion runs in `finally` on training/copy failure, timeout and Ctrl+C, and polls
until the API returns 404. The parent waits through interrupts so it does not kill
the cleanup controller. Transient transport failures, including truncated HTTP
responses during DELETE or confirmation GET, are retried within the five-minute
cleanup window. A lost or ambiguous create response (including HTTP 5xx and 408)
is reconciled by the unique run name, without repeating the create request or
deleting unrelated Pods. A definite HTTP 4xx rejection other than 408 reports the
status without claiming that cleanup of an allocated Pod failed. Fixed controller
failure reasons are shown; arbitrary transport/subprocess exception text is hidden.
Keep the controller running; force-killing it or losing API access can leave cleanup
unconfirmed. An unconfirmed cleanup exits nonzero and names the run to check in
the RunPod console. A single interrupted SSH training session is failed rather
than retried, to avoid duplicate training. Keepalive probes run every 30 seconds
with six missed probes allowed (roughly three minutes); the training command has
a 25-hour timeout. Artifacts are copied only after a successful training command,
with one SCP attempt capped at 15 minutes. A failed copy still terminates the Pod,
so an incomplete download cannot be recovered from that Pod afterward. Remote log
tail retrieval and a copy retry window are not implemented in this controller.

**Not yet live-validated.** Tests use local mocked API/SSH transports. The code
PR does not include a paid run; provider execution, GPU behavior, retrieval and
billing termination remain a separate live-validation task, per #264's ruling.
API contract: [RunPod REST v1 OpenAPI](https://rest.runpod.io/v1/openapi.json),
[SSH setup](https://docs.runpod.io/pods/configuration/use-ssh).

### Lambda Cloud

Lambda uses an instance rather than a serverless function. The generated local controller sends a
secret-free cloud-init script as API `user_data`, waits for it over SSH, copies the configured
output back, and requests instance termination in a `finally` block. The controller polls training
status over short SSH connections with keepalive probes rather than holding a single persistent
session open. It tolerates transient network drops and disconnects for up to 15 minutes with exponential
backoff before terminating the instance to prevent runaway billing. Keep the controller running
until it reports that termination succeeded; shutting down the guest does not terminate billing.
Pressing Ctrl+C during `--cloud-submit` interrupts the controller, which still runs that `finally`
block, and `soup` waits for it to terminate the instance and exit rather than killing it; the only
way to skip that cleanup is to kill the controller process itself (for example with `kill` from
another terminal), which can leave the instance running. If remote training fails with a non-zero exit
code, the controller fetches the last 50 lines of `/home/ubuntu/soup/train.log` over SSH (capped at 50 KB
and sanitized of terminal control sequences) and prints them before terminating the instance so
the root cause is immediately visible (the log file holds only `soup train` output; an earlier failure
during cloud-init or `pip install` is not captured in it).

Register the public half of an SSH key with Lambda first, then set:

```bash
export LAMBDA_API_KEY=...
export LAMBDA_SSH_KEY_NAME=my-lambda-key
export LAMBDA_SSH_PRIVATE_KEY=/path/to/private-key
export LAMBDA_REGION=us-tx-1  # optional; defaults to us-tx-1
soup train --config soup.yaml --cloud lambda --gpu a100 --cloud-submit
```

`--gpu` accepts: `a10` / `a100` / `a6000` / `h100`. Lambda output paths must be relative, without
`..` and without a path part starting with `-`, so the controller can copy the artifact back
safely; planning refuses any other output path. SCP separates options from its operands with `--`.
`ssh` and `scp` run by the absolute path found in an absolute `PATH` entry (the current directory
and relative entries are not searched), with the controller's environment minus `LAMBDA_API_KEY`.
When either is missing, the controller prints "OpenSSH ssh and scp must be installed and on PATH
before Lambda submission." and exits 2 before launching an instance. The API key stays on the
caller and is never embedded in cloud-init or instance logs.

The Lambda submission path still requires the paid live-validation checklist in #1280 before
it can be described as provider-validated. Plan-only rendering and the lifecycle boundaries are
covered by offline tests.


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


## Resume Training

Resume a training run from a checkpoint:

```bash
# Auto-detect latest checkpoint in output directory
soup train --config soup.yaml --resume auto

# Resume from a specific checkpoint
soup train --config soup.yaml --resume ./output/checkpoint-500
```

`backend: mlx` writes and resumes a different checkpoint shape: a
step-numbered `NNNNNNN_adapters.safetensors` file (or the final
`adapters.safetensors`) directly under `output`, not a `checkpoint-N`
directory. `--resume auto` and `--resume ./output/0011800_adapters.safetensors`
both work; `--resume ./output/checkpoint-500` does not, because MLX never
writes that shape. This is a weights-only warm start — mlx-lm's LoRA trainer
exposes no optimizer state or step count, so the resumed run starts counting
from step 0 regardless of how far the checkpoint got.

`task: unlearn` refuses `--resume` / `--hf-resume`, because it writes no checkpoints.


## Run Management & Cleanup

LLM training generates massive checkpoint files. Soup automatically manages an SQLite database of your training loss and metrics, empowering you to safely reclaim disk space once training is complete.

```bash
# List all historical training runs
soup runs

# Compare two differing experiments side-by-side
soup runs compare run_202611... run_202612...

# Intelligently clean up redundant checkpoints
# (Preserves the final model and the checkpoint with the lowest loss)
soup runs clean run_202611...

# Preview space that would be reclaimed across ALL experiments
soup runs clean --all --dry-run
```

By default, the `clean` command operates in "surgical mode" (`--keep-weights`), deleting huge optimizer state files (`optimizer.pt`) from lesser checkpoints to save gigabytes, but keeping their lightweight evaluation weights just in case you want to load them later. Pass `--no-keep-weights` to delete whole non-best checkpoints instead (the checkpoint with the lowest loss is always kept); combine it with `--dry-run` to see what would go first. The checkpoint kept whole is the one with the lowest loss recorded at its step. When no checkpoint has a recorded loss, for example in a run shorter than `logging_steps`, which logs none, it is the latest checkpoint, and `clean` says so.


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

ModelScope and Modelers downloads and uploads route through live, lazy-imported SDK adapters. The
HF path remains the default, and MLX remains HF-only.

`soup push --hub modelscope` authenticates with `--token` or `MODELSCOPE_API_TOKEN`; `--hub modelers`
with `--token` or `MODELERS_TOKEN`. Neither reads `HF_TOKEN`.


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

# Check every shipped config can attach its LoRA adapter (no weights downloaded)
soup recipes verify
soup recipes verify --config soup.yaml
soup recipes verify --json    # one row per config on stdout, for CI
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
| **Multimodal reasoning** | Llama-3.2-Vision GRPO |
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

# Preview without running — validates first (#642): an unknown config key is
# refused exactly as `train --dry-run` refuses it, and a --param naming no
# config field exits non-zero before any grid is printed
soup sweep --config soup.yaml --param lr=1e-5,2e-5 --param epochs=2,3 --dry-run

# Early stopping: skip remaining runs if loss exceeds 1.5x best
soup sweep --config soup.yaml --param lr=1e-5,2e-5,5e-5 --early-stop 1.5
```

Every arm builds the trainer `soup train` builds for the config's `task` and
`backend`, so an arm on a `task: distill` config distills, and an arm on
`backend: mlx` uses the MLX wrapper.


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

With `soup doctor --config soup.yaml`, the declared support report lists only
explicitly written, active settings that differ from their schema defaults.
Writing a default counts as leaving it unwritten, so a full YAML dump and its
minimal source have the same report; `seed: 0` still counts because the seed
default is unset. Settings refused at config load are not listed as doctor rows:
their active values fail before the report can run. This report describes the
reviewed gaps, not a guarantee that every unlisted setting is supported.

### GPU diagnostics

When an NVIDIA GPU is available, `soup doctor` reports its compute capability
and architecture family, whether the installed PyTorch build contains native
or PTX support for that architecture, and the hardware/software gates for
BF16, FP8, and NVFP4. A missing architecture warning includes a CUDA-enabled
PyTorch reinstall hint based on the CUDA version reported by the driver. When
the driver version cannot be mapped to a supported wheel, the hint remains
actionable without constructing a `whl/None` URL.


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


## Error Handling

Soup shows friendly error messages by default (2-3 lines with a fix suggestion). For full tracebacks:

```bash
# Global flag goes BEFORE the command
soup --verbose train --config soup.yaml

# Works with any command
soup --verbose eval benchmark --model ./output --benchmarks mmlu
```

> **Note:** `--verbose` is a global flag — it must go **before** the command name, not after.


## Unknown config keys

Every config model used to run with Pydantic's default `extra="ignore"`, so a key the
schema did not declare was dropped without a word. `soup train --dry-run` printed
"Config valid. Ready to train!", the run exited 0, and the requested setting was simply
never applied — `quantizaton: none` trained 4-bit quantized when full precision was
what you asked for, `gradient_checkpoint: true` did no checkpointing, `max_len: 512`
truncated at 2048.

Loading a config now refuses every key it cannot place, in **one** report per load,
with the field you probably meant:

```
Config validation error:

  unknown config key 'data.max_len' - did you mean 'max_length' or 'video_maxlen'? Refused.
  unknown config key 'training.learning_rate' - use training.lr. Refused.
  unknown config key 'training.quantizaton' - did you mean 'quantization' or 'quantization_aware'? Refused.
```

Known foreign parameter names (such as Hugging Face `learning_rate` or `num_train_epochs`) get an authoritative `use <field>` hint instead of a guess.

**Since v0.75 an unknown config key refuses the load.** v0.74 shipped the same report
as a warning that named this deadline, so there was exactly one release of notice —
deliberately, because a config written against a newer Soup has to keep running on an
older wheel for at least one release. The refusal is the same everywhere a `SoupConfig`
is built from a file or a string: `soup train` exits 1 before the training stack is
imported, `soup sweep` / `soup doctor --config` / `soup ship --config` / `soup plan` /
`soup apply` refuse the same way, and the Web UI / API loader raises `ValueError` with
the same text (the Web UI shows it). Nothing is defaulted and nothing is guessed: the
suggestion is a hint for you, not a substitution the loader makes. A root-level `lora:`
block is not an unknown key — the schema has accepted that spelling and moved it under
`training` since v0.40.1, and the detector applies the same remap before it looks.

A config that names a key your installed Soup does not have usually means one of two
things: a typo (take the suggestion), or a field added after your version shipped
(`soup version` against the [changelog](../CHANGELOG.md) will say which). A config
that must stay loadable on v0.74 as well needs the key removed, not renamed — v0.74
warns and ignores it, v0.75 refuses it, and neither applies it.

**`soup sweep` never had the warning period.** A `--param` that matches no config
field has been a hard error since v0.74:

```bash
soup sweep --config soup.yaml --param lora_rank=8,16   # the field is training.lora.r
# sweep parameter does not match any config field: unknown config key 'lora_rank' - refused.
echo $?   # 1
```

The whole sweep is refused before the first arm starts, and the command exits non-zero, so
a scripted or CI-driven sweep fails rather than reporting a grid of arms that each failed
for the same reason. A sweep whose swept knob is never applied produces arms that are all
identical, so there is no partially-useful result to preserve by continuing.


## Config size limits

A config file may be at most 1 MiB, nest mappings and lists at most 100 levels,
and expand through anchors, aliases and `<<` merge keys to at most one node per byte plus 10,000.
A file over a limit is refused with a message naming the file and the limit (`soup ship`: 4 MiB).


## External programs

Soup starts the external programs it uses (`nvidia-smi`, `git`, `docker`, `ollama`, `gh`, the
Modal CLI, `trtllm-build`, the llama.cpp binaries, `accelerate` for `soup train --gpus`, the
`soup adapters bisect --eval-command` program, and `ssh` / `scp` in the RunPod and Lambda
controllers) by the absolute path found in an absolute `PATH` directory, searching the directories
in order. Empty and relative `PATH` entries (`.`, `bin`, and on Windows `\bin` or `C:bin`) are not
searched, and neither is the current directory. A program that is only in the current directory
or in such an entry is not found: put the program's directory on `PATH` as an absolute path.
On Windows `soup` also sets `NoDefaultCurrentDirectoryInExePath=1` for itself when it is not
set, so a program started by bare name by Soup's libraries or by a program Soup starts (an eval
or reward script included) is not looked for in the current directory either, as on Linux and
macOS. A value already set is kept, and Windows leaves the current directory out for any
non-empty value, `0` included, so no setting brings the old search back: to run a program from
the current directory, give its path (`.\tool.exe`) or put the directory on `PATH` as an absolute
path.


## Experiment Tracking

Every `soup train` run is automatically tracked in a local SQLite database (`~/.soup/experiments.db`).

### Recorded base identity

For single-process Hugging Face loads on the Transformers backend (resident or
layer-streamed), `soup train` resolves
the base's commit once, immediately before its first model/tokenizer/processor/config
load, and passes that commit to the run's subsequent base loads. The original `base`
name stays in the config and experiments row; unrelated teacher, reward and scorer
repositories do not receive its revision. `soup runs show` displays the revision and
its source.

Layer streaming pins its tokenizer, architecture config, probe/final meta
skeletons and both checkpoint download passes. Known Hub commits use separate
shard and materialized-weight cache slots; repeated loads of one commit reuse
its slot. Local paths and unknown revisions retain the existing cache behavior.
The config, PEFT base name and run metadata continue to name the original base.
The first pinned streamed run after upgrading re-shards and re-copies the base
into the new `<slug>@<commit>` cache slots, costing disk space and time; old slots
are left in place.

The new nullable `runs.base_revision`, `runs.base_revision_source` and
`runs.training_data_sha256` columns upgrade existing databases in place. Historical
rows stay unknown; Soup never fills them from today's Hub state or dataset bytes.
`soup train` fills `runs.training_data_sha256` at startup for a bounded, complete local
text-training source (#2024); every other input stays unknown.

An unavailable/offline Hub metadata lookup preserves the existing load flow and
records an unknown revision. A local `snapshots/<40-hex>` directory records
`base_revision_source: cache_path`: this is a hint derived from the directory name,
**not proof that its local bytes match the Hub commit**. Other local directories
also have unknown content identity. There is no new config field or new default
offline, legacy-export or `--base` refusal.

The saving rank writes `soup_run.json` beside the final training artifact, containing
the schema version, run ID, original base ID, nullable revision/source and nullable
training-data hash. Keep that file when copying an adapter. Shared live-eval loading
and GGUF's adapter-merge path can then reuse a matching recorded Hub commit without
access to the original experiments database. An explicit different base or missing,
malformed or future-version metadata keeps the existing flow, without guessing a
historical commit. This metadata is provenance, not a signed integrity attestation.
Loading a local adapter folder directly also reuses that commit when its
`adapter_config.json` names the same base and has no conflicting explicit revision;
the tokenizer still comes from the adapter folder.

For a multi-process Hugging Face Transformers `soup train` launch, global rank zero
resolves the base once and broadcasts its nullable result before trainer setup.
Soup initializes or reuses Accelerate's shared process group with Trainer's default
1800-second timeout. The first initialization sets the timeout for every collective,
including later Trainer operations; an existing group keeps its original timeout.
Offline/failed metadata requests
are broadcast as unknown, preserving existing loading behavior. A collective
failure stops the run rather than letting workers resolve independent moving heads.
Ranks compare identity receipts after setup and training; a conflicting loaded
model commit on any rank clears the bound revision for every rank before recording
the final artifact. CPU/Gloo coordination is covered by real two-process tests on
Linux; simulated-rank checks run on every platform. The real cases skip on macOS
and Windows because env:// rendezvous can time out on loaded macOS runners. Rank
failures include both child logs after cleanup. NCCL/FSDP hardware execution is
not established by those tests.

**Remaining boundaries:** Non-HF hubs, Unsloth and MLX loads remain unbound and
are recorded as such; a cache-path hint on any backend is still unverified. Standalone
wrapper/sweep invocations do not automatically establish a run scope. Metadata is
written for the final artifact, not each intermediate checkpoint. Other exporters
(including ONNX/TensorRT and native torchao loaders), remote adapter sidecars
and independent teacher/reward/scorer repositories do not gain a binding guarantee.
Evaluation engines that bypass the shared live-eval loader (including lm-eval's
adapter benchmark path and vLLM) also remain unbound in this stage.

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

As of v0.71.5, the metric-series lookup that powers replay (`ExperimentTracker.get_metric_series`)
transparently falls back to the `eval_results` table when a metric has no per-step
rows — so you can plot a benchmark-score curve (e.g. `mmlu`, `gsm8k`) the same way
you plot `loss`, without caring which table holds the series.

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
Python major.minor + OS + arch + duration + anonymous distinct ID). It is **off by default** and never
sends model names, dataset paths, or config contents.

To opt in, set the environment variable:

```bash
SOUP_TELEMETRY=1 soup train --config soup.yaml
```

When `SOUP_TELEMETRY` is unset, `0`, or any value other than `1`/`true`/`yes`/`on`, Soup performs
zero telemetry network requests.

You can also explicitly disable telemetry for a specific invocation using the `--no-telemetry` flag:

```bash
soup --no-telemetry train --config soup.yaml
```

As released, opting in sends nothing: the bundled PostHog key is a placeholder, so the command prints `Notice: telemetry is not yet live.` and makes no network request. Only with an operator's own `SOUP_POSTHOG_KEY` (and optionally an HTTPS `SOUP_POSTHOG_ENDPOINT`) does telemetry perform a synchronous fire-and-forget HTTPS POST with a 1-second connect and read timeout (DNS resolution excluded) on command exit. The anonymous identifier is stored at `~/.soup/telemetry_id`; deleting `~/.soup/telemetry_id` regenerates it on the next opt-in. See [Privacy Policy](#privacy-policy) for details.


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
pip install "soup-cli[tui]"
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


## GPU Live Monitor

```bash
soup monitor                # 2s refresh, Util / Mem / VRAM / Temp / Power per GPU
soup monitor --refresh 0.5  # faster polling
soup monitor --once         # single snapshot, no Live panel
```

On NVIDIA systems, Soup calls `nvidia-smi` via a list-args subprocess (no shell)
with a 5-second timeout. On Apple Silicon, it reads GPU utilization and power
from `/usr/bin/powermetrics --samplers gpu_power --format plist`. Run `sudo -v`
in a terminal before starting the monitor: Soup uses `sudo -n`, so it can reuse
the cached credential without ever prompting for or reading a password. If the
credential or utility is unavailable, the command exits with an Activity
Monitor fallback rather than reporting an NVIDIA error.

macOS does not expose NVIDIA-style dedicated VRAM, memory-utilization, or GPU
temperature fields through this sampler. Those columns therefore remain `—`
instead of guessing values from unified memory or unrelated thermal sensors.


## Soup Fetch — Bundled Examples

```bash
soup fetch examples                          # list bundled entries
soup fetch examples llama-3.1-8b-lora        # write to ./llama-3.1-8b-lora.yaml
soup fetch examples qwen2.5-7b-dpo -o ./my-config.yaml --force
soup fetch deepspeed_configs zero3-cpu-offload
```

Closed catalog (`MappingProxyType`) of ready-to-edit YAML / JSON. Output path cwd-contained, bundled-source `os.path.commonpath` check (defends against catalog escape), `os.lstat + S_ISLNK` symlink-reject at the write target.


## Llama 4 Delinearizer

```bash
soup delinearize-llama4 ./llama4-checkpoint --target ./out-delinearized [--num-experts N] [--plan-only]
```

LIVE (v0.71.21): reshapes fused Llama-4 expert weights `[E*din, dout]` → `[E, din, dout]` shard-by-shard (atomic writes, per-shard 16 GiB cap, cwd containment) and copies the JSON sidecars so the target stays loadable. The expert count defaults from `config.json` (`text_config.num_local_experts`); pass `--num-experts` when the config doesn't carry it (exit 2 otherwise). `--plan-only` keeps the original preview flow and writes nothing. `is_llama4_model` uses a word-boundary regex matching the `is_gemma4_model` pattern — `ungemma-llama-4ish` is rejected.


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


## Standalone Sweep Config

`--config` stays the base `soup.yaml` to train against; `--sweep-config` points at a
separate file holding the strategy, run count, seed and swept parameters, in place of
repeating `--param` on the command line.

```bash
soup sweep --config soup.yaml --sweep-config sweep.yaml
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

`--param` and `--sweep-config` are mutually exclusive. With `--sweep-config`:

- The file's `strategy` and `seed` win over `--strategy`.
- A non-zero `n_runs` wins over `--max-runs`; `0` or absent falls back to `--max-runs`.
- `n_runs` only caps the run count, so a grid smaller than `n_runs` runs every combination
  (the example above is 3 x 3 = 9, so it runs 9 trials, not 20).
- `seed` seeds the random sampler.

A file that omits `strategy`, combined with `--strategy random --max-runs 2` on the command
line, silently runs grid, because the file's default `grid` wins over both flags.


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


## Telemetry & Privacy Policy

Soup contains opt-in, hardware-info-only telemetry in `utils/trackers.py` (`build_telemetry_payload` / `send_telemetry_payload`).

### Privacy Policy

Soup's telemetry is strictly anonymous and hardware-focused. When opted in via `SOUP_TELEMETRY=1`, the payload carries only the following fields to understand what environments we need to support (it is sent only with an operator's own `SOUP_POSTHOG_KEY`, see below):

- `soup_version`: the version of Soup being run
- `command`: the top-level command executed (e.g. `train`, `data`, validated against known commands; unknown commands or paths are masked as `(unknown)`)
- `python`: Python major.minor version
- `os`: OS platform name (`platform.system()`)
- `arch`: System architecture (`platform.machine()`)
- `duration_seconds`: Command execution duration in seconds
- `distinct_id`: Anonymous UUID4 generated locally on first run and stored at `~/.soup/telemetry_id` to deduplicate events. Deleting `~/.soup/telemetry_id` regenerates it on the next opt-in.

We **NEVER** collect:
- Dataset paths or contents
- Model names or architectures
- Config file contents or hyperparameters
- Usernames, local file paths, or directory names
- IP addresses (the payload carries none; the receiving server sees the client address of any HTTPS request), tokens, or credentials

With the bundled placeholder key nothing is uploaded: an opted-in command prints `Notice: telemetry is not yet live.` and makes no request. With an operator's own `SOUP_POSTHOG_KEY`, uploads use HTTPS, a 1-second connect and read timeout (DNS resolution excluded), and defensive SSRF validation. Any network or filesystem exception is silently swallowed so telemetry can never fail or interrupt your work.


## Plugin System

Soup discovers bundled modules under `soup_cli.plugins` and installed Python
distributions that publish the `soup_cli.plugins` entry-point group. An external
plugin exposes a zero-argument registration function:

```python
from soup_cli.plugins import register_plugin

class MyPlugin:
    def pre_train(self, ctx):
        ...
    def post_train(self, ctx):
        ...

def register():
    register_plugin(
        name="my-plugin",
        version="1.0.0",
        plugin=MyPlugin(),
        description="Hooks into pre/post-train",
        templates=["my-template"],         # optional metadata
        model_groups=["my-arch-family"],   # optional metadata
    )
```

Declare it in the plugin distribution's `pyproject.toml`:

```toml
[project.entry-points."soup_cli.plugins"]
my-plugin = "my_package.soup_plugin:register"
```

Bundled Soup plugin modules are enabled by default. Installed third-party entry points
are **disabled by default**: discovery reads their names and versions from package
metadata without importing or executing their modules. `soup plugins enable <name>` is
the explicit boundary that loads the selected entry point; its entry-point name must
match the plugin name it registers. The choice is stored atomically in
`~/.soup/plugins.json` and is reused by later Soup processes.

Set `SOUP_PLUGIN_STATE_PATH` to use a different trusted local state file, for example in
an isolated test environment. This explicit path is not confined to the current
workspace; Soup rejects NULs, oversized paths, symlink state files, oversized content,
and malformed JSON.

```bash
soup plugins                       # discover and list plugins
soup plugins enable my-plugin      # opt in persistently
soup plugins disable my-plugin
```

`soup plugins install` deliberately exits with status 2: Soup does not run a package
installer on the user's behalf. Install the distribution with your trusted Python
package workflow, then enable it explicitly.

Plugin names are kebab-case (`^[a-z0-9][a-z0-9-]{0,39}$`); versions are semver-ish (`MAJOR.MINOR.PATCH`); registry caps `_MAX_PLUGINS=64`, `_MAX_TEMPLATES_PER_PLUGIN=32`, `_MAX_MODEL_GROUPS_PER_PLUGIN=32`. Re-registering the same `(name, version, plugin, templates, model_groups, description)` is idempotent; any field mismatch is rejected with a clear error. `templates` and `model_groups` are descriptive metadata surfaced by `soup plugins`; Soup does not apply them to model configuration automatically.


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


## Soup Plugin Callbacks

Register a plugin once via the v0.45.0 registry API; v0.53.6 wires it into every
transformer-backend trainer as a real HF `TrainerCallback`:

```python
# my_package/soup_plugin.py — discovered only by plugin-aware commands/training
from soup_cli.plugins import register_plugin

class MyPlugin:
    def pre_train(self, ctx):
        print("training about to start, args =", ctx["args"])

    def post_step(self, ctx):
        if ctx["state"].global_step % 100 == 0:
            print(f"step {ctx['state'].global_step}")

def register():
    register_plugin(name="my-plugin", version="0.1.0", plugin=MyPlugin())
```

A misbehaving plugin hook is swallowed at WARNING — one bad plugin must never crash
a multi-hour training run. The hook snapshot is taken at callback-construction time,
so a plugin registered MID-run does not retroactively receive events. Discovery is
lazy, so importing the CLI or running an unrelated light command does not load plugin
entry points.


## Terraform-Style Plan & Apply (`soup plan` / `soup apply`)

A training run is a one-shot infrastructure-shaped operation: spot price, expected cost, base SHA, dataset SHA, peak VRAM. v0.64 borrows Terraform's plan-apply split so you can review the numbers before committing money.

```bash
# Render a pre-flight summary + write soup.tfstate
soup plan --config soup.yaml

# Apply — refuses on drift (exit 3) if the YAML changed since `plan`
soup apply --config soup.yaml

# Validate without actually running anything
soup apply --config soup.yaml --dry-run
```

The state file is a thin JSON envelope; the actual training is still driven by `soup train`. The gate prevents the "wait, why did I spend another $0.50 on the wrong config" surprise.

`peak_vram_gb` is the `soup train` pre-flight's own estimate for the run, before its 10% margin ([Hardware-Fit Calculator](#hardware-fit-calculator)). When the plan cannot price the run, it records `null` and says why in `peak_vram_reason`: the model's `config.json` could not be read, `batch_size: auto`, a quantization with no memory model, layer streaming (sized by the streaming planner when it starts) or `backend: mlx`. Until #2049 it recorded a fixed 8.0 GB for every run. A state file written before that still loads, and `peak_vram_gb` is not part of the drift check.


## Hermetic Env Lockfile (`soup env`)

The "CUDA hell" problem: a fine-tune that worked on Friday breaks on Monday because PyPI silently upgraded `transformers` past the trainer's compat band. v0.34 `soup doctor` surfaces some of this; v0.64 makes it lockable.

```bash
# Snapshot the current env into soup-env.lock
soup env lock

# Print the locked env summary
soup env status

# Compare current env to the lock — exit 3 on ABI-sensitive drift
soup env check
```

`soup-env.lock` captures Python + platform + CUDA + 15 ABI-sensitive packages (torch / transformers / peft / trl / accelerate / bitsandbytes / flash-attn / xformers / deepspeed / unsloth / vllm / sentencepiece / tokenizers / datasets / huggingface-hub). Wire `soup env check` into your CI to refuse silent ABI breakage.


## Hardware-Fit Calculator

One peak-VRAM estimate (`soup_cli/utils/vram_estimate.py`) serves the `soup train` pre-flight, `soup profile`, the MCP `profile` tool and `soup plan`, so the same config gets the same number from each (#2049). The pre-flight adds a 10% margin and refuses a run whose estimate does not fit the card (`--allow-oom-attempt` launches it anyway); it skips layer-streaming runs, which the streaming planner sizes, and `backend: mlx`.

The estimate is per term, in bytes (GB in the output is 1e9 bytes):

| term | what it prices |
|---|---|
| `weights` | the base at its loaded precision: 2 bytes a parameter in bf16, 4 under full fine-tuning (fp32 master weights, #339). Under `4bit` and `8bit` loading only the decoder's linear layers are quantized (0.53 and 1.02 bytes a parameter); the embeddings, output head and norms are cast to fp32 for training |
| `adapters` | LoRA's A and B, `r * (in + out)` per adapted module per layer, from the model's own dims and `lora.target_modules` (`auto` is peft's default for the architecture), in fp32 |
| `gradients`, `optimizer` | per trainable parameter: 4 bytes of gradient, and 8 for AdamW, 2 for the 8-bit optimizers, 4 for Adafactor |
| `activations` | what the backward pass keeps per layer; with `gradient_checkpointing`, one hidden-sized tensor per layer plus one layer recomputed. `medium` keeps every other layer; `selective` is priced as no checkpointing, an upper bound. `auto` is resolved on the card as the `sft` trainer resolves it (`full` below 24 GB, `medium` to 80 GB, `selective` above), and only `sft` and `tts` apply the tiers: every other task checkpoints fully |
| `logits` | the output head's logits and the loss over them, 12 bytes an element (14 under `4bit` / `8bit`, whose output head is fp32). Checkpointing does not reduce it, and on a 150k vocabulary it is often the largest term. Not charged for the tasks without an LM head (`classifier`, `reranker`, `cross_encoder`, `embedding`, `reward_model`). With `use_liger` or `use_cut_ce` on `sft` or `tts` it is still charged, since the fused loss may not apply, and the output says it may overstate the peak |
| `dequant` | under `4bit`, the bf16 copy of the largest weight a matmul dequantizes |
| `overhead` | the CUDA context and allocator slack, which `max_memory_allocated()` does not see |

The coefficients come from 39 two-step `soup train` runs on an RTX 3060 Laptop across seven models from 135M to 3B parameters ([`gate-2049-resident-vram.md`](../benchmarks/gate-2049-resident-vram.md)); `tests/test_vram_estimate_calibration.py` holds the estimate to a stated band of every measured peak (0.98 to 1.10 of it for bf16 LoRA, wider for 4-bit, 8-bit and full fine-tuning).

What those numbers do not cover: the rows are seven dense decoders of 135M to 3B parameters, at up to 4,096 tokens and batch 4, on one 6 GB card under WSL2 with one software stack (torch 2.14.0+cu130, transformers 5.17.0). The coefficients are fitted on the same 39 rows the test checks, so the bands show how well they fit those rows, not an error on runs they were not fitted to. Mixture-of-experts models are sized as dense ones, no model of 7B or more was measured, and neither were the `medium` and `selective` checkpointing tiers. The pre-flight refuses runs on this basis; `--allow-oom-attempt` launches one anyway.

The model's shape comes from its `config.json` (a local directory, then the Hugging Face cache, then the Hub unless `HF_HUB_OFFLINE` or `TRANSFORMERS_OFFLINE` is set), and its exact parameter count from the safetensors headers (local files, a cached snapshot, or the Hub's header-only metadata call on `huggingface-hub` 0.20 or later; otherwise it is counted from `config.json`). No weight is downloaded and torch is not imported. Reaching the Hub writes the model's `config.json` into the Hugging Face cache; the size is checked with a HEAD request first, each request has a timeout, and the metadata lookup as a whole has a deadline. The MCP `profile` tool never reaches the Hub: it reads a local directory or the local cache, and prices any other model from a guessed shape. When no `config.json` can be read, the shape is guessed from the parameter count (`model_size_from_name`), the logits term is not counted, and the output says so. That count:
- A local checkpoint path is measured directly from safetensors metadata headers. The files measured are the checkpoint's own: the ones `model.safetensors.index.json` names; without an index `model.safetensors`, else the `model-*-of-*.safetensors` shards, else every `*.safetensors` except `consolidated*`. The Hub download for layer streaming and `soup spectrum scan` fetches by the same rule, so what is downloaded is what is counted. A `consolidated.safetensors` beside the checkpoint (the mistral-inference copy Mistral's repositories ship, which transformers never loads) is not counted, not downloaded for layer streaming or `soup spectrum scan`, not copied out of the HF cache, and not sharded; a directory holding only such a file is refused with its name (#1806).
- A Hub id is parsed: e.g. `72B`, `8x7B` → `46.7` total, `17B-16E` → `109` total, with active-parameter markers (`-A22B`) ignored.
- An id with no size token (e.g. `DeepSeek-V3`, `Kimi-K2`, `GLM-5`, `MiniMax-M2`) still falls back to `7B`, which under-predicts these models (they have 229B to 1T parameters). Point `base:` at a local download to have the checkpoint measured instead.

```python
from soup_cli.utils.hardware_fit import HardwareFitInput, decide_hardware_fit

inp = HardwareFitInput(
    params_b=7.0, seq_len=2048, batch_size=4,
    optimizer="adamw_torch", quant="4bit", peft="lora",
    gradient_checkpointing=True,
)
report = decide_hardware_fit(inp, available_vram_gb=24.0)
print(report.ok, report.reason)
# True | 'fits: peak 9.09 GB + 10% margin <= 24.00 GB available'
```

That input carries no shape, so it is guessed from `params_b` and the logits term is left out; `hardware_fit_input_from_config(cfg)` builds the input `soup train` uses, with the model's own shape. When the run doesn't fit, the report names actionable knobs: `--batch-size halve`, `--quantization 4bit`, `--gradient-checkpointing auto`. Composes with v0.40.3 live CUDA OOM probe (`make_cuda_probe_fn`) which still runs when `auto_batch_size_strategy: probe`.

`mxfp4` and `fp8` are dequantized on load, so they are priced as bf16. `gptq`, `awq`, `aqlm` and `eetq` stay packed and are priced at their own figures (0.55, 0.55, 0.5 and 1.0 bytes a parameter). `soup train` has no figure for `hqq:*` and says it skipped the check ([Quant Menu](performance-and-quantization.md#quant-menu--9-quantization-formats)).


## Shell Completions (`soup completions`)

Tab-completion for `soup` + every subcommand. The generated script is Click/Typer-backed so new commands are picked up automatically.

```bash
# Bash
eval "$(soup completions bash)"        # current shell
soup completions bash >> ~/.bashrc     # permanent

# Zsh
soup completions zsh > "${fpath[1]}/_soup"

# Fish
soup completions fish > ~/.config/fish/completions/soup.fish
```

Command names and option names complete, straight from the live command tree: `rec` + Tab gives `recipes`, and after `soup train`, `--res` + Tab gives `--resume`. Values do not complete yet — recipe names after `soup recipes show`, for instance — because no option or argument is wired to a value completer. Two are ready for that wiring in `soup_cli/utils/completions.py`: `complete_recipe_name`, and `complete_target_modules`, which since v0.71.1 looks the base model's architecture up in its Hugging Face config when that config is already in the local cache (config only, never a download) and offers that architecture's module names, falling back to the canonical Llama-shape defaults (`q_proj` / `k_proj` / `v_proj` / etc.).


## License Advisor (`soup license-advisor`)

Picking a license-clean base for a specific deployment target is a recurring legal-review pain point. v0.64 captures the three most common deploy contexts as a closed allowlist and surfaces the per-license downstream risk.

```bash
# What licenses are safe for a B2C consumer product?
soup license-advisor --target b2c

# Defense — restricted-use community licenses forbidden
soup license-advisor --target defense

# Per-license check: Llama community license + 800M MAU = block (exit 3)
soup license-advisor --target b2c --license llama-3 --monthly-active-users 800000000
```

The Llama-family allowlist is tight (no `.startswith` over-match), so a hypothetical future `llama-permissive-2030` won't false-trigger the 700M-MAU gate. Composes with v0.60 `soup adapters merge --license <id>` for the merge-time conflict gate.

## Troubleshooting

```bash
soup doctor    # GPU, system resources, dependencies, and version in one place
```

- **`ImportError: DLL load failed while importing _C` (Windows).** PyPI's torch
  wheel is CPU-only. Reinstall a CUDA build; `soup doctor` prints the
  `pip install` command for the wheel your driver can run.
- **`soup version` ≠ `pip show soup-cli`** — multiple Python installs; use a virtualenv.
