# Config reference

[← Back to the docs index](README.md)

> **Generated file — do not edit by hand.**
> It is rendered from `SoupConfig` in `src/soup_cli/config/schema.py`. After adding or
> removing a field, or changing its type, default or bounds, run
> `python scripts/gen_reference_docs.py` and commit the result; the test suite fails
> until the page matches. Descriptions are refreshed when a release is cut, so between
> releases the wording here can trail the code.

Every key `soup.yaml` accepts. A key that is not on this page **refuses the load** — see
[Unknown config keys](backends-and-ops.md#unknown-config-keys) for the error and the
nearest-name hint. A key whose description ends "Refused at load when set away from
its default" is declared but read by nothing: leave it out, or at its default.
The guides linked from the [docs index](README.md) explain what the settings do
together; this page is the lookup table.

How to read a table:

- **Key** is the full YAML path: `training.lora.r` is `r:` under `lora:` under
  `training:`.
- **Type** uses `str`, `int`, `float`, `bool`, `list[...]` and `dict[...]`; `null` means
  the key may be left out or set to `null`; `section` is a nested block with a table of
  its own. A value in quotes (`"auto"`) is the one literal accepted next to the type.
- **Default** is written the way YAML reads it (`null`, `true`, `"auto"`, `["a"]`).
  **required** means there is none. *(set at runtime)* marks a default computed on
  the machine, which this page does not print.
- **Constraints** are the rules on the value itself: bounds, lengths, the allowed
  choices. Rules that span keys (a setting valid for one task or backend only) are
  enforced by the schema's validators and named in the description.

Sections:

- [Top-level keys](#top-level-keys)
- [`data`](#data)
- [`training`](#training)
- [`training.lora`](#traininglora)
- [`training.eval_gate`](#trainingeval_gate)
- [`eval`](#eval)
- [`eval.ship`](#evalship)
- [`advise`](#advise)

## Top-level keys

| Key | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `base` | `str` | **required** |  | Base model name or path (HF model ID) |
| `task` | `str` | `"sft"` | one of `"sft"`, `"dpo"`, `"grpo"`, `"ppo"`, `"reward_model"`, `"kto"`, `"orpo"`, `"simpo"`, `"ipo"`, `"bco"`, `"preference"`, `"pretrain"`, `"embedding"`, `"prm"`, `"tts"`, `"classifier"`, `"reranker"`, `"cross_encoder"`, `"distill"`, `"unlearn"`, `"moe_lora_routing"`, `"online_dpo"`, `"asr"` | Training task type. v0.50.0 added 'prm'; v0.52.0 adds 'tts' (TTS fine-tuning), 'classifier' / 'reranker' / 'cross_encoder' (classification heads), and 'distill' (knowledge distillation). v0.61.0 adds 'unlearn' (NPO / SimNPO / RMU). v0.67.0 adds 'moe_lora_routing' (per-token gating over N task LoRAs). |
| `modality` | `str` | `"text"` | one of `"text"`, `"vision"`, `"audio"`, `"audio_out"` | Training modality: text (default), vision (multimodal), audio (audio-input), or audio_out (audio-output — paired with task='tts', v0.52.0). |
| `backend` | `str` | `"transformers"` | one of `"transformers"`, `"unsloth"`, `"mlx"` | Training backend: transformers (default), unsloth (2-5x faster on CUDA), or mlx (Apple Silicon M1-M4) |
| `data` | `section` | **required** |  | Dataset source, format, tokenization, and preprocessing settings. See [`data`](#data). |
| `training` | `section` | section defaults |  | Training hyperparameters, optimizer, scheduler, and hardware settings. See [`training`](#training). |
| `output` | `str` | `"./output"` |  | Output directory for trained model |
| `experiment_name` | `str \| null` | `null` |  | Experiment name for tracking |
| `eval` | `section \| null` | `null` |  | Evaluation configuration for auto-eval after training. See [`eval`](#eval). |
| `advise` | `section \| null` | `null` |  | Pre-flight decision settings consumed by `soup advise`. See [`advise`](#advise). |

## `data`

| Key | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `data.train` | `str \| list[str]` | **required** |  | Path to training data or HF dataset name, or a list of >= 2 local file paths to combine via data.interleave. (#443) |
| `data.format` | `str` | `"auto"` | one of `"alpaca"`, `"sharegpt"`, `"chatml"`, `"dpo"`, `"kto"`, `"llava"`, `"sharegpt4v"`, `"plaintext"`, `"embedding"`, `"audio"`, `"tool-calling"`, `"auto"`, `"prm"`, `"pre_tokenized"`, `"input_output"`, `"video"`, `"multimodal"`, `"raft"`, `"asr"`, `"cross_encoder"` | Data format |
| `data.val_split` | `float` | `0.1` | `>= 0.0`, `<= 0.5` | Validation split ratio |
| `data.max_length` | `int` | `2048` | `>= 64`, `<= 1048576` | Max sequence length in tokens |
| `data.image_dir` | `str \| null` | `null` |  | Base directory for resolving relative image paths in vision datasets |
| `data.audio_dir` | `str \| null` | `null` |  | Base directory for resolving relative audio paths in audio datasets |
| `data.train_on_responses_only` | `bool` | `true` |  | Mask non-assistant tokens with IGNORE_INDEX (-100). When True, only assistant content contributes to the SFT loss. Mirrors LlamaFactory + Axolotl default — replaces TRL's heuristic. (v0.36.0) |
| `data.train_on_messages_with_train_field` | `bool` | `false` |  | Per-message training mask via messages\[i].train: bool. Mutually exclusive with train_on_responses_only. (v0.36.0) |
| `data.chat_template` | `str \| null` | `null` |  | Override the tokenizer chat template. Accepts a registered name (chatml, llama3, qwen2.5, mistral, gemma3, phi4, deepseek-r1) or a raw Jinja string. None = use the tokenizer's shipped template (errors loudly if absent). (v0.36.0) |
| `data.raft_shuffle_seed` | `int \| null` | `null` | `>= 0`, `<= 2147483647` | Seed for the RAFT golden/distractor document shuffle (data.format='raft'). Documents are always shuffled for distractor robustness; this knob fixes which reproducible permutation. None = seed 0. (v0.71.10 #199) |
| `data.raft_epoch_shuffle` | `bool` | `false` |  | Re-permute RAFT golden/distractor documents EACH training epoch (data.format='raft'). When False (default) the document order is baked once at tokenisation time and fixed across epochs; when True the trainer re-composes + re-tokenises rows per epoch with an epoch salt so the model cannot memorise a fixed golden-doc slot. (v0.71.17 #253) |
| `data.replay` | `str \| null` | `null` |  | Path to an OLD dataset to interleave into training as continual-learning rehearsal, so fine-tuning on a new task does not erase the previous one. Rows are mixed into train ONLY (never val, which stays pure new-task). sft / pretrain only; incompatible with packing / multipack. (v0.71.36) |
| `data.replay_ratio` | `float` | `0.1` | `> 0.0`, `<= 0.5` | Fraction of the FINAL mixed train set that is replay rows: n_replay = round(r/(1-r) \* n_new). At 0.1 over 1000 new rows that is 111 replay rows -> 1111 total -> 10.0%. (v0.71.36) |
| `data.replay_seed` | `int \| null` | `null` | `>= 0`, `<= 2147483647` | Seed for the replay sample + interleave. None = seed 0. (v0.71.36) |
| `data.video_dir` | `str \| null` | `null` |  | Base directory for resolving relative video paths in video datasets. Mirrors image_dir / audio_dir. (v0.42.0). Refused at load when set away from its default (v0.77, #808). |
| `data.tokenized_path` | `str \| null` | `null` |  | Path to a pre-tokenized cache produced by `soup data preprocess`. When set, the trainer skips the tokenize stage and reads tensors directly. Mirrors LF tokenized_path / Axolotl `empty` type. (v0.42.0) |
| `data.streaming` | `bool` | `false` |  | Pass-through to HF datasets `streaming=True`. Use for datasets that don't fit on disk. Pairs with `buffer_size`. (v0.42.0) |
| `data.buffer_size` | `int \| null` | `null` |  | Shuffle buffer size for streaming datasets. None = HF default. Bounds \[1, 1_000_000]. (v0.42.0) |
| `data.shards` | `int \| null` | `null` |  | Number of shards for HF dataset splits (axolotl `shards`). Bounds \[1, 1024]. (v0.42.0) |
| `data.interleave` | `str \| dict \| null` | `null` |  | Multi-dataset interleave strategy: 'concat' / 'under' / 'over' / {strategy: 'probs', probs: \[...]}. (v0.42.0) |
| `data.mask_history` | `bool` | `false` |  | LF mask_history — mask all but the last assistant turn during loss computation. (v0.42.0) |
| `data.train_on_prompt` | `bool` | `false` |  | LF train_on_prompt — include the prompt tokens in the loss. Inverse of train_on_responses_only. (v0.42.0) |
| `data.eval_on_each_dataset` | `bool` | `false` |  | LF eval_on_each_dataset — when interleaving, run eval on every constituent dataset separately. (v0.42.0). Refused at load when set away from its default (v0.77, #808). |
| `data.split_thinking` | `bool` | `false` |  | Axolotl split_thinking — separate `<think>` reasoning blocks from the final answer for fine-grained masking. Qwen3-style. (v0.42.0). Refused at load when set away from its default (v0.77, #808). |
| `data.image_min_pixels` | `int \| null` | `null` |  | Per-image min pixel count for vision data. (v0.42.0). Refused at load when set away from its default (v0.77, #808). |
| `data.image_max_pixels` | `int \| null` | `null` |  | Per-image max pixel count for vision data. (v0.42.0). Refused at load when set away from its default (v0.77, #808). |
| `data.image_resize_algorithm` | `str \| null` | `null` | one of `"nearest"`, `"bilinear"`, `"bicubic"`, `"lanczos"` | Pillow resize algorithm for image preprocessing. (v0.42.0). Refused at load when set away from its default (v0.77, #808). |
| `data.video_fps` | `float \| null` | `null` |  | Target frames-per-second for video preprocessing. (v0.42.0). Refused at load when set away from its default (v0.77, #808). |
| `data.video_maxlen` | `int \| null` | `null` |  | Max number of frames per video clip. Bounds (0, 4096]. (v0.42.0). Refused at load when set away from its default (v0.77, #808). |
| `data.add_new_tokens` | `list[str] \| null` | `null` |  | Add these tokens to the tokenizer vocab + resize embeddings. Cap 10_000 entries; per-token &lt;= 256 chars; no duplicates. (v0.42.0) |
| `data.new_special_tokens` | `list[str] \| null` | `null` |  | Like add_new_tokens but registered as additional_special_tokens so they are not split by the tokenizer. (v0.42.0) |
| `data.resize_vocab` | `bool` | `false` |  | Resize the model's input/output embedding matrix when add_new_tokens / new_special_tokens grew the vocab. (v0.42.0). Refused at load when set away from its default (v0.77, #808). |
| `data.extend_conversation` | `bool` | `false` |  | Unsloth-style conversation extension — extend the last assistant turn with N more tokens for 'continue' prompts. (v0.42.0). Refused at load when set away from its default (v0.77, #808). |
| `data.skip_prepare_dataset` | `bool` | `false` |  | Axolotl skip_prepare_dataset — escape hatch when the input is already in the trainer's expected schema. (v0.42.0). Refused at load when set away from its default (v0.77, #808). |
| `data.remove_unused_columns` | `bool` | `false` |  | HF Trainer remove_unused_columns. No trainer reads this field; the trainers that set it pass False so a custom collator can still see the extra columns. An explicit `true` is refused at config load (it warned for one release first); delete the line. (v0.42.0; #759) |
| `data.prompt_strategy` | `str \| null` | `null` |  | Axolotl-style 'module.path:function_name' Python transform, resolved lazily and applied to each row during SFT formatting. |
| `data.forget_set` | `str \| null` | `null` |  | Path or HF dataset name for the forget set (rows to unlearn). Required when task='unlearn'. Null-byte rejected, capped at 4096 chars. Containment is deferred to the trainer-side loader so HF dataset IDs (e.g. ``locuslab/TOFU``) still pass schema. (v0.61.0) |
| `data.retain_set` | `str \| null` | `null` |  | Path or HF dataset name for the retain set (rows whose performance must be preserved). Optional but recommended — NPO/SimNPO/RMU all degrade without one. Same validation as forget_set. (v0.61.0) |

## `training`

| Key | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `training.epochs` | `int` | `3` | `>= 1` | Number of training epochs |
| `training.lr` | `float` | `2e-05` | `> 0` | Learning rate |
| `training.batch_size` | `int \| "auto"` | `"auto"` |  | Batch size. 'auto' = find max that fits in memory. |
| `training.auto_batch_size_strategy` | `str` | `"auto"` | one of `"auto"`, `"static"`, `"probe"` | How to pick the auto batch size: 'static' (fast formula), 'probe' (real OOM try/halve loop), 'auto' (probe on CUDA, static on CPU). Default 'auto' (v0.36.0). |
| `training.gradient_accumulation_steps` | `int` | `4` | `>= 1` | Number of micro-batches whose gradients are summed before each optimizer update. Effective batch size per device is batch_size \* gradient_accumulation_steps (times the number of GPUs). |
| `training.seed` | `int \| null` | `null` | `>= 0`, `<= 4294967295` | Training seed (weight init of new params, data order, dropout). Reaches TrainingArguments.seed and the multipack sampler. Unset = HF's default of 42. |
| `training.data_seed` | `int \| null` | `null` | `>= 0`, `<= 4294967295` | Separate seed for data sampling only (TrainingArguments.data_seed). Set it alongside `seed` to vary data order while holding initialisation fixed. Unset = follow `seed`. |
| `training.warmup_ratio` | `float` | `0.03` | `>= 0.0`, `<= 0.5` | Fraction of total training optimizer steps used for linear learning rate warmup. Resolved over real optimizer steps. |
| `training.weight_decay` | `float` | `0.01` | `>= 0.0` | Weight decay factor (decoupled, AdamW-style) applied to weights during optimization; 0 disables it. |
| `training.max_grad_norm` | `float` | `1.0` | `> 0` | Maximum gradient norm for gradient clipping. |
| `training.lora` | `section` | section defaults |  | LoRA (Low-Rank Adaptation) adapter architecture and target modules. See [`training.lora`](#traininglora). |
| `training.quantization` | `str` | `"4bit"` | one of `"4bit"`, `"8bit"`, `"none"`, `"gptq"`, `"awq"`, `"hqq:1bit"`, `"hqq:2bit"`, `"hqq:3bit"`, `"hqq:4bit"`, `"hqq:5bit"`, `"hqq:6bit"`, `"hqq:8bit"`, `"aqlm"`, `"eetq"`, `"mxfp4"`, `"fp8"`, `"bitnet_1.58"` | Quantization (v0.38.0 — Quant Menu): 4bit (BNB QLoRA), 8bit (BNB), none, gptq / awq (load pre-quantized checkpoint, train LoRA on top), hqq:Nbit (HQQ 1-8 bit, N in {1..6, 8}), aqlm (extreme 2-bit), eetq (8-bit fast), mxfp4 (MXFP4 pre-quantized base, dequantized on load), fp8 (load FP8 checkpoint with dequantize-on-load), bitnet_1.58 (reserved until BitNet trainer routing is available). A task whose trainer never quantises the base (distill, classifier, reranker, cross_encoder, prm, moe_lora_routing, unlearn, asr) resolves an unset value to none, and an explicit value other than none, 4bit and 8bit included, is refused at load (#795). |
| `training.gptq_disable_exllama` | `bool` | `true` |  | v0.38.0 — disable exllama backend for GPTQ. PEFT requires triton backend; exllama silently breaks adapter training. |
| `training.bnb_4bit_quant_storage` | `str \| null` | `null` | one of `"uint8"`, `"float16"`, `"bfloat16"`, `"float32"` | v0.38.0 — BNB 4-bit storage dtype. FSDP+QLoRA resolves this automatically to the effective floating compute dtype; outside FSDP, None keeps BNB's legacy uint8 default. |
| `training.quantization_aware` | `bool \| "fp8" \| "quest"` | `false` |  | Quantization-Aware Training. False=off, True=refused at load (int8 QAT is not implemented, #1222), 'fp8'=FP8 training on H100/B100 (v0.28.0), 'quest'=experimental mixed W4/A4+A16 QuEST route (#674). |
| `training.fp8_recipe` | `str` | `"tensorwise"` | one of `"tensorwise"`, `"rowwise"`, `"rowwise_with_gw_hp"` | FP8 scaling recipe (only used when quantization_aware='fp8'). 'tensorwise' (fastest, default), 'rowwise' (more accurate, CUTLASS rowwise), 'rowwise_with_gw_hp' (most accurate, grad_weight in high precision). (v0.28.1). |
| `training.optimizer` | `str` | `"adamw_torch"` |  | Optimizer name (v0.41.0 expanded the allowlist: HF-native, bnb-8bit, lomo/adalomo, apollo_adamw, grokadamw, schedule_free). Note: ten names transformers 5.x rejects in TrainingArguments (e.g. badam, adam_mini, muon, dion, came_pytorch, ao_adamw\_{fp8,4bit,8bit}) are refused at config load (#1269); adamw_hf and muon still work on backend: mlx. See soup_cli.utils.optimizer_zoo.SUPPORTED_OPTIMIZERS for the full list. |
| `training.lr_groups` | `list[dict[str, str \| float]] \| null` | `null` |  | Staged, not applied: a per-module LR override (list of {pattern, lr} entries or a {pattern: lr} dict, capped at 32) that no optimizer reads, so setting it is refused at load (#761). (v0.41.0). Refused at load when set away from its default (v0.77, #808). |
| `training.expand_layers` | `int \| null` | `null` | `>= 1`, `<= 64` | LLaMA Pro: append N zero-init transformer blocks and freeze the original ones, so only the new blocks train. A LoRA rank is refused with it, since get_peft_model would freeze the new blocks again (#1326): set lora.r: 0 and quantization: none on task: sft. |
| `training.freeze_trainable_layers` | `int \| null` | `null` |  | LLaMA Pro: applies only together with expand_layers, must equal it, and is refused otherwise. It freezes every parameter of the original model so only the appended blocks train; it does not select top-N or bottom-N layers. Magnitude capped at 1000. (v0.41.0) |
| `training.use_mod` | `bool` | `false` |  | Enable Mixture-of-Depths selective-token routing (arXiv:2404.02258). Each decoder layer gets a router that passes only the top-k tokens (k = floor(seq_len \* mod_capacity_factor)) through the block. Applied on the resident transformers LoRA path for sft and tts (not vision, audio, layer streaming, LISA, Spectrum, lora.r=0, unsloth, or mlx) and on transformers pretrain including LISA; refused at config load anywhere else (#1752). Unsupported architectures warn and skip. (v0.71.12 #84) |
| `training.mod_capacity_factor` | `float` | `0.125` | `> 0.0`, `<= 1.0` | Fraction of tokens routed through each block when use_mod=True. Bounded (0, 1]; default 0.125 (per the MoD paper). (v0.71.12 #84) |
| `training.load_in_8bit` | `bool \| null` | `null` |  | Friendly alias for quantization='8bit' / 'none'. When True, rewrites quantization to '8bit' if currently 'none'/'4bit'. Conflicts with load_in_16bit. On a task whose trainer never quantises the base (see quantization), true is refused at load (#795). (v0.41.0) |
| `training.load_in_16bit` | `bool \| null` | `null` |  | Friendly alias: when True, sets quantization='none' (full bf16/fp16 LoRA). Conflicts with load_in_8bit. (v0.41.0) |
| `training.scheduler` | `str` | `"cosine"` |  | LR scheduler type |
| `training.save_steps` | `int` | `100` |  | Save checkpoint every N steps |
| `training.logging_steps` | `int` | `10` |  | Log metrics every N steps |
| `training.eval_steps` | `int \| null` | `null` | `>= 1` | Evaluate the validation split every N optimizer steps (counted after gradient accumulation, like save_steps). Unset = evaluate at the end of every epoch. grpo evaluates only when this is set, because an evaluation there generates completions. (#1223) |
| `training.dpo_beta` | `float` | `0.1` | `> 0` | DPO beta — KL penalty coefficient |
| `training.kto_beta` | `float` | `0.1` | `> 0` | KTO beta — KL penalty coefficient |
| `training.orpo_beta` | `float` | `0.1` | `> 0` | ORPO beta — odds ratio weight |
| `training.simpo_gamma` | `float` | `0.5` | `>= 0` | SimPO gamma — reward margin term |
| `training.cpo_alpha` | `float` | `1.0` | `> 0` | CPO/SimPO alpha — NLL loss weight |
| `training.ipo_tau` | `float` | `0.1` | `> 0` | IPO tau — regularization strength |
| `training.bco_beta` | `float` | `0.1` | `> 0` | BCO beta — KL penalty coefficient |
| `training.preference_loss` | `str \| null` | `null` | one of `"dpo"`, `"simpo"`, `"orpo"`, `"ipo"`, `"bco"` | Preference loss for task='preference'. One of: dpo, simpo, orpo, ipo, bco. Mutually exclusive with task in {dpo, simpo, orpo, ipo, bco}. |
| `training.dpo_beta_schedule` | `str \| null` | `null` | one of `"linear"`, `"cosine"`, `"exponential"` | Anneal DPO β over training. None = constant β (default). Requires dpo_beta_end. DPO-family tasks only (dpo, ipo, preference+dpo). |
| `training.dpo_beta_end` | `float \| null` | `null` | `> 0` | Target β at the end of training when dpo_beta_schedule is set. Must be > 0. The starting β is dpo_beta. |
| `training.dpo_ref_regen_epochs` | `int \| null` | `null` | `>= 1`, `<= 1000` | Replace the frozen ref adapter with the current active adapter every N epochs. Copies .default. adapter weights into .ref. under torch.no_grad(). Requires LoRA (lora.r >= 1); full fine-tuning is not supported. |
| `training.preference_loss_weights` | `dict[str, float] \| null` | `null` |  | Weighted blend of preference losses, e.g. {'dpo': 0.7, 'bco': 0.3}. Each weight ∈ (0, 1]; weights must sum to 1.0 (±1e-6). All keys must be members of {dpo, simpo, orpo, ipo, bco}. Requires task='preference'; mutually exclusive with preference_loss (the scalar form). Capped at 5 components (the supported set). |
| `training.grpo_beta` | `float` | `0.1` | `>= 0.0`; finite | GRPO beta - KL penalty coefficient (default 0.1). Set to 0 to disable the KL penalty entirely (KL-free recipes like DAPO and Dr. GRPO, #1247). |
| `training.num_generations` | `int` | `4` | `>= 2` | Number of generations per prompt for GRPO |
| `training.reward_fn` | `str \| null` | `"accuracy"` |  | Reward function: 'accuracy', 'format', 'verifiable', 'judge_rank' (GRPO only, an LLM judge ranks each prompt's completions; see grpo_judge, #2047), a path to a custom .py file, or a comma-separated ensemble of the above (e.g. 'accuracy,format') — the comma form is GRPO-only (v0.71.40). |
| `training.verifiable_domain` | `str \| null` | `null` | one of `"math"`, `"code"`, `"json_schema"` | RLVR verifiable reward domain: math \| code \| json_schema. Required when reward_fn='verifiable'. |
| `training.prm_reward` | `str \| null` | `null` |  | Path (or HF id) to a Soup-trained PRM (task='prm') used as the GRPO per-step reward (v0.71.30). When set, the PRM replaces reward_fn. Requires task='grpo', backend='transformers', modality='text'. |
| `training.prm_aggregate` | `str` | `"min"` | one of `"min"`, `"prod"`, `"last"` | How PRM per-step scores fold into one reward: min (weakest-link, default) \| prod \| last. Only meaningful when prm_reward is set. NOTE: 'prod' assumes per-step scores are bounded in \~\[0,1] (a probability of step-correctness). Soup's PRM head is trained with unconstrained MSE regression, so 'prod' can blow up / flip sign on unbounded labels — prefer the default 'min' unless your PRM labels are calibrated to \[0,1]. |
| `training.grpo_judge` | `str \| null` | `null` |  | Judge URL for reward_fn judge_rank (ollama://model \| https://host/model \| http://localhost:port/model), the form online_dpo_judge takes. Required with judge_rank (#2047). |
| `training.grpo_judge_rubric` | `str \| null` | `null` |  | Rubric YAML for judge_rank, under the working directory. Only each criterion's name and description are used: a ranking scores from 0 to 1, so weights and scale are ignored. Unset = Soup's default criteria (helpfulness, accuracy, safety). |
| `training.grpo_judge_max_tokens` | `int \| null` | `null` | `>= 1`, `<= 32768` | Judge reply budget per judge_rank call. Unset = 1024. |
| `training.grpo_judge_max_calls` | `int \| null` | `null` | `>= 1` | Cost guard for judge_rank: the run is refused before the model loads when it could make more judge calls than this ((train rows x epochs + held-out rows x evaluations) x 2 for the retry). Required with judge_rank; no default. |
| `training.grpo_judge_max_failure_rate` | `float \| null` | `null` | `>= 0.0`, `<= 1.0` | judge_rank stops the run when more than this share of its training groups got no scores (checked from the 20th group). Unset = 0.2. |
| `training.online_dpo_judge` | `str \| null` | `null` |  | Judge URL (ollama://model \| https://... \| http://localhost) for task='online_dpo'. Mutually exclusive with reward_model. |
| `training.online_dpo_loss_type` | `str` | `"sigmoid"` | one of `"sigmoid"`, `"ipo"` | Online DPO loss type (maps to OnlineDPOConfig.loss_type). |
| `training.online_dpo_max_new_tokens` | `int` | `64` | `>= 1`, `<= 4096` | Max new tokens generated per online-DPO step. |
| `training.asr_language` | `str \| null` | `null` |  | Target language for task='asr' (Whisper), e.g. 'en' or 'spanish'. Sets the forced decoder prompt; None uses the model default / language detection. |
| `training.asr_task` | `str` | `"transcribe"` | one of `"transcribe"`, `"translate"` | Whisper decoding objective for task='asr': 'transcribe' (same language) or 'translate' (to English). |
| `training.asr_lora` | `bool` | `false` |  | Opt-in LoRA for task='asr' (adapts q/v attention projections). Default False = full fine-tune (tiny Whisper fits a 4 GB GPU). Mirrors classifier_lora — the standard training.lora block still supplies r/alpha/dropout when enabled. |
| `training.grpo_variant` | `str \| null` | `null` | one of `"standard"`, `"gspo"`, `"dapo"`, `"dr_grpo"`, `"bnpo"`, `"two_sided"`, `"rft"` | GRPO objective variant: standard \| gspo \| dapo \| dr_grpo \| bnpo \| two_sided \| rft. Defaults to None (legacy GRPO). Requires task='grpo'; the selected loss is installed by the GRPO trainer. |
| `training.grpo_delta` | `float \| null` | `null` | `> 0.0`, `<= 1.0` | Symmetric clipping radius for grpo_variant='two_sided' (required) or grpo_variant='gspo' (optional sequence clipping radius, defaults to 0.2). Rejected for all other variants. |
| `training.grpo_fp16` | `bool` | `false` |  | Force FP16 mixed precision for GRPO/RL (unsloth parity). On CUDA, the GRPO trainer forwards fp16=True and bf16=False; on non-CUDA devices this flag does not enable FP16. |
| `training.long_context_grpo` | `bool` | `false` |  | Enable long-context GRPO (unsloth: 380K B200 / 110K H100). Compatibility validation only: no trainer consumes this flag yet. Requires task='grpo' on a non-mlx backend and use_ring_attention=False. Refused at load when set away from its default (v0.77, #808). |
| `training.vllm_sleep_mode` | `bool` | `false` |  | Enable vLLM sleep/standby between rollouts (memory savings during the optimisation step). Requires backend in {transformers, unsloth}; forwarded to compatible TRL and vLLM runtimes by the GRPO trainer. |
| `training.rollout_backend` | `str \| null` | `null` | one of `"art"`, `"ruler"`, `"nemo_gym"`, `"openenv"` | Multi-turn agent rollout backend (unsloth / axolotl parity): art (OpenPipe ART) / ruler / nemo_gym / openenv. Requires task='grpo'. openenv runs live (v0.71.21 #125) via training.rollout_func; art/ruler/nemo_gym are lazy-import gated. |
| `training.rollout_func` | `str \| null` | `null` |  | OpenEnv rollout function as 'module.path:function_name' (v0.71.21 #125). Requires rollout_backend='openenv'. The callable receives the seed prompts list and returns rollout rows ({'prompt': str\|messages, 'answer'?: str}) that replace the GRPO prompt dataset. Trusted-input policy: names operator-controlled code (mirrors data.prompt_strategy). |
| `training.ref_model_ema_alpha` | `float \| null` | `null` | `> 0.0`, `<= 1.0` | Exponential moving average coefficient for ref-model sync (policy → reference). Must be in (0, 1]. None = disabled. Axolotl parity. |
| `training.replay_buffer_size` | `int \| null` | `null` | `>= 1`, `<= 1000000` | Bounded replay buffer size for GRPO rollouts. None = disabled. Axolotl parity. |
| `training.async_grpo_prefetch` | `bool` | `false` |  | Overlap rollout + train via async prefetch (axolotl). Requires backend in {transformers, unsloth}. |
| `training.tis_threshold` | `float \| null` | `null` | `> 0.0`, `<= 100.0` | Truncated importance sampling threshold (unsloth, axolotl). Must be in (0, 100]. None = disabled. |
| `training.mask_truncated_completions` | `bool` | `false` |  | Mask out truncated completions when computing the policy gradient (paired with tis_threshold). Unsloth + axolotl parity. |
| `training.defer_rerolling` | `bool` | `false` |  | Defer re-rolling identical prompts across optimisation steps (axolotl). Saves rollouts on repeat prompts. |
| `training.skip_zero_advantage` | `bool` | `false` |  | Skip backward pass on samples whose advantage is exactly zero (axolotl). Avoids wasted compute on no-signal samples. |
| `training.off_policy_mask_threshold` | `float \| null` | `null` | `>= 0.0`, `<= 1.0` | Off-policy mask threshold for token/sequence gating (axolotl). Must be in \[0, 1]. None = disabled. |
| `training.vision_grpo` | `bool` | `false` |  | Enable Vision RL / VLM RL — extends GRPO/PPO to vision modality (Qwen2-VL / Pixtral / InternVL). Requires modality='vision', task in {grpo, ppo}, backend in {transformers, unsloth}. Compatibility validation is available, but no trainer consumes this flag yet. Refused at load when set away from its default (v0.77, #808). |
| `training.hub` | `str` | `"hf"` | one of `"hf"`, `"modelscope"`, `"modelers"` | Training-time model-download hub. 'hf' (default), 'modelscope' (China-hosted; mirrors most Llama/Qwen/etc.), 'modelers' (Openmind hub). `soup push --hub` selects its upload destination independently of this field. |
| `training.tts_family` | `str \| null` | `null` | one of `"orpheus"`, `"sesame_csm"`, `"llasa"`, `"spark"`, `"oute"` | TTS model family — required when task='tts'. orpheus, llasa, spark and oute use codec-string SFT; sesame_csm uses a native multimodal trainer with raw audio and 32 parallel Mimi codebooks. |
| `training.tts_emotion` | `str \| null` | `null` |  | Optional emotion tag for emotion-conditioned families (Orpheus / Oute). Allowlisted per-family. (v0.52.0) |
| `training.num_labels` | `int \| null` | `null` | `>= 1`, `<= 1024` | Number of output labels for task in (classifier, reranker, cross_encoder). Required when task is one of those. (v0.52.0) |
| `training.classifier_kind` | `str \| null` | `null` | one of `"single_label"`, `"multi_label"` | Sequence-classification head kind: single_label (default for task='classifier') or multi_label. (v0.52.0) |
| `training.label_names` | `list[str] \| null` | `null` |  | Optional human-readable label names. Length must match num_labels. Capped at 1024 entries. (v0.52.0) |
| `training.mole_task_adapters` | `list[str] \| null` | `null` |  | Paths (HF ids or local dirs) to the N pre-trained task LoRA adapters routed over by the MoLE gate. Required when task='moe_lora_routing'; 2-64 entries, deduplicated. The base model + every task adapter stay frozen — only the gate trains. (v0.71.12 #222) |
| `training.mole_top_k` | `int \| null` | `null` |  | Number of task adapters each token is routed to (sparse top-k dispatch). Defaults to num_task_adapters (dense) when unset. 1 &lt;= top_k &lt;= len(mole_task_adapters). (v0.71.12 #222) |
| `training.mole_temperature` | `float \| null` | `null` |  | Softmax temperature for the MoLE router (default 1.0). (1e-6, 100.0]. (v0.71.12 #222) |
| `training.teacher_model` | `str \| null` | `null` |  | Teacher model HF id or local path — required when task='distill'. Loaded frozen by the distillation trainer. Null-byte rejected, capped at 512 chars. |
| `training.distill_divergence` | `str \| null` | `null` | one of `"forward_kl"`, `"reverse_kl"`, `"js"` | Divergence used for distillation loss. 'kl' is an alias for 'forward_kl' (canonical form). (v0.52.0) |
| `training.distill_temperature` | `float \| null` | `null` |  | Softmax temperature applied to teacher and student logits before the divergence. Bounded \[0.05, 100.0]. (v0.52.0) |
| `training.distill_mode` | `str` | `"token"` | one of `"token"`, `"sequence"` | Distillation mode. 'token' (default, v0.53.2) = column-aligned logit KL — requires a shared tokenizer (or set uld_strategy for the cross-tokenizer logit path). 'sequence' (v0.71.12) = sequence-level KD: the teacher GENERATES a completion per prompt and the student does plain CE on the re-tokenised output, which works across ANY tokenizer pair (e.g. Llama-3 student / Qwen-2 teacher). |
| `training.distill_chunk_size` | `int \| null` | `null` |  | Token chunk size for evaluating the distillation divergence kernel. Chunks the active supervised tokens to reduce peak memory retention. Unset (None) processes all tokens in a single chunk. (#722) |
| `training.distill_checkpoint` | `bool` | `false` |  | Enable non-reentrant activation checkpointing per token chunk during distillation divergence computation to minimize autograd retained memory. (#722) |
| `training.classifier_lora` | `bool` | `false` |  | Wrap the sequence-classification head with LoRA (task_type='SEQ_CLS') instead of full fine-tuning. Opt-in (default False preserves the v0.53.2 full-finetune behaviour). Reuses the training.lora block (r / alpha / dropout / target_modules). Only honored for task in (classifier, reranker, cross_encoder). (v0.71.12 #146) |
| `training.ebft_variant` | `str \| null` | `null` | one of `"structured"`, `"strided"` | Refused at config load (#1230): EBFT is not yet a distinct objective. The added term had no causal shift, so it rewarded copying the input token, and shifted it duplicates the cross-entropy. |
| `training.ebft_temperature` | `float \| null` | `null` |  | Sampling temperature for EBFT energy proxy. Bounded \[1e-4, 100.0]. Only applies with ebft_variant, which is refused at config load (#1230). (v0.52.0) |
| `training.gdpo_variant` | `str \| null` | `null` | one of `"standard"`, `"length_normalized"`, `"margin"` | Refused at config load (#1309): supported TRL versions lack the DPO loss hook. Remove this field; use plain task: dpo for standard, or task: simpo as the nearest (not identical) length-normalized objective. |
| `training.moe_expert_quant` | `str \| null` | `null` | one of `"nf4"`, `"int8_rowwise"` | Per-expert quantization for fused-MoE Linear blocks. Requires moe_lora=true. (v0.52.0) |
| `training.train_router_only` | `bool` | `false` |  | Freeze every expert + train only the gating router (unsloth MoE recipe). Requires moe_lora=true. (v0.52.0) |
| `training.reasoning_effort` | `str \| null` | `null` | one of `"low"`, `"medium"`, `"high"` | gpt-oss train-time reasoning effort level. Injected as a prompt prefix by the SFT data formatter; task sft, tts or distill only. |
| `training.train_on_eot` | `bool` | `false` |  | Include explicit EOT / EOS control tokens in the SFT loss (axolotl ``train_on_eot``). Default False matches HF Trainer convention. Applied by the SFT data formatter; task sft, tts or distill only. (v0.52.0) |
| `training.kv_cache_type` | `str \| null` | `null` | one of `"q8_0"`, `"bf16"`, `"f16"`, `"fp8"` | KV-cache element type for inference (q8_0 / bf16 / f16 / fp8). Applied by soup serve on the transformers backend; backend and hardware compatibility are validated before model loading. |
| `training.fp8_attention` | `bool` | `false` |  | Extend the v0.28.0 FP8 menu to FP8 attention (axolotl-parity flag). Requires quantization_aware='fp8'. DPO-family, GRPO/PPO, pre-training, reward-model, and embedding wrappers route it through the shared advanced-precision setup; the default SFT path does not consume it. |
| `training.nvfp4` | `bool` | `false` |  | Blackwell-only NVFP4 training (unsloth + axolotl). DPO-family, GRPO/PPO, pre-training, reward-model, and embedding wrappers route it through torchao conversion; the default SFT path does not consume it. |
| `training.unsloth_bnb_4bit` | `bool` | `false` |  | Promote Unsloth Dynamic 4-bit from 'inferable' to a native flag. Requires backend='unsloth' and quantization='4bit'. (v0.53.0) |
| `training.bnb_4bit_use_double_quant` | `bool \| null` | `null` |  | Apply BNB 4-bit double-quantization (LF / Axolotl parity). Unset means the shipped default (on, to match every 4-bit load path); set false to disable it. Only meaningful when quantization='4bit'. (v0.53.0) |
| `training.llm_int8` | `bool` | `false` |  | Explicit 8-bit LLM.int8 alias for quantization='8bit'. When True, requires quantization='8bit'. (v0.53.0) |
| `training.quantize_ref_model` | `bool` | `false` |  | Apply the same Quant Menu config to the reference model (DPO/IPO/SimPO/ORPO/BCO ref model) — extends v0.40.5. (v0.53.0). Refused at load when set away from its default (v0.77, #808). |
| `training.quantize_reward_model` | `bool` | `false` |  | Apply the same Quant Menu config to the reward model (PPO/reward_model task) — extends v0.40.5. (v0.53.0) |
| `training.echo_trap_enabled` | `bool` | `false` |  | Enable RAGEN-style echo-trap detection during multi-turn agent RL. Requires task in {'grpo', 'ppo'} on a non-mlx backend; installs the shared RL signal callback. |
| `training.echo_trap_threshold` | `float` | `0.6` | `>= 0.0`, `<= 1.0` | Threshold on the aggregate echo signal. Above this = TRAP. Bounded \[0.0, 1.0]. (v0.70.0) |
| `training.echo_trap_halt` | `bool` | `false` |  | Auto-halt training on TRAP verdict. Requires echo_trap_enabled=True. (v0.70.0) |
| `training.echo_trap_tokenizer_aware` | `bool` | `false` |  | Use tokenizer-id n-grams for echo-trap scoring instead of whitespace tokens. More sensitive to subword repetition but bound to the active tokenizer vocabulary. Requires echo_trap_enabled=True. (v0.70.x) |
| `training.rl_checkpoint_save_every_steps` | `int \| null` | `null` | `>= 1`, `<= 10000000` | Save an RL-aware mid-epoch checkpoint every N steps. None = use HF Trainer's per-epoch checkpoint only. Requires task in {'grpo', 'ppo'}; the callback persists the policy adapter, optional optimizer state, and a manifest. |
| `training.rl_checkpoint_keep_last` | `int` | `3` | `>= 1`, `<= 100` | Number of recent RL checkpoints to retain. Older ones are pruned at write time. (v0.70.0) |
| `training.rl_checkpoint_include_optimizer` | `bool` | `true` |  | Include AdamW / Lion optimizer state in the mid-epoch RL checkpoint. (v0.70.0) |
| `training.rl_checkpoint_include_ref_model` | `bool` | `false` |  | Not saved yet (#808): the RL checkpoint holds no reference model state whatever this says, and its manifest reports has_ref_model: false. (v0.70.0). Refused at load when set away from its default (v0.77, #808). |
| `training.rl_checkpoint_include_rollout_buffer` | `bool` | `false` |  | Not saved yet (#808): the RL checkpoint holds no rollout or replay buffer whatever this says, and its manifest reports has_rollout_buffer: false. (v0.70.0). Refused at load when set away from its default (v0.77, #808). |
| `training.minillm_enabled` | `bool` | `false` |  | Enable MiniLLM-style on-policy distillation (Gu et al. 2024). Requires task='distill' on a non-mlx backend; installs the teacher-mixed sampling and reverse-KL loss path. |
| `training.minillm_teacher_mix_ratio` | `float` | `0.0` | `>= 0.0`, `<= 1.0` | Teacher mixture used by MiniLLM. Offline (minillm_on_policy=false): weight of the teacher distribution in the reverse-KL target (ratio \* teacher + (1 - ratio) \* stopgrad(student)). 0.0 is rejected when MiniLLM is enabled because the teacher then has zero weight. On-policy: probability of sampling from the teacher at rollout time; 0.0 = student-only sampling, and the loss is still KL(student \|\| teacher). Typical range 0.2-0.5. (v0.70.0; offline zero rejected in #692) |
| `training.minillm_length_normalize` | `bool` | `true` |  | Length-normalise the rollout log-probability before the reverse-KL term. Prevents long completions from dominating the gradient. (v0.70.0) |
| `training.minillm_pretrain_anchor_weight` | `float` | `0.0` | `>= 0.0`, `<= 1.0` | Weight on the pretrain-loss anchor term (SFT on a small pretrain corpus). Prevents drift away from coherent language. Requires minillm_pretrain_anchor_path when > 0. (v0.70.0) |
| `training.minillm_pretrain_anchor_path` | `str \| null` | `null` |  | Path to the pretrain JSONL used by the anchor term. Required when minillm_pretrain_anchor_weight > 0. Null-byte rejected; capped at 4096 chars. (v0.70.0) |
| `training.minillm_on_policy` | `bool` | `false` |  | Use the TRUE on-policy MiniLLM teacher-mixed rollout (Gu et al. 2024 §3.1): sample a fresh autoregressive rollout per step (per-token teacher/student mix) then compute length-normalised reverse-KL on it. Default off = the cheap offline distribution blend. Requires minillm_enabled=True. (v0.71.18 #257) |
| `training.minillm_rollout_length` | `int \| null` | `null` | `>= 1`, `<= 512` | On-policy rollout length (number of generated tokens per step). When None the distill trainer auto-derives min(max_length, 32) for the consumer-GPU budget. The autoregressive loop re-forwards the full growing prefix each step (\~O(L^2) graph), so keep this small. Requires minillm_on_policy=True. (v0.71.18 #257) |
| `training.uld_strategy` | `str \| null` | `null` | one of `"wasserstein"`, `"topk_align"`, `"wasserstein_aligned"` | Cross-tokenizer distillation strategy: 'wasserstein' (no alignment needed), 'topk_align' (requires uld_top_k), or 'wasserstein_aligned' (token-sequence alignment for fully disjoint tokenizers, v0.71.18 #258). Requires task='distill' on a non-mlx backend. |
| `training.uld_top_k` | `int \| null` | `null` | `>= 1`, `<= 262144` | Top-K teacher logits to align (uld_strategy='topk_align' only). Bounded \[1, 262144] to cap pathological vocabs. (v0.70.0) |
| `training.reward_hack_detector` | `str \| null` | `null` | one of `"info_rm"`, `"rm_ensemble"` | Reward-hacking detector for GRPO/PPO. 'info_rm' tracks InfoRM cluster-separation across training; 'rm_ensemble' tracks pairwise variance across an RM ensemble. Requires task in {'grpo', 'ppo'} on a non-mlx backend; installs the shared reward-signal callback. |
| `training.reward_hack_halt` | `bool` | `false` |  | Auto-halt training on HACK verdict (drop_pct >= 30% in cluster separation). Requires reward_hack_detector to be set. (v0.70.0) |
| `training.reward_hack_mitigation` | `str` | `"off"` | one of `"off"`, `"log_only"`, `"kl_control"`, `"pid_lagrangian"` | Closed-loop reward-hacking mitigation mode (v0.71.26). 'off' = no mitigation; 'log_only' = instrument + append a mitigation log without touching training; 'kl_control' = reversible bang-bang KL/β controller; 'pid_lagrangian' = PID-Lagrangian controller + rollback escalation ladder. Any non-'off' mode requires reward_hack_detector (the signal source) and task in {'grpo', 'ppo'} on a non-mlx backend. |
| `training.reward_hack_beta_floor` | `float` | `0.02` | `> 0.0` | v0.71.26 — lower β/kl_coef bound for the mitigation controller. Must be > 0 (β=0 gates off the ref-log-prob path at generation). |
| `training.reward_hack_beta_ceil` | `float` | `1.0` | `> 0.0`, `<= 1000.0` | v0.71.26 — upper β/kl_coef bound for the mitigation controller. Must be > reward_hack_beta_floor. |
| `training.reward_hack_trip_band` | `float` | `0.3` | `>= 0.0`, `<= 1.0` | v0.71.26 — hacking drop_pct at/above which the controller wants to RAISE β. Must be > reward_hack_release_band. |
| `training.reward_hack_release_band` | `float` | `0.1` | `>= 0.0`, `<= 1.0` | v0.71.26 — hacking drop_pct at/below which the controller wants to RELAX β. Must be &lt; reward_hack_trip_band. |
| `training.reward_hack_dwell_steps` | `int` | `2` | `>= 1`, `<= 100000` | v0.71.26 — consecutive trip-band steps required before the controller raises β (hysteresis). |
| `training.reward_hack_release_patience` | `int` | `3` | `>= 1`, `<= 100000` | v0.71.26 — consecutive release-band steps required before the controller relaxes β. |
| `training.reward_hack_kl_gain` | `float` | `1.5` | `> 1.0`, `<= 100.0` | v0.71.26 — multiplicative β step per trip (>1). β is multiplied on trip / divided on release, clamped to \[floor, ceil]. |
| `training.reward_hack_signals` | `list[str]` | `["info_rm"]` | max length 4 | v0.71.26 — signals combined into the controller's multi-signal vote. Allowlist (max 4): info_rm, rm_ensemble, length_trend, repetition. |
| `training.reward_hack_pid_kp` | `float` | `0.5` | `>= 0.0`, `<= 1000.0` | v0.71.26 — PID proportional gain (pid_lagrangian mode). |
| `training.reward_hack_pid_ki` | `float` | `0.1` | `>= 0.0`, `<= 1000.0` | v0.71.26 — PID integral gain (pid_lagrangian mode). |
| `training.reward_hack_pid_kd` | `float` | `0.05` | `>= 0.0`, `<= 1000.0` | v0.71.26 — PID derivative gain (pid_lagrangian mode). |
| `training.reward_hack_signal_target` | `float` | `0.15` | `>= 0.0`, `< 1.0` | v0.71.26 — target hacking drop_pct the PID controller holds (pid_lagrangian mode). |
| `training.reward_hack_integral_clamp` | `float` | `1.0` | `> 0.0`, `<= 1000.0` | v0.71.26 — PID anti-windup bound on the integral accumulator (pid_lagrangian mode). Independent of beta_ceil. |
| `training.reward_hack_rollback` | `bool` | `false` |  | v0.71.26 — enable rollback to the last-good RL checkpoint in the escalation ladder. Requires rl_checkpoint_save_every_steps set. |
| `training.reward_hack_rollback_patience` | `int` | `3` | `>= 1`, `<= 100000` | v0.71.26 — consecutive HACK steps before a rollback is triggered. |
| `training.reward_hack_max_recovery_attempts` | `int` | `2` | `>= 0`, `<= 1000` | v0.71.26 — max rollbacks before the controller early-stops training (terminal rung of the escalation ladder). |
| `training.reward_hack_signal_smoothing` | `str` | `"none"` | one of `"none"`, `"ema"`, `"median"` | v0.71.26 — per-signal smoothing before the controller vote: 'none', 'ema' (0.5·prev+0.5·new), or 'median' over a window. |
| `training.reward_hack_smoothing_window` | `int` | `8` | `>= 2`, `<= 256` | v0.71.26 — window length for signal smoothing. |
| `training.reward_hack_conservative_on_disagreement` | `bool` | `false` |  | v0.71.26 — when detectors disagree, keep KL high (use the MAX signal) instead of relaxing. |
| `training.reward_hack_reward_shaping` | `bool` | `false` |  | v0.71.26 — apply a bounded penalty on the gamed proxy (length/repetition/sentinel) via a shaping shim over the reward fn. Requires reward_hack_shaping_strength > 0 and a control mode. |
| `training.reward_hack_shaping_kind` | `str` | `"length"` | one of `"length"`, `"repetition"`, `"sentinel"` | v0.71.26 — which gamed proxy the reward-shaping shim penalises. |
| `training.reward_hack_shaping_strength` | `float` | `0.0` | `>= 0.0`, `<= 1.0` | v0.71.26 — magnitude of the bounded reward-shaping penalty \[0, 1]. |
| `training.ppo_epochs` | `int` | `4` | `>= 1` | Number of PPO optimization epochs per batch |
| `training.ppo_clip_ratio` | `float` | `0.2` | `> 0`, `<= 1.0` | PPO clipping range for policy ratio |
| `training.ppo_kl_penalty` | `float` | `0.05` | `>= 0` | KL divergence penalty coefficient for PPO |
| `training.reward_model` | `str \| null` | `null` |  | Path or HF ID of a trained reward model for PPO |
| `training.loraplus_lr_ratio` | `float \| null` | `null` | `> 0` | LoRA+ lr ratio: lr_B = lr × ratio. None = disabled (standard LoRA). |
| `training.use_lorafa` | `bool` | `false` |  | Enable LoRA-FA (Frozen-A LoRA) optimizer: freezes LoRA A matrices to reduce activation memory |
| `training.use_galore` | `bool` | `false` |  | Enable GaLore (Gradient Low-Rank Projection) for memory-efficient training |
| `training.galore_rank` | `int` | `128` | `>= 1` | GaLore projection rank |
| `training.galore_update_proj_gap` | `int` | `200` | `>= 1` | GaLore projection update interval (steps) |
| `training.galore_scale` | `float` | `0.25` | `> 0` | GaLore gradient scaling factor |
| `training.moe_lora` | `bool` | `false` |  | Enable MoE-aware LoRA (ScatterMoE) — applies LoRA to expert FFN layers |
| `training.moe_aux_loss_coeff` | `float` | `0.01` | `>= 0` | Auxiliary load-balancing loss coefficient for MoE models |
| `training.use_liger` | `bool` | `false` |  | Enable Liger Kernel fused operations (20-60% memory savings, 20-40% speedup) |
| `training.fast_lora` | `bool` | `false` |  | Run LoRA's q/k/v, MLP and other projections through Soup's hand-written Fast-LoRA autograd functions instead of peft's modules. Same math; no speedup is claimed. task='sft' on the transformers backend with text, quantization none or 4bit, and lora.dropout 0; refused beside use_liger, DoRA, VeRA, target_parameters, Spectrum, LISA and FP8 training. |
| `training.use_flash_attn` | `bool` | `false` |  | Enable FlashAttention (auto-detects v2/v3/v4 for faster attention) |
| `training.use_ring_attention` | `bool` | `false` |  | Enable Ring FlashAttention for sequence parallelism across GPUs |
| `training.rope_scaling_type` | `str \| null` | `null` | one of `"linear"`, `"dynamic"`, `"yarn"`, `"llama3"` | RoPE scaling method for long-context: linear, dynamic, yarn or llama3 (v0.49.0). 'longrope' is refused at config load (#1239). Applied by task sft, tts and pretrain on backend: transformers (sft: the text path without layer streaming); refused at config load anywhere else (#1697). |
| `training.yarn_factor` | `float \| null` | `null` | `> 1.0`, `<= 1024.0` | YaRN scaling factor (s). Optional — when omitted, the runtime falls back to ``target_length / original_length`` (HF default behaviour). |
| `training.yarn_attn_factor` | `float \| null` | `null` | `>= 0.0`, `<= 10.0` | YaRN attention temperature multiplier (default 1.0 in HF). |
| `training.yarn_beta_fast` | `int \| null` | `null` | `>= 1`, `<= 1024` | YaRN beta_fast cutoff (HF default 32). |
| `training.yarn_beta_slow` | `int \| null` | `null` | `>= 1`, `<= 1024` | YaRN beta_slow cutoff (HF default 1). |
| `training.use_longlora` | `bool` | `false` |  | Refused at config load (#1240): the LongLoRA override rolled the Q/K projections of half the heads with wrap-around under full causal attention, leaking future tokens, and applied no S² grouped attention. Use rope_scaling_type with plain LoRA to extend the context. |
| `training.gradient_checkpointing` | `bool \| "selective" \| "medium" \| "full" \| "auto"` | `false` |  | Gradient checkpointing for memory savings on long sequences. False/True (legacy bool) or tier: 'selective' (attention only), 'medium' (every other block), 'full' (all blocks), 'auto' (picks based on available VRAM). (v0.28.0). |
| `training.use_cut_ce` | `bool` | `false` |  | Enable Cut Cross-Entropy (CCE) for large-vocab models. Saves 8-24GB VRAM on Llama 3.1 128k vocab. Requires cut_cross_entropy. Mutually exclusive with Unsloth/MLX backends. |
| `training.kernel_auto_compose` | `bool` | `false` |  | Unsupported compatibility field. Set use_liger and/or use_flash_attn explicitly instead. |
| `training.packing_cross_doc_attn_mask` | `bool` | `false` |  | Never worked: Soup mapped this to packing_strategy='attention_free', which is not in TRL's allowlist (bfd / bfd-requeue / wrapped) on any released trl. Use packing: true with FlashAttention instead. (v0.28.0 / #691 / #709). |
| `training.activation_offloading` | `str \| null` | `null` | one of `"cpu"`, `"disk"` | Offload activations to CPU or disk during backward pass. None=off, 'cpu'=offload to RAM, 'disk'=offload to tmp file. (v0.28.0). |
| `training.embedding_loss` | `str` | `"contrastive"` | one of `"contrastive"`, `"triplet"`, `"cosine"` | Loss function for embedding training: contrastive, triplet, or cosine |
| `training.embedding_margin` | `float` | `0.5` | `> 0` | Margin for contrastive/triplet loss (higher = stricter separation) |
| `training.embedding_pooling` | `str` | `"mean"` | one of `"mean"`, `"cls"`, `"last"` | Pooling strategy for sentence embeddings: mean, cls, or last token |
| `training.embedding_temperature` | `float` | `0.05` | `> 0` | Temperature for contrastive (InfoNCE) loss — lower = stricter similarity |
| `training.curriculum` | `bool` | `false` |  | Enable curriculum learning (sort dataset by difficulty, easy → hard) |
| `training.curriculum_metric` | `str` | `"length"` | one of `"length"`, `"perplexity"`, `"loss"` | Metric for curriculum difficulty: length, perplexity, or loss |
| `training.curriculum_buckets` | `int` | `4` | `>= 1`, `<= 20` | Number of difficulty stages for curriculum learning |
| `training.curriculum_dynamic` | `bool` | `false` |  | BETA: dynamically re-weight curriculum buckets every N steps via online uncertainty estimation (per-sample loss + grad norm). Requires curriculum=true. Multi-rank launches must wire an all_reduce hook on per-bucket stats (see utils.curriculum_dynamic.validate_distributed_curriculum). |
| `training.curriculum_dynamic_recompute_steps` | `int` | `50` | `>= 1`, `<= 100000` | Recompute curriculum bucket sampler weights every N global training steps. |
| `training.curriculum_dynamic_floor` | `float` | `0.05` | `> 0.0`, `<= 0.5` | Minimum normalised per-bucket weight after softmax. Must be in (0.0, 1/curriculum_buckets]; the cross-validator tightens this to the per-config ceiling. Prevents bucket starvation. |
| `training.curriculum_dynamic_temperature` | `float` | `1.0` | `> 0.0`, `<= 100.0` | Softmax temperature on the uncertainty signal. Higher = flatter distribution; lower = concentrate on hardest buckets. |
| `training.loss_watchdog` | `bool` | `false` |  | Enable loss spike detection (auto-stop if loss exceeds threshold) |
| `training.loss_watchdog_threshold` | `float` | `3.0` | `> 0`, `<= 100.0` | Stop training if loss exceeds this threshold |
| `training.loss_watchdog_patience` | `int` | `5` | `>= 1`, `<= 1000` | Consecutive high-loss steps before stopping |
| `training.rewind_log` | `bool` | `true` |  | Write &lt;output>/rewind.jsonl: the dataset rows in every training micro-batch with each row's loss, read by `soup rewind` to name the rows behind a loss spike. SFT only (transformers and MLX, single process); resumed runs and packing turn it off with one warning. |
| `training.loss_spike_recovery` | `bool` | `false` |  | On watchdog trigger: write &lt;output>/spike_recovery.json with decayed LR and attempt count for re-launch (tracks attempts and caps recovery). Requires loss_watchdog=true. |
| `training.loss_spike_recovery_max_attempts` | `int` | `3` | `>= 1`, `<= 10` | Max number of spike-recovery attempts before giving up |
| `training.loss_spike_recovery_lr_decay` | `float` | `0.5` | `> 0.0`, `< 1.0` | Multiply LR by this factor on each spike recovery (0.5 = halve) |
| `training.relora_steps` | `int \| null` | `null` | `>= 1`, `<= 10000000` | Fire a paper-faithful ReLoRA restart every N global steps: merge the LoRA update into the base weight, reinitialize lora_A/lora_B, and optionally clear optimizer state. None disables. Requires float LoRA (training.quantization defaults to '4bit' and must be set to 'none'; incompatible with VeRA, DoRA, layer streaming, and FSDP2 compile). |
| `training.relora_warmup_ratio` | `float` | `0.1` | `>= 0.0`, `<= 1.0` | Skip ReLoRA firings during the first warmup_ratio fraction of training |
| `training.relora_reset_optimizer` | `bool` | `true` |  | Clear optimizer state for lora_A/lora_B parameters after each ReLoRA merge+reinit restart |
| `training.relora_prune_ratio` | `float` | `0.9` | `> 0.0`, `< 1.0` | Deprecated: kept so older YAML still loads. Restarts no longer magnitude-prune adapter weights; relora_reset_optimizer controls optimizer clearing instead. |
| `training.convergence_detection` | `bool` | `false` |  | Watch for loss plateau / oscillation and surface advice (continue / early_stop / lower_lr) at the end of training. |
| `training.convergence_window` | `int` | `50` | `>= 5`, `<= 10000` | Staged, not enforced during training: number of recent losses to inspect for plateau / oscillation. Refused at load when set away from its default (v0.77, #808). |
| `training.convergence_rel_tol` | `float` | `0.005` | `> 0.0`, `<= 1.0` | Staged, not enforced during training: relative range threshold below which the window is a plateau. Refused at load when set away from its default (v0.77, #808). |
| `training.warmup_auto` | `bool` | `false` |  | Auto-pick warmup_steps from dataset_size × epochs × warmup_ratio. Overrides any manual warmup_steps in the trainer. |
| `training.auto_mixed_precision` | `bool` | `false` |  | Pick bf16/fp16 based on model + GPU compute capability. Applied by task='sft' (and 'tts', which trains through the SFT trainer); refused on every other task, whose trainers pick their own precision. |
| `training.grad_accum_auto_tune` | `bool` | `false` |  | Monitor VRAM each step; warn (and recommend new batch/accum) when memory pressure is high. Advisory only: it does not rebuild the DataLoader or change accumulation during a run. |
| `training.grad_accum_pressure_threshold` | `float` | `0.92` | `> 0.05`, `< 0.99` | VRAM utilisation fraction that triggers a recommendation |
| `training.freeze_layers` | `int \| null` | `null` | `>= 1`, `<= 1000` | Freeze first N layers (from bottom). Train only remaining layers. Applied by tasks sft and tts on the text path with backend transformers and no layer streaming; refused everywhere else. |
| `training.freeze_ratio` | `float \| null` | `null` | `> 0.0`, `< 1.0` | Freeze this fraction of layers (0.75 = freeze 75% from bottom). Applied by tasks sft and tts on the text path with backend transformers and no layer streaming; refused everywhere else. |
| `training.unfrozen_parameters` | `list[str] \| null` | `null` |  | Spectrum (#266) targeted training: regex patterns of parameter names to keep trainable — every other parameter is frozen. Full fine-tuning, LoRA off. Generate with `soup spectrum scan`. Mutually exclusive with LoRA features / freeze_layers / freeze_ratio / train_router_only; sft + transformers backend only. |
| `training.lisa_enabled` | `bool` | `false` |  | Enable LISA layerwise importance sampling (#267): every lisa_interval_steps, freeze all decoder layers except a random lisa_num_layers set (embeddings + head always trainable). Full fine-tuning, LoRA off. sft or pretrain + transformers + text + quantization=none only; mutually exclusive with LoRA features / freeze_layers / freeze_ratio / unfrozen_parameters. |
| `training.lisa_num_layers` | `int` | `2` | `>= 1`, `<= 64` | LISA: number of decoder layers kept trainable per interval (clamped to the model's layer count). Small = LoRA-like memory. |
| `training.lisa_interval_steps` | `int` | `20` | `>= 1`, `<= 1000000` | LISA: re-sample the active decoder layers every N global steps. |
| `training.lisa_reset_optimizer` | `bool` | `true` |  | Clear optimizer state for decoder layers that LISA re-freezes on each interval (avoids stale Adam moments). Mirrors relora_reset_optimizer. |
| `training.lisa_train_embeddings` | `bool` | `true` |  | Keep the always-on group (input embeddings + LM head + final norm) trainable every LISA interval. Default true is LISA as published (#267). Set false to freeze that group so only the sampled lisa_num_layers decoder layers train — the always-on set is the majority of LISA's memory, so this is the knob that buys the memory saving (#377). It is a real trade: the always-on set may be load-bearing for quality, so measure both ways. |
| `training.stream_layers` | `bool` | `false` |  | BETA — stream the frozen base layer-by-layer from CPU RAM so a model larger than VRAM can be fine-tuned. sft + transformers + text + quantization=none only; batch_size 1, no gradient accumulation. Slower than resident training, but these models did not run on the card at all. |
| `training.stream_source` | `str` | `"auto"` | one of `"auto"`, `"ram"`, `"disk"` | Where the streamed base lives. 'ram' (the only tier implemented in v0.72.0) pins the base in CPU RAM; 'disk' is the v0.72.3 overflow tier; 'auto' picks RAM only when the store fits free-RAM headroom and the store plus resident extras fits the physical RAM ceiling. |
| `training.stream_ngram_source` | `str` | `"auto"` | one of `"auto"`, `"ram"`, `"disk"` | Where Qwen4-Exp's frozen PLE N-gram embedding lives while layer streaming. 'disk' gathers only requested rows from the original safetensors through a read-only mmap; 'ram' keeps the table in CPU RAM; 'auto' uses RAM only when the table and selected base tier plus resident extras fit within free-RAM headroom and the physical RAM ceiling. oMLX/oQ affine PLE tables require 'disk' (or 'auto') so only selected rows are dequantized. A non-default value warns when the checkpoint has no external PLE table. |
| `training.stream_buffers` | `int` | `2` | `>= 2`, `<= 8` | Pre-allocated VRAM layer buffers. 2 = double buffering, which is what lets the next layer load while the current one computes. A single buffer cannot overlap load with compute. |
| `training.stream_read_ahead` | `int` | `2` | `>= 1`, `<= 8` | Layers the async disk tier stages ahead on its background thread. One staging slot is always held by the layer currently being consumed, so a setting of N stages N-1 layers ahead, not N: 1 = the read merely leaves the compute thread (nothing staged ahead of the one in use), 2 = one layer in flight while one is consumed, up to 7 staged ahead at the maximum setting of 8. Each level costs one layer of pinned host memory, which is 441 MB on a 70B. |
| `training.stream_pin` | `bool \| null` | `null` |  | Force the page-locked (pinned) host memory on or off — the RAM tier's base store, or the disk tier's async-reader host staging (#971). None (the default) lets the box decide: it attempts to page-lock that memory and falls back to pageable — announcing the cost — when the host cannot page-lock it. 'false' forces the pageable store, the only known escape hatch when a pinning path is suspect (it was the sole mitigation while #331 was live). 'true' forces pinning and REFUSES the run rather than silently degrading if the box cannot page-lock it: on the RAM tier naming the store size, on the disk tier naming training.stream_read_ahead as the depth that decides how much staging gets locked — page-locking is worth up to 6.56x measured throughput, so a silent fallback spends the whole margin the feature exists to provide. Non-CUDA targets (CPU or MPS) are the one case pinning stays INAPPLICABLE: 'true' is announced there and the run proceeds with a pageable CPU source, so the key stays committable to a config shared between CUDA and non-CUDA boxes. |
| `training.stream_vram_override` | `int \| null` | `null` | `>= 0` | Bytes to assume free instead of measuring torch.cuda.mem_get_info(). mem_get_info() is a device-level driver query, so it cannot see a per-process cap (set_per_process_memory_fraction, a MIG slice, a card another process is also using), so on that hardware the pre-flight sees the whole device's free VRAM regardless of what this process can actually use. Set this to replace the measured figure in either direction: raise it to let a known-safe over-prediction through, or lower it to make the pre-flight enforce a cap the driver itself cannot report. |
| `training.stream_vram_probe` | `bool` | `false` |  | Decide the layer-streaming VRAM pre-flight on a MEASUREMENT instead of the fitted formula. One real forward+backward runs at the configured shape after the streamed model is built, and its peak decides whether the run proceeds; the prediction is printed beside it so a divergence is visible. The formula under-predicts past seq 4352 (measured 0.934x the real peak at seq 5120 and 0.787x at 6144), which is the direction that does not announce itself — an OOM on Linux, a silent spill to host memory on Windows. Against the probe the same formula reads 0.992x at seq 4096 and 0.830x at 5120, because the probe itself runs 12.5-14.3% above the real training step — the direction that makes it safe as a gate. Off by default: it costs one step (1-5 s measured) and it can refuse a run the formula accepts. task='sft' only: the probe runs a plain causal-LM forward+backward, which IS the SFT step but not a preference loss. At the one preference shape measured it reads +13.5% high (probe 6.021 GB against a real DPO step's 5.304 GB) — the same safe direction it shows for SFT — but one shape is not a validation, so the restriction stands until it is measured across shapes. |
| `training.stream_nf4_scale_cache` | `bool` | `false` |  | Keep each streamed NF4 weight's expanded fp32 block scales on the compute device after its first use. A double-quantised weight stores its scales as 8-bit codes plus a nested scale and offset, and bitsandbytes' dequantise + linear path expands them on every visit (forward and checkpoint recompute): two extra kernels and a temporary per projection. The base is frozen, so the expansion is done once per (layer, weight) and reused; results are bit-identical. Costs 4 bytes per 64-element block of VRAM for the whole run (416 MiB on Mistral-7B), which the pre-flight charges up front. bitsandbytes' fused 4-bit GEMM path expands the scales inside its kernel and is left unchanged. Needs quantization='4bit' with double quantization (the default). Off by default. |
| `training.stream_recompute` | `str` | `"on"` | one of `"on"`, `"off"`, `"auto"` | Whether a streamed decoder layer runs its forward a second time in the backward. 'on' (the default) is the shipped checkpoint recompute: activations are dropped after the forward and recomputed. 'off': every streamed layer keeps its activations instead (the backward still re-reads each layer's weights): a faster step where the step is compute-bound (the RAM tier), at the cost of every layer's activations in VRAM, which the formula pre-flight does not count (stream_vram_probe measures it). 'auto': before training, one synthetic forward+backward at the configured batch_size x data.max_length measures one layer's retained bytes and the step's reserved peak; the LAST K layers keep their activations and the rest recompute, K = (the smallest of 85% of the card, free + reserved VRAM and stream_vram_override) minus that peak, divided by the layer's bytes times the step's reserved/allocated ratio. Costs one step at setup; RAM tier and CUDA only, otherwise the recompute is kept. Gradients are the recompute's: measured bit-identical under deterministic algorithms. task='sft' only for 'off' and 'auto', and refused beside activation_offloading. YAML reads an unquoted on/off (and true/false, yes/no) as a boolean; true is taken as 'on' and false as 'off'. |
| `training.stream_expert_layout` | `str` | `"fused"` | one of `"fused"`, `"per_expert"` | How a MoE base's routed experts are streamed. 'fused' (the default) keeps each MoE layer's experts inside its layer file, so the largest layer bounds the card. 'per_expert' writes them to their own file, one pair of matrices per expert (NF4 per matrix under quantization='4bit'), and reads an expert only when the router picks it, again in the backward: the decoder-layer buffers hold the dense part alone, so the VRAM bound no longer grows with the expert count. Each routed expert is read about three times per step from the cache volume, so expect slower steps (an NVMe volume is recommended). Under quantization='4bit' the routed experts are NF4 here but stay at the stream dtype with 'fused', so the two layouts train against numerically different bases. Admitted for DeepSeek-V3-family bases (model_type deepseek_v3); adapters on the routed experts (moe_lora) are refused with it. |
| `training.stream_loss_chunk` | `int \| null` | `null` | `>= 16`, `<= 8192` | Tokens per chunk for the output head and the loss under layer streaming (#1693). Unset, the head builds the full \[batch, seq, vocab] logits and the loss works on all of them, which is what caps the context length on a small card. Set, each chunk runs the head, takes its cross-entropy and the gradient on its hidden states, and drops its logits, so the logits term is vocab x chunk x batch whatever data.max_length is. It costs one more \[batch, seq, hidden] tensor and a loop over the head. It gives up outputs.logits on a labelled forward, and equality to the last bit with an unchunked step: the per-token terms are the same, the order of the final sum is not. task='sft' only; refused beside use_cut_ce. |
| `training.stream_disk_kind` | `str \| null` | `null` | one of `"nvme"`, `"ssd"`, `"hdd"` | Override the auto-detected disk media type for the streaming disk overflow tier. Detection classifies a device by a measured sequential read when the kernel's rotational flag is unreliable — virtio and other paravirtual disks report spinning with no media hint, so a fast cloud disk is otherwise refused (#365). Set this for the case where even that is wrong: 'nvme' forces the disk tier on, 'ssd'/'hdd' force it off. The resolved value is printed beside what was detected. |
| `training.packing` | `bool` | `false` |  | Pack multiple short samples into one sequence for faster training |
| `training.multipack` | `bool` | `false` |  | Use FFD bin-packing sampler to maximise tokens-per-batch on uneven-length data. Mutually exclusive with packing. Only supported for sft / pretrain tasks (transformers backend). (v0.37.0). |
| `training.neftune_alpha` | `float \| null` | `null` | `>= 0.0`, `<= 50.0` | NEFTune noise alpha (0-50). Adds noise to embeddings for better chat quality. |
| `training.forgetting_detection` | `bool` | `false` |  | Staged, not enforced during training: enable periodic general-knowledge eval to detect catastrophic forgetting |
| `training.forgetting_eval_steps` | `int` | `100` | `>= 10`, `<= 10000` | Staged, not enforced during training: run forgetting eval every N steps. Refused at load when set away from its default (v0.77, #808). |
| `training.forgetting_threshold` | `float` | `0.1` | `>= 0.01`, `<= 0.5` | Staged, not enforced during training: warn if accuracy drops beyond this threshold (0.01-0.50) |
| `training.forgetting_benchmark` | `str` | `"mini_mmlu"` | one of `"mini_mmlu"`, `"mini_common_sense"`, `"mini_instruction"` | Staged, not enforced during training: built-in mini benchmark for forgetting detection. Refused at load when set away from its default (v0.77, #808). |
| `training.forgetting_stop` | `bool` | `false` |  | Staged, not enforced during training: stop on severe forgetting. Refused at load when set away from its default (v0.77, #808). |
| `training.checkpoint_intelligence` | `bool` | `false` |  | Staged, not enforced during training: track best checkpoints by quality |
| `training.checkpoint_eval_steps` | `int` | `200` | `>= 50`, `<= 10000` | Staged, not enforced during training: evaluate checkpoint quality. Refused at load when set away from its default (v0.77, #808). |
| `training.checkpoint_eval_metric` | `str` | `"composite"` | one of `"judge"`, `"mmlu"`, `"custom"`, `"composite"` | Staged, not enforced during training: checkpoint quality metric. Refused at load when set away from its default (v0.77, #808). |
| `training.checkpoint_eval_tasks` | `str \| null` | `null` |  | Staged, not enforced during training: JSONL tasks for checkpoint scoring. Refused at load when set away from its default (v0.77, #808). |
| `training.checkpoint_keep_top` | `int` | `3` | `>= 1`, `<= 20` | Staged, not enforced during training: keep top-N quality checkpoints. Refused at load when set away from its default (v0.77, #808). |
| `training.early_stop_on_regression` | `bool` | `false` |  | Staged, not enforced during training: stop after consecutive regressions |
| `training.early_stop_patience` | `int` | `2` | `>= 1`, `<= 10` | Staged, not enforced during training: regressions before stopping (1-10). Refused at load when set away from its default (v0.77, #808). |
| `training.eval_gate` | `section \| null` | `null` |  | Optional EvalGateConfig — block training on regressions. See [`training.eval_gate`](#trainingeval_gate). |
| `training.use_fsdp2_compile` | `bool` | `false` |  | Enable torch.compile on top of FSDP2 for +20-30% training speed. Requires --fsdp, CUDA, and backend=transformers. |
| `training.parallelism` | `str` | `"data"` | one of `"data"`, `"pipeline"` | Distributed strategy: 'data' (DDP/FSDP/DeepSpeed) or 'pipeline' (pipeline parallel, v0.27.0 wiring only). |
| `training.pipeline_stages` | `int` | `1` | `>= 1`, `<= 16` | Number of pipeline parallel stages. Ignored when parallelism='data'. |
| `training.unlearn_method` | `str \| null` | `null` | one of `"npo"`, `"simnpo"`, `"rmu"` | Unlearning method backend — required when task='unlearn'. npo (Negative Preference Optimization, DPO-shaped negative-only loss); simnpo (length-normalised NPO without ref model); rmu (Representation Misdirection Unlearning, residual-stream noise). |
| `training.unlearn_alpha` | `float \| null` | `null` |  | Retain-set weighting in the unlearn loss (forget vs retain mixing coefficient). 0.0 = pure forget loss; higher values increasingly favour the retain set. Bounded \[0.0, 10.0]. (v0.61.0) |
| `training.ra_dit_stage` | `str \| null` | `null` | one of `"retriever"`, `"generator"` | RA-DIT pipeline stage. 'retriever' trains the sentence-transformer via the v0.16 embedding trainer; 'generator' runs RAFT-style SFT on `data.format='raft'`. Composes with the v0.62.0 RAFT recipe. (v0.62.0) |
| `training.ra_dit_retriever_model` | `str \| null` | `null` |  | Optional retriever model id (e.g. `sentence-transformers/all-mpnet-base-v2`) used by the generator stage to pre-encode distractor docs. (v0.62.0) |
| `training.citation_faithful` | `bool` | `false` |  | Opt INTO citation-precision / recall scoring + a loss-mask rule that emphasises citation spans. Requires `data.format='raft'`; SFT and RAFT trainers apply the weighted citation-span mask. |
| `training.citation_style` | `str \| null` | `null` | one of `"bracket"`, `"inline"`, `"footnote"` | Citation rendering style. 'bracket' = `[doc-1]` inline tag (canonical RAFT default), 'inline' = `(doc-1)`, and 'footnote' = `[^doc-1]`. Used by citation scoring and span masking. |
| `training.citation_recall_threshold` | `float \| null` | `null` |  | Staged, not applied: a citation-recall threshold in \[0.0, 1.0] that nothing gates a save or a run on, so setting it is refused at load (#761). (v0.62.0). Refused at load when set away from its default (v0.77, #808). |
| `training.grace_codebook` | `bool` | `false` |  | Reserve GRACE codebook configuration for training. Compatibility validation only: no trainer or knowledge-edit command consumes this field yet. Refused at load when set away from its default (v0.77, #808). |
| `training.grace_codebook_size` | `int \| null` | `null` |  | Reserved GRACE codebook entry count. Required when grace_codebook=True and bounded \[1, 100_000], but not consumed at runtime yet. Refused at load when set away from its default (v0.77, #808). |
| `training.grace_codebook_dim` | `int \| null` | `null` |  | Reserved GRACE codebook entry dimension. Required when grace_codebook=True and bounded \[1, 16_384], but not consumed at runtime yet. Refused at load when set away from its default (v0.77, #808). |

## `training.lora`

| Key | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `training.lora.r` | `int` | `64` | `>= 0` | LoRA rank. 0 = full fine-tuning: no adapter, every base parameter trains (sft / embedding + transformers + text + quantization='none' only). |
| `training.lora.alpha` | `int` | `16` |  | LoRA alpha |
| `training.lora.dropout` | `float` | `0.05` |  | LoRA dropout |
| `training.lora.target_modules` | `str \| list[str]` | `"auto"` |  | Target modules for LoRA. 'auto' = let peft decide. |
| `training.lora.target_parameters` | `"auto" \| list[str] \| null` | `null` |  | Raw 2-D/3-D nn.Parameter tensors to adapt with PEFT LoRA. Use 'auto' to select architecture-specific parameter targets (currently Qwen4-Exp routed experts), a list of parameter-name suffixes for explicit control, or omit to disable. Requires dropout=0 and is wired for transformers SFT/pretrain. |
| `training.lora.use_dora` | `bool` | `false` |  | Enable DoRA (Weight-Decomposed Low-Rank Adaptation) |
| `training.lora.use_rslora` | `bool` | `false` |  | Enable rank-stabilized LoRA scaling (better for high ranks) |
| `training.lora.use_vera` | `bool` | `false` |  | Enable VeRA (Vector-based Random Matrix Adaptation). Shared random matrices — much smaller memory than LoRA. Mutually exclusive with use_dora and use_olora. |
| `training.lora.use_olora` | `bool` | `false` |  | Enable OLoRA (Orthogonal LoRA init via QR decomposition). Passes init_lora_weights='olora' to peft. Mutually exclusive with use_dora and use_vera. Equivalent to init_strategy='olora'. |
| `training.lora.rank_pattern` | `dict[str, int] \| null` | `null` |  | Per-target-module-pattern LoRA rank override. Maps module name patterns (e.g. 'q_proj', 'experts.\*.w1') to integer rank values. Useful for MoE configs where expert FFNs need lower rank than attn. Incompatible with use_vera (VeRA shares one rank across modules). |
| `training.lora.alpha_pattern` | `dict[str, int] \| null` | `null` |  | Per-target-module-pattern LoRA alpha override. Maps module name patterns to integer alpha values. Pairs with rank_pattern. Incompatible with use_vera. |
| `training.lora.init_strategy` | `str` | `"random"` | one of `"random"`, `"pissa"`, `"olora"`, `"loftq"` | LoRA init strategy. 'random' (default) is standard Kaiming init. 'pissa' (PiSSA) initializes A/B from the SVD of the base weight — faster early convergence but adds an SVD pass on the first epoch. 'olora' is equivalent to use_olora=True (orthogonal QR init). 'loftq' (v0.41.0) initialises A/B + a low-bit base together, useful with QLoRA. Cannot be combined with use_dora or use_vera. |
| `training.lora.loftq_iter` | `int` | `1` | `>= 1`, `<= 10` | LoftQ iteration count (1-10). Higher = better quant-aware init at the cost of one-time setup latency. Used only when init_strategy='loftq'. |
| `training.lora.loftq_bits` | `int` | `4` | one of `2`, `4`, `8` | LoftQ target bitwidth — must be one of {2, 4, 8}. Used only when init_strategy='loftq'. |

## `training.eval_gate`

| Key | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `training.eval_gate.enabled` | `bool` | `false` |  | Turn the eval gate on |
| `training.eval_gate.suite` | `str \| null` | `null` |  | Path to eval-suite YAML (evals/gate.yaml) |
| `training.eval_gate.every_n_epochs` | `int` | `1` | `>= 1`, `<= 100` | Run gate every N epochs (1-100) |
| `training.eval_gate.regression_threshold` | `float` | `0.05` | `>= 0.0`, `<= 1.0` | Max absolute drop vs baseline before regression fires |
| `training.eval_gate.baseline` | `str \| null` | `null` |  | registry://&lt;id> \| 'previous' \| file path - scores to compare against |
| `training.eval_gate.on_regression` | `str` | `"stop"` | one of `"stop"`, `"warn"`, `"continue"` | Action on regression: stop training \| warn only \| continue |

## `eval`

| Key | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `eval.auto_eval` | `bool` | `false` |  | Run evaluation automatically after training completes |
| `eval.benchmarks` | `list[str] \| null` | `null` |  | lm-evaluation-harness benchmark names to run |
| `eval.custom_tasks` | `str \| null` | `null` |  | Path to custom eval JSONL file |
| `eval.judge` | `dict \| null` | `null` |  | LLM-as-a-judge config: model, rubric, provider |
| `eval.ship` | `section \| null` | `null` |  | soup ship verdict defaults (task_eval / task_mode / general_suite / forgetting_threshold / judge_model / baseline / noise_floor). See [`eval.ship`](#evalship). |

## `eval.ship`

| Key | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `eval.ship.task_eval` | `str \| null` | `null` |  | JSONL of leg-1 task-win eval tasks |
| `eval.ship.task_mode` | `str` | `"metric"` | one of `"metric"`, `"judge_score"`, `"pairwise"` | Leg-1 mode: metric \| judge_score \| pairwise (judge win-rate) |
| `eval.ship.general_suite` | `str \| null` | `null` |  | Comma list of leg-2 benchmarks (default: bundled offline suite) |
| `eval.ship.forgetting_threshold` | `float` | `0.05` | `>= 0.0`, `<= 1.0` | Max allowed leg-2 drop in absolute points before DON'T SHIP |
| `eval.ship.judge_model` | `str \| null` | `null` |  | Judge model URL for task_mode judge_score / pairwise |
| `eval.ship.baseline` | `str \| null` | `null` |  | registry://&lt;id> \| stamped baseline JSON ({scores, provenance.scorer_revision}) of base leg-2 scores |
| `eval.ship.noise_floor` | `int \| null` | `null` | `>= 2`, `<= 10` | Repeats of the BASE run used to measure a leg-2 noise floor; a leg-2 drop within the floor is treated as noise, not regression. Bounded \[2, 10]. A live input: measured when producing evidence, refused under --evidence. |

## `advise`

| Key | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `advise.goal` | `str \| null` | `null` | max length 4096 | Default goal string for `soup advise`. Sharpens task classification when set. |
| `advise.probe` | `bool` | `false` |  | Run the 10-minute ROI probe by default when `soup advise` is invoked through this config. Heuristic stubs in v0.54.0. |
| `advise.record` | `bool` | `false` |  | Append every verdict from this config to \~/.soup/advise_history.jsonl with accepted=True. |
