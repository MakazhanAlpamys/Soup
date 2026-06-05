# Changelog

All notable changes to **Soup CLI** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Detailed, per-release notes for every published version live on the
[GitHub Releases page](https://github.com/MakazhanAlpamys/Soup/releases). This
file tracks unreleased changes and links out for historical detail rather than
reproducing 70+ versions of notes.

## [Unreleased]

## [0.71.14] - 2026-06-05

### Added
- **Live FSDP shard consolidation** (closes #96). `soup merge-sharded-fsdp-weights`
  lifts the v0.44.0 plan-only stub: it now streams each `pytorch_model_fsdp_*.bin`
  shard via `torch.load(weights_only=True)` (no arbitrary pickle exec), unions the
  per-rank parameter fragments into one state-dict, and writes a single
  `.safetensors` atomically. Memory-friendly (one shard loaded at a time). New
  `--plan-only` flag prints the plan without writing. Single-process — no
  multi-GPU needed to MERGE. (Per-rank disjoint-parameter / FULL_STATE_DICT
  shards; DCP sharded-tensor reconstruction is out of scope — use
  `accelerate merge-weights` for those.)
- **Live `kv_cache_type` wiring on the transformers serve backend** (closes #140).
  `soup serve --kv-cache-type q8_0 | bf16 | f16 | fp8` lifts the v0.53.1
  `apply_kv_cache_type` `NotImplementedError` stub: `bf16`/`f16` load the model in
  that dtype (the KV cache inherits it); `q8_0` routes an 8-bit HQQ quantized KV
  cache through `model.generate` (needs `pip install hqq`); `fp8` raises a friendly
  runtime error (vLLM + Hopper-only — the transformers backend has no fp8 KV
  path). vLLM / SGLang KV-cache-dtype routing stays in the infra-blocked tail.
- **ONNX export QA verified** (closes #71) — `soup export --format onnx` exercised
  end-to-end on a tiny model: export exits 0, `model.onnx` loads in ONNX Runtime
  with `input_ids` present, and a forward pass produces a real output. Recorded in
  `tests/qa/v07114_qa.md`.

### Notes
- GGUF export (#70), AWQ/GPTQ export (#72), the CUDA + llama.cpp QA doc (#144),
  HF Hub push/Spaces deploy (#74), and the Community-QA tracking meta-issue (#79)
  remain open with `infra-blocked` labels — they need a built llama.cpp toolchain,
  `autoawq`/`auto-gptq` Windows wheels, or HF credentials the QA box lacks. See
  `tests/qa/v07114_qa.md`.

## [0.71.13] - 2026-06-04

### Added
- **Prompt-compile family — live wiring** (closes #225, #226, #227, #229). Four
  `soup` commands that shipped as deferred-stub `NotImplementedError` in v0.68.0
  are now real, validated end-to-end (real DPO train on SmolLM2-135M + real
  Ollama teacher distillation on RTX 3050).
- **`soup local-rl train` runs a real nightly DPO/KTO/ORPO train** (#229).
  `--once` harvests the latest thumbs-up/down DPO pairs from the local-RL SQLite
  and trains them via a `soup train` subprocess (argv list, no shell); a `state`
  table tracks `last_train_at` so a re-run with no new feedback skips, and a run
  with fewer than `--min-pairs` (default 10) skips. Without `--once` it renders a
  systemd `.service`/`.timer` + launchd `.plist` scheduler scaffold into
  `--scheduler-dir` for the user to install. New flags: `--once`, `--min-pairs`,
  `--output/-o`, `--scheduler-dir`, `--hour`, `--minute`.
- **`soup distill-prompt` prepares a real distillation dataset** (#226). For
  each prompt in the traces JSONL the teacher is called once via the v0.20
  provider helpers (Ollama / Anthropic / vLLM); `sft`/`kl` emit
  `{messages:[user, assistant=teacher]}` and `preference` emits
  `{prompt, chosen=teacher, rejected=student}`. New flags: `--provider`,
  `--base-url`, `--temperature`, `--max-rows`.
- **`soup compile` runs DSPy / GEPA / TextGrad prompt-program optimisation** (#225)
  and **`soup compile-tools` runs the TextGrad / GEPA tool-schema optimiser** (#227),
  both lazy-importing the optimiser libraries behind the new `[compile]` extra
  (`pip install 'soup-cli[compile]'`) with a friendly `ImportError` naming the
  extra when absent. `--plan-only` still renders the plan and exits 0.

### Security
- **systemd / launchd injection defence** (#229). `local-rl` and the scheduler
  renderers reject `\n` / `\r` in the model id and shell-quote every `ExecStart`
  argument, so a crafted model id cannot inject extra unit directives.

### Fixed
- **`local-rl` train config rendered `output` as a mapping** (#229). The nightly
  `soup train` YAML now emits `output: <dir>` (a plain string the schema accepts)
  instead of `output: {dir: <dir>}`; a regression test validates the rendered
  config against `SoupConfig`.

## [0.71.12] - 2026-06-04

### Added
- **Architecture + distillation + adapter-training — live wiring** (closes #145,
  #146, #148, #158, #84, #221, #222). Seven surfaces that shipped schema-only in
  earlier releases are now real, validated end-to-end on tiny models
  (SmolLM2-135M / a locally-built tiny Llama).
- **Sequence-level knowledge distillation is live** (#145). `task: distill` now
  accepts `distill_mode: token|sequence`; sequence mode trains the student on the
  teacher's generated continuations (cross-tokenizer-friendly hard-label KD)
  instead of per-token logit matching. `sequence` mode is mutually exclusive with
  the v0.70 cross-tokenizer ULD logit path.
- **Classifier LoRA is live** (#146). `task: classifier|reranker|cross_encoder`
  now attaches a LoRA adapter to the sequence-classification head when `lora` is
  configured, so a frozen encoder + small adapter can be trained instead of the
  full model.
- **LLaMA Pro block expansion is per-architecture** (#148). `expand_layers`
  now interleaves zero-initialised identity blocks for Llama / Qwen / Mistral
  decoder stacks (was Llama-shaped only), with `freeze_trainable_layers`
  freezing the original blocks so only the new ones train.
- **LongLoRA S² shifted-sparse attention is live** (#158). `use_longlora: true`
  now installs the shifted-sparse-attention forward override on the Q/K
  projections (Llama / Mistral / Qwen / Phi), restoring the patched forwards on
  context exit.
- **Mixture-of-Depths is live** (#84). `use_mod: true` attaches a per-layer
  top-k token router (`mod_capacity_factor`) so only a subset of tokens receive
  each block's residual update. Architecture allowlist: Llama / Qwen / Mistral;
  unsupported bases warn and skip.
- **VeRA / VB-LoRA multi-tenant serving is live** (#221). `soup serve --bank
  <bank.json> [--bank-strength S]` reconstructs the shared projection + per-user
  scaling vectors and installs a decode-time forward hook; the active user is
  selected per request via the `X-User-Id` header (an unknown/absent id is a
  zero-delta no-op, so there is no cross-request leak). Serves N personas at
  ~KB-per-user instead of a full LoRA each.
- **MoLE per-token adapter routing is live** (#222). `task: moe_lora_routing`
  with `mole_task_adapters: [...]` trains a per-token gating network that blends
  N frozen task LoRAs (`mole_top_k` / `mole_temperature`); only the router
  trains. The gate is saved as `mole_gate.pt` alongside the run.

### Changed
- `apply_bank_to_serve` (#221) and `build_gating_kernel` (#222) now return live
  objects (a `LoadedVectorBank` and a `torch.nn.Module` router) instead of the
  v0.67.0 deferred-stub `NotImplementedError`.

## [0.71.11] - 2026-06-04

### Added
- **GRPO / RL callbacks — live wiring** (closes #235, #236, #237, #238, #239,
  #240, #159, #160). The reward-hacking, cross-tokenizer distillation, MiniLLM,
  mid-epoch RL checkpoint, iterative-DPO and echo-trap surfaces that shipped
  schema-only in v0.70.0 are now real, validated end-to-end on SmolLM2-135M.
- **Reward-hacking detector is live** (#235). `--reward-hack-detector
  info_rm|rm_ensemble` now installs a GRPO `TrainerCallback` that reads the
  per-step rewards (via a shared, thread-safe reward-fn capture buffer),
  computes an InfoRM cluster-separation drop (`info_rm`) or RM-ensemble
  divergence (`rm_ensemble`), classifies OK/WARN/HACK, logs the verdict to
  `state.log_history`, and halts training on HACK when `--reward-hack-halt` is
  set. `rm_ensemble` requires ≥2 reward functions.
- **Cross-tokenizer ULD distillation is live** (#236). `task: distill` with
  `--uld-strategy wasserstein|topk_align` now computes a real Wasserstein-1
  (sorted-CDF) or top-k-aligned distillation loss inside the distill trainer,
  handling student/teacher vocab-size mismatch by clamping teacher ids to the
  teacher vocab.
- **MiniLLM reverse-KL distillation is live** (#237). `--minillm-enabled` adds
  a teacher-mixed, length-normalised reverse-KL term plus an optional
  pretrain-anchor SFT term (`--minillm-pretrain-anchor-path` /
  `--minillm-pretrain-anchor-weight`) that keeps the student near coherent
  language. The anchor corpus reader is cwd-contained + symlink-rejecting with
  a per-line byte cap.
- **Mid-epoch RL checkpoint is live** (#238). `--rl-checkpoint-save-every-steps
  N` writes a real adapter + optimizer state + JSON manifest every N steps
  during PPO/GRPO and prunes to `--rl-checkpoint-keep-last`, so a long RL run
  survives a crash without losing the optimizer momentum.
- **`soup iterative-dpo` orchestrator is live** (#239). Runs the full
  sample → reward-score → build-pairs → DPO-train loop across rounds: each
  round samples completions from the previous round's adapter, the next round
  trains a fresh LoRA from the base on that round's harvested pairs.
  `--plan-only` still renders the plan without running.
- **Echo-trap detector is live** (#240). `--echo-trap-enabled` installs a GRPO
  callback that scores per-trajectory n-gram repetition, classifies
  OK/WARN/TRAP against `--echo-trap-threshold`, logs the verdict, and halts on
  TRAP when `--echo-trap-halt` is set (catches RAGEN-style degenerate
  repetition in multi-turn agent RL).
- **GRPO variant fallback now warns once** (#159). When a `--grpo-variant`
  custom `compute_loss` falls back to the base trainer (because the installed
  TRL renamed the loss inputs), the trainer logs a one-shot WARNING instead of
  silently degrading to the default objective.

### Changed
- **GRPO reference-model EMA no longer materialises full state dicts** (#160).
  `--ref-model-ema-alpha` now updates the reference model in place by iterating
  `named_parameters()` (`ref = (1-α)·ref + α·policy`), eliminating the three
  model-sized allocations per step the v0.53.11 path made. A total
  name/shape-mismatch (0 shared parameters) logs a one-shot WARNING so a
  misconfigured EMA can't silently no-op.

## [0.71.10] - 2026-06-03

### Added
- **RAG family — live wiring** (closes #199, #200, #201, #202). The four
  retrieval / steering surfaces that shipped schema-only in v0.62.0 are now
  real, validated on SmolLM2-135M.
- **RAFT span-mask training is live** (#199). `data.format: raft` rows
  (`{query, golden_doc, distractor_docs, answer}`) now train answer-only: the
  prompt span is masked to `-100` and each document is labelled `[doc-N]` so
  the model learns to cite the supporting document. Documents are shuffled
  reproducibly (`data.raft_shuffle_seed`). Rows whose prompt fills
  `max_length` (answer fully truncated) are dropped with a warning rather than
  silently shrinking the effective dataset.
- **`soup ra-dit` — one-shot two-stage orchestrator** (#200). Trains the
  retriever (stage 1, embedding/contrastive) then the generator (stage 2,
  RAFT-SFT) in a single command, recording the trained retriever as the
  generator's paired retriever. A `soup train` of a generator-stage config
  with no retriever model set now auto-links the most-recent RA-DIT retriever
  run from the Registry. `--plan-only` validates both configs without
  training; `--retriever-model` overrides the auto-link.
- **`soup steer train` / `apply` + `soup serve --steer` are live** (#201).
  Fit a CAA (contrastive activation addition), ITI (inference-time
  intervention) or RepE (representation-engineering PCA) control vector from
  `{positive, negative}` contrastive pairs, persist it as a safetensors +
  config artifact, and apply it at decode time via a forward hook
  (`soup serve --steer <name> --steer-strength <s>`).
- **`soup eval citation` + citation-span loss boost are live** (#202). Score
  citation precision / recall / F1 over `{predicted, expected_ids}` or RAFT
  rows (`--shuffle-seed` aligns the golden `[doc-N]` id with what the model
  saw at train time). When `citation_faithful: true`, bracketed `[doc-id]`
  spans in the answer get a boosted per-token loss weight. A new `citation`
  failure mode is available in `soup diagnose`.

## [0.71.9] - 2026-06-03

### Added
- **Knowledge edit + unlearn — live wiring** (closes #193, #194, #196, #197,
  #203). The v0.61.0 / v0.62.0 schema-only stubs are now live, validated on
  SmolLM2-135M.
- **`soup edit set` (ROME / MEMIT / AlphaEdit) is live** (#194). New
  `soup_cli/utils/edit_kernels.py` ships covariance-free rank-1 weight-edit
  kernels: ROME (single-layer `W += δ·kᵀ/‖k‖²`), MEMIT (residual distributed
  across a layer band), AlphaEdit (ROME update projected orthogonal to the
  down-proj's top singular direction). `apply_edit` loads the model, optimises
  the target residual, applies the rank-1 update, and optionally saves with
  cwd-containment + symlink rejection. `--output`, `--device`, `--governor/
  --no-governor` flags added. On a tiny model a ROME edit moved
  `P("Lyon" | "The capital of France is")` from 0.0016 → 0.96.
- **`soup edit diff` live before/after generation** (#194). Pass
  `--before-model` + `--after-model` (+ `--probes`) to generate completions
  through both models and surface the probes whose output changed.
- **EditGovernor SQLite persistence + cross-process locking** (#196). New
  `EditGovernorStore` (mirrors `namespace_pin.NamespacePinStore` —
  $HOME/$CWD/$TMPDIR containment, TOCTOU symlink rejection, WAL +
  busy_timeout, `fcntl`/`msvcrt` sidecar lock, POSIX 0600). `save_governor` /
  `load_governor` / `default_governor_db_path` (env override
  `SOUP_EDIT_GOVERNOR_DB`) persist per-base-model edit-count + verdict across
  separate `soup edit set` runs.
- **`apply_edit` consults the EditGovernor automatically** (#197). When a
  governor is supplied, `check_can_edit()` runs BEFORE the model load (refusing
  on norm blowup / edit cap) and `record_edit()` runs AFTER with the measured
  Frobenius delta.
- **Live GRACE codebook** (#203). `GraceCodebook` (epsilon-ball nearest-key
  lookup), `apply_grace_edit` (captures a residual key + optimises a value +
  appends to a codebook sidecar), `save_codebook` / `load_codebook` (atomic,
  cwd-contained, symlink-rejected), `install_grace_hook` (decode-time residual
  substitution). New `edited_model` / `grace_codebook` Registry artifact kinds.
- **`soup train --task unlearn` is live (NPO / SimNPO / RMU)** (#193). New
  `soup_cli/utils/unlearn_kernels.py` (NPO `(2/β)·mean(-logσ(-β(πlp-reflp)))`,
  length-normalised SimNPO, RMU representation steering) + a self-contained
  `UnlearnTrainerWrapper` loop loading a LoRA policy, a frozen reference
  (NPO/RMU), and forget/retain JSONL datasets. NPO/SimNPO forget loss
  decreased on the tiny-model smoke. Warns when run without a retain set.

### Security
- `_save_edited_model` / `UnlearnTrainerWrapper` output dirs + `save_codebook`
  / `load_codebook` + `_load_unlearn_rows` enforce cwd-containment, raw-path
  symlink rejection (TOCTOU), null-byte rejection, and file-size / per-line
  caps. `apply_grace_edit` honours the governor for direct callers.

## [0.71.8] - 2026-06-03

### Added
- **Probes & SAE — real weights + live downloads** (closes #215, #216, #217,
  #218, #219). A new shared `soup_cli/utils/probe_kernel.py` provides the
  linear-probe math (contrast-pair derivation, apply, flag-rate, verdict bands,
  operator-supplied weight loading, deterministic synthetic fallback); every
  heavy import (`numpy` / `torch` / `safetensors`) is lazy.
- **`soup probe sleeper --weights <w.npz|.npy|.safetensors>`** (#215) — load a
  real calibrated probe direction instead of the synthetic fallback. Weights are
  cwd-contained, symlink-rejected, `O_NOFOLLOW`-opened, `allow_pickle=False`,
  and size-capped. `compute_contrast_probe(positive, negative)` derives a probe
  from contrast-pair activations.
- **`soup probe sae-diff <repo> --auto-download`** (#216) — fetch an
  allowlisted SAE from the HF Hub into `~/.soup/sae-cache/` (validated against
  `HF_HUB_ALLOWLIST` BEFORE any network call) via a new SSRF-hardened
  `soup_cli.utils.hubs.snapshot_download` (repo-id shape + home/cwd/tmp cache
  containment + namespace-pin TOFU gate).
- **`soup probe truth` / `soup probe harm`** (#217) — TruthfulQA-style honesty
  and HarmBench-style misuse activation probes (6 bundled bases each, 5% / 20%
  verdict bands, `--weights` to skip the allowlist with a real probe). The
  probe pack now ships truth + harm entries per base.
- **`soup probe interference --measure <eval_suite> --base-model <m> --adapter
  name=path ...`** (#218) — auto-measure the N×N interference matrix by actually
  loading the base + each LoRA adapter (PEFT multi-adapter), measuring loss for
  each adapter alone (diagonal) and each co-loaded pair
  (`add_weighted_adapter(combination_type="cat")`, off-diagonal). Exit 2 on a
  MAJOR worst-pair.
- **`soup train --capture-activations <layer> --capture-prompts <jsonl>`** (#219)
  — a post-training hook writes an SAE-diff-ready per-token activation snapshot
  to `<output>/activations/activations.json`. `resolve_layer_module` resolves
  the same `model.layers.N` path whether or not a LoRA adapter is loaded
  (PEFT-wrapper fallback).

### Security
- Probe / SAE / capture file I/O is cwd-contained + `O_NOFOLLOW` (TOCTOU close)
  + size-capped; SAE weight loads use `allow_pickle=False`. SAE auto-download
  validates the allowlist before any network call and rejects a glob result
  that resolves outside the snapshot dir (symlink-escape guard).

### Notes
- #215 is partial: the operator-supplied / contrast-pair / synthetic paths ship
  now, but the 6 large-base Anthropic-calibrated probe vectors remain
  upstream-gated (no public calibrated artifact exists). Documented as a known
  limitation.

## [0.71.7] - 2026-06-02

### Added
- **Eval live runners** — six probe surfaces that previously emitted heuristic
  / neutral stubs now load a real model and run live (closes #161, #162, #208,
  #211, #212, #165). New shared `soup_cli/utils/live_eval.py` provides the
  model-loading primitives (generator / multi-generator closures, masked
  cross-entropy eval-loss, a short-LoRA probe, and held-out logit agreement);
  every heavy import (`torch` / `transformers` / `peft` / `lm_eval`) is lazy.
- **`soup advise --probe-model <id>`** — runs a LIVE ROI probe: zero/few-shot
  token-F1 baselines, a short LoRA probe (relative held-out-loss improvement +
  real wall-clock), and base-model proximity (held-out logit agreement) folded
  into the dataset profile. Without `--probe-model`, `--probe` stays the offline
  heuristic.
- **`soup tunability --live`** — replaces the offline heuristic with a real
  per-candidate LoRA probe (loads each `repo_id`, trains `--probe-steps` on a
  held-out-excluded slice, reports the held-out-loss drop).
- **`soup eval capability --live --model <id>`** — invokes lm-eval-harness per
  resolved task (or a `--tasks` override) with `--limit` / `--device`, isolating
  per-task failures and surfacing a no-metric result as an explicit error.
- **`soup eval behavior --base-model <id> [--adapter <path>]`** — generates
  pre/post responses on the bundled behaviour battery and scores the live diff.
- **`soup diagnose --base-model <id> [--adapter <path>] [--dataset <jsonl>]
  [--tokenizer <id>]`** — runs all six failure-mode probes (forgetting / refusal
  / format / mode_collapse / memorization / contamination) live via
  `soup_cli.utils.diagnose.live.run_live_diagnose`; falls back to neutral OK or
  `--evidence` JSON when no model is supplied.

### Security
- The two new JSONL dataset readers (`diagnose.live._load_dataset_rows`,
  `tunability._load_jsonl_rows`) open with `O_NOFOLLOW` after the cwd-containment
  check, closing the check→open TOCTOU window (matches the v0.65 / v0.67 reader
  policy).

## [0.71.6] - 2026-06-02

### Added
- **`soup build` live runner** — the dbt-for-SFT DAG (`soup build <manifest>`) now
  *materialises* datasets instead of only dry-running the plan. Five built-in
  transforms ship live (`identity`, `drop_empty`, `lowercase`, `strip`,
  `dedup_exact`); `table` rebuilds from scratch, `view` re-derives on every run,
  and `incremental` re-transforms only the rows whose content hash changed
  (tracked in a SQLite state store, keyed by row hash **and** the model's
  transform+config fingerprint so a transform change re-runs everything). Custom
  transforms are passed per-run via the Python API's `transforms=` map. Outputs
  are written atomically; the `--output-dir` is symlink-checked before any
  directory is created.
- **`soup data gen-magpie` live generator** — the Magpie synthetic generator
  (Xu et al. 2024) now actually generates. It feeds an aligned model its
  chat-template prefix (chatml / llama3 / gemma / mistral families auto-detected)
  and harvests the self-generated user instruction + assistant response via raw
  completion. Live providers: `ollama` (`/api/generate` raw) and `vllm`
  (`/v1/completions`) — both SSRF-hardened (loopback-only HTTP); `anthropic` is
  rejected (no raw-completion endpoint). Optional `--quality-filter` drops
  low-quality rows via the v0.47 toxicity/educational scorers; exact-duplicate
  instructions are de-duplicated.
- **`soup eval irt-subset --model {1pl,2pl,3pl}`** — the IRT eval-cost optimiser
  gained 2PL (per-item discrimination) and 3PL (+guessing floor) joint
  coordinate-ascent MLE fits alongside the existing 1PL Rasch. `1pl` keeps the
  closed-form path for back-compat; `2pl`/`3pl` route through the new `fit_irt`.
- **Tokenizer-aware memorization probe** — `score_memorization(..., tokenizer=...)`
  and `split_prefix(..., tokenizer=...)` (used by `soup diagnose`) now split the
  prefix/suffix on real token-id boundaries and measure echo-overlap over
  sub-word tokens when a tokenizer (HF id / path / duck-typed object) is supplied,
  catching BPE-level memorization that whitespace tokenisation misses. Default
  (no tokenizer) keeps the whitespace behaviour.

### Fixed
- **`soup data augment --provider ollama|vllm` no longer crashes** — the command
  imported a non-existent `OllamaProvider` symbol and raised `ImportError` on
  every non-OpenAI provider. It now routes through the shared, SSRF-hardened
  provider factory; `--model` / `--base-url` are honoured, the output path is
  containment- and symlink-checked, and the write is atomic.

### Security
- **Ollama / vLLM provider URLs reject `0.0.0.0`** — `validate_ollama_url` /
  `validate_vllm_url` dropped the bind-any wildcard from their loopback allow-set
  (now `localhost` / `127.0.0.1` / `::1` only), matching the newer
  `validate_hub_endpoint` / `validate_webhook_url` SSRF validators. Reachable now
  that Magpie threads a user-supplied `--base-url` through these providers.

## [0.71.5] - 2026-06-02

### Added
- **`soup eval against` now reads eval metrics** — `ExperimentTracker.get_metric_series`
  falls back to the `eval_results` table when the metric is not a per-step
  training column (`loss` / `lr` / `grad_norm` / `speed` / `gpu_mem`). So
  `soup eval against <base> --candidate <run> --metric task_accuracy` returns a
  real score series (benchmark scores live in `eval_results`, not `metrics`)
  instead of "Empty series". Per-step columns still read from `metrics` — no
  regression for existing callers.
- **`soup advise` learns from past project outcomes** — `soup advise` now reads
  this project's accepted-verdict history (`~/.soup/advise_history.jsonl`) and
  biases the rubric: 3+ successful SFT precedents flip a marginal RAG call to
  SFT; 3+ negative GRPO outcomes suppress GRPO in favour of SFT-on-traces; an
  encouraged choice gets a small confidence nudge. Scoped per-project (one
  project's record never biases another). No history → identical to before.
- **Slack/Discord webhooks on four more commands** — `--slack-url` / `--discord-url`
  (SSRF-hardened, loopback-only HTTP, RFC1918 rejected, never crashes the
  command) now ship on `soup ingest`, `soup prune-prompt`, `soup ab` (fires only
  on a `reject_h0` / `accept_h0` decision, not `continue`), and
  `soup data active-sample` — not just `soup drift-alarm`. The validator + sender
  moved to a shared `soup_cli/utils/webhooks.py`.
- **Tokenizer-aware `soup prune-prompt`** — `--tokenizer <model_or_path>` detects
  and strips the shared system-prompt prefix on **token** boundaries instead of
  characters, so a multi-byte UTF-8 prefix can never be split mid-code-point.
  Default (no `--tokenizer`) keeps the whitespace-character behaviour.
- **Curriculum bucketing by loss percentile** — `DynamicCurriculumCallback` now
  buckets samples by the percentile rank of the live loss (or perplexity)
  signal within a rolling window when `data.curriculum_metric` is `loss` /
  `perplexity`, so a consistently-hard sample is routed to the same difficulty
  bucket across recomputes. `length` and warm-up still use round-robin.
- **`--hub` on `soup data push` and `soup data forge`** — `soup data push
  --hub modelscope|modelers` uploads a dataset via the matching SDK
  (`repo_type=dataset`, commit message sanitised); `soup data forge --hub
  <non-hf> --teacher owner/name` pre-fetches the teacher model from that hub
  (and warns when the teacher is not a repo id so `--hub` is never silently
  ignored). HF stays the default.

### Notes
- Live SaaS *pull* adapters for `soup ingest` (Langfuse / LangSmith / Helicone /
  OpenPipe / OpenAI SDKs, issue #204) remain deferred: they need credentialed
  vendor accounts with populated trace data to validate honestly. Tracked as an
  open, `infra-blocked` (external-account) item. `soup ingest` continues to parse
  the JSONL export you pull from your dashboard.

## [0.71.4] - 2026-06-02

### Added
- **Live canary verdict for `soup adapters merge`** — `--canary <suite.json>`
  scores the merged adapter against the first input and classifies
  **OK / MINOR / MAJOR** using the Quant-Lobotomy taxonomy (drop <2% OK, <5%
  MINOR, else MAJOR). `--strict-verdict` exits 2 on MAJOR. Pre-scored
  `{"baseline_scores","candidate_scores"}` suites run with no model load; a
  `{"tasks":[...]}` suite uses an injectable scorer. Replaces the v0.57 `UNKNOWN`
  stub.
- **Live evolutionary merge** — `soup adapters merge --strategy cmaes --eval
  <suite> --budget <t>` now runs the full CMA-ES loop: each candidate is merged,
  materialised, scored against the eval suite, and the best-weighted merge is
  written to `--output`. Replaces the v0.67 plan-only stub.
- **Publish an adapter PR to GitHub** — `soup adapters pr <title> --base-sha
  <hex> --adapter <path> --push owner/repo#N` posts the rendered PR Markdown as a
  GitHub PR comment via `gh api` (argv-list, body over JSON stdin; no shell).
  Token resolves from `GITHUB_TOKEN` / `GH_TOKEN`.
- **Pre-wired `soup loop` production stages** — `soup loop init --pre-wired` (or
  `soup loop watch --pre-wired`) swaps the v0.58 no-op stage stubs for real
  harvest (traces → preference pairs) → DPO train → eval-gate → canary-deploy
  callables. `soup loop status` now shows the `pre_wired` flag.
- **Loop iterations as Soup Cans + Registry lineage** — `soup loop watch
  --pack-cans` packs each successful iteration as a v0.26 Soup Can and appends a
  Registry entry (tag `loop-iter`), chaining a real lineage DAG across
  iterations visible through `soup history`. `soup loop replay <id> --extract
  <dir>` unpacks a recorded iteration.
- **Branch pointers into the Registry** — `soup adapters branch <name>
  --attach-to-registry <id>` links a branch snapshot to a Registry entry (shown
  as a `branches` node in `soup history`); `soup adapters branch <name>
  --from-registry <id>` derives a fresh snapshot's config + base from an entry.

### Security
- The backdoor-scan gate (v0.71.2 #192) and license-conflict gate (v0.60 Part E)
  now run for **all** merge strategies, including `--strategy cmaes` (previously
  bypassed because cmaes returned before the gates).
- `soup loop` canary deploy restricts `SOUP_LOOP_SERVE_ENDPOINT` to loopback /
  RFC1918-private hosts (a serve endpoint is the operator's own box/LAN), beyond
  the general webhook SSRF policy which permits any HTTPS host.
- `soup adapters pr --push` builds the `gh` child environment from an allowlist
  so unrelated secrets (`HF_TOKEN` / `OPENAI_API_KEY` / …) never reach the
  subprocess.
- The canary-suite JSON read uses `O_NOFOLLOW` + `os.fstat` (size cap enforced on
  the same fd) to close the symlink/size-cap TOCTOU window.

## [0.71.3] - 2026-06-01

### Added
- **Energy & CO2 measurement for training** — `soup train --track-energy` wraps
  the training window in a codecarbon **offline** tracker (no IP-geolocation
  network call) and reports kWh / CO2 / grid intensity, feeding those numbers
  into `--annex-xi`. New `EnergyTracker` context manager; graceful no-op when
  codecarbon is absent (`pip install soup-cli[carbon]`). `--energy-country`
  picks the ISO-3166 alpha-3 grid for the CO2 estimate (default `USA`).
- **PDF Annex XI/XII documents** — `soup train --annex-xi report.pdf` now renders
  a reportlab PDF (a `.md` path still renders markdown). `pip install
  soup-cli[pdf]`.
- **Auto-populated training-corpus domains in Annex XI/XII** — the top crawled
  domains (with shares) are now extracted from the training JSONL and listed in
  the EU AI Act docs, replacing the previous empty placeholder.
- **Soup Can manifest v3 with embedded attestations** — `soup can pack --attest
  <statement.json>` (repeatable) embeds in-toto Statements into a v3 can
  manifest; v1/v2 cans still load. Each statement is shape- and size-validated.
- **Local audit log auto-instrumentation** — every `soup` command now appends one
  HIPAA/SOC2-shaped record to `~/.soup/audit.jsonl` (secrets redacted, args
  capped). Opt out per-invocation with `--no-audit-log` or globally with
  `SOUP_NO_AUDIT_LOG=1`. Tail/rotate with `soup audit-log`.
- **Reproducibility receipt in airgap bundles** — `soup airgap-bundle
  --repro-receipt <receipt.json>` embeds an SR 11-7 receipt as
  `repro-receipt.json`; auto-detected from `<model>/repro-receipt.json` when not
  supplied.

### Security
- `soup can pack --attest` now rejects oversize attestation files by their raw
  size *before* parsing them into memory (defence against memory-exhaustion).
- The new file-loading paths (attestation JSON, airgap receipt, training-corpus
  scan, PDF write) are all cwd-contained + TOCTOU symlink-rejected and
  size-capped; the audit auto-log redacts `hf_`/`sk-`/`Bearer` tokens and never
  crashes the CLI on a broken log.

## [0.71.2] - 2026-06-01

### Added
- **ed25519 signing for `soup adapters sign` / `soup attest`** — real detached
  signatures (over the adapter Merkle root / the in-toto statement) via a new
  `[sign]` extra (`pip install soup-cli[sign]`, pulling `cryptography`).
  `soup adapters sign --backend ed25519 --key <priv.pem>` (or `--generate-key
  <out.pem>`, or `SOUP_SIGNING_KEY`); `soup adapters verify [--public-key
  <trusted.pem>]` does a cryptographic verify and, with a trusted key, genuine
  authentication. `soup attest emit --sign ed25519 --key <priv.pem>` writes a
  `<output>.sig` sidecar; new `soup attest verify <statement> --signature <sig>`
  verifies it (canonical-JSON, so it's platform/newline-independent). Sigstore
  keyless signing stays infra-blocked (needs an OIDC provider + Fulcio/Rekor
  network — can't be honestly validated offline).
- **Anti-AI-Jacking namespace pin on Hub downloads** — HF model fetches now
  consult a trust-on-first-use pin store: a repo whose author changes (or whose
  `created_at` jumps backward) is refused unless the namespace shift is explicitly
  allowed. Fails open when repo metadata is unavailable.
- **License auto-detection at `soup adapters merge`** — when `--license` isn't
  given, the license is read from each adapter's `adapter_config.json` /
  `config.json` / model-card frontmatter (HF `llama3.1`-style ids mapped to
  canonical) and the conflict gate runs automatically.
- **Backdoor-scan gate at `soup adapters merge`** — refuses to merge any input
  whose `soup adapters scan` returns FAIL (or can't be scanned) unless
  `--allow-unscanned` is passed; WARN is advisory.

### Changed
- License-conflict overrides (`--license-override <reason>`) are now recorded to
  the audit log for legal review.
- The namespace-pin store now uses SQLite WAL + busy-timeout and a cross-process
  file lock around its get+insert, so concurrent writers don't lose the trust
  anchor.

### Security
- ed25519 verification fails closed (any tamper / wrong key / missing key ⇒
  invalid). Signing keys + trusted public keys are symlink-rejected and
  size-capped via a shared reader (no cwd-containment — keys live outside the
  project). `--generate-key` refuses to overwrite any existing path.

## [0.71.1] - 2026-06-01

### Added
- `soup env fix` — render a reproducible install plan from `soup-env.lock`.
  Emits copy/paste `uv pip install` commands (`--format uv-pip`, default) or a
  `requirements.txt` body (`--format requirements`); `--output` optionally writes
  a `requirements.txt` under cwd. Print-only by design — never shells out to a
  package manager.
- `soup lock write --env-lock <path>` — auto-derive `--env-hash` from a
  `soup-env.lock` so operators who ran `soup env lock` don't copy the hash by
  hand. `--env-hash` still wins when passed explicitly.
- `soup serve --record-thumbs <db>` — capture thumbs-up/down feedback into a
  local-RL SQLite at startup, plus a new `POST /v1/thumbs` endpoint (transformers
  backend). Returns 404 when the flag isn't set.
- Judge-calibration persistence: `JudgeCalibrationReport.to_dict`,
  `write_judge_calibration`, and `load_judge_calibration`, backed by a new
  `judge_calibration` registry artifact kind. Loading re-validates the report so
  a corrupt on-disk field is rejected.
- Bundled MUSE and WMDP unlearning eval fixtures so
  `soup eval unlearning --benchmark muse|wmdp` runs out of the box. WMDP
  forget-set probes ship **redacted** (placeholder prompts + `REFUSED` responses)
  — Soup never ships verbatim hazardous content.

### Changed
- `soup completions` now introspects a cached base model's actual LoRA target
  modules (config-only `AutoConfig` load, `local_files_only=True`, never networks
  or raises) and falls back to the canonical default shape when the base isn't
  cached locally.
- `build_dag` exposes a `validate_build_source` helper (cwd-containment +
  symlink rejection) for build-manifest source paths.

## [0.71.0] - 2026-06-01

### Changed
- **Breaking — install split.** The heavy training stack (`torch`,
  `transformers`, `peft`, `trl`, `datasets`, `bitsandbytes`, `accelerate`) moved
  out of the core install into a new `[train]` extra. `pip install soup-cli` is
  now a light CLI + data-tools install with **no PyTorch**; run
  `pip install 'soup-cli[train]'` (or `[all]`) to fine-tune. Existing users who
  train must reinstall with `[train]`. Version pins are unchanged.
- Trimmed `README.md` to a ~238-line front door; the full feature reference now
  lives under `docs/` (one topic page per area, indexed from the README).
- Raised the pytest coverage gate from 50% to 77% (`--cov-fail-under=77`).
- Migrated to a `src/` layout (`src/soup_cli/`) for cleaner packaging and to
  stop tests accidentally importing the in-tree package.

### Added
- `[train]` and `[all]` optional-dependency extras (`[all]` pulls
  `train`, `serve`, `ui`, `data`). `[dev]` self-references `[train]` so CI and
  contributors still get the full stack from `pip install -e ".[dev]"`.
- Friendly error mapping: a missing heavy dependency (`torch`, `transformers`,
  `peft`, `trl`, `datasets`, `bitsandbytes`, `accelerate`) now surfaces
  "Training needs the [train] extra. Run: pip install 'soup-cli[train]'".
- `py.typed` marker (PEP 561) so downstream type checkers pick up Soup's inline
  type hints.
- `.pre-commit-config.yaml` with ruff (lint + format) and standard file-hygiene
  hooks.
- Lenient `mypy` configuration and a non-blocking `type-check` CI job.
- This `CHANGELOG.md`.

### Removed
- The historical, per-version security-fix log that had grown inside
  `SECURITY.md` (~220 KB). `SECURITY.md` is now a concise security policy; the
  detailed hardening notes remain in git history and the GitHub Releases notes.

[Unreleased]: https://github.com/MakazhanAlpamys/Soup/compare/v0.71.0...HEAD
[0.71.0]: https://github.com/MakazhanAlpamys/Soup/compare/v0.70.0...v0.71.0
