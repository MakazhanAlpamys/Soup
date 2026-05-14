# Contributing to Soup

Thank you for your interest in contributing to Soup! We welcome bug reports, feature requests, and pull requests from the community.

## Getting Started

### 1. Fork & Clone

```bash
git clone https://github.com/MakazhanAlpamys/Soup.git
cd Soup
```

### 2. Set Up Development Environment

**Requirements:** Python 3.9+

Install the project in editable mode with dev dependencies:

```bash
pip install -e ".[dev]"
```

This installs:
- `pytest` for testing
- `ruff` for linting
- `pytest-cov` for coverage
- `httpx` for HTTP testing

### 3. Verify Setup

Run the test suite to confirm everything works:

```bash
pytest tests/ -v --tb=short
```

Run the linter:

```bash
ruff check soup_cli/ tests/
```

If both pass — you're ready to contribute!

## Code Style

We use **ruff** for all code style and linting. Before committing, run:

```bash
# Check for issues
ruff check soup_cli/ tests/

# Auto-fix issues
ruff check --fix soup_cli/ tests/
```

### Style Guidelines

- **Line length:** 100 characters (enforced by ruff)
- **Imports:** Sorted and organized (ruff I rule)
- **Naming:** No single-letter variable names (ruff E741) — use `entry`, `part`, `length` instead of `l`, `p`, etc.
- **Lazy imports:** Heavy dependencies (torch, transformers, peft, trl, etc.) should be imported inside functions, not at module level, to keep the CLI responsive
- **Config validation:** Always use Pydantic v2 with `BaseModel` and `Field`
- **Output:** Use `rich.console.Console` for all output — never bare `print()`
- **Type hints:** Always include type hints for function parameters and return values

Example:

```python
# WRONG
from torch import cuda
import transformers

def train():
    print("Starting training")
    model = transformers.AutoModel.from_pretrained("llama-7b")

# CORRECT
def train():
    from torch import cuda
    import transformers

    console = Console()
    console.print("Starting training")
    model = transformers.AutoModel.from_pretrained("llama-7b")
```

## Project Structure

Key directories:

```
soup_cli/
  cli.py              - Main entry point, command routing
  commands/           - Command implementations (train, chat, eval, deploy, etc.)
  config/             - Config schema (schema.py) and loader (loader.py)
  data/               - Data loading, format conversion, providers, templates
  trainer/            - Training wrappers (SFT, DPO, GRPO, PPO, KTO, ORPO, SimPO, IPO, Pretrain, Reward Model, Embedding)
  monitoring/         - Callbacks and live dashboard
  experiment/         - SQLite experiment tracking
  eval/               - Eval platform (custom tasks, LLM judge, human eval, leaderboard)
  migrate/            - Config migration (LLaMA-Factory, Axolotl, Unsloth)
  recipes/            - Ready-made configs for popular models (80 recipes)
  autopilot/          - Zero-config decision engine (v0.25.0)
  registry/           - Model Registry (hashing, store, diff, attach) (v0.26.0 + v0.33.0)
  cans/               - Shareable .can artifact format + run/publish orchestrator (v0.26.0 + v0.33.0)
  data/traces/        - Trace-to-Preference harvester (v0.26.0)
  data/collators.py   - CrossDocCollator for sample packing (v0.33.0)
  utils/              - GPU, errors, MoE, GaLore, QAT, Unsloth, vLLM, SGLang, Liger, FlashAttn, FSDP, Ring Attention, long-context, quality, curriculum, freeze, dataset-registry, mlx, peft_builder, paths, topology, launcher, mii, pipeline, cut_ce, fp8, gradient_ckpt, kernel_picker, cross_doc_attn, activation_offload, hf, spec_pairing, structured_output, metrics, tracing, auto_quant, lr_finder, grad_accum, mixed_precision, warmup, spike_recovery, convergence, v028_features, multipack_sampler, multipack, neat_packing, jinja_analyzer, quant_menu, relora, peft_patches, peft_wiring, dpo_variants, optimizer_zoo, lr_groups, loftq_init, block_expansion, tts, classifier, distill, bitnet, ebft_gdpo, moe_quant, reasoning_effort, gguf_quant, kv_cache, advanced_precision, save_formats, deploy_measure
  templates/          - 17 built-in soup.yaml templates (YAML + manifest.json) with load_template loader (v0.39.0, +bco v0.40.0)
  ui/                 - Web UI (FastAPI + HTML/JS SPA)

tests/                - Test suite (190 files, 8330 tests)
examples/             - Real-world config examples and datasets
```

## Running Tests

### All Tests

```bash
pytest tests/ -v --tb=short
```

### Single Test File

```bash
pytest tests/test_config.py -v
```

### Single Test

```bash
pytest tests/test_data.py::test_detect_alpaca_format -v
```

### With Coverage

```bash
pytest tests/ --cov=soup_cli --cov-report=html
```

### Test Files (86 files)

| File | Covers |
|------|--------|
| test_config.py | Config loading, validation, defaults |
| test_data.py | Format detection, conversion, validation |
| test_gpu.py | GPU detection, batch size estimation |
| test_cli.py | CLI commands, version --full |
| test_tracker.py | SQLite experiment tracker |
| test_runs.py | `soup runs` CLI commands |
| test_data_tools.py | Data convert/merge/dedup/stats commands |
| test_eval.py | Eval command |
| test_smoke_train.py | Full pipeline smoke tests (GPU) |
| test_chat.py | Chat command, `_detect_base_model` |
| test_push.py | Push command, `_format_size`, `_generate_model_card` |
| test_init.py | Init command, templates, overwrite logic |
| test_callback.py | `SoupTrainerCallback` (mock-based) |
| test_display.py | `TrainingDisplay` rendering |
| test_loader.py | Data loading (JSONL/JSON/CSV, edge cases) |
| test_validator.py | `validate_and_stats`, `extended_stats`, `_percentile` |
| test_formats.py | Reverse conversion, round-trips, edge cases |
| test_merge.py | Merge command, adapter detection, validation |
| test_export.py | Export command, GGUF quant types, validation |
| test_resume.py | Resume checkpoint resolution, W&B flag |
| test_serve.py | Serve command, FastAPI app, endpoints, streaming |
| test_generate.py | Data generate, JSON parsing, validation, prompts |
| test_sweep.py | Sweep params parsing, combinations, nested config |
| test_diff.py | Diff prompts collection, metrics, CLI |
| test_deepspeed.py | DeepSpeed configs, multi-GPU detection, trainer integration |
| test_errors.py | Friendly error messages, --verbose flag, error mapping |
| test_doctor.py | `soup doctor` command, version checking, system resources, dependency table |
| test_quickstart.py | `soup quickstart` demo, data/config creation, --dry-run |
| test_grpo.py | GRPO config, rewards, data prep, template, sweep shortcuts |
| test_progress.py | Rich download progress bar, `_enable_hf_transfer_progress` |
| test_unsloth.py | Unsloth backend config, detection, trainer integration, templates |
| test_vision.py | Vision modality config, LLaVA/ShareGPT4V formats, loader, trainer, templates |
| test_qat.py | QAT config, validation, trainer integration, export compatibility |
| test_ui.py | Web UI command, FastAPI endpoints, auth, static files, config validation |
| test_vllm_serve.py | vLLM backend detection, engine creation, serve --backend flag, FastAPI app |
| test_ppo.py | PPO config, reward model config, data prep, RLHF template, routing, sweep |
| test_kto.py | KTO config, data format, template, routing, sweep, train guard, wizard |
| test_orpo.py | ORPO config, template, routing, sweep, train guard, wizard |
| test_simpo.py | SimPO config, template, routing, sweep, train guard |
| test_ipo.py | IPO config, template, routing, sweep, train guard |
| test_advanced_peft.py | DoRA, LoRA+, GaLore config, validation, sweep shortcuts |
| test_infer.py | Batch inference command, prompt reading, CLI validation |
| test_tensorboard.py | TensorBoard flag, wandb conflict, report_to routing |
| test_pretrain.py | Pretrain task, plaintext format, MoE config, templates, routing |
| test_moe.py | MoE detection, ScatterMoE LoRA targets, MoE info extraction |
| test_bugfixes.py | v0.10.1-v0.14.3 regression fixes |
| test_cli_subprocess.py | Subprocess CLI tests: entry point, encoding, paths, platform regressions |
| test_performance.py | Liger Kernel, FlashAttention, FSDP2, Ring Attention, long-context, RoPE scaling |
| test_embedding.py | Embedding task config, format, template, routing, sweep, pooling |
| test_onnx_tensorrt_export.py | ONNX export, TensorRT-LLM export, format support |
| test_speculative_decoding.py | Speculative decoding CLI, draft model, vLLM integration |
| test_server_generate.py | Server provider for data generate, SSRF validation |
| test_quality_filter.py | Perplexity + coherence scoring, `soup data filter` |
| test_audio.py | Audio modality config, format, template, routing, loader |
| test_sglang_serve.py | SGLang backend detection, runtime creation, serve --backend |
| test_deploy_ollama.py | Ollama deploy, Modelfile gen, template mapping, security validation |
| test_eval_platform.py | Custom eval, judge, human eval (Elo), leaderboard, compare, auto-eval, security |
| test_synth_data_pro.py | Providers (Ollama, Anthropic, vLLM), templates, quality pipeline, SSRF |
| test_migrate.py | LLaMA-Factory/Axolotl/Unsloth migration, path traversal, round-trip validation |
| test_recipes.py | Recipe catalog, search, CLI (list/show/use), path traversal |
| test_neftune_rslora.py | NEFTune config/validation/sweep, rsLoRA config/validation/sweep |
| test_profile.py | Training profiler: memory estimation, speed, GPU recommendations, CLI |
| test_multi_adapter.py | Multi-adapter serving: validation, parsing, FastAPI endpoints, CLI |
| test_data_sample.py | Data sampling: random/diverse/hard strategies, CLI, edge cases |
| test_adapters.py | Adapter management: list/info/compare, discovery, metadata |
| test_awq_gptq_export.py | AWQ/GPTQ export: format support, CLI, quantize mocks, calibration, security |
| test_packing.py | Sample packing: config, YAML, trainer integration, sweep |
| test_data_split.py | Data split: ratio/absolute/stratified splits, seed, edge cases |
| test_curriculum.py | Curriculum learning: config, length sort, buckets, sweep |
| test_dataset_hub.py | HF dataset search, preview, download, format conversion, security |
| test_freeze_training.py | Freeze training: config, layer freezing, GPT-2 naming, sweep |
| test_loss_watchdog.py | Loss watchdog: config, callback behavior, patience, sweep |
| test_dataset_registry.py | Dataset registry: CRUD, CLI, name validation, error handling |
| test_tool_calling.py | Tool-calling format detection, normalization, eval scoring, recipes (v0.25.0) |
| test_rlvr.py | RLVR verifiable rewards: math_verify, code_exec sandbox, json_schema (v0.25.0) |
| test_peft_methods.py | VeRA + OLoRA LoraConfig, peft_builder, sweep integration (v0.25.0) |
| test_mlx_backend.py | Apple Silicon MLX backend: detection, trainers, routing (v0.25.0) |
| test_data_augment.py | Data augmentation: rephrase/translate/style strategies, CLI, security (v0.25.0) |
| test_training_intelligence.py | Forgetting detection + checkpoint intelligence + SQLite (v0.25.0) |
| test_autopilot.py | Autopilot: analyzers, decision engine, CLI (v0.25.0) |
| test_registry.py | Model Registry: hashing, store CRUD, artifacts, lineage DAG, diff, CLI, history (v0.26.0) |
| test_eval_gate.py | Eval-Gated Training: config, suite loading, baseline, callback, CLI (v0.26.0) |
| test_trace_to_pref.py | Trace-to-Preference: LangChain/OpenAI/Soup-serve parsers, pair builder, CLI (v0.26.0) |
| test_quant_check.py | Quant-Lobotomy: classify_delta, resolve_model_ref, render formats, CLI (v0.26.0) |
| test_cans.py | Soup Cans: manifest schema, pack/unpack, tar traversal, fork security, CLI (v0.26.0) |
| test_multi_gpu.py | Multi-GPU Mastery: topology, --gpus, accelerate launcher, ZeRO++, FSDP2+compile, pipeline (v0.27.0) |
| test_training_speed.py | Training Speed & Memory: CCE, FP8, grad-ckpt tiers, kernel picker, cross-doc attn, activation offload (v0.28.0) |
| test_hf_integration.py | HF Hub Deep Integration: token/endpoint/repo_id, auto-push callback, model card v2, collections, data push, HF Spaces, private-IP SSRF (v0.29.0) |
| test_inference_advanced.py | Inference Excellence: prefix caching, spec-decoding auto-pairing, LoRA hot-swap, structured output, dashboard + /metrics, OpenTelemetry tracing, auto-quant picker (v0.30.0) |
| test_recipes_v031.py | Model & Recipe Breadth: 34 new recipes (vision/audio/reasoning/edge/domain/multimodal); catalog-wide invariants; CI workflow validation (v0.31.0) |
| test_auto_tuning.py | Training Stability & Auto-Tuning: LR range finder, grad-accum monitor, auto mixed-precision, auto warmup, spike recovery, convergence detector, autopilot wiring (v0.32.0) |
| test_part_f_hardening.py | Live Wire Part F: RLVR OS-level sandbox isolation + prune_checkpoints TOCTOU (v0.33.0) |
| test_part_a_wave1.py | Live Wire Part A: live eval-gate scoring + registry attach (v0.33.0) |
| test_part_a_wave2.py | Live Wire Part A: soup can run / publish + DeployTarget schema (v0.33.0) |
| test_part_e.py | Live Wire Part E: --find-lr live loop + spike recovery hint + auto mixed-precision push + grad-accum advisory (v0.33.0) |
| test_part_d.py | Live Wire Part D: structured-output LogitsProcessor + auto-quant live picker + HF push integration smoke (v0.33.0) |
| test_part_c.py | Live Wire Part C: multi-trainer v0.28.0 features + selective ckpt hooks + CrossDocCollator (v0.33.0) |
| test_part_b.py | Live Wire Part B: auto-reexec under accelerate launch + DeepSpeed-MII live serve (v0.33.0) |
| test_log_level.py | Smart logging tiers `--log-level quiet/normal/verbose/debug` (v0.34.0 Part A) |
| test_run_cost.py | Per-run cost: GPU-rate lookup + estimate + format + tracker integration (v0.34.0 Part B) |
| test_why.py | `soup why` heuristic explainer: NaN / plateau / divergence / grad-norm / LR bounds (v0.34.0 Part C) |
| test_crash_reporter.py | `.crash` bundle: classify, redact secrets, write under cwd, oversize truncation (v0.34.0 Part D) |
| test_replay.py | `soup runs replay`: summarise + downsample + CLI rendering (v0.34.0 Part E) |
| test_profiling.py | Auto-profiling: ProfilerSchedule + path containment + torch-less degradation (v0.34.0 Part F) |
| test_tui.py | `soup tui`: CLI bounds + missing-textual error + row builders (v0.34.0 Part G) |
| test_trainer_coverage_v035.py | Multi-trainer v0.28.0 wiring smoke matrix: every trainer × every speed/memory feature + auto-quant translators + try_reload_with_fallback + benchmark_kernel_combos + schema-gate lift (v0.35.0 Parts A / B / C / D — #60, #61, #45) |
| test_v0470_part_a.py | Synthetic Data Forge: ForgePlan / ProvenanceRecord / ForgeRow frozen dataclasses + VALID_TASKS allowlist + chunk_document + score_uncertainty + discover_documents (cwd-contained, symlink-rejecting, extension allowlist) + build_forge_plan validators (task / target_rows / teacher / NaN+Inf threshold) + synthesise_forge_rows (judge-exception swallow at DEBUG) + write_forge_dataset + write_provenance (atomic, TOCTOU-safe) + CLI smoke (v0.47.0) |
| test_v0470_part_b.py | Data Quality Moat: BENCHMARKS MappingProxyType + ScoreReport frozen + ngram_set / ngram_overlap_ratio / decontaminate_rows (containment ratio) + detect_pii (ReDoS-hardened regexes + 50 KB pre-cap) + detect_language (6-language stopword heuristic) + score_toxicity (keyword baseline) + score_educational_value + compute_scorecard + load_jsonl_rows / write_jsonl_rows (cwd-contained, symlink-rejecting, atomic) + CLI smoke per subcommand (v0.47.0) |
| test_v0480_part_a.py | Curriculum-Aware Trainer (BETA): DynamicCurriculumPolicy frozen + bounds; compute_bucket_weights softmax + water-fill (floor-strict invariant); validate_distributed_curriculum DDP gate; render_curve + parse_history_jsonl with 100k-row DoS cap; SoupConfig cross-validators (requires-curriculum / mlx-rejected / non-SFT-rejected / floor ≤ 1/buckets); `soup runs curriculum-curve` CLI (TOCTOU symlink reject + 50 MB cap + corrupt-JSONL exit 2) (v0.48.0) |
| test_v0490.py | v0.49.0 Long Context & Architecture: YaRN math kernels (yarn_find_correction_dim / yarn_find_correction_range / yarn_linear_ramp_mask / yarn_get_mscale — bool/NaN/Inf rejection, auto-disambiguated low==high); YaRN schema fields (yarn_factor / yarn_attn_factor / yarn_beta_fast / yarn_beta_slow) + cross-validator rejecting yarn fields without rope_scaling_type=yarn; Dynamic NTK harden + verify; LongLoRA schema gate (Llama-family + sft + transformers + !ring_attn); `is_llama_model` word-boundary regex (mirrors v0.39.0 `is_gemma4_model` / v0.44.0 `is_llama4_model` policy); Llama 3.1 NTK-aware (`scale_inv_freq_llama3` smooth-transition / bool-rejected on every param / zero-`old_context_len` rejected; `detect_llama3_rope_in_config` with `rope_type` alias + explicit `is None` check); public-boundary input validation on `get_rope_scaling_config` (target_length / original_length / yarn_factor — security review fix) (v0.49.0) |
| test_v0480_part_b.py | Data Mixing Optimizer (BETA): parse_budget (digits + s/m/h suffix [60s, 24h]) + validate_datasets (containment + symlink + dedup + 32-cap) + MixCandidate (simplex + finite + bool-rejected) + BudgetTracker (injectable clock) + run_mix_optimizer (isolated proxy failures + KeyboardInterrupt propagation + NaN skip + partial budget trip) + render_mix_recipe_yaml (YAML injection defence) + write_mix_recipe / load_mix_recipe (atomic + TOCTOU + 256 KB cap) + `soup data mix --optimize / --apply` CLI (v0.48.0) |

| test_v0500_part_a.py | v0.50.0 Part A — GRPO objective variants (gspo / dapo / dr_grpo / bnpo / two_sided / rft / standard); closed allowlist + frozen `GRPOVariantSpec` + `MappingProxyType` immutability; `validate_grpo_variant` (bool / null-byte / oversize / unknown rejected); `validate_grpo_delta` (bool / NaN / Inf / out-of-range rejected); `apply_variant_loss` deferred-live stub; TrainingConfig integration (Literal accept + two_sided requires delta cross-validator + NaN/Inf field_validator) (v0.50.0 Part A) |
| test_v0500_part_b.py | v0.50.0 Part B — Long-context GRPO + vLLM sleep; `validate_long_context_grpo_compat` (task / backend / ring-attention exclusivity + null-byte + bool guards); `validate_vllm_sleep_mode_compat` (transformers / unsloth allowlist + null-byte); `apply_vllm_sleep_mode` deferred stub with v0.50.1 marker; SoupConfig integration (vllm_sleep_mode requires task='grpo' — code-review HIGH fix) (v0.50.0 Part B) |
| test_v0500_part_c.py | v0.50.0 Part C — Multi-turn agent rollout backend allowlist (art / ruler / nemo_gym / openenv); frozen `RolloutBackendSpec` + `MappingProxyType` immutability; `validate_rollout_backend` (bool rejected); `required_rollout_package` per-entry mapping; `launch_rollout` deferred stub; SoupConfig task-gate + mlx rejection (v0.50.0 Part C) |
| test_v0500_part_d.py | v0.50.0 Part D — 7 stability/efficiency knobs (`ref_model_ema_alpha` / `replay_buffer_size` / `async_grpo_prefetch` / `tis_threshold` / `mask_truncated_completions` / `defer_rerolling` / `skip_zero_advantage` / `off_policy_mask_threshold`); explicit bool-rejection field_validator across all numeric fields (tdd-guide HIGH fix); `mask_truncated_completions` requires `tis_threshold` cross-validator; SoupConfig task-gate naming every offending field; `grpo_fp16` task-gate (code-review HIGH fix) (v0.50.0 Part D) |
| test_v0500_part_e.py | v0.50.0 Part E — `task='prm'` (Process Reward Model) + `vision_grpo` flag; `validate_prm_compat` (data.format / modality / mlx gates); `validate_vision_grpo_compat` (task ∈ {grpo, ppo} / modality='vision' / non-mlx); `build_prm_trainer` deferred stub; SoupConfig integration with all rejection paths exercised (v0.50.0 Part E) |
| test_v0520.py | v0.52.0 Modality II — TTS / classifier / distill / BitNet / EBFT-GDPO / MoE quant / reasoning_effort: TTS family allowlist + per-family emotion allowlists (Orpheus + Oute) + validate_tts_compat; classifier / reranker / cross_encoder tasks + num_labels (with field_validator bool guard) + label_names dedup + classifier-only field gates; distill divergence (kl alias canonicalised, Literal excludes alias) + teacher_model + distill_temperature bounds; BitNet 1.58 quant + bitnet/tq1_0 export-format stubs + Falcon-E recipe + is_bitnet_model org-prefix detect; EBFT (structured/strided) + GDPO (standard/length_normalized/margin) variant allowlists + task gates; MoE expert quant (nf4/int8_rowwise) + train_router_only requiring moe_lora=true; reasoning_effort + train_on_eot with SFT-family task gate; 6 new recipes (5 TTS + Falcon-E BitNet); review-fix coverage (num_labels bool guard, Oute emotion allowlist, lazy-import in classifier validator, task gates, oversize / NaN / Inf matrices). Test count: 272 (v0.52.0) |
| test_v0531_82.py | v0.53.1 #82 autopilot pre-quantized detection: `detect_prequantized_format` + `decide_quantization(prequantized=...)` + `detect_prequantized_format_from_path` with cwd-containment + config.json symlink rejection + name/config aliases + word-boundary regex (v0.53.1) |
| test_v0531_142.py | v0.53.1 #142 merge_4bit + export_torchao live wiring: BNB-4bit single-stage merge + TorchAO PTQ with per-scheme kwarg allowlist + CLI `--save-format` + `--quant-config` + `load_quant_config` (yaml.safe_load + 256 KB cap + extension allowlist) + path TOCTOU (v0.53.1) |
| test_v0531_139.py | v0.53.1 #139 export_advanced_gguf live: 3-stage llama.cpp pipeline (convert → imatrix → quantize) + UD-prefix strip + subprocess argv shape + `_prepare_calibration_text` JSONL alias fallback + null-byte strip + 50 MB cap + POSIX O_NOFOLLOW + `_safe_stderr` Rich escape (v0.53.1) |
| test_v0531_109.py | v0.53.1 #109 deploy autopilot --measure: `compute_cache_key` + `sha_of_file` + `measure_candidate` OK/MINOR/MAJOR bands + `pick_best` soft-fallback (max-by-delta) + cache round-trip with symlink rejection on load AND save + CLI integration + `_MAX_CANDIDATES=32` cap + `render_measure_table` markup escape regression (v0.53.1) |
| test_v0530.py | v0.53.0 Quant Menu II — UD GGUFs + KV cache + NVFP4 + LF parity + save formats: Parts A+B GGUF (UD ladder 14 entries + IQ 12 + Apple/ARM 10 frozensets + non-overlap invariant + `validate_*` case-insensitive + rejection matrix + `is_advanced_gguf_format` union + `_LOWER_INDEX` O(1) lookup + MappingProxyType immutability + `validate_calibration_data_path` shape rejection + 4096-boundary + `export_advanced_gguf` v0.53.1 deferred stub); Part C KV cache (`KV_CACHE_TYPES` frozenset + `validate_kv_cache_type` case + bool/null/oversize/non-string rejection + `requires_hopper` delegates to spec + `get_kv_cache_spec` frozen + schema fp8-on-mlx rejected with specific message + q8_0-on-mlx allowed); Part D advanced precision (`fp8_attention` requires `quantization_aware='fp8'` BEFORE mlx-gate ordering + bool guards on every string param + schema rejects-without-fp8-qat; `nvfp4` mlx + vision rejection + bool guards; `unsloth_bnb_4bit` backend='unsloth' + quantization='4bit' rejection matrix; `apply_*` deferred); Part E LF parity (`bnb_4bit_use_double_quant` rejects none/8bit/gptq parametrize; `llm_int8` rejects default-none + 4bit; `quantize_ref_model` happy on dpo/grpo/kto + rejects sft/pretrain; `quantize_reward_model` happy on ppo/reward_model + rejects dpo; explicit `TypeError("v0.53.0 flag must be bool")` from `_validate_v053_bool_fields`; explicit-null surfaces as `valid boolean` ValidationError); Part F save formats (`MERGE_SAVE_FORMATS` lowercase normalisation + rejection matrix; `TORCHAO_PTQ_SCHEMES` CASE-SENSITIVE — `int4weightonly` rejected; `validate_quant_config_path` 4096-boundary; `MergeSaveSpec` + `TorchAOPTQSpec` frozen + MappingProxyType immutability; `merge_4bit` + `export_torchao` deferred); Cross-cutting (full 5-field YAML round-trip + cardinality invariant + tautological-assert replaced with allowlist + idempotent re-validate + `get_gguf_spec` unknown raises + bool guards on backend/modality/quantization across every Part D validator). Test count: 154 (v0.53.0) |
| test_v0510.py | v0.51.0 Model Catalog Expansion + Alternative Model Hubs: Part E hubs.py (`SUPPORTED_HUBS` + `validate_hub_name` + `validate_hub_endpoint` SSRF parity / CRLF rejection / IPv6 mapped private rejected / IPv6 loopback ok / control chars; `resolve_endpoint` env-var override; `default_endpoint` + `endpoint_env_var` + `required_hub_package` + `is_hf` with bool guards; MappingProxyType immutability); TrainingConfig `hub` field (default + Literal accept + None reject + case-insensitive normalisation + YAML round-trip) + SoupConfig `_validate_hub_supported` (mlx + non-hf rejected; mlx + hf accepted; modelers + transformers accepted); Part D MULTIPACK_ARCHITECTURES extension (20 new arches parametrize + legacy preserved + exact count=38 + frozenset immutability); Parts A/B/C 26 new recipes (parametrize over every name × {get_recipe / RecipeMeta / SoupConfig load / yaml.safe_load / model id no null/whitespace/empty parts / max_length bounds / GRPO required fields}); baichuan-sft uses `hub: modelscope`; total recipe count >= 105 (v0.51.0) |

(Note: the test-file table above covers v0.25.0–v0.35.0 + v0.47.0 + v0.48.0 + v0.49.0 + v0.50.0 only; full per-release table lives in `.claude/CLAUDE.md`.)

## Making Changes

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Your Changes

- Write code following the style guidelines above
- **Write tests first** (TDD) — then implement to pass them
- Keep commits focused and logical

### 3. Run Tests & Lint

Before pushing, ensure everything passes:

```bash
# Lint first
ruff check --fix soup_cli/ tests/

# Then run tests
pytest tests/ -v --tb=short
```

### 4. Commit

Write clear, descriptive commit messages following [Conventional Commits](https://www.conventionalcommits.org/):

```bash
git add <specific-files>
git commit -m "feat: add support for X"
# or
git commit -m "fix: resolve Y when Z"
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`

### 5. Push & Open a PR

```bash
git push origin feature/your-feature-name
```

Then open a pull request on GitHub with:
- Clear title describing the change
- Description of what and why
- Reference any related issues (e.g., "Closes #123")
- Test results

## Pull Request Checklist

When you open a PR, the GitHub template will show this checklist:

- [ ] `ruff check soup_cli/ tests/` passes
- [ ] `pytest tests/ -v` passes
- [ ] Updated relevant docs (README, CLAUDE.md) if needed
- [ ] New tests added for new functionality
- [ ] No breaking changes (or documented in PR description)

## Architecture & Design Decisions

### Lazy Imports for Speed

Heavy ML imports (torch, transformers, trl) are imported inside command handlers so the CLI stays fast. Users can run `soup version` or `soup --help` instantly without waiting for PyTorch to load.

### Pydantic for Config Validation

All YAML configs are validated using Pydantic v2 models. These models are the single source of truth for valid fields and defaults. See `config/schema.py`.

### Trainers as Wrappers

`trainer/sft.py`, `trainer/dpo.py`, `trainer/grpo.py`, `trainer/ppo.py` wrap HuggingFace TRL trainers with:
- Auto quantization (BitsAndBytes, torchao QAT)
- Auto LoRA setup (PEFT)
- Auto batch size estimation
- Progress bar integration

### Experiment Tracking is SQLite

No external dependencies required. All runs, metrics, and eval results go to `~/.soup/experiments.db`.

### Data Format Normalization

Multiple formats (Alpaca, ShareGPT, ChatML, LLaVA, ShareGPT4V) are normalized to a unified `{"messages": [...]}` structure in `data/formats.py`.

## Adding a New Feature

### 1. New Training Task Type

If adding a new training algorithm:

1. Create `trainer/your_trainer.py` wrapping the appropriate TRL trainer
2. Add config fields to `config/schema.py` (Pydantic v2)
3. Add template to `config/schema.py` (see existing 15 templates)
4. Update `commands/train.py` to route to your trainer
5. Add 30+ tests in `tests/test_your_trainer.py`
6. Update `CLAUDE.md`, `README.md`, and `CONTRIBUTING.md`

### 2. New Data Format

1. Add detection and conversion logic to `data/formats.py`
2. Add tests in `tests/test_formats.py`
3. Update `data/loader.py` if needed
4. Document in `CLAUDE.md`

### 3. New Command

1. Create `commands/your_command.py` with a handler function
2. Register in `soup_cli/cli.py` with `@app.command()`
3. Add tests in `tests/test_your_command.py`
4. Update help text and README

### 4. New Recipe

1. Add a `RecipeMeta` entry in `recipes/catalog.py`
2. Add tests in `tests/test_recipes.py`
3. Update `README.md` recipes section

## Good First Issues

Look for issues labeled [`good first issue`](https://github.com/MakazhanAlpamys/Soup/labels/good%20first%20issue) on GitHub. These are beginner-friendly tasks that help you get familiar with the codebase.

Great areas for first contributions:
- **New recipes** — add a ready-made config for a popular model (see `recipes/catalog.py`)
- **Documentation** — improve docstrings, README examples, or example configs
- **Tests** — increase coverage for existing commands
- **Bug fixes** — check [open issues](https://github.com/MakazhanAlpamys/Soup/issues) labeled `bug`

## CI/CD

GitHub Actions runs on every push and PR:
- **ruff** linting on Python 3.11 (must pass)
- **pytest** on Python 3.9, 3.11, 3.12 across Ubuntu, Windows, macOS (must pass)

See `.github/workflows/ci.yml`.

## Releases

The project follows semantic versioning: `MAJOR.MINOR.PATCH`

### Version Bump Process

1. Update version in `pyproject.toml` and `soup_cli/__init__.py`
2. Run full test suite and linting
3. Update `CLAUDE.md`, `README.md`, `SECURITY.md` (if security-related), `CONTRIBUTING.md` (if workflow changed)
4. Commit with message: `Release v0.X.0`
5. Tag: `git tag v0.X.0 && git push --tags`
6. GitHub Actions auto-publishes to PyPI

See `CLAUDE.md` for the complete release checklist.

## Community

- **Issues:** Report bugs and request features on [GitHub Issues](https://github.com/MakazhanAlpamys/Soup/issues)
- **Discussions:** Ask questions on [GitHub Discussions](https://github.com/MakazhanAlpamys/Soup/discussions)
- **Code of Conduct:** Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- **Security:** Report security issues via [SECURITY.md](SECURITY.md)

## Questions?

- Check the [README](README.md) for quick start and features
- Check [CLAUDE.md](.claude/CLAUDE.md) for detailed architecture
- Open a GitHub Discussion for questions

Thank you for contributing!
