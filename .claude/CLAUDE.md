# Soup CLI — Project CLAUDE.md

Soup is a CLI-first LLM fine-tuning tool (v0.24.3). Python 3.9+, MIT license.

## Build & Development

```bash
pip install -e ".[dev]"          # Install editable + test deps
pytest tests/ -v --tb=short      # Run all tests (2130 tests)
ruff check soup_cli/ tests/      # Lint (must pass before commit)
ruff check --fix soup_cli/ tests/  # Auto-fix lint issues
```

## Project Structure

```
soup_cli/
  cli.py              # Entry point, Typer app, all command registration
  __init__.py          # __version__ = "0.20.1"
  config/
    schema.py          # Pydantic models (SoupConfig, DataConfig, TrainingConfig, LoraConfig, EvalConfig)
    loader.py          # YAML -> SoupConfig, load_config_from_string()
  data/
    loader.py          # Local files (JSONL/JSON/CSV/Parquet) + HF datasets
    formats.py         # Auto-detect + normalize to {"messages": [...]} structure
    validator.py       # Dataset stats, quality checks, extended_stats()
    providers/
      ollama.py        # Ollama provider for synth data gen (localhost-only)
      anthropic.py     # Anthropic Claude provider (env-only API key)
      vllm.py          # vLLM provider for synth data gen (SSRF-protected)
    templates/
      code.py          # Code instruction pairs template
      conversation.py  # Multi-turn conversation template
      qa.py            # QA from context template
      preference.py    # DPO/KTO/ORPO preference data template
      reasoning.py     # Chain-of-thought / GRPO reasoning template
  trainer/
    sft.py             # SFTTrainerWrapper (415 lines, supports vision + unsloth + QAT)
    dpo.py             # DPOTrainerWrapper (253 lines)
    grpo.py            # GRPOTrainerWrapper (349 lines, reasoning/DeepSeek-R1 style)
    kto.py             # KTOTrainerWrapper (255 lines, unpaired preference)
    orpo.py            # ORPOTrainerWrapper (230 lines, odds ratio preference)
    simpo.py           # SimPOTrainerWrapper (233 lines, simple preference via CPOTrainer)
    ipo.py             # IPOTrainerWrapper (233 lines, identity preference via DPOTrainer)
    ppo.py             # PPOTrainerWrapper (682 lines, full RLHF stage 3)
    pretrain.py        # PretrainTrainerWrapper (294 lines, continued pre-training)
    reward_model.py    # RewardModelTrainerWrapper (272 lines, RLHF stage 2)
    embedding.py       # EmbeddingTrainerWrapper (contrastive/triplet/cosine loss)
    rewards.py         # Built-in reward fns (accuracy, format) + custom .py loader
  commands/
    ...
    data.py            # soup data (inspect/validate/convert/merge/dedup/stats/filter/split/search/preview/download/register)
  monitoring/
    callback.py        # HF TrainerCallback -> Rich display + SQLite tracker
    display.py         # Rich Live terminal dashboard (2Hz refresh)
  eval/
    custom.py          # Custom eval task runner (JSONL tasks, scoring functions)
    judge.py           # LLM-as-a-judge evaluator (OpenAI/Ollama/server backends)
    human.py           # Human evaluation (A/B comparison, Elo ratings)
    leaderboard.py     # Leaderboard aggregation, run comparison, export
  experiment/
    tracker.py         # SQLite at ~/.soup/experiments.db (runs, metrics, eval_results)
  migrate/
    common.py          # Shared utilities (path validation, YAML output, to_number)
    llamafactory.py    # LLaMA-Factory YAML → SoupConfig migration
    axolotl.py         # Axolotl YAML → SoupConfig migration
    unsloth.py         # Unsloth .ipynb → SoupConfig migration (AST-only, no exec)
  recipes/
    catalog.py         # 29 ready-made RecipeMeta configs for popular models
  commands/
    train.py           # soup train (routes to SFT/DPO/GRPO/PPO/Reward/KTO/ORPO/SimPO/IPO)
    deploy.py          # soup deploy ollama (deploy GGUF to Ollama)
    init.py            # soup init (interactive wizard + 10 templates)
    chat.py            # soup chat (terminal REPL)
    serve.py           # soup serve (OpenAI-compatible API, transformers/vllm backends)
    export.py          # soup export (GGUF conversion via llama.cpp)
    merge.py           # soup merge (LoRA + base -> full model)
    push.py            # soup push (HuggingFace Hub upload)
    eval.py            # soup eval (benchmark, custom, judge, compare, leaderboard, human, auto)
    data.py            # soup data (inspect/validate/convert/merge/dedup/stats)
    generate.py        # soup data generate (synthetic data via LLM APIs)
    migrate.py         # soup migrate (import configs from competitors)
    profile.py         # soup profile (memory/speed estimator before training)
    adapters.py        # soup adapters (list/info/compare LoRA adapters)
    recipes.py         # soup recipes (list/show/use/search ready-made configs)
    infer.py           # soup infer (batch inference on JSONL prompts)
    runs.py            # soup runs (list/show/compare/delete experiments)
    sweep.py           # soup sweep (grid/random hyperparameter search)
    diff.py            # soup diff (compare two models side-by-side)
    doctor.py          # soup doctor (resources, dependency + GPU checker)
    quickstart.py      # soup quickstart (20-example TinyLlama demo)
    ui.py              # soup ui (launches FastAPI web UI)
  ui/
    app.py             # FastAPI REST API (auth token, CORS, path traversal protection)
    static/            # SPA: Dashboard, New Training, Data Explorer, Model Chat
  utils/
    gpu.py             # detect_device(), estimate_batch_size(), get_gpu_info()
    errors.py          # Friendly error messages with fix suggestions
    deepspeed.py       # ZeRO Stage 2/3 config templates
    qat.py             # Quantization-Aware Training (torchao)
    unsloth.py         # FastLanguageModel backend (2-5x speedup)
    vllm.py            # AsyncLLMEngine backend (2-4x inference throughput)
    galore.py          # GaLore optimizer config + validation
    moe.py             # MoE model detection, ScatterMoE LoRA target modules
    liger.py           # Liger Kernel detection + fused ops (RMSNorm, SwiGLU, etc.)
    flash_attn.py      # FlashAttention v2/v3 auto-detection
    fsdp.py            # FSDP2 config templates (full_shard, shard_grad, offload)
    ring_attention.py   # Ring FlashAttention for sequence parallelism
    long_context.py    # RoPE scaling for 128k+ context fine-tuning
    quality.py         # Perplexity + coherence scoring for data quality filters
    sglang.py          # SGLang runtime backend (high-throughput serving)
    ollama.py          # Ollama integration (detect, deploy, list, remove, Modelfile gen)
    profiler.py        # Training memory/speed estimator (GPU lookup, model arch)
    curriculum.py      # Curriculum learning: sort by difficulty, create buckets
    freeze.py          # Freeze training: freeze bottom N layers
    registry.py        # Dataset registry: name → path + format mapping
    constants.py       # APP_NAME, paths, default chat template
tests/                 # 78 test files, 2130 tests
examples/
  configs/             # 7 production-ready YAML examples
  data/                # Sample datasets
```

## CLI Commands

```
soup init              # Create config (interactive or --template)
soup train             # Main training (--config, --resume, --wandb, --tensorboard, --deepspeed, --fsdp, --yes)
soup infer             # Batch inference (--model, --input, --output)
soup chat              # Terminal chat with model
soup serve             # OpenAI-compatible inference server (--backend transformers|vllm|sglang, --speculative-decoding, --adapters)
soup export            # Convert to GGUF/ONNX/TensorRT/AWQ/GPTQ for deployment
soup export --deploy ollama  # Export GGUF + auto-deploy to Ollama
soup deploy ollama     # Deploy GGUF model to local Ollama instance
soup deploy ollama --list    # List Soup-deployed models in Ollama
soup deploy ollama --remove  # Remove model from Ollama
soup merge             # Merge LoRA adapter with base model
soup push              # Upload to HuggingFace Hub
soup eval benchmark    # Run benchmarks (mmlu, gsm8k, etc.)
soup eval custom       # Run custom eval tasks from JSONL
soup eval judge        # LLM-as-a-judge evaluation
soup eval auto         # Auto-eval from soup.yaml config
soup eval compare      # Compare eval results between runs
soup eval leaderboard  # Local leaderboard across models
soup eval human        # Human A/B evaluation with Elo ratings
soup data inspect      # Dataset stats + sample rows
soup data validate     # Format compliance check
soup data convert      # Transform between alpaca/sharegpt/chatml
soup data merge        # Combine multiple datasets
soup data dedup        # MinHash deduplication
soup data stats        # Extended statistics with histograms
soup data generate     # Synthetic data via LLM APIs (--provider openai|local|server|ollama|anthropic|vllm)
soup migrate           # Import config from LLaMA-Factory/Axolotl/Unsloth (--from, --dry-run)
soup recipes list      # List all ready-made recipes (30 configs)
soup recipes show      # Print recipe YAML to stdout
soup recipes use       # Copy recipe to soup.yaml
soup recipes search    # Search recipes by keyword, task, or model size
soup data filter       # Quality filter (perplexity + coherence scoring)
soup data sample       # Sample subset: random, diverse (TF-IDF + clusters), hard (by length)
soup data split        # Split dataset into train/val/test files (random or stratified)
soup data search       # Search HuggingFace Hub for datasets
soup data preview      # Preview remote HF dataset metadata, splits, features
soup data download     # Download HF dataset to local JSONL (streaming)
soup data register     # Register local dataset by name for use in soup.yaml
soup data unregister   # Remove dataset from local registry
soup data registry     # List all registered datasets
soup profile           # Estimate memory, speed, GPU requirements before training
soup adapters list     # Scan directory for LoRA adapters
soup adapters info     # Show adapter metadata (base model, rank, size)
soup adapters compare  # Side-by-side adapter comparison
soup runs              # List experiment runs
soup runs show <id>    # Detailed run info + metrics
soup runs compare      # Side-by-side loss curves
soup runs delete <id>  # Remove from DB
soup sweep             # Hyperparameter search (grid/random)
soup diff              # Compare two models' outputs
soup doctor            # Check system resources, GPU, + dependencies
soup quickstart        # One-command demo
soup ui                # Web UI (Dashboard, Training, Data Explorer, Chat)
soup version           # Show version (--full for details)
```

## Config System

`config/schema.py` is the single source of truth. Pydantic v2 models:

- **SoupConfig**: base (required), task (sft/dpo/kto/orpo/simpo/ipo/grpo/ppo/reward_model/pretrain/embedding), modality (text/vision/audio), backend (transformers/unsloth), data, training, output, eval
- **EvalConfig**: auto_eval, benchmarks, custom_tasks, judge
- **DataConfig**: train, format (alpaca/sharegpt/chatml/dpo/kto/llava/sharegpt4v/plaintext/embedding/audio/auto), val_split, max_length, image_dir, audio_dir
- **TrainingConfig**: epochs, lr, batch_size (int or "auto"), quantization (4bit/8bit/none), quantization_aware, optimizer, scheduler, dpo_beta, kto_beta, orpo_beta, simpo_gamma, cpo_alpha, ipo_tau, grpo_beta, num_generations, reward_fn, ppo_epochs, ppo_clip_ratio, ppo_kl_penalty, reward_model, loraplus_lr_ratio, use_galore, galore_rank, galore_update_proj_gap, galore_scale, moe_lora, moe_aux_loss_coeff, use_liger, use_flash_attn, use_ring_attention, rope_scaling_type, gradient_checkpointing, embedding_loss, embedding_margin, embedding_pooling, embedding_temperature, neftune_alpha, packing, curriculum, curriculum_metric, curriculum_buckets, loss_watchdog, loss_watchdog_threshold, loss_watchdog_patience, freeze_layers, freeze_ratio
- **LoraConfig**: r, alpha, dropout, target_modules, use_dora, use_rslora

15 built-in templates: chat, code, medical, reasoning, vision, audio, kto, orpo, simpo, ipo, embedding, rlhf, pretrain, moe, longcontext.
29 ready-made recipes via `soup recipes` (Llama 3.1/3.2, Qwen 2.5/3, Mistral, Gemma 3, Phi-4, DeepSeek R1).

## Training Tasks

| Task | Trainer | Data Format | Use Case |
|------|---------|-------------|----------|
| sft | SFTTrainerWrapper | alpaca/sharegpt/chatml/llava | Instruction tuning |
| dpo | DPOTrainerWrapper | prompt+chosen+rejected | Preference alignment |
| grpo | GRPOTrainerWrapper | prompts + reward fns | Reasoning (DeepSeek-R1) |
| kto | KTOTrainerWrapper | prompt+completion+label | Unpaired preference alignment |
| orpo | ORPOTrainerWrapper | prompt+chosen+rejected | Reference-free alignment |
| simpo | SimPOTrainerWrapper | prompt+chosen+rejected | Length-normalized preference |
| ipo | IPOTrainerWrapper | prompt+chosen+rejected | Regularized preference (squared hinge) |
| pretrain | PretrainTrainerWrapper | plaintext (raw text) | Continued pre-training |
| ppo | PPOTrainerWrapper | prompts + reward model/fn | Full RLHF stage 3 |
| embedding | EmbeddingTrainerWrapper | anchor+positive(+negative) | Sentence embeddings |
| reward_model | RewardModelTrainerWrapper | prompt+chosen+rejected | RLHF stage 2 |

## Key Design Patterns

- **Lazy imports**: torch/transformers/peft inside functions, not module level (fast CLI)
- **Wrapper pattern**: All trainers wrap HF TRL with auto quantization + LoRA + batch estimation
- **Unified messages format**: All data normalizes to `{"messages": [...]}` (+ "image" for vision)
- **Format auto-detection**: `detect_format()` checks first row keys
- **Rich output**: All UX via `rich.console.Console`, never bare `print()`
- **Callback bridge**: `SoupTrainerCallback` connects HF Trainer events to Rich display + SQLite
- **Backend abstraction**: `--backend` flag for training (transformers/unsloth) and serving (transformers/vllm/sglang)
- **Error mapping**: Pattern-based friendly errors with fix suggestions in `utils/errors.py`

## Security (v0.10.10+)

- **Web UI auth**: Bearer token (secrets.token_urlsafe(32)) on mutating endpoints, printed at startup
- **CORS**: Restricted to served origin (not wildcard)
- **Path traversal**: `/api/data/inspect` validates paths stay under cwd
- **Config validation**: `/api/train/start` validates YAML before writing, uses fixed temp path
- **No secret leaks**: HTTP error responses return generic messages, details logged server-side
- **SSRF protection**: `--api-base` blocks non-HTTPS for remote URLs
- **Supply chain**: llama.cpp cloned from pinned tag (b5270), not HEAD
- **Deprecated CLI secrets**: `--api-key` and `--token` flags read from env vars, marked deprecated
- **Custom reward warning**: Prominent warning before executing arbitrary .py reward files
- **max_tokens bound**: Capped at 16384 on inference endpoints
- **experiment_name validation**: Path separators and null bytes blocked (v0.12.0)
- **GaLore params**: Type-enforced before string interpolation (v0.12.0)
- **Batch inference**: max_tokens capped at 16384, trust_remote_code warning (v0.13.0)
- **Plaintext loader**: .txt files read with encoding="utf-8", empty lines skipped (v0.14.0)
- **MoE config**: moe_aux_loss_coeff validated ge=0, moe_lora is boolean only (v0.14.0)
- **Data validate**: auto-detects format when `--format` not specified (v0.14.3)
- **Data stats**: UTF-8 stdout redirect on Windows for plotext histograms (v0.14.3)
- **Web UI**: `--show-token` flag + auth token documented in `--help` (v0.14.3)
- **RoPE scaling**: `rope_scaling_type` validated via Literal constraint (v0.15.0)
- **max_length bounds**: ge=64, le=1048576 prevents OOM/corruption from extreme values (v0.15.0)
- **FSDP config**: key allowlist prevents injection of unexpected TrainingArguments (v0.15.0)
- **Liger Kernel**: exception handling narrowed to prevent silent CUDA error swallowing (v0.15.0)
- **ONNX export**: uses optimum main_export without trust_remote_code (v0.16.0)
- **TensorRT export**: subprocess calls use list args (no shell injection) (v0.16.0)
- **Embedding config**: embedding_loss Literal constraint, embedding_margin gt=0 (v0.16.0)
- **Speculative decoding**: draft model SSRF-protected (URL blocked), warning panel before load (v0.16.0)
- **vLLM speculative**: speculative_model URL validation, rejects http:// schemes (v0.16.0)
- **Server provider**: SSRF validation — scheme whitelist (http/https only), localhost-only HTTP (v0.17.0)
- **Audio model**: trust_remote_code warning panel before loading audio models (v0.17.0)
- **Audio paths**: path traversal protection — resolved paths confined to audio_dir (v0.17.0)
- **SGLang**: trust_remote_code warning panel before runtime creation (v0.17.0)
- **Ollama deploy**: GGUF path traversal protection + `.gguf` extension validation (v0.18.0)
- **Ollama deploy**: model name validation — no path separators or null bytes (v0.18.0)
- **Ollama deploy**: subprocess calls use list args (no shell injection) (v0.18.0)
- **Ollama deploy**: Modelfile parameter key allowlist prevents directive injection (v0.18.0)
- **Ollama deploy**: parameter value newline/null sanitization prevents Modelfile injection (v0.18.0)
- **Ollama deploy**: warning panel before `ollama create` (overwrites existing model) (v0.18.0)
- **Custom eval**: JSONL schema validation, capped at 10k tasks (v0.19.0)
- **Custom eval**: regex scoring ReDoS guard — pattern + input length caps (v0.19.0)
- **Judge eval**: SSRF protection on `--api-base` (localhost-only HTTP) (v0.19.0)
- **Judge eval**: API key isolation — OpenAI key not leaked to non-OpenAI providers (v0.19.0)
- **Human eval**: local-only terminal UI, no network access (v0.19.0)
- **Human eval**: prompts file capped at 10k entries (v0.19.0)
- **Leaderboard**: read-only SQLite queries, no user input in SQL (v0.19.0)
- **Ollama provider**: localhost-only validation — remote Ollama instances blocked (v0.20.0)
- **Anthropic provider**: API key from env only (ANTHROPIC_API_KEY), never CLI arg (v0.20.0)
- **vLLM provider**: SSRF protection — scheme whitelist, localhost-only HTTP (v0.20.0)
- **Output path**: path traversal protection — resolve + relative_to(cwd) on output (v0.20.0)
- **Input paths**: seed, dedup, context files confined to cwd via resolve + relative_to (v0.20.0)
- **Rate limiting**: configurable `--requests-per-minute` (default: 60) (v0.20.0)
- **Migrate input**: resolve + relative_to(cwd) — path traversal protection (v0.21.0)
- **Migrate output**: resolve + relative_to(cwd) — path traversal protection (v0.21.0)
- **Unsloth .ipynb**: AST-only parsing (no exec/eval of notebook code) (v0.21.0)
- **Recipes output**: resolve + relative_to(cwd) — path traversal protection (v0.21.0)
- **NEFTune config**: neftune_alpha bounded ge=0.0, le=50.0 (v0.21.0)
- **Multi-adapter serving**: adapter paths resolve + relative_to(cwd) — path traversal protection (v0.22.0)
- **Multi-adapter serving**: adapter name validation — alphanumeric + hyphens only (v0.22.0)
- **Multi-adapter serving**: unknown adapter → 404 (not 500) (v0.22.0)
- **AWQ/GPTQ export**: calibration data path traversal protection — resolve + relative_to(cwd) (v0.23.0)
- **AWQ/GPTQ export**: output path stays under cwd (v0.23.0)
- **Curriculum config**: curriculum_buckets bounded ge=1, le=20 (v0.23.0)
- **AWQ/GPTQ export**: trust_remote_code warning panel before model loading (v0.23.0)
- **Data split**: output files written to input's parent dir (consistent with sample_data) (v0.23.0)
- **HF download**: trust_remote_code=False + warning panel before download (v0.24.0)
- **HF download**: default output path sanitized via Path.name (v0.24.0)
- **HF download**: --samples capped at 1,000,000 (v0.24.0)
- **Dataset registry**: name validation — no path separators or null bytes (v0.24.0)
- **Dataset registry**: register path traversal protection — resolve + relative_to(cwd) (v0.24.0)
- **Dataset registry**: Rich markup escaped in table output (v0.24.0)
- **Dataset registry**: JSON validation on load — catches corruption + type mismatch (v0.24.0)
- **Loss watchdog**: threshold bounded le=100.0, patience bounded le=1000 (v0.24.0)
- **Freeze training**: freeze_layers bounded le=1000 (v0.24.0)
- **AWQ/GPTQ export**: output path traversal validation before import check (v0.24.3)
- **Windows Unicode**: Rich console symbols replaced with ASCII equivalents (v0.24.3)
- **Chat proxy**: SSRF protection — localhost-only HTTP, HTTPS for remote (v0.24.3)
- **Chat proxy**: max_tokens capped at 16384, temperature 0-2, top_p 0-1 (v0.24.3)
- **Chat proxy**: Bearer token auth required on POST endpoint (v0.24.3)
- **Chat proxy**: XSS prevention — HTML-escape before markdown render (v0.24.3)
- **Runs compare**: max 5 runs per comparison (v0.24.3)
- **Config from-form**: validates via load_config_from_string before returning YAML (v0.24.3)
- **SSE endpoints**: read-only GET, no auth required (consistent with other GET endpoints) (v0.24.3)

## Code Conventions

- **Line length**: 100 chars (ruff enforced)
- **Linter**: ruff with E, F, I, N, W rules
- **Config validation**: Always Pydantic v2 (BaseModel + Field)
- **CLI framework**: Typer with `rich_markup_mode="rich"`
- **Output**: Use `rich.console.Console` — never bare `print()`
- **Lazy imports**: Heavy deps imported inside functions, not at module level
- **Variable naming**: No single-letter names (ruff E741)
- **Testing**: Rich Panel objects rendered via `Console(file=StringIO())`, not `str(panel)`

## Dependencies

**Core**: torch, transformers, peft, trl, datasets, bitsandbytes, accelerate, huggingface-hub, plotext, typer, rich, pydantic, pyyaml

**Optional extras**:
- `[serve]`: FastAPI + uvicorn
- `[serve-fast]`: vLLM
- `[eval]`: lm-evaluation-harness
- `[data]`: datasketch (dedup)
- `[wandb]`: W&B logging
- `[deepspeed]`: ZeRO distributed training
- `[fast]`: unsloth (2-5x training speedup)
- `[vision]`: Pillow
- `[qat]`: torchao
- `[liger]`: liger-kernel (fused ops)
- `[ring-attn]`: ring-flash-attn (sequence parallelism)
- `[onnx]`: optimum + onnxruntime (ONNX export)
- `[tensorrt]`: tensorrt_llm (TensorRT-LLM export)
- `[awq]`: autoawq (AWQ quantization export)
- `[gptq]`: auto-gptq (GPTQ quantization export)
- `[audio]`: librosa + soundfile
- `[sglang]`: sglang + FastAPI + uvicorn
- `[ui]`: FastAPI + uvicorn + static SPA
- `[dev]`: pytest, ruff, pytest-cov, httpx

## Git & CI

- Repo: https://github.com/MakazhanAlpamys/Soup
- Branch: `main`
- CI: GitHub Actions — ruff lint (3.11) + pytest (3.9/3.11/3.12) × (ubuntu/windows/macos)
- PyPI: auto-publish on `git tag v*` via Trusted Publisher (OIDC)
- Always run `ruff check soup_cli/ tests/` and `pytest tests/ -v` before committing

## Release Checklist

**Required for every phase. Do not skip steps. Every phase = full cycle from code to PyPI.**

**If the phase has multiple parts (A, B, C…):** implement part by part. Write tests FIRST (TDD), then implement to pass them. Run `ruff check soup_cli/ tests/` + `pytest tests/ -v --tb=short` after each part to catch issues early. Only proceed to the Release Checklist below after ALL parts pass lint + tests.

1. **Code**: implement the feature following project conventions
2. **Tests**: write tests, add the test file to the test table below
3. **Lint**: `ruff check soup_cli/ tests/` — must be clean
4. **Pytest**: `pytest tests/ -v --tb=short` — all tests must pass
5. **Reviews** (run in order, fix all CRITICAL and HIGH before next):
   - `/everything-claude-code:python-review`
   - `/everything-claude-code:code-review`
   - `/security-review`
   - `/everything-claude-code:tdd`
   - `/everything-claude-code:verification-loop`
6. **Version**: bump version in `pyproject.toml` + `soup_cli/__init__.py`
7. **CLAUDE.md**: update architecture, test table, and any new sections
8. **README.md**: add docs for new feature, update Features / All Commands / Data Formats
9. **SECURITY.md**: update if new security fixes or supported versions changed
10. **CONTRIBUTING.md**: update if dev workflow, deps, or project structure changed
11. **examples/README.md**: update if new example configs or datasets added
12. **plan.md**: mark the phase as complete, update version/test counters
13. **Commit**: one commit per phase with a descriptive message
14. **Push**: `git push origin main`
15. **Tag**: `git tag v0.X.Y && git push origin v0.X.Y`
16. **Release**: `gh release create v0.X.Y` with changelog (What's New, Install/Upgrade)

## Tests (78 test files, 2065 tests)

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
| test_awq_gptq_export.py | AWQ/GPTQ export: format support, CLI, quantize mocks, calibration, output path traversal, security |
| test_packing.py | Sample packing: config, YAML, trainer integration, sweep |
| test_data_split.py | Data split: ratio/absolute/stratified splits, seed, edge cases |
| test_curriculum.py | Curriculum learning: config, length sort, buckets, sweep |
| test_dataset_hub.py | HF dataset search, preview, download, format conversion, security |
| test_freeze_training.py | Freeze training: config, layer freezing, GPT-2 naming, sweep |
| test_loss_watchdog.py | Loss watchdog: config, callback behavior, patience, sweep |
| test_dataset_registry.py | Dataset registry: CRUD, CLI, name validation, error handling |
| test_ui_live_monitor.py | Web UI: SSE log streaming, live metrics SSE, progress endpoint |
| test_ui_metrics.py | Web UI: metrics full fields, runs compare, eval results display |
| test_ui_chat.py | Web UI: chat proxy SSE, SSRF protection, param bounds, auth |
| test_ui_config_builder.py | Web UI: config schema, recipes API, form-to-YAML endpoint |
