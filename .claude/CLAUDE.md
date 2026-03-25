# Soup CLI — Project CLAUDE.md

Soup is a CLI-first LLM fine-tuning tool (v0.13.0). Python 3.9+, MIT license.

## Build & Development

```bash
pip install -e ".[dev]"          # Install editable + test deps
pytest tests/ -v --tb=short      # Run all tests (906 tests)
ruff check soup_cli/ tests/      # Lint (must pass before commit)
ruff check --fix soup_cli/ tests/  # Auto-fix lint issues
```

## Project Structure

```
soup_cli/
  cli.py              # Entry point, Typer app, all command registration
  __init__.py          # __version__ = "0.12.0"
  config/
    schema.py          # Pydantic models (SoupConfig, DataConfig, TrainingConfig, LoraConfig)
    loader.py          # YAML -> SoupConfig, load_config_from_string()
  data/
    loader.py          # Local files (JSONL/JSON/CSV/Parquet) + HF datasets
    formats.py         # Auto-detect + normalize to {"messages": [...]} structure
    validator.py       # Dataset stats, quality checks, extended_stats()
  trainer/
    sft.py             # SFTTrainerWrapper (415 lines, supports vision + unsloth + QAT)
    dpo.py             # DPOTrainerWrapper (253 lines)
    grpo.py            # GRPOTrainerWrapper (349 lines, reasoning/DeepSeek-R1 style)
    kto.py             # KTOTrainerWrapper (255 lines, unpaired preference)
    orpo.py            # ORPOTrainerWrapper (230 lines, odds ratio preference)
    simpo.py           # SimPOTrainerWrapper (233 lines, simple preference via CPOTrainer)
    ipo.py             # IPOTrainerWrapper (233 lines, identity preference via DPOTrainer)
    ppo.py             # PPOTrainerWrapper (682 lines, full RLHF stage 3)
    reward_model.py    # RewardModelTrainerWrapper (272 lines, RLHF stage 2)
    rewards.py         # Built-in reward fns (accuracy, format) + custom .py loader
  monitoring/
    callback.py        # HF TrainerCallback -> Rich display + SQLite tracker
    display.py         # Rich Live terminal dashboard (2Hz refresh)
  experiment/
    tracker.py         # SQLite at ~/.soup/experiments.db (runs, metrics, eval_results)
  commands/
    train.py           # soup train (routes to SFT/DPO/GRPO/PPO/Reward/KTO/ORPO/SimPO/IPO)
    init.py            # soup init (interactive wizard + 10 templates)
    chat.py            # soup chat (terminal REPL)
    serve.py           # soup serve (OpenAI-compatible API, transformers/vllm backends)
    export.py          # soup export (GGUF conversion via llama.cpp)
    merge.py           # soup merge (LoRA + base -> full model)
    push.py            # soup push (HuggingFace Hub upload)
    eval.py            # soup eval (lm-evaluation-harness wrapper)
    data.py            # soup data (inspect/validate/convert/merge/dedup/stats)
    generate.py        # soup data generate (synthetic data via LLM APIs)
    infer.py           # soup infer (batch inference on JSONL prompts)
    runs.py            # soup runs (list/show/compare/delete experiments)
    sweep.py           # soup sweep (grid/random hyperparameter search)
    diff.py            # soup diff (compare two models side-by-side)
    doctor.py          # soup doctor (dependency + GPU checker)
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
    constants.py       # APP_NAME, paths, default chat template
tests/                 # 44 test files, 906 tests
examples/
  configs/             # 7 production-ready YAML examples
  data/                # Sample datasets
```

## CLI Commands

```
soup init              # Create config (interactive or --template)
soup train             # Main training (--config, --resume, --wandb, --tensorboard, --deepspeed, --yes)
soup infer             # Batch inference (--model, --input, --output)
soup chat              # Terminal chat with model
soup serve             # OpenAI-compatible inference server (--backend transformers|vllm)
soup export            # Convert to GGUF for Ollama/llama.cpp
soup merge             # Merge LoRA adapter with base model
soup push              # Upload to HuggingFace Hub
soup eval              # Run benchmarks (mmlu, gsm8k, etc.)
soup data inspect      # Dataset stats + sample rows
soup data validate     # Format compliance check
soup data convert      # Transform between alpaca/sharegpt/chatml
soup data merge        # Combine multiple datasets
soup data dedup        # MinHash deduplication
soup data stats        # Extended statistics with histograms
soup data generate     # Synthetic data via LLM APIs
soup runs              # List experiment runs
soup runs show <id>    # Detailed run info + metrics
soup runs compare      # Side-by-side loss curves
soup runs delete <id>  # Remove from DB
soup sweep             # Hyperparameter search (grid/random)
soup diff              # Compare two models' outputs
soup doctor            # Check system + dependencies
soup quickstart        # One-command demo
soup ui                # Web UI (Dashboard, Training, Data Explorer, Chat)
soup version           # Show version (--full for details)
```

## Config System

`config/schema.py` is the single source of truth. Pydantic v2 models:

- **SoupConfig**: base (required), task (sft/dpo/kto/orpo/simpo/ipo/grpo/ppo/reward_model), modality (text/vision), backend (transformers/unsloth), data, training, output
- **DataConfig**: train, format (alpaca/sharegpt/chatml/dpo/kto/llava/sharegpt4v/auto), val_split, max_length, image_dir
- **TrainingConfig**: epochs, lr, batch_size (int or "auto"), quantization (4bit/8bit/none), quantization_aware, optimizer, scheduler, dpo_beta, kto_beta, orpo_beta, simpo_gamma, cpo_alpha, ipo_tau, grpo_beta, num_generations, reward_fn, ppo_epochs, ppo_clip_ratio, ppo_kl_penalty, reward_model, loraplus_lr_ratio, use_galore, galore_rank, galore_update_proj_gap, galore_scale
- **LoraConfig**: r, alpha, dropout, target_modules, use_dora

10 built-in templates: chat, code, medical, reasoning, vision, kto, orpo, simpo, ipo, rlhf.

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
| ppo | PPOTrainerWrapper | prompts + reward model/fn | Full RLHF stage 3 |
| reward_model | RewardModelTrainerWrapper | prompt+chosen+rejected | RLHF stage 2 |

## Key Design Patterns

- **Lazy imports**: torch/transformers/peft inside functions, not module level (fast CLI)
- **Wrapper pattern**: All trainers wrap HF TRL with auto quantization + LoRA + batch estimation
- **Unified messages format**: All data normalizes to `{"messages": [...]}` (+ "image" for vision)
- **Format auto-detection**: `detect_format()` checks first row keys
- **Rich output**: All UX via `rich.console.Console`, never bare `print()`
- **Callback bridge**: `SoupTrainerCallback` connects HF Trainer events to Rich display + SQLite
- **Backend abstraction**: `--backend` flag for training (transformers/unsloth) and serving (transformers/vllm)
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
- **experiment_name validation**: Path separators and null bytes blocked (v0.13.0)
- **GaLore params**: Type-enforced before string interpolation (v0.13.0)

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
- `[ui]`: FastAPI + uvicorn + static SPA
- `[dev]`: pytest, ruff, pytest-cov, httpx

## Git & CI

- Repo: https://github.com/MakazhanAlpamys/Soup
- Branch: `main`
- CI: GitHub Actions — ruff lint (3.11) + pytest (3.9/3.11/3.12)
- PyPI: auto-publish on `git tag v*` via Trusted Publisher (OIDC)
- Always run `ruff check soup_cli/ tests/` and `pytest tests/ -v` before committing

## Release Checklist

**Required for every phase. Do not skip steps. Every phase = full cycle from code to PyPI.**

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
9. **plan.md**: mark the phase as complete, update version/test counters
10. **Commit**: one commit per phase with a descriptive message
11. **Push**: `git push origin main`
12. **Tag**: `git tag v0.X.Y && git push origin v0.X.Y`
13. **Release**: `gh release create v0.X.Y` with changelog (What's New, Install/Upgrade)

## Tests (44 test files, 906 tests)

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
| test_doctor.py | `soup doctor` command, version checking, dependency table |
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
| test_bugfixes.py | v0.10.1-v0.10.8 regression fixes |
