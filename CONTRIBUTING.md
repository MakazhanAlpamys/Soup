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
  recipes/            - Ready-made configs for popular models (43 recipes)
  autopilot/          - Zero-config decision engine (v0.25.0)
  registry/           - Model Registry (hashing, store, diff) (v0.26.0)
  cans/               - Shareable .can artifact format (v0.26.0)
  data/traces/        - Trace-to-Preference harvester (v0.26.0)
  utils/              - GPU, errors, MoE, GaLore, QAT, Unsloth, vLLM, SGLang, Liger, FlashAttn, FSDP, Ring Attention, long-context, quality, curriculum, freeze, dataset-registry, mlx, peft_builder, paths, topology, launcher, mii, pipeline, cut_ce, fp8, gradient_ckpt, kernel_picker, cross_doc_attn, activation_offload, hf, spec_pairing, structured_output, metrics, tracing, auto_quant
  ui/                 - Web UI (FastAPI + HTML/JS SPA)

tests/                - Test suite (95 files, 2886 tests)
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
