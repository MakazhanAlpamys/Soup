# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
# Install in dev mode (editable + test deps)
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v --tb=short

# Run a single test file
pytest tests/test_config.py -v

# Run a single test
pytest tests/test_data.py::test_detect_alpaca_format -v

# Lint
ruff check soup_cli/ tests/

# Lint with auto-fix
ruff check --fix soup_cli/ tests/
```

## Architecture

Soup is a CLI-first tool for LLM fine-tuning. The core flow:

```
soup train --config soup.yaml
  → config/loader.py    (YAML → Pydantic SoupConfig)
  → utils/gpu.py        (detect CUDA/MPS/CPU, estimate batch size)
  → data/loader.py      (load file or HF dataset → normalize format)
  → trainer/sft.py      (load model → quantize → apply LoRA → train)
  → monitoring/callback.py + display.py  (live Rich dashboard)
  → experiment/tracker.py  (log run + metrics to SQLite)
  → save LoRA adapter to output/
```

**Config system:** `config/schema.py` is the single source of truth. All YAML fields are validated by Pydantic models (`SoupConfig` → `TrainingConfig` → `LoraConfig`, `DataConfig`). Templates (chat/code/medical) live as YAML strings in this file.

**Data pipeline:** `data/loader.py` handles local files (JSONL/JSON/CSV/Parquet) and HuggingFace datasets. `data/formats.py` auto-detects and normalizes alpaca/sharegpt/chatml formats into a unified `{"messages": [...]}` structure. Also supports reverse conversion via `messages_to_format()`.

**Trainer:** `trainer/sft.py` (`SFTTrainerWrapper`) and `trainer/dpo.py` (`DPOTrainerWrapper`) wrap HuggingFace's SFTTrainer/DPOTrainer with auto quantization (BitsAndBytes), LoRA (PEFT), and batch size estimation. Heavy ML imports are lazy (inside methods) so CLI stays fast for non-training commands.

**Monitoring:** `monitoring/callback.py` is a HuggingFace `TrainerCallback` that streams metrics to `monitoring/display.py` (Rich Live panel at 2Hz) and optionally to the experiment tracker.

**Experiment tracking:** `experiment/tracker.py` (`ExperimentTracker`) stores runs, per-step metrics, and eval results in SQLite at `~/.soup/experiments.db`. Automatically integrated into `soup train`. Commands: `soup runs`, `soup runs show`, `soup runs compare`, `soup runs delete`.

**Data tools:** `commands/data.py` provides inspect, validate, convert (between alpaca/sharegpt/chatml), merge, dedup (MinHash via datasketch), and stats (extended statistics with plotext histograms).

**Eval:** `commands/eval.py` wraps lm-evaluation-harness for model evaluation on standard benchmarks (mmlu, gsm8k, etc.) with results saved to the experiment tracker.

## Code Conventions

- **Line length:** 100 chars (ruff enforced)
- **Linter:** ruff with E, F, I, N, W rules
- **Config validation:** Always Pydantic v2 (BaseModel + Field)
- **CLI framework:** Typer with `rich_markup_mode="rich"`
- **Output:** Use `rich.console.Console` — never bare `print()`
- **Lazy imports:** Heavy deps (torch, transformers, peft, datasketch, lm_eval, plotext) are imported inside functions, not at module level
- **Variable naming:** Avoid single-letter names (ruff E741) — use `entry`, `part`, `length` instead of `l`
- **Testing:** Rich Panel objects must be rendered via `Console(file=StringIO())` for string assertions, not `str(panel)`

## Git Workflow

- Repo: https://github.com/MakazhanAlpamys/Soup
- Branch: `main`
- CI: GitHub Actions runs ruff lint + pytest on Python 3.9/3.11/3.12
- Always run `ruff check soup_cli/ tests/` before committing
- Always run `pytest tests/ -v` before committing

## Publishing

- **PyPI:** https://pypi.org/project/soup-cli/ — `pip install soup-cli`
- **Auto-publish:** `.github/workflows/publish.yml` triggers on `git tag v*`
- **How to release:** bump version in `pyproject.toml`, then `git tag v0.X.0 && git push --tags`
- **Auth:** PyPI Trusted Publisher (OIDC) — no tokens needed in GitHub secrets

## Tests

Test suite (~147 tests) lives in `tests/`:

| File | Covers |
|---|---|
| `test_config.py` | Config loading, validation, defaults |
| `test_data.py` | Format detection, conversion, validation |
| `test_gpu.py` | GPU detection, batch size estimation |
| `test_cli.py` | CLI commands basic validation |
| `test_tracker.py` | SQLite experiment tracker |
| `test_runs.py` | `soup runs` CLI commands |
| `test_data_tools.py` | Data convert/merge/dedup/stats commands |
| `test_eval.py` | Eval command |
| `test_smoke_train.py` | Full pipeline smoke tests (GPU) |
| `test_chat.py` | Chat command, `_detect_base_model` |
| `test_push.py` | Push command, `_format_size`, `_generate_model_card` |
| `test_init.py` | Init command, templates, overwrite logic |
| `test_callback.py` | `SoupTrainerCallback` (mock-based) |
| `test_display.py` | `TrainingDisplay` rendering |
| `test_loader.py` | Data loading (JSONL/JSON/CSV, edge cases) |
| `test_validator.py` | `validate_and_stats`, `extended_stats`, `_percentile` |
| `test_formats.py` | Reverse conversion, round-trips, edge cases |
