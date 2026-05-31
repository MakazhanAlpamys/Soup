# AGENTS.md

Tool-agnostic entry point for AI coding agents (Codex, Cursor, Aider, Claude Code, etc.).

**Soup** is a CLI-first LLM fine-tuning tool. Python 3.10+, Apache-2.0.

## Build & test

```bash
pip install -e ".[dev]"          # Editable install + test deps
pytest tests/ -v --tb=short      # Run the suite (smoke tests are excluded by default)
ruff check src/soup_cli/ tests/  # Lint — must be clean before any commit
```

- `pytest -m smoke` runs the slow tests that download models and train (skipped by default).
- CI matrix: Python 3.10 / 3.11 / 3.12 × Ubuntu / Windows / macOS.

## Conventions (must follow)

- **Config** is Pydantic v2 in `src/soup_cli/config/schema.py` — single source of truth.
- **Heavy deps** (`torch`, `transformers`, `peft`, `trl`, `mlx`) are lazy-imported inside functions, never at module top.
- **Output** via `rich.console.Console`, never bare `print()`.
- **Path containment**: use `os.path.realpath` + `os.path.commonpath`, not `Path.resolve() + relative_to()` (breaks on Windows short names).
- **Line length** 100, ruff rules `E, F, I, N, W`.

## Full instructions

The authoritative, detailed guide lives in [`.claude/CLAUDE.md`](.claude/CLAUDE.md) — architecture map, every CLI command, the config schema, the security model, and the release checklist. Read it before making non-trivial changes.
