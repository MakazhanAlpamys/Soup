# PRM-guided GRPO + bundled rollout envs — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:test-driven-development, task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Let a trained Soup PRM score each reasoning step of a GRPO completion as the reward signal, and bundle 3 pure-python toy environments that seed the live `openenv` rollout path out-of-the-box.

**Architecture:** A pure-kernel + torch-lazy `PRMScorer` in `utils/prm_reward.py` becomes the GRPO `reward_fn` when `training.prm_reward` is set (replacing the configured reward, riding the existing shaping+buffer seam). Three `soup_cli/envs/*.py` modules expose `rollout(prompts)` entry points wired via `training.rollout_func`. Schema fields + cross-validators gate both.

**Tech Stack:** Python 3.10+, Pydantic v2, torch/transformers/safetensors (lazy), TRL GRPO, pytest.

## Global Constraints

- Line length 100 (ruff E,F,I,N,W). No bare `print()` — use `rich.console.Console`.
- Lazy imports for torch/transformers/peft/safetensors inside functions, never module-level.
- Path containment: `os.path.realpath` + `os.path.commonpath`, NOT `Path.resolve()+relative_to()`.
- Pydantic v2 models in `config/schema.py`; assert specific error keywords in tests.
- Test file: `tests/test_v07130.py`. Use `assert result.exit_code == 0, (result.output, repr(result.exception))`.
- Version target: `0.71.29 → 0.71.30` (bump in `pyproject.toml` + `src/soup_cli/__init__.py`).
- Proof-of-mechanism honesty: tiny model / synthetic data only; scale ask extends #286.

---

### Task 1: PRM reward pure kernels

**Files:**
- Create: `src/soup_cli/utils/prm_reward.py`
- Test: `tests/test_v07130.py`

**Interfaces:**
- Produces: `split_steps(text: str) -> list[str]`, `aggregate_step_scores(scores: list[float], mode: str) -> float`, module constants `_MAX_STEPS`, `_MAX_STEP_CHARS`, `AGGREGATE_MODES: tuple[str,...]`.

- [ ] Step 1: Write failing tests `TestSplitSteps` (newline split; drop empty/whitespace; count cap `_MAX_STEPS`; per-step char cap; non-str→`[]`) and `TestAggregate` (min/prod/last correctness; empty→0.0; single; non-finite→treated safe; bad mode `ValueError` names modes; bool reject).
- [ ] Step 2: Run — expect ImportError / fail.
- [ ] Step 3: Implement kernels (no top-level torch). `split_steps`: `str.splitlines()`, strip, drop empties, truncate each to `_MAX_STEP_CHARS`, cap list to `_MAX_STEPS`. `aggregate_step_scores`: validate mode in `AGGREGATE_MODES`; empty→0.0; `min`→min, `last`→scores[-1], `prod`→math.prod; coerce non-finite to 0.0.
- [ ] Step 4: Run tests → PASS.
- [ ] Step 5: `ruff check` + commit `feat(prm): PRM reward pure kernels (split_steps/aggregate) (v0.71.30)`.

---

### Task 2: Schema fields + cross-validators

**Files:**
- Modify: `src/soup_cli/config/schema.py` (TrainingConfig fields near `reward_fn` ~L1043; field_validator; SoupConfig cross-validator near `_validate_rollout_backend` ~L4183)
- Test: `tests/test_v07130.py`

**Interfaces:**
- Produces: `TrainingConfig.prm_reward: Optional[str]`, `TrainingConfig.prm_aggregate: Literal["min","prod","last"]`; a `SoupConfig` `model_validator(mode="after")` `_validate_prm_reward`.

- [ ] Step 1: Write failing `TestPrmSchema` — happy config (task=grpo, transformers, text, prm_reward set, prm_aggregate=prod) parses via `load_config_from_string`; rejects: task≠grpo ("requires task='grpo'"), backend=mlx/unsloth ("transformers"), modality≠text ("modality='text'"), `prm_aggregate` non-default while `prm_reward None` (footgun keyword), null-byte prm_reward, oversize prm_reward.
- [ ] Step 2: Run → fail (fields absent).
- [ ] Step 3: Add fields + `field_validator("prm_reward", mode="before")` (null-byte/non-str/len≤512 reject) + SoupConfig `_validate_prm_reward` cross-validator (gates task/backend/modality; footgun on aggregate). Mirror `_validate_rollout_backend`.
- [ ] Step 4: Run → PASS.
- [ ] Step 5: ruff + commit `feat(prm): schema prm_reward/prm_aggregate + cross-validators (v0.71.30)`.

---

### Task 3: PRMScorer + build_prm_reward_fn (torch-lazy load + scoring)

**Files:**
- Modify: `src/soup_cli/utils/prm_reward.py`
- Test: `tests/test_v07130.py`

**Interfaces:**
- Consumes: `split_steps`, `aggregate_step_scores` (Task 1); `TrainingConfig` (Task 2).
- Produces: `PRMScorer(prm_path, aggregate, device="cpu", trust_remote_code=False)` with `__call__(completions, **kwargs) -> list[float]` and `__name__="prm_reward"`; `build_prm_reward_fn(tcfg, device, trust_remote_code) -> PRMScorer`; `load_reward_head_weights(prm_path) -> dict` helper.

- [ ] Step 1: Write failing `TestPRMScorer` — fixture: build a tiny `AutoModelForCausalLM` from a 2-layer config, attach `reward_head=nn.Linear(hidden,1)`, `save_pretrained(tmp)`; `PRMScorer(tmp,"min","cpu")` scores `[[{"role":"assistant","content":"step1\nstep2"}]]` → list len 1 of finite float. "not a PRM" fixture (base with no reward_head) → friendly `ValueError`/`RuntimeError` naming "reward_head". `build_prm_reward_fn` returns callable with `__name__=="prm_reward"`; outside-cwd path → containment reject.
- [ ] Step 2: Run → fail.
- [ ] Step 3: Implement. `load_reward_head_weights`: iterate `*.safetensors` in dir via `safetensors.safe_open`, collect `reward_head.*` tensors; empty→raise. `PRMScorer._ensure_loaded`: lazy tokenizer + `AutoModelForCausalLM.from_pretrained`, `reward_head=nn.Linear(hidden,1)`, `load_state_dict` the head, `eval()`+`requires_grad_(False)`+`.to(device)`. `__call__`: per completion → render prompt (from `kwargs["prompts"]`) + steps, tokenize, `no_grad` `output_hidden_states` forward, gather boundary hidden, `reward_head`→scalars, `aggregate_step_scores`. `build_prm_reward_fn`: realpath+commonpath containment for local paths, trust_remote_code probe+warn.
- [ ] Step 4: Run → PASS.
- [ ] Step 5: ruff + commit `feat(prm): PRMScorer + build_prm_reward_fn (safetensors head load) (v0.71.30)`.

---

### Task 4: GRPO integration

**Files:**
- Modify: `src/soup_cli/trainer/grpo.py` (`setup()`, the `load_reward_fn` block ~L224-229)
- Test: `tests/test_v07130.py`

**Interfaces:**
- Consumes: `build_prm_reward_fn` (Task 3).

- [ ] Step 1: Write failing `TestGrpoPrmWiring` — construct a SoupConfig with `prm_reward` set, monkeypatch `build_prm_reward_fn` to a sentinel, call the reward-selection path (extract a small helper `_select_reward_fn(tcfg, device, trust)` if cleaner, or assert via the setup path with mocked model load), assert the PRM scorer is chosen (not `load_reward_fn`).
- [ ] Step 2: Run → fail.
- [ ] Step 3: In `setup()`: `if tcfg.prm_reward is not None: reward_fn = build_prm_reward_fn(tcfg, self.device, self._trust_remote_code) else: reward_fn = load_reward_fn(...)`. Keep the shaping+buffer wrapping downstream unchanged.
- [ ] Step 4: Run → PASS.
- [ ] Step 5: ruff + commit `feat(prm): wire PRM reward into GRPO setup (v0.71.30)`.

---

### Task 5: Bundled rollout envs

**Files:**
- Create: `src/soup_cli/envs/__init__.py`, `calculator.py`, `retrieval_qa.py`, `guess_number.py`
- Test: `tests/test_v07130.py`

**Interfaces:**
- Produces: `soup_cli.envs.calculator:rollout`, `...retrieval_qa:rollout`, `...guess_number:rollout` — each `rollout(prompts) -> list[{"prompt": str, "answer": str}]`, deterministic (seeded `random.Random`), no I/O.

- [ ] Step 1: Write failing `TestEnvs` — each `rollout([])` returns non-empty rows; every row passes `agent_rollout._normalise_rollout_rows(rows, "openenv")`; determinism (two calls equal); calculator answers verified (parse "a OP b" from prompt, compute, assert == answer).
- [ ] Step 2: Run → fail.
- [ ] Step 3: Implement 3 modules. Each builds a fixed count of {prompt, answer} rows via `random.Random(seed)` (fixed seed → deterministic; no `Math.random`/wall-clock). calculator: arithmetic; retrieval_qa: doc+question, answer=span; guess_number: constraint deduction puzzles with a unique integer answer.
- [ ] Step 4: Run → PASS.
- [ ] Step 5: ruff + commit `feat(envs): bundled calculator/retrieval-qa/guess-number rollout envs (v0.71.30)`.

---

### Task 6: Recipes + no-top-level-torch AST test

**Files:**
- Modify: `src/soup_cli/recipes/catalog.py` (+3 GRPO env recipes)
- Test: `tests/test_v07130.py`

- [ ] Step 1: Write failing `TestRecipes` (3 new recipe names resolve + their YAML parses via `load_config_from_string`) + `TestNoTopLevelTorch` (AST-parse `utils/prm_reward.py`, assert no top-level `import torch/transformers/peft`).
- [ ] Step 2: Run → fail.
- [ ] Step 3: Add `grpo-env-calculator` / `grpo-env-retrieval-qa` / `grpo-env-guess-number` RecipeMeta entries (SmolLM2-135M, task=grpo, rollout_backend=openenv, rollout_func=`soup_cli.envs.*:rollout`, reward_fn=math/accuracy). Confirm the no-top-level-torch already holds.
- [ ] Step 4: Run → PASS.
- [ ] Step 5: ruff + commit `feat(recipes): 3 GRPO env recipes + AST purity test (v0.71.30)`.

---

### Task 7: Release docs (Release Checklist steps 7-13)

Version bump (`pyproject.toml` + `__init__.py` → 0.71.30); CHANGELOG; CLAUDE.md (arch line for `utils/prm_reward.py` + `envs/`, schema field list, CLI/train notes, recipe count 134→137, test-count refs, history-roll to `.claude/history/release-notes.md` + `tests-index.md`); README What's New; `docs/training.md` + `docs/commands.md`; CONTRIBUTING counts; plan.md flip. Handled after Tasks 1-6 pass the 5-review round + Step-6 smoke.

---

## Self-review

- **Spec coverage:** Part A schema (T2) + kernels (T1) + scorer (T3) + wiring (T4); Part B envs (T5) + recipes (T6); docs (T7). ✅
- **Type consistency:** `split_steps`/`aggregate_step_scores` (T1) consumed by `PRMScorer` (T3); `build_prm_reward_fn` (T3) consumed by grpo (T4); `rollout` signature uniform (T5). ✅
- **Placeholders:** none — each task has concrete test targets + implementation sketch (full code written during TDD RED/GREEN).
