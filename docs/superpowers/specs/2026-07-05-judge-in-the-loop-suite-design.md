# v0.71.31 — Judge-in-the-loop suite (design)

**Date:** 2026-07-05
**Version target:** 0.71.30 → 0.71.31
**Closes:** #284 (ship pairwise). Scope items (no dedicated issue): `task='online_dpo'`, `soup data best-of-n`, `soup data evolve`.
**Honest state:** Live (all four validatable on Windows + RTX 3050 4 GB / CPU).

## Pitch

Put an LLM judge *in the loop* across the whole workflow: train against a judge
(`task='online_dpo'`), mine winners from a base model with a judge
(`soup data best-of-n`), grow instruction diversity (`soup data evolve`), and
decide SHIP with a true pairwise judge win-rate (`soup ship --task-mode
pairwise`). No competitor (Unsloth / Axolotl / LLaMA-Factory / OpenPipe) ships
this as an integrated CLI suite.

## Decisions locked (do NOT re-litigate)

- **Scope:** all four features ship in v0.71.31 (one slot). One commit per
  feature for bisect clarity; one tag.
- **best-of-N sampler:** local `transformers` sampling (`--base` = model id/path;
  `do_sample` + `num_return_sequences=N`). Fully offline on the dev box; the
  best-of-N smoke needs no provider. Judge picks the winner **pointwise**
  (score each candidate via `JudgeEvaluator`, argmax), NOT a pairwise
  tournament — one judge call per candidate, simplest reuse.
- **online-DPO judge:** a Soup `SoupPairwiseJudge(BasePairwiseJudge)` adapter
  over the project's existing httpx `JudgeEvaluator` (ollama / server / openai),
  so it works with local ollama on the dev box — NOT TRL's `OpenAIPairwiseJudge`
  (which needs the `openai` package + OpenAI API). `reward_model=` is the
  alternative leg.
- **online-DPO smoke:** a **synthetic length-preferring judge** injected via a
  module-level test seam (`_ONLINE_DPO_JUDGE_OVERRIDE`) — fully offline,
  proof-of-mechanism only (mirrors the v0.71.26 synthetic-reward-hacking
  constraint). Scale ask extends #286.

## Verified in-repo (2026-07-05, installed stack)

- **TRL 0.19.1** `OnlineDPOTrainer.__init__(model, ref_model, reward_model,
  judge, args, ..., peft_config, ...)`. `OnlineDPOConfig` defaults:
  `max_new_tokens=64`, `max_length=512`, `temperature=0.9`,
  `loss_type='sigmoid'`, `missing_eos_penalty=None`, `use_vllm=False`; **`beta`
  is required** (no default) → the wrapper must pass it (reuse `dpo_beta`).
- **`BasePairwiseJudge.judge(prompts: list[str], completions: list[list[str]],
  shuffle_order: bool = True) -> list[int]`** — returns, per prompt, the **index
  (0 or 1) of the best** completion; **`-1`** signals an inner failure. Both
  `PairRMJudge` and `OpenAIPairwiseJudge` subclass it.
- **`eval/judge.JudgeEvaluator`**: `evaluate(prompt, response, category) ->
  JudgeScore(.weighted_score)`, `evaluate_batch(items) ->
  JudgeResults(.overall_score)`; provider allowlist `{openai, server, ollama}`;
  SSRF via `validate_judge_api_base`; OpenAI-compatible `_call_llm`.
- **`eval/gate._parse_judge_url(url) -> (provider, model, api_base)`** — the
  shared judge-URL parser used by `soup ship`; reused by all four features.
- **`utils/magpie.make_magpie_generate_fn(provider, model=, base_url=,
  temperature=, max_tokens=) -> generate(prompt)->str`** — SSRF-guarded
  ollama/vllm raw-completion primitive (reused by `evolve`).
- **`utils/ship_verdict.py`**: `TASK_MODES=("metric","judge_score","pairwise")`
  already includes pairwise; `SUPPORTED_TASK_MODES=("metric","judge_score")`
  gates it off. `build_task_win(mode, base, tuned)`, `decide_ship(...)`.
- **`commands/ship.py`**: `_leg1_metric`, `_leg1_judge`, `_verdict_live`,
  `_verdict_from_evidence`, `_validate_task_mode_flag`,
  `_validate_judge_model_url`.
- **`commands/train.py`** task routing `if cfg.task == "dpo": ... elif ...`
  (~L1230); **`config/schema.py`** `task` Literal (~L3496) + `TrainingConfig`.
- **`commands/data.py`** `@app.command(name="gen-magpie")` pattern for new
  `soup data <subcommand>` registration.

## Feature 1 — `task='online_dpo'`

### Files
- **NEW `src/soup_cli/trainer/online_dpo.py`** — `OnlineDPOTrainerWrapper`.
  Mirror `dpo.py`: same `__init__(config, device, report_to, deepspeed_config,
  fsdp_config, trust_remote_code)` + trust resolution; `_setup_transformers`
  loads tokenizer + model + Quant-Menu quant + vocab-expansion + kbit-prepare;
  builds `LoraConfig` but does **NOT** `get_peft_model` (passes `peft_config=`
  to the trainer instead); `train()` mirrors dpo's callbacks / save / metrics.
  Module-level `_ONLINE_DPO_JUDGE_OVERRIDE: Optional[BasePairwiseJudge] = None`
  test seam.
- **`eval/judge.py`** — add `SoupPairwiseJudge(BasePairwiseJudge)` (lazy TRL
  import guarded) + `pairwise_compare` + `pairwise_winrate` (shared with
  Feature 4). `SoupPairwiseJudge.judge` honors `shuffle_order` (per-pair
  order shuffle + un-shuffle), returns `0/1` or `-1`.
- **`config/schema.py`** — task Literal `+"online_dpo"`; `TrainingConfig` +
  `online_dpo_judge: Optional[str]`, `online_dpo_loss_type:
  Literal["sigmoid","ipo"]="sigmoid"`, `online_dpo_max_new_tokens: int` (ge=1,
  le=4096, default 64). Field-validators: `online_dpo_judge` non-empty /
  NUL-free / ≤512 chars when set. `SoupConfig._validate_online_dpo_compat`
  cross-validator (see below).
- **`commands/train.py`** — `elif cfg.task == "online_dpo":
  OnlineDPOTrainerWrapper(...)`; prompt-only dataset path (normalize
  `{"messages":[...]}` → `prompt` column, mirror grpo).
- **`recipes/catalog.py`** — +1 `online-dpo-smollm2-135m` (optional; bumps
  137→138). If dropped, say so and keep 137.

### Cross-validator `_validate_online_dpo_compat`
- `task == "online_dpo"` ⇒ `backend == "transformers"` (reject mlx + unsloth
  with a named error), `modality == "text"`.
- Exactly one of `{online_dpo_judge set, reward_model set}` — reject **both
  unset** ("online_dpo needs a judge or a reward_model") and **both set**.
- Footgun: any `online_dpo_*` field non-default while `task != "online_dpo"` →
  reject (mirrors the `prm_aggregate`-while-`prm_reward`-None guard).

### Judge URL
- Reuse `eval/gate._parse_judge_url` + `validate_judge_api_base`. Supported
  schemes: `ollama://model`, `https://host/...`, `http://localhost...`.

## Feature 2 — `soup data best-of-n`

### Files
- **NEW `src/soup_cli/utils/best_of_n.py`** (torch-lazy; pure judge/build half):
  - `sample_candidates(model, tok, prompt, *, n, temperature, max_new_tokens)
    -> list[str]` — transformers `do_sample=True, num_return_sequences=n`,
    decode only the continuation.
  - `judge_pick_best(prompt, candidates, evaluator) -> BestOfNPick` (frozen:
    `winner_idx, scores, winner`) — pointwise `evaluator.evaluate`, argmax;
    ties → lowest index.
  - `build_sft_row(prompt, pick) -> dict` — `{"messages":[{user},{assistant:
    winner}], "_best_of_n":{n, winner_idx, judge_model, scores}}`.
  - `build_dpo_pair(prompt, pick) -> dict` — `{prompt, chosen=winner,
    rejected=lowest-scored}`; `None` if only one distinct candidate.
- **`commands/data.py`** — `@app.command(name="best-of-n")`:
  `--base --prompts --n(8) --judge <url> -o [--emit-pairs dpo.jsonl]
  [--temperature(1.0) --max-new-tokens(256) --device --seed --plan-only]`.

### Bounds / security
- `n` in `[2, 64]`; prompts cap (`_MAX_PROMPTS`, e.g. 100_000);
  `max_new_tokens` in `[1, 4096]`; `temperature` in `[0, 2]`.
- `--prompts`, `-o`, `--emit-pairs` cwd-contained + symlink-rejected.
- `--judge` via `_parse_judge_url` + `validate_judge_api_base` (SSRF).
- `--base` `trust_remote_code=False` probe + warn (mirror chat/diff/merge).
- Rich-escape prompt/candidate text before any terminal echo.

## Feature 3 — `soup data evolve`

### Files
- **NEW `src/soup_cli/utils/evolve.py`** (pure; httpx-lazy via magpie primitive):
  - `DEPTH_TEMPLATES` (add-constraints / deepen / concretize / increase-steps),
    `BREADTH_TEMPLATE` (new in-domain instruction). WizardLM-style.
  - `evolve_instruction(seed, strategy, generate_fn) -> str` (one op).
  - `run_evolve(seeds, strategy, rounds, generate_fn) -> list[EvolvedRow]`
    (frozen: `instruction, seed, strategy, round`). Each round evolves every
    current instruction. Elimination: drop empty / whitespace-only / unchanged /
    degenerate (contains the meta-prompt back) evolutions.
- **`commands/data.py`** — `@app.command(name="evolve")`:
  `--input seeds.jsonl --provider ollama|vllm --model --strategy depth|breadth
  --rounds N -o [--base-url --temperature(1.0) --max-tokens(512) --plan-only]`.
  Output rows: `{"messages":[{"role":"user","content":evolved}],
  "_evolve":{seed,strategy,round}}`.

### Bounds / security
- `rounds` in `[1, 5]`; seeds cap; `--input`/`-o` cwd-contained + symlink-reject.
- Provider SSRF via magpie's `validate_ollama_url` / `validate_vllm_url`.
- `anthropic` rejected (no raw-completion endpoint — same as magpie).

## Feature 4 — `soup ship --task-mode pairwise` (#284)

### Files
- **`utils/ship_verdict.py`** — add `"pairwise"` to `SUPPORTED_TASK_MODES`;
  extend the module docstring: pairwise `TaskWin(base=0.5 coin-flip,
  tuned=win-rate)`, `won = tuned > 0.5`.
- **`eval/judge.py`** — `pairwise_compare` / `pairwise_winrate` (shared with
  Feature 1). Pairwise rubric prompt ("prompt + response A + response B → answer
  A or B"); parse A/B; swap-debiased (judge A,B and B,A; disagreement/unparseable
  → tie=0.5).
- **`commands/ship.py`** — new `_leg1_pairwise(base_gen, tuned_gen, task_eval,
  judge_model) -> TaskWin`: load tasks, generate base + tuned per prompt,
  `pairwise_winrate` → `build_task_win("pairwise", 0.5, winrate)`. Route
  `task_mode == "pairwise"` in `_verdict_live` (requires `--judge-model`,
  validated via `_validate_judge_model_url`). The `--evidence` offline path
  accepts pairwise automatically once it's in `SUPPORTED_TASK_MODES` (evidence
  `task = {mode:"pairwise", base:0.5, tuned:<winrate>}`). Update `--task-mode`
  help text to `metric | judge_score | pairwise`.

## Testing — `tests/test_v07131.py`

- **online_dpo:** schema defaults + happy parse; every cross-validator reject
  (mlx/unsloth backend, non-text modality, both judge+reward, neither,
  footgun-while-off); `SoupPairwiseJudge.judge` returns correct indices +
  `-1` on failure + `shuffle_order` un-shuffle; `OnlineDPOTrainerWrapper.setup`
  builds a trainer on a CPU tiny model (real or mocked); `train()` result shape;
  synthetic length-judge picks the longer completion.
- **best-of-n:** `sample_candidates` (mocked model → N strings);
  `judge_pick_best` argmax + ties; `build_sft_row`/`build_dpo_pair` shapes +
  provenance; bounds + cwd/SSRF/trust guards; CLI help + happy (fixture judge) +
  reject paths; `--emit-pairs` writes DPO rows.
- **evolve:** depth/breadth template rendering; `run_evolve` rounds + lineage +
  elimination (unchanged dropped); provider reuse (mocked generate_fn); bounds;
  CLI help + `--plan-only` + reject.
- **ship pairwise:** `pairwise_compare` swap-debias (fixture judge);
  `pairwise_winrate` in [0,1]; `_leg1_pairwise` → `TaskWin("pairwise",0.5,wr)`;
  `--evidence` pairwise accepted; `decide_ship` with a pairwise TaskWin;
  `soup ship --task-mode pairwise` no longer rejected; help text updated.
- **no-top-level-torch** AST guard on `utils/best_of_n.py` (judge/build half),
  `utils/evolve.py`.

## Step-6 local smoke (RTX 3050 / CPU)

- **online_dpo:** ~30–50 steps on SmolLM2-135M, judge = synthetic
  length-preferring (via `_ONLINE_DPO_JUDGE_OVERRIDE`), 4-bit or CPU LoRA →
  confirm the run completes + a preference/reward metric is logged and moves;
  reject configs (mlx, both-judge-and-reward, neither).
- **best-of-n:** SmolLM2-135M local sampling `--n 4` on a tiny prompts.jsonl with
  a fixture/pointwise judge → `sft.jsonl` written with `_best_of_n` provenance;
  `--emit-pairs` writes DPO pairs; bad path / bad n rejected.
- **evolve:** depth + breadth 2 rounds against a local ollama small model
  (or a mocked-generate unit path if ollama is unavailable) → evolved
  instructions with lineage; `--plan-only`; anthropic rejected.
- **ship pairwise:** fixture base/tuned responses + a fixture judge →
  win-rate + SHIP/DON'T + exit 0/2; `--evidence` pairwise; bad `--judge-model`
  URL rejected (exit 2).

## Risks

- **OnlineDPO API drift across TRL versions** — the wrapper reads
  `OnlineDPOConfig`/`OnlineDPOTrainer` by name; guard the import with a friendly
  "requires trl>=0.19 with OnlineDPO" error so an older/newer TRL fails loudly,
  not cryptically.
- **Pairwise judge position bias** — mitigated by swap-debiasing; note the
  residual (a judge that always says "A" yields 0.5 after the swap).
- **best-of-N generation cost** — one forward *per candidate per prompt*; caps
  bound it; document that large N × many prompts is slow (fine for tiny models).
- **Proof-of-mechanism only** — online-DPO validated with a synthetic judge on
  SmolLM2-135M; not a production-scale claim (extends #286).

## Release Checklist mapping

Per `.claude/CLAUDE.md`: TDD per feature → `ruff` + `pytest` green → 5 sequential
ECC reviews (python / code / security / tdd / verification-loop), every finding
fixed → Step-6 local smoke (the real runs above) → docs pass (version bump both
files, CHANGELOG, CLAUDE.md arch+CLI+schema+security+`.claude/history/*`
append+inline-roll+≥3 test counts, README What's New, `docs/training.md` +
`docs/data.md` + `docs/evaluation.md` + `docs/commands.md`, CONTRIBUTING counts)
→ one commit per feature → push → CI green (Windows PYTHONUTF8, realpath) → tag
`v0.71.31` → `gh release create` → verify PyPI + GHCR → close #284 → file
follow-ups only for actionable code limitations.
