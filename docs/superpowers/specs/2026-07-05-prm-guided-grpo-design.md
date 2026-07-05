# v0.71.30 — PRM-guided GRPO + bundled rollout envs — Design

**Status:** approved 2026-07-05
**Slot:** v0.71.30 (first `[ ]` patch in `.claude/plan.md`)
**Type:** net-new mechanism (PRM-as-per-step-reward adapter + bundled envs package)

## Pitch

Use a trained PRM (the v0.53.11 `PRMTrainerWrapper`, live) as a **per-step reward inside
GRPO** — the o1-era process-supervision training signal. Plus bundle 3 pure-python toy
"environments" so the live `openenv` rollout path (#125) runs out-of-the-box.

**Honesty constraint (this is an RL feature — cannot ship schema-only):** validation is
**proof-of-mechanism only** on a tiny model (SmolLM2-135M) + a tiny synthetic PRM. It is NOT
a 7B / real-PRM / production claim. Scale ask extends #286. State this loudly in the PR,
CHANGELOG, release notes, and CLAUDE.md known-limitations. Mirrors the v0.71.26 framing.

## Decisions locked

- **PRM reward role = sole reward (replace).** When `training.prm_reward` is set, the PRM
  scorer becomes THE reward function, replacing `reward_fn`. Simplest wiring — drops into the
  existing single-fn shaping + buffer seam. Matches the "PRM reward curve" smoke.
- **PRM artifact = Soup-trained PRM only (v1).** Load base `AutoModelForCausalLM`, re-attach
  `reward_head=nn.Linear(hidden,1)`, and load its weights from the saved `model.safetensors`
  (`reward_head.*` keys). HF `AutoModelForSequenceClassification(num_labels=1)` interop is a
  deferred follow-up.
- **Envs = deterministic single-shot seeders**, not interactive multi-turn episodes. The live
  `openenv` contract calls `fn(seed_prompts)` — model/tokenizer are NOT passed — so an env is a
  prompt+answer *generator* scored by the existing `accuracy`/`math` reward.
- **Aggregation default = `min`** (weakest-link; Lightman et al. "Let's Verify Step by Step").
- **Step split = newline heuristic (v1).**

## Reuse map (verified in-repo 2026-07-05)

- Reward seam: `trainer/grpo.py::setup()` loads a single `reward_fn` via
  `rewards.load_reward_fn(spec)`, then optionally wraps with
  `reward_hack_control.apply_reward_shaping` + `rl_signal_buffer.wrap_reward_funcs(buffer)`
  (the v0.71.26 mitigation seam). A PRM reward is just a `(completions, **kwargs) -> list[float]`
  callable → rides that seam unchanged.
- PRM artifact: `trainer/prm.py::PRMTrainerWrapper` sets `base_model.reward_head =
  nn.Linear(hidden,1,bias=True)` and `save_model`s the full model → the head weights live inside
  `model.safetensors` under `reward_head.*` (base `from_pretrained` drops them as "unexpected").
- Scoring math mirror: `PRMTrainerWrapper.compute_loss` — forward `output_hidden_states=True`,
  gather `last_hidden` at step-boundary positions, project via `reward_head`.
- Rollout: `utils/agent_rollout.py::launch_rollout` — `openenv` resolves `rollout_func`
  (`module:fn`) and calls `fn(seed_prompts)`; rows go through `_normalise_rollout_rows`
  (caps + anti-smuggle → `{prompt, answer?}`-only).
- Schema patterns: `rollout_func` field_validator + `_validate_rollout_backend` cross-validator;
  `verifiable_domain` `reward_fn` cross-validator (`schema.py`).

## Part A — PRM reward

### Schema (`config/schema.py`, `TrainingConfig`)

- `prm_reward: Optional[str] = None` — path (or HF id) to a Soup-trained PRM.
  `field_validator(mode="before")` rejects null-byte / non-str / oversize (shape only;
  filesystem containment enforced at load, TOCTOU-safe).
- `prm_aggregate: Literal["min","prod","last"] = "min"`.
- `SoupConfig` cross-validator: `prm_reward` set ⇒ `task=='grpo'`, `backend=='transformers'`
  (reward runs a transformers forward — mlx/unsloth rejected), `modality=='text'`;
  `prm_aggregate != "min"` while `prm_reward is None` → footgun reject.

### New module `utils/prm_reward.py` (NO top-level torch)

- `split_steps(text: str) -> list[str]` — newline heuristic v1; drop empty/whitespace; cap
  count (`_MAX_STEPS`) and per-step chars. Pure.
- `aggregate_step_scores(scores: list[float], mode: str) -> float` — min/prod/last;
  empty→0.0; non-finite-safe; bad-mode `ValueError`; bool guard. Pure kernel.
- `PRMScorer` — stateful callable:
  - `__init__(prm_path, aggregate, device, trust_remote_code)` — stores config, no load yet.
  - `_ensure_loaded()` — lazy: tokenizer + base `AutoModelForCausalLM` (dtype per device),
    re-attach `reward_head`, load `reward_head.*` from any `*.safetensors` in the dir via
    `safetensors.safe_open` (friendly error if absent), `eval()` + `requires_grad_(False)`,
    move to device.
  - `__call__(completions, **kwargs) -> list[float]` — per completion: render + prepend the
    `prompts` kwarg as leading context (best-effort), `split_steps`, one `no_grad`
    `output_hidden_states` forward, gather boundary hidden, `reward_head` → per-step scalars,
    `aggregate_step_scores`. Per-completion loop (no batching — fine for tiny; perf follow-up).
- `build_prm_reward_fn(tcfg, device, trust_remote_code) -> PRMScorer` — validates local-path
  containment (realpath + commonpath under cwd), `trust_remote_code` probe+warn, returns scorer.

### Integration (`trainer/grpo.py::setup()`)

One insertion before `load_reward_fn`: if `tcfg.prm_reward is not None` →
`reward_fn = build_prm_reward_fn(tcfg, self.device, self._trust_remote_code)` instead of
`load_reward_fn(...)`. The existing `apply_reward_shaping` + `wrap_reward_funcs` wrapping applies
unchanged → the v0.71.26 mitigation controller + `mitigation_log.jsonl` observe the PRM reward.

## Part B — bundled rollout envs

### New package `soup_cli/envs/`

Each module exposes `rollout(prompts) -> list[{"prompt", "answer"}]`, deterministic (seeded RNG,
no I/O / no network), usable as `soup_cli.envs.<name>:rollout` in `training.rollout_func`:

- `calculator.py` — arithmetic problems, answer = the number (`math`/`accuracy` reward).
- `retrieval_qa.py` — short document + question, answer = a span (`accuracy` reward).
- `guess_number.py` — single-shot number-**deduction** puzzles (constraints → unique answer;
  honestly framed as deducible, not interactive guessing).

Rows flow through the existing `_normalise_rollout_rows` → replace the GRPO prompt dataset.

### Recipes (`recipes/catalog.py`, +3 → 137)

`grpo-env-calculator` / `grpo-env-retrieval-qa` / `grpo-env-guess-number` — SmolLM2-135M base,
`rollout_backend=openenv` + `rollout_func=soup_cli.envs.*:rollout` + reward. Parse-clean.

## Data flow

- **PRM:** dataset prompts → GRPO generates N completions → `PRMScorer(completions, prompts)` →
  per completion: split steps → PRM forward → gather boundary hidden → `reward_head` → per-step
  scalars → `aggregate(min/prod/last)` → scalar reward list → GRPO advantages.
- **Envs:** `rollout_func` seeds `{prompt, answer}` rows → replace GRPO prompt dataset → GRPO
  trains with `accuracy`/`math` reward.

## Tests (`tests/test_v07130.py`)

- `split_steps` (newlines / empties / whitespace / count cap / char cap / non-str guard).
- `aggregate_step_scores` (min/prod/last correctness, empty→0, single, non-finite defensive,
  bad-mode reject, bool guard).
- Schema parse + every cross-validator (task≠grpo, mlx/unsloth, modality≠text, footgun,
  null-byte/oversize) — assert specific keywords.
- `PRMScorer` against a tiny **real saved PRM fixture** (2-layer CausalLM + reward_head →
  `save_pretrained` → load → score → finite list of right length); "not a PRM" friendly error.
- grpo wiring: `prm_reward` set → PRM scorer built (inject/monkeypatch, CPU tiny).
- Each env's rows pass `_normalise_rollout_rows`; determinism; answers correct
  (calculator arithmetic verified).
- 3 recipes parse via `load_config_from_string`; count bump.
- No-top-level-torch AST test on `utils/prm_reward.py`.

## Security

- `prm_reward` local path: realpath + commonpath containment under cwd at load; `safe_open`
  (safetensors — no pickle) for the reward_head weights; O_NOFOLLOW where reading. HF id (no
  local path) allowed via `from_pretrained`. `trust_remote_code=False` probe+warn.
- Bounds: `_MAX_STEPS`, per-step char cap, aggregate mode allowlist.
- Envs: pure python, deterministic, no I/O / no network; rows re-validated by
  `_normalise_rollout_rows` (caps + anti-smuggle).

## Known limitations (release notes + CLAUDE.md)

1. Proof-of-mechanism only — tiny PRM, synthetic data; scale ask extends #286.
2. Envs are deterministic single-shot seeders, not interactive multi-turn model-in-the-loop
   episodes — the live `openenv` contract passes only `prompts`.
3. Step split = newline heuristic (v1); PRM scored per-completion, no batching (perf follow-up).
4. PRM artifact v1 = Soup-trained format only (HF SeqClassification interop deferred).

## Smoke (RTX 3050 / CPU)

Train a tiny PRM on synthetic step-labelled math → GRPO on SmolLM2-135M with `prm_reward`,
reward curve logged (mitigation log reused); one env end-to-end GRPO run (mirrors #125). Plus
every validator-reject config.
