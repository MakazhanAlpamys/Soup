# Evaluation, Diagnostics & Probes

[← Back to the Soup README](../README.md)

> Eval design/gate, eval-gated training, benchmarks, NLG metrics, calibration, the Elo arena, diagnose, post-train X-ray probes, A/B testing, drift alarms, tunability, and `soup advise`.

**Contents:**

- [Post-train X-rays (`soup probe`, `soup adapters blame --live`)](#post-train-x-rays-soup-probe-soup-adapters-blame---live)
- [Pre-flight Decision (`soup advise`)](#pre-flight-decision-soup-advise)
- [Eval Design Pipeline (`soup eval design / discover / lock / coverage`)](#eval-design-pipeline-soup-eval-design--discover--lock--coverage)
- [Pre-Push Regression Gate (`soup eval gate-install`)](#pre-push-regression-gate-soup-eval-gate-install)
- [Eval-Gated Training](#eval-gated-training)
- [Sequential A/B Harness (`soup ab`)](#sequential-ab-harness-soup-ab)
- [Drift Alarm (`soup drift-alarm`)](#drift-alarm-soup-drift-alarm)
- [Diagnose (Post-Training Report Card)](#diagnose-post-training-report-card)
- [Ship Verdict (`soup ship`)](#ship-verdict-soup-ship)
- [NLG Evaluation Metrics (BLEU + ROUGE)](#nlg-evaluation-metrics-bleu--rouge)
- [Quant Calibration (KL Divergence)](#quant-calibration-kl-divergence)
- [Model Arena (Elo Tournament)](#model-arena-elo-tournament)
- [Model Evaluation](#model-evaluation)
- [Tunability Probe (`soup tunability`)](#tunability-probe-soup-tunability)
- [Eval Depth (`soup eval behavior / capability / checklist / irt-subset`)](#eval-depth-soup-eval-behavior--capability--checklist--irt-subset)

---

## Post-train X-rays (`soup probe`, `soup adapters blame --live`)

Five surfaces that extend `soup diagnose` from 6 failure modes to 10. Mechanistic interpretability has been research-grade for years; v0.66.0 ships the wiring CLI-first so anyone can probe their FT without the SaaS unit-economics tax.

```bash
# 1. Sparse-Autoencoder feature diff: which SAE features moved during FT?
soup probe sae-diff path/to/sae.safetensors pre.json post.json --top-k 20

# 2. Live influence-function blame: which 50 training rows pulled toward this output?
soup adapters blame ./my-adapter --dataset ./train.jsonl --layer q_proj.7 \
    --budget 1h --shards 10 --top-k 50

# 3. Sleeper-agent defection probe: per-token defection rate via calibrated linear probe
soup probe sleeper meta-llama/Llama-3-8B --evidence activations.json
soup probe sleeper my/model --weights probe.npz --evidence activations.json  # real calibrated probe (v0.71.8)

# 3b. Honesty + misuse probes (v0.71.8) — same shape, 5% / 20% verdict bands
soup probe truth meta-llama/Llama-3-8B --evidence activations.json
soup probe harm  meta-llama/Llama-3-8B --evidence activations.json

# 4. Pairwise adapter interference matrix: which pairs can't be deployed together?
soup probe interference losses.json    # exit 2 if worst-pair score ≥ 20%
# v0.71.8: auto-measure the matrix live instead of supplying losses.json
soup probe interference --measure eval.jsonl --base-model meta-llama/Llama-3-8B \
    --adapter a=./adapter-a --adapter b=./adapter-b --device cpu

# 5. Probe pack: list/assemble calibrated probes per base (sleeper + truth + harm per base)
soup probe pack --list                 # list bundled bases
soup probe pack meta-llama/Llama-3-8B  # render the per-base manifest

# 6. SAE auto-download + capture pipeline (v0.71.8)
soup train --config soup.yaml --capture-activations model.layers.5 \
    --capture-prompts probes.jsonl    # writes <output>/activations/activations.json
soup probe sae-diff google/gemma-scope-2b-pt-res pre.json post.json --auto-download
```

Every probe uses the OK / MINOR / MAJOR taxonomy from v0.26 (Quant-Lobotomy) / v0.56 (Diagnose) / v0.65 (Eval Depth). Sleeper / truth / harm / interference exit 2 on MAJOR for CI gating. **v0.71.8** ships real probe weights: `--weights <w.npz|.npy|.safetensors>` loads a calibrated direction (cwd-contained, `O_NOFOLLOW`, `allow_pickle=False`, size-capped); `compute_contrast_probe(positive, negative)` derives one from contrast-pair activations; the bundled specs fall back to a deterministic synthetic seed (the large-base Anthropic-calibrated vectors remain upstream-gated). `soup probe interference --measure` loads the base + each LoRA adapter via PEFT and measures loss per adapter alone (diagonal) and per co-loaded pair (`add_weighted_adapter(combination_type="cat")`, off-diagonal). `soup train --capture-activations` writes an SAE-diff-ready per-token snapshot (the `model.layers.N` path resolves whether or not a LoRA adapter is loaded). The blame runner closes the v0.57 `NotImplementedError` stub via a DataInf-style influence approximation: `cos(grad_row, grad_probe) × |grad_row|`. Operators supply a `probe_fn` returning `(row_grads, probe_grad)`, or the runner falls back to a deterministic synthetic probe so the surface always returns a real `BlameResult` (no exception leaks). SAE feature diff is pure-numpy; the safetensors loader is `O_NOFOLLOW`-protected (TOCTOU defence — closes the symlink swap window between containment check and read); `--auto-download` validates the `HF_HUB_ALLOWLIST` before any network call and rejects a glob result escaping the snapshot dir.


## Pre-flight Decision (`soup advise`)

Run BEFORE you spend 8 hours on a GPU. `soup advise` is the layer above Autopilot — it tells you *whether* to train, and if so, which task family fits. Pure-Python heuristic, no GPU required for the verdict itself.

```bash
# Headline UX — one line gives you a verdict.
soup advise data.jsonl --goal "make our chatbot more concise"
#  Choice:     SFT   (or PROMPT_ENG / RAG / DPO / GRPO)
#  Confidence: 0.71
#  Why:        Task is summarization with 120 rows and healthy diversity ...
#  Flip when:  the prompt-engineering baseline already meets your target ...

# Optional ROI probe (offline heuristic: zero/few-shot + RAG + LoRA estimate).
soup advise data.jsonl --goal "summarize my reports" --probe

# LIVE ROI probe (v0.71.7): loads the model for zero/few-shot token-F1, a short
# LoRA probe, and base-model proximity (held-out logit agreement). Implies --probe.
soup advise data.jsonl --goal "..." --probe-model HuggingFaceTB/SmolLM2-135M

# Print the rubric / evidence trail of the last verdict.
soup advise explain

# Record this verdict to ~/.soup/advise_history.jsonl for later compare.
soup advise data.jsonl --goal "..." --record

# Show prior verdicts (newest first), with per-choice counts.
soup advise compare
```

**The rubric** (advisory, encoded explicitly so `explain` can print it):

1. Dataset rows expose paired `chosen` + `rejected` fields → **DPO**.
2. Task is `reasoning`, dataset has ≥500 rows AND carries `<think>` traces → **GRPO**.
3. Fewer than 50 rows → **PROMPT_ENG** (below the floor for meaningful fine-tuning).
4. Task is `factual_lookup` with high output variance → **RAG**.
5. Otherwise → **SFT**.

**Cross-project confidence bias (v0.71.5).** When `~/.soup/advise_history.jsonl` holds ≥3 prior verdicts for the *same choice* in the *same project*, `soup advise` nudges its confidence (not its decision) toward what worked before: a net-positive precedent record (you accepted it AND its recorded outcome was good) bumps confidence up by a small constant; a net-negative one bumps it down. The rubric verdict itself never changes — only how sure Soup is. Verdicts must be `--record`ed for the bias to kick in.

**Why this command exists.** "Choose fine-tuning vs RAG vs prompt-engineering" is the most-mis-made decision in the space. Reddit, HN, IBM, and Google Cloud all converge on the same advice (start with prompts, escalate to RAG, fine-tune as last resort) and almost everyone ignores it because nobody has the data to prove their case is the exception. Soup `autopilot` picks hyperparameters AFTER you've decided to train; `soup advise` owns the layer above. No trainer library has an incentive to tell users *not to train* — Unsloth's funnel, Axolotl's hosted business, LLaMA-Factory's Alibaba alignment all monetise the training event.


## Eval Design Pipeline (`soup eval design / discover / lock / coverage`)

Trainer libraries help you RUN evals — none help you DEFINE them. The eval-design
pipeline closes that gap with four CPU-only subcommands.

```bash
# 1. Draft a goal-conditioned suite from your training data.
soup eval design data.jsonl --goal "better at SQL" --output evals/design.json

# 2. Discover held-out canaries + memorization probes.
soup eval discover data.jsonl --num-clusters 5 --output evals/canaries.json

# 3. Freeze the design as a checksummed eval_suite artifact.
soup eval lock evals/design.json --output evals/locked.json

# 4. Heuristic gap analysis vs the task taxonomy.
soup eval coverage evals/design.json --task reasoning
```

`soup eval design` clusters training rows by TF-IDF salience, picks a scorer
per dimension (`exact_match` / `regex` / `judge` / `rlvr`) via a goal-keyword
dispatch matrix, and writes a versioned `evals/design.json` of frozen
`EvalDimension` rows.

`soup eval discover` runs farthest-first Jaccard clustering and emits a
`CanarySet` with three groups:

- `held_out` — cluster representatives that test generalisation.
- `adjacent_skills` — rare clusters that catch catastrophic forgetting.
- `memorization_probes` — 25 %-prefix truncations that catch verbatim regurgitation.

`soup eval lock` canonicalises the suite (sorted-key JSON, no whitespace),
computes a SHA-256 over the bytes that hit disk, and optionally attaches the
artifact to a Registry entry as `eval_suite`. Two designs hash identically
iff their semantic content matches.

`soup eval coverage` does heuristic gap analysis against the task taxonomy:
`reasoning` benefits from a `rlvr` dimension, `format_conversion` benefits
from both `regex` and `rlvr`, etc. Missing scorers surface as named
recommendations so operators can spot gaps before shipping the gate.


## Pre-Push Regression Gate (`soup eval gate-install`)

Install a portable pre-push git hook that blocks the push when an adapter
regresses past a tolerance. Threshold checks use paired-bootstrap 95 % CI
so a single outlier row doesn't flip the gate.

```bash
soup eval gate-install --baseline run-abc-123 --suite evals/locked.json
```

The generated `.git/hooks/pre-push` script:

- Compares against a baseline run id from the Soup registry.
- Watches four metrics: `task_accuracy`, `refusal_rate`, `format_validity`,
  `p95_latency_ms`.
- Treats `task_accuracy` / `refusal_rate` / `format_validity` as higher-is-better
  and `p95_latency_ms` as lower-is-better; regression is decided per metric on the
  paired-bootstrap CI bound (upper bound for higher-better, lower for lower-better).
- Uses `shlex.quote` on every embedded value — no shell-injection surface from a
  crafted run id or suite path.
- Refuses to overwrite an existing hook without `--force`; rejects pre-placed
  symlinks at the hook path (TOCTOU defence).

The hook is portable bash (`#!/usr/bin/env bash` shebang) and works under
Git-for-Windows' bundled bash on Windows.


## Eval-Gated Training

Halt training automatically if a declarative eval suite regresses beyond a threshold vs a baseline. The gate runs at epoch boundaries — no wasted compute on runs that are already worse.

**Configure in `soup.yaml`:**

```yaml
training:
  epochs: 5
  eval_gate:
    enabled: true
    suite: ./evals/gate.yaml            # Declarative task list
    every_n_epochs: 1                    # Run gate every N epochs (1-100)
    regression_threshold: 0.05           # Allow 5% drop before halting (0.0-1.0)
    baseline: registry://llama31-chat-v1 # Or a file path, or omit for first run
    on_regression: stop                  # stop | warn | continue
```

**Or pass on the command line:**

```bash
soup train --config soup.yaml --gate ./evals/gate.yaml
```

**Run a gate suite post-hoc (no training):**

```bash
soup eval gate --suite ./evals/gate.yaml --model ./output \
  --baseline registry://llama31-chat-v1
```

**`evals/gate.yaml` example:**

```yaml
tasks:
  - name: math_sanity
    prompts: ./evals/math.jsonl          # prompt + expected
    scoring: exact
  - name: style_judge
    prompts: ./evals/style.jsonl
    scoring: judge
    judge_model: ollama://llama3.1        # SSRF-allowlisted scheme
```

Baselines may be a registry reference (`registry://<name-or-id>`), a file path, or omitted for the first run. Any structured exception (`ValueError`, `FileNotFoundError`, `OSError`) during the gate is treated as a regression under `on_regression: stop`.


## Sequential A/B Harness (`soup ab`)

Proper sequential testing with early-stop guarantees on `latency` / `judge_score` / `retry_rate`. Uses Wald's classic SPRT for the point alternative — the log-likelihood ratio is a martingale under H0, so Type-I error is controlled at every stopping time per the optional stopping theorem (unlike a naive repeated t-test, which inflates Type-I if you peek at the data).

```bash
soup ab --input ab.jsonl --metric latency --effect-size 0.5
# Or with custom alpha / beta
soup ab --input ab.jsonl --metric judge_score --alpha 0.01 --beta 0.10 --effect-size 0.1
```

Input rows look like `{"arm": "control", "latency": 1.23}` or `{"arm": "treatment", "judge_score": 0.91}`. Decision is one of `continue` (keep collecting samples), `reject_h0` (real difference detected), `accept_h0` (no significant difference). Composes with `soup loop canary` (v0.58) — promote or roll back as soon as the LLR clears a decision boundary.

`soup ab` accepts `--slack-url` / `--discord-url` (v0.71.5) and pings the webhook **only when the test actually decides** (`reject_h0` / `accept_h0`) — a still-running `continue` stays quiet so you're not paged on every peek. Same SSRF-hardened validator as `soup drift-alarm`.


## Drift Alarm (`soup drift-alarm`)

Rolling KL divergence on the whitespace-tokenised output distribution catches both behavioural drift ("model now outputs JSON when it used to output prose") and vocabulary drift ("model has started repeating the same 20 phrases"). Cheaper than perplexity — runs in ms over a day of traces.

```bash
soup drift-alarm --reference ft-time.jsonl --live yesterday.jsonl --threshold 0.2

# Optional webhook on drift detected
soup drift-alarm --reference ft-time.jsonl --live yesterday.jsonl --threshold 0.2 \
                 --slack-url   https://hooks.slack.com/services/... \
                 --discord-url https://discord.com/api/webhooks/...
```

Default threshold 0.2 matches v0.43.0 KL-delta quant-check thresholds. Webhooks are SSRF-validated (loopback HTTP only, RFC1918 / 169.254.x / 0.0.0.0 rejected). On drift the CLI exits with code 3 — cron-friendly automation.


## Diagnose (Post-Training Report Card)

`soup diagnose` scores seven independent failure modes for a trained adapter and renders an OK / MINOR / MAJOR verdict per mode plus an overall headline — same taxonomy as Quant-Lobotomy. Useful for catching adapter regressions that a loss curve cannot distinguish from a healthy run.

```bash
# Neutral report (no model load — runs as a sanity check)
soup diagnose my-run-id

# LIVE (v0.71.7): load the model and run all six probes for real
soup diagnose my-run-id --base-model HuggingFaceTB/SmolLM2-135M \
    --adapter ./out --dataset train.jsonl --tokenizer HuggingFaceTB/SmolLM2-135M

# RAFT models: tell the citation probe which style + shuffle seed the run used
# so the golden [doc-N] ids line up with what the model saw at train time (v0.71.17)
soup diagnose my-run-id --base-model HuggingFaceTB/SmolLM2-135M --adapter ./raft_out \
    --dataset raft.jsonl --citation-style bracket --shuffle-seed 1

# Compute scores from a pre-built evidence JSON
soup diagnose my-run-id --evidence evidence.json --output diag.json

# Twitter-shareable SVG badge embeddable in a model card
soup diagnose my-run-id --badge diag.svg

# Attach the report to a Model Registry entry as a first-class artifact
soup diagnose my-run-id --output diag.json --attach-to-registry abc123
```

**Live runners (v0.71.7).** With `--base-model` the six probes run against the loaded model
(+ optional `--adapter` LoRA path, `--dataset` for the forgetting / format / memorization probes,
`--tokenizer` for a sub-word memorization variant) instead of emitting neutral OK. `refusal` uses
a built-in probe set; `format` only fires when the dataset's own targets look like JSON;
`contamination` stays neutral unless a benchmark corpus is supplied. Validated on SmolLM2-135M.

**Seven failure-mode probes:**

| Mode | What it catches | Score range |
|------|-----------------|-------------|
| `forgetting` | Catastrophic forgetting on MMLU / HellaSwag / domain hold-outs | Δ accuracy vs base, tolerance band |
| `refusal` | Refusal-rate regression on harmful / benign probe sets | abs(Δ harmful) + abs(Δ benign) |
| `format` | JSON / regex / tool-call validity drift | fraction of valid outputs |
| `mode_collapse` | Diversity collapse at T=0 and T=1 | pairwise n-gram Jaccard distance |
| `memorization` | Verbatim training-prefix echo on partial prompts | 1 − echo_rate |
| `contamination` | Training data overlapping public benchmarks | 1 − contamination_rate |
| `citation` | RAFT model stopped citing the supporting `[doc-N]` (v0.71.10) | fraction of answers citing the golden doc |

**Verdict pill colours:** OK (≥ 0.85) green / MINOR (≥ 0.60) amber / MAJOR (< 0.60) red. `soup diagnose` exits 2 when the overall verdict is MAJOR — wire into CI to fail the build on regression.

**Post-training gate:** `soup train --diagnose-gate <evidence.json>` runs the same scorer after training finishes and refuses to mark the run successful when any mode comes back MAJOR. Composes with `--gate <eval-suite>` (v0.26) — the eval gate catches accuracy regressions vs a baseline; the diagnose gate catches behaviour regressions the eval suite is blind to.


## Ship Verdict (`soup ship`)

After fine-tuning, `soup ship` answers one question — did the model get better, or did I
break it? — as a single binary **SHIP / DON'T SHIP** plus a one-screen reason. It fuses two
legs into one decision:

- **Leg 1 (task win):** the task metric *strictly* improved, base → tuned.
- **Leg 2 (the moat):** *no* general benchmark regressed past the forgetting threshold
  (default `0.05` **absolute** points — same semantics as `EvalGateConfig.regression_threshold`).

```
SHIP  ⇔  task_tuned > task_base  AND  ∀ benchmark: base − tuned ≤ forgetting_threshold
else DON'T SHIP — even if the task metric looks great.
```

Exit codes are CI-gateable: **0 = SHIP, 2 = DON'T SHIP, 1 = runtime error**. A tie on leg 1, a
single regressed benchmark, or a missing baseline all yield DON'T SHIP (a missing baseline
*refuses* rather than silently shipping).

```bash
# Live: base vs a LoRA adapter, metric leg + default mini benchmarks
soup ship --base HuggingFaceTB/SmolLM2-135M-Instruct --adapter ./out \
  --task-eval tasks.jsonl --device cuda

# Leg-1 via LLM-as-a-judge instead of an accuracy metric
soup ship --base <m> --adapter ./out --task-eval tasks.jsonl \
  --task-mode judge_score --judge-model ollama://llama3.1

# Leg-1 via a true pairwise judge win-rate (v0.71.31 #284) — the judge picks
# base vs tuned per prompt (swap-debiased); base = 0.5 coin-flip, won <=> winrate > 0.5
soup ship --base <m> --adapter ./out --task-eval tasks.jsonl \
  --task-mode pairwise --judge-model ollama://llama3.1

# Leg-2 via lm-eval benchmarks, base scores supplied by --baseline
soup ship --base <m> --tuned ./out --task-eval tasks.jsonl \
  --general-suite mmlu,hellaswag --baseline registry://abc123

# Offline: decide from pre-computed scores, persist the verdict JSON
soup ship --evidence evidence.json --output verdict.json
```

**`--evidence` shape** (no model load — the offline / CI path):

```json
{
  "task": {"mode": "metric", "base": 0.40, "tuned": 0.55},
  "benchmarks": {
    "mini_mmlu": {"base": 0.80, "tuned": 0.79},
    "mini_common_sense": {"base": 0.60, "tuned": 0.62}
  }
}
```

Leg-2 defaults to the built-in mini benchmarks (`mini_mmlu` / `mini_common_sense` /
`mini_instruction`) — offline, CPU, instant. `--general-suite <names>` with non-mini names
routes through the lm-eval harness. Pairwise judge win-rate (`--task-mode pairwise`) is planned
for a later release. The engine lives in `soup_cli.utils.ship_verdict` (`decide_ship` is a pure
function — the whole truth table is CPU-testable).


## NLG Evaluation Metrics (BLEU + ROUGE)

Pure-Python BLEU + ROUGE-1 / ROUGE-2 / ROUGE-L for `soup eval custom`:

```python
from soup_cli.utils.nlg_metrics import (
    bleu_score, rouge_l_score, compute_nlg_metric, NLG_METRICS,
    effective_tokens_per_second,
)

bleu_score(["the cat sat on the mat"], ["the cat sat on the mat"])
# 1.0
rouge_l_score(["the quick brown fox"], ["a quick brown dog"])
# 0.5
compute_nlg_metric("rouge_2", preds, refs)
# generic dispatch by canonical name

effective_tokens_per_second(unmasked_tokens=12_500_000, wall_clock_seconds=600.0)
# 20833.33  — None when wall_clock <= 0 (no fabrication)
```

Smoothed BLEU uses Chen & Cherry epsilon for zero-correct buckets where
`total[n] > 0`; empty buckets (e.g. predictions shorter than `max_n` tokens)
force the score to 0.0.


## Quant Calibration (KL Divergence)

Compare a quantized model to a full-precision baseline on a small fixed prompt
set. OK / MINOR / MAJOR thresholds at 0.05 / 0.20 mean KL — same scale as
`soup eval quant-check`.

```python
from soup_cli.eval.calibrate import run_calibration

# baseline_logits / quantized_logits: list[list[float]] aligned per-prompt
report = run_calibration(baseline_logits, quantized_logits)
print(report.delta_status, report.mean_kl)
# OK 0.012
```

The kernel is pure-math and capped at 10 000 prompts to defend against
accidental OOM. `CalibrationReport` is a frozen dataclass.


## Model Arena (Elo Tournament)

Local leaderboard with Elo ratings (K=32, base 1500). Bring your own pairwise
winners — Soup just keeps the books:

```python
from soup_cli.eval.arena import Tournament

t = Tournament()
t.record("llama-3.1-8b-finetune", "qwen2.5-7b-finetune", winner="a")
t.record("llama-3.1-8b-finetune", "mistral-7b-finetune", winner="draw")
for row in t.leaderboard():
    print(row)
```

Caps: 256 models per tournament, 1M matches. Model names with `[` or `]`
characters are rejected so leaderboard rows can't be markup-injected.


## Model Evaluation

Full-featured evaluation platform with standard benchmarks, custom evals, LLM-as-a-judge, and human evaluation:

```bash
# Install eval dependencies
pip install 'soup-cli[eval]'

# Standard benchmarks (wraps lm-evaluation-harness)
soup eval benchmark --model ./output --benchmarks mmlu,gsm8k,hellaswag

# Custom eval tasks from JSONL
soup eval custom --tasks eval_tasks.jsonl --model ./output

# LLM-as-a-judge (score model outputs using GPT-4o, Ollama, etc.)
soup eval judge --target responses.jsonl --model gpt-4o-mini --provider openai
soup eval judge --target responses.jsonl --model llama3.1 --provider ollama

# Auto-eval after training (configure in soup.yaml)
soup eval auto --config soup.yaml

# Compare eval results between two training runs
soup eval compare run_20260301_143052_a1b2 run_20260315_091023_c3d4

# Local leaderboard across all evaluated models
soup eval leaderboard
soup eval leaderboard --format json
soup eval leaderboard --format csv

# Human A/B evaluation with Elo ratings
soup eval human --input prompts.jsonl --model-a ./model_a --model-b ./model_b
```

### Quant-Lobotomy Checker

Before you ship a quantized model, verify it didn't lose skills. The checker runs the same task list against the `--before` and `--after` models and renders a per-task OK / MINOR / MAJOR verdict.

```bash
# Compare a pre-quant model with its post-quant version
soup eval quant-check \
  --before ./output \
  --after  ./output/quantized.q4_k_m.gguf \
  --tasks  ./evals/sanity.jsonl

# Both sides may be registry refs
soup eval quant-check \
  --before registry://llama31-chat-v1 \
  --after  registry://llama31-chat-v1-q4 \
  --tasks  ./evals/sanity.jsonl

# Render as JSON for CI integration
soup eval quant-check --before X --after Y --tasks t.jsonl --format json
```

**Verdict thresholds (per task):**
- `OK` — score delta ≤ 2%
- `MINOR` — delta 2-10% (investigate)
- `MAJOR` — delta > 10% (do NOT ship)

Paths are containment-checked, and `registry://` refs are resolved with an optional `kinds` filter so you never pick the wrong artifact.

### Custom Eval Format

```jsonl
{"prompt": "What is 2+2?", "expected": "4", "category": "math", "scoring": "exact"}
{"prompt": "Explain gravity", "expected": "force.*attraction", "scoring": "regex"}
{"prompt": "Capital of France?", "expected": "Paris", "scoring": "contains"}
```

### Auto-Eval Config (soup.yaml)

```yaml
eval:
  auto_eval: true
  benchmarks: [mmlu, gsm8k]
  custom_tasks: eval_tasks.jsonl
  judge:
    model: gpt-4o-mini
    provider: openai
```


## Tunability Probe (`soup tunability`)

Before committing to a single base model, run a short LoRA probe on every reasonable candidate against your held-out slice. v0.64.0 ships an 8-entry default catalogue covering Qwen3, Llama-3.2, Gemma 3, Phi-4, SmolLM3, and Qwen2.5 across the 0.6 B – 3.8 B band.

```bash
# List the built-in catalogue
soup tunability --list

# Dry-run a sweep across a subset
soup tunability --dataset ./eval.jsonl --candidates qwen3-0.6b,phi-4-mini --plan-only

# Run the full sweep + write a JSON report
soup tunability --dataset ./eval.jsonl --probe-steps 100 --output ./tunability.json
```

The report is a Pareto frontier over (eval delta from base, train cost, license) — candidates that nothing dominates on both axes survive, so you see a clean shortlist instead of a noisy single-leaderboard score. By default the probe is a deterministic offline heuristic; pass `--live` (v0.71.7) to run a real per-candidate LoRA probe (loads each `repo_id`, trains `--probe-steps` on a held-out-excluded slice, reports the held-out-loss drop). `--device` selects cuda / cpu.


## Eval Depth (`soup eval behavior / capability / checklist / irt-subset`)

v0.65 ships five new evaluation surfaces that close the "judges are biased, suites are arbitrary, eval costs are high" gaps that SaaS evals (Galileo, Braintrust) don't address.

**Judge calibration** — refuse to use an uncalibrated judge in production:

```python
from soup_cli.eval.calibrate import (
    PairwiseJudgement, run_pairwise_calibration, ensure_judge_calibrated,
)

# Run your judge on a calibration set with positions swapped.
judgements = [PairwiseJudgement(...) for _ in oracle_set]
report = run_pairwise_calibration(judgements, scores=confidence_scores)
ensure_judge_calibrated(report)  # raises RuntimeError if not calibrated
```

The report carries `position_bias` ∈ [-1, 1] (0 = no slot preference), a conformal abstention threshold from the score quantile, agreement-rate vs the oracle, and a `calibrated` bool. `ensure_judge_calibrated` refuses on missing report, low agreement, or extreme bias — so production scoring code can fail loud, not silent.

Persist a calibration once and reuse it across runs (v0.71.1): `write_judge_calibration(report, "calib.json")` writes a cwd-contained JSON, and `load_judge_calibration("calib.json")` re-validates it on load (a corrupt or out-of-range field on disk is rejected, since it's the production-gate safety net). The artifact attaches to the registry under the new `judge_calibration` kind.

**Behaviour battery** — pre/post diff on bundled safety / refusal / sycophancy probe sets:

```bash
# Score over-refusal regression on XSTest (operator supplies evidence JSON)
soup eval behavior my_run --battery xstest --evidence ev.json --output diff.json

# LIVE (v0.71.7): generate pre/post responses on the bundled battery + score the diff
soup eval behavior my_run --battery xstest \
    --base-model HuggingFaceTB/SmolLM2-135M --adapter ./out

# Bundled batteries: xstest, harmbench, jailbreakbench, elephant, syceval
# Harmful prompts ship REDACTED — pull real sets from upstream papers.
```

Word-boundary regex agreement (no `"safe" in "unsafe"` false positives); OK/MINOR/MAJOR thresholds match the v0.26 / v0.56 taxonomy.

**Capability auto-suite** — pre-bundled profile selector with friendly `lm-eval-harness` task ids:

```bash
soup eval capability my_run --suite math --output cap.json   # AIME + MATH-500
soup eval capability my_run --suite code --output cap.json   # HumanEval+ + SWE-bench-Verified
soup eval capability my_run --suite fast --output cap.json   # MMLU-Pro + HumanEval+
soup eval capability my_run --suite full --output cap.json   # all 7 benchmarks

# LIVE (v0.71.7): invoke lm-eval-harness per task against a real model
soup eval capability my_run --live --model HuggingFaceTB/SmolLM2-135M \
    --tasks arc_easy --limit 1 --device cpu
```

Without `--live` it emits the (benchmark, lm-eval task) manifest; chain into the existing `soup eval benchmark` surface. With `--live --model <id>` (v0.71.7) it runs lm-eval-harness per resolved task — or a `--tasks` override — isolating per-task failures and capping examples with `--limit`.

**CheckList behavioural DSL** — Ribeiro et al. 2020 MFT / INV / DIR tests:

```yaml
# tests.yaml
tests:
  - name: capital-france
    kind: mft
    prompts: ["What is the capital of France?"]
    expected: ["paris"]
  - name: paraphrase-add
    kind: inv
    prompts:
      - "Add 2 and 2."
      - "Add two and two."
```

```bash
soup eval checklist tests.yaml --evidence responses.json
```

`mft` = response must contain a keyword as a whole word (`"sand"` won't pass for `"and"`); `inv` = all paraphrases must agree; `dir` = directional expectation under perturbation.

**IRT subset selection** — pick a smaller eval set that preserves ranking power:

```bash
# Pick top-info 30% of items (5-10x eval-bill cut without losing power)
soup eval irt-subset per_item_correctness.jsonl --size small --output plan.json

# Richer item models: 2PL learns per-item discrimination, 3PL adds a guessing floor
soup eval irt-subset per_item_correctness.jsonl --size small --model 2pl
```

`--model` picks the item-response model (default `1pl`). `1pl` is the closed-form Rasch fit (`β̂_i = -log(p̂_i / (1 - p̂_i))`); `2pl` adds a per-item discrimination parameter and `3pl` a guessing floor, both via joint coordinate-ascent MLE (v0.71.6). Items rank by Fisher information at θ=0 (`p̂ · (1-p̂)` in 1PL — maximised at 50/50 items, since extremes carry no new ranking information). `full` keeps 100%, `small` keeps 30%, `tiny` keeps 10%.


