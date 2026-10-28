# CLI reference

[← Back to the docs index](README.md)

> **Generated file — do not edit by hand.**
> It is rendered from the live Typer command tree (`soup_cli.cli:app`). After adding or
> removing a command, an option or an argument, or changing its type, its default or
> whether it is required, run `python scripts/gen_reference_docs.py` and commit the
> result; the test suite fails until the page matches. Help text is refreshed when a
> release is cut, so between releases the wording here can trail the code.

Every `soup` command with every option and argument. For a one-line-per-task listing
see the [Command reference](commands.md); for the keys of `soup.yaml` see the
[Config reference](config-reference.md).

How to read a table:

- **Name** is an option with its short form, or an argument in capitals (`PATH`;
  `PATH...` takes any number of values). `--flag` / `--no-flag` is one switch with
  both spellings.
- **Type**: a `flag` takes no value; `text`, `integer`, `float` and `path` take one;
  a `repeatable` option may be given more than once.
- **Default** is the value used when the option is left out (`on` / `off` for a flag);
  an empty cell means there is none. *(set at runtime)* marks a default computed on
  the machine, which this page does not print.
- **Required** says `yes` when the command refuses to run without it.

`--help` works on every command and is not listed. Global options go before the
command name: `soup --verbose train --config soup.yaml`.

## Global options

Usage: `soup [OPTIONS] <command> [ARGS]...`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--version` | flag | off |  | Show Soup CLI version. |
| `--verbose`, `-V` | flag | off |  | Show full traceback on errors |
| `--log-level` | text | `normal` |  | Logging tier: quiet \| normal \| verbose \| debug |
| `--no-audit-log` | flag | off |  | Disable the local HIPAA/SOC2 audit log for this invocation (also via SOUP_NO_AUDIT_LOG=1). Default: a one-line record per command under \~/.soup/audit.jsonl. v0.71.3. |
| `--no-telemetry` | flag | off |  | Opt-out of anonymous hardware-only telemetry for this run. |
| `--install-completion` | flag |  |  | Install shell completion for the current shell. |
| `--show-completion` | flag |  |  | Print the shell completion script, to copy or customise it. |

## Commands

| Command | Summary |
|---|---|
| [`soup ab`](#soup-ab) | Two-sided mSPRT sequential A/B harness on latency / judge_score / retry_rate: reports whether the treatment is better or worse, valid under re-running after every new row (v0.63.0). |
| [`soup adapters`](#soup-adapters) | Adapter management: list, info, compare LoRA adapters. |
| [`soup advise`](#soup-advise) | Pre-flight decision: PROMPT_ENG / RAG / SFT / DPO / GRPO. Run BEFORE you spend 8 hours on a GPU (v0.54.0). |
| [`soup agent`](#soup-agent) | Agent Forge: spec -> tool-calling dataset / train / eval (v0.46.0). |
| [`soup airgap-bundle`](#soup-airgap-bundle) | Build a signed tarball with model + datasets + wheels + kernels (v0.60.0). |
| [`soup apple-adapter`](#soup-apple-adapter) | Convert / sign HF / MLX / Apple FoundationModels adapters (v0.68.0). |
| [`soup apply`](#soup-apply) | Execute the planned training run, refusing on drift between soup.yaml and soup.tfstate (v0.64.0). |
| [`soup attest`](#soup-attest) | In-toto + SLSA-3 attestations per Soup Can stage (v0.59.0). |
| [`soup audit-log`](#soup-audit-log) | HIPAA/SOC2-shaped JSONL audit log: tail + rotate (v0.59.0). |
| [`soup autopilot`](#soup-autopilot) | Autopilot: give model+data+goal, Soup picks optimal hyperparameters. |
| [`soup bench`](#soup-bench) | Benchmark a model: inference speed (infer, the default) or training (train). |
| [`soup bom`](#soup-bom) | CycloneDX ML-BOM + SPDX AI bill-of-materials emitter (v0.59.0). |
| [`soup build`](#soup-build) | dbt-for-SFT DAG: validate, plan, and materialise dataset transforms. |
| [`soup can`](#soup-can) | Soup Cans: pack/inspect/verify/fork shareable .can artifacts. |
| [`soup card`](#soup-card) | Generate a HuggingFace model card from a registry entry. |
| [`soup chat`](#soup-chat) | Chat with a fine-tuned model in the terminal. |
| [`soup ci`](#soup-ci) | Fine-tuning CI: init a PR gate workflow. |
| [`soup compile`](#soup-compile) | Compile a DSPy / GEPA prompt program against an eval suite (v0.68.0). |
| [`soup compile-tools`](#soup-compile-tools) | Compile / optimize tool schemas + descriptions (v0.68.0). |
| [`soup completions`](#soup-completions) | Emit a bash / zsh / fish completion script. Use with `eval "$(soup completions bash)"` (v0.64.0). |
| [`soup cost`](#soup-cost) | Estimate training cost in USD across cloud providers. |
| [`soup data`](#soup-data) | Dataset tools: inspect, convert, merge, dedup, validate, stats. |
| [`soup delinearize-llama4`](#soup-delinearize-llama4) | Delinearize Llama 4 fused expert weights to 3-D form. |
| [`soup deploy`](#soup-deploy) | Deploy models: Ollama integration (deploy, list, remove). |
| [`soup diagnose`](#soup-diagnose) | Post-training report card: forgetting / refusal / format / mode_collapse / memorization / contamination (v0.56.0). |
| [`soup diff`](#soup-diff) | Compare outputs of two models side-by-side on the same prompts. |
| [`soup distill-prompt`](#soup-distill-prompt) | Distill prompt-heavy traces into a small FT plan (v0.68.0). |
| [`soup doctor`](#soup-doctor) | Check system dependencies, GPU, and compatibility. |
| [`soup draft`](#soup-draft) | Train + measure a speculative-decoding draft model: distil your tuned model into a tiny draft, then serve it with --auto-spec (v0.71.33). |
| [`soup drift-alarm`](#soup-drift-alarm) | Rolling-KL drift alarm on output-token distribution with optional Slack/Discord webhook (v0.63.0). |
| [`soup edit`](#soup-edit) | Knowledge editing (ROME / MEMIT / AlphaEdit) - patch facts without re-training (v0.61.0). |
| [`soup env`](#soup-env) | Hermetic env lockfile + ABI drift detection (v0.64.0). |
| [`soup eval`](#soup-eval) | Evaluate models: benchmarks, custom evals, LLM judge, leaderboard. |
| [`soup expect`](#soup-expect) | Run an expectations suite against a JSONL dataset. |
| [`soup export`](#soup-export) | Export a model to GGUF, ONNX, TensorRT-LLM, BitNet, or TorchAO format. |
| [`soup fetch`](#soup-fetch) | Fetch a ready-to-edit example config from the bundled catalog. |
| [`soup history`](#soup-history) | Show the lineage tree for all entries with a given name. |
| [`soup infer`](#soup-infer) | Run batch inference on a JSONL file of prompts. |
| [`soup ingest`](#soup-ingest) | Universal trace importer: Langfuse / LangSmith / Helicone / OpenPipe / OTel / OpenAI Stored Completions (v0.63.0). |
| [`soup init`](#soup-init) | Create a new soup.yaml config interactively or from a template. |
| [`soup iterative-dpo`](#soup-iterative-dpo) | Iterative DPO sample, score, pair, and train loop. |
| [`soup license-advisor`](#soup-license-advisor) | Recommend a license-clean base for a deploy target (b2c / defense / embedded) + flag downstream risk (v0.64.0). |
| [`soup llama`](#soup-llama) | Proxy to llama.cpp binaries (cli / mtmd-cli / gguf-split / server). |
| [`soup local-rl`](#soup-local-rl) | Personal-LLM flywheel daemon (init / status / record / harvest / train) (v0.68.0). |
| [`soup lock`](#soup-lock) | Shared run lockfile (write / show / check). |
| [`soup loop`](#soup-loop) | Data flywheel: traces -> preference pairs -> DPO -> gate -> canary deploy -> rollback, all from the CLI (v0.58.0). |
| [`soup mcp`](#soup-mcp) | Model Context Protocol server - drive Soup from any MCP client (v0.71.28). |
| [`soup merge`](#soup-merge) | Merge a LoRA adapter with its base model into a full model. |
| [`soup merge-sharded-fsdp-weights`](#soup-merge-sharded-fsdp-weights) | Consolidate FSDP shard files into a single safetensors file. |
| [`soup migrate`](#soup-migrate) | Import a config from LLaMA-Factory, Axolotl, or Unsloth notebook. |
| [`soup monitor`](#soup-monitor) | Live GPU monitor: Util / Temp / VRAM / Power per GPU. |
| [`soup plan`](#soup-plan) | Render a pre-flight training plan (cost / ETA / SHA / VRAM) and write soup.tfstate for `soup apply` to consult (v0.64.0). |
| [`soup plugins`](#soup-plugins) | Discover, list, enable, and disable Soup plugins. |
| [`soup probe`](#soup-probe) | Activation probes: sleeper-agent defection / honesty / misuse / pairwise interference / SAE feature diff / probe pack (v0.66.0, truth+harm v0.71.8). |
| [`soup profile`](#soup-profile) | Estimate memory, speed, and GPU requirements BEFORE training. |
| [`soup prune-prompt`](#soup-prune-prompt) | Detect + strip a shared system-prompt prefix across training data so the FT model internalises it (v0.63.0). |
| [`soup push`](#soup-push) | Push a trained model to HuggingFace Hub (or alternate hub). |
| [`soup quantize`](#soup-quantize) | Quantize a model - ergonomic alias for `soup export --format <to>`. |
| [`soup quickstart`](#soup-quickstart) | Run a complete demo: create sample data, config, and train. |
| [`soup ra-dit`](#soup-ra-dit) | RA-DIT two-stage orchestrator: retriever -> generator (v0.71.10). |
| [`soup recipes`](#soup-recipes) | Ready-made configs: list, show, use, search recipes for popular models. |
| [`soup registry`](#soup-registry) | Model Registry: push, list, show, diff, search, promote, delete. |
| [`soup reward`](#soup-reward) | Synthesize a deterministic reward verifier from reference outputs, with a calibration report that refuses degenerate verifiers (v0.71.40). |
| [`soup rewind`](#soup-rewind) | Name the dataset rows behind a loss spike, from the training flight recorder. |
| [`soup runs`](#soup-runs) | Experiment tracking: list, show, compare runs. |
| [`soup serve`](#soup-serve) | Start a local inference server with OpenAI-compatible API. |
| [`soup ship`](#soup-ship) | SHIP / DON'T SHIP verdict after fine-tuning: task win AND no catastrophic forgetting, fused into one decision (v0.71.25). |
| [`soup shrink`](#soup-shrink) | Depth-prune a model (drop the least-important contiguous layer block by residual angular distance) + optional distill-heal (v0.71.29). |
| [`soup spectrum`](#soup-spectrum) | Spectrum SNR scan for targeted training (v0.71.23). |
| [`soup steer`](#soup-steer) | Activation steering (CAA / ITI / RepE) - inference-time intervention without retraining (v0.62.0). |
| [`soup sweep`](#soup-sweep) | Run hyperparameter sweep: grid or random search over training parameters. |
| [`soup tokenizer`](#soup-tokenizer) | Tokenizer tools: train a BPE tokenizer from JSONL (v0.53.9). |
| [`soup train`](#soup-train) | Start training from a soup.yaml config. |
| [`soup tui`](#soup-tui) | Full-screen terminal dashboard: list runs, inspect metrics, tail logs. |
| [`soup tunability`](#soup-tunability) | Probe-train 6-10 small bases on a held-out slice + report Pareto frontier of (eval delta, train cost, license) (v0.64.0). |
| [`soup ui`](#soup-ui) | Launch the Soup Web UI. |
| [`soup version`](#soup-version) | Show Soup CLI version. |
| [`soup why`](#soup-why) | Explain training anomalies for a run in plain English. |

## `soup ab`

Two-sided mSPRT sequential A/B harness on latency / judge_score / retry_rate: reports whether the treatment is better or worse, valid under re-running after every new row (v0.63.0).

Usage: `soup ab [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--input`, `-i` | text |  | yes | JSONL with {arm, &lt;metric>} per row. |
| `--metric` | text |  | yes | Metric: latency \| judge_score \| retry_rate. |
| `--alpha` | float | `0.05` |  | Error rate (0, 1) of both verdicts, kept for a test re-run after every new row, at any number of rows: with no difference the test ends in reject_h0, and with a true difference of --effect-size or more in accept_h0, each in at most this share of runs. |
| `--beta` | float |  |  | Retired (#1418): refused with a usage error. --alpha now bounds wrong accept_h0 verdicts too. |
| `--effect-size` | float | `0.1` |  | Minimum detectable difference in means. |
| `--slack-url` | text |  |  | Optional Slack webhook URL — POSTed on a reject_h0 / accept_h0 decision (not on continue). SSRF-validated. |
| `--discord-url` | text |  |  | Optional Discord webhook URL — POSTed on a reject_h0 / accept_h0 decision (not on continue). SSRF-validated. |

## `soup adapters`

Adapter management: list, info, compare LoRA adapters.

Usage: `soup adapters <command> [ARGS]...`

Subcommands:

- [`arithmetic`](#soup-adapters-arithmetic)
- [`audit`](#soup-adapters-audit)
- [`bisect`](#soup-adapters-bisect)
- [`blame`](#soup-adapters-blame)
- [`branch`](#soup-adapters-branch)
- [`branches`](#soup-adapters-branches)
- [`check`](#soup-adapters-check)
- [`check-safetensors`](#soup-adapters-check-safetensors)
- [`checkout`](#soup-adapters-checkout)
- [`compare`](#soup-adapters-compare)
- [`diff`](#soup-adapters-diff)
- [`info`](#soup-adapters-info)
- [`list`](#soup-adapters-list)
- [`merge`](#soup-adapters-merge)
- [`pr`](#soup-adapters-pr)
- [`scan`](#soup-adapters-scan)
- [`sign`](#soup-adapters-sign)
- [`verify`](#soup-adapters-verify)

### `soup adapters arithmetic`

Task-vector arithmetic over LoRA adapters (add / scale / negate) (v0.71.34).

Usage: `soup adapters arithmetic [OPTIONS] EXPRESSION`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `EXPRESSION` | text |  | yes | Task-vector expression, e.g. "coder + 0.5\*math - toxic". |
| `--adapter` | text, repeatable |  | yes | name=path mapping for each adapter referenced in the expression (repeatable). Names use \[A-Za-z0-9\_.-]. |
| `--output`, `-o` | text |  | yes | Output directory for the merged adapter |
| `--rank` | integer, at least 1 |  |  | Truncated-SVD rank cap. Passing it routes the merge through the exact concat path and caps the output rank at N via SVD (works for same-rank AND mixed-rank inputs). Without it, mixed-rank inputs keep the full concatenated rank (Σ rᵢ) and same-rank inputs use the fast element-wise path. |
| `--allow-unscanned` | flag | off |  | Skip the backdoor-scan gate (v0.71.2 #192). By default refuses an input whose `adapters scan` returns FAIL or cannot be scanned. |
| `--allow-cross-base` | flag | off |  | Allow combining adapters trained on different base models. By default a base-model mismatch is refused (task vectors from different bases are not comparable). |

### `soup adapters audit`

Check whether a finished run did what its config asked for (#762).

Usage: `soup adapters audit [OPTIONS] ADAPTER`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `ADAPTER` | text |  | yes | Path to adapter directory |
| `--config`, `-c` | text |  | yes | soup.yaml the run was started from |
| `--json` | flag | off |  | Emit machine-readable JSON |

### `soup adapters bisect`

Binary-search a training history to find the first failing checkpoint.

Usage: `soup adapters bisect [OPTIONS] HISTORY...`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `HISTORY...` | text, any number |  | yes | Ordered checkpoint identifiers (oldest first). At least 2. Example: `soup adapters bisect ckpt-100 ckpt-200 ckpt-300 ckpt-400` |
| `--eval-command` | text |  | yes | Shell command template; '{ckpt}' is replaced with each checkpoint id. Exit code 0 means PASS, non-zero means FAIL. Example: 'soup eval custom --model {ckpt} --tasks tasks.jsonl' |
| `--plan-only` | flag | off |  | Print the plan and exit without running the bisect |
| `--output`, `-o` | text |  |  | Write JSON result to path |

### `soup adapters blame`

Attribute weight movement to dataset rows via DataInf influence (v0.66.0).

Usage: `soup adapters blame [OPTIONS] ADAPTER_DIR`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `ADAPTER_DIR` | text |  | yes | Path to trained adapter |
| `--dataset` | text |  | yes | Training JSONL the adapter was built on |
| `--layer` | text |  | yes | Layer to attribute (e.g. q_proj.7) |
| `--budget` | text | `4h` |  | Wall-clock budget (e.g. 4h, 30m) |
| `--shards` | integer, 2 to 100 | `10` |  | Number of dataset shards for leave-one-out |
| `--top-k` | integer, 1 to 10000 | `50` |  | Number of top influencer rows to report (v0.66.0) |
| `--plan-only` | flag | off |  | Print plan and exit without running |
| `--output`, `-o` | text |  |  | Write JSON blame result to path (v0.66.0) |

### `soup adapters branch`

Snapshot a training environment as a comparable branch (v0.57.0).

Usage: `soup adapters branch [OPTIONS] NAME`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `NAME` | text |  | yes | Branch name (alphanumeric + .\_-) |
| `--config`, `-c` | text |  |  | Path to soup.yaml (required unless --from-registry) |
| `--base` | text |  |  | Base model id (required unless --from-registry) |
| `--dataset` | text |  |  | Training dataset path (optional) |
| `--attach-to-registry` | text |  |  | Registry entry id/ref to attach this branch pointer to as a branch_ref artifact + link via registry_entry_id (v0.71.4). |
| `--from-registry` | text |  |  | Derive config + base_model + dataset hash from a Registry entry instead of --config/--base (v0.71.4). |

### `soup adapters branches`

List all snapshotted branches (v0.57.0).

Usage: `soup adapters branches`

Takes no options or arguments.

### `soup adapters check`

Audit whether an adapter is trained: Frobenius delta, zero B, inner keys (#1721).

Usage: `soup adapters check [OPTIONS] ADAPTER`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `ADAPTER` | text |  | yes | Path to adapter directory |
| `--json` | flag | off |  | Emit machine-readable JSON document |

### `soup adapters check-safetensors`

Refuse pickle / PyTorch-classic weights at the boundary (v0.60.0).

Usage: `soup adapters check-safetensors [OPTIONS] ADAPTER`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `ADAPTER` | text |  | yes | Path to adapter / model directory |
| `--strict` | flag | off |  | Exit 3 on any unsafe (pickle / PyTorch-classic) weight file |

### `soup adapters checkout`

Restore a snapshotted branch's config into cwd (v0.57.0).

Usage: `soup adapters checkout [OPTIONS] NAME`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `NAME` | text |  | yes | Branch name to check out |
| `--output`, `-o` | text | `soup.yaml` |  | Where to write the restored config |

### `soup adapters compare`

Compare two LoRA adapters side-by-side.

Usage: `soup adapters compare ADAPTER1 ADAPTER2`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `ADAPTER1` | text |  | yes | Path to first adapter |
| `ADAPTER2` | text |  | yes | Path to second adapter |

### `soup adapters diff`

Per-layer ΔW Frobenius diff + effective-rank drift (v0.57.0).

Usage: `soup adapters diff [OPTIONS] ADAPTER_A ADAPTER_B`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `ADAPTER_A` | text |  | yes | Path to first adapter |
| `ADAPTER_B` | text |  | yes | Path to second adapter |
| `--top-k` | integer, 1 to 200 | `10` |  | Number of top changed projections to report |
| `--format` | text | `table` |  | Output format: table \| json \| markdown |
| `--output`, `-o` | text |  |  | Write report to file (json/markdown only) |

### `soup adapters info`

Show detailed metadata for a LoRA adapter.

Usage: `soup adapters info ADAPTER`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `ADAPTER` | text |  | yes | Path to adapter directory |

### `soup adapters list`

Scan a directory for LoRA adapters and list them.

Usage: `soup adapters list [DIRECTORY]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `DIRECTORY` | text | `.` |  | Directory to scan for adapters (recursive) |

### `soup adapters merge`

Merge LoRA adapters via linear / ties / dare / svd (v0.57.0, v0.60.0 license gate).

Usage: `soup adapters merge [OPTIONS] ADAPTERS...`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `ADAPTERS...` | text, any number |  | yes | Two or more adapter paths to merge |
| `--output`, `-o` | text |  | yes | Output directory for merged adapter |
| `--strategy` | text | `linear` |  | linear \| ties \| dare \| svd \| cmaes |
| `--weights` | text |  |  | Comma-separated weights (default: equal) |
| `--density` | float | `0.2` |  | Trim density for ties/dare in (0, 1] |
| `--seed` | integer | `0` |  | Random seed for dare / cmaes |
| `--rank` | integer |  |  | SVD rank (svd strategy only) |
| `--eval` | text |  |  | Path to eval suite (required for --strategy cmaes). Used by the evolutionary loop to score candidate merges. |
| `--budget` | text | `1h` |  | Wall-clock budget for cmaes (60s..24h, e.g. 1h, 30m) |
| `--population` | integer, 2 to 256 | `8` |  | cmaes population size per generation |
| `--max-generations` | integer, 1 to 10000 | `20` |  | cmaes generation cap |
| `--license` | text, repeatable |  |  | SPDX license id per adapter (repeatable; same order as inputs). v0.60.0 refuses merges with conflicting licenses unless --license-override &lt;reason> is passed. |
| `--license-override` | text |  |  | Free-text justification (>=8 chars) to merge across a license conflict. Logged to the audit log for legal review (v0.71.2). |
| `--allow-unscanned` | flag | off |  | Skip the v0.71.2 backdoor-scan gate. By default `adapters merge` refuses to merge an input whose `adapters scan` returns FAIL (or that cannot be scanned). |
| `--canary` | text |  |  | Canary-suite JSON to compute a live OK/MINOR/MAJOR verdict for the merged adapter vs the first input (v0.71.4 #172). Shape: {"baseline_scores": \[...], "candidate_scores": \[...]}. |
| `--strict-verdict` | flag | off |  | Exit 2 when the canary verdict is MAJOR (CI gate). |

### `soup adapters pr`

Render a GitHub-style PR for an adapter.

Usage: `soup adapters pr [OPTIONS] TITLE`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `TITLE` | text |  | yes | Short PR title (e.g. 'add-support-tone') |
| `--base-sha` | text |  | yes | 64-hex SHA of the base model |
| `--adapter` | text |  | yes | Path to candidate adapter |
| `--eval` | text |  |  | Path to JSON eval-deltas: list of {"metric": ..., "baseline": ..., "candidate": ...} |
| `--samples` | text |  |  | Path to JSON sample diffs: list of {"prompt": ..., "baseline_output": ..., "candidate_output": ...} |
| `--dataset-diff` | text |  |  | Path to a text file containing the dataset diff |
| `--output`, `-o` | text |  |  | Write rendered Markdown to path (default: stdout) |
| `--format`, `-f` | text | `markdown` |  | markdown \| json |
| `--push` | text |  |  | Post the rendered Markdown as a comment on a GitHub PR: owner/repo#N (e.g. MakazhanAlpamys/Soup#42). Auth via GITHUB_TOKEN / GH_TOKEN (v0.71.4). |

### `soup adapters scan`

Spectral backdoor scan over LoRA adapter weights (v0.60.0).

Usage: `soup adapters scan [OPTIONS] ADAPTER`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `ADAPTER` | text |  | yes | Path to adapter directory |
| `--format` | text | `text` |  | Output format: text \| json |

### `soup adapters sign`

Compute manifest + write ``.soup-signature.json`` (v0.60.0, ed25519 v0.71.2).

Usage: `soup adapters sign [OPTIONS] ADAPTER`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `ADAPTER` | text |  | yes | Path to adapter directory |
| `--backend` | text | `unsigned` |  | Signing backend: unsigned \| ed25519 \| sigstore (keyless OIDC) |
| `--key` | text |  |  | ed25519 private-key PEM path (or set SOUP_SIGNING_KEY). |
| `--generate-key` | text |  |  | Generate a fresh ed25519 keypair, persist the private key here (PEM, 0600), and sign with it. |
| `--interactive-oidc` | flag | off |  | Allow Sigstore browser OIDC when no ambient credential exists. Off by default so headless runners fail instead of hanging. |

### `soup adapters verify`

Verify ``.soup-signature.json`` against current files (v0.60.0, ed25519 v0.71.2).

Usage: `soup adapters verify [OPTIONS] ADAPTER`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `ADAPTER` | text |  | yes | Path to adapter directory |
| `--strict` | flag | off |  | Exit 3 on any verification failure (CI-friendly) |
| `--public-key` | text |  |  | Trusted ed25519 public-key PEM. When set, the embedded signing key must match it and requires an ed25519 signature (genuine authentication, not just consistency). |
| `--cert-identity` | text |  |  | Trusted Sigstore certificate identity (SAN). Requires --cert-oidc-issuer and is required for sigstore-signed adapters. |
| `--cert-oidc-issuer` | text |  |  | Trusted OIDC issuer. Required together with --cert-identity. |

## `soup advise`

Pre-flight decision: PROMPT_ENG / RAG / SFT / DPO / GRPO. Run BEFORE you spend 8 hours on a GPU (v0.54.0).

Usage: `soup advise <command> [ARGS]...`

Subcommands:

- [`compare`](#soup-advise-compare)
- [`explain`](#soup-advise-explain)
- [`run`](#soup-advise-run)

### `soup advise compare`

Show prior verdicts from advise_history.jsonl.

Usage: `soup advise compare [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--limit` | integer, 1 to 1000 | `20` |  | How many recent verdicts to show. |

### `soup advise explain`

Print the rubric, weights, and evidence trail of the last verdict.

Usage: `soup advise explain`

Takes no options or arguments.

### `soup advise run`

Render a Verdict for the supplied dataset.

Usage: `soup advise run [OPTIONS] DATA`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `DATA` | text |  | yes | Path to JSONL dataset to advise on (must stay under cwd). |
| `--goal`, `-g` | text |  |  | One-line goal statement: e.g. "make our chatbot more concise". Sharpens task classification. |
| `--probe` | flag | off |  | Also run an ROI probe (zero/few-shot + RAG baseline + N-step LoRA). Heuristic by default; pass --probe-model &lt;id> for a LIVE probe that loads the model + LoRA-trains on a tiny held-out slice. |
| `--probe-model` | text |  |  | Base model id for a LIVE probe (e.g. HuggingFaceTB/SmolLM2-135M). When set, the probe loads this model, scores zero/few-shot, LoRA-trains, and measures base-model proximity. Implies --probe. |
| `--probe-device` | text |  |  | Device for the live probe (cuda / cpu). Auto-detected when omitted. |
| `--record` | flag | off |  | Append this verdict to advise_history.jsonl with accepted=True. Omit --record to skip. Stored under \~/.soup/. |
| `--notes` | text | `""` |  | Optional notes attached to the recorded verdict (&lt;=4096 chars). |

## `soup agent`

Agent Forge: spec -> tool-calling dataset / train / eval (v0.46.0).

Usage: `soup agent <command> [ARGS]...`

Subcommands:

- [`eval`](#soup-agent-eval)
- [`synth`](#soup-agent-synth)
- [`train`](#soup-agent-train)

### `soup agent eval`

Score predicted tool-calls against the spec's tool catalog.

Usage: `soup agent eval [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--spec`, `-s` | text |  | yes | API spec (under cwd). |
| `--predictions`, `-p` | text |  | yes | JSONL of model outputs, one per line, with at minimum {tool: &lt;name>, arguments: {...}}. |
| `--sandbox` | flag | off |  | Execute each heuristic-passing prediction against a sandboxed mock of the endpoint (v0.71.18 #110): classify ok / tool_error / timeout / arg_error. Strong isolation is POSIX-only (reduced isolation on Windows; a friendly advisory is printed). |

### `soup agent synth`

Parse an API spec and synthesise a tool-calling SFT dataset.

Usage: `soup agent synth [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--spec`, `-s` | text |  | yes | Path to OpenAPI / MCP / GraphQL spec (YAML or JSON, under cwd). |
| `--output`, `-o` | text | `agent_dataset.jsonl` |  | Where to write the synthesised JSONL dataset (under cwd). |
| `--kind`, `-k` | text |  |  | Spec kind override: openapi \| mcp \| graphql. Auto-detected if omitted. |
| `--examples-per-endpoint`, `-n` | integer, 1 to 32 | `1` |  | Number of synthetic rows to emit per endpoint. |

### `soup agent train`

One-shot wrapper: synth + planned soup train invocation (printed).

Usage: `soup agent train [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--spec`, `-s` | text |  | yes | API spec (under cwd). |
| `--base`, `-b` | text |  | yes | Base model HF repo id to fine-tune. |
| `--dataset-out` | text | `agent_dataset.jsonl` |  | Where the synth step writes its dataset (under cwd). |
| `--output-dir` | text | `./agent_train_output` |  | Where the planned soup train run will store checkpoints. |
| `--examples-per-endpoint`, `-n` | integer, 1 to 32 | `4` |  | Number of synthetic examples to create per API endpoint. |

## `soup airgap-bundle`

Build a signed tarball with model + datasets + wheels + kernels (v0.60.0).

Usage: `soup airgap-bundle [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--output`, `-o` | text |  | yes | Output tarball path (cwd-contained) |
| `--model` | text |  | yes | Path to model directory |
| `--dataset` | text, repeatable |  |  | Dataset directory (repeatable) |
| `--wheel` | text, repeatable |  |  | Wheel directory (repeatable) |
| `--kernel` | text, repeatable |  |  | CUDA / kernel directory (repeatable) |
| `--bundle-size-cap` | float | `100.0` |  | Cap in GiB (default 100). Build aborts when exceeded. |
| `--repro-receipt` | text |  |  | Embed a reproducibility receipt JSON (from `soup train --repro-receipt`) into the bundle. Auto-detected from &lt;model>/repro-receipt.json when not given. v0.71.3 #188. |

## `soup apple-adapter`

Convert / sign HF / MLX / Apple FoundationModels adapters (v0.68.0).

Usage: `soup apple-adapter [OPTIONS] SOURCE`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `SOURCE` | text |  | yes | Source adapter directory |
| `--direction` | text | `hf-to-mlx` |  | Allowed: hf-to-apple, hf-to-mlx, mlx-to-apple, mlx-to-hf |
| `--output`, `-o` | text |  | yes | Output directory |
| `--sign` / `--no-sign` | flag | off |  | Sign converted adapter via v0.60 Merkle-root signing. |
| `--plan-only` | flag | off |  | Render plan + exit 0 (no live conversion). |

## `soup apply`

Execute the planned training run, refusing on drift between soup.yaml and soup.tfstate (v0.64.0).

Usage: `soup apply [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--config`, `-c` | text | `soup.yaml` |  | Path to soup.yaml. |
| `--state` | text | `soup.tfstate` |  | Path to the state file written by `soup plan`. |
| `--dry-run` | flag | off |  | Verify drift and exit without invoking training. |

## `soup attest`

In-toto + SLSA-3 attestations per Soup Can stage (v0.59.0).

Usage: `soup attest <command> [ARGS]...`

Subcommands:

- [`emit`](#soup-attest-emit)
- [`verify`](#soup-attest-verify)

### `soup attest emit`

Emit a per-stage in-toto/SLSA-3 attestation.

Usage: `soup attest emit [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--stage` | text |  | yes | Stage: extract / train / eval / export / publish. |
| `--subject` | text |  | yes | Artefact name. |
| `--sha` | text |  | yes | 64-hex SHA-256 of the artefact. |
| `--builder` | text | `soup-cli` |  | Builder identity (default: soup-cli). |
| `--invocation` | text |  |  | Free-form invocation marker (e.g. command line). |
| `--sign` | text | `unsigned` |  | Signature backend: unsigned (default) \| ed25519 \| sigstore (keyless OIDC + Fulcio/Rekor). |
| `--key` | text |  |  | ed25519 private-key PEM path (or set SOUP_SIGNING_KEY). |
| `--output`, `-o` | text |  |  | Output file path (cwd-contained). |
| `--attach-to-registry` | text |  |  | Attach the emitted attestation statement to a registry entry id (needs --output). |
| `--interactive-oidc` | flag | off |  | Allow Sigstore browser OIDC when no ambient credential exists. Off by default so headless runners fail instead of hanging. |

### `soup attest verify`

Verify an ed25519 or Sigstore-signed attestation.

Usage: `soup attest verify [OPTIONS] STATEMENT`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `STATEMENT` | text |  | yes | Path to the in-toto Statement JSON. |
| `--signature`, `-s` | text |  | yes | Path to the .sig JSON sidecar written by `attest emit --sign`. |
| `--public-key` | text |  |  | Trusted ed25519 public-key PEM. When set, the embedded key must match it (genuine authentication). |
| `--cert-identity` | text |  |  | Trusted Sigstore certificate identity (SAN). Requires --cert-oidc-issuer and is required for sigstore-signed attestations. |
| `--cert-oidc-issuer` | text |  |  | Trusted OIDC issuer. Required together with --cert-identity. |

## `soup audit-log`

HIPAA/SOC2-shaped JSONL audit log: tail + rotate (v0.59.0).

Usage: `soup audit-log <command> [ARGS]...`

Subcommands:

- [`rotate`](#soup-audit-log-rotate)
- [`tail`](#soup-audit-log-tail)

### `soup audit-log rotate`

Force a rotation pass at the current cap.

Usage: `soup audit-log rotate [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--path` | text |  |  | Override audit log path. |
| `--cap-mb` | integer, 1 to 10000 | `100` |  | Rotation cap in MiB. |

### `soup audit-log tail`

Show the most recent audit records.

Usage: `soup audit-log tail [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--limit` | integer | `50` |  | Max records to show (1-100000). |
| `--path` | text |  |  | Override audit log path (default: \$SOUP_AUDIT_LOG_PATH or \~/.soup/audit.jsonl). |
| `--json` | flag | off |  | Emit raw JSONL instead of a table. |

## `soup autopilot`

Autopilot: give model+data+goal, Soup picks optimal hyperparameters.

Usage: `soup autopilot [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--model`, `-m` | text |  |  | Base model (HF model id) |
| `--data`, `-d` | text |  |  | Dataset path (JSONL) |
| `--goal`, `-g` | text |  |  | Goal: chat \| reasoning \| code \| classification \| tool-calling \| alignment \| domain-adapt |
| `--gpu-budget` | text | `""` |  | VRAM budget, e.g. 24GB (default: auto-detect) |
| `--output`, `-o` | text | `soup.yaml` |  | Output config path |
| `--dry-run` | flag | off |  | Show decisions but don't write the config |
| `--yes`, `-y` | flag | off |  | Skip confirmation prompts |

## `soup bench`

Benchmark a model: inference speed (infer, the default) or training (train).

Usage: `soup bench <command> [ARGS]...`

Subcommands:

- [`infer`](#soup-bench-infer)
- [`train`](#soup-bench-train)

### `soup bench infer`

Run an inference benchmark (speed and memory) on a loaded model.

Usage: `soup bench infer [OPTIONS] MODEL`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `MODEL` | text |  | yes | Path to model (LoRA adapter or full model) to benchmark |
| `--base`, `-b` | text |  |  | Base model for LoRA adapter (auto-detected if not set) |
| `--max-tokens` | integer | `128` |  | Maximum tokens to generate per prompt |
| `--num-prompts`, `-n` | integer | `3` |  | Number of prompts to run for averaging (ignored when --prompts-file is set) |
| `--prompts-file` | text |  |  | Path to custom prompts file (.txt or .jsonl) |
| `--p50` | flag | off |  | Print p50 (median) per-prompt latency. v0.53.9 #26. |
| `--p95` | flag | off |  | Print p95 per-prompt latency. v0.53.9 #26. |
| `--backend` | text | `auto` |  | Inference backend hint: auto (default) \| transformers \| mlx. v0.53.9 #28. |
| `--cuda-graphs` | flag | off |  | Experimental CUDA graph decode for resident, unquantized Qwen2/Llama models. |

### `soup bench train`

Train for a fixed number of steps and report timing, memory and tokens -- failing when the model was not actually training.

Usage: `soup bench train [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--config`, `-c` | text | `soup.yaml` |  | Config to train |
| `--steps` | integer, at least 1 | `20` |  | Optimizer steps to run |
| `--warmup` | integer, at least 0 | `3` |  | Leading steps measured but left out of the timing |
| `--output`, `-o` | text | `bench-train.json` |  | Where to write the JSON report |

## `soup bom`

CycloneDX ML-BOM + SPDX AI bill-of-materials emitter (v0.59.0).

Usage: `soup bom <command> [ARGS]...`

Subcommands:

- [`emit`](#soup-bom-emit)

### `soup bom emit`

Emit a CycloneDX + SPDX BOM from CLI-supplied SHAs.

Usage: `soup bom emit [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--name` | text |  | yes | Model / adapter name. |
| `--version` | text | `0.1.0` |  | Model version string. |
| `--base-model` | text |  | yes | HF repo id of the base model. |
| `--base-sha` | text |  | yes | SHA-256 of the base model. |
| `--config-sha` | text |  | yes | SHA-256 of the resolved soup.yaml config. |
| `--data-sha` | text |  |  | SHA-256 of the training dataset. |
| `--task` | text | `sft` |  | Training task (sft / dpo / grpo / ...). |
| `--license` | text |  |  | License: an SPDX id (e.g. Apache-2.0, MIT; case-insensitive), an SPDX expression (Apache-2.0 OR MIT), or any other name. |
| `--format`, `-f` | text | `cyclonedx` |  | Output BOM format: cyclonedx \| spdx \| both. |
| `--output`, `-o` | text |  |  | Output file path (cwd-contained). When --format=both, this is the prefix and Soup writes &lt;prefix>.cdx.json + &lt;prefix>.spdx.json. |
| `--energy`, `-e` | text |  |  | Path to energy measurement JSON. |
| `--attach-to-registry` | text |  |  | Attach the emitted BOM file(s) to a registry entry id (needs --output). |

## `soup build`

dbt-for-SFT DAG: validate, plan, and materialise dataset transforms.

Usage: `soup build [OPTIONS] CONFIG`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `CONFIG` | text |  | yes | Path to build manifest (YAML) |
| `--output-dir`, `-o` | text | `build_out` |  | Directory (under cwd) for materialised model JSONL + state store. |
| `--dry-run` | flag | off |  | Validate the manifest + print the topological plan; do not execute. |

## `soup can`

Soup Cans: pack/inspect/verify/fork shareable .can artifacts.

Usage: `soup can <command> [ARGS]...`

Subcommands:

- [`fork`](#soup-can-fork)
- [`inspect`](#soup-can-inspect)
- [`pack`](#soup-can-pack)
- [`publish`](#soup-can-publish)
- [`run`](#soup-can-run)
- [`verify`](#soup-can-verify)

### `soup can fork`

Fork a can with config modifications and re-pack.

Usage: `soup can fork [OPTIONS] SOURCE`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `SOURCE` | text |  | yes | Source .can path |
| `--out`, `-o` | text |  | yes | Output forked .can path |
| `--modify` | text, repeatable |  |  | Modification like 'training.lr=5e-5' (repeatable) |
| `--author` | text | `unknown` |  | Author name recorded in the forked manifest. |

### `soup can inspect`

Preview a .can file's manifest without extracting.

Usage: `soup can inspect PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `PATH` | text |  | yes | Path to .can file |

### `soup can pack`

Pack a registry entry into a shareable .can file.

Usage: `soup can pack [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--entry-id` | text |  | yes | Registry entry id / prefix / name:tag |
| `--out`, `-o` | text |  | yes | Output .can path |
| `--author` | text | `unknown` |  | Author handle |
| `--description` | text | `""` |  | Free-form description |
| `--attest` | text, repeatable |  |  | Embed an in-toto Statement JSON into the can manifest (v3, repeatable). Each file is shape-validated. v0.71.3. |

### `soup can publish`

Publish a .can file to HF Hub as a dataset.

Usage: `soup can publish [OPTIONS] CAN_PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `CAN_PATH` | text |  | yes | Path to .can file |
| `--hf-hub` | text |  | yes | HF Hub dataset repo (e.g. 'me/my-recipe-can') |
| `--private` | flag | off |  | Create a private HF Hub dataset repository. |
| `--message`, `-m` | text |  |  | Commit message (first line only, capped at 200 chars) |

### `soup can run`

Run a can end-to-end: extract -> train (-> optional deploy).

Usage: `soup can run [OPTIONS] CAN_PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `CAN_PATH` | text |  | yes | Path to .can file |
| `--yes`, `-y` | flag | off |  | Skip the security confirmation panel (auto-trains, auto-fetches data) |
| `--deploy` | flag | off |  | Run the can's deploy_targets after a successful train |
| `--extract-dir` | text |  |  | Where to extract (default: fresh tmp dir) |
| `--env-capture` | text |  |  | Write env summary (pip freeze + GPU info) to this path |

### `soup can verify`

Verify a .can file's schema and config parseability.

Usage: `soup can verify PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `PATH` | text |  | yes | Path to .can file |

## `soup card`

Generate a HuggingFace model card from a registry entry.

Usage: `soup card [OPTIONS] REF`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `REF` | text |  | yes | Registry entry id / prefix / name:tag |
| `-o`, `--output` | text | `MODELCARD.md` |  | Output markdown path (must stay under the current directory) |

## `soup chat`

Chat with a fine-tuned model in the terminal.

Usage: `soup chat [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--model`, `-m` | text |  | yes | Path to LoRA adapter directory or full model |
| `--base`, `-b` | text |  |  | Base model ID. Auto-detected from adapter_config.json if not set. |
| `--device` | text |  |  | Device: cuda, mps, cpu. Auto-detected if not set. |
| `--max-tokens` | integer | `512` |  | Maximum number of tokens to generate |
| `--temperature`, `-t` | float | `0.7` |  | Sampling temperature (0.0 = greedy, 1.0 = creative) |
| `--system`, `-s` | text |  |  | System prompt for the conversation |
| `--trust-remote-code` | flag | off |  | Allow loading models that ship custom Python via auto_map. Default deny (v0.36.0). Only enable if you trust the source. |
| `--hub` | text | `hf` |  | Source hub for the base model: hf (default) / modelscope / modelers. Non-HF hubs require the matching SDK; the base model is snapshotted to a cwd-contained cache before chat starts (v0.53.10 #152). |

## `soup ci`

Fine-tuning CI: init a PR gate workflow.

Usage: `soup ci <command> [ARGS]...`

Subcommands:

- [`init`](#soup-ci-init)

### `soup ci init`

Write a GitHub Actions fine-tuning gate workflow.

Usage: `soup ci init [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--data` | text | `./data/train.jsonl` |  | Training data path (repo-relative) |
| `--suite` | text | `expectations.yaml` |  | Expectations suite YAML (repo-relative) |
| `--evidence` | text | `ship_evidence.json` |  | Ship evidence JSON (repo-relative) |
| `--python` | text | `3.11` |  | Python version for the CI runner (e.g. 3.11) |
| `--branch` | text | `main` |  | Branch the workflow triggers on |
| `--config` | text |  |  | Bind the ship gate to a committed soup.yaml (refuses stale evidence) |
| `-o`, `--output` | text | `.github/workflows/soup-gate.yml` |  | Workflow output path |
| `--force`, `-f` | flag | off |  | Overwrite an existing workflow |

## `soup compile`

Compile a DSPy / GEPA prompt program against an eval suite (v0.68.0).

Usage: `soup compile [OPTIONS] PROGRAM`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `PROGRAM` | text |  | yes | Path to DSPy / GEPA program (.py) |
| `--eval` | text |  | yes | Path to eval-suite JSON / JSONL file |
| `--optimizer` | text | `mipro` |  | Optimizer name. Allowed: bootstrap_fewshot, copro, gepa, mipro, textgrad |
| `--max-iters` | integer | `10` |  | Maximum optimizer iterations |
| `--output`, `-o` | text | `compiled_program.py` |  | Output program path |
| `--plan-only` | flag | off |  | Render the resolved plan + exit 0 (no live compile). |

## `soup compile-tools`

Compile / optimize tool schemas + descriptions (v0.68.0).

Usage: `soup compile-tools [OPTIONS] SPEC`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `SPEC` | text |  | yes | OpenAPI / MCP / GraphQL spec path |
| `--eval` | text |  | yes | Path to tool-call eval JSONL |
| `--optimizer` | text | `textgrad` |  | Allowed: gepa, textgrad |
| `--output`, `-o` | text | `compiled_tools.json` |  | Output schema path |
| `--task-lm` | text |  |  | Task language model for GEPA optimizer |
| `--reflection-lm` | text |  |  | Reflection language model for GEPA optimizer |
| `--engine` | text |  |  | Backward engine for TextGrad optimizer (e.g. gpt-4o) |
| `--plan-only` | flag | off |  | Render plan + exit 0 (no live optimise). |

## `soup completions`

Emit a bash / zsh / fish completion script. Use with `eval "$(soup completions bash)"` (v0.64.0).

Usage: `soup completions SHELL`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `SHELL` | text |  | yes | One of: bash, zsh, fish. |

## `soup cost`

Estimate training cost in USD across cloud providers.

Usage: `soup cost [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--config`, `-c` | text | `soup.yaml` |  | Path to soup.yaml config file |
| `--gpu`, `-g` | text |  |  | Filter by specific GPU (e.g., A100, H100, RTX 4090) |
| `--json` | flag | off |  | Output as JSON for scripting |

## `soup data`

Dataset tools: inspect, convert, merge, dedup, validate, stats.

Usage: `soup data <command> [ARGS]...`

Subcommands:

- [`active-sample`](#soup-data-active-sample)
- [`augment`](#soup-data-augment)
- [`best-of-n`](#soup-data-best-of-n)
- [`brain-rot`](#soup-data-brain-rot)
- [`canary`](#soup-data-canary)
- [`clean`](#soup-data-clean)
- [`convert`](#soup-data-convert)
- [`decontaminate`](#soup-data-decontaminate)
- [`dedup`](#soup-data-dedup)
- [`demo`](#soup-data-demo)
- [`doctor`](#soup-data-doctor)
- [`download`](#soup-data-download)
- [`educational`](#soup-data-educational)
- [`evolve`](#soup-data-evolve)
- [`filter`](#soup-data-filter)
- [`forge`](#soup-data-forge)
- [`from-traces`](#soup-data-from-traces)
- [`gen-magpie`](#soup-data-gen-magpie)
- [`generate`](#soup-data-generate)
- [`ingest`](#soup-data-ingest)
- [`inspect`](#soup-data-inspect)
- [`langdetect`](#soup-data-langdetect)
- [`lint`](#soup-data-lint)
- [`merge`](#soup-data-merge)
- [`mix`](#soup-data-mix)
- [`persona-mix`](#soup-data-persona-mix)
- [`pii`](#soup-data-pii)
- [`preprocess`](#soup-data-preprocess)
- [`preview`](#soup-data-preview)
- [`push`](#soup-data-push)
- [`recipe`](#soup-data-recipe)
- [`register`](#soup-data-register)
- [`registry`](#soup-data-registry)
- [`review`](#soup-data-review)
- [`sample`](#soup-data-sample)
- [`score`](#soup-data-score)
- [`search`](#soup-data-search)
- [`split`](#soup-data-split)
- [`stats`](#soup-data-stats)
- [`topics`](#soup-data-topics)
- [`toxicity`](#soup-data-toxicity)
- [`unregister`](#soup-data-unregister)
- [`validate`](#soup-data-validate)

### `soup data active-sample`

Surface the most uncertain prod traces for human review.

Usage: `soup data active-sample [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--input`, `-i` | text |  | yes | JSONL of trace rows with rm_score or rm_scores fields. |
| `--output`, `-o` | text | `active_samples.jsonl` |  | Output JSONL of the top-uncertainty rows (default: active_samples.jsonl). |
| `--budget` | integer | `100` |  | Max rows to surface for human review (1 - 100_000). |
| `--slack-url` | text |  |  | Optional Slack webhook URL — POSTed on completion. SSRF-validated. |
| `--discord-url` | text |  |  | Optional Discord webhook URL — POSTed on completion. SSRF-validated. |

### `soup data augment`

Augment a dataset via LLM (rephrase / translate / style).

Usage: `soup data augment [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--input`, `-i` | text |  | yes | Source JSONL file |
| `--output`, `-o` | text | `augmented.jsonl` |  | Output JSONL path |
| `--strategy`, `-s` | text | `rephrase` |  | Augmentation strategy: rephrase, translate, style |
| `--provider`, `-p` | text | `ollama` |  | LLM provider: ollama, anthropic, vllm |
| `--model` | text | `""` |  | Provider model id (default per provider) |
| `--base-url` | text | `""` |  | Provider base URL (ollama/vllm; defaults to loopback) |
| `--count`, `-c` | integer, 1 to 10 | `2` |  | Augmentation multiplier (1-10) |
| `--lang` | text | `""` |  | Comma-separated target languages for translate |
| `--styles` | text | `""` |  | Comma-separated styles for style strategy |
| `--requests-per-minute` | integer, 1 to 600 | `60` |  | Rate limit for provider requests |
| `--dedup` | flag | off |  | Deduplicate augmented + original data |

### `soup data best-of-n`

Best-of-N rejection sampling: sample N per prompt, a judge picks the winner.

Usage: `soup data best-of-n [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--base` | text | `""` |  | Local base model id/path to sample from |
| `--prompts` | text | `""` |  | JSONL of {prompt\|messages} rows |
| `--judge` | text | `""` |  | Judge URL (ollama://\|https://\|http://localhost) |
| `--provider` | text | `""` |  | Provider sampler: ollama \| vllm |
| `--model` | text | `""` |  | Provider sampler model id |
| `--base-url` | text | `""` |  | Provider sampler base URL |
| `--n` | integer | `8` |  | Candidates per prompt \[2, 64] |
| `--output`, `-o` | text | `""` |  | Output SFT JSONL (required unless --plan-only) |
| `--emit-pairs` | text | `""` |  | Also write winner/loser DPO pairs to this JSONL |
| `--checkpoint` | text | `""` |  | Recovery journal (default: &lt;output-or-artifact>.checkpoint.jsonl) |
| `--manifest` | text | `""` |  | Commit manifest (default: &lt;output>.manifest.json) |
| `--resume` | flag | off |  | Resume a matching checkpoint |
| `--export-candidates` | text | `""` |  | Sampling-only candidate artifact JSONL |
| `--candidate-artifact` | text | `""` |  | Offline candidate artifact to materialize |
| `--judgments` | text | `""` |  | Offline verified judgments JSONL |
| `--temperature` | float | `1.0` |  | Sampling temp \[0, 2] |
| `--max-new-tokens` | integer | `256` |  | Max new tokens per candidate \[1, 4096] |
| `--device` | text | `""` |  | Local sampler device: cuda \| cpu |
| `--seed` | integer | `0` |  | Local sampling seed |
| `--revision` | text | `""` |  | Pinned local model revision |
| `--trust-remote-code` | flag | off |  | Allow custom model code on local --base |
| `--plan-only` | flag | off |  | Validate + print plan |

### `soup data brain-rot`

Score a dataset for brain-rot per arXiv 2510.13928.

Usage: `soup data brain-rot [OPTIONS] DATA`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `DATA` | text |  | yes | Path to JSONL dataset |
| `--strict` | flag | off |  | Exit 3 on MAJOR verdict (CI gate mode) |
| `--max-major-fraction` | float | `0.25` |  | Strict-mode MAJOR-row fraction cap \[0, 1] |
| `--lang` | text | `en` |  | Per-language heuristic bundle: en \| es \| fr \| de \| ru \| auto. Closes issue #234. |

### `soup data canary`

Dataset canaries (memorization probe).

Usage: `soup data canary <command> [ARGS]...`

Subcommands:

- [`check`](#soup-data-canary-check)
- [`insert`](#soup-data-canary-insert)

#### `soup data canary check`

Check a model for memorization of the manifest's canaries.

Usage: `soup data canary check [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--manifest` | text |  | yes | Canary manifest |
| `--base` | text |  | yes | Base model id or path |
| `--adapter` | text |  |  | LoRA adapter to check |
| `--controls` | integer | `128` |  | Never-inserted controls to rank against |
| `--device` | text | `auto` |  | auto/cpu/cuda |
| `--output`, `-o` | text |  |  | Write the report as JSON |

#### `soup data canary insert`

Insert K unique canaries into a dataset and record them.

Usage: `soup data canary insert [OPTIONS] PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `PATH` | text |  | yes | Path to dataset file |
| `--output`, `-o` | text |  | yes | Output file: .jsonl, or .json for a JSON array |
| `--manifest` | text |  | yes | Where to write the canary manifest (CONTAINS THE SECRETS) |
| `--count`, `-k` | integer | `16` |  | Number of canaries |
| `--seed` | integer | `0` |  | Seed for the canaries and the rows they go to |
| `--format` | text | `auto` |  | Dataset format to write the canaries in: auto (detect), alpaca, sharegpt, chatml |

### `soup data clean`

Clean and repair a fine-tuning dataset: control characters, whitespace, format sanity.

Usage: `soup data clean [OPTIONS] PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `PATH` | text |  | yes | Path to input dataset file (JSONL) |
| `--output`, `-o` | text |  |  | Path to write cleaned dataset (default: &lt;input>\_cleaned.jsonl) |
| `--force`, `-f` | flag | off |  | Overwrite output file if it already exists |
| `--dry-run` | flag | off |  | Preview modifications without writing output file |
| `--min-tokens` | integer | `1` |  | Minimum character/token length for assistant turns (shorter turns are dropped) |
| `--prune-echo` / `--no-prune-echo` | flag | off |  | Prune assistant turns that merely echo the user prompt (opt-in) |
| `--strip-boilerplate` / `--no-strip-boilerplate` | flag | off |  | Strip canned AI disclaimers ('As an AI...', 'Certainly!...') |
| `--repair-code` / `--no-repair-code` | flag | off |  | Auto-close unclosed markdown \`\`\` code fences in assistant completions |
| `--repair-json` / `--no-repair-json` | flag | off |  | Repair trailing commas and markdown fences in tool-call arguments |
| `--drop-invalid-json` | flag | off |  | Drop tool-calling rows with unrepairable JSON syntax errors |
| `--json` | flag | off |  | Output machine-readable JSON summary |

### `soup data convert`

Convert a dataset between formats (alpaca, sharegpt, chatml).

Usage: `soup data convert [OPTIONS] PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `PATH` | text |  | yes | Input dataset file |
| `--to`, `-t` | text |  | yes | Target format: alpaca, sharegpt, chatml |
| `--output`, `-o` | text |  |  | Output file path (default: &lt;input>\_&lt;format>.jsonl) |

### `soup data decontaminate`

Drop rows that overlap public benchmarks (n-gram heuristic).

Usage: `soup data decontaminate [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--input`, `-i` | text |  | yes | Input JSONL file to decontaminate. |
| `--benchmarks`, `-b` | text |  |  | Comma-separated benchmark names (label only; corpora not bundled). |
| `--benchmark-file` | text |  |  | v0.53.7: operator-supplied benchmark JSONL (under cwd). Each row's text, prompt, Alpaca, preference or conversation fields contribute to the n-gram contamination set. Compatible with --benchmarks (extends). |
| `--output`, `-o` | text | `clean.jsonl` |  | Output JSONL of rows that survived decontamination. |
| `--threshold` | float, 0.0 to 1.0 | `0.8` |  | Minimum n-gram overlap ratio for removing a row. |
| `--n` | integer, 1 to 32 | `8` |  | N-gram size used for overlap detection. |

### `soup data dedup`

Remove near-duplicate rows: MinHash (default) or embeddings (--semantic).

Usage: `soup data dedup [OPTIONS] PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `PATH` | text |  | yes | Path to dataset file |
| `--output`, `-o` | text |  |  | Output file path (default: &lt;input>\_deduped.jsonl) |
| `--threshold` | float | `0.8` |  | Similarity threshold (0.0-1.0): MinHash Jaccard by default, embedding cosine under --semantic. |
| `--field`, `-f` | text |  |  | Field to hash/embed (default: all text fields concatenated) |
| `--semantic` | flag | off |  | Use embedding cosine (SemDeDup) instead of MinHash. Catches paraphrases MinHash misses. Requires soup-cli\[train]. |
| `--embed-model` | text | `sentence-transformers/all-MiniLM-L6-v2` |  | Embedding model used by --semantic. |
| `--device` | text | `auto` |  | Device for --semantic (auto/cpu/cuda). |

### `soup data demo`

List or copy a bundled demo dataset (v0.43.0).

Usage: `soup data demo [OPTIONS] [NAME]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `NAME` | text |  |  | Bundle name (alpaca_demo / sharegpt_demo / dpo_demo / grpo_demo). Omit to list available bundles. |
| `--output`, `-o` | text |  |  | Destination JSONL path (defaults to ./&lt;name>.jsonl). |

### `soup data doctor`

Chat-template compatibility report + loss-mask X-ray.

Usage: `soup data doctor [OPTIONS] PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `PATH` | text |  | yes | Path to dataset file (jsonl/json/csv/parquet). |
| `--model`, `-m` | text |  | yes | Tokenizer id or local path to check chat-template compatibility against. |
| `--format`, `-f` | text | `auto` |  | Dataset format (auto-detects by default). |
| `--max-length` | integer | `2048` |  | Training max_length — used for the truncation-risk check. |
| `--sample` | integer | `200` |  | Number of rows to sample for the checks. |
| `--show-mask` | integer |  |  | Render N sample rows with per-token trained/masked colouring (the real collator path). |
| `--train-on-responses-only` / `--no-train-on-responses-only` | flag | on |  | Mask non-assistant tokens (mirrors data.train_on_responses_only). |
| `--train-on-messages-with-train-field` | flag | off |  | Per-message train:bool field masking (mirrors the same soup.yaml flag). |
| `--mask-history` / `--no-mask-history` | flag | off |  | Train only the LAST assistant turn (mirrors data.mask_history). |
| `--train-on-eot` | flag | off |  | Extend the trained span through the trailing EOS/EOT token. |
| `--trust-remote-code` | flag | off |  | Allow loading a tokenizer that ships custom code. |
| `--output`, `-o` | text |  |  | Write the report JSON to this path. |

### `soup data download`

Download a HuggingFace dataset and save as JSONL.

Usage: `soup data download [OPTIONS] DATASET_ID`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `DATASET_ID` | text |  | yes | HuggingFace dataset ID (e.g. teknium/OpenHermes-2.5) |
| `--output`, `-o` | text |  |  | Output file path (default: &lt;dataset-name>.jsonl) |
| `--split` | text | `train` |  | Dataset split to download (e.g. train, test, train\[:1000]) |
| `--samples`, `-n` | integer |  |  | Max number of samples to download (streams, no full download) |
| `--format`, `-f` | text |  |  | Convert to Soup format after download: alpaca, sharegpt, chatml |
| `--trust-remote-code` | flag | off |  | Allow datasets that ship custom Python loaders. Default deny (v0.36.0). Only enable if you trust the source. |
| `--hub` | text | `hf` |  | Source hub: hf (default) / modelscope / modelers. Non-HF hubs require the matching SDK (v0.53.8 #130). |

### `soup data educational`

Tag each row with an educational-value score \[0, 1].

Usage: `soup data educational [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--input`, `-i` | text |  | yes | Input JSONL file to score. |
| `--output`, `-o` | text | `educational.jsonl` |  | Output JSONL file with educational-value scores. |

### `soup data evolve`

Evol-Instruct: deepen (depth) or diversify (breadth) seed instructions.

Usage: `soup data evolve [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--input` | text |  | yes | Seed JSONL ({prompt\|instruction\|messages} per row) |
| `--provider` | text |  | yes | ollama \| vllm |
| `--model` | text |  | yes | Provider model id |
| `--strategy` | text | `depth` |  | depth \| breadth |
| `--rounds` | integer | `1` |  | Evolution rounds \[1, 5] |
| `--output`, `-o` | text | `""` |  | Output JSONL (required unless --plan-only) |
| `--base-url` | text | `""` |  | Provider base URL |
| `--temperature` | float | `1.0` |  | Sampling temp \[0, 2] |
| `--max-tokens` | integer | `512` |  | Max tokens \[1, 16384] |
| `--plan-only` | flag | off |  | Validate + print plan |

### `soup data filter`

Filter dataset by quality: perplexity and/or coherence scoring.

Usage: `soup data filter [OPTIONS] PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `PATH` | text |  | yes | Path to dataset file |
| `--output`, `-o` | text |  |  | Output file path (default: &lt;input>\_filtered.jsonl) |
| `--perplexity`, `--ppl` | float |  |  | Max perplexity threshold (rows above this are removed) |
| `--coherence`, `--min-coherence` | float |  |  | Min coherence threshold 0.0-1.0 (rows below this are removed) |
| `--ppl-model` | text | `gpt2` |  | Model for perplexity scoring (default: gpt2) |
| `--field`, `-f` | text |  |  | Field to score (default: all text fields concatenated) |
| `--score-only` | flag | off |  | Add scores to data without filtering (writes \_scored.jsonl) |

### `soup data forge`

Run the multi-stage synthetic data pipeline with provenance.

Usage: `soup data forge [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--docs`, `-d` | text |  | yes | Directory of documents (txt/md/json/jsonl, under cwd). |
| `--task`, `-t` | text | `sft` |  | Forge task: sft \| preference \| tool. |
| `--target-rows`, `-r` | integer, 1 to 1000000 | `100` |  | Maximum number of synthesised rows. |
| `--teacher` | text | `local-judge` |  | Label recorded in provenance for the judge backend. |
| `--output`, `-o` | text | `forge_dataset.jsonl` |  | JSONL output path (under cwd). |
| `--provenance` | text | `forge_provenance.json` |  | Provenance manifest output path (under cwd). |
| `--uncertainty-threshold` | float, 0.0 to 1.0 | `0.0` |  | Minimum Jaccard-distance score required to keep a synthesised row. |
| `--max-chunk-chars` | integer, 1 to 64000 | `1000` |  | Maximum chars per document chunk before judge call. |
| `--judge-provider` | text |  |  | v0.53.7 #111: live judge backend. One of ollama / anthropic / vllm. Omit for the deterministic offline stub. |
| `--judge-model` | text | `llama3.1` |  | Model name for the live judge provider (default llama3.1). |
| `--judge-base-url` | text |  |  | Override base URL for Ollama (localhost-only) / vLLM (scheme allowlist + loopback). Ignored for Anthropic. |
| `--hub` | text | `hf` |  | Teacher hub: hf (default) / modelscope / modelers. When non-HF and --teacher is a repo id (owner/name), the teacher is pre-fetched from that hub (v0.71.5 #157). |

### `soup data from-traces`

Harvest preference pairs from production traces.

Usage: `soup data from-traces [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--logs` | text |  | yes | Trace source: a JSONL file for langchain/openai, or the directory `soup serve --trace-log` writes for soup-serve (a single file is refused) |
| `--format` | text |  | yes | Trace format: langchain \| openai \| soup-serve |
| `--signal` | text | `thumbs_up` |  | Signal to extract pairs from: thumbs_up \| regenerations \| user_edit |
| `--output`, `-o` | text | `prefs.jsonl` |  | Output path for preference pairs (JSONL) |
| `--judge` | flag | off |  | Filter pairs via LLM-as-a-judge confidence (v0.40.3 #33). |
| `--judge-provider` | text | `openai` |  | Judge backend: openai \| server \| ollama. Used with --judge. |
| `--judge-model` | text | `gpt-4o-mini` |  | Judge model id (e.g. 'gpt-4o-mini', 'llama3', 'qwen2.5'). Used with --judge. |
| `--judge-api-base` | text |  |  | Judge API base URL. SSRF-protected. Used with --judge. |
| `--min-confidence` | float | `0.7` |  | Drop pairs with judge-confidence below this threshold (0.0 - 1.0). |

### `soup data gen-magpie`

Generate synthetic data from a chat-template prefix.

Usage: `soup data gen-magpie [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--base` | text |  | yes | Aligned chat-tuned base model id |
| `--provider` | text |  | yes | ollama \| anthropic \| vllm |
| `--target` | integer |  | yes | Target row count \[1, 1_000_000] |
| `--output`, `-o` | text | `""` |  | Output JSONL path (required unless --plan-only) |
| `--base-url` | text | `""` |  | Provider base URL (ollama/vllm; defaults to loopback) |
| `--max-tokens` | integer | `512` |  | Max tokens per generation \[1, 16384] |
| `--quality-filter` / `--no-quality-filter` | flag | on |  | Apply non-empty + educational heuristics (not a toxicity classifier) |
| `--plan-only` | flag | off |  | Validate + print plan; do not generate. |

### `soup data generate`

Generate synthetic training data using an LLM.

Usage: `soup data generate [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--prompt`, `-p` | text |  | yes | System prompt describing what kind of data to generate |
| `--count`, `-n` | integer | `100` |  | Number of examples to generate |
| `--output`, `-o` | text | `generated.jsonl` |  | Output file path |
| `--format`, `-f` | text | `alpaca` |  | Output format: alpaca, sharegpt, chatml |
| `--provider` | text | `openai` |  | LLM provider: openai, local, server, ollama, anthropic, vllm |
| `--model`, `-m` | text | `gpt-4o-mini` |  | Model name (OpenAI model ID, Ollama model, or local model path) |
| `--api-key` | text |  |  | \[deprecated] Use OPENAI_API_KEY env var instead. Env: `OPENAI_API_KEY`. |
| `--api-base` | text |  |  | Custom API base URL (must use HTTPS for remote APIs) |
| `--batch-size` | integer | `5` |  | Number of examples per API call |
| `--temperature`, `-t` | float | `0.8` |  | Sampling temperature for generation |
| `--dedup-with` | text |  |  | Path to existing dataset to deduplicate against |
| `--seed` | text |  |  | Path to seed examples file (JSONL) to guide generation |
| `--template` | text |  |  | Domain template: code, conversation, qa, preference, reasoning |
| `--language` | text | `Python` |  | Language for code template (Python, JavaScript, Go, Rust, Java) |
| `--task-type` | text | `function` |  | Task type for code template (function, debug, explain, refactor, test) |
| `--turns` | integer | `4` |  | Number of turns for conversation template (2-10) |
| `--topic` | text | `general knowledge` |  | Topic for conversation template |
| `--context` | text |  |  | Path to context document for QA template |
| `--pref-task` | text | `dpo` |  | Preference task format: dpo, kto, orpo |
| `--domain` | text | `math` |  | Domain for reasoning template (math, logic, code) |
| `--ollama-model` | text |  |  | Ollama model name (e.g. llama3.1). Shorthand for --provider ollama --model X |
| `--validate` | flag | off |  | Auto-validate generated data after generation |
| `--filter` | flag | off |  | Auto-filter generated data by quality (perplexity + coherence) |
| `--dedup` | flag | off |  | Auto-dedup generated data (MinHash) |
| `--quality-pipeline` | flag | off |  | Run full quality pipeline: validate + filter + dedup |
| `--requests-per-minute`, `--rpm` | integer | `60` |  | Rate limit for API requests (default: 60) |
| `--trust-remote-code` | flag | off |  | Allow loading local-provider models that ship custom Python via auto_map. Default deny (v0.36.0). |

### `soup data ingest`

Ingest a document into JSONL with one row per page / heading.

Usage: `soup data ingest [OPTIONS] FILE`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `FILE` | text |  | yes | PDF / DOCX / MD / TXT file |
| `--output`, `-o` | text | `./ingested.jsonl` |  | Output JSONL path |

### `soup data inspect`

Inspect a dataset: show stats and sample rows.

Usage: `soup data inspect [OPTIONS] PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `PATH` | text |  | yes | Path to dataset file (jsonl, csv, parquet) |
| `--rows`, `-r` | integer | `5` |  | Number of sample rows to show |
| `--image-dir` | text |  |  | Directory containing referenced images |

### `soup data langdetect`

Tag each row with a 2-letter language code (heuristic).

Usage: `soup data langdetect [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--input`, `-i` | text |  | yes | Input JSONL file to tag. |
| `--output`, `-o` | text | `langdetect.jsonl` |  | Output JSONL file with language tags. |

### `soup data lint`

Preference-data linter (dpo/orpo/simpo/ipo/bco/kto).

Usage: `soup data lint [OPTIONS] PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `PATH` | text |  | yes | Path to preference dataset file (dpo/kto). |
| `--format`, `-f` | text | `auto` |  | Dataset format: auto, dpo, kto. |
| `--model`, `-m` | text |  |  | Optional tokenizer for exact token-length bias (default: word count). |
| `--sample` | integer | `2000` |  | Number of rows to sample for the checks. |
| `--trust-remote-code` | flag | off |  | Allow loading a tokenizer that ships custom code. |
| `--output`, `-o` | text |  |  | Write the report JSON to this path. |

### `soup data merge`

Merge multiple datasets into a single file.

Usage: `soup data merge [OPTIONS] FILES...`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `FILES...` | text, any number |  | yes | Paths to dataset files to merge |
| `--output`, `-o` | text | `merged.jsonl` |  | Output file path |
| `--shuffle` | flag | off |  | Shuffle after merging |

### `soup data mix`

BETA: optimise per-dataset mixture weights against a proxy run.

Usage: `soup data mix [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--optimize` | flag | off |  | Run Bayesian search over dataset mixture weights. |
| `--apply` | text |  |  | Re-print a previously written mix-recipe (path under cwd). |
| `--datasets` | text |  |  | Comma-separated list of dataset JSONL paths (>= 2, all under cwd). |
| `--budget` | text | `1h` |  | Wall-clock cap: digits + optional s/m/h suffix (e.g. 1h, 30m). |
| `--num-probes`, `-n` | integer, 1 to 256 | `8` |  | Maximum number of proxy runs. |
| `--seed` | integer, 0 to 2147483647 | `42` |  | RNG seed for the optimiser. |
| `--output`, `-o` | text | `mix_recipe.yaml` |  | YAML recipe output path (under cwd). |
| `--overwrite` | flag | off |  | Overwrite the output path when it exists. |
| `--live` | flag | off |  | v0.53.5 #116 — run live short `soup train` proxy runs per candidate (requires --base-yaml). Defaults to the offline synthetic proxy. |
| `--base-yaml` | text |  |  | Path to a base soup.yaml (under cwd) supplying base/task/training/output. Required with --live. |

### `soup data persona-mix`

Sample a prompt × persona × style matrix.

Usage: `soup data persona-mix [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--prompts` | text |  | yes | JSONL file with one prompt per row |
| `--n` | integer |  | yes | Number of rows to emit \[1, 1_000_000] |
| `--output` | text |  | yes | Output JSONL path |
| `--personas` | text | `""` |  | Optional JSONL of personas (defaults to bundled set) |
| `--styles` | text | `""` |  | Optional JSONL of styles (defaults to bundled set) |
| `--seed` | integer | `0` |  | Deterministic seed |

### `soup data pii`

Flag rows containing email / phone / SSN / credit-card patterns.

Usage: `soup data pii [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--input`, `-i` | text |  | yes | Input JSONL file to scan. |
| `--output`, `-o` | text | `pii.jsonl` |  | Output JSONL file containing rows with detected PII. |

### `soup data preprocess`

AOT-tokenize a dataset and cache to disk for reuse across runs.

Usage: `soup data preprocess [OPTIONS] CONFIG_PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `CONFIG_PATH` | text |  | yes | Path to soup.yaml — uses data.train + tokenizer + max_length |
| `--output`, `-o` | text | `./.soup-tokenized` |  | Cache directory under cwd |
| `--yes` | flag | off |  | Skip confirmation prompt |

### `soup data preview`

Preview a remote HuggingFace dataset: metadata, splits, features.

Usage: `soup data preview DATASET_ID`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `DATASET_ID` | text |  | yes | HuggingFace dataset ID (e.g. teknium/OpenHermes-2.5) |

### `soup data push`

Upload a local JSONL dataset to a model hub as a dataset repo.

Usage: `soup data push [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--input`, `-i` | text |  | yes | Path to local JSONL dataset file |
| `--hf-dataset` | text |  | yes | HuggingFace dataset repo id (e.g. user/my-dataset) |
| `--private` | flag | off |  | Make the HF dataset repo private |
| `--message` | text | `Upload dataset with Soup CLI` |  | Commit message for the dataset upload |
| `--hub` | text | `hf` |  | Target hub: hf (default) / modelscope / modelers. Non-HF hubs upload via the matching SDK (v0.71.5 #157) — repo_type=dataset, commit message sanitised to first line + 200 chars. |

### `soup data recipe`

Validate a Data Recipe DAG and optionally execute every node.

Usage: `soup data recipe [OPTIONS] PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `PATH` | text |  | yes | Path to recipe.yaml under cwd |
| `--execute` | flag | off |  | Run the validated DAG end-to-end (v0.53.7 #106 — live per-node execution: seed / llm_text / code / judge / validator / sampler). |
| `--output`, `-o` | text |  |  | Output dir for sampler node (required with --execute). |
| `--provider` | text |  |  | LLM provider for llm_text/judge nodes: ollama, anthropic, or vllm. |
| `--model` | text |  |  | Provider model name (defaults to llama3.1). |
| `--base-url` | text |  |  | Provider base URL (Ollama/vLLM; defaults to loopback). |
| `--offline` | flag | off |  | Explicitly allow placeholder generation and accept-all judging. |

### `soup data register`

Register a local dataset by name for use in soup.yaml.

Usage: `soup data register [OPTIONS] [NAME_ARG] [PATH_ARG]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `NAME_ARG` | text |  |  | Dataset name (positional alternative to --name) |
| `PATH_ARG` | text |  |  | Path to dataset file (positional alternative to --path) |
| `--name`, `-n` | text |  |  | Dataset name |
| `--path`, `-p` | text |  |  | Path to dataset file |
| `--format`, `-f` | text | `auto` |  | Dataset format: alpaca, sharegpt, chatml, dpo, kto, auto |

### `soup data registry`

List all registered datasets.

Usage: `soup data registry`

Takes no options or arguments.

### `soup data review`

Preview preference pairs for manual review.

Usage: `soup data review [OPTIONS] INPUT`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `INPUT` | text |  | yes | Path to preference JSONL (chosen/rejected) |
| `--sample`, `-s` | integer | `10` |  | How many pairs to preview (1-100) |

### `soup data sample`

Sample a subset of a dataset using various strategies.

Usage: `soup data sample [OPTIONS] PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `PATH` | text |  | yes | Path to dataset file |
| `--output`, `-o` | text |  |  | Output file path (default: &lt;input>\_sampled.jsonl) |
| `--n`, `-n` | integer |  |  | Number of samples to select |
| `--pct` | float |  |  | Percentage of dataset to sample (0-100) |
| `--strategy`, `-s` | text | `random` |  | Sampling strategy: random, diverse (TF-IDF + clusters), hard (by length) |
| `--seed` | integer |  |  | Random seed for reproducibility |

### `soup data score`

Composite data-quality scorecard (PII + keyword triage + lang + edu + dec).

Usage: `soup data score [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--input`, `-i` | text |  | yes | JSONL input under cwd. |
| `--benchmarks`, `-b` | text |  |  | Benchmark labels (e.g. mmlu,gsm8k), validated only; they do not select or filter --benchmark-file texts. Requires --benchmark-file; corpora are not bundled. |
| `--benchmark-file` | text |  |  | Operator-supplied benchmark JSONL under cwd, with usable 8-gram text in every row. |
| `--threshold` | float, 0.0 to 1.0 | `0.8` |  | Decontamination overlap threshold. |

### `soup data search`

Search HuggingFace Hub for datasets.

Usage: `soup data search [OPTIONS] QUERY`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `QUERY` | text |  | yes | Search query for HuggingFace datasets |
| `--limit`, `-l` | integer | `20` |  | Maximum results to show |
| `--sort`, `-s` | text | `downloads` |  | Sort by: downloads, likes, lastModified, trending, createdAt |

### `soup data split`

Split dataset into train/val/test files.

Usage: `soup data split [OPTIONS] PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `PATH` | text |  | yes | Path to dataset file |
| `--val` | integer |  |  | Validation split: percentage (default) or absolute count (with --absolute) |
| `--test` | integer |  |  | Test split: percentage (default) or absolute count (with --absolute) |
| `--train` | integer |  |  | Train split (informational; the train remainder is implied by --val + --test). Accepted for command parity. |
| `--absolute` | flag | off |  | Treat --val/--test as absolute sample counts instead of percentages |
| `--seed` | integer |  |  | Random seed for reproducible splits |
| `--stratify` | text |  |  | Field name for stratified splitting (preserves category distribution) |
| `--stratify-semantic` | flag | off |  | Use semantic clustering (TF-IDF + K-Means) to perform stratified splitting without requiring a category field |
| `--num-clusters` | integer |  |  | Number of semantic clusters to use for semantic stratified splitting (default: 5) |

### `soup data stats`

Extended dataset statistics: length distribution, token counts, languages.

Usage: `soup data stats PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `PATH` | text |  | yes | Path to dataset file |

### `soup data topics`

Cluster a dataset into topics and show a coverage table.

Usage: `soup data topics [OPTIONS] PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `PATH` | text |  | yes | Path to dataset file (jsonl/json/csv) |
| `--clusters`, `-k` | text | `auto` |  | Number of topic clusters, or 'auto' (sqrt heuristic). |
| `--embed-model` | text | `sentence-transformers/all-MiniLM-L6-v2` |  | Embedding model. |
| `--device` | text | `auto` |  | auto/cpu/cuda |
| `--seed` | integer | `0` |  | Clustering seed. |
| `--output`, `-o` | text |  |  | Write the report as JSON. |

### `soup data toxicity`

Run an abuse-keyword heuristic (not a toxicity classifier).

Usage: `soup data toxicity [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--input`, `-i` | text |  | yes | Input JSONL file to score. |
| `--output`, `-o` | text | `toxicity.jsonl` |  | Output JSONL file for rows meeting the threshold. |
| `--threshold` | float, 0.0 to 1.0 | `0.05` |  | Rows meeting the abuse-keyword threshold are kept. |

### `soup data unregister`

Remove a dataset from the local registry.

Usage: `soup data unregister [OPTIONS] [NAME_ARG]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `NAME_ARG` | text |  |  | Dataset name (positional alternative to --name) |
| `--name`, `-n` | text |  |  | Dataset name to remove |

### `soup data validate`

Validate a dataset, returning exit 3 for input errors and 2 for unusable data.

Usage: `soup data validate [OPTIONS] PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `PATH` | text |  | yes | Path to dataset file |
| `--format`, `-f` | text | `auto` |  | Expected format: auto, alpaca, sharegpt, chatml, dpo, kto, plaintext |
| `--min-valid-fraction` | float, 0.0 to 1.0 | `0.0` |  | Exit 2 when the fraction of valid rows is below this value |

## `soup delinearize-llama4`

Delinearize Llama 4 fused expert weights to 3-D form.

Usage: `soup delinearize-llama4 [OPTIONS] SOURCE_DIR`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `SOURCE_DIR` | text |  | yes | Llama 4 checkpoint directory (containing \*.safetensors). |
| `--target`, `-o` | text |  | yes | Destination directory for the delinearized weights (under cwd). |
| `--model-id` | text |  |  | Optional model id; warn if it doesn't look like a Llama 4 model. |
| `--num-experts` | integer |  |  | Expert count for the reshape. Defaults to config.json's num_local_experts. |
| `--plan-only` | flag | off |  | Render the plan and exit without reshaping (v0.44.0 flow). |

## `soup deploy`

Deploy models: Ollama integration (deploy, list, remove).

Usage: `soup deploy <command> [ARGS]...`

Subcommands:

- [`autopilot`](#soup-deploy-autopilot)
- [`hf-space`](#soup-deploy-hf-space)
- [`ollama`](#soup-deploy-ollama)

### `soup deploy autopilot`

Pick PEFT + quant + spec-decoding combo for a hardware target.

Usage: `soup deploy autopilot [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--target`, `-t` | text |  |  | Profile name (e.g. mac-m3, rtx-4090-24gb, ollama-local). |
| `--base`, `-b` | text | `meta-llama/Llama-3.2-1B` |  | Base model HF repo id or local path to embed in the recipe. |
| `--recipe-out` | text | `deploy_autopilot.yaml` |  | Where to write the rendered soup.yaml recipe (under cwd). |
| `--script-out` | text | `deploy_autopilot.sh` |  | Where to write the planned deploy shell script (under cwd). |
| `--output-dir` | text | `./output` |  | Recipe's training output directory. |
| `--list`, `-l` | flag | off |  | List all known deploy profiles. |
| `--measure` | flag | off |  | Run Quant-Lobotomy measurement on candidate quants. Requires --tasks. Results cached at \~/.soup/deploy_autopilot_cache.json (v0.53.1 #109). |
| `--tasks` | text |  |  | JSONL eval tasks for --measure (one prompt + expected per line). |
| `--measure-candidates` | text |  |  | Comma-separated quant candidates to measure (default: profile's primary quant). Example: 4bit,8bit,fp8 |

### `soup deploy hf-space`

Create a HuggingFace Space wrapping a fine-tuned model.

Usage: `soup deploy hf-space [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--model`, `-m` | text |  | yes | HuggingFace model repo id to wrap in the Space (e.g. user/my-model) |
| `--space`, `-s` | text |  | yes | HuggingFace Space repo id to create (e.g. user/my-space) |
| `--template`, `-t` | text | `gradio-chat` |  | Space template: gradio-chat, streamlit-chat |
| `--template-dir` | text |  |  | Custom template directory (overrides --template). Must contain app.py + README.md, optionally requirements.txt. Use {MODEL_REPO} placeholder for substitution. |
| `--private` | flag | off |  | Create the Space as private |
| `--yes`, `-y` | flag | off |  | Skip confirmation |

### `soup deploy ollama`

Deploy a GGUF model to local Ollama instance.

Usage: `soup deploy ollama [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--model`, `-m` | text |  |  | Path to GGUF model file |
| `--name`, `-n` | text |  |  | Ollama model name (e.g. soup-my-model) |
| `--system`, `-s` | text |  |  | System prompt for the model |
| `--template`, `-t` | text | `auto` |  | Chat template: auto, chatml, llama, mistral, vicuna, zephyr |
| `--parameter`, `-p` | text, repeatable |  |  | Ollama parameter (repeatable): temperature=0.7, top_p=0.9, etc. |
| `--list`, `-l` | flag | off |  | List Soup-deployed models in Ollama |
| `--remove`, `-r` | text |  |  | Remove a model from Ollama by name |
| `--yes`, `-y` | flag | off |  | Skip confirmation prompt |

## `soup diagnose`

Post-training report card: forgetting / refusal / format / mode_collapse / memorization / contamination (v0.56.0).

Usage: `soup diagnose [OPTIONS] RUN_ID`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `RUN_ID` | text |  | yes | Registry run id (or any opaque tag). |
| `--base` | text | `""` |  | Base model name (informational). |
| `--adapter` | text | `""` |  | Adapter name (informational). |
| `--evidence` | text |  |  | JSON file with pre-computed probe scores (see README). |
| `--output`, `-o` | text |  |  | Write the report JSON to this path. |
| `--badge` | text |  |  | Write an SVG badge to this path. |
| `--attach-to-registry` | text |  |  | Attach the --output report to a registry entry id (needs --output). Exits 1 when the attach fails. |
| `--base-model` | text |  |  | Base model id to LOAD for a live diagnose run (e.g. HuggingFaceTB/SmolLM2-135M). When set, the 6 probes run live against the model instead of emitting neutral scores. |
| `--dataset` | text |  |  | Training JSONL for live format / memorization / citation probes (must stay under cwd). |
| `--holdout` | text |  |  | Held-out evaluation JSONL for the live forgetting probe (must stay under cwd; falls back to built-in general prompts if unset). |
| `--tokenizer` | text |  |  | Tokenizer id/path for a sub-word memorization probe (live). |
| `--device` | text |  |  | Device for the live run (cuda / cpu). |
| `--citation-style` | text | `bracket` |  | Citation style for the live RAFT citation probe: bracket (\[doc-id]) / inline ((doc-id)) / footnote (\[^id]). Must match the style the model was trained to emit. (v0.71.17 #254) |
| `--allow-not-run` | flag | off |  | Exit 0 even when a requested probe did not run (verdict NOT_RUN). Without it such a run exits 3. |
| `--shuffle-seed` | integer |  |  | RAFT document-shuffle seed for the live citation probe (composes the \[doc-N] ids the model is scored against). (v0.71.17 #254) |

## `soup diff`

Compare outputs of two models side-by-side on the same prompts.

Usage: `soup diff [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--model-a`, `-a` | text |  | yes | Path to first model (LoRA adapter or full model) |
| `--model-b`, `-b` | text |  | yes | Path to second model (LoRA adapter or full model) |
| `--prompts`, `-p` | text |  |  | Path to prompts file (JSONL with 'prompt' field, or one prompt per line) |
| `--prompt` | text, repeatable |  |  | Single prompt to compare (can be repeated) |
| `--base-a` | text |  |  | Base model for model A (auto-detected for LoRA adapters) |
| `--base-b` | text |  |  | Base model for model B (auto-detected for LoRA adapters) |
| `--max-tokens` | integer | `256` |  | Maximum tokens to generate per response |
| `--temperature`, `-t` | float | `0.7` |  | Sampling temperature |
| `--device` | text |  |  | Device: cuda, mps, cpu. Auto-detected if not set. |
| `--output`, `-o` | text |  |  | Save results to JSONL file |
| `--trust-remote-code` | flag | off |  | Allow loading models that ship custom Python via auto_map. Default deny (v0.36.0). Only enable if you trust the source. |

## `soup distill-prompt`

Distill prompt-heavy traces into a small FT plan (v0.68.0).

Usage: `soup distill-prompt [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--traces` | text |  | yes | Path to traces JSONL |
| `--teacher` | text |  | yes | Teacher model id |
| `--student` | text |  | yes | Student model id |
| `--strategy` | text | `sft` |  | Distill strategy. Allowed: kl, preference, sft |
| `--provider` | text | `ollama` |  | Teacher/student provider: ollama / anthropic / vllm. |
| `--base-url` | text |  |  | Provider base URL (ollama / vllm). |
| `--temperature` | float | `0.0` |  | Sampling temperature for teacher/student. |
| `--max-rows` | integer |  |  | Cap the number of distilled rows. |
| `--output`, `-o` | text | `distilled.jsonl` |  | Output JSONL path |
| `--plan-only` | flag | off |  | Render plan + exit 0 (no live preparation). |

## `soup doctor`

Check system dependencies, GPU, and compatibility.

Usage: `soup doctor [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--nccl` | flag | off |  | Measure NCCL bandwidth and check against reference table. |
| `--disk` | flag | off |  | Detect the disk media type (NVMe / SSD / HDD). Layer streaming's disk overflow tier needs NVMe. Off by default: the probe costs \~9 s on Windows, and on Linux it WRITES a \~64 MiB scratch file (.soup-diskprobe-\*, git-ignored) into the current directory to measure sequential read throughput. |
| `--config`, `-c` | text |  |  | Also check a soup.yaml: report which of the settings it actually sets are not read on its task and backend (#755). |

## `soup draft`

Train + measure a speculative-decoding draft model: distil your tuned model into a tiny draft, then serve it with --auto-spec (v0.71.33).

Usage: `soup draft <command> [ARGS]...`

Subcommands:

- [`distill`](#soup-draft-distill)
- [`list`](#soup-draft-list)
- [`measure`](#soup-draft-measure)

### `soup draft distill`

Distil a target model into a tiny dense speculative-decoding draft.

Usage: `soup draft distill [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--target` | text |  | yes | The tuned model to speed up (the teacher). |
| `--draft-base` | text |  | yes | Tiny model to distil into (the student). Must share the target's tokenizer. A `soup shrink` output qualifies by construction. |
| `--data` | text |  | yes | JSONL distillation set (chat / instruction rows). |
| `-o`, `--output` | text | `draft` |  | Directory for the dense draft model. |
| `--steps` | integer, 1 to 1000000 | `500` |  | Approximate training steps. |
| `--device` | text |  |  | cpu \| cuda (default: auto-detect). |
| `--no-register` | flag | off |  | Do not record the draft in \~/.soup/drafts.json (it then won't be picked up by `soup serve --auto-spec`). |
| `--trust-remote-code` | flag | off |  | Allow custom modelling code from the target / draft-base (required for architectures that ship an auto_map). |
| `--force` | flag | off |  | Overwrite -o even if it already contains a non-draft directory. |
| `--plan-only` | flag | off |  | Print the distill config and exit; write nothing. |

### `soup draft list`

List locally-trained drafts that `soup serve --auto-spec` can use.

Usage: `soup draft list`

Takes no options or arguments.

### `soup draft measure`

Report a draft's acceptance rate + throughput against its target.

Usage: `soup draft measure [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--target` | text |  | yes | The model being served. |
| `--draft` | text |  | yes | The draft model (a `soup draft distill` output). |
| `--prompts` | text |  | yes | JSONL prompts representative of production traffic. |
| `--max-new-tokens` | integer, 1 to 4096 | `64` |  | Tokens to generate per prompt. |
| `--num-assistant-tokens` | integer, 1 to 64 | `5` |  | Tokens the draft proposes per step (the assisted-generation knob). |
| `--device` | text |  |  | cpu \| cuda. |
| `--min-acceptance` | float, 0.0 to 1.0 |  |  | Exit 2 if the acceptance rate falls below this (for CI gating). |
| `--trust-remote-code` | flag | off |  | Allow custom modelling code from the target / draft (required for architectures that ship an auto_map). |
| `-o`, `--output` | text |  |  | Write the report as JSON. |
| `--sweep-k` | text |  |  | Also time assisted generation at each of these draft lengths, e.g. 1,2,3,5,8 (at most 8, each in 1..64). |

## `soup drift-alarm`

Rolling-KL drift alarm on output-token distribution with optional Slack/Discord webhook (v0.63.0).

Usage: `soup drift-alarm [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--reference` | text |  | yes | JSONL of FT-time reference outputs. |
| `--live` | text |  | yes | JSONL of live production outputs. |
| `--threshold` | float | `0.2` |  | Drift threshold (KL divergence). Default 0.2. |
| `--slack-url` | text |  |  | Optional Slack webhook URL — POSTed on drift. SSRF-validated. |
| `--discord-url` | text |  |  | Optional Discord webhook URL — POSTed on drift. SSRF-validated. |

## `soup edit`

Knowledge editing (ROME / MEMIT / AlphaEdit) - patch facts without re-training (v0.61.0).

Usage: `soup edit <command> [ARGS]...`

Subcommands:

- [`diff`](#soup-edit-diff)
- [`set`](#soup-edit-set)

### `soup edit diff`

Knowledge-injection diff: facts changed between before / after.

Usage: `soup edit diff [OPTIONS] BEFORE AFTER`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `BEFORE` | text |  | yes | Label or run id of the model BEFORE the edit (metadata). |
| `AFTER` | text |  | yes | Label or run id of the model AFTER the edit (metadata). |
| `--probes` | text |  |  | JSONL file with probe prompts; required when both --before-model and --after-model are given. Each row should have a 'prompt' field. Capped at 1000 rows. |
| `--output`, `-o` | text |  |  | Where to write the rendered diff JSON. |
| `--top-k`, `-k` | integer | `10` |  | Number of changed facts to surface (1-100). |
| `--before-model` | text |  |  | Model path / HF id of the BEFORE model (live generation with --after-model and --probes). |
| `--after-model` | text |  |  | Model path / HF id of the AFTER model (live generation with --before-model and --probes). |
| `--device` | text |  |  | torch device (cpu / cuda). Defaults to CUDA when available. |

### `soup edit set`

Apply a single surgical knowledge edit.

Usage: `soup edit set [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--base`, `-b` | text |  | yes | Base model HF id or local path. |
| `--method`, `-m` | text | `rome` |  | Edit method: rome / memit / alphaedit. |
| `--subject`, `-s` | text |  | yes | Prefix sentence (e.g. "Paris is the capital of France"). |
| `--target`, `-t` | text |  | yes | New completion target (e.g. "Lyon"). |
| `--layer`, `-l` | integer |  |  | MLP layer index to edit (defaults to method-specific recommended layer). |
| `--output`, `-o` | text |  |  | Directory to save the edited model / GRACE codebook (cwd-contained). |
| `--device` | text |  |  | torch device (cpu / cuda). Defaults to CUDA when available. |
| `--governor` / `--no-governor` | flag | on |  | Consult the sequential-edit governor (refuse on norm blowup). |
| `--cov-corpus` | text |  |  | ROME only: JSONL/text corpus to estimate the key covariance C for a C^{-1}-preconditioned update (reduces collateral damage). |
| `--plan-only` | flag | off |  | Print the resolved EditPlan and exit without applying. |
| `--registry-id` | text |  |  | Registry entry id to attach the saved result to (needs --output): the weight file(s) as edited_model, or the GRACE codebook as grace_codebook. Exits 1 when the attach fails. |

## `soup env`

Hermetic env lockfile + ABI drift detection (v0.64.0).

Usage: `soup env <command> [ARGS]...`

Subcommands:

- [`check`](#soup-env-check)
- [`fix`](#soup-env-fix)
- [`lock`](#soup-env-lock)
- [`status`](#soup-env-status)

### `soup env check`

Compare the current env against the lock and report drift.

Usage: `soup env check [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--lock` | text | `soup-env.lock` |  | Path to the lock file to compare against. |

### `soup env fix`

Render a reproducible install plan from ``soup-env.lock``.

Usage: `soup env fix [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--lock` | text | `soup-env.lock` |  | Path to the lock file to render an install plan from. |
| `--format` | text | `uv-pip` |  | Install-plan format: uv-pip (copy/paste uv commands) \| requirements. |
| `--output`, `-o` | text |  |  | Optionally also write a requirements.txt to this path (under cwd). |

### `soup env lock`

Snapshot the current env into a lock file.

Usage: `soup env lock [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--output`, `-o` | text | `soup-env.lock` |  | Path to write the lock file (default: ./soup-env.lock). |

### `soup env status`

Print currently-locked env summary.

Usage: `soup env status [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--lock` | text | `soup-env.lock` |  | Path to the lock file (default: ./soup-env.lock). |

## `soup eval`

Evaluate models: benchmarks, custom evals, LLM judge, leaderboard.

Usage: `soup eval <command> [ARGS]...`

Subcommands:

- [`against`](#soup-eval-against)
- [`aider`](#soup-eval-aider)
- [`auto`](#soup-eval-auto)
- [`behavior`](#soup-eval-behavior)
- [`benchmark`](#soup-eval-benchmark)
- [`capability`](#soup-eval-capability)
- [`checklist`](#soup-eval-checklist)
- [`citation`](#soup-eval-citation)
- [`compare`](#soup-eval-compare)
- [`coverage`](#soup-eval-coverage)
- [`custom`](#soup-eval-custom)
- [`design`](#soup-eval-design)
- [`discover`](#soup-eval-discover)
- [`gate`](#soup-eval-gate)
- [`gate-install`](#soup-eval-gate-install)
- [`human`](#soup-eval-human)
- [`irt-subset`](#soup-eval-irt-subset)
- [`judge`](#soup-eval-judge)
- [`leaderboard`](#soup-eval-leaderboard)
- [`lock`](#soup-eval-lock)
- [`quant-check`](#soup-eval-quant-check)
- [`unlearning`](#soup-eval-unlearning)

### `soup eval against`

Run-vs-run regression check (paired-bootstrap CI).

Usage: `soup eval against [OPTIONS] BASELINE_RUN_ID`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `BASELINE_RUN_ID` | text |  | yes | Baseline run id (from `soup runs`). |
| `--candidate` | text |  | yes | Candidate run id whose metrics are compared to the baseline. |
| `--metric` | text | `task_accuracy` |  | Metric to check: task_accuracy \| refusal_rate \| format_validity \| p95_latency_ms \| custom \| aider_polyglot \| judge:&lt;model> \| benchmark:&lt;lm-eval-task>. |
| `--n-samples` | integer | `1000` |  | Paired-bootstrap samples (100-100000). |
| `--seed` | integer | `0` |  | Deterministic seed. |
| `--json-only` | flag | off |  | Suppress Rich output; emit a single JSON verdict line. |
| `--suite` | text |  |  | Locked eval suite (from `soup eval lock`) to validate as a gate precondition — a missing / unparseable suite BLOCKS the check (exit 3). cwd-contained. The generated pre-push hook passes \$GATE_SUITE here so the locked suite is actually enforced. |

### `soup eval aider`

Run Aider's Polyglot coding benchmark in its official Docker image.

Usage: `soup eval aider [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--model`, `-m` | text |  | yes | Aider model identifier (for example, openai/gpt-4.1) |
| `--output`, `-o` | text |  | yes | Contained directory for upstream and Soup result files |
| `--exercises-dir` | text | `tmp.benchmarks/polyglot-benchmark` |  | Prepared Aider-AI/polyglot-benchmark checkout |
| `--image` | text | `aider-benchmark` |  | Locally built upstream Aider benchmark image |
| `--threads`, `-t` | integer, 1 to 64 | `1` |  | Number of benchmark exercises to run concurrently |
| `--num-tests`, `-n` | integer | `-1` |  | Number of exercises to run (-1 runs all exercises) |
| `--timeout` | integer, 60 to 86400 | `86400` |  | Maximum benchmark runtime in seconds |
| `--run-id` | text |  |  | Save the aggregate score under an existing Soup run ID |
| `--allow-host-services` | flag | off |  | Let the benchmark container reach services on the host |

### `soup eval auto`

Run automatic evaluation using config from soup.yaml.

Usage: `soup eval auto [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--config`, `-c` | text | `soup.yaml` |  | Path to soup.yaml config file |
| `--benchmarks`, `-b` | text |  |  | Comma-separated benchmarks (overrides config) |
| `--tasks`, `-t` | text |  |  | Path to custom eval JSONL (overrides config) |
| `--trust-remote-code` | flag | off |  | Allow loading models that ship custom Python via auto_map. Default deny (v0.36.0). Only enable if you trust the source. |

### `soup eval behavior`

Score a run on a bundled behaviour battery (pre/post diff).

Usage: `soup eval behavior [OPTIONS] RUN_ID`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `RUN_ID` | text |  | yes | Run identifier. |
| `--battery`, `-b` | text | `xstest` |  | Battery: xstest / harmbench / jailbreakbench / elephant / syceval. |
| `--evidence`, `-e` | text |  |  | Path to a JSON file with {pre_responses, post_responses, oracle} arrays. Here each oracle entry is a word the matching response must contain; --base-model scores refusal behaviour instead. |
| `--output`, `-o` | text |  |  | Where to write the rendered BehaviorDiffReport JSON. |
| `--base-model` | text |  |  | Base model id for a LIVE diff (generates pre/post responses). |
| `--adapter` | text |  |  | LoRA adapter path for the 'post' model in a live diff. |
| `--device` | text |  |  | Device for the live diff (cuda / cpu). |

### `soup eval benchmark`

Evaluate on standard benchmarks (wraps lm-evaluation-harness).

Usage: `soup eval benchmark [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--model`, `-m` | text |  | yes | Path to model or LoRA adapter directory |
| `--benchmarks`, `-b` | text | `mmlu` |  | Comma-separated benchmark names (mmlu, gsm8k, hellaswag, etc.) |
| `--fewshot`, `-f` | integer |  |  | Number of few-shot examples (benchmark default if not set) |
| `--batch-size` | integer | `8` |  | Batch size for evaluation |
| `--run-id` | text |  |  | Link results to an existing training run |
| `--device` | text |  |  | Device: cuda, mps, cpu. Auto-detected if not set. |
| `--trust-remote-code` | flag | off |  | Allow loading models that ship custom Python via auto_map. Default deny (v0.36.0). Only enable if you trust the source. |

### `soup eval capability`

Run a bundled capability profile (MMLU-Pro / GPQA / AIME / ...).

Usage: `soup eval capability [OPTIONS] RUN_ID`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `RUN_ID` | text |  | yes | Run identifier. |
| `--suite`, `-s` | text | `fast` |  | Profile: full / fast / math / code. |
| `--output`, `-o` | text |  |  | Where to write the rendered CapabilityReport JSON. |
| `--live` | flag | off |  | Run the suite LIVE via lm-eval-harness against --model. |
| `--model`, `-m` | text |  |  | HF model id for a live run (required with --live). |
| `--tasks` | text |  |  | Comma-separated lm-eval task override (live; bypasses --suite). |
| `--limit` | integer |  |  | Cap eval examples per task (live; use 1-5 for a smoke). |
| `--device` | text |  |  | Device for the live run (cuda / cpu). |

### `soup eval checklist`

Run CheckList MFT / INV / DIR behavioural tests.

Usage: `soup eval checklist [OPTIONS] SPEC_PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `SPEC_PATH` | text |  | yes | Path to CheckList DSL YAML. |
| `--evidence`, `-e` | text |  |  | Required JSON with operator-supplied per-test responses. |
| `--output`, `-o` | text |  |  | Where to write the rendered CheckListReport JSON. |

### `soup eval citation`

Score citation precision / recall / F1 over a JSONL (#202).

Usage: `soup eval citation [OPTIONS] DATA`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `DATA` | text |  | yes | JSONL of {predicted, expected_ids} rows OR RAFT rows {query, golden_doc, distractor_docs, answer}. |
| `--style` | text | `bracket` |  | Citation style: bracket / inline / footnote. |
| `--shuffle-seed` | integer |  |  | For RAFT-shaped rows: the data.raft_shuffle_seed used at train time, so the golden \[doc-N] id matches what the model saw. Ignored for {predicted, expected_ids} rows. |
| `--output`, `-o` | text |  |  | Write the per-row + aggregate CitationScore JSON here. |

### `soup eval compare`

Compare eval results between two training runs.

Usage: `soup eval compare RUN1 RUN2`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `RUN1` | text |  | yes | First run ID |
| `RUN2` | text |  | yes | Second run ID |

### `soup eval coverage`

Heuristic coverage / gap analysis for a locked or drafted suite.

Usage: `soup eval coverage [OPTIONS] DESIGN_PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `DESIGN_PATH` | text |  | yes | Path to an EvalDesign JSON |
| `--task` | text |  | yes | Task category from v0.54.0 taxonomy: factual_lookup \| style_shaping \| format_conversion \| reasoning \| tool_use \| summarization \| classification. |

### `soup eval custom`

Run custom evaluation tasks from a JSONL file.

Usage: `soup eval custom [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--tasks`, `-t` | text |  | yes | Path to eval tasks JSONL file |
| `--model`, `-m` | text |  | yes | Path to model directory |
| `--run-id` | text |  |  | Link results to an existing training run |
| `--attach-to-registry` | text |  |  | Attach the eval JSON to a registry entry as kind=eval_results |
| `--output`, `-o` | text |  |  | Path to write the eval JSON output. Honored independently of --attach-to-registry (v0.40.1 / G10). |
| `--trust-remote-code` | flag | off |  | Allow loading models that ship custom Python via auto_map. Default deny (v0.36.0). Only enable if you trust the source. |

### `soup eval design`

Draft an evaluation suite from training data + a one-line goal.

Usage: `soup eval design [OPTIONS] DATA`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `DATA` | text |  | yes | Training-data JSONL path |
| `--goal`, `-g` | text |  | yes | One-line goal description (e.g. 'better at SQL'). |
| `--num-dimensions`, `-n` | integer | `5` |  | Number of eval dimensions to draft (1-20). |
| `--output`, `-o` | text | `evals/design.json` |  | Where to write the rendered EvalDesign JSON. |

### `soup eval discover`

Discover a held-out canary set for regression detection.

Usage: `soup eval discover [OPTIONS] DATA`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `DATA` | text |  | yes | Training-data JSONL path |
| `--base` | text |  |  | Base model id (recorded; consumed by `soup diagnose`). |
| `--num-clusters` | integer | `5` |  | Number of behavioural clusters to discover (1-64). |
| `--per-cluster` | integer | `3` |  | Held-out canaries to draw per cluster (1-64). |
| `--seed` | integer | `0` |  | Deterministic seed. |
| `--output`, `-o` | text | `evals/canaries.json` |  | Where to write the rendered CanarySet JSON. |

### `soup eval gate`

Run an eval-gate suite standalone (post-hoc verdict).

Usage: `soup eval gate [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--suite`, `-s` | text |  | yes | Path to eval suite YAML (see evals/gate.yaml example) |
| `--baseline`, `-b` | text |  |  | Baseline: registry://&lt;id> or path to stamped/flat score JSON |
| `--regression-threshold` | float | `0.05` |  | Max absolute drop vs baseline before regression fires (0.0-1.0) |
| `--model`, `-m` | text |  |  | Model path or HF id to evaluate (required for live scoring) |
| `--write-baseline` | text |  |  | Write this run's task scores as a stamped baseline JSON ({'scores', 'provenance'}) consumable by --baseline (#404). |

### `soup eval gate-install`

Install a pre-push regression gate for the selected baseline.

Usage: `soup eval gate-install [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--baseline` | text |  | yes | Baseline run id the pre-push hook compares against. |
| `--suite` | text | `evals/locked.json` |  | Path to the locked eval suite (cwd-contained). |
| `--hook-path` | text | `.git/hooks/pre-push` |  | Hook target — usually .git/hooks/pre-push. |
| `--force` | flag | off |  | Overwrite an existing hook. |

### `soup eval human`

Run human A/B evaluation between two models.

Usage: `soup eval human [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--input`, `-i` | text |  | yes | Path to JSONL file with evaluation prompts |
| `--model-a`, `-a` | text |  | yes | Path to first model |
| `--model-b`, `-b` | text |  | yes | Path to second model |
| `--output`, `-o` | text | `human_eval_results.json` |  | Output file for results |

### `soup eval irt-subset`

Pick a minimum-cost eval subset that preserves ranking power.

Usage: `soup eval irt-subset [OPTIONS] RESPONSES_PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `RESPONSES_PATH` | text |  | yes | Path to JSONL with rows {item_id, correct(bool), respondent_id? (required for 2pl/3pl)}. |
| `--size`, `-z` | text | `small` |  | Subset profile: full / small / tiny. |
| `--model`, `-m` | text | `1pl` |  | IRT model: 1pl (Rasch) / 2pl (discrimination) / 3pl (+ guessing). |
| `--output`, `-o` | text |  |  | Where to write the rendered IrtSubsetPlan JSON. |

### `soup eval judge`

Evaluate model outputs using LLM-as-a-judge.

Usage: `soup eval judge [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--target` | text |  | yes | Path to JSONL with prompt+response pairs to evaluate |
| `--model`, `-m` | text | `gpt-4o-mini` |  | Judge model name |
| `--provider` | text | `openai` |  | Judge provider: openai, server, ollama |
| `--rubric`, `-r` | text |  |  | Path to rubric YAML file |
| `--api-base` | text |  |  | API base URL for judge model |
| `--run-id` | text |  |  | Link results to an existing training run |

### `soup eval leaderboard`

Show local leaderboard across all evaluated models.

Usage: `soup eval leaderboard [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--sort-by`, `-s` | text |  |  | Sort by specific benchmark (default: average) |
| `--format`, `-f` | text | `table` |  | Output format: table, json, csv |

### `soup eval lock`

Freeze an EvalDesign as a checksummed eval_suite artifact.

Usage: `soup eval lock [OPTIONS] DESIGN_PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `DESIGN_PATH` | text |  | yes | Path to an EvalDesign JSON (from `soup eval design`) |
| `--output`, `-o` | text | `evals/locked.json` |  | Where to write the canonicalised locked suite. |
| `--attach-to-registry` | text |  |  | Registry entry id or name to attach the locked suite to. |

### `soup eval quant-check`

Compare accuracy before vs after quantization on the same eval suite.

Usage: `soup eval quant-check [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--before` | text |  | yes | Before-quantization model path or registry://&lt;id> |
| `--after` | text |  | yes | After-quantization model path or registry://&lt;id> |
| `--tasks` | text |  | yes | JSONL eval tasks file (see 'soup eval custom') |
| `--format` | text | `table` |  | Output format: table \| json \| markdown |
| `--allow-stub` | flag | off |  | Allow deterministic stub generators if live model loading fails. |

### `soup eval unlearning`

Score an unlearn run on TOFU / MUSE / WMDP.

Usage: `soup eval unlearning [OPTIONS] RUN_ID`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `RUN_ID` | text |  | yes | Run identifier (e.g. registry id of the unlearn run). |
| `--benchmark`, `-b` | text | `tofu` |  | Benchmark: tofu / muse / wmdp. |
| `--evidence`, `-e` | text |  |  | Path to a JSON file with pre-computed evidence. Schema: {forget_quality: {pre_loss, post_loss}, model_utility: {pre_acc, post_acc}, priv_leak: {mia_auc}}. Missing keys fall through to a neutral OK score. |
| `--output`, `-o` | text |  |  | Where to write the rendered UnlearnReport JSON. |
| `--attach-to-registry` | text |  |  | Registry entry id to attach the --output report to as an eval_results artifact (needs --output). Exits 1 when the attach fails. |

## `soup expect`

Run an expectations suite against a JSONL dataset.

Usage: `soup expect [OPTIONS] DATA SUITE`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `DATA` | text |  | yes | Path to JSONL dataset |
| `SUITE` | text |  | yes | Path to expectations suite YAML |
| `--judge` | text |  |  | Judge model URL (e.g. ollama://llama3.1, https://api.openai.com/gpt-4o-mini) |

## `soup export`

Export a model to GGUF, ONNX, TensorRT-LLM, BitNet, or TorchAO format.

Usage: `soup export [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--model`, `-m` | text |  | yes | Path to model directory (full model or LoRA adapter) |
| `--format`, `-f` | text | `gguf` |  | Export format: gguf, onnx, tensorrt, bitnet, tq1_0, torchao, gguf-ud (torchao needs --quant-config) |
| `--quant`, `-q` | text | `q4_k_m` |  | Quantization type: q4_0, q4_k_m, q5_k_m, q8_0, f16, f32 |
| `--output`, `-o` | text |  |  | Output file path. Default: &lt;model-name>.&lt;quant>.gguf |
| `--base`, `-b` | text |  |  | Base model ID (for LoRA adapters). Auto-detected if not set. |
| `--llama-cpp` | text |  |  | Path to llama.cpp directory. Auto-detected or cloned to \~/.soup/llama.cpp |
| `--onnx-task` | text | `text-generation` |  | ONNX export task: text-generation (causal LM) or feature-extraction (embedding) |
| `--deploy` | text |  |  | Auto-deploy after export. Currently supported: ollama |
| `--deploy-name` | text |  |  | Model name for deployment (used with --deploy) |
| `--calibration-data` | text |  |  | Path to calibration JSONL for imatrix GGUF export. |
| `--registry-id` | text |  |  | Attach exported artifact to this registry entry (default: auto-match by source --model output dir) |
| `--trust-remote-code` | flag | off |  | Allow loading models that ship custom Python via auto_map. Default deny (v0.36.0). Only enable if you trust the source. |
| `--quant-config` | text |  |  | Path to YAML for torchao PTQ export (v0.53.1 #142). Required when --format=torchao. |
| `--gguf-flavour` | text |  |  | Advanced GGUF format flag — UD-Q\*\_K_XL / IQ\*\_M / Q4_0_4_4 / etc. Required when --format=gguf-ud (v0.53.1 #139). |
| `--hub` | text | `hf` |  | Source hub for the base model when --model is a LoRA adapter: hf (default) / modelscope / modelers (v0.53.10 #152). |
| `--verbose` | flag | off |  | Stream GGUF converter and quantizer logs while they run. |
| `--no-install-deps` | flag | off |  | GGUF only: never invoke pip; refuse missing conversion dependencies before work. |

## `soup fetch`

Fetch a ready-to-edit example config from the bundled catalog.

Usage: `soup fetch [OPTIONS] NAMESPACE [NAME]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `NAMESPACE` | text |  | yes | One of: examples, configs, deepspeed_configs. |
| `NAME` | text |  |  | Catalog entry name (omit to list). |
| `--output`, `-o` | text |  |  | Destination path (default: ./&lt;filename> in cwd). |
| `--force` | flag | off |  | Overwrite if --output already exists. |

## `soup history`

Show the lineage tree for all entries with a given name.

Usage: `soup history NAME`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `NAME` | text |  | yes | Registry entry name to trace (e.g. 'medical-chat') |

## `soup infer`

Run batch inference on a JSONL file of prompts.

Usage: `soup infer [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--model`, `-m` | text |  | yes | Path to model (LoRA adapter or full model) |
| `--input`, `-i` | text |  | yes | Path to input JSONL file (each line: {"prompt": "..."}) |
| `--output`, `-o` | text |  | yes | Path to output JSONL file for results |
| `--backend` | text | `transformers` |  | Text backend: transformers or mlx |
| `--base`, `-b` | text |  |  | Base model for LoRA adapter (auto-detected if not set) |
| `--max-tokens` | integer, 1 to 16384 | `256` |  | Maximum tokens to generate per response (1-16384) |
| `--temperature`, `-t` | float | `0.7` |  | Sampling temperature (0 = greedy) |
| `--batch-size` | integer, at least 1 | `1` |  | Number of causal-LM prompts to generate together (default: 1; batching is opt-in; fp16/bf16 batching may not be bit-exact with batch size 1) |
| `--device` | text |  |  | Device: cuda, mps, cpu. Auto-detected if not set. |
| `--task` | text | `text` |  | Inference task: 'text' (default, chat generation) or 'asr' (Whisper transcription; input rows are {"audio": path\[, "text": reference]}). |
| `--asr-language` | text |  |  | ASR decode language (overrides the training sidecar). --task asr only. |
| `--asr-task` | text |  |  | ASR decode objective: transcribe \| translate. --task asr only. |
| `--audio-dir` | text |  |  | Base directory audio paths in --input must stay under (--task asr; defaults to the --input file's directory). Traversal / UNC paths are rejected. |
| `--trust-remote-code` | flag | off |  | Allow loading models that ship custom Python via auto_map. Default deny (v0.36.0). Only enable if you trust the source. |
| `--hub` | text | `hf` |  | Source hub for the base model: hf (default) / modelscope / modelers. Non-HF hubs require the matching SDK (v0.53.10 #152). |
| `--cuda-graphs` | flag | off |  | Experimental CUDA graph decode for resident, unquantized Qwen2/Llama models. |

## `soup ingest`

Universal trace importer: Langfuse / LangSmith / Helicone / OpenPipe / OTel / OpenAI Stored Completions (v0.63.0).

Usage: `soup ingest [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--source` | text |  | yes | Trace source: langfuse \| langsmith \| helicone \| openpipe \| otel \| openai-stored |
| `--logs` | text |  |  | Path to JSONL trace export (one event per line). Required unless --pull. |
| `--output`, `-o` | text |  |  | Output JSONL (default: traces.jsonl in cwd). |
| `--pull` | flag | off |  | Fetch generations live from the Langfuse API instead of reading --logs (--source langfuse only). Reads LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY, plus LANGFUSE_HOST for another region or a self-hosted instance. |
| `--since` | text |  |  | With --pull: how far back to fetch, e.g. 30m, 24h, 7d (default 7d, max 365d). |
| `--max-pages` | integer, 1 to 10000 |  |  | With --pull: stop with an error, writing nothing, if results are still pending after this many pages of 100 generations (default 100). |
| `--allow-private-host` | flag | off |  | With --pull: allow LANGFUSE_HOST to be a private or loopback address (self-hosted Langfuse). HTTPS is still required. |
| `--slack-url` | text |  |  | Optional Slack webhook URL — POSTed on completion. SSRF-validated. |
| `--discord-url` | text |  |  | Optional Discord webhook URL — POSTed on completion. SSRF-validated. |

## `soup init`

Create a new soup.yaml config interactively or from a template.

Usage: `soup init [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--template`, `-t` | text |  |  | Template: audio, bco, chat, code, embedding, eu-ai-act, hipaa, ipo, kto, longcontext, medical, moe, orpo, pretrain, reasoning, rlhf, simpo, soc2, sr-11-7, tool-calling, vision |
| `--output`, `-o` | text | `soup.yaml` |  | Output config file path |
| `--force`, `-f` | flag | off |  | Overwrite existing config without prompting (v0.40.1 / M2). |

## `soup iterative-dpo`

Iterative DPO sample, score, pair, and train loop.

Usage: `soup iterative-dpo [OPTIONS] [<command>] [ARGS]...`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--base-model` | text |  | yes | HF id / local path |
| `--reward-model` | text |  | yes | RM HF id / path |
| `--prompts` | text |  | yes | Source prompts JSONL |
| `--output-dir` | text |  | yes | Output dir |
| `--rounds` | integer | `3` |  | Number of iterative-DPO rounds |
| `--pairs-per-round` | integer | `500` |  | Number of (chosen, rejected) pairs per round |
| `--plan-only` | flag | off |  | Print the resolved plan and exit (no training). |

## `soup license-advisor`

Recommend a license-clean base for a deploy target (b2c / defense / embedded) + flag downstream risk (v0.64.0).

Usage: `soup license-advisor [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--target`, `-t` | text |  | yes | Deploy target: b2c \| defense \| embedded. |
| `--license` | text |  |  | Optional: per-license risk check (e.g. apache-2.0, llama-3). |
| `--monthly-active-users`, `--mau` | integer | `0` |  | Expected MAU for the per-license risk check (default 0). |

## `soup llama`

Proxy to llama.cpp binaries (cli / mtmd-cli / gguf-split / server).

Usage: `soup llama [<command>] [ARGS]...`

Subcommands:

- [`cli`](#soup-llama-cli)
- [`gguf-split`](#soup-llama-gguf-split)
- [`mtmd-cli`](#soup-llama-mtmd-cli)
- [`quantize`](#soup-llama-quantize)
- [`server`](#soup-llama-server)

### `soup llama cli`

Forward args to llama.cpp binary llama-cli.

Usage: `soup llama cli [ARGS]...`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `ARGS...` | text, any number |  |  | Args forwarded to llama-cli. |

### `soup llama gguf-split`

Forward args to llama.cpp binary llama-gguf-split.

Usage: `soup llama gguf-split [ARGS]...`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `ARGS...` | text, any number |  |  | Args forwarded to llama-gguf-split. |

### `soup llama mtmd-cli`

Forward args to llama.cpp binary llama-mtmd-cli.

Usage: `soup llama mtmd-cli [ARGS]...`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `ARGS...` | text, any number |  |  | Args forwarded to llama-mtmd-cli. |

### `soup llama quantize`

Forward args to llama.cpp binary llama-quantize.

Usage: `soup llama quantize [ARGS]...`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `ARGS...` | text, any number |  |  | Args forwarded to llama-quantize. |

### `soup llama server`

Forward args to llama.cpp binary llama-server.

Usage: `soup llama server [ARGS]...`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `ARGS...` | text, any number |  |  | Args forwarded to llama-server. |

## `soup local-rl`

Personal-LLM flywheel daemon (init / status / record / harvest / train) (v0.68.0).

Usage: `soup local-rl <command> [ARGS]...`

Subcommands:

- [`harvest`](#soup-local-rl-harvest)
- [`init`](#soup-local-rl-init)
- [`record`](#soup-local-rl-record)
- [`status`](#soup-local-rl-status)
- [`train`](#soup-local-rl-train)

### `soup local-rl harvest`

Harvest DPO pairs from thumbs into a JSONL file.

Usage: `soup local-rl harvest [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--db` | text | `local_rl.db` |  | Path to local-RL SQLite database |
| `--output`, `-o` | text | `dpo_pairs.jsonl` |  | Output JSONL path |

### `soup local-rl init`

Create the local-RL SQLite schema.

Usage: `soup local-rl init [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--db` | text | `local_rl.db` |  | Path to local-RL SQLite database |

### `soup local-rl record`

Append a thumbs-up / thumbs-down record.

Usage: `soup local-rl record [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--db` | text | `local_rl.db` |  | Path to local-RL SQLite database |
| `--prompt` | text |  | yes | Prompt text |
| `--response` | text |  | yes | Model response |
| `--thumb` | text |  | yes | up / down |

### `soup local-rl status`

Print counters from the local-RL database.

Usage: `soup local-rl status [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--db` | text | `local_rl.db` |  | Path to local-RL SQLite database |

### `soup local-rl train`

Run (``--once``) or schedule the nightly DPO/KTO/ORPO train.

Usage: `soup local-rl train [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--db` | text | `local_rl.db` |  | Path to local-RL SQLite database |
| `--backend` | text |  |  | Accepted for compatibility; has no effect on training. Allowed: mlx, ollama |
| `--model` | text |  | yes | Training base — HF repo id or local path (NOT the Ollama tag) |
| `--train-method` | text | `dpo` |  | Allowed: dpo, kto, orpo |
| `--once` | flag | off |  | Run the train now (ad-hoc); skip scheduler install. |
| `--min-pairs` | integer | `10` |  | Skip training when fewer pairs are harvested. |
| `--output`, `-o` | text | `local_rl_adapter` |  | Adapter output directory |
| `--scheduler-dir` | text | `local-rl-scheduler` |  | Directory to render the systemd / launchd scaffold into. |
| `--hour` | integer | `3` |  | Daily train hour (local time). |
| `--minute` | integer | `0` |  | Daily train minute. |

## `soup lock`

Shared run lockfile (write / show / check).

Usage: `soup lock <command> [ARGS]...`

Subcommands:

- [`check`](#soup-lock-check)
- [`show`](#soup-lock-show)
- [`write`](#soup-lock-write)

### `soup lock check`

Refuse with exit 2 if the lock has drifted from current state.

Usage: `soup lock check [OPTIONS] [PATH]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `PATH` | text | `soup.lock` |  | Path to tracked soup.lock |
| `--base-sha` | text |  | yes | 64-hex current base-model SHA |
| `--dataset-sha` | text |  | yes | 64-hex current dataset SHA |
| `--env-hash` | text |  | yes | 64-hex current env hash |
| `--base-model` | text |  | yes | Current base model id |

### `soup lock show`

Print a tracked lock file.

Usage: `soup lock show [PATH]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `PATH` | text | `soup.lock` |  | Path to soup.lock |

### `soup lock write`

Render a ``soup.lock`` from the base/dataset/env hashes.

Usage: `soup lock write [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--base-model` | text |  | yes | HF model id / path |
| `--base-sha` | text |  | yes | 64-hex base-model SHA |
| `--dataset-sha` | text |  | yes | 64-hex dataset SHA |
| `--env-hash` | text |  |  | 64-hex env hash. If omitted, auto-derived from --env-lock (soup-env.lock) via `soup env lock`. |
| `--env-lock` | text | `soup-env.lock` |  | Path to a soup-env.lock to auto-derive --env-hash from (used only when --env-hash is omitted). |
| `--output`, `-o` | text | `soup.lock` |  | Output path |

## `soup loop`

Data flywheel: traces -> preference pairs -> DPO -> gate -> canary deploy -> rollback, all from the CLI (v0.58.0).

Usage: `soup loop <command> [ARGS]...`

Subcommands:

- [`canary`](#soup-loop-canary)
- [`init`](#soup-loop-init)
- [`pause`](#soup-loop-pause)
- [`replay`](#soup-loop-replay)
- [`resume`](#soup-loop-resume)
- [`status`](#soup-loop-status)
- [`watch`](#soup-loop-watch)

### `soup loop canary`

Promote ``new_adapter`` as the canary with a traffic split.

Usage: `soup loop canary [OPTIONS] NEW_ADAPTER`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `NEW_ADAPTER` | text |  | yes | Adapter id/path to canary. |
| `--traffic` | text | `5%` |  | Traffic share, e.g. "5%" or "5". |
| `--autoroll-on-regress` / `--no-autoroll-on-regress` | flag | on |  | Roll back automatically on MAJOR verdict. |

### `soup loop init`

Create the .soup/loop.yaml control file (one-time setup).

Usage: `soup loop init [OPTIONS] SERVED_MODEL`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `SERVED_MODEL` | text |  | yes | Served model id (e.g. registry://abc12). |
| `--eval` | text |  | yes | Eval suite path or registry ref. |
| `--baseline` | text |  | yes | Baseline registry id or file. |
| `--monthly-budget` | text |  |  | Monthly USD cap (e.g. 50usd, 100). |
| `--max-runs-per-day` | integer |  |  | Cap on iteration starts per UTC day. |
| `--pre-wired` | flag | off |  | Use the pre-wired harvest/train/gate/deploy stages (utils/loop_stages.py) on watch instead of no-op stubs (v0.71.4). |
| `--thumbs-db` | text |  |  | Local-RL SQLite written by `soup serve --record-thumbs`; the pre-wired harvest turns its thumbs into preference pairs. Must stay under cwd. SOUP_LOOP_THUMBS_DB overrides it. |
| `--force` | flag | off |  | Overwrite existing loop.yaml. |

### `soup loop pause`

Pause the watch daemon at the next iteration boundary.

Usage: `soup loop pause`

Takes no options or arguments.

### `soup loop replay`

Replay a recorded loop iteration manifest.

Usage: `soup loop replay [OPTIONS] [ITERATION_ID]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `ITERATION_ID` | text |  |  | Iteration id (omit to list all). |
| `--extract` | text |  |  | Extract the iteration's .can to this directory for a what-if re-run (v0.71.4). |

### `soup loop resume`

Resume a paused loop (next watch iteration picks up automatically).

Usage: `soup loop resume`

Takes no options or arguments.

### `soup loop status`

Show counters and current status.

Usage: `soup loop status`

Takes no options or arguments.

### `soup loop watch`

Run the harvest → train → gate → deploy daemon.

Usage: `soup loop watch [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--foreground` | flag | off |  | Run in foreground (default). |
| `--detach` | flag | off |  | Spawn a background subprocess running --foreground. |
| `--max-iterations` | integer |  |  | Stop after N iterations (test/demo use). |
| `--poll-interval` | float | `60.0` |  | Seconds between iterations \[1, 3600]. |
| `--pre-wired` | flag | off |  | Force the pre-wired production stages even if loop.yaml didn't set it. |
| `--pack-cans` | flag | off |  | Pack each iteration as a v0.26 Soup Can + Registry entry (v0.71.4). |
| `--tool-auth-token` | text |  |  | The soup serve --tool-auth-token value, sent as a Bearer token when the deploy stage activates an adapter. Defaults to \$SOUP_TOOL_AUTH_TOKEN, which keeps it out of the shell history. Env: `SOUP_TOOL_AUTH_TOKEN`. |

## `soup mcp`

Model Context Protocol server - drive Soup from any MCP client (v0.71.28).

Usage: `soup mcp <command> [ARGS]...`

Subcommands:

- [`runs`](#soup-mcp-runs)
- [`serve`](#soup-mcp-serve)

### `soup mcp runs`

Recover persisted MCP execution runs.

Usage: `soup mcp runs <command> [ARGS]...`

Subcommands:

- [`reconcile`](#soup-mcp-runs-reconcile)

#### `soup mcp runs reconcile`

Recover execution capacity left blocked by a crashed MCP server.

Usage: `soup mcp runs reconcile [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--expunge-launching` | flag | off |  | Delete stale launching rows after checking that no recorded PID is alive. |
| `--older-than-seconds` | integer, at least 1 | `300` |  | Only consider launching rows at least this many seconds old. |

### `soup mcp serve`

Start the MCP server (read-only tools by default).

Usage: `soup mcp serve [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--allow-mutating` | flag | off |  | Enable the plan-only mutating tools (train_start / export). Even when enabled they only render the command that would run - v1 never executes training or export. Off by default: those tools refuse. |
| `--allow-execute` | flag | off |  | Reserve the execution gate for future MCP tools. Implies --allow-mutating, but train_start and export remain plan-only and never execute in this version. |
| `--transport` | text | `stdio` |  | stdio (default), sse, or http. stdio speaks JSON-RPC over the process pipes and is what a client that spawns Soup wants. sse and http open a local listener instead, for remote or multi-client setups, and require a Bearer token. |
| `--host` | text |  |  | Interface to bind for --transport sse/http. Defaults to 127.0.0.1. Binding anything else exposes the server to the network, where the Bearer token is the only gate. |
| `--port` | integer |  |  | Port to bind for --transport sse/http. Defaults to 8765. |
| `--auth-token` | text |  |  | Override the auto-generated Bearer token (16-128 urlsafe base64 chars) for --transport sse/http. When omitted, a fresh token is generated each startup and printed. |

## `soup merge`

Merge a LoRA adapter with its base model into a full model.

Usage: `soup merge [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--adapter`, `-a` | text |  | yes | Path to the LoRA adapter directory |
| `--base`, `-b` | text |  |  | Base model ID. Auto-detected from adapter_config.json if not set. |
| `--output`, `-o` | text | `./merged` |  | Output directory for the merged model |
| `--dtype` | text | `float16` |  | Data type for the merged model: float16, bfloat16, float32 |
| `--trust-remote-code` | flag | off |  | Allow loading models that ship custom Python via auto_map. Default deny (v0.36.0). Only enable if you trust the source. |
| `--save-format` | text | `fp16` |  | Merged-checkpoint save format. fp16 (default) writes a standard fp16 merge. 4bit / 4bit_forced write a single BNB-4bit-quantized merge without the dequant-merge-requant cycle (v0.53.1 #142). |
| `--double-quant` / `--no-double-quant` | flag | on |  | BNB double-quantization for the 4bit / 4bit_forced save formats (#321). Default on, matching the training loaders; pass --no-double-quant to disable. Ignored for the fp16 save format. |
| `--low-memory` | flag | off |  | Merge tensor by tensor straight from the base safetensors files instead of loading the whole model into RAM. Needs the fp16 save format and a plain LoRA adapter. |
| `--hub` | text | `hf` |  | Source hub for the base model: hf (default) / modelscope / modelers. Non-HF hubs require the matching SDK (v0.53.10 #152). |

## `soup merge-sharded-fsdp-weights`

Consolidate FSDP shard files into a single safetensors file.

Usage: `soup merge-sharded-fsdp-weights [OPTIONS] SHARD_DIR`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `SHARD_DIR` | text |  | yes | Directory containing pytorch_model_fsdp\_\*.bin shard files. |
| `--output`, `-o` | text |  | yes | Destination .safetensors file path (under cwd). |
| `--plan-only` | flag | off |  | Print the consolidation plan and exit without writing anything. |

## `soup migrate`

Import a config from LLaMA-Factory, Axolotl, or Unsloth notebook.

Usage: `soup migrate [OPTIONS] CONFIG_FILE`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--from` | text |  | yes | Source tool: llamafactory, axolotl, or unsloth |
| `CONFIG_FILE` | text |  | yes | Path to the source config file (.yaml or .ipynb) |
| `--output`, `-o` | text | `soup.yaml` |  | Output path for generated soup.yaml |
| `--dry-run` | flag | off |  | Print generated config without writing to file |
| `--yes`, `-y` | flag | off |  | Skip confirmation prompts |

## `soup monitor`

Live GPU monitor: Util / Temp / VRAM / Power per GPU.

Usage: `soup monitor [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--refresh`, `-r` | float | `2.0` |  | Refresh interval in seconds (0.25 to 30). |
| `--once` | flag | off |  | Print one snapshot and exit (skip the live panel). |

## `soup plan`

Render a pre-flight training plan (cost / ETA / SHA / VRAM) and write soup.tfstate for `soup apply` to consult (v0.64.0).

Usage: `soup plan [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--config`, `-c` | text | `soup.yaml` |  | Path to soup.yaml. |
| `--state` | text | `soup.tfstate` |  | Path to write the state file (default: ./soup.tfstate). |

## `soup plugins`

Discover, list, enable, and disable Soup plugins.

Usage: `soup plugins [<command>] [ARGS]...`

Subcommands:

- [`disable`](#soup-plugins-disable)
- [`enable`](#soup-plugins-enable)
- [`install`](#soup-plugins-install)
- [`list`](#soup-plugins-list)

### `soup plugins disable`

Disable a registered Soup plugin.

Usage: `soup plugins disable NAME`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `NAME` | text |  | yes | Plugin name |

### `soup plugins enable`

Enable a registered Soup plugin.

Usage: `soup plugins enable NAME`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `NAME` | text |  | yes | Plugin name |

### `soup plugins install`

Explain how to install plugins without pretending an install occurred.

Usage: `soup plugins install NAME`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `NAME` | text |  | yes | Plugin name |

### `soup plugins list`

List all registered plugins.

Usage: `soup plugins list`

Takes no options or arguments.

## `soup probe`

Activation probes: sleeper-agent defection / honesty / misuse / pairwise interference / SAE feature diff / probe pack (v0.66.0, truth+harm v0.71.8).

Usage: `soup probe <command> [ARGS]...`

Subcommands:

- [`harm`](#soup-probe-harm)
- [`interference`](#soup-probe-interference)
- [`pack`](#soup-probe-pack)
- [`sae-diff`](#soup-probe-sae-diff)
- [`sleeper`](#soup-probe-sleeper)
- [`truth`](#soup-probe-truth)

### `soup probe harm`

Apply the HarmBench-style misuse probe to activations (v0.71.8 #217).

Usage: `soup probe harm [OPTIONS] BASE`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `BASE` | text |  | yes | Base model id (must have a bundled probe) |
| `--evidence` | text |  |  | JSON with 'activations': \[\[...], ...] (2D float matrix) |
| `--weights` | text |  |  | Operator-supplied calibrated probe weights (.npz / .npy / .safetensors) |
| `--output`, `-o` | text |  |  | Write probe result JSON to path |

### `soup probe interference`

Pairwise N×N adapter interference matrix (v0.66.0; --measure v0.71.8 #218).

Usage: `soup probe interference [OPTIONS] [LOSSES]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `LOSSES` | text |  |  | JSON: {adapters: \[...], losses: {'a\|b': loss_value, ...}} |
| `--measure` | text |  |  | Auto-measure losses against this eval-suite JSONL instead of supplying a losses JSON. Requires --base-model + ≥2 --adapter (v0.71.8 #218). |
| `--base-model` | text |  |  | Base model id for --measure |
| `--adapter` | text, repeatable |  |  | name=path adapter spec for --measure (repeatable) |
| `--device` | text |  |  | Device for --measure (cpu / cuda) |
| `--output`, `-o` | text |  |  | Write matrix JSON to path |

### `soup probe pack`

Assemble or list calibrated probe packs for a base (v0.66.0).

Usage: `soup probe pack [OPTIONS] [BASE]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `BASE` | text |  |  | Base model id (e.g. meta-llama/Llama-3-8B) |
| `--list` | flag | off |  | List all bases that ship a probe pack |
| `--output`, `-o` | text |  |  | Write pack manifest JSON to path |

### `soup probe sae-diff`

SAE feature diff between pre- and post-FT activations (v0.66.0).

Usage: `soup probe sae-diff [OPTIONS] SAE PRE_ACTS POST_ACTS`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `SAE` | text |  | yes | Path to SAE safetensors checkpoint (or repo id with --auto-download) |
| `PRE_ACTS` | text |  | yes | JSON: {'activations': \[\[...]]} |
| `POST_ACTS` | text |  | yes | JSON: {'activations': \[\[...]]} |
| `--top-k` | integer, 1 to 10000 | `20` |  | Top-K changed features to report |
| `--auto-download` | flag | off |  | Treat &lt;sae> as an HF Hub repo id and download it into \~/.soup/sae-cache/ first (must be in HF_HUB_ALLOWLIST; v0.71.8 #216). |
| `--output`, `-o` | text |  |  | Write diff report JSON to path |

### `soup probe sleeper`

Apply the sleeper-agent defection probe to activations (v0.66.0).

Usage: `soup probe sleeper [OPTIONS] BASE`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `BASE` | text |  | yes | Base model id (must have a bundled probe) |
| `--evidence` | text |  |  | JSON with 'activations': \[\[...], ...] (2D float matrix) |
| `--weights` | text |  |  | Operator-supplied calibrated probe weights (.npz / .npy / .safetensors). When set, replaces the synthetic seed fallback and accepts any base id (v0.71.8 #215). |
| `--output`, `-o` | text |  |  | Write probe result JSON to path |

### `soup probe truth`

Apply the TruthfulQA-style honesty probe to activations (v0.71.8 #217).

Usage: `soup probe truth [OPTIONS] BASE`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `BASE` | text |  | yes | Base model id (must have a bundled probe) |
| `--evidence` | text |  |  | JSON with 'activations': \[\[...], ...] (2D float matrix) |
| `--weights` | text |  |  | Operator-supplied calibrated probe weights (.npz / .npy / .safetensors) |
| `--output`, `-o` | text |  |  | Write probe result JSON to path |

## `soup profile`

Estimate memory, speed, and GPU requirements BEFORE training.

Usage: `soup profile [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--config`, `-c` | text | `soup.yaml` |  | Path to soup.yaml config file |
| `--gpu`, `-g` | text |  |  | Target GPU for recommendations (e.g., rtx3090, rtx4090, a100, h100). Auto-detects if not set. |
| `--json` | flag | off |  | Output as JSON for scripting |

## `soup prune-prompt`

Detect + strip a shared system-prompt prefix across training data so the FT model internalises it (v0.63.0).

Usage: `soup prune-prompt [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--input`, `-i` | text |  | yes | Input JSONL with {prompt, output} rows (e.g. from `soup ingest`). |
| `--output`, `-o` | text |  | yes | Output JSONL with the shared prefix stripped from `prompt`. |
| `--min-frequency` | float | `0.95` |  | Prefix must appear in >= this fraction of rows to be stripped (0.0 - 1.0). |
| `--tokenizer` | text |  |  | Optional HF tokenizer (model id or local path) for token-aware prefix detection. Default: whitespace-character level. |
| `--slack-url` | text |  |  | Optional Slack webhook URL — POSTed on completion. SSRF-validated. |
| `--discord-url` | text |  |  | Optional Discord webhook URL — POSTed on completion. SSRF-validated. |

## `soup push`

Push a trained model to HuggingFace Hub (or alternate hub).

Usage: `soup push [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--model`, `-m` | text |  | yes | Path to the trained model / LoRA adapter directory |
| `--repo`, `-r` | text |  | yes | HuggingFace repo ID, e.g. username/my-model |
| `--private` | flag | off |  | Make the HuggingFace repo private |
| `--token`, `-t` | text |  |  | Upload token. For --hub hf: \[deprecated] use HF_TOKEN or cached login. For other hubs: that hub's token (or MODELSCOPE_API_TOKEN / MODELERS_TOKEN). Env: `HF_TOKEN`. |
| `--message` | text | `Upload model trained with Soup CLI` |  | Commit message for the upload |
| `--collection` | text |  |  | Add the pushed repo to an existing HF Collection (slug: 'owner/title-hash') |
| `--hub` | text | `hf` |  | Destination hub: hf (default) / modelscope / modelers. Non-HF hubs require the matching SDK and skip the HF-specific Collections / model-card auto-render path (v0.53.10 #152). |
| `--card` | text |  |  | Registry entry id / prefix / name:tag — render its provenance-rich model card (via `soup card`) and upload it as README.md, overriding the auto-generated card (v0.71.35). HF hub only. |

## `soup quantize`

Quantize a model - ergonomic alias for `soup export --format <to>`.

Usage: `soup quantize [OPTIONS] MODEL_PATH`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `MODEL_PATH` | text |  | yes | Source checkpoint (safetensors directory or single .safetensors). |
| `--to` | text | `gguf` |  | Target format: gguf \| onnx \| tensorrt. |
| `--bits` | integer | `4` |  | Quantization bits (1-16; respected by gguf). |
| `--output`, `-o` | text |  |  | Destination directory (default: &lt;model_path>-&lt;to>). |

## `soup quickstart`

Run a complete demo: create sample data, config, and train.

Usage: `soup quickstart [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--yes`, `-y` | flag | off |  | Skip confirmation prompt |
| `--dry-run` | flag | off |  | Create data and config only, do not train |
| `--output`, `-o` | text |  |  | Output directory for data, config, and run artifacts (default: current directory). |

## `soup ra-dit`

RA-DIT two-stage orchestrator: retriever -> generator (v0.71.10).

Usage: `soup ra-dit [OPTIONS] [<command>] [ARGS]...`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--retriever-config`, `-r` | text |  | yes | Stage-1 embedding/contrastive recipe (cwd-contained). |
| `--generator-config`, `-g` | text |  | yes | Stage-2 RAFT-SFT recipe (cwd-contained). |
| `--retriever-model` | text |  |  | Manual retriever-model override; skips Registry auto-link. |
| `--timeout` | integer | `21600` |  | Per-stage hard timeout in seconds (60..21600). |
| `--plan-only` | flag | off |  | Validate config paths + render the plan; skip training. |

## `soup recipes`

Ready-made configs: list, show, use, search recipes for popular models.

Usage: `soup recipes <command> [ARGS]...`

Subcommands:

- [`list`](#soup-recipes-list)
- [`search`](#soup-recipes-search)
- [`show`](#soup-recipes-show)
- [`use`](#soup-recipes-use)
- [`verify`](#soup-recipes-verify)

### `soup recipes list`

List all available recipes.

Usage: `soup recipes list`

Takes no options or arguments.

### `soup recipes search`

Search recipes by keyword, task, or model size.

Usage: `soup recipes search [OPTIONS] [QUERY]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `QUERY` | text |  |  | Search keyword |
| `--task` | text |  |  | Filter by task |
| `--size` | text |  |  | Filter by model size |

### `soup recipes show`

Show a recipe config (print YAML to stdout).

Usage: `soup recipes show NAME`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `NAME` | text |  | yes | Recipe name |

### `soup recipes use`

Copy a recipe to soup.yaml (or custom path).

Usage: `soup recipes use [OPTIONS] NAME`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `NAME` | text |  | yes | Recipe name |
| `--output`, `-o` | text | `soup.yaml` |  | Output path for config file |
| `--yes`, `-y` | flag | off |  | Skip confirmation prompts |

### `soup recipes verify`

Check that a config can attach a LoRA adapter, without downloading weights.

Usage: `soup recipes verify [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--config`, `-c` | text |  |  | Verify one config instead of the whole catalogue. |
| `--json` | flag | off |  | Machine-readable rows on stdout. |
| `--templates` / `--no-templates` | flag | on |  | Also verify src/soup_cli/templates and examples/ (catalogue mode). |

## `soup registry`

Model Registry: push, list, show, diff, search, promote, delete.

Usage: `soup registry <command> [ARGS]...`

Subcommands:

- [`delete`](#soup-registry-delete)
- [`diff`](#soup-registry-diff)
- [`list`](#soup-registry-list)
- [`promote`](#soup-registry-promote)
- [`push`](#soup-registry-push)
- [`search`](#soup-registry-search)
- [`show`](#soup-registry-show)

### `soup registry delete`

Delete a registry entry and its lineage / artifact rows.

Usage: `soup registry delete [OPTIONS] REF`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `REF` | text |  | yes | Entry ref to delete |
| `--yes`, `-y` | flag | off |  | Skip confirmation |

### `soup registry diff`

Side-by-side config + eval delta between two entries.

Usage: `soup registry diff LEFT RIGHT`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `LEFT` | text |  | yes | Left entry ref |
| `RIGHT` | text |  | yes | Right entry ref |

### `soup registry list`

List registry entries.

Usage: `soup registry list [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--name`, `-n` | text |  |  | Filter by registry entry name. |
| `--tag`, `-t` | text |  |  | Filter by tag. |
| `--base`, `-b` | text |  |  | Filter by base model. |
| `--task` | text |  |  | Filter by training task. |
| `--limit`, `-l` | integer | `100` |  | Maximum number of entries to show. |

### `soup registry promote`

Add a tag to an existing entry (e.g. promote to 'prod').

Usage: `soup registry promote [OPTIONS] REF`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `REF` | text |  | yes | Entry ref to promote |
| `--tag`, `-t` | text |  | yes | Tag to assign |

### `soup registry push`

Register a completed training run as a registry entry.

Usage: `soup registry push [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--run-id` | text |  | yes | Training run id to register |
| `--name`, `-n` | text |  | yes | Registry entry name |
| `--tag`, `-t` | text | `v1` |  | Tag (e.g. v1, prod-2024) |
| `--notes` | text | `""` |  | Free-form notes |
| `--data` | text |  |  | Training data file to record: its SHA-256 is stored as data_hash. This is the file registered as the training input, not proof of what the run trained on. Must be a file under the current directory. |
| `--idempotency-key` | text |  |  | Make a retry safe: a later push with the same key and the same run, name, tag and data returns the existing entry instead of a new one; anything different is refused. 1-128 characters from letters, digits, '.', '\_', ':', '-'. |

### `soup registry search`

Search registry entries by name/base/task/notes.

Usage: `soup registry search [OPTIONS] QUERY`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `QUERY` | text |  | yes | Keyword to search for |
| `--limit`, `-l` | integer | `50` |  | Maximum number of matching entries to show. |

### `soup registry show`

Show full details of a registry entry.

Usage: `soup registry show REF`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `REF` | text |  | yes | Entry id / prefix / name:tag |

## `soup reward`

Synthesize a deterministic reward verifier from reference outputs, with a calibration report that refuses degenerate verifiers (v0.71.40).

Usage: `soup reward <command> [ARGS]...`

Subcommands:

- [`stress`](#soup-reward-stress)
- [`synth`](#soup-reward-synth)

### `soup reward stress`

Adversarially probe a reward verifier for gameability (exit 0 robust / 2 gameable).

Usage: `soup reward stress [OPTIONS] REWARD_TARGET`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `REWARD_TARGET` | text |  | yes | Verifier to probe: a .py path, a builtin name, or 'verifiable'. |
| `--references` | text |  |  | JSONL of gold outputs (enables gold-aware probing). |
| `--field` | text | `answer` |  | Gold field (default: answer). |
| `--verifiable-domain` | text |  |  | Domain for a 'verifiable' target. |
| `--sentinel` | text | `GOLD` |  | Sentinel-spam token. |
| `--threshold` | float | `0.5` |  | Reward >= this = accept. |
| `--max-gameable` | float | `0.0` |  | Max junk accept-rate allowed before the verdict flips to gameable. |
| `--attacks` | text | `empty,length,repetition,sentinel,wrapped_junk,answer_spray` |  | Comma list: empty,length,repetition,sentinel,wrapped_junk,answer_spray. |
| `--output-report` | text |  |  | Also write the stress report as JSON. |

### `soup reward synth`

Synthesize a deterministic reward verifier from reference outputs.

Usage: `soup reward synth [OPTIONS] REFERENCES`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `REFERENCES` | text |  | yes | JSONL of reference (gold) outputs — a gold field or chat messages. |
| `-o`, `--output` | text |  |  | Where to write the verifier .py. |
| `--kind` | text | `auto` |  | Verifier family: auto \| numeric \| json_schema \| regex \| tool_call. |
| `--field` | text | `answer` |  | Gold-output field (default: answer). |
| `--tolerance` | float |  |  | Numeric match tolerance (numeric kind only). |
| `--min-discrimination` | float | `0.5` |  | Refuse to emit unless (accept_rate refs - accept_rate negatives) >= this. |
| `--force` | flag | off |  | Overwrite an existing output. |
| `--plan-only` | flag | off |  | Detect + report the induced spec; write nothing. |
| `--output-report` | text |  |  | Also write the calibration report as JSON. |

## `soup rewind`

Name the dataset rows behind a loss spike, from the training flight recorder.

Usage: `soup rewind [OPTIONS] [RUN_ID]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `RUN_ID` | text |  |  | Run ID (or prefix). Defaults to the most recent run. |
| `--step` | integer |  |  | Detail this step instead of the worst spike. |
| `--top` | integer, at least 1 | `10` |  | How many rows to list for the detailed step (at least 1). |
| `--json` | text |  |  | Write the spikes and the ranked rows to this path as JSON (must stay under the current directory). |
| `--no-preview` | flag | off |  | Skip the row previews (does not load the dataset). |

## `soup runs`

Experiment tracking: list, show, compare runs.

Usage: `soup runs [OPTIONS] [<command>] [ARGS]...`

Subcommands:

- [`clean`](#soup-runs-clean)
- [`compare`](#soup-runs-compare)
- [`curriculum-curve`](#soup-runs-curriculum-curve)
- [`delete`](#soup-runs-delete)
- [`replay`](#soup-runs-replay)
- [`show`](#soup-runs-show)

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--limit`, `-l` | integer | `20` |  | Max runs to show |
| `--cwd-only` | flag | off |  | Only show runs whose output_dir is under the current working directory (default: show all in \~/.soup/experiments.db). |

### `soup runs clean`

Intelligently clean up redundant checkpoint files to reclaim disk space.

Usage: `soup runs clean [OPTIONS] [RUN_ID]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `RUN_ID` | text |  |  | Run ID (or prefix) to clean. Omit if --all is used. |
| `--all` | flag | off |  | Cleanup all historical runs. |
| `--dry-run` | flag | off |  | Estimate space savings without deleting. |
| `--force`, `-f` | flag | off |  | Skip confirmation |
| `--keep-weights` / `--no-keep-weights` | flag | on |  | Keep intermediate model weights and delete only optimizer and scheduler states (default). --no-keep-weights deletes whole non-best checkpoints. |

### `soup runs compare`

Compare two training runs side by side.

Usage: `soup runs compare RUN_1 RUN_2`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `RUN_1` | text |  | yes | First run ID (or prefix) |
| `RUN_2` | text |  | yes | Second run ID (or prefix) |

### `soup runs curriculum-curve`

v0.48.0 (BETA) — visualise dynamic curriculum bucket weights.

Usage: `soup runs curriculum-curve [OPTIONS] RUN_ID`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `RUN_ID` | text |  | yes | Run ID (or prefix). |
| `--history` | text |  |  | Override path to curriculum_history.jsonl (default: under run output_dir). |
| `--width` | integer, 4 to 200 | `10` |  | Per-bucket column width. |

### `soup runs delete`

Delete a training run and its metrics.

Usage: `soup runs delete [OPTIONS] RUN_ID`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `RUN_ID` | text |  | yes | Run ID (or prefix) to delete |
| `--force`, `-f` | flag | off |  | Skip confirmation |

### `soup runs replay`

Replay a completed run's metrics — re-renders summary + loss curve.

Usage: `soup runs replay [OPTIONS] RUN_ID`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `RUN_ID` | text |  | yes | Run ID (or prefix) to replay. |
| `--plot` / `--no-plot` | flag | on |  | Render loss curve |

### `soup runs show`

Show detailed info about a specific run, including loss curve.

Usage: `soup runs show [OPTIONS] RUN_ID`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `RUN_ID` | text |  | yes | Run ID (or prefix) to show |
| `--plot` / `--no-plot` | flag | on |  | Show loss curve |

## `soup serve`

Start a local inference server with OpenAI-compatible API.

Usage: `soup serve [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--model`, `-m` | text |  | yes | Path to LoRA adapter directory or full model |
| `--base`, `-b` | text |  |  | Base model ID. Auto-detected from adapter_config.json if not set. |
| `--port`, `-p` | integer | `8000` |  | Port to serve on |
| `--host` | text | `127.0.0.1` |  | Host to bind to. Defaults to loopback (127.0.0.1). Any other host needs --tool-auth-token, which is checked on the tool, thumbs and adapter routes only: the generation routes take no token. |
| `--device` | text |  |  | Device: cuda, cuda:&lt;index>, mps or cpu. Auto-detected if not set. With the transformers backend, cpu loads on the CPU in float32 (unless --kv-cache-type bf16/f16 picks the dtype); cuda:&lt;index> and mps are pinned to that device; cuda uses device_map=auto, all in float16. |
| `--max-tokens` | integer | `512` |  | Default max tokens for generation |
| `--backend` | text | `transformers` |  | Inference backend: transformers (default), vllm, sglang, or mii |
| `--tensor-parallel`, `--tp` | integer | `1` |  | Number of GPUs for tensor parallelism (vLLM only) |
| `--gpu-memory` | float | `0.9` |  | Fraction of GPU memory to use (vLLM only, 0.0-1.0) |
| `--max-model-len` | integer |  |  | Maximum sequence length for the vLLM engine (vLLM only). Lower this when the engine refuses to start because the KV cache does not fit. Default: the model's own maximum. |
| `--speculative-decoding` | text |  |  | Draft model for speculative decoding (smaller/faster model ID or path) |
| `--num-speculative-tokens` | integer | `5` |  | Number of tokens the draft model generates per step (speculative decoding) |
| `--adapters` | text, repeatable |  |  | LoRA adapters as name=path pairs (repeatable). E.g. chat=./chat-adapter |
| `--prefix-cache` | flag | off |  | Enable vLLM prefix caching for shared system prompts (RAG/agent workloads). |
| `--auto-spec` | flag | off |  | Auto-pair draft model for speculative decoding based on target model. |
| `--structured-output` | text | `off` |  | Constrain generation: off (default) \| json \| regex. |
| `--json-schema` | text |  |  | Path to JSON schema file (used with --structured-output json). |
| `--regex-pattern` | text |  |  | Regex pattern (used with --structured-output regex). |
| `--dashboard` | flag | off |  | Enable live continuous-batching dashboard + /metrics endpoint. |
| `--trace` | flag | off |  | Enable OpenTelemetry request tracing (requires opentelemetry-sdk). |
| `--trace-endpoint` | text |  |  | OTLP endpoint URL (default: http://localhost:4317). |
| `--auto-quant` | flag | off |  | Unavailable until Soup can measure each loaded candidate; the command refuses instead of guessing a format. |
| `--trust-remote-code` | flag | off |  | Allow loading models that ship custom Python via auto_map. Default deny (v0.36.0). Only enable if you trust the source. |
| `--trace-log` | text |  |  | Append per-request {prompt, response, latency_ms, tokens, ts} to JSONL at this path. Path must stay under cwd. Rotates at 100 MB (one backup retained). Added in v0.40.3 (#33). |
| `--trace-log-cap-mb` | integer | `100` |  | Rotation cap in MB for --trace-log (1 - 10000). Default 100. |
| `--record-thumbs` | text |  |  | Auto-capture thumbs-up/down feedback into a local-rl SQLite at this path via POST /v1/thumbs. Path must stay under cwd. Transformers backend only. v0.71.1 (#230). |
| `--reasoning-parser` | text |  |  | Strip reasoning-trace blocks from responses. One of: deepseek-r1 \| qwen3 \| phi4 \| openthinker. v0.53.9 #98. |
| `--steer` | text |  |  | Apply a stored activation-steering vector at decode time (CAA / ITI / RepE). Pass the name registered via `soup steer train`; the transformers backend installs the decode hook after loading the model. |
| `--steer-strength` | float | `1.0` |  | Steering strength multiplier (\|s\| &lt;= 10.0). Ignored when --steer is unset. |
| `--hub` | text | `hf` |  | Source hub for the base model: hf (default) / modelscope / modelers. Non-HF hubs require the matching SDK (v0.53.10 #152). |
| `--bank` | text |  |  | Path to a VeRA / VB-LoRA vector bank (JSON). Multi-tenant LoRA serving at MB-per-user: the per-token delta is v_u ⊙ Px, routed by the X-User-Id request header. Requires --backend transformers. (v0.71.12 #221) |
| `--bank-strength` | float | `1.0` |  | Vector-bank delta strength multiplier. Ignored when --bank is unset. |
| `--mole` | text |  |  | Path to a MoLE training output dir (mole_gate.pt + mole_manifest.json). Serves the base + N frozen task LoRAs with per-token gate blending at decode time. Requires --backend transformers; not combinable with --bank / --steer / --adapters. (v0.71.17 #259) |
| `--kv-cache-type` | text |  |  | KV-cache type for decoding: q8_0 (8-bit quantized, needs hqq) / bf16 / f16 (cache dtype) / fp8 (vLLM+Hopper only). Transformers backend only — vLLM / SGLang routing is in the blocked tail. v0.71.14 (#140). |
| `--tool-auth-token` | text |  |  | Require 'Authorization: Bearer &lt;token>' on the tool endpoints (/v1/tools/python, /v1/tools/bash, /v1/tools/web_search). Without it, /v1/tools/python and /v1/tools/bash answer 401 on every bind and a non-loopback --host is refused. Env: `SOUP_TOOL_AUTH_TOKEN`. |

## `soup ship`

SHIP / DON'T SHIP verdict after fine-tuning: task win AND no catastrophic forgetting, fused into one decision (v0.71.25).

Usage: `soup ship [OPTIONS] [<command>] [ARGS]...`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--progress` | text |  |  | Opt-in JSONL progress on stderr (jsonl). |
| `--base` | text |  |  | Base model id/path (the 'before'). |
| `--tuned` | text |  |  | Tuned model id/path (a separate 'after' model). |
| `--adapter` | text |  |  | LoRA adapter path (tuned = base + this adapter). |
| `--task-eval` | text |  |  | JSONL of leg-1 task-win eval tasks. |
| `--task-mode` | text | `metric` |  | Leg-1 mode: metric \| judge_score \| pairwise (judge win-rate). |
| `--noise-floor` | integer |  |  | Re-run the BASE model N times (2-10) to measure what this instrument can resolve, print it beside the verdict, and refuse to call any delta smaller than the measured floor significant. Costs N extra base passes. The leg-1 task floor is measured in every --task-mode; in judge_score / pairwise each repeat is N extra JUDGE passes (N x the judge API calls), and the floor is labelled decode + judge. |
| `--judge-model` | text |  |  | Judge model URL for --task-mode judge_score or pairwise. |
| `--general-suite` | text |  |  | Comma list of leg-2 benchmarks (default: the bundled offline suite — MCQ/arithmetic + tool-call/JSON/safety; non-bundled names route through lm-eval). |
| `--baseline` | text |  |  | registry://&lt;id> or stamped JSON of base leg-2 scores (skips base run). Unstamped or scorer_revision-mismatched baselines warn once — re-measure so both sides share one scorer scale (#404). |
| `--forgetting-threshold` | float | `0.05` |  | Max allowed leg-2 drop in absolute points before DON'T SHIP. |
| `--evidence` | text |  |  | JSON of pre-computed scores (offline, no model load). |
| `--output`, `-o` | text |  |  | Write the verdict JSON to this path. |
| `--emit-evidence` | text |  |  | Write the scores in the --evidence INPUT schema so this run can be replayed offline (output-is-input). With --config it also STAMPS the config's provenance onto the emitted evidence. |
| `--config` | text |  |  | soup.yaml whose eval.ship block supplies defaults (CLI flags win). With --evidence alone it GATES (refuses evidence whose config_sha drifted from this config, or whose numerics family does not match); with --emit-evidence it STAMPS this config's provenance. |
| `--require-provenance` | flag | off |  | Strict offline drift check: requires --evidence, --config, --adapter and task eval. Compare artifact/task/source bytes, suite version and effective verdict policy. |
| `--data-sha256` | text |  |  | Caller-asserted training-data SHA-256 for emitted/strict evidence; not proof of which bytes trained the model. |
| `--push` | text |  |  | Post the verdict as a GitHub PR comment (owner/repo#N). Auth via GITHUB_TOKEN / GH_TOKEN; needs the `gh` CLI. |
| `--device` | text |  |  | Device for the live run (cuda / cpu). |
| `--chat-template-kwargs` | text |  |  | Live chat-template variables as a JSON object, e.g. {"enable_thinking":false}. |

## `soup shrink`

Depth-prune a model (drop the least-important contiguous layer block by residual angular distance) + optional distill-heal (v0.71.29).

Usage: `soup shrink [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--model` | text |  | yes | Model id or local path to shrink. |
| `--drop-ratio` | float |  |  | Fraction of layers to drop (0-1); e.g. 0.25. |
| `--drop-layers` | integer |  |  | Explicit number of contiguous layers to drop. |
| `--calib` | text |  | yes | Calibration JSONL (prompts) - must stay under cwd. |
| `--heal` | text |  |  | Heal JSONL (chat rows) - distill the original into the pruned model. |
| `--heal-steps` | integer | `200` |  | Approx. optimiser steps for the distill heal. |
| `--tolerance` | float | `0.1` |  | Perplexity-regression tolerance for the verdict. |
| `--output-dir`, `-o` | text | `./shrunk` |  | Directory for the shrunk model + report. |
| `--device` | text |  |  | Device for the importance/ppl passes (cuda / cpu). |
| `--trust-remote-code` | flag | off |  | Allow custom modeling code (auto_map). |
| `--attach-to-registry` | text |  |  | Attach the shrink report to a registry entry id. |
| `--plan-only` | flag | off |  | Print the importance table + chosen block and exit. |

## `soup spectrum`

Spectrum SNR scan for targeted training (v0.71.23).

Usage: `soup spectrum <command> [ARGS]...`

Subcommands:

- [`scan`](#soup-spectrum-scan)

### `soup spectrum scan`

Scan a model's layer SNR and emit a ``training.unfrozen_parameters`` patch.

Usage: `soup spectrum scan [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--model`, `-m` | text |  | yes | HF Hub id or local model directory. |
| `--top-percent` | float | `50.0` |  | Keep the top N% of layers (by SNR) within each module type. |
| `--modules` | text | `all` |  | Comma list of module types to scan: e.g. 'mlp,attn' — or 'all'. |
| `--output`, `-o` | text |  |  | Write the YAML patch to this path (must stay under cwd). |
| `--no-cache` | flag | off |  | Skip the \~/.soup/spectrum scan cache. |

## `soup steer`

Activation steering (CAA / ITI / RepE) - inference-time intervention without retraining (v0.62.0).

Usage: `soup steer <command> [ARGS]...`

Subcommands:

- [`apply`](#soup-steer-apply)
- [`list`](#soup-steer-list)
- [`train`](#soup-steer-train)

### `soup steer apply`

Resolve + load a stored steering vector and print its metadata.

Usage: `soup steer apply [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--name`, `-n` | text |  | yes | Identifier of a stored steering vector. |
| `--strength`, `-s` | float | `1.0` |  | Steering strength multiplier (\|s\| &lt;= 10.0). |

### `soup steer list`

List locally-stored steering vectors (Registry artifact kind ``steering_vector``).

Usage: `soup steer list`

Takes no options or arguments.

### `soup steer train`

Train a steering vector from contrastive prompt pairs.

Usage: `soup steer train [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--base`, `-b` | text |  | yes | Base model HF id or local path. |
| `--method`, `-m` | text | `caa` |  | Steering method: caa / iti / repe. |
| `--name`, `-n` | text |  | yes | Identifier for the trained vector (e.g. 'safety-v1'). |
| `--pairs`, `-p` | text |  | yes | Path to JSONL of contrastive (positive, negative) prompt pairs. |
| `--layer`, `-l` | integer |  |  | Decoder layer index to extract the residual-stream vector from (default: the middle layer). |
| `--output`, `-o` | text |  |  | Directory to write the steering vector to (default: ./steering/&lt;name>). Cwd-contained. |
| `--device` | text |  |  | torch device (cpu / cuda). Defaults to CUDA when available. |
| `--top-k`, `-k` | integer | `8` |  | ITI only: number of attention heads to intervene on (1-256). |
| `--plan-only` | flag | off |  | Validate inputs + print the resolved plan; skip the live fit. |
| `--registry-id` | text |  |  | Registry entry id to attach the trained vector file to, as a steering_vector artifact. Exits 1 when the attach fails. |

## `soup sweep`

Run hyperparameter sweep: grid or random search over training parameters.

Usage: `soup sweep [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--config`, `-c` | text | `soup.yaml` |  | Path to base soup.yaml config file |
| `--param`, `-p` | text, repeatable |  |  | Parameter to sweep: key=val1,val2,val3 (e.g., lr=1e-5,2e-5,5e-5) |
| `--sweep-config` | text |  |  | Path to a standalone sweep YAML (strategy/n_runs/seed/params) instead of --param; its strategy, n_runs and seed take precedence over --strategy/--max-runs |
| `--strategy`, `-s` | text | `grid` |  | Search strategy: grid, random |
| `--max-runs` | integer |  |  | Max number of runs (useful for random strategy) |
| `--name`, `-n` | text |  |  | Sweep experiment name prefix |
| `--dry-run` | flag | off |  | Show planned runs without executing |
| `--early-stop` | float |  |  | Stop early if run's loss exceeds best loss by this factor (e.g. 1.5 = 50% worse) |
| `--yes`, `-y` | flag | off |  | Skip confirmation prompt |

## `soup tokenizer`

Tokenizer tools: train a BPE tokenizer from JSONL (v0.53.9).

Usage: `soup tokenizer <command> [ARGS]...`

Subcommands:

- [`train`](#soup-tokenizer-train)

### `soup tokenizer train`

Train a BPE tokenizer from a local corpus.

Usage: `soup tokenizer train [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--input`, `-i` | text |  | yes | Path to JSONL or .txt corpus (must stay under cwd). |
| `--vocab-size`, `-v` | integer | `32000` |  | Target vocabulary size (256-200000). |
| `--output`, `-o` | text | `tokenizer_out` |  | Output directory for tokenizer.json + vocab.json (under cwd). |
| `--min-frequency` | integer | `2` |  | Minimum pair frequency (>=1). |
| `--special-token` | text, repeatable |  |  | Add special token (repeatable). Defaults: &lt;pad>, &lt;s>, &lt;/s>, &lt;unk>. |

## `soup train`

Start training from a soup.yaml config.

Usage: `soup train [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--config`, `-c` | text | `soup.yaml` |  | Path to soup.yaml config file |
| `--name`, `-n` | text |  |  | Experiment name (auto-generated if not set) |
| `--dry-run` | flag | off |  | Validate config and data without training |
| `--resume`, `-r` | text |  |  | Resume from checkpoint: path to checkpoint dir ('auto' for latest); on the MLX backend, a path to a .safetensors adapter file instead. Not supported when training.relora_steps is configured |
| `--wandb` | flag | off |  | Enable Weights &amp; Biases logging |
| `--tensorboard` | flag | off |  | Enable TensorBoard logging (logs to output_dir/runs/) |
| `--tracker` | text |  |  | Experiment tracker: mlflow / swanlab / trackio (v0.43.0). Mutually exclusive with --wandb / --tensorboard. |
| `--deepspeed` | text |  |  | Enable DeepSpeed: zero2, zero3, zero2_offload, zero3_offload (stage 3 + CPU parameter offload), zero++ (ZeRO++), or path to config JSON |
| `--fsdp` | text |  |  | Enable FSDP2: full_shard, shard_grad, or full_offload |
| `--gpus` | text |  |  | GPUs per node for distributed training ('auto' or integer) |
| `--nodes` | integer | `1` |  | Number of machines; each must use the same --gpus count |
| `--node-rank` | integer |  |  | This machine's rank, from 0 to nodes minus 1 (default 0) |
| `--master-addr` | text |  |  | Rank 0 hostname or IP reachable by all nodes; required with --nodes > 1 |
| `--master-port` | integer |  |  | Shared coordinator port for multiple nodes (default 29500) |
| `--no-reexec` | flag | off |  | Print the distributed launch command instead of launching it |
| `--gate` | text |  |  | Enable eval-gated training with a suite file (shortcut for training.eval_gate.enabled=true + suite=&lt;path>) |
| `--push-as` | text |  |  | Auto-push each save_steps checkpoint to HF Hub as 'checkpoint-&lt;step>' branch of the given repo (e.g. user/my-model) |
| `--hf-resume` | flag | off |  | Download the latest checkpoint branch from the --push-as repo and resume from it. Requires --push-as; not supported when training.relora_steps is configured. |
| `--find-lr` | flag | off |  | LR range finder (v0.32.0): run a short geometric LR sweep, write a JSON report with the recommended LR, then exit without training. |
| `--find-lr-start` | float | `1e-07` |  | LR range finder: starting LR (default 1e-7) |
| `--find-lr-end` | float | `0.1` |  | LR range finder: ending LR (default 1e-1) |
| `--find-lr-steps` | integer | `100` |  | LR range finder: number of sweep steps (default 100) |
| `--find-lr-output` | text | `lr_finder.json` |  | LR range finder: JSON report path (default ./lr_finder.json) |
| `--yes`, `-y` | flag | off |  | Skip the start-training confirmation. Required when stdin is not a TTY (scripts, CI, pipes). |
| `--trust-remote-code` | flag | off |  | Allow loading models that ship custom Python via auto_map. Default deny (v0.36.0). Only enable if you trust the source. |
| `--echo-trap-tokenizer-aware` | flag | off |  | Use tokenizer-id n-grams for echo-trap scoring. Requires training.echo_trap_enabled=true on grpo/ppo. |
| `--reward-hack-detector` | text |  |  | Reward-hacking detector for GRPO/PPO: info_rm \| rm_ensemble. Overrides training.reward_hack_detector. (v0.71.26) |
| `--reward-hack-halt` | flag | off |  | Auto-halt training on a HACK verdict. Requires --reward-hack-detector (or training.reward_hack_detector). (v0.71.26) |
| `--replay` | text |  |  | Old dataset to interleave as continual-learning rehearsal, so training on the new task does not erase the previous one. sft/pretrain only; incompatible with packing/multipack. Overrides data.replay. (v0.71.36) |
| `--replay-ratio` | float |  |  | Fraction of the FINAL mixed train set that is replay rows (default 0.1). Overrides data.replay_ratio. (v0.71.36) |
| `--replay-seed` | integer |  |  | Seed for the replay sample + interleave. Overrides data.replay_seed. (v0.71.36) |
| `--reward-hack-mitigation` | text |  |  | Closed-loop reward-hacking mitigation mode: off \| log_only \| kl_control \| pid_lagrangian. Requires training.reward_hack_detector on grpo/ppo. Overrides training.reward_hack_mitigation. (v0.71.26) |
| `--minillm-on-policy` | flag | off |  | Use the TRUE on-policy MiniLLM teacher-mixed rollout (v0.71.18 #257) instead of the offline distribution blend. Requires training.minillm_enabled=true on task='distill'. |
| `--profile` | flag | off |  | Record a torch.profiler trace (Chrome trace JSON) during early training steps. Output: &lt;output>/profiles/&lt;run_id>.trace.json |
| `--allow-oom-attempt` | flag | off |  | Bypass the analytical hardware-fit VRAM gate and launch even when the run is predicted to run out of GPU memory (opt-out). |
| `--diagnose-gate` | text |  |  | After training, run `soup diagnose` against the supplied evidence JSON (or scratch evidence). Refuses to mark the run successful if any failure mode returns MAJOR (exit 2) or NOT_RUN (exit 3). |
| `--annex-xi` | text |  |  | After training, render an EU AI Act Annex XI/XII auto-doc to the given output path (cwd-contained). Markdown body now; PDF in v0.59.1. |
| `--repro-receipt` | text |  |  | After training, write an SR 11-7-style reproducibility receipt (seeds + kernel versions + GPU + OS) to the given path. v0.59.0. |
| `--capture-activations` | text |  |  | After training, capture residual-stream activations at the named decoder layer (e.g. model.layers.5) on --capture-prompts and write them to &lt;output>/activations/activations.json for soup probe sae-diff / sleeper. v0.71.8 #219. |
| `--capture-prompts` | text |  |  | JSONL (or .txt) of prompts to run for --capture-activations (one prompt per line; 'prompt'/'text' field or raw text). |
| `--track-energy` | flag | off |  | Measure the training window's energy + CO2 via codecarbon (offline; requires `pip install soup-cli[carbon]`). Feeds the kWh / CO2 into --annex-xi. v0.71.3. |
| `--energy-country` | text | `USA` |  | ISO 3166-1 alpha-3 country code for the CO2 grid-intensity estimate used by --track-energy (default USA). |
| `--energy-out` | text |  |  | Write the --track-energy measurement to this JSON file (cwd-contained) so `soup bom emit --energy <path>` can consume it. v0.71.15. |
| `--cloud` | text |  |  | Train on a cloud GPU instead of locally (v0.71.18 #16). Supported: modal, lambda, runpod. Renders a cloud app stub from the config (plan-only); use --cloud-submit to submit live. |
| `--gpu` | text | `a100` |  | Cloud GPU type for --cloud (t4 / l4 / a10 / a10g / a100 / a100-80gb / l40s / h100 / a6000). Default a100. Provider-specific allowlists apply. |
| `--cloud-submit` | flag | off |  | With --cloud, submit the rendered run live via the cloud's SDK or API (gated on respective provider token/API key). Default is plan-only (render + print the command). Lambda requires a registered SSH key. |

## `soup tui`

Full-screen terminal dashboard: list runs, inspect metrics, tail logs.

Usage: `soup tui [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--refresh` | float | `1.0` |  | Refresh interval in seconds (0.25 - 10.0). |
| `--limit` | integer | `50` |  | Maximum runs to load into the dashboard. |

## `soup tunability`

Probe-train 6-10 small bases on a held-out slice + report Pareto frontier of (eval delta, train cost, license) (v0.64.0).

Usage: `soup tunability [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--dataset`, `-d` | text |  |  | Path to held-out JSONL slice used for the probe. |
| `--candidates` | text |  |  | Comma-separated subset of default candidates by name. |
| `--probe-steps` | integer | `100` |  | Steps per candidate probe (10-10000). |
| `--holdout-size` | integer | `64` |  | Number of holdout rows used for delta scoring (10-100000). |
| `--output`, `-o` | text |  |  | Optional path for the JSON report. |
| `--plan-only` | flag | off |  | Print planned candidates + cost estimate, exit without probing. |
| `--list` | flag | off |  | List built-in candidate catalogue + exit. |
| `--live` | flag | off |  | Run a LIVE LoRA probe per candidate (loads each repo + trains --probe-steps on a tiny held-out slice). Without it, a deterministic offline heuristic is used. |
| `--device` | text |  |  | Device for the live probe (cuda / cpu). Auto-detected when omitted. |

## `soup ui`

Launch the Soup Web UI.

Usage: `soup ui [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--port`, `-p` | integer | `7860` |  | Port to serve on |
| `--host` | text | `127.0.0.1` |  | Host to bind to |
| `--no-browser` | flag | off |  | Don't open browser automatically |
| `--show-token` | flag | off |  | Print the auth token and exit (for scripting) |
| `--public` | flag | off |  | Bind to 0.0.0.0 so the UI is reachable from a phone on the same LAN. Prints a phone-scannable URL + ASCII QR at startup. Loopback HTTP is auto-upgraded to require a Bearer token in the query string. v0.53.9 #95. |
| `--auth-token` | text |  |  | Override the auto-generated Bearer token (16-128 urlsafe base64 chars). Useful for stable phone bookmarks. When omitted, a fresh token is generated each startup. |

## `soup version`

Show Soup CLI version.

Usage: `soup version [OPTIONS]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `--full`, `-f` | flag | off |  | Show system info and extras |
| `--json` | flag | off |  | Output in JSON format |

## `soup why`

Explain training anomalies for a run in plain English.

Usage: `soup why [RUN_ID]`

| Name | Type | Default | Required | Help |
|---|---|---|---|---|
| `RUN_ID` | text |  |  | Run ID (or prefix). Defaults to the most recent run. |
