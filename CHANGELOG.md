# Changelog

All notable changes to **Soup CLI** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Detailed, per-release notes for every published version live on the
[GitHub Releases page](https://github.com/MakazhanAlpamys/Soup/releases). This
file tracks unreleased changes and links out for historical detail rather than
reproducing 70+ versions of notes.

## [Unreleased]

## [0.71.5] - 2026-06-02

### Added
- **`soup eval against` now reads eval metrics** — `ExperimentTracker.get_metric_series`
  falls back to the `eval_results` table when the metric is not a per-step
  training column (`loss` / `lr` / `grad_norm` / `speed` / `gpu_mem`). So
  `soup eval against <base> --candidate <run> --metric task_accuracy` returns a
  real score series (benchmark scores live in `eval_results`, not `metrics`)
  instead of "Empty series". Per-step columns still read from `metrics` — no
  regression for existing callers.
- **`soup advise` learns from past project outcomes** — `soup advise` now reads
  this project's accepted-verdict history (`~/.soup/advise_history.jsonl`) and
  biases the rubric: 3+ successful SFT precedents flip a marginal RAG call to
  SFT; 3+ negative GRPO outcomes suppress GRPO in favour of SFT-on-traces; an
  encouraged choice gets a small confidence nudge. Scoped per-project (one
  project's record never biases another). No history → identical to before.
- **Slack/Discord webhooks on four more commands** — `--slack-url` / `--discord-url`
  (SSRF-hardened, loopback-only HTTP, RFC1918 rejected, never crashes the
  command) now ship on `soup ingest`, `soup prune-prompt`, `soup ab` (fires only
  on a `reject_h0` / `accept_h0` decision, not `continue`), and
  `soup data active-sample` — not just `soup drift-alarm`. The validator + sender
  moved to a shared `soup_cli/utils/webhooks.py`.
- **Tokenizer-aware `soup prune-prompt`** — `--tokenizer <model_or_path>` detects
  and strips the shared system-prompt prefix on **token** boundaries instead of
  characters, so a multi-byte UTF-8 prefix can never be split mid-code-point.
  Default (no `--tokenizer`) keeps the whitespace-character behaviour.
- **Curriculum bucketing by loss percentile** — `DynamicCurriculumCallback` now
  buckets samples by the percentile rank of the live loss (or perplexity)
  signal within a rolling window when `data.curriculum_metric` is `loss` /
  `perplexity`, so a consistently-hard sample is routed to the same difficulty
  bucket across recomputes. `length` and warm-up still use round-robin.
- **`--hub` on `soup data push` and `soup data forge`** — `soup data push
  --hub modelscope|modelers` uploads a dataset via the matching SDK
  (`repo_type=dataset`, commit message sanitised); `soup data forge --hub
  <non-hf> --teacher owner/name` pre-fetches the teacher model from that hub
  (and warns when the teacher is not a repo id so `--hub` is never silently
  ignored). HF stays the default.

### Notes
- Live SaaS *pull* adapters for `soup ingest` (Langfuse / LangSmith / Helicone /
  OpenPipe / OpenAI SDKs, issue #204) remain deferred: they need credentialed
  vendor accounts with populated trace data to validate honestly. Tracked as an
  open, `infra-blocked` (external-account) item. `soup ingest` continues to parse
  the JSONL export you pull from your dashboard.

## [0.71.4] - 2026-06-02

### Added
- **Live canary verdict for `soup adapters merge`** — `--canary <suite.json>`
  scores the merged adapter against the first input and classifies
  **OK / MINOR / MAJOR** using the Quant-Lobotomy taxonomy (drop <2% OK, <5%
  MINOR, else MAJOR). `--strict-verdict` exits 2 on MAJOR. Pre-scored
  `{"baseline_scores","candidate_scores"}` suites run with no model load; a
  `{"tasks":[...]}` suite uses an injectable scorer. Replaces the v0.57 `UNKNOWN`
  stub.
- **Live evolutionary merge** — `soup adapters merge --strategy cmaes --eval
  <suite> --budget <t>` now runs the full CMA-ES loop: each candidate is merged,
  materialised, scored against the eval suite, and the best-weighted merge is
  written to `--output`. Replaces the v0.67 plan-only stub.
- **Publish an adapter PR to GitHub** — `soup adapters pr <title> --base-sha
  <hex> --adapter <path> --push owner/repo#N` posts the rendered PR Markdown as a
  GitHub PR comment via `gh api` (argv-list, body over JSON stdin; no shell).
  Token resolves from `GITHUB_TOKEN` / `GH_TOKEN`.
- **Pre-wired `soup loop` production stages** — `soup loop init --pre-wired` (or
  `soup loop watch --pre-wired`) swaps the v0.58 no-op stage stubs for real
  harvest (traces → preference pairs) → DPO train → eval-gate → canary-deploy
  callables. `soup loop status` now shows the `pre_wired` flag.
- **Loop iterations as Soup Cans + Registry lineage** — `soup loop watch
  --pack-cans` packs each successful iteration as a v0.26 Soup Can and appends a
  Registry entry (tag `loop-iter`), chaining a real lineage DAG across
  iterations visible through `soup history`. `soup loop replay <id> --extract
  <dir>` unpacks a recorded iteration.
- **Branch pointers into the Registry** — `soup adapters branch <name>
  --attach-to-registry <id>` links a branch snapshot to a Registry entry (shown
  as a `branches` node in `soup history`); `soup adapters branch <name>
  --from-registry <id>` derives a fresh snapshot's config + base from an entry.

### Security
- The backdoor-scan gate (v0.71.2 #192) and license-conflict gate (v0.60 Part E)
  now run for **all** merge strategies, including `--strategy cmaes` (previously
  bypassed because cmaes returned before the gates).
- `soup loop` canary deploy restricts `SOUP_LOOP_SERVE_ENDPOINT` to loopback /
  RFC1918-private hosts (a serve endpoint is the operator's own box/LAN), beyond
  the general webhook SSRF policy which permits any HTTPS host.
- `soup adapters pr --push` builds the `gh` child environment from an allowlist
  so unrelated secrets (`HF_TOKEN` / `OPENAI_API_KEY` / …) never reach the
  subprocess.
- The canary-suite JSON read uses `O_NOFOLLOW` + `os.fstat` (size cap enforced on
  the same fd) to close the symlink/size-cap TOCTOU window.

## [0.71.3] - 2026-06-01

### Added
- **Energy & CO2 measurement for training** — `soup train --track-energy` wraps
  the training window in a codecarbon **offline** tracker (no IP-geolocation
  network call) and reports kWh / CO2 / grid intensity, feeding those numbers
  into `--annex-xi`. New `EnergyTracker` context manager; graceful no-op when
  codecarbon is absent (`pip install soup-cli[carbon]`). `--energy-country`
  picks the ISO-3166 alpha-3 grid for the CO2 estimate (default `USA`).
- **PDF Annex XI/XII documents** — `soup train --annex-xi report.pdf` now renders
  a reportlab PDF (a `.md` path still renders markdown). `pip install
  soup-cli[pdf]`.
- **Auto-populated training-corpus domains in Annex XI/XII** — the top crawled
  domains (with shares) are now extracted from the training JSONL and listed in
  the EU AI Act docs, replacing the previous empty placeholder.
- **Soup Can manifest v3 with embedded attestations** — `soup can pack --attest
  <statement.json>` (repeatable) embeds in-toto Statements into a v3 can
  manifest; v1/v2 cans still load. Each statement is shape- and size-validated.
- **Local audit log auto-instrumentation** — every `soup` command now appends one
  HIPAA/SOC2-shaped record to `~/.soup/audit.jsonl` (secrets redacted, args
  capped). Opt out per-invocation with `--no-audit-log` or globally with
  `SOUP_NO_AUDIT_LOG=1`. Tail/rotate with `soup audit-log`.
- **Reproducibility receipt in airgap bundles** — `soup airgap-bundle
  --repro-receipt <receipt.json>` embeds an SR 11-7 receipt as
  `repro-receipt.json`; auto-detected from `<model>/repro-receipt.json` when not
  supplied.

### Security
- `soup can pack --attest` now rejects oversize attestation files by their raw
  size *before* parsing them into memory (defence against memory-exhaustion).
- The new file-loading paths (attestation JSON, airgap receipt, training-corpus
  scan, PDF write) are all cwd-contained + TOCTOU symlink-rejected and
  size-capped; the audit auto-log redacts `hf_`/`sk-`/`Bearer` tokens and never
  crashes the CLI on a broken log.

## [0.71.2] - 2026-06-01

### Added
- **ed25519 signing for `soup adapters sign` / `soup attest`** — real detached
  signatures (over the adapter Merkle root / the in-toto statement) via a new
  `[sign]` extra (`pip install soup-cli[sign]`, pulling `cryptography`).
  `soup adapters sign --backend ed25519 --key <priv.pem>` (or `--generate-key
  <out.pem>`, or `SOUP_SIGNING_KEY`); `soup adapters verify [--public-key
  <trusted.pem>]` does a cryptographic verify and, with a trusted key, genuine
  authentication. `soup attest emit --sign ed25519 --key <priv.pem>` writes a
  `<output>.sig` sidecar; new `soup attest verify <statement> --signature <sig>`
  verifies it (canonical-JSON, so it's platform/newline-independent). Sigstore
  keyless signing stays infra-blocked (needs an OIDC provider + Fulcio/Rekor
  network — can't be honestly validated offline).
- **Anti-AI-Jacking namespace pin on Hub downloads** — HF model fetches now
  consult a trust-on-first-use pin store: a repo whose author changes (or whose
  `created_at` jumps backward) is refused unless the namespace shift is explicitly
  allowed. Fails open when repo metadata is unavailable.
- **License auto-detection at `soup adapters merge`** — when `--license` isn't
  given, the license is read from each adapter's `adapter_config.json` /
  `config.json` / model-card frontmatter (HF `llama3.1`-style ids mapped to
  canonical) and the conflict gate runs automatically.
- **Backdoor-scan gate at `soup adapters merge`** — refuses to merge any input
  whose `soup adapters scan` returns FAIL (or can't be scanned) unless
  `--allow-unscanned` is passed; WARN is advisory.

### Changed
- License-conflict overrides (`--license-override <reason>`) are now recorded to
  the audit log for legal review.
- The namespace-pin store now uses SQLite WAL + busy-timeout and a cross-process
  file lock around its get+insert, so concurrent writers don't lose the trust
  anchor.

### Security
- ed25519 verification fails closed (any tamper / wrong key / missing key ⇒
  invalid). Signing keys + trusted public keys are symlink-rejected and
  size-capped via a shared reader (no cwd-containment — keys live outside the
  project). `--generate-key` refuses to overwrite any existing path.

## [0.71.1] - 2026-06-01

### Added
- `soup env fix` — render a reproducible install plan from `soup-env.lock`.
  Emits copy/paste `uv pip install` commands (`--format uv-pip`, default) or a
  `requirements.txt` body (`--format requirements`); `--output` optionally writes
  a `requirements.txt` under cwd. Print-only by design — never shells out to a
  package manager.
- `soup lock write --env-lock <path>` — auto-derive `--env-hash` from a
  `soup-env.lock` so operators who ran `soup env lock` don't copy the hash by
  hand. `--env-hash` still wins when passed explicitly.
- `soup serve --record-thumbs <db>` — capture thumbs-up/down feedback into a
  local-RL SQLite at startup, plus a new `POST /v1/thumbs` endpoint (transformers
  backend). Returns 404 when the flag isn't set.
- Judge-calibration persistence: `JudgeCalibrationReport.to_dict`,
  `write_judge_calibration`, and `load_judge_calibration`, backed by a new
  `judge_calibration` registry artifact kind. Loading re-validates the report so
  a corrupt on-disk field is rejected.
- Bundled MUSE and WMDP unlearning eval fixtures so
  `soup eval unlearning --benchmark muse|wmdp` runs out of the box. WMDP
  forget-set probes ship **redacted** (placeholder prompts + `REFUSED` responses)
  — Soup never ships verbatim hazardous content.

### Changed
- `soup completions` now introspects a cached base model's actual LoRA target
  modules (config-only `AutoConfig` load, `local_files_only=True`, never networks
  or raises) and falls back to the canonical default shape when the base isn't
  cached locally.
- `build_dag` exposes a `validate_build_source` helper (cwd-containment +
  symlink rejection) for build-manifest source paths.

## [0.71.0] - 2026-06-01

### Changed
- **Breaking — install split.** The heavy training stack (`torch`,
  `transformers`, `peft`, `trl`, `datasets`, `bitsandbytes`, `accelerate`) moved
  out of the core install into a new `[train]` extra. `pip install soup-cli` is
  now a light CLI + data-tools install with **no PyTorch**; run
  `pip install 'soup-cli[train]'` (or `[all]`) to fine-tune. Existing users who
  train must reinstall with `[train]`. Version pins are unchanged.
- Trimmed `README.md` to a ~238-line front door; the full feature reference now
  lives under `docs/` (one topic page per area, indexed from the README).
- Raised the pytest coverage gate from 50% to 77% (`--cov-fail-under=77`).
- Migrated to a `src/` layout (`src/soup_cli/`) for cleaner packaging and to
  stop tests accidentally importing the in-tree package.

### Added
- `[train]` and `[all]` optional-dependency extras (`[all]` pulls
  `train`, `serve`, `ui`, `data`). `[dev]` self-references `[train]` so CI and
  contributors still get the full stack from `pip install -e ".[dev]"`.
- Friendly error mapping: a missing heavy dependency (`torch`, `transformers`,
  `peft`, `trl`, `datasets`, `bitsandbytes`, `accelerate`) now surfaces
  "Training needs the [train] extra. Run: pip install 'soup-cli[train]'".
- `py.typed` marker (PEP 561) so downstream type checkers pick up Soup's inline
  type hints.
- `.pre-commit-config.yaml` with ruff (lint + format) and standard file-hygiene
  hooks.
- Lenient `mypy` configuration and a non-blocking `type-check` CI job.
- This `CHANGELOG.md`.

### Removed
- The historical, per-version security-fix log that had grown inside
  `SECURITY.md` (~220 KB). `SECURITY.md` is now a concise security policy; the
  detailed hardening notes remain in git history and the GitHub Releases notes.

[Unreleased]: https://github.com/MakazhanAlpamys/Soup/compare/v0.71.0...HEAD
[0.71.0]: https://github.com/MakazhanAlpamys/Soup/compare/v0.70.0...v0.71.0
