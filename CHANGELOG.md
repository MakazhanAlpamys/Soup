# Changelog

All notable changes to **Soup CLI** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Detailed, per-release notes for every published version live on the
[GitHub Releases page](https://github.com/MakazhanAlpamys/Soup/releases). This
file tracks unreleased changes and links out for historical detail rather than
reproducing 70+ versions of notes.

## [Unreleased]

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
