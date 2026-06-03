# Adapters, Registry & Governance

[← Back to the Soup README](../README.md)

> Adapter lifecycle/management, the model registry, Soup Cans, the data flywheel (`soup loop`), knowledge editing, activation steering, and the supply-chain / governance surfaces (scan, sign, BOM, attest, audit, airgap).

**Contents:**

- [Adapter Lifecycle (`soup adapters {merge,pr,bisect}`, `soup lock`)](#adapter-lifecycle-soup-adapters-mergeprbisect-soup-lock)
- [Data Flywheel (`soup loop`)](#data-flywheel-soup-loop)
- [Knowledge Editing (`soup edit set`, ROME / MEMIT / AlphaEdit)](#knowledge-editing-soup-edit-set-rome--memit--alphaedit)
- [Activation Steering (`soup steer`)](#activation-steering-soup-steer)
- [GRACE Codebook — Lifelong Knowledge Edits](#grace-codebook--lifelong-knowledge-edits)
- [Model Registry & Lineage](#model-registry--lineage)
- [Adapter Management (git for LoRA)](#adapter-management-git-for-lora)
- [Soup Cans (Shareable Recipes)](#soup-cans-shareable-recipes)
- [Bill of Materials (`soup bom emit`)](#bill-of-materials-soup-bom-emit)
- [Provenance Attestations (`soup attest emit`)](#provenance-attestations-soup-attest-emit)
- [EU AI Act Annex XI/XII Auto-Doc (`soup train --annex-xi`)](#eu-ai-act-annex-xixii-auto-doc-soup-train---annex-xi)
- [Audit Log (`soup audit-log`)](#audit-log-soup-audit-log)
- [Reproducibility Receipt (`soup train --repro-receipt`)](#reproducibility-receipt-soup-train---repro-receipt)
- [Adapter Backdoor Scanner (`soup adapters scan`)](#adapter-backdoor-scanner-soup-adapters-scan)
- [Adapter Sign + Verify (`soup adapters sign` / `verify`)](#adapter-sign--verify-soup-adapters-sign--verify)
- [Strict Safetensors Mode (`soup adapters check-safetensors`)](#strict-safetensors-mode-soup-adapters-check-safetensors)
- [Namespace Pinning (Anti-AI-Jacking)](#namespace-pinning-anti-ai-jacking)
- [License-Conflict Matrix at Merge](#license-conflict-matrix-at-merge)
- [Airgap Bundle (`soup airgap-bundle`)](#airgap-bundle-soup-airgap-bundle)

---

## Adapter Lifecycle (`soup adapters {merge,pr,bisect}`, `soup lock`)

v0.57 shipped `adapters diff / merge / blame / branch`. v0.67 finishes the lifecycle: evolutionary merge driven by your eval, GitHub-shaped PRs for adapter review, a shared `soup.lock` for team reproducibility, and binary-search bisect over training history.

```bash
# 1. Evolutionary merge: search the simplex of merge weights via CMA-ES.
#    v0.71.4 makes this a LIVE loop — each candidate is merged, scored against
#    --eval, and the best blend is written to --output.
soup adapters merge \
  adapter-finance/ adapter-medical/ adapter-legal/ \
  --strategy cmaes \
  --eval evals/domain_mix.yaml \
  --budget 1h \
  --population 8 \
  --max-generations 20 \
  --output merged/

# 1b. One-shot merge with a live canary verdict (v0.71.4). Scores the merged
#     adapter vs the first input → OK / MINOR / MAJOR; --strict-verdict exits 2
#     on a MAJOR regression. A pre-scored canary suite needs no model load.
soup adapters merge adapter-a/ adapter-b/ -o merged/ \
  --canary evals/canary.json --strict-verdict

# 2. Render the merge as a GitHub PR for review (eval deltas + sample diffs).
soup adapters pr "merge: 3-domain blend" \
  --base-sha $(git rev-parse HEAD) \
  --adapter merged/ \
  --eval evals/deltas.json \
  --samples evals/samples.json \
  --dataset-diff data/diff.txt \
  --format markdown -o pr.md

# 2b. Post it straight to a GitHub PR comment (v0.71.4). Auth via GITHUB_TOKEN
#     / GH_TOKEN; uses `gh api` under the hood (no secret leaks to the child).
soup adapters pr "merge: 3-domain blend" \
  --base-sha $(git rev-parse HEAD) --adapter merged/ \
  --eval evals/deltas.json --push your-org/your-repo#42

# 3. Lock a reproducible run state. Closure = sha(base + dataset + env).
soup env lock                                     # v0.64 — capture env hash
soup lock write \
  --base-model meta-llama/Llama-3.1-8B \
  --base-sha $BASE_SHA \
  --dataset-sha $DATA_SHA \
  --env-hash $(jq -r .closure soup-env.lock) \
  -o soup.lock
# Teammates re-check the lock; exit 3 on drift.
soup lock check soup.lock \
  --base-model meta-llama/Llama-3.1-8B \
  --base-sha $BASE_SHA --dataset-sha $DATA_SHA --env-hash $ENV_HASH

# 4. Bisect a training history to find the step that broke an eval.
soup adapters bisect \
  ckpt-step-100 ckpt-step-200 ckpt-step-400 ckpt-step-800 \
  --eval-command "soup eval custom --model {ckpt} --tasks eval.jsonl" \
  -o bisect.json
# Exits 3 on BROKEN_AT — pipe into `soup adapters blame` for attribution.
```

CMA-ES is pure-Python (no `cma` dependency); the eval is operator-supplied via a closure so any scoring code works. PR rendering escapes Markdown table cells, so crafted metric names cannot inject table rows or links. The lockfile composes with v0.64 `soup env lock` — drift in any of `{base_model, base_model_sha, dataset_sha, env_hash, closure_sha}` exits 3 (`soup_version` and `created_at` are advisory-only). Bisect uses `shlex.split` + `shlex.quote(ckpt)` in argv-list mode (no `shell=True`), so checkpoint ids cannot inject shell metacharacters.

VeRA / VB-LoRA bank storage (`soup_cli.utils.vector_bank`) and MoLE per-token routing (`task='moe_lora_routing'`) ship as schema-only in v0.67.0 — live multi-tenant serving and gating-kernel training land in v0.67.1.


## Data Flywheel (`soup loop`)

The full *production traces → preference pairs → Eval-Gated DPO → canary deploy → rollback* loop, driven from a single CLI. Connects v0.26 Trace-to-Preference + Eval-Gated Training + Registry lineage + Quant-Lobotomy verdicts + Soup Cans + v0.25 Autopilot + v0.54 Advise + v0.55 Eval Design + v0.56 Diagnose.

```bash
# One-time setup
soup loop init registry://abc12 --eval evals/lock.json --baseline registry://prod \
    --monthly-budget 50usd --max-runs-per-day 3

# Inspect counters + status
soup loop status

# Run the daemon (foreground)
soup loop watch --poll-interval 300

# Pre-wired production stages (v0.71.4): real traces → DPO → eval-gate → canary,
# instead of the v0.58 no-op stubs. Opt in once via `loop init --pre-wired` or
# per-run via --pre-wired. Pack each iteration as a shareable Soup Can with
# Registry lineage via --pack-cans.
soup loop init registry://abc12 --eval evals/lock.json --baseline registry://prod --pre-wired
soup loop watch --pre-wired --pack-cans --poll-interval 300

# Unpack a recorded iteration's Soup Can for what-if analysis (v0.71.4)
soup loop replay iter-20260515T120000-abcdef01 --extract ./iter-dump

# Background subprocess (writes PID, no shell)
soup loop watch --detach

# Promote a canary at 5% traffic with auto-rollback on MAJOR verdict
soup loop canary registry://candidate --traffic 5% --autoroll-on-regress

# Pause/resume the daemon between iterations (atomic state flip)
soup loop pause
soup loop resume

# Replay any recorded iteration
soup loop replay iter-20260515T120000-abcdef01
```

State lives in `.soup/loop.yaml` (atomic write, cwd-contained, symlink-rejected). Per-iteration manifests under `.soup-loops/<iter-id>/iteration.json` are laid out so a v0.26 Soup Can can wrap them directly. The canary router is deterministic (SHA-256 hash of conversation id) and sticky-on-rollback — a flaky verdict can't ping-pong traffic between adapters.


## Knowledge Editing (`soup edit set`, ROME / MEMIT / AlphaEdit)

Surgical factual patches WITHOUT a full fine-tuning loop. Hospital data team correcting a misattributed drug interaction, lab fixing a wrong historical date, security team responding to a hallucinated CVE — all one CLI invocation.

```bash
# Live edit: optimises a residual + applies a rank-1 weight update, then saves.
soup edit set \
  --base HuggingFaceTB/SmolLM2-135M \
  --method rome \
  --subject "The capital of France is" \
  --target "Lyon" \
  --output ./edited --device cuda

# Plan-only mode validates the request + prints the resolved EditPlan + exits 0.
soup edit set --base ./model --method rome --subject "..." --target "..." --plan-only

# Diff what the model says before vs after the edit (live generation).
soup edit diff before after \
  --before-model ./base --after-model ./edited \
  --probes probes.jsonl --output diff.json
```

The kernels are a covariance-free (`C = I`) variant of the ROME family — well-defined, tractable on a 4 GB box, and validated on SmolLM2-135M (a ROME edit moved `P("Lyon" | "The capital of France is")` from 0.0016 → 0.96). **ROME** edits a single layer; **MEMIT** distributes the residual across a layer band; **AlphaEdit** projects the update orthogonal to the down-proj's top singular direction.

The sequential edit governor is persisted (SQLite, cross-process-locked, `SOUP_EDIT_GOVERNOR_DB` override) so the per-base-model edit count + norm-blowup verdict survive across separate `soup edit set` runs. `soup edit set` consults it automatically: it refuses BEFORE the model load past the per-base cap or after a BLOWUP verdict, and records the measured `||ΔW||_F` after each edit. Pass `--no-governor` to opt out. `--registry-id <id>` attaches the edited model (or GRACE codebook) into the Registry lineage.


## Activation Steering (`soup steer`)

Sometimes you don't want to retrain — you want to *push* the model along a learned direction at decode time. Soup ships three control-vector backends:

- **CAA** (Contrastive Activation Addition) — add a contrastive vector to the residual stream.
- **ITI** (Inference-Time Intervention) — shift specific attention heads along a learned direction.
- **RepE** (Representation Engineering) — PCA-based direction in the residual stream.

```bash
# Train a steering vector from contrastive (positive, negative) prompt pairs
soup steer train --base meta-llama/Llama-3.1-8B-Instruct \
                 --method caa --name safety-v1 \
                 --pairs ./data/pairs.jsonl

# Apply at decode time via soup serve
soup serve --model ./adapter --steer safety-v1 --steer-strength 1.5

# List locally-stored steering vectors
soup steer list
```

Steering names are validated against a strict regex (`^[A-Za-z0-9][A-Za-z0-9._\-]{0,127}$` — no path separators, no shell metacharacters); strength is bounded `|s| <= 10.0`. The trained vectors land in the Soup Registry under the `steering_vector` artifact kind so lineage is preserved.


## GRACE Codebook — Lifelong Knowledge Edits

Vanilla ROME / MEMIT degrade after dozens of sequential edits — the model's norms blow up. GRACE (Hartvigsen et al., 2023) stores each edit in a discrete latent codebook so thousands of sequential patches survive:

```bash
soup edit set --base ./model --method grace \
              --subject "The CEO of Acme is" --target "Jane Doe"
```

```yaml
# Or via soup.yaml when training a model with GRACE-aware lookups
training:
  grace_codebook: true
  grace_codebook_size: 1024    # codebook entries (max 100k)
  grace_codebook_dim: 768      # residual-stream width
```

`grace` joins the existing `rome` / `memit` / `alphaedit` allowlist on `soup edit set`; the sequential edit governor still gates the call when the per-base-model edit count or norm-blowup verdict trips. GRACE is live: `soup edit set --method grace --output ./ckpt` captures the residual key at the subject's last token, optimises a replacement value, and appends a `(key, value)` triple to a `grace_codebook.json` sidecar (atomic, cwd-contained). At inference the codebook is applied via a forward hook that substitutes the residual whenever it falls within an epsilon ball of a stored key — so the base weights are never modified and thousands of edits survive without norm blowup.


## Model Registry & Lineage

Every fine-tune you ship should be reproducible. Soup's local registry (`~/.soup/registry.db`) tracks each entry by a content hash of its config + data + base model, plus lineage pointers to parent entries.

```bash
# Register a completed run
soup registry push --run-id run_202611_abc123 --name llama31-chat --tag v1

# List entries (filter by name, tag, base model, task)
soup registry list
soup registry list --name llama31-chat --tag prod

# Show full details: config, eval baseline, artifacts, ancestors
soup registry show llama31-chat-v1

# Side-by-side config diff + eval delta between two entries
soup registry diff llama31-chat-v1 llama31-chat-v2

# Full-text search across name / base model / task / notes
soup registry search "medical reasoning"

# Promote an entry (add a tag, e.g. "prod")
soup registry promote llama31-chat-v1 --tag prod

# Delete (cascades to artifacts + lineage links)
soup registry delete llama31-chat-v1 --yes
```

**Lineage DAG** — every entry can point to a parent (its ancestor run). Walk the DAG for any name with:

```bash
soup history llama31-chat
```

**Refs resolve flexibly** — you can use a registry ID, a name (latest), or `name:tag`. Ambiguous prefixes raise an error rather than silently picking the wrong entry. Registry files are stored with `600` perms on POSIX; override the path with `SOUP_REGISTRY_DB_PATH`.


## Adapter Management (git for LoRA)

`soup adapters` is the git-for-LoRA surface: weight-aware diff, four merge strategies, leave-one-out blame, and SHA-256 branch snapshots. All commands operate on `adapter_model.safetensors` directories (peft-compatible).

```bash
# Per-layer ΔW Frobenius diff + effective-rank drift + top-K changed projections
soup adapters diff ./run-v17 ./run-v18

# Machine-readable JSON for CI
soup adapters diff ./run-v17 ./run-v18 --format json --output diff.json

# Weighted merge with linear / ties / dare / svd strategies
soup adapters merge ./run-v17 ./run-v18 ./run-v19 -o ./merged --strategy ties \
  --weights 0.5,0.3,0.2 --density 0.2

# DARE merge (deterministic via --seed)
soup adapters merge ./run-v17 ./run-v18 -o ./merged --strategy dare \
  --density 0.5 --seed 42

# Leave-one-out ablation plan against a 4-hour wall-clock budget
soup adapters blame ./run-v18 --dataset train.jsonl --layer q_proj.7 \
  --budget 4h --shards 10 --plan-only

# Snapshot a training environment as a comparable branch
soup adapters branch v18 --config soup.yaml --base meta-llama/Llama-3.1-8B \
  --dataset train.jsonl

# Link a branch into the Registry lineage DAG (v0.71.4) — shows as a
# `branches` node under the entry in `soup history`.
soup adapters branch v18 --config soup.yaml --base meta-llama/Llama-3.1-8B \
  --attach-to-registry reg_20260601_abc123

# Or derive a fresh snapshot's config + base straight from a Registry entry (v0.71.4)
soup adapters branch v18-from-reg --from-registry reg_20260601_abc123

# Restore the snapshot's config (refuses if source SHA drifted)
soup adapters checkout v18 --output soup.yaml

# List all snapshotted branches
soup adapters branches
```

**Four merge strategies (pure numpy, no torch import at module level):**

| Strategy | Math | Use case |
|----------|------|----------|
| `linear` | Weighted average per layer | Baseline; tasks share a basis |
| `ties` | Trim by density → elect majority sign → disjoint average | Conflicting task adapters (Yadav et al. 2023) |
| `dare` | Random drop with `density` + rescale `1/density`, then average | Sparse-merge; reduces parameter interference (Yu et al. 2024) |
| `svd` | Linear-merge → low-rank reconstruction via SVD (`--rank`) | Constrain effective rank of the merged delta |

**Defaults & safety:**

- Output paths are containment-checked under cwd and reject pre-placed symlinks (TOCTOU defence).
- Safetensors writes are atomic via `tempfile.mkstemp` + `os.replace` — a crash mid-write never leaves a partial adapter at the target path.
- `.bin` (PyTorch pickle) adapter format is rejected with an explicit "re-save as safetensors" message.
- Branch pointers live under `~/.soup/branches/` (override via `SOUP_BRANCHES_DIR`, constrained to `$HOME` / `$CWD` / `$TMPDIR`).
- `soup adapters checkout` SHA-checks the source config — refuses to restore when the source has drifted from the snapshot, so reproducibility never silently lies.

**v0.66.0:** `soup adapters blame` is now LIVE — the v0.57 `NotImplementedError` stub (#171) is lifted via a DataInf-style influence-function approximation. Pass `--top-k 50` to control the reported top-influencer count; pass a real `probe_fn` (Python API) to feed real gradients, or use the default deterministic synthetic probe for offline planning.

**v0.71.4:** the merge verdict is now LIVE — `soup adapters merge … --canary suite.json` lifts the `MergeReport.verdict` `UNKNOWN` stub. A pre-scored `{"baseline_scores","candidate_scores"}` suite classifies the blend OK / MINOR / MAJOR with no model load; a `{"tasks":[...]}` suite uses an injectable scorer. `--strict-verdict` exits 2 on MAJOR. The backdoor-scan and license-conflict gates now run for **every** strategy, including `cmaes`. `soup adapters branch <name> --from-registry <id>` / `--attach-to-registry <id>` link training-env snapshots into the Registry lineage DAG (shown as a `branches` node in `soup history`).


## Soup Cans (Shareable Recipes)

Share a reproducible recipe as a single `.can` file — a tarball of the manifest, full config, and a reference to the training data (URL or HF dataset). Not the weights, not the dataset bytes: just enough for someone else to re-run the same training.

```bash
# Pack a registry entry into a .can
soup can pack --entry-id llama31-chat-v1 --out ./llama31-chat.can

# Embed in-toto attestations into the manifest (v3 cans; repeatable)
soup can pack --entry-id llama31-chat-v1 --out ./signed.can --attest att.json

# Preview the manifest without extracting
soup can inspect ./llama31-chat.can

# Verify schema + config parseability
soup can verify ./llama31-chat.can

# Fork with modifications (dotted-path overrides) and re-pack
soup can fork ./llama31-chat.can --out ./llama31-chat-hot.can \
  --modify training.lr=5e-5 --modify training.epochs=5

# Run a .can end-to-end: extract → train (→ optional deploy)
soup can run ./llama31-chat.can --yes
soup can run ./llama31-chat.can --yes --deploy --env-capture ./env.txt

# Publish a .can to HF Hub as a dataset
soup can publish ./llama31-chat.can --hf-hub me/llama31-chat-recipe
```

**Security** — tar extraction uses `filter="data"` on Python 3.12+ with symlink/hardlink rejection fallback for older runtimes. Size cap: 100 MB. `DataRef.url` must be HTTPS. Fork overrides reject dunder keys (`__class__`, `__init__`) and null bytes. Manifest format version supports `1` and `2` (additive bump in v0.33.0 added `deploy_targets`). `soup can run` requires `--yes` (mandatory consent — auto-downloads data + auto-trains). `soup can publish` validates `repo_id` and resolves the HF token via env / cache files; commit messages are first-line + 200-char capped.



## Bill of Materials (`soup bom emit`)

Emit a **CycloneDX 1.6 ML-BOM** or **SPDX 2.3 + AI profile** bill of materials from any
training run. Procurement teams and compliance auditors can ingest the BOM directly into
their existing tooling — no custom parser required.

```bash
soup bom emit \
  --name adapter-v1 --version 0.1.0 \
  --base-model meta-llama/Llama-3.1-8B \
  --base-sha aaaa...64hex \
  --config-sha bbbb...64hex \
  --task sft --license apache-2.0 \
  --format both --output bom
# writes bom.cdx.json + bom.spdx.json
```

Root component is `type=machine-learning-model` (per CycloneDX ML-BOM extension). Base
model + parent adapters + per-artifact files appear as components with SHA-256 hashes.
License chain uses SPDX identifiers. Energy + CO₂ properties (when attached via the
energy schema) ship under `metadata.properties`.


## Provenance Attestations (`soup attest emit`)

Emit an **in-toto v1 Statement** wrapping a **SLSA-3 provenance v1 predicate** for each
Soup Can lifecycle stage:

```bash
soup attest emit \
  --stage train \
  --subject adapter-v1 \
  --sha aaaa...64hex \
  --builder soup-cli@0.59.0 \
  --output att.json
```

Stages are a closed allowlist: `extract` / `train` / `eval` / `export` / `publish`.
Subject SHA must be 64-hex (sha256). The default `--sign unsigned` backend ships now;
the **`ed25519` backend is live** (`pip install soup-cli[sign]`):

```bash
soup attest emit --stage train --subject adapter-v1 --sha aaaa...64hex \
  --sign ed25519 --key signing.pem --output att.json   # writes att.json.sig
soup attest verify att.json --signature att.json.sig                 # exit 0 valid
soup attest verify att.json --signature att.json.sig --public-key trusted.pub
```

`verify` re-canonicalises the statement JSON (so it's platform/newline-independent)
and checks the ed25519 signature — exit 3 on tamper, key mismatch, or a sidecar with
no public key. A valid signature proves the signer *asserted* the statement; it does
not re-verify the subject digest against an artifact. Sigstore keyless signing remains
infra-blocked (needs an OIDC identity provider + Fulcio/Rekor network).


## EU AI Act Annex XI/XII Auto-Doc (`soup train --annex-xi`)

Render an EU AI Act Annex XI (technical documentation, Sections 1+2) or Annex XII
(Article 53(1)(d) public training summary) directly from a training run:

```bash
soup train --config soup.yaml --annex-xi annex.md           # markdown
soup train --config soup.yaml --annex-xi annex.pdf           # PDF (pip install soup-cli[pdf])
soup train --config soup.yaml --track-energy --annex-xi annex.md  # + measured kWh/CO2
```

Top-10 domains by share, modality breakdown, training compute / kWh / CO₂, model
description, base model, run id. A `.pdf` output path renders a reportlab PDF (a `.md`
path renders markdown). The **top crawled domains** are auto-extracted from the training
JSONL (`cfg.data.train`). With `--track-energy`, the measured energy is recorded in the
doc. Operator-controlled fields are escape-neutralised (`|[](){}!<>` + newline / CR / tab)
so a malicious model name can't inject a forged heading into downstream renderers.

### Energy & CO₂ measurement (`--track-energy`)

`soup train --track-energy` wraps the training window in a codecarbon **offline** tracker
(no IP-geolocation network call). It reports kWh / CO₂ / grid intensity and feeds them into
`--annex-xi`. `--energy-country <ISO3>` picks the grid for the CO₂ estimate (default `USA`;
the kWh figure itself is country-independent). Requires `pip install soup-cli[carbon]`;
without it, `--track-energy` is a graceful no-op.


## Audit Log (`soup audit-log`)

**Every `soup` command** now appends one HIPAA/SOC2-shaped record to the JSONL audit log at
`~/.soup/audit.jsonl` (override via `SOUP_AUDIT_LOG_PATH`, containment-checked to
`$HOME / $CWD / $TMPDIR`). Opt out per-invocation with `soup --no-audit-log <cmd>` or
globally with `SOUP_NO_AUDIT_LOG=1`. Tail or rotate:

```bash
soup audit-log tail --limit 50          # Rich table view
soup audit-log tail --json              # raw JSONL for SIEM ingestion
soup audit-log rotate --cap-mb 100      # force a rotation pass
soup --no-audit-log version             # skip the audit line for this command
```

PII redaction across **every** string field (`hf_*` / `sk-*` / `Bearer …` → `<redacted>`)
via the v0.40.3 `_SECRET_RE` policy. POSIX `O_NOFOLLOW` + `0o600` perms, atomic-append,
rotation at 100 MiB with symlink rejection at the backup path. The auto-instrumentation is
best-effort — a broken audit log never crashes the CLI.


## Reproducibility Receipt (`soup train --repro-receipt`)

SR 11-7-style reproducibility receipt captures seeds (torch + numpy + python), kernel
versions (CUDA + cuDNN + NCCL), GPU model + driver, OS + arch:

```bash
soup train --config soup.yaml --repro-receipt repro.json
```

Bank model-risk teams and regulated-org auditors get a single JSON file that fingerprints
the exact environment the run executed in. Atomic write, cwd-contained.


## Adapter Backdoor Scanner (`soup adapters scan`)

Spectral analysis of LoRA adapter weights pre-load. Flags rank-1 dominance
(the canonical weight-space trojan pattern), top-1 singular-vector energy
concentration, NaN/Inf in weights, and Frobenius-norm outliers via robust
median + MAD bucketing. Pure numpy, no torch. Exit codes 0=OK / 1=WARN /
3=FAIL so CI can grep specifically for security failures. Reuses the
v0.57.0 `adapter_diff` loader so the on-disk surface stays single-source.


## Adapter Sign + Verify (`soup adapters sign` / `verify`)

Deterministic Merkle-root manifest over every file in the adapter dir
(including nested `tokenizer/` / `processor/` subdirs). Tamper any file
and `verify` fails. The `unsigned` backend gives offline tamper detection
(the Merkle-root hash is the trust anchor). The **`ed25519` backend is live**
(`pip install soup-cli[sign]`): it adds a real detached signature over the
Merkle root for *authentication* — proof the signer holds the private key.

```bash
# generate a keypair + sign in one shot
soup adapters sign ./adapter --backend ed25519 --generate-key signing.pem
# or sign with an existing key (or set SOUP_SIGNING_KEY)
soup adapters sign ./adapter --backend ed25519 --key signing.pem
# self-consistency verify (embedded public key)
soup adapters verify ./adapter
# genuine authentication: require the signer's key match a trusted one
soup adapters verify ./adapter --public-key trusted.pub
```

`verify` fails closed — any tamper, wrong/missing/unreadable key, or a
signature from an untrusted key marks the adapter invalid. Signing keys and
trusted public keys are symlink-rejected and size-capped but **not**
cwd-contained (keys are secrets that live outside the project). Signature
persists as `.soup-signature.json` (atomic write). `sigstore` keyless signing
stays infra-blocked (needs an OIDC identity provider + Fulcio/Rekor network —
it can't be honestly validated offline). `--strict` mode exits 3 on any verify
failure (CI gate code distinct from generic errors).


## Strict Safetensors Mode (`soup adapters check-safetensors`)

Refuses pickle / PyTorch-classic weights at the boundary — closed 8-entry
unsafe-extension allowlist (`.bin` / `.pt` / `.pth` / `.ckpt` / `.pkl` /
`.pickle` / `.joblib` / `.msgpack`). Picklemod gives every loader the
right to execute arbitrary code on load; refusing the file at the
boundary is the only sound mitigation. Friendly advisory names the
offending file and the canonical `from safetensors.torch import save_file`
recipe. Exit code 3 under `--strict` for CI gating.


## Namespace Pinning (Anti-AI-Jacking)

Trust-on-first-use SQLite cache: records `(repo_id, author, created_at)`
the first time Soup sees a HuggingFace repo. Subsequent loads compare
the current Hub fingerprint to the recorded pin — refuses author change
or backward `created_at` jump unless the operator passes
`--allow-namespace-shift <new-author>` (case-insensitive author match;
bool rejected so callers can't smuggle a free-for-all bypass). Threat
model: an attacker watches a popular repo, waits for the original owner
to delete or expire it, then re-creates the same `owner/name` with
malicious weights. Without this control, anyone with Soup pinned to that
namespace silently pulls poison on the next run. **The gate is wired into
Hub model downloads** — it consults the pin before fetching and refuses a
mismatch (raising before any weights land). It fails **open** when repo
metadata can't be fetched (offline / rate-limited) so a transient hiccup
never blocks a legitimate download. The pin store uses SQLite WAL + a
cross-process file lock so concurrent fetches don't lose the trust anchor.


## License-Conflict Matrix at Merge

Closed compatibility table over 33 SPDX-ish licenses spanning Apache /
MIT / BSD / LGPL / MPL / GPL / AGPL / CC-BY / CC-BY-NC / Llama-2/3.x /
Gemma / Qwen-research / Mistral-research / OpenRAIL / OpenAI-ToS /
Anthropic-AUP. `soup adapters merge --license apache-2.0 --license
cc-by-nc-4.0 -o merged` refuses (non-commercial cannot combine with
permissive). When you don't pass `--license`, the license is
**auto-detected** from each adapter's `adapter_config.json` / `config.json`
/ model-card frontmatter (HF `llama3.1`-style ids map to canonical) and the
gate runs automatically when every adapter resolves to a known license. To
proceed past a flagged conflict, pass `--license-override "legal-cleared
2026-05-19 by alice"` (8-char min, 4096-char max reason) — the decision is
**recorded to the audit log** for later legal review.

Merge is additionally gated by the backdoor scanner: any input that
`soup adapters scan` flags FAIL (or that can't be scanned) refuses the
merge with exit 3 unless you pass `--allow-unscanned`.


## Airgap Bundle (`soup airgap-bundle`)

Single signed tarball with model + datasets + wheels + CUDA kernels +
embedded `manifest.json` listing SHA-256 per file. Sized for one-way
physical-media transfer through a data diode. A reproducibility receipt is
embedded as a top-level `repro-receipt.json` when you pass
`--repro-receipt <receipt.json>` (or auto-detected from
`<model>/repro-receipt.json`):

```bash
soup airgap-bundle --model ./out --output bundle.tar --repro-receipt repro.json
```

Default 100 GiB cap;
refuses oversize. Deterministic dataset labeling by sorted basename
(NOT argv order) so the same inputs in different argv order produce
identical manifests. TOCTOU defence: `os.lstat + S_ISLNK` re-check on
parent + final output path before `mkstemp`; atomic `os.replace` from a
sibling tempfile. `tarfile.data_filter` set on Python 3.12+ so any
future caller adding `tar.extractall` automatically gets the safe-mode
extraction filter. `soup airgap-bundle` is intentionally a top-level
command (not `soup deploy airgap-bundle`) — it's an export operation,
not a deploy target.


