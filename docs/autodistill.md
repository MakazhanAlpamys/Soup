# AutoDistill artifact contract (Milestone A)

> Status: design-and-contract slice for [issue #580](https://github.com/MakazhanAlpamys/Soup/issues/580).
> This document is normative for artifact version 1. It does **not** add a task,
> configuration fields, model loading, capture, training, adaptive allocation, or a benchmark runner.

## 1. Goal and hard boundary

AutoDistill is a future bounded loop that discovers student weaknesses, asks a local
teacher for targeted material, verifies it, captures reusable teacher probabilities,
trains the student, and repeats only while measured gains justify the cost.

The invariant that motivates the offline format is:

> A teacher-only process must publish and release all teacher memory before a
> student-only process may consume its artifacts.

Version 1 has a second hard boundary: teacher and student use the **same tokenizer
fingerprint**. A repository name, equal vocabulary size, or similar token strings is
not sufficient. Cross-tokenizer visible-span alignment, ULD, and sequence KD remain
later and explicitly different objectives.

## 2. Non-goals and claim boundary

Milestone A does not choose a default `top_k`, compression method, probability dtype,
teacher quantization, benchmark sample size, or quality threshold. It does not claim
that sparse KD reproduces dense KD when `k < vocab_size`, or that a lower training loss
means a better model.

The v1 code is intentionally independent from `SoupConfig` and imports no ML runtime.
`task: autodistill` remains an illustrative name until a later review accepts a CLI and
configuration surface.

## 3. Closed-loop architecture

The eventual workflow has separate, bounded processes:

1. Freeze a student fingerprint, a private capability probe, selected public benchmark
   manifests, budgets, and stop rules.
2. Probe teacher and student on generated tasks without exposing private evaluation
   prompts or answers.
3. Admit only rows where verification establishes useful teacher advantage.
4. Run teacher-only capture and transactionally publish immutable shards.
5. Exit the teacher process and verify that the artifact fingerprints and hashes match.
6. Run student-only training against read-only shards and commit consumption only with
   a durable checkpoint hash.
7. Re-evaluate the frozen probe and any selected public benchmarks with the same item
   IDs, harness, scaffold, token budget, and version.
8. Allocate a new bounded generation budget or stop.

Milestone A implements only the contracts needed around steps 4-6 and the arithmetic
needed to plan step 4.

## 4. Versioned artifact family

Every JSON object has an exact `schema` discriminator. Readers reject unknown major
versions and unknown fields. Canonical JSON is UTF-8, sorted by key, has no insignificant
whitespace, permits no NaN/Infinity values, and is hashed with SHA-256.

| Artifact | Schema | Purpose |
|---|---|---|
| Plan | `soup.autodistill.plan.v1` | Immutable identities, capture policy, consumption policy, and estimate |
| Token row | `soup.autodistill.capture-token.v1` | Exact prefix/target IDs and teacher selected/tail probabilities |
| Shard manifest | `soup.autodistill.shard-manifest.v1` | Transaction state and exact payload commitments |
| Consumption event | `soup.autodistill.consumption-event.v1` | Append-only reserve, release, commit, and explicit replay ledger |

The implementation is in `soup_cli.autodistill.contract`. Pydantic models are frozen
and reject extra fields. JSON aliases keep the serialized key `schema` even though the
Python attribute is `schema_id`.

### 4.1 Plan

The plan binds all inputs that can alter captured bytes:

- teacher and student model identifiers, immutable 40-64 hex revisions,
  `config.json` hashes, and the sorted weight-file path/size/hash list;
- one shared tokenizer identifier, immutable revision, vocabulary size, tokenizer-file
  path/size/hash list, chat-template hash, and renderer name/version;
- ordered source-data path/size/hash list, row count, normalization algorithm, and
  normalized-content hash;
- backend and version, dtype, quantization label, sequence limit, truncation direction,
  planned token count, and maximum extra forced IDs per position;
- the complete probability and consumption policies;
- an optional previously measured throughput profile and its hash;
- the exact model-free storage/runtime estimate.

The SHA-256 of the plan's canonical JSON is the `plan_sha256` embedded in every shard.
Changing any identity or policy creates a different run; it is never a resume.

### 4.2 Data normalization

`soup-jsonl-c14n-v1` is deterministic:

1. decode bytes as UTF-8, accepting and removing one leading UTF-8 BOM;
2. split universal newline forms while preserving row order;
3. reject blank rows, duplicate object keys, malformed JSON, and non-object roots;
4. serialize each object as canonical JSON without changing Unicode codepoints;
5. terminate every row with one LF byte.

Both source-byte hashes and the normalized hash are retained. Source hashes detect an
exact file change; the normalized hash lets CRLF/formatting-only variants identify the
same ordered logical rows.

### 4.3 Capture token

Each v1 token row binds:

- example ID, trajectory kind, position, vocabulary size;
- exact context token IDs;
- teacher target token ID or student-sampled token ID, depending on the trajectory;
- top-k IDs, forced IDs, their sorted union, and one teacher log-probability per selected ID;
- residual tail mass, full teacher entropy, and temperature.

Production shards may use a denser binary column layout later, but it must preserve
these semantics exactly and be identified by a new schema if its interpretation changes.

### 4.4 Shard manifest

A shard manifest contains a portable shard/transaction ID, state, plan hash, preceding
manifest hash, logical row/token counts, and a sorted list of payload path/byte/row/token/
SHA-256 commitments. A `staging` manifest has no predecessor. Every subsequent state
must name the canonical hash of its preceding manifest so a transition cannot silently
replace history.

### 4.5 Consumption event

Consumption is an append-only ledger over an immutable source artifact. Every event
binds the artifact hash, trajectory view, prior/next state, run/reservation IDs, sequence,
optional checkpoint hash, and optional replay reference. The sequence begins at zero,
is contiguous, never mixes artifacts or views, and must form a valid state chain.

## 5. Missing-probability policy

For a teacher distribution `P` at one position, define:

```text
S = teacher_top_k_ids union forced_token_ids
```

Forced IDs are the ground-truth target on teacher-expert trajectories and the sampled
student token on student-rollout trajectories. If such an ID is already in top-k it is
stored once.

The artifact stores, explicitly and without defaults:

- exact teacher `log p_i` for every `i in S`;
- `p_tail = 1 - sum(exp(log p_i) for i in S)`;
- full teacher entropy, temperature, ID width, probability width, tail width, and entropy width;
- `renormalize_selected = false`.

Top-k probabilities are never renormalized to one. Doing so discards whether the teacher
assigned 1%, 20%, or 80% to the omitted vocabulary and changes the objective silently.

Given student probabilities over the same selected IDs and its residual mass, the v1
coarse-tail forward KL is:

```text
D_tail(P || Q)
  = sum(i in S) p_i * log(p_i / q_i)
  + p_tail * log(p_tail / q_tail)
```

When `S` is the complete vocabulary, both tails are zero and this equals dense forward
KL. When `S` is smaller, the tail is one coarse bucket: this is a data-processing
lower-resolution approximation, not dense equivalence and not a reconstruction of the
distribution inside the tail.

No default `top_k`, polynomial/residual compression, or “minimal quality loss” claim may
be introduced until dense-vs-sparse reconstruction and downstream ablations cover the
actual model pair, dtype, backend, and task mix.

## 6. Teacher fidelity tiers

BF16 teacher inference is preferred when it fits, but it cannot be required universally.
The eventual capture controller must classify evidence before choosing a loss:

| Tier | Evidence | Allowed use |
|---|---|---|
| Trusted distributions | BF16/FP16 or calibrated quantized logits pass a tiny A/B fidelity probe | Sparse KD plus verified CE/preferences |
| Trusted outputs only | Answers verify, but quantized logit fidelity does not | Verified SFT and preference/repair data; no logit-KD claim |
| No verified advantage | Teacher does not reliably beat the student/verifier | Task proposal only or disable the teacher for that stratum |

A quantized teacher therefore degrades by evidence, not by a hard-coded ban. A pilot must
compare verified SFT alone with verified SFT plus KD at equal student training tokens.

## 7. State machines

### 7.1 Example admission

```text
proposed -> probed -> captured -> verified -> admitted
                    |            |          -> rejected
                    |            -> quarantined
                    -> rejected / quarantined
```

Skipped transitions fail. `admitted`, `rejected`, and `quarantined` are terminal records;
reconsideration creates a new example/version instead of rewriting the old decision.

The later adaptive generator should interpret probe outcomes as follows:

| Student | Teacher | Decision |
|---|---|---|
| pass | pass | mastered; reduce sampling or increase difficulty |
| fail | pass | ideal frontier sample |
| fail | fail | simplify, change teacher/verifier, or quarantine |
| pass | fail | do not distill the teacher answer |

### 7.2 Shard publication

```text
staging -> complete -> verified -> available
    |          |           |           |
    +----------+-----------+-----------+-> quarantined
```

Only `available` shards may be reserved by a student. Corruption discovered after
publication moves the registry view to `quarantined`; payload bytes are not repaired in
place.

### 7.3 Consumption and replay

```text
available -> reserved -> committed
               |
               +-> available   (release after crash before durable checkpoint)
```

- A commit requires the exact durable student checkpoint SHA-256.
- A reservation alone never proves consumption.
- Student-rollout trajectories are one-round, consume-once signals. Replay after commit
  is forbidden because the prefixes are stale with respect to the updated student.
- Teacher-expert artifacts may be replayed only through a new reservation whose
  `replay_of` equals the prior committed event hash. The source artifact stays immutable.
- Replay policy is explicit in every plan; it is not inferred from a filename or task.

### 7.4 Milestone B1 MLX teacher boundary

The first Apple Silicon capture worker is an internal, opt-in implementation surface; it
does not add a task, CLI command, or `SoupConfig` field. The controller starts a fresh Python
process whose request accepts teacher, tokenizer, dataset, and publication roots but no student
root. The controller writes its final receipt only after that child has exited successfully and
the exact `available` manifest has been reopened. The plan still contains the immutable student
fingerprint as future-consumer metadata; it is never resolved or loaded during capture.

For this boundary, MLX-LM loads the teacher model from its immutable local checkpoint root and
loads the canonical shared tokenizer separately from a tokenizer-only root. That tokenizer root
may be derived from the student's tokenizer bytes, but it contains no student weights and the
worker never resolves a student model. `capture.backend_version` is the exact installed `mlx-lm`
distribution version and the tokenizer renderer is `mlx-lm@<version>`. Both must match the plan
before publication. MLX/MLX-LM remain lazy imports inside the worker.

The worker also verifies the loaded floating-parameter dtypes against `capture.dtype`. An
unquantized checkpoint must expose exactly that dtype and uses the literal quantization identity
`none`. A quantized checkpoint must expose the declared base dtype; float32 auxiliary parameters
are allowed, while any other undeclared floating dtype fails closed. If `config.json` contains
`quantization` or `quantization_config`, its identity is
`config-sha256:<sha256(canonical JSON of those active fields)>`. This binds the plan to the exact
quantization recipe without pretending that names such as `4-bit` uniquely identify a runtime.
The declared inference dtype, all observed floating-parameter dtypes, and the quantization
identity are repeated in the child receipt.

The bound dataset is canonical JSONL with versioned, already-tokenized rows:

```json
{"schema":"soup.autodistill.tokenized-teacher-example.v1","example_id":"ex-1","prompt_token_ids":[1,2],"target_token_ids":[3,4]}
```

Prompt IDs are non-empty because a causal next-token distribution needs a preceding position.
Every source byte, normalized row, tokenizer file, teacher config, and listed teacher weight is
verified before model load. A worker request selects a half-open example range, but the complete
dataset fingerprint and total planned target-token count are verified before that shard subset is
accepted. Ranges never split one example trajectory.

Without truncation, one causal forward captures all target positions in an example. Once the
declared sequence limit requires truncation, the worker evaluates the exact recorded context for
each position. Only the final vocabulary row is converted to float32 host values; selected IDs are
not renormalized. Process exit, rather than cache reclamation alone, is the memory-isolation gate.

## 8. Transactional publication, resume, and corruption

The capture writer follows a write-last manifest protocol on one filesystem:

1. Create a transaction-specific staging directory that consumers never scan.
2. Write payloads to temporary sibling files; flush and sync before replacement.
3. Compute hashes and logical counts from the exact bytes on disk.
4. Write the `complete` manifest atomically and chain it to the staging manifest hash.
5. Reopen every payload, verify membership, byte count, SHA-256, parseability, row count,
   token count, plan hash, and tokenizer/data fingerprints.
6. Write the `verified` manifest atomically.
7. Publish payload names as a group, then atomically write the `available` manifest last.

If grouped publication fails, the previous available group is restored or no group is
visible. A manifest without all matching payloads is corruption, never a partial success.

Resume decisions are closed and deterministic:

| Observed state | Integrity/fingerprints | Action |
|---|---|---|
| `staging` | valid/matching | resume the staging transaction |
| `complete` | valid/matching | verify, then publish |
| `verified` | valid/matching | publish |
| `available` | valid/matching | reuse without recapture |
| `quarantined` | any | refuse |
| any other state | corrupt or mismatched | quarantine and start a new transaction |

An uncommitted final fragment may be truncated only when its bytes are outside a committed
record boundary. A parseable changed row, duplicate/reordered position, mismatched plan,
changed model revision, tokenizer mismatch, or payload hash mismatch fails closed. Resume
must never mix rows from two plans or duplicate an already committed prefix.

## 9. Plan-only estimate

Plan-only accepts metadata and optional cached throughput; it does not resolve a hub,
instantiate a tokenizer, inspect a device, or import Torch, Transformers, or MLX.

For:

- `N`: planned captured token count;
- `V`: vocabulary size;
- `K`: explicit top-k;
- `F`: maximum additional forced IDs per position;
- `B_id`, `B_logp`, `B_tail`, `B_entropy`: explicit storage widths;

the exact raw dense-log-probability bytes are:

```text
dense_bytes = N * V * B_logp
```

The raw v1 sparse upper bound is:

```text
S_max = min(V, K + F)
sparse_upper_bytes
  = N * (S_max * (B_id + B_logp) + B_tail + B_entropy)
```

This deliberately excludes container/index/JSON metadata and reports that exclusion.
Actual output must report both planned and observed bytes/token.

Runtime is `unknown` when no compatible measured profile is provided. Given a previously
measured end-to-end range `[throughput_min, throughput_max]`, plan-only reports:

```text
seconds_min = N / throughput_max
seconds_max = N / throughput_min
```

The profile hash, backend, model fingerprint, dtype/quantization, sequence-length band,
and hardware class must accompany a real future profile. Plan-only must not benchmark a
model to make the estimate look complete.

## 10. Adaptive curriculum (later milestone)

Tasks must not come from one permanent list. A later allocator should maintain strata by
domain, subskill, difficulty, language, format, and verifier type. It should combine:

- normalized student weakness and uncertainty;
- proximity to the teacher-student frontier;
- verified teacher advantage;
- verifier reliability;
- novelty/coverage deficit;
- measured gain per student training token;
- teacher-generation, storage, and training cost;
- a bounded exploration floor.

Difficulty increases only after easier strata pass. Repeated both-fail or verifier-
disagreement rows are simplified or quarantined rather than force-labeled. Every loop has
mandatory teacher-token, student-token, storage, wall-clock, energy, and iteration ceilings,
plus plateau and general/safety regression stops.

The eventual objective family may combine verified CE, teacher-trajectory sparse forward
KD, entropy-aware sparse KD on fresh student prefixes, preference/repair loss, and replay.
Those weights remain experimental and are not part of v1 artifact defaults.

## 11. Evaluation firewall and benchmark registry (later milestone)

At minimum, every run freezes a small private capability probe before generation. Its raw
prompts and answers never reach the teacher, task generator, allocator, or training data;
only aggregate weakness bands may cross that boundary.

Public benchmark execution is selectable because full agentic suites can dominate runtime.
The planned registry exposes `quick`, `standard`, `full`, and deterministic `custom` samples.
Sampling is stratified and selected by a stable hash of `(benchmark version, seed, item id)`;
the exact sorted item IDs are stored. Pre/post comparisons require the same benchmark version,
items, harness, scaffold, tools, token/time budget, and scorer.

Candidate catalog families include:

- small/general: ARC, HellaSwag, PIQA, WinoGrande, GSM8K, MBPP, and HumanEval;
- general/reasoning: MMLU-Pro, GPQA Diamond, MATH-500/AIME, and LiveCodeBench;
- software engineering: SWE-bench Verified/Pro and DeepSWE-style evaluations;
- agentic: Terminal-Bench (pin v4.0.0 when selected), Agents' Last Exam, and other
  reproducible harnessed task suites.

Names above are roadmap candidates, not v1 schema keys. Each integration needs its own
license, availability, harness, version, contamination, and cost review. Agentic scores
measure the model plus the pinned harness/scaffold, not the model in isolation.

A user may skip an expensive public pre-baseline, but then Soup may report only an absolute
post score, not a before/after gain. Scientific comparison requires at least:

1. untouched student;
2. fixed verified SFT;
3. adaptive verified SFT;
4. adaptive sparse KD;

with equal student training tokens. Report absolute deltas and teacher-gap closure, confidence/
noise floors, three seeds where practical, peak memory, capture/training time, energy, bytes/token,
and dense-vs-sparse reconstruction/KL error. Loss reduction alone is not an acceptance criterion.

## 12. Threat and failure model

| Threat/failure | Detection | Required response |
|---|---|---|
| Interrupted or partial publication | Missing state/payload, count/hash mismatch | Ignore as reusable; resume a matching transaction or quarantine |
| Parseable payload tampering/bit rot | SHA-256 over exact bytes | Quarantine; never repair or overwrite in place |
| Mixed run/model/tokenizer/data | Plan and immutable fingerprint mismatch | Refuse resume/consumption and create a new run |
| Moving hub revision | Revision is not immutable or files no longer match | Refuse capture/reuse |
| Cross-tokenizer token-ID collision | Tokenizer fingerprint differs | Fail clearly; no silent ID matching or ULD fallback |
| Stale student on-policy prefixes | Student rollout already committed/round changed | Forbid replay; recapture from the new student |
| Crash after reserve, before checkpoint | No committed checkpoint hash | Release reservation back to available |
| Crash after checkpoint, before ledger commit | Checkpoint exists but event absent | Reconcile exact hash, then append one commit; never retrain blindly |
| Private-eval leakage | Provenance/lineage intersects frozen eval IDs | Quarantine data and invalidate the run |
| Benchmark contamination | N-gram/exact lineage checks against pinned public corpora | Remove/quarantine and report affected strata |
| Verifier/reward hacking | Deterministic checks, adversarial controls, disagreement rates | Quarantine disagreement; stop after repeated failures |
| Teacher self-confirmation | Teacher is sole generator and judge | Require deterministic or independent evidence where possible |
| Quantized-logit drift | Tiny BF16/quantized fidelity and downstream A/B pilot | Fall back to output-only SFT/preferences |
| Budget runaway | Mandatory hard ceilings and preflight estimate | Stop before exceeding the first exhausted budget |
| Path traversal/symlink swap | Normalized relative paths, `realpath` + `commonpath`, no-follow reads | Refuse access and quarantine the transaction |
| Non-finite/invalid probabilities | Strict schema, mass/entropy checks | Reject row before shard completion |
| Duplicate/reordered records | Contiguous IDs/positions and committed counts | Fail closed; resume verified prefix only |

The system does not defend against a malicious kernel or compromised model host. Local model
weights, generated code, and verifier execution remain untrusted inputs and require the same
sandboxing and path/size caps as Soup's existing code-execution and data surfaces.

## 13. Milestone boundaries

- **A (this slice):** v1 schemas, fingerprints, probability/replay contracts, deterministic
  estimate, state machines, integrity/resume semantics, fixtures, and threat model.
- **B:** same-tokenizer teacher-only capture plus student-only CE+sparse-KL, process-memory
  separation, interrupted-run equivalence, and CUDA plus an Apple Silicon path.
- **C:** adaptive generation, verification, capability map, bounded allocator, replay and stops.
- **D:** explicit cross-tokenizer visible-span ULD with Unicode/control/chat-template attacks.
- **E:** equal-token reproducible benchmark and ablation report.

Milestone B must first prove dense-vs-offline equivalence on a tiny deterministic full-vocabulary
fixture and exact resume behavior. Only after that should a real teacher/student scientific smoke
be attempted.

## 14. Deterministic fixtures

`tests/fixtures/autodistill/v1/` contains:

- a fully explicit plan with no runtime profile;
- teacher-expert and student-rollout capture rows over a three-token vocabulary, where selected
  mass is 0.9 and residual tail is 0.1;
- a byte-counted and SHA-256-committed JSONL shard reconstructed from canonical capture data so
  checkout newline conversion cannot alter its committed bytes;
- a student-rollout reservation/commit ledger.

The tests cover frozen/strict models, missing policy fields, tampered estimates, no heavy import,
dataset canonicalization, student-rollout requirements, top-k/forced union and residual-tail mass,
`k = vocab` dense equivalence, path traversal, corruption, checkout-newline independence, state
skips, contiguous exactly-once consumption, chained expert replay, and every resume decision.
