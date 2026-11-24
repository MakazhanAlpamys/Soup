# Docs-search lexical candidate for #3192

Draft implementation PR: [#3242](https://github.com/MakazhanAlpamys/Soup/pull/3242).

The original-28 pinned-docs gate and the friend-supplied 30-question comparison
both pass their declared checks. Friend-set Hit@5 improves from 29/30 to 30/30;
its outside-tuning authorship is user-attested, as recorded below.
The maintainer's [fixture ruling](https://github.com/MakazhanAlpamys/Soup/issues/3192#issuecomment-7126471247)
replaces the original-corpus gate with a separately versioned fixture rebuilt
entirely from pinned Soup docs. Option B applies to this new fixture. The
original-corpus comparison remains historical evidence.

## Scope and selection

The tokenizer, Okapi BM25 formula, index configuration, serialized cache, CLI
flags, JSON keys and wheel mappings are unchanged. `retriever.py` combines
literal scores with query-side inflections/aliases and a bounded leaf-heading
signal before selecting results. `docs/commands.md` lists every rule, weight
and inactive example. The reference-free rule matches all qualifying headings;
it does not name ORPO. Generic preference queries retain DPO, KTO and ORPO.

Development selection used the ten AI-drafted, maintainer-reviewed questions.
The already inspected 8+6 AI-written questions remain development evidence.
They are not a fresh evaluation for this candidate. The three rule groups were
measured separately and together; `development-ablation.json` records the rows.
All three give the highest development MRR@10 among candidates tied at nine
top-five hits. Unneeded prototype aliases for near-duplicate, GPU-memory and
OpenAI-compatible phrases were not retained. No fresh questions were used to
choose these rules.

## Pinned-docs v1 gate

`tests/fixtures/docs_search/pinned_docs_v1.json` is a new fixture, version 1,
built from all 12 Markdown files at Soup commit
`8ab26641ed34e724018e696a93a06460f26c153e`. Its 295 sections use Soup's existing
fence-aware heading chunker, with no model tokenizer, model download or new
dependency. The original 1,068 passage IDs are all retired for this fixture;
model-based sub-splitting, decoded WordPiece normalization and code-comment
pseudo-headings are removed. These section scores cannot be compared directly
with the historical passage scores.

All 28 original question IDs and wording are preserved. Their 69 positive
passage targets map to 34 unique owning-section targets, with duplicates
collapsed per question. No relevance judgments were added. Every question
remains resolved, and no question was dropped. `pinned_docs_v1-migration.json`
records every historical passage-to-section mapping and all 28 old/new target
lists and statuses. `original28-source.json` preserves the original questions,
annotations, source-file hashes and metadata/body hashes for all historical
passages, without copying their bodies.

The two disputed IDs are:

- `backends-and-ops::replay-an-old-run-s-summary-loss-curve-from-history::sub_000`
- `backends-and-ops::replay-an-old-run-s-summary-loss-curve-from-history::sub_001`

Neither was listed as relevant in any of the 28 original annotations; their
old label status was unjudged. Their clean owning section is
`backends-and-ops::section_020` (Experiment Tracking), which is also unjudged
by this set. Thus no query targets or label statuses changed because of these
two passages. Their old bodies are excluded; the replacement section contains
only pinned Soup docs text. The migration file names both passages and records
empty old/new affected-query lists. This is a full rebuild, rather than deletion
of only those two passages.

`pinned_docs_v1-lock.json` fixes the fixture, source metadata, migration audit,
rebuild script, evaluation script and candidate source hashes. Candidate
`f40be434fb6b4dbd1b1688cf12c633de5f1f6ef5` is unchanged. The lock and fixture
were committed and pushed in `a5ab5484`, then declared in PR #3242 at
2026-11-24 09:13:47 UTC before either baseline or candidate was scored on v1.
`pinned_docs_v1-execution.json` records the declaration identity and first-run
start/end times. No locked file changed after the declaration or scoring.
`candidate.json` remains the original development-freeze record, including the
historical provenance status at that time.

Reproduction requires the pinned Soup Git object in the local clone. In a
shallow clone, fetch it with
`git fetch https://github.com/MakazhanAlpamys/Soup.git 8ab26641ed34e724018e696a93a06460f26c153e`.
The committed fixture can be scored offline without Git history or the
prototype repository. From a base Soup environment:

```bash
python scripts/rebuild_docs_search_fixture.py
python scripts/check_docs_search_ranking.py tests/fixtures/docs_search/pinned_docs_v1.json
```

The first command verifies a byte-identical rebuild and complete migration
without scoring. `--write` regenerates the two artifacts. The second runs the
option B gate. The first declared execution returned exit code 0:

| Pinned-docs v1, original 28 | Baseline | Candidate |
| --- | --- | --- |
| Hit@1 | 24/28 | 25/28 |
| Hit@5 | 27/28 | 27/28 |
| MRR@10 | 0.9151785714 | 0.9345238095 |
| Recall@5 | 0.9642857143 | 0.9642857143 |

There are no first-relevant-rank losses. `pinned_docs_v1-results.json` contains
all 28 rows and both top-ten rankings. Exact aggregate metrics were recomputed
from the saved rows without rerunning ranking; all four pass option B. This
gate result does not satisfy the separate fresh-query Hit@5 requirement.

## Reproduce the separate current-section and development comparisons

From this revision, in a base Soup environment:

```bash
python scripts/check_docs_search_ranking.py tests/fixtures/docs_search/maintainer_dev.json
python scripts/check_docs_search_ranking.py tests/fixtures/docs_search/current_sections.json
python scripts/ablate_docs_search.py
pytest -o addopts='' tests/test_docs_search.py
```

The fixtures contain original queries, labels, baseline top-ten IDs and exact
document/chunk hashes. Corpus text is read from this checkout's `docs/`.
The runner refuses changed documentation, changed chunking, invalid labels or
a changed recorded baseline. The 69 historical label mappings are recorded in
`tests/fixtures/docs_search/current_mapping.json`; mappings were fixed before
candidate scoring. Several old passages collapse to the same section, including
old code-fence comments. These current-section metrics remain a separate
comparison; the pinned-docs v1 fixture above is the gate.

| Cohort | Baseline Hit@1 | Candidate Hit@1 | Baseline Hit@5 | Candidate Hit@5 | Baseline MRR@10 | Candidate MRR@10 |
| --- | --- | --- | --- | --- | --- | --- |
| Ten development questions, 702 current sections | 2/10 | 5/10 | 7/10 | 9/10 | 0.3833 | 0.6560 |
| Original 28, mapped onto 702 current sections | 17/28 | 18/28 | 27/28 | 27/28 | 0.7857 | 0.8075 |
| Original 28, exact 1,068 frozen passages (historical evidence only) | 23/28 | 24/28 | 27/28 | 28/28 | 0.8845 | 0.9167 |

The source baseline is Soup `bf62781f0dc8cb509e3dae16a3818f8dba28627c`.
The candidate documentation adds one section to its 701-section corpus. Each
comparison uses identical corpus text and labels for both rankers. The Mac
target moves from 21 to 7 in the current development snapshot and remains
outside the default five results. Original ten-question targets remain strict;
two source-heading spelling differences are explicit in `maintainer_dev.json`.

The original-passage replay reproduced the archived baseline top-ten IDs on
all 28 queries. Recall@5 moves from 0.7167 to 0.8000. The full rank-loss list is:

| ID | Query | Old first-relevant rank | New first-relevant rank |
| --- | --- | --- | --- |
| `soup_v3_023` | find mlx recipes for apple silicon | 2 | 3 |

The current-section comparison has no individual rank losses and unchanged
Recall@5 of 0.9464. Full comparison outputs are stored alongside this report.

## Option B and fresh-query requirement

The comparison uses exact rational arithmetic. Hit@1, Hit@5, MRR@10 and mean
per-query Recall@5 must each stay at least at baseline. MRR@10 is zero when
the first relevant result is beyond ten or absent. Every individual first-rank
loss is printed, including losses beyond ten, but does not itself fail option B.
No bootstrap interval is used for this deterministic comparison.

The runner exits 0 for a passing comparison, 1 for a failed numerical gate and
2 for invalid inputs. These are development-tool exit codes; `soup docs search`
keeps its existing successful exit code of 0. A fresh fixture can additionally
be scored with `--require-hit5-gain`. A passing numerical result is necessary
but does not establish fresh authorship or fixture provenance.

The [supplemental ten questions](question-review.md) were approved by the user
with their original wording and target sections unchanged, then frozen in
commit `f1f507ad` before scoring against candidate `f40be434`. Their authorship
and lack of outside-tuning independence remain recorded in the fixture and
`supplemental10-lock.json`.

| Supplemental 10 | Baseline | Candidate |
| --- | --- | --- |
| Hit@1 | 3/10 | 3/10 |
| Hit@5 | 7/10 | 7/10 |
| MRR@10 | 0.4750 | 0.4533 |
| Recall@5 | 0.6000 | 0.6000 |

The check returned exit code 1: Hit@5 did not improve and MRR@10 decreased.
Every loss is retained: R01 (38 → 39), R05 (13 → 15), R07 (2 → 3), and R08
(4 → 5). R06 improved from 62 to 38 while remaining outside ten. Full rankings
are in `supplemental10-results.json`; `supplemental10-execution.json` records the
first run and pre-scoring fixture hash. No rule, query or label was changed in
response. This result does not justify promotion of the candidate.

The friend-supplied 30 below provide the fresh-set top-five gain on the user’s
outside-tuning authorship attestation. The supplemental ten remain separate
negative evidence; user approval of their text does not change their recorded
authorship.

Before the fresh set is scored, record the candidate commit, rule/source hash,
documentation hashes, query provenance and target labels. Have a contributor
who is not tuning the rules supply the questions. Keep the set separate and
require a Hit@5 increase before proposing a default change. If inspected fresh
results are used to revise the candidate, obtain another fresh set.

## Friend-supplied 30: frozen comparison

On November 24 the user supplied 30 questions and target descriptions,
attributed to a friend outside the project/tuning work. That attribution is
user-attested; individual attribution and writing assistance were not supplied.
All 30 questions and their supplied target descriptions are preserved exactly
in `tests/fixtures/docs_search/friend30.json`. No question was selected or
rewritten after inspecting a ranking.

The supplied targets map to 25 existing owning sections in the unchanged
17-guide, 702-section candidate snapshot. FR13 retains both supplied targets.
The full mapping is in `friend30-targets.md`; it was prepared without scoring.
This cohort is separate from both the supplemental ten and the older pinned-docs
v1 original-28 gate. It covers 23 questions in adapters/governance, four in
AutoDistill and three in backends/ops, and is not a representative random sample.
No normalized literal repeat was found against the 62 questions in the original
28, maintainer ten, supplemental ten and previously inspected 8+6 sets. Shared
intents are not claimed to be novel.

`friend30-lock.json` fixes the question/target fixture, mapping audit,
evaluation tool and unchanged candidate source identities. Candidate `f40be434`
was frozen before these questions were received. The fixture and lock were
committed and pushed in `fe823e9d`, then declared in PR #3242 at
`2026-11-24T09:49:48Z` before either arm was scored.
`friend30-execution.json` records the declaration and first-run identities.
No locked file changed after scoring.

The maintainer's fresh-query requirement is a strict Hit@5 gain, with the
original-28 option B gate kept separate. The existing command below also checks
aggregate preservation on this fresh set. Its utility exit status and the
maintainer's fresh Hit@5 condition are both reported, with all metrics and
every first-relevant-rank loss retained:

The command must run in the exact scored checkout, `d6422869d54e624b6b0e7cd8fe5fd7fbe245e82c`,
whose docs match the frozen 17-guide hashes. Current main and this branch after
merging main contain newer guides and are intentionally rejected by the runner.
From an up-to-date clone of this PR branch, create an isolated checkout so the
working branch is not changed:

```bash
git worktree add --detach ../soup-3242-friend30 d6422869d54e624b6b0e7cd8fe5fd7fbe245e82c
cd ../soup-3242-friend30
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python scripts/check_docs_search_ranking.py tests/fixtures/docs_search/friend30.json --require-hit5-gain
```

On Windows, use `.venv\Scripts\python.exe` for the virtualenv commands. The
existing result and pre-scoring lock remain the evidence for the scored
snapshot; merging main does not relabel or rescore these questions.

The first declared run returned exit code 0. All four aggregates are preserved
and the strict fresh Hit@5 gain is satisfied:

| Friend 30 | Baseline | Candidate |
| --- | --- | --- |
| Hit@1 | 23/30 | 23/30 |
| Hit@5 | 29/30 | 30/30 |
| MRR@10 | 0.8381481481 | 0.8666666667 |
| Recall@5 | 0.9500 | 1.0000 |

The complete first-relevant-rank loss list has one row:

| ID | Question | Old rank | New rank |
| --- | --- | --- | --- |
| FR11 | registry push idempotency key same key different config error | 1 | 2 |

The new top-five hit is FR24, “mlx backend on mac training lora setup,” which
moves 9 → 2. FR13 improves 5 → 1, and FR15 improves 3 → 2. The other 26
first-relevant ranks are unchanged. Both supplied FR13 targets are retained;
its second target also enters the top five. `friend30-results.json` stores all
30 rows and both top-ten rankings. The original-28 pinned-docs v1 gate remains
passing. Exact metrics and the complete loss list were recalculated from saved
rows without rerunning ranking; candidate, questions and labels remain frozen.

## Historical corpus provenance and the approved rebuild

The original corpus SHA-256 is
`d787703e048a641884274e57f5802e328e43c637a0971217601e6a37128393e3`.
Its metadata names Soup docs commit
`8ab26641ed34e724018e696a93a06460f26c153e`. The original queries, labels and corpus
were read from prototype commit `e884b9ef684533d9fd81996f4a63c42488b943ee` without
changing them.

Of 1,068 corpus passages, 1,066 can be traced to that Soup snapshot after
accounting for whitespace/case/accent normalization, decoded WordPiece prefixes
and omitted emoji. Two passages contain extra loss-reporting prose not found
in that recorded snapshot:

- `backends-and-ops::replay-an-old-run-s-summary-loss-curve-from-history::sub_000`
- `backends-and-ops::replay-an-old-run-s-summary-loss-curve-from-history::sub_001`

The extra prose discusses loss reporting and NULL loss measurements. Its exact
source revision has not been established. The maintainer reports finding no
matching commit and has ruled that further revision searches are unnecessary.
Line endings also differ between the vendored docs and upstream, and the saved
backends source-manifest hash does not match its vendored file. The original
corpus and frozen-manifest hashes themselves do match.

The maintainer requires every committed corpus text to come from Soup docs.
The unresolved passage bodies are therefore not included in this PR candidate.
The local exact-corpus replay remains historical numerical evidence. The
maintainer's November 24 ruling makes pinned-docs v1 the gate instead. All old
identities and hashes are preserved as metadata, with every replacement and
target migration disclosed above. The historical output was renamed to
`historical-original28-results.json` without changing its bytes; its legacy
cohort identifier and numerical `gate` field describe that historical run,
which is no longer the PR's gate.

## Verification and cost

`.gitattributes` pins LF checkout for the files covered by frozen byte hashes.
A `core.autocrlf=true` checkout originally changed all 11 locked file hashes;
after the attribute fix, all 11 match the frozen identities. This verifies Git
checkout conversion locally; it is not a Windows runtime test. No candidate,
fixture or evaluation-script contents changed for this fix.

Each of the six retained rule components was removed in turn: Mac/Macs,
continue/resume, batch-size, contextual MLX, reference-free phrase and heading
weight. A focused retrieval test failed for every removal. The source was
restored after each run; `mutation-results.json` names the failed tests.

Tests also cover technical-only unchanged rankings, reference-free ambiguity,
literal multiplicity, unique inferred terms, heading bounds and pre-cutoff
ranking, exact option-B comparisons, invalid inputs and cache preservation.
The query-only patch reuses an existing cache without changing its bytes or
mtime. CLI-output assertions use ANSI stripping and whitespace collapse.

The base wheel contains 17 Markdown guides exactly once and no docs media.
It searched successfully from an empty directory with offline flags, an isolated
cache and no Torch/Transformers installed or imported.

Local Apple Silicon, Python 3.12.13; timings use the same 702-section corpus and
the query “will this model fit in 24GB of VRAM” in both arms. Fresh processes
invoke the real Typer app through a Python wrapper that redirects the cache;
they are not measurements of the bare installed executable. Twenty interleaved
runs per arm/condition; 500 interleaved warm rankings per arm:

| Measurement | Baseline median / p95 | Candidate median / p95 |
| --- | --- | --- |
| Empty index cache, full process | 630.70 / 640.66 ms | 636.32 / 649.76 ms |
| Populated index cache, full process | 317.34 / 335.32 ms | 322.93 / 343.19 ms |
| Warm in-process ranking | 0.228 / 0.245 ms | 2.513 / 2.649 ms |

The candidate spends more time examining section headings. These are local
observations, not hardware-independent guarantees. `timing-results.json`
records the method, sample counts and unrounded summaries.
