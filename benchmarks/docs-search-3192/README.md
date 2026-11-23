# Docs-search lexical candidate for #3192

Draft implementation PR: [#3242](https://github.com/MakazhanAlpamys/Soup/pull/3242).

This is a development candidate for draft implementation review. An eligible
fresh query set with a Hit@5 gain and resolution of the historical corpus
provenance gap below are required before requesting review or merge of a default
change. The maintainer's confirmed gate is option B, not the earlier
zero-individual-loss proposal.

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

## Reproduce the current-section and development comparisons

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
old code-fence comments. These section metrics are a separate comparison and
do not replace the original passage benchmark.

| Cohort | Baseline Hit@1 | Candidate Hit@1 | Baseline Hit@5 | Candidate Hit@5 | Baseline MRR@10 | Candidate MRR@10 |
| --- | --- | --- | --- | --- | --- | --- |
| Ten development questions, 702 current sections | 2/10 | 5/10 | 7/10 | 9/10 | 0.3833 | 0.6560 |
| Original 28, mapped onto 702 current sections | 17/28 | 18/28 | 27/28 | 27/28 | 0.7857 | 0.8075 |
| Original 28, exact 1,068 frozen passages (local replay; provenance pending) | 23/28 | 24/28 | 27/28 | 28/28 | 0.8845 | 0.9167 |

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

An eligible fresh set with a top-five gain or an explicit maintainer amendment
is still required. User approval of supplemental question text does not change
its recorded authorship.

Before the fresh set is scored, record the candidate commit, rule/source hash,
documentation hashes, query provenance and target labels. Have a contributor
who is not tuning the rules supply the questions. Keep the set separate and
require a Hit@5 increase before proposing a default change. If inspected fresh
results are used to revise the candidate, obtain another fresh set.

## Historical corpus provenance gap

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

The extra prose includes “The loss candidate based on 0.73.3…” and discussion
of NULL loss measurements. Its exact source revision has not been established.
Line endings also differ between the vendored docs and upstream, and the saved
backends source-manifest hash does not match its vendored file. The original
corpus and frozen-manifest hashes themselves do match.

The maintainer requires every committed corpus text to come from Soup docs.
The unresolved passage bodies are therefore not included in this PR candidate.
The local exact-corpus replay is numerical evidence, not a completed
repository-local fixture deliverable. No passages, labels or baseline results
were silently replaced. A source revision or explicit maintainer decision is
needed to complete this acceptance criterion.

## Verification and cost

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
