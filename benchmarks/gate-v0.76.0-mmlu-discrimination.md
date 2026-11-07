# v0.76.0 gate: MMLU-style discrimination (#1192)

`mini_mmlu` had reached a ceiling: the #1111 record measured the strong
reference at **1.000 (26/26)**, and Qwen2.5-1.5B-Instruct scored 1.000 (26/26)
in the arithmetic and common-sense records. Its answer key was also lopsided:
12 of 26 answers were B, so a model that only ever answered "B" scored 0.462.
This record measures two 24-row candidate pools, the 14 rows selected from them,
and the revised 40-row fixture on two models.

## Environment and method

- Apple M1, 32 GB unified memory; MPS
- checkpoint-native dtype (`dtype="auto"`), greedy decoding,
  `BEHAVIOURAL_MAX_NEW_TOKENS` (256) new-token cap
- strong reference: `Qwen/Qwen2.5-1.5B-Instruct`
- weak reference: `HuggingFaceTB/SmolLM2-135M-Instruct`

Each model was loaded once through `live_eval.make_generator`. Candidates were
prompted with `build_mcq_prompt` and scored with `score_answer`, the same path
`score_bundled_suite` uses; the final fixture was scored with
`score_bundled_suite` over every bundled suite, on the same machine.

**The strong reference is not the #1111 one**, as in the arithmetic and
common-sense records. These numbers show the fixture leaves the rails for a 1.5B
model. They do not show that a 7B model is off the ceiling.

## Candidate pools and selection

48 hand-authored, original four-option rows in two rounds, each key carrying a
one-line justification. Correct letters were spread six to a letter by option
position before measurement, so the pools measured are the pools shipped.

| Pool | Rows | Qwen2.5-1.5B-Instruct | SmolLM2-135M-Instruct |
|---|---:|---:|---:|
| round 1: general science, history, literature, maths | 24 | 21/24 | 6/24 |
| round 2: rows with a common misconception as a distractor | 24 | 19/24 | 8/24 |

Round 1 alone was too easy to move the suite: added to the 26 original rows it
would have left Qwen at about 0.94. Qwen answered every candidate in both rounds
with a bare or parenthesised letter, so no miss was an extraction or truncation
failure.

Selection kept 14 rows:

| Decision | Candidates | Reason |
|---|---|---|
| keep: strong-model misses | fastest-rotating planet, hexagon interior angles (round 1); longest coastline, octopus hearts, tallest mountain base to peak, bits in a nibble (round 2) | each miss chose a plausible wrong answer: Mercury, 1080 degrees, Australia, 4, Everest, 2 |
| keep: strong-model passes | mitosis phase (round 1); hottest planet, symbol K, smallest ocean, Saturn's largest moon, hardest body tissue, Gibraltar, capital of Turkey (round 2) | keeps the suite from measuring only the traps |
| reject: strong-model misses | country with most time zones, largest desert | correct keys, but each turns on a definition a reviewer can argue (territories counted; Antarctica as a desert) |
| reject | the remaining 32 passes | not needed to place the strong model off the rail |

Two of the six kept misses chose option A (Mercury, Everest), as did both
rejected misses (Russia, the Sahara). Unlike the rejected "A" misses in the
common-sense record, A held the misconception in each of these four rows, so
they read as the trap working rather than a letter preference. The other four
kept misses chose B or D.

## Result

The final fixture keeps the 26 original rows, appends the 14 selected rows, and
swaps options on two original rows (square root of 81 B -> A, author of *Romeo
and Juliet* B -> C) so no letter answers more than 30% of the suite. The key is
now A 11, B 12, C 11, D 6.

| Fixture | Rows | Qwen2.5-1.5B-Instruct | SmolLM2-135M-Instruct |
|---|---:|---:|---:|
| v0.73.2 fixture (common-sense record) | 26 | 1.000 (26/26) | 0.269 (7/26) |
| revised fixture | 40 | **0.850 (34/40)** | **0.350 (14/40)** |

Qwen2.5-1.5B missed 6 rows, all new. Row numbers are 0-based indices into
`MINI_MMLU`; index 26 is the first new row.

| Row | Expected | Chosen |
|---:|---|---|
| 26 | D (Jupiter) | A (Mercury) |
| 27 | C (720 degrees) | D (1080 degrees) |
| 28 | C (Canada) | B (Australia) |
| 29 | A (3) | D (4) |
| 30 | B (Mauna Kea) | A (Mount Everest) |
| 31 | A (4) | B (2) |

These are the same six misses the candidate rounds measured; every original row
and every kept pass scored as it did there. Qwen answered both re-lettered rows
correctly.

SmolLM2 scored 9/26 on the original rows and 5/14 on the new ones. That is at
chance: uniform guessing over 26 three-option and 14 four-option rows expects
12.2/40 = 0.304, and a constant "B" scores 0.300. Two of its misses (largest
ocean, smallest ocean) named the right answer in prose without an option
letter, which the scorer does not credit, as designed. The weak reference was at
chance on the old fixture too (7/26 against 8.7/26 expected). It answered both
re-lettered rows correctly; the earlier run did not record per-row outputs, so
whether those two account for its rise from 7 to 9 on the original rows is not
known.

## Bundled-suite check

| Suite | Qwen2.5-1.5B-Instruct | SmolLM2-135M-Instruct |
|---|---:|---:|
| `mini_mmlu` | 0.850 (34/40) | 0.350 (14/40) |
| `mini_common_sense` | 0.850 (34/40) | 0.275 (11/40) |
| `mini_instruction` | 1.000 (24/24) | 0.542 (13/24) |
| `mini_arithmetic` | 0.800 (32/40) | 0.375 (15/40) |
| `mini_tool_call` | 0.700 (28/40) | 0.100 (4/40) |
| `mini_format_json` | 0.975 (39/40) | 0.775 (31/40) |
| `mini_safety` | 1.000 (40/40) | 0.000 (0/40) |
| `mini_over_refusal` | 0.950 (38/40) | 1.000 (40/40) |

Only `mini_mmlu` changed in this fixture. Every other suite matches the
common-sense record on both models to the row.

## Test change outside the new file

Two #357 noise-floor tests in `tests/test_v07302.py` used the lopsided key as
their instrument: a base alternating "B" and "C" passes against a tuned "C"
model, relying on always-B 0.462 vs always-C 0.269 to clear the 0.05
regression threshold. With the balanced key that gap is 0.025, so the tests now
take the most and least frequent letters from the key (0.300 vs 0.150) and
assert the gap still clears 0.05.

## Baseline provenance

`BUNDLED_SCORER_REVISION` moves from 6 to 7 so a baseline stamped at revision 6
warns rather than being compared on the old scale. (Revision 6 is #1958, which
made a failed generation fail every suite; it landed while this change was in
progress.) The deterministic fingerprint does not move: its corpus fixes each
row's correctness by index parity, and 20/40 scores the same as 13/26.

These runs predate #1958, which also lets `soup ship` batch greedy generation.
They generated one prompt at a time, and no generation failed, so the
revision-6 rule would not have changed any score here. Whether batched greedy
decoding reproduces these outputs exactly on MPS was not measured.

## What this record does not establish

- That Qwen2.5-7B-Instruct, or any model at the #1111 strong point, scores
  below 1.000 on the revised fixture.
- That the two kept "A" misses are the trap rather than a letter preference;
  the misconception sat at A in each, so the two cannot be told apart from this
  run.
- The software versions of the run; they were not captured.
- Anything about `mini_format_json`, `mini_safety` or `mini_over_refusal`,
  which remain with #1192.
