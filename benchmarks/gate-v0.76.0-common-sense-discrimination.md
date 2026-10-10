# v0.76.0 gate: common-sense discrimination (#1192)

`mini_common_sense` had reached a ceiling: the #1111 record measured the strong
reference at **1.000 (24/24)**, and Qwen2.5-1.5B-Instruct scored 0.958 (23/24)
in the arithmetic record. Its answer key was also lopsided: 14 of 24 answers
were B, so a model that only ever answered "B" scored 0.583. This record
measures a 24-row candidate pool, the 16 rows selected from it, and the revised
40-row fixture on two models.

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

**The strong reference is not the #1111 one**, as in the arithmetic record.
These numbers show the fixture leaves the rails for a 1.5B model. They do not
show that a 7B model is off the ceiling.

## Candidate pool and selection

24 hand-authored, original four-option rows: physical cause and effect,
time and sequence, relational reasoning, and "which would NOT" questions. Each
key carries a one-line justification. Correct letters were spread across A-D
by swapping option positions before measurement, so the pool measured is the
pool shipped.

| Pool | Rows | Qwen2.5-1.5B-Instruct | SmolLM2-135M-Instruct |
|---|---:|---:|---:|
| candidate pool | 24 | 14/24 | 5/24 |

Qwen answered every candidate with a bare or parenthesised letter, so no miss
was an extraction or truncation failure.

Selection kept 16 rows:

| Decision | Candidates | Reason |
|---|---|---|
| keep: strong-model misses | departure time, force and mass, falling bodies, time zone, clock arithmetic | reasoning traps a capable model can still get wrong |
| keep: strong-model passes | frozen bottle, candle in a jar, battery drain, spoon handles, two height-order rows, wool shrinkage, stargazing, umbrella in wind, open fish tank, borrowed book | keeps the suite from measuring only the traps |
| reject | black seat, steel key, closed shop, dark-grown seedling | Qwen chose A on each; four "A" misses on facts it plainly knows read as a letter preference, not a common-sense failure |
| reject | foam cooler, wet-floor sign | trivially easy |
| reject | fridge banana, missing sock | weakest answer keys of the pool |

## Result

The final fixture keeps the 24 original rows, appends the 16 selected rows,
and swaps the B and C options of four original rows (door key, falling glass,
plant sunlight, old bread) so no letter answers more than 30% of the suite.
The key is now A 11, B 11, C 12, D 6.

| Fixture | Rows | Qwen2.5-1.5B-Instruct | SmolLM2-135M-Instruct |
|---|---:|---:|---:|
| v0.73.2 fixture (arithmetic record) | 24 | 0.958 (23/24) | 0.208 (5/24) |
| revised fixture | 40 | **0.850 (34/40)** | **0.275 (11/40)** |

Qwen2.5-1.5B missed 6 rows. Row numbers are 0-based indices into
`MINI_COMMON_SENSE`; index 24 is the first new row.

| Row | Kind | Expected | Chosen |
|---:|---|---|---|
| 15 | original | A (grandparent) | B (newborn) |
| 26 | new | A (leave at 4:00) | B (3:00) |
| 28 | new | A (light box moves faster) | D (same speed) |
| 33 | new | C (both land together) | A (the ball) |
| 35 | new | C (6 p.m.) | B (5 p.m.) |
| 39 | new | D (10 o'clock) | B (7 o'clock) |

Row 15 is the same original-row miss the arithmetic record measured. Qwen
answered all four re-lettered rows correctly; SmolLM2 answered two of them.
SmolLM2 scored 7/24 on the original rows and 4/16 on the new ones.
That is chance level, not a skill level: uniform guessing over 24 three-option
and 16 four-option rows expects 12/40 = 0.300, and a constant "A" or "B" scores
0.275, exactly SmolLM2's score. The weak reference was at chance on the old
fixture too (5/24 against 8/24 expected).

## Bundled-suite check

| Suite | Qwen2.5-1.5B-Instruct | SmolLM2-135M-Instruct |
|---|---:|---:|
| `mini_mmlu` | 1.000 (26/26) | 0.269 (7/26) |
| `mini_common_sense` | 0.850 (34/40) | 0.275 (11/40) |
| `mini_instruction` | 1.000 (24/24) | 0.542 (13/24) |
| `mini_arithmetic` | 0.800 (32/40) | 0.375 (15/40) |
| `mini_tool_call` | 0.700 (28/40) | 0.100 (4/40) |
| `mini_format_json` | 0.975 (39/40) | 0.775 (31/40) |
| `mini_safety` | 1.000 (40/40) | 0.000 (0/40) |
| `mini_over_refusal` | 0.950 (38/40) | 1.000 (40/40) |

Only `mini_common_sense` changed in this fixture. Every other suite matches the
arithmetic record on both models to the row, so the run reproduces on this
machine. The `mini_safety` and `mini_over_refusal` rows here are on the
post-#1335 refusal classifier and still agree with the earlier run.

## Baseline provenance

`BUNDLED_SCORER_REVISION` moves from 4 to 5 so a baseline stamped at revision 4
warns rather than being compared on the old scale. The deterministic
fingerprint does not move: its corpus fixes each row's correctness by index
parity, and 20/40 scores the same as 12/24.

## What this record does not establish

- That Qwen2.5-7B-Instruct, or any model at the #1111 strong point, scores
  below 1.000 on the revised fixture.
- That the four rejected "A" misses are a letter preference rather than error;
  that is the reading of four misses on easy rows, not a measured bias.
- The software versions of the run; they were not captured.
- Anything about `mini_mmlu`, `mini_format_json`, `mini_safety` or
  `mini_over_refusal`, which remain with #1192.
