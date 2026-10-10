# v0.76.0 gate: arithmetic discrimination (#1192)

`mini_arithmetic` had reached a ceiling: the #1111 record measured the strong
reference at **1.000 (36/36)** on the shipped fixture. This record measures the
revised 40-row fixture on two models, neither of which sits on a rail.

## Environment and method

- Apple M1, 32 GB unified memory; macOS
- checkpoint-native dtype (`dtype="auto"`), greedy decoding,
  `BEHAVIOURAL_MAX_NEW_TOKENS` (256) new-token cap
- strong reference: `Qwen/Qwen2.5-1.5B-Instruct`
- weak reference: `HuggingFaceTB/SmolLM2-135M-Instruct`

Each model was loaded once through `live_eval.make_generator` and scored with
`score_bundled_suite` over every bundled suite, on the same machine.

**The strong reference is not the #1111 one.** Qwen2.5-7B-Instruct did not run
on this machine, so these numbers show that the fixture leaves the rails for a
1.5B model. They do not show that a 7B model is off the ceiling.

## Fixture

The first 24 single-step rows of the v0.73.2 fixture are kept. The remaining
12 are replaced by 16 multi-step or large-operand rows (multi-digit products,
order of operations, compound percentages, remainders, LCM, fraction sums,
digit counting). No new answer appears as a standalone token in its own
question, so a model that echoes the prompt cannot score.

## Result

| Fixture | Rows | Qwen2.5-1.5B-Instruct | SmolLM2-135M-Instruct |
|---|---:|---:|---:|
| revised fixture | 40 | **0.800 (32/40)** | **0.375 (15/40)** |

SmolLM2 solved 15 of the 24 kept rows and none of the 16 new ones.

Qwen2.5-1.5B missed 8 rows. Row numbers are 0-based indices into
`MINI_ARITHMETIC`; index 24 is the first new row (`47 x 38`).

| Row | Kind | Failure |
|---:|---|---|
| 21 | kept | wrong answer (10 percent of 200 -> 5) |
| 24 | new | wrong product (47 x 38 -> 1796) |
| 25 | new | wrong product (1234 x 56 -> 70064) |
| 31 | new | wrong product (999 x 999 -> 994009) |
| 29 | new | truncated at 256 tokens before stating an answer |
| 37 | new | truncated at 256 tokens before stating an answer |
| 38 | new | truncated at 256 tokens before stating an answer |
| 39 | new | truncated at 256 tokens before stating an answer |

Half of the strong-model misses are the token budget, not the arithmetic. The
leg-2 gate generates under the same 256-token cap, so this is how `soup ship`
scores these rows, but a verbose model can lose them without an arithmetic
error.

## Bundled-suite check

| Suite | Qwen2.5-1.5B-Instruct | SmolLM2-135M-Instruct |
|---|---:|---:|
| `mini_mmlu` | 1.000 (26/26) | 0.269 (7/26) |
| `mini_common_sense` | 0.958 (23/24) | 0.208 (5/24) |
| `mini_instruction` | 1.000 (24/24) | 0.542 (13/24) |
| `mini_arithmetic` | 0.800 (32/40) | 0.375 (15/40) |
| `mini_tool_call` | 0.700 (28/40) | 0.100 (4/40) |
| `mini_format_json` | 0.975 (39/40) | 0.775 (31/40) |
| `mini_safety` | 1.000 (40/40) | 0.000 (0/40) |
| `mini_over_refusal` | 0.950 (38/40) | 1.000 (40/40) |

Only `mini_arithmetic` changed in this fixture. The run predates #1335, which
changed the refusal classifier behind `mini_safety` and `mini_over_refusal`,
so those two rows are on the pre-#1335 scorer; `mini_arithmetic` does not use
that classifier and is unaffected. SmolLM2's untouched suites
differ from the #1111 record by one or two rows (for example `mini_mmlu` 7/26
here, 8/26 there), which is machine and stack drift between an M1 and an
M4 Max. Compare scores within one record, not across records.

## Baseline provenance

`BUNDLED_SCORER_REVISION` moves from 3 to 4 so a baseline stamped at revision 3
warns rather than being compared on the old scale. The deterministic
fingerprint does not move: its corpus fixes each row's correctness by index
parity, and 20/40 scores the same as 18/36.

## What this record does not establish

- That Qwen2.5-7B-Instruct, or any model at the #1111 strong point, scores
  below 1.000 on the revised fixture.
- That the four truncated rows measure arithmetic rather than verbosity.
- Anything about the other saturated suites, which remain with #1192.
