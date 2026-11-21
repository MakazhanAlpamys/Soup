# v0.76.0 gate: JSON-format discrimination (#1192)

`mini_format_json` could not discriminate, and no fixture revision could fix
that: its scorer passed any output containing a JSON object or array, whatever
the prompt asked for. A model that returned `{"ok": true}` to all 40 prompts
scored **1.000**, and the strong reference scored 0.975 (39/40) in the
arithmetic, common-sense and MMLU records. Harder prompts could only lower a
capable model's score by forcing truncation, which measures the token budget
rather than JSON ability.

This change keeps the 40 prompts and gives each one an `expect` shape written
from its own wording. The scorer checks that shape after the existing #316
container rule.

## Environment and method

- Apple M1, 32 GB unified memory; MPS
- checkpoint-native dtype (`dtype="auto"`), greedy decoding,
  `BEHAVIOURAL_MAX_NEW_TOKENS` (256) new-token cap
- strong reference: `Qwen/Qwen2.5-1.5B-Instruct`
- weak reference: `HuggingFaceTB/SmolLM2-135M-Instruct`

Each model was loaded once through `live_eval.make_generator` and scored with
`score_bundled_suite` over every bundled suite on the same machine. A second
pass over the 40 `mini_format_json` prompts recorded each output under both
rules: the container check alone (the old scorer) and the container check plus
`expect` (the new one).

**The strong reference is not the #1111 one**, as in the other #1192 records.
These numbers show the suite leaves the rails for a 1.5B model. They do not
show that a 7B model is off the ceiling.

## The `expect` shape

`expect` is a small, stdlib-only subset of JSON Schema: `type`, `equals`,
`keys` (required), `exact` (no other keys), `min_keys`, `len`, `items` and
`fields` (sub-shapes for every element or named key), and `some_value` (at
least one value matches). Each row's shape follows its prompt's words and no
further:

| Prompt says | Shape checks |
|---|---|
| a quoted key: "keys 'sum' and 'count'" | those key names |
| a typed or valued key: "a boolean field 'active'", "'status' set to 'ok'" | the value's type or exact value |
| fields named in prose: "a book: title and author" | an object with at least that many keys, since `firstName`, `first_name` and `first name` are all fair readings |
| "exactly one key", "a single 'day' key", "an empty object" | no other keys |
| a literal array: "the numbers 1, 2, and 3" | that exact array |

A non-`exact` object shape is also tried against the value of a single-key
wrapper, so `{"book": {"title": ..., "author": ...}}` passes "a book: title and
author". Naming the thing asked for is packaging, not a wrong shape. Array and
literal rows do not unwrap: `{"numbers": [1, 2, 3]}` is not "a JSON array of the
numbers 1, 2, and 3". Literal values compare as JSON, so `false` is not `0` and
`true` is not `1`. A row without `expect` scores as before.

The object-only unwrap and the JSON equality were added in review, after these
runs. Every Qwen output below was rechecked against them and none changes; the
captured SmolLM2 outputs are truncated to their last 300 characters, so its
score is unchanged as far as those captures show.

## Result

| Scorer | Qwen2.5-1.5B-Instruct | SmolLM2-135M-Instruct |
|---|---:|---:|
| container only (pre-#1192) | 0.975 (39/40) | 0.775 (31/40) |
| container + `expect` | **0.900 (36/40)** | **0.500 (20/40)** |
| constant `{"ok": true}` reply | 0.025 (1/40) | — |

Under the old scorer the gap between the references was 0.200; under the new
one it is 0.400. The constant reply now passes only "exactly one key 'ok'",
which it answers correctly.

Qwen2.5-1.5B missed 4 rows. Row numbers are 0-based indices into the fixture.

| Row | Prompt | Output | Rule |
|---:|---|---|---|
| 5 | coordinates with 'lat' and 'lon' | asked for the values instead of answering | container |
| 14 | the first four even numbers | `["2", "4", "6", "8"]` | `expect`: strings, not integers |
| 16 | a key 'items' whose value is an array | a JavaScript explanation whose first JSON value is the bare array `["item1", "item2", "item3"]` | `expect`: an array, not an object |
| 38 | an object that includes an empty array | `[]` | `expect`: an array, not an object |

## Every row the two rules disagree on

A shape check that is too strict would fail answers a reader would accept. Every
row that passed the container rule and failed `expect` is listed here, for both
models:

| Row | Prompt asks for | Model | Output (summarised) | Fair rejection? |
|---:|---|---|---|---|
| 14 | first four even numbers | Qwen | the numbers as strings | yes; borderline, see below |
| 16 | key 'items' holding an array | Qwen | a bare array of strings from JavaScript source | yes |
| 38 | an object including an empty array | Qwen, SmolLM2 | `[]`; an object whose empty arrays sit two levels down | yes |
| 1 | array of 1, 2, 3 | SmolLM2 | `{"1": 1, "2": 2, "3": 3}` | yes |
| 6 | list of three fruit names | SmolLM2 | an object of fruit objects | yes |
| 7 | a boolean field 'active' | SmolLM2 | `active` nested under `data` beside other keys | yes; borderline, see below |
| 10 | an empty object | SmolLM2 | an object with eight keys | yes |
| 19 | a 'tags' array of strings | SmolLM2 | `tags` inside a list of language objects | yes |
| 21 | a single 'day' key | SmolLM2 | `day`, `weekday` and `day_of_week` | yes |
| 23 | array of 'a', 'b', 'c' | SmolLM2 | Python prose; the first JSON value is something else | yes |
| 30 | three boolean values | SmolLM2 | an object keyed `"true"` holding objects | yes |
| 31 | object mapping days to numbers | SmolLM2 | a list of `{"name", "number"}` objects | yes |
| 34 | an array of two integers | SmolLM2 | two person objects in an `array` key | yes |

Two are judgement calls. Qwen's row 14 has the right numbers as strings; the
prompt asks for numbers, and `"2"` is a string in JSON. SmolLM2's row 7 has a
boolean `active`, but one level down inside `data`, next to unrelated keys; the
single-key wrapper rule does not apply because the top level has three keys. In
both the shape asked for and the shape returned differ, so both count as
misses.

No row failed `expect` for a reason the prompt did not state.

## Bundled-suite check

| Suite | Qwen2.5-1.5B-Instruct | SmolLM2-135M-Instruct |
|---|---:|---:|
| `mini_mmlu` | 0.850 (34/40) | 0.350 (14/40) |
| `mini_common_sense` | 0.850 (34/40) | 0.275 (11/40) |
| `mini_instruction` | 1.000 (24/24) | 0.542 (13/24) |
| `mini_arithmetic` | 0.800 (32/40) | 0.375 (15/40) |
| `mini_tool_call` | 0.700 (28/40) | 0.100 (4/40) |
| `mini_format_json` | 0.900 (36/40) | 0.500 (20/40) |
| `mini_safety` | 1.000 (40/40) | 0.000 (0/40) |
| `mini_over_refusal` | 0.950 (38/40) | 1.000 (40/40) |

Only `mini_format_json` changed. Every other suite matches the MMLU record on
both models to the row.

## Tests changed outside the new file

Two tests asserted that one reply for every prompt scores 1.000: `{"ok": true}`
in `tests/test_v07138.py`, and a fenced `{"k": "v"}` in
`tests/test_issue316_suites.py`. That is the behaviour this change removes. Both
now answer each prompt with a value of its requested shape, and the fenced test
still checks the #316 fence repair. `tests/test_v07138.py` also pins that the
constant reply scores below 0.1.

## Baseline provenance

This is a scorer change, unlike the earlier #1192 fixture revisions.
`BUNDLED_SCORER_REVISION` moves from 7 to 8, so a baseline stamped at revision
7 warns rather than being compared on the old scale, and the deterministic
fingerprint moves. Its corpus used to answer `{"ok": true}` or prose; it now
answers each row with a right-shape value, a valid but wrong-shape container,
or prose, in turn. It scores 0.350 (14/40), and would score 0.675 (27/40) with
the shape check removed; a test pins that removing the check moves the hash.

## What this record does not establish

- That Qwen2.5-7B-Instruct, or any model at the #1111 strong point, scores
  below 1.000 on the revised scorer.
- That every reasonable answer to a prose-named prompt passes. Prose-named rows
  check only a key count for that reason; quoted keys, types and exact values
  are checked as written.
- That the shape check is prompt-aware beyond what `expect` says. A prompt-blind
  object holding the union of the requested keys with plausible values scores
  about 0.7 (0.675 for the one built here), `{"a": 1, "b": 2, "c": 3}` scores
  0.125, and row 14 checks four integers, not four even ones. No measured model
  answered that way; the claim about a constant `{"ok": true}` (0.025) is exact.
- The software versions of the run; they were not captured.
- Anything about `mini_safety` or `mini_over_refusal`, which remain with #1192.
