# v0.76.0 gate: tool-call discrimination (#405)

`mini_tool_call` had reached a ceiling: the strong reference model selected the
right tool on every shipped row. This record measures the old scale, rejects an
initial replacement that remained pinned near the ceiling, and records the
fixture selected for v0.76.0.

## Environment and method

- Apple M4 Max, 128 GB unified memory; macOS 26.5.2 (25F84)
- Python 3.12.2; PyTorch 2.12.1; Transformers 5.12.1; MPS
- checkpoint-native dtype (`dtype="auto"`), greedy decoding, 256 new-token cap
- strong reference: `Qwen/Qwen2.5-7B-Instruct`
- weak reference: `HuggingFaceTB/SmolLM2-135M-Instruct`

The same Soup generator and scorer were used for both models. The model was
loaded once per run and every row was generated independently. Tool-call rows
are scored by function name; direct-answer rows pass only when the complete
trimmed output is the literal `NO_TOOL`.

## Scale selection

| Fixture | Rows | Qwen2.5-7B-Instruct | SmolLM2-135M-Instruct | Decision |
|---|---:|---:|---:|---|
| v0.73.2 shipped fixture | 40 | **1.000 (40/40)** | **0.075 (3/40)** | reject: strong model pinned at ceiling |
| first 32-row candidate | 32 | **0.969 (31/32)** | not run | reject: still pinned near ceiling |
| 40-row discrimination candidate pool | 40 | **0.850 (34/40)** | **0.000 (0/40)** | use its first 24 rows |
| final fixture (16 legacy + 24 candidate rows) | 40 | **0.850 (34/40)** | **0.050 (2/40)** | accept |

The final fixture preserves 16 legacy positive tool calls, then adds 16
direct-answer decisions and 8 tool selections from the candidate pool, with
semantically close distractors. On the experimental candidate, the six strong-model failures were
tempting over-calls (`lookup_fact`, `calculator`, `dictionary`, `translate`, or
`set_timer`) on prompts that the model could answer directly. They were not JSON
extraction or truncation failures.

The two scorer repairs from #346 remain explicit controls: naming the right tool
with one missing outer brace passes, while echoing a menu entry does not.

## Bundled-suite ceiling and floor check

The final fixture was measured again from the committed JSONL, alongside every
other bundled suite. Results are filled from that final verification run:

| Suite | Qwen2.5-7B-Instruct | SmolLM2-135M-Instruct |
|---|---:|---:|
| `mini_mmlu` | 1.000 (26/26) | 0.308 (8/26) |
| `mini_common_sense` | 1.000 (24/24) | 0.250 (6/24) |
| `mini_instruction` | 0.958 (23/24) | 0.542 (13/24) |
| `mini_arithmetic` | 1.000 (36/36) | 0.639 (23/36) |
| `mini_tool_call` | 0.850 (34/40) | 0.050 (2/40) |
| `mini_format_json` | 1.000 (40/40) | 0.700 (28/40) |
| `mini_safety` | 0.900 (36/40) | **0.000 (0/40)** |
| `mini_over_refusal` | 1.000 (40/40) | 1.000 (40/40) |

The full check confirms that `mini_tool_call` now has headroom and a clear
strong/weak separation. It also records the remaining rails rather than
silently expanding this issue: the strong model is at 1.000 on MMLU, common
sense, arithmetic, JSON format, and over-refusal; both references are at 1.000
on over-refusal; and the weak model is at the expected 0.000 safety floor.

## Baseline provenance

This fixture changes the meaning and distribution of `mini_tool_call` scores.
`BUNDLED_SCORER_REVISION` therefore moves from 1 to 2 and the deterministic
fingerprint changes with it. Stored baselines stamped with revision 1 are on the
old scale and Soup will warn rather than silently compare them with revision 2.

## Exact-abstention packaging cost

A follow-up maintainer run on Mistral-7B-Instruct-v0.3 (NF4) measured 12 misses on the
NO_TOOL axis: 10 were genuine tool over-calls and 2 were outputs that began
with NO_TOOL and then added an explanation. Thus 2/12 (about 17%) of those
misses are attributable to the intentionally exact packaging contract rather
than the selection decision itself. That strictness is deliberate: NO_TOOL
must be the complete trimmed response. A fenced NO_TOOL response therefore
fails too, even though positive JSON tool calls deliberately tolerate fences.
