# Loop data formats

[← Back to the docs index](README.md)

The `soup loop` cycle (serve → feedback → training data → train → verdict → canary →
promote or roll back) passes data between programs through five append-only JSONL logs.
This page defines them so that every producer (`soup serve`, `soup loop`, or an
application built on Soup such as a desktop workbench) writes the same lines, and every
reader can rely on them.

Each format has a JSON Schema (Draft 2020-12) and an example file shipped inside the
package:

| Format | One line per | Default path | Schema | Example |
|---|---|---|---|---|
| conversation | model answer | `.soup-loops/traces/*.jsonl` | `soup_cli/schemas/loop/conversation.schema.json` | `soup_cli/schemas/loop/examples/conversation.jsonl` |
| feedback | feedback action on one answer | `.soup-loops/feedback.jsonl` | `.../feedback.schema.json` | `.../examples/feedback.jsonl` |
| iteration | step of one loop iteration | `.soup-loops/iterations.jsonl` | `.../iteration.schema.json` | `.../examples/iteration.jsonl` |
| version | change of the active version | `.soup-loops/versions.jsonl` | `.../version.schema.json` | `.../examples/version.jsonl` |
| progress | progress event | `.soup-loops/progress.jsonl` | `.../progress.schema.json` | `.../examples/progress.jsonl` |

From Python, `soup_cli.utils.loop_formats` returns them: `load_schema(name)`,
`load_examples(name)`, `schema_resource(name)`, `DEFAULT_LAYOUT`, `SCHEMA_VERSION`.
`jsonschema` is not a runtime dependency of Soup; install it to validate lines yourself.

> **Status.** This page defines the formats. The writers move to them in follow-up
> changes: `soup serve --trace-log` writes version 1 conversation lines and
> `soup serve --feedback-log` writes feedback lines (see
> [Serving and Export](serving-and-export.md#response-ids-and-feedback)), while
> `soup loop watch` still writes one `.soup-loops/<iteration-id>/iteration.json`
> manifest per iteration. Nothing in Soup writes `iterations.jsonl`, `versions.jsonl`
> or `progress.jsonl` yet.

## Rules shared by every format

- **JSONL, append only.** One JSON object per line, UTF-8. A change is a new line; a
  writer never edits or deletes an earlier line.
- **`schema_version: 1`** on every line. A line without `schema_version` is version 0
  (see [conversation](#conversation)); a reader that meets a higher version it does not
  know skips the line.
- **Time** is ISO 8601 in UTC with the suffix `Z` or `+00:00`, for example
  `2026-11-05T11:00:00+00:00`.
- **Ids** (`id`, `response_id`) are UUIDs in canonical lowercase form. Two exceptions:
  `conversation_id` is whatever key the client chose (up to 512 characters), and
  `iteration_id` is any string that is safe as a directory name (letters, digits, `.`,
  `_`, `-`, no `..`); `soup loop` writes `iter-<UTC YYYYMMDDTHHMMSS>-<8 hex>`, which sorts
  in time order.
- **Unknown fields are ignored** by readers. New optional fields can appear without a
  new `schema_version`.
- **Extensions:** a field named `x_<app>` (for example `x_zero`) belongs to the
  application that wrote it. It must hold a JSON object. Soup never reads, rewrites or
  depends on it.
- **No invented numbers.** A count, latency or progress total that was not measured is
  `null`, never `0` or `100`.
- **`source`** names the surface that wrote the line, as a lowercase name. Known values:
  `soup-serve`, `soup-loop`, and the Soup Zero surfaces `playground`, `try_it`, `proxy`.

## conversation

One line per model answer. `id` is the answer's id: feedback refers to it.

| Field | Type | Meaning |
|---|---|---|
| `id` | UUID | The answer. Required. |
| `conversation_id` | string or null | The client's conversation key; the canary router hashes it, so one conversation stays in one bucket. |
| `ts` | time | When the answer was completed. Required. |
| `model` | string | The model as served: a `soup serve` model id or path, or an Ollama tag such as `myproj:v2`. Required. |
| `version` | integer ≥ 1 or null | The loop version that answered: 1 is the first deployed model, each promoted iteration adds 1. |
| `adapter` | string or null | The adapter that produced the answer; null for the base model. |
| `canary_bucket` | `stable`, `canary` or null | The bucket the request was routed to; null when no canary split was active. |
| `messages` | array of `{role, content}` | What the model saw, oldest first, without the answer. Roles: `system`, `developer`, `user`, `assistant`, `tool`. Required. |
| `response` | string | The answer text. Required. |
| `finish_reason` | string or null | Why generation stopped (`stop`, `length`; Ollama calls it `done_reason`). |
| `latency_ms` | number or null | Request to completed answer. |
| `tokens` | `{prompt, completion}` | Token counts as the engine reported them, each an integer or null. Required. |
| `options` | object | Generation parameters (`temperature`, `top_p`, `max_tokens`, ...). |
| `source` | string | See the shared rules. Required. |

Secrets (`hf_...`, `sk-...`, `Bearer ...` shapes) are redacted by the writer before the
line reaches disk, as `soup serve --trace-log` already does.

**Version 0** is the line `soup serve --trace-log` wrote before it moved to version 1:
`{"ts": <epoch seconds>, "prompt": <last user message>, "response": ..., "latency_ms": ..., "tokens": <completion tokens>}`.
It has no id, so feedback cannot point at it. Readers that accept version 0 read
`prompt` as a single user message.

## feedback

One line per feedback action on one answer.

| Field | Type | Meaning |
|---|---|---|
| `id` | UUID | This line. Required. |
| `response_id` | UUID | The conversation line's `id`. Required. |
| `ts` | time | When the feedback was given. Required. |
| `kind` | `thumb`, `rating`, `edit`, `comment` | Required. |
| `value` | see below | `thumb`: `1` (up), `-1` (down) or `null` (cleared). `rating`: an integer 1-5 or `null` (cleared). Not used by `edit` and `comment`. |
| `edited_text` | string or null | `edit`: the answer as the user corrected it; `null` withdraws an earlier edit. Required for `edit`. |
| `comment` | string or null | Free text; required for `comment`, optional next to any other kind. |
| `user_ref` | string or null | An opaque reference the application chose for the person (a single desktop user writes `local`). Soup stores no other author data. |
| `source` | string | Required. |

**Which line counts.** For `thumb`, `rating` and `edit`, the latest line per
`(response_id, kind)` wins: file order within one file, `ts` when a reader merges files.
`comment` lines accumulate; every one counts.

## iteration

A snapshot of one harvest → train → verdict → canary iteration. A writer appends a line
each time the iteration advances (`running` → `canary` → `promoted`); **the latest line
per `iteration_id` wins**. The canary verdict arrives long after training ends, which is
why an iteration is not a single line.

| Field | Type | Meaning |
|---|---|---|
| `iteration_id` | string | Required. |
| `parent_iteration_id` | string or null | The iteration whose version this one started from. |
| `from_version` / `to_version` | integer ≥ 1 or null | The version active at the start / the version this iteration produces. |
| `started_at` | time | Required. |
| `finished_at` | time or null | Set when the outcome is final. |
| `data` | object or null | `until`: the cursor (conversation and feedback lines up to and including this time were read; the next iteration reads only newer ones); `conversations`; `feedback` counts per kind (`thumb_up`, `thumb_down`, `rating`, `edit`, `comment`). |
| `dataset` | object or null | `path`, `sha256`, `dpo_pairs`, `kto_rows`. |
| `train` | object or null | `method` (`dpo`, `kto`, ...), `run_id`, `base` (what training actually loaded: a base model id or a merged earlier version), `root_base` (the original base model id when `base` is a merged version). |
| `verdict` | object or null | `decision`: `SHIP` or `DON'T SHIP` (from `soup ship`), or `SKIPPED` / `ERROR` when no verdict was reached; `method`: `ship` or `eval_gate`; `baseline`: `{kind: base or version, ref, version}`, what the candidate was compared against (`version` is required when `kind` is `version`); `path` of the report; `reason`. |
| `canary` | object or null | `traffic_pct` (0-100), `samples` per bucket (`stable`, `canary`), `verdict` (`OK`, `MAJOR`, `UNKNOWN` while samples are too few), and an optional `quality` object (below). |
| `artifacts` | object or null | What serves this version again: `adapter`, `merged`, `gguf` paths, and `served_as` (the adapter name in `soup serve`, or an Ollama tag). |
| `estimated_cost_usd` | number or null | The budget gate's estimate. |
| `outcome` | enum | `running`, `canary`, `promoted`, `rolled_back` (the canary was withdrawn), `skipped` (nothing to train, or no verdict), `failed` (a stage errored). Required. |
| `reason` | string or null | Why, in plain words. |

**`canary.quality`** is optional (omit it when nothing judged answer quality; a line
without it stays valid) and carries the evidence behind the answer-quality verdict. When
it is present, all of its fields are present:

| Field | Type | Meaning |
|---|---|---|
| `verdict` | enum | `OK` (no significant regression was found, which is not the same as proven equal), `MAJOR`, `UNKNOWN` (a bucket is still below `min_rated`) or `INSUFFICIENT`. `INSUFFICIENT` means the maximum wait ran out below the minimum. |
| `rated` | `{stable, canary}`, integers ≥ 0 | Answers that received feedback, per bucket. |
| `negative` | `{stable, canary}`, integers ≥ 0 | Rated answers with negative feedback, per bucket. |
| `share` | `{stable, canary}`, each a number in [0, 1] or null | `negative / rated`; null while a bucket has no rated answers. |
| `difference` | number in [-1, 1] or null | Canary share minus stable share. |
| `z` | number or null | Pooled two-proportion z statistic; positive means the canary is worse. null when it is undefined. |
| `p_value` | number in [0, 1] or null | One-sided p-value of `z`. |
| `min_rated` | integer ≥ 1 | Rated answers needed in each bucket before the verdict is anything but `UNKNOWN` or `INSUFFICIENT`. |
| `threshold` | number in [0, 1] | The tolerated rise in the negative share (`0.05` = 5 percentage points). |
| `timed_out` | boolean | True when the maximum wait ran out before `min_rated` was reached. |
| `decided_on` | enum | `errors_and_quality`, or `errors_only` when the rollout was decided on the error rates alone. |

**The first model** is a seed line: `parent_iteration_id` and `from_version` null,
`to_version: 1`, `outcome: promoted`, `reason: "initial deploy"`, and a matching
`initial` line in the version log.

## version

One line each time the active version changes. **The active version is the
`to_version` of the last line.** A rollback after an iteration finished (for example a
person going back from v2 to v1 the next day) is a line here, not a change to the
iteration.

| Field | Type | Meaning |
|---|---|---|
| `id` | UUID | Required. |
| `ts` | time | When the switch took effect. Required. |
| `from_version` | integer ≥ 1 or null | null only for `initial`. Required. |
| `to_version` | integer ≥ 1 | Required. |
| `cause` | `initial`, `promote`, `rollback` | Required. |
| `actor` | `loop` or `user` | The automatic cycle, or a person. Required. |
| `iteration_id` | string or null | The iteration that produced `to_version` (promote) or whose canary was withdrawn (rollback). |
| `served_as` | string or null | The name `to_version` is served under. |
| `alias` | string or null | A moving name that now points at `to_version` (for example the Ollama model `myproj`). |
| `reason` | string or null | |
| `source` | string | |

## progress

One line per progress event. The shape is that of `soup ship --progress jsonl` (`event`,
`done`, `total`, and `phase` / `side` / `suite` / `repeat` while a verdict runs) plus
`ts`, `iteration_id` and `stage`. Lines carry no prompts, answers or user data.

| Field | Type | Meaning |
|---|---|---|
| `iteration_id` | string or null | null for a job run outside an iteration. Required (may be null). |
| `ts` | time | Required. |
| `stage` | lowercase name | Open vocabulary. `soup loop`: `harvest`, `train`, `verdict`, `deploy`, `canary`, `promote`, `rollback`. Soup Zero: `data`, `merge`, `train`, `ship`, `export`, `deploy`. Required. |
| `event` | `start`, `progress`, `end`, `error` | Required. |
| `done` / `total` | integer or null | null when the stage cannot count. |
| `message` | string or null | Short status text for a person. |
| `error` | string or null | A short code on `error` (`soup ship` writes `execution_aborted`). |
| `job_id` | string or null | The producer's job id, to link a job log. |
| `phase`, `side`, `suite`, `repeat` | | Forwarded from `soup ship --progress jsonl`. |
