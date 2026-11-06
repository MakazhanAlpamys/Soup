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

> **Status.** Soup writes all five formats. `soup serve --trace-log` writes version 1
> conversation lines and `soup serve --feedback-log` writes feedback lines (see
> [Serving and Export](serving-and-export.md#response-ids-and-feedback)). `soup loop`
> writes `iterations.jsonl`, `versions.jsonl` and `progress.jsonl` (see
> [What `soup loop` writes](#what-soup-loop-writes)), next to the
> `.soup-loops/<iteration-id>/iteration.json` manifest it has always written per iteration.
> **One active loop per project:** the files take no lock, and two `soup loop watch`
> processes on one `.soup/loop.yaml` would race on the version counter. Other programs may
> append to the same files; Soup never assumes it is the only writer.

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
| `dataset` | object or null | `path`, `sha256`, `dpo_pairs`, `kto_rows`, `sft_rows` (row counts of the three datasets a harvest builds). |
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

## What `soup loop` writes

`soup loop` appends to `iterations.jsonl`, `versions.jsonl` and `progress.jsonl` under
`.soup-loops/` (under `WatchConfig.iteration_dir` when a program embeds the daemon with
another directory). Every line has `source: soup-loop` where the format has a source.

**How a line is written.** One JSON object, UTF-8 without a BOM, `\n` ending, emitted by a
single `write` call on a file opened in append mode (a short write is continued). An
earlier line is never rewritten: a correction is a new line, and the last line for a key
wins. If the file ends in a line another writer left unfinished, a newline is written
first so the new line stays readable. A path outside the project directory, a symlink, a
junction or a hard link in the way is refused. A line that cannot be written is a warning
and a note on the iteration, and never stops the loop.

**Several writers, no lock.** A desktop application may append to the same files. Soup
decides everything from `.soup/loop.yaml` and never reads a line back, so it cannot be
confused by lines it did not write; but two `soup loop watch` processes on one project
would both change the version counter, and nothing prevents that. Run one.

**Iteration ids.** `soup loop` already names an iteration
`iter-<UTC YYYYMMDDTHHMMSS>-<8 hex>`, the contract's form. That id is the directory of the
manifest `.soup-loops/<id>/iteration.json` and the `iteration_id` of every line, so
`soup loop replay <id>` and the log refer to the same iteration. The first model's seed
line has an id of the same form and no manifest.

**One source per iteration.** The `iterations.jsonl` line and the manifest are built from
the same `IterationRecord` in the same cycle (the record, plus the state the iteration
started from and what the stages returned), so they cannot disagree. The line is also
what the loop keeps in `loop.yaml` (`canary_iteration`) while the iteration's canary runs,
so that the line written when the canary is promoted or withdrawn is a complete snapshot
and not a fragment; it then replaces the first line, because the latest line wins. The
manifest is not rewritten.

### `iterations.jsonl`

A line is appended when an iteration is recorded (a cycle that only waits for ratings, or
that the budget skips, records no iteration of its own) and again when its outcome changes:

| Outcome | When |
|---|---|
| `canary` | The adapter was trained, passed the check and its canary started (`to_version` is the version it will take if promoted; `finished_at` is null). |
| `promoted` | A later cycle promoted the canary; `to_version` is now a version. |
| `rolled_back` | The canary was withdrawn by a regression, by `--canary-on-timeout rollback`, by `soup loop rollback`, or because `soup loop canary` replaced it. `to_version` becomes null. |
| `skipped` | Nothing to train, no new data, a `DON'T SHIP` verdict (`verdict.decision` says so), no verdict reached (`ERROR` or `SKIPPED`), or a deploy that was not attempted (no serve endpoint). |
| `failed` | The training run died, or the deploy was attempted and refused. |

A promotion or a rollback after the iteration was promoted (a person going back the next
day) is a line in `versions.jsonl`, not a change to the iteration. The seed line written by
`soup loop init --pre-wired` (or the first watch cycle) is a promoted iteration with
`reason: "initial deploy"`.

What the stages returned fills the other fields, and what they did not report is null or
left out, never invented: `dataset` has the `path`, its `sha256` and the row counts the
harvest reported (`dpo_pairs`, `kto_rows`, `sft_rows`); `train` the method, `run_id` and
base model, only when a training run happened; `verdict` the decision, the method, the
baseline (`version` is the production version at the start, for a `soup ship` against the
production adapter) and the verdict file; `artifacts.adapter` the trained adapter's path;
`data.until` the data cursor and `data.conversations` the answers in the trace directory
(the per-kind feedback counts are not reported by the harvest and are left out).
`canary.quality` is the evidence of the quality verdict, copied from the record's
`canary_quality`, on the line written when the canary is promoted or withdrawn; the first
`canary` line has none because nothing was rated yet. Paths are written with `/`.

### `versions.jsonl`

| Line | `cause` | `actor` | Written by |
|---|---|---|---|
| The first version | `initial` | `user` | `soup loop init --pre-wired` |
| The first version of a loop whose init wrote none | `initial` | `loop` | the first watch cycle (also the first `soup loop rollback` of a `loop.yaml` from before the log) |
| A canary became production | `promote` | `loop` | the watch daemon |
| Production went back | `rollback` | `user` | `soup loop rollback` |

Only a change of the active version is a line. A canary that is withdrawn (by a regression,
a timeout or `soup loop rollback` with nothing promoted) leaves production where it was,
and so does `soup loop canary`, so neither writes a version line; the withdrawn iteration
ends `rolled_back` in `iterations.jsonl`. The loop never changes production on its own
except by promotion, so there is no `rollback` line with actor `loop`.

Version numbers are integers from 1, kept in `loop.yaml`: `production_version` (the active
one), `previous_production_version` (what `soup loop rollback` returns to) and
`latest_version` (the highest ever promoted). The served model is version 1; a promotion
takes `latest_version + 1`, so a number is never reused after a rollback (v1, v2, back to
v1, then the next promotion is v3). A canary started with `soup loop canary` was started by
no iteration: its promotion is a `promote` line with `iteration_id` null. A `loop.yaml`
written before the counter existed is numbered by the names it remembers: version 2 once
an adapter was promoted, one more when a previous adapter is remembered (the lowest number
consistent with the file; the file kept no count). `iteration_id` is the iteration that
produced `to_version` for a `promote`, and the iteration that produced the version being
left for a `rollback`.

### `progress.jsonl`

Stages `harvest`, `train`, `verdict` (the check before rollout), `deploy`, `canary`,
`promote` and `rollback`, each with `start` and `end` (or `error`, with the code
`train_failed`, `verdict_not_reached`, `deploy_failed` or `promote_refused`). The harvest
and train events are written only for an iteration that has something to train on or a
canary to start; an idle poll adds only its `skipped` iteration line. `done` and `total`
are null for stages that cannot count.

The `canary` stage is the one a person watches: a `start` line with `done: 0` and
`total: 30` when the canary starts (also for `soup loop canary`, with a null
`iteration_id`), then one `progress` line per watch cycle while it waits, then the
`promote` or `rollback` events. `done` is the number of canary answers that were rated and
`total` the minimum the quality verdict needs (both taken from `canary.quality`); when
nobody rates the answers (no feedback log) `done` counts the canary's answers out of 30,
and both are null when the canary's statistics could not be read. At most one `canary` line
is written per cycle while waiting.
