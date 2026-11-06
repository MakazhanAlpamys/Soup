# Loop data formats

[← Back to the docs index](README.md)

The `soup loop` cycle (serve → feedback → training data → train → verdict → canary →
promote or roll back) passes data between programs through six append-only JSONL logs.
This page defines them so that every producer (`soup serve`, `soup loop`, or an
application built on Soup such as a desktop workbench) writes the same lines, and every
reader can rely on them.

Each format has a JSON Schema (Draft 2020-12) and an example file shipped inside the
package:

| Format | One line per | Default path | Schema | Example |
|---|---|---|---|---|
| conversation | model answer | `.soup-loops/traces/*.jsonl` | `soup_cli/schemas/loop/conversation.schema.json` | `soup_cli/schemas/loop/examples/conversation.jsonl` |
| feedback | feedback action on one answer | `.soup-loops/feedback.jsonl` | `.../feedback.schema.json` | `.../examples/feedback.jsonl` |
| paraphrase | edited answer | `.soup-loops/paraphrases.jsonl` | `.../paraphrase.schema.json` | `.../examples/paraphrase.jsonl` |
| iteration | step of one loop iteration | `.soup-loops/iterations.jsonl` | `.../iteration.schema.json` | `.../examples/iteration.jsonl` |
| version | change of the active version | `.soup-loops/versions.jsonl` | `.../version.schema.json` | `.../examples/version.jsonl` |
| progress | progress event | `.soup-loops/progress.jsonl` | `.../progress.schema.json` | `.../examples/progress.jsonl` |

From Python, `soup_cli.utils.loop_formats` returns them: `load_schema(name)`,
`load_examples(name)`, `schema_resource(name)`, `FORMATS`, `DEFAULT_LAYOUT`,
`SCHEMA_VERSION`, and `prompt_hash(messages)` (see [paraphrase](#paraphrase)).
`jsonschema` is not a runtime dependency of Soup; install it to validate lines yourself.

> **Status.** Soup writes all six formats. `soup serve --trace-log` writes version 1
> conversation lines and `soup serve --feedback-log` writes feedback lines (see
> [Serving and Export](serving-and-export.md#response-ids-and-feedback)). `soup loop`
> writes `iterations.jsonl`, `versions.jsonl` and `progress.jsonl` (see
> [What `soup loop` writes](#what-soup-loop-writes)), next to the
> `.soup-loops/<iteration-id>/iteration.json` manifest it has always written per iteration,
> and, before it trains, `paraphrases.jsonl` (see [Who writes it](#who-writes-it)).
> The `verdict.corrections` / `verdict.held_out` evidence is measured by
> `soup ship --gate loop-v2` and written into the iteration line by `soup loop` (see
> [The loop-v2 verdict](#the-loop-v2-verdict)); Soup Zero writes the same objects.
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

## paraphrase

One line per edited answer: stable paraphrases of the prompt of an answer a person
corrected. The loop and the Zero desktop app use them to check whether a new model
version learned the correction (the version is asked the paraphrases, not the original
prompt). They are generated once per answer and reused, so every version is checked
against the same questions. **They are never training data.**

| Field | Type | Meaning |
|---|---|---|
| `response_id` | UUID | The conversation line's `id`: the edited answer. Required. |
| `prompt_hash` | 64 lowercase hex characters | SHA-256 of the normalised prompt (below). Required. |
| `paraphrases` | array of non-empty strings, at least 1 | Rewordings of the prompt. Required. |
| `generator` | `{model, template: {id, version}}` | The model that wrote the paraphrases and the instruction template it was given, all non-empty strings. Required. |
| `created_at` | time | When the paraphrases were generated. Required. |
| `dropped` | integer ≥ 0 | Optional. How many generated paraphrases the generator's filters dropped. |
| `dropped_reasons` | array of `same_as_prompt`, `leaks_answer` | Optional. Why paraphrases were dropped. |

**`prompt_hash`** is the lowercase SHA-256 hex digest of the *normalised prompt*: the
conversation line's `messages` (the prompt without the answer), serialised as JSON with
sorted keys and no extra whitespace (`json.dumps(messages, sort_keys=True,
separators=(",", ":"), ensure_ascii=False)` in Python), encoded as UTF-8. Non-ASCII
characters are written as themselves, not as `\uXXXX` escapes. Nothing is trimmed,
case-folded or reordered. `soup_cli.utils.loop_formats.prompt_hash(messages)` computes
it; a reader recomputes it to see whether the prompt of the answer is still the one that
was paraphrased.

**Which line counts.** The last line per `response_id` wins (file order). A writer that
regenerates the paraphrases (a new template version, say) appends a new line.

**What the generator drops.** It rejects a paraphrase that normalises to the original
prompt (`same_as_prompt`) or contains the words the correction added (`leaks_answer`),
regenerates once, and drops it if it fails again. `paraphrases` can therefore hold fewer
items than were requested, but never none: if nothing survives, **no line is written**.

### Who writes it

`soup loop`'s pre-wired train stage, before it trains, for every edit in the training set that
has no current line: current means the last line for its `response_id` has the `prompt_hash` of
the edit's prompt, so an edit is generated once and every later iteration reuses it. It
follows [Soup Zero](https://github.com/SoupTeam/SoupZero)'s method: the prompt that is
paraphrased is the last user message; one request asks for two paraphrases as a numbered list
(template `loop-paraphrase`, version `1`, temperature 0.7); a paraphrase is dropped when it
equals the prompt after case-folding and removing punctuation and spacing (`same_as_prompt`) or
holds at least half of the words the correction added (`leaks_answer`: the content words of the
edit that the answer it corrected lacked, all of the edit's content words when that is fewer
than two); when fewer than two survive the same request is made once more, never a third time;
`dropped` counts every paraphrase dropped across both requests and `dropped_reasons` names each
reason once, in the order it first happened. The model is `loop.yaml`'s `paraphraser_model` at
its `paraphraser_endpoint`, else the loop's own serve endpoint, which answers the call with the
`X-Soup-Internal: loop-check` header and the tool token so that it is not recorded (see
[Internal Calls](serving-and-export.md#internal-calls-x-soup-internal) and
[the flywheel](adapters-and-governance.md#data-flywheel-soup-loop)). The paraphrases are never
training data.

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
| `dataset` | object or null | `path`, `sha256`, `dpo_pairs`, `kto_rows`, `sft_rows` (row counts of the three datasets a harvest builds), `graduated` (edits that moved from the held-out set into training in this iteration: an edit whose prompt hash puts it in the held-out 30% is kept aside for exactly its first iteration, then trained on). |
| `train` | object or null | `method` (`dpo`, `kto`, ...), `run_id`, `base` (what training actually loaded: a base model id or a merged earlier version), `root_base` (the original base model id when `base` is a merged version). |
| `verdict` | object or null | `decision`: `SHIP` or `DON'T SHIP` (from `soup ship`), or `SKIPPED` / `ERROR` when no verdict was reached; `method`: `ship` or `eval_gate`; `gate`: optional, `loop-v2` when `soup ship --gate loop-v2` made the decision (absent = the plain verdict); `baseline`: `{kind: base or version, ref, version}`, what the candidate was compared against (`version` is required when `kind` is `version`); `path` of the report; `reason`; and the optional evidence objects `corrections` and `held_out` (below). |
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

**`verdict.corrections`** (check 1, *corrections learned*) and **`verdict.held_out`**
(check 2, *non-inferiority on held-out edits*) are optional and independent: absent means
that check was not measured, and a line without them stays valid. When one is present, all
of its fields are present.

An edit is **fixed** when the new version's hits on its paraphrases (see
[paraphrase](#paraphrase)) exceed the old version's, **broken** when they are lower, and a
**tie** otherwise; `n_edits` counts the edits. A *hit* is decided by `scoring` (below).

| `corrections` field | Type | Meaning |
|---|---|---|
| `fixed`, `broken`, `ties` | integers ≥ 0 | Edits by the rule above. |
| `n_edits` | integer ≥ 0 | The edits checked: `fixed + broken + ties`. |
| `n_paraphrases` | integer ≥ 0 | The paraphrases used, summed over the edits. |
| `p_value` | number in [0, 1] or null | One-sided sign test over `fixed` vs `broken`: the chance of at least this many `fixed` if fixing and breaking were equally likely. null when `fixed + broken` is 0. |
| `scoring` | object | How a hit was decided (below). |
| `outcome` | enum | `pass`, `fail`, `insufficient` (the iteration had edits but fewer than 5 discordant ones, `fixed + broken`) or `not_applicable` (the training set held no edits, for example a thumbs-only loop, so there were no corrections to check; promotion is then decided by the canary's ratings). |

`held_out` checks that learning the corrections did not cost anything else. The held-out
edits are edits kept out of training; they are scored with the same metric against their
corrected text and paired by prompt. **Non-inferiority** means the new version is not
significantly below the old one on that metric.

| `held_out` field | Type | Meaning |
|---|---|---|
| `better`, `worse`, `ties` | integers ≥ 0 | Held-out edits the new version scored higher on, lower on, or the same as the old version. |
| `n` | integer ≥ 0 | The held-out edits scored: `better + worse + ties`. |
| `p_value` | number in [0, 1] or null | One-sided sign test over `worse` vs `better`: the chance of at least this many `worse` if the two versions were equally good. null when `better + worse` is 0. |
| `scoring` | object | The same shape as in `corrections`. Required. |
| `outcome` | enum | `pass` (not significantly below the old version), `fail` (significantly below), `weak` (fewer than 10 held-out edits: too few to trust a pass) or `not_measured` (fewer than 5 held-out edits). |
| `style` | object | Optional, information only: `preferred_new`, `preferred_old`, `n` (integers ≥ 0) and `p_value` (number in [0, 1] or null) of the pairwise style comparison. It is recorded but never part of the outcome. |

**`scoring`** (in both objects) is `{method, threshold, template}`, all three present:
`method` is `added_recall` or `reference_judge`; `threshold` is a number in [0, 1] or null;
`template` is `{id, version}` or null.

- `added_recall` scores what the correction *added*. The added words are the edit's
  content words that the original answer lacked, and an answer is a hit when at least
  `threshold` (0.5) of them are present in it. When an edit adds fewer than 2 content
  words, all of its content words are used instead.
- `reference_judge` asks a judge model to compare the answer with the corrected text;
  `template` names the judge's instruction template.

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
harvest reported (`dpo_pairs`, `kto_rows`, `sft_rows`) and, for a loop whose check is
`soup ship`, `graduated` (below); `train` the method, `run_id` and
base model, only when a training run happened; `verdict` the decision, the method, the
baseline (`version` is the production version at the start, for a `soup ship` against the
production adapter), the verdict file and, for `soup ship --gate loop-v2`, `gate`,
`corrections`, `held_out` and the gate's `reason` for a `DON'T SHIP`; `artifacts.adapter` the
trained adapter's path;
`data.until` the data cursor and `data.conversations` the answers in the trace directory
(the per-kind feedback counts are not reported by the harvest and are left out). For an
iteration that trained, `data.until` is the newest feedback it read, which is the cursor the
data cursor moves to if its canary is promoted; for one that trained nothing it is the data
cursor (below).
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

A loop whose check is `soup ship` can also wait for corrections before it trains: a
`harvest` line with the event `progress`, `done` the trainable corrections and `total` the 5
needed, and the words in `message` (see [The loop-v2 verdict](#the-loop-v2-verdict)). It is
written once per change of the message, not once per poll.

### The loop-v2 verdict

A loop whose check is `soup ship` (`soup loop init --verify ship`, the default) calls
`soup ship --gate loop-v2 --corrections FILE --held-out FILE` (docs/evaluation.md), so
the verdict asks whether the new version **learned the corrections** people made and **kept
the edits held out of training**. `verdict.gate` is `loop-v2`, `verdict.corrections` and
`verdict.held_out` are what `soup ship` measured (`--output` writes them in these exact
shapes; a field that does not follow this page is left out of the line, never trusted), and a
`DON'T SHIP` carries the gate's reason in `verdict.reason`. The rules are Soup Zero's
(`zero_server/loop/harvest.py`), so both programs split the same edits:

- **The split.** The key of an edit is the content of the last user message of the corrected
  answer's conversation, `text.strip().casefold()`; the edit is held out iff
  `int(sha256(key.encode("utf-8")).hexdigest(), 16) % 100 < 30`. Fixtures (text, value):
  `How do I reset my password?` 87, `  How do I RESET my password?  ` 87, `Straße öffnen – wie?`
  12, `Line one\nline two\twith tab` 5, `Привет, как сбросить пароль?` 83. An answer with no
  user message that has text has no key and is never held out. Thumbs and ratings without an
  edit are never held out: they train as they always did.
- **One iteration aside, then graduation.** A held-out prompt is kept out of training in
  exactly one iteration, the first one that trains; every edit on it then is that iteration's
  held-out check (`--held-out`: `edit_id`, `prompt`, the last user message as written,
  `expected`, `original`; identical corrections count once). From the next iteration on it
  trains and joins the corrections check, and is never kept aside again, whatever the verdict
  of the iteration that checked it was. An iteration that trained nothing (it waited for more
  corrections) or whose training failed does not use the keep-aside up. `dataset.graduated`
  counts the edits that moved from the held-out set into training in the iteration.
  Which edit was kept aside and which graduated, in which iteration, is
  `.soup-loops/holdout.jsonl`, written by the iteration that trained (it is the loop's own
  small log, not one of the six formats): `schema_version`, `ts`, `iteration_id`, `event`
  (`kept_aside` or `graduated`), `edit_id`, `prompt_key` (the SHA-256 of the key, not the
  prompt) and `until` (the cursor the iteration would set). Append only; the last row per
  `edit_id` wins.
- **The corrections check** (`--corrections`: `edit_id`, `expected`, `original`,
  `paraphrases`) covers the edits trained in the iteration, the new ones and the graduated
  ones, that have a current line in `paraphrases.jsonl`; a trained edit without one is left out
  and the iteration's notes count it.
- **The lineage cursor.** The data cursor (`harvest_cursor_ts` in `.soup/loop.yaml`; the `data.until`
  of an iteration is what that iteration read up to) follows the version in production: it moves
  when the canary that used the rows is promoted, and `soup loop rollback` returns it to the
  previous production's. The rows
  of a rejected candidate (a `DON'T SHIP`, a withdrawn or rolled-back canary, a canary that
  never started) are new again, so they are trained and counted again. What triggers training
  still counts feedback newer than the later of that cursor and the newest feedback a training
  run read (`attempt_cursor_ts`), so a rejected run is not repeated on its own rows. A
  `loop.yaml` from before takes its cursor as production's.
- **The plan-time wait.** The edits that would train in a cycle are the new ones (newer than the
  lineage cursor) left after the keep-aside, plus the graduated ones. With none, there are no
  corrections to check and the usual trigger (8 new edits or 30 new rated answers) decides. With
  one to four the cycle trains nothing and spends no GPU: `Waiting: 4/5 trainable corrections
  (2 more kept aside for checking) · 41 new ratings`. That is the iteration's note, a
  `harvest` progress line and a `waiting` row of `soup loop status`; the edits keep
  accumulating. With five or more it trains, as long as the usual trigger allows it.
- **When nothing is left to check.** With no corrections to check and the general suites
  holding, `soup ship` answers `DON'T SHIP` with `canary_decides: true`. The loop still
  deploys the candidate to a canary and its ratings decide (note: `No corrections to check - the
  canary's ratings decide`); the iteration line says `decision: "DON'T SHIP"`, `outcome:
  "canary"`. A loop without a feedback log has no ratings, so such a canary is judged by request
  errors alone. Any other `DON'T SHIP` stops the rollout.

A loop whose check is the eval gate (`--verify eval-gate`) keeps no edit aside and does not
wait for corrections; only the lineage cursor applies to it.
