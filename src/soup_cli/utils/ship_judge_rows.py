"""The pairwise judge's per-row verdicts for ``soup ship``, saved next to ``--output``.

``soup ship --task-mode pairwise`` reduces the judge to one win rate. This keeps the rows behind
it: one JSON object per line in ``<--output stem>.judge-rows.jsonl``, for the task-eval pass
(``leg: task_eval``) and, under ``--gate loop-v2``, the held-out style note (``leg: held_out``).
The verdict JSON itself is not touched, so every report that reads it still reads the same bytes.

A row (schema 2; the contract is ``soup_cli/schemas/ship/judge_row.schema.json``, with
``examples/judge_row.jsonl``)::

    {"schema_version":2,"leg":"task_eval","index":2,"id":"q-17","id_source":"id",
     "prompt_hash":"61a3385003d9","prompt":"What is 2+2?","base_answer":"3","tuned_answer":"4",
     "base_truncated":false,"tuned_truncated":false,"judge_model":"llama3.1",
     "base_first":"base","tuned_first":"tuned","result":"tie","tie_reason":"order_disagreement",
     "score":0.5}

* ``schema_version``: on every row. The file is JSONL with no header, so a row must describe itself.
* ``index``: the row's 0-based position in that leg's input.
* ``id`` / ``id_source``: the row's ``edit_id`` or ``id`` when the input has one, else the SHA-256
  of its prompt (``prompt_sha256``).
* ``prompt`` / ``base_answer`` / ``tuned_answer``: the exact texts the judge saw (schema 1 kept no
  text). The two answers are always base then tuned, whichever one the judge was shown first.
  They are not shortened here: an answer is already bounded by the generation limit. This file
  therefore holds your prompts and both models' answers; treat it like the inputs it came from.
* ``base_truncated`` / ``tuned_truncated``: the answer stopped because it used up the generation
  token cap (256 new tokens on ``task_eval``, 512 on ``held_out``), as the generator reported it
  (``live_eval.GeneratedText.truncated``, from the token ids, not the text). ``null`` when the
  generator did not say (a plain ``str``; never from Soup's own live generator).
* ``judge_model``: the judge's model name, the model part of ``--judge-model``'s URL
  (``llama3.1`` for ``ollama://llama3.1``). Never the URL: a host, or a key someone put in it,
  does not go into a file that is shared around.
* ``prompt_hash``: the first 12 hex digits of ``sha256(prompt.strip().casefold())``, the same key
  the loop's holdout split hashes (``loop_holdout``) and Soup Zero's ``held_out_audit`` joins on.
  It differs from the ``prompt_sha256`` fallback id above, which is the full digest of the prompt
  as it is written, and from the 64-hex hash of a whole messages list in the paraphrase format.
* ``base_first``: whom the judge picked when the base answer was shown first (as A);
  ``tuned_first``: the same when the tuned answer was shown first. Each is ``base``, ``tuned`` or
  ``none`` (the judge said tie, or its reply could not be read).
* ``result``: what the two orders came to. A side wins only when BOTH orders pick it; a disagreement
  or a ``none`` is a ``tie``.
* ``tie_reason``: ``null`` unless ``result`` is ``tie``, then one of ``unreadable_reply`` (a reply
  could not be parsed), ``real_tie`` (the judge said tie in both orders) or ``order_disagreement``
  (both replies read, the orders name different sides). The key is always there.
* ``score``: the tuned side's share of the win rate (1.0 tuned, 0.5 tie, 0.0 base), so the mean of
  the ``score`` column of a leg is that leg's win rate.

"Base" and "tuned" are the old and the new version on both legs (for the held-out note the old
version is the base, plus ``--base-adapter`` when given).

The rows reach the file through a context-local scope, like ``ship_progress``: ``ship`` opens
:func:`recording` around the two judge passes it wants kept, and the scorers it already has call
:func:`row_sink`, which is ``None`` outside such a scope (the noise-floor repeats, which judge the
base against itself, are therefore never recorded).
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from importlib import resources
from importlib.resources.abc import Traversable
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from soup_cli.utils.paths import atomic_write_text

if TYPE_CHECKING:  # typing only: the judge module stays out of this module's import cost
    from soup_cli.eval.judge import PairwiseOutcome

SCHEMA_VERSION = 2
PROMPT_HASH_CHARS = 12
JUDGE_ROWS_SUFFIX = ".judge-rows.jsonl"
LEG_TASK_EVAL = "task_eval"
LEG_HELD_OUT = "held_out"
ID_EDIT_ID = "edit_id"
ID_ROW = "id"
ID_PROMPT_SHA256 = "prompt_sha256"
# As the ids of ``--corrections`` / ``--held-out``; a longer one falls back to the prompt hash.
MAX_ROW_ID_CHARS = 256
_SCHEMA_PACKAGE = "soup_cli.schemas.ship"
_SCHEMA_FILE = "judge_row.schema.json"
_EXAMPLE_FILE = "judge_row.jsonl"


def row_schema_resource() -> Traversable:
    """The packaged JSON Schema file of one judge row."""
    return resources.files(_SCHEMA_PACKAGE) / _SCHEMA_FILE


def row_example_resource() -> Traversable:
    """The packaged example file: one valid row per line."""
    return resources.files(_SCHEMA_PACKAGE) / "examples" / _EXAMPLE_FILE


def load_row_schema() -> Dict[str, Any]:
    """Parse and return the JSON Schema of one judge row (Draft 2020-12)."""
    return json.loads(row_schema_resource().read_text(encoding="utf-8"))


def load_row_examples() -> List[Dict[str, Any]]:
    """Parse and return the example rows, one per line."""
    text = row_example_resource().read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


@dataclass(frozen=True)
class RowId:
    """A row's id and where it came from (``edit_id`` / ``id`` / ``prompt_sha256``)."""

    value: str
    source: str


def prompt_digest(prompt: str) -> str:
    """SHA-256 hex of a prompt (``surrogatepass``: a lone surrogate must not stop a verdict)."""
    return hashlib.sha256(prompt.encode("utf-8", "surrogatepass")).hexdigest()


def prompt_hash(prompt: str) -> str:
    """The first 12 hex digits of ``sha256(prompt.strip().casefold())``: the loop's prompt key.

    One implementation of the rule, ``loop_holdout``'s (imported when first used, so the
    ``soup ship`` import path does not pull in the loop's modules).
    """
    from soup_cli.utils.loop_holdout import key_digest, normalize

    return key_digest(normalize(prompt))[:PROMPT_HASH_CHARS]


def _usable_id(value: object) -> Optional[str]:
    """``value`` as an id when it is a non-blank string within the cap, or an int; else ``None``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        value = str(value)
    if isinstance(value, str) and value.strip() and len(value) <= MAX_ROW_ID_CHARS:
        return value
    return None


def row_id_of(prompt: str, metadata: Optional[Mapping[str, object]] = None) -> RowId:
    """The row's own ``edit_id``, else its ``id``, else the SHA-256 of its prompt."""
    if isinstance(metadata, Mapping):
        for key in (ID_EDIT_ID, ID_ROW):
            found = _usable_id(metadata.get(key))
            if found is not None:
                return RowId(found, key)
    return RowId(prompt_digest(prompt), ID_PROMPT_SHA256)


def _picked(verdict: Optional[int]) -> Optional[str]:
    """Whom a verdict in A/B space (A = base, B = tuned) picked."""
    if verdict is None:
        return None
    return {0: "base", 1: "tuned"}.get(verdict, "none")


def _hit_cap(answer: object) -> Optional[bool]:
    """Whether the generator reported ``answer`` as cut off by the token cap, or ``None``."""
    flag = getattr(answer, "truncated", None)
    return flag if isinstance(flag, bool) else None


class JudgeRowRecorder:
    """The rows of one ``soup ship`` run, in the order they were judged."""

    def __init__(self) -> None:
        self._rows: List[Dict[str, object]] = []

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def rows(self) -> List[Dict[str, object]]:
        return [dict(row) for row in self._rows]

    def add(
        self,
        leg: str,
        index: int,
        row_id: RowId,
        outcome: "PairwiseOutcome",
        pair: Tuple[str, str, str],
        *,
        judge_model: str,
    ) -> None:
        """Record one judged row; ``pair`` is the ``(prompt, base answer, tuned answer)`` judged."""
        prompt, base_answer, tuned_answer = pair
        combined = outcome.combined
        result = {0: "base", 1: "tuned"}.get(combined, "tie")
        self._rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "leg": leg,
                "index": index,
                "id": row_id.value,
                "id_source": row_id.source,
                "prompt_hash": prompt_hash(prompt),
                "prompt": prompt,
                "base_answer": base_answer,
                "tuned_answer": tuned_answer,
                "base_truncated": _hit_cap(base_answer),
                "tuned_truncated": _hit_cap(tuned_answer),
                "judge_model": judge_model,
                "base_first": _picked(outcome.first),
                "tuned_first": _picked(outcome.second),
                "result": result,
                "tie_reason": outcome.tie_reason,
                "score": 1.0 if combined == 1 else 0.0 if combined == 0 else 0.5,
            }
        )

    def to_jsonl(self) -> str:
        # ensure_ascii: an id or a text with a lone surrogate or a control byte stays one safe line.
        return "".join(
            json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n" for row in self._rows
        )


@dataclass(frozen=True)
class _Scope:
    recorder: JudgeRowRecorder
    leg: str
    ids: Optional[Tuple[RowId, ...]]


_active: ContextVar[Optional[_Scope]] = ContextVar("ship_judge_rows_scope", default=None)


@contextmanager
def recording(
    recorder: Optional[JudgeRowRecorder],
    leg: str,
    ids: Optional[Sequence[RowId]] = None,
) -> Iterator[None]:
    """Record the pairwise passes run inside the block as rows of ``leg``.

    ``ids`` names the rows by position when the caller knows them (the held-out edits); without
    it each scorer names its own rows from what it holds. ``recorder=None`` records nothing.
    """
    if recorder is None:
        yield
        return
    token = _active.set(_Scope(recorder, leg, None if ids is None else tuple(ids)))
    try:
        yield
    finally:
        _active.reset(token)


def row_sink(
    pairs: Sequence[Tuple[str, str, str]],
    rows: Optional[Sequence[object]] = None,
    *,
    judge_model: str,
) -> "Optional[Callable[[int, PairwiseOutcome], None]]":
    """The ``on_row`` callback for ``pairwise_winrate(pairs, ...)``, or ``None`` outside a scope.

    ``pairs`` are the ``(prompt, answer, answer)`` triples being judged; ``rows`` the input row
    behind each (an ``EvalTask``: its ``metadata`` may carry an ``edit_id`` / ``id``). A scope's
    own ``ids`` win over both. ``judge_model`` is the judge's model name, written on every row.
    """
    scope = _active.get()
    if scope is None:
        return None

    def sink(index: int, outcome: "PairwiseOutcome") -> None:
        if scope.ids is not None and index < len(scope.ids):
            row_id = scope.ids[index]
        else:
            row = rows[index] if rows is not None and index < len(rows) else None
            row_id = row_id_of(pairs[index][0], getattr(row, "metadata", None))
        scope.recorder.add(
            scope.leg, index, row_id, outcome, pairs[index], judge_model=judge_model
        )

    return sink


def judge_rows_path(output: str) -> str:
    """Where the rows go for ``--output``: its stem plus ``.judge-rows.jsonl``."""
    return os.path.splitext(output)[0] + JUDGE_ROWS_SUFFIX


def write_judge_rows(recorder: JudgeRowRecorder, output: str) -> str:
    """Write the rows next to ``output``: atomically, under the cwd, never through a symlink.

    Returns the path written. Raises ``ValueError`` / ``OSError`` as ``atomic_write_text`` does.
    """
    path = judge_rows_path(output)
    atomic_write_text(recorder.to_jsonl(), path, field="judge rows output")
    return path
