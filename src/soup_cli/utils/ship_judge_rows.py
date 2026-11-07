"""The pairwise judge's per-row verdicts for ``soup ship``, saved next to ``--output``.

``soup ship --task-mode pairwise`` reduces the judge to one win rate. This keeps the rows behind
it: one JSON object per line in ``<--output stem>.judge-rows.jsonl``, for the task-eval pass
(``leg: task_eval``) and, under ``--gate loop-v2``, the held-out style note (``leg: held_out``).
The verdict JSON itself is not touched, so every report that reads it still reads the same bytes.

A row::

    {"schema_version":1,"leg":"task_eval","index":2,"id":"q-17","id_source":"id",
     "base_first":"base","tuned_first":"tuned","result":"tie","score":0.5}

* ``index``: the row's 0-based position in that leg's input.
* ``id`` / ``id_source``: the row's ``edit_id`` or ``id`` when the input has one, else the SHA-256
  of its prompt (``prompt_sha256``). No prompt or answer text is written, as in the other reports.
* ``base_first``: whom the judge picked when the base answer was shown first (as A);
  ``tuned_first``: the same when the tuned answer was shown first. Each is ``base``, ``tuned`` or
  ``none`` (a tie, or a reply that could not be read: the judge reports both as one "no pick").
* ``result``: what the two orders came to. A side wins only when BOTH orders pick it; a disagreement
  or a ``none`` is a ``tie``.
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
from typing import TYPE_CHECKING, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from soup_cli.utils.paths import atomic_write_text

if TYPE_CHECKING:  # typing only: the judge module stays out of this module's import cost
    from soup_cli.eval.judge import PairwiseOutcome

SCHEMA_VERSION = 1
JUDGE_ROWS_SUFFIX = ".judge-rows.jsonl"
LEG_TASK_EVAL = "task_eval"
LEG_HELD_OUT = "held_out"
ID_EDIT_ID = "edit_id"
ID_ROW = "id"
ID_PROMPT_SHA256 = "prompt_sha256"
# As the ids of ``--corrections`` / ``--held-out``; a longer one falls back to the prompt hash.
MAX_ROW_ID_CHARS = 256


@dataclass(frozen=True)
class RowId:
    """A row's id and where it came from (``edit_id`` / ``id`` / ``prompt_sha256``)."""

    value: str
    source: str


def prompt_digest(prompt: str) -> str:
    """SHA-256 hex of a prompt (``surrogatepass``: a lone surrogate must not stop a verdict)."""
    return hashlib.sha256(prompt.encode("utf-8", "surrogatepass")).hexdigest()


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


class JudgeRowRecorder:
    """The rows of one ``soup ship`` run, in the order they were judged."""

    def __init__(self) -> None:
        self._rows: List[Dict[str, object]] = []

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def rows(self) -> List[Dict[str, object]]:
        return [dict(row) for row in self._rows]

    def add(self, leg: str, index: int, row_id: RowId, outcome: "PairwiseOutcome") -> None:
        combined = outcome.combined
        result = {0: "base", 1: "tuned"}.get(combined, "tie")
        self._rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "leg": leg,
                "index": index,
                "id": row_id.value,
                "id_source": row_id.source,
                "base_first": _picked(outcome.first),
                "tuned_first": _picked(outcome.second),
                "result": result,
                "score": 1.0 if combined == 1 else 0.0 if combined == 0 else 0.5,
            }
        )

    def to_jsonl(self) -> str:
        # ensure_ascii: an id with a lone surrogate or a control byte stays one safe line.
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
) -> "Optional[Callable[[int, PairwiseOutcome], None]]":
    """The ``on_row`` callback for ``pairwise_winrate(pairs, ...)``, or ``None`` outside a scope.

    ``pairs`` are the ``(prompt, answer, answer)`` triples being judged; ``rows`` the input row
    behind each (an ``EvalTask``: its ``metadata`` may carry an ``edit_id`` / ``id``). A scope's
    own ``ids`` win over both.
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
        scope.recorder.add(scope.leg, index, row_id, outcome)

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
