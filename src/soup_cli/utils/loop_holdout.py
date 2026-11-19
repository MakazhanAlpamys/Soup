"""The loop's holdout split, graduation, plan-time wait and lineage cursor (the "loop-v2" rules).

The loop's verdict (``soup ship --gate loop-v2``, ``utils/loop_gate.py``) asks whether a new
version learned the corrections people made and kept the edits held out of training. Two rules,
shared with the consuming application, decide which edits those are:

- **The holdout split.** The key of an edit is the last user message of the corrected answer's
  conversation, ``text.strip().casefold()``. The edit is held out iff
  ``int(sha256(key), 16) % 100 < 30``. An answer with no user message that has text has no key
  and is never held out; thumbs and ratings without an edit are never held out either.
- **One iteration aside, then graduation.** A held-out prompt is kept out of training in exactly
  ONE iteration, the first one that trains (an iteration that trains nothing, or whose training
  failed, does not use it up): its edits are that iteration's held-out check. From the next
  iteration on they train, count in the corrections check (they *graduate*), and never go back.
  Which edit was kept aside and which graduated, in which iteration, is an append-only log,
  ``.soup-loops/holdout.jsonl`` (last row per ``edit_id`` wins), written by the iteration that
  trains; a row carries ``until``, the data cursor that iteration would set.
- **The plan-time wait.** The edits that would TRAIN this cycle are the new ones (newer than the
  lineage cursor, ``LoopState.harvest_cursor_ts``) left after the keep-aside, plus the ones that
  graduate. Fewer than ``MIN_TRAINABLE_EDITS`` (5) and at least one: the cycle trains nothing and
  says what is missing.

An edit has *trained in the lineage* when the version in production, or an ancestor, was trained
on it. That is read off the cursor, not off a list of versions: a ``graduated`` row whose
``until`` the lineage cursor has reached was trained into a promoted candidate (the cursor moves
only on promotion, and back on rollback). A candidate that is thrown away leaves the cursor
behind its rows, so they are trainable again.

Pure functions and one small file. Nothing here imports torch or reads a model.
"""

from __future__ import annotations

import hashlib
import math
import os
import posixpath
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from soup_cli.utils.loop_feedback import EditedAnswer, LogStats, _iter_objects, is_new
from soup_cli.utils.loop_formats import SCHEMA_VERSION
from soup_cli.utils.loop_log import LOG_DIR, append_json_line, utc_now_iso
from soup_cli.utils.loop_state import MAX_CURSOR_TS

#: A prompt is held out iff its bucket (0-99) is under this.
HOLDOUT_PERCENT = 30
#: Fewer trainable corrections than this cannot give the corrections check a verdict
#: (``soup ship``'s minimum of edits that changed either way); the loop does not train for them.
MIN_TRAINABLE_EDITS = 5
#: The ledger of kept-aside and graduated edits.
HOLDOUT_LOG = posixpath.join(LOG_DIR, "holdout.jsonl")
KEPT_ASIDE = "kept_aside"
GRADUATED = "graduated"
_EVENTS = (KEPT_ASIDE, GRADUATED)
_DIGEST = re.compile(r"[0-9a-f]{64}")
_MAX_ID_CHARS = 256
_MAX_ITERATION_CHARS = 128


# ---------------------------------------------------------------------------
# The split
# ---------------------------------------------------------------------------


def normalize(text: str) -> str:
    """The key form of a prompt: outer whitespace trimmed, case-folded (not lower-cased)."""
    return text.strip().casefold()


def _digest(key: str) -> str:
    # ``surrogatepass``: a lone surrogate in a logged message must not stop the loop. Text that
    # encodes at all hashes exactly as the strict encoding does.
    return hashlib.sha256(key.encode("utf-8", "surrogatepass")).hexdigest()


def holdout_bucket(text: str) -> int:
    """Where a prompt falls, 0-99: ``int(sha256(key), 16) % 100``, ``key = strip().casefold()``."""
    return int(_digest(normalize(text)), 16) % 100


def is_held_out(text: str) -> bool:
    """Whether a prompt is on the held-out side of the split (a bucket under 30)."""
    return holdout_bucket(text) < HOLDOUT_PERCENT


def last_user_text(messages: object) -> Optional[str]:
    """The text of the last user message of a prompt, or ``None`` when it has none.

    The last user message decides, as in the consuming application: if it holds no text the
    prompt has no key, whatever an earlier user message said.
    """
    if not isinstance(messages, (list, tuple)):
        return None
    for message in reversed(messages):
        if isinstance(message, Mapping) and message.get("role") == "user":
            content = message.get("content")
            return content if isinstance(content, str) and content.strip() else None
    return None


def holdout_key(messages: object) -> Optional[str]:
    """The key the split hashes: the last user message, trimmed and case-folded; or ``None``."""
    text = last_user_text(messages)
    return None if text is None else normalize(text)


def key_digest(key: str) -> str:
    """SHA-256 hex of a key: what the ledger keeps instead of the prompt."""
    return _digest(key)


# ---------------------------------------------------------------------------
# The ledger: holdout.jsonl
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LedgerEntry:
    """The row in force for one edited answer."""

    event: str
    edit_id: str
    iteration_id: str
    #: The data cursor the iteration would set (what it read), or ``None`` when it had none.
    until: Optional[float]
    #: ``key_digest`` of the edit's holdout key.
    prompt_key: str


@dataclass
class Ledger:
    """A ledger read: the row in force per edit and every prompt digest ever recorded."""

    latest: "dict[str, LedgerEntry]" = field(default_factory=dict)
    keys: "set[str]" = field(default_factory=set)
    stats: LogStats = field(default_factory=LogStats)


def _text_field(value: object, limit: int) -> Optional[str]:
    if isinstance(value, str) and value and len(value) <= limit and "\x00" not in value:
        return value
    return None


def _entry(obj: Mapping) -> Optional[LedgerEntry]:
    """``obj`` as a version 1 ledger row, or ``None`` when it is not one."""
    version = obj.get("schema_version")
    if type(version) is not int or version != SCHEMA_VERSION:
        return None
    event = obj.get("event")
    edit_id = _text_field(obj.get("edit_id"), _MAX_ID_CHARS)
    iteration_id = _text_field(obj.get("iteration_id"), _MAX_ITERATION_CHARS)
    prompt_key = obj.get("prompt_key")
    if event not in _EVENTS or edit_id is None or iteration_id is None:
        return None
    if not (isinstance(prompt_key, str) and _DIGEST.fullmatch(prompt_key)):
        return None
    until = obj.get("until")
    if until is not None:
        if isinstance(until, bool) or not isinstance(until, (int, float)):
            return None
        if not (0 <= until <= MAX_CURSOR_TS):  # False for NaN too
            return None
        until = float(until)
    return LedgerEntry(event, edit_id, iteration_id, until, prompt_key)


def read_ledger(path: Optional[str] = None) -> Ledger:
    """Read ``holdout.jsonl``: the last row per ``edit_id`` wins (file order).

    A missing file is an empty ledger; a line that is not a version 1 ledger row is skipped and
    counted in ``stats.skipped``. The read is bounded like the other logs.
    """
    target = HOLDOUT_LOG if path is None else path
    ledger = Ledger()
    if not os.path.isfile(target):
        return ledger
    for obj in _iter_objects(target, ledger.stats):
        row = _entry(obj)
        if row is None:
            ledger.stats.skipped += 1
            continue
        ledger.latest[row.edit_id] = row
        ledger.keys.add(row.prompt_key)
    return ledger


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HoldoutPlan:
    """Which corrected answers this cycle keeps aside, trains and graduates."""

    #: Kept out of training in this iteration: the first sighting of a held-out prompt. They are
    #: the iteration's held-out check, every edit on such a prompt together.
    kept_aside: "tuple[EditedAnswer, ...]" = ()
    #: Newer than the lineage cursor and not kept aside.
    new_trainable: "tuple[EditedAnswer, ...]" = ()
    #: Held-out-side edits that train in this iteration and had not trained in the lineage.
    graduated: "tuple[EditedAnswer, ...]" = ()
    #: ``new_trainable`` and ``graduated`` together, each edit once: the corrections check.
    trainable: "tuple[EditedAnswer, ...]" = ()


def _trained_in_lineage(edit: EditedAnswer, cursor: Optional[float], ledger: Ledger) -> bool:
    """Whether the version in production (or an ancestor) was trained on this edit.

    With a ledger row: it graduated in an iteration whose cursor the lineage cursor has reached
    (that candidate, or a later one that read the same rows, was promoted). Without one, the edit
    was never kept aside: it trained as any edit does once the cursor passed it.
    """
    row = ledger.latest.get(edit.id)
    if row is None:
        return not is_new(edit.ts, cursor)
    return (
        row.event == GRADUATED
        and row.until is not None
        and cursor is not None
        and row.until <= cursor
    )


def plan_holdout(
    edits: Sequence[EditedAnswer], cursor: Optional[float], ledger: Ledger
) -> HoldoutPlan:
    """Split the corrected answers into kept aside, new, graduating and trainable.

    ``cursor`` is the lineage cursor (``LoopState.harvest_cursor_ts``). A prompt is *seen* when the
    ledger holds it (an iteration that trained already kept it aside) or an edit on it is not new
    (the cursor has passed it). An edit whose prompt hashes under 30 and is not seen is kept aside,
    together with every other edit on that prompt; one on a seen prompt trains.
    """
    keyed = []
    seen = set(ledger.keys)
    for edit in edits:
        key = holdout_key(edit.messages)
        digest = None if key is None else key_digest(key)
        keyed.append((edit, digest, key is not None and is_held_out(key)))
        if digest is not None and not is_new(edit.ts, cursor):
            seen.add(digest)
    kept, new, graduated = [], [], []
    for edit, digest, held in keyed:
        if held and digest not in seen:
            kept.append(edit)
            continue
        if is_new(edit.ts, cursor):
            new.append(edit)
        if held and not _trained_in_lineage(edit, cursor, ledger):
            graduated.append(edit)
    wanted = {edit.id for edit in new} | {edit.id for edit in graduated}
    trainable = tuple(edit for edit in edits if edit.id in wanted)
    return HoldoutPlan(tuple(kept), tuple(new), tuple(graduated), trainable)


def waiting_note(trainable: int, kept_aside: int, new_rated: int) -> Optional[str]:
    """What a cycle says when it will not train for want of corrections, else ``None``.

    From one to four trainable corrections: ``Waiting: 4/5 trainable corrections (2 more kept
    aside for checking) · 41 new ratings``. None at zero (nothing to check: the usual
    trigger decides) and from five up.
    """
    if not 1 <= trainable < MIN_TRAINABLE_EDITS:
        return None
    text = f"Waiting: {trainable}/{MIN_TRAINABLE_EDITS} trainable corrections"
    if kept_aside:
        text += f" ({kept_aside} more kept aside for checking)"
    ratings = f"{new_rated} new rating" + ("" if new_rated == 1 else "s")
    return f"{text} · {ratings}"


def record_plan(
    plan: HoldoutPlan,
    *,
    iteration_id: str,
    until: Optional[float],
    path: Optional[str] = None,
) -> "list[str]":
    """Append the plan of the iteration that trained to the ledger; returns notes for failures.

    One ``kept_aside`` row for each edit kept out of training and one ``graduated`` row for each
    edit that graduated, all carrying the iteration and the cursor it would set. Nothing earlier
    is touched. A row that cannot be written ends the writing and gives one note (the exception's
    class, never a path): the loop goes on.
    """
    target = HOLDOUT_LOG if path is None else path
    stamp = utc_now_iso()
    rows = [(KEPT_ASIDE, edit) for edit in plan.kept_aside]
    rows += [(GRADUATED, edit) for edit in plan.graduated]
    for event, edit in rows:
        key = holdout_key(edit.messages)
        if key is None:
            continue
        line = {
            "schema_version": SCHEMA_VERSION,
            "ts": stamp,
            "iteration_id": iteration_id,
            "event": event,
            "edit_id": edit.id,
            "prompt_key": key_digest(key),
            "until": until,
        }
        try:
            append_json_line(target, line)
        except (OSError, ValueError, TypeError) as exc:
            return [f"holdout.jsonl not written ({type(exc).__name__})"]
    return []


# ---------------------------------------------------------------------------
# The lineage cursor
# ---------------------------------------------------------------------------


def valid_cursor(value: object) -> Optional[float]:
    """``value`` as a data cursor (epoch seconds in range), or ``None`` for anything else."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if not (0 <= value <= MAX_CURSOR_TS):
        return None
    return float(value)


def later(left: Optional[float], right: Optional[float]) -> Optional[float]:
    """The larger of two optional cursors."""
    if left is None:
        return right
    return left if right is None or left >= right else right


def promotion_cursor_changes(state: Any) -> "dict[str, Any]":
    """The cursor fields a promotion of the running canary changes.

    The canary's rows are now in production's lineage: the data cursor moves to the canary's (it
    never moves back), the old one is kept for ``soup loop rollback``, and the canary's own is
    spent. A canary no iteration started has no cursor, so the data cursor stays.
    """
    return {
        "harvest_cursor_ts": later(state.harvest_cursor_ts, state.canary_cursor_ts),
        "previous_harvest_cursor_ts": state.harvest_cursor_ts,
        "canary_cursor_ts": None,
    }


def rollback_cursor_changes(state: Any) -> "dict[str, Any]":
    """The cursor fields ``soup loop rollback`` changes: back to the previous production's.

    A running canary is withdrawn with it, and so is its cursor. A ``loop.yaml`` written before
    the previous cursor was kept has none: when rolling back to an earlier adapter the cursor is
    left as it is, because a cursor of ``None`` there would train on everything again.
    """
    changes: "dict[str, Any]" = {"canary_cursor_ts": None}
    if state.previous_harvest_cursor_ts is None and state.previous_production_name is not None:
        return changes
    changes["harvest_cursor_ts"] = state.previous_harvest_cursor_ts
    changes["previous_harvest_cursor_ts"] = None
    return changes
