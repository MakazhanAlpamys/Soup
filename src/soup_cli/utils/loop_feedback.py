"""Training data from the conversation and feedback logs (the `soup loop` data contract).

Two kinds of append-only JSONL log feed the loop:

- **conversation records**, one line per answer, in the trace directory (``*.jsonl``,
  default ``.soup-loops/traces``): ``id``, ``conversation_id``, ``ts``, ``messages`` (what
  the model saw, without the answer), ``response``, ``adapter``, ``canary_bucket``, ...
  A line without ``schema_version`` is version 0, the line ``soup serve --trace-log`` wrote
  before the format existed (``prompt`` / ``response``, no id): feedback cannot point at it.
- **feedback records**, one line per action on one answer, in the feedback log (default
  ``.soup-loops/feedback.jsonl``): ``response_id`` (the conversation record's ``id``),
  ``ts``, ``kind`` (``thumb``, ``rating``, ``edit``, ``comment``), ``value``,
  ``edited_text``, ``comment``.

Soup Zero and ``soup serve`` both write these files, so this module reads them by the
rules of the contract, not by one producer's code:

- for ``thumb``, ``rating`` and ``edit`` the latest line per ``(response_id, kind)`` wins
  (file order); a thumb or rating with ``value: null`` and an edit with
  ``edited_text: null`` clear the earlier one; ``comment`` lines accumulate;
- readers ignore unknown fields and skip a line whose ``schema_version`` they do not know;
- feedback about an id no conversation record has is ignored, and counted.

From one reading it builds three datasets, whatever method the loop trains with. Each row
carries ``ts``, the time of the newest feedback line it came from (``None`` when that line
has no usable ``ts``), which is what the loop's data cursor compares:

``dpo``
    An edit is a preference: ``chosen`` is the edited text, ``rejected`` the original answer.
``kto``
    One row labelled for the answer as given: a thumb up (+1) or a rating of 4-5 is
    ``true``, a thumb down (-1) or a rating of 1-2 is ``false`` (a rating of 3 says
    nothing), an edit is ``false``. When an answer has several of these, the newest line
    decides, so the dataset never contradicts itself. An edit adds a second row: its
    edited text, ``true``.
``sft``
    An edit is the answer the model should have given: the prompt plus the edited text.

The prompt of every row is the conversation's ``messages`` as logged (a list of
``{"role", "content"}``), the layout ``soup train`` accepts for ``dpo`` and ``kto`` rows
(``prompt`` as a message list) and, with the edited answer appended, for the ``chatml``
format that ``sft`` reads. An edit identical to the original answer (apart from
surrounding whitespace) changes nothing and gives no row.

Only the standard library is used (plus ``parse_ts`` for the ``ts`` fields). A bad line is
skipped and counted, never raised: a log is written by other programs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Collection, Iterator, Mapping, Optional

from soup_cli.data.traces.parsers import parse_ts

SCHEMA_VERSION = 1
#: A line longer than this many bytes is skipped (a chat history with a long document fits).
MAX_LINE_BYTES = 1024 * 1024
#: Bytes read from one log file; the rest of a larger file is ignored.
MAX_FILE_BYTES = 256 * 1024 * 1024
_MAX_ID_LEN = 256
_MAX_COMMENT_CHARS = 10_000
_FEEDBACK_KINDS = ("thumb", "rating", "edit", "comment")
_BUCKETS = ("stable", "canary")
_BOM = b"\xef\xbb\xbf"


@dataclass
class LogStats:
    """What one reading of a log (or a directory of logs) saw."""

    #: Non-blank lines read.
    lines: int = 0
    #: Lines not used: not JSON, not an object, too long, an unknown ``schema_version``, or
    #: a record missing something the contract requires.
    skipped: int = 0
    #: A file was longer than ``MAX_FILE_BYTES`` and its tail was not read.
    truncated: bool = False


def _iter_objects(path: str, stats: LogStats) -> Iterator[dict]:
    """The JSON objects of one JSONL file, in file order; anything else is counted and skipped.

    Memory is bounded: a line is read at most ``MAX_LINE_BYTES + 1`` bytes at a time, and a
    longer one is drained without being kept. An unreadable file reads as empty.
    """
    try:
        handle = open(path, "rb")
    except OSError:
        return
    with handle:
        read = 0
        first = True
        try:
            while True:
                raw = handle.readline(MAX_LINE_BYTES + 1)
                if not raw:
                    return
                read += len(raw)
                oversize = len(raw) > MAX_LINE_BYTES and not raw.endswith(b"\n")
                if oversize:
                    while True:  # drain the rest of the line
                        chunk = handle.readline(MAX_LINE_BYTES)
                        read += len(chunk)
                        if not chunk or chunk.endswith(b"\n"):
                            break
                if first:
                    first = False
                    if raw.startswith(_BOM):
                        raw = raw[len(_BOM):]
                if oversize:
                    stats.lines += 1
                    stats.skipped += 1
                else:
                    obj = _decode(raw, stats)
                    if obj is not None:
                        yield obj
                if read >= MAX_FILE_BYTES:
                    if handle.read(1):
                        stats.truncated = True
                    return
        except OSError:
            return


def _decode(raw: bytes, stats: LogStats) -> Optional[dict]:
    """One line as a JSON object, or ``None`` (counted, unless the line is blank)."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        stats.lines += 1
        stats.skipped += 1
        return None
    if not text.strip():
        return None
    stats.lines += 1
    try:
        obj = json.loads(text)
    except (ValueError, RecursionError):  # JSONDecodeError is a ValueError
        stats.skipped += 1
        return None
    if not isinstance(obj, dict):
        stats.skipped += 1
        return None
    return obj


def _is_int(value: object) -> bool:
    return type(value) is int  # a bool is not a number here


def _clean_id(value: object) -> Optional[str]:
    if isinstance(value, str) and value and len(value) <= _MAX_ID_LEN and "\x00" not in value:
        return value
    return None


# ---------------------------------------------------------------------------
# Feedback records
# ---------------------------------------------------------------------------


@dataclass
class ResponseFeedback:
    """The feedback in force for one answer, after the latest-wins / clear rules.

    ``*_seq`` is the line number (file order) of the line the value came from, so the
    newest of several signals can be told apart even when their ``ts`` are equal or missing.
    """

    thumb: Optional[int] = None
    thumb_ts: Optional[float] = None
    thumb_seq: int = -1
    rating: Optional[int] = None
    rating_ts: Optional[float] = None
    rating_seq: int = -1
    edit: Optional[str] = None
    edit_ts: Optional[float] = None
    edit_seq: int = -1
    comments: "list[str]" = field(default_factory=list)
    #: Valid feedback lines about this answer, whatever their kind.
    lines: int = 0


@dataclass
class FeedbackLog:
    """A feedback log read: ``by_response`` keeps the order its ids first appeared in."""

    by_response: "dict[str, ResponseFeedback]" = field(default_factory=dict)
    stats: LogStats = field(default_factory=LogStats)


def _read_value(obj: dict, kind: str) -> "tuple[bool, object]":
    """``(valid, value)`` of a thumb / rating / edit line. ``None`` is a clear."""
    if kind == "edit":
        if "edited_text" not in obj:
            return False, None
        text = obj["edited_text"]
        if text is None:
            return True, None
        if isinstance(text, str) and text.strip() and "\x00" not in text:
            return True, text
        return False, None
    if "value" not in obj:
        return False, None
    value = obj["value"]
    if value is None:
        return True, None
    if kind == "thumb":
        return (True, value) if _is_int(value) and value in (1, -1) else (False, None)
    return (True, value) if _is_int(value) and 1 <= value <= 5 else (False, None)


def read_feedback(path: Optional[str]) -> FeedbackLog:
    """Read a feedback log by the contract's rules. A missing file is an empty log."""
    log = FeedbackLog()
    if not path or not os.path.isfile(path):
        return log
    for seq, obj in enumerate(_iter_objects(path, log.stats)):
        version = obj.get("schema_version", SCHEMA_VERSION)  # no feedback version 0 exists
        response_id = _clean_id(obj.get("response_id"))
        kind = obj.get("kind")
        if not (_is_int(version) and version == SCHEMA_VERSION):
            log.stats.skipped += 1
            continue
        if response_id is None or kind not in _FEEDBACK_KINDS:
            log.stats.skipped += 1
            continue
        ts = parse_ts(obj.get("ts"))
        if kind == "comment":
            text = obj.get("comment")
            if not (isinstance(text, str) and text.strip()):
                log.stats.skipped += 1
                continue
            record = log.by_response.setdefault(response_id, ResponseFeedback())
            record.comments.append(text[:_MAX_COMMENT_CHARS])
            record.lines += 1
            continue
        valid, value = _read_value(obj, kind)
        if not valid:
            log.stats.skipped += 1
            continue
        record = log.by_response.setdefault(response_id, ResponseFeedback())
        record.lines += 1
        if kind == "thumb":
            record.thumb, record.thumb_ts = value, (ts if value is not None else None)
            record.thumb_seq = seq if value is not None else -1
        elif kind == "rating":
            record.rating, record.rating_ts = value, (ts if value is not None else None)
            record.rating_seq = seq if value is not None else -1
        else:
            record.edit, record.edit_ts = value, (ts if value is not None else None)
            record.edit_seq = seq if value is not None else -1
    return log


# ---------------------------------------------------------------------------
# Conversation records
# ---------------------------------------------------------------------------


@dataclass
class Conversation:
    """One logged answer (a version 1 conversation record)."""

    id: str
    ts: Optional[float]
    messages: "list[dict]"
    response: str
    adapter: Optional[str] = None
    canary_bucket: Optional[str] = None
    conversation_id: Optional[str] = None


@dataclass
class ConversationLog:
    """The conversation records of a trace directory that were asked for."""

    #: By ``id``; only the ids asked for when ``wanted`` was given. The first line wins.
    records: "dict[str, Conversation]" = field(default_factory=dict)
    stats: LogStats = field(default_factory=LogStats)
    #: Valid version 1 records seen, kept or not.
    answers: int = 0
    #: Version 0 lines (no ``schema_version``): they have no id, so feedback cannot name them.
    legacy_lines: int = 0


def _messages(value: object) -> "Optional[list[dict]]":
    if not isinstance(value, list) or not value:
        return None
    cleaned = []
    for message in value:
        if not isinstance(message, dict):
            return None
        role, content = message.get("role"), message.get("content")
        if not (isinstance(role, str) and role and isinstance(content, str)):
            return None
        cleaned.append({"role": role, "content": content})
    return cleaned


def _conversation(obj: dict) -> Optional[Conversation]:
    answer_id = _clean_id(obj.get("id"))
    messages = _messages(obj.get("messages"))
    response = obj.get("response")
    if answer_id is None or messages is None or not isinstance(response, str):
        return None
    adapter = obj.get("adapter")
    bucket = obj.get("canary_bucket")
    return Conversation(
        id=answer_id,
        ts=parse_ts(obj.get("ts")),
        messages=messages,
        response=response,
        adapter=adapter if isinstance(adapter, str) and adapter else None,
        canary_bucket=bucket if bucket in _BUCKETS else None,
        conversation_id=_clean_id(obj.get("conversation_id")),
    )


def conversation_files(directory: Optional[str]) -> "list[str]":
    """The ``*.jsonl`` files of ``directory`` in name order (what ``parse_soup_serve`` reads)."""
    if not directory or not os.path.isdir(directory):
        return []
    try:
        names = sorted(name for name in os.listdir(directory) if name.endswith(".jsonl"))
    except OSError:
        return []
    paths = [os.path.join(directory, name) for name in names]
    return [path for path in paths if os.path.isfile(path)]


def read_conversations(
    directory: Optional[str], wanted: Optional[Collection[str]] = None
) -> ConversationLog:
    """Read the conversation records of a trace directory.

    ``wanted`` limits which records are kept (their ids), so a large log is read without
    holding it; ``None`` keeps all. A missing directory is an empty log.
    """
    log = ConversationLog()
    for path in conversation_files(directory):
        for obj in _iter_objects(path, log.stats):
            version = obj.get("schema_version")
            if "schema_version" not in obj:
                log.legacy_lines += 1
                continue
            if not (_is_int(version) and version == SCHEMA_VERSION):
                log.stats.skipped += 1
                continue
            record = _conversation(obj)
            if record is None:
                log.stats.skipped += 1
                continue
            log.answers += 1
            if wanted is not None and record.id not in wanted:
                continue
            log.records.setdefault(record.id, record)
    return log


# ---------------------------------------------------------------------------
# The three datasets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetRow:
    """One training row and the ts of the newest feedback line it came from."""

    row: dict
    ts: Optional[float]


@dataclass
class FeedbackDatasets:
    dpo: "list[DatasetRow]" = field(default_factory=list)
    kto: "list[DatasetRow]" = field(default_factory=list)
    sft: "list[DatasetRow]" = field(default_factory=list)
    #: Valid feedback lines about an answer that no conversation record has: ignored.
    unknown_response: int = 0


def prompt_key(prompt: object) -> str:
    """A text that two spellings of one prompt share.

    A plain string, and a list holding the one user message with that text, give the same
    key, so a pair from a serve trace and one from a conversation record can be told to be
    the same pair.
    """
    if isinstance(prompt, str):
        return prompt
    if (
        isinstance(prompt, list)
        and len(prompt) == 1
        and isinstance(prompt[0], dict)
        and prompt[0].get("role") == "user"
        and isinstance(prompt[0].get("content"), str)
    ):
        return prompt[0]["content"]
    return json.dumps(prompt, sort_keys=True, ensure_ascii=False)


def _newer(left: Optional[float], right: Optional[float]) -> Optional[float]:
    if left is None:
        return right
    return left if right is None or left >= right else right


def _unique(rows: "list[tuple[tuple, DatasetRow]]") -> "list[DatasetRow]":
    """Drop repeated rows (same key), keeping the first and the newest ts of the copies."""
    kept: "dict[tuple, DatasetRow]" = {}
    for key, row in rows:
        old = kept.get(key)
        if old is None:
            kept[key] = row
        elif row.ts is not None and (old.ts is None or row.ts > old.ts):
            kept[key] = DatasetRow(old.row, row.ts)
    return list(kept.values())


def _decision(record: ResponseFeedback) -> "Optional[tuple[int, bool, Optional[float]]]":
    """The newest thumb or rating on the answer as given: ``(seq, label, ts)`` or ``None``.

    A thumb up and a rating of 4-5 are ``True``; a thumb down and a rating of 1-2 are
    ``False``; a rating of 3 is not a signal. The caller adds an edit (``False``).
    """
    signals = []
    if record.thumb is not None:
        signals.append((record.thumb_seq, record.thumb == 1, record.thumb_ts))
    if record.rating is not None and record.rating != 3:
        signals.append((record.rating_seq, record.rating >= 4, record.rating_ts))
    return max(signals, key=lambda signal: signal[0]) if signals else None


def build_datasets(
    records: Mapping[str, Conversation], feedback: FeedbackLog
) -> FeedbackDatasets:
    """The dpo / kto / sft rows the feedback gives, in the order its ids first appeared."""
    dpo, kto, sft = [], [], []
    unknown = 0
    for answer_id, record in feedback.by_response.items():
        conversation = records.get(answer_id)
        if conversation is None:
            unknown += record.lines
            continue
        prompt = conversation.messages
        original = conversation.response
        edited = record.edit
        if edited is not None and edited.strip() == original.strip():
            edited = None  # nothing was corrected
        deciding = _decision(record)
        if edited is not None and (deciding is None or record.edit_seq > deciding[0]):
            deciding = (record.edit_seq, False, record.edit_ts)

        if edited is not None:
            ts = record.edit_ts
            if original.strip():
                key = (prompt_key(prompt), edited, original)
                dpo.append((key, DatasetRow(_dpo_row(prompt, edited, original), ts)))
            key = (prompt_key(prompt), edited)
            sft.append((key, DatasetRow(_sft_row(prompt, edited), ts)))
            kto.append(
                ((prompt_key(prompt), edited, True), DatasetRow(_kto_row(prompt, edited, True), ts))
            )
        if deciding is not None and original.strip():
            _, label, ts = deciding
            kto.append(
                (
                    (prompt_key(prompt), original, label),
                    DatasetRow(_kto_row(prompt, original, label), ts),
                )
            )
    return FeedbackDatasets(
        dpo=_unique(dpo), kto=_unique(kto), sft=_unique(sft), unknown_response=unknown
    )


def _copy(messages: "list[dict]") -> "list[dict]":
    return [dict(message) for message in messages]


def _dpo_row(prompt: "list[dict]", chosen: str, rejected: str) -> dict:
    return {
        "prompt": _copy(prompt),
        "chosen": chosen,
        "rejected": rejected,
        "source": "feedback-edit",
    }


def _kto_row(prompt: "list[dict]", completion: str, label: bool) -> dict:
    return {
        "prompt": _copy(prompt),
        "completion": completion,
        "label": label,
        "source": "feedback",
    }


def _sft_row(prompt: "list[dict]", answer: str) -> dict:
    return {
        "messages": [*_copy(prompt), {"role": "assistant", "content": answer}],
        "source": "feedback-edit",
    }


# ---------------------------------------------------------------------------
# One call for the harvest
# ---------------------------------------------------------------------------


@dataclass
class FeedbackHarvest:
    datasets: FeedbackDatasets
    #: Version 1 conversation records in the trace directory (the answers logged).
    answers: int = 0
    #: Version 0 lines in the trace directory (read by the serve-trace parser instead).
    legacy_lines: int = 0
    feedback_lines: int = 0
    #: Lines of both logs that were not used (malformed, unknown ``schema_version``, ...).
    skipped_lines: int = 0
    truncated: bool = False


def harvest_feedback(
    trace_dir: Optional[str], feedback_path: Optional[str]
) -> FeedbackHarvest:
    """Read the feedback log, join it to the conversation records, build the datasets."""
    feedback = read_feedback(feedback_path)
    conversations = read_conversations(trace_dir, wanted=set(feedback.by_response))
    datasets = build_datasets(conversations.records, feedback)
    return FeedbackHarvest(
        datasets=datasets,
        answers=conversations.answers,
        legacy_lines=conversations.legacy_lines,
        feedback_lines=feedback.stats.lines,
        skipped_lines=feedback.stats.skipped + conversations.stats.skipped,
        truncated=feedback.stats.truncated or conversations.stats.truncated,
    )


def is_new(ts: Optional[float], cursor: Optional[float]) -> bool:
    """Whether a row with feedback time ``ts`` is news after the data ``cursor``.

    Everything is news until a training run has set the cursor; after that a row is new when
    its feedback is later than the cursor. A row without a ``ts`` cannot be placed in time,
    so it counts as new only while there is no cursor (else it would be new forever).
    """
    return cursor is None or (ts is not None and ts > cursor)


def newest_ts(rows: "Collection[DatasetRow]") -> Optional[float]:
    """The largest ``ts`` among ``rows``, or ``None`` when none has one."""
    newest: Optional[float] = None
    for row in rows:
        newest = _newer(newest, row.ts)
    return newest
