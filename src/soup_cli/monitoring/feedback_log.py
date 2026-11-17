"""Append-only feedback log for ``soup serve --feedback-log`` (``POST /v1/feedback``).

One JSON object per feedback action, in the version 1 format of
``soup_cli/schemas/loop/feedback.schema.json`` (described in ``docs/loop-formats.md``)::

    {"schema_version": 1, "id": <feedback id>, "response_id": <answer id>, "ts": ...,
     "kind": "thumb" | "rating" | "edit" | "comment", "value": ..., "edited_text": ...,
     "comment": ..., "user_ref": ..., "source": "soup-serve"}

:func:`validate_feedback` applies the schema's rules to a request body and names the
first field that breaks one; :func:`build_record` adds what the server owns (ids, time,
source) and redacts secrets in the free text; :class:`FeedbackLogWriter` appends the
line. The log is never rotated: a feedback line is the durable record of what a person
said about an answer, and renaming the file away would hide it from the readers.
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

from soup_cli.monitoring.trace_logger import (
    SERVE_SOURCE,
    open_private_append,
    redact_value,
    resolve_log_path,
    utc_timestamp,
)

#: ``schema_version`` of the feedback record.
FEEDBACK_SCHEMA_VERSION = 1

#: The feedback kinds ``feedback.schema.json`` allows.
FEEDBACK_KINDS: tuple[str, ...] = ("thumb", "rating", "edit", "comment")

#: Largest request body ``POST /v1/feedback`` reads, in bytes.
MAX_BODY_BYTES = 64 * 1024

MAX_COMMENT_CHARS = 10_000
MAX_USER_REF_CHARS = 256
#: Longest ``edited_text`` (characters). Far below the body cap: an edit is one answer.
MAX_EDITED_TEXT_CHARS = 20_000

#: The feedback log stops accepting posts once the file is this big (bytes). It is never
#: rotated (a line is the durable record of what a person said), so the cap is the bound.
MAX_LOG_BYTES = 256 * 1024 * 1024

#: How many response ids the server remembers having issued (oldest forgotten first).
MAX_ISSUED_RESPONSE_IDS = 100_000

_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


class FeedbackValidationError(ValueError):
    """A feedback request body broke a schema rule; ``field`` names the field."""

    def __init__(self, field: str) -> None:
        super().__init__(field)
        self.field = field


def _whole_number(value: Any) -> Any:
    """``value`` as an int when it is a whole number (``3`` or ``3.0``), else ``None``.

    JSON Schema counts ``3.0`` as the integer 3; a bool is not a number.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _optional_text(payload: dict[str, Any], field: str, *, max_chars: int | None) -> str | None:
    """A text field that may be absent or ``null``; when set it is 1..``max_chars`` long."""
    value = payload.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise FeedbackValidationError(field)
    if max_chars is not None and len(value) > max_chars:
        raise FeedbackValidationError(field)
    return value


def validate_feedback(payload: Any) -> dict[str, Any]:
    """Check a request body against ``feedback.schema.json`` and return its fields.

    The result holds ``response_id`` and ``kind``, plus ``value`` (thumb, rating),
    ``edited_text`` (edit), ``comment`` and ``user_ref`` when they apply. A key the
    schema does not use for the kind is dropped (``value`` of an edit or a comment is
    "ignored" in the schema), as is any key the schema does not name.

    Raises :class:`FeedbackValidationError` naming the first broken field, checked in
    the order ``response_id``, ``kind``, the kind's own field, ``comment``, ``user_ref``.
    """
    if not isinstance(payload, dict):
        raise FeedbackValidationError("body")
    response_id = payload.get("response_id")
    if not isinstance(response_id, str) or _UUID_RE.fullmatch(response_id) is None:
        raise FeedbackValidationError("response_id")
    kind = payload.get("kind")
    if kind not in FEEDBACK_KINDS:  # also refuses a non-string: only the four names equal
        raise FeedbackValidationError("kind")

    fields: dict[str, Any] = {"response_id": response_id, "kind": kind}
    if kind == "thumb":
        if "value" not in payload:
            raise FeedbackValidationError("value")
        value = payload["value"]
        number = _whole_number(value)
        if value is not None and number not in (1, -1):
            raise FeedbackValidationError("value")
        fields["value"] = number
    elif kind == "rating":
        if "value" not in payload:
            raise FeedbackValidationError("value")
        value = payload["value"]
        number = _whole_number(value)
        if value is not None and (number is None or not 1 <= number <= 5):
            raise FeedbackValidationError("value")
        fields["value"] = number
    elif kind == "edit":
        if "edited_text" not in payload:
            raise FeedbackValidationError("edited_text")
        fields["edited_text"] = _optional_text(
            payload, "edited_text", max_chars=MAX_EDITED_TEXT_CHARS
        )

    comment = _optional_text(payload, "comment", max_chars=MAX_COMMENT_CHARS)
    if kind == "comment" and comment is None:
        raise FeedbackValidationError("comment")
    if comment is not None:
        fields["comment"] = comment
    user_ref = _optional_text(payload, "user_ref", max_chars=MAX_USER_REF_CHARS)
    if user_ref is not None:
        fields["user_ref"] = user_ref
    return fields


def build_record(fields: dict[str, Any]) -> dict[str, Any]:
    """The line to append for validated ``fields``: server-owned keys added, text redacted."""
    record: dict[str, Any] = {
        "schema_version": FEEDBACK_SCHEMA_VERSION,
        "id": str(uuid.uuid4()),
        "response_id": fields["response_id"],
        "ts": utc_timestamp(),
        "kind": fields["kind"],
    }
    for key in ("value", "edited_text", "comment", "user_ref"):
        if key in fields:
            record[key] = fields[key]
    # Free text is redacted exactly as the trace log redacts answers.
    for key in ("edited_text", "comment", "user_ref"):
        if key in record:
            record[key] = redact_value(record[key])
    record["source"] = SERVE_SOURCE
    return record


class FeedbackLogFullError(Exception):
    """The feedback log reached its size cap; ``path`` is the file, nothing was appended."""

    def __init__(self, path: Path, max_bytes: int) -> None:
        super().__init__(f"feedback log is full: {path} has reached {max_bytes} bytes")
        self.path = path
        self.max_bytes = max_bytes


class IssuedResponseIds:
    """The response ids this server handed out, bounded: the oldest are forgotten first.

    ``POST /v1/feedback`` accepts only an id in here, so a client cannot attach feedback to
    an answer this process never gave. Held in memory (the trace log is not read per
    request); after a restart the new process does not know the old ids.
    """

    def __init__(self, capacity: int = MAX_ISSUED_RESPONSE_IDS) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError(f"capacity must be a positive int, got {capacity!r}")
        self._capacity = capacity
        self._ids: "OrderedDict[str, None]" = OrderedDict()
        self._lock = threading.Lock()

    def add(self, response_id: str) -> None:
        with self._lock:
            self._ids[response_id] = None
            self._ids.move_to_end(response_id)
            while len(self._ids) > self._capacity:
                self._ids.popitem(last=False)

    def __contains__(self, response_id: object) -> bool:
        with self._lock:
            return response_id in self._ids

    def __len__(self) -> int:
        with self._lock:
            return len(self._ids)


class FeedbackLogWriter:
    """Thread-safe append-only JSONL writer for feedback records. No rotation.

    The file is created ``0o600`` and refuses new lines once it is ``max_bytes`` or more.
    """

    def __init__(self, path: str, *, max_bytes: int = MAX_LOG_BYTES) -> None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
            raise ValueError(f"max_bytes must be a positive int, got {max_bytes!r}")
        self._path: Path = resolve_log_path(path, flag="--feedback-log")
        self._max_bytes = max_bytes
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def append(self, record: dict[str, Any]) -> None:
        """Append ``record`` as one line. Raises ``OSError`` when it cannot be written.

        Unlike the passive trace log, a failure is reported: the caller tells the
        client its feedback was not stored instead of answering ``ok``. Raises
        :class:`FeedbackLogFullError` (nothing written) when the file is at the size cap.
        """
        try:
            line = json.dumps(record, ensure_ascii=False, allow_nan=False)
            data = (line + "\n").encode("utf-8")
        except UnicodeEncodeError:
            # A lone surrogate cannot be written as UTF-8: \u-escape it instead.
            data = (json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n").encode(
                "ascii"
            )
        with self._lock, open_private_append(self._path) as handle:
            if os.fstat(handle.fileno()).st_size >= self._max_bytes:
                raise FeedbackLogFullError(self._path, self._max_bytes)
            handle.write(data)
