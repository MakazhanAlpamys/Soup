"""Trace parsers for LangChain / OpenAI-style / Soup-serve JSONL logs."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from soup_cli.data.traces.messages import content_text, conversation_prompt

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS: tuple[str, ...] = ("langchain", "openai", "soup-serve")

# The per-trace signal vocabulary a producer may write. Distinct from
# ``pair_builder.SUPPORTED_SIGNALS``, which names pair *modes*: a trace can carry
# ``thumbs_down`` (a signal) even though no ``--signal thumbs_down`` exists (#1440).
TRACE_SIGNALS: frozenset[str] = frozenset({
    "thumbs_up",
    "thumbs_down",
    "regenerated",
    "user_edit",
})


@dataclass(frozen=True)
class Trace:
    """A single request/response pair extracted from a log."""

    trace_id: str
    prompt: str
    output: str
    signal: str = "none"
    regen_order: Optional[int] = None
    edited_output: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # When the exchange was logged, as epoch seconds; None when the log gives none
    # (see ``parse_ts``). ``soup loop`` compares it with its data cursor: a trace is
    # new when its ts is later; one without a ts counts as new only while the cursor
    # is unset, else it would be new forever (``loop_stages._trace_pairs``).
    ts: Optional[float] = None


# Epoch seconds up to the year 9999 (datetime.max). Larger numbers are not seconds (a
# millisecond clock gives ~1.7e12) and would pin a data cursor in the far future.
_MAX_TS = 253402300799.0
_UTC_SUFFIXES = ("Z", "+00:00")


def parse_ts(value: Any) -> Optional[float]:
    """A log line's ``ts`` as epoch seconds, or ``None`` when it has none.

    A number is epoch seconds (what ``soup serve --trace-log`` writes). An
    ISO-8601 string ending ``Z`` or ``+00:00`` is converted. Anything else is
    ``None``: a bool, NaN or infinity, a negative or beyond-year-9999 number, an
    ISO string in another zone or with none, other text, or a missing field.
    """
    if isinstance(value, bool):
        return None
    try:
        if isinstance(value, (int, float)):
            seconds = float(value)
        elif isinstance(value, str) and value.endswith(_UTC_SUFFIXES):
            seconds = datetime.fromisoformat(value).timestamp()  # aware: a UTC suffix
        else:
            return None
    except (ValueError, OverflowError):  # unparsable text, or an int too large for a float
        return None
    if not 0.0 <= seconds <= _MAX_TS:  # False for NaN too
        return None
    return seconds


#: A log line dated more than this many seconds ahead of now is not used: one such line
#: would become the newest ``ts`` and freeze ``soup loop``'s data cursor for good (every
#: later row would be "not new"). A day tolerates a client whose clock runs ahead.
MAX_FUTURE_TS_SEC = 24 * 60 * 60


def ts_is_in_the_future(ts: Optional[float], now: Optional[float] = None) -> bool:
    """Whether ``ts`` (epoch seconds) lies more than ``MAX_FUTURE_TS_SEC`` after ``now``.

    ``now`` defaults to the current time. ``None`` (no usable ts) is not in the future.
    """
    if ts is None:
        return False
    return ts > (time.time() if now is None else now) + MAX_FUTURE_TS_SEC


# ---------------------------------------------------------------------------
# LangChain
# ---------------------------------------------------------------------------


def _langchain_prompt(inputs: Any) -> str:
    """Flatten LangChain ``inputs`` into a string prompt."""
    if isinstance(inputs, list):
        return conversation_prompt(inputs)
    if not isinstance(inputs, dict):
        return ""
    messages = inputs.get("messages")
    if isinstance(messages, list):
        return conversation_prompt(messages)
    for key in ("prompt", "input", "query"):
        value = inputs.get(key)
        if isinstance(value, str):
            return value
    return json.dumps(inputs, default=str)


def _langchain_output(outputs: Any) -> str:
    """Extract the assistant response from LangChain ``outputs``."""
    if not isinstance(outputs, dict):
        return ""
    gens = outputs.get("generations")
    if isinstance(gens, list) and gens:
        first = gens[0]
        if isinstance(first, list) and first:
            item = first[0]
        else:
            item = first
        if isinstance(item, dict):
            text = item.get("text") or item.get("content") or ""
            return content_text(text)
    for key in ("output", "response", "text", "content"):
        value = outputs.get(key)
        if isinstance(value, (str, list)):
            return content_text(value)
    return ""


def _langchain_signal(feedback: Any) -> str:
    """Map LangChain feedback to our signal vocabulary."""
    if not isinstance(feedback, list):
        return "none"
    for entry in feedback:
        if not isinstance(entry, dict):
            continue
        if entry.get("key") == "thumbs":
            score = entry.get("score")
            if score == 1 or score == "up":
                return "thumbs_up"
            if score == 0 or score == "down":
                return "thumbs_down"
    return "none"


def parse_langchain(events: Iterable[Any]) -> Iterator[Trace]:
    for idx, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        outputs = event.get("outputs")
        if not outputs:
            continue
        prompt = _langchain_prompt(event.get("inputs"))
        output = _langchain_output(outputs)
        if not prompt or not output:
            continue
        trace_id = str(event.get("id") or f"lc-{idx}")
        signal = _langchain_signal(event.get("feedback"))
        yield Trace(
            trace_id=trace_id, prompt=prompt, output=output, signal=signal,
        )


# ---------------------------------------------------------------------------
# OpenAI-style
# ---------------------------------------------------------------------------


def _openai_prompt(messages: Any) -> str:
    return conversation_prompt(messages)


def _openai_output(choices: Any) -> str:
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    msg = first.get("message")
    if isinstance(msg, dict):
        return content_text(msg.get("content"))
    return content_text(first.get("text"))


def _openai_signal(feedback: Any) -> str:
    if not isinstance(feedback, dict):
        return "none"
    rating = feedback.get("rating")
    if rating in ("up", 1, "thumbs_up"):
        return "thumbs_up"
    if rating in ("down", 0, "thumbs_down"):
        return "thumbs_down"
    return "none"


def parse_openai(events: Iterable[Any]) -> Iterator[Trace]:
    # Track regenerations for the same prompt via "regenerated_from"
    regen_orders: dict[str, int] = {}
    for idx, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        prompt = _openai_prompt(event.get("messages"))
        output = _openai_output(event.get("choices"))
        if not prompt or not output:
            continue
        trace_id = str(event.get("id") or f"oai-{idx}")
        signal = _openai_signal(event.get("feedback"))
        regen_from = event.get("regenerated_from")
        if regen_from:
            order = regen_orders.get(prompt, 0) + 1
            regen_orders[prompt] = order
            signal = "regenerated" if signal == "none" else signal
            yield Trace(
                trace_id=trace_id, prompt=prompt, output=output,
                signal=signal, regen_order=order,
            )
        else:
            regen_orders.setdefault(prompt, 0)
            yield Trace(
                trace_id=trace_id, prompt=prompt, output=output,
                signal=signal, regen_order=regen_orders[prompt],
            )


# ---------------------------------------------------------------------------
# Soup-serve (directory of JSONL files)
# ---------------------------------------------------------------------------


def _record_signal(event: dict[str, Any]) -> str:
    """The signal a ``soup ingest`` record carries.

    ``soup ingest`` writes a top-level ``signal`` field in the canonical
    vocabulary (``TraceRecord.to_dict``); ``soup serve --trace-log`` records
    none, and a hand-written log may still carry the older
    ``feedback.rating``. #1440: reading only ``feedback`` meant Soup's own
    ingest stream was read as ``none`` for every row, so
    ``soup data from-traces`` wrote 0 pairs from it and exited 0.
    """
    signal = event.get("signal")
    if isinstance(signal, str) and signal in TRACE_SIGNALS:
        return signal
    return _openai_signal(event.get("feedback"))


def _edited_output(event: dict[str, Any]) -> Optional[str]:
    """The user's edited answer, under the names a log would use."""
    feedback = event.get("feedback")
    candidates = [
        event.get("edited_output"),
        event.get("edited_response"),
        feedback.get("edited_output") if isinstance(feedback, dict) else None,
    ]
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value
    return None


#: The conversation-record ``schema_version`` this reader understands. A line without
#: one is version 0 (``soup serve --trace-log`` before the loop formats).
_KNOWN_SCHEMA_VERSION = 1


def _serve_prompt(event: dict[str, Any]) -> str:
    """The prompt of a ``soup serve`` line.

    Version 0 and version 1 lines both carry a top-level ``prompt`` (the last user
    message). A version 1 line from another writer may carry only ``messages``; its
    prompt is then the last user message there.
    """
    prompt = event.get("prompt")
    if isinstance(prompt, str) and prompt:
        return prompt
    messages = event.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "user":
                return content_text(message.get("content"))
    return ""


def _known_schema_version(event: dict[str, Any]) -> bool:
    """Whether this reader understands the line's ``schema_version`` (absent = version 0)."""
    if "schema_version" not in event:
        return True
    version = event["schema_version"]
    return type(version) is int and version == _KNOWN_SCHEMA_VERSION


def parse_soup_serve(path: str) -> Iterator[Trace]:
    """Traces from a directory of ``soup serve --trace-log`` JSONL files (``Trace.ts`` per line)."""
    dir_path = Path(path)
    if not dir_path.is_dir():
        return
    future = 0
    now = time.time()
    for file_path in sorted(dir_path.glob("*.jsonl")):
        # utf-8-sig strips a leading BOM (common on Windows-written JSONL);
        # plain utf-8 left the BOM on line 1, so json.loads failed and the
        # first record of every BOM'd file was silently dropped.
        for line in file_path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            # A line written by a newer schema is not guessed at (docs/loop-formats.md).
            if not _known_schema_version(event):
                continue
            prompt = _serve_prompt(event)
            output = str(event.get("response") or event.get("output") or "")
            if not prompt or not output:
                continue
            ts = parse_ts(event.get("ts"))
            if ts_is_in_the_future(ts, now):
                future += 1
                continue
            signal = _record_signal(event)
            yield Trace(
                trace_id=str(event.get("id") or file_path.stem),
                prompt=prompt, output=output, signal=signal,
                edited_output=_edited_output(event),
                ts=ts,
            )
    if future:
        logger.warning(
            "%d trace line(s) skipped: ts more than 24 h in the future (a later line would "
            "never count as new)", future,
        )
