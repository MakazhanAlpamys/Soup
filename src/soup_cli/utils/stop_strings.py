"""OpenAI ``stop`` / Anthropic ``stop_sequences`` for ``soup serve`` (#2549).

Pure Python, no torch: the parser that refuses a bad request, the matcher that
cuts a finished reply, and the filter that does the same on a reply that arrives
in pieces. ``soup serve`` runs both forms on the same text and they agree, so the
joined streamed deltas equal the non-streamed reply for the same request.

What counts as the reply: the text the client would otherwise receive, i.e. the
decoded reply with ``str.strip()`` applied. A stop string is matched in that text
and the reply is cut before it, so the stop string and everything after it are
never returned. Whitespace left dangling before the cut is dropped, as ``strip()``
drops it from the end of an uncut reply.

When several stop strings occur, the one that *completes first* ends the reply,
because generation stops the moment any stop string is complete. That is the
earliest end, not the earliest start: with ``("abcd", "bc")`` the reply
``"xabcd"`` ends at ``"bc"``. Two that complete together are cut at the one that
starts first, so neither survives.
"""

from __future__ import annotations

from typing import Optional, Sequence

#: OpenAI accepts at most four stop sequences.
MAX_STOP_STRINGS = 4
#: One stop string may be at most this long. The streaming filter holds back one
#: character fewer than the longest, so this also bounds how late text is sent.
MAX_STOP_LENGTH = 256


class StopStringError(ValueError):
    """A ``stop`` / ``stop_sequences`` value that is refused. The message names the field."""


def parse_stop(
    value: object, *, field: str = "stop", allow_string: bool = True
) -> tuple[str, ...]:
    """The stop strings a request asks for, or raise ``StopStringError`` naming ``field``.

    Accepted: ``None`` (absent or ``null``), a non-empty string (OpenAI ``stop`` only:
    Anthropic's ``stop_sequences`` is a list, so its caller passes ``allow_string=False``),
    a list of at most ``MAX_STOP_STRINGS`` non-empty strings, and an empty list (no stop).
    Anything else is refused: an empty string, a non-string entry, too many entries, an
    entry longer than ``MAX_STOP_LENGTH``, or a value that is neither a string nor a list.
    The message never repeats the value, only the field and the rule it broke.
    """
    if value is None:
        return ()
    if isinstance(value, str) and allow_string:
        entries: Sequence[object] = [value]
        single = True
    elif isinstance(value, (list, tuple)):
        # The count first: a list of a million entries is refused without walking it.
        if len(value) > MAX_STOP_STRINGS:
            raise StopStringError(
                f"{field} accepts at most {MAX_STOP_STRINGS} stop strings, got {len(value)}"
            )
        entries = list(value)
        single = False
    else:
        expected = "a string or a list of strings" if allow_string else "a list of strings"
        raise StopStringError(f"{field} must be {expected}")

    parsed: list[str] = []
    for index, entry in enumerate(entries):
        label = field if single else f"{field}[{index}]"
        if not isinstance(entry, str):
            raise StopStringError(f"{label} must be a string")
        if not entry:
            raise StopStringError(f"{label} must not be an empty string")
        if len(entry) > MAX_STOP_LENGTH:
            raise StopStringError(f"{label} is longer than {MAX_STOP_LENGTH} characters")
        if entry not in parsed:
            parsed.append(entry)
    return tuple(parsed)


def find_stop(text: str, stops: Sequence[str]) -> Optional[tuple[int, str]]:
    """``(start, stop)`` of the stop string that completes first in ``text``, or ``None``.

    Ties on the end go to the string that starts first (the longer one).
    """
    best: Optional[tuple[int, int, str]] = None
    for stop in stops:
        start = text.find(stop)
        if start < 0:
            continue
        key = (start + len(stop), start, stop)
        if best is None or key[:2] < best[:2]:
            best = key
    if best is None:
        return None
    return best[1], best[2]


def cut_at_stop(text: str, stops: Sequence[str]) -> tuple[str, Optional[str]]:
    """``(reply, matched)``: ``text`` before the first stop string, or ``text`` unchanged.

    ``matched`` is the stop string that ended the reply, ``None`` if none occurs. A cut
    reply has the whitespace before the stop string dropped.
    """
    hit = find_stop(text, stops)
    if hit is None:
        return text, None
    start, stop = hit
    return text[:start].rstrip(), stop


class StopFilter:
    """``cut_at_stop`` for a reply that arrives in pieces.

    ``feed(piece)`` returns the text that is safe to send now. Two things are held
    back so that text after a stop string never leaves the server: the last
    ``len(longest stop) - 1`` characters, which may be the start of a stop string that
    is not complete yet, and any whitespace at the end of what could be sent, which a
    cut would drop. Nothing is flushed when the pieces end: the held text is the part of
    the finished reply that follows what was sent, and the caller sends it from the cut
    reply (``cut_at_stop`` of the whole text), which is the same text this filter would
    have produced.

    Once a stop string has been seen, ``matched`` is set and every later ``feed`` returns
    an empty string, so the caller can stop generation and discard what is left.
    """

    def __init__(self, stops: Sequence[str]) -> None:
        self._stops = tuple(stops)
        self._hold = max((len(stop) for stop in self._stops), default=1) - 1
        self._pending = ""
        self.matched: Optional[str] = None

    def feed(self, piece: str) -> str:
        if not self._stops:
            return piece
        if self.matched is not None:
            return ""
        self._pending += piece
        hit = find_stop(self._pending, self._stops)
        if hit is not None:
            start, self.matched = hit
            out = self._pending[:start].rstrip()
            self._pending = ""
            return out
        safe = len(self._pending) - self._hold
        if safe <= 0:
            return ""
        out = self._pending[:safe].rstrip()
        self._pending = self._pending[len(out):]
        return out
