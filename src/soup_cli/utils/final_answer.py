"""One parser for the final answer a text states (#1226).

GRPO's ``accuracy`` and ``verifiable/math`` rewards compare the final answer of a completion with
the final answer of a gold reference. Both sides are read here, so a spelling understood on one
side is understood on the other. Before #1226 only the completion was parsed: an Alpaca or
ShareGPT reference such as ``"6*7=42\\n#### 42"`` was compared raw, and every correct completion
scored 0.0.

An answer is stated explicitly in one of three forms, tried in this order:

1. ``####`` (GSM8K): the rest of the line after the last marker;
2. ``\\boxed{...}``: the last box that closes; whitespace is allowed before the brace and nested
   braces are balanced, so ``\\boxed {\\frac{1}{2}}`` reads ``\\frac{1}{2}``;
3. ``answer is`` / ``answer:`` in any case: the last such phrase, up to the end of its clause
   (a line end, ``". "``, ``", "``, ``"; "`` or ``" ("``).

The two delimited forms are authoritative: what they enclose IS the answer, number or not. A
phrase has no closing delimiter, so its extent is a guess; when the guess is not a number, a
completion's number is read from its free text instead.

A text that states no explicit answer is free text. A reference is then its whole value, which
must fit on one line (a bare answer such as ``"42"`` or ``"Paris"``). A completion reads as its
last non-empty line, and its number is the last number anywhere in it, which is how
``math_verify`` has always read an unmarked completion.

Both sides are normalised the same way: surrounding whitespace, every ``$`` and ``\\$``, LaTeX
thousands separators (``1{,}000``, ``1,\\!000``, ``1\\,000``), markdown ``*`` at either end,
trailing ``. , ; : !``, and the Unicode minus sign. A number is then exactly one literal: an
optional sign, digits with optional ``,`` groups of three, an optional fraction and an optional
exponent (``-1,000.5``, ``.5``, ``1e5``). ``42 apples``, ``1/2`` and ``x^2`` are not numbers;
they compare as text.

Every scan is linear in its input, because a completion is untrusted model output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

__all__ = [
    "FinalAnswer",
    "answers_match",
    "extract_final_answer",
    "normalize_answer",
    "parse_completion",
    "parse_number",
    "parse_reference",
]

_MARKER = "####"
# LaTeX allows whitespace between a command and its argument: ``\boxed {42}``.
_BOXED_OPEN_RE = re.compile(r"\\boxed\s*\{")
_BRACE_RE = re.compile(r"[{}]")
# ``\bis\b`` keeps "the answer isn't 41" from reading as an answer.
_PHRASE_RE = re.compile(r"\banswer\b\s*(?:is\b\s*:?|:)", re.IGNORECASE)
# ``[.,;]\s`` rather than ``[.,;]``, so ``3.5`` and ``1,000`` stay whole.
_CLAUSE_END_RE = re.compile(r"[.,;]\s|\s\(")
_LATEX_SPACING_RE = re.compile(r"\\[,!]")
_NUMBER_RE = re.compile(r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?")

_EDGE_CHARS = " \t\r\n*"
_TRAILING_CHARS = _EDGE_CHARS + ".,;:!"


@dataclass(frozen=True)
class FinalAnswer:
    """A final answer: its normalised ``text`` and, when it has one, its ``number``."""

    text: str
    number: Decimal | None = None


def _canonical_symbols(text: str) -> str:
    """Rewrite spellings that change no value: the Unicode minus, LaTeX thousands separators."""
    text = text.replace("\u2212", "-").replace("{,}", ",")
    return _LATEX_SPACING_RE.sub("", text)


def normalize_answer(text: str) -> str:
    """Normalise an answer for comparison; see the module docstring for the rules."""
    text = _canonical_symbols(text).replace("\\$", "").replace("$", "")
    text = text.lstrip(_EDGE_CHARS).rstrip(_TRAILING_CHARS)
    return " ".join(text.split())


def _to_decimal(literal: str) -> Decimal | None:
    try:
        return Decimal(literal.replace(",", ""))
    except InvalidOperation:  # pragma: no cover - _NUMBER_RE admits only valid literals
        return None


def parse_number(text: str) -> Decimal | None:
    """Return the value of ``text`` when, normalised, it is exactly one number; else ``None``."""
    candidate = normalize_answer(text)
    if _NUMBER_RE.fullmatch(candidate) is None:
        return None
    return _to_decimal(candidate)


def _marker_answer(text: str) -> str | None:
    index = text.rfind(_MARKER)
    if index < 0:
        return None
    start = index + len(_MARKER)
    end = text.find("\n", start)
    return normalize_answer(text[start:] if end < 0 else text[start:end]) or None


def _closing_brace(text: str, start: int, limit: int) -> int | None:
    depth = 1
    for match in _BRACE_RE.finditer(text, start, limit):
        depth += 1 if match.group() == "{" else -1
        if depth == 0:
            return match.start()
    return None


def _boxed_answer(text: str) -> str | None:
    # Walk the boxes from the last one back. When a box never closes, an earlier box can only
    # close BEFORE that box's brace (it would have to close the later box first), so each scan
    # stops at the next box's brace and the whole walk is linear.
    starts = [match.end() for match in _BOXED_OPEN_RE.finditer(text)]
    limit = len(text)
    for start in reversed(starts):
        close = _closing_brace(text, start, limit)
        if close is not None:
            return normalize_answer(text[start:close]) or None
        limit = start - 1
    return None


def _phrase_answer(text: str) -> str | None:
    last = None
    for last in _PHRASE_RE.finditer(text):
        pass
    if last is None:
        return None
    start = last.end()
    end = text.find("\n", start)
    clause = (text[start:] if end < 0 else text[start:end]).lstrip()
    cut = _CLAUSE_END_RE.search(clause)
    if cut is not None:
        clause = clause[: cut.start()]
    return normalize_answer(clause) or None


def _explicit_answer(text: str) -> tuple[str, bool] | None:
    """Return ``(answer, delimited)`` for the answer ``text`` states explicitly, if any."""
    for reader in (_marker_answer, _boxed_answer):
        answer = reader(text)
        if answer is not None:
            return answer, True
    answer = _phrase_answer(text)
    return None if answer is None else (answer, False)


def _names_an_answer_form(text: str) -> bool:
    return (
        _MARKER in text
        or _BOXED_OPEN_RE.search(text) is not None
        or _PHRASE_RE.search(text) is not None
    )


def _last_number(text: str) -> Decimal | None:
    last = None
    for last in _NUMBER_RE.finditer(_canonical_symbols(text)):
        pass
    return None if last is None else _to_decimal(last.group())


def extract_final_answer(text: str) -> str | None:
    """Return the normalised answer ``text`` states explicitly, or ``None`` when it states none."""
    found = _explicit_answer(text)
    return None if found is None else found[0]


def parse_reference(text: str) -> FinalAnswer | None:
    """Read a gold reference; ``None`` when it states no answer a reward could compare against.

    A reference is never read as free text: without an explicit answer it must be a bare,
    one-line answer. A reference that names an answer form but leaves it empty
    (``"...\\n####"``, ``"\\boxed{}"``) is ``None`` too.
    """
    found = _explicit_answer(text)
    if found is not None:
        return FinalAnswer(found[0], parse_number(found[0]))
    bare = text.strip()
    if not bare or "\n" in bare or _names_an_answer_form(bare):
        return None
    answer = normalize_answer(bare)
    return FinalAnswer(answer, parse_number(answer)) if answer else None


def parse_completion(text: str) -> FinalAnswer | None:
    """Read a model completion; ``None`` when it is empty."""
    found = _explicit_answer(text)
    if found is not None:
        answer, delimited = found
        number = parse_number(answer)
        if number is None and not delimited:
            number = _last_number(text)
        return FinalAnswer(answer, number)
    stripped = text.rstrip()
    if not stripped:
        return None
    last_line = stripped[stripped.rfind("\n") + 1 :]
    return FinalAnswer(normalize_answer(last_line), _last_number(text))


def answers_match(completion: FinalAnswer | None, reference: FinalAnswer) -> bool:
    """Exact match: by value when the reference is a number, else as case-insensitive text."""
    if completion is None:
        return False
    if reference.number is not None:
        return completion.number is not None and completion.number == reference.number
    return bool(reference.text) and completion.text.casefold() == reference.text.casefold()
