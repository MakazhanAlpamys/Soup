"""One parser for the final answer a text states (#1226).

GRPO's ``accuracy`` and ``verifiable/math`` rewards compare the final answer of a completion with
the final answer of a gold reference. Both sides are read here, so a spelling understood on one
side is understood on the other. Before #1226 only the completion was parsed: an Alpaca or
ShareGPT reference such as ``"6*7=42\\n#### 42"`` was compared raw, and every correct completion
scored 0.0.

An answer is stated explicitly in one of three forms:

1. ``####`` (GSM8K): the rest of the line after the last marker. A ``####`` line that only says
   ``Final Answer`` is a markdown heading, and its answer is the next non-empty line;
2. ``\\boxed{...}``: the last box that closes; whitespace is allowed before the brace and nested
   braces are balanced, so ``\\boxed {\\frac{1}{2}}`` reads ``\\frac{1}{2}``;
3. ``answer is`` / ``answer:`` in any case, with markdown emphasis allowed around the word
   (``**Answer**:``) and the answer allowed on the next line: the last such phrase, up to the end
   of its clause. The clause ends at a line end, or at ``". "``, ``", "`` or ``"; "`` outside
   brackets, except that a comma or semicolon followed by a number continues a list
   (``41, 42 or 43``). A ``" ("`` aside belongs to the clause but not to the answer:
   ``42 (i.e. 42.0)`` answers ``42``.

A box outranks a phrase. A ``####`` line outranks both, unless one of them comes after it: then
the ``####`` line was a markdown heading, or an answer the text went on to correct.

The two delimited forms are authoritative: what they enclose IS the answer, number or not, so a
list or a tuple there is one answer. A phrase has no closing delimiter, so its values are read
from its clause: the stand-alone numbers outside brackets (the digits of ``(3, 4)``,
``\\frac{14}{3}``, ``2^{10}`` or ``2x`` belong to one expression, not to several values) and
every stand-alone number in its aside. A clause with MORE THAN ONE distinct value is a hedge
(``The answer is either 41 or 42.``): it states no answer, so a completion that hedges scores
0.0 and a gold that hedges is refused. One value, even repeated (``42 or 42.0``), is the
clause's number; a clause with no digits at all falls back to a completion's last number.

A text that states no explicit answer is free text. A reference is then its whole value, which
must fit on one line (a bare answer such as ``"42"`` or ``"Paris"``). A completion reads as its
last non-empty line, and its number is the last number anywhere in it, which is how
``math_verify`` has always read an unmarked completion.

Both sides are normalised the same way: surrounding whitespace; every ``$``, ``\\$``, ``\\(``,
``\\)``, ``\\[`` and ``\\]``; ``\\left`` / ``\\right``; ``\\dfrac`` / ``\\tfrac`` read as
``\\frac``; LaTeX spacing (``\\,``, ``\\!``, ``\\;``, ``\\:``, ``\\ ``) and ``{,}`` thousands
separators; markdown ``*`` at either end; trailing ``. , ; : !``; and the Unicode minus sign. A
number is then exactly one literal: an optional sign, digits with optional ``,`` groups of three,
an optional fraction and an optional exponent (``-1,000.5``, ``.5``, ``1e5``). Anything else,
such as ``42 apples``, ``\\frac{14}{3}`` or ``p - q``, is compared as text, ignoring case and
whitespace.

Every scan is linear in its input, because a completion is untrusted model output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import NamedTuple

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
# "answer is", "answer:", "**Answer**:", "Answer:\n42". Look-arounds rather than \b, so that
# "__Answer__:" matches while "answers:" and "the answer isn't" do not.
_PHRASE_RE = re.compile(
    r"(?<![A-Za-z0-9])answer(?![A-Za-z0-9])[*_\s]*(?:is\b(?:[*_\s]*:)?|:)[*_\s]*",
    re.IGNORECASE,
)
# The rest of a '####' line that is only an answer label ("#### Final Answer"): a heading.
_ANSWER_LABEL_RE = re.compile(r"[*_\s]*(?:final\s+)?answer[*_\s]*(?::[*_\s]*)?", re.IGNORECASE)
_NON_SPACE_RE = re.compile(r"\S")
# The characters that can end a phrase's clause, or open or close a bracket inside it.
_CLAUSE_CHAR_RE = re.compile(r"[()\[\]{}.,;]")
_LATEX_SPACING_RE = re.compile(r"\\[,!;: ]")
_MATH_DELIMITER_RE = re.compile(r"\\[()\[\]]")
# Not followed by a letter, so ``\leftarrow`` and ``\rightarrow`` stay whole.
_SIZING_RE = re.compile(r"\\(?:left|right)(?![A-Za-z])")
_FRAC_VARIANT_RE = re.compile(r"\\[dt]frac(?![A-Za-z])")
_NUMBER_PATTERN = r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
_NUMBER_RE = re.compile(_NUMBER_PATTERN)
# A bracket, or a number that stands alone: not part of a word ("2x", "3rd") or of a LaTeX
# expression ("\frac{14}{3}", "2^{10}", "4/5"). Brackets let a scan track depth.
_VALUE_TOKEN_RE = re.compile(
    r"[()\[\]{}]|(?<![\w\\{}^/.])(" + _NUMBER_PATTERN + r")(?![\w{}^/])"
)
# After ", " or "; " inside a clause: a number next means a list that goes on ("41, 42 or 43").
_NUMBER_AHEAD_RE = re.compile(r"\s*(?:\$|\\[(\[])?\s*[-+\u2212]?\.?\d")
_DIGIT_RE = re.compile(r"\d")

_EDGE_CHARS = " \t\r\n*"
_TRAILING_CHARS = _EDGE_CHARS + ".,;:!"
_OPENERS = "([{"
_CLOSERS = ")]}"


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
    text = _MATH_DELIMITER_RE.sub("", _canonical_symbols(text))
    text = text.replace("\\$", "").replace("$", "")
    text = _FRAC_VARIANT_RE.sub(r"\\frac", _SIZING_RE.sub("", text))
    text = text.lstrip(_EDGE_CHARS).rstrip(_TRAILING_CHARS)
    return " ".join(text.split())


def _text_key(text: str) -> str:
    """The form in which two non-numeric answers are compared: no whitespace, any case."""
    return "".join(text.split()).casefold()


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


def _line_end(text: str, start: int) -> int:
    end = text.find("\n", start)
    return len(text) if end < 0 else end


def _split_clause(line: str) -> tuple[str, str]:
    """Split the text after an answer phrase into its clause's ``(answer, aside)``.

    The clause ends at ". ", ", " or "; " outside brackets, unless a number follows the comma
    or semicolon ("41, 42 or 43" is one clause). A " (" aside is in the clause, not the answer.
    """
    line = line.lstrip()
    depth = 0
    aside = None
    for match in _CLAUSE_CHAR_RE.finditer(line):
        char, index = match.group(), match.start()
        if char in _OPENERS:
            if char == "(" and depth == 0 and aside is None and index and line[index - 1].isspace():
                aside = index
            depth += 1
        elif char in _CLOSERS:
            depth = max(0, depth - 1)
        elif depth == 0 and line[index + 1 : index + 2].isspace():
            if char in ",;" and _NUMBER_AHEAD_RE.match(line, index + 1):
                continue
            line = line[:index]
            break
    return (line, "") if aside is None else (line[:aside], line[aside:])


def _clause_values(answer: str, aside: str) -> frozenset[Decimal]:
    """The distinct values a clause names: the answer's stand-alone numbers outside brackets,
    and every stand-alone number in its aside. Both texts are normalised."""
    values: set[Decimal | None] = set()
    depth = 0
    for match in _VALUE_TOKEN_RE.finditer(answer):
        if match.group(1) is None:
            depth = depth + 1 if match.group() in _OPENERS else max(0, depth - 1)
        elif depth == 0:
            values.add(_to_decimal(match.group(1)))
    values.update(_to_decimal(m.group(1)) for m in _VALUE_TOKEN_RE.finditer(aside) if m.group(1))
    values.discard(None)
    return frozenset(values)


def _clause_reading(text: str, start: int) -> tuple[str, frozenset[Decimal]]:
    """Return ``(answer, values)`` for the clause that starts at ``start``."""
    answer, aside = _split_clause(text[start : _line_end(text, start)])
    answer = normalize_answer(answer)
    return answer, _clause_values(answer, normalize_answer(aside))


class _Explicit(NamedTuple):
    """An answer a text states explicitly."""

    answer: str
    delimited: bool  # after '####' or inside \boxed{}
    values: frozenset[Decimal]  # a phrase clause's distinct values; empty when delimited


def _marker_answer(text: str) -> tuple[int, _Explicit] | None:
    """Return ``(position, explicit answer)`` for the last ``####`` line, if it has one."""
    index = text.rfind(_MARKER)
    if index < 0:
        return None
    start = index + len(_MARKER)
    end = _line_end(text, start)
    if _ANSWER_LABEL_RE.fullmatch(text, start, end):
        # "#### Final Answer" is a markdown heading; the answer is the next non-empty line,
        # read like an answer phrase's clause.
        found = _NON_SPACE_RE.search(text, end)
        if found is None:
            return None
        answer, values = _clause_reading(text, found.start())
        return (found.start(), _Explicit(answer, False, values)) if answer else None
    answer = normalize_answer(text[start:end])
    return (start, _Explicit(answer, True, frozenset())) if answer else None


def _closing_brace(text: str, start: int, limit: int) -> int | None:
    depth = 1
    for match in _BRACE_RE.finditer(text, start, limit):
        depth += 1 if match.group() == "{" else -1
        if depth == 0:
            return match.start()
    return None


def _boxed_answer(text: str) -> tuple[int, str] | None:
    """Return ``(position, answer)`` for the last ``\\boxed{...}`` that closes, if any."""
    # Walk the boxes from the last one back. When a box never closes, an earlier box can only
    # close BEFORE that box's brace (it would have to close the later box first), so each scan
    # stops at the next box's brace and the whole walk is linear.
    matches = list(_BOXED_OPEN_RE.finditer(text))
    limit = len(text)
    for match in reversed(matches):
        close = _closing_brace(text, match.end(), limit)
        if close is not None:
            answer = normalize_answer(text[match.end() : close])
            return (match.start(), answer) if answer else None
        limit = match.end() - 1
    return None


def _phrase_answer(text: str) -> tuple[int, _Explicit] | None:
    """Return ``(position, explicit answer)`` for the last answer phrase, if its clause is not
    empty."""
    last = None
    for last in _PHRASE_RE.finditer(text):
        pass
    if last is None:
        return None
    answer, values = _clause_reading(text, last.end())
    return (last.end(), _Explicit(answer, False, values)) if answer else None


def _explicit_answer(text: str) -> _Explicit | None:
    """Return ``(answer, delimited, values)`` for the answer ``text`` states explicitly."""
    marker = _marker_answer(text)
    boxed = _boxed_answer(text)
    phrase = _phrase_answer(text)
    if marker is not None:
        position, explicit = marker
        # A box or a phrase that comes after the '####' line supersedes it.
        boxed = boxed if boxed is not None and boxed[0] >= position else None
        phrase = phrase if phrase is not None and phrase[0] >= position else None
        if boxed is None and phrase is None:
            return explicit
    if boxed is not None:
        return _Explicit(boxed[1], True, frozenset())
    if phrase is not None:
        return phrase[1]
    return None


def _is_hedge(explicit: _Explicit) -> bool:
    """An answer clause that names more than one distinct value ("either 41 or 42")."""
    return not explicit.delimited and len(explicit.values) > 1


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
    """Return the normalised answer ``text`` states explicitly, or ``None`` when it states none
    (a hedge such as "The answer is either 41 or 42." states none)."""
    found = _explicit_answer(text)
    return None if found is None or _is_hedge(found) else found.answer


def parse_reference(text: str) -> FinalAnswer | None:
    """Read a gold reference; ``None`` when it states no answer a reward could compare against.

    A reference is never read as free text: without an explicit answer it must be a bare,
    one-line answer. A reference that names an answer form but leaves it empty
    (``"...\\n####"``, ``"\\boxed{}"``), or whose answer phrase hedges between several values,
    is ``None`` too.
    """
    found = _explicit_answer(text)
    if found is not None:
        if _is_hedge(found):
            return None
        return FinalAnswer(found.answer, parse_number(found.answer))
    bare = text.strip()
    if not bare or "\n" in bare or _names_an_answer_form(bare):
        return None
    answer = normalize_answer(bare)
    return FinalAnswer(answer, parse_number(answer)) if answer else None


def parse_completion(text: str) -> FinalAnswer | None:
    """Read a model completion; ``None`` when it is empty or its answer phrase hedges."""
    found = _explicit_answer(text)
    if found is not None:
        if _is_hedge(found):
            return None  # "The answer is either 41 or 42." states no answer
        number = parse_number(found.answer)
        if number is None and not found.delimited:
            # A phrase's clause names at most one value here: "The answer is 41 apples, not
            # 42." is 41, as the clause ends at the comma. An expression ("\frac{14}{3}") has
            # no value of its own; a clause with no digits at all reads the text's last number.
            if found.values:
                (number,) = found.values
            elif _DIGIT_RE.search(found.answer) is None:
                number = _last_number(text)
        return FinalAnswer(found.answer, number)
    stripped = text.rstrip()
    if not stripped:
        return None
    last_line = stripped[stripped.rfind("\n") + 1 :]
    return FinalAnswer(normalize_answer(last_line), _last_number(text))


def answers_match(completion: FinalAnswer | None, reference: FinalAnswer) -> bool:
    """Exact match: by value when the reference is a number, else as normalised text."""
    if completion is None:
        return False
    if reference.number is not None:
        return completion.number is not None and completion.number == reference.number
    key = _text_key(reference.text)
    return bool(key) and _text_key(completion.text) == key
