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
   of its clause. The clause ends at a line end, at ``". "``, ``", "`` or ``"; "`` outside
   brackets, or at the connective ``because`` or ``since`` (case-insensitive, requiring whitespace
   on both sides; not ``as``), except that a comma or semicolon followed by a number continues
   a list (``41, 42 or 43``). A ``" ("`` aside belongs to the clause but not to the answer:
   ``42 (i.e. 42.0)`` answers ``42``.

A box outranks a phrase. A ``####`` line outranks both, unless one of them comes after it: then
the ``####`` line was a markdown heading, or an answer the text went on to correct. Any later
phrase counts, chatter such as ``I hope this answer is helpful!`` included, because it cannot
be told apart from an answer under a heading. A reference is held to more: a ``####`` line that
is not a number and has more text after it may be a heading over an unmarked body, so such a
reference states no answer.

The two delimited forms are authoritative: what they enclose IS the answer, number or not, so a
list or a tuple there is one answer. A phrase has no closing delimiter, so its values are read
from its clause: the stand-alone numbers outside brackets and LaTeX environments (the digits of
``(3, 4)``, of a ``\\begin{pmatrix} 3 \\\\ 4 \\end{pmatrix}`` matrix, of ``\\frac{14}{3}``,
``2^{10}`` or ``2x``, and of a time or a ratio such as ``3:45`` or ``1:1,000``, belong to one
expression, not to several values) and every stand-alone number in its aside. A clause with
MORE THAN ONE distinct value is a hedge (``The answer is either 41 or 42.``): it states no
answer, so a completion that hedges scores 0.0 and a gold that hedges is refused. A justification
inline or after punctuation ends the clause (``42 because 6*7=42`` and ``42, because 6*7=42`` both
answer 42), while parenthetical justifications belong to the clause and hedge it (``42 (6*7=42)``).
One value, even repeated (``42 or 42.0``), is the clause's number; a clause with no digits at all
falls back to
a completion's last number.

A text that states no explicit answer is free text. A reference is then its whole value, which
must fit on one line (a bare answer such as ``"42"`` or ``"Paris"``). A completion reads as its
last non-empty line, and its number is the last number anywhere in it, which is how
``math_verify`` has always read an unmarked completion.

Both sides are normalised the same way: surrounding whitespace; every ``$``, ``\\$``, ``\\(``,
``\\)``, ``\\[`` and ``\\]``; ``\\left`` / ``\\right``; ``\\dfrac`` / ``\\tfrac`` read as
``\\frac``; one ``\\text{}`` / ``\\textbf{}`` / ``\\mathrm{}`` / ``\\mbox{}`` wrapper, unwrapped to
its contents; ``^\\circ``, ``^{\\circ}`` and ``\N{DEGREE SIGN}``, dropped; a compact ``\\frac``
argument, braced to match, whether it is a single bare character or an already-braced group
without nested braces, and whether or not a space separates it from ``\\frac`` (``\\frac12``,
``\\frac1{2}``, ``\\frac{1}2``, ``\\frac 34`` and ``\\frac9{19}`` all read ``\\frac{N}{D}``);
LaTeX spacing (``\\,``, ``\\!``, ``\\;``, ``\\:``, ``\\ ``, but never the second backslash of a
``\\\\`` line break) and ``{,}`` thousands separators; markdown ``*`` at either end; trailing
``. , ; : !``; and the Unicode minus sign. A number is then exactly one literal: an optional
sign, digits with optional ``,`` groups of three, an optional fraction and an optional exponent
(``-1,000.5``, ``.5``, ``1e5``). Anything else, such as ``42 apples``, ``\\frac{14}{3}`` or
``p - q``, is compared as text, ignoring case and whitespace. Units are not stripped
(``42 apples`` against ``42``), and nothing is evaluated (``\\frac{1}{2}`` against ``0.5``).

A single-letter variable prefix (``x = 7``) reads its right-hand side as the answer, on either
side, but the letter itself is kept: when BOTH sides name one and the names differ (``x = 3``
against ``y = 3``), the two answers conflict and the pair scores 0.0 even though their
right-hand sides are equal, because in an asymptote, directrix or axis problem the variable is
part of the answer. A side that names no variable never conflicts, so ``7`` still accepts
``\\boxed{x = 7}`` and ``x = 7`` still accepts ``\\boxed{7}``.

Every scan is linear in its input, because a completion is untrusted model output.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import NamedTuple

__all__ = [
    "BoxedAnswer",
    "FinalAnswer",
    "answers_match",
    "extract_final_answer",
    "iter_boxed_answers",
    "normalize_answer",
    "parse_completion",
    "parse_number",
    "parse_reference",
    "variables_conflict",
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
_CLAUSE_CHAR_RE = re.compile(r"[()\[\]{}.,;]|\b(?:because|since)\b", re.IGNORECASE)
# A LaTeX line break '\\' is consumed whole, so its second backslash is never read as the
# control space '\ ' ('\\ -14' must stay '\\ -14', not become '\-14').
_LATEX_SPACING_RE = re.compile(r"\\\\|\\[,!;: ]")
_MATH_DELIMITER_RE = re.compile(r"\\[()\[\]]")
# Not followed by a letter, so ``\leftarrow`` and ``\rightarrow`` stay whole.
_SIZING_RE = re.compile(r"\\(?:left|right)(?![A-Za-z])")
_FRAC_VARIANT_RE = re.compile(r"\\[dt]frac(?![A-Za-z])")
# One \text{}/\textbf{}/\mathrm{}/\mbox{} wrapper: what it encloses IS the answer (#1346).
_TEXT_WRAP_OPEN_RE = re.compile(r"\\(?:text|textbf|mathrm|mbox)\s*\{")
# "^\circ", "^{\circ}" or the degree sign; no unit beyond that is stripped (#1346). The two
# braces are matched as one alternative each, never independently optional, so this cannot
# eat the closing brace of an enclosing group (`\frac{180^\circ}{3}`'s outer `}`).
_DEGREE_RE = re.compile(r"\^(?:\{\\circ\}|\\circ)|\N{DEGREE SIGN}")
# A \frac argument: a braced group without nested braces, or a single bare character (TeX skips
# whitespace before an undelimited macro argument, so a space before it is allowed too).
# \frac12, \frac1{2}, \frac{1}2, \frac 34 and \frac9{19} all read \frac{N}{D} (#1346).
_FRAC_COMPACT_RE = re.compile(r"\\frac\s*(\{[^{}]*\}|[^\s{}\\])\s*(\{[^{}]*\}|[^\s{}\\])")
# A single-letter variable prefix at the start of an answer: "x = 7" splits into ("x", "7")
# (#1346). The name is kept on ``FinalAnswer.variable`` rather than discarded here, so a
# reference and a completion that name DIFFERENT variables can still be told apart.
_VAR_PREFIX_RE = re.compile(r"^([A-Za-z])\s*=\s*")
_UNSIGNED_PATTERN = r"(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
_NUMBER_PATTERN = r"[-+]?" + _UNSIGNED_PATTERN + r"(?:[eE][-+]?\d+)?"
_NUMBER_RE = re.compile(_NUMBER_PATTERN)
# Not glued to a word ("2x", "3rd") or to a LaTeX expression ("\frac{14}{3}", "2^{10}", "4/5").
_STANDS_ALONE = r"(?<![\w\\{}^/.])"
# A bracket, or a LaTeX environment's "\begin{...}" / "\end{...}", which lets a scan track depth
# (a matrix is one answer, not one value per entry); a time or a ratio ("3:45", "1:1,000"), which
# is ONE answer and is matched whole so that none of its numbers reads as a value; or, as group
# 1, a number that stands alone.
_VALUE_TOKEN_RE = re.compile(
    r"\\(?:begin|end)\{[A-Za-z*]*\}|[()\[\]{}]"
    + "|" + _STANDS_ALONE + "[-+]?" + _UNSIGNED_PATTERN + "(?::" + _UNSIGNED_PATTERN + ")+"
    + "|" + _STANDS_ALONE + r"(?<!\d:)(" + _NUMBER_PATTERN + r")(?![\w{}^/]|:\d)"
)
# After ", " or "; " inside a clause: a number next means a list that goes on ("41, 42 or 43",
# "41, $42$"). Only one ``\s*`` can match a given whitespace run, because the second one follows
# a mandatory "$", "\(" or "\[", so a run that no number follows costs linear time. With
# ``\s*(?:...)?\s*`` every split of the run was retried: one completion ending in a comma and
# 40,000 spaces took over 90 s to score.
_NUMBER_AHEAD_RE = re.compile(r"\s*(?:(?:\$|\\[(\[])\s*)?[-+\u2212]?\.?\d")
_DIGIT_RE = re.compile(r"\d")

_EDGE_CHARS = " \t\r\n*"
_TRAILING_CHARS = _EDGE_CHARS + ".,;:!"
_OPENERS = "([{"
_CLOSERS = ")]}"


@dataclass(frozen=True)
class FinalAnswer:
    """A final answer: its normalised ``text``, its ``number`` when it has one, and the
    single-letter ``variable`` a prefix such as ``x = 7`` named, when it named one."""

    text: str
    number: Decimal | None = None
    variable: str | None = None


@dataclass(frozen=True)
class BoxedAnswer:
    """One closed ``\\boxed{...}``: the braces' positions and the normalised content.

    ``end`` is the index just past the closing brace, so a caller can rank a box against other
    matches by position and read the last committed one.
    """

    start: int
    end: int
    answer: str


def _canonical_symbols(text: str) -> str:
    """Rewrite spellings that change no value: the Unicode minus, LaTeX thousands separators."""
    text = text.replace("\u2212", "-").replace("{,}", ",")
    return _LATEX_SPACING_RE.sub(lambda match: "\\\\" if match.group() == "\\\\" else "", text)


def _unwrap_text_command(text: str) -> str:
    """Unwrap one ``\\text{}``/``\\textbf{}``/``\\mathrm{}``/``\\mbox{}``: what it encloses IS
    the answer, so ``\\text{Paris}`` reads ``Paris``."""
    match = _TEXT_WRAP_OPEN_RE.search(text)
    if match is None:
        return text
    depth = 1
    for brace in _BRACE_RE.finditer(text, match.end()):
        depth += 1 if brace.group() == "{" else -1
        if depth == 0:
            close = brace.start()
            return text[: match.start()] + text[match.end() : close] + text[close + 1 :]
    return text


def _frac_compact_repl(match: re.Match[str]) -> str:
    numerator, denominator = match.group(1), match.group(2)
    numerator = numerator[1:-1] if numerator.startswith("{") else numerator
    denominator = denominator[1:-1] if denominator.startswith("{") else denominator
    return f"\\frac{{{numerator}}}{{{denominator}}}"


def normalize_answer(text: str) -> str:
    """Normalise an answer for comparison; see the module docstring for the rules."""
    text = _MATH_DELIMITER_RE.sub("", _canonical_symbols(text))
    text = text.replace("\\$", "").replace("$", "")
    text = _FRAC_VARIANT_RE.sub(r"\\frac", _SIZING_RE.sub("", text))
    text = _unwrap_text_command(text)
    text = _DEGREE_RE.sub("", text)
    text = _FRAC_COMPACT_RE.sub(_frac_compact_repl, text)
    text = text.lstrip(_EDGE_CHARS).rstrip(_TRAILING_CHARS)
    return " ".join(text.split())


def _split_variable(answer: str) -> tuple[str | None, str]:
    """Split a one-letter variable prefix off a normalised answer: ``x = 7`` -> ``("x", "7")``.

    The split happens after normalisation rather than inside it, so a name is available to
    :func:`variables_conflict` instead of being discarded (#1346)."""
    match = _VAR_PREFIX_RE.match(answer)
    return (None, answer) if match is None else (match.group(1), answer[match.end() :])


def variables_conflict(completion: FinalAnswer, reference: FinalAnswer) -> bool:
    """Both sides name a variable (``x = 3`` against ``y = 3``) and they are not the same one.

    A side that names no variable never conflicts, so a gold of ``7`` still accepts
    ``\\boxed{x = 7}`` and a gold of ``x = 7`` still accepts ``\\boxed{7}`` (#1346). The
    comparison is case-sensitive: a variable's case is part of its name, so ``x = 5``
    conflicts with ``\\boxed{X = 5}``."""
    named = reference.variable is not None and completion.variable is not None
    return named and completion.variable != reference.variable


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
    or semicolon ("41, 42 or 43" is one clause), or at the connective "because" or "since"
    (case-insensitive, requiring whitespace on both sides; not "as").
    A " (" aside is in the clause, not the answer.
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
        elif depth == 0:
            if char in ".,;":
                if line[index + 1 : index + 2].isspace():
                    if char in ",;" and _NUMBER_AHEAD_RE.match(line, index + 1):
                        continue
                    line = line[:index]
                    break
            elif (index > 0 and line[index - 1].isspace()) and (
                index + len(char) == len(line) or line[index + len(char)].isspace()
            ):
                line = line[:index].rstrip()
                break
    return (line, "") if aside is None else (line[:aside], line[aside:])


def _clause_values(answer: str, aside: str) -> frozenset[Decimal]:
    """The distinct values a clause names: the answer's stand-alone numbers outside brackets and
    LaTeX environments, and every stand-alone number in its aside. Both texts are normalised."""
    values: set[Decimal | None] = set()
    depth = 0
    for match in _VALUE_TOKEN_RE.finditer(answer):
        token = match.group()
        if match.group(1) is not None:
            if depth == 0:
                values.add(_to_decimal(match.group(1)))
        elif token in _OPENERS or token.startswith("\\begin"):
            depth += 1
        elif token in _CLOSERS or token.startswith("\\end"):
            depth = max(0, depth - 1)
        # Any other token is a time or a ratio: one answer, not a value.
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
    text_follows: bool = False  # a '####' answer with more text after its line


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
    if not answer:
        return None
    text_follows = _NON_SPACE_RE.search(text, end) is not None
    return start, _Explicit(answer, True, frozenset(), text_follows)


def iter_boxed_answers(text: str) -> Iterator[BoxedAnswer]:
    """Yield the innermost ``\\boxed{...}`` answers in ``text``, in the order they close.

    One pass over the braces. A box that holds another box is skipped: its content is the inner
    box, so it is never itself an answer, and skipping it stops the normalised contents from
    overlapping, which is what keeps the pass linear. A box that never closes is skipped too,
    instead of swallowing the rest of the text.
    """
    opens = {match.end() - 1: match.start() for match in _BOXED_OPEN_RE.finditer(text)}
    open_braces: list[int] = []
    open_boxes: list[int] = []  # the braces of the boxes still open, innermost last
    holds_a_box: set[int] = set()
    for match in _BRACE_RE.finditer(text):
        brace_at = match.start()
        if match.group() == "{":
            open_braces.append(brace_at)
            if brace_at in opens:
                if open_boxes:
                    holds_a_box.add(open_boxes[-1])
                open_boxes.append(brace_at)
        elif open_braces:
            brace = open_braces.pop()
            if brace in opens:
                open_boxes.pop()
                if brace not in holds_a_box:
                    answer = normalize_answer(text[brace + 1 : brace_at])
                    yield BoxedAnswer(opens[brace], match.end(), answer)


def _boxed_answer(text: str) -> tuple[int, str] | None:
    """Return ``(position, answer)`` for the last ``\\boxed{...}`` that closes, if any."""
    last = None
    for last in iter_boxed_answers(text):
        pass
    if last is None or not last.answer:
        return None
    return last.start, last.answer


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
    is ``None`` too. So is one whose ``####`` line is not a number and has more text after it
    (``"#### Solution\\nSix times seven is 42."``): that line may be a markdown heading over an
    unmarked body, and a reference must not be ambiguous. A completion's ``####`` line is its
    answer either way (``"#### Paris\\nHope this helps!"``).
    """
    found = _explicit_answer(text)
    if found is not None:
        variable, answer = _split_variable(found.answer)
        number = parse_number(answer)
        if not answer or _is_hedge(found) or (found.text_follows and number is None):
            return None
        return FinalAnswer(answer, number, variable)
    bare = text.strip()
    if not bare or "\n" in bare or _names_an_answer_form(bare):
        return None
    variable, answer = _split_variable(normalize_answer(bare))
    return FinalAnswer(answer, parse_number(answer), variable) if answer else None


def parse_completion(text: str) -> FinalAnswer | None:
    """Read a model completion; ``None`` when it is empty or its answer phrase hedges."""
    found = _explicit_answer(text)
    if found is not None:
        if _is_hedge(found):
            return None  # "The answer is either 41 or 42." states no answer
        variable, answer = _split_variable(found.answer)
        number = parse_number(answer)
        if number is None and not found.delimited:
            # A phrase's clause names at most one value here: "The answer is 41 apples, not
            # 42." is 41, as the clause ends at the comma. An expression ("\frac{14}{3}") has
            # no value of its own; a clause with no digits at all reads the text's last number.
            if found.values:
                (number,) = found.values
            elif _DIGIT_RE.search(answer) is None:
                number = _last_number(text)
        return FinalAnswer(answer, number, variable)
    stripped = text.rstrip()
    if not stripped:
        return None
    last_line = stripped[stripped.rfind("\n") + 1 :]
    variable, answer = _split_variable(normalize_answer(last_line))
    return FinalAnswer(answer, _last_number(text), variable)


def answers_match(completion: FinalAnswer | None, reference: FinalAnswer) -> bool:
    """Exact match: by value when the reference is a number, else as normalised text; ``False``
    when the two sides name different variables (#1346)."""
    if completion is None:
        return False
    if variables_conflict(completion, reference):
        return False
    if reference.number is not None:
        return completion.number is not None and completion.number == reference.number
    key = _text_key(reference.text)
    return bool(key) and _text_key(completion.text) == key
