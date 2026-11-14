"""``soup ship --gate loop-v2``: learned the corrections, kept the held-out edits?

Today's verdict judges the new version on a task set and a general suite; it never sees a
correction a person made. This gate adds the two checks the continual-learning loop needs and makes
them decide:

* **scoring, ``added_recall``.** The *added words* of a correction are its content words
  (lowercased, punctuation stripped, stop words removed) that the answer it corrected lacked; with
  fewer than two of them, all of its content words. An answer is a *hit* when at least half of those
  words are in it, its ``<think>`` block removed first.
* **check 1, corrections learned** (``--corrections``): for each edit, ask both versions every
  paraphrase of its question and count the hits of each. More hits for the new version is *fixed*,
  fewer is *broken*, equal is a tie. A one-sided exact sign test over fixed vs broken, with at least
  five edits that changed either way.
* **check 2, held-out edits** (``--held-out``): non-inferiority. Both versions are asked the
  original question and the share of the added words each answer holds is compared (the recall
  value, not the hit). A one-sided exact sign test over worse vs better.

The rule is that of the consuming application's reference implementation, word for word:
the tokenizer, ``lower()`` (not ``casefold``), the stop list and the 0.5 threshold. That
application only records the two checks; here they decide (see ``decide_loop_gate``).

This module is pure python and light: no torch, no model. Model loading and the live run stay in
``commands/ship.py``; the verdict, panel and JSON live in ``utils/ship_verdict.py``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from math import comb
from typing import Callable, Dict, List, Optional, Sequence, Tuple

GATE_LOOP_V2 = "loop-v2"
GATES: Tuple[str, ...] = (GATE_LOOP_V2,)

# Failed-rule codes the gate adds to ship's own (missing_baseline / task_win / regression /
# adapter_noop). ``task_win`` is never one of them: under this gate the task win is information.
FAILED_HELD_OUT = "held_out"  # check 2: significantly worse on the held-out edits
FAILED_CORRECTIONS = "corrections"  # check 1: the corrections were not learned
FAILED_CORRECTIONS_INSUFFICIENT = "corrections_insufficient"  # check 1: < 5 edits changed
FAILED_NO_CORRECTIONS = "no_corrections"  # check 1: there is nothing to check

# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------

THRESHOLD = 0.5  # the share of the added words an answer must hold to be a hit
# New tokens an answer to a corrections or held-out question may run to, as the consuming
# application's ``ANSWER_TOKENS``. The live eval's 256 (``BEHAVIOURAL_MAX_NEW_TOKENS``) can cut a
# verbose answer before it reaches the corrected fact, which scores as a miss the model did not
# earn.
ANSWER_MAX_NEW_TOKENS = 512
MIN_ADDED = 2  # an edit that adds fewer content words than this is scored on all of them
MIN_DISCORDANT = 5  # check 1: fewer edits than this that changed either way is not enough evidence
MIN_HELD_OUT = 5  # check 2: fewer held-out edits than this is not measured
WEAK_BELOW = 10  # check 2: fewer held-out edits than this is weak evidence
SIGNIFICANT = 0.05

# One small fixed English stop list. Negations (not, no, never) and numbers are not on it: they can
# be the whole correction. There are no stop words for other languages: "в" in a Russian answer
# counts as a content word (a known limitation, shared with the consuming application).
STOP_WORDS = frozenset("""
a an the and or but if so as of at by for in on to up with from into onto than then that
this these those is are was were be been being am do does did doing have has had having will
would can could shall should may might must it its it's i me my we our you your he him his she
her they them their what which who whom whose when where why how there here also just very too
about any some such own same each all both
""".split())

_TOKEN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE | re.ASCII)
_THINK_OPEN = re.compile(r"<think>", re.IGNORECASE | re.ASCII)


def scoring() -> Dict[str, object]:
    """The ``scoring`` object of the loop iteration format for this method (a fresh copy)."""
    return {"method": "added_recall", "threshold": THRESHOLD, "template": None}


def strip_think(text: Optional[str]) -> str:
    """The answer without its reasoning.

    Closed ``<think>`` blocks go; a lone ``</think>`` means the opening was in the chat template, so
    what follows the last one is the answer; an unclosed ``<think>`` is a thought cut off, so what
    precedes it is.
    """
    text = _THINK_BLOCK.sub("", text or "")
    # Offsets are taken on ``text`` itself: lower-casing first would move them (U+0130 lower-cases
    # to two code points) and cut the answer one character off.
    closes = list(_THINK_CLOSE.finditer(text))
    if closes:
        text = text[closes[-1].end():]
    opening = _THINK_OPEN.search(text)
    if opening is not None:
        text = text[: opening.start()]
    return text.strip()


def content_words(text: Optional[str]) -> frozenset:
    """Lowercase words without punctuation, minus the stop words."""
    return frozenset(
        word for word in _TOKEN.findall((text or "").lower()) if word not in STOP_WORDS
    )


def added_words(expected: str, original: str) -> frozenset:
    """What the correction added.

    The corrected text's content words that the original answer lacked; all of the corrected text's
    content words when it adds fewer than ``MIN_ADDED``.
    """
    words = content_words(expected)
    added = words - content_words(original)
    return added if len(added) >= MIN_ADDED else words


def words_present(answer: str, words: frozenset) -> int:
    """How many of ``words`` the answer holds (its thinking removed)."""
    return len(words & content_words(strip_think(answer)))


def added_recall_hit(expected: str, original: str, answer: str) -> bool:
    """Whether ``answer`` holds at least half of what ``expected`` added to ``original``.

    With no words to look for there is nothing to find: never a hit.
    """
    words = added_words(expected, original)
    return bool(words) and words_present(answer, words) / len(words) >= THRESHOLD


def sign_test(hits: int, total: int) -> Optional[float]:
    """One-sided exact sign test: P(X >= hits) for X ~ Binomial(total, 1/2).

    Integer arithmetic until the final division. ``None`` when ``total`` is 0 (nothing to test).
    """
    if total <= 0:
        return None
    if hits <= 0:
        return 1.0
    if hits > total:
        return 0.0
    return sum(comb(total, count) for count in range(hits, total + 1)) / 2 ** total


# ---------------------------------------------------------------------------
# The two input files
# ---------------------------------------------------------------------------

MAX_ROWS = 10_000  # per file; every row is two or more model generations
MAX_TEXT_CHARS = 20_000  # one text field
MAX_ID_CHARS = 256
MAX_PARAPHRASES = 20  # per edit


class GateInputError(ValueError):
    """A corrections or held-out file the gate refuses. The message names the file and the line."""


@dataclass(frozen=True)
class CorrectionEdit:
    """One correction: the corrected text, the answer it corrected, and the questions to ask."""

    edit_id: str
    expected: str
    original: str
    paraphrases: Tuple[str, ...]


@dataclass(frozen=True)
class HeldOutEdit:
    """One edit kept out of training: the original question and its corrected text."""

    edit_id: str
    prompt: str
    expected: str
    original: str


@dataclass(frozen=True)
class GateInputs:
    """The parsed ``--corrections`` and ``--held-out`` files (either may be empty)."""

    corrections: Tuple[CorrectionEdit, ...] = ()
    held_out: Tuple[HeldOutEdit, ...] = ()


def _json_kind(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def _rows(text: str, source: str):
    """Yield ``(line number, JSON object)`` for each non-blank line, within ``MAX_ROWS`` rows."""
    if text.startswith("\ufeff"):
        text = text[1:]
    count = 0
    # Split on "\n" only: a JSON string may hold U+2028, U+0085 and friends, which
    # ``str.splitlines`` would cut a row at. A trailing "\r" is a CRLF line end.
    for number, line in enumerate(text.split("\n"), 1):
        line = line.rstrip("\r")
        if not line.strip():
            continue
        count += 1
        if count > MAX_ROWS:
            raise GateInputError(f"{source} line {number}: more than {MAX_ROWS} rows")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GateInputError(f"{source} line {number}: invalid JSON: {exc.msg}") from None
        except RecursionError:
            raise GateInputError(
                f"{source} line {number}: invalid JSON: nested too deeply"
            ) from None
        if not isinstance(row, dict):
            raise GateInputError(
                f"{source} line {number}: expected a JSON object, got {_json_kind(row)}"
            )
        yield number, row


def _text(row: dict, key: str, source: str, number: int, *, limit: Optional[int] = None,
          allow_empty: bool = False) -> str:
    limit = MAX_TEXT_CHARS if limit is None else limit
    if key not in row:
        raise GateInputError(f'{source} line {number}: missing "{key}"')
    value = row[key]
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        kind = "a string" if allow_empty else "a non-empty string"
        raise GateInputError(f'{source} line {number}: "{key}" must be {kind}')
    if len(value) > limit:
        raise GateInputError(f'{source} line {number}: "{key}" is longer than {limit} characters')
    return value


def _paraphrases(row: dict, source: str, number: int) -> Tuple[str, ...]:
    if "paraphrases" not in row:
        raise GateInputError(f'{source} line {number}: missing "paraphrases"')
    value = row["paraphrases"]
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_PARAPHRASES:
        raise GateInputError(
            f'{source} line {number}: "paraphrases" must be a list of 1 to {MAX_PARAPHRASES} '
            "non-empty strings"
        )
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise GateInputError(
                f'{source} line {number}: "paraphrases"[{index}] must be a non-empty string'
            )
        if len(item) > MAX_TEXT_CHARS:
            raise GateInputError(
                f'{source} line {number}: "paraphrases"[{index}] is longer than '
                f"{MAX_TEXT_CHARS} characters"
            )
    return tuple(value)


def _unique(seen: Dict[str, int], edit_id: str, source: str, number: int) -> None:
    if edit_id in seen:
        raise GateInputError(
            f'{source} line {number}: duplicate edit_id "{edit_id}" (first on line {seen[edit_id]})'
        )
    seen[edit_id] = number


def parse_corrections(text: str, source: str) -> Tuple[CorrectionEdit, ...]:
    """Parse a ``--corrections`` JSONL: ``{edit_id, expected, original, paraphrases}`` per line.

    ``source`` is how the file is named in an error ("--corrections 'c.jsonl'"). Raises
    ``GateInputError`` at the first bad row with ``<source> line <n>: <problem>``.
    """
    edits: List[CorrectionEdit] = []
    seen: Dict[str, int] = {}
    for number, row in _rows(text, source):
        edit_id = _text(row, "edit_id", source, number, limit=MAX_ID_CHARS)
        expected = _text(row, "expected", source, number)
        original = _text(row, "original", source, number, allow_empty=True)
        paraphrases = _paraphrases(row, source, number)
        _unique(seen, edit_id, source, number)
        edits.append(CorrectionEdit(edit_id, expected, original, paraphrases))
    return tuple(edits)


def parse_held_out(text: str, source: str) -> Tuple[HeldOutEdit, ...]:
    """Parse a ``--held-out`` JSONL: ``{edit_id, prompt, expected, original}`` per line."""
    edits: List[HeldOutEdit] = []
    seen: Dict[str, int] = {}
    for number, row in _rows(text, source):
        edit_id = _text(row, "edit_id", source, number, limit=MAX_ID_CHARS)
        prompt = _text(row, "prompt", source, number)
        expected = _text(row, "expected", source, number)
        original = _text(row, "original", source, number, allow_empty=True)
        _unique(seen, edit_id, source, number)
        edits.append(HeldOutEdit(edit_id, prompt, expected, original))
    return tuple(edits)


# ---------------------------------------------------------------------------
# The two checks
# ---------------------------------------------------------------------------

Generator = Callable[[str], str]


@dataclass(frozen=True)
class CorrectionsCheck:
    """``verdict.corrections`` of the loop iteration format."""

    fixed: int
    broken: int
    ties: int
    n_edits: int
    n_paraphrases: int
    p_value: Optional[float]
    outcome: str  # pass | fail | insufficient | not_applicable

    def to_dict(self) -> Dict[str, object]:
        return {
            "fixed": self.fixed, "broken": self.broken, "ties": self.ties,
            "n_edits": self.n_edits, "n_paraphrases": self.n_paraphrases,
            "p_value": self.p_value, "scoring": scoring(), "outcome": self.outcome,
        }


@dataclass(frozen=True)
class StyleNote:
    """How a pairwise judge preferred the new version's held-out answers. Information only."""

    preferred_new: int
    preferred_old: int
    n: int
    p_value: Optional[float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "preferred_new": self.preferred_new, "preferred_old": self.preferred_old,
            "n": self.n, "p_value": self.p_value,
        }


@dataclass(frozen=True)
class HeldOutCheck:
    """``verdict.held_out`` of the loop iteration format."""

    better: int
    worse: int
    ties: int
    n: int
    p_value: Optional[float]
    outcome: str  # pass | fail | weak | not_measured
    style: Optional[StyleNote] = None

    def to_dict(self) -> Dict[str, object]:
        found: Dict[str, object] = {
            "better": self.better, "worse": self.worse, "ties": self.ties, "n": self.n,
            "p_value": self.p_value, "scoring": scoring(), "outcome": self.outcome,
        }
        if self.style is not None:
            found["style"] = self.style.to_dict()
        return found


def corrections_check(fixed: int, broken: int, ties: int, n_paraphrases: int, *,
                      rows: int) -> CorrectionsCheck:
    """Check 1 from its counts. ``rows`` is how many corrections the file held.

    ``not_applicable`` when there are none; ``insufficient`` when fewer than ``MIN_DISCORDANT`` of
    them changed either way (also when rows exist but nothing could be asked); ``pass`` when the
    sign test over fixed vs broken is below 0.05; ``fail`` otherwise.
    """
    discordant = fixed + broken
    p_value = sign_test(fixed, discordant)
    if rows == 0:
        outcome = "not_applicable"
    elif discordant < MIN_DISCORDANT:
        outcome = "insufficient"
    else:
        outcome = "pass" if p_value is not None and p_value < SIGNIFICANT else "fail"
    return CorrectionsCheck(
        fixed, broken, ties, fixed + broken + ties, n_paraphrases, p_value, outcome,
    )


def held_out_check(
    better: int, worse: int, ties: int, style: Optional[StyleNote] = None,
) -> HeldOutCheck:
    """Check 2 from its counts: ``not_measured`` below 5 edits, ``fail`` when the new version is
    significantly worse, ``weak`` below 10 edits, ``pass`` otherwise. ``style`` never decides."""
    total = better + worse + ties
    p_value = sign_test(worse, better + worse)
    if total < MIN_HELD_OUT:
        outcome = "not_measured"
    elif p_value is not None and p_value < SIGNIFICANT:
        outcome = "fail"
    else:
        outcome = "weak" if total < WEAK_BELOW else "pass"
    return HeldOutCheck(better, worse, ties, total, p_value, outcome, style)


def check_corrections(old_gen: Generator, new_gen: Generator,
                      edits: Sequence[CorrectionEdit]) -> CorrectionsCheck:
    """Check 1 over ``edits``: every paraphrase goes to the old version first, then to the new one.

    The hits of each version are counted per edit and compared as numbers (one hit against two is
    fixed), not as booleans.
    """
    old_hits = [
        [added_recall_hit(edit.expected, edit.original, old_gen(q)) for q in edit.paraphrases]
        for edit in edits
    ]
    new_hits = [
        [added_recall_hit(edit.expected, edit.original, new_gen(q)) for q in edit.paraphrases]
        for edit in edits
    ]
    fixed = broken = ties = 0
    for before, after in zip(old_hits, new_hits):
        if sum(after) > sum(before):
            fixed += 1
        elif sum(after) < sum(before):
            broken += 1
        else:
            ties += 1
    n_paraphrases = sum(len(edit.paraphrases) for edit in edits)
    return corrections_check(fixed, broken, ties, n_paraphrases, rows=len(edits))


def style_from_win_rate(win_rate: object, n: int) -> Optional[StyleNote]:
    """What a pairwise judge preferred, from the new version's win rate over ``n`` held-out edits.

    A tie counts as half a win, so the counts are the rate times ``n``, rounded. The p-value is the
    two-sided sign test of the split. ``None`` when the rate is not a number in [0, 1] or nothing
    was compared.
    """
    if isinstance(win_rate, bool) or not isinstance(win_rate, (int, float)):
        return None
    if not 0 <= win_rate <= 1 or n < 1:
        return None
    new = min(n, int(win_rate * n + 0.5 + 1e-9))
    old = n - new
    tail = sign_test(max(new, old), n)  # n >= 1 here, so this is a number
    return StyleNote(new, old, n, None if tail is None else min(1.0, 2 * tail))


def check_held_out(old_gen: Generator, new_gen: Generator, edits: Sequence[HeldOutEdit], *,
                   judge_rate: Optional[Callable[[List[Tuple[str, str, str]]], float]] = None,
                   ) -> HeldOutCheck:
    """Check 2 over the held-out edits: both versions get the original question, and the number of
    added words each answer holds is compared (the recall value, not the hit).

    ``judge_rate``, when given, is a pairwise judge over ``(prompt, old answer, new answer)``
    triples that returns the new version's win rate; it fills ``style`` and never decides.
    """
    words = [added_words(edit.expected, edit.original) for edit in edits]
    old_answers = [old_gen(edit.prompt) for edit in edits]
    new_answers = [new_gen(edit.prompt) for edit in edits]
    better = worse = ties = 0
    for edit_words, before, after in zip(words, old_answers, new_answers):
        held_before = words_present(before, edit_words)
        held_after = words_present(after, edit_words)
        if held_after > held_before:
            better += 1
        elif held_after < held_before:
            worse += 1
        else:
            ties += 1
    style = None
    if judge_rate is not None and edits:
        triples = [(edit.prompt, before, after)
                   for edit, before, after in zip(edits, old_answers, new_answers)]
        style = style_from_win_rate(judge_rate(triples), len(edits))
    return held_out_check(better, worse, ties, style)


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoopGate:
    """What the gate measured, carried on the verdict."""

    corrections: CorrectionsCheck
    held_out: HeldOutCheck
    canary_decides: bool
    gate: str = GATE_LOOP_V2


@dataclass(frozen=True)
class GateDecision:
    ship: bool
    failed_rule: Optional[str]
    canary_decides: bool


def decide_loop_gate(corrections: CorrectionsCheck, held_out: HeldOutCheck, *,
                     suites_failed_rule: Optional[str]) -> GateDecision:
    """SHIP iff check 1 passed AND check 2 is not a fail AND the suites hold.

    ``suites_failed_rule`` is ship's own rule code when the general suites did not hold
    (``missing_baseline``: nothing measured, ``regression``: a benchmark dropped), else ``None``.
    The task win and the style judge are information only.

    When several things are wrong the one reported is the most decisive: the suites, then the
    held-out edits, then the corrections. A check that is merely short of evidence
    (``insufficient``, ``not_applicable``) never outranks a real failure.

    ``canary_decides`` is true only for "there are no corrections to check" with nothing else wrong:
    the verdict is DON'T SHIP, but the canary's ratings, not this gate, settle the rollout.
    """
    if suites_failed_rule is not None:
        return GateDecision(False, suites_failed_rule, False)
    if held_out.outcome == "fail":
        return GateDecision(False, FAILED_HELD_OUT, False)
    if corrections.outcome == "fail":
        return GateDecision(False, FAILED_CORRECTIONS, False)
    if corrections.outcome == "insufficient":
        return GateDecision(False, FAILED_CORRECTIONS_INSUFFICIENT, False)
    if corrections.outcome == "not_applicable":
        return GateDecision(False, FAILED_NO_CORRECTIONS, True)
    return GateDecision(True, None, False)


def unmeasured_loop_gate(inputs: GateInputs) -> LoopGate:
    """The gate when nothing was asked (an all-zero adapter ends the run before any model loads)."""
    return LoopGate(
        corrections=corrections_check(0, 0, 0, 0, rows=len(inputs.corrections)),
        held_out=held_out_check(0, 0, 0),
        canary_decides=False,
    )


# ---------------------------------------------------------------------------
# Words for the verdict
# ---------------------------------------------------------------------------


def _p(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def gate_reason(failed_rule: Optional[str], loop_gate: LoopGate) -> Optional[str]:
    """The one-line reason for a gate rule, or ``None`` for a rule that is ship's own."""
    corrections, held_out = loop_gate.corrections, loop_gate.held_out
    if failed_rule == FAILED_CORRECTIONS_INSUFFICIENT:
        changed = corrections.fixed + corrections.broken
        return f"not enough corrections yet ({changed}/{MIN_DISCORDANT})"
    if failed_rule == FAILED_NO_CORRECTIONS:
        return "no corrections to check - the canary's ratings decide"
    if failed_rule == FAILED_CORRECTIONS:
        return (
            "the new version did not learn the corrections "
            f"(fixed {corrections.fixed}, broken {corrections.broken}, p={_p(corrections.p_value)})"
        )
    if failed_rule == FAILED_HELD_OUT:
        return (
            "the new version is significantly worse on the held-out edits "
            f"(worse {held_out.worse}, better {held_out.better}, p={_p(held_out.p_value)})"
        )
    return None


def held_out_note(check: HeldOutCheck) -> Optional[str]:
    """In words: the held-out check did not run (or ran on few edits), or ``None`` if it did.

    The check can only fail on a handful of discordant edits, and the held-out set is a share of
    one iteration's new prompts, so it is often below ``MIN_HELD_OUT`` edits. A SHIP then says
    nothing about the held-out edits and must not be read as "held-out passed".
    """
    if check.outcome == "not_measured":
        return (
            f"held-out edits not checked: {check.n} of {MIN_HELD_OUT} needed; this verdict "
            "says nothing about them"
        )
    if check.outcome == "weak":
        return (
            f"held-out edits checked on few edits ({check.n}; {WEAK_BELOW} for a firm check): "
            "weak evidence"
        )
    return None


def gate_lines(loop_gate: LoopGate) -> List[str]:
    """The gate's measurements as plain lines, for the panel and the rubric."""
    corrections, held_out = loop_gate.corrections, loop_gate.held_out
    lines = [
        f"Corrections learned: fixed {corrections.fixed}, broken {corrections.broken}, "
        f"ties {corrections.ties}"
        + (f" (p={_p(corrections.p_value)})" if corrections.p_value is not None else "")
        + f" - {corrections.outcome}",
        f"Held-out edits: better {held_out.better}, worse {held_out.worse}, ties {held_out.ties}"
        + (f" (p={_p(held_out.p_value)})" if held_out.p_value is not None else "")
        + f" - {held_out.outcome}",
    ]
    note = held_out_note(held_out)
    if note is not None:
        lines.append(f"  Note: {note}")
    if held_out.style is not None:
        style = held_out.style
        lines.append(
            f"Style judge (information only): new {style.preferred_new}, "
            f"old {style.preferred_old} of {style.n}"
        )
    return lines
