"""Structural complexity check for regular expressions taken from config.

Patterns in ``soup.yaml`` (``training.unfrozen_parameters``,
``training.lr_groups[*].pattern``, ``lora.rank_pattern`` /
``lora.alpha_pattern`` keys) are matched against every parameter name of a
model. A pattern whose structure lets Python's backtracking engine explore an
exponential number of paths can stall config loading or training, so such
patterns are refused before they are ever matched.

The check parses the pattern with the stdlib regex parser and walks the tree.
It is O(pattern length): no timing, no probe match, no subprocess. It refuses:

* a repeat with a maximum above 1 whose body contains another such repeat
  ("nested repetition") or an alternation ("repeated alternation");
* any backreference, including a conditional group reference;
* more than ``MAX_UNBOUNDED_REPEATS`` repeats whose maximum is unbounded or
  above ``LARGE_REPEAT`` in the whole pattern;
* more than ``MAX_TOTAL_REPEATS`` repeats of any kind in the whole pattern
  ("too many repetitions");
* two repeats that sit next to each other in the same sequence and can match a
  common character ("ambiguous adjacent repetition").

The last two rules close a gap the first three did not see: a chain of SIBLING
bounded repeats over overlapping character sets, such as ``("a{1,20}" * 12) +
"z"``, is neither nested nor alternated nor unbounded, yet the engine has to
try every way of splitting the subject between the repeats before it can
report a failure. Matching that pattern against forty ``a`` characters does
not finish.

Only stdlib imports, so the config schema can use it without slowing the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass

try:  # Python 3.11+
    from re import _constants as _c  # type: ignore[attr-defined]
    from re import _parser as _p  # type: ignore[attr-defined]
except ImportError:  # Python 3.10: importing these warns only on 3.11+
    import sre_constants as _c  # type: ignore[no-redef]
    import sre_parse as _p  # type: ignore[no-redef]

MAX_UNBOUNDED_REPEATS: int = 3

# The deepest legitimate pattern this project has to accept is a fully
# qualified MoE parameter name -- one repeat per numeric segment, three of them
# in ``model\.layers\.\d+\.mlp\.experts\.\d+\.(gate|up|down)_proj\.\d+\.weight``
# -- and a four-segment name is plausible. Six is double the known worst case
# and, measured, costs nothing today: no shipped recipe, template or example
# config sets any of the four regex-bearing fields at all.
#
# The ceiling matters because the work a chain of k repeats can force is
# combinatorial in k. With "ambiguous adjacent repetition" also refused, the
# remaining reachable shape needs a nullable separator between every pair, and
# its cost is a polynomial of degree k-1 in the length of the parameter name.
# Six keeps that degree at five on subjects that are under a few hundred
# characters; raising it buys nothing a real pattern needs.
MAX_TOTAL_REPEATS: int = 6

LARGE_REPEAT: int = 64
_MAX_WALK_DEPTH = 100
# Splicing groups open for the adjacency scan is bounded separately from the
# walk: patterns nested deeper than this are refused by the walk's own depth
# guard, so stopping early here can only miss a refusal, never invent one.
_MAX_FLATTEN_DEPTH = 20

_REPEAT_OPS = frozenset(
    {_c.MAX_REPEAT, _c.MIN_REPEAT}
    | ({_c.POSSESSIVE_REPEAT} if hasattr(_c, "POSSESSIVE_REPEAT") else set())
)
_BACKREF_OPS = frozenset({_c.GROUPREF, _c.GROUPREF_EXISTS})
_ASSERT_OPS = frozenset({_c.ASSERT, _c.ASSERT_NOT})
_ATOMIC_GROUP = getattr(_c, "ATOMIC_GROUP", None)

NESTED_REPETITION = "nested repetition"
REPEATED_ALTERNATION = "repeated alternation"
BACKREFERENCE = "backreference"
TOO_MANY_UNBOUNDED = "too many unbounded repeats"
TOO_MANY_REPEATS = "too many repetitions"
AMBIGUOUS_ADJACENT = "ambiguous adjacent repetition"

# --- character sets ---------------------------------------------------------
#
# Overlap between two repeat bodies is decided over ASCII, plus one bit saying
# "this atom can also match something outside ASCII". Parameter names are
# ASCII, so the approximation only loses precision for atoms that overlap
# exclusively outside it.

_ASCII = frozenset(range(128))
_ASCII_DIGIT = frozenset(range(ord("0"), ord("9") + 1))
_ASCII_WORD = (
    _ASCII_DIGIT
    | frozenset(range(ord("a"), ord("z") + 1))
    | frozenset(range(ord("A"), ord("Z") + 1))
    | frozenset({ord("_")})
)
_ASCII_SPACE = frozenset({0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20})
_ASCII_LINEBREAK = frozenset({0x0A})

# Value -> (ASCII members, can also match a non-ASCII character). The negated
# categories carry the non-ASCII bit; the positive ones do not, which is the
# ASCII approximation above.
_CATEGORIES: dict[object, tuple[frozenset[int], bool]] = {}
for _name, _members in (
    ("CATEGORY_DIGIT", _ASCII_DIGIT),
    ("CATEGORY_WORD", _ASCII_WORD),
    ("CATEGORY_SPACE", _ASCII_SPACE),
    ("CATEGORY_LINEBREAK", _ASCII_LINEBREAK),
):
    _value = getattr(_c, _name, None)
    if _value is not None:
        _CATEGORIES[_value] = (_members, False)
    _negated = getattr(_c, _name.replace("CATEGORY_", "CATEGORY_NOT_"), None)
    if _negated is not None:
        _CATEGORIES[_negated] = (frozenset(_ASCII - _members), True)

_CharSet = tuple[frozenset[int], bool]


def _literal_charset(code: int) -> _CharSet:
    return (frozenset({code}), False) if code < 128 else (frozenset(), True)


def _atom_charset(op, av) -> _CharSet | None:
    """The characters a single-character atom can match, or ``None``.

    ``None`` means "cannot be decided here" and callers fail closed.
    """
    if op == _c.LITERAL:
        return _literal_charset(av)
    if op == _c.NOT_LITERAL:
        return (frozenset(_ASCII - _literal_charset(av)[0]), True)
    if op == _c.ANY:
        return (frozenset(_ASCII - _ASCII_LINEBREAK), True)
    if op == _c.RANGE:
        low, high = av
        return (frozenset(range(low, min(high, 127) + 1)), high > 127)
    if op == _c.CATEGORY:
        return _CATEGORIES.get(av)
    if op == _c.IN:
        return _in_charset(av)
    return None


def _in_charset(items) -> _CharSet | None:
    negated = False
    members: set[int] = set()
    non_ascii = False
    for op, av in items:
        if op == _c.NEGATE:
            negated = True
            continue
        part = _atom_charset(op, av)
        if part is None:
            return None
        members |= part[0]
        non_ascii = non_ascii or part[1]
    if negated:
        return (frozenset(_ASCII - members), True)
    return (frozenset(members), non_ascii)


def _overlaps(left: _CharSet | None, right: _CharSet | None) -> bool:
    """Whether two repeat bodies can match a common character (fail closed)."""
    if left is None or right is None:
        return True
    return bool(left[0] & right[0]) or (left[1] and right[1])


# --- sequence shape ---------------------------------------------------------


def _is_multi(hi: int) -> bool:
    return hi == _c.MAXREPEAT or hi > 1


def _is_unbounded(hi: int) -> bool:
    return hi == _c.MAXREPEAT or hi > LARGE_REPEAT


def _is_zero_width(op) -> bool:
    """Anchors and lookarounds consume nothing, so they do not separate."""
    return op == _c.AT or op in _ASSERT_OPS


def _flatten(items, depth: int) -> list:
    """Splice group contents into their sequence.

    ``(a{1,20})(a{1,20})`` is two groups at the top level and two adjacent
    repeats once they are spliced; without this, wrapping a chain in groups
    would hide it from the adjacency rule.
    """
    out: list = []
    for entry in items:
        op, av = entry
        if op == _c.SUBPATTERN and depth < _MAX_FLATTEN_DEPTH:
            out.extend(_flatten(av[-1], depth + 1))
        else:
            out.append(entry)
    return out


def _repeat_body_charset(body) -> _CharSet | None:
    """The character set of a repeat whose body is one atom, else ``None``."""
    atoms = [entry for entry in _flatten(body, 0) if not _is_zero_width(entry[0])]
    if len(atoms) != 1:
        return None
    return _atom_charset(*atoms[0])


def _has_adjacent_ambiguity(items) -> bool:
    """Whether two repeats sit next to each other over a shared character.

    Two such repeats make the boundary between them a free choice, which is
    what turns a chain of them into a combinatorial search. Items that consume
    nothing (anchors, lookarounds) and items that can match nothing at all
    (``b?``, ``b*``) do not separate the pair.
    """
    entries = _flatten(items, 0)
    for index, (op, av) in enumerate(entries):
        if op not in _REPEAT_OPS or not _is_multi(av[1]):
            continue
        left = _repeat_body_charset(av[2])
        for op_right, av_right in entries[index + 1 :]:
            if _is_zero_width(op_right):
                continue
            if op_right not in _REPEAT_OPS:
                break
            lo_right, hi_right, body_right = av_right
            if _is_multi(hi_right) and _overlaps(left, _repeat_body_charset(body_right)):
                return True
            if lo_right != 0:
                break
    return False


@dataclass
class _Tally:
    """Counts collected by one walk of the tree."""

    unbounded: int = 0
    total: int = 0
    adjacent: bool = False


def _walk(items, in_repeat: bool, depth: int, tally: _Tally) -> str | None:
    if depth > _MAX_WALK_DEPTH:
        return NESTED_REPETITION
    if not tally.adjacent and _has_adjacent_ambiguity(items):
        tally.adjacent = True
    for op, av in items:
        child: str | None = None
        if op in _REPEAT_OPS:
            _lo, hi, body = av
            if _is_unbounded(hi):
                tally.unbounded += 1
            multi = _is_multi(hi)
            if multi:
                tally.total += 1
            if multi and in_repeat:
                return NESTED_REPETITION
            child = _walk(body, in_repeat or multi, depth + 1, tally)
        elif op == _c.BRANCH:
            if in_repeat:
                return REPEATED_ALTERNATION
            for alternative in av[1]:
                child = _walk(alternative, in_repeat, depth + 1, tally)
                if child is not None:
                    return child
        elif op in _BACKREF_OPS:
            return BACKREFERENCE
        elif op == _c.SUBPATTERN:
            child = _walk(av[-1], in_repeat, depth + 1, tally)
        elif op in _ASSERT_OPS:
            child = _walk(av[1], in_repeat, depth + 1, tally)
        elif _ATOMIC_GROUP is not None and op == _ATOMIC_GROUP:
            child = _walk(av, in_repeat, depth + 1, tally)
        if child is not None:
            return child
    return None


def regex_complexity_problem(pattern: str) -> str | None:
    """Return ``None`` when *pattern* is safe to match, else a short reason.

    Raises ``re.error`` when the pattern is not a valid regular expression.
    """
    tree = _p.parse(pattern)
    tally = _Tally()
    problem = _walk(tree, False, 0, tally)
    if problem is not None:
        return problem
    # Fixed order: the counts are the more specific answer, and a reason string
    # that moved between releases would break configs people grep for.
    if tally.unbounded > MAX_UNBOUNDED_REPEATS:
        return TOO_MANY_UNBOUNDED
    if tally.total > MAX_TOTAL_REPEATS:
        return TOO_MANY_REPEATS
    if tally.adjacent:
        return AMBIGUOUS_ADJACENT
    return None


def check_config_regex(pattern: str, field: str) -> None:
    """Raise ``ValueError`` naming *field* when *pattern* is too complex."""
    reason = regex_complexity_problem(pattern)
    if reason is not None:
        raise ValueError(
            f"{field}: pattern {pattern!r} is too complex to match safely "
            f"({reason}); use a literal name prefix or a simpler pattern"
        )
