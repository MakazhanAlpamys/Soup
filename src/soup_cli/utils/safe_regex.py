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
  above ``LARGE_REPEAT`` in the whole pattern.

Only stdlib imports, so the config schema can use it without slowing the CLI.
"""

from __future__ import annotations

try:  # Python 3.11+
    from re import _constants as _c  # type: ignore[attr-defined]
    from re import _parser as _p  # type: ignore[attr-defined]
except ImportError:  # Python 3.10: importing these warns only on 3.11+
    import sre_constants as _c  # type: ignore[no-redef]
    import sre_parse as _p  # type: ignore[no-redef]

MAX_UNBOUNDED_REPEATS: int = 3
LARGE_REPEAT: int = 64
_MAX_WALK_DEPTH = 100

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


def _is_multi(hi: int) -> bool:
    return hi == _c.MAXREPEAT or hi > 1


def _is_unbounded(hi: int) -> bool:
    return hi == _c.MAXREPEAT or hi > LARGE_REPEAT


def _walk(items, in_repeat: bool, depth: int, unbounded: list[int]) -> str | None:
    if depth > _MAX_WALK_DEPTH:
        return NESTED_REPETITION
    for op, av in items:
        child: str | None = None
        if op in _REPEAT_OPS:
            _lo, hi, body = av
            if _is_unbounded(hi):
                unbounded[0] += 1
            multi = _is_multi(hi)
            if multi and in_repeat:
                return NESTED_REPETITION
            child = _walk(body, in_repeat or multi, depth + 1, unbounded)
        elif op == _c.BRANCH:
            if in_repeat:
                return REPEATED_ALTERNATION
            for alternative in av[1]:
                child = _walk(alternative, in_repeat, depth + 1, unbounded)
                if child is not None:
                    return child
        elif op in _BACKREF_OPS:
            return BACKREFERENCE
        elif op == _c.SUBPATTERN:
            child = _walk(av[-1], in_repeat, depth + 1, unbounded)
        elif op in _ASSERT_OPS:
            child = _walk(av[1], in_repeat, depth + 1, unbounded)
        elif _ATOMIC_GROUP is not None and op == _ATOMIC_GROUP:
            child = _walk(av, in_repeat, depth + 1, unbounded)
        if child is not None:
            return child
    return None


def regex_complexity_problem(pattern: str) -> str | None:
    """Return ``None`` when *pattern* is safe to match, else a short reason.

    Raises ``re.error`` when the pattern is not a valid regular expression.
    """
    tree = _p.parse(pattern)
    unbounded = [0]
    problem = _walk(tree, False, 0, unbounded)
    if problem is not None:
        return problem
    if unbounded[0] > MAX_UNBOUNDED_REPEATS:
        return TOO_MANY_UNBOUNDED
    return None


def check_config_regex(pattern: str, field: str) -> None:
    """Raise ``ValueError`` naming *field* when *pattern* is too complex."""
    reason = regex_complexity_problem(pattern)
    if reason is not None:
        raise ValueError(
            f"{field}: pattern {pattern!r} is too complex to match safely "
            f"({reason}); use a literal name prefix or a simpler pattern"
        )
