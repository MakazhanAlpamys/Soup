r"""#1528: the custom-eval ``contains`` scorer stays linear in output and expected length.

#1490 made ``contains`` match ``expected`` only as a whole token. Its pattern started with a
lookbehind and ran under ``re.IGNORECASE``, so ``re`` tried it at every position: quadratic in
``len(output) * len(expected)``. The time at 8N is compared with the time at N: a linear scan
grows about 8x, a quadratic one about 64x. ``tests/_growth_timing.py`` runs the comparison on a
reference long enough to time, so a loaded runner cannot decide it.
"""

from __future__ import annotations

import random
import re

import pytest

from soup_cli.eval.custom import score_contains
from tests._growth_timing import assert_grows_linearly

_SCALE_N = 250  # the starting size; the helper doubles it until the run is long enough to time

SHAPES = {
    "word-starts-never-match": lambda k: ("a " * k, "a " * (k // 2) + "b"),
    "word-starts-match-at-end": lambda k: ("a " * k + "b", "a " * (k // 2) + "b"),
    "one-letter-run": lambda k: ("a" * (2 * k), "a" * k),
    "hyphen-pairs": lambda k: ("a-" * k, "-a" * (k // 2)),
}


@pytest.mark.parametrize("build", list(SHAPES.values()), ids=list(SHAPES))
def test_contains_grows_linearly(build):
    assert_grows_linearly(lambda pair: score_contains(*pair), build, _SCALE_N)


def _contains_before_1528(output: str, expected: str) -> bool:
    """The #1490 scorer, kept as the reference the linear rewrite must agree with."""
    needle = expected.strip()
    if not needle:
        return True
    pattern = r"(?<![A-Za-z0-9])" + re.escape(needle) + r"(?![A-Za-z0-9])"
    return re.search(pattern, output, re.IGNORECASE) is not None


_ALPHABET = "abAB01 .-#+(|"


def test_contains_agrees_with_the_previous_regex_on_short_strings():
    rng = random.Random(1528)
    for _ in range(20_000):
        output = "".join(rng.choices(_ALPHABET, k=rng.randint(0, 12)))
        expected = "".join(rng.choices(_ALPHABET, k=rng.randint(0, 4)))
        assert score_contains(output, expected) == _contains_before_1528(output, expected), (
            output,
            expected,
        )


@pytest.mark.parametrize(
    "output, expected",
    [("Welcome to İstanbul", "istanbul"), ("ſ", "s")],
)
def test_contains_folds_case_with_str_lower(output, expected):
    """``str.lower()`` and ``re.IGNORECASE`` disagree here; ``False`` matches v0.75.1."""
    assert score_contains(output, expected) is False


def test_a_gold_token_past_60000_characters_still_scores():
    output = "x " * 30_000 + "42" + " " * 14
    assert len(output) == 60_016 and output.index("42") == 60_000
    assert score_contains(output, "42") is True
