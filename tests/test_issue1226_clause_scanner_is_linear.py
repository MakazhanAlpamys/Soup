r"""The #1226 answer-clause scanner is linear in a whitespace run (found reviewing the fix).

An answer clause ends at ", " or "; " unless a number follows ("41, 42 or 43"). The look-ahead
for that number had two ``\s*`` around an optional ``$`` / ``\(`` / ``\[``, so a whitespace run
that no number follows was re-split between them at every length: "The answer is 42," plus
40,000 spaces took over 90 s across the two rewards, and that text is a GRPO policy's own
completion.

The test compares the time at N with the time at 8N instead of checking an absolute time, so it
holds on any machine and fails fast: a linear scan grows about 8x, the quadratic one grew
61-66x, and at N = 1,000 the quadratic scan failed each case in 10-13 s. The absolute 2 s
bound on 20,000 spaces is in ``TestAdversarialInputsStayLinear`` of
``test_issue1226_reward_gold_parsing.py``.

The second class pins what the look-ahead matches, so a faster rewrite cannot change which
clauses continue a list.
"""

from __future__ import annotations

import time
from decimal import Decimal

import pytest

from soup_cli.trainer.rewards import accuracy_reward, math_verify_reward
from soup_cli.utils.final_answer import parse_completion


def _msg(text: str) -> list[list[dict]]:
    return [[{"role": "assistant", "content": text}]]


def _scores(completion: str, gold: str) -> tuple[float, float]:
    return (
        accuracy_reward(_msg(completion), answer=[gold])[0],
        math_verify_reward(_msg(completion), answer=[gold])[0],
    )


# A whitespace run after ", " or "; " in an answer clause, with no number after it. The first
# three were quadratic. The last two put the run after "$" or "\(", where the look-ahead's
# second whitespace match sits, so a rewrite cannot move the problem there unnoticed.
WHITESPACE_RUNS = {
    "comma-spaces": lambda k: "The answer is 42," + " " * k + "x",
    "semicolon-tabs": lambda k: "Answer: 42;" + "\t" * k + "x",
    "comma-unicode-whitespace": lambda k: (
        "The answer is 42," + " \t\N{NO-BREAK SPACE}\N{IDEOGRAPHIC SPACE}" * (k // 4) + "x"
    ),
    "comma-dollar-spaces": lambda k: "The answer is 42, $" + " " * k + "x",
    "comma-math-delimiter-spaces": lambda k: "The answer is 42, \\(" + " " * k + "x",
}

_SCALE_N = 1_000
# A linear scan takes about 8x as long on 8x the input: 6.5-7.4x measured, the fixed cost of a
# call pulling it below 8. The quadratic scan took 61-66x. The limit leaves 3x of slack over
# linear, and every time is a best of several runs, so a stall on a busy runner would have to
# hit every run of the larger input to fail the test.
_SCALE_LIMIT = 24.0


def _best_time(text: str, repeats: int) -> float:
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        parse_completion(text)
        best = min(best, time.perf_counter() - start)
    return best


class TestTheClauseScannerIsLinear:
    @pytest.mark.parametrize("build", list(WHITESPACE_RUNS.values()), ids=list(WHITESPACE_RUNS))
    def test_time_grows_linearly_with_a_whitespace_run(self, build):
        small, large = build(_SCALE_N), build(8 * _SCALE_N)
        # The clause ends at the comma or semicolon, so the answer is 42 at every size.
        assert parse_completion(large).number == Decimal(42)
        t_small = max(_best_time(small, repeats=5), 1e-7)
        t_large = float("inf")
        for _ in range(5):  # stop as soon as the larger input is shown to be fast enough
            t_large = min(t_large, _best_time(large, repeats=1))
            if t_large <= _SCALE_LIMIT * t_small:
                break
        assert t_large <= _SCALE_LIMIT * t_small, (
            f"8x the whitespace took {t_large / t_small:.0f}x as long "
            f"({t_small * 1e3:.2f} ms -> {t_large * 1e3:.2f} ms)"
        )


# After ", " or "; " a number continues the list in every spelling the look-ahead accepts: a
# sign, a leading ".", a "$", "\(" or "\[" before it, and whitespace in between.
CONTINUED_LISTS = [
    ("The answer is 41, 42 or 43.", ["41", "42", "43"]),
    ("The answer is 41,\t42.", ["41", "42"]),
    ("The answer is 41, $42$.", ["41", "42"]),
    ("The answer is 41, $ 42$.", ["41", "42"]),
    ("The answer is 41, \\(42\\).", ["41", "42"]),
    ("The answer is 41, \\[ 42 \\].", ["41", "42"]),
    ("The answer is 41, -42.", ["41", "-42"]),
    ("The answer is 41, +42.", ["41", "42"]),
    ("The answer is 41; \N{MINUS SIGN}42.", ["41", "-42"]),
    ("The answer is 41, .5.", ["41", "0.5"]),
]
# ...and anything else after the comma ends the clause, whose one value is the answer.
ENDED_CLAUSES = [
    "The answer is 41, not 42.",
    "The answer is 41, $x$ is 42.",
    "The answer is 41, \\(x\\) is 42.",
    "The answer is 41, -x is 42.",
    "The answer is 41, . 42",
]


class TestWhatContinuesAList:
    @pytest.mark.parametrize(("completion", "values"), CONTINUED_LISTS)
    def test_a_number_after_the_comma_continues_the_list(self, completion, values):
        assert parse_completion(completion) is None  # the list is a hedge
        for value in values:
            assert _scores(completion, value) == (0.0, 0.0), value

    @pytest.mark.parametrize("completion", ENDED_CLAUSES)
    def test_anything_else_after_the_comma_ends_the_clause(self, completion):
        assert parse_completion(completion).number == Decimal(41)
        assert _scores(completion, "41") == (1.0, 1.0)
        assert _scores(completion, "42") == (0.0, 0.0)
