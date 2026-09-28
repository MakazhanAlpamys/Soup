r"""#1346: the four normalisation rewrites stay linear in a completion's length.

A completion is untrusted model output, so each rewrite gets an input aimed at it, and the time
at 8N is compared with the time at N, as in ``test_issue1226_clause_scanner_is_linear.py``: a
linear scan grows about 8x, a quadratic one about 64x.
"""

from __future__ import annotations

import time

import pytest

from soup_cli.trainer.rewards import accuracy_reward, math_verify_reward

_SCALE_N = 1_000
_SCALE_LIMIT = 24.0  # the same 3x slack over linear as the #1226 growth test

REWRITE_INPUTS = {
    "text-wrapper-never-closes": lambda k: "\\text{" * k,
    "text-wrapper-deep-braces": lambda k: "The answer is \\text{" + "{" * k,
    "text-command-then-spaces": lambda k: "The answer is \\text" + " " * k + "x",
    "many-closed-text-wrappers": lambda k: "\\boxed{" + "\\text{a}" * k + "}",
    "degree-carets": lambda k: "#### " + "^\\circ" * k,
    "degree-open-braces": lambda k: "#### " + "^{\\circ" * k,
    "degree-signs": lambda k: "#### " + "\N{DEGREE SIGN}" * k,
    "compact-fracs": lambda k: "#### " + "\\frac12" * k,
    "bare-fracs": lambda k: "#### " + "\\frac" * k,
    "variable-then-spaces": lambda k: "#### x" + " " * k,
    "variable-spaces-equals-spaces": lambda k: "#### x" + " " * k + "=" + " " * k,
}


def _msg(text: str) -> list[list[dict]]:
    return [[{"role": "assistant", "content": text}]]


def _best_time(text: str, repeats: int) -> float:
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        accuracy_reward(_msg(text), answer=["42"])
        math_verify_reward(_msg(text), answer=["42"])
        best = min(best, time.perf_counter() - start)
    return best


@pytest.mark.parametrize("build", list(REWRITE_INPUTS.values()), ids=list(REWRITE_INPUTS))
def test_each_rewrite_grows_linearly(build):
    small, large = build(_SCALE_N), build(8 * _SCALE_N)
    t_small = max(_best_time(small, repeats=5), 1e-7)
    t_large = float("inf")
    for _ in range(5):  # stop as soon as the larger input is shown to be fast enough
        t_large = min(t_large, _best_time(large, repeats=1))
        if t_large / t_small < _SCALE_LIMIT:
            break
    assert t_large / t_small < _SCALE_LIMIT, (t_small, t_large)
