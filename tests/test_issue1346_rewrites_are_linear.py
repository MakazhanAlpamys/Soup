r"""#1346: the four normalisation rewrites stay linear in a completion's length.

A completion is untrusted model output, so each rewrite gets an input aimed at it, and the time
at 8N is compared with the time at N, as in ``test_issue1226_clause_scanner_is_linear.py``: a
linear scan grows about 8x, a quadratic one about 64x. ``tests/_growth_timing.py`` runs the
comparison on a reference long enough to time.
"""

from __future__ import annotations

import pytest

from soup_cli.trainer.rewards import accuracy_reward, math_verify_reward
from tests._growth_timing import assert_grows_linearly

_SCALE_N = 100  # the starting size; the helper doubles it until the run is long enough to time

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
    "spaced-fracs": lambda k: "#### \\frac" + " " * k + "1",
    "open-frac-group": lambda k: "#### \\frac{" + "a" * k,
    "variable-then-spaces": lambda k: "#### x" + " " * k,
    "variable-spaces-equals-spaces": lambda k: "#### x" + " " * k + "=" + " " * k,
}


def _msg(text: str) -> list[list[dict]]:
    return [[{"role": "assistant", "content": text}]]


def _score(text: str) -> None:
    accuracy_reward(_msg(text), answer=["42"])
    math_verify_reward(_msg(text), answer=["42"])


@pytest.mark.parametrize("build", list(REWRITE_INPUTS.values()), ids=list(REWRITE_INPUTS))
def test_each_rewrite_grows_linearly(build):
    assert_grows_linearly(_score, build, _SCALE_N)
