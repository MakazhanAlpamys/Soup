"""Structural complexity check for regexes taken from config."""

from __future__ import annotations

import os
import re
import string
import subprocess
import sys
import warnings

import pytest

from soup_cli.utils.safe_regex import (
    MAX_TOTAL_REPEATS,
    MAX_UNBOUNDED_REPEATS,
    check_config_regex,
    regex_complexity_problem,
)

# The payload that proved the walker's gap: a chain of sibling bounded repeats
# over the SAME character. It is neither nested nor alternated nor unbounded,
# so every pre-v0.75.1 rule accepted it, yet matching it against 40 'a's does
# not finish. It is never matched here -- only refused.
CHAIN_PAYLOAD = "a{1,20}" * 12 + "z"


def _disjoint_chain(count: int) -> str:
    """``a{2}b{2}c{2}...`` -- *count* repeats that are adjacent but disjoint.

    Nothing here backtracks (each boundary is forced by a different literal),
    so this isolates the total-repeat cap from the adjacency rule.
    """
    assert count <= len(string.ascii_lowercase)
    return "".join(f"{ch}{{2}}" for ch in string.ascii_lowercase[:count])


REFUSED = [
    ("(a+)+b", "nested repetition"),
    ("(x+)+y", "nested repetition"),
    ("(a*)*", "nested repetition"),
    ("(.*)*", "nested repetition"),
    ("((.+))+z", "nested repetition"),
    ("(.+){2,}z", "nested repetition"),
    ("(?:a{1,1000}){1,1000}", "nested repetition"),
    ("(?:.|.)+z", "repeated alternation"),
    ("(a|aa)+b", "repeated alternation"),
    # A multi-character branch survives parsing as a BRANCH node inside the repeat.
    ("(?:(?:qa|k)_proj\\.)*x", "repeated alternation"),
    ("(a)\\1", "backreference"),
    ("(?P<x>a)(?P=x)", "backreference"),
    ("(x)(?(1)a|b)", "backreference"),
    (".*.*.*.*z", "too many unbounded repeats"),
    ("(?=(a+)+)b", "nested repetition"),
    # --- chains of sibling repeats (v0.75.1) -------------------------------
    (CHAIN_PAYLOAD, "too many repetitions"),
    (_disjoint_chain(MAX_TOTAL_REPEATS + 1), "too many repetitions"),
    ("a{1,20}" * 7 + "z", "too many repetitions"),
    # --- adjacent repeats whose bodies share a character --------------------
    ("a{1,20}a{1,20}", "ambiguous adjacent repetition"),
    (r"\d+\d*", "ambiguous adjacent repetition"),
    (".{2,}.{2,}", "ambiguous adjacent repetition"),
    (r"[0-9]{1,3}\d{1,3}", "ambiguous adjacent repetition"),
    (r"a{1,20}[ab]{1,20}", "ambiguous adjacent repetition"),
    # A capturing group is spliced into its sequence, so wrapping the repeats
    # does not hide the adjacency.
    ("(a{1,20})(a{1,20})", "ambiguous adjacent repetition"),
    # Zero-width items between the two repeats do not separate them.
    (r"a{1,20}\ba{1,20}", "ambiguous adjacent repetition"),
    (r"a{1,20}(?=a)a{1,20}", "ambiguous adjacent repetition"),
    # Neither does an item that can match nothing at all.
    ("a{1,20}b?a{1,20}", "ambiguous adjacent repetition"),
    ("a{1,20}b*a{1,20}", "ambiguous adjacent repetition"),
    # Bodies that are not a single character cannot be decided, and two
    # adjacent repeats are exactly the shape that has to fail closed.
    ("(?:ab){1,20}(?:cd){1,20}", "ambiguous adjacent repetition"),
]

ACCEPTED = [
    "model.layers.0.mlp.down_proj",
    r"model\.layers\.\d+\.mlp\..*",
    r"^model\.layers\.(1[0-9]|2[0-3])\.self_attn\.(q|k|v|o)_proj\.weight$",
    r"lm_head",
    r".*embed_tokens.*",
    r"[ab]+c",
    r"(ab){2}",
    r"layers\.[0-9]{1,3}\.",
    # Single-character alternation is folded into a character class by the
    # parser, so the repeat body is linear.
    r"(?:(?:q|k)_proj\.)*x",
    # --- must stay accepted after the v0.75.1 chain rules -------------------
    r"model\.layers\.\d+\.mlp\.experts\.\d+\.(gate|up|down)_proj\.\d+\.weight",
    # Adjacent, but the two bodies cannot match a common character, so the
    # split between them is forced and nothing backtracks.
    "a{1,20}b{1,20}",
    r"\d+\.\d+",
    r"\d+_\w+",
    # Exactly the total cap.
    _disjoint_chain(MAX_TOTAL_REPEATS),
    # Bounded digit runs are the documented way to write a deep MoE name
    # without spending the unbounded-repeat budget (see MAX_UNBOUNDED_REPEATS).
    r"model\.layers\.[0-9]{1,3}\.mlp\.experts\.[0-9]{1,4}\.w[0-9]{1,2}"
    r"\.lora_A\.[0-9]{1,3}\.weight",
]


@pytest.mark.parametrize(("pattern", "reason"), REFUSED)
def test_refused(pattern, reason):
    assert regex_complexity_problem(pattern) == reason


@pytest.mark.parametrize("pattern", ACCEPTED)
def test_accepted(pattern):
    assert regex_complexity_problem(pattern) is None


def test_check_config_regex_message_names_field_and_pattern():
    expected = r"training\.lr_groups: pattern '\(a\+\)\+b' is too complex"
    with pytest.raises(ValueError, match=expected):
        check_config_regex("(a+)+b", "training.lr_groups")


def test_check_config_regex_accepts_linear_pattern():
    assert check_config_regex(r"model\.layers\.\d+", "training.lr_groups") is None


def test_exactly_the_unbounded_limit_is_accepted():
    # Separated, so only the unbounded cap is in play. The adjacent spelling of
    # the same count (".*.*.*z") is refused by the adjacency rule instead --
    # pinned below -- because three repeats sharing one alphabet make every
    # boundary between them a free choice.
    assert regex_complexity_problem(".*a.*b.*c") is None
    assert regex_complexity_problem(".*.*.*z") == "ambiguous adjacent repetition"


def test_large_bounded_repeat_counts_as_unbounded():
    assert regex_complexity_problem("a{65}b{65}c{65}d{65}") == "too many unbounded repeats"
    assert regex_complexity_problem("a{64}b{64}c{64}d{64}") is None


def test_deep_group_nesting_is_refused():
    # Capturing groups: the parser flattens flag-less non-capturing groups.
    pattern = "(" * 150 + "a" + ")" * 150
    assert regex_complexity_problem(pattern) == "nested repetition"


def test_invalid_regex_raises_re_error():
    with pytest.raises(re.error):
        regex_complexity_problem("(unclosed")


# --- the repetition-chain rules (v0.75.1) -----------------------------------


def test_exactly_the_total_repeat_limit_is_accepted():
    pattern = _disjoint_chain(MAX_TOTAL_REPEATS)
    assert pattern.count("{2}") == MAX_TOTAL_REPEATS
    assert regex_complexity_problem(pattern) is None


def test_one_repeat_over_the_total_limit_is_refused():
    assert regex_complexity_problem(_disjoint_chain(MAX_TOTAL_REPEATS + 1)) == (
        "too many repetitions"
    )


def test_optional_single_repeats_do_not_spend_the_total_budget():
    """``x?`` cannot multiply the search space, so it is not counted."""
    pattern = "a?b?c?d?e?f?g?h?i?j?k?"
    assert pattern.count("?") > MAX_TOTAL_REPEATS
    assert regex_complexity_problem(pattern) is None


def test_the_unbounded_cap_is_reported_before_the_total_cap():
    """Precedence is fixed so a reason string cannot drift between releases."""
    assert regex_complexity_problem(".*.*.*.*z") == "too many unbounded repeats"


def test_the_total_cap_is_reported_before_the_adjacency_rule():
    # Both rules fire on this pattern; the count is the more specific answer.
    assert regex_complexity_problem("a{1,20}" * 7) == "too many repetitions"


def test_adjacent_repeats_with_disjoint_bodies_are_accepted():
    assert regex_complexity_problem("a{1,20}b{1,20}c{1,20}") is None


def test_structural_refusals_still_win_over_the_chain_rules():
    assert regex_complexity_problem("(a+)+" + "b{1,20}" * 9) == "nested repetition"


def test_the_proven_payload_is_refused():
    assert regex_complexity_problem(CHAIN_PAYLOAD) == "too many repetitions"


def test_the_proven_payload_is_refused_far_faster_than_it_would_match():
    """The refusal is structural: it never touches the matching engine.

    ``re.compile(CHAIN_PAYLOAD).match('a' * 40)`` does not finish in minutes,
    which is why this runs in a subprocess with a timeout and only ever calls
    the checker -- the payload is never matched against anything.
    """
    code = (
        "import time\n"
        "from soup_cli.utils.safe_regex import regex_complexity_problem\n"
        f"pattern = {CHAIN_PAYLOAD!r}\n"
        "start = time.perf_counter()\n"
        "reason = regex_complexity_problem(pattern)\n"
        "print(reason, time.perf_counter() - start)\n"
    )
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=5,
        env=os.environ,
    )
    assert res.returncode == 0, (res.stdout, res.stderr)
    reason, _, elapsed = res.stdout.strip().rpartition(" ")
    assert reason == "too many repetitions", res.stdout
    assert float(elapsed) < 0.1, res.stdout


def test_the_unbounded_cap_is_still_three():
    """Kept deliberately: it is the only rule that sees NON-adjacent ``.*``.

    ``.*a.*b.*c.*d`` has four unbounded repeats, is under the total cap and is
    not adjacent anywhere, yet it is polynomial of degree four on a failing
    match. Relaxing the cap would also delete two refusals this module already
    ships (``.*.*.*.*z`` and ``a{65}b{65}c{65}d{65}``).
    """
    assert MAX_UNBOUNDED_REPEATS == 3
    assert regex_complexity_problem(".*a.*b.*c.*d") == "too many unbounded repeats"
    assert regex_complexity_problem(".*a.*b.*c") is None


def test_no_deprecation_warning_on_import_and_use():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert regex_complexity_problem("a+") is None
