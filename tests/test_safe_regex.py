"""Structural complexity check for regexes taken from config."""

from __future__ import annotations

import re
import warnings

import pytest

from soup_cli.utils.safe_regex import check_config_regex, regex_complexity_problem

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
    assert regex_complexity_problem(".*.*.*z") is None


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


def test_no_deprecation_warning_on_import_and_use():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert regex_complexity_problem("a+") is None
