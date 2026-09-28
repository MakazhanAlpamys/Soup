"""Regression coverage for structural guards on runtime regex sinks (#1137)."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

_UNSAFE = "(a+)+b"
_SAFE = "a|b"


def test_custom_eval_refuses_complex_regex_before_matching(monkeypatch) -> None:
    from soup_cli.eval import custom

    def unexpected_match(*_args, **_kwargs):
        raise AssertionError("unsafe custom-eval regex reached re.search")

    monkeypatch.setattr(
        custom,
        "re",
        SimpleNamespace(search=unexpected_match, error=re.error, IGNORECASE=re.IGNORECASE),
    )
    with pytest.raises(
        ValueError,
        match=r"eval\.custom\.expected: pattern .* too complex",
    ):
        custom.score_regex("aaaa", _UNSAFE)


def test_custom_eval_accepts_simple_alternation() -> None:
    from soup_cli.eval.custom import score_regex

    assert score_regex("b", _SAFE)


def test_diagnose_refuses_complex_regex_before_compiling(monkeypatch) -> None:
    from soup_cli.utils.diagnose import format as diagnose_format

    def unexpected_compile(*_args, **_kwargs):
        raise AssertionError("unsafe diagnose regex reached re.compile")

    monkeypatch.setattr(
        diagnose_format,
        "re",
        SimpleNamespace(compile=unexpected_compile, error=re.error),
    )
    with pytest.raises(
        ValueError,
        match=r"diagnose\.regex_pattern: pattern .* too complex",
    ):
        diagnose_format.matches_regex("aaaa", _UNSAFE)


def test_diagnose_accepts_simple_alternation() -> None:
    from soup_cli.utils.diagnose.format import matches_regex

    assert matches_regex("b", _SAFE)


def test_recipe_validator_refuses_complex_regex_before_compiling(monkeypatch) -> None:
    from soup_cli.utils import recipe_run
    from soup_cli.utils.recipe_dag import RecipeNode

    def unexpected_compile(*_args, **_kwargs):
        raise AssertionError("unsafe recipe regex reached re.compile")

    monkeypatch.setattr(
        recipe_run,
        "re",
        SimpleNamespace(compile=unexpected_compile, error=re.error),
    )
    node = RecipeNode(name="guarded", kind="validator", config={"regex": _UNSAFE})
    with pytest.raises(
        ValueError,
        match=r"recipe\.nodes\['guarded'\]\.config\.regex: pattern .* too complex",
    ):
        recipe_run._node_validator(node, (({"text": "aaaa"},),))


def test_recipe_validator_accepts_simple_alternation() -> None:
    from soup_cli.utils.recipe_dag import RecipeNode
    from soup_cli.utils.recipe_run import _node_validator

    node = RecipeNode(name="guarded", kind="validator", config={"regex": _SAFE})
    rows = ({"text": "a"}, {"text": "c"})

    assert _node_validator(node, (rows,)) == [{"text": "a"}]
