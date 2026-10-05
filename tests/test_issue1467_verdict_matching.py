"""#1467: model output is read by its structure, not by a pattern that fails silently.

Four readers used to fail in the direction that looks like a real result: the LLM
judge floored a valid reply to the scale minimum when its reasoning quoted a nested
object, keyword matching with ``\\b`` could never match a keyword such as ``-5`` or
``(B)``, and the recipe judge kept any reply containing the letters ``OK``.
"""

from __future__ import annotations

import json

import pytest
import yaml
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.eval.judge import DEFAULT_RUBRIC, _parse_judge_response
from soup_cli.utils.behavior_battery import _agreement_rate
from soup_cli.utils.checklist_dsl import _mft_pass

runner = CliRunner()

NESTED_REASONING = 'It returns {"function": {"name": "get_weather"}} as asked.'
FIVES = {"helpfulness": 5.0, "accuracy": 5.0, "safety": 5.0}
#: ``_MAX_JUDGE_REPLY_CHARS`` in ``soup_cli.eval.judge``, spelled out so the limit is pinned.
JUDGE_REPLY_LIMIT = 65_536


class TestJudgeReadsTheWholeObject:
    def test_nested_reasoning_scores_first(self) -> None:
        reply = json.dumps(
            {
                "scores": {"helpfulness": 5, "accuracy": 5, "safety": 5},
                "reasoning": NESTED_REASONING,
            }
        )
        assert _parse_judge_response(reply, DEFAULT_RUBRIC) == (FIVES, NESTED_REASONING)

    def test_nested_reasoning_before_scores(self) -> None:
        reply = json.dumps(
            {
                "reasoning": NESTED_REASONING,
                "scores": {"helpfulness": 5, "accuracy": 5, "safety": 5},
            }
        )
        assert _parse_judge_response(reply, DEFAULT_RUBRIC) == (FIVES, NESTED_REASONING)

    def test_code_like_reasoning_and_surrounding_prose(self) -> None:
        body = json.dumps(
            {
                "scores": {"helpfulness": 4, "accuracy": 3, "safety": 5},
                "reasoning": "cfg = {'a': {'b': 1}}",
            }
        )
        scores, reasoning = _parse_judge_response(
            f"Here is my verdict {{draft}}:\n```json\n{body}\n```", DEFAULT_RUBRIC
        )
        assert scores == {"helpfulness": 4.0, "accuracy": 3.0, "safety": 5.0}
        assert reasoning == "cfg = {'a': {'b': 1}}"

    def test_capitalised_keys_and_score_objects_are_read(self) -> None:
        reply = json.dumps(
            {
                "scores": {"Helpfulness": 4, "ACCURACY": {"score": 2}, "safety": {"score": "5"}},
            }
        )
        scores, _ = _parse_judge_response(reply, DEFAULT_RUBRIC)
        assert scores == {"helpfulness": 4.0, "accuracy": 2.0, "safety": 5.0}

    def test_missing_criterion_names_it(self) -> None:
        reply = json.dumps({"scores": {"helpfulness": 5, "accuracy": 5}})
        with pytest.raises(ValueError, match="no score for criterion 'safety'"):
            _parse_judge_response(reply, DEFAULT_RUBRIC)

    @pytest.mark.parametrize("value", ["high", None, True, [5], {"points": 5}, "NaN", 10**400])
    def test_unreadable_score_names_the_criterion(self, value: object) -> None:
        reply = json.dumps({"scores": {"helpfulness": 5, "accuracy": value, "safety": 5}})
        with pytest.raises(ValueError, match="criterion 'accuracy' is not a number"):
            _parse_judge_response(reply, DEFAULT_RUBRIC)


def test_degenerate_nesting_is_an_unreadable_reply_not_a_crash() -> None:
    # _first_json_object, without `except RecursionError` the error escapes. 13,000
    # levels exceed the decoder's nesting limit on every supported Python (1,000 on 3.10
    # and 3.11, 10,000 on 3.12) and the reply stays under the size limit.
    reply = '{"a":' * 13_000
    assert len(reply) <= JUDGE_REPLY_LIMIT
    with pytest.raises(ValueError, match="No JSON object found"):
        _parse_judge_response(reply, DEFAULT_RUBRIC)


def test_a_capitalised_rubric_criterion_is_found_under_any_reply_casing() -> None:
    # _parse_judge_response, fold only the reply side
    rubric = {"criteria": [{"name": "Helpfulness", "weight": 1.0}], "scale": {"min": 1, "max": 5}}
    for reply_key in ("Helpfulness", "helpfulness", "HELPFULNESS"):
        scores, _ = _parse_judge_response(json.dumps({"scores": {reply_key: 4}}), rubric)
        assert scores == {"Helpfulness": 4.0}


def test_the_object_with_a_scores_key_wins_over_an_earlier_object() -> None:
    # _first_json_object, drop the preference for an object that has "scores"
    reply = '{"note": "draft"} then ' + json.dumps(
        {"scores": {"helpfulness": 5, "accuracy": 5, "safety": 5}, "reasoning": "final"}
    )
    scores, reasoning = _parse_judge_response(reply, DEFAULT_RUBRIC)
    assert scores == {"helpfulness": 5.0, "accuracy": 5.0, "safety": 5.0}
    assert reasoning == "final"


@pytest.mark.parametrize("scores", [[5, 5, 5], "5,5,5", 5, None])
def test_scores_that_is_not_an_object_is_a_value_error(scores: object) -> None:
    # _parse_judge_response, without the guard `.items()` raises AttributeError
    with pytest.raises(ValueError, match="'scores' must be a JSON object"):
        _parse_judge_response(json.dumps({"scores": scores}), DEFAULT_RUBRIC)


def test_a_criterion_scored_under_two_spellings_is_refused() -> None:
    # _parse_judge_response, without the check the later spelling's score wins
    reply = json.dumps({"scores": {"Helpfulness": 5, "helpfulness": 1, "accuracy": 5, "safety": 5}})
    with pytest.raises(
        ValueError,
        match="criterion 'helpfulness' more than once, as 'Helpfulness', 'helpfulness'",
    ):
        _parse_judge_response(reply, DEFAULT_RUBRIC)


def test_a_reply_over_the_size_limit_is_refused_before_it_is_scanned(monkeypatch) -> None:
    # _parse_judge_response, without the length check the reply is scanned
    import soup_cli.eval.judge as judge

    def scan(_text: str) -> None:
        raise AssertionError("an oversized reply was scanned")

    monkeypatch.setattr(judge, "_first_json_object", scan)
    with pytest.raises(ValueError, match="100,000 characters, over the 65,536-character limit"):
        _parse_judge_response("{ " * 50_000, DEFAULT_RUBRIC)


def test_a_reply_at_the_size_limit_is_still_read() -> None:
    # _parse_judge_response, `>=` in the length check refuses this reply
    body = json.dumps({"scores": {"helpfulness": 5, "accuracy": 5, "safety": 5}})
    reply = body + " " * (JUDGE_REPLY_LIMIT - len(body))
    assert len(reply) == JUDGE_REPLY_LIMIT
    assert _parse_judge_response(reply, DEFAULT_RUBRIC) == (FIVES, "")


PUNCTUATION_EDGES = (
    ("The answer is -5.", "-5"),
    ("-5", "-5"),
    ("Answer: (B)", "(B)"),
    ("I use C++ daily", "C++"),
    ("I use C# at work", "C#"),
    ("It costs $5 now", "$5"),
    ("Made in the U.S. today", "U.S."),
    ("Built on .NET 8", ".NET"),
    ("Coverage is 100% here", "100%"),
    ("Well, yes! Of course", "yes!"),
    ("Answer: (B)", "B"),
)


class TestKeywordsWithPunctuationEdges:
    @pytest.mark.parametrize(("response", "keyword"), PUNCTUATION_EDGES)
    def test_checklist_and_battery_match(self, response: str, keyword: str) -> None:
        assert _mft_pass(response, [keyword])
        assert _agreement_rate([response], [keyword]) == 1.0

    @pytest.mark.parametrize(
        ("response", "keyword"),
        [
            ("sand castle", "and"),
            ("unsafe", "safe"),
            ("C++ and C", "C#"),
            ("x-5y", "-5"),
            ("the safety net", "safe"),
            ("uses C++11 here", "C++"),
            ("andy came", "and"),
            ("5th", "5"),
        ],
    )
    def test_whole_word_rule_still_holds(self, response: str, keyword: str) -> None:
        assert not _mft_pass(response, [keyword])
        assert _agreement_rate([response], [keyword]) == 0.0

    def test_both_readers_share_one_matcher(self) -> None:
        from soup_cli.utils.behavior_battery import contains_whole_word

        assert contains_whole_word("answer: (b)", "(b)")
        assert not contains_whole_word("sand", "and")

    def test_checklist_cli_passes_a_correct_negative_answer(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "spec.yaml").write_text(
            yaml.safe_dump(
                {
                    "tests": [
                        {
                            "name": "neg",
                            "kind": "mft",
                            "prompts": ["What is 2 - 7?"],
                            "expected": ["-5"],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "evidence.json").write_text(
            json.dumps({"neg": ["The answer is -5."]}), encoding="utf-8"
        )
        result = runner.invoke(app, ["eval", "checklist", "spec.yaml", "-e", "evidence.json"])
        assert result.exit_code == 0, (result.output, repr(result.exception))

    def test_behavior_evidence_cli_matches_lettered_options(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "behavior.json").write_text(
            json.dumps(
                {
                    "pre_responses": ["Answer: (B)", "Answer: (A)"],
                    "post_responses": ["Answer: (B)", "Answer: (A)"],
                    "oracle": ["(B)", "(A)"],
                }
            ),
            encoding="utf-8",
        )
        result = runner.invoke(
            app,
            ["eval", "behavior", "run1", "--battery", "syceval", "--evidence", "behavior.json"],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))


def _run_judge_node(monkeypatch, replies: list[str]):
    import soup_cli.utils.data_forge as data_forge
    from soup_cli.utils.recipe_dag import RecipeNode
    from soup_cli.utils.recipe_run import _node_judge

    answers = iter(replies)
    monkeypatch.setattr(
        data_forge,
        "make_judge_provider_fn",
        lambda *a, **k: lambda prompt: {"text": next(answers)},
    )
    rows = [[{"text": f"row {i}"} for i in range(len(replies))]]
    return _node_judge(
        RecipeNode(name="j", kind="judge", config={}),
        rows,
        judge_provider="ollama",
        judge_model=None,
        judge_base_url=None,
    )


class TestRecipeJudgeReadsTheVerdict:
    @pytest.mark.parametrize(
        "reply",
        ["OK", "OK, nothing to REJECT", "ok.", "**OK**", "Okay", "OK - no reason to reject it"],
    )
    def test_positive_verdict_keeps_the_row(self, monkeypatch, reply: str) -> None:
        result = _run_judge_node(monkeypatch, [reply])
        assert len(result.rows) == 1
        assert result.failure_count == 0

    @pytest.mark.parametrize("reply", ["REJECT", "NOT OK", "not okay", "Rejected: off topic"])
    def test_negative_verdict_drops_the_row(self, monkeypatch, reply: str) -> None:
        result = _run_judge_node(monkeypatch, [reply])
        assert result.rows == ()
        assert result.failure_count == 0

    @pytest.mark.parametrize(
        "reply",
        [
            "This looks broken",
            "BAD - token limit",
            "The answer is a joke",
            "Books are fine? No.",
            "OKAYISH",
            "NOT REJECT",
            "",
        ],
    )
    def test_reply_without_a_verdict_is_a_failure(self, monkeypatch, reply: str) -> None:
        result = _run_judge_node(monkeypatch, [reply])
        assert result.rows == ()
        assert result.failure_count == 1
        assert result.call_count == 1


def test_the_recipe_judge_default_prompt_asks_for_one_word(monkeypatch) -> None:
    # _node_judge, revert the default prompt to the old wording
    import soup_cli.utils.data_forge as data_forge
    from soup_cli.utils.recipe_dag import RecipeNode
    from soup_cli.utils.recipe_run import _node_judge

    seen: list[str] = []

    def factory(*_args, **_kwargs):
        def call(prompt: str) -> dict:
            seen.append(prompt)
            return {"text": "OK"}

        return call

    monkeypatch.setattr(data_forge, "make_judge_provider_fn", factory)
    _node_judge(
        RecipeNode(name="j", kind="judge", config={}),
        [[{"text": "row"}]],
        judge_provider="ollama",
        judge_model=None,
        judge_base_url=None,
    )
    assert "one word" in seen[0]


def test_judge_accepts_a_rubric_name_that_is_not_a_string() -> None:
    rubric = {"criteria": [{"name": 1, "weight": 1.0}], "scale": {"min": 1, "max": 5}}
    scores, _ = _parse_judge_response(json.dumps({"scores": {"1": 4}}), rubric)
    assert scores == {1: 4.0}
