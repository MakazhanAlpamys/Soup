"""Keep parse failures distinct from empty tool arguments (#1734)."""

import json

import pytest

from soup_cli.eval.custom import EvalTask, score_task, tool_call_args_subset

BAD_ARGUMENTS = ["not-json", "[]", "null", [], None, 17]


def call(arguments, name="lookup"):
    return json.dumps({"function": {"name": name, "arguments": arguments}})


@pytest.mark.parametrize("arguments", BAD_ARGUMENTS)
@pytest.mark.parametrize("gold_arguments", [{}, {"query": "synthetic"}])
def test_provided_malformed_arguments_earn_no_argument_credit(arguments, gold_arguments):
    assert tool_call_args_subset(call(arguments), call(gold_arguments)) == 0.5


@pytest.mark.parametrize("arguments", [{}, "{}"])
def test_valid_empty_arguments_keep_full_credit(arguments):
    assert tool_call_args_subset(call(arguments), call({})) == 1.0


def test_extra_arguments_against_empty_gold_keep_only_name_credit():
    assert tool_call_args_subset(call({"extra": 1}), call({})) == 0.5


def test_valid_expected_subset_keeps_full_credit():
    assert tool_call_args_subset(
        call({"query": "synthetic", "limit": 2}), call({"query": "synthetic"}),
    ) == 1.0


@pytest.mark.parametrize("gold_arguments, score", [({}, 1.0), ({"query": "synthetic"}, 0.5)])
def test_omitted_arguments_keep_the_existing_treatment(gold_arguments, score):
    output = json.dumps({"function": {"name": "lookup"}})
    assert tool_call_args_subset(output, call(gold_arguments)) == score


def test_malformed_gold_handling_is_unchanged():
    assert tool_call_args_subset(call({}), call("not-json")) == 1.0


def test_score_task_keeps_the_half_credit_match_threshold():
    result = score_task(
        EvalTask(prompt="Synthetic question", expected=call({}), scoring="tool_call_args_subset"),
        call("not-json"),
    )
    assert result.score == 0.5
    assert result.matched is True  # The existing >= 0.5 threshold is a separate contract.


def test_invalid_arguments_cannot_add_credit_to_a_wrong_name():
    assert tool_call_args_subset(call("not-json", name="other"), call({})) == 0.0


def test_empty_arguments_with_wrong_name_keep_only_argument_credit():
    assert tool_call_args_subset(call({}, name="other"), call({})) == 0.5
