"""Tool-call scan text contains arguments, while call names stay excluded."""

import json

import pytest

from soup_cli.utils.row_text import extract_row_parts


@pytest.mark.parametrize("location", ["top-level", "message"])
@pytest.mark.parametrize("shape", ["function", "call"])
@pytest.mark.parametrize("arguments", ['{"id": 7}', {"id": 7}, ["seven", 7]])
def test_tool_call_parts_are_arguments_only(location: str, shape: str, arguments: object) -> None:
    name = "CALL_NAME_MUST_NOT_ENTER_SCAN"
    call = {"name": name, "arguments": arguments}
    if shape == "function":
        call = {"function": call}
    row = {"tool_calls": [call]}
    if location == "message":
        row = {"messages": [{"role": "assistant", "tool_calls": [call]}]}

    parts = extract_row_parts(row)
    expected = arguments if isinstance(arguments, str) else json.dumps(arguments)
    assert parts == [expected]
    assert all(name not in part for part in parts)


@pytest.mark.parametrize("location", ["top-level", "message"])
@pytest.mark.parametrize("shape", ["function", "call"])
def test_a_call_name_without_arguments_produces_no_parts(location: str, shape: str) -> None:
    call = {"name": "CALL_NAME_MUST_NOT_ENTER_SCAN"}
    if shape == "function":
        call = {"function": call}
    row = {"tool_calls": [call]}
    if location == "message":
        row = {"messages": [{"role": "assistant", "tool_calls": [call]}]}
    assert extract_row_parts(row) == []
