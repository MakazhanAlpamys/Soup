"""Tool-call text extraction returns arguments only, never call names (#2212)."""

import json

from soup_cli.utils import row_text
from soup_cli.utils.row_text import extract_row_parts


def test_tool_calls_docstring_says_names_are_not_extracted():
    doc = row_text._extract_tool_calls_text.__doc__
    assert "arguments" in doc
    assert "not extracted" in doc


def test_top_level_tool_calls_return_arguments_not_names():
    row = {
        "tool_calls": [
            {"function": {"name": "send_email", "arguments": '{"to": "a@b.c"}'}},
            {"name": "lookup_user", "arguments": {"id": 7}},
            {"function": {"name": "search_docs", "arguments": ["alpha", "beta"]}},
        ]
    }
    parts = extract_row_parts(row)
    assert '{"to": "a@b.c"}' in parts
    assert json.dumps({"id": 7}) in parts
    assert json.dumps(["alpha", "beta"]) in parts
    assert not any(
        name in part
        for part in parts
        for name in ("send_email", "lookup_user", "search_docs")
    )


def test_message_tool_calls_return_arguments_not_names():
    row = {
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "send_email", "arguments": '{"to": "a@b.c"}'}},
                    {"name": "lookup_user", "arguments": {"id": 7}},
                ],
            }
        ]
    }
    parts = extract_row_parts(row)
    assert '{"to": "a@b.c"}' in parts
    assert json.dumps({"id": 7}) in parts
    assert not any(
        name in part for part in parts for name in ("send_email", "lookup_user")
    )
