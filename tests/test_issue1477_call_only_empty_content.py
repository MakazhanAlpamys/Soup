"""#1477: `soup data clean` keeps a call-only assistant turn whose content is empty.

A call-only turn carries its payload in ``tool_calls``. The unified tool-calling
format writes ``"content": ""`` on it, and the loader accepts the row, but the
``--min-tokens`` check in the ``chatml`` branch of ``clean_row`` ran on every
assistant turn and dropped it as "Empty / Whitespace Turns". The same turn with
``"content": null`` was kept. Turns with no calls (``tool_calls`` missing, null
or ``[]``) keep the ``--min-tokens`` rule.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.data.formats import format_to_messages
from soup_cli.utils.data_clean import clean_row

CALL = {
    "id": "c1",
    "type": "function",
    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
}


def _row(content, tool_calls):
    """The issue's trajectory: a call turn, its tool result, then the answer."""
    turn = {"role": "assistant", "content": content}
    if tool_calls is not ...:
        turn["tool_calls"] = tool_calls
    return {
        "messages": [
            {"role": "user", "content": "Weather in Paris?"},
            turn,
            {"role": "tool", "tool_call_id": "c1", "content": '{"temp": 18}'},
            {"role": "assistant", "content": "It is 18C in Paris."},
        ]
    }


def _call_turn(cleaned):
    return cleaned["messages"][1]


class TestChatmlBranch:
    """Nested-call rows with no ``tools`` key are detected as ``chatml``."""

    @pytest.mark.parametrize("content", ["", "   ", None], ids=["empty", "whitespace", "null"])
    def test_a_call_only_turn_is_kept(self, content):
        row = _row(content, [CALL])
        cleaned, rules = clean_row(row, "chatml")
        assert cleaned is not None, rules
        assert _call_turn(cleaned)["tool_calls"] == [CALL]
        assert "Empty / Whitespace Turns" not in rules

    def test_empty_content_is_left_as_written(self):
        cleaned, _ = clean_row(_row("", [CALL]), "chatml")
        assert _call_turn(cleaned)["content"] == ""

    def test_a_turn_with_calls_is_not_held_to_min_tokens(self):
        cleaned, rules = clean_row(_row("ok", [CALL]), "chatml", min_tokens=5)
        assert cleaned is not None, rules

    @pytest.mark.parametrize(
        "tool_calls",
        [[], None, ..., CALL],
        ids=["empty-list", "null", "missing", "bare-dict-not-a-list"],
    )
    def test_an_empty_turn_without_calls_is_still_dropped(self, tool_calls):
        """A bare dict is not a call list: the loader rejects it too."""
        cleaned, rules = clean_row(_row("", tool_calls), "chatml")
        assert cleaned is None
        assert rules == ["Empty / Whitespace Turns"]

    def test_a_short_turn_without_calls_is_still_held_to_min_tokens(self):
        cleaned, rules = clean_row(_row("ok", ...), "chatml", min_tokens=5)
        assert cleaned is None
        assert rules == ["Empty / Whitespace Turns"]

    def test_null_content_with_an_empty_list_is_kept_as_on_main(self):
        """``--min-tokens`` reads string content only; this is unchanged."""
        cleaned, rules = clean_row(_row(None, []), "chatml")
        assert cleaned is not None, rules


class TestToolCallingBranch:
    """``{messages, tools}`` rows are detected as ``tool-calling`` and never dropped
    for an empty turn; pinned so the two branches keep agreeing."""

    @pytest.mark.parametrize("content", ["", None], ids=["empty", "null"])
    @pytest.mark.parametrize("tool_calls", [[CALL], []], ids=["one-call", "empty-list"])
    def test_the_row_is_kept(self, content, tool_calls):
        row = {
            **_row(content, tool_calls),
            "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
        }
        cleaned, rules = clean_row(row, "tool-calling")
        assert cleaned is not None, rules
        assert _call_turn(cleaned)["content"] == content
        assert _call_turn(cleaned)["tool_calls"] == tool_calls


@pytest.mark.parametrize("content", ["", None], ids=["empty", "null"])
def test_every_kept_call_row_loads(content):
    """What clean keeps, the loader accepts: the issue's premise, pinned."""
    row = _row(content, [CALL])
    cleaned, _ = clean_row(row, "chatml")
    assert cleaned is not None
    assert format_to_messages(cleaned, "tool-calling") is not None


def test_cli_keeps_the_call_row(tmp_path, monkeypatch):
    """The issue's CLI run: the ``""`` row plus one plain chat row, no flags."""
    monkeypatch.chdir(tmp_path)
    plain = {
        "messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
    }
    rows = [_row("", [CALL]), plain]
    Path("raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    result = CliRunner().invoke(app, ["data", "clean", "raw.jsonl", "-o", "clean.jsonl", "-f"])
    assert result.exit_code == 0, result.output
    out = [json.loads(line) for line in Path("clean.jsonl").read_text().splitlines()]
    assert out == rows
