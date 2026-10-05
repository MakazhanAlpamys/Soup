"""#1338: `soup data clean` repairs and checks every tool call the loader reads.

The tool-calling loader reads calls in the documented ``function`` shape, both at
the row's top level and inside each assistant turn (#1217). ``clean_row`` used to
read only a flat ``arguments`` key on the top-level list, so nested calls were
never repaired or checked, and a valid documented-shape row had its missing flat
``arguments`` parsed as ``""`` and was dropped.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.data.formats import format_to_messages
from soup_cli.utils.data_clean import clean_dataset, clean_row

BROKEN = '```json\n{"city": "Paris",}\n```'  # fenced, trailing comma
REPAIRED = '{"city": "Paris"}'
UNREPAIRABLE = '{"city": "Paris"'
TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]


def _call(args):
    return {
        "id": "call_0",
        "type": "function",
        "function": {"name": "get_weather", "arguments": args},
    }


def _nested_row(args):
    """A call inside the assistant turn, no top-level keys (``soup agent synth``)."""
    return {
        "messages": [
            {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "tool_calls": [_call(args)]},
        ]
    }


def _openai_row(args):
    """The OpenAI fine-tuning shape."""
    return {
        "messages": [
            {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "content": None, "tool_calls": [_call(args)]},
            {"role": "tool", "tool_call_id": "call_0", "content": "21"},
            {"role": "assistant", "content": "21 C."},
        ],
        "tools": TOOLS,
    }


def _top_level_row(args):
    """The documented shape, calls at the row's top level."""
    return {
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
        "tools": TOOLS,
        "tool_calls": [_call(args)],
    }


def _flat_row(args):
    """The flat shape the existing tests use: the control."""
    return {
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
        "tool_calls": [{"name": "get_weather", "arguments": args}],
    }


def _all_args(row):
    """Every call's arguments in the row, nested first, in order."""
    out = []
    for msg in row.get("messages", []):
        for c in msg.get("tool_calls") or []:
            out.append(c["function"]["arguments"])
    for c in row.get("tool_calls", []):
        out.append(c["function"]["arguments"] if "function" in c else c["arguments"])
    return out


SHAPES = {
    "nested": (_nested_row, "chatml"),
    "openai": (_openai_row, "tool-calling"),
    "top_level": (_top_level_row, "tool-calling"),
    "flat": (_flat_row, "tool-calling"),
}


@pytest.mark.parametrize("shape", list(SHAPES))
class TestRepairJson:
    def test_repairs_the_call(self, shape):
        make, fmt = SHAPES[shape]
        cleaned, rules = clean_row(make(BROKEN), fmt, repair_json=True)
        assert cleaned is not None
        assert _all_args(cleaned) == [REPAIRED]
        assert "Malformed JSON in Tools" in rules

    def test_leaves_the_call_alone_without_the_flag(self, shape):
        make, fmt = SHAPES[shape]
        cleaned, rules = clean_row(make(BROKEN), fmt)
        assert _all_args(cleaned) == [BROKEN]
        assert "Malformed JSON in Tools" not in rules

    def test_keeps_the_other_call_keys(self, shape):
        if shape == "flat":
            pytest.skip("the flat call has no function dict")
        make, fmt = SHAPES[shape]
        cleaned, _ = clean_row(make(BROKEN), fmt, repair_json=True)
        calls = [c for m in cleaned["messages"] for c in (m.get("tool_calls") or [])]
        calls += cleaned.get("tool_calls", [])
        assert calls == [_call(REPAIRED)]

    def test_does_not_mutate_the_input(self, shape):
        make, fmt = SHAPES[shape]
        row = make(BROKEN)
        before = json.dumps(row, sort_keys=True)
        clean_row(row, fmt, repair_json=True)
        assert json.dumps(row, sort_keys=True) == before


@pytest.mark.parametrize("shape", list(SHAPES))
class TestDropInvalidJson:
    def test_keeps_a_valid_row(self, shape):
        make, fmt = SHAPES[shape]
        cleaned, rules = clean_row(make(REPAIRED), fmt, drop_invalid_json=True)
        assert cleaned is not None, rules
        assert _all_args(cleaned) == [REPAIRED]

    def test_drops_an_invalid_row(self, shape):
        make, fmt = SHAPES[shape]
        cleaned, rules = clean_row(make(UNREPAIRABLE), fmt, drop_invalid_json=True)
        assert cleaned is None
        assert rules == ["Invalid JSON in Tool Calls"]

    def test_repair_runs_before_the_check(self, shape):
        make, fmt = SHAPES[shape]
        cleaned, _ = clean_row(make(BROKEN), fmt, repair_json=True, drop_invalid_json=True)
        assert cleaned is not None
        assert _all_args(cleaned) == [REPAIRED]

    def test_drops_what_repair_cannot_fix(self, shape):
        make, fmt = SHAPES[shape]
        cleaned, rules = clean_row(
            make(UNREPAIRABLE), fmt, repair_json=True, drop_invalid_json=True
        )
        assert cleaned is None
        assert rules == ["Invalid JSON in Tool Calls"]


@pytest.mark.parametrize("shape", ["nested", "openai", "top_level"])
def test_every_row_kept_by_drop_invalid_json_loads(shape):
    """The rows ``--drop-invalid-json`` keeps are the rows the loader accepts."""
    make, fmt = SHAPES[shape]
    for args in (REPAIRED, UNREPAIRABLE):
        row = make(args)
        cleaned, _ = clean_row(row, fmt, drop_invalid_json=True)
        loads = format_to_messages(row, "tool-calling") is not None
        assert (cleaned is not None) == loads


def test_dict_arguments_are_left_alone():
    row = _top_level_row({"city": "Paris"})
    cleaned, rules = clean_row(row, "tool-calling", repair_json=True, drop_invalid_json=True)
    assert cleaned == row
    assert rules == []


def test_a_function_call_without_arguments_is_kept():
    """The loader reads a missing ``arguments`` as ``{}``."""
    row = _top_level_row("{}")
    del row["tool_calls"][0]["function"]["arguments"]
    assert format_to_messages(row, "tool-calling") is not None
    cleaned, _ = clean_row(row, "tool-calling", drop_invalid_json=True)
    assert cleaned == row


def test_a_later_invalid_nested_call_drops_the_row():
    row = _openai_row(REPAIRED)
    row["messages"].append({"role": "assistant", "tool_calls": [_call(UNREPAIRABLE)]})
    cleaned, rules = clean_row(row, "tool-calling", drop_invalid_json=True)
    assert cleaned is None
    assert rules == ["Invalid JSON in Tool Calls"]


def test_calls_outside_an_assistant_turn_are_not_touched():
    """The loader only reads calls on assistant turns."""
    row = _nested_row(REPAIRED)
    row["messages"][0]["tool_calls"] = [_call(UNREPAIRABLE)]
    cleaned, _ = clean_row(row, "chatml", drop_invalid_json=True)
    assert cleaned == row


def test_a_nested_call_row_gains_no_top_level_tool_calls():
    row = _openai_row(REPAIRED)
    cleaned, rules = clean_row(row, "tool-calling")
    assert "tool_calls" not in cleaned
    assert cleaned == row
    assert rules == []


@pytest.mark.parametrize(
    "flags",
    [{}, {"repair_json": True}, {"drop_invalid_json": True}],
    ids=["none", "repair", "drop"],
)
def test_a_null_top_level_tool_calls_is_kept(flags):
    """The loader reads ``"tool_calls": null`` as no calls (``top_level_calls is None``)."""
    row = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
        "tools": TOOLS,
        "tool_calls": None,
    }
    assert format_to_messages(row, "tool-calling") is not None
    cleaned, rules = clean_row(row, "tool-calling", **flags)
    assert cleaned == row, rules
    assert rules == []


def test_nested_control_chars_are_sanitised():
    row = _nested_row('{"city": "Pa​ris"}')
    cleaned, rules = clean_row(row, "chatml")
    assert _all_args(cleaned) == [REPAIRED]
    assert "Invisible & Control Chars" in rules


def test_clean_dataset_counts_the_nested_repair():
    rows = [_nested_row(BROKEN), _nested_row(REPAIRED)]
    cleaned, report = clean_dataset(rows, "chatml", repair_json=True)
    assert [_all_args(r) for r in cleaned] == [[REPAIRED], [REPAIRED]]
    assert report.total_modified == 1
    assert report.total_clean == 1


def test_the_issue_repro_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rows = [_nested_row(BROKEN), _openai_row(BROKEN), _top_level_row(BROKEN)]
    Path("raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(
        app, ["data", "clean", "raw.jsonl", "-o", "clean.jsonl", "-f", "--repair-json"]
    )
    assert result.exit_code == 0, result.output
    out = [json.loads(line) for line in Path("clean.jsonl").read_text().splitlines()]
    assert [_all_args(r) for r in out] == [[REPAIRED]] * 3
    for row in out:
        assert format_to_messages(row, "tool-calling") is not None

    Path("raw.jsonl").write_text(json.dumps(_top_level_row(REPAIRED)) + "\n", encoding="utf-8")
    result = runner.invoke(
        app, ["data", "clean", "raw.jsonl", "-o", "clean.jsonl", "-f", "--drop-invalid-json"]
    )
    assert result.exit_code == 0, result.output
    assert len(Path("clean.jsonl").read_text().splitlines()) == 1
