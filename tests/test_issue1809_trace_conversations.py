"""Trace and ingest prompts retain conversation identity without binary parts."""

import json

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.data.traces import build_pairs, parse_langchain, parse_openai
from soup_cli.utils.ingest_sources import (
    _coerce_str,
    parse_helicone,
    parse_langfuse,
    parse_langsmith,
    parse_openai_stored,
    parse_openpipe,
)


def _messages(question="Q2"):
    return [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": question},
    ]


def _event(messages, rating="up", output="A2"):
    return {"messages": messages, "choices": [{"message": {"content": output}}],
            "feedback": {"rating": rating}}


def test_two_turns_cannot_form_a_false_preference_pair() -> None:
    first = _event(_messages()[:2], "down", "A1")
    second = _event(_messages())
    traces = list(parse_openai([first, second]))
    assert len(traces) == 2
    assert traces[0].prompt != traces[1].prompt
    assert list(build_pairs(traces, signal="thumbs_up")) == []


def test_shared_opening_does_not_merge_different_conversations() -> None:
    traces = list(parse_openai([_event(_messages("2+2?"), "down"),
                                _event(_messages("Capital of France?"))]))
    assert len({trace.prompt for trace in traces}) == 2
    assert list(build_pairs(traces, signal="thumbs_up")) == []


def test_context_is_role_preserving_and_stops_at_last_user() -> None:
    messages = _messages()
    request = messages + [{"role": "assistant", "content": "leak"}]
    prompt = next(parse_openai([_event(request)])).prompt
    assert json.loads(prompt) == messages
    assert "leak" not in prompt


@pytest.mark.parametrize("parser", [parse_langfuse, parse_langsmith, parse_openai_stored])
@pytest.mark.parametrize("envelope", [False, True])
def test_ingest_parsers_share_prompt_rule(parser, envelope) -> None:
    messages = _messages()
    source = {"messages": messages} if envelope else messages
    event = {"input": source, "inputs": source, "output": [{"role": "assistant", "content": "A2"}],
             "outputs": {"generations": [[{"text": "A2"}]]}}
    trace = next(parser([event]))
    assert trace.prompt == next(parse_openai([_event(messages)])).prompt
    assert trace.output == "A2"


@pytest.mark.parametrize("envelope", [None, "messages", "input", "content", "prompt", "output"])
def test_coercion_uses_same_rule_in_every_list_branch(envelope) -> None:
    messages = _messages()
    value = {envelope: messages} if envelope else messages
    assert _coerce_str(value) == next(parse_openai([_event(messages)])).prompt


def test_langchain_and_other_ingest_sources_share_conversation_identity() -> None:
    messages = _messages()
    expected = next(parse_openai([_event(messages)])).prompt
    lc = next(parse_langchain([{"inputs": {"messages": messages},
                               "outputs": {"generations": [[{"text": "A2"}]]}}]))
    helicone = next(parse_helicone([{"request": {"body": {"messages": messages}},
                                     "response": {"body": {"choices": [{"message": {
                                         "content": "A2"}}]}}}]))
    openpipe = next(parse_openpipe([{"messages": messages, "response": "A2"}]))
    assert lc.prompt == helicone.prompt == openpipe.prompt == expected


def test_text_parts_are_flattened_and_binary_parts_dropped_everywhere() -> None:
    parts = [{"type": "text", "text": "Hi"},
             {"type": "image_url", "image_url": {"url": "data:image/png;base64,SECRET"}}]
    messages = [{"role": "user", "content": parts}]
    oai = next(parse_openai([{"messages": messages, "choices": [{"message": {"content": parts}}]}]))
    lc = next(parse_langchain([{"inputs": {"messages": messages},
                               "outputs": {"generations": [[{"text": parts}]]}}]))
    stored = next(parse_openai_stored([{"input": messages,
                                       "output": [{"role": "assistant", "content": parts}]}]))
    for trace in (oai, lc, stored):
        assert trace.prompt == "Hi"
        assert trace.output == "Hi"
    assert _coerce_str(parts) == "Hi"


@pytest.mark.parametrize("messages", [[], [{"role": "system", "content": "System only"}],
                                     [{"role": "user", "content": [{"type": "image_url"}]}]])
def test_no_text_user_turn_does_not_become_a_training_prompt(messages) -> None:
    assert list(parse_openai([_event(messages)])) == []
    assert _coerce_str(messages) == ""


def test_same_conversation_still_forms_a_real_preference_pair() -> None:
    traces = list(parse_openai([_event(_messages(), "down", "Bad"),
                                _event(_messages(), "up", "Good")]))
    pairs = list(build_pairs(traces, signal="thumbs_up"))
    assert len(pairs) == 1
    assert pairs[0].prompt == traces[0].prompt
    assert (pairs[0].chosen, pairs[0].rejected) == ("Good", "Bad")


def test_ingest_cli_preserves_the_user_question(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    event = {"input": _messages(), "output": [{"role": "assistant", "content": "A2"}]}
    (tmp_path / "logs.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
    result = CliRunner().invoke(app, ["ingest", "--source", "openai-stored", "--logs", "logs.jsonl",
                                      "--output", "traces.jsonl"])
    assert result.exit_code == 0, result.output
    row = json.loads((tmp_path / "traces.jsonl").read_text(encoding="utf-8"))
    assert json.loads(row["prompt"]) == _messages()
    assert row["output"] == "A2"
