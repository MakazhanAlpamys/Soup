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


def _output_messages() -> list[dict[str, str]]:
    return [
        {"role": "user", "content": "Response context"},
        {"role": "assistant", "content": "Earlier answer"},
        {"role": "assistant", "content": "Final answer"},
        {"role": "user", "content": "Trailing user is not the answer"},
    ]


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
def test_system_plus_one_user_keeps_plain_prompt_across_ingest(parser, envelope) -> None:
    messages = _messages()[:2] + [{"role": "assistant", "content": "Trailing answer"}]
    source = {"messages": messages} if envelope else messages
    event = {"input": source, "inputs": source, "output": "A1",
             "outputs": {"generations": [[{"text": "A1"}]]}}
    trace = next(parser([event]))
    assert trace.prompt == "Q1"
    assert trace.output == "A1"


def test_system_plus_one_user_keeps_plain_prompt_across_trace_sources() -> None:
    messages = _messages()[:2] + [{"role": "assistant", "content": "Trailing answer"}]
    traces = [
        next(parse_openai([_event(messages, output="A1")])),
        next(parse_langchain([{"inputs": {"messages": messages}, "outputs": {"output": "A1"}}])),
        next(parse_helicone([{"request": {"body": {"messages": messages}},
                              "response": {"body": {"choices": [{"message": {
                                  "content": "A1"}}]}}}])),
        next(parse_openpipe([{"messages": messages, "response": "A1"}])),
    ]
    assert all(trace.prompt == "Q1" and trace.output == "A1" for trace in traces)


@pytest.mark.parametrize("role", ["user", "assistant"])
def test_either_earlier_conversation_role_requires_json(role: str) -> None:
    messages = [{"role": "system", "content": "Be helpful."},
                {"role": role, "content": "Earlier turn"},
                {"role": "user", "content": "Question"}]
    assert json.loads(next(parse_openai([_event(messages)])).prompt) == messages


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


def test_langchain_bare_list_input_preserves_context_through_last_user() -> None:
    messages = _messages() + [{"role": "assistant", "content": "Trailing answer"}]
    trace = next(parse_langchain([{"inputs": messages, "outputs": {"output": "A2"}}]))
    assert trace.prompt == (
        '[{"role":"system","content":"Be helpful."},{"role":"user","content":"Q1"},'
        '{"role":"assistant","content":"A1"},{"role":"user","content":"Q2"}]'
    )
    assert trace.output == "A2"


@pytest.mark.parametrize("envelope", [False, True])
@pytest.mark.parametrize("multi_turn", [False, True])
def test_langchain_type_keyed_messages_follow_conversation_rule(envelope, multi_turn) -> None:
    expected = _messages() if multi_turn else _messages()[:2]
    types = {"system": "system", "user": "human", "assistant": "ai"}
    messages = [{"type": types[turn["role"]], "content": turn["content"]} for turn in expected]
    messages.append({"type": "ai", "content": "Trailing answer"})
    inputs = {"messages": messages} if envelope else messages
    traces = list(parse_langchain([{"inputs": inputs, "outputs": {"output": "Answer"}}]))
    assert len(traces) == 1
    assert traces[0].output == "Answer"
    if multi_turn:
        assert json.loads(traces[0].prompt) == expected
    else:
        assert traces[0].prompt == "Q1"
    assert _coerce_str(inputs) == traces[0].prompt


def test_type_keyed_output_messages_select_last_assistant() -> None:
    messages = [{"type": "ai", "content": "Earlier answer"},
                {"type": "ai", "content": "Final answer"},
                {"type": "human", "content": "Trailing user"}]
    assert _coerce_str(messages, prompt=False) == "Final answer"


def test_explicit_role_takes_precedence_over_langchain_type() -> None:
    messages = [{"role": "user", "type": "ai", "content": "Question"}]
    assert _coerce_str(messages) == "Question"


@pytest.mark.parametrize("invalid_type", [None, [], {}])
def test_invalid_message_type_is_ignored(invalid_type) -> None:
    messages = [{"type": invalid_type, "content": "Ignored"},
                {"role": "user", "content": "Question"}]
    assert _coerce_str(messages) == "Question"


@pytest.mark.parametrize("invalid_role", [None, [], {}])
def test_invalid_role_message_does_not_become_a_content_part(invalid_role) -> None:
    assert _coerce_str([{"role": invalid_role, "text": "Not message content"}]) == ""


@pytest.mark.parametrize("parser", [parse_openai, parse_langchain])
@pytest.mark.parametrize("role", ["system", "user", "assistant"])
@pytest.mark.parametrize("blank", ["", " \t\n "])
def test_blank_context_turn_is_omitted_from_multi_turn_json(parser, role, blank) -> None:
    messages = _messages()
    messages.insert(2, {"role": role, "content": blank})
    event = (_event(messages) if parser is parse_openai else
             {"inputs": {"messages": messages}, "outputs": {"output": "A2"}})
    assert json.loads(next(parser([event])).prompt) == _messages()


@pytest.mark.parametrize("output_key", ["output", "response"])
def test_langchain_output_keys_flatten_content_parts(output_key: str) -> None:
    parts = [{"type": "text", "text": "First line"},
             {"type": "output_text", "text": "Second line"},
             {"type": "image_url", "image_url": {"url": "data:image/png;base64,SECRET"}}]
    trace = next(parse_langchain([{"inputs": [{"role": "user", "content": "Question"}],
                                   "outputs": {output_key: parts}}]))
    assert trace.prompt == "Question"
    assert trace.output == "First line\nSecond line"


@pytest.mark.parametrize("output_path", ["response", "body", "text"])
def test_helicone_message_output_fallbacks_choose_final_assistant(output_path: str) -> None:
    messages = _output_messages()
    responses = {
        "response": messages,
        "body": {"body": messages},
        "text": {"body": {"choices": [{"text": messages}]}},
    }
    event = {"request": {"body": {"messages": [{"role": "user", "content": "Question"}]}},
             "response": responses[output_path]}
    trace = next(parse_helicone([event]))
    assert trace.prompt == "Question"
    assert trace.output == "Final answer"


def test_helicone_message_content_flattens_content_parts() -> None:
    parts = [{"type": "text", "text": "First line"},
             {"type": "output_text", "text": "Second line"},
             {"type": "image_url", "image_url": {"url": "data:image/png;base64,SECRET"}}]
    event = {"request": {"body": {"messages": [{"role": "user", "content": "Question"}]}},
             "response": {"body": {"choices": [{"message": {"content": parts}}]}}}
    trace = next(parse_helicone([event]))
    assert trace.prompt == "Question"
    assert trace.output == "First line\nSecond line"


@pytest.mark.parametrize("content_key", ["text", "content"])
def test_langsmith_generations_flatten_content_parts(content_key: str) -> None:
    parts = [{"type": "text", "text": "First line"},
             {"type": "output_text", "text": "Second line"},
             {"type": "image_url", "image_url": {"url": "data:image/png;base64,SECRET"}}]
    event = {"inputs": {"messages": [{"role": "user", "content": "Question"}]},
             "outputs": {"generations": [[{content_key: parts}]]}}
    trace = next(parse_langsmith([event]))
    assert trace.prompt == "Question"
    assert trace.output == "First line\nSecond line"


def test_langsmith_output_fallback_selects_final_assistant() -> None:
    event = {"inputs": {"messages": [{"role": "user", "content": "Question"}]},
             "outputs": {"messages": _output_messages()}}
    trace = next(parse_langsmith([event]))
    assert trace.prompt == "Question"
    assert trace.output == "Final answer"


def test_output_message_lists_select_final_assistant_before_trailing_user() -> None:
    event = {"input": [{"role": "user", "content": "Question"}],
             "output": _output_messages()}
    trace = next(parse_openai_stored([event]))
    assert trace.prompt == "Question"
    assert trace.output == "Final answer"


def test_conversation_json_keeps_non_ascii_text_literal() -> None:
    messages = [{"role": "system", "content": "Қазақша жауап бер."},
                {"role": "user", "content": "Алдыңғы сұрақ"},
                {"role": "assistant", "content": "Алдыңғы жауап"},
                {"role": "user", "content": "Сәлем 👋"}]
    trace = next(parse_openai([_event(messages, output="Жауап")]))
    assert trace.prompt == (
        '[{"role":"system","content":"Қазақша жауап бер."},'
        '{"role":"user","content":"Алдыңғы сұрақ"},'
        '{"role":"assistant","content":"Алдыңғы жауап"},'
        '{"role":"user","content":"Сәлем 👋"}]'
    )
    assert trace.output == "Жауап"


def test_text_parts_are_flattened_and_binary_parts_dropped_everywhere() -> None:
    parts = [{"type": "text", "text": "Hi"},
             {"type": "image_url", "text": "Not a text part",
              "image_url": {"url": "data:image/png;base64,SECRET"}}]
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
