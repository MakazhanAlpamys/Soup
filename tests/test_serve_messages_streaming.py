"""``/v1/messages`` with ``"stream": true`` sends events as the model generates (#2515).

Before, the route ran the whole non-streamed ``chat_completions`` first and only then cut the
finished text into one ``content_block_delta`` per word, so an Anthropic client (Claude Code,
the SDKs) saw nothing until the reply was complete. The events now come from the same
generator ``/v1/chat/completions`` streams with, in the order the SDKs expect:
``message_start``, ``content_block_start``, a ``content_block_delta`` per piece of text,
``content_block_stop``, ``message_delta`` (``stop_reason``, ``stop_sequence`` and the output
token count, which are only known at the end) and ``message_stop``.

``stop_sequences`` (#2549) ends a stream with ``stop_reason`` ``stop_sequence`` and
text after the match never leaves the server; ``test_serve_stop_strings`` holds the rest.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from soup_cli.commands import serve  # noqa: E402
from tests.test_serve_token_streaming import (  # noqa: E402
    PROMPT_IDS,
    _drain,
    _PieceTokenizer,
    _script,
    _ScriptedModel,
    _stream,
)

URL = "/v1/messages"
ABSENT = object()
SEQUENCE = [
    "message_start",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
    "message_delta",
    "message_stop",
]


def _client(model, tokenizer, **kwargs):
    app = serve._create_app(model, tokenizer, "cpu", "m", 64, **kwargs)
    return TestClient(app, base_url="http://localhost")


def _post(client, *, stop=ABSENT, stream=False, **extra):
    body = {
        "model": "m",
        "max_tokens": 64,
        "temperature": 0,
        "messages": [{"role": "user", "content": "hi"}],
        **extra,
    }
    if stop is not ABSENT:
        body["stop_sequences"] = stop
    if stream:
        body["stream"] = True
    return client.post(URL, json=body)


def _events(text):
    """``[(event name, data)]`` of an SSE body: ``event:`` line, ``data:`` line, blank line."""
    events = []
    for block in text.split("\n\n"):
        if not block:
            continue
        lines = block.split("\n")
        assert len(lines) == 2, repr(block)
        assert lines[0].startswith("event: ") and lines[1].startswith("data: "), repr(block)
        name = lines[0][len("event: "):]
        data = json.loads(lines[1][len("data: "):])
        assert data["type"] == name
        events.append((name, data))
    return events


def _names(events):
    """The event names with runs of ``content_block_delta`` folded into one."""
    names = []
    for name, _ in events:
        if not (names and names[-1] == name == "content_block_delta"):
            names.append(name)
    return names


def _text(events):
    return "".join(
        data["delta"]["text"] for name, data in events if name == "content_block_delta"
    )


def _final(events):
    """``(stop_reason, stop_sequence, output_tokens)`` of the stream's ``message_delta``."""
    deltas = [data for name, data in events if name == "message_delta"]
    assert len(deltas) == 1
    return (
        deltas[0]["delta"]["stop_reason"],
        deltas[0]["delta"]["stop_sequence"],
        deltas[0]["usage"]["output_tokens"],
    )


def _plain(client, **kwargs):
    response = _post(client, **kwargs)
    assert response.status_code == 200, response.text
    return response.json()


# --------------------------------------------------------------------------- #
# The events, in order, with the shape the SDKs read
# --------------------------------------------------------------------------- #


def test_event_sequence_and_shape():
    tokenizer, ids = _script("Hello", " brave", " new", " world")
    client = _client(_ScriptedModel(ids), tokenizer)

    plain = _plain(client)
    response = _post(client, stream=True)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-accel-buffering"] == "no"
    events = _events(response.text)
    assert _names(events) == SEQUENCE
    assert events[0][0] == "message_start" and events[1][0] == "content_block_start"
    assert [name for name, _ in events[-3:]] == [
        "content_block_stop", "message_delta", "message_stop",
    ]

    message = events[0][1]["message"]
    assert message["id"].startswith("chatcmpl-")
    assert message["type"] == "message" and message["role"] == "assistant"
    assert message["content"] == [] and message["model"] == "m"
    assert message["stop_reason"] is None and message["stop_sequence"] is None
    assert message["usage"] == {"input_tokens": plain["usage"]["input_tokens"], "output_tokens": 0}
    assert plain["usage"]["input_tokens"] == len(PROMPT_IDS)
    assert events[1][1] == {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }
    for name, data in events:
        if name == "content_block_delta":
            assert data["index"] == 0 and data["delta"]["type"] == "text_delta"
    assert events[-3][1] == {"type": "content_block_stop", "index": 0}
    assert _text(events) == plain["content"][0]["text"] == "Hello brave new world"
    assert _final(events) == ("end_turn", None, plain["usage"]["output_tokens"])


def test_first_events_are_the_message_start_before_any_text():
    tokenizer, ids = _script("Hello", " world")
    model = _ScriptedModel(ids, hold_after=1)
    stream = _stream(model, tokenizer, frames=serve._AnthropicStreamFrames("m"))

    first = next(stream)
    second = next(stream)

    assert first.startswith("event: message_start\n")
    assert second.startswith("event: content_block_start\n")
    assert not model.generate_returned.is_set()
    model.gate.set()
    _drain(stream)


def test_a_delta_leaves_the_server_while_generate_is_still_running():
    tokenizer, ids = _script("Hello", " brave", " new", " world")
    model = _ScriptedModel(ids, hold_after=3)
    stream = _stream(model, tokenizer, frames=serve._AnthropicStreamFrames("m"))

    frames = [next(stream)]
    while not frames[-1].startswith("event: content_block_delta\n"):
        frames.append(next(stream))

    # Pinned on the unfixed code: the whole reply was generated before the first delta.
    assert not model.generate_returned.is_set(), (
        "the first content_block_delta only arrived after generate() had finished"
    )
    model.gate.set()
    frames.extend(_drain(stream))
    events = _events("".join(frames))
    assert _names(events) == SEQUENCE
    assert _text(events) == "Hello brave new world"
    assert not model.gate_timed_out


def test_the_route_streams_from_the_incremental_generator(monkeypatch):
    """Through the real route: the first delta is yielded while ``generate`` is parked."""
    tokenizer, ids = _script("Hello", " brave", " new", " world")
    model = _ScriptedModel(ids, hold_after=3)
    observed = []
    real = serve._stream_response

    def spy(*args, **kwargs):
        for frame in real(*args, **kwargs):
            if frame.startswith("event: content_block_delta\n") and not observed:
                observed.append(model.generate_returned.is_set())
                model.gate.set()
            yield frame

    monkeypatch.setattr(serve, "_stream_response", spy)
    client = _client(model, tokenizer)

    response = _post(client, stream=True)

    assert observed == [False], "the first delta was only sent after generate() had finished"
    assert not model.gate_timed_out
    assert _text(_events(response.text)) == "Hello brave new world"
    from transformers import TextIteratorStreamer

    assert isinstance(model.calls[0]["streamer"], TextIteratorStreamer)
    assert model.threads[0].name == "soup-serve-generate"


def test_stop_reason_is_max_tokens_when_the_budget_ran_out():
    tokenizer, ids = _script("a", " b", " c", eos=False)
    client = _client(_ScriptedModel(ids), tokenizer)

    plain = _plain(client, max_tokens=3)
    events = _events(_post(client, stream=True, max_tokens=3).text)

    assert plain["stop_reason"] == "max_tokens"
    assert _final(events) == ("max_tokens", None, 3)
    assert _text(events) == plain["content"][0]["text"] == "a b c"


def test_an_error_after_the_stream_started_is_an_error_event():
    tokenizer, ids = _script("Hello", " brave", " world")
    client = _client(_ScriptedModel(ids, fail_at=1), tokenizer)

    response = _post(client, stream=True)

    assert response.status_code == 200
    events = _events(response.text)
    names = [name for name, _ in events]
    assert names[:2] == ["message_start", "content_block_start"]
    assert names[-1] == "error"
    assert "message_stop" not in names and "message_delta" not in names
    assert events[-1][1] == {
        "type": "error",
        "error": {"type": "api_error", "message": "Internal server error"},
    }
    assert "boom in generate" not in response.text


@pytest.mark.parametrize(
    "options",
    [
        {"reasoning_parser": "deepseek-r1"},
        {"draft_model": MagicMock()},
    ],
    ids=["reasoning_parser", "draft_model"],
)
def test_requests_generated_in_full_still_send_the_whole_event_sequence(options):
    tokenizer, ids = _script("<think>", "plan", " it</think>", "Hello", " brave", " world")
    model = _ScriptedModel(ids)
    client = _client(model, tokenizer, **options)

    plain = _plain(client)
    events = _events(_post(client, stream=True).text)

    assert "streamer" not in model.calls[-1]
    assert _names(events) == SEQUENCE
    assert _text(events) == plain["content"][0]["text"]
    assert _final(events)[0] == "end_turn"


def test_mole_reply_sends_the_whole_event_sequence():
    tokenizer, ids = _script("unused")
    runtime = MagicMock()
    runtime.generate_text.return_value = ("Blended reply here", 4, 3, "stop")
    client = _client(_ScriptedModel(ids), tokenizer, mole_runtime=runtime)

    events = _events(_post(client, stream=True).text)

    assert _names(events) == SEQUENCE
    assert _text(events) == "Blended reply here"
    assert _final(events) == ("end_turn", None, 3)


def test_model_name_cannot_inject_an_event():
    frames = serve._AnthropicStreamFrames("m\r\nevent: hijack\r\ndata: {}\r\n")

    out = frames.start(3) + [frames.text("hi")] + frames.end("stop", 1, None)

    assert all(frame.count("\n\n") == 1 and frame.endswith("\n\n") for frame in out)
    for frame in out:
        assert [line for line in frame.split("\n") if line.startswith("event: hijack")] == []
    start = json.loads(out[0].split("\ndata: ", 1)[1])
    assert start["message"]["model"] == "mevent: hijackdata: {}"


# --------------------------------------------------------------------------- #
# stop_sequences on a stream
# --------------------------------------------------------------------------- #

_REPLY = ["Hello", " there", "\nUser:", " hi", " again"]


def test_a_stop_sequence_ends_the_stream_with_stop_reason_stop_sequence():
    tokenizer, ids = _script(*_REPLY)
    client = _client(_ScriptedModel(ids), tokenizer)

    events = _events(_post(client, stop=["\nUser:"], stream=True).text)

    assert _names(events) == SEQUENCE
    assert _text(events) == "Hello there"
    assert _final(events)[:2] == ("stop_sequence", "\nUser:")


def test_a_stream_ends_generation_at_the_stop_sequence():
    tokenizer, ids = _script(*_REPLY)
    model = _ScriptedModel(ids, wait_for_stop=True)
    client = _client(model, tokenizer)

    events = _events(_post(client, stop=["\nUser:"], stream=True).text)

    assert model.stopped_early, "generation was never told to stop"
    assert _text(events) == "Hello there"
    assert _final(events)[:2] == ("stop_sequence", "\nUser:")


def test_no_delta_carries_text_after_the_stop_sequence():
    tokenizer, ids = _script(*_REPLY)
    client = _client(_ScriptedModel(ids), tokenizer)

    events = _events(_post(client, stop=["\nUser:"], stream=True).text)

    sent = ""
    for name, data in events:
        if name == "content_block_delta":
            sent += data["delta"]["text"]
            assert "Hello there".startswith(sent), sent


# --------------------------------------------------------------------------- #
# The helpers
# --------------------------------------------------------------------------- #


def test_the_prompt_token_count_is_what_generate_counts():
    tokenizer, _ = _script("Hello")

    assert serve._count_prompt_tokens(tokenizer, [{"role": "user", "content": "hi"}]) == len(
        PROMPT_IDS
    )


def test_a_tokenizer_that_cannot_count_reports_zero():
    class Broken(_PieceTokenizer):
        def __call__(self, *args, **kwargs):
            raise RuntimeError("no")

    assert serve._count_prompt_tokens(Broken({}), [{"role": "user", "content": "hi"}]) == 0
