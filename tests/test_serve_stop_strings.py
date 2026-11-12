"""``soup serve --backend transformers`` honours the OpenAI ``stop`` parameter (#2549).

``ChatCompletionRequest`` had no ``stop`` field, so pydantic dropped it and a client
that sent ``"stop": ["\\nUser:"]`` got the text after the stop string, with
``finish_reason`` set as if no stop had been asked for. Now the reply is cut before the
first stop string, the stop string is not returned, ``finish_reason`` is ``stop``, and
a streamed reply holds back anything that could be the start of a stop string so text
after one never leaves the server. The joined streamed deltas equal the plain reply.

The fake models come from ``test_serve_token_streaming``: they replay chosen text pieces
through ``generate`` the way ``transformers`` does. The last tests run a real ``generate``
on a tiny random Llama.
"""

from __future__ import annotations

import json
import threading
import types
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from soup_cli.commands import serve  # noqa: E402
from tests.test_serve_token_streaming import (  # noqa: E402
    EOS_ID,
    MESSAGES,
    _content,
    _drain,
    _frames,
    _script,
    _ScriptedModel,
    _stream,
    _word_tokenizer,
)

CHAT = "/v1/chat/completions"
ABSENT = object()


def _client(model, tokenizer, max_tokens_default=64, **kwargs):
    app = serve._create_app(model, tokenizer, "cpu", "m", max_tokens_default, **kwargs)
    return TestClient(app, base_url="http://localhost")


def _post(client, *, stop=ABSENT, stream=False, **extra):
    body = {"model": "m", "messages": MESSAGES, "temperature": 0, **extra}
    if stop is not ABSENT:
        body["stop"] = stop
    if stream:
        body["stream"] = True
    return client.post(CHAT, json=body)


def _sse(text):
    return _frames([part + "\n\n" for part in text.split("\n\n") if part])


def _plain(response):
    assert response.status_code == 200, response.text
    choice = response.json()["choices"][0]
    return choice["message"]["content"], choice["finish_reason"]


def _streamed(response):
    assert response.status_code == 200, response.text
    frames = _sse(response.text)
    assert frames[-1] == "[DONE]"
    return _content(frames), frames[-2]["choices"][0]["finish_reason"], frames


# --------------------------------------------------------------------------- #
# The request: every spelling of ``stop``
# --------------------------------------------------------------------------- #

_REPLY = ["Hello", " there", "\nUser:", " hi", " again"]


@pytest.mark.parametrize(
    "stop",
    ["\nUser:", ["\nUser:"], ["nothing here", "\nUser:"], ["\nUser:", "never"]],
    ids=["string", "list_of_one", "list_second_matches", "list_first_matches"],
)
@pytest.mark.parametrize("stream", [False, True], ids=["reply", "stream"])
def test_a_string_or_a_list_cuts_the_reply_before_the_stop_string(stop, stream):
    tokenizer, ids = _script(*_REPLY)
    client = _client(_ScriptedModel(ids), tokenizer)

    response = _post(client, stop=stop, stream=stream)

    if stream:
        text, reason, _ = _streamed(response)
    else:
        text, reason = _plain(response)
    assert text == "Hello there"
    assert reason == "stop"


@pytest.mark.parametrize("stop", [ABSENT, None, []], ids=["absent", "null", "empty_list"])
@pytest.mark.parametrize("stream", [False, True], ids=["reply", "stream"])
def test_no_stop_leaves_the_reply_alone(stop, stream):
    tokenizer, ids = _script(*_REPLY)
    client = _client(_ScriptedModel(ids), tokenizer)

    response = _post(client, stop=stop, stream=stream)

    if stream:
        text, reason, _ = _streamed(response)
    else:
        text, reason = _plain(response)
    assert text == "Hello there\nUser: hi again"
    assert reason == "stop"


_BAD_STOP = [
    ("", "stop must not be an empty string"),
    ([""], "stop[0] must not be an empty string"),
    (["ok", ""], "stop[1] must not be an empty string"),
    (["a", "b", "c", "d", "e"], "stop accepts at most 4 stop strings, got 5"),
    ("x" * 257, "stop is longer than 256 characters"),
    (["x" * 257], "stop[0] is longer than 256 characters"),
    ([1], "stop[0] must be a string"),
    (["a", None], "stop[1] must be a string"),
    (7, "stop must be a string or a list of strings"),
    (True, "stop must be a string or a list of strings"),
    ({"a": "b"}, "stop must be a string or a list of strings"),
]


@pytest.mark.parametrize(("stop", "message"), _BAD_STOP)
@pytest.mark.parametrize("stream", [False, True], ids=["reply", "stream"])
def test_a_bad_stop_is_a_400_that_names_the_field(stop, message, stream):
    tokenizer, ids = _script(*_REPLY)
    model = _ScriptedModel(ids)
    client = _client(model, tokenizer)

    response = _post(client, stop=stop, stream=stream)

    assert response.status_code == 400, response.text
    assert response.json() == {"detail": message}
    assert model.calls == [], "a refused request must not start generating"


def test_a_stop_that_never_matches_keeps_the_reason_the_run_ended_with():
    tokenizer, ids = _script("a", " b", " c", eos=False)
    client = _client(_ScriptedModel(ids), tokenizer)

    for stream in (False, True):
        response = _post(client, stop="zzz", stream=stream, max_tokens=3)
        if stream:
            text, reason, _ = _streamed(response)
        else:
            text, reason = _plain(response)
        assert text == "a b c"
        assert reason == "length"


@pytest.mark.parametrize("stream", [False, True], ids=["reply", "stream"])
def test_a_match_is_stop_even_when_the_run_also_used_its_budget(stream):
    tokenizer, ids = _script("a", " b", " STOP", " c", eos=False)
    client = _client(_ScriptedModel(ids), tokenizer)

    response = _post(client, stop="STOP", stream=stream, max_tokens=4)

    if stream:
        text, reason, _ = _streamed(response)
    else:
        text, reason = _plain(response)
    assert text == "a b"
    assert reason == "stop"


# --------------------------------------------------------------------------- #
# The streamed reply equals the plain reply for the same stop
# --------------------------------------------------------------------------- #

_CASES = {
    "string": (_REPLY, "\nUser:", "Hello there"),
    "split_across_tokens": (["Hello", "\nUs", "er:", " hi"], "\nUser:", "Hello"),
    "match_at_position_zero": (["END", " rest"], "END", ""),
    "stop_is_the_whole_reply": (["END"], "END", ""),
    "second_of_two_stops": (["a", " b", " END", " c", " STOP"], ["STOP", "END"], "a b"),
    "overlapping_stops": (["x", "ab", "c", "d"], ["abcd", "bc"], "xa"),
    "overlapping_stops_other_order": (["x", "ab", "c", "d"], ["bc", "abcd"], "xa"),
    "no_match_sends_the_held_tail": (["Hello", " world", " STO"], "STOP", "Hello world STO"),
    "partial_match_then_diverges": (["Hello", " ST", "ART now"], "STOP", "Hello START now"),
    "whitespace_before_the_stop_is_dropped": (
        ["Hello", " world ", "\n", "User:", " x"], "User:", "Hello world",
    ),
    "leading_whitespace": (["  ", "Hello", " END"], "END", "Hello"),
    "unicode": (["日本", "語", "END", "続き"], "END", "日本語"),
    "one_character_stop": (["Hello", " world", ".", " More"], ".", "Hello world"),
    "newline_stop": (["Line one", "\n", "line two"], "\n", "Line one"),
    "stop_longer_than_the_reply": (["Hello", " world"], "Hello world and more", "Hello world"),
    "four_stops": (["a", " b", " c", " d", " e"], ["x", "y", " d", "z"], "a b c"),
}


@pytest.mark.parametrize("name", sorted(_CASES))
def test_streamed_deltas_equal_the_plain_reply(name):
    pieces, stop, expected = _CASES[name]
    tokenizer, ids = _script(*pieces)
    writer = MagicMock()
    client = _client(_ScriptedModel(ids), tokenizer, trace_log_writer=writer)

    plain_text, plain_reason = _plain(_post(client, stop=stop))
    text, reason, frames = _streamed(_post(client, stop=stop, stream=True))

    assert plain_text == expected
    assert text == plain_text
    assert reason == plain_reason
    if expected != "".join(pieces).strip():
        assert plain_reason == "stop"
    # The request log carries the cut reply, for the plain call and the streamed one.
    assert [call.kwargs["response"] for call in writer.record.call_args_list] == [expected] * 2
    for stop_string in [stop] if isinstance(stop, str) else stop:
        assert stop_string not in text


@pytest.mark.parametrize("name", sorted(_CASES))
def test_no_chunk_carries_text_after_the_stop_string(name):
    pieces, stop, expected = _CASES[name]
    tokenizer, ids = _script(*pieces)
    client = _client(_ScriptedModel(ids), tokenizer)

    _, _, frames = _streamed(_post(client, stop=stop, stream=True))

    sent = ""
    for frame in frames[:-2]:
        sent += frame["choices"][0]["delta"].get("content", "")
        assert expected.startswith(sent), (sent, expected)


def test_the_first_chunk_is_sent_while_generate_is_still_running():
    """The hold-back is a few characters, not the whole reply."""
    tokenizer, ids = _script("Hello", " brave", " new", " world")
    model = _ScriptedModel(ids, hold_after=3)
    stream = _stream(model, tokenizer, stop=("STOP",))

    first = next(stream)

    assert not model.generate_returned.is_set(), "nothing was sent until generate() had finished"
    sent_first = _content(_frames([first]))
    assert sent_first and "Hello brave new world".startswith(sent_first)
    model.gate.set()
    frames = _frames([first] + _drain(stream))
    assert _content(frames) == "Hello brave new world"
    assert not model.gate_timed_out


# --------------------------------------------------------------------------- #
# Generation ends at the stop string
# --------------------------------------------------------------------------- #


def test_a_plain_reply_stops_generating_after_the_token_that_completes_the_stop_string():
    tokenizer, ids = _script("Hello", " there", "\nUser:", " hi", " again")
    model = _ScriptedModel(ids)
    client = _client(model, tokenizer)

    body = _post(client, stop="\nUser:").json()

    assert model.stopped_early
    # prompt + "Hello" + " there" + "\nUser:": the three tokens that make the stop string.
    assert body["usage"]["completion_tokens"] == 3
    assert body["choices"][0]["message"]["content"] == "Hello there"

    full = _post(client).json()
    assert full["usage"]["completion_tokens"] == len(ids)


def test_a_streamed_reply_ends_generation_when_a_stop_string_is_reached():
    """The model keeps going until told to stop; the stop event is what tells it."""
    tokenizer, ids = _script("Hello", " there", "\nUser:", " hi", " again")
    model = _ScriptedModel(ids, wait_for_stop=True)
    client = _client(model, tokenizer)

    text, reason, _ = _streamed(_post(client, stop="\nUser:", stream=True))

    assert model.stopped_early, "generation was never told to stop"
    assert (text, reason) == ("Hello there", "stop")


def test_generate_gets_the_stop_strings_only_on_the_plain_path():
    tokenizer, ids = _script("a", " END", " b")
    cases = {
        "plain": ({}, True),
        "draft_model": ({"draft_model": MagicMock()}, False),
        "ngram": ({"ngram_config": types.SimpleNamespace(num_draft_tokens=7)}, False),
        "reasoning_parser": ({"reasoning_parser": "deepseek-r1"}, False),
    }
    for name, (options, early) in cases.items():
        model = _ScriptedModel(ids)
        client = _client(model, tokenizer, **options)

        text, reason = _plain(_post(client, stop="END"))

        assert text == "a", name
        assert reason == "stop", name
        assert ("stopping_criteria" in model.calls[0]) is early, name


def test_beam_search_keeps_generating_and_the_reply_is_still_cut():
    tokenizer, ids = _script("a", " END", " b")
    model = _ScriptedModel(ids)
    model.generation_config = types.SimpleNamespace(eos_token_id=EOS_ID, num_beams=4)
    client = _client(model, tokenizer)

    for stream in (False, True):
        response = _post(client, stop="END", stream=stream)
        if stream:
            text, reason, _ = _streamed(response)
        else:
            text, reason = _plain(response)
        assert (text, reason) == ("a", "stop")
    assert all("stopping_criteria" not in call for call in model.calls)


def test_a_reasoning_block_is_removed_before_the_stop_string_is_looked_for():
    tokenizer, ids = _script("<think>", "plan END", " it</think>", "Hello", " END", " tail")
    client = _client(_ScriptedModel(ids), tokenizer, reasoning_parser="deepseek-r1")

    plain_text, plain_reason = _plain(_post(client, stop="END"))
    text, reason, frames = _streamed(_post(client, stop="END", stream=True))

    assert (plain_text, plain_reason) == ("Hello", "stop")
    assert (text, reason) == ("Hello", "stop")
    assert "plan" not in "".join(str(frame) for frame in frames)


def test_mole_reply_is_cut_too():
    tokenizer, ids = _script("unused")
    runtime = MagicMock()
    runtime.generate_text.return_value = ("Blended END more", 4, 6, "length")
    client = _client(_ScriptedModel(ids), tokenizer, mole_runtime=runtime)

    plain_text, plain_reason = _plain(_post(client, stop="END"))
    text, reason, _ = _streamed(_post(client, stop="END", stream=True))

    assert (plain_text, plain_reason) == ("Blended", "stop")
    assert (text, reason) == ("Blended", "stop")


def test_text_the_streamer_never_released_is_still_cut():
    tokenizer, ids = _script("Hello", " brave", " new", " world")
    model = _ScriptedModel(ids, stream_only=2)
    client = _client(model, tokenizer)

    text, reason, _ = _streamed(_post(client, stop=" new", stream=True))

    assert (text, reason) == ("Hello brave", "stop")


def test_a_model_that_ignores_the_streamer_is_still_cut():
    tokenizer, ids = _script("Hello", " brave", " new", " world")
    client = _client(_ScriptedModel(ids, use_streamer=False), tokenizer)

    text, reason, _ = _streamed(_post(client, stop=" new", stream=True))

    assert (text, reason) == ("Hello brave", "stop")


def test_generate_response_returns_the_uncut_reply_and_adds_one_criterion_per_source():
    tokenizer, ids = _script("a", " END", " b")
    model = _ScriptedModel(ids)

    text, _, _, _ = serve._generate_response(
        model, tokenizer, MESSAGES, max_tokens=8, temperature=0.0, stop_strings=("END",),
    )
    assert text == "a END"  # generation ended after the stop string; the caller cuts
    assert len(model.calls[0]["stopping_criteria"]) == 1

    both = _ScriptedModel(ids)
    serve._generate_response(
        both, tokenizer, MESSAGES, max_tokens=8, temperature=0.0,
        stop_event=threading.Event(), stop_strings=("END",),
    )
    assert len(both.calls[0]["stopping_criteria"]) == 2

    plain = _ScriptedModel(ids)
    serve._generate_response(plain, tokenizer, MESSAGES, max_tokens=8, temperature=0.0)
    assert "stopping_criteria" not in plain.calls[0]


# --------------------------------------------------------------------------- #
# A real generate() on a tiny random Llama
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(3)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=128,
            pad_token_id=0,
            eos_token_id=1,
        )
    ).eval()
    # Plain words only and no end token, so every run uses its whole budget.
    model.generation_config.suppress_tokens = [0, 1, 2]
    return model, _word_tokenizer()


@pytest.mark.parametrize("stream", [False, True], ids=["reply", "stream"])
def test_real_generate_stops_at_the_stop_string(tiny_model, stream):
    from soup_cli.utils.stop_strings import cut_at_stop

    model, tokenizer = tiny_model
    client = _client(model, tokenizer, max_tokens_default=12)
    free = _post(client, max_tokens=12).json()
    words = free["choices"][0]["message"]["content"].split()
    assert len(words) == 12
    stop = " ".join(words[4:6])
    expected, matched = cut_at_stop(" ".join(words), (stop,))
    assert matched == stop

    response = _post(client, stop=stop, stream=stream, max_tokens=12)

    if stream:
        text, reason, _ = _streamed(response)
    else:
        body = response.json()
        text, reason = _plain(response)
        # Generation ended at the token that completed the stop string, not at the budget.
        assert body["usage"]["completion_tokens"] < 12
    assert text == expected
    assert reason == "stop"
    assert stop not in text


# --------------------------------------------------------------------------- #
# Anthropic ``stop_sequences`` on /v1/messages
# --------------------------------------------------------------------------- #

URL = "/v1/messages"


def _messages_post(client, *, stop=ABSENT, stream=False, headers=None, **extra):
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
    return client.post(URL, json=body, headers=headers)


def _messages_events(text):
    """``[(event name, data)]`` of an SSE body: ``event:`` line, ``data:`` line, blank line."""
    events = []
    for block in text.split("\n\n"):
        if not block:
            continue
        lines = block.split("\n")
        assert len(lines) == 2, repr(block)
        assert lines[0].startswith("event: ") and lines[1].startswith("data: "), repr(block)
        data = json.loads(lines[1][len("data: "):])
        assert data["type"] == lines[0][len("event: "):]
        events.append((data["type"], data))
    return events


def _messages_result(client, *, stream, **kwargs):
    """``(text, stop_reason, stop_sequence)`` of a reply, plain or streamed."""
    response = _messages_post(client, stream=stream, **kwargs)
    assert response.status_code == 200, response.text
    if not stream:
        body = response.json()
        return body["content"][0]["text"], body["stop_reason"], body["stop_sequence"]
    events = _messages_events(response.text)
    text = "".join(d["delta"]["text"] for name, d in events if name == "content_block_delta")
    deltas = [d for name, d in events if name == "message_delta"]
    assert len(deltas) == 1
    return text, deltas[0]["delta"]["stop_reason"], deltas[0]["delta"]["stop_sequence"]


@pytest.mark.parametrize(
    "stop",
    [["\nUser:"], ["nothing here", "\nUser:"], ["\nUser:", "never"]],
    ids=["one", "second_matches", "first_matches"],
)
@pytest.mark.parametrize("stream", [False, True], ids=["reply", "stream"])
def test_messages_stop_sequence_ends_the_reply_with_stop_reason_stop_sequence(stop, stream):
    tokenizer, ids = _script(*_REPLY)
    client = _client(_ScriptedModel(ids), tokenizer)

    result = _messages_result(client, stream=stream, stop=stop)

    assert result == ("Hello there", "stop_sequence", "\nUser:")


@pytest.mark.parametrize("stop", [ABSENT, None, []], ids=["absent", "null", "empty_list"])
@pytest.mark.parametrize("stream", [False, True], ids=["reply", "stream"])
def test_messages_without_stop_sequences_leaves_the_reply_alone(stop, stream):
    tokenizer, ids = _script(*_REPLY)
    client = _client(_ScriptedModel(ids), tokenizer)

    result = _messages_result(client, stream=stream, stop=stop)

    assert result == ("Hello there\nUser: hi again", "end_turn", None)


@pytest.mark.parametrize("name", sorted(_CASES))
def test_messages_streamed_reply_equals_the_plain_reply(name):
    from soup_cli.utils.stop_strings import cut_at_stop

    pieces, stop, expected = _CASES[name]
    stops = [stop] if isinstance(stop, str) else stop
    tokenizer, ids = _script(*pieces)
    client = _client(_ScriptedModel(ids), tokenizer)
    _, matched = cut_at_stop("".join(pieces).strip(), stops)
    reason = "stop_sequence" if matched else "end_turn"

    plain = _messages_result(client, stream=False, stop=stops)
    streamed = _messages_result(client, stream=True, stop=stops)

    assert plain == (expected, reason, matched)
    assert streamed == plain


@pytest.mark.parametrize("stream", [False, True], ids=["reply", "stream"])
def test_messages_stop_sequence_beats_the_token_budget(stream):
    tokenizer, ids = _script("a", " b", " STOP", " c", eos=False)
    client = _client(_ScriptedModel(ids), tokenizer)

    result = _messages_result(client, stream=stream, stop=["STOP"], max_tokens=4)

    assert result == ("a b", "stop_sequence", "STOP")


_BAD_SEQUENCES = [
    ("END", "stop_sequences must be a list of strings"),
    ("", "stop_sequences must be a list of strings"),
    ([""], "stop_sequences[0] must not be an empty string"),
    (["ok", ""], "stop_sequences[1] must not be an empty string"),
    (["a", "b", "c", "d", "e"], "stop_sequences accepts at most 4 stop strings, got 5"),
    (["x" * 257], "stop_sequences[0] is longer than 256 characters"),
    ([1], "stop_sequences[0] must be a string"),
    ([None], "stop_sequences[0] must be a string"),
    (7, "stop_sequences must be a list of strings"),
    ({"a": "b"}, "stop_sequences must be a list of strings"),
    (True, "stop_sequences must be a list of strings"),
]


@pytest.mark.parametrize(("stop", "message"), _BAD_SEQUENCES)
@pytest.mark.parametrize("stream", [False, True], ids=["reply", "stream"])
def test_messages_bad_stop_sequences_is_a_400_that_names_the_field(stop, message, stream):
    tokenizer, ids = _script(*_REPLY)
    model = _ScriptedModel(ids)
    client = _client(model, tokenizer)

    response = _messages_post(client, stop=stop, stream=stream)

    assert response.status_code == 400, response.text
    assert response.json() == {"detail": message}
    assert model.calls == []


def test_messages_other_invalid_requests_still_answer_a_bare_invalid_request():
    tokenizer, ids = _script(*_REPLY)
    client = _client(_ScriptedModel(ids), tokenizer)

    response = client.post(URL, json={"model": "m", "max_tokens": 0, "messages": []})

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid request"}


def test_messages_plain_reply_always_carries_stop_sequence():
    tokenizer, ids = _script("Hello", " world")
    client = _client(_ScriptedModel(ids), tokenizer)

    body = _messages_post(client).json()

    assert body["stop_sequence"] is None
    assert body["stop_reason"] == "end_turn"
    assert set(body) == {
        "id", "type", "role", "model", "content", "stop_reason", "stop_sequence", "usage",
    }


@pytest.mark.parametrize(
    ("finish_reason", "stop_sequence", "expected"),
    [
        ("stop", None, "end_turn"),
        (None, None, "end_turn"),
        ("length", None, "max_tokens"),
        ("stop", "END", "stop_sequence"),
        ("length", "END", "stop_sequence"),
    ],
)
def test_anthropic_stop_reason(finish_reason, stop_sequence, expected):
    assert serve._anthropic_stop_reason(finish_reason, stop_sequence) == expected


@pytest.mark.parametrize("stream", [False, True], ids=["reply", "stream"])
def test_x_user_id_selects_the_bank_user_on_messages(stream):
    """The per-user headers were not read on this route; they reach the generation now."""
    tokenizer, ids = _script("Hello", " world")
    bank = MagicMock()
    client = _client(_ScriptedModel(ids), tokenizer, loaded_bank=bank)

    _messages_post(client, stream=stream, headers={"X-User-Id": "alice"})
    assert bank.set_active_user.call_args_list[-1].args == ("alice",)

    _messages_post(client, stream=stream)
    assert bank.set_active_user.call_args_list[-1].args == (None,)
