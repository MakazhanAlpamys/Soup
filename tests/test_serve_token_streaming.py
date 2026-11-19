"""``soup serve --backend transformers`` streams tokens as ``generate`` produces them.

Before, a ``"stream": true`` request ran the whole generation first and only then
sent the first SSE chunk, so a client saw nothing until the reply was finished and
then got a burst. ``generate`` now runs in a worker thread with a
``TextIteratorStreamer`` and the SSE generator yields chunks as text arrives.

The wire format is unchanged: the same chunk fields, final chunk, ``[DONE]`` and
``finish_reason`` for the same generated text. Only the timing moves. Requests
whose output is post-processed on the full text (reasoning parser), or whose decode
loop is not a plain ``generate`` call (draft model, n-gram lookup, MoLE), or whose
model sets beam search, keep the generate-then-send path; each has a test pinning that.

The fake models below call the streamer the way ``transformers`` does: the prompt
first (dropped by ``skip_prompt``), then one ``put`` per new token, then ``end``.
"""

from __future__ import annotations

import contextvars
import copy
import gc
import inspect
import json
import logging
import threading
import time
import types
import weakref
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")
pytest.importorskip("fastapi")

from soup_cli.commands import serve  # noqa: E402

PROMPT_IDS = [5, 6, 7]
EOS_ID = 1
FIRST_PIECE_ID = 10
WORKER_NAME = "soup-serve-generate"
MESSAGES = [{"role": "user", "content": "hi"}]
# How long a fake generate() waits for the test (a gate to open, a stop event). Nothing
# waits this long on a passing run; it only keeps a broken test from hanging for good.
GATE_TIMEOUT = 30.0


class _Abort(BaseException):
    """Not an ``Exception``: only an ``except BaseException`` sees it."""


class _PieceTokenizer:
    """A tokenizer whose pieces are chosen by the test, so whitespace is exact."""

    pad_token_id = 0
    eos_token_id = EOS_ID
    chat_template = None

    def __init__(self, pieces):
        self.pieces = dict(pieces)

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = torch.tensor([PROMPT_IDS])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, ids, skip_special_tokens=True, **kwargs):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return "".join(
            self.pieces[token]
            for token in ids
            if not (skip_special_tokens and token in (0, EOS_ID))
        )


def _script(*pieces, eos=True):
    """``(tokenizer, token_ids)`` for a reply made of exactly these text pieces."""
    table = {FIRST_PIECE_ID + index: piece for index, piece in enumerate(pieces)}
    ids = list(table)
    if eos:
        ids.append(EOS_ID)
    return _PieceTokenizer(table), ids


class _ScriptedModel:
    """Stands in for a HF model: replays ``token_ids`` through ``generate``."""

    device = "cpu"
    generation_config = types.SimpleNamespace(eos_token_id=EOS_ID)

    def __init__(
        self,
        token_ids,
        *,
        hold_after=None,
        fail_at=None,
        use_streamer=True,
        stream_only=None,
        streamed_ids=None,
        on_generate=None,
        fail_with=None,
        wait_for_stop=False,
        after_stop_seconds=0.0,
        linger=None,
    ):
        self.token_ids = list(token_ids)
        self.hold_after = hold_after
        self.fail_at = fail_at
        self.use_streamer = use_streamer
        # Only the first ``stream_only`` tokens reach the streamer (the rest are still
        # returned), or ``streamed_ids`` reach it in place of the real ones.
        self.stream_only = stream_only
        self.streamed_ids = streamed_ids
        self.on_generate = on_generate
        self.fail_with = fail_with
        # After its tokens, generate() keeps running until the stop criteria fire, then
        # spends ``after_stop_seconds`` finishing the decode step it was in.
        self.wait_for_stop = wait_for_stop
        self.after_stop_seconds = after_stop_seconds
        # Set after streamer.end(): generate() has produced everything but not returned.
        self.linger = linger
        self.gate = threading.Event()
        self.gate_timed_out = False
        self.generate_returned = threading.Event()
        self.stopped_early = False
        self.calls = []
        self.threads = []

    def _streams(self, index):
        return self.stream_only is None or index < self.stream_only

    @staticmethod
    def _wait_until_stopped(criteria, sequence):
        deadline = time.monotonic() + GATE_TIMEOUT
        while time.monotonic() < deadline:
            if bool(criteria(torch.tensor([sequence]), None).any()):
                return True
            time.sleep(0.002)
        return False

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        self.threads.append(threading.current_thread())
        if self.on_generate is not None:
            self.on_generate(self)
        streamer = kwargs.get("streamer") if self.use_streamer else None
        criteria = kwargs.get("stopping_criteria")
        sequence = kwargs["input_ids"][0].tolist()
        try:
            if streamer is not None:
                streamer.put(kwargs["input_ids"])
            for index, token in enumerate(self.token_ids):
                if criteria is not None and bool(criteria(torch.tensor([sequence]), None).any()):
                    self.stopped_early = True
                    break
                if index == self.fail_at:
                    raise self.fail_with or RuntimeError("boom in generate")
                sequence.append(token)
                if streamer is not None and self._streams(index):
                    shown = token if self.streamed_ids is None else self.streamed_ids[index]
                    streamer.put(torch.tensor([shown]))
                if self.hold_after is not None and index + 1 == self.hold_after:
                    # A generate() that is still running: the reply is not finished.
                    self.gate_timed_out = not self.gate.wait(timeout=GATE_TIMEOUT)
            if self.wait_for_stop:
                self.stopped_early = self._wait_until_stopped(criteria, sequence)
                if self.after_stop_seconds:
                    time.sleep(self.after_stop_seconds)
            if streamer is not None:
                streamer.end()
            if self.linger is not None:
                self.linger.wait(timeout=GATE_TIMEOUT)
            return torch.tensor([sequence])
        finally:
            self.generate_returned.set()


def _stream(model, tokenizer, **overrides):
    return serve._stream_response(
        model,
        tokenizer,
        MESSAGES,
        max_tokens=overrides.pop("max_tokens", 64),
        temperature=0.0,
        top_p=0.9,
        model_name="m",
        **overrides,
    )


def _drain(generator, timeout=10.0):
    """Consume ``generator`` on a helper thread so a hang fails instead of freezing pytest."""
    chunks: list = []
    failure: list = []

    def run():
        try:
            chunks.extend(generator)
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            failure.append(exc)

    runner = threading.Thread(target=run, daemon=True)
    runner.start()
    runner.join(timeout)
    assert not runner.is_alive(), "the SSE generator never finished (hang)"
    if failure:
        raise failure[0]
    return chunks


def _frames(chunks):
    """Parse SSE strings into dicts; the sentinel stays the string ``[DONE]``."""
    parsed = []
    for chunk in chunks:
        assert chunk.startswith("data: ") and chunk.endswith("\n\n"), repr(chunk)
        body = chunk[len("data: "):-2]
        parsed.append(body if body == "[DONE]" else json.loads(body))
    return parsed


def _content(frames):
    return "".join(
        frame["choices"][0]["delta"].get("content", "")
        for frame in frames
        if isinstance(frame, dict) and "choices" in frame
    )


def _worker_threads():
    return [thread for thread in threading.enumerate() if thread.name == WORKER_NAME]


def _wait_for_workers_to_exit():
    deadline = time.monotonic() + 5.0
    while _worker_threads() and time.monotonic() < deadline:
        time.sleep(0.01)
    return _worker_threads()


# --------------------------------------------------------------------------- #
# The reply is sent while generate() is still running
# --------------------------------------------------------------------------- #


def test_first_chunk_is_sent_while_generate_is_still_running():
    tokenizer, ids = _script("Hello", " brave", " new", " world")
    model = _ScriptedModel(ids, hold_after=3)
    stream = _stream(model, tokenizer)

    first = next(stream)

    # Pinned on the unfixed code: generate() finishes (its gate times out) before the
    # first chunk is sent, so it is already over when this assertion runs.
    assert not model.generate_returned.is_set(), (
        "the first SSE chunk only arrived after generate() had finished"
    )
    frame = json.loads(first[len("data: "):])
    assert frame["choices"][0]["delta"] == {"content": "Hello"}
    assert frame["choices"][0]["finish_reason"] is None

    model.gate.set()
    rest = _frames(_drain(stream))
    assert _content([frame]) + _content(rest) == "Hello brave new world"
    assert rest[-1] == "[DONE]"
    assert rest[-2]["choices"][0] == {"index": 0, "delta": {}, "finish_reason": "stop"}
    assert not model.gate_timed_out


def test_sse_chunk_shape_is_unchanged():
    tokenizer, ids = _script("Hello", " world")
    frames = _frames(_drain(_stream(_ScriptedModel(ids), tokenizer)))

    chunk_ids = {frame["id"] for frame in frames[:-1]}
    assert len(chunk_ids) == 1 and chunk_ids.pop().startswith("chatcmpl-")
    # Every chunk also carries the answer's response_id (#2532), the same in all of them.
    assert len({frame["response_id"] for frame in frames[:-1]}) == 1
    for frame in frames[:-1]:
        assert set(frame) == {"id", "object", "created", "model", "choices", "response_id"}
        assert frame["object"] == "chat.completion.chunk"
        assert frame["model"] == "m"
        assert len(frame["choices"]) == 1 and frame["choices"][0]["index"] == 0
    assert all(set(f["choices"][0]) == {"index", "delta", "finish_reason"} for f in frames[:-1])
    assert all(f["choices"][0]["finish_reason"] is None for f in frames[:-2])
    assert frames[-2]["choices"][0]["delta"] == {}
    assert frames[-1] == "[DONE]"


def test_generate_gets_a_streamer_and_runs_on_a_worker_thread():
    tokenizer, ids = _script("Hello", " world")
    model = _ScriptedModel(ids)

    _drain(_stream(model, tokenizer))

    from transformers import TextIteratorStreamer

    kwargs = model.calls[0]
    assert isinstance(kwargs["streamer"], TextIteratorStreamer)
    assert type(kwargs["streamer"]).__name__ == "_ReplyStreamer"
    assert model.threads[0].name == WORKER_NAME
    assert model.threads[0] is not threading.current_thread()
    assert not _wait_for_workers_to_exit()


def test_the_worker_thread_is_a_daemon():
    """A generate() that never returns must not keep the interpreter from exiting."""
    tokenizer, ids = _script("Hello", " world")
    model = _ScriptedModel(ids)

    _drain(_stream(model, tokenizer))

    assert model.threads[0].daemon is True


# --------------------------------------------------------------------------- #
# Same text, finish_reason and token count as the non-streaming reply
# --------------------------------------------------------------------------- #

_REPLIES = {
    "words": ["Hello", " brave", " new", " world", "."],
    "leading_and_trailing_whitespace": ["\n\n  ", "Hello", " there", "  \n "],
    "multi_line": ["Line one", "\n", "line two", "\n\n", "line three"],
    "unicode": ["héllo", " wörld", " ✓", " 日本語"],
    "single_word": ["Hello"],
    "whitespace_only": ["  ", "\n"],
    "interior_runs": ["a", "  ", "b", "\t", "c"],
}


@pytest.mark.parametrize("name", sorted(_REPLIES))
def test_streamed_deltas_equal_the_non_streamed_reply(name):
    from fastapi.testclient import TestClient

    tokenizer, ids = _script(*_REPLIES[name])
    writer = MagicMock()
    app = serve._create_app(
        _ScriptedModel(ids), tokenizer, "cpu", "m", 64, trace_log_writer=writer
    )
    client = TestClient(app, base_url="http://localhost")
    body = {"model": "m", "messages": MESSAGES, "temperature": 0}

    plain = client.post("/v1/chat/completions", json=body).json()
    streamed = client.post("/v1/chat/completions", json={**body, "stream": True})

    assert streamed.status_code == 200
    frames = _frames([part + "\n\n" for part in streamed.text.split("\n\n") if part])
    expected = plain["choices"][0]["message"]["content"]
    assert expected == "".join(_REPLIES[name]).strip()
    assert _content(frames) == expected
    assert frames[-2]["choices"][0]["finish_reason"] == plain["choices"][0]["finish_reason"]
    assert frames[-1] == "[DONE]"
    # The conversation record of the streamed call carries the same text and token
    # count, under the response_id its chunks carried (#2532).
    record = writer.record_conversation.call_args.kwargs
    assert record["response"] == expected
    assert record["completion_tokens"] == plain["usage"]["completion_tokens"] == len(ids)
    assert record["response_id"] == frames[0]["response_id"]


def test_finish_reason_length_when_the_budget_is_used():
    tokenizer, ids = _script("a", " b", " c", " d", eos=False)
    frames = _frames(_drain(_stream(_ScriptedModel(ids), tokenizer, max_tokens=4)))

    assert _content(frames) == "a b c d"
    assert frames[-2]["choices"][0]["finish_reason"] == "length"


def test_a_model_that_ignores_the_streamer_still_gets_its_reply_sent():
    """Not every ``generate`` path calls the streamer: the reply is then sent after it."""
    tokenizer, ids = _script("Hello", " brave", " world")
    model = _ScriptedModel(ids, use_streamer=False)

    frames = _frames(_drain(_stream(model, tokenizer)))

    assert _content(frames) == "Hello brave world"
    assert frames[-2]["choices"][0]["finish_reason"] == "stop"
    assert frames[-1] == "[DONE]"


def test_text_the_streamer_did_not_release_is_sent_at_the_end():
    """The finished reply is what counts: anything the streamer left out follows."""
    tokenizer, ids = _script("Hello", " brave", " new", " world")
    model = _ScriptedModel(ids, stream_only=2)

    frames = _frames(_drain(_stream(model, tokenizer)))

    assert _content(frames) == "Hello brave new world"
    assert frames[-2]["choices"][0]["finish_reason"] == "stop"
    assert frames[-1] == "[DONE]"


def test_text_already_sent_is_never_taken_back():
    """If the streamed decode ever disagrees with the final one, what went out stays."""
    tokenizer, ids = _script("Hello", " world", "Hullo")
    model = _ScriptedModel(ids[:2] + [EOS_ID], streamed_ids=[ids[2], ids[1], EOS_ID])

    frames = _frames(_drain(_stream(model, tokenizer)))

    assert _content(frames) == "Hullo world"
    assert frames[-2]["choices"][0]["finish_reason"] == "stop"
    assert frames[-1] == "[DONE]"


def test_a_longer_final_reply_is_not_spliced_onto_text_that_is_not_its_prefix():
    """The text sent is not a prefix of the final decode, and the final decode is longer.

    Appending the rest of the final reply would join two different decodes ("Hullo
    big" and " world", from "Hello big world"), so nothing more is sent.
    """
    tokenizer, ids = _script("Hello", " big", " world", "Hullo")
    hello, big, world, hullo, _eos = ids
    model = _ScriptedModel(
        [hello, big, world, EOS_ID],
        stream_only=2,
        streamed_ids=[hullo, big, world, EOS_ID],
    )

    frames = _frames(_drain(_stream(model, tokenizer)))

    assert _content(frames) == "Hullo big"
    assert frames[-2]["choices"][0]["finish_reason"] == "stop"
    assert frames[-1] == "[DONE]"


def test_an_empty_reply_keeps_the_single_empty_chunk():
    tokenizer, ids = _script()
    frames = _frames(_drain(_stream(_ScriptedModel(ids), tokenizer)))

    assert [f["choices"][0]["delta"] for f in frames[:-2]] == [{"content": ""}]
    assert frames[-1] == "[DONE]"


# --------------------------------------------------------------------------- #
# Whitespace handling matches str.strip() on the whole reply
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "pieces, expected_text, expected_chunks",
    [
        (["  ", "\n", "Hello"], "Hello", ["Hello"]),
        (["Hello", "  ", "world"], "Hello  world", ["Hello", "  world"]),
        (["Hello", "  "], "Hello", ["Hello"]),
        (["", "  ", ""], "", []),
        (["a ", " ", "b"], "a  b", ["a", "  b"]),
    ],
)
def test_strip_emitter_matches_str_strip(pieces, expected_text, expected_chunks):
    emitter = serve._StripEmitter()
    emitted = [emitter.feed(piece) for piece in pieces]

    assert "".join(emitted) == expected_text == "".join(pieces).strip()
    assert [chunk for chunk in emitted if chunk] == expected_chunks


def test_strip_emitter_holds_trailing_whitespace_until_more_text_arrives():
    emitter = serve._StripEmitter()

    assert emitter.feed("Hello ") == "Hello"
    assert emitter.feed("  ") == ""
    assert emitter.feed("world") == "   world"


# --------------------------------------------------------------------------- #
# Errors and early close
# --------------------------------------------------------------------------- #


def test_error_in_the_generation_thread_reaches_the_client_and_ends_the_stream(caplog):
    tokenizer, ids = _script("Hello", " brave", " new", " world")
    model = _ScriptedModel(ids, fail_at=3)
    outcomes = []

    with caplog.at_level(logging.ERROR, logger=serve.logger.name):
        chunks = _drain(_stream(model, tokenizer, canary_outcome=outcomes.append))

    assert chunks[-1] == 'data: {"error": "Internal server error"}\n\n'
    frames = _frames(chunks[:-1])
    assert _content(frames) == "Hello brave"
    assert "data: [DONE]\n\n" not in chunks
    assert all(frame["choices"][0]["finish_reason"] is None for frame in frames)
    assert outcomes == [False]
    # The operator's log carries the error the thread raised, not a bookkeeping one.
    logged = [record.exc_info[1] for record in caplog.records if record.exc_info]
    assert [str(error) for error in logged] == ["boom in generate"]
    assert not _wait_for_workers_to_exit()


def test_error_before_any_text_is_the_only_frame():
    tokenizer, ids = _script("Hello", " world")
    model = _ScriptedModel(ids, fail_at=0)

    chunks = _drain(_stream(model, tokenizer))

    assert chunks == ['data: {"error": "Internal server error"}\n\n']
    assert not _wait_for_workers_to_exit()


def test_successful_stream_records_one_canary_success():
    tokenizer, ids = _script("Hello", " world")
    outcomes = []

    _drain(_stream(_ScriptedModel(ids), tokenizer, canary_outcome=outcomes.append))

    assert outcomes == [True]


@pytest.mark.parametrize("reasoning_parser", [None, "deepseek-r1"], ids=["streamed", "inline"])
def test_a_base_exception_in_generate_reaches_the_stream_as_it_does_inline(reasoning_parser):
    """The worker catches ``BaseException``: an exit signal is not turned into a KeyError."""
    tokenizer, ids = _script("Hello", " world")
    abort = _Abort("stop it")
    model = _ScriptedModel(ids, fail_at=1, fail_with=abort)

    with pytest.raises(_Abort) as raised:
        _drain(_stream(model, tokenizer, reasoning_parser=reasoning_parser))

    assert raised.value is abort
    assert not _wait_for_workers_to_exit()


class _Leftovers:
    """Stands in for what a failed generation leaves in its frames (ids, KV cache)."""


class _FailsHoldingState(_ScriptedModel):
    def generate(self, **kwargs):
        held = _Leftovers()
        self.held = weakref.ref(held)
        raise RuntimeError("boom in generate")


def test_a_failed_generation_is_freed_without_a_garbage_collection(monkeypatch):
    """The failure list must not keep the exception, and with it the worker's frames."""
    # A log record carries the exception and would keep it alive on its own.
    monkeypatch.setattr(serve, "logger", MagicMock())
    tokenizer, ids = _script("Hello")
    model = _FailsHoldingState(ids)
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        chunks = _drain(_stream(model, tokenizer))

        assert chunks == ['data: {"error": "Internal server error"}\n\n']
        assert model.held() is None, "the failed generation's frames are still alive"
    finally:
        if was_enabled:
            gc.enable()


def test_a_client_that_leaves_mid_stream_still_leaves_its_conversation_record():
    tokenizer, ids = _script("w", " w", " w", eos=False)
    model = _ScriptedModel(ids, wait_for_stop=True)
    writer = MagicMock()
    stream = _stream(
        model, tokenizer, max_tokens=1000, trace_log_writer=writer,
        started=time.perf_counter(), response_id="11111111-2222-4333-8444-555555555555",
    )

    first = json.loads(next(stream)[len("data: "):])
    assert writer.record_conversation.call_count == 0  # nothing is logged mid-stream
    stream.close()

    assert model.generate_returned.wait(timeout=5.0)
    record = writer.record_conversation.call_args.kwargs
    assert record["response_id"] == "11111111-2222-4333-8444-555555555555"
    assert record["finish_reason"] == "client_disconnected"
    assert record["response"] == first["choices"][0]["delta"]["content"]
    assert not _wait_for_workers_to_exit()


def test_closing_the_stream_stops_generation_after_the_current_token():
    tokenizer, ids = _script("w", " w", " w", eos=False)
    model = _ScriptedModel(ids, wait_for_stop=True)
    stream = _stream(model, tokenizer, max_tokens=1000)

    next(stream)
    stream.close()

    # The stop criteria see the event, so generate() ends on its own thread.
    assert model.generate_returned.wait(timeout=5.0), "generate() never saw the stop event"
    assert model.stopped_early
    assert not _wait_for_workers_to_exit()


def test_closing_the_stream_does_not_wait_for_the_worker_and_the_lock_is_released_later():
    """``close()`` can run on the server's event-loop thread, so it returns at once."""
    tokenizer, ids = _script("w", " w", " w", eos=False)
    lock = threading.Lock()
    model = _ScriptedModel(ids, wait_for_stop=True, after_stop_seconds=2.0)
    model.set_adapter = MagicMock()
    stream = _stream(
        model, tokenizer, max_tokens=1000,
        adapter_lock=lock, adapter_names={"a"}, requested_adapter="a",
    )

    next(stream)
    assert lock.locked()
    started = time.monotonic()
    stream.close()
    elapsed = time.monotonic() - started

    # The worker is still inside the decode step it was in when the stream closed.
    assert elapsed < 1.0, f"close() took {elapsed:.2f}s: it waited for the worker"
    assert not model.generate_returned.is_set()
    assert lock.locked()
    # It ends that step, returns, and releases the lock on its own thread.
    deadline = time.monotonic() + 5.0
    while lock.locked() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not lock.locked()
    assert model.stopped_early
    assert not _wait_for_workers_to_exit()


def test_closing_the_stream_closes_the_inner_iterator_even_if_it_is_kept_alive(monkeypatch):
    """The stop event is set when the stream closes, not whenever the iterator is freed.

    Another reference (a traceback, a debugger, an interpreter without refcounting)
    can keep the inner iterator alive after the SSE generator is gone.
    """
    opened = []
    real = serve._iter_generated_text

    def spy(*args, **kwargs):
        iterator = real(*args, **kwargs)
        opened.append(iterator)
        return iterator

    monkeypatch.setattr(serve, "_iter_generated_text", spy)
    tokenizer, ids = _script("w", " w", " w", eos=False)
    model = _ScriptedModel(ids, wait_for_stop=True)
    stream = _stream(model, tokenizer, max_tokens=1000)

    next(stream)
    stream.close()

    assert inspect.getgeneratorstate(opened[0]) == inspect.GEN_CLOSED
    assert model.generate_returned.wait(timeout=5.0)
    assert model.stopped_early


def test_a_worker_still_running_after_the_reply_ended_is_logged(monkeypatch, caplog):
    monkeypatch.setattr(serve, "_STREAM_JOIN_SECONDS", 0.05)
    tokenizer, ids = _script("Hello", " world")
    linger = threading.Event()
    model = _ScriptedModel(ids, linger=linger)

    try:
        with caplog.at_level(logging.WARNING, logger=serve.logger.name):
            _drain(_stream(model, tokenizer))
    finally:
        linger.set()

    messages = [record.getMessage() for record in caplog.records]
    assert any("stream generation thread still running" in message for message in messages), (
        messages
    )
    assert not _wait_for_workers_to_exit()


def test_an_exception_thrown_into_the_text_iterator_also_stops_generation_at_once():
    """Not only a close: any exception thrown in sets the stop event and does not wait."""
    tokenizer, _ = _script("a", " b")
    seen = {}

    def generate(streamer, stop_event):
        seen["event"] = stop_event
        streamer.put(torch.tensor([PROMPT_IDS]))
        streamer.put(torch.tensor([FIRST_PIECE_ID]))
        streamer.put(torch.tensor([FIRST_PIECE_ID + 1]))
        stop_event.wait(timeout=GATE_TIMEOUT)
        streamer.end()

    pieces = serve._iter_generated_text(tokenizer, generate, {})

    assert next(pieces) == "a "
    with pytest.raises(RuntimeError, match="thrown in"):
        pieces.throw(RuntimeError("thrown in"))
    assert seen["event"].is_set()
    assert not _wait_for_workers_to_exit()


def test_the_reply_is_finished_only_after_generate_has_returned():
    """``streamer.end()`` runs inside generate(); its result lands in the outcome on return."""
    tokenizer, ids = _script("Hello", " world")
    linger = threading.Event()
    timer = threading.Timer(0.3, linger.set)
    model = _ScriptedModel(ids, linger=linger)

    timer.start()
    try:
        frames = _frames(_drain(_stream(model, tokenizer)))
    finally:
        timer.cancel()
        linger.set()

    assert _content(frames) == "Hello world"
    assert frames[-2]["choices"][0]["finish_reason"] == "stop"
    assert frames[-1] == "[DONE]"


def test_a_copy_of_the_stop_criteria_watches_the_same_event():
    stop = threading.Event()
    criteria = serve._stop_event_criteria(stop)
    clone = copy.deepcopy(criteria)
    sequence = torch.tensor([[1, 2, 3]])

    assert not bool(clone(sequence, None).any())
    stop.set()
    assert bool(clone(sequence, None).all())
    assert bool(criteria(sequence, None).all())


# --------------------------------------------------------------------------- #
# Request features that stream
# --------------------------------------------------------------------------- #


def test_logits_processors_and_kv_cache_kwargs_reach_a_streaming_generate():
    tokenizer, ids = _script("Hello", " world")
    model = _ScriptedModel(ids)
    processors = ["processor"]

    _drain(_stream(
        model, tokenizer,
        logits_processor=processors,
        kv_cache_generate_kwargs={"cache_implementation": "quantized"},
    ))

    kwargs = model.calls[0]
    assert kwargs["logits_processor"] is processors
    assert kwargs["cache_implementation"] == "quantized"
    assert "streamer" in kwargs and "stopping_criteria" in kwargs


def test_per_user_bank_selection_is_visible_on_the_worker_thread():
    selected = contextvars.ContextVar("selected_user", default=None)
    seen = []

    class Bank:
        def set_active_user(self, user_id):
            selected.set(user_id)

    tokenizer, ids = _script("Hello", " world")
    model = _ScriptedModel(ids, on_generate=lambda _model: seen.append(selected.get()))

    _drain(_stream(model, tokenizer, loaded_bank=Bank(), x_user_id="alice"))

    assert seen == ["alice"]


def test_adapter_is_selected_on_the_worker_thread_and_the_lock_is_held_while_it_runs():
    lock = threading.Lock()
    adapter_calls = []
    held_during_generate = []

    tokenizer, ids = _script("Hello", " world")
    model = _ScriptedModel(
        ids, on_generate=lambda _model: held_during_generate.append(lock.locked())
    )
    model.set_adapter = lambda name: adapter_calls.append((name, threading.current_thread().name))

    chunks = _drain(_stream(
        model, tokenizer,
        adapter_lock=lock, adapter_names={"sql"}, requested_adapter="sql",
    ))

    assert _content(_frames(chunks)) == "Hello world"
    assert adapter_calls == [("sql", WORKER_NAME)]
    assert held_during_generate == [True]
    assert not lock.locked()


# --------------------------------------------------------------------------- #
# Requests that keep the generate-then-send path
# --------------------------------------------------------------------------- #


def _assert_not_streamed(model):
    assert len(model.calls) == 1
    assert "streamer" not in model.calls[0]
    assert "stopping_criteria" not in model.calls[0]
    assert model.threads[0].name != WORKER_NAME


def test_reasoning_parser_keeps_full_generation():
    """The reasoning block is cut from the finished text, so it cannot be streamed."""
    tokenizer, ids = _script("<think>", "plan the", " answer</think>", "Hello", " world")
    model = _ScriptedModel(ids)

    frames = _frames(_drain(_stream(model, tokenizer, reasoning_parser="deepseek-r1")))

    _assert_not_streamed(model)
    assert _content(frames) == "Hello world"
    assert "plan" not in "".join(str(frame) for frame in frames)


def test_draft_model_keeps_full_generation():
    tokenizer, ids = _script("Hello", " world")
    model = _ScriptedModel(ids)
    draft = object()

    frames = _frames(_drain(_stream(
        model, tokenizer, assistant_model=draft, num_assistant_tokens=3,
    )))

    _assert_not_streamed(model)
    assert model.calls[0]["assistant_model"] is draft
    assert _content(frames) == "Hello world"


def test_ngram_lookup_keeps_full_generation():
    tokenizer, ids = _script("Hello", " world")
    model = _ScriptedModel(ids)

    frames = _frames(_drain(_stream(
        model, tokenizer, ngram_config=types.SimpleNamespace(num_draft_tokens=7),
    )))

    _assert_not_streamed(model)
    assert model.calls[0]["prompt_lookup_num_tokens"] == 7
    assert _content(frames) == "Hello world"


def test_mole_runtime_keeps_full_generation():
    tokenizer, ids = _script("unused")
    model = _ScriptedModel(ids)
    runtime = MagicMock()
    runtime.generate_text.return_value = ("Blended reply here", 4, 3, "stop")

    frames = _frames(_drain(_stream(model, tokenizer, mole_runtime=runtime)))

    assert model.calls == []
    runtime.generate_text.assert_called_once()
    assert _content(frames) == "Blended reply here"
    assert frames[-2]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.parametrize(
    "settings",
    [
        {"num_beams": 4},
        {"num_return_sequences": 2},
        {"num_beams": 2, "num_return_sequences": 2},
    ],
    ids=["beams", "return_sequences", "both"],
)
def test_beam_search_or_several_return_sequences_keep_full_generation(settings):
    """transformers refuses a streamer with beam search, and TextStreamer takes one sequence."""
    tokenizer, ids = _script("Hello", " world")
    model = _ScriptedModel(ids)
    model.generation_config = types.SimpleNamespace(eos_token_id=EOS_ID, **settings)

    frames = _frames(_drain(_stream(model, tokenizer)))

    _assert_not_streamed(model)
    assert _content(frames) == "Hello world"
    assert frames[-2]["choices"][0]["finish_reason"] == "stop"
    assert frames[-1] == "[DONE]"


@pytest.mark.parametrize(
    "config, expected",
    [
        (None, True),
        (types.SimpleNamespace(), True),
        (types.SimpleNamespace(num_beams=1, num_return_sequences=1), True),
        (types.SimpleNamespace(num_beams=None, num_return_sequences=None), True),
        (types.SimpleNamespace(num_beams=2), False),
        (types.SimpleNamespace(num_return_sequences=3), False),
    ],
)
def test_can_stream_follows_the_generation_config(config, expected):
    assert serve._can_stream(types.SimpleNamespace(generation_config=config)) is expected


def test_can_stream_without_a_generation_config():
    assert serve._can_stream(object()) is True


# --------------------------------------------------------------------------- #
# A SentencePiece-style decoder drops the leading space of the text it is given
# --------------------------------------------------------------------------- #

SPIECE = chr(0x2581)  # the SentencePiece word-start marker
# Tokens are written with "_" for the marker. The reply is what the whole token list
# decodes to: the decoder drops the marker of the very first token and turns every
# other one into a space.
_SENTENCEPIECE_REPLIES = {
    "indented_code": (
        ["_def", "_f():", "\n", "___", "_return", "_1"],
        "def f():\n    return 1",
    ),
    "numbered_list": (
        ["_Steps:", "\n", "_1.", "_mix", "\n", "_2.", "_bake"],
        "Steps:\n 1. mix\n 2. bake",
    ),
}


def _sentencepiece_tokenizer():
    from tokenizers import Tokenizer, decoders, pre_tokenizers
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "<eos>": EOS_ID, "<unk>": 2}
    for tokens, _ in _SENTENCEPIECE_REPLIES.values():
        for token in tokens:
            vocab.setdefault(token.replace("_", SPIECE), len(vocab))
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Metaspace(prepend_scheme="first")
    backend.decoder = decoders.Metaspace(prepend_scheme="first")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
    )
    tokenizer.chat_template = (
        "{% for m in messages %}{{ m['role'] }} {{ m['content'] }} {% endfor %}assistant"
    )
    return tokenizer


def _sentencepiece_ids(tokenizer, name):
    tokens, reply = _SENTENCEPIECE_REPLIES[name]
    ids = [tokenizer.convert_tokens_to_ids(token.replace("_", SPIECE)) for token in tokens]
    ids.append(EOS_ID)
    assert tokenizer.decode(ids, skip_special_tokens=True).strip() == reply
    return ids, reply


def _text_released_by(streamer_class, tokenizer, ids):
    """The text ``streamer_class`` releases for ``ids``, the prompt first as generate() sends it."""
    streamer = streamer_class(tokenizer, skip_prompt=True, skip_special_tokens=True)
    streamer.put(torch.tensor([PROMPT_IDS]))
    for token in ids:
        streamer.put(torch.tensor([token]))
    streamer.end()
    parts = []
    while True:
        part = streamer.text_queue.get_nowait()
        if part is streamer.stop_signal:
            return "".join(parts)
        parts.append(part)


@pytest.mark.parametrize("name", sorted(_SENTENCEPIECE_REPLIES))
def test_reply_streamer_keeps_the_leading_space_of_a_line(name):
    tokenizer = _sentencepiece_tokenizer()
    ids, reply = _sentencepiece_ids(tokenizer, name)

    assert _text_released_by(serve._reply_streamer_class(), tokenizer, ids) == reply


@pytest.mark.parametrize("name", sorted(_SENTENCEPIECE_REPLIES))
def test_the_stock_streamer_loses_that_space_on_the_same_tokenizer(name):
    """Control: the tokenizer above is one the stock streamer gets wrong."""
    from transformers import TextIteratorStreamer

    tokenizer = _sentencepiece_tokenizer()
    ids, reply = _sentencepiece_ids(tokenizer, name)

    stock = _text_released_by(TextIteratorStreamer, tokenizer, ids)

    if stock == reply:
        pytest.skip("the stock streamer keeps the space here; _ReplyStreamer is not needed")
    assert stock.replace(" ", "") == reply.replace(" ", "")  # only spaces are lost
    assert len(stock) < len(reply)


CJK = chr(0x65E5) + chr(0x672C)


def _reply_streamer(tokenizer):
    return serve._reply_streamer_class()(tokenizer, skip_prompt=True, skip_special_tokens=True)


@pytest.mark.parametrize(
    "pieces, released",
    [
        (["Hello"], ""),
        (["Hello", " wor"], "Hello "),
        (["Line", "\n"], "Line\n"),
        (["abc", CJK], "abc" + CJK),
    ],
    ids=["unfinished_word", "space", "newline", "cjk"],
)
def test_reply_streamer_releases_text_at_a_space_a_newline_or_a_cjk_character(pieces, released):
    tokenizer, ids = _script(*pieces, eos=False)
    streamer = _reply_streamer(tokenizer)
    streamer.put(torch.tensor([PROMPT_IDS]))
    for token in ids:
        streamer.put(torch.tensor([token]))

    parts = []
    while not streamer.text_queue.empty():
        parts.append(streamer.text_queue.get_nowait())
    assert "".join(parts) == released


def test_reply_streamer_refuses_a_batch():
    tokenizer, _ = _script("Hello")

    with pytest.raises(ValueError, match="batch size 1"):
        _reply_streamer(tokenizer).put(torch.tensor([[1, 2], [3, 4]]))


@pytest.mark.parametrize("name", sorted(_SENTENCEPIECE_REPLIES))
def test_streamed_deltas_keep_the_leading_space_of_a_line(name):
    tokenizer = _sentencepiece_tokenizer()
    ids, reply = _sentencepiece_ids(tokenizer, name)

    frames = _frames(_drain(_stream(_ScriptedModel(ids), tokenizer)))

    assert _content(frames) == reply
    assert frames[-2]["choices"][0]["finish_reason"] == "stop"


# --------------------------------------------------------------------------- #
# Real transformers: tiny random Llama, real generate + TextIteratorStreamer
# --------------------------------------------------------------------------- #


def _word_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, "hi": 3, "user": 4, "assistant": 5}
    vocab.update({f"w{index}": index for index in range(6, 64)})
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
    )
    tokenizer.chat_template = (
        "{% for m in messages %}{{ m['role'] }} {{ m['content'] }} {% endfor %}assistant"
    )
    return tokenizer


@pytest.fixture(scope="module")
def tiny_llama():
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
    # Keep the random model on plain words and off the end token, so every run
    # uses its whole budget.
    model.generation_config.suppress_tokens = [0, 1, 2]
    return model, _word_tokenizer()


def test_real_generate_streams_the_same_reply_as_the_plain_call(tiny_llama, monkeypatch):
    from fastapi.testclient import TestClient

    model, tokenizer = tiny_llama
    got_streamer = []
    real_generate = model.generate

    def spy(**kwargs):
        got_streamer.append("streamer" in kwargs)
        return real_generate(**kwargs)

    monkeypatch.setattr(model, "generate", spy)
    app = serve._create_app(model, tokenizer, "cpu", "tiny", 12)
    client = TestClient(app, base_url="http://localhost")
    body = {"model": "tiny", "messages": MESSAGES, "temperature": 0, "max_tokens": 12}

    plain = client.post("/v1/chat/completions", json=body).json()
    streamed = client.post("/v1/chat/completions", json={**body, "stream": True})

    assert streamed.status_code == 200
    frames = _frames([part + "\n\n" for part in streamed.text.split("\n\n") if part])
    expected = plain["choices"][0]["message"]["content"]
    assert len(expected.split()) == 12, expected
    assert _content(frames) == expected
    assert frames[-2]["choices"][0]["finish_reason"] == plain["choices"][0]["finish_reason"]
    assert plain["usage"]["completion_tokens"] == 12
    assert got_streamer == [False, True]
    assert not _wait_for_workers_to_exit()


def test_real_model_with_beam_search_still_returns_its_reply(tiny_llama, monkeypatch):
    """The real ``generate`` refuses a streamer with beam search; the gate keeps it away."""
    from fastapi.testclient import TestClient

    model, tokenizer = tiny_llama
    config = copy.deepcopy(model.generation_config)
    config.num_beams = 2
    monkeypatch.setattr(model, "generation_config", config)
    app = serve._create_app(model, tokenizer, "cpu", "tiny", 6)
    client = TestClient(app, base_url="http://localhost")
    body = {"model": "tiny", "messages": MESSAGES, "temperature": 0, "max_tokens": 6}

    plain = client.post("/v1/chat/completions", json=body).json()
    streamed = client.post("/v1/chat/completions", json={**body, "stream": True})

    assert streamed.status_code == 200
    frames = _frames([part + "\n\n" for part in streamed.text.split("\n\n") if part])
    expected = plain["choices"][0]["message"]["content"]
    assert expected
    assert not [frame for frame in frames if isinstance(frame, dict) and "error" in frame]
    assert _content(frames) == expected


def test_real_generate_stops_when_the_stop_event_is_set(tiny_llama):
    model, tokenizer = tiny_llama
    stop = threading.Event()
    stop.set()

    _, _, completion_tokens, _ = serve._generate_response(
        model, tokenizer, MESSAGES, max_tokens=12, temperature=0.0, stop_event=stop,
    )

    assert completion_tokens == 1

    _, _, completion_tokens, _ = serve._generate_response(
        model, tokenizer, MESSAGES, max_tokens=12, temperature=0.0,
    )
    assert completion_tokens == 12
