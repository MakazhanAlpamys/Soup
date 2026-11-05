"""``soup serve --backend transformers`` streams tokens as ``generate`` produces them.

Before, a ``"stream": true`` request ran the whole generation first and only then
sent the first SSE chunk, so a client saw nothing until the reply was finished and
then got a burst. ``generate`` now runs in a worker thread with a
``TextIteratorStreamer`` and the SSE generator yields chunks as text arrives.

The wire format is unchanged: the same chunk fields, final chunk, ``[DONE]`` and
``finish_reason`` for the same generated text. Only the timing moves. Requests
whose output is post-processed on the full text (reasoning parser), or whose decode
loop is not a plain ``generate`` call (draft model, n-gram lookup, MoLE), keep the
generate-then-send path; each has a test pinning that.

The fake models below call the streamer the way ``transformers`` does: the prompt
first (dropped by ``skip_prompt``), then one ``put`` per new token, then ``end``.
"""

from __future__ import annotations

import contextvars
import json
import logging
import threading
import time
import types
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
        step_delay=0.0,
        use_streamer=True,
        stream_only=None,
        streamed_ids=None,
        on_generate=None,
    ):
        self.token_ids = list(token_ids)
        self.hold_after = hold_after
        self.fail_at = fail_at
        self.step_delay = step_delay
        self.use_streamer = use_streamer
        # Only the first ``stream_only`` tokens reach the streamer (the rest are still
        # returned), or ``streamed_ids`` reach it in place of the real ones.
        self.stream_only = stream_only
        self.streamed_ids = streamed_ids
        self.on_generate = on_generate
        self.gate = threading.Event()
        self.gate_timed_out = False
        self.generate_returned = threading.Event()
        self.stopped_early = False
        self.calls = []
        self.threads = []
        self.produced = 0

    def _streams(self, index):
        return self.stream_only is None or index < self.stream_only

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
                    raise RuntimeError("boom in generate")
                if self.step_delay:
                    time.sleep(self.step_delay)
                sequence.append(token)
                self.produced += 1
                if streamer is not None and self._streams(index):
                    shown = token if self.streamed_ids is None else self.streamed_ids[index]
                    streamer.put(torch.tensor([shown]))
                if self.hold_after is not None and index + 1 == self.hold_after:
                    # A generate() that is still running: the reply is not finished.
                    self.gate_timed_out = not self.gate.wait(timeout=3.0)
            if streamer is not None:
                streamer.end()
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
    for frame in frames[:-1]:
        assert set(frame) == {"id", "object", "created", "model", "choices"}
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

    kwargs = model.calls[0]
    assert type(kwargs["streamer"]).__name__ == "TextIteratorStreamer"
    assert model.threads[0].name == WORKER_NAME
    assert model.threads[0] is not threading.current_thread()
    assert not _wait_for_workers_to_exit()


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
    # The request log of the streamed call carries the same text and token count.
    record = writer.record.call_args.kwargs
    assert record["response"] == expected
    assert record["tokens"] == plain["usage"]["completion_tokens"] == len(ids)
    assert record["extra"] == {"stream": True}


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


def test_closing_the_stream_stops_generation_and_joins_the_worker():
    pieces = ["w"] + [" w"] * 399
    tokenizer, ids = _script(*pieces, eos=False)
    model = _ScriptedModel(ids, step_delay=0.01)
    stream = _stream(model, tokenizer, max_tokens=1000)

    next(stream)
    stream.close()

    # close() joins the worker, so generate() has returned by the time it comes back.
    assert model.generate_returned.is_set()
    assert model.stopped_early, "the stop criteria never saw the stop event"
    assert model.produced < len(ids)
    assert not model.threads[0].is_alive()
    assert not _worker_threads()


def test_closing_the_stream_releases_the_adapter_lock():
    pieces = ["w"] + [" w"] * 399
    tokenizer, ids = _script(*pieces, eos=False)
    lock = threading.Lock()
    model = _ScriptedModel(ids, step_delay=0.01)
    model.set_adapter = MagicMock()
    stream = _stream(
        model, tokenizer, max_tokens=1000,
        adapter_lock=lock, adapter_names={"a"}, requested_adapter="a",
    )

    next(stream)
    assert lock.locked()
    stream.close()

    assert not lock.locked()


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
