"""``soup serve`` gives every answer an id, logs v1 conversation records, takes feedback.

* every ``/v1/chat/completions`` and ``/v1/messages`` answer carries a fresh canonical
  UUID4 as body field ``response_id`` and header ``X-Response-Id`` (a stream: the header
  and every chunk); the OpenAI-style ``id`` stays as it was;
* ``--trace-log`` writes one conversation record per answer in the version 1 format of
  ``conversation.schema.json``, its ``id`` the response id;
* ``--feedback-log`` + ``POST /v1/feedback`` append feedback records in the format of
  ``feedback.schema.json``;
* flags the vLLM, SGLang and MII backends never act on are refused, not ignored.

Every line the server writes is validated against the packaged JSON Schema.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi")
jsonschema = pytest.importorskip("jsonschema")

from fastapi.testclient import TestClient  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

import soup_cli.commands.serve as serve  # noqa: E402
from soup_cli.cli import app as cli_app  # noqa: E402
from soup_cli.data.traces.parsers import parse_soup_serve  # noqa: E402
from soup_cli.monitoring.feedback_log import (  # noqa: E402
    MAX_BODY_BYTES,
    FeedbackLogWriter,
    FeedbackValidationError,
    build_record,
    validate_feedback,
)
from soup_cli.monitoring.trace_logger import TraceLogWriter  # noqa: E402
from soup_cli.utils import loop_formats  # noqa: E402
from soup_cli.utils.canary_router import CanaryPolicy, route  # noqa: E402
from soup_cli.utils.loop_state import LoopState, write_state  # noqa: E402
from tests.conftest import strip_ansi  # noqa: E402

runner = CliRunner()

MODEL_NAME = "test-model"
RESPONSE_ID_HEADER = "X-Response-Id"
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")
TOKEN = "s3cret-feedback-token"

# Secret shapes the trace log redacts. Built from parts so no scanner mistakes the file
# for a leaked credential.
HF_SECRET = "hf_" + "abcdef0123456789"
SK_SECRET = "sk-" + "abcdefghijklmnopqrstuvwxyz"
BEARER_SECRET = "Bearer " + "abc12345xyz"


def _plain(text: str) -> str:
    """ANSI-stripped, whitespace-collapsed CLI output, safe to substring-match."""
    return " ".join(strip_ansi(text).split())


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


def _validator(name: str):
    return jsonschema.Draft202012Validator(
        loop_formats.load_schema(name),
        format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER,
    )


def _schema_errors(name: str, record: dict) -> list[str]:
    return [error.message for error in _validator(name).iter_errors(record)]


def _read_lines(path) -> list[dict]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _conversation_records(path) -> list[dict]:
    """Every line of a trace log, each checked against conversation.schema.json."""
    records = _read_lines(path)
    for record in records:
        assert _schema_errors("conversation", record) == [], record
    return records


def _feedback_records(path) -> list[dict]:
    """Every line of a feedback log, each checked against feedback.schema.json."""
    records = _read_lines(path)
    for record in records:
        assert _schema_errors("feedback", record) == [], record
    return records


# ---------------------------------------------------------------------------
# App builders
# ---------------------------------------------------------------------------


class _AdapterModel:
    """A model whose adapter selection can be observed while it generates."""

    def __init__(self) -> None:
        self.current = None

    def set_adapter(self, name: str) -> None:
        self.current = name

    @contextmanager
    def disable_adapter(self):
        previous = self.current
        self.current = None
        try:
            yield
        finally:
            self.current = previous


def _fake_generate(model, tokenizer, messages, **kwargs):
    return "hello there", 5, 2, "stop"


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _client(
    cwd,
    monkeypatch,
    *,
    generate=_fake_generate,
    trace=True,
    feedback=True,
    host="127.0.0.1",
    **kwargs,
) -> TestClient:
    monkeypatch.setattr(serve, "_generate_response", generate)
    app = serve._create_app(
        model_obj=kwargs.pop("model_obj", MagicMock()),
        tokenizer=MagicMock(),
        device="cpu",
        model_name=MODEL_NAME,
        max_tokens_default=8,
        trace_log_writer=TraceLogWriter(str(cwd / "trace.jsonl")) if trace else None,
        feedback_log_writer=FeedbackLogWriter(str(cwd / "feedback.jsonl")) if feedback else None,
        host=host,
        **kwargs,
    )
    return TestClient(app, base_url=f"http://{host}")


def _chat(client, content="hi", **extra):
    body = {"messages": [{"role": "user", "content": content}], **extra}
    return client.post("/v1/chat/completions", json=body)


def _sse_chunks(text: str) -> list[dict]:
    chunks = []
    for line in text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            chunks.append(json.loads(line[len("data: "):]))
    return chunks


def _trace(cwd) -> list[dict]:
    return _conversation_records(cwd / "trace.jsonl")


def _feedback(cwd) -> list[dict]:
    return _feedback_records(cwd / "feedback.jsonl")


# ---------------------------------------------------------------------------
# 1. The response id
# ---------------------------------------------------------------------------


class TestResponseId:
    def test_body_field_is_a_canonical_lowercase_uuid4(self, cwd, monkeypatch):
        resp = _chat(_client(cwd, monkeypatch))

        assert resp.status_code == 200, resp.text
        response_id = resp.json()["response_id"]
        assert UUID_RE.fullmatch(response_id), response_id
        assert str(uuid.UUID(response_id)) == response_id

    def test_header_carries_the_same_id(self, cwd, monkeypatch):
        resp = _chat(_client(cwd, monkeypatch))

        assert resp.headers[RESPONSE_ID_HEADER] == resp.json()["response_id"]

    def test_the_openai_style_id_field_is_unchanged(self, cwd, monkeypatch):
        body = _chat(_client(cwd, monkeypatch)).json()

        assert re.fullmatch(r"chatcmpl-[0-9a-f]{8}", body["id"]), body["id"]
        assert body["id"] != body["response_id"]
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"] == {"role": "assistant", "content": "hello there"}
        assert body["usage"] == {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}

    def test_every_answer_gets_a_fresh_id(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch)

        ids = {_chat(client).json()["response_id"] for _ in range(5)}

        assert len(ids) == 5

    def test_ids_come_without_any_log_enabled(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch, trace=False, feedback=False)

        resp = _chat(client)

        assert resp.headers[RESPONSE_ID_HEADER] == resp.json()["response_id"]
        assert not (cwd / "trace.jsonl").exists()

    def test_a_failed_generation_returns_no_id(self, cwd, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("cuda out of memory")

        resp = _chat(_client(cwd, monkeypatch, generate=boom))

        assert resp.status_code == 500
        assert resp.json() == {"detail": "Internal server error"}
        assert RESPONSE_ID_HEADER not in resp.headers
        assert _trace(cwd) == []

    def test_the_header_is_readable_by_a_page_on_localhost(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch)

        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"Origin": "http://localhost:3000"},
        )

        assert resp.status_code == 200, resp.text
        exposed = resp.headers["access-control-expose-headers"].lower()
        assert RESPONSE_ID_HEADER.lower() in exposed


class TestStreamedResponseId:
    def _stream(self, client, **extra):
        return _chat(client, stream=True, **extra)

    def test_header_and_every_chunk_carry_the_id(self, cwd, monkeypatch):
        resp = self._stream(_client(cwd, monkeypatch))

        assert resp.status_code == 200, resp.text
        header_id = resp.headers[RESPONSE_ID_HEADER]
        assert UUID_RE.fullmatch(header_id), header_id
        chunks = _sse_chunks(resp.text)
        assert len(chunks) >= 3  # "hello", " there", the final chunk
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        assert {chunk["response_id"] for chunk in chunks} == {header_id}
        assert resp.text.rstrip().endswith("data: [DONE]")

    def test_the_openai_style_chunk_id_is_unchanged(self, cwd, monkeypatch):
        chunks = _sse_chunks(self._stream(_client(cwd, monkeypatch)).text)

        chunk_ids = {chunk["id"] for chunk in chunks}
        assert len(chunk_ids) == 1
        assert re.fullmatch(r"chatcmpl-[0-9a-f]{8}", chunk_ids.pop())
        assert chunks[0]["choices"][0]["delta"] == {"content": "hello"}

    def test_the_streamed_answer_is_logged_under_its_id(self, cwd, monkeypatch):
        resp = self._stream(_client(cwd, monkeypatch), conversation_id="chat-9")

        (record,) = _trace(cwd)
        assert record["id"] == resp.headers[RESPONSE_ID_HEADER]
        assert record["response"] == "hello there"
        assert record["conversation_id"] == "chat-9"
        assert record["tokens"] == {"prompt": 5, "completion": 2}

    def test_each_stream_gets_its_own_id(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch)

        first = self._stream(client).headers[RESPONSE_ID_HEADER]
        second = self._stream(client).headers[RESPONSE_ID_HEADER]

        assert first != second

    def test_a_failed_stream_is_not_logged(self, cwd, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("cuda out of memory")

        resp = self._stream(_client(cwd, monkeypatch, generate=boom))

        assert "Internal server error" in resp.text
        assert _trace(cwd) == []

    def test_the_record_exists_before_the_first_chunk_is_sent(self, cwd, monkeypatch):
        # A reader who stops halfway has the id already; feedback on it needs the record.
        monkeypatch.setattr(serve, "_generate_response", _fake_generate)
        writer = TraceLogWriter(str(cwd / "trace.jsonl"))
        stream = serve._stream_response(
            MagicMock(), MagicMock(), [{"role": "user", "content": "hi"}],
            max_tokens=8, temperature=0.7, top_p=0.9, model_name=MODEL_NAME,
            trace_log_writer=writer, started=0.0,
            response_id="3f2b8c1e-0d4a-4e6f-9a1b-2c3d4e5f6a7b",
        )

        first = next(stream)

        assert '"response_id": "3f2b8c1e-0d4a-4e6f-9a1b-2c3d4e5f6a7b"' in first
        (record,) = _trace(cwd)
        assert record["id"] == "3f2b8c1e-0d4a-4e6f-9a1b-2c3d4e5f6a7b"
        stream.close()

    def test_a_stream_without_a_given_id_makes_one(self, cwd, monkeypatch):
        monkeypatch.setattr(serve, "_generate_response", _fake_generate)

        chunks = list(serve._stream_response(
            MagicMock(), MagicMock(), [{"role": "user", "content": "hi"}],
            max_tokens=8, temperature=0.7, top_p=0.9, model_name=MODEL_NAME,
        ))

        ids = {c["response_id"] for c in _sse_chunks("".join(chunks))}
        assert len(ids) == 1
        assert UUID_RE.fullmatch(ids.pop())

    def test_a_stream_without_a_given_id_makes_a_new_one_each_time(self, cwd, monkeypatch):
        monkeypatch.setattr(serve, "_generate_response", _fake_generate)

        def stream_id() -> str:
            chunks = serve._stream_response(
                MagicMock(), MagicMock(), [{"role": "user", "content": "hi"}],
                max_tokens=8, temperature=0.7, top_p=0.9, model_name=MODEL_NAME,
            )
            return _sse_chunks("".join(chunks))[0]["response_id"]

        assert len({stream_id() for _ in range(4)}) == 4


class TestAnthropicMessagesRoute:
    BODY = {"model": MODEL_NAME, "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}

    def test_the_reply_carries_the_id_in_body_and_header(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch)

        resp = client.post("/v1/messages", json=self.BODY)

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert UUID_RE.fullmatch(body["response_id"]), body
        assert resp.headers[RESPONSE_ID_HEADER] == body["response_id"]
        # The Anthropic envelope is as it was: id is still the generated "chatcmpl-" one.
        assert re.fullmatch(r"chatcmpl-[0-9a-f]{8}", body["id"])
        assert body["type"] == "message"
        assert body["content"] == [{"type": "text", "text": "hello there"}]
        assert body["usage"] == {"input_tokens": 5, "output_tokens": 2}

    def test_the_answer_is_logged_under_that_id(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch)

        resp = client.post("/v1/messages", json=self.BODY)

        (record,) = _trace(cwd)
        assert record["id"] == resp.headers[RESPONSE_ID_HEADER]
        assert record["messages"] == [{"role": "user", "content": "hi"}]
        # No X-Conversation-Id header and /v1/messages has no body field for one.
        assert record["conversation_id"] is None

    @pytest.mark.parametrize("stream", [False, True])
    def test_the_conversation_header_is_read_as_on_chat_completions(
        self, cwd, monkeypatch, stream
    ):
        # #2630 made /v1/messages forward X-Conversation-Id (and X-User-Id), so the answer
        # is logged under the conversation the client named, as on /v1/chat/completions.
        client = _client(cwd, monkeypatch)

        client.post(
            "/v1/messages",
            json={**self.BODY, "stream": stream},
            headers={"X-Conversation-Id": "conv-header"},
        )

        assert _trace(cwd)[0]["conversation_id"] == "conv-header"

    def test_a_streamed_reply_has_the_header_and_the_id_in_message_start(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch)

        resp = client.post("/v1/messages", json={**self.BODY, "stream": True})

        assert resp.status_code == 200, resp.text
        header_id = resp.headers[RESPONSE_ID_HEADER]
        start = next(
            json.loads(line[len("data: "):])
            for line in resp.text.splitlines()
            if line.startswith("data: ") and '"message_start"' in line
        )
        assert start["message"]["response_id"] == header_id
        (record,) = _trace(cwd)
        assert record["id"] == header_id

    def test_a_chat_completion_and_a_message_get_different_ids(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch)

        chat_id = _chat(client).json()["response_id"]
        message_id = client.post("/v1/messages", json=self.BODY).json()["response_id"]

        assert chat_id != message_id
        assert {record["id"] for record in _trace(cwd)} == {chat_id, message_id}


# ---------------------------------------------------------------------------
# 2. The v1 trace record
# ---------------------------------------------------------------------------


class TestConversationRecord:
    def test_one_schema_valid_record_per_answer(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch)

        resp = _chat(client, "What is Soup?")

        (record,) = _trace(cwd)
        assert record["schema_version"] == 1
        assert record["id"] == resp.json()["response_id"]
        assert record["model"] == MODEL_NAME
        assert record["source"] == "soup-serve"
        assert record["messages"] == [{"role": "user", "content": "What is Soup?"}]
        assert record["response"] == "hello there"
        assert record["finish_reason"] == "stop"
        assert record["tokens"] == {"prompt": 5, "completion": 2}

    def test_unknown_things_are_null(self, cwd, monkeypatch):
        _chat(_client(cwd, monkeypatch))

        (record,) = _trace(cwd)
        assert record["conversation_id"] is None
        assert record["version"] is None
        assert record["adapter"] is None
        assert record["canary_bucket"] is None

    def test_ts_is_iso_8601_utc_and_latency_is_measured(self, cwd, monkeypatch):
        _chat(_client(cwd, monkeypatch))

        (record,) = _trace(cwd)
        assert re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,9})?\+00:00", record["ts"]
        ), record["ts"]
        assert isinstance(record["latency_ms"], float)
        assert record["latency_ms"] >= 0

    def test_the_whole_conversation_is_recorded_and_prompt_is_its_last_user_turn(
        self, cwd, monkeypatch
    ):
        client = _client(cwd, monkeypatch)
        messages = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
        ]

        client.post("/v1/chat/completions", json={"messages": messages})

        (record,) = _trace(cwd)
        assert record["messages"] == messages
        assert record["prompt"] == "second question"
        assert record["response"] == "hello there"  # the answer is not among the messages

    def test_every_role_the_schema_allows_is_logged(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch)
        messages = [
            {"role": role, "content": f"{role} text"}
            for role in ("system", "developer", "user", "assistant", "tool", "user")
        ]

        resp = client.post("/v1/chat/completions", json={"messages": messages})

        assert resp.status_code == 200, resp.text
        (record,) = _trace(cwd)
        assert [m["role"] for m in record["messages"]] == [m["role"] for m in messages]

    def test_a_role_outside_the_schema_is_answered_but_not_logged(
        self, cwd, monkeypatch, caplog
    ):
        client = _client(cwd, monkeypatch)
        messages = [{"role": "function", "content": "42"}, {"role": "user", "content": "hi"}]

        with caplog.at_level("WARNING", logger="soup_cli.monitoring.trace_logger"):
            resp = client.post("/v1/chat/completions", json={"messages": messages})

        assert resp.status_code == 200, resp.text
        assert UUID_RE.fullmatch(resp.json()["response_id"])
        assert _trace(cwd) == []
        assert "message role 'function' is not one of" in caplog.text
        assert resp.json()["response_id"] in caplog.text

    def test_no_user_message_leaves_the_legacy_prompt_empty(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch)

        client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "system", "content": "only a system prompt"}]},
        )

        (record,) = _trace(cwd)
        assert record["prompt"] == ""

    def test_finish_reason_length_is_recorded(self, cwd, monkeypatch):
        client = _client(
            cwd, monkeypatch, generate=lambda *a, **k: ("cut off", 4, 8, "length")
        )

        _chat(client)

        assert _trace(cwd)[0]["finish_reason"] == "length"

    def test_the_reasoning_parser_strips_before_the_answer_is_logged(self, cwd, monkeypatch):
        client = _client(
            cwd,
            monkeypatch,
            generate=lambda *a, **k: ("<think>secret plan</think>The answer", 4, 8, "stop"),
            reasoning_parser="deepseek-r1",
        )

        body = _chat(client).json()

        (record,) = _trace(cwd)
        assert body["choices"][0]["message"]["content"] == record["response"]
        assert "secret plan" not in record["response"]

    def test_no_trace_log_writes_nothing(self, cwd, monkeypatch):
        _chat(_client(cwd, monkeypatch, trace=False))

        assert not (cwd / "trace.jsonl").exists()


class TestConversationId:
    def test_from_the_request_body(self, cwd, monkeypatch):
        _chat(_client(cwd, monkeypatch), conversation_id="conv-body")

        assert _trace(cwd)[0]["conversation_id"] == "conv-body"

    def test_from_the_x_conversation_id_header(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch)

        client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"X-Conversation-Id": "conv-header"},
        )

        assert _trace(cwd)[0]["conversation_id"] == "conv-header"

    def test_the_body_wins_over_the_header(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch)

        client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "conversation_id": "conv-body",
            },
            headers={"X-Conversation-Id": "conv-header"},
        )

        assert _trace(cwd)[0]["conversation_id"] == "conv-body"

    def test_a_streamed_answer_takes_it_from_the_header_too(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch)

        client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
            headers={"X-Conversation-Id": "conv-stream"},
        )

        assert _trace(cwd)[0]["conversation_id"] == "conv-stream"

    @pytest.mark.parametrize("value", ["", "k" * 513], ids=["empty", "513 chars"])
    def test_a_header_value_the_schema_cannot_hold_is_recorded_as_null(
        self, cwd, monkeypatch, value
    ):
        client = _client(cwd, monkeypatch)

        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"X-Conversation-Id": value},
        )

        assert resp.status_code == 200, resp.text
        assert _trace(cwd)[0]["conversation_id"] is None

    def test_a_512_char_key_is_kept(self, cwd, monkeypatch):
        _chat(_client(cwd, monkeypatch), conversation_id="k" * 512)

        assert _trace(cwd)[0]["conversation_id"] == "k" * 512


class TestTokens:
    def test_a_count_the_engine_did_not_report_is_null_not_zero(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch, generate=lambda *a, **k: ("hi", None, None, "stop"))

        _chat(client, stream=True)

        (record,) = _trace(cwd)
        assert record["tokens"] == {"prompt": None, "completion": None}

    def test_each_count_is_judged_alone(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch, generate=lambda *a, **k: ("hi", None, 3, "stop"))

        _chat(client, stream=True)

        assert _trace(cwd)[0]["tokens"] == {"prompt": None, "completion": 3}

    def test_a_real_zero_is_kept(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch, generate=lambda *a, **k: ("", 0, 0, "stop"))

        _chat(client)

        assert _trace(cwd)[0]["tokens"] == {"prompt": 0, "completion": 0}

    @pytest.mark.parametrize("bad", [True, -1, 2.5, "7", [3]])
    def test_a_value_that_is_not_a_count_is_null(self, cwd, bad):
        writer = TraceLogWriter(str(cwd / "t.jsonl"))

        writer.record_conversation(
            response_id=str(uuid.uuid4()), model="m",
            messages=[{"role": "user", "content": "q"}], response="a",
            prompt_tokens=bad, completion_tokens=bad,
        )

        (record,) = _conversation_records(cwd / "t.jsonl")
        assert record["tokens"] == {"prompt": None, "completion": None}

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, True, "3", None])
    def test_a_latency_that_was_not_measured_is_null(self, cwd, bad):
        writer = TraceLogWriter(str(cwd / "t.jsonl"))

        writer.record_conversation(
            response_id=str(uuid.uuid4()), model="m",
            messages=[{"role": "user", "content": "q"}], response="a", latency_ms=bad,
        )

        (record,) = _conversation_records(cwd / "t.jsonl")
        assert record["latency_ms"] is None

    @pytest.mark.parametrize(
        ("given", "written"),
        [(3, 3), (1, 1), (0, None), (-2, None), (True, None), ("2", None), (None, None)],
    )
    def test_a_loop_version_is_kept_only_when_it_is_one_or_more(self, cwd, given, written):
        writer = TraceLogWriter(str(cwd / "t.jsonl"))

        writer.record_conversation(
            response_id=str(uuid.uuid4()), model="m",
            messages=[{"role": "user", "content": "q"}], response="a", version=given,
        )

        (record,) = _conversation_records(cwd / "t.jsonl")
        assert record["version"] == written

    @pytest.mark.parametrize("bad", ["", "x" * 65, 3, None])
    def test_a_finish_reason_the_schema_cannot_hold_is_null(self, cwd, bad):
        writer = TraceLogWriter(str(cwd / "t.jsonl"))

        writer.record_conversation(
            response_id=str(uuid.uuid4()), model="m",
            messages=[{"role": "user", "content": "q"}], response="a", finish_reason=bad,
        )

        (record,) = _conversation_records(cwd / "t.jsonl")
        assert record["finish_reason"] is None


class TestRedaction:
    def test_secrets_never_reach_the_trace_log(self, cwd, monkeypatch):
        client = _client(
            cwd,
            monkeypatch,
            generate=lambda *a, **k: (f"your key is {SK_SECRET}", 5, 2, "stop"),
        )
        messages = [
            {"role": "system", "content": f"Authorization: {BEARER_SECRET}."},
            {"role": "user", "content": f"my token is {HF_SECRET}"},
        ]

        client.post("/v1/chat/completions", json={"messages": messages})

        raw = (cwd / "trace.jsonl").read_text(encoding="utf-8")
        for secret in (HF_SECRET, SK_SECRET, "abc12345xyz"):
            assert secret not in raw, secret
        (record,) = _trace(cwd)
        assert record["messages"][0]["content"] == "Authorization: <redacted>."
        assert record["messages"][1]["content"] == "my token is <redacted>"
        assert record["response"] == "your key is <redacted>"
        assert record["prompt"] == "my token is <redacted>"

    def test_a_streamed_answer_is_redacted_too(self, cwd, monkeypatch):
        client = _client(
            cwd, monkeypatch, generate=lambda *a, **k: (f"key {SK_SECRET}", 5, 2, "stop")
        )

        _chat(client, f"token {HF_SECRET}", stream=True)

        raw = (cwd / "trace.jsonl").read_text(encoding="utf-8")
        assert HF_SECRET not in raw
        assert SK_SECRET not in raw
        assert _trace(cwd)[0]["response"] == "key <redacted>"


class TestTraceWriter:
    def test_the_version_0_entry_is_still_written_by_record(self, cwd):
        writer = TraceLogWriter(str(cwd / "t.jsonl"))

        writer.record(prompt="p", response="r", latency_ms=1.5, tokens=3)

        (line,) = _read_lines(cwd / "t.jsonl")
        assert "schema_version" not in line
        assert line["tokens"] == 3
        assert isinstance(line["ts"], float)

    def test_version_1_records_rotate_like_version_0_ones(self, cwd):
        path = cwd / "t.jsonl"
        writer = TraceLogWriter(str(path), cap_mb=1)
        messages = [{"role": "user", "content": "q"}]

        writer.record_conversation(
            response_id=str(uuid.uuid4()), model="m", messages=messages, response="x" * (1 << 20)
        )
        second = str(uuid.uuid4())
        writer.record_conversation(response_id=second, model="m", messages=messages, response="y")

        assert path.with_suffix(".jsonl.1").exists()
        assert [record["id"] for record in _conversation_records(path)] == [second]

    def test_a_path_outside_cwd_is_refused_by_flag_name(self, cwd):
        with pytest.raises(ValueError, match="--trace-log path must stay under cwd"):
            TraceLogWriter(str(cwd.parent / "elsewhere.jsonl"))

    def test_a_lone_surrogate_is_escaped_instead_of_raising(self, cwd):
        writer = TraceLogWriter(str(cwd / "t.jsonl"))

        writer.record_conversation(
            response_id=str(uuid.uuid4()), model="m",
            messages=[{"role": "user", "content": "q"}], response="broken \ud800 text",
        )

        (record,) = _conversation_records(cwd / "t.jsonl")
        assert record["response"] == "broken \ud800 text"

    def test_concurrent_writes_keep_every_line_whole(self, cwd):
        writer = TraceLogWriter(str(cwd / "t.jsonl"))
        ids = [str(uuid.uuid4()) for _ in range(40)]

        def write(response_id):
            writer.record_conversation(
                response_id=response_id, model="m",
                messages=[{"role": "user", "content": "q" * 2000}], response="a" * 2000,
            )

        threads = [threading.Thread(target=write, args=(rid,)) for rid in ids]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert sorted(r["id"] for r in _conversation_records(cwd / "t.jsonl")) == sorted(ids)

    def test_a_write_failure_never_reaches_the_client(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch)
        (cwd / "trace.jsonl").unlink(missing_ok=True)
        monkeypatch.setattr(
            TraceLogWriter, "_append", lambda self, entry: (_ for _ in ()).throw(OSError("disk"))
        )

        resp = _chat(client)

        assert resp.status_code == 200, resp.text
        assert UUID_RE.fullmatch(resp.json()["response_id"])

    def test_a_request_without_messages_is_not_logged(self, cwd, caplog):
        writer = TraceLogWriter(str(cwd / "t.jsonl"))

        with caplog.at_level("WARNING", logger="soup_cli.monitoring.trace_logger"):
            writer.record_conversation(
                response_id=str(uuid.uuid4()), model="m", messages=[], response="a",
            )

        assert _conversation_records(cwd / "t.jsonl") == []
        assert "the request has no messages" in caplog.text

    def test_an_answer_to_an_empty_message_list_still_carries_an_id(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch)

        resp = client.post("/v1/chat/completions", json={"messages": []})

        assert resp.status_code == 200, resp.text
        assert UUID_RE.fullmatch(resp.json()["response_id"])
        assert _trace(cwd) == []

    @pytest.mark.parametrize(
        ("field", "value"),
        [("model", ""), ("model", "m" * 513), ("adapter", ""), ("adapter", "a" * 257)],
    )
    def test_a_name_the_schema_cannot_hold_is_not_logged(self, cwd, field, value):
        writer = TraceLogWriter(str(cwd / "t.jsonl"))
        arguments = {"model": "m", "adapter": None, field: value}

        writer.record_conversation(
            response_id=str(uuid.uuid4()),
            messages=[{"role": "user", "content": "q"}], response="a", **arguments,
        )

        assert _conversation_records(cwd / "t.jsonl") == []

    @pytest.mark.parametrize(("field", "value"), [("model", "m" * 512), ("adapter", "a" * 256)])
    def test_a_name_at_the_schema_limit_is_logged(self, cwd, field, value):
        writer = TraceLogWriter(str(cwd / "t.jsonl"))
        arguments = {"model": "m", "adapter": None, field: value}

        writer.record_conversation(
            response_id=str(uuid.uuid4()),
            messages=[{"role": "user", "content": "q"}], response="a", **arguments,
        )

        (record,) = _conversation_records(cwd / "t.jsonl")
        assert record[field] == value

    @pytest.mark.parametrize("bucket", ["stable", "canary"])
    def test_a_known_canary_bucket_is_kept(self, cwd, bucket):
        writer = TraceLogWriter(str(cwd / "t.jsonl"))

        writer.record_conversation(
            response_id=str(uuid.uuid4()), model="m",
            messages=[{"role": "user", "content": "q"}], response="a", canary_bucket=bucket,
        )

        assert _conversation_records(cwd / "t.jsonl")[0]["canary_bucket"] == bucket

    @pytest.mark.parametrize("bucket", ["shadow", "", "STABLE", 1, None])
    def test_an_unknown_canary_bucket_is_null(self, cwd, bucket):
        writer = TraceLogWriter(str(cwd / "t.jsonl"))

        writer.record_conversation(
            response_id=str(uuid.uuid4()), model="m",
            messages=[{"role": "user", "content": "q"}], response="a", canary_bucket=bucket,
        )

        assert _conversation_records(cwd / "t.jsonl")[0]["canary_bucket"] is None


# ---------------------------------------------------------------------------
# The adapter and the canary bucket
# ---------------------------------------------------------------------------


class TestAdapterAndBucket:
    def _adapter_client(self, cwd, monkeypatch, observed, **kwargs):
        def generate(model, tokenizer, messages, **kw):
            observed.append(model.current)
            return "ok", 3, 1, "stop"

        return _client(
            cwd,
            monkeypatch,
            generate=generate,
            model_obj=_AdapterModel(),
            adapter_map={"chat": "unused", "code": "unused"},
            peft_adapter_names={"chat", "code"},
            **kwargs,
        )

    def test_the_base_model_is_recorded_as_no_adapter(self, cwd, monkeypatch):
        observed = []
        client = self._adapter_client(cwd, monkeypatch, observed)

        _chat(client)

        assert observed == [None]
        assert _trace(cwd)[0]["adapter"] is None

    def test_the_adapter_a_request_asks_for_is_recorded(self, cwd, monkeypatch):
        observed = []
        client = self._adapter_client(cwd, monkeypatch, observed)

        _chat(client, adapter="chat")

        assert observed == ["chat"]
        assert _trace(cwd)[0]["adapter"] == "chat"

    def test_the_activated_adapter_is_recorded(self, cwd, monkeypatch):
        observed = []
        client = self._adapter_client(cwd, monkeypatch, observed)
        assert client.post("/v1/adapters/activate/code").status_code == 200

        _chat(client)

        assert observed == ["code"]
        assert _trace(cwd)[0]["adapter"] == "code"

    def test_a_request_adapter_beats_the_activated_one(self, cwd, monkeypatch):
        observed = []
        client = self._adapter_client(cwd, monkeypatch, observed)
        client.post("/v1/adapters/activate/code")

        _chat(client, adapter="chat")

        assert observed == ["chat"]
        assert _trace(cwd)[0]["adapter"] == "chat"

    def test_each_answer_names_its_own_adapter(self, cwd, monkeypatch):
        observed = []
        client = self._adapter_client(cwd, monkeypatch, observed)

        _chat(client, adapter="chat")
        _chat(client, adapter="code")
        _chat(client)

        assert observed == ["chat", "code", None]
        assert [record["adapter"] for record in _trace(cwd)] == observed

    def test_a_streamed_answer_names_its_adapter(self, cwd, monkeypatch):
        observed = []
        client = self._adapter_client(cwd, monkeypatch, observed)

        _chat(client, adapter="code", stream=True)

        assert observed == ["code"]
        assert _trace(cwd)[0]["adapter"] == "code"

    def test_an_unknown_adapter_is_refused_and_not_logged(self, cwd, monkeypatch):
        observed = []
        client = self._adapter_client(cwd, monkeypatch, observed)

        resp = _chat(client, adapter="nope")

        assert resp.status_code == 404
        assert observed == []
        assert _trace(cwd) == []

    def test_a_model_that_cannot_switch_adapters_is_recorded_as_base(self, cwd, monkeypatch):
        # peft names loaded but the model has no set_adapter: _adapter_scope runs the base.
        model = MagicMock(spec=["generate"])
        client = _client(
            cwd, monkeypatch, model_obj=model,
            adapter_map={"chat": "unused"}, peft_adapter_names={"chat"},
        )

        _chat(client, adapter="chat")

        assert _trace(cwd)[0]["adapter"] is None

    def test_the_effective_adapter_rule_matches_the_adapter_scope_rule(self):
        names = {"a", "b"}
        model = _AdapterModel()
        for requested, active in [
            (None, None), (None, "a"), ("b", None), ("b", "a"), ("", "a"),
            ("zzz", "a"), (None, "zzz"),
        ]:
            with serve._adapter_scope(model, threading.Lock(), names, requested, active):
                ran = model.current
            assert serve._effective_adapter(model, names, requested, active) == ran, (
                requested, active,
            )
        assert serve._effective_adapter(model, set(), "a", "a") is None


class TestCanaryBucket:
    def _canary_client(self, cwd, monkeypatch, observed, stable="base"):
        state_path = cwd / ".soup" / "loop.yaml"
        write_state(
            LoopState(
                served_model=stable,
                eval_suite="evals.yaml",
                baseline="prod",
                canary_active="candidate",
                canary_traffic_pct=50.0,
                canary_rollout_id="rollout-ids",
            ),
            str(state_path),
        )

        def generate(model, tokenizer, messages, **kw):
            observed.append(model.current)
            return "ok", 3, 1, "stop"

        client = _client(
            cwd,
            monkeypatch,
            generate=generate,
            model_obj=_AdapterModel(),
            adapter_map={"candidate": "unused", "stable-ad": "unused"},
            peft_adapter_names={"candidate", "stable-ad"},
            canary_state_path=str(state_path),
            canary_stats_path=str(cwd / ".soup" / "canary-stats.json"),
        )
        return client, CanaryPolicy(stable=stable, canary="candidate", traffic_pct=50.0)

    def test_the_bucket_and_adapter_each_answer_was_routed_to_are_recorded(
        self, cwd, monkeypatch
    ):
        observed = []
        client, policy = self._canary_client(cwd, monkeypatch, observed)
        keys = [f"conversation-{index}" for index in range(40)]

        for key in keys:
            assert _chat(client, conversation_id=key).status_code == 200

        records = _trace(cwd)
        assert [record["conversation_id"] for record in records] == keys
        assert [record["canary_bucket"] for record in records] == [
            route(policy, key).bucket for key in keys
        ]
        assert {record["canary_bucket"] for record in records} == {"stable", "canary"}
        # The adapter named is the one that was selected while the model generated.
        assert [record["adapter"] for record in records] == observed
        for record in records:
            expected = "candidate" if record["canary_bucket"] == "canary" else None
            assert record["adapter"] == expected
        client.app.state.canary_outcomes.close()

    def test_a_stable_bucket_served_by_a_loaded_adapter_names_that_adapter(
        self, cwd, monkeypatch
    ):
        observed = []
        client, policy = self._canary_client(cwd, monkeypatch, observed, stable="stable-ad")

        for index in range(40):
            _chat(client, conversation_id=f"conversation-{index}")

        records = _trace(cwd)
        assert [record["adapter"] for record in records] == observed
        by_bucket = {record["canary_bucket"]: record["adapter"] for record in records}
        assert by_bucket == {"stable": "stable-ad", "canary": "candidate"}
        client.app.state.canary_outcomes.close()

    def test_a_conversation_from_the_header_is_routed_and_recorded(self, cwd, monkeypatch):
        observed = []
        client, policy = self._canary_client(cwd, monkeypatch, observed)
        keys = [f"header-conversation-{index}" for index in range(30)]

        for key in keys:
            client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
                headers={"X-Conversation-Id": key},
            )

        records = _trace(cwd)
        assert [record["conversation_id"] for record in records] == keys
        assert [record["canary_bucket"] for record in records] == [
            route(policy, key).bucket for key in keys
        ]
        client.app.state.canary_outcomes.close()

    def test_a_streamed_answer_records_its_bucket(self, cwd, monkeypatch):
        observed = []
        client, policy = self._canary_client(cwd, monkeypatch, observed)
        keys = [f"stream-conversation-{index}" for index in range(30)]

        for key in keys:
            _chat(client, conversation_id=key, stream=True)

        records = _trace(cwd)
        assert [record["canary_bucket"] for record in records] == [
            route(policy, key).bucket for key in keys
        ]
        assert [record["adapter"] for record in records] == observed
        client.app.state.canary_outcomes.close()

    def test_no_conversation_key_means_no_bucket(self, cwd, monkeypatch):
        observed = []
        client, _ = self._canary_client(cwd, monkeypatch, observed)

        _chat(client)

        record = _trace(cwd)[0]
        assert record["canary_bucket"] is None
        assert record["adapter"] is None
        client.app.state.canary_outcomes.close()

    def test_an_explicit_adapter_skips_routing_and_the_bucket(self, cwd, monkeypatch):
        observed = []
        client, _ = self._canary_client(cwd, monkeypatch, observed)

        _chat(client, conversation_id="conversation-1", adapter="stable-ad")

        record = _trace(cwd)[0]
        assert record["canary_bucket"] is None
        assert record["adapter"] == "stable-ad"
        client.app.state.canary_outcomes.close()


# ---------------------------------------------------------------------------
# The in-repo reader of the trace line
# ---------------------------------------------------------------------------


def _v1_line(**overrides) -> dict:
    line = {
        "schema_version": 1,
        "id": "11111111-2222-4333-8444-555555555555",
        "conversation_id": None,
        "ts": "2026-11-05T11:00:00.000+00:00",
        "model": "m",
        "version": None,
        "adapter": None,
        "canary_bucket": None,
        "messages": [{"role": "user", "content": "q1"}],
        "response": "a1",
        "finish_reason": "stop",
        "latency_ms": 3.0,
        "tokens": {"prompt": 4, "completion": 5},
        "source": "soup-serve",
        "prompt": "q1",
    }
    line.update(overrides)
    return line


def _write_traces(directory, name, *lines) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(
        "".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8"
    )


class TestParseSoupServe:
    def test_a_version_0_line_is_read_as_before(self, tmp_path):
        v0 = {"ts": 1.0, "prompt": "q0", "response": "a0", "latency_ms": 2.0, "tokens": 9}
        _write_traces(tmp_path / "t", "serve.jsonl", v0)

        (trace,) = parse_soup_serve(str(tmp_path / "t"))

        assert (trace.trace_id, trace.prompt, trace.output) == ("serve", "q0", "a0")

    def test_a_version_1_line_is_read_by_its_id(self, tmp_path):
        _write_traces(tmp_path / "t", "serve.jsonl", _v1_line())

        (trace,) = parse_soup_serve(str(tmp_path / "t"))

        assert trace.trace_id == "11111111-2222-4333-8444-555555555555"
        assert trace.prompt == "q1"
        assert trace.output == "a1"

    def test_the_legacy_prompt_wins_over_messages(self, tmp_path):
        _write_traces(
            tmp_path / "t", "serve.jsonl",
            _v1_line(prompt="legacy prompt", messages=[{"role": "user", "content": "other"}]),
        )

        (trace,) = parse_soup_serve(str(tmp_path / "t"))

        assert trace.prompt == "legacy prompt"

    def test_without_a_legacy_prompt_the_last_user_message_is_the_prompt(self, tmp_path):
        line = _v1_line(
            messages=[
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": "last question"},
                {"role": "assistant", "content": "trailing"},
            ]
        )
        del line["prompt"]
        _write_traces(tmp_path / "t", "serve.jsonl", line)

        (trace,) = parse_soup_serve(str(tmp_path / "t"))

        assert trace.prompt == "last question"

    def test_an_empty_legacy_prompt_falls_back_to_messages(self, tmp_path):
        _write_traces(
            tmp_path / "t", "serve.jsonl",
            _v1_line(prompt="", messages=[{"role": "user", "content": "from messages"}]),
        )

        (trace,) = parse_soup_serve(str(tmp_path / "t"))

        assert trace.prompt == "from messages"

    def test_a_v1_line_with_no_user_message_and_no_prompt_is_skipped(self, tmp_path):
        line = _v1_line(messages=[{"role": "system", "content": "s"}])
        del line["prompt"]
        _write_traces(tmp_path / "t", "serve.jsonl", line)

        assert list(parse_soup_serve(str(tmp_path / "t"))) == []

    @pytest.mark.parametrize("version", [2, 99, True, "1", None, 1.5])
    def test_a_line_of_a_version_this_reader_does_not_know_is_skipped(self, tmp_path, version):
        _write_traces(
            tmp_path / "t", "serve.jsonl",
            _v1_line(schema_version=version), _v1_line(id="22222222-2222-4333-8444-555555555555"),
        )

        traces = list(parse_soup_serve(str(tmp_path / "t")))

        assert [trace.trace_id for trace in traces] == ["22222222-2222-4333-8444-555555555555"]

    def test_version_0_and_version_1_lines_mix_in_one_directory(self, tmp_path):
        v0 = {"ts": 1.0, "prompt": "q0", "response": "a0", "latency_ms": 2.0, "tokens": 9}
        _write_traces(tmp_path / "t", "a-old.jsonl", v0)
        _write_traces(tmp_path / "t", "b-new.jsonl", _v1_line())

        traces = list(parse_soup_serve(str(tmp_path / "t")))

        assert [(t.trace_id, t.prompt) for t in traces] == [
            ("a-old", "q0"),
            ("11111111-2222-4333-8444-555555555555", "q1"),
        ]

    def test_what_the_server_writes_is_what_the_reader_reads(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch)
        ids = [_chat(client, f"question {n}").json()["response_id"] for n in range(3)]
        (cwd / "traces").mkdir()
        (cwd / "trace.jsonl").replace(cwd / "traces" / "serve.jsonl")

        traces = list(parse_soup_serve(str(cwd / "traces")))

        assert [t.trace_id for t in traces] == ids
        assert [t.prompt for t in traces] == [f"question {n}" for n in range(3)]
        assert {t.output for t in traces} == {"hello there"}

    def test_from_traces_counts_version_1_lines(self, cwd):
        v0 = {"ts": 1.0, "prompt": "q0", "response": "a0", "latency_ms": 2.0, "tokens": 9}
        _write_traces(cwd / "logs", "old.jsonl", v0)
        _write_traces(cwd / "logs", "new.jsonl", _v1_line())

        result = runner.invoke(
            cli_app,
            ["data", "from-traces", "--logs", "logs", "--format", "soup-serve",
             "--output", "prefs.jsonl"],
        )

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "Read 2 trace(s) but built no pairs" in _plain(result.output)


# ---------------------------------------------------------------------------
# 3. Feedback: validation
# ---------------------------------------------------------------------------

RESPONSE_ID = "0de3d93e-f426-4096-8ba2-14f7fb992421"


def _body(**fields) -> dict:
    return {"response_id": RESPONSE_ID, **fields}


VALID_BODIES = [
    pytest.param(_body(kind="thumb", value=1), id="thumb up"),
    pytest.param(_body(kind="thumb", value=-1), id="thumb down"),
    pytest.param(_body(kind="thumb", value=None), id="thumb cleared"),
    pytest.param(_body(kind="thumb", value=1.0), id="thumb 1.0"),
    pytest.param(_body(kind="rating", value=1), id="rating 1"),
    pytest.param(_body(kind="rating", value=5), id="rating 5"),
    pytest.param(_body(kind="rating", value=3.0), id="rating 3.0"),
    pytest.param(_body(kind="rating", value=None), id="rating cleared"),
    pytest.param(_body(kind="edit", edited_text="the corrected answer"), id="edit"),
    pytest.param(_body(kind="edit", edited_text=None), id="edit withdrawn"),
    pytest.param(_body(kind="comment", comment="too long"), id="comment"),
    pytest.param(_body(kind="comment", comment="c" * 10_000), id="comment at the limit"),
    pytest.param(_body(kind="thumb", value=-1, comment="wrong units"), id="thumb + comment"),
    pytest.param(_body(kind="rating", value=2, comment=None), id="null comment"),
    pytest.param(_body(kind="thumb", value=1, user_ref="local"), id="user_ref"),
    pytest.param(_body(kind="thumb", value=1, user_ref="u" * 256), id="user_ref at the limit"),
    pytest.param(_body(kind="thumb", value=1, user_ref=None), id="null user_ref"),
    pytest.param(_body(kind="edit", edited_text="x", comment="why", user_ref="a"), id="all"),
]

INVALID_BODIES = [
    # (body, the field the error names)
    pytest.param([], "body", id="array body"),
    pytest.param("thumb", "body", id="string body"),
    pytest.param(None, "body", id="null body"),
    pytest.param({"kind": "thumb", "value": 1}, "response_id", id="no response_id"),
    pytest.param(_body(kind="thumb", value=1) | {"response_id": 7}, "response_id", id="int id"),
    pytest.param(
        _body(kind="thumb", value=1) | {"response_id": RESPONSE_ID.upper()},
        "response_id", id="upper-case uuid",
    ),
    pytest.param(
        _body(kind="thumb", value=1) | {"response_id": "chatcmpl-6ac61b8f"},
        "response_id", id="the OpenAI-style id",
    ),
    pytest.param(
        _body(kind="thumb", value=1) | {"response_id": RESPONSE_ID.replace("-", "")},
        "response_id", id="uuid without dashes",
    ),
    pytest.param(
        _body(kind="thumb", value=1) | {"response_id": RESPONSE_ID + "\n"},
        "response_id", id="trailing newline",
    ),
    pytest.param({"response_id": RESPONSE_ID, "value": 1}, "kind", id="no kind"),
    pytest.param(_body(kind="like", value=1), "kind", id="unknown kind"),
    pytest.param(_body(kind="Thumb", value=1), "kind", id="kind case"),
    pytest.param(_body(kind=["thumb"], value=1), "kind", id="list kind"),
    pytest.param(_body(kind="thumb"), "value", id="thumb without value"),
    pytest.param(_body(kind="thumb", value=0), "value", id="thumb 0"),
    pytest.param(_body(kind="thumb", value=2), "value", id="thumb 2"),
    pytest.param(_body(kind="thumb", value="1"), "value", id="thumb string"),
    pytest.param(_body(kind="thumb", value=True), "value", id="thumb true"),
    pytest.param(_body(kind="thumb", value=1.5), "value", id="thumb 1.5"),
    pytest.param(_body(kind="thumb", value=[1]), "value", id="thumb list"),
    pytest.param(_body(kind="rating"), "value", id="rating without value"),
    pytest.param(_body(kind="rating", value=0), "value", id="rating 0"),
    pytest.param(_body(kind="rating", value=6), "value", id="rating 6"),
    pytest.param(_body(kind="rating", value=3.5), "value", id="rating 3.5"),
    pytest.param(_body(kind="rating", value="3"), "value", id="rating string"),
    pytest.param(_body(kind="rating", value=True), "value", id="rating true"),
    pytest.param(_body(kind="rating", value=float("nan")), "value", id="rating nan"),
    pytest.param(_body(kind="edit"), "edited_text", id="edit without text"),
    pytest.param(_body(kind="edit", edited_text=""), "edited_text", id="edit empty"),
    pytest.param(_body(kind="edit", edited_text=5), "edited_text", id="edit int"),
    pytest.param(_body(kind="comment"), "comment", id="comment without text"),
    pytest.param(_body(kind="comment", comment=None), "comment", id="comment null"),
    pytest.param(_body(kind="comment", comment=""), "comment", id="comment empty"),
    pytest.param(_body(kind="comment", comment=5), "comment", id="comment int"),
    pytest.param(
        _body(kind="comment", comment="c" * 10_001), "comment", id="comment over the limit"
    ),
    pytest.param(
        _body(kind="thumb", value=1, comment=""), "comment", id="empty comment next to a thumb"
    ),
    pytest.param(
        _body(kind="thumb", value=1, comment="c" * 10_001), "comment",
        id="long comment next to a thumb",
    ),
    pytest.param(_body(kind="thumb", value=1, user_ref=""), "user_ref", id="empty user_ref"),
    pytest.param(_body(kind="thumb", value=1, user_ref=3), "user_ref", id="int user_ref"),
    pytest.param(
        _body(kind="thumb", value=1, user_ref="u" * 257), "user_ref", id="long user_ref"
    ),
]


class TestValidateFeedback:
    @pytest.mark.parametrize("body", VALID_BODIES)
    def test_a_valid_body_becomes_a_schema_valid_record(self, body):
        record = build_record(validate_feedback(body))

        assert _schema_errors("feedback", record) == [], record
        assert record["response_id"] == RESPONSE_ID
        assert record["kind"] == body["kind"]
        assert record["schema_version"] == 1
        assert record["source"] == "soup-serve"
        assert UUID_RE.fullmatch(record["id"])
        assert record["id"] != RESPONSE_ID

    @pytest.mark.parametrize(("body", "field"), INVALID_BODIES)
    def test_a_body_that_breaks_the_schema_is_refused_naming_the_field(self, body, field):
        with pytest.raises(FeedbackValidationError) as excinfo:
            validate_feedback(body)

        assert excinfo.value.field == field

    @pytest.mark.parametrize(("body", "field"), INVALID_BODIES)
    def test_what_is_refused_would_fail_the_schema_or_is_not_a_record(self, body, field):
        # The rules are the schema's: a body this module refuses becomes an invalid
        # record when copied into one verbatim (the body check aside, which has no record).
        if field == "body":
            return
        if str(body.get("response_id")).endswith("\n"):
            # jsonschema matches patterns with Python's re.search, where `$` also matches
            # before a trailing newline; ECMA-262 (what the schema means) does not.
            # The writer follows ECMA-262, so this one case is stricter than the library.
            pytest.skip("Python's `$` accepts a trailing newline; ECMA-262 does not")
        record = {
            "schema_version": 1, "id": str(uuid.uuid4()), "ts": "2026-11-05T11:00:00+00:00",
            "source": "soup-serve", **body,
        }
        assert _schema_errors("feedback", record) != [], record

    def test_whole_numbers_are_written_as_ints(self):
        assert validate_feedback(_body(kind="thumb", value=1.0))["value"] == 1
        assert isinstance(validate_feedback(_body(kind="rating", value=4.0))["value"], int)

    def test_a_cleared_value_is_kept_as_null(self):
        fields = validate_feedback(_body(kind="thumb", value=None))

        assert "value" in fields and fields["value"] is None

    def test_value_is_ignored_for_edits_and_comments(self):
        edit = validate_feedback(_body(kind="edit", edited_text="x", value="anything"))
        comment = validate_feedback(_body(kind="comment", comment="x", value=99))

        assert "value" not in edit
        assert "value" not in comment

    def test_keys_the_schema_does_not_name_are_dropped(self):
        record = build_record(validate_feedback(
            _body(kind="thumb", value=1, id="forged", source="evil", ts="never", extra=1)
        ))

        assert record["id"] != "forged"
        assert record["source"] == "soup-serve"
        assert record["ts"] != "never"
        assert "extra" not in record

    def test_the_first_broken_field_is_named(self):
        assert _first_field({"response_id": "bad", "kind": "bad"}) == "response_id"
        assert _first_field(_body(kind="bad")) == "kind"
        assert _first_field(_body(kind="thumb", value=9, comment="", user_ref="")) == "value"
        assert _first_field(_body(kind="thumb", value=1, comment="", user_ref="")) == "comment"

    def test_secrets_in_free_text_are_redacted(self):
        record = build_record(validate_feedback(_body(
            kind="edit", edited_text=f"use {HF_SECRET} and {BEARER_SECRET}.", comment=SK_SECRET,
        )))

        assert record["edited_text"] == "use <redacted> and <redacted>."
        assert record["comment"] == "<redacted>"
        assert _schema_errors("feedback", record) == []


def _first_field(body) -> str:
    with pytest.raises(FeedbackValidationError) as excinfo:
        validate_feedback(body)
    return excinfo.value.field


class TestFeedbackLogWriter:
    def test_a_path_outside_cwd_is_refused_by_flag_name(self, cwd):
        with pytest.raises(ValueError, match="--feedback-log path must stay under cwd"):
            FeedbackLogWriter(str(cwd.parent / "elsewhere.jsonl"))

    @pytest.mark.parametrize("path", ["", "a\x00b.jsonl"])
    def test_an_empty_or_nul_path_is_refused(self, cwd, path):
        with pytest.raises(ValueError, match="non-empty string with no null bytes"):
            FeedbackLogWriter(path)

    @pytest.mark.requires_symlink
    @pytest.mark.parametrize("writer_class", [FeedbackLogWriter, TraceLogWriter])
    def test_a_symlink_that_leads_out_of_cwd_is_refused(self, cwd, monkeypatch, writer_class):
        project = cwd / "project"
        project.mkdir()
        outside = cwd / "outside"
        outside.mkdir()
        (project / "link.jsonl").symlink_to(outside / "target.jsonl")

        monkeypatch.chdir(project)
        with pytest.raises(ValueError, match="path must stay under cwd"):
            writer_class("link.jsonl")
        assert not (outside / "target.jsonl").exists()

    def test_the_parent_directory_is_created(self, cwd):
        writer = FeedbackLogWriter(str(cwd / ".soup-loops" / "feedback.jsonl"))

        assert writer.path.parent.is_dir()

    def test_lines_are_appended_never_rewritten(self, cwd):
        writer = FeedbackLogWriter(str(cwd / "fb.jsonl"))
        writer.append(build_record(validate_feedback(_body(kind="thumb", value=1))))
        first_bytes = (cwd / "fb.jsonl").read_bytes()

        writer.append(build_record(validate_feedback(_body(kind="thumb", value=-1))))

        assert (cwd / "fb.jsonl").read_bytes().startswith(first_bytes)
        assert [r["value"] for r in _feedback_records(cwd / "fb.jsonl")] == [1, -1]

    def test_the_log_is_not_rotated(self, cwd):
        writer = FeedbackLogWriter(str(cwd / "fb.jsonl"))
        big = build_record(validate_feedback(_body(kind="edit", edited_text="e" * 60_000)))

        for _ in range(20):  # 1.2 MB: past the trace log's smallest rotation cap
            writer.append(big)

        assert len(_feedback_records(cwd / "fb.jsonl")) == 20
        assert not (cwd / "fb.jsonl.1").exists()

    def test_concurrent_appends_keep_every_line_whole(self, cwd):
        writer = FeedbackLogWriter(str(cwd / "fb.jsonl"))
        records = [
            build_record(validate_feedback(_body(kind="comment", comment=f"{n} " * 3000)))
            for n in range(40)
        ]

        threads = [threading.Thread(target=writer.append, args=(rec,)) for rec in records]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        written = _feedback_records(cwd / "fb.jsonl")
        assert sorted(r["id"] for r in written) == sorted(r["id"] for r in records)

    def test_only_one_append_is_in_flight_at_a_time(self, cwd):
        # Whole lines survive without a lock only by luck of the platform's append
        # semantics; the lock is what makes it a guarantee. Watch the writes overlap.
        import time

        in_flight = []
        peak = []

        class _SlowHandle:
            def write(self, data):
                in_flight.append(1)
                peak.append(len(in_flight))
                time.sleep(0.02)
                in_flight.pop()

            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return False

        class _SlowPath:
            def open(self, mode):
                assert mode == "ab"
                return _SlowHandle()

        writer = FeedbackLogWriter(str(cwd / "fb.jsonl"))
        writer._path = _SlowPath()
        record = build_record(validate_feedback(_body(kind="thumb", value=1)))

        threads = [threading.Thread(target=writer.append, args=(record,)) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(peak) == 8
        assert max(peak) == 1

    def test_a_failed_write_raises(self, cwd):
        writer = FeedbackLogWriter(str(cwd / "gone" / "fb.jsonl"))
        (cwd / "gone").rmdir()

        with pytest.raises(OSError):
            writer.append(build_record(validate_feedback(_body(kind="thumb", value=1))))


# ---------------------------------------------------------------------------
# 3. Feedback: the route
# ---------------------------------------------------------------------------


def _post_feedback(client, body=None, *, raw=None, headers=None):
    if raw is not None:
        return client.post(
            "/v1/feedback", content=raw, headers={"Content-Type": "application/json",
                                                  **(headers or {})},
        )
    return client.post("/v1/feedback", json=body, headers=headers)


class TestFeedbackRoute:
    @pytest.mark.parametrize("body", VALID_BODIES)
    def test_a_valid_body_is_stored_and_acknowledged(self, cwd, monkeypatch, body):
        client = _client(cwd, monkeypatch)

        resp = _post_feedback(client, body)

        assert resp.status_code == 200, resp.text
        answer = resp.json()
        assert answer["status"] == "ok"
        assert UUID_RE.fullmatch(answer["id"])
        (record,) = _feedback(cwd)
        assert record["id"] == answer["id"]
        assert record["response_id"] == RESPONSE_ID
        assert record["kind"] == body["kind"]

    def test_each_kind_writes_the_fields_of_its_kind(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch)
        for body in (
            _body(kind="thumb", value=-1),
            _body(kind="rating", value=4, comment="nearly", user_ref="local"),
            _body(kind="edit", edited_text="fixed"),
            _body(kind="comment", comment="note"),
        ):
            assert _post_feedback(client, body).status_code == 200

        thumb, rating, edit, comment = _feedback(cwd)
        assert (thumb["value"], thumb["kind"]) == (-1, "thumb")
        assert (rating["value"], rating["comment"], rating["user_ref"]) == (4, "nearly", "local")
        assert edit["edited_text"] == "fixed"
        assert comment["comment"] == "note"
        assert "value" not in edit and "value" not in comment
        assert "edited_text" not in thumb

    def test_a_cleared_thumb_is_stored_as_null(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch)

        _post_feedback(client, _body(kind="thumb", value=None))

        (record,) = _feedback(cwd)
        assert "value" in record and record["value"] is None

    def test_every_post_appends_its_own_line_with_its_own_id(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch)

        ids = [
            _post_feedback(client, _body(kind="thumb", value=1)).json()["id"] for _ in range(4)
        ]

        assert [record["id"] for record in _feedback(cwd)] == ids
        assert len(set(ids)) == 4

    def test_ts_is_iso_8601_utc(self, cwd, monkeypatch):
        _post_feedback(_client(cwd, monkeypatch), _body(kind="thumb", value=1))

        assert re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,9})?\+00:00", _feedback(cwd)[0]["ts"]
        )

    def test_the_id_of_a_real_answer_round_trips(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch)
        response_id = _chat(client).json()["response_id"]

        _post_feedback(client, {"response_id": response_id, "kind": "thumb", "value": 1})

        assert _feedback(cwd)[0]["response_id"] == response_id
        assert _trace(cwd)[0]["id"] == response_id

    def test_the_response_id_is_not_checked_against_the_trace_log(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch, trace=False)

        resp = _post_feedback(client, _body(kind="thumb", value=1))

        assert resp.status_code == 200, resp.text

    def test_free_text_is_redacted_on_disk(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch)

        _post_feedback(client, _body(
            kind="edit", edited_text=f"key {SK_SECRET}", comment=f"{HF_SECRET} leaked",
        ))

        raw = (cwd / "feedback.jsonl").read_text(encoding="utf-8")
        assert SK_SECRET not in raw and HF_SECRET not in raw
        (record,) = _feedback(cwd)
        assert record["edited_text"] == "key <redacted>"
        assert record["comment"] == "<redacted> leaked"

    def test_non_ascii_text_is_kept(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch)

        _post_feedback(client, _body(kind="comment", comment="ответ неверный \U0001f600"))

        assert _feedback(cwd)[0]["comment"] == "ответ неверный \U0001f600"

    def test_a_lone_surrogate_in_the_body_does_not_break_the_log(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch)
        raw = ('{"response_id": "%s", "kind": "comment", "comment": "bad \\ud800 text"}'
               % RESPONSE_ID).encode()

        resp = _post_feedback(client, raw=raw)

        assert resp.status_code == 200, resp.text
        assert _feedback(cwd)[0]["comment"] == "bad \ud800 text"

    @pytest.mark.parametrize(("body", "field"), INVALID_BODIES)
    def test_an_invalid_body_is_a_400_naming_the_field_and_writes_nothing(
        self, cwd, monkeypatch, body, field
    ):
        client = _client(cwd, monkeypatch)
        raw = json.dumps(body).encode()

        resp = _post_feedback(client, raw=raw)

        assert resp.status_code == 400, resp.text
        assert resp.json() == {"detail": f"Invalid request: {field}"}
        assert _feedback(cwd) == []

    @pytest.mark.parametrize(
        "raw", [b"", b"not json", b"{", b'{"kind": "thumb"', b"\xff\xfe\x00", b"[" * 60_000],
        ids=["empty", "text", "truncated", "unterminated", "bad bytes", "deeply nested"],
    )
    def test_a_body_that_is_not_json_is_a_400(self, cwd, monkeypatch, raw):
        resp = _post_feedback(_client(cwd, monkeypatch), raw=raw)

        assert resp.status_code == 400, resp.text
        assert resp.json() == {"detail": "Invalid request: body"}
        assert _feedback(cwd) == []

    def test_the_error_never_echoes_what_was_sent(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch)

        resp = _post_feedback(client, _body(kind="comment", comment=HF_SECRET * 2000))

        assert resp.status_code == 400, resp.text
        assert HF_SECRET not in resp.text

    def test_a_failed_write_is_a_500_not_an_ok(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch)
        (cwd / "feedback.jsonl").unlink(missing_ok=True)
        writer = client.app.state.feedback_log_writer
        monkeypatch.setattr(writer, "_path", cwd / "gone" / "feedback.jsonl")

        resp = _post_feedback(client, _body(kind="thumb", value=1))

        assert resp.status_code == 500, resp.text
        assert resp.json() == {"detail": "Internal server error"}

    def test_thumbs_keeps_its_own_switch(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch)

        resp = client.post("/v1/thumbs", json={"prompt": "q", "response": "a", "thumb": "up"})

        assert resp.status_code == 404
        assert resp.json() == {"detail": "thumbs recording not enabled"}
        assert _feedback(cwd) == []

    def test_thumbs_still_records_to_its_database(self, cwd, monkeypatch):
        from soup_cli.utils.local_rl import init_local_rl_db

        init_local_rl_db("rl.db")
        client = _client(cwd, monkeypatch, record_thumbs_db="rl.db")

        resp = client.post("/v1/thumbs", json={"prompt": "q", "response": "a", "thumb": "up"})

        assert resp.status_code == 200, resp.text
        assert resp.json() == {"status": "ok", "thumb": "up"}
        assert _feedback(cwd) == []


class TestFeedbackDisabled:
    DETAIL = "feedback recording not enabled; start soup serve with --feedback-log"

    def test_without_the_flag_the_route_is_a_404(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch, feedback=False)

        resp = _post_feedback(client, _body(kind="thumb", value=1))

        assert resp.status_code == 404
        assert resp.json() == {"detail": self.DETAIL}

    def test_disabled_is_answered_before_the_body_is_read(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch, feedback=False)

        resp = _post_feedback(client, raw=b"not json")

        assert resp.status_code == 404
        assert resp.json() == {"detail": self.DETAIL}

    def test_the_app_state_holds_no_writer(self, cwd, monkeypatch):
        assert _client(cwd, monkeypatch, feedback=False).app.state.feedback_log_writer is None


class TestFeedbackAuth:
    def test_a_configured_token_is_required(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch, auth_token=TOKEN)
        body = _body(kind="thumb", value=1)

        missing = _post_feedback(client, body)
        wrong = _post_feedback(client, body, headers={"Authorization": "Bearer wrong"})
        right = _post_feedback(client, body, headers={"Authorization": f"Bearer {TOKEN}"})

        assert (missing.status_code, wrong.status_code, right.status_code) == (401, 401, 200)
        assert missing.json() == {"detail": "Invalid or missing bearer token"}
        assert len(_feedback(cwd)) == 1

    def test_thumbs_and_feedback_check_the_token_the_same_way(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch, auth_token=TOKEN)
        thumbs = client.post("/v1/thumbs", json={"prompt": "q", "response": "a", "thumb": "up"})
        feedback = _post_feedback(client, _body(kind="thumb", value=1))

        assert thumbs.status_code == feedback.status_code == 401
        assert thumbs.json() == feedback.json()
        assert thumbs.headers["www-authenticate"] == feedback.headers["www-authenticate"]

    def test_the_token_is_compared_in_constant_time(self, cwd, monkeypatch):
        import secrets

        calls = []
        real = secrets.compare_digest
        monkeypatch.setattr(
            secrets, "compare_digest", lambda a, b: calls.append((a, b)) or real(a, b)
        )
        client = _client(cwd, monkeypatch, auth_token=TOKEN)

        _post_feedback(
            client, _body(kind="thumb", value=1), headers={"Authorization": f"Bearer {TOKEN}"}
        )

        assert calls, "the bearer token was not compared with secrets.compare_digest"

    def test_the_token_is_checked_before_the_route_says_it_is_disabled(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch, feedback=False, auth_token=TOKEN)

        resp = _post_feedback(client, _body(kind="thumb", value=1))

        assert resp.status_code == 401

    def test_the_token_is_checked_before_the_body_is_read(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch, auth_token=TOKEN)

        resp = _post_feedback(client, raw=b"not json")

        assert resp.status_code == 401

    def test_a_lan_bind_without_a_token_refuses(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch, host="192.168.1.5")

        resp = _post_feedback(client, _body(kind="thumb", value=1))

        assert resp.status_code == 401
        assert resp.json() == {"detail": "Authentication required on non-loopback host"}
        assert _feedback(cwd) == []

    def test_a_lan_bind_with_the_token_accepts(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch, host="192.168.1.5", auth_token=TOKEN)

        resp = _post_feedback(
            client, _body(kind="thumb", value=1), headers={"Authorization": f"Bearer {TOKEN}"}
        )

        assert resp.status_code == 200, resp.text

    def test_a_foreign_host_is_refused_even_with_the_token(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch, auth_token=TOKEN)

        resp = _post_feedback(
            client, _body(kind="thumb", value=1),
            headers={"Host": "evil.example", "Authorization": f"Bearer {TOKEN}"},
        )

        assert resp.status_code == 421
        assert _feedback(cwd) == []

    def test_a_foreign_origin_is_refused(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch)

        resp = _post_feedback(
            client, _body(kind="thumb", value=1), headers={"Origin": "http://evil.example"}
        )

        assert resp.status_code == 403
        assert _feedback(cwd) == []

    def test_a_loopback_origin_is_accepted(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch)

        resp = _post_feedback(
            client, _body(kind="thumb", value=1), headers={"Origin": "http://localhost:3000"}
        )

        assert resp.status_code == 200, resp.text


class TestFeedbackBodyCap:
    def _body_of(self, size: int) -> bytes:
        """A valid feedback body of exactly ``size`` bytes."""
        head = ('{"response_id": "%s", "kind": "edit", "edited_text": "' % RESPONSE_ID).encode()
        tail = b'"}'
        raw = head + b"e" * (size - len(head) - len(tail)) + tail
        assert len(raw) == size
        return raw

    def test_the_cap_is_64_kib(self):
        assert MAX_BODY_BYTES == 64 * 1024

    def test_a_body_of_exactly_the_cap_is_accepted(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch)

        resp = _post_feedback(client, raw=self._body_of(MAX_BODY_BYTES))

        assert resp.status_code == 200, resp.text
        assert len(_feedback(cwd)[0]["edited_text"]) > 65_000

    def test_one_byte_more_is_a_413(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch)

        resp = _post_feedback(client, raw=self._body_of(MAX_BODY_BYTES + 1))

        assert resp.status_code == 413, resp.text
        assert resp.json() == {"detail": "Request body too large"}
        assert _feedback(cwd) == []

    def test_a_chunked_upload_without_a_length_is_capped_by_the_bytes_received(
        self, cwd, monkeypatch
    ):
        client = _client(cwd, monkeypatch)
        raw = self._body_of(MAX_BODY_BYTES + 1)

        def chunks():
            for start in range(0, len(raw), 8192):
                yield raw[start:start + 8192]

        resp = client.post(
            "/v1/feedback", content=chunks(), headers={"Content-Type": "application/json"}
        )

        assert resp.status_code == 413, resp.text
        assert _feedback(cwd) == []

    def test_a_chunked_upload_within_the_cap_is_accepted(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch)
        raw = self._body_of(20_000)

        def chunks():
            for start in range(0, len(raw), 4096):
                yield raw[start:start + 4096]

        resp = client.post(
            "/v1/feedback", content=chunks(), headers={"Content-Type": "application/json"}
        )

        assert resp.status_code == 200, resp.text

    def test_a_declared_length_over_the_cap_is_refused_without_reading_the_body(self):
        import asyncio

        from fastapi import HTTPException

        class _Request:
            headers = {"content-length": str(MAX_BODY_BYTES + 1)}

            async def stream(self):
                raise AssertionError("the body was read although its declared length is too big")
                yield b""  # pragma: no cover - makes this an async generator

        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(serve._read_capped_body(_Request(), MAX_BODY_BYTES))

        assert excinfo.value.status_code == 413

    def test_a_length_header_that_is_not_a_number_does_not_skip_the_byte_count(self):
        import asyncio

        from fastapi import HTTPException

        class _Request:
            headers = {"content-length": "lots"}

            async def stream(self):
                yield b"x" * (MAX_BODY_BYTES + 1)

        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(serve._read_capped_body(_Request(), MAX_BODY_BYTES))

        assert excinfo.value.status_code == 413

    def test_the_cap_is_judged_after_the_token(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch, auth_token=TOKEN)

        resp = _post_feedback(client, raw=self._body_of(MAX_BODY_BYTES + 1))

        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# 4 + 5. The command line
# ---------------------------------------------------------------------------


class TestServeHelp:
    def test_help_lists_feedback_log_with_its_description(self, monkeypatch):
        monkeypatch.setenv("COLUMNS", "200")

        result = runner.invoke(cli_app, ["serve", "--help"])

        assert result.exit_code == 0, (result.output, repr(result.exception))
        out = _plain(result.output.replace("│", " "))
        assert "--feedback-log" in out
        assert (
            "Append feedback (POST /v1/feedback) as JSONL records to this path; "
            "must stay under cwd."
        ) in out


class TestBackendRefusals:
    FLAGS = [
        ("--trace-log", "trace.jsonl"),
        ("--record-thumbs", "rl.db"),
        ("--feedback-log", "feedback.jsonl"),
    ]

    @pytest.mark.parametrize("backend", ["vllm", "sglang", "mii"])
    @pytest.mark.parametrize(("flag", "value"), FLAGS)
    def test_a_flag_the_backend_would_ignore_is_refused(self, cwd, backend, flag, value):
        result = runner.invoke(
            cli_app, ["serve", "--model", "does-not-exist", "--backend", backend, flag, value]
        )

        assert result.exit_code == 2, (result.output, repr(result.exception))
        out = _plain(result.output)
        assert f"{flag} requires --backend transformers" in out
        assert f"the {backend} backend does not support it" in out
        assert not (cwd / value).exists()

    def test_the_backend_name_is_case_insensitive(self, cwd):
        result = runner.invoke(
            cli_app,
            ["serve", "--model", "x", "--backend", "VLLM", "--feedback-log", "fb.jsonl"],
        )

        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "--feedback-log requires --backend transformers" in _plain(result.output)

    def test_the_first_ignored_flag_is_named(self, cwd):
        result = runner.invoke(
            cli_app,
            ["serve", "--model", "x", "--backend", "vllm", "--feedback-log", "fb.jsonl",
             "--trace-log", "t.jsonl"],
        )

        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "--trace-log requires --backend transformers" in _plain(result.output)

    @pytest.mark.parametrize(("flag", "value"), FLAGS)
    def test_without_the_flag_the_backend_is_not_refused_for_it(self, cwd, flag, value):
        result = runner.invoke(cli_app, ["serve", "--model", "does-not-exist", "--backend", "vllm"])

        assert "requires --backend transformers" not in _plain(result.output)

    def test_an_unknown_backend_is_reported_as_such_first(self, cwd):
        result = runner.invoke(
            cli_app,
            ["serve", "--model", "x", "--backend", "tgi", "--feedback-log", "fb.jsonl"],
        )

        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert "Unknown backend: tgi" in _plain(result.output)


class _Launched:
    """What ``soup serve`` hands the app and uvicorn once the model is 'loaded'."""

    def __init__(self) -> None:
        self.app_kwargs: dict = {}
        self.ran = False


@pytest.fixture
def launched(cwd, monkeypatch):
    """Run the real ``serve`` command with the model load and the server stubbed out."""
    import uvicorn

    state = _Launched()
    (cwd / "model").mkdir()

    def fake_create_app(**kwargs):
        state.app_kwargs = kwargs
        return object()

    monkeypatch.setattr(serve, "_create_app", fake_create_app)
    monkeypatch.setattr(serve, "_load_model", lambda **kwargs: (MagicMock(), MagicMock()))
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: setattr(state, "ran", True))
    return state


class TestServeStartup:
    def _serve(self, *args):
        return runner.invoke(
            cli_app, ["serve", "--model", "model", "--device", "cpu", *args]
        )

    def test_the_flag_builds_a_feedback_writer_at_that_path(self, cwd, launched):
        result = self._serve("--feedback-log", ".soup-loops/feedback.jsonl")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        writer = launched.app_kwargs["feedback_log_writer"]
        assert isinstance(writer, FeedbackLogWriter)
        assert writer.path == (cwd / ".soup-loops" / "feedback.jsonl").resolve()
        assert launched.ran

    def test_without_the_flag_there_is_no_writer(self, cwd, launched):
        result = self._serve()

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert launched.app_kwargs["feedback_log_writer"] is None
        assert launched.app_kwargs["trace_log_writer"] is None

    def test_both_logs_can_be_on_together(self, cwd, launched):
        result = self._serve("--trace-log", "traces/t.jsonl", "--feedback-log", "fb.jsonl")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert isinstance(launched.app_kwargs["trace_log_writer"], TraceLogWriter)
        assert isinstance(launched.app_kwargs["feedback_log_writer"], FeedbackLogWriter)

    def test_a_path_outside_cwd_is_refused_by_flag_name(self, cwd, launched):
        result = self._serve("--feedback-log", "../outside.jsonl")

        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert "--feedback-log path must stay under cwd" in _plain(result.output)
        assert not launched.ran
        assert not (cwd.parent / "outside.jsonl").exists()
