"""Decoded response limits exercise httpx streaming, then each production caller."""

from __future__ import annotations

import gzip
import json
import logging
from contextlib import contextmanager
from typing import Any, Iterator

import httpx
import pytest


class _Body(httpx.SyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.reads = 0
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self.chunks:
            self.reads += 1
            yield chunk

    def close(self) -> None:
        self.closed = True


def _stream_reply(
    monkeypatch: pytest.MonkeyPatch,
    chunks: list[bytes],
    *,
    headers: dict[str, str] | None = None,
    status: int = 200,
) -> tuple[_Body, list[tuple[str, str, dict[str, Any]]]]:
    body = _Body(chunks)
    calls: list[tuple[str, str, dict[str, Any]]] = []

    @contextmanager
    def stream(method: str, url: str, **kwargs: Any) -> Iterator[httpx.Response]:
        calls.append((method, url, kwargs))
        response = httpx.Response(
            status,
            headers=headers,
            stream=body,
            request=httpx.Request(method, url),
        )
        try:
            yield response
        finally:
            response.close()

    monkeypatch.setattr(httpx, "stream", stream)
    monkeypatch.setattr(httpx, "post", lambda *a, **k: pytest.fail("uncapped httpx.post"))
    return body, calls


@pytest.mark.parametrize("header", [None, "8", "0", "nonsense", "-1"])
def test_exact_decoded_boundary_is_buffered(
    monkeypatch: pytest.MonkeyPatch,
    header: str | None,
) -> None:
    from soup_cli.utils.http_response import post_capped

    headers = {"Retry-After": "2", "Content-Type": "application/json"}
    if header is not None:
        headers["Content-Length"] = header
    body, calls = _stream_reply(monkeypatch, [b'{"a":', b"12}"], headers=headers, status=429)
    response = post_capped(
        "https://example.com/api",
        max_bytes=8,
        endpoint_label="--api-base",
        timeout=3.0,
    )

    assert response.status_code == 429
    assert response.json() == {"a": 12}
    assert response.headers["Retry-After"] == "2"
    assert response.headers["Content-Length"] == "8"
    assert response.request.url == "https://example.com/api"
    assert body.closed
    assert calls == [("POST", "https://example.com/api", {"timeout": 3.0})]


def test_content_length_refuses_without_reading_and_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    from soup_cli.utils.http_response import ResponseTooLargeError, post_capped

    body, _ = _stream_reply(monkeypatch, [b"secret body"], headers={"Content-Length": "9"})
    with pytest.raises(ResponseTooLargeError, match=r"--api-base.*8 bytes"):
        post_capped("https://example.com", max_bytes=8, endpoint_label="--api-base")
    assert body.reads == 0
    assert body.closed


def test_unknown_length_stops_before_remaining_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    from soup_cli.utils.http_response import ResponseTooLargeError, post_capped

    body, _ = _stream_reply(monkeypatch, [b"a" * 65536, b"b" * 65536, b"unread"])
    with pytest.raises(ResponseTooLargeError, match="65536 bytes"):
        post_capped("https://example.com", max_bytes=65536, endpoint_label="--base-url")
    assert body.reads == 2
    assert body.closed


@pytest.mark.parametrize("size", [64, 65], ids=["at-cap", "above-cap"])
def test_gzip_limit_counts_decoded_bytes(monkeypatch: pytest.MonkeyPatch, size: int) -> None:
    from soup_cli.utils.http_response import ResponseTooLargeError, post_capped

    payload = b"a" * size
    compressed = gzip.compress(payload)
    assert len(compressed) < 64
    body, _ = _stream_reply(
        monkeypatch,
        [compressed],
        headers={
            "Content-Encoding": "gzip",
            "Content-Length": str(len(compressed)),
        },
    )
    if size > 64:
        with pytest.raises(ResponseTooLargeError, match="64 bytes"):
            post_capped("https://example.com", max_bytes=64, endpoint_label="--api-base")
    else:
        response = post_capped("https://example.com", max_bytes=64, endpoint_label="--api-base")
        assert response.content == payload
        assert "Content-Encoding" not in response.headers
        assert response.headers["Content-Length"] == "64"
    assert body.closed


def test_stream_error_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    from soup_cli.utils.http_response import post_capped

    class BrokenBody(_Body):
        def __iter__(self) -> Iterator[bytes]:
            yield b"first"
            raise httpx.ReadError("connection ended")

    body = BrokenBody([])

    @contextmanager
    def stream(*args: Any, **kwargs: Any) -> Iterator[httpx.Response]:
        response = httpx.Response(200, stream=body)
        try:
            yield response
        finally:
            response.close()

    monkeypatch.setattr(httpx, "stream", stream)
    with pytest.raises(httpx.ReadError, match="connection ended"):
        post_capped("https://example.com", max_bytes=64, endpoint_label="--api-base")
    assert body.closed


TARGETS = [
    "judge",
    "ollama",
    "vllm",
    "anthropic",
    "openai",
    "server",
    "magpie-ollama",
    "magpie-vllm",
]


def _call_target(target: str) -> Any:
    if target == "judge":
        from soup_cli.eval.judge import JudgeEvaluator

        return JudgeEvaluator(
            provider="server", model="m", api_base="http://localhost:8000"
        )._call_llm("p")
    if target.startswith("magpie-"):
        from soup_cli.utils.magpie import make_magpie_generate_fn

        return make_magpie_generate_fn(target.removeprefix("magpie-"), model="m")("p")
    common = dict(prompt="p", count=1, fmt="alpaca", model_name="m", temperature=0.0)
    if target == "ollama":
        from soup_cli.data.providers.ollama import generate_ollama

        return generate_ollama(**common, base_url="http://localhost:11434", generation_prompt="p")
    if target == "vllm":
        from soup_cli.data.providers.vllm import generate_vllm

        return generate_vllm(**common, base_url="http://localhost:8000", generation_prompt="p")
    if target == "anthropic":
        from soup_cli.data.providers.anthropic import generate_anthropic

        return generate_anthropic(**common, generation_prompt="p")
    from soup_cli.commands.generate import _generate_openai, _generate_server

    if target == "openai":
        return _generate_openai(**common, api_key="synthetic-key", api_base=None, seed_examples=[])
    return _generate_server(**common, api_base="http://localhost:8000", seed_examples=[])


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("compressed", [False, True], ids=["plain", "gzip"])
def test_every_target_limits_oversize_before_parsing(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    target: str,
    compressed: bool,
) -> None:
    from soup_cli.eval.judge import JudgeUnavailableError
    from soup_cli.utils import http_response, magpie

    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-secret-key")
    monkeypatch.setattr(http_response, "JUDGE_RESPONSE_LIMIT_BYTES", 64)
    monkeypatch.setattr(http_response, "GENERATION_RESPONSE_LIMIT_BYTES", 64)
    monkeypatch.setattr(magpie, "_MAX_RESPONSE_BYTES", 64)
    payload = b"secret-body-" * 20
    headers = {"Content-Encoding": "gzip"} if compressed else {}
    body, calls = _stream_reply(
        monkeypatch, [gzip.compress(payload) if compressed else payload], headers=headers
    )

    expected_error = (
        JudgeUnavailableError if target == "judge" else http_response.ResponseTooLargeError
    )
    with caplog.at_level(logging.DEBUG):
        if target.startswith("magpie-"):
            assert _call_target(target) == ""
            error_text = ""
        else:
            with pytest.raises(expected_error, match=r"64 bytes") as caught:
                _call_target(target)
            error_text = str(caught.value)
            assert (
                "--api-base" in error_text
                or "--base-url" in error_text
                or "--provider anthropic" in error_text
            )
    assert "secret-body" not in error_text + caplog.text
    assert "synthetic-secret-key" not in error_text + caplog.text
    assert len(calls) == 1  # An oversized judge response is not a transient outage.
    assert body.closed


@pytest.mark.parametrize("provider", ["ollama", "vllm"])
@pytest.mark.parametrize("compressed", [False, True], ids=["plain", "gzip"])
@pytest.mark.parametrize("oversized_stage", ["instruction", "response", "all"])
def test_magpie_drops_oversized_sample_and_respects_attempt_bound(
    monkeypatch: pytest.MonkeyPatch, tmp_path, caplog: pytest.LogCaptureFixture,
    provider: str, compressed: bool, oversized_stage: str,
) -> None:
    from soup_cli.utils import magpie

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(magpie, "_MAX_RESPONSE_BYTES", 64)

    def reply(text: str) -> bytes:
        payload = {"response": text} if provider == "ollama" else {"choices": [{"text": text}]}
        return json.dumps(payload).encode()

    oversized = b"secret-body-" * 20
    payloads = {
        "instruction": [oversized, reply("q2"), reply("a2")],
        "response": [reply("q1"), oversized, reply("q2"), reply("a2")],
        "all": [oversized, oversized],
    }[oversized_stage]
    bodies = [_Body([gzip.compress(p) if compressed else p]) for p in payloads]
    calls: list[dict[str, Any]] = []

    @contextmanager
    def stream(method: str, url: str, **kwargs: Any) -> Iterator[httpx.Response]:
        body = bodies[len(calls)]
        calls.append(kwargs)
        response = httpx.Response(
            200, stream=body, request=httpx.Request(method, url),
            headers={"Content-Encoding": "gzip"} if compressed else {},
        )
        try:
            yield response
        finally:
            response.close()

    monkeypatch.setattr(httpx, "stream", stream)
    monkeypatch.setattr(httpx, "post", lambda *a, **kw: pytest.fail("uncapped httpx.post"))
    config = magpie.MagpieConfig("Qwen/Qwen2.5-7B-Instruct", provider, 1, False)
    with caplog.at_level(logging.DEBUG):
        result = magpie.run_magpie(config, output_path="rows.jsonl", max_attempts=2)
    assert result.attempts == 2
    assert result.rows_kept == (0 if oversized_stage == "all" else 1)
    assert len(calls) == len(bodies)
    assert all(body.closed for body in bodies)
    assert "secret-body" not in caplog.text
    rows = [json.loads(line) for line in (tmp_path / "rows.jsonl").read_text().splitlines()]
    assert rows == ([] if oversized_stage == "all" else [{"messages": [
        {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"},
    ]}])


@pytest.mark.parametrize("target", TARGETS)
def test_every_target_preserves_valid_reply(monkeypatch: pytest.MonkeyPatch, target: str) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-secret-key")
    generated = '[{"instruction": "q", "output": "a"}]'
    payload = {
        "choices": [{"message": {"content": generated}, "text": generated}],
        "response": generated,
        "content": [{"type": "text", "text": generated}],
    }
    body, calls = _stream_reply(monkeypatch, [json.dumps(payload).encode()])
    result = _call_target(target)
    assert result == (
        generated
        if target == "judge" or target.startswith("magpie-")
        else [{"instruction": "q", "output": "a"}]
    )
    assert len(calls) == 1
    assert body.closed


def test_limits_and_judge_url_redaction(monkeypatch: pytest.MonkeyPatch) -> None:
    from soup_cli.eval.judge import JudgeUnavailableError, _judge_request
    from soup_cli.utils.http_response import (
        GENERATION_RESPONSE_LIMIT_BYTES,
        JUDGE_RESPONSE_LIMIT_BYTES,
    )

    assert JUDGE_RESPONSE_LIMIT_BYTES == 1024 * 1024
    assert GENERATION_RESPONSE_LIMIT_BYTES == 8 * 1024 * 1024
    body, calls = _stream_reply(
        monkeypatch, [], headers={"Content-Length": str(JUDGE_RESPONSE_LIMIT_BYTES + 1)}
    )
    with pytest.raises(JudgeUnavailableError) as caught:
        _judge_request(
            "https://user:password@example.com/api?api_key=query-secret#fragment", {}, {}
        )
    text = str(caught.value)
    assert "1048576 bytes" in text
    assert "--api-base" in text
    assert caught.value.retries_spent is False
    assert "password" not in text
    assert "query-secret" not in text
    assert "fragment" not in text
    assert body.reads == 0
    assert body.closed
    assert len(calls) == 1


@pytest.mark.parametrize("target", ["ollama", "vllm", "anthropic", "openai", "server"])
def test_provider_error_does_not_log_response_body(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, target: str,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-secret-key")
    body, _ = _stream_reply(monkeypatch, [b"secret-response-body"], status=500)
    with caplog.at_level(logging.DEBUG), pytest.raises(ValueError, match="500") as caught:
        _call_target(target)
    assert "secret-response-body" not in str(caught.value) + caplog.text
    assert "synthetic-secret-key" not in str(caught.value) + caplog.text
    assert body.closed


def test_judge_retry_preserves_streamed_status_and_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    from soup_cli.eval import judge

    delays: list[float] = []
    monkeypatch.setattr(judge, "_sleep", delays.append)
    bodies = [
        _Body([b'{"error": "busy"}']),
        _Body([b'{"choices": [{"message": {"content": "ok"}}]}']),
    ]
    attempts: list[str] = []

    @contextmanager
    def stream(method: str, url: str, **kwargs: Any) -> Iterator[httpx.Response]:
        index = len(attempts)
        attempts.append(url)
        response = httpx.Response(
            429 if index == 0 else 200,
            headers={"Retry-After": "2"}, stream=bodies[index],
            request=httpx.Request(method, url),
        )
        try:
            yield response
        finally:
            response.close()

    monkeypatch.setattr(httpx, "stream", stream)
    assert judge._judge_request("https://example.com/api", {}, {}) == "ok"
    assert delays == [2.0]
    assert len(attempts) == 2
    assert all(body.closed for body in bodies)
