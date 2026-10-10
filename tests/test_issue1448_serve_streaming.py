"""Tests for issue #1448: soup serve streaming ignores structured-output and reasoning-parser.

Acceptance criteria:
- Streaming SSE of a <think> reply contains no <think> when reasoning_parser is configured.
- The output constraint (logits processor) reaches generation on the streaming path.
"""

from __future__ import annotations

import json
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import soup_cli.commands.serve as serve
from soup_cli.commands.serve import _create_app
from soup_cli.utils import structured_output


def _make_app(**kwargs):
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    defaults = dict(
        model_obj=model,
        tokenizer=tokenizer,
        device="cpu",
        model_name="test-model",
        max_tokens_default=64,
    )
    defaults.update(kwargs)
    return _create_app(**defaults)


def test_streaming_sse_strips_reasoning():
    """Streaming SSE chunks must not contain <think> blocks when reasoning_parser is active."""
    app = _make_app(reasoning_parser="deepseek-r1")
    client = TestClient(app)

    raw_reply = "<think>internal reasoning steps</think>Hello world!"
    with patch(
        "soup_cli.commands.serve._generate_response",
        return_value=(raw_reply, 4, 10),
    ):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    body = response.text
    assert "<think>" not in body
    assert "internal reasoning steps" not in body
    assert "Hello" in body
    assert "world!" in body


def test_streaming_forwards_logits_processors():
    """Streaming chat completions must pass built logits processors to generation."""
    dummy_processor = MagicMock()
    app = _make_app(output_constraint={"type": "json_schema", "schema": "{}"})
    client = TestClient(app)

    with (
        patch(
            "soup_cli.utils.structured_output.build_logits_processors",
            return_value=[dummy_processor],
        ),
        patch(
            "soup_cli.commands.serve._generate_response",
            return_value=("output", 2, 5),
        ) as mock_generate,
    ):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    mock_generate.assert_called_once()
    _, kwargs = mock_generate.call_args
    assert kwargs.get("logits_processor") == [dummy_processor]


REGEX = {"kind": "regex", "pattern": "a+"}


def _sse_text(body: str) -> str:
    return "".join(
        json.loads(line[len("data: "):])["choices"][0]["delta"].get("content", "")
        for line in body.splitlines()
        if line.startswith("data: {") and '"choices"' in line
    )


@pytest.mark.parametrize(
    "parser, raw, visible",
    [
        ("deepseek-r1", "<think>plan</think>aaa and more", "aaa and more"),
        ("openthinker", "<|begin_of_thought|>t<|end_of_thought|>aaa", "aaa"),
        (None, "<think>kept</think>aaa", "<think>kept</think>aaa"),
    ],
)
def test_streaming_matches_non_streaming(parser, raw, visible):
    seen, built_for = [], []

    def fake_generate(model, tokenizer, messages, **kwargs):
        seen.append((kwargs.get("logits_processor"), kwargs.get("ngram_config")))
        return (raw, 3, 2)

    def fake_build(constraint, tokenizer):
        built_for.append(constraint)
        return ["processor-for", constraint]

    ngram = object()
    with mock.patch.object(serve, "_generate_response", fake_generate), \
            mock.patch.object(structured_output, "build_logits_processors", fake_build):
        client = TestClient(serve._create_app(
            model_obj=object(), tokenizer=object(), device="cpu", model_name="m",
            max_tokens_default=8, output_constraint=REGEX, reasoning_parser=parser,
            ngram_config=ngram,
        ))
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        plain = client.post("/v1/chat/completions", json=body)
        streamed = client.post("/v1/chat/completions", json={**body, "stream": True})

    assert plain.status_code == streamed.status_code == 200
    assert plain.json()["choices"][0]["message"]["content"] == visible
    assert _sse_text(streamed.text) == visible
    assert built_for == [REGEX, REGEX]
    assert seen[0] == seen[1] == (["processor-for", REGEX], ngram)
