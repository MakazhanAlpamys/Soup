"""Tests for issue #1448: soup serve streaming ignores structured-output and reasoning-parser.

Acceptance criteria:
- Streaming SSE of a <think> reply contains no <think> when reasoning_parser is configured.
- The output constraint (logits processor) reaches generation on the streaming path.
"""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from soup_cli.commands.serve import _create_app


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
