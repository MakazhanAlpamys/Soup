"""Regression: /v1/messages forwards temperature and top_p (#2381)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from soup_cli.commands import serve


def test_messages_forwards_temperature_and_top_p_like_chat_completions():
    seen: list[tuple[object, object]] = []

    def fake(model, tok, messages, **kw):
        seen.append((kw.get("temperature"), kw.get("top_p")))
        return ("ok", 1, 1)

    with patch.object(serve, "_generate_response", fake):
        app = serve._create_app(MagicMock(), MagicMock(), "cpu", "m", 16)
        client = TestClient(app, base_url="http://localhost")
        for temperature in (0, 0.2, 1.5):
            body = {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 4,
                "temperature": temperature,
                "top_p": 0.5,
            }
            client.post("/v1/chat/completions", json=body)
            chat = seen[-1]
            client.post("/v1/messages", json=body)
            messages = seen[-1]
            assert messages == chat == (float(temperature), 0.5)


def test_messages_uses_defaults_when_sampling_omitted():
    seen: list[tuple[object, object]] = []

    def fake(model, tok, messages, **kw):
        seen.append((kw.get("temperature"), kw.get("top_p")))
        return ("ok", 1, 1)

    with patch.object(serve, "_generate_response", fake):
        app = serve._create_app(MagicMock(), MagicMock(), "cpu", "m", 16)
        client = TestClient(app, base_url="http://localhost")
        body = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
        }
        client.post("/v1/messages", json=body)
        assert seen[-1] == (0.7, 0.9)
