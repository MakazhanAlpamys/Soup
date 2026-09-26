"""Judge URLs: the OpenAI key is attached only for the OpenAI API host."""

from __future__ import annotations

from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-canary")


def _auth_header_for(url: str):
    from soup_cli.eval.gate import _parse_judge_url
    from soup_cli.eval.judge import JudgeEvaluator

    provider, model, base = _parse_judge_url(url)
    evaluator = JudgeEvaluator(provider=provider, model=model, api_base=base)
    with mock.patch("httpx.post") as post:
        post.return_value.json.return_value = {
            "choices": [{"message": {"content": '{"winner": "A"}'}}]
        }
        evaluator.compare_pair("p", "a", "b")
    assert post.call_args is not None, "the judge was never called"
    return post.call_args.args[0], post.call_args.kwargs["headers"].get("Authorization")


@pytest.mark.parametrize(
    "url",
    [
        "https://judge.example.com/gpt-4o-mini",
        "https://api.openai.com.example.net/gpt-4o-mini",
        "https://api.groq.com/openai/llama3-70b-8192",
        "https://example.com/api.openai.com/gpt-4o-mini",
    ],
)
def test_other_https_hosts_get_no_openai_key(url):
    _target, auth = _auth_header_for(url)
    assert "sk-canary" not in str(auth)
    assert auth is None


@pytest.mark.parametrize(
    "url", ["https://api.openai.com/gpt-4o-mini", "https://API.OPENAI.COM/gpt-4o-mini"]
)
def test_openai_host_keeps_key(url):
    target, auth = _auth_header_for(url)
    assert target == "https://" + url.split("/")[2] + "/v1/chat/completions"
    assert auth == "Bearer sk-canary"


def test_other_https_host_is_server_provider():
    from soup_cli.eval.gate import _parse_judge_url

    assert _parse_judge_url("https://judge.example.com/m") == (
        "server", "m", "https://judge.example.com"
    )
