"""Provider failures must not look like a successful empty generation run."""

import json

import httpx
import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils import http_response
from tests.conftest import strip_ansi

ENDPOINT = "http://127.0.0.1:9123"
SENTINEL = b"existing output\r\n\x00must survive"


def _reply(provider, text):
    return {"response": text} if provider == "ollama" else {"choices": [{"text": text}]}


def _run(command, provider, tmp_path, monkeypatch, *extra):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "seeds.jsonl").write_text(
        '{"prompt":"Write a poem"}\n{"prompt":"Explain gravity"}\n', encoding="utf-8"
    )
    args = ["data", command, "--provider", provider, "--base-url", ENDPOINT,
            "--output", "out.jsonl"]
    if command == "evolve":
        args += ["--input", "seeds.jsonl", "--model", "llama3"]
    else:
        args += ["--base", "meta-llama/Llama-3.1-8B-Instruct", "--target", "1"]
    return CliRunner().invoke(app, [*args, *extra])


def _plain(result):
    return " ".join(strip_ansi(result.output).split())


@pytest.mark.parametrize("command", ["evolve", "gen-magpie"])
@pytest.mark.parametrize("provider", ["ollama", "vllm"])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure,diagnostic", [
    ("transport", "connection refused"), ("status", "HTTP 503"),
    ("json", "JSON"), ("shape", "response"), ("type", "string"),
    ("oversize", "byte limit"), ("empty", "empty"),
])
def test_failed_calls_preserve_output(
    command, provider, existing, failure, diagnostic, tmp_path, monkeypatch
):
    output = tmp_path / "out.jsonl"
    if existing:
        output.write_bytes(SENTINEL)

    def post(*args, **kwargs):
        if failure == "transport":
            raise httpx.ConnectError("connection refused")
        if failure == "oversize":
            raise http_response.ResponseTooLargeError("response exceeds byte limit")
        if failure == "status":
            return httpx.Response(503)
        if failure == "json":
            return httpx.Response(200, content=b"invalid JSON")
        body = {} if failure == "shape" else _reply(provider, 7 if failure == "type" else " ")
        return httpx.Response(200, json=body)

    monkeypatch.setattr(http_response, "post_capped", post)
    result = _run(command, provider, tmp_path, monkeypatch)
    text = _plain(result)
    assert result.exit_code == 1, (text, result.exception)
    assert "No usable rows produced:" in text
    assert f"--provider {provider}" in text
    assert ENDPOINT in text
    assert "first error:" in text and diagnostic.lower() in text.lower()
    if existing:
        assert output.read_bytes() == SENTINEL
    else:
        assert not output.exists()


@pytest.mark.parametrize("command", ["evolve", "gen-magpie"])
@pytest.mark.parametrize("provider", ["ollama", "vllm"])
def test_partial_failure_writes_rows_and_reports_counts(command, provider, tmp_path, monkeypatch):
    calls = 0

    def post(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("first request failed [bold]literally[/bold]")
        return httpx.Response(200, json=_reply(provider, "Explain orbital motion in two steps"))

    monkeypatch.setattr(http_response, "post_capped", post)
    result = _run(command, provider, tmp_path, monkeypatch)
    text = _plain(result)
    assert result.exit_code == 0, (text, result.exception)
    rows = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["messages"][0]["content"] == "Explain orbital motion in two steps"
    assert f"1 of {calls} provider calls failed" in text
    assert "Warning:" in text and ENDPOINT in text
    assert "[bold]literally[/bold]" in text


@pytest.mark.parametrize("command", ["evolve", "gen-magpie"])
@pytest.mark.parametrize("provider", ["ollama", "vllm"])
def test_successful_but_filtered_replies_keep_empty_success(
    command, provider, tmp_path, monkeypatch
):
    # Meta-prompt echoes are eliminated by evolve. Magpie's quality filter refuses the pair.
    monkeypatch.setattr(http_response, "post_capped", lambda *a, **k: httpx.Response(
        200, json=_reply(provider, "#Given Prompt#")
    ))
    monkeypatch.setattr("soup_cli.utils.magpie.default_quality_fn", lambda *a: False)
    result = _run(command, provider, tmp_path, monkeypatch)
    assert result.exit_code == 0, (result.output, result.exception)
    assert (tmp_path / "out.jsonl").read_bytes() == b""
    assert "0 rows" in _plain(result)
    assert "provider calls failed" not in _plain(result)


@pytest.mark.parametrize("command", ["evolve", "gen-magpie"])
def test_plan_only_makes_no_requests(command, tmp_path, monkeypatch):
    def post(*args, **kwargs):
        pytest.fail("plan contacted the provider")

    monkeypatch.setattr(http_response, "post_capped", post)
    result = _run(command, "ollama", tmp_path, monkeypatch, "--plan-only")
    assert result.exit_code == 0, result.output
    assert not (tmp_path / "out.jsonl").exists()


@pytest.mark.parametrize("provider", ["ollama", "vllm"])
def test_sdk_tracks_failures_by_default_and_preserves_output(provider, tmp_path, monkeypatch):
    from soup_cli.utils.magpie import MagpieProviderError, build_magpie_config, run_magpie

    monkeypatch.chdir(tmp_path)
    output = tmp_path / "out.jsonl"
    output.write_bytes(SENTINEL)
    monkeypatch.setattr(http_response, "post_capped", lambda *a, **k: httpx.Response(503))
    config = build_magpie_config(base="llama", provider=provider, target=1, quality_filter=False)
    with pytest.raises(MagpieProviderError, match="3 of 3 provider calls failed"):
        run_magpie(config, output_path="out.jsonl", max_attempts=3)
    assert output.read_bytes() == SENTINEL


@pytest.mark.parametrize("provider", ["ollama", "vllm"])
def test_factory_counts_calls_and_keeps_the_first_failure(provider, monkeypatch):
    from soup_cli.utils.data_forge import ForgeJudgeStats
    from soup_cli.utils.magpie import make_magpie_generate_fn

    replies = iter([httpx.Response(503), httpx.Response(500),
                    httpx.Response(200, json=_reply(provider, "useful answer"))])
    monkeypatch.setattr(http_response, "post_capped", lambda *a, **k: next(replies))
    stats = ForgeJudgeStats()
    generate = make_magpie_generate_fn(provider, model="llama", stats=stats)
    assert generate(None) == ""  # invalid caller input makes no HTTP request
    assert generate("first") == ""
    assert generate("second") == ""
    assert generate("third") == "useful answer"
    assert (stats.calls, stats.failures, stats.first_error) == (3, 2, "HTTP 503")
