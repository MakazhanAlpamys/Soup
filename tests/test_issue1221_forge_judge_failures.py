"""#1221 — ``soup data forge --judge-provider`` must not turn failed judge calls into rows.

Before the fix the forge judge was built without ``raise_on_error``, so every
transport / HTTP / parse failure came back as ``{"text": ""}``.
``score_uncertainty`` rates an empty reply 1.0, so no ``--uncertainty-threshold``
could prune it: the command wrote one empty-assistant row per failed call,
printed the green "synth complete" panel and exited 0.
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator, List

import pytest
from typer.testing import CliRunner

from tests.conftest import strip_ansi

_HEALTHY_ANSWER = "Customers receive refunds within five business days."


def _terminal_text(result) -> str:
    return re.sub(r"\s+", " ", strip_ansi(result.output))


def _write_docs(tmp_path: Path) -> None:
    """3 documents x 2 paragraphs = 6 chunks (6 judge calls)."""
    docs = tmp_path / "docs"
    docs.mkdir()
    for name in ("a", "b", "c"):
        (docs / f"{name}.txt").write_text(
            f"Refunds for {name} take five days.\n\nShipping for {name} takes two days.\n",
            encoding="utf-8",
        )


def _run_forge(tmp_path: Path, monkeypatch, *extra: str):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    _write_docs(tmp_path)
    return CliRunner().invoke(
        app,
        ["data", "forge", "--docs", "docs", "--target-rows", "10", *extra],
    )


def _dataset_rows(tmp_path: Path) -> List[dict]:
    path = tmp_path / "forge_dataset.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _empty_answer_rows(rows: List[dict]) -> List[dict]:
    return [r for r in rows if not r["messages"][-1]["content"].strip()]


# --------------------------------------------------------------------------- stub judge


class _StubJudge:
    def __init__(self) -> None:
        self.mode = "healthy"
        self.calls = 0
        self._lock = threading.Lock()

    def respond(self) -> tuple[int, bytes]:
        with self._lock:
            self.calls += 1
            n = self.calls
        mode = self.mode
        if mode == "alternate":
            mode = "healthy" if n % 2 else "500"
        if mode in {"404", "429", "500"}:
            return int(mode), b'{"error": "stub"}'
        if mode == "malformed":
            return 200, b"this is not json"
        content = "" if mode == "empty" else _HEALTHY_ANSWER
        body = {"choices": [{"message": {"role": "assistant", "content": content}}]}
        return 200, json.dumps(body).encode("utf-8")


@pytest.fixture
def stub_judge() -> Iterator[tuple[_StubJudge, str]]:
    stub = _StubJudge()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 — http.server API
            length = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(length)
            status, payload = stub.respond()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield stub, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


# --------------------------------------------------------------------------- tests


def test_unreachable_judge_exits_nonzero_and_writes_no_rows(tmp_path, monkeypatch) -> None:
    # Nothing listens on port 9: every call fails with "connection refused".
    result = _run_forge(
        tmp_path, monkeypatch,
        "--judge-provider", "ollama", "--judge-base-url", "http://127.0.0.1:9",
    )
    output = _terminal_text(result)

    assert result.exit_code != 0, output
    assert _empty_answer_rows(_dataset_rows(tmp_path)) == []
    assert not (tmp_path / "forge_dataset.jsonl").exists()
    assert not (tmp_path / "forge_provenance.json").exists()
    assert "synth complete" not in output
    assert "6 of 6 judge calls failed" in output
    assert "--judge-provider ollama (http://127.0.0.1:9)" in output
    assert "ollama provider request failed" in output


_FAILURES = [
    ("404", "provider returned HTTP 404"),
    ("429", "provider returned HTTP 429"),
    ("500", "provider returned HTTP 500"),
    ("empty", "judge returned an empty reply"),
    ("malformed", "provider returned a malformed response"),
]


@pytest.mark.parametrize("provider", ["ollama", "vllm"])
@pytest.mark.parametrize(("mode", "first_error"), _FAILURES)
def test_failing_stub_judge_exits_nonzero(
    tmp_path, monkeypatch, stub_judge, provider: str, mode: str, first_error: str
) -> None:
    stub, url = stub_judge
    stub.mode = mode
    result = _run_forge(
        tmp_path, monkeypatch,
        "--judge-provider", provider, "--judge-base-url", url,
    )
    output = _terminal_text(result)

    assert stub.calls == 6
    assert result.exit_code == 1, output
    assert _dataset_rows(tmp_path) == []
    assert "synth complete" not in output
    assert f"6 of 6 judge calls failed for --judge-provider {provider}" in output
    assert first_error in output


@pytest.mark.parametrize("provider", ["ollama", "vllm"])
def test_partial_outage_keeps_only_successful_rows(
    tmp_path, monkeypatch, stub_judge, provider: str
) -> None:
    stub, url = stub_judge
    stub.mode = "alternate"
    result = _run_forge(
        tmp_path, monkeypatch,
        "--judge-provider", provider, "--judge-base-url", url,
    )
    output = _terminal_text(result)

    assert result.exit_code == 0, output
    rows = _dataset_rows(tmp_path)
    assert len(rows) == 3
    assert all(r["messages"][-1]["content"] == _HEALTHY_ANSWER for r in rows)
    assert "Judge calls: 3 of 6 failed" in output
    assert "3 of 6 judge calls failed" in output
    assert "HTTP 500" in output


@pytest.mark.parametrize("threshold", ["0.0", "1.0"])
def test_empty_replies_pruned_at_any_threshold_cli(
    tmp_path, monkeypatch, stub_judge, threshold: str
) -> None:
    stub, url = stub_judge
    stub.mode = "empty"
    result = _run_forge(
        tmp_path, monkeypatch,
        "--judge-provider", "ollama", "--judge-base-url", url,
        "--uncertainty-threshold", threshold,
    )
    assert result.exit_code == 1, _terminal_text(result)
    assert _dataset_rows(tmp_path) == []


@pytest.mark.parametrize("threshold", [0.0, 1.0])
@pytest.mark.parametrize("reply", ["", "   \n", None])
def test_synthesise_never_keeps_empty_reply(tmp_path, monkeypatch, threshold, reply) -> None:
    from soup_cli.utils.data_forge import (
        ForgeJudgeStats,
        discover_documents,
        synthesise_forge_rows,
    )

    monkeypatch.chdir(tmp_path)
    _write_docs(tmp_path)
    stats = ForgeJudgeStats()
    rows = synthesise_forge_rows(
        discover_documents("docs"),
        task="sft",
        target_rows=10,
        judge=lambda _prompt: {"text": reply},
        uncertainty_threshold=threshold,
        stats=stats,
    )
    assert rows == []
    assert (stats.calls, stats.failures) == (6, 6)
    assert stats.first_error == "judge returned an empty reply"


def test_synthesise_counts_raising_judge(tmp_path, monkeypatch) -> None:
    from soup_cli.utils.data_forge import (
        ForgeJudgeStats,
        discover_documents,
        synthesise_forge_rows,
    )

    monkeypatch.chdir(tmp_path)
    _write_docs(tmp_path)
    calls = {"n": 0}

    def judge(_prompt: str) -> dict:
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            raise RuntimeError("boom")
        return {"text": _HEALTHY_ANSWER}

    stats = ForgeJudgeStats()
    rows = synthesise_forge_rows(
        discover_documents("docs"), task="sft", target_rows=10, judge=judge, stats=stats
    )
    assert len(rows) == 3
    assert (stats.calls, stats.failures, stats.first_error) == (6, 3, "boom")


def test_synthesise_rejects_wrong_stats_type(tmp_path, monkeypatch) -> None:
    from soup_cli.utils.data_forge import discover_documents, synthesise_forge_rows

    monkeypatch.chdir(tmp_path)
    _write_docs(tmp_path)
    with pytest.raises(TypeError, match="stats"):
        synthesise_forge_rows(
            discover_documents("docs"), task="sft", target_rows=1,
            judge=lambda _p: {"text": "x"}, stats={},  # type: ignore[arg-type]
        )


class _AnthropicResponse:
    def __init__(self, status_code: int, text: str = _HEALTHY_ANSWER) -> None:
        self.status_code = status_code
        self._text = text

    def json(self) -> dict:
        return {"content": [{"type": "text", "text": self._text}]}


@pytest.mark.parametrize(
    ("status", "text", "first_error"),
    [
        (500, _HEALTHY_ANSWER, "anthropic provider returned HTTP 500"),
        (200, "", "judge returned an empty reply"),
    ],
)
def test_anthropic_failures_exit_nonzero(
    tmp_path, monkeypatch, status: int, text: str, first_error: str
) -> None:
    import httpx

    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-test-value")
    monkeypatch.setattr(
        httpx, "post", lambda *_a, **_k: _AnthropicResponse(status, text)
    )
    result = _run_forge(tmp_path, monkeypatch, "--judge-provider", "anthropic")
    output = _terminal_text(result)

    assert result.exit_code == 1, output
    assert _dataset_rows(tmp_path) == []
    assert "6 of 6 judge calls failed for --judge-provider anthropic" in output
    assert first_error in output


def test_anthropic_partial_outage_keeps_successful_rows(tmp_path, monkeypatch) -> None:
    import httpx

    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-test-value")
    calls = {"n": 0}

    def post(*_a, **_k):
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            raise httpx.ConnectError("refused")
        return _AnthropicResponse(200)

    monkeypatch.setattr(httpx, "post", post)
    result = _run_forge(tmp_path, monkeypatch, "--judge-provider", "anthropic")
    output = _terminal_text(result)

    assert result.exit_code == 0, output
    rows = _dataset_rows(tmp_path)
    assert len(rows) == 3
    assert _empty_answer_rows(rows) == []
    assert "3 of 6 judge calls failed" in output
    assert "anthropic provider request failed" in output


def test_healthy_judge_control(tmp_path, monkeypatch, stub_judge) -> None:
    stub, url = stub_judge
    result = _run_forge(
        tmp_path, monkeypatch,
        "--judge-provider", "ollama", "--judge-base-url", url,
    )
    output = _terminal_text(result)

    assert result.exit_code == 0, output
    assert len(_dataset_rows(tmp_path)) == 6
    assert "Data Forge — synth complete" in output
    assert "judge calls failed" not in output
    assert "Judge calls:" not in output


def test_offline_stub_unchanged(tmp_path, monkeypatch) -> None:
    result = _run_forge(tmp_path, monkeypatch)
    output = _terminal_text(result)

    assert result.exit_code == 0, output
    rows = _dataset_rows(tmp_path)
    assert len(rows) == 6
    assert all(r["messages"][-1]["content"].startswith("Synthesised answer") for r in rows)
    assert "judge calls failed" not in output
