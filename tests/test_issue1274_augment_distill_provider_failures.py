"""#1274 — ``soup data augment`` / ``soup distill-prompt`` must not turn failed
provider calls into rows.

Same defect class as #1221 (fixed for ``soup data forge`` in #1261): both
commands built their LLM provider without ``raise_on_error=True``, so every
transport/HTTP/malformed-response failure came back as ``{"text": ""}``.
``augment`` wrote a row with the field rewritten to an empty string, and
``distill-prompt`` silently dropped every failed row — both reported success
(exit 0) even when every call failed.
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

_HEALTHY_ANSWER = "Rewritten text."

runner = CliRunner()


def _terminal_text(result) -> str:
    return re.sub(r"\s+", " ", strip_ansi(result.output))


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


# --------------------------------------------------------------------------- augment


def _write_augment_data(tmp_path: Path) -> None:
    # One string field per row so each row costs exactly one provider call
    # (parity-sensitive tests below rely on that 1:1 mapping).
    rows = [{"text": f"q{i}"} for i in range(2)]
    (tmp_path / "data.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


def _run_augment(tmp_path: Path, monkeypatch, *extra: str):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    _write_augment_data(tmp_path)
    return runner.invoke(
        app,
        [
            "data", "augment",
            "--input", "data.jsonl",
            "--output", "augmented.jsonl",
            "--strategy", "rephrase",
            "--count", "1",
            *extra,
        ],
    )


def _augment_rows(tmp_path: Path) -> List[dict]:
    path = tmp_path / "augmented.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_augment_unreachable_provider_exits_nonzero_no_file(tmp_path, monkeypatch) -> None:
    result = _run_augment(
        tmp_path, monkeypatch,
        "--provider", "ollama", "--base-url", "http://127.0.0.1:9",
    )
    output = _terminal_text(result)

    assert result.exit_code == 1, (output, repr(result.exception))
    assert not (tmp_path / "augmented.jsonl").exists()
    assert "of" in output and "provider calls failed" in output
    assert "--provider ollama (http://127.0.0.1:9)" in output


@pytest.mark.parametrize(
    ("mode", "first_error"),
    [
        ("500", "provider returned HTTP 500"),
        ("empty", "provider returned an empty reply"),
        ("malformed", "provider returned a malformed response"),
    ],
)
def test_augment_failing_provider_exits_nonzero(
    tmp_path, monkeypatch, stub_judge, mode: str, first_error: str
) -> None:
    stub, url = stub_judge
    stub.mode = mode
    result = _run_augment(
        tmp_path, monkeypatch,
        "--provider", "ollama", "--base-url", url,
    )
    output = _terminal_text(result)

    assert result.exit_code == 1, (output, repr(result.exception))
    assert not (tmp_path / "augmented.jsonl").exists()
    assert first_error in output


def test_augment_partial_outage_keeps_only_successful_rows(
    tmp_path, monkeypatch, stub_judge
) -> None:
    stub, url = stub_judge
    stub.mode = "alternate"
    result = _run_augment(
        tmp_path, monkeypatch,
        "--provider", "ollama", "--base-url", url,
    )
    output = _terminal_text(result)

    assert result.exit_code == 0, output
    rows = _augment_rows(tmp_path)
    # 2 source rows + only the augmented row whose rewrite succeeded
    # (call 1 healthy, call 2 fails under "alternate").
    assert len(rows) == 3
    assert rows[2]["text"] == _HEALTHY_ANSWER
    assert "1 of 2 provider calls failed" in output
    assert "Augmentation complete with provider failures" in output


def test_augment_healthy_provider_control(tmp_path, monkeypatch, stub_judge) -> None:
    stub, url = stub_judge
    result = _run_augment(
        tmp_path, monkeypatch,
        "--provider", "ollama", "--base-url", url,
    )
    output = _terminal_text(result)

    assert result.exit_code == 0, output
    rows = _augment_rows(tmp_path)
    assert len(rows) == 4
    assert "provider calls failed" not in output
    assert "Augmentation complete:" in output


@pytest.mark.parametrize(
    ("strategy", "strategy_args"),
    [("translate", ["--lang", "es"]), ("style", ["--styles", "formal"])],
)
def test_augment_translate_and_style_fail_loudly_too(
    tmp_path, monkeypatch, stub_judge, strategy: str, strategy_args: list
) -> None:
    stub, url = stub_judge
    stub.mode = "500"
    result = _run_augment(
        tmp_path, monkeypatch,
        "--strategy", strategy, *strategy_args,
        "--provider", "ollama", "--base-url", url,
    )
    output = _terminal_text(result)

    assert result.exit_code == 1, (output, repr(result.exception))
    assert not (tmp_path / "augmented.jsonl").exists()
    assert "provider returned HTTP 500" in output


@pytest.mark.parametrize(
    ("strategy", "strategy_args"),
    [("translate", ["--lang", "es"]), ("style", ["--styles", "formal"])],
)
def test_augment_translate_and_style_partial_outage_keeps_only_good_rows(
    tmp_path, monkeypatch, stub_judge, strategy: str, strategy_args: list
) -> None:
    stub, url = stub_judge
    stub.mode = "alternate"
    result = _run_augment(
        tmp_path, monkeypatch,
        "--strategy", strategy, *strategy_args,
        "--provider", "ollama", "--base-url", url,
    )
    output = _terminal_text(result)

    assert result.exit_code == 0, (output, repr(result.exception))
    rows = _augment_rows(tmp_path)
    assert len(rows) == 3  # 2 source rows + the one variant whose call succeeded
    assert all(row["text"] for row in rows)
    assert "1 of 2 provider calls failed" in output


def test_augment_failure_summary_is_not_parsed_as_rich_markup(tmp_path, monkeypatch) -> None:
    import soup_cli.commands.data as data_mod
    from soup_cli.utils.data_forge import ProviderCallError

    class Boom:
        def generate(self, prompt: str, max_tokens: int = 512) -> str:
            raise ProviderCallError("upstream said [/oops]")

    monkeypatch.setattr(data_mod, "_load_augment_provider", lambda *a, **k: Boom())
    result = _run_augment(
        tmp_path, monkeypatch, "--provider", "ollama", "--base-url", "http://127.0.0.1:9"
    )

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "[/oops]" in _terminal_text(result)


def test_augment_partial_failure_warning_is_not_parsed_as_rich_markup(
    tmp_path, monkeypatch
) -> None:
    # The partial-failure path writes the file FIRST and only then prints the warning, so an
    # unescaped "[/oops]" in the provider's error crashes the command after a successful write.
    import soup_cli.commands.data as data_mod
    from soup_cli.utils.data_forge import ProviderCallError

    class FailsOnce:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompt: str, max_tokens: int = 512) -> str:
            self.calls += 1
            if self.calls == 1:
                raise ProviderCallError("upstream said [/oops]")
            return "Rewritten text."

    monkeypatch.setattr(data_mod, "_load_augment_provider", lambda *a, **k: FailsOnce())
    result = _run_augment(
        tmp_path, monkeypatch, "--provider", "ollama", "--base-url", "http://127.0.0.1:9"
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert len(_augment_rows(tmp_path)) == 3  # 2 source rows + the one variant that succeeded
    assert "[/oops]" in _terminal_text(result)


# --------------------------------------------------------------------------- distill-prompt


def _write_traces(tmp_path: Path, prompts: List[str]) -> None:
    with open(tmp_path / "traces.jsonl", "w", encoding="utf-8") as fh:
        for p in prompts:
            fh.write(json.dumps({"prompt": p}) + "\n")


def _run_distill(tmp_path: Path, monkeypatch, *extra: str):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    _write_traces(tmp_path, ["q1", "q2"])
    return runner.invoke(
        app,
        [
            "distill-prompt",
            "--traces", "traces.jsonl",
            "--teacher", "qwen2.5:0.5b",
            "--student", "smol",
            "--strategy", "sft",
            "--output", "distilled.jsonl",
            *extra,
        ],
    )


def test_distill_unreachable_provider_exits_nonzero_no_file(tmp_path, monkeypatch) -> None:
    result = _run_distill(
        tmp_path, monkeypatch,
        "--provider", "ollama", "--base-url", "http://127.0.0.1:9",
    )
    output = _terminal_text(result)

    assert result.exit_code == 1, (output, repr(result.exception))
    assert not (tmp_path / "distilled.jsonl").exists()
    assert "No usable rows produced" in output
    assert "2 of 2 provider calls failed" in output


def test_distill_failing_provider_exits_nonzero(tmp_path, monkeypatch, stub_judge) -> None:
    stub, url = stub_judge
    stub.mode = "500"
    result = _run_distill(
        tmp_path, monkeypatch,
        "--provider", "ollama", "--base-url", url,
    )
    output = _terminal_text(result)

    assert result.exit_code == 1, (output, repr(result.exception))
    assert not (tmp_path / "distilled.jsonl").exists()
    assert "provider returned HTTP 500" in output


def test_distill_partial_outage_keeps_only_successful_rows(
    tmp_path, monkeypatch, stub_judge
) -> None:
    stub, url = stub_judge
    stub.mode = "alternate"
    result = _run_distill(
        tmp_path, monkeypatch,
        "--provider", "ollama", "--base-url", url,
    )
    output = _terminal_text(result)

    assert result.exit_code == 0, output
    with open(tmp_path / "distilled.jsonl", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    assert len(rows) == 1
    assert "1 of 2 failed" in output
    assert "done with provider failures" in output


def test_distill_healthy_provider_control(tmp_path, monkeypatch, stub_judge) -> None:
    stub, url = stub_judge
    result = _run_distill(
        tmp_path, monkeypatch,
        "--provider", "ollama", "--base-url", url,
    )
    output = _terminal_text(result)

    assert result.exit_code == 0, output
    with open(tmp_path / "distilled.jsonl", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    assert len(rows) == 2
    assert "provider failures" not in output


def test_distill_preference_strategy_exercises_teacher_and_student(
    tmp_path, monkeypatch, stub_judge
) -> None:
    stub, url = stub_judge
    stub.mode = "alternate"
    result = _run_distill(
        tmp_path, monkeypatch,
        "--provider", "ollama", "--base-url", url,
        "--strategy", "preference",
    )
    output = _terminal_text(result)

    # alternate: call 1 ok (teacher q1), call 2 fails (student q1),
    # call 3 ok (teacher q2), call 4 fails (student q2) -> both rows drop
    # for lack of a student reply, even though only the student calls failed.
    assert result.exit_code == 1, (output, repr(result.exception))
    assert not (tmp_path / "distilled.jsonl").exists()
    assert "2 of 4 provider calls failed" in output


def test_distill_failure_summary_is_not_parsed_as_rich_markup(tmp_path, monkeypatch) -> None:
    import soup_cli.utils.prompt_distill as distill_mod
    from soup_cli.utils.data_forge import ProviderCallError

    def boom(prompt: str):
        raise ProviderCallError("upstream said [/oops]")

    monkeypatch.setattr(distill_mod, "_build_provider_fn", lambda *a, **k: boom)
    result = _run_distill(
        tmp_path, monkeypatch, "--provider", "ollama", "--base-url", "http://127.0.0.1:9"
    )

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "[/oops]" in _terminal_text(result)
