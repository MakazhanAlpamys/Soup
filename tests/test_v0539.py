"""Tests for v0.53.9 — Live Dashboard + UX + Bench + Standalone CLIs.

Covers #94 SSE stream, #95 ui --public + QR, #98 reasoning-parser strip,
#100 tool-outputs API, #15 tokenizer train, #26 bench percentiles,
#28 bench backend auto-detect, #12 example doc.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

import soup_cli
from soup_cli.cli import app

# ----------------------------------------------------------------- version

def test_version_bump_to_0_53_9():
    assert soup_cli.__version__ == "0.53.9"


# ----------------------------------------------------- #94 SSE event buffer

def test_train_event_buffer_push_and_drain():
    from soup_cli.utils.sse_train_stream import TrainEvent
    from soup_cli.utils.train_event_buffer import TrainEventBuffer

    buffer = TrainEventBuffer()
    cursor = buffer.push(TrainEvent(type="metric", step=1, loss=0.5))
    assert cursor == 1
    cursor = buffer.push(TrainEvent(type="metric", step=2, loss=0.4))
    assert cursor == 2
    drained = buffer.drain()
    assert len(drained) == 2
    assert drained[0].step == 1
    # After draining, buffer is empty.
    assert buffer.drain() == []


def test_train_event_buffer_rejects_non_event():
    from soup_cli.utils.train_event_buffer import TrainEventBuffer

    buffer = TrainEventBuffer()
    with pytest.raises(TypeError):
        buffer.push("not an event")
    with pytest.raises(TypeError):
        buffer.push({"type": "metric"})


def test_train_event_buffer_maxlen_validation():
    from soup_cli.utils.train_event_buffer import TrainEventBuffer

    with pytest.raises(TypeError):
        TrainEventBuffer(maxlen=True)
    with pytest.raises(ValueError):
        TrainEventBuffer(maxlen=0)
    with pytest.raises(ValueError):
        TrainEventBuffer(maxlen=-3)


def test_train_event_buffer_snapshot_limits():
    from soup_cli.utils.sse_train_stream import TrainEvent
    from soup_cli.utils.train_event_buffer import TrainEventBuffer

    buffer = TrainEventBuffer()
    for step in range(5):
        buffer.push(TrainEvent(type="metric", step=step))
    snap = buffer.snapshot(limit=3)
    assert len(snap) == 3
    assert [e.step for e in snap] == [2, 3, 4]
    # Snapshot does not drain.
    assert len(buffer.snapshot()) == 5
    assert buffer.snapshot(limit=0) == []
    with pytest.raises(TypeError):
        buffer.snapshot(limit=True)
    with pytest.raises(ValueError):
        buffer.snapshot(limit=-1)


def test_train_event_buffer_overflow_drops_oldest():
    from soup_cli.utils.sse_train_stream import TrainEvent
    from soup_cli.utils.train_event_buffer import TrainEventBuffer

    buffer = TrainEventBuffer(maxlen=3)
    for step in range(5):
        buffer.push(TrainEvent(type="metric", step=step))
    # Only latest three retained.
    drained = buffer.drain()
    assert [e.step for e in drained] == [2, 3, 4]


def test_push_train_event_silent_on_bad_input():
    from soup_cli.utils.train_event_buffer import (
        get_global_buffer,
        push_train_event,
        reset_global_buffer,
    )

    reset_global_buffer()
    assert push_train_event("nope") is None
    assert push_train_event(None) is None
    assert get_global_buffer().drain() == []


def test_push_train_event_happy():
    from soup_cli.utils.sse_train_stream import TrainEvent
    from soup_cli.utils.train_event_buffer import (
        get_global_buffer,
        push_train_event,
        reset_global_buffer,
    )

    reset_global_buffer()
    cursor = push_train_event(TrainEvent(type="metric", step=42, loss=0.1))
    assert cursor == 1
    events = get_global_buffer().drain()
    assert len(events) == 1
    assert events[0].step == 42


# ---------------------------------------------------- #94 SSE FastAPI route

def test_api_train_stream_emits_pending_events():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from soup_cli.ui.app import create_app
    from soup_cli.utils.sse_train_stream import TrainEvent
    from soup_cli.utils.train_event_buffer import (
        push_train_event,
        reset_global_buffer,
    )

    reset_global_buffer()
    push_train_event(TrainEvent(type="metric", step=1, loss=0.5))
    push_train_event(TrainEvent(type="status", message="started"))

    app_inst = create_app()
    client = TestClient(app_inst)
    response = client.get("/api/train/stream")
    assert response.status_code == 200
    body = response.text
    # Frames are W3C SSE.
    assert "data:" in body
    assert '"loss":0.5' in body or '"loss": 0.5' in body
    # Closes with a done status event.
    assert '"message":"done"' in body or '"message": "done"' in body


# ------------------------------------------------- #100 tool-outputs API

def test_global_tool_buffer_round_trip():
    from soup_cli.utils.tool_outputs import (
        get_global_tool_buffer,
        reset_global_tool_buffer,
    )

    reset_global_tool_buffer()
    buf = get_global_tool_buffer()
    buf.record_call(
        name="search",
        started_ts=1.0,
        duration_ms=12.5,
        success=True,
        output_preview="hits=3",
    )
    snap = buf.snapshot()
    assert len(snap) == 1
    assert snap[0].name == "search"


def test_api_tool_outputs_endpoint():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from soup_cli.ui.app import create_app
    from soup_cli.utils.tool_outputs import (
        get_global_tool_buffer,
        reset_global_tool_buffer,
    )

    reset_global_tool_buffer()
    buf = get_global_tool_buffer()
    buf.record_call(
        name="calculator",
        started_ts=1.23,
        duration_ms=4.5,
        success=True,
        output_preview="result=42",
    )
    buf.record_call(
        name="search",
        started_ts=2.34,
        duration_ms=99.9,
        success=False,
        output_preview="",
        error="timeout",
    )

    app_inst = create_app()
    client = TestClient(app_inst)
    response = client.get("/api/tool-outputs?limit=5")
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 2
    names = [r["name"] for r in payload["records"]]
    assert names == ["calculator", "search"]
    assert payload["records"][1]["error"] == "timeout"


def test_api_tool_outputs_rejects_out_of_bounds_limit():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from soup_cli.ui.app import create_app

    client = TestClient(create_app())
    # limit must be 1..1000
    assert client.get("/api/tool-outputs?limit=0").status_code == 422
    assert client.get("/api/tool-outputs?limit=1001").status_code == 422


# ---------------------------------------------------- #98 reasoning-parser

@pytest.mark.parametrize(
    "parser,raw,expected",
    [
        (
            "deepseek-r1",
            "<think>secret</think>final answer",
            "final answer",
        ),
        (
            "qwen3",
            "<think>step1\nstep2</think>\nresponse",
            "response",
        ),
        (
            "phi4",
            "no think tags here",
            "no think tags here",
        ),
        (
            "openthinker",
            "<|begin_of_thought|>cot<|end_of_thought|>visible",
            "visible",
        ),
    ],
)
def test_strip_reasoning_per_parser(parser, raw, expected):
    from soup_cli.utils.reasoning_parser import strip_reasoning

    assert strip_reasoning(raw, parser) == expected


def test_strip_reasoning_no_op_paths():
    from soup_cli.utils.reasoning_parser import strip_reasoning

    # None parser short-circuits.
    assert strip_reasoning("<think>x</think>y", None) == "<think>x</think>y"
    # Empty string parser short-circuits.
    assert strip_reasoning("hello", "") == "hello"
    # Unknown parser is silently no-op.
    assert strip_reasoning("<think>x</think>y", "unknown-parser") == (
        "<think>x</think>y"
    )
    # Non-string input passes through.
    assert strip_reasoning(123, "deepseek-r1") == 123  # type: ignore[arg-type]


def test_strip_reasoning_oversize_passthrough():
    from soup_cli.utils.reasoning_parser import strip_reasoning

    big = "<think>x</think>" + ("a" * 1_100_000)
    out = strip_reasoning(big, "deepseek-r1")
    assert out == big  # >1MiB returned unchanged


def test_strip_reasoning_idempotent():
    from soup_cli.utils.reasoning_parser import strip_reasoning

    out = strip_reasoning("<think>a</think>final", "deepseek-r1")
    assert strip_reasoning(out, "deepseek-r1") == out


def test_serve_create_app_accepts_reasoning_parser_kwarg():
    """`_create_app` must accept the new `reasoning_parser=` kwarg."""
    import inspect

    from soup_cli.commands.serve import _create_app

    sig = inspect.signature(_create_app)
    assert "reasoning_parser" in sig.parameters


def test_serve_help_lists_reasoning_parser_flag():
    runner = CliRunner()
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "--reasoning-parser" in result.output


# --------------------------------------------------- #15 tokenizer train

def test_tokenizer_train_help_listed():
    runner = CliRunner()
    result = runner.invoke(app, ["tokenizer", "--help"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "train" in result.output


def test_tokenizer_train_subcommand_help():
    runner = CliRunner()
    result = runner.invoke(app, ["tokenizer", "train", "--help"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "--vocab-size" in result.output
    assert "--input" in result.output
    assert "--output" in result.output


def test_tokenizer_train_rejects_out_of_cwd_input(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    outside = tmp_path.parent / "outside.jsonl"
    outside.write_text('{"text": "hi"}\n', encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "tokenizer", "train",
            "--input", str(outside),
            "--output", "out",
        ],
    )
    assert result.exit_code != 0
    assert "current working directory" in result.output


def test_tokenizer_train_rejects_vocab_size_bounds(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fixture = tmp_path / "corpus.jsonl"
    fixture.write_text('{"text": "hello world"}\n', encoding="utf-8")
    runner = CliRunner()
    # Too small.
    result = runner.invoke(
        app,
        [
            "tokenizer", "train",
            "--input", "corpus.jsonl",
            "--vocab-size", "10",
        ],
    )
    assert result.exit_code != 0
    assert "vocab-size" in result.output
    # Too large.
    result = runner.invoke(
        app,
        [
            "tokenizer", "train",
            "--input", "corpus.jsonl",
            "--vocab-size", "999999",
        ],
    )
    assert result.exit_code != 0


def test_tokenizer_train_happy_path(tmp_path, monkeypatch):
    pytest.importorskip("tokenizers")
    monkeypatch.chdir(tmp_path)
    fixture = tmp_path / "corpus.jsonl"
    rows = [
        {"text": "the quick brown fox jumps over the lazy dog"},
        {"text": "pack my box with five dozen liquor jugs"},
        {"text": "how vexingly quick daft zebras jump"},
    ] * 20
    fixture.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "tokenizer", "train",
            "--input", "corpus.jsonl",
            "--vocab-size", "300",
            "--output", "bpe_out",
            "--min-frequency", "1",
        ],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert (tmp_path / "bpe_out" / "tokenizer.json").is_file()
    assert (tmp_path / "bpe_out" / "vocab.json").is_file()
    vocab = json.loads(
        (tmp_path / "bpe_out" / "vocab.json").read_text(encoding="utf-8")
    )
    assert isinstance(vocab, dict)
    assert len(vocab) >= 1


# -------------------------------------------------- #28 backend auto-detect

def test_detect_backend_fallback_transformers():
    from soup_cli.utils.backend_detect import detect_backend

    assert detect_backend("does-not-exist") == "transformers"
    assert detect_backend("") == "transformers"
    assert detect_backend("abc\x00def") == "transformers"


def test_detect_backend_env_hint(monkeypatch):
    from soup_cli.utils.backend_detect import detect_backend

    monkeypatch.setenv("SOUP_BENCH_BACKEND", "VLLM")
    assert detect_backend("anything") == "vllm"
    monkeypatch.setenv("SOUP_BENCH_BACKEND", "bogus-name")
    # bogus env hint falls back to detection.
    out = detect_backend("anything")
    assert out in {"transformers", "mlx"}


def test_detect_backend_mlx_weights(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model_dir = tmp_path / "mlx_model"
    model_dir.mkdir()
    (model_dir / "weights.npz").write_bytes(b"\x00\x00")
    from soup_cli.utils.backend_detect import detect_backend

    assert detect_backend(str(model_dir)) == "mlx"


def test_detect_backend_transformers_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model_dir = tmp_path / "hf_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}),
        encoding="utf-8",
    )
    from soup_cli.utils.backend_detect import detect_backend

    assert detect_backend(str(model_dir)) == "transformers"


def test_detect_backend_config_malformed_falls_back(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model_dir = tmp_path / "junk"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("not-json{", encoding="utf-8")
    from soup_cli.utils.backend_detect import detect_backend

    assert detect_backend(str(model_dir)) == "transformers"


# ----------------------------------------------------- #26 bench percentiles

def test_bench_help_lists_percentile_flags():
    runner = CliRunner()
    result = runner.invoke(app, ["bench", "--help"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "--p50" in result.output
    assert "--p95" in result.output
    assert "--backend" in result.output


def test_bench_invalid_backend_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["bench", "fake-model", "--backend", "evilbackend"],
    )
    # Either exit 2 (Typer reject) or exit 1 (resolve failure later).
    assert result.exit_code != 0


# --------------------------------------------------- #95 ui --public + QR

def test_ui_help_includes_public_and_auth_token():
    runner = CliRunner()
    result = runner.invoke(app, ["ui", "--help"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "--public" in result.output
    assert "--auth-token" in result.output


def test_ui_show_token_via_custom_token(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    valid = "A" * 32  # urlsafe-base64 shape
    result = runner.invoke(
        app,
        [
            "ui",
            "--show-token",
            "--auth-token", valid,
            "--no-browser",
        ],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert valid in result.output


def test_ui_rejects_malformed_auth_token():
    pytest.importorskip("fastapi")
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "ui",
            "--auth-token", "short",  # < 16 chars
            "--show-token",
            "--no-browser",
        ],
    )
    assert result.exit_code != 0
    assert "auth-token" in result.output or "token" in result.output


# ------------------------------------------------ #12 example workflow doc

def test_train_event_buffer_snapshot_since_returns_only_new_events():
    from soup_cli.utils.sse_train_stream import TrainEvent
    from soup_cli.utils.train_event_buffer import TrainEventBuffer

    buffer = TrainEventBuffer()
    buffer.push(TrainEvent(type="metric", step=0))
    buffer.push(TrainEvent(type="metric", step=1))
    events, cursor = buffer.snapshot_since(0)
    assert [e.step for e in events] == [0, 1]
    assert cursor == 2

    buffer.push(TrainEvent(type="metric", step=2))
    events, cursor = buffer.snapshot_since(cursor)
    assert [e.step for e in events] == [2]
    assert cursor == 3

    # Caller cursor caught up — no new events.
    events, cursor = buffer.snapshot_since(cursor)
    assert events == []
    assert cursor == 3


def test_train_event_buffer_snapshot_since_rejects_non_int():
    from soup_cli.utils.train_event_buffer import TrainEventBuffer

    buffer = TrainEventBuffer()
    with pytest.raises(TypeError):
        buffer.snapshot_since("0")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        buffer.snapshot_since(True)


def test_train_event_buffer_concurrent_subscribers_isolated():
    """Multiple subscribers using snapshot_since each see every event."""
    from soup_cli.utils.sse_train_stream import TrainEvent
    from soup_cli.utils.train_event_buffer import TrainEventBuffer

    buffer = TrainEventBuffer()
    for step in range(3):
        buffer.push(TrainEvent(type="metric", step=step))

    a_events, _ = buffer.snapshot_since(0)
    b_events, _ = buffer.snapshot_since(0)
    assert [e.step for e in a_events] == [0, 1, 2]
    assert [e.step for e in b_events] == [0, 1, 2]


def test_tokenizer_train_rejects_symlink_input(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    real = tmp_path / "real.jsonl"
    real.write_text('{"text": "hello"}\n', encoding="utf-8")
    link = tmp_path / "link.jsonl"
    try:
        os.symlink(real, link)
    except (OSError, NotImplementedError, AttributeError):
        pytest.skip("symlinks not supported on this platform")
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "tokenizer", "train",
            "--input", "link.jsonl",
            "--output", "out",
        ],
    )
    assert result.exit_code != 0
    assert "symlink" in result.output.lower()


def test_train_event_buffer_cursor_and_clear():
    from soup_cli.utils.sse_train_stream import TrainEvent
    from soup_cli.utils.train_event_buffer import TrainEventBuffer

    buffer = TrainEventBuffer()
    assert buffer.cursor() == 0
    buffer.push(TrainEvent(type="metric", step=0))
    buffer.push(TrainEvent(type="metric", step=1))
    assert buffer.cursor() == 2
    buffer.clear()
    assert buffer.cursor() == 2  # cursor is monotonic; clear does NOT reset
    assert buffer.snapshot() == []


def test_train_event_buffer_snapshot_since_rollover():
    """`maxlen=3` then push 5 events — `snapshot_since(0)` returns the 3 retained."""
    from soup_cli.utils.sse_train_stream import TrainEvent
    from soup_cli.utils.train_event_buffer import TrainEventBuffer

    buffer = TrainEventBuffer(maxlen=3)
    for step in range(5):
        buffer.push(TrainEvent(type="metric", step=step))
    # Latest cursor is 5; only the last 3 events are retained.
    events, cursor = buffer.snapshot_since(0)
    assert [e.step for e in events] == [2, 3, 4]
    assert cursor == 5
    # Asking for events newer than cursor 4 (step=3) returns the tail.
    events, cursor = buffer.snapshot_since(4)
    assert [e.step for e in events] == [4]


def test_set_auth_token_concurrent_rotation_safe():
    """Concurrent set_auth_token + get_auth_token never observes a partial token."""
    import threading

    from soup_cli.ui.app import get_auth_token, set_auth_token

    valid_a = "A" * 32
    valid_b = "B" * 32
    set_auth_token(valid_a)

    errors: list[str] = []

    def _writer():
        for _ in range(100):
            set_auth_token(valid_a)
            set_auth_token(valid_b)

    def _reader():
        for _ in range(100):
            tok = get_auth_token()
            if tok not in (valid_a, valid_b):
                errors.append(f"torn token: {tok!r}")

    threads = [threading.Thread(target=_writer) for _ in range(2)]
    threads += [threading.Thread(target=_reader) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


def test_set_auth_token_rejects_bool_and_non_str():
    from soup_cli.ui.app import get_auth_token, set_auth_token

    before = get_auth_token()
    with pytest.raises(TypeError):
        set_auth_token(True)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        set_auth_token(42)  # type: ignore[arg-type]
    # Token must be unchanged after rejected rotation.
    assert get_auth_token() == before


def test_strip_reasoning_swallows_null_byte_parser():
    from soup_cli.utils.reasoning_parser import strip_reasoning

    out = strip_reasoning("<think>x</think>final", "\x00")
    assert out == "<think>x</think>final"


def test_strip_reasoning_crlf_residue():
    from soup_cli.utils.reasoning_parser import strip_reasoning

    out = strip_reasoning("<think>a</think>\r\n\r\nfinal", "deepseek-r1")
    # Both \r and \n stripped — content preserved.
    assert out == "final"


def test_strip_reasoning_fast_path_skips_regex():
    """Inputs without the marker token bypass the regex entirely."""
    from soup_cli.utils.reasoning_parser import strip_reasoning

    # Long input WITHOUT <think marker — returns unchanged in O(n).
    big = "a" * 200_000
    assert strip_reasoning(big, "deepseek-r1") == big
    # OpenThinker marker absent — unchanged.
    assert strip_reasoning(big, "openthinker") == big


def test_strip_reasoning_multiple_blocks():
    from soup_cli.utils.reasoning_parser import strip_reasoning

    out = strip_reasoning(
        "<think>a</think>mid<think>b</think>final", "deepseek-r1"
    )
    assert out == "midfinal"


def test_detect_backend_rejects_symlinked_mlx_weights(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    real_weights = tmp_path / "real.npz"
    real_weights.write_bytes(b"\x00")
    link = model_dir / "weights.npz"
    try:
        os.symlink(real_weights, link)
    except (OSError, NotImplementedError, AttributeError):
        pytest.skip("symlinks not supported on this platform")
    from soup_cli.utils.backend_detect import detect_backend

    # Symlinked weights.npz must NOT trigger MLX dispatch.
    assert detect_backend(str(model_dir)) == "transformers"


def test_detect_backend_mlx_model_type_in_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "mlx_lm_llama"}),
        encoding="utf-8",
    )
    from soup_cli.utils.backend_detect import detect_backend

    assert detect_backend(str(model_dir)) == "mlx"


def test_detect_backend_config_non_dict_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("[1, 2, 3]", encoding="utf-8")
    from soup_cli.utils.backend_detect import detect_backend

    assert detect_backend(str(model_dir)) == "transformers"


def test_tokenizer_train_plaintext_corpus(tmp_path, monkeypatch):
    """`.txt` extractor branch (different from JSONL)."""
    pytest.importorskip("tokenizers")
    monkeypatch.chdir(tmp_path)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "\n".join(["the quick brown fox jumps over the lazy dog"] * 80) + "\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "tokenizer", "train",
            "--input", "corpus.txt",
            "--vocab-size", "300",
            "--output", "bpe_out",
            "--min-frequency", "1",
        ],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert (tmp_path / "bpe_out" / "tokenizer.json").is_file()


def test_tokenizer_train_rejects_min_frequency_zero(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "corpus.jsonl").write_text(
        '{"text": "hello"}\n', encoding="utf-8"
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "tokenizer", "train",
            "--input", "corpus.jsonl",
            "--min-frequency", "0",
        ],
    )
    assert result.exit_code != 0
    assert "min-frequency" in result.output


def test_tokenizer_train_special_token_dedup_and_validation(tmp_path, monkeypatch):
    pytest.importorskip("tokenizers")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "corpus.jsonl").write_text(
        "\n".join(['{"text": "hello world"}'] * 20) + "\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    # NUL byte in special-token rejected.
    result = runner.invoke(
        app,
        [
            "tokenizer", "train",
            "--input", "corpus.jsonl",
            "--special-token", "<pad>\x00bad",
            "--vocab-size", "300",
            "--min-frequency", "1",
        ],
    )
    assert result.exit_code != 0
    assert "special-token" in result.output


def test_bench_p50_p95_runs_with_help_only():
    """Smoke: --p50/--p95 flags are wired (without spinning up a real model)."""
    runner = CliRunner()
    result = runner.invoke(app, ["bench", "--help"])
    assert result.exit_code == 0
    # The flag descriptions must mention the v0.53.9 release tag so future
    # patches don't silently drop the percentile rows.
    assert "p50" in result.output and "p95" in result.output


def test_reset_global_tool_buffer_clears_state():
    from soup_cli.utils.tool_outputs import (
        get_global_tool_buffer,
        reset_global_tool_buffer,
    )

    buf = get_global_tool_buffer()
    buf.record_call(
        name="ping",
        started_ts=1.0,
        duration_ms=0.5,
        success=True,
        output_preview="ok",
    )
    assert len(buf.snapshot()) >= 1
    reset_global_tool_buffer()
    fresh = get_global_tool_buffer()
    assert fresh.snapshot() == []


def test_synthetic_workflow_doc_exists():
    repo_root = Path(__file__).resolve().parent.parent
    doc = repo_root / "examples" / "synthetic_workflow.md"
    assert doc.is_file()
    body = doc.read_text(encoding="utf-8")
    # End-to-end walkthrough names every step.
    assert "soup data generate" in body
    assert "soup data filter" in body
    assert "soup data score" in body
    assert "soup train" in body
