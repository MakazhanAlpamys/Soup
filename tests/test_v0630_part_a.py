"""v0.63.0 Part A — soup ingest universal trace importer tests."""

from __future__ import annotations

import dataclasses
import json

import pytest
from typer.testing import CliRunner

runner = CliRunner()


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_module_imports():
    from soup_cli.utils import ingest_sources  # noqa: F401

    assert hasattr(ingest_sources, "SUPPORTED_INGEST_SOURCES")
    assert hasattr(ingest_sources, "TraceRecord")
    assert hasattr(ingest_sources, "validate_source_name")
    assert hasattr(ingest_sources, "parse_langfuse")
    assert hasattr(ingest_sources, "parse_langsmith")
    assert hasattr(ingest_sources, "parse_helicone")
    assert hasattr(ingest_sources, "parse_openpipe")
    assert hasattr(ingest_sources, "parse_otel")
    assert hasattr(ingest_sources, "parse_openai_stored")
    assert hasattr(ingest_sources, "ingest_traces")
    assert hasattr(ingest_sources, "resolve_auth_env")


def test_supported_sources_exact():
    from soup_cli.utils.ingest_sources import SUPPORTED_INGEST_SOURCES

    assert SUPPORTED_INGEST_SOURCES == frozenset(
        {"langfuse", "langsmith", "helicone", "openpipe", "otel", "openai-stored"}
    )


def test_trace_record_frozen():
    from soup_cli.utils.ingest_sources import TraceRecord

    rec = TraceRecord(
        trace_id="abc",
        prompt="hello",
        output="world",
        source="langfuse",
        signal="none",
        metadata={},
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.prompt = "tampered"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# validate_source_name
# ---------------------------------------------------------------------------


def test_validate_source_name_happy():
    from soup_cli.utils.ingest_sources import validate_source_name

    for name in ["langfuse", "langsmith", "helicone", "openpipe", "otel", "openai-stored"]:
        assert validate_source_name(name) == name


def test_validate_source_name_case_insensitive():
    from soup_cli.utils.ingest_sources import validate_source_name

    assert validate_source_name("LANGFUSE") == "langfuse"
    assert validate_source_name("OpenPipe") == "openpipe"


def test_validate_source_name_unknown():
    from soup_cli.utils.ingest_sources import validate_source_name

    with pytest.raises(ValueError, match="unknown"):
        validate_source_name("evilcorp")


@pytest.mark.parametrize("bad", [None, 123, True, "", "x" * 33, "lang\x00fuse"])
def test_validate_source_name_rejects(bad):
    from soup_cli.utils.ingest_sources import validate_source_name

    with pytest.raises((TypeError, ValueError)):
        validate_source_name(bad)


# ---------------------------------------------------------------------------
# Per-source parsers (offline, no network)
# ---------------------------------------------------------------------------


def test_parse_langfuse_basic():
    from soup_cli.utils.ingest_sources import parse_langfuse

    events = [
        {
            "id": "trace-1",
            "input": {"messages": [{"role": "user", "content": "hello"}]},
            "output": "hi there",
        },
        {
            "id": "trace-2",
            "input": "raw prompt",
            "output": {"content": "raw response"},
        },
    ]
    rows = list(parse_langfuse(events))
    assert len(rows) == 2
    assert rows[0].trace_id == "trace-1"
    assert rows[0].prompt == "hello"
    assert rows[0].output == "hi there"
    assert rows[0].source == "langfuse"
    assert rows[1].prompt == "raw prompt"
    assert rows[1].output == "raw response"


def test_parse_langfuse_skips_non_dict():
    from soup_cli.utils.ingest_sources import parse_langfuse

    rows = list(parse_langfuse(["string-row", 42, None, {"input": "x", "output": "y"}]))
    assert len(rows) == 1


def test_parse_langfuse_missing_fields():
    from soup_cli.utils.ingest_sources import parse_langfuse

    rows = list(parse_langfuse([{"id": "x"}, {"input": "only-input"}, {"output": "only-output"}]))
    assert rows == []


def test_parse_langsmith_basic():
    from soup_cli.utils.ingest_sources import parse_langsmith

    events = [
        {
            "id": "run-1",
            "inputs": {"messages": [{"role": "user", "content": "Q"}]},
            "outputs": {"generations": [[{"text": "A"}]]},
            "feedback_stats": {"thumbs": {"avg": 1.0}},
        },
    ]
    rows = list(parse_langsmith(events))
    assert len(rows) == 1
    assert rows[0].prompt == "Q"
    assert rows[0].output == "A"
    assert rows[0].source == "langsmith"
    assert rows[0].signal == "thumbs_up"


def test_parse_helicone_basic():
    from soup_cli.utils.ingest_sources import parse_helicone

    events = [
        {
            "request_id": "req-1",
            "request": {"body": {"messages": [{"role": "user", "content": "hi"}]}},
            "response": {"body": {"choices": [{"message": {"content": "bye"}}]}},
        }
    ]
    rows = list(parse_helicone(events))
    assert len(rows) == 1
    assert rows[0].trace_id == "req-1"
    assert rows[0].prompt == "hi"
    assert rows[0].output == "bye"
    assert rows[0].source == "helicone"


def test_parse_openpipe_basic():
    from soup_cli.utils.ingest_sources import parse_openpipe

    events = [
        {
            "id": "op-1",
            "messages": [{"role": "user", "content": "hello"}],
            "response": "world",
        }
    ]
    rows = list(parse_openpipe(events))
    assert len(rows) == 1
    assert rows[0].source == "openpipe"


def test_parse_otel_basic():
    from soup_cli.utils.ingest_sources import parse_otel

    spans = [
        {
            "traceId": "abc123",
            "attributes": {
                "llm.prompt": "what is 2+2",
                "llm.completion": "4",
                "llm.model": "gpt-4o",
            },
        }
    ]
    rows = list(parse_otel(spans))
    assert len(rows) == 1
    assert rows[0].prompt == "what is 2+2"
    assert rows[0].output == "4"
    assert rows[0].source == "otel"
    assert rows[0].metadata.get("model") == "gpt-4o"


def test_parse_otel_skips_non_llm_spans():
    from soup_cli.utils.ingest_sources import parse_otel

    spans = [
        {"attributes": {"http.method": "GET"}},
        {"attributes": {"llm.prompt": "Q", "llm.completion": "A"}},
    ]
    rows = list(parse_otel(spans))
    assert len(rows) == 1


def test_parse_openai_stored_basic():
    from soup_cli.utils.ingest_sources import parse_openai_stored

    events = [
        {
            "id": "chatcmpl-1",
            "input": [{"role": "user", "content": "hello"}],
            "output": [{"role": "assistant", "content": "hi"}],
        }
    ]
    rows = list(parse_openai_stored(events))
    assert len(rows) == 1
    assert rows[0].prompt == "hello"
    assert rows[0].output == "hi"
    assert rows[0].source == "openai-stored"


# ---------------------------------------------------------------------------
# ingest_traces dispatch + file containment
# ---------------------------------------------------------------------------


def test_ingest_traces_dispatches_to_parser(tmp_path, monkeypatch):
    """Smoke: ingest_traces routes a JSONL log through the matching parser."""
    from soup_cli.utils.ingest_sources import ingest_traces

    monkeypatch.chdir(tmp_path)
    log_file = tmp_path / "langfuse.jsonl"
    payload = [
        {"id": "t1", "input": "hi", "output": "there"},
        {"id": "t2", "input": "x", "output": "y"},
    ]
    log_file.write_text("\n".join(json.dumps(r) for r in payload), encoding="utf-8")

    rows = list(ingest_traces(source="langfuse", path=str(log_file)))
    assert len(rows) == 2
    assert all(r.source == "langfuse" for r in rows)


def test_ingest_traces_rejects_outside_cwd(tmp_path, monkeypatch):
    from soup_cli.utils.ingest_sources import ingest_traces

    monkeypatch.chdir(tmp_path)
    elsewhere = tmp_path.parent / "stray.jsonl"
    elsewhere.write_text('{"input":"x","output":"y"}\n', encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="outside"):
            list(ingest_traces(source="langfuse", path=str(elsewhere)))
    finally:
        if elsewhere.exists():
            elsewhere.unlink()


def test_ingest_traces_rejects_unknown_source(tmp_path, monkeypatch):
    from soup_cli.utils.ingest_sources import ingest_traces

    monkeypatch.chdir(tmp_path)
    log_file = tmp_path / "x.jsonl"
    log_file.write_text('{"input":"x","output":"y"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        list(ingest_traces(source="bogus", path=str(log_file)))


def test_ingest_traces_missing_file(tmp_path, monkeypatch):
    from soup_cli.utils.ingest_sources import ingest_traces

    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        list(ingest_traces(source="langfuse", path=str(tmp_path / "missing.jsonl")))


def test_ingest_traces_rejects_null_byte_path(tmp_path, monkeypatch):
    from soup_cli.utils.ingest_sources import ingest_traces

    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        list(ingest_traces(source="langfuse", path="bad\x00path.jsonl"))


def test_ingest_traces_caps_lines(tmp_path, monkeypatch):
    """Should not hang on extremely large files; respect _MAX_INGEST_LINES."""
    from soup_cli.utils import ingest_sources

    monkeypatch.chdir(tmp_path)
    log_file = tmp_path / "huge.jsonl"
    # Write enough valid rows to trip cap (use very small cap via monkeypatch).
    monkeypatch.setattr(ingest_sources, "_MAX_INGEST_LINES", 5)
    rows = [{"input": "x", "output": "y"} for _ in range(20)]
    log_file.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    result = list(ingest_sources.ingest_traces(source="langfuse", path=str(log_file)))
    assert len(result) <= 5


def test_ingest_traces_skips_malformed_lines(tmp_path, monkeypatch):
    from soup_cli.utils.ingest_sources import ingest_traces

    monkeypatch.chdir(tmp_path)
    log_file = tmp_path / "mixed.jsonl"
    log_file.write_text(
        '{"input":"x","output":"y"}\n'
        "not-json\n"
        '{"input":"a","output":"b"}\n',
        encoding="utf-8",
    )
    rows = list(ingest_traces(source="langfuse", path=str(log_file)))
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# Auth env resolution
# ---------------------------------------------------------------------------


def test_resolve_auth_env_each_source(monkeypatch):
    from soup_cli.utils.ingest_sources import resolve_auth_env

    monkeypatch.setenv("LANGFUSE_KEY", "lf-key")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-key")
    monkeypatch.setenv("HELICONE_API_KEY", "h-key")
    monkeypatch.setenv("OPENPIPE_API_KEY", "op-key")
    monkeypatch.setenv("OPENAI_API_KEY", "oai-key")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "Authorization=Bearer x")

    assert resolve_auth_env("langfuse") == "lf-key"
    assert resolve_auth_env("langsmith") == "ls-key"
    assert resolve_auth_env("helicone") == "h-key"
    assert resolve_auth_env("openpipe") == "op-key"
    assert resolve_auth_env("openai-stored") == "oai-key"
    assert resolve_auth_env("otel") is not None


def test_resolve_auth_env_missing(monkeypatch):
    from soup_cli.utils.ingest_sources import resolve_auth_env

    for env in [
        "LANGFUSE_KEY",
        "LANGSMITH_API_KEY",
        "HELICONE_API_KEY",
        "OPENPIPE_API_KEY",
        "OPENAI_API_KEY",
        "OTEL_EXPORTER_OTLP_HEADERS",
    ]:
        monkeypatch.delenv(env, raising=False)
    assert resolve_auth_env("langfuse") is None


def test_resolve_auth_env_unknown_raises():
    from soup_cli.utils.ingest_sources import resolve_auth_env

    with pytest.raises(ValueError):
        resolve_auth_env("evilcorp")


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_cli_ingest_help():
    from soup_cli.cli import app

    result = runner.invoke(app, ["ingest", "--help"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    # Sources + path must surface
    assert "langfuse" in result.output.lower() or "source" in result.output.lower()


def test_cli_ingest_happy(tmp_path, monkeypatch):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    log = tmp_path / "lf.jsonl"
    log.write_text(
        '{"id":"t1","input":"a","output":"b"}\n{"id":"t2","input":"c","output":"d"}\n',
        encoding="utf-8",
    )
    out = tmp_path / "traces.jsonl"

    result = runner.invoke(
        app,
        ["ingest", "--source", "langfuse", "--logs", str(log), "--output", str(out)],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert out.exists()
    content = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 2
    for line in content:
        row = json.loads(line)
        assert row["source"] == "langfuse"
        assert "prompt" in row and "output" in row


def test_cli_ingest_unknown_source(tmp_path, monkeypatch):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    log = tmp_path / "x.jsonl"
    log.write_text("{}\n", encoding="utf-8")
    result = runner.invoke(
        app,
        ["ingest", "--source", "bogus", "--logs", str(log)],
    )
    assert result.exit_code != 0
    assert "unknown" in result.output.lower() or "source" in result.output.lower()


def test_cli_ingest_outside_cwd_rejected(tmp_path, monkeypatch):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    outside = tmp_path.parent / "stray_log.jsonl"
    outside.write_text('{"input":"x","output":"y"}\n', encoding="utf-8")
    try:
        result = runner.invoke(
            app,
            ["ingest", "--source", "langfuse", "--logs", str(outside)],
        )
        assert result.exit_code != 0
        assert "outside" in result.output.lower()
    finally:
        if outside.exists():
            outside.unlink()


def test_cli_ingest_pii_panel_prints(tmp_path, monkeypatch):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    log = tmp_path / "lf.jsonl"
    log.write_text('{"input":"x","output":"y"}\n', encoding="utf-8")
    out = tmp_path / "out.jsonl"

    result = runner.invoke(
        app,
        ["ingest", "--source", "langfuse", "--logs", str(log), "--output", str(out)],
    )
    assert result.exit_code == 0
    # PII reminder shown
    assert "pii" in result.output.lower() or "sensitive" in result.output.lower()
