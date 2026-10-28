"""#1854 — ``soup distill-prompt`` must not report "done" and empty an existing
output when no trace row yields a prompt.

``prepare_distill_dataset`` skipped every row ``extract_prompt`` could not read
without counting it. With no prompt there was no provider call, so the #1274
guard (``stats.failures``) never fired: the command printed ``Rows: 0``, exited
0 and replaced the output path with a 0-byte file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

import pytest
from typer.testing import CliRunner

from tests.conftest import strip_ansi

runner = CliRunner()

# The three shapes from the issue: none matches ``extract_prompt``.
_UNREADABLE_ROWS = [
    {"id": "t1", "input": [{"role": "user", "content": "What is 2+2?"}], "output": "4"},
    {
        "id": "t2",
        "inputs": {"messages": [{"role": "user", "content": "Capital of France?"}]},
        "outputs": {"text": "Paris"},
    },
    {"id": "t3", "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]},
]
_PRIOR_OUTPUT = b'{"messages": [{"role": "user", "content": "kept from an earlier run"}]}\n'


def _terminal_text(result) -> str:
    return re.sub(r"\s+", " ", strip_ansi(result.output))


def _write_traces(tmp_path: Path, rows: List[dict]) -> Path:
    path = tmp_path / "traces.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


@pytest.fixture
def provider_calls(monkeypatch) -> List[str]:
    """Stub provider: records each prompt and answers it."""
    import soup_cli.utils.prompt_distill as distill_mod

    calls: List[str] = []

    def reply(prompt: str) -> dict:
        calls.append(prompt)
        return {"text": f"answer to {prompt}"}

    monkeypatch.setattr(distill_mod, "_build_provider_fn", lambda *a, **k: reply)
    return calls


def _run_distill(tmp_path: Path, monkeypatch, *extra: str):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    return runner.invoke(
        app,
        [
            "distill-prompt",
            "--traces", "traces.jsonl",
            "--teacher", "big-model",
            "--student", "small-model",
            "--output", "distilled.jsonl",
            *extra,
        ],
    )


@pytest.mark.parametrize("rows", [_UNREADABLE_ROWS, []], ids=["unreadable", "empty"])
def test_no_prompt_refuses_and_keeps_existing_output(
    tmp_path, monkeypatch, provider_calls, rows
) -> None:
    _write_traces(tmp_path, rows)
    output = tmp_path / "distilled.jsonl"
    output.write_bytes(_PRIOR_OUTPUT)

    result = _run_distill(tmp_path, monkeypatch)
    text = _terminal_text(result)

    assert result.exit_code == 3, (text, repr(result.exception))
    assert output.read_bytes() == _PRIOR_OUTPUT
    assert provider_calls == []
    assert f"0 of {len(rows)} trace rows had a prompt" in text
    assert "prompt / input / instruction / question / query" in text
    assert "messages" in text
    assert "done" not in text


def test_lines_without_a_json_object_are_not_rows(tmp_path, monkeypatch, provider_calls) -> None:
    (tmp_path / "traces.jsonl").write_text('not json\n[1, 2]\n"text"\n', encoding="utf-8")
    output = tmp_path / "distilled.jsonl"
    output.write_bytes(_PRIOR_OUTPUT)

    result = _run_distill(tmp_path, monkeypatch)
    text = _terminal_text(result)

    assert result.exit_code == 3, (text, repr(result.exception))
    assert "0 of 0 trace rows had a prompt" in text
    assert "only lines holding a JSON object count as rows" in text
    assert output.read_bytes() == _PRIOR_OUTPUT
    assert provider_calls == []


@pytest.mark.parametrize("rows", [_UNREADABLE_ROWS, []], ids=["unreadable", "empty"])
def test_no_prompt_creates_no_output(tmp_path, monkeypatch, provider_calls, rows) -> None:
    _write_traces(tmp_path, rows)

    result = _run_distill(tmp_path, monkeypatch)

    assert result.exit_code == 3, (result.output, repr(result.exception))
    assert not (tmp_path / "distilled.jsonl").exists()


@pytest.mark.parametrize("strategy", ["sft", "preference"])
def test_no_prompt_refuses_before_building_a_provider(tmp_path, monkeypatch, strategy) -> None:
    import soup_cli.utils.prompt_distill as distill_mod

    def no_provider(*args, **kwargs):
        pytest.fail("no provider should be built when no row has a prompt")

    monkeypatch.setattr(distill_mod, "_build_provider_fn", no_provider)
    _write_traces(tmp_path, _UNREADABLE_ROWS)

    result = _run_distill(tmp_path, monkeypatch, "--strategy", strategy)

    assert result.exit_code == 3, (result.output, repr(result.exception))


def test_mixed_rows_write_readable_ones_and_report_skipped(
    tmp_path, monkeypatch, provider_calls
) -> None:
    _write_traces(tmp_path, [_UNREADABLE_ROWS[0], {"prompt": "What is 3+3?"}, _UNREADABLE_ROWS[2]])

    result = _run_distill(tmp_path, monkeypatch)
    text = _terminal_text(result)

    assert result.exit_code == 0, (text, repr(result.exception))
    lines = (tmp_path / "distilled.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["messages"][0]["content"] for line in lines] == ["What is 3+3?"]
    assert provider_calls == ["What is 3+3?"]
    assert "Rows: 1 (2 of 3 trace rows skipped: no prompt found)" in text


def test_all_readable_control_reports_no_skipped_rows(
    tmp_path, monkeypatch, provider_calls
) -> None:
    _write_traces(tmp_path, [{"prompt": "q1"}, {"messages": [{"role": "user", "content": "q2"}]}])

    result = _run_distill(tmp_path, monkeypatch)
    text = _terminal_text(result)

    assert result.exit_code == 0, (text, repr(result.exception))
    assert len((tmp_path / "distilled.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert "skipped" not in text


def test_max_rows_still_caps_rows_written_around_skipped_rows(
    tmp_path, monkeypatch, provider_calls
) -> None:
    _write_traces(
        tmp_path, [_UNREADABLE_ROWS[0], {"prompt": "q1"}, _UNREADABLE_ROWS[1], {"prompt": "q2"}]
    )

    result = _run_distill(tmp_path, monkeypatch, "--max-rows", "1")
    text = _terminal_text(result)

    assert result.exit_code == 0, (text, repr(result.exception))
    assert len((tmp_path / "distilled.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert provider_calls == ["q1"]
    assert "2 of 4 trace rows skipped" in text


def test_prepare_counts_rows_and_raises_without_touching_output(tmp_path, monkeypatch) -> None:
    from soup_cli.utils.prompt_distill import (
        NoTracePromptsError,
        TraceRowStats,
        build_distill_prompt_plan,
        prepare_distill_dataset,
    )

    monkeypatch.chdir(tmp_path)
    _write_traces(tmp_path, _UNREADABLE_ROWS)
    (tmp_path / "out.jsonl").write_bytes(_PRIOR_OUTPUT)
    plan = build_distill_prompt_plan(
        traces_path="traces.jsonl",
        teacher="big-model",
        student="small-model",
        strategy="sft",
        output_path="out.jsonl",
    )
    trace_stats = TraceRowStats()

    with pytest.raises(NoTracePromptsError, match="0 of 3 trace rows had a prompt"):
        prepare_distill_dataset(
            plan,
            teacher_fn=lambda prompt: pytest.fail("teacher must not be called"),
            trace_stats=trace_stats,
        )

    assert (trace_stats.read, trace_stats.skipped) == (3, 3)
    assert (tmp_path / "out.jsonl").read_bytes() == _PRIOR_OUTPUT
