"""#3247: a malformed judge JSONL row reports file-level line and column.

The decoder is fed one row per line, so its coordinates are row-local only
when the line ending is kept; the loader strips the line ending alone, so
leading indentation and trailing spaces stay in the coordinates, which
therefore match the physical file position the way they already do for
malformed objects and fields.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from soup_cli.eval import gate


def task() -> gate.GateTask:
    return gate.GateTask(
        name="judge-control", type="judge", threshold=0.0,
        prompts="prompts.jsonl", judge_model="ollama://synthetic",
    )


def test_syntax_error_reports_the_physical_line_and_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    Path("prompts.jsonl").write_text(
        '{"prompt":"valid"}\n'
        "\n"
        "\n"
        '  {"prompt": }\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as caught:
        gate._run_judge_task(task(), lambda prompt: "synthetic")
    assert str(caught.value) == (
        "judge task 'judge-control' file 'prompts.jsonl' line 4 column 14: "
        "Expecting value"
    )


def test_syntax_error_on_an_indented_row_with_trailing_spaces_reports_row_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    Path("prompts.jsonl").write_text(
        json.dumps({"prompt": "valid"}) + "\n"
        + '  {"prompt":  \n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as caught:
        gate._run_judge_task(task(), lambda prompt: "synthetic")
    assert "line 2 column 15: Expecting value" in str(caught.value)


def test_run_gate_returns_task_error_without_generation_or_judging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    Path("prompts.jsonl").write_text(
        json.dumps({"prompt": "valid"}) + "\n" + '{"prompt": }\n',
        encoding="utf-8",
    )
    suite = gate.EvalSuite(suite="s", tasks=[task()])
    seen: list[str] = []

    def generate_fn(prompt: str) -> str:
        seen.append(prompt)
        return "synthetic"

    result = gate.run_gate(suite, generate_fn=generate_fn)
    assert seen == []
    assert len(result.task_results) == 1
    task_result = result.task_results[0]
    assert task_result.score is None
    assert task_result.error is not None
    assert task_result.error.startswith("invalid JSONL: judge task 'judge-control'")
    assert "file 'prompts.jsonl' line 2 column 12: Expecting value" in task_result.error
    assert not task_result.passed


def test_syntax_error_on_an_unindented_row_keeps_column_1_based(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch: pytest.MonkeyPatch
    monkeypatch.chdir(tmp_path)
    Path("prompts.jsonl").write_text(
        json.dumps({"prompt": "valid"}) + "\n" + "{oops}\n", encoding="utf-8",
    )
    with pytest.raises(ValueError) as caught:
        gate._run_judge_task(task(), lambda prompt: "synthetic")
    assert "line 2 column 2: Expecting property name" in str(caught.value)


def test_valid_rows_still_load_after_the_rebase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    Path("prompts.jsonl").write_text(
        "\n"
        + json.dumps({"prompt": "p"}) + "\n"
        + "   " + json.dumps({"prompt": "q", "category": "c"}) + "\n",
        encoding="utf-8",
    )
    from soup_cli.eval.judge import JudgeEvaluator

    monkeypatch.setattr(
        JudgeEvaluator, "_call_llm",
        lambda self, prompt: json.dumps(
            {"scores": {"helpfulness": 5, "accuracy": 5, "safety": 5}}
        ),
    )
    seen: list[str] = []
    gate._run_judge_task(task(), lambda prompt: seen.append(prompt) or "ok")
    assert seen == ["p", "q"]


def test_row_wrapped_in_unicode_blanks_still_parses_and_keeps_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    Path("prompts.jsonl").write_text(
        "\u00a0" + json.dumps({"prompt": "valid"}) + "\u00a0\r\n",
        encoding="utf-8",
    )
    # previously accepted input keeps passing: surrounding unicode blanks
    # are stripped before decoding, exactly like the base loader.
    from soup_cli.eval.judge import JudgeEvaluator

    monkeypatch.setattr(
        JudgeEvaluator, "_call_llm",
        lambda self, prompt: json.dumps(
            {"scores": {"helpfulness": 5, "accuracy": 5, "safety": 5}},
        ),
    )
    gate._run_judge_task(task(), lambda prompt: "synthetic")
    Path("prompts.jsonl").write_text(
        "\u00a0" + '{"prompt": }' + "\u00a0\n", encoding="utf-8"
    )
    # a malformed row reports the physical column past the leading blank.
    with pytest.raises(ValueError) as caught:
        gate._run_judge_task(task(), lambda prompt: "synthetic")
    assert "line 1 column 13: Expecting value" in str(caught.value)
