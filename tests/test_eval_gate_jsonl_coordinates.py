"""#3247: a malformed judge JSONL row reports file-level line and column.

The decoder is fed one stripped line at a time, so its coordinates are
row-local (always line 1, column relative to the stripped text). The
loader must rebase them onto the physical file position the way it
already does for malformed objects and fields.
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
