"""Judge JSONL validates row shape and prompt type before that row's generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import eval as eval_command
from soup_cli.eval import gate, quant_check
from soup_cli.eval.judge import JudgeEvaluator
from tests.conftest import strip_ansi

BAD_ROWS = [None, 1, True, [], "text"]
BAD_PROMPTS = [None, 1, True, [], {}]
INVALID = [
    *((row, "row", type(row).__name__) for row in BAD_ROWS),
    *(({"prompt": value}, "prompt", type(value).__name__) for value in BAD_PROMPTS),
]
IDS = ["null-row", "number-row", "bool-row", "array-row", "string-row",
       "null-prompt", "number-prompt", "bool-prompt", "array-prompt", "object-prompt"]


@pytest.fixture
def calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    monkeypatch.chdir(tmp_path)
    recorded: dict[str, list] = {"generate": [], "judge": []}

    def judge(self: JudgeEvaluator, prompt: str) -> str:
        recorded["judge"].append(prompt)
        return json.dumps({"scores": {"helpfulness": 5, "accuracy": 5, "safety": 5}})

    monkeypatch.setattr(JudgeEvaluator, "_call_llm", judge)
    monkeypatch.setattr(
        quant_check, "make_model_generator",
        lambda model: lambda prompt: recorded["generate"].append(prompt) or "synthetic",
    )
    monkeypatch.setattr(eval_command, "console", Console(width=240))
    return recorded


def task() -> gate.GateTask:
    return gate.GateTask(
        name="judge-control", type="judge", threshold=0.0,
        prompts="prompts.jsonl", judge_model="ollama://synthetic",
    )


def write_input(row: object) -> None:
    Path("prompts.jsonl").write_text("\n \n" + json.dumps(row) + "\n", encoding="utf-8")


def expected_message(field: str, kind: str, line: int = 3) -> str:
    location = f"judge task 'judge-control' file 'prompts.jsonl' line {line}"
    detail = "row must be an object" if field == "row" else "field 'prompt' must be a string"
    return f"{location}: {detail}, got {kind}"


@pytest.mark.parametrize("row,field,kind", INVALID, ids=IDS)
def test_helper_refuses_invalid_row_before_generation(
    row: object, field: str, kind: str, calls: dict[str, list],
) -> None:
    write_input(row)
    with pytest.raises(ValueError) as caught:
        gate._run_judge_task(task(), lambda prompt: calls["generate"].append(prompt) or "synthetic")
    assert str(caught.value) == expected_message(field, kind)
    assert calls == {"generate": [], "judge": []}


@pytest.mark.parametrize("row,field,kind", INVALID, ids=IDS)
def test_run_gate_keeps_invalid_row_unscored(
    row: object, field: str, kind: str, calls: dict[str, list],
) -> None:
    write_input(row)
    result = gate.run_gate(
        gate.EvalSuite(suite="judge-control", tasks=[task()]),
        generate_fn=lambda prompt: calls["generate"].append(prompt) or "synthetic",
    )
    assert not result.passed
    scored = result.task_results[0]
    assert scored.score is None and scored.delta is None and not scored.passed
    assert scored.error == expected_message(field, kind)
    assert calls == {"generate": [], "judge": []}


def write_suite() -> None:
    Path("gate.yaml").write_text(yaml.safe_dump({
        "suite": "judge-control", "tasks": [task().model_dump(exclude_none=True)],
    }), encoding="utf-8")


@pytest.mark.parametrize("row,field,kind", INVALID, ids=IDS)
def test_cli_keeps_existing_failed_gate_exit_and_names_input(
    row: object, field: str, kind: str, calls: dict[str, list],
) -> None:
    write_input(row)
    write_suite()
    result = CliRunner().invoke(app, [
        "eval", "gate", "--suite", "gate.yaml", "--model", "synthetic",
    ])
    output = " ".join(strip_ansi(result.output).split())
    assert result.exit_code == 2, (output, repr(result.exception))
    assert isinstance(result.exception, SystemExit)
    assert expected_message(field, kind) in output
    assert "GATE FAILED" in output and "GATE PASSED" not in output
    assert calls == {"generate": [], "judge": []}


@pytest.mark.parametrize("row,prompt", [({"prompt": "valid"}, "valid"), ({}, ""),
                                      ({"prompt": ""}, "")])
def test_valid_missing_and_empty_prompts_are_preserved(
    row: dict, prompt: str, calls: dict[str, list],
) -> None:
    write_input(row)
    assert gate._run_judge_task(
        task(), lambda value: calls["generate"].append(value) or "synthetic",
    ) == 1.0
    assert calls["generate"] == [prompt] and len(calls["judge"]) == 1


def test_valid_row_before_bad_row_does_not_forward_bad_value(calls: dict[str, list]) -> None:
    Path("prompts.jsonl").write_text(
        '{"prompt":"valid"}\n\n{"prompt":17}\n', encoding="utf-8",
    )
    with pytest.raises(ValueError) as caught:
        gate._run_judge_task(task(), lambda prompt: calls["generate"].append(prompt) or "synthetic")
    assert str(caught.value) == expected_message("prompt", "int")
    assert calls == {"generate": ["valid"], "judge": []}


def test_cli_invalid_input_does_not_replace_baseline(calls: dict[str, list]) -> None:
    write_input({"prompt": None})
    write_suite()
    original = b'{"scores":{"older":0.75}}\n'
    Path("baseline.json").write_bytes(original)
    result = CliRunner().invoke(app, [
        "eval", "gate", "--suite", "gate.yaml", "--model", "synthetic",
        "--write-baseline", "baseline.json",
    ])
    output = " ".join(strip_ansi(result.output).split())
    assert result.exit_code == 1 and isinstance(result.exception, SystemExit)
    assert expected_message("prompt", "NoneType") in output
    assert "Wrote stamped baseline" not in output
    assert Path("baseline.json").read_bytes() == original
    assert calls == {"generate": [], "judge": []}
