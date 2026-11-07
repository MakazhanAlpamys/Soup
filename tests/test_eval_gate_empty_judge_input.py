"""A judge task with no prompts has no measured score or baseline (#1910)."""

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


def _task(path: Path, threshold: float = 0.0, kind: str = "judge") -> gate.GateTask:
    return gate.GateTask(
        name="input-control", type=kind, threshold=threshold,
        prompts=str(path) if kind == "judge" else None,
        tasks=str(path) if kind == "custom" else None,
        judge_model="ollama://synthetic" if kind == "judge" else None,
    )


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    recorded: dict[str, list[str]] = {"generate": [], "judge": []}

    def judge(self: JudgeEvaluator, prompt: str) -> str:
        recorded["judge"].append(prompt)
        return json.dumps({
            "scores": {"helpfulness": 5, "accuracy": 5, "safety": 5},
            "reasoning": "valid measured control",
        })

    monkeypatch.setattr(JudgeEvaluator, "_call_llm", judge)
    monkeypatch.setattr(
        quant_check, "make_model_generator",
        lambda model: lambda prompt: recorded["generate"].append(prompt) or "gold",
    )
    monkeypatch.setattr(eval_command, "console", Console(width=200))
    return recorded


@pytest.mark.parametrize("body", ["", " \n\t\n"], ids=["zero-byte", "whitespace"])
@pytest.mark.parametrize("threshold", [0.0, 0.5])
@pytest.mark.parametrize("kind", ["judge", "custom"])
def test_empty_input_has_no_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: dict[str, list[str]],
    body: str, threshold: float, kind: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "prompts.jsonl"
    path.write_text(body, encoding="utf-8")
    task = _task(path, threshold, kind)
    result = gate.run_gate(
        gate.EvalSuite(suite="input-control", tasks=[task]),
        generate_fn=lambda prompt: calls["generate"].append(prompt) or "gold",
    )
    assert result.passed is False
    row = result.task_results[0]
    assert row.score is None
    assert row.passed is False
    assert row.delta is None
    assert row.error is not None and str(path) in row.error
    if kind == "judge":
        assert "input-control" in row.error
        assert "No judge prompts found" in row.error
    else:
        assert "No eval tasks found" in row.error
    assert calls == {"generate": [], "judge": []}


def _suite(tmp_path: Path, task: gate.GateTask) -> Path:
    path = tmp_path / "gate.yaml"
    path.write_text(yaml.safe_dump({
        "suite": "input-control", "tasks": [task.model_dump(exclude_none=True)],
    }), encoding="utf-8")
    return path


@pytest.mark.parametrize("body", ["", " \n\t\n"], ids=["zero-byte", "whitespace"])
@pytest.mark.parametrize("threshold", [0.0, 0.5])
@pytest.mark.parametrize("write", [False, True], ids=["gate-only", "write-baseline"])
@pytest.mark.parametrize("existing", [False, True], ids=["new-baseline", "existing-baseline"])
def test_cli_empty_judge_cannot_pass_or_mint_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: dict[str, list[str]],
    body: str, threshold: float, write: bool, existing: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(body, encoding="utf-8")
    suite = _suite(tmp_path, _task(prompts, threshold))
    baseline = tmp_path / "baseline.json"
    original = b'{"scores": {"older": 0.75}}\n'
    if existing:
        baseline.write_bytes(original)
    args = ["eval", "gate", "--suite", str(suite), "--model", "synthetic"]
    if write:
        args.extend(["--write-baseline", str(baseline)])
    result = CliRunner().invoke(app, args)
    output = " ".join(strip_ansi(result.output).split())
    assert result.exit_code == (1 if write else 2), output
    assert "GATE FAILED" in output
    assert "GATE PASSED" not in output
    assert "No judge prompts found" in output
    assert "prompts.jsonl" in output
    assert "Wrote stamped baseline" not in output
    assert calls == {"generate": [], "judge": []}
    if existing:
        assert baseline.read_bytes() == original
    else:
        assert not baseline.exists()


@pytest.mark.parametrize("kind", ["judge", "custom"])
@pytest.mark.parametrize("passes", [False, True], ids=["measured-failure", "measured-pass"])
def test_nonempty_cli_measures_and_writes_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: dict[str, list[str]], kind: str,
    passes: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        json.dumps({"prompt": "question", "expected": "gold"}) + "\n", encoding="utf-8",
    )
    suite = _suite(tmp_path, _task(prompts, 0.5, kind))
    baseline = tmp_path / "baseline.json"
    if not passes:
        monkeypatch.setattr(
            quant_check, "make_model_generator",
            lambda model: lambda prompt: calls["generate"].append(prompt) or "wrong",
        )

        def low_judge(self: JudgeEvaluator, prompt: str) -> str:
            calls["judge"].append(prompt)
            return json.dumps({"scores": {"helpfulness": 1, "accuracy": 1, "safety": 1}})

        monkeypatch.setattr(JudgeEvaluator, "_call_llm", low_judge)
    result = CliRunner().invoke(app, [
        "eval", "gate", "--suite", str(suite), "--model", "synthetic",
        "--write-baseline", str(baseline),
    ])
    assert result.exit_code == (0 if passes else 2), result.output
    assert ("GATE PASSED" if passes else "GATE FAILED") in strip_ansi(result.output)
    payload = json.loads(baseline.read_text(encoding="utf-8"))
    assert payload["scores"] == {"input-control": 1.0 if passes else 0.0}
    assert payload["provenance"]
    assert calls["generate"] == ["question"]
    assert len(calls["judge"]) == (1 if kind == "judge" else 0)


@pytest.mark.parametrize("body", ["{broken\n", '{"prompt": "question"}\n'])
def test_existing_invalid_json_and_valid_judge_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: dict[str, list[str]], body: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "prompts.jsonl"
    path.write_text(body, encoding="utf-8")
    result = gate.run_gate(
        gate.EvalSuite(suite="control", tasks=[_task(path)]),
        generate_fn=lambda prompt: calls["generate"].append(prompt) or "gold",
        baseline={"input-control": 1.0},
    )
    row = result.task_results[0]
    if body.startswith("{broken"):
        assert row.score is None and "invalid JSONL" in row.error
        assert calls == {"generate": [], "judge": []}
    else:
        assert result.passed is True and row.score == 1.0 and row.error is None
        assert calls["generate"] == ["question"] and len(calls["judge"]) == 1
