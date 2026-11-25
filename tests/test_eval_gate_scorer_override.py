"""#3327: an eval gate custom task validates its rows against the scorer it grades with.

The gate loaded the rows with their own ``scoring`` checks and applied its ``scorer``
afterwards, so a row marked ``answer`` whose multiline reference the gate meant to grade
``exact`` was refused before any generation.
"""

import json
from pathlib import Path

import pytest

from soup_cli.eval.custom import load_eval_tasks
from soup_cli.eval.gate import EvalSuite, GateTask, run_gate

EXPECTED = "First line.\nSecond line."


def _gate(tmp_path: Path, row: dict, scorer: str):
    path = tmp_path / "tasks.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    suite = EvalSuite(
        suite="synthetic",
        tasks=[
            GateTask(type="custom", name="sample", tasks=str(path), scorer=scorer, threshold=1.0)
        ],
    )
    calls: list[str] = []

    def generate(prompt: str) -> str:
        calls.append(prompt)
        return EXPECTED

    (result,) = run_gate(suite, generate_fn=generate).task_results
    return result, calls


@pytest.mark.parametrize("row_scoring", ["answer", "exact"])
def test_an_exact_override_grades_a_multiline_reference(tmp_path, monkeypatch, row_scoring):
    monkeypatch.chdir(tmp_path)
    result, calls = _gate(
        tmp_path, {"prompt": "q", "expected": EXPECTED, "scoring": row_scoring}, "exact"
    )
    assert result.error is None
    assert result.score == 1.0
    assert calls == ["q"]


def test_an_answer_override_still_refuses_a_multiline_reference(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result, calls = _gate(
        tmp_path, {"prompt": "q", "expected": EXPECTED, "scoring": "exact"}, "answer"
    )
    assert result.score is None
    assert "scoring 'answer'" in result.error
    assert calls == []


def test_an_unknown_row_scorer_is_still_refused_under_an_override(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result, calls = _gate(
        tmp_path, {"prompt": "q", "expected": EXPECTED, "scoring": "fuzzy"}, "exact"
    )
    assert result.score is None
    assert "invalid scoring 'fuzzy'" in result.error
    assert calls == []


def test_the_standalone_loader_still_refuses_a_multiline_answer_row(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text(json.dumps({"prompt": "q", "expected": EXPECTED, "scoring": "answer"}) + "\n")
    with pytest.raises(ValueError, match=r"^Line 1: .*scoring 'answer'"):
        load_eval_tasks(path)


# The loader's side of the contract: ``graded_by`` is a hint, not an override.
# Tasks keep the row's own scorer; the hint only lifts the row scorer's reference
# rule, and the caller (the gate) substitutes the scorer and runs its checks.
def _loader(tmp_path: Path, row: dict, graded_by: str):
    path = tmp_path / "tasks.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return load_eval_tasks(path, graded_by=graded_by)


def test_the_loader_keeps_the_row_scorer_and_lifts_its_rule_under_a_hint(tmp_path):
    (task,) = _loader(
        tmp_path, {"prompt": "q", "expected": EXPECTED, "scoring": "answer"}, "exact"
    )
    assert task.scoring == "answer"


def test_the_loader_leaves_the_hinted_scorer_rule_to_the_caller(tmp_path):
    (task,) = _loader(tmp_path, {"prompt": "q", "expected": EXPECTED, "scoring": "exact"}, "answer")
    assert task.scoring == "exact"
