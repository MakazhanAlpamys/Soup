"""#1342: custom eval scores the answer a model states, not its raw string.

``scoring: "answer"`` reads ``expected`` and the output with ``soup_cli.utils.final_answer``
(the parser GRPO's ``accuracy`` reward uses). ``contains`` only matches ``expected`` as a whole
alnum-bounded token, so ``14`` no longer pays for ``4``. ``exact`` still compares the output.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from soup_cli.eval.custom import EvalTask, load_eval_tasks, score_contains, score_task
from soup_cli.eval.gate import EvalSuite, GateTask, run_gate

# (expected, output, truth) -- the table in #1342.
_TABLE = [
    ("4", "4.", True),
    ("4", "2 + 2 = 4", True),
    ("4", "The answer is 4.", True),
    ("4", "#### 4", True),
    ("4", "14", False),
    ("42", "#### 420", False),
    ("42", "It is not 42, so the answer is 41.", False),
    ("6*7=42\n#### 42", "#### 42", True),
]


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


@pytest.mark.parametrize(("expected", "output", "truth"), _TABLE)
def test_answer_mode_gets_every_row_right(expected, output, truth):
    result = score_task(EvalTask(prompt="q", expected=expected, scoring="answer"), output)
    assert result.matched is truth
    assert result.score == (1.0 if truth else 0.0)


@pytest.mark.parametrize(("expected", "output"), [("4", "14"), ("42", "#### 420")])
def test_contains_no_longer_pays_digits_inside_another_number(expected, output):
    assert score_contains(output, expected) is False


@pytest.mark.parametrize(
    ("expected", "output"),
    [("4", "2 + 2 = 4"), ("4", "#### 4"), ("Paris", "It is paris."), ("-3", "x = -3")],
)
def test_contains_still_pays_a_standalone_token(expected, output):
    assert score_contains(output, expected) is True


def test_contains_negation_is_a_documented_limit():
    # A token match cannot tell "not 42" from "42"; use scoring "answer" for that.
    assert score_contains("It is not 42, so the answer is 41.", "42") is True


def test_contains_with_an_empty_expected_keeps_matching():
    assert score_contains("anything", "") is True


def test_answer_mode_loads_a_readable_expected(tmp_path):
    path = _write(tmp_path / "t.jsonl", [{"prompt": "2+2?", "expected": "4", "scoring": "answer"}])
    assert load_eval_tasks(str(path))[0].scoring == "answer"


@pytest.mark.parametrize(
    "expected",
    ["", "Step 1.\nStep 2.", "6*7=42\n####", r"\boxed{}", "The answer is either 41 or 42."],
)
def test_answer_mode_refuses_an_expected_with_no_single_answer(tmp_path, expected):
    rows = [
        {"prompt": "ok", "expected": "4", "scoring": "answer"},
        {"prompt": "bad", "expected": expected, "scoring": "answer"},
    ]
    with pytest.raises(ValueError, match=r"^Line 2: .*scoring 'answer'"):
        load_eval_tasks(str(_write(tmp_path / "t.jsonl", rows)))


def test_gate_answer_scorer_overrides_the_rows(tmp_path):
    path = _write(tmp_path / "t.jsonl", [{"prompt": "2+2?", "expected": "4"}])
    suite = EvalSuite(
        suite="s",
        tasks=[GateTask(type="custom", name="m", threshold=1.0, tasks=str(path), scorer="answer")],
    )
    result = run_gate(suite, generate_fn=lambda _prompt: "The answer is 4.")
    assert result.task_results[0].score == 1.0


def test_gate_answer_scorer_refuses_an_unreadable_expected(tmp_path):
    path = _write(tmp_path / "t.jsonl", [{"prompt": "q", "expected": "Step 1.\nStep 2."}])
    suite = EvalSuite(
        suite="s",
        tasks=[GateTask(type="custom", name="m", threshold=1.0, tasks=str(path), scorer="answer")],
    )
    (task,) = run_gate(suite, generate_fn=lambda _prompt: "4").task_results
    assert task.score is None and not task.passed
    assert "row 1" in task.error and "scoring 'answer'" in task.error


@pytest.mark.parametrize(
    ("expected", "output", "truth"),
    [
        ("C++", "I like C++ a lot", True),
        ("a.b", "axb", False),
        ("a.b", "see a.b here", True),
        ("4|5", "5", False),
        ("[0]", "x [0] y", True),
        ("(", "a ( b", True),
    ],
)
def test_contains_reads_expected_literally(expected, output, truth):
    """`expected` is data, never a pattern: regex metacharacters match themselves."""
    assert score_contains(output, expected) is truth
