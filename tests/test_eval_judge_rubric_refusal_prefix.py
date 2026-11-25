"""``soup eval judge --rubric`` names the rubric in front of a shape error (#3088, B-4).

The refusals of ``JudgeEvaluator`` (a weight that is not finite) were printed as
``Cannot use rubric '<path>': <message>``, while the ones of ``load_rubric`` (a duplicate
criterion, a rubric that is not a mapping) were printed bare. A shape or criterion error of
``load_rubric`` now gets the same prefix. An error of reading the file already starts with
``Rubric file <path>``, and the not-found one quotes the path, so those keep their text.
Every refusal exits 1 before any request is made.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.eval.judge import JudgeEvaluator
from soup_cli.utils.yaml_limits import MAX_YAML_NESTING
from tests.conftest import strip_ansi

runner = CliRunner()

TWO_CRITERIA = (
    "criteria:\n"
    "  - name: quality\n"
    "    description: response quality\n"
    "  - name: clarity\n"
    "    description: response clarity\n"
)


def _flat(text: str) -> str:
    # Rich may fold a long path anywhere, so lines are compared with all whitespace removed.
    return "".join(strip_ansi(text).split())


@pytest.fixture
def no_request(monkeypatch) -> MagicMock:
    reply = MagicMock(return_value='{"scores":{"quality":5,"clarity":1}}')
    monkeypatch.setattr(JudgeEvaluator, "_call_llm", reply)
    return reply


def _judge(tmp_path, rubric: str):
    target = tmp_path / "responses.jsonl"
    target.write_text('{"prompt":"question","response":"answer"}\n', encoding="utf-8")
    return runner.invoke(app, ["eval", "judge", "--target", str(target), "--rubric", rubric])


def _assert_refused(result, line: str, reply: MagicMock) -> None:
    assert result.exit_code == 1, result.output
    assert _flat(line) in _flat(result.output), strip_ansi(result.output)
    reply.assert_not_called()


@pytest.mark.parametrize(
    ("text", "error"),
    [
        (
            TWO_CRITERIA + "  - name: Quality\n    description: again\n",
            "Criterion 2 ('Quality') duplicates criterion 0",
        ),
        ("- quality\n- clarity\n", "Rubric must be a YAML mapping"),
        ("criteria: []\n", "Rubric 'criteria' must be a non-empty list"),
        ("criteria:\n  - name: quality\n", "Criterion 0 must have 'name' and 'description'"),
    ],
)
def test_a_shape_error_of_the_rubric_is_prefixed(tmp_path, no_request, text, error):
    rubric = tmp_path / "rubric.yaml"
    rubric.write_text(text, encoding="utf-8")

    result = _judge(tmp_path, str(rubric))

    _assert_refused(result, f"Cannot use rubric {str(rubric)!r}: {error}", no_request)


def test_a_non_finite_weight_keeps_the_same_prefix(tmp_path, no_request):
    rubric = tmp_path / "rubric.yaml"
    rubric.write_text(TWO_CRITERIA + "    weight: .nan\n", encoding="utf-8")

    result = _judge(tmp_path, str(rubric))

    _assert_refused(
        result, f"Cannot use rubric {str(rubric)!r}: Criterion 1 ('clarity') weight", no_request
    )


def test_an_error_of_reading_the_file_keeps_its_text(tmp_path, no_request):
    rubric = tmp_path / "rubric.yaml"
    depth = MAX_YAML_NESTING + 5
    rubric.write_text("criteria: " + "[" * depth + "]" * depth + "\n", encoding="utf-8")

    result = _judge(tmp_path, str(rubric))

    _assert_refused(
        result,
        f"Rubric file {rubric} nests collections more than {MAX_YAML_NESTING} levels deep",
        no_request,
    )
    assert "Cannot use rubric" not in strip_ansi(result.output)


def test_a_missing_rubric_keeps_its_text(tmp_path, no_request):
    rubric = tmp_path / "missing.yaml"

    result = _judge(tmp_path, str(rubric))

    _assert_refused(result, f"Rubric file not found: {rubric}", no_request)
    assert "Cannot use rubric" not in strip_ansi(result.output)
