"""Run the rendered validator command against actual dataset quality boundaries."""

import json
import shlex

import pytest
import yaml
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.ci_workflow import render_soup_gate_workflow
from tests.conftest import strip_ansi


def _validation_args(body):
    steps = yaml.safe_load(body)["jobs"]["soup-gate"]["steps"]
    step, = [item for item in steps if item.get("name") == "Validate training data"]
    command, = [line.strip() for line in step["run"].splitlines()
                if line.strip().startswith("soup data validate ")]
    args = shlex.split(command)
    assert args[:3] == ["soup", "data", "validate"]
    return args[1:]


@pytest.mark.parametrize("name", ["train.jsonl", "train's set.jsonl"])
@pytest.mark.parametrize("valid,total,expected", [(1, 10, 2), (9, 10, 2), (10, 10, 0)])
def test_rendered_gate_refuses_any_partial_drop(
    tmp_path, monkeypatch, name, valid, total, expected,
):
    monkeypatch.chdir(tmp_path)
    rows = [{"instruction": "hi", "output": "hello"}] * valid
    rows += [{"instruction": "missing output"}] * (total - valid)
    (tmp_path / name).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    body = render_soup_gate_workflow(
        data_path=name, suite_path="expectations.yaml", evidence_path="evidence.json",
    )
    runner = CliRunner()
    result = runner.invoke(app, _validation_args(body))
    assert result.exit_code == expected, (result.output, repr(result.exception))
    # The strict policy belongs to the generated gate, not standalone validation.
    standalone = runner.invoke(app, ["data", "validate", name])
    assert standalone.exit_code == 0, (standalone.output, repr(standalone.exception))


@pytest.mark.parametrize(("content", "code", "message"), [
    ("", 3, "Empty dataset - cannot detect format"),
    ("\n", 3, "Empty dataset - cannot detect format"),
    (
        '{"instruction":"hi","output":"ok"}\n42\n',
        2,
        "Validation failed: valid fraction 0.500 is below --min-valid-fraction 1.000.",
    ),
], ids=["empty", "blank-line", "skipped-row"])
def test_rendered_gate_refuses_empty_or_skipped_rows(
    tmp_path, monkeypatch, content, code, message,
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.jsonl").write_text(content, encoding="utf-8")
    body = render_soup_gate_workflow(
        data_path="train.jsonl", suite_path="expectations.yaml", evidence_path="evidence.json",
    )
    result = CliRunner().invoke(app, _validation_args(body))
    assert result.exit_code == code, (result.output, repr(result.exception))
    assert message in " ".join(strip_ansi(result.output).split()), result.output


def test_ci_init_writes_strict_command_and_preserves_existing_workflow(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["ci", "init"])
    assert result.exit_code == 0, result.output
    workflow = tmp_path / ".github/workflows/soup-gate.yml"
    args = _validation_args(workflow.read_text(encoding="utf-8"))
    assert args[-2:] == ["--min-valid-fraction", "1.0"]
    workflow.write_text("operator-owned workflow\n", encoding="utf-8")
    result = runner.invoke(app, ["ci", "init"])
    assert result.exit_code == 1, result.output
    assert workflow.read_text(encoding="utf-8") == "operator-owned workflow\n"
