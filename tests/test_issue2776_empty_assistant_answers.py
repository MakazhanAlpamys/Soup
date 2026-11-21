"""The data commands surface empty answers without changing their gates."""

import json

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi
from tests.test_v07127 import _FakeTokenizer

QUESTION = {"role": "user", "content": "A question long enough to validate"}


def _chat(*answers):
    return {"messages": [QUESTION, *({"role": "assistant", **answer} for answer in answers)]}


def _write_rows(path, *rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def doctor_model(monkeypatch, tmp_path):
    from soup_cli.commands import data_doctor

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(data_doctor, "_load_tokenizer", lambda *_args: _FakeTokenizer())
    return "local/tokenizer"


@pytest.mark.parametrize(
    ("fmt", "row"),
    [
        pytest.param("chatml", _chat({"content": ""}), id="chatml-empty-string"),
        pytest.param("chatml", _chat({"content": " \r\n\t "}), id="chatml-whitespace"),
        pytest.param("chatml", _chat({"content": None}), id="chatml-null"),
        pytest.param("chatml", _chat({}), id="chatml-missing-content"),
        pytest.param("chatml", _chat({"content": []}), id="chatml-empty-parts"),
        pytest.param(
            "chatml",
            _chat({"content": [{"type": "text", "text": ""}, {"type": "text", "text": "\t"}]}),
            id="chatml-empty-text-parts",
        ),
        pytest.param(
            "chatml", _chat({"content": "\u200b\x07\u200c\u200d\ufeff"}), id="chatml-invisible"
        ),
        pytest.param(
            "sharegpt",
            {"conversations": [
                {"from": "human", "value": "A question"},
                {"from": "gpt", "value": ""},
            ]},
            id="sharegpt-empty-answer",
        ),
        pytest.param(
            "alpaca", {"instruction": "A question", "input": "", "output": ""}, id="alpaca-empty"
        ),
        pytest.param(
            "alpaca", {"instruction": "A question", "output": " \t\u200b "}, id="alpaca-whitespace"
        ),
    ],
)
def test_both_commands_flag_empty_answers(tmp_path, doctor_model, fmt, row):
    data = tmp_path / "train.jsonl"
    _write_rows(data, row)
    runner = CliRunner()

    result = runner.invoke(
        app, ["data", "validate", str(data), "--format", fmt, "--min-valid-fraction", "1"]
    )
    output = " ".join(strip_ansi(result.output).split())
    assert result.exit_code == 0, output
    assert "Empty assistant answers in 1 row(s) (1 turn(s)). First rows: 1." in output
    assert "1/1 rows valid" in output

    report = tmp_path / "doctor.json"
    result = runner.invoke(
        app, ["data", "doctor", str(data), "--format", fmt, "--model", doctor_model,
              "--output", str(report)],
    )
    assert result.exit_code == 0, result.output
    checks = {check["name"]: check for check in json.loads(report.read_text())["checks"]}
    assert checks["empty_answers"]["verdict"] == "MINOR"
    assert checks["empty_answers"]["message"] == (
        "1/1 sampled rows (100.0%) contain empty assistant answers (1 turn(s))."
    )


def test_validate_reports_source_lines_and_counts_each_affected_row_once(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text(
        "\n".join([
            json.dumps(_chat({"content": "A valid answer"})),
            "",
            "42",
            "not JSON",
            json.dumps(_chat({"content": ""}, {"content": None})),
            json.dumps(_chat({"content": " "})),
            json.dumps(_chat({"content": []})),
            json.dumps(_chat({})),
        ]) + "\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["data", "validate", str(data), "--format", "chatml"])
    output = " ".join(strip_ansi(result.output).split())
    assert result.exit_code == 0, output
    assert "Empty assistant answers in 4 row(s) (5 turn(s)). First rows: 5, 6, 7." in output
    assert "5/7 rows valid" in output


def test_validate_reports_json_array_positions_after_skipped_rows(tmp_path):
    data = tmp_path / "train.json"
    data.write_text(
        json.dumps([42, _chat({"content": "A valid answer"}), _chat({"content": ""})]),
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["data", "validate", str(data), "--format", "chatml"])
    output = " ".join(strip_ansi(result.output).split())
    assert result.exit_code == 0, output
    assert "Empty assistant answers in 1 row(s) (1 turn(s)). First rows: 3." in output
    assert "2/3 rows valid" in output


def test_doctor_counts_only_converted_sample_rows(tmp_path, doctor_model):
    data = tmp_path / "train.jsonl"
    good_row = _chat({"content": "A valid answer"})
    empty_row = _chat({"content": ""})
    _write_rows(data, good_row, empty_row, empty_row, good_row)
    report = tmp_path / "doctor.json"
    result = CliRunner().invoke(
        app, ["data", "doctor", str(data), "--format", "chatml", "--model", doctor_model,
              "--sample", "2", "--output", str(report)],
    )
    assert result.exit_code == 0, result.output
    checks = {check["name"]: check for check in json.loads(report.read_text())["checks"]}
    assert checks["empty_answers"]["verdict"] == "MINOR"
    assert checks["empty_answers"]["message"] == (
        "1/2 sampled rows (50.0%) contain empty assistant answers (1 turn(s))."
    )


@pytest.mark.parametrize(
    ("fmt", "row"),
    [
        ("chatml", _chat({
            "content": None,
            "tool_calls": [{"function": {"name": "lookup", "arguments": "{}"}}],
        })),
        ("chatml", _chat({"content": [{"type": "text", "text": "\u200bUseful answer"}]})),
        ("chatml", _chat({"content": [{"type": "image", "image": "answer.png"}]})),
        ("alpaca", {"instruction": "A question", "input": "", "output": "A valid answer"}),
    ],
)
def test_valid_content_does_not_trigger_the_warning(tmp_path, doctor_model, fmt, row):
    data = tmp_path / "train.jsonl"
    _write_rows(data, row)
    runner = CliRunner()
    result = runner.invoke(app, ["data", "validate", str(data), "--format", fmt])
    assert result.exit_code == 0, result.output
    assert "Empty assistant answers" not in result.output

    report = tmp_path / "doctor.json"
    result = runner.invoke(
        app, ["data", "doctor", str(data), "--format", fmt, "--model", doctor_model,
              "--output", str(report)],
    )
    assert result.exit_code == 0, result.output
    checks = {check["name"]: check for check in json.loads(report.read_text())["checks"]}
    assert checks["empty_answers"]["verdict"] == "OK"
