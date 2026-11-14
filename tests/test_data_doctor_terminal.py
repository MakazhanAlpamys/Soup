"""Doctor and lint preserve literal diagnostics and report values."""

import json
import sys
from pathlib import Path

import pytest
from rich.console import Console

from soup_cli.cli import app
from soup_cli.commands import data_doctor as command
from soup_cli.utils import data_doctor as doctor
from soup_cli.utils import data_lint as lint
from tests.conftest import make_cli_runner, strip_ansi

RAW = r"label:warning:[red]\[end" + "\x9b\u202e"
SHOWN = r"label:warning:[red]\[end"


@pytest.fixture(autouse=True)
def cwd_and_console(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        command,
        "console",
        Console(
            width=500,
            height=100,
            force_terminal=True,
            no_color=False,
            color_system="standard",
            legacy_windows=False,
        ),
    )


def run(name, path, *args):
    options = ["--model", "owner/model"] if name == "doctor" else []
    return make_cli_runner().invoke(app, ["data", name, str(path), *options, *args])


@pytest.fixture(params=["doctor", "lint"])
def report_case(request, monkeypatch):
    name = request.param
    path = Path("rows.jsonl")
    if name == "doctor":
        row = {"instruction": "Say hi", "output": "Hello"}
        check = doctor.DoctorCheck("chat_template", "OK", "message " + RAW, "evidence " + RAW)
        report = doctor.DoctorReport((check,), "OK", 1, 1)
        monkeypatch.setattr(doctor, "resolve_tokenizer", lambda *args, **kwargs: object())
        monkeypatch.setattr(doctor, "run_doctor", lambda *args, **kwargs: report)
    else:
        row = {"prompt": "Say hi", "chosen": "Hello", "rejected": "Goodbye"}
        check = lint.LintCheck("length_bias", "OK", "message " + RAW, "evidence " + RAW)
        report = lint.LintReport((check,), "OK", "dpo", 1, 1)
        monkeypatch.setattr(lint, "run_lint", lambda *args, **kwargs: report)
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return name, path, report


@pytest.mark.parametrize("name,code", [("doctor", 1), ("lint", 3)])
def test_missing_path_diagnostic(name, code):
    path = Path("missing[/x]\x9b\u202e.jsonl")
    result = run(name, path)
    assert result.exit_code == code, (result.output, result.exception)
    expected = str(path).replace("\x9b\u202e", "")
    assert f"\x1b[31mFile not found: {expected}\x1b[0m" in result.output


@pytest.mark.parametrize("column", ["Message", "Evidence"])
def test_report_cells_remain_literal(report_case, column):
    name, path, _ = report_case
    result = run(name, path)
    assert result.exit_code == 0, (result.output, result.exception)
    assert f"{column.lower()} {SHOWN}" in strip_ansi(result.output)
    assert "\x1b[32mOK\x1b[0m" in result.output
    assert "\x9b" not in result.output and "\u202e" not in result.output


@pytest.mark.parametrize("spelling", [r"result-\[draft].json", "result-:warning:.json"])
def test_report_receipt_keeps_name_and_json_values(report_case, spelling):
    if sys.platform == "win32" and ":" in spelling:
        pytest.skip("colon is not a Windows filename character")
    name, path, report = report_case
    output = Path(spelling)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = run(name, path, "-o", str(output))
    assert result.exit_code == 0, (result.output, result.exception)
    assert f"\x1b[32mWrote\x1b[0m {output}\n" in result.output
    assert json.loads(output.read_text(encoding="utf-8")) == report.to_dict()


@pytest.mark.parametrize("stage", ["detect", "tokenizer", "report", "mask", "write"])
def test_doctor_failures_keep_literal_message(report_case, monkeypatch, stage):
    name, path, _ = report_case
    if name == "lint" and stage in {"tokenizer", "mask"}:
        pytest.skip("doctor-only stage")
    module, function = {
        "detect": (command, "detect_format"),
        "tokenizer": (doctor, "resolve_tokenizer"),
        "report": (doctor, "run_doctor") if name == "doctor" else (lint, "run_lint"),
        "mask": (doctor, "render_mask_preview"),
        "write": (command, "_write_report_json"),
    }[stage]

    def fail(*args, **kwargs):
        raise ValueError(RAW)

    monkeypatch.setattr(module, function, fail)
    args = {"mask": ["--show-mask", "1"], "write": ["-o", "out.json"]}.get(stage, [])
    result = run(name, path, *args)
    code = 3 if name == "lint" and stage in {"report", "write"} else 1
    assert result.exit_code == code, (result.output, result.exception)
    label = "cannot write --output: " if stage == "write" else ""
    assert f"\x1b[31mError:\x1b[0m {label}{SHOWN}" in result.output
    assert not Path("out.json").exists()


def test_lint_refuses_unknown_format_literally(report_case):
    name, path, _ = report_case
    if name == "doctor":
        pytest.skip("lint-only format check")
    value = r"format:warning:\[test"
    result = run("lint", path, "--format", value)
    assert result.exit_code == 3
    assert f"format={value!r} is not preference data" in strip_ansi(result.output)
    assert "\x1b[1msoup data doctor\x1b[0m" in result.output
