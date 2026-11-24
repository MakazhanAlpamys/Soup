"""Standalone eval gate reports unreadable suite and baseline inputs before model creation."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.eval import quant_check
from tests.conftest import strip_ansi

runner = CliRunner()


@pytest.fixture
def gate_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Mock:
    monkeypatch.chdir(tmp_path)
    Path("gate.yaml").write_text(
        "suite: synthetic\ntasks:\n"
        "  - {type: custom, name: sample, threshold: 0, tasks: tasks.jsonl, scorer: exact}\n",
        encoding="utf-8",
    )
    Path("tasks.jsonl").write_text(
        '{"prompt":"Question","expected":"correct"}\n',
        encoding="utf-8",
    )
    factory = Mock(return_value=lambda prompt: "correct")
    monkeypatch.setattr(quant_check, "make_model_generator", factory)
    return factory


@pytest.mark.parametrize(
    ("flag", "path", "diagnostic", "missing_message"),
    [
        ("--suite", "input.yaml", "Cannot load suite", "eval-gate suite not found"),
        ("--baseline", "input.json", "Cannot resolve baseline", "baseline file not found"),
    ],
    ids=["suite", "baseline"],
)
@pytest.mark.parametrize("is_directory", [True, False], ids=["directory", "missing"])
def test_gate_names_bad_file_input_before_loading_model(
    gate_inputs: Mock, flag: str, path: str, diagnostic: str, missing_message: str,
    is_directory: bool,
) -> None:
    if is_directory:
        directory = Path(path)
        directory.mkdir()
        # Opening a directory has different OS error text on Windows and POSIX.
        with pytest.raises(OSError) as error:
            directory.read_text(encoding="utf-8")
        detail = str(error.value)
    else:
        detail = f"{missing_message}: {path}"

    args = ["eval", "gate", "--model", "synthetic", flag, path]
    if flag == "--baseline":
        args.extend(["--suite", "gate.yaml"])
    result = runner.invoke(app, args)
    output = " ".join(strip_ansi(result.output).split())
    expected = " ".join(f"{diagnostic}: {detail}".split())

    assert result.exit_code == 3, (output, result.exception)
    assert output == expected
    gate_inputs.assert_not_called()


def test_gate_scores_with_a_readable_baseline(gate_inputs: Mock) -> None:
    Path("baseline.json").write_text('{"sample": 0.0}\n', encoding="utf-8")
    result = runner.invoke(
        app, ["eval", "gate", "--suite", "gate.yaml", "--baseline", "baseline.json",
              "--model", "synthetic"],
    )
    output = " ".join(strip_ansi(result.output).split())

    assert result.exit_code == 0, (output, result.exception)
    assert "GATE PASSED" in output
    gate_inputs.assert_called_once_with("synthetic")
