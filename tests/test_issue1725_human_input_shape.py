"""Reject non-object human-eval rows before either model is loaded (#1725)."""

import json
from unittest.mock import MagicMock

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.commands import eval as eval_command
from soup_cli.eval import custom, human
from tests.conftest import strip_ansi

BAD_ROWS = ["prompt", ["prompt"], None, 17, True]
OBJECT_ROW = {"prompt": "Synthetic question?", "category": "test", "metadata": {"keep": 1}}


@pytest.mark.parametrize("row", [*BAD_ROWS, OBJECT_ROW])
def test_loader_requires_json_object(tmp_path, row):
    path = tmp_path / "prompts.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    if isinstance(row, dict):
        assert human.load_prompts(path) == [row]
    else:
        with pytest.raises(ValueError) as exc:
            human.load_prompts(path)
        assert str(exc.value) == f"Line 1: expected JSON object, got {type(row).__name__}"


@pytest.mark.parametrize("prefix", ["\n", '{"prompt": "valid"}\n'])
def test_loader_error_uses_physical_line_number(tmp_path, prefix):
    path = tmp_path / "prompts.jsonl"
    path.write_text(prefix + '["prompt"]\n', encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        human.load_prompts(path)
    assert str(exc.value) == "Line 2: expected JSON object, got list"


@pytest.mark.parametrize("row", [*BAD_ROWS, OBJECT_ROW])
def test_cli_rejects_bad_row_before_model_construction(tmp_path, monkeypatch, row):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "model_a").mkdir()
    (tmp_path / "model_b").mkdir()
    (tmp_path / "prompts.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    factory = MagicMock(return_value=lambda _prompt: "Synthetic response")
    monkeypatch.setattr(custom, "_create_default_generator", factory)
    monkeypatch.setattr(eval_command, "console", Console(width=200))
    monkeypatch.setattr(eval_command.console, "input", lambda *_args, **_kwargs: "a")
    result = CliRunner().invoke(eval_command.app, [
        "human", "--input", "prompts.jsonl", "--model-a", "model_a",
        "--model-b", "model_b", "--output", "results.json",
    ])
    output = strip_ansi(result.output)
    if isinstance(row, dict):
        assert result.exit_code == 0, output
        assert factory.call_count == 2
        judgments = json.loads((tmp_path / "results.json").read_text())["judgments"]
        assert len(judgments) == 1 and judgments[0]["prompt"] == row["prompt"]
    else:
        assert result.exit_code == 1, output
        factory.assert_not_called()
        assert f"Line 1: expected JSON object, got {type(row).__name__}" in output
        assert not (tmp_path / "results.json").exists()
