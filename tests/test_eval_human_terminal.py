"""Human eval renders literal terminal text and preserves the saved vote and ratings."""

import json
import os
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import eval as eval_command
from soup_cli.eval import custom as custom_eval
from tests.conftest import strip_ansi

SPELLINGS = [
    pytest.param("plain", "plain", id="plain"),
    pytest.param(":rocket:", ":rocket:", id="emoji"),
    pytest.param("[red]name", "[red]name", id="markup"),
    pytest.param(r"name\[red]tag", r"name\[red]tag", id="backslash"),
    pytest.param("name\x1b\x07\x9b\u202e[end]", "name[end]", id="control-and-bidi"),
]


@pytest.fixture(autouse=True)
def terminal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        eval_command,
        "console",
        Console(width=260, height=40, force_terminal=True, color_system="standard", no_color=False),
    )


def _run(prompts, model_a, model_b, output="results.json"):
    return CliRunner().invoke(
        app,
        ["eval", "human", "--input", prompts, "--model-a", model_a, "--model-b", model_b,
         "--output", output],
        input="a\n",
    )


def _lines(result, exit_code=0):
    assert result.exit_code == exit_code, (result.output, repr(result.exception))
    plain = strip_ansi(result.stdout)
    assert not any(char in plain for char in ("\x1b", "\x07", "\x9b", "\u202e")), repr(plain)
    cells = str.maketrans("│┃|", "   ")
    return [" ".join(line.translate(cells).split()) for line in plain.splitlines() if line.strip()]


def _require_filename(value):
    if os.name == "nt" and any(char in value for char in (":", "\\", "\x1b")):
        pytest.skip("POSIX filename spelling")


def _write_prompt(prompt):
    Path("prompts.jsonl").write_text(json.dumps({"prompt": prompt}) + "\n", encoding="utf-8")


def _assert_red_refusal(result, message):
    assert not Path("results.json").exists()
    assert _lines(result, 1) == [message]
    assert result.stdout.startswith("\x1b[31m")


@pytest.mark.parametrize("value,shown", SPELLINGS)
def test_human_missing_prompt_file_displays_literal_path(value, shown):
    _require_filename(value)
    result = _run(f"missing-{value}.jsonl", "unused-a", "unused-b")
    _assert_red_refusal(result, f"Prompts file not found: missing-{shown}.jsonl")


@pytest.mark.parametrize("value,shown", SPELLINGS)
def test_human_missing_model_displays_literal_path(value, shown):
    _require_filename(value)
    _write_prompt("Which answer is better?")
    Path("present-model").mkdir()
    result = _run("prompts.jsonl", "present-model", f"missing-{value}")
    _assert_red_refusal(result, f"Model not found: missing-{shown}")


@pytest.mark.parametrize("value,shown", SPELLINGS)
def test_human_displays_literal_judgment_and_preserves_report(monkeypatch, value, shown):
    _require_filename(value)
    prompt = f"Question {value}"
    model_a, model_b = f"fixtures/models/model-a-{value}", f"fixtures/models/model-b-{value}"
    response_a, response_b = f"Answer A {value}", f"Answer B {value}"
    output = f"results-{value}.json"
    _write_prompt(prompt)
    for path in (model_a, model_b):
        Path(path).mkdir(parents=True)
    responses = {model_a: response_a, model_b: response_b}

    def generator(model):
        def generate(received):
            assert received == prompt
            return responses[model]

        return generate

    monkeypatch.setattr(custom_eval, "_create_default_generator", generator)
    result = _run("prompts.jsonl", model_a, model_b, output)
    assert result.exit_code == 0, (result.output, repr(result.exception))

    report = {
        "judgments": [{
            "prompt": prompt,
            "response_a": response_a,
            "response_b": response_b,
            "model_a": model_a,
            "model_b": model_b,
            "winner": "a",
        }],
        "ratings": {
            model_a: {"rating": 1516.0, "wins": 1, "losses": 0, "ties": 0, "total": 1},
            model_b: {"rating": 1484.0, "wins": 0, "losses": 1, "ties": 0, "total": 1},
        },
    }
    expected = json.dumps(report, indent=2).replace("\n", os.linesep).encode("utf-8")
    assert Path(output).read_bytes() == expected

    lines = _lines(result)
    for line in (
        "Human Evaluation",
        "Prompts: 1",
        f"Model A: fixtures/models/model-a-{shown}",
        f"Model B: fixtures/models/model-b-{shown}",
        "--- Prompt 1/1 ---",
        f"Question {shown}",
        f"Answer A {shown}",
        f"Answer B {shown}",
        f"models/model-a-{shown} 1516 1 0 0",
        f"models/model-b-{shown} 1484 0 1 0",
        f"Results saved to results-{shown}.json",
    ):
        assert line in lines

    heading = next(
        line for line in result.stdout.splitlines() if strip_ansi(line) == "--- Prompt 1/1 ---"
    )
    assert heading.startswith("\x1b[1m")
