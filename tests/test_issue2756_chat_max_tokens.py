"""#2756: ``soup chat --max-tokens`` refuses a count below 1 before loading.

``--max-tokens`` had no lower bound on the transformers backend, so ``0`` or a
negative value loaded the model and printed the prompt, and the first message
then ended the session with ``ValueError: max_new_tokens must be greater
than 0``.  ``soup infer --max-tokens`` already refuses the same values at
parse time, and the MLX backend refused them before loading; the transformers
path now does the same.

The bound is a parse-time one, so 0 and negative values are refused on every
backend, and the MLX backend keeps its own 1-16384 check for the upper end.
"""

from __future__ import annotations

import pytest

from soup_cli.cli import app
from tests.conftest import make_cli_runner, strip_ansi

runner = make_cli_runner()


@pytest.fixture
def model_dir(tmp_path) -> str:
    path = tmp_path / "model"
    path.mkdir()
    return str(path)


@pytest.fixture
def never_loaded(monkeypatch) -> list[str]:
    """Fail loudly if the command reaches the model or the generator."""
    calls: list[str] = []

    def load(*args, **kwargs):
        calls.append("load")
        raise AssertionError("the model must not be loaded")

    def generate(*args, **kwargs):
        calls.append("generate")
        raise AssertionError("no generation may happen")

    monkeypatch.setattr("soup_cli.commands.chat._load_model", load)
    monkeypatch.setattr("soup_cli.commands.chat._generate", generate)
    return calls


@pytest.mark.parametrize("value", ["0", "-5"])
def test_a_count_below_one_is_refused_before_the_model_loads(
    model_dir, never_loaded, value
):
    result = runner.invoke(app, ["chat", "--model", model_dir, "--max-tokens", value])

    assert result.exit_code == 2, result.output
    plain = " ".join(strip_ansi(result.output).split())
    assert "--max-tokens" in plain
    assert never_loaded == []


@pytest.mark.parametrize("value", ["1", "16384", "16385"])
def test_the_counts_that_work_today_still_reach_generation(model_dir, monkeypatch, value):
    seen: list[int] = []

    def load(*args, **kwargs):
        return object(), object()

    def generate(*args, **kwargs):
        seen.append(kwargs.get("max_tokens"))
        return "answer"

    monkeypatch.setattr("soup_cli.commands.chat._load_model", load)
    monkeypatch.setattr("soup_cli.commands.chat._generate", generate)

    result = runner.invoke(
        app, ["chat", "--model", model_dir, "--max-tokens", value], input="hello\n/quit\n"
    )

    assert result.exit_code == 0, result.output
    assert seen == [int(value)]
