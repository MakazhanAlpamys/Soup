"""#2757: ``soup bench infer`` refuses counts below 1 before the model loads.

``--max-tokens`` and ``--num-prompts`` had no lower bound, so the model was
loaded and the warm-up then died: ``IndexError: list index out of range`` for
``--num-prompts 0``, and ``ValueError: max_new_tokens must be greater than 0``
for ``--max-tokens 0``.  ``soup infer --max-tokens`` and ``soup bench train
--steps`` already refuse the same values at parse time.

``--num-prompts`` is ignored when ``--prompts-file`` is set, and that call
works today, so it is refused only without a prompts file.
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

    monkeypatch.setattr("soup_cli.commands.infer._load_model", load)
    monkeypatch.setattr("soup_cli.commands.infer._generate", generate)
    return calls


@pytest.mark.parametrize(
    "options,option",
    [
        (["--num-prompts", "0"], "--num-prompts"),
        (["--num-prompts", "-2"], "--num-prompts"),
        (["--max-tokens", "0"], "--max-tokens"),
        (["--max-tokens", "-5"], "--max-tokens"),
    ],
)
def test_a_count_below_one_is_refused_before_the_model_loads(
    model_dir, never_loaded, options, option
):
    result = runner.invoke(app, ["bench", "infer", model_dir, *options])

    assert result.exit_code == 2, result.output
    plain = " ".join(strip_ansi(result.output).split())
    assert option in plain
    assert never_loaded == []


@pytest.mark.parametrize("value", ["1", "16384", "16385"])
def test_the_max_token_counts_that_work_today_still_run(model_dir, monkeypatch, value):
    seen: list[int] = []

    def load(*args, **kwargs):
        return object(), object()

    def generate(model_obj, tokenizer, messages, **kwargs):
        seen.append(kwargs.get("max_tokens"))
        return "answer", 5

    monkeypatch.setattr("soup_cli.commands.infer._load_model", load)
    monkeypatch.setattr("soup_cli.commands.infer._generate", generate)

    result = runner.invoke(
        app,
        ["bench", "infer", model_dir, "--num-prompts", "1", "--max-tokens", value],
    )

    assert result.exit_code == 0, result.output
    assert seen


@pytest.mark.parametrize("count", ["0", "-2"])
def test_a_count_below_one_still_runs_with_a_prompts_file(tmp_path, monkeypatch, count):
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("hello\nworld\n")
    # The prompts file has to stay under the working directory.
    monkeypatch.chdir(tmp_path)
    seen: list[str] = []

    def load(*args, **kwargs):
        return object(), object()

    def generate(model_obj, tokenizer, messages, **kwargs):
        seen.append(messages[0]["content"])
        return "answer", 5

    monkeypatch.setattr("soup_cli.commands.infer._load_model", load)
    monkeypatch.setattr("soup_cli.commands.infer._generate", generate)

    result = runner.invoke(
        app,
        [
            "bench",
            "infer",
            str(tmp_path),
            "--num-prompts",
            count,
            "--prompts-file",
            "prompts.txt",
        ],
    )

    assert result.exit_code == 0, result.output
    # Every prompt in the file ran, as it does today. The warm-up is discarded
    # from the timing and runs the first prompt, so it comes first here.
    assert seen[-2:] == ["hello", "world"]
