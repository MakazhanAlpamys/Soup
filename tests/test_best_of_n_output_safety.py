from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi


@pytest.mark.parametrize("plan_only", [True, False], ids=["plan", "live"])
@pytest.mark.parametrize(
    "output_option",
    ["--output", "--emit-pairs", "--checkpoint", "--manifest"],
)
@pytest.mark.parametrize(
    "alias",
    ["prompts.jsonl", "./prompts.jsonl"],
    ids=["same-path", "dot-alias"],
)
def test_best_of_n_rejects_output_aliasing_prompts_before_sampling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output_option: str,
    alias: str,
    plan_only: bool,
) -> None:
    from soup_cli.eval import judge
    from soup_cli.utils import magpie

    monkeypatch.chdir(tmp_path)
    source = Path("prompts.jsonl")
    original = b'{"prompt": "Explain gravity."}\n'
    source.write_bytes(original)

    sampler_calls = []

    def fake_sampler_factory(*args, **kwargs):
        sampler_calls.append((args, kwargs))
        return lambda *args, **kwargs: "unused"

    monkeypatch.setattr(
        judge,
        "JudgeEvaluator",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        magpie,
        "make_magpie_generate_fn",
        fake_sampler_factory,
    )

    options = {
        "--output": "selected.jsonl",
        "--emit-pairs": "pairs.jsonl",
        "--checkpoint": "run.checkpoint.jsonl",
        "--manifest": "run.manifest.json",
    }
    options[output_option] = alias

    arguments = [
        "data",
        "best-of-n",
        "--prompts",
        "prompts.jsonl",
        "--provider",
        "ollama",
        "--model",
        "test-model",
        "--judge",
        "ollama://test-judge",
    ]
    if plan_only:
        arguments.append("--plan-only")

    for option, value in options.items():
        arguments.extend([option, value])

    result = CliRunner().invoke(app, arguments)
    message = " ".join(strip_ansi(result.output).split())

    assert source.read_bytes() == original
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert "--prompts" in message
    assert output_option in message
    assert "same file" in message
    assert sampler_calls == []
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "prompts.jsonl"
    ]

@pytest.mark.requires_hardlink
@pytest.mark.parametrize("plan_only", [True, False], ids=["plan", "live"])
@pytest.mark.parametrize(
    "output_option",
    ["--output", "--emit-pairs", "--manifest"],
)
@pytest.mark.parametrize(
    "input_option",
    ["--candidate-artifact", "--judgments"],
)
def test_offline_rejects_output_hardlink_before_indexing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    input_option: str,
    output_option: str,
    plan_only: bool,
) -> None:
    from soup_cli.utils import best_of_n_stream

    monkeypatch.chdir(tmp_path)
    candidates = Path("candidates.jsonl")
    judgments = Path("judgments.jsonl")
    candidates.write_bytes(b"candidate input must remain unchanged\n")
    judgments.write_bytes(b"judgment input must remain unchanged\n")

    inputs = {
        "--candidate-artifact": candidates,
        "--judgments": judgments,
    }
    alias = Path("alias.jsonl")
    alias.hardlink_to(inputs[input_option])

    def unexpected_indexing(*args, **kwargs):
        pytest.fail("Offline indexing started before rejecting an input alias")

    monkeypatch.setattr(
        best_of_n_stream,
        "index_offline_artifacts",
        unexpected_indexing,
    )

    options = {
        "--candidate-artifact": str(candidates),
        "--judgments": str(judgments),
        "--output": "selected.jsonl",
        "--emit-pairs": "pairs.jsonl",
        "--manifest": "run.manifest.json",
    }
    options[output_option] = str(alias)

    before = {
        path.name: path.read_bytes()
        for path in tmp_path.iterdir()
    }
    arguments = ["data", "best-of-n"]
    if plan_only:
        arguments.append("--plan-only")
    for option, value in options.items():
        arguments.extend([option, value])

    result = CliRunner().invoke(app, arguments)
    message = " ".join(strip_ansi(result.output).split())

    after = {
        path.name: path.read_bytes()
        for path in tmp_path.iterdir()
    }
    assert after == before
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert input_option in message
    assert output_option in message
    assert "same file" in message

@pytest.mark.parametrize("plan_only", [True, False], ids=["plan", "live"])
@pytest.mark.parametrize(
    "output_option",
    ["--export-candidates", "--checkpoint"],
)
@pytest.mark.parametrize(
    "alias_kind",
    [
        "same-path",
        "dot-alias",
        pytest.param("hardlink", marks=pytest.mark.requires_hardlink),
    ],
)
def test_export_rejects_output_aliasing_prompts_before_sampling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output_option: str,
    alias_kind: str,
    plan_only: bool,
) -> None:
    from soup_cli.utils import magpie

    monkeypatch.chdir(tmp_path)
    source = Path("prompts.jsonl")
    source.write_bytes(b'{"prompt": "Explain gravity."}\n')

    if alias_kind == "hardlink":
        alias_path = Path("alias.jsonl")
        alias_path.hardlink_to(source)
        alias = str(alias_path)
    elif alias_kind == "dot-alias":
        alias = "./prompts.jsonl"
    else:
        alias = "prompts.jsonl"

    def unexpected_sampler(*args, **kwargs):
        pytest.fail("Sampler constructed before rejecting an input alias")

    monkeypatch.setattr(
        magpie,
        "make_magpie_generate_fn",
        unexpected_sampler,
    )

    options = {
        "--export-candidates": "candidates.jsonl",
        "--checkpoint": "run.checkpoint.jsonl",
    }
    options[output_option] = alias

    before = {
        path.name: path.read_bytes()
        for path in tmp_path.iterdir()
    }
    arguments = [
        "data",
        "best-of-n",
        "--prompts",
        "prompts.jsonl",
        "--provider",
        "ollama",
        "--model",
        "test-model",
    ]
    if plan_only:
        arguments.append("--plan-only")
    for option, value in options.items():
        arguments.extend([option, value])

    result = CliRunner().invoke(app, arguments)
    message = " ".join(strip_ansi(result.output).split())

    after = {
        path.name: path.read_bytes()
        for path in tmp_path.iterdir()
    }
    assert after == before
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert (
        "must be distinct" in message
        or "same file" in message
    )
