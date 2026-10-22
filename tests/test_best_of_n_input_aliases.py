import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi


@pytest.mark.parametrize("resume", [False, True], ids=["new-run", "resume"])
@pytest.mark.parametrize(
    "output_option",
    ["--output", "--emit-pairs", "--checkpoint", "--manifest"],
)
@pytest.mark.parametrize(
    "alias_kind",
    [
        "same-path",
        "dot-alias",
        pytest.param("hardlink", marks=pytest.mark.requires_hardlink),
        "windows-case",
    ],
)
def test_online_input_alias_is_rejected_without_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resume: bool,
    output_option: str,
    alias_kind: str,
) -> None:
    from soup_cli.eval import judge
    from soup_cli.utils import magpie

    monkeypatch.chdir(tmp_path)
    source = Path("prompts.jsonl")
    source.write_bytes(b'{"prompt":"Explain gravity."}')

    alias = str(source)
    if alias_kind == "dot-alias":
        alias = "./prompts.jsonl"
    elif alias_kind == "hardlink":
        linked = Path("linked-prompts.jsonl")
        linked.hardlink_to(source)
        alias = str(linked)
    elif alias_kind == "windows-case":
        if os.name != "nt":
            pytest.skip("Windows case-alias test")
        alias = "PROMPTS.JSONL"
        if not os.path.exists(alias) or not os.path.samefile(source, alias):
            pytest.skip("Working directory is case-sensitive")

    sampler_calls = []

    def forbidden_sampler(*args, **kwargs):
        sampler_calls.append((args, kwargs))
        pytest.fail("Sampler constructed before rejecting an input alias")

    monkeypatch.setattr(judge, "JudgeEvaluator", lambda **kwargs: object())
    monkeypatch.setattr(magpie, "make_magpie_generate_fn", forbidden_sampler)

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
        str(source),
        "--provider",
        "ollama",
        "--model",
        "test-model",
        "--judge",
        "ollama://test-judge",
    ]
    for option, value in options.items():
        arguments.extend([option, value])
    if resume:
        arguments.append("--resume")

    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    result = CliRunner().invoke(app, arguments)
    after = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    message = " ".join(strip_ansi(result.output).split())

    assert after == before
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert "--prompts" in message
    assert output_option in message
    assert "same file" in message
    assert sampler_calls == []
