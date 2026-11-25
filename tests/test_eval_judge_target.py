"""Tests for reading target rows through the standalone judge CLI."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.eval.judge import DEFAULT_RUBRIC, JudgeEvaluator


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
def test_judge_scores_target_rows_with_optional_utf8_bom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, encoding: str
) -> None:
    target = tmp_path / "responses.jsonl"
    prompts = ["first\ufeffprompt", "second prompt"]
    payload = "".join(
        json.dumps({"prompt": prompt, "response": "answer"}) + "\r\n" for prompt in prompts
    )
    target.write_bytes(payload.encode(encoding))
    requests: list[str] = []
    scores = {criterion["name"]: 3 for criterion in DEFAULT_RUBRIC["criteria"]}

    def judge_reply(self, prompt: str) -> str:
        requests.append(prompt)
        return json.dumps({"scores": scores})

    monkeypatch.setattr(JudgeEvaluator, "_call_llm", judge_reply)
    result = CliRunner().invoke(app, ["eval", "judge", "--target", str(target)])

    assert result.exit_code == 0, result.output
    assert len(requests) == len(prompts)
    for prompt, request in zip(prompts, requests, strict=True):
        assert prompt in request
