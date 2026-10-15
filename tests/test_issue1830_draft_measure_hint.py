"""Regression tests for issue #1830: soup draft measure and distill missing transformers hint."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi

runner = CliRunner()


def test_draft_measure_missing_transformers_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """soup draft measure without transformers prints install hint with
    soup-cli[train] and exits 1.
    """
    monkeypatch.chdir(tmp_path)
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"prompt": "Hello world"}) + "\n")

    def _mock_vocab_size(*args: object, **kwargs: object) -> int:
        raise ModuleNotFoundError("No module named 'transformers'")

    monkeypatch.setattr("soup_cli.commands.draft._vocab_size_of", _mock_vocab_size)

    result = runner.invoke(
        app,
        [
            "draft",
            "measure",
            "--target",
            "meta-llama/Llama-3-8B",
            "--draft",
            "meta-llama/Llama-3-8B-draft",
            "--prompts",
            "prompts.jsonl",
        ],
    )
    assert result.exit_code == 1
    out = strip_ansi(result.output)
    assert "could not read model config: No module named 'transformers'" in out
    assert 'soup-cli[train]' in out
    assert 'Install with: pip install "soup-cli[train]"' in out


def test_draft_distill_missing_transformers_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """soup draft distill without transformers prints install hint with
    soup-cli[train] and exits 1.
    """
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"instruction": "hi", "output": "bye"}) + "\n")

    def _mock_vocab_size(*args: object, **kwargs: object) -> int:
        raise ModuleNotFoundError("No module named 'transformers'")

    monkeypatch.setattr("soup_cli.commands.draft._vocab_size_of", _mock_vocab_size)

    result = runner.invoke(
        app,
        [
            "draft",
            "distill",
            "--target",
            "meta-llama/Llama-3-8B",
            "--draft-base",
            "meta-llama/Llama-3-8B-draft",
            "--data",
            "data.jsonl",
            "--output",
            "draft_out",
        ],
    )
    assert result.exit_code == 1
    out = strip_ansi(result.output)
    assert "could not read model config: No module named 'transformers'" in out
    assert 'soup-cli[train]' in out
    assert 'Install with: pip install "soup-cli[train]"' in out
