"""Regression coverage for issue #821's toxicity-heuristic claims."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi


def test_abuse_keywords_do_not_restore_ambiguous_action_words() -> None:
    from soup_cli.utils.data_score import score_toxicity

    benign = (
        "heart attack symptoms",
        "Why does my Python thread die?",
        "How do I kill a zombie process in Linux?",
    )

    assert score_toxicity("You are a worthless idiot.") > 0.0
    assert all(score_toxicity(text) == 0.0 for text in benign)


def test_magpie_default_quality_keeps_benign_process_management_text() -> None:
    from soup_cli.utils.magpie import default_quality_fn

    assert default_quality_fn(
        "How do I kill a hung process?",
        "Run kill -9 on its PID.",
    )


def test_magpie_default_quality_never_calls_legacy_toxicity_scorer(monkeypatch) -> None:
    from soup_cli.utils import data_score
    from soup_cli.utils.magpie import default_quality_fn

    def fail_if_called(_text):
        raise AssertionError("Magpie must not use the keyword heuristic as a safety gate")

    monkeypatch.setattr(data_score, "score_abuse_keywords", fail_if_called)
    assert default_quality_fn(
        "Explain why a baseline matters in model evaluation.",
        "A baseline makes changes measurable and helps distinguish signal from noise.",
    )


def test_toxicity_command_names_and_serialises_the_actual_heuristic(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "rows.jsonl"
    input_path.write_text(
        json.dumps({"text": "You are a worthless idiot."}) + "\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    help_result = runner.invoke(app, ["data", "toxicity", "--help"])
    assert help_result.exit_code == 0, help_result.output
    assert "abuse-keyword heuristic" in strip_ansi(help_result.output).lower()

    result = runner.invoke(
        app,
        ["data", "toxicity", "--input", str(input_path)],
    )
    assert result.exit_code == 0, result.output
    assert "abuse-keyword heuristic" in strip_ansi(result.output).lower()
    row = json.loads((tmp_path / "toxicity.jsonl").read_text(encoding="utf-8"))
    assert row["_abuse_keyword_score"] > 0.0
    assert row["_toxicity"] == row["_abuse_keyword_score"]


def test_mcp_scorecard_preserves_legacy_key_and_adds_honest_alias(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from soup_cli.mcp_server.registry import tool_data_score

    monkeypatch.chdir(tmp_path)
    (tmp_path / "rows.jsonl").write_text(
        json.dumps({"text": "You are a worthless idiot."}) + "\n",
        encoding="utf-8",
    )

    result = tool_data_score({"data": "rows.jsonl"})
    assert result["toxic_flagged"] == 1
    assert result["abuse_keyword_flagged"] == result["toxic_flagged"]


def test_docs_do_not_claim_unshipped_data_classifiers() -> None:
    docs = (
        Path(__file__).resolve().parents[1] / "docs" / "data.md"
    ).read_text(encoding="utf-8")

    assert "Llama-Guard-3-1B variant + FineWeb-Edu classifier ship" not in docs
    assert "not a toxicity classifier" in docs
