"""Tests for Issue #1435 (PR 1): a probe that never ran must not score OK 1.00.

Covers the ``NOT_RUN`` verdict, its ranking, the live-path sites that used to fall
back to ``neutral_score``, the exit code / ``--allow-not-run`` flag, the badge colour,
and resolving ``--tokenizer`` before any model is loaded.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest
from typer.testing import CliRunner

import soup_cli.utils.diagnose.live as live
from soup_cli.cli import app
from soup_cli.utils.diagnose.report import (
    VERDICTS,
    FailureReport,
    FailureScore,
    overall_verdict,
)
from tests.conftest import strip_ansi

DATASET_MODES = ("forgetting", "format", "mode_collapse", "memorization")

RECOGNISED = [
    {"prompt": f"Write story number {i}.", "completion": f"Once upon a time {i}."}
    for i in range(12)
]
# No key any row builder reads, now or after the PR 2 converter rewrite.
UNRECOGNISED = [{"foo": f"bar {i}"} for i in range(12)]


def _gens(**overrides):
    gens = {
        "base_gen": lambda prompt: "ok",
        "adapter_gen": lambda prompt: "ok",
        "base_multi": lambda prompt, k: [f"base{i} x{i}y z{i}w" for i in range(k)],
        "adapter_multi": lambda prompt, k: [f"adapter{i} p{i}q r{i}s" for i in range(k)],
    }
    gens.update(overrides)
    return gens


def _write_rows(tmp_path: Path, rows) -> None:
    (tmp_path / "d.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )


def _run_live(tmp_path, monkeypatch, rows, gens, **kwargs):
    """Run ``run_live_diagnose`` with a stand-in model pair; returns (report, load_calls)."""
    monkeypatch.chdir(tmp_path)
    _write_rows(tmp_path, rows)
    calls: list = []

    def fake_pair(base, adapter=None, **kw):
        calls.append(base)
        return gens

    monkeypatch.setattr(live, "load_adapter_pair", fake_pair)
    kwargs.setdefault("dataset_path", "d.jsonl")
    report = live.run_live_diagnose(run_id="r", base="b", adapter="a", **kwargs)
    return report, calls


def _cli(tmp_path, monkeypatch, rows, gens, *extra):
    monkeypatch.chdir(tmp_path)
    _write_rows(tmp_path, rows)
    calls: list = []

    def fake_pair(base, adapter=None, **kw):
        calls.append(base)
        return gens

    monkeypatch.setattr(live, "load_adapter_pair", fake_pair)
    args = ["diagnose", "r", "--base-model", "b", "--adapter", "a", "--dataset", "d.jsonl"]
    result = CliRunner().invoke(app, [*args, *extra])
    return result, strip_ansi(result.output), calls


# ---------------------------------------------------------------------------
# Verdict taxonomy
# ---------------------------------------------------------------------------


class TestNotRunVerdict:
    def test_verdicts_include_not_run_and_keep_the_old_three(self):
        assert "NOT_RUN" in VERDICTS
        assert {"OK", "MINOR", "MAJOR"} <= set(VERDICTS)

    def test_not_run_score_helper(self):
        from soup_cli.utils.diagnose.runner import neutral_score, not_run_score

        score = not_run_score("memorization", "no usable rows")
        assert score.verdict == "NOT_RUN"
        assert score.score == 0.0
        assert "probe not run" in score.evidence
        assert "no usable rows" in score.evidence
        # The by-design "not applicable" helper keeps its documented behaviour.
        assert neutral_score("memorization", "x").verdict == "OK"

    def test_failure_score_accepts_not_run_only_with_zero_score(self):
        ok = FailureScore(mode="format", score=0.0, verdict="NOT_RUN", evidence="probe not run")
        assert ok.verdict == "NOT_RUN"
        with pytest.raises(ValueError):
            FailureScore(mode="format", score=0.5, verdict="NOT_RUN", evidence="x")

    def test_old_verdicts_still_validate(self):
        assert FailureScore(mode="format", score=0.9, verdict="OK", evidence="x").verdict == "OK"
        with pytest.raises(ValueError, match="disagrees"):
            FailureScore(mode="format", score=0.9, verdict="MAJOR", evidence="x")

    @pytest.mark.parametrize(
        ("verdicts", "expected"),
        [
            (("OK", "NOT_RUN"), "NOT_RUN"),
            (("MINOR", "NOT_RUN"), "NOT_RUN"),
            (("MAJOR", "NOT_RUN"), "MAJOR"),
            (("OK", "MINOR"), "MINOR"),
        ],
    )
    def test_overall_ranking(self, verdicts, expected):
        modes = ["forgetting", "format", "mode_collapse"]
        scores = {}
        for mode, verdict in zip(modes, verdicts):
            value = {"OK": 1.0, "MINOR": 0.7, "MAJOR": 0.1, "NOT_RUN": 0.0}[verdict]
            scores[mode] = FailureScore(mode=mode, score=value, verdict=verdict, evidence="x")
        assert overall_verdict(scores) == expected

    def test_report_serialises_not_run(self):
        from soup_cli.utils.diagnose.runner import build_report, not_run_score

        report = build_report(
            run_id="r", base="b", adapter="a", scores={"format": not_run_score("format", "why")}
        )
        assert isinstance(report, FailureReport)
        assert report.overall == "NOT_RUN"
        assert report.to_dict()["scores"]["format"]["verdict"] == "NOT_RUN"


# ---------------------------------------------------------------------------
# Live path
# ---------------------------------------------------------------------------


class TestLivePath:
    def test_dataset_without_usable_pairs_is_not_run(self, tmp_path, monkeypatch):
        report, _ = _run_live(tmp_path, monkeypatch, UNRECOGNISED, _gens())
        for mode in DATASET_MODES:
            assert report.scores[mode].verdict == "NOT_RUN", mode
        assert report.overall == "NOT_RUN"

    def test_probe_exception_is_not_run_and_keeps_the_reason(self, tmp_path, monkeypatch):
        def boom(prompt, k):
            raise ValueError("generator exploded")

        report, _ = _run_live(
            tmp_path, monkeypatch, RECOGNISED, _gens(adapter_multi=boom)
        )
        score = report.scores["mode_collapse"]
        assert score.verdict == "NOT_RUN"
        assert "ValueError" in score.evidence
        assert "generator exploded" in score.evidence
        assert report.overall == "NOT_RUN"

    def test_exception_text_is_sanitised_and_capped(self, tmp_path, monkeypatch):
        def boom(prompt, k):
            raise ValueError("a\x00b" + "x" * 5000)

        report, _ = _run_live(
            tmp_path, monkeypatch, RECOGNISED, _gens(adapter_multi=boom)
        )
        evidence = report.scores["mode_collapse"].evidence
        assert "\x00" not in evidence
        assert len(evidence) < 400

    def test_major_outranks_not_run(self, tmp_path, monkeypatch):
        collapsed = _gens(adapter_multi=lambda prompt, k: ["same reply"] * k)
        with mock.patch(
            "soup_cli.utils.diagnose.memorization.score_memorization",
            side_effect=ValueError("probe broke"),
        ):
            report, _ = _run_live(tmp_path, monkeypatch, RECOGNISED, collapsed)
        assert report.scores["mode_collapse"].verdict == "MAJOR"
        assert report.scores["memorization"].verdict == "NOT_RUN"
        assert report.overall == "MAJOR"

    def test_not_applicable_probes_stay_neutral_ok(self, tmp_path, monkeypatch):
        report, _ = _run_live(tmp_path, monkeypatch, RECOGNISED, _gens())
        assert report.scores["format"].verdict == "OK"
        assert "not JSON" in report.scores["format"].evidence
        assert report.scores["contamination"].verdict == "OK"
        assert report.scores["citation"].verdict == "OK"

    def test_no_dataset_keeps_dataset_probes_neutral(self, tmp_path, monkeypatch):
        report, _ = _run_live(
            tmp_path, monkeypatch, RECOGNISED, _gens(), dataset_path=None
        )
        for mode in DATASET_MODES:
            assert report.scores[mode].verdict == "OK", mode
        assert "NOT_RUN" not in {score.verdict for score in report.scores.values()}

    def test_bad_tokenizer_fails_before_any_model_load(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_rows(tmp_path, RECOGNISED)
        calls: list = []
        monkeypatch.setattr(
            live, "load_adapter_pair", lambda base, adapter=None, **kw: calls.append(base)
        )
        with mock.patch(
            "soup_cli.utils.diagnose._common.resolve_tokenizer",
            side_effect=ValueError("could not load tokenizer 'typo/tok'"),
        ):
            with pytest.raises(ValueError, match="could not load tokenizer"):
                live.run_live_diagnose(
                    run_id="r", base="b", dataset_path="d.jsonl", tokenizer="typo/tok"
                )
        assert calls == []


# ---------------------------------------------------------------------------
# CLI: exit code, opt-out flag, badge, tokenizer
# ---------------------------------------------------------------------------


class TestCli:
    def test_not_run_exits_3(self, tmp_path, monkeypatch):
        result, out, _ = _cli(tmp_path, monkeypatch, UNRECOGNISED, _gens())
        assert result.exit_code == 3
        assert "NOT_RUN" in out

    def test_allow_not_run_exits_0(self, tmp_path, monkeypatch):
        result, out, _ = _cli(
            tmp_path, monkeypatch, UNRECOGNISED, _gens(), "--allow-not-run"
        )
        assert result.exit_code == 0
        assert "NOT_RUN" in out  # still visible, only the exit code changes

    def test_major_still_exits_2_with_not_run_present(self, tmp_path, monkeypatch):
        collapsed = _gens(adapter_multi=lambda prompt, k: ["same reply"] * k)
        with mock.patch(
            "soup_cli.utils.diagnose.memorization.score_memorization",
            side_effect=ValueError("probe broke"),
        ):
            result, _, _ = _cli(tmp_path, monkeypatch, RECOGNISED, collapsed)
        assert result.exit_code == 2

    def test_allow_not_run_does_not_hide_a_major(self, tmp_path, monkeypatch):
        collapsed = _gens(adapter_multi=lambda prompt, k: ["same reply"] * k)
        with mock.patch(
            "soup_cli.utils.diagnose.memorization.score_memorization",
            side_effect=ValueError("probe broke"),
        ):
            result, _, _ = _cli(
                tmp_path, monkeypatch, RECOGNISED, collapsed, "--allow-not-run"
            )
        assert result.exit_code == 2

    def test_badge_is_grey_not_green_for_not_run(self, tmp_path, monkeypatch):
        result, _, _ = _cli(
            tmp_path, monkeypatch, UNRECOGNISED, _gens(), "--badge", "b.svg", "--allow-not-run"
        )
        assert result.exit_code == 0
        svg = (tmp_path / "b.svg").read_text(encoding="utf-8")
        assert "NOT_RUN" in svg  # the overall pill
        assert "#8b949e" in svg
        # The four dataset probes show "not run", never a stored 0.00 placeholder.
        assert svg.count("not run") == 4
        assert "0.00" not in svg
        # Only refusal, contamination and citation are OK (green); the pill is grey.
        assert svg.count("#3fb950") == 3

    def test_bad_tokenizer_exits_3_before_model_load(self, tmp_path, monkeypatch):
        with mock.patch(
            "soup_cli.utils.diagnose._common.resolve_tokenizer",
            side_effect=ValueError("could not load tokenizer 'typo/tok'"),
        ):
            result, out, calls = _cli(
                tmp_path, monkeypatch, RECOGNISED, _gens(), "--tokenizer", "typo/tok"
            )
        assert result.exit_code == 3
        assert "could not load tokenizer" in out
        assert calls == []

    def test_evidence_only_path_is_unchanged(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(app, ["diagnose", "r1"])
        assert result.exit_code == 0
        assert "NOT_RUN" not in strip_ansi(result.output)


# ---------------------------------------------------------------------------
# Evidence files carrying NOT_RUN (CLI + MCP)
# ---------------------------------------------------------------------------


def _evidence_file(tmp_path: Path, scores: dict) -> str:
    (tmp_path / "ev.json").write_text(json.dumps({"scores": scores}), encoding="utf-8")
    return "ev.json"


class TestEvidenceNotRun:
    def test_cli_evidence_not_run_exits_3(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = _evidence_file(tmp_path, {"format": {"verdict": "NOT_RUN", "evidence": "never"}})
        result = CliRunner().invoke(app, ["diagnose", "r", "--evidence", path])
        assert result.exit_code == 3
        assert "NOT_RUN" in strip_ansi(result.output)

    def test_cli_evidence_not_run_allowed_exits_0(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = _evidence_file(tmp_path, {"format": {"verdict": "NOT_RUN", "evidence": "never"}})
        result = CliRunner().invoke(app, ["diagnose", "r", "--evidence", path, "--allow-not-run"])
        assert result.exit_code == 0

    def test_cli_not_run_with_a_score_is_a_clear_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = _evidence_file(
            tmp_path, {"format": {"verdict": "NOT_RUN", "score": 0.4, "evidence": "x"}}
        )
        result = CliRunner().invoke(app, ["diagnose", "r", "--evidence", path])
        # A usage error (2/3) with a message, not a ValueError traceback (exit 1).
        assert result.exit_code in (2, 3)
        assert isinstance(result.exception, SystemExit)
        assert "NOT_RUN score must be 0.0" in strip_ansi(result.output)

    def test_not_run_entry_needs_no_score(self):
        from soup_cli.utils.diagnose.report import evidence_default_score

        assert evidence_default_score({"verdict": "NOT_RUN"}) == 0.0
        assert evidence_default_score({"verdict": "OK"}) == 1.0
        assert evidence_default_score({}) == 1.0

    def test_mcp_evidence_round_trips_not_run(self, tmp_path, monkeypatch):
        from soup_cli.mcp_server.registry import tool_diagnose_evidence

        monkeypatch.chdir(tmp_path)
        path = _evidence_file(tmp_path, {"format": {"verdict": "NOT_RUN", "evidence": "never"}})
        out = tool_diagnose_evidence({"run_id": "r", "evidence": path})
        assert out["overall"] == "NOT_RUN"
        assert out["scores"]["format"]["verdict"] == "NOT_RUN"


# ---------------------------------------------------------------------------
# Every probe site reports NOT_RUN, and a probe with nothing to measure is not OK
# ---------------------------------------------------------------------------

JSON_ROWS = [
    {"prompt": f"Return object {i}.", "completion": f'{{"id": {i}}}'} for i in range(12)
]
RAFT_ROWS = [
    {
        "query": f"question {i}?",
        "golden_doc": f"golden doc {i}",
        "answer": f"answer {i} [doc-1]",
        "text": f"document {i} has several separate words that can be split",
    }
    for i in range(12)
]
# Pairs exist (the pair builder reads these keys) but score_memorization reads none of them
# and used to answer "nothing to check" with OK 1.00.
QUESTION_ANSWER = [
    {"question": f"What is topic {i} about in detail?", "answer": f"Topic {i} is about things."}
    for i in range(12)
]
QUERY_RESPONSE = [
    {"query": f"Tell me about topic {i} in detail.", "response": f"Topic {i} is about things."}
    for i in range(12)
]
INPUT_OUTPUT = [
    {"input": f"translate sentence number {i} to French", "output": f"phrase {i}"}
    for i in range(12)
]

PROBE_SITES = [
    ("refusal", "soup_cli.utils.diagnose.refusal.score_refusal", RECOGNISED),
    ("forgetting", "soup_cli.utils.diagnose.forgetting.score_forgetting", RECOGNISED),
    ("format", "soup_cli.utils.diagnose.format.score_format", JSON_ROWS),
    ("mode_collapse", "soup_cli.utils.diagnose.mode_collapse.score_mode_collapse", RECOGNISED),
    ("memorization", "soup_cli.utils.diagnose.memorization.score_memorization", RECOGNISED),
    ("citation", "soup_cli.utils.diagnose.citation.score_citation", RAFT_ROWS),
]


class TestEveryProbeSite:
    @pytest.mark.parametrize(
        ("mode", "target", "rows"), PROBE_SITES, ids=[site[0] for site in PROBE_SITES]
    )
    def test_a_probe_that_raises_is_not_run_with_its_reason(
        self, tmp_path, monkeypatch, mode, target, rows
    ):
        with mock.patch(target, side_effect=ValueError("probe exploded")):
            report, _ = _run_live(tmp_path, monkeypatch, rows, _gens())
        score = report.scores[mode]
        assert score.verdict == "NOT_RUN", (mode, score)
        assert "ValueError: probe exploded" in score.evidence
        assert report.overall == "NOT_RUN"

    @pytest.mark.parametrize(
        "rows",
        [QUESTION_ANSWER, QUERY_RESPONSE, INPUT_OUTPUT],
        ids=["question_answer", "query_response", "input_output"],
    )
    def test_memorization_with_no_splittable_text_is_not_run(self, tmp_path, monkeypatch, rows):
        report, _ = _run_live(tmp_path, monkeypatch, rows, _gens())
        score = report.scores["memorization"]
        assert score.verdict == "NOT_RUN", score
        assert "text, content" in score.evidence  # the new guard, not "no usable rows"
        assert report.scores["forgetting"].verdict != "NOT_RUN"  # the rows do form pairs
        assert report.overall == "NOT_RUN"


# ---------------------------------------------------------------------------
# Review follow-ups: table cell, empty --tokenizer, grey style
# ---------------------------------------------------------------------------


class TestReviewFollowUps:
    def test_table_shows_a_dash_not_a_zero_score_for_not_run(self, tmp_path, monkeypatch):
        result, out, _ = _cli(
            tmp_path, monkeypatch, UNRECOGNISED, _gens(), "--allow-not-run"
        )
        assert result.exit_code == 0
        modes = ("forgetting", "format", "mode_collapse", "memorization")
        rows = [
            line for line in out.splitlines()
            if "NOT_RUN" in line and any(mode in line for mode in modes)
        ]
        assert len(rows) == 4  # one table row per NOT_RUN probe, not the overall panel
        for line in rows:
            assert "\u2014" in line
            assert "0.000" not in line

    def test_empty_tokenizer_is_an_input_error(self, tmp_path, monkeypatch):
        result, out, calls = _cli(
            tmp_path, monkeypatch, RECOGNISED, _gens(), "--tokenizer", ""
        )
        assert result.exit_code == 3
        assert "non-empty" in out
        assert calls == []

    def test_not_run_is_styled_grey(self):
        from soup_cli.commands.diagnose import _verdict_style

        assert _verdict_style("NOT_RUN") == "bright_black"
