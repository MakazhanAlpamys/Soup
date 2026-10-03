"""Tests for Issue #1525 (item 1): ``--require-all-modes`` for evidence-only diagnose.

A mode missing from the evidence reads as a neutral OK by default (documented, kept).
With the opt-in flag it is ``NOT_RUN``, so a gate can prove coverage. The rule lives in
``build_report(missing_as_not_run=...)`` and is exposed by the CLI, the MCP tool and
``soup train --diagnose-gate``.
"""

from __future__ import annotations

import json
import re

import pytest
import typer
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.diagnose.report import FAILURE_MODES, FailureScore
from soup_cli.utils.diagnose.runner import build_report

runner = CliRunner()


def _ok(mode: str) -> FailureScore:
    return FailureScore(mode=mode, score=1.0, verdict="OK", evidence="measured")


def _evidence(tmp_path, scores: dict, name: str = "ev.json") -> str:
    (tmp_path / name).write_text(json.dumps({"scores": scores}), encoding="utf-8")
    return name


ONE_MODE = {"format": {"score": 1.0, "verdict": "OK"}}
FULL = {mode: {"score": 1.0, "verdict": "OK"} for mode in FAILURE_MODES}


class TestBuildReport:
    def test_default_fills_neutral_ok(self):
        report = build_report(run_id="r", base="b", adapter="a", scores={"format": _ok("format")})
        assert report.overall == "OK"

    def test_flag_fills_missing_modes_with_not_run(self):
        report = build_report(
            run_id="r",
            base="b",
            adapter="a",
            scores={"format": _ok("format")},
            missing_as_not_run=True,
        )
        assert report.overall == "NOT_RUN"
        assert report.scores["format"].verdict == "OK"
        others = [m for m in FAILURE_MODES if m != "format"]
        assert all(report.scores[m].verdict == "NOT_RUN" for m in others)
        assert "no evidence supplied" in report.scores[others[0]].evidence

    def test_flag_with_every_mode_present_is_ok(self):
        scores = {mode: _ok(mode) for mode in FAILURE_MODES}
        report = build_report(
            run_id="r", base="b", adapter="a", scores=scores, missing_as_not_run=True
        )
        assert report.overall == "OK"

    def test_major_still_wins(self):
        major = FailureScore(mode="format", score=0.0, verdict="MAJOR", evidence="broken")
        report = build_report(
            run_id="r", base="b", adapter="a", scores={"format": major}, missing_as_not_run=True
        )
        assert report.overall == "MAJOR"


class TestDiagnoseCli:
    def _run(self, tmp_path, monkeypatch, scores, *flags):
        monkeypatch.chdir(tmp_path)
        name = _evidence(tmp_path, scores)
        return runner.invoke(app, ["diagnose", "r", "--evidence", name, *flags])

    def test_empty_evidence_exits_3(self, tmp_path, monkeypatch):
        result = self._run(tmp_path, monkeypatch, {}, "--require-all-modes")
        assert result.exit_code == 3, result.output

    def test_one_mode_exits_3(self, tmp_path, monkeypatch):
        result = self._run(tmp_path, monkeypatch, ONE_MODE, "--require-all-modes")
        assert result.exit_code == 3, result.output

    def test_allow_not_run_accepts_it(self, tmp_path, monkeypatch):
        result = self._run(
            tmp_path, monkeypatch, ONE_MODE, "--require-all-modes", "--allow-not-run"
        )
        assert result.exit_code == 0, result.output

    def test_major_exits_2(self, tmp_path, monkeypatch):
        scores = {"format": {"score": 0.0, "verdict": "MAJOR", "evidence": "broken"}}
        result = self._run(tmp_path, monkeypatch, scores, "--require-all-modes")
        assert result.exit_code == 2, result.output

    def test_complete_evidence_exits_0(self, tmp_path, monkeypatch):
        result = self._run(tmp_path, monkeypatch, FULL, "--require-all-modes")
        assert result.exit_code == 0, result.output

    def test_default_is_unchanged(self, tmp_path, monkeypatch):
        result = self._run(tmp_path, monkeypatch, ONE_MODE)
        assert result.exit_code == 0, result.output

    def test_report_marks_missing_modes_not_run(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        name = _evidence(tmp_path, ONE_MODE)
        result = runner.invoke(
            app,
            ["diagnose", "r", "--evidence", name, "--require-all-modes", "-o", "out.json"],
        )
        assert result.exit_code == 3, result.output
        scores = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))["scores"]
        assert scores["format"]["verdict"] == "OK"
        assert scores["refusal"]["verdict"] == "NOT_RUN"

    def test_flag_with_base_model_is_a_usage_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        name = _evidence(tmp_path, ONE_MODE)
        result = runner.invoke(
            app,
            ["diagnose", "r", "--evidence", name, "--base-model", "b", "--require-all-modes"],
        )
        assert result.exit_code == 3, result.output
        assert "--base-model" in result.output

    def test_flag_without_evidence_is_a_usage_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["diagnose", "r", "--require-all-modes"])
        assert result.exit_code == 3, result.output
        assert "--evidence" in result.output


class TestMcpTool:
    def _call(self, tmp_path, monkeypatch, scores, **extra):
        from soup_cli.mcp_server import registry as reg

        monkeypatch.chdir(tmp_path)
        name = _evidence(tmp_path, scores)
        return reg.tool_diagnose_evidence({"run_id": "r", "evidence": name, **extra})

    def test_flag_gives_not_run(self, tmp_path, monkeypatch):
        out = self._call(tmp_path, monkeypatch, {}, require_all_modes=True)
        assert out["overall"] == "NOT_RUN"

    def test_default_is_unchanged(self, tmp_path, monkeypatch):
        assert self._call(tmp_path, monkeypatch, {})["overall"] == "OK"

    def test_false_is_the_default(self, tmp_path, monkeypatch):
        assert self._call(tmp_path, monkeypatch, {}, require_all_modes=False)["overall"] == "OK"

    @pytest.mark.parametrize("bad", ["yes", 1, None, []])
    def test_non_bool_is_rejected(self, tmp_path, monkeypatch, bad):
        from soup_cli.mcp_server import registry as reg

        with pytest.raises(reg.McpToolError):
            self._call(tmp_path, monkeypatch, {}, require_all_modes=bad)

    def test_schema_lists_the_argument(self):
        from soup_cli.mcp_server import registry as reg

        spec = next(
            tool for tool in reg.build_registry(allow_mutating=False)
            if tool.name == "diagnose_evidence"
        )
        assert spec.input_schema["properties"]["require_all_modes"]["type"] == "boolean"


class TestTrainGate:
    def test_empty_evidence_with_flag_exits_3(self, tmp_path, monkeypatch):
        from soup_cli.commands.train import _run_diagnose_gate

        monkeypatch.chdir(tmp_path)
        name = _evidence(tmp_path, {})
        with pytest.raises(typer.Exit) as excinfo:
            _run_diagnose_gate(name, "run1", "base", "adapter", require_all_modes=True)
        assert excinfo.value.exit_code == 3

    def test_empty_evidence_default_passes(self, tmp_path, monkeypatch):
        from soup_cli.commands.train import _run_diagnose_gate

        monkeypatch.chdir(tmp_path)
        _run_diagnose_gate(_evidence(tmp_path, {}), "run1", "base", "adapter")

    def test_complete_evidence_with_flag_passes(self, tmp_path, monkeypatch):
        from soup_cli.commands.train import _run_diagnose_gate

        monkeypatch.chdir(tmp_path)
        name = _evidence(tmp_path, FULL)
        _run_diagnose_gate(name, "run1", "base", "adapter", require_all_modes=True)

    def test_flag_without_gate_is_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "cfg.yaml").write_text(
            "base: HuggingFaceTB/SmolLM2-135M\n"
            "task: grpo\n"
            "data:\n  train: ./train.jsonl\n  format: chatml\n"
            "training:\n  reward_fn: accuracy\n",
            encoding="utf-8",
        )
        result = runner.invoke(
            app,
            ["train", "--config", "cfg.yaml", "--diagnose-gate-require-all-modes", "--yes"],
        )
        assert result.exit_code == 1, result.output
        assert "--diagnose-gate" in result.output

    def test_launcher_forwards_the_switch(self):
        from soup_cli.utils.launcher import collect_reexec_passthrough

        extra = collect_reexec_passthrough(
            diagnose_gate="ev.json", diagnose_gate_require_all_modes=True
        )
        assert "--diagnose-gate-require-all-modes" in extra
        assert "--diagnose-gate-require-all-modes" not in collect_reexec_passthrough(
            diagnose_gate="ev.json"
        )

    def test_help_lists_the_flag(self):
        from tests.conftest import strip_ansi

        result = runner.invoke(app, ["train", "--help"], env={"COLUMNS": "250"})
        assert "--diagnose-gate-require-all-modes" in strip_ansi(result.output)

    def test_train_passes_the_switch_to_the_gate(self):
        # Running a full train is too heavy; pin the call site the way the repo
        # pins other train() wiring (inspect the source).
        import inspect

        from soup_cli.commands import train as train_mod

        src = inspect.getsource(train_mod.train)
        # Not preceded by a word char: the launcher kwarg ends in the same text.
        assert re.search(r"(?<!\w)require_all_modes=diagnose_gate_require_all_modes", src)

    def test_train_passes_the_switch_to_the_reexec(self):
        # Dropping this kwarg makes a --gpus child run the gate without the flag.
        import inspect

        from soup_cli.commands import train as train_mod

        src = inspect.getsource(train_mod.train)
        assert re.search(
            r"\bdiagnose_gate_require_all_modes=diagnose_gate_require_all_modes\b", src
        )
