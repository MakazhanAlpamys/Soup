"""#1227 — `soup ab` runs a two-sided mSPRT and reports the direction.

The statistic was Wald's SPRT for the single point alternative
``mean_t - mean_c = +effect_size``, evaluated on the signed z. A treatment that
moved the other way drove the log-likelihood ratio towards minus infinity, i.e.
across the ACCEPT boundary: the larger the regression, the more confidently the
command said "No significant difference". On ``latency`` and ``retry_rate``
(lower is better) that made a real improvement look like no difference too.

The #1227 fix was a symmetric two-point mixture with a plug-in variance and a
burn-in. #1265 replaced that statistic with the normal-inverse-gamma mixture,
whose prior on the standardised difference is symmetric too, so both signs of
a difference are equal evidence; its own tests (the Bayes factor, Type-I under
peeking, power) are in test_issue1265_ab_nig.py. What this file pins is the
#1227 contract, whatever the statistic: a shift either way is rejected, the
direction is read through the metric's polarity, and the CLI panel, table and
webhook payload carry it.
"""

from __future__ import annotations

import json
import math
import re

import pytest
from typer.testing import CliRunner

runner = CliRunner()

# Rich colours per character on Linux CI, and wraps; compare on plain text.
# Box-drawing characters go too, so a table row reads "direction worse" and a
# panel sentence wrapped across two lines reads as one sentence.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_BOX_RE = re.compile(f"[{chr(0x2500)}-{chr(0x257F)}|]")  # the box-drawing block, and "|"


def _plain(text: str) -> str:
    return " ".join(_BOX_RE.sub(" ", _ANSI_RE.sub("", text)).split())


# The reproduction data from the issue, rows around 0.80, repeated to 40 rows per
# arm (the #1227 burn-in needed that many; since #1265 the first 5 set the prior
# scale and the rest are tested).
_JUDGE_CONTROL = [0.80, 0.82, 0.78, 0.81, 0.79] * 8
_LATENCY_CONTROL = [1.20, 1.25, 1.18, 1.22, 1.21] * 8
_RETRY_CONTROL = [0.20, 0.22, 0.18, 0.21, 0.19] * 8


def _shifted(values, shift):
    return [x + shift for x in values]


def _step(metric, control, treatment, **kwargs):
    from soup_cli.utils.ab_test import MsprtConfig, msprt_step

    return msprt_step(MsprtConfig(metric=metric, **kwargs), control=control, treatment=treatment)


def _upper(alpha=0.05):
    return math.log(1.0 / alpha)  # the reject boundary since #1265


# ---------------------------------------------------------------------------
# Acceptance: judge_score (higher is better)
# ---------------------------------------------------------------------------


class TestJudgeScoreIsTwoSided:
    @pytest.mark.parametrize("shift", [-0.30, -0.10])
    def test_regression_is_rejected_as_worse(self, shift):
        verdict = _step("judge_score", _JUDGE_CONTROL, _shifted(_JUDGE_CONTROL, shift))
        assert verdict.decision == "reject_h0", verdict
        assert verdict.direction == "worse", verdict
        assert verdict.log_likelihood_ratio >= _upper()

    @pytest.mark.parametrize("shift", [0.10, 0.30])
    def test_improvement_is_rejected_as_better(self, shift):
        verdict = _step("judge_score", _JUDGE_CONTROL, _shifted(_JUDGE_CONTROL, shift))
        assert verdict.decision == "reject_h0", verdict
        assert verdict.direction == "better", verdict

    def test_statistic_is_symmetric_in_the_sign_of_the_difference(self):
        """A regression is exactly as much evidence as the mirror-image improvement.

        The one-sided statistic gave +1142.8 for +0.30 and -1574.6 for -0.30.
        """
        up = _step("judge_score", _JUDGE_CONTROL, _shifted(_JUDGE_CONTROL, 0.30))
        down = _step("judge_score", _JUDGE_CONTROL, _shifted(_JUDGE_CONTROL, -0.30))
        assert math.isfinite(up.log_likelihood_ratio)
        assert math.isfinite(down.log_likelihood_ratio)
        assert down.log_likelihood_ratio == pytest.approx(up.log_likelihood_ratio, rel=1e-9)


# ---------------------------------------------------------------------------
# Acceptance: latency and retry_rate (lower is better)
# ---------------------------------------------------------------------------


class TestLowerIsBetterMetrics:
    @pytest.mark.parametrize(
        ("metric", "control", "shift", "expected"),
        [
            ("latency", _LATENCY_CONTROL, -0.30, "better"),
            ("latency", _LATENCY_CONTROL, +0.30, "worse"),
            ("retry_rate", _RETRY_CONTROL, -0.10, "better"),
            ("retry_rate", _RETRY_CONTROL, +0.10, "worse"),
        ],
    )
    def test_direction_follows_polarity(self, metric, control, shift, expected):
        verdict = _step(metric, control, _shifted(control, shift))
        assert verdict.decision == "reject_h0", verdict
        assert verdict.direction == expected, verdict

    @pytest.mark.parametrize(
        ("spelling", "control", "shift", "expected"),
        [
            (" LATENCY ", _LATENCY_CONTROL, +0.30, "worse"),
            ("Retry_Rate", _RETRY_CONTROL, -0.10, "better"),
            ("JUDGE_SCORE", _JUDGE_CONTROL, -0.10, "worse"),
        ],
    )
    def test_polarity_uses_the_canonical_metric_name(self, spelling, control, shift, expected):
        verdict = _step(spelling, control, _shifted(control, shift))
        assert verdict.decision == "reject_h0", verdict
        assert verdict.direction == expected, verdict

    def test_every_supported_metric_declares_a_polarity(self):
        """A new metric must say which way is better, or it inherits a wrong direction."""
        from soup_cli.utils.ab_test import HIGHER_IS_BETTER, SUPPORTED_METRICS

        assert set(HIGHER_IS_BETTER) == set(SUPPORTED_METRICS)
        assert HIGHER_IS_BETTER["judge_score"] is True
        assert HIGHER_IS_BETTER["latency"] is False
        assert HIGHER_IS_BETTER["retry_rate"] is False

    def test_polarity_table_is_read_only(self):
        from soup_cli.utils.ab_test import HIGHER_IS_BETTER

        with pytest.raises(TypeError):
            HIGHER_IS_BETTER["latency"] = True  # type: ignore[index]


# ---------------------------------------------------------------------------
# Acceptance: no shift never rejects; undecided verdicts carry no direction
# ---------------------------------------------------------------------------


class TestNoDifference:
    @pytest.mark.parametrize(
        ("metric", "control"),
        [
            ("judge_score", _JUDGE_CONTROL),
            ("latency", _LATENCY_CONTROL),
            ("retry_rate", _RETRY_CONTROL),
        ],
    )
    def test_shift_zero_never_rejects(self, metric, control):
        verdict = _step(metric, control, list(control))
        assert verdict.decision in ("accept_h0", "continue"), verdict
        assert verdict.direction is None

    def test_too_few_rows_continue_without_direction(self):
        verdict = _step("latency", [1.0], [9.0])
        assert verdict.decision == "continue"
        assert verdict.direction is None

    def test_zero_variance_continues_without_direction(self):
        verdict = _step("latency", [1.0] * 30, [2.0] * 30)
        assert verdict.decision == "continue"
        assert verdict.direction is None
        assert verdict.log_likelihood_ratio == 0.0


# ---------------------------------------------------------------------------
# MsprtVerdict: the direction field
# ---------------------------------------------------------------------------


def _verdict(**overrides):
    from soup_cli.utils.ab_test import MsprtVerdict

    fields = {
        "decision": "reject_h0",
        "log_likelihood_ratio": 5.0,
        "n_control": 10,
        "n_treatment": 10,
        "mean_control": 1.0,
        "mean_treatment": 1.5,
        "direction": "worse",
    }
    fields.update(overrides)
    return MsprtVerdict(**fields)


class TestVerdictDirectionField:
    @pytest.mark.parametrize("direction", ["better", "worse"])
    def test_reject_h0_accepts_a_direction(self, direction):
        assert _verdict(direction=direction).direction == direction

    @pytest.mark.parametrize("decision", ["accept_h0", "continue"])
    def test_undecided_verdicts_default_to_no_direction(self, decision):
        from soup_cli.utils.ab_test import MsprtVerdict

        verdict = MsprtVerdict(
            decision=decision,
            log_likelihood_ratio=0.0,
            n_control=3,
            n_treatment=3,
            mean_control=1.0,
            mean_treatment=1.0,
        )
        assert verdict.direction is None

    @pytest.mark.parametrize("bad", ["sideways", "BETTER", "", True, 1])
    def test_unknown_direction_is_refused(self, bad):
        with pytest.raises(ValueError, match="direction must be one of"):
            _verdict(direction=bad)

    @pytest.mark.parametrize("decision", ["accept_h0", "continue"])
    def test_direction_without_a_rejection_is_refused(self, decision):
        with pytest.raises(ValueError, match="only a reject_h0 verdict has a direction"):
            _verdict(decision=decision, direction="better")

    def test_rejection_without_a_direction_is_refused(self):
        with pytest.raises(ValueError, match="reject_h0 verdict must carry a direction"):
            _verdict(direction=None)


# ---------------------------------------------------------------------------
# CLI: panel, table and webhook payload carry the direction
# ---------------------------------------------------------------------------


def _write_ab(path, metric, control, treatment):
    rows = [{"arm": "control", metric: x} for x in control]
    rows += [{"arm": "treatment", metric: round(x, 6)} for x in treatment]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def _run_cli(tmp_path, monkeypatch, metric, control, treatment, extra=()):
    from soup_cli.cli import app
    from soup_cli.utils import webhooks

    captured = []
    monkeypatch.setattr(webhooks, "post_webhook", lambda **kw: (captured.append(kw), True)[1])
    monkeypatch.chdir(tmp_path)
    _write_ab(tmp_path / "ab.jsonl", metric, control, treatment)
    result = runner.invoke(
        app,
        [
            "ab",
            "--input",
            "ab.jsonl",
            "--metric",
            metric,
            "--slack-url",
            "https://hooks.slack.com/x",
            *extra,
        ],
    )
    return result, captured


class TestCli:
    def test_regression_file_reports_worse_and_rollback(self, tmp_path, monkeypatch):
        result, captured = _run_cli(
            tmp_path,
            monkeypatch,
            "judge_score",
            _JUDGE_CONTROL,
            _shifted(_JUDGE_CONTROL, -0.30),
        )
        # `soup ab` promises no gating exit code (docs/evaluation.md); a detected
        # regression is reported, not turned into a failure status.
        assert result.exit_code == 0, (result.output, repr(result.exception))
        out = _plain(result.output)
        assert "reject_h0" in out
        assert "direction worse" in out
        assert "prior_scale first 5 rows per arm" in out
        assert "rollback" in out.lower()
        assert "No significant difference" not in out
        assert "promote" not in out.lower()
        assert "Warning" not in out

        assert len(captured) == 1
        payload = captured[0]["payload"]
        assert payload["decision"] == "reject_h0"
        assert payload["direction"] == "worse"
        assert payload["mean_treatment"] == pytest.approx(0.5)

    def test_improvement_file_reports_better_without_rollback(self, tmp_path, monkeypatch):
        result, captured = _run_cli(
            tmp_path,
            monkeypatch,
            "judge_score",
            _JUDGE_CONTROL,
            _shifted(_JUDGE_CONTROL, 0.30),
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        out = _plain(result.output)
        assert "direction better" in out
        assert "promote" in out.lower()
        assert "rollback" not in out.lower()
        assert captured[0]["payload"]["direction"] == "better"

    @pytest.mark.parametrize(("shift", "expected"), [(-0.30, "better"), (+0.30, "worse")])
    def test_latency_polarity_in_output_and_payload(self, tmp_path, monkeypatch, shift, expected):
        result, captured = _run_cli(
            tmp_path,
            monkeypatch,
            "latency",
            _LATENCY_CONTROL,
            _shifted(_LATENCY_CONTROL, shift),
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        out = _plain(result.output)
        assert f"direction {expected}" in out
        assert "lower is better" in out
        assert ("rollback" in out.lower()) is (expected == "worse")
        assert captured[0]["payload"]["direction"] == expected

    def test_accept_h0_payload_has_no_direction(self, tmp_path, monkeypatch):
        result, captured = _run_cli(
            tmp_path,
            monkeypatch,
            "judge_score",
            _JUDGE_CONTROL,
            list(_JUDGE_CONTROL),
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        out = _plain(result.output)
        assert "accept_h0" in out
        assert "direction n/a" in out
        assert "No significant difference" in out
        assert "rollback" not in out.lower()
        payload = captured[0]["payload"]
        assert payload["decision"] == "accept_h0"
        assert "direction" in payload
        assert payload["direction"] is None

    def test_too_few_rows_file_says_why_there_is_no_verdict(self, tmp_path, monkeypatch):
        control = _JUDGE_CONTROL[:6]  # the 5 prior-scale rows and 1 more: 2 are needed
        result, captured = _run_cli(
            tmp_path,
            monkeypatch,
            "judge_score",
            control,
            _shifted(control, -0.30),
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        out = _plain(result.output)
        assert "decision continue" in out
        assert "direction n/a" in out
        assert "Not enough rows yet: the first 5 rows of each arm set the scale" in out
        assert "control has 6, treatment 6" in out
        assert "rollback" not in out.lower()
        assert captured == []  # a `continue` never pages anyone
