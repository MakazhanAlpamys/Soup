"""#1227 — `soup ab` runs a two-sided mSPRT and reports the direction.

The statistic was Wald's SPRT for the single point alternative
``mean_t - mean_c = +effect_size``, evaluated on the signed z. A treatment that
moved the other way drove the log-likelihood ratio towards minus infinity, i.e.
across the ACCEPT boundary: the larger the regression, the more confidently the
command said "No significant difference". On ``latency`` and ``retry_rate``
(lower is better) that made a real improvement look like no difference too.

The fix is the symmetric two-point mixture: average the likelihood RATIOS for
``+effect_size`` and ``-effect_size`` (in log space, with logsumexp). Each point
ratio is a martingale under H0 and so is their average, which is why the
``log((1 - beta) / alpha)`` boundary keeps its meaning. Averaging the
log-ratios, or substituting ``|z|``, does not have that property; the peeking
simulations below are what catch the ``|z|`` shortcut.
"""

from __future__ import annotations

import json
import math
import re
import statistics

import pytest
from typer.testing import CliRunner

runner = CliRunner()

# Rich colours per character on Linux CI, and wraps; compare on plain text.
# Box-drawing characters go too, so a table row reads "direction worse" and a
# panel sentence wrapped across two lines reads as one sentence.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_BOX_RE = re.compile(r"[─-╿|]")


def _plain(text: str) -> str:
    return " ".join(_BOX_RE.sub(" ", _ANSI_RE.sub("", text)).split())


# The reproduction data from the issue: twenty control rows around 0.80.
_JUDGE_CONTROL = [0.80, 0.82, 0.78, 0.81, 0.79] * 4
_LATENCY_CONTROL = [1.20, 1.25, 1.18, 1.22, 1.21] * 4
_RETRY_CONTROL = [0.20, 0.22, 0.18, 0.21, 0.19] * 4


def _shifted(values, shift):
    return [x + shift for x in values]


def _step(metric, control, treatment, **kwargs):
    from soup_cli.utils.ab_test import MsprtConfig, msprt_step

    return msprt_step(MsprtConfig(metric=metric, **kwargs), control=control, treatment=treatment)


def _upper(alpha=0.05, beta=0.20):
    return math.log((1.0 - beta) / alpha)


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
# The statistic itself: log of the AVERAGE of the two likelihood ratios
# ---------------------------------------------------------------------------


def _documented_point_llrs(control, treatment, effect_size):
    """Wald's point-alternative log-ratios for +effect_size and -effect_size.

    Written from the formula in the module docstring, not from the code.
    """
    n_c, n_t = len(control), len(treatment)
    pooled = ((n_c - 1) * statistics.variance(control)
              + (n_t - 1) * statistics.variance(treatment)) / (n_c + n_t - 2)
    se = math.sqrt(pooled * (1.0 / n_c + 1.0 / n_t))
    z = (statistics.fmean(treatment) - statistics.fmean(control)) / se
    mu = effect_size / se
    n_eff = n_c * n_t / (n_c + n_t)
    ratio = n_eff / (n_eff + 1.0)
    shift = z * mu * math.sqrt(ratio)
    drift = 0.5 * mu**2 * ratio
    return shift - drift, -shift - drift


class TestStatistic:
    # Moderate evidence, so exp() of either point log-ratio is representable.
    _CONTROL = [1.00, 1.30, 0.80, 1.10, 0.90, 1.20, 0.95, 1.05]
    _TREATMENT = [1.10, 1.45, 0.85, 1.25, 1.00, 1.30, 1.00, 1.20]

    def test_llr_is_log_of_the_average_likelihood_ratio(self):
        plus, minus = _documented_point_llrs(self._CONTROL, self._TREATMENT, 0.1)
        expected = math.log((math.exp(plus) + math.exp(minus)) / 2.0)
        # Discriminates the tempting wrong answers at this operating point:
        assert abs(expected - max(plus, minus)) > 0.1  # |z| / max of the two
        assert abs(expected - (plus + minus) / 2.0) > 0.1  # mean of the log-ratios
        assert abs(expected - plus) > 0.1  # the old one-sided statistic

        verdict = _step("judge_score", self._CONTROL, self._TREATMENT, effect_size=0.1)
        assert verdict.log_likelihood_ratio == pytest.approx(expected, rel=1e-9, abs=1e-12)

    def test_huge_evidence_stays_finite(self):
        """exp(1142) overflows; the average must be taken in log space."""
        verdict = _step("judge_score", _JUDGE_CONTROL, _shifted(_JUDGE_CONTROL, -0.30))
        plus, minus = _documented_point_llrs(
            _JUDGE_CONTROL, _shifted(_JUDGE_CONTROL, -0.30), 0.1
        )
        assert max(plus, minus) > 710  # math.exp overflows above ~709.78
        assert verdict.log_likelihood_ratio == pytest.approx(
            max(plus, minus) - math.log(2.0), rel=1e-9
        )


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


def _run_cli(tmp_path, monkeypatch, metric, control, treatment):
    from soup_cli.cli import app
    from soup_cli.utils import webhooks

    captured = []
    monkeypatch.setattr(
        webhooks, "post_webhook", lambda **kw: (captured.append(kw), True)[1]
    )
    monkeypatch.chdir(tmp_path)
    _write_ab(tmp_path / "ab.jsonl", metric, control, treatment)
    result = runner.invoke(
        app,
        ["ab", "--input", "ab.jsonl", "--metric", metric,
         "--slack-url", "https://hooks.slack.com/x"],
    )
    return result, captured


class TestCli:
    def test_regression_file_reports_worse_and_rollback(self, tmp_path, monkeypatch):
        result, captured = _run_cli(
            tmp_path, monkeypatch, "judge_score",
            _JUDGE_CONTROL, _shifted(_JUDGE_CONTROL, -0.30),
        )
        # `soup ab` promises no gating exit code (docs/evaluation.md); a detected
        # regression is reported, not turned into a failure status.
        assert result.exit_code == 0, (result.output, repr(result.exception))
        out = _plain(result.output)
        assert "reject_h0" in out
        assert "direction worse" in out
        assert "rollback" in out.lower()
        assert "No significant difference" not in out
        assert "promote" not in out.lower()

        assert len(captured) == 1
        payload = captured[0]["payload"]
        assert payload["decision"] == "reject_h0"
        assert payload["direction"] == "worse"
        assert payload["mean_treatment"] == pytest.approx(0.5)

    def test_improvement_file_reports_better_without_rollback(self, tmp_path, monkeypatch):
        result, captured = _run_cli(
            tmp_path, monkeypatch, "judge_score",
            _JUDGE_CONTROL, _shifted(_JUDGE_CONTROL, 0.30),
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        out = _plain(result.output)
        assert "direction better" in out
        assert "promote" in out.lower()
        assert "rollback" not in out.lower()
        assert captured[0]["payload"]["direction"] == "better"

    @pytest.mark.parametrize(
        ("shift", "expected"), [(-0.30, "better"), (+0.30, "worse")]
    )
    def test_latency_polarity_in_output_and_payload(self, tmp_path, monkeypatch, shift,
                                                    expected):
        result, captured = _run_cli(
            tmp_path, monkeypatch, "latency",
            _LATENCY_CONTROL, _shifted(_LATENCY_CONTROL, shift),
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        out = _plain(result.output)
        assert f"direction {expected}" in out
        assert "lower is better" in out
        assert ("rollback" in out.lower()) is (expected == "worse")
        assert captured[0]["payload"]["direction"] == expected

    def test_accept_h0_payload_has_no_direction(self, tmp_path, monkeypatch):
        result, captured = _run_cli(
            tmp_path, monkeypatch, "judge_score", _JUDGE_CONTROL, list(_JUDGE_CONTROL),
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


# ---------------------------------------------------------------------------
# Type-I error under peeking (seeded Monte-Carlo, vectorised with numpy)
# ---------------------------------------------------------------------------
#
# The operator re-runs `soup ab` after every new (control, treatment) pair,
# n = 2..200 rows per arm, and stops at the first terminal verdict. Calling
# `msprt_step` on the raw rows at every peek is O(n) per call (about 45 s for
# a few thousand runs), so the rows' running means and Bessel-corrected
# variances are computed with numpy and handed to `_verdict_from_summary`,
# the decision code `msprt_step` itself delegates to. The anchor at the end of
# the first simulation proves those are `msprt_step`'s verdicts.

_REPS = 3000
_ALPHA = 0.05
# Monte-Carlo tolerance: a one-sided 3-sigma binomial allowance on the rate,
# 3 * sqrt(0.05 * 0.95 / 3000) = 0.0119, so the bound is 0.0619.
_MC_TOLERANCE = 3.0 * math.sqrt(_ALPHA * (1.0 - _ALPHA) / _REPS)


def _peek_until_decided(np, config, *, sigma, reps, seed, n_max=200, known_variance=False):
    """Peek after every pair; return the terminal verdict (or None) per run.

    Also returns the raw rows and the running summaries that were fed to the
    decision code, so a caller can check them against ``msprt_step``.
    """
    from soup_cli.utils.ab_test import _verdict_from_summary

    rng = np.random.default_rng(seed)
    # H0: both arms from the same distribution. The statistic is
    # location-invariant, so the common mean is 0 (no cancellation in the
    # one-pass variance below).
    control = rng.normal(0.0, sigma, size=(reps, n_max))
    treatment = rng.normal(0.0, sigma, size=(reps, n_max))
    rows = np.arange(1, n_max + 1, dtype=float)
    mean_c = np.cumsum(control, axis=1) / rows
    mean_t = np.cumsum(treatment, axis=1) / rows
    if known_variance:
        pooled = np.full_like(mean_c, sigma * sigma)
    else:
        dof = np.maximum(rows - 1.0, 1.0)
        var_c = (np.cumsum(control * control, axis=1) - rows * mean_c * mean_c) / dof
        var_t = (np.cumsum(treatment * treatment, axis=1) - rows * mean_t * mean_t) / dof
        pooled = (var_c + var_t) / 2.0  # equal arms: the Bessel-pooled variance

    mean_c_rows, mean_t_rows, pooled_rows = mean_c.tolist(), mean_t.tolist(), pooled.tolist()
    outcomes = []
    for row_c, row_t, row_v in zip(mean_c_rows, mean_t_rows, pooled_rows):
        verdict = None
        for idx in range(1, n_max):  # idx + 1 = 2..n_max rows per arm
            candidate = _verdict_from_summary(
                config,
                n_control=idx + 1,
                n_treatment=idx + 1,
                mean_control=row_c[idx],
                mean_treatment=row_t[idx],
                pooled_variance=row_v[idx],
            )
            if candidate.decision != "continue":
                verdict = candidate
                break
        outcomes.append(verdict)
    return {
        "outcomes": outcomes,
        "control": control,
        "treatment": treatment,
        "summaries": (mean_c_rows, mean_t_rows, pooled_rows),
    }


def _rejection_rate(outcomes):
    return sum(1 for v in outcomes if v is not None and v.decision == "reject_h0") / len(outcomes)


class TestTypeOneErrorUnderPeeking:
    def test_rejects_at_most_alpha_when_the_effect_is_small_against_noise(self):
        """Acceptance: H0, a peek after every pair, rejection rate <= alpha.

        effect_size / sigma = 0.18: the effect you look for is small next to the
        row-to-row noise, the usual A/B regime. Measured over 40,000 runs of this
        statistic: 0.043 for the mixture, 0.083 for a |z| substitution, 0.041 for
        the old one-sided test. 3000 runs put the bound at 0.0619.
        """
        np = pytest.importorskip("numpy")
        from soup_cli.utils.ab_test import MsprtConfig, _verdict_from_summary, msprt_step

        config = MsprtConfig(metric="latency", alpha=_ALPHA, beta=0.20, effect_size=0.1)
        sim = _peek_until_decided(np, config, sigma=0.1 / 0.18, reps=_REPS, seed=1227)
        outcomes = sim["outcomes"]
        rate = _rejection_rate(outcomes)
        assert rate <= _ALPHA + _MC_TOLERANCE, (
            f"Type-I rate {rate:.4f} under peeking exceeds alpha {_ALPHA} "
            f"+ tolerance {_MC_TOLERANCE:.4f}"
        )

        # Two-sided under H0: false rejections fall on both sides.
        directions = [v.direction for v in outcomes if v is not None and v.decision == "reject_h0"]
        assert len(directions) >= 50, directions
        for side in ("better", "worse"):
            share = directions.count(side) / len(directions)
            assert 0.25 <= share <= 0.75, (side, share, len(directions))

        # Anchor: the summaries the simulation decided on give msprt_step's own
        # verdict on the raw rows.
        mean_c_rows, mean_t_rows, pooled_rows = sim["summaries"]
        rng = np.random.default_rng(7)
        for run, idx in zip(rng.integers(0, _REPS, 150).tolist(),
                            rng.integers(1, 200, 150).tolist()):
            from_rows = msprt_step(
                config,
                control=sim["control"][run, : idx + 1].tolist(),
                treatment=sim["treatment"][run, : idx + 1].tolist(),
            )
            from_summary = _verdict_from_summary(
                config,
                n_control=idx + 1,
                n_treatment=idx + 1,
                mean_control=mean_c_rows[run][idx],
                mean_treatment=mean_t_rows[run][idx],
                pooled_variance=pooled_rows[run][idx],
            )
            assert from_rows.decision == from_summary.decision, (run, idx)
            assert from_rows.direction == from_summary.direction, (run, idx)
            assert from_rows.log_likelihood_ratio == pytest.approx(
                from_summary.log_likelihood_ratio, rel=1e-9, abs=1e-9
            ), (run, idx)

    def test_mixture_keeps_the_wald_boundary_with_a_known_variance(self):
        """The martingale argument itself, with the variance estimate taken out.

        effect_size / sigma = 1.0. Measured over 40,000 runs with the true
        variance: 0.035 for the mixture, 0.071 for |z|. 6000 runs put the bound at
        0.05 + 3 * sqrt(0.05 * 0.95 / 6000) = 0.0584.
        """
        np = pytest.importorskip("numpy")
        from soup_cli.utils.ab_test import MsprtConfig

        reps = 6000
        tolerance = 3.0 * math.sqrt(_ALPHA * (1.0 - _ALPHA) / reps)
        config = MsprtConfig(metric="judge_score", alpha=_ALPHA, beta=0.20, effect_size=0.1)
        sim = _peek_until_decided(
            np, config, sigma=0.1, reps=reps, seed=2027, known_variance=True
        )
        rate = _rejection_rate(sim["outcomes"])
        assert rate <= _ALPHA + tolerance, (rate, _ALPHA + tolerance)

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=(
            "Pre-existing, not caused by #1227: the pooled variance is estimated from "
            "as few as 2 rows per arm, and with effect_size close to the row noise that "
            "estimate is too noisy for the likelihood ratio to stay a martingale. The "
            "old one-sided statistic measures 0.11 here; the two-sided one 0.16."
        ),
    )
    def test_few_rows_per_arm_inflate_type_one_error_when_effect_matches_noise(self):
        np = pytest.importorskip("numpy")
        from soup_cli.utils.ab_test import MsprtConfig

        reps = 2000
        tolerance = 3.0 * math.sqrt(_ALPHA * (1.0 - _ALPHA) / reps)
        config = MsprtConfig(metric="judge_score", alpha=_ALPHA, beta=0.20, effect_size=0.1)
        sim = _peek_until_decided(np, config, sigma=0.1, reps=reps, seed=2028)
        assert _rejection_rate(sim["outcomes"]) <= _ALPHA + tolerance
