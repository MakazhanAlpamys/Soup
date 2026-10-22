"""#1265 — `soup ab` decides with a variance-robust statistic and needs no burn-in.

The #1227 statistic plugged the estimated variance into a likelihood ratio built
for a known one. From a handful of rows that estimate is too noisy, so a burn-in
of 30 / 40 rows per arm held verdicts back, and at alpha 0.01 the Type-I rate
still sat near 0.0108, with no calibration below 0.01 or past 1000 rows.

The statistic is now the normal-inverse-gamma Bayes factor in its right-Haar
limit: flat on the common mean, 1/sigma on sigma, N(0, g) on the standardised
difference d = delta / sigma. It is a function of the two-sample t statistic
only, so under H0 it is a martingale whatever sigma is, and Ville's inequality
bounds the chance it ever reaches 1/alpha by alpha, at any horizon. The first
``PRIOR_SCALE_ROWS`` rows per arm are held out to put effect_size in standard
deviations: g = (effect_size / s0) ** 2, s0 their pooled SD. The sweep behind
it is benchmarks/gate-1265-ab-nig.md; the simulations here stop at 200 rows per
arm to stay CI-sized.
"""

from __future__ import annotations

import math
import re

import pytest
from typer.testing import CliRunner

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_BOX_RE = re.compile(f"[{chr(0x2500)}-{chr(0x257F)}|]")


def _plain(text: str) -> str:
    return " ".join(_BOX_RE.sub(" ", _ANSI_RE.sub("", text)).split())


def _step(control, treatment, metric="judge_score", **kwargs):
    from soup_cli.utils.ab_test import MsprtConfig, msprt_step

    return msprt_step(MsprtConfig(metric=metric, **kwargs), control=control, treatment=treatment)


def _log_bf(control, treatment, prior_variance):
    from soup_cli.utils.ab_test import _nig_log_bayes_factor

    n_c, n_t = len(control), len(treatment)
    mean_c, mean_t = sum(control) / n_c, sum(treatment) / n_t
    sum_sq = sum((x - mean_c) ** 2 for x in control) + sum((x - mean_t) ** 2 for x in treatment)
    return _nig_log_bayes_factor(
        n_control=n_c,
        n_treatment=n_t,
        mean_difference=mean_t - mean_c,
        pooled_variance=sum_sq / (n_c + n_t - 2),
        prior_variance=prior_variance,
    )


# ---------------------------------------------------------------------------
# The Bayes factor itself
# ---------------------------------------------------------------------------


def _log_bf_by_integration(np, control, treatment, g):
    """The Bayes factor from the likelihoods, by quadrature, without the t form.

    With a flat prior the common mean integrates out exactly:
    m(delta, sigma) is proportional to sigma^(1 - N) * exp(-(SS + n_eff *
    (D - delta)^2) / (2 sigma^2)), D the difference of the means, SS the
    within-arm sum of squares. H0 sets delta = 0; H1 has delta = d * sigma with
    d ~ N(0, g). Both integrate over sigma with d(sigma) / sigma = d(log sigma).
    """
    control, treatment = np.asarray(control), np.asarray(treatment)
    n_c, n_t = len(control), len(treatment)
    n_total = n_c + n_t
    n_eff = n_c * n_t / n_total
    diff = treatment.mean() - control.mean()
    sum_sq = ((control - control.mean()) ** 2).sum() + ((treatment - treatment.mean()) ** 2).sum()
    log_sigma = np.linspace(math.log(1e-3), math.log(1e3), 4001)[:, None]
    sigma = np.exp(log_sigma)
    d = np.linspace(-10 * math.sqrt(g), 10 * math.sqrt(g), 2001)[None, :]

    def log_m(delta):
        return (1 - n_total) * log_sigma - (sum_sq + n_eff * (diff - delta) ** 2) / (2 * sigma**2)

    h0 = log_m(0.0)[:, 0]
    prior = np.exp(-d * d / (2 * g)) / math.sqrt(2 * math.pi * g)
    shift = h0.max()
    h0_mass = np.trapezoid(np.exp(h0 - shift), log_sigma[:, 0])
    h1_inner = np.trapezoid(np.exp(log_m(d * sigma) - shift) * prior, d[0], axis=1)
    h1_mass = np.trapezoid(h1_inner, log_sigma[:, 0])
    return math.log(h1_mass / h0_mass)


class TestBayesFactor:
    @pytest.mark.parametrize(
        ("control", "treatment", "g"),
        [
            ([1.0, 1.4], [0.7, 1.9], 1.0),  # 2 + 2 rows: nu = 2
            ([0.2, -0.4, 0.9], [1.3, 0.8], 0.5),  # unequal arms
            ([3.1, 2.7, 3.3, 2.9], [2.2, 2.6, 2.5, 2.1, 2.8], 4.0),  # clear difference
            ([0.5, 0.52, 0.49, 0.51, 0.5, 0.48], [0.5, 0.49, 0.51, 0.5, 0.52, 0.5], 0.05),
        ],
    )
    def test_closed_form_matches_the_integrated_likelihoods(self, control, treatment, g):
        np = pytest.importorskip("numpy")
        expected = _log_bf_by_integration(np, control, treatment, g)
        assert _log_bf(control, treatment, g) == pytest.approx(expected, abs=2e-4)

    def test_no_difference_gives_the_occam_penalty(self):
        """t = 0: log BF = -0.5 * log(1 + n_eff * g), whatever the spread."""
        control = [1.0, 2.0, 3.0, 4.0]
        treatment = [4.0, 1.0, 3.0, 2.0]
        assert _log_bf(control, treatment, 0.3) == pytest.approx(-0.5 * math.log(1 + 2 * 0.3))

    def test_statistic_is_scale_and_location_invariant(self):
        """What makes it a martingale under H0 for every sigma."""
        control = [0.80, 0.83, 0.77, 0.81, 0.79, 0.84]
        treatment = [0.78, 0.74, 0.79, 0.76, 0.80, 0.73]
        moved_c = [1000.0 * x - 55.0 for x in control]
        moved_t = [1000.0 * x - 55.0 for x in treatment]
        assert _log_bf(moved_c, moved_t, 0.7) == pytest.approx(
            _log_bf(control, treatment, 0.7), rel=1e-9
        )

    def test_statistic_is_symmetric_in_the_sign_of_the_difference(self):
        control = [0.80, 0.83, 0.77, 0.81, 0.79, 0.84]
        up = [x + 0.05 for x in control]
        down = [x - 0.05 for x in control]
        assert _log_bf(control, up, 1.0) == pytest.approx(_log_bf(control, down, 1.0))

    def test_huge_evidence_stays_finite(self):
        control = [0.80, 0.82, 0.78, 0.81, 0.79] * 200
        verdict = _step(control, [x - 0.30 for x in control])
        assert math.isfinite(verdict.log_likelihood_ratio)
        assert verdict.decision == "reject_h0"


# ---------------------------------------------------------------------------
# The held-out rows set the prior scale and nothing else
# ---------------------------------------------------------------------------


_CONTROL = [0.80, 0.83, 0.77, 0.81, 0.79, 0.84, 0.78, 0.82, 0.80, 0.76, 0.83, 0.79]
_TREATMENT = [0.78, 0.74, 0.79, 0.76, 0.80, 0.73, 0.77, 0.75, 0.79, 0.74, 0.76, 0.78]


def _sd(values_c, values_t):
    import statistics

    return math.sqrt((statistics.variance(values_c) + statistics.variance(values_t)) / 2)


class TestPriorScale:
    def test_the_first_five_rows_per_arm_are_held_out(self):
        """Chosen by the sweep in benchmarks/gate-1265-ab-nig.md; re-run it to change."""
        from soup_cli.utils.ab_test import PRIOR_SCALE_ROWS

        assert PRIOR_SCALE_ROWS == 5

    def test_the_docs_give_the_same_row_counts(self):
        """docs/evaluation.md states both counts in words; they follow the constant."""
        from pathlib import Path

        from soup_cli.utils.ab_test import PRIOR_SCALE_ROWS

        docs = (Path(__file__).parent.parent / "docs" / "evaluation.md").read_text(
            encoding="utf-8"
        )
        assert f"the first {PRIOR_SCALE_ROWS} rows of each arm" in docs
        assert f"from {PRIOR_SCALE_ROWS + 2} rows per arm" in docs

    @pytest.mark.parametrize("effect_size", [0.01, 0.1, 2.0])
    def test_statistic_runs_on_the_rows_after_them(self, effect_size):
        s0 = _sd(_CONTROL[:5], _TREATMENT[:5])
        expected = _log_bf(_CONTROL[5:], _TREATMENT[5:], (effect_size / s0) ** 2)
        verdict = _step(_CONTROL, _TREATMENT, effect_size=effect_size)
        assert verdict.log_likelihood_ratio == pytest.approx(expected, rel=1e-12)

    def test_held_out_values_enter_only_through_their_spread(self):
        """Moving the held-out rows (not their spread) leaves the statistic alone."""
        moved_c = [x + 7.0 for x in _CONTROL[:5]] + _CONTROL[5:]
        moved_t = [x - 3.0 for x in _TREATMENT[:5]] + _TREATMENT[5:]
        assert _step(moved_c, moved_t).log_likelihood_ratio == pytest.approx(
            _step(_CONTROL, _TREATMENT).log_likelihood_ratio, rel=1e-12
        )

    def test_whole_arm_means_are_reported(self):
        verdict = _step(_CONTROL, _TREATMENT)
        assert verdict.n_control == len(_CONTROL)
        assert verdict.mean_control == pytest.approx(sum(_CONTROL) / len(_CONTROL))
        assert verdict.mean_treatment == pytest.approx(sum(_TREATMENT) / len(_TREATMENT))

    @pytest.mark.parametrize(("n_control", "n_treatment"), [(6, 12), (12, 6), (6, 6)])
    def test_each_arm_needs_two_rows_after_the_held_out_ones(self, n_control, n_treatment):
        verdict = _step(_CONTROL[:n_control], [x - 0.3 for x in _TREATMENT[:n_treatment]])
        assert verdict.decision == "continue"
        assert verdict.log_likelihood_ratio == 0.0

    def test_seven_rows_per_arm_can_already_decide(self):
        """No burn-in: 5 held out + 2 tested rows per arm can reject a clear shift."""
        control = [0.80, 0.82, 0.78, 0.81, 0.79, 0.80, 0.81]
        verdict = _step(control, [x - 0.30 for x in control])
        assert verdict.decision == "reject_h0"
        assert verdict.direction == "worse"

    def test_constant_held_out_rows_extend_until_one_arm_varies(self):
        """A pooled SD of 0 would make the prior infinitely wide: hold out more rows."""
        control = [1.0] * 7 + [1.1, 0.9, 1.05, 0.95, 1.0, 1.1, 0.9]
        treatment = [1.0] * 8 + [1.3, 1.2, 1.25, 1.35, 1.3, 1.2]
        # control first varies at row 8, so 8 rows per arm are held out.
        s0 = _sd(control[:8], treatment[:8])
        expected = _log_bf(control[8:], treatment[8:], (0.1 / s0) ** 2)
        assert _step(control, treatment).log_likelihood_ratio == pytest.approx(expected)

    def test_never_varying_arms_give_no_verdict(self):
        verdict = _step([1.0] * 50, [2.0] * 50)
        assert verdict.decision == "continue"
        assert verdict.log_likelihood_ratio == 0.0

    def test_constant_tested_rows_give_no_verdict(self):
        """Varying held-out rows, but t is 0 / 0 on the rows after them."""
        verdict = _step([1.0, 1.1, 0.9, 1.0, 1.0] + [1.0] * 10, [1.0] * 5 + [2.0] * 10)
        assert verdict.decision == "continue"
        assert verdict.log_likelihood_ratio == 0.0


# ---------------------------------------------------------------------------
# Boundaries: reject at 1/alpha (Ville). The accept rule is #1418's confidence
# sequence, pinned in tests/test_issue1418_ab_cs_accept.py.
# ---------------------------------------------------------------------------


class TestBoundaries:
    def test_reject_boundary_is_one_over_alpha(self, monkeypatch):
        from soup_cli.utils import ab_test

        # No accept, so the only way out of continue is the reject boundary.
        monkeypatch.setattr(ab_test, "_nig_confidence_half_width", lambda **_: math.inf)
        for alpha in (0.05, 0.01, 0.001):
            edge = math.log(1.0 / alpha)
            for llr, decision in ((edge, "reject_h0"), (edge - 1e-9, "continue")):
                monkeypatch.setattr(ab_test, "_nig_log_bayes_factor", lambda **_: llr)
                got = _step(_CONTROL, _TREATMENT, alpha=alpha)
                assert got.decision == decision, (alpha, llr)

    def test_a_near_constant_held_out_column_is_refused_not_accepted(self):
        """#1339 in the NIG path: the scale comes from the held-out rows only.

        Rows 0 and 1e-160 put ``effect_size / s0`` past the float range while the
        tested rows vary normally. Without the guard, squaring that ratio raises a
        bare OverflowError; and ``effect_size**2 / s0**2``, the form before the
        guard, does not raise at all: the prior variance is inf, the Bayes factor
        -inf, and the verdict a silent ``accept_h0``.
        """
        held_out = [0.0, 1e-160, 0.0, 1e-160, 0.0]
        with pytest.raises(ValueError) as exc:
            _step(held_out + _CONTROL[5:], held_out + _TREATMENT[5:])
        message = str(exc.value)
        assert "--effect-size" in message
        assert "judge_score" in message

    def test_an_overflowing_effect_is_refused_before_the_prior_scale_is_complete(self):
        """Three rows per arm: the scale is checked on the rows so far."""
        with pytest.raises(ValueError, match="--effect-size"):
            _step(_CONTROL[:3], _TREATMENT[:3], effect_size=1e200)

    def test_too_few_rows_for_any_scale_still_continue(self):
        assert _step(_CONTROL[:1], _TREATMENT[:1], effect_size=1e200).decision == "continue"

    @pytest.mark.parametrize(("shift", "direction"), [(-0.1, "worse"), (0.1, "better")])
    def test_direction_comes_from_the_tested_rows(self, shift, direction):
        """Held-out rows pulling the whole-arm means the other way do not flip it."""
        control = [5.0, 5.02, 4.98, 5.01, 4.99] + _CONTROL[5:]
        treatment = [0.0, 0.02, -0.02, 0.01, -0.01] + [x + shift for x in _CONTROL[5:]]
        verdict = _step(control, treatment, effect_size=0.05)
        assert verdict.mean_treatment < verdict.mean_control  # whole arms: always lower
        assert verdict.decision == "reject_h0", verdict
        assert verdict.direction == direction


class TestNonFiniteStatistic:
    """Values a float cannot carry are refused, not decided on (#1419 review)."""

    def test_huge_means_are_refused_not_a_bare_overflow(self):
        control = [-1e154 * (1 + k * 1e-10) for k in range(12)]
        treatment = [1e154 * (1 + k * 1e-10) for k in range(12)]
        with pytest.raises(ValueError, match="judge_score"):
            _step(control, treatment)

    def test_near_constant_tested_rows_are_refused_not_a_nan_continue(self):
        held = [0.1, 0.2, 0.3, 0.4, 0.5]
        with pytest.raises(ValueError, match="judge_score"):
            _step(held + [0.0, 1e-160] * 10, held + [1e-5] * 20)

    def test_a_prior_scale_just_under_the_bound_is_never_minus_inf(self):
        held = [0.0, 1.83e-155, 0.0, 1.83e-155, 0.0]  # effect_size / s0 ~ 1.0e154
        rest = _CONTROL[5:]
        try:
            verdict = _step(held + rest, held + [x + 5.0 for x in rest])
        except ValueError as exc:
            assert "--effect-size" in str(exc) or "judge_score" in str(exc)
        else:
            assert math.isfinite(verdict.log_likelihood_ratio), verdict

    def test_held_out_rows_whose_variance_underflows_to_zero_are_refused(self):
        """They differ, so no more rows are held out, but their variance is 0.0 in floats."""
        held = [0.0, 1e-170, 0.0, 1e-170, 0.0]
        rest = _CONTROL[5:]
        with pytest.raises(ValueError, match="spread is below"):
            _step(held + rest, held + [x + 5.0 for x in rest])

    def test_one_arm_differing_is_enough_for_that_refusal(self):
        """The other arm's held-out rows all equal, as a warm cache gives."""
        rest = _CONTROL[5:]
        with pytest.raises(ValueError, match="spread is below"):
            _step([0.0, 1e-170, 0.0, 1e-170, 0.0] + rest, [0.0] * 5 + [x + 5.0 for x in rest])

    def test_the_treatment_arm_differing_is_enough_too(self):
        """The mirror of the control-arm case: control's held-out rows all equal."""
        rest = _CONTROL[5:]
        with pytest.raises(ValueError, match="spread is below"):
            _step([0.0] * 5 + rest, [0.0, 1e-170, 0.0, 1e-170, 0.0] + [x + 5.0 for x in rest])

    def test_the_round_one_f3_input_never_accepts_h0_at_any_peek(self):
        """The same rows, replayed one row at a time, the way soup ab is meant to be run."""
        from soup_cli.utils.ab_test import PRIOR_SCALE_ROWS

        held = [0.0, 1.83e-155, 0.0, 1.83e-155, 0.0]
        rest = _CONTROL[5:]
        control, treatment = held + rest, held + [x + 5.0 for x in rest]
        for n in range(PRIOR_SCALE_ROWS + 2, len(control) + 1):
            try:
                verdict = _step(control[:n], treatment[:n])
            except ValueError:
                continue
            assert verdict.decision != "accept_h0", (n, verdict)

    def test_an_oversized_flag_on_ordinary_rows_names_the_flag_not_the_spread(self):
        """The two --effect-size refusals are told apart by the held-out variance."""
        with pytest.raises(ValueError) as exc:
            _step(_CONTROL, _TREATMENT, effect_size=1e160)
        message = str(exc.value)
        assert "Lower --effect-size" in message
        assert "spread is below" not in message


@pytest.mark.parametrize(("factor", "refused"), [(0.99, False), (1.01, True)])
def test_the_prior_bound_sits_at_the_largest_n_eff_the_tool_accepts(factor, refused):
    """sqrt(float max / (1,000,000 / 2)) standard deviations, about 1.9e151, and no further."""
    import sys

    held = [0.0, 1.0, 0.0, 1.0, 0.0]  # pooled variance 0.3 in both arms
    bound = math.sqrt(sys.float_info.max / (1_000_000 / 2))
    effect_size = factor * bound * math.sqrt(0.3)
    control, treatment = held + _CONTROL[5:], held + _CONTROL[5:]
    if refused:
        with pytest.raises(ValueError, match="Lower --effect-size"):
            _step(control, treatment, effect_size=effect_size)
    else:
        _step(control, treatment, effect_size=effect_size)


# A saturated start: the first rows barely vary (a warm cache, a judge that gives
# 1.0 early), then ordinary rows. Seeded Gaussian rows, sd 0.03, rounded.
_SATURATED_C = [1.0, 1.0, 1.0, 1.0, 0.99]
_SATURATED_T = [1.0] * 5
_ORDINARY_C = [0.792, 0.815, 0.793, 0.791, 0.772, 0.794, 0.833, 0.813, 0.831, 0.807,
               0.812, 0.806, 0.75, 0.826, 0.815, 0.815, 0.749, 0.748, 0.773, 0.786]
_ORDINARY_T = [0.818, 0.782, 0.786, 0.762, 0.771, 0.784, 0.839, 0.739, 0.756, 0.807,
               0.843, 0.817, 0.743, 0.724, 0.811, 0.778, 0.766, 0.829, 0.833, 0.805]


def _peeks(control, treatment, **kwargs):
    """Every verdict from 7 rows per arm on, one row at a time."""
    return [_step(control[:n], treatment[:n], **kwargs) for n in range(7, len(control) + 1)]


class TestAcceptHold:
    """accept_h0 waits while the tested rows spread far more than the held-out ones."""

    @pytest.mark.parametrize("shift", [0.0, -0.03], ids=["no-difference", "worse"])
    def test_a_saturated_start_never_accepts_h0(self, shift, monkeypatch):
        from soup_cli.utils import ab_test

        control = _SATURATED_C + _ORDINARY_C[5:]
        treatment = _SATURATED_T + [x + shift for x in _ORDINARY_T[5:]]
        assert all(v.decision != "accept_h0" for v in _peeks(control, treatment))
        # The premise: without the hold, both accept within these rows. Under
        # #1418's accept rule that takes 12 and 19 rows per arm, not the first
        # peek, and both shifts are below the default --effect-size 0.1.
        monkeypatch.setattr(ab_test, "ACCEPT_HOLD_SPREAD_RATIO", math.inf)
        assert any(v.decision == "accept_h0" for v in _peeks(control, treatment))

    def test_the_same_tested_rows_with_ordinary_held_out_rows_still_accept(self):
        decided = [v for v in _peeks(_ORDINARY_C, _ORDINARY_T) if v.decision != "continue"]
        assert decided and decided[0].decision == "accept_h0", decided[:1]

    def test_a_reject_is_never_held_back(self):
        """A saturated start with a clear shift still rejects, with its direction."""
        control = _SATURATED_C + _ORDINARY_C[5:]
        treatment = _SATURATED_T + [x - 0.3 for x in _ORDINARY_T[5:]]
        verdicts = [v for v in _peeks(control, treatment) if v.decision != "continue"]
        assert verdicts and verdicts[0].decision == "reject_h0"
        assert verdicts[0].direction == "worse"

    def test_the_ratio_is_three(self):
        from soup_cli.utils.ab_test import ACCEPT_HOLD_SPREAD_RATIO

        assert ACCEPT_HOLD_SPREAD_RATIO == 3.0

    def test_gaussian_rows_rarely_change_and_rejects_never_do(self, monkeypatch):
        """600 seeded runs of Gaussian rows, first verdict with and without the hold."""
        import random

        from soup_cli.utils import ab_test

        def first(control, treatment, effect_size, ratio):
            monkeypatch.setattr(ab_test, "ACCEPT_HOLD_SPREAD_RATIO", ratio)
            for v in _peeks(control, treatment, effect_size=effect_size):
                if v.decision != "continue":
                    return v.decision, v.direction, v.n_control
            return None

        changed = 0
        for effect_size, shift in ((1.0, 0.0), (2.0, 0.0), (1.0, 1.0)):
            for seed in range(200):
                rng = random.Random(1_265_000 + seed)
                control = [rng.gauss(0.0, 1.0) for _ in range(60)]
                treatment = [rng.gauss(shift, 1.0) for _ in range(60)]
                held = first(control, treatment, effect_size, 3.0)
                free = first(control, treatment, effect_size, math.inf)
                if free is not None and free[0] == "reject_h0":
                    assert held == free, (effect_size, shift, seed)
                changed += held != free
        assert changed <= 6, changed  # 1% of 600; 2 on this seed range


# ---------------------------------------------------------------------------
# Type-I error and power under peeking (seeded Monte-Carlo)
# ---------------------------------------------------------------------------
#
# The operator re-runs `soup ab` after every new (control, treatment) pair and
# stops at the first terminal verdict; runs stop at 200 rows per arm. The
# statistic is vectorised with numpy (msprt_step on raw rows at every peek is
# O(n) per call); test_msprt_step_replays_the_simulated_runs ties it back to
# the shipped code run by run.

_N_MAX = 200


def _simulate(np, *, ratio, shift_in_effects, reps, seed, alpha, effect_size=0.1):
    """First verdict per run: (decision, direction, rows per arm), or None if undecided."""
    from soup_cli.utils.ab_test import PRIOR_SCALE_ROWS

    held = PRIOR_SCALE_ROWS
    rng = np.random.default_rng(seed)
    sigma = effect_size / ratio
    control = rng.normal(0.0, sigma, size=(reps, _N_MAX))
    treatment = rng.normal(shift_in_effects * effect_size, sigma, size=(reps, _N_MAX))
    s0_sq = (control[:, :held].var(axis=1, ddof=1) + treatment[:, :held].var(axis=1, ddof=1)) / 2
    g = (effect_size**2 / s0_sq)[:, None]
    rest_c, rest_t = control[:, held:], treatment[:, held:]
    rows = np.arange(1, rest_c.shape[1] + 1, dtype=float)
    mean_c = np.cumsum(rest_c, axis=1) / rows
    mean_t = np.cumsum(rest_t, axis=1) / rows
    sum_sq = (np.cumsum(rest_c**2, axis=1) - rows * mean_c**2) + (
        np.cumsum(rest_t**2, axis=1) - rows * mean_t**2
    )
    rows, mean_c, mean_t, sum_sq = rows[1:], mean_c[:, 1:], mean_t[:, 1:], sum_sq[:, 1:]
    nu, n_eff = 2 * rows - 2, rows / 2
    t_sq = (mean_t - mean_c) ** 2 * n_eff * nu / sum_sq
    spread = 1 + n_eff * g
    log_bf = -0.5 * np.log(spread) - 0.5 * (nu + 1) * (
        np.log1p(t_sq / (nu * spread)) - np.log1p(t_sq / nu)
    )
    upper = math.log(1 / alpha)
    # #1418: accept once the confidence sequence at coverage 1 - alpha lies inside
    # +-effect_size, unless the accept hold is on.
    c = np.exp(-(2 * math.log(1 / alpha) + np.log(spread)) / (nu + 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_edge = np.where(c > 1 / spread, nu * (1 - c) / (c - 1 / spread), np.inf)
    pooled = sum_sq / nu
    half_width = np.sqrt(t_edge * pooled / n_eff)
    hold = np.sqrt(pooled) > 3.0 * np.sqrt(s0_sq)[:, None]
    accept = (np.abs(mean_t - mean_c) + half_width < effect_size) & ~hold
    outcomes = []
    for run in range(reps):
        hits = np.flatnonzero((log_bf[run] >= upper) | accept[run])
        if hits.size == 0:
            outcomes.append(None)
            continue
        col = int(hits[0])
        if log_bf[run, col] >= upper:
            higher = mean_t[run, col] > mean_c[run, col]
            outcomes.append(("reject_h0", "better" if higher else "worse", held + col + 2))
        else:
            outcomes.append(("accept_h0", None, held + col + 2))
    return {"outcomes": outcomes, "control": control, "treatment": treatment}


def _rate(outcomes, decision="reject_h0"):
    return sum(1 for o in outcomes if o is not None and o[0] == decision) / len(outcomes)


class TestUnderPeeking:
    @pytest.mark.parametrize(
        ("ratio", "alpha", "seed"),
        [
            (0.35, 0.05, 3101),  # the burn-in design's worst ratio at alpha 0.05
            (1.0, 0.05, 3102),  # 0.164 at alpha 0.05 with the #1227 statistic and no burn-in
            (0.4, 0.01, 3103),  # the burn-in design's worst ratio at alpha 0.01
            (1.0, 0.005, 3104),  # below 0.01, where the burn-in was not calibrated
        ],
    )
    def test_type_one_error_stays_within_alpha_from_the_first_rows(self, ratio, alpha, seed):
        """H0, a peek after every pair from 7 rows per arm on, no burn-in.

        benchmarks/gate-1265-ab-nig.md records the full sweep (100,000 runs per
        ratio, horizons 200 and 1000). 3000 runs put the bound at alpha + 3 SE.
        """
        np = pytest.importorskip("numpy")
        reps = 3000
        sim = _simulate(np, ratio=ratio, shift_in_effects=0.0, reps=reps, seed=seed, alpha=alpha)
        rate = _rate(sim["outcomes"])
        bound = alpha + 3 * math.sqrt(alpha * (1 - alpha) / reps)
        assert rate <= bound, (rate, bound)

    def test_false_rejections_fall_on_both_sides(self):
        np = pytest.importorskip("numpy")
        sim = _simulate(np, ratio=0.35, shift_in_effects=0.0, reps=6000, seed=3105, alpha=0.10)
        sides = [o[1] for o in sim["outcomes"] if o is not None and o[0] == "reject_h0"]
        assert len(sides) >= 60, len(sides)
        assert 0.3 <= sides.count("better") / len(sides) <= 0.7

    def test_verdicts_come_well_before_the_old_burn_in(self):
        """The #1227 design gave nothing before 30 rows per arm; this one need not wait."""
        np = pytest.importorskip("numpy")
        sim = _simulate(np, ratio=2.0, shift_in_effects=1.0, reps=1000, seed=3106, alpha=0.05)
        decided = [o for o in sim["outcomes"] if o is not None]
        assert sum(o[2] < 30 for o in decided) >= 0.9 * len(sim["outcomes"])

    @pytest.mark.parametrize(("ratio", "min_power"), [(0.5, 0.75), (1.0, 0.95)])
    def test_power_against_the_asked_for_effect(self, ratio, min_power):
        """True difference = +effect_size: reject as `better` (judge_score) within 200 rows.

        Full numbers side by side with the burn-in design: benchmarks/gate-1265-ab-nig.md.
        """
        np = pytest.importorskip("numpy")
        sim = _simulate(np, ratio=ratio, shift_in_effects=1.0, reps=1000, seed=3107, alpha=0.05)
        right = sum(1 for o in sim["outcomes"] if o is not None and o[:2] == ("reject_h0",
                                                                               "better"))
        assert right / len(sim["outcomes"]) >= min_power

    def test_msprt_step_replays_the_simulated_runs(self):
        """Whole runs through the real `msprt_step` give the simulation's first verdicts."""
        np = pytest.importorskip("numpy")
        sim = _simulate(np, ratio=2.0, shift_in_effects=0.3, reps=300, seed=3108, alpha=0.05)
        picked = np.random.default_rng(4).choice(300, 30, replace=False).tolist()
        expected = [sim["outcomes"][run] for run in picked]
        assert sum(o is not None for o in expected) >= 25
        assert {o[0] for o in expected if o is not None} == {"reject_h0", "accept_h0"}
        for run, want in zip(picked, expected):
            replayed = None
            for rows in range(2, _N_MAX + 1):
                verdict = _step(
                    sim["control"][run, :rows].tolist(), sim["treatment"][run, :rows].tolist()
                )
                if verdict.decision != "continue":
                    replayed = (verdict.decision, verdict.direction, verdict.n_control)
                    break
            assert replayed == want, run


# ---------------------------------------------------------------------------
# CLI: the #1227 calibration warnings are gone with the burn-in
# ---------------------------------------------------------------------------


def _run_cli(tmp_path, monkeypatch, control, treatment, extra=()):
    import json

    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    rows = [{"arm": "control", "judge_score": x} for x in control]
    rows += [{"arm": "treatment", "judge_score": round(x, 6)} for x in treatment]
    (tmp_path / "ab.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return runner.invoke(app, ["ab", "--input", "ab.jsonl", "--metric", "judge_score", *extra])


class TestCli:
    @pytest.mark.parametrize(
        ("rows", "extra"),
        [(40, ("--alpha", "0.005")), (1001, ()), (12, ("--alpha", "0.001"))],
    )
    def test_no_calibration_warning_at_any_alpha_or_length(self, tmp_path, monkeypatch, rows,
                                                           extra):
        pattern = ([0.80, 0.82, 0.78, 0.81, 0.79] * 201)[:rows]
        result = _run_cli(tmp_path, monkeypatch, pattern, [x - 0.30 for x in pattern], extra)
        assert result.exit_code == 0, (result.output, repr(result.exception))
        out = _plain(result.output)
        assert "Warning" not in out
        assert "calibrated" not in out
        assert "burn" not in out.lower()
        assert "direction worse" in out
        assert "prior_scale first 5 rows per arm" in out

    def test_alpha_help_no_longer_describes_a_burn_in(self):
        from soup_cli.cli import app

        result = runner.invoke(app, ["ab", "--help"])
        out = _plain(result.output)
        assert "burn-in" not in out
        assert "re-run after every new row" in out
