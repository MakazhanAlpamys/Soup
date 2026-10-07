"""#1418 — `soup ab` accepts H0 with an always-valid confidence sequence.

Under #1265 `soup ab` accepted H0 once the normal-inverse-gamma Bayes factor fell
to ``log(beta / (1 - alpha))``. A mixture Bayes factor gathers evidence for H0
only like ``-0.5 * log(n_eff * g)``, so with no difference the test took 281 pairs
at 0.5 standard deviations and 889 at 0.2 to say so.

The reject rule is unchanged. The accept rule now reads the same mixture the
other way: shift the treatment by a candidate difference delta0 and the Bayes
factor for "delta0 is the difference" is a martingale when delta0 is the true
one, so the differences it has not rejected at ``1 / alpha`` form a confidence
sequence that covers the true difference at every look with probability at
least ``1 - alpha``. ``accept_h0`` comes once that sequence lies inside
(-effect_size, +effect_size), so a true difference of ``effect_size`` or more
ends in ``accept_h0`` in at most an ``alpha`` share of runs, however often the
operator looks. ``alpha`` bounds both wrong verdicts, and ``beta`` is retired:
passed, it is ignored with a warning. The sweep is
benchmarks/gate-1418-ab-cs-accept.md.
"""

from __future__ import annotations

import math

import pytest

from tests.test_issue1265_ab_nig import (
    _CONTROL,
    _TREATMENT,
    _log_bf,
    _plain,
    _run_cli,
    _simulate,
    _step,
)


def _half_width(n_control, n_treatment, pooled_variance, prior_variance, level):
    from soup_cli.utils.ab_test import _nig_confidence_half_width

    return _nig_confidence_half_width(
        n_control=n_control,
        n_treatment=n_treatment,
        pooled_variance=pooled_variance,
        prior_variance=prior_variance,
        level=level,
    )


def _tested_difference(control, treatment):
    from soup_cli.utils.ab_test import PRIOR_SCALE_ROWS

    rest_c, rest_t = control[PRIOR_SCALE_ROWS:], treatment[PRIOR_SCALE_ROWS:]
    return sum(rest_t) / len(rest_t) - sum(rest_c) / len(rest_c)


# ---------------------------------------------------------------------------
# The confidence sequence itself
# ---------------------------------------------------------------------------


class TestHalfWidth:
    @pytest.mark.parametrize(
        ("n_c", "n_t", "g", "level"),
        [(10, 10, 1.0, 0.2), (50, 30, 0.25, 0.05), (400, 400, 0.04, 0.2), (7, 12, 9.0, 0.01)],
    )
    def test_its_edge_is_where_the_bayes_factor_reaches_one_over_level(
        self, n_c, n_t, g, level
    ):
        """The closed form against `_nig_log_bayes_factor` at the edge's t."""
        from soup_cli.utils.ab_test import _nig_log_bayes_factor

        pooled = 0.3
        edge = _half_width(n_c, n_t, pooled, g, level)
        assert math.isfinite(edge)

        def log_bf(difference):
            return _nig_log_bayes_factor(
                n_control=n_c,
                n_treatment=n_t,
                mean_difference=difference,
                pooled_variance=pooled,
                prior_variance=g,
            )

        assert log_bf(edge) == pytest.approx(math.log(1.0 / level), rel=1e-9)
        assert log_bf(edge * (1 - 1e-6)) < math.log(1.0 / level) < log_bf(edge * (1 + 1e-6))

    def test_it_is_the_set_of_shifts_the_rows_do_not_reject(self):
        """Shifting the treatment rows themselves gives the same interval.

        This is the step the guarantee rests on: the Bayes factor of the shifted
        rows is the test of "delta0 is the difference", and only the rows, not
        the closed form, are used here.
        """
        control, treatment = _CONTROL, _TREATMENT
        n = len(control)
        mean_c, mean_t = sum(control) / n, sum(treatment) / n
        diff = mean_t - mean_c
        sum_sq = sum((x - mean_c) ** 2 for x in control) + sum(
            (x - mean_t) ** 2 for x in treatment
        )
        g, level = 2.0, 0.2
        edge = _half_width(n, n, sum_sq / (2 * n - 2), g, level)
        bound = math.log(1.0 / level)
        for side in (-1.0, 1.0):
            for scale, inside in ((1 - 1e-6, True), (1 + 1e-6, False), (0.5, True), (2.0, False)):
                delta0 = diff + side * scale * edge
                shifted = [x - delta0 for x in treatment]
                assert (_log_bf(control, shifted, g) < bound) is inside, (side, scale)

    def test_too_few_rows_for_the_prior_leave_every_difference_in(self):
        """2 rows per arm and a narrow prior: no t reaches 1 / level."""
        assert _half_width(2, 2, 0.3, 0.01, 0.2) == math.inf

    def test_a_lower_level_widens_it(self):
        widths = [_half_width(40, 40, 0.3, 1.0, level) for level in (0.5, 0.2, 0.05, 0.01)]
        assert widths == sorted(widths)
        assert widths[0] < widths[-1]

    def test_it_is_in_the_metric_s_units(self):
        """Scaling every row by k scales the pooled variance by k**2 and the width by k."""
        base = _half_width(30, 30, 0.3, 1.0, 0.2)
        assert _half_width(30, 30, 0.3 * 100.0, 1.0, 0.2) == pytest.approx(10.0 * base)

    def test_it_narrows_as_rows_come_in(self):
        widths = [_half_width(n, n, 0.3, 1.0, 0.2) for n in (10, 40, 160, 640)]
        assert widths == sorted(widths, reverse=True)


# ---------------------------------------------------------------------------
# The accept rule in msprt_step
# ---------------------------------------------------------------------------


class TestAcceptRule:
    def test_accept_needs_the_sequence_strictly_inside_effect_size(self, monkeypatch):
        from soup_cli.utils import ab_test

        effect_size = 0.1
        room = effect_size - abs(_tested_difference(_CONTROL, _TREATMENT))
        assert room > 0
        monkeypatch.setattr(ab_test, "_nig_log_bayes_factor", lambda **_: 0.0)
        for width, decision in ((room, "continue"), (room * (1 - 1e-9), "accept_h0")):
            monkeypatch.setattr(ab_test, "_nig_confidence_half_width", lambda **_: width)
            got = _step(_CONTROL, _TREATMENT, effect_size=effect_size)
            assert got.decision == decision, width

    @pytest.mark.parametrize("alpha", [0.05, 0.01, 0.2])
    def test_the_sequence_is_at_coverage_one_minus_alpha(self, alpha, monkeypatch):
        from soup_cli.utils import ab_test

        levels = []
        real = ab_test._nig_confidence_half_width

        def spy(**kwargs):
            levels.append(kwargs["level"])
            return real(**kwargs)

        monkeypatch.setattr(ab_test, "_nig_confidence_half_width", spy)
        monkeypatch.setattr(ab_test, "_nig_log_bayes_factor", lambda **_: 0.0)
        _step(_CONTROL, _TREATMENT, alpha=alpha)
        assert levels == [alpha]

    def test_a_reject_wins_when_both_rules_fire(self, monkeypatch):
        """A difference that is real but smaller than effect_size is reported as real."""
        from soup_cli.utils import ab_test

        monkeypatch.setattr(ab_test, "_nig_log_bayes_factor", lambda **_: math.log(1 / 0.05))
        monkeypatch.setattr(ab_test, "_nig_confidence_half_width", lambda **_: 0.0)
        got = _step(_CONTROL, _TREATMENT)
        assert got.decision == "reject_h0"
        assert got.direction == "worse"

    def test_the_old_bayes_factor_boundary_no_longer_accepts(self, monkeypatch):
        """A Bayes factor far below beta / (1 - alpha) is not, on its own, an accept."""
        from soup_cli.utils import ab_test

        monkeypatch.setattr(ab_test, "_nig_log_bayes_factor", lambda **_: -50.0)
        monkeypatch.setattr(ab_test, "_nig_confidence_half_width", lambda **_: math.inf)
        assert _step(_CONTROL, _TREATMENT).decision == "continue"

    def test_identical_arms_accept_once_the_sequence_is_narrow_enough(self):
        rows = [0.80, 0.83, 0.77, 0.81, 0.79, 0.84, 0.78, 0.82, 0.80, 0.76] * 3
        verdicts = [_step(rows[:n], rows[:n]) for n in range(7, len(rows) + 1)]
        decided = [v for v in verdicts if v.decision != "continue"]
        assert decided and decided[0].decision == "accept_h0", decided[:1]
        assert all(v.decision == "accept_h0" for v in decided)

    def test_a_difference_of_effect_size_is_not_accepted_from_these_rows(self):
        """The treatment 0.1 below control, effect_size 0.1: a reject, never an accept."""
        rows = [0.80, 0.83, 0.77, 0.81, 0.79, 0.84, 0.78, 0.82, 0.80, 0.76] * 3
        lower = [x - 0.1 for x in rows]
        verdicts = [_step(rows[:n], lower[:n]) for n in range(7, len(rows) + 1)]
        assert all(v.decision != "accept_h0" for v in verdicts)
        assert any(v.decision == "reject_h0" for v in verdicts)


# ---------------------------------------------------------------------------
# Under peeking (seeded Monte-Carlo, runs to 200 rows per arm)
# ---------------------------------------------------------------------------


def _accepts(outcomes):
    return [o for o in outcomes if o is not None and o[0] == "accept_h0"]


class TestUnderPeeking:
    @pytest.mark.parametrize(
        ("ratio", "alpha", "shift", "seed"),
        [
            (0.5, 0.20, 1.0, 4101),
            (0.5, 0.20, -1.0, 4102),
            (1.0, 0.05, 1.0, 4103),
            (2.0, 0.10, -1.0, 4104),
        ],
    )
    def test_a_difference_of_effect_size_is_accepted_in_at_most_alpha_of_runs(
        self, ratio, alpha, shift, seed
    ):
        """The full sweep (20,000 runs per ratio, to 1000 rows): gate-1418-ab-cs-accept.md."""
        np = pytest.importorskip("numpy")
        reps = 2000
        sim = _simulate(
            np, ratio=ratio, shift_in_effects=shift, reps=reps, seed=seed, alpha=alpha
        )
        rate = len(_accepts(sim["outcomes"])) / reps
        assert rate <= alpha + 3 * math.sqrt(alpha * (1 - alpha) / reps), rate

    def test_with_no_difference_accept_comes_far_sooner_than_under_1265(self):
        """1 standard deviation: 82 pairs on average under the old rule (1000-row runs)."""
        np = pytest.importorskip("numpy")
        reps = 1000
        sim = _simulate(np, ratio=1.0, shift_in_effects=0.0, reps=reps, seed=4105, alpha=0.05)
        accepted = _accepts(sim["outcomes"])
        assert len(accepted) >= 0.9 * reps
        assert sum(o[2] for o in accepted) / len(accepted) <= 40


# ---------------------------------------------------------------------------
# What the operator reads
# ---------------------------------------------------------------------------


class TestWording:
    def test_beta_help_says_it_is_retired(self):
        from soup_cli.cli import app
        from tests.test_issue1265_ab_nig import runner

        result = runner.invoke(app, ["ab", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        out = _plain(result.output)
        assert "Retired (#1418)" in out
        assert "accept_h0, each in at most this share of runs" in out

    def test_the_accept_panel_names_the_effect_size_and_confidence(self, tmp_path, monkeypatch):
        rows = [0.80, 0.83, 0.77, 0.81, 0.79, 0.84, 0.78, 0.82, 0.80, 0.76] * 3
        result = _run_cli(tmp_path, monkeypatch, rows, rows, ("--alpha", "0.1"))
        assert result.exit_code == 0, (result.output, repr(result.exception))
        out = _plain(result.output)
        assert "decision accept_h0" in out
        assert "smaller than --effect-size 0.1, at confidence 0.9" in out


# ---------------------------------------------------------------------------
# --beta is retired: ignored, and never silently
# ---------------------------------------------------------------------------

_ROWS = [0.80, 0.83, 0.77, 0.81, 0.79, 0.84, 0.78, 0.82, 0.80, 0.76] * 3
_LOWER = [x - 0.1 for x in _ROWS]


class TestRetiredBeta:
    @pytest.mark.parametrize("beta", [0.01, 0.2, 0.95])
    def test_a_passed_beta_warns_and_changes_no_verdict(self, beta):
        from soup_cli.config.deprecation import SoupConfigDeprecationWarning

        for control, treatment in ((_ROWS, _ROWS), (_ROWS, _LOWER), (_CONTROL, _TREATMENT)):
            for n in (7, 12, 20, len(control)):
                want = _step(control[:n], treatment[:n])
                with pytest.warns(SoupConfigDeprecationWarning) as record:
                    got = _step(control[:n], treatment[:n], beta=beta)
                assert got == want, (beta, n)
                message = str(record[0].message)
                assert message.startswith("beta no longer does anything (#1418)")
                assert "lower alpha" in message and "will refuse it" in message

    def test_no_beta_no_warning(self, recwarn):
        from soup_cli.utils.ab_test import MsprtConfig

        assert MsprtConfig(metric="judge_score").beta is None
        assert not [w for w in recwarn if "beta" in str(w.message)]

    def test_alpha_plus_beta_past_1_is_no_longer_refused(self):
        """#1339's check guarded the old accept boundary; with beta unused it went."""
        from soup_cli.config.deprecation import SoupConfigDeprecationWarning
        from soup_cli.utils.ab_test import MsprtConfig

        with pytest.warns(SoupConfigDeprecationWarning):
            MsprtConfig(metric="judge_score", alpha=0.05, beta=0.95)

    def test_the_cli_says_what_changed_and_points_at_alpha(self, tmp_path, monkeypatch, recwarn):
        without = _run_cli(tmp_path, monkeypatch, _ROWS, _ROWS, ())
        result = _run_cli(tmp_path, monkeypatch, _ROWS, _ROWS, ("--beta", "0.95"))
        # Said once, by the CLI: --beta is not passed on, so MsprtConfig does not
        # warn a second time.
        assert not [w for w in recwarn if "no longer does anything" in str(w.message)]
        assert result.exit_code == 0, (result.output, repr(result.exception))
        out = _plain(result.output)
        assert "Warning: --beta no longer does anything (#1418)" in out
        assert "at level --alpha" in out and "lower --alpha" in out
        assert "will refuse it" in out
        assert out.count("no longer does anything") == 1
        # And it is ignored: the verdict is the one without --beta.
        assert "decision accept_h0" in out
        assert "decision accept_h0" in _plain(without.output)

    def test_the_cli_without_beta_does_not_warn(self, tmp_path, monkeypatch):
        result = _run_cli(tmp_path, monkeypatch, _ROWS, _ROWS, ())
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "no longer does anything" not in _plain(result.output)
