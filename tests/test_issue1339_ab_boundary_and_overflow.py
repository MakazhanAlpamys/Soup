"""#1339 — `soup ab` refuses `alpha + beta >= 1` and an overflowing standardised effect.

Two gaps listed as pre-existing in #1310's description and left out of it.

**The boundaries cross.** ``MsprtConfig`` checked ``alpha`` and ``beta`` in
(0, 1) one at a time. The reject boundary ``log((1 - beta) / alpha)`` stays above
the accept boundary ``log(beta / (1 - alpha))`` only while ``alpha + beta < 1``;
past that there is no ``continue`` band, and since reject is tested first, the
gap between them rejects. ``--alpha 0.6 --beta 0.5`` rejected on two identical
arms, and ``--alpha 0.05 --beta 0.95`` (a power of 0.95 typed as beta) put both
boundaries at 0 and rejected a true H0 in 255 of 1000 single looks.

Since #1418 ``beta`` is retired: the accept rule is a confidence sequence at
level ``alpha`` and the reject boundary is ``log(1 / alpha) > 0``, so there is
no pair of boundaries left to cross, and the check went with ``beta``. A passed
``beta`` is ignored with a warning. The symptom stays pinned below: two
identical arms never reject, at any ``alpha`` in the old example.

**The statistic overflows.** ``drift = 0.5 * mu_h1**2 * n_ratio`` raises
``OverflowError`` rather than returning ``inf``, and ``commands/ab.py`` catches
only ``FileNotFoundError`` / ``TypeError`` / ``ValueError``, so the run ended on
a bare ``Error: OverflowError: (34, 'Result too large')`` that named no flag. It
is the ratio ``effect_size / pooled_se`` that overflows, so a near-constant
column reaches it at the default ``--effect-size`` too.
"""

from __future__ import annotations

import math
import sys

import pytest
from typer.testing import CliRunner

from soup_cli.utils.ab_test import MsprtConfig, msprt_step
from tests.test_issue1227_ab_two_sided import _plain, _write_ab

runner = CliRunner()

# Rows around 0.80 with real noise, 40 per arm — past the 30-row burn-in at
# alpha 0.05 and the 40-row one below it, so a verdict is actually reached.
_H0_CONTROL = [0.80, 0.82, 0.78, 0.81, 0.79] * 8

# The same, spread wider (SD 0.095): at alpha 0.6 / beta 0.5 and
# --effect-size 0.01 these land the LLR of two IDENTICAL arms at -0.129, inside
# the inverted band (reject at -0.182, accept at +0.223). Measured on main:
# "decision reject_h0 / direction worse / mean_control 0.8000 /
# mean_treatment 0.8000", exit 0.
_WIDE_CONTROL = [0.68, 0.92, 0.74, 0.86, 0.80] * 8


def _run_cli(tmp_path, monkeypatch, control, treatment, extra=()):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    _write_ab(tmp_path / "ab.jsonl", "judge_score", control, treatment)
    return runner.invoke(
        app, ["ab", "--input", "ab.jsonl", "--metric", "judge_score", *extra]
    )


class TestAlphaPlusBeta:
    """Retired with beta in #1418: the pairs once refused load, beta unused."""

    @pytest.mark.parametrize(
        ("alpha", "beta"),
        [
            (0.6, 0.5),  # reject at -0.182, accept at +0.223: the gap rejected
            (0.05, 0.95),  # a power of 0.95 typed as beta: both boundaries at 0
            (0.5, 0.5),  # exactly 1.0, the degenerate edge
        ],
    )
    def test_a_once_crossed_pair_loads_with_beta_ignored(self, alpha, beta):
        from soup_cli.config.deprecation import SoupConfigDeprecationWarning

        with pytest.warns(SoupConfigDeprecationWarning, match="beta no longer does anything"):
            cfg = MsprtConfig(metric="judge_score", alpha=alpha, beta=beta)
        assert cfg.alpha == alpha

    @pytest.mark.parametrize("alpha", [0.6, 0.5, 0.05])
    def test_identical_arms_never_reject(self, alpha):
        """The reported symptom, now unreachable by construction: on main before
        #1339, alpha 0.6 / beta 0.5 returned ``reject_h0`` with ``direction worse``
        on two byte-identical arms. A Bayes factor at t = 0 is below 1, and the
        reject boundary ``1 / alpha`` is above it."""
        config = MsprtConfig(metric="judge_score", alpha=alpha, effect_size=0.01)
        for n in range(7, len(_WIDE_CONTROL) + 1):
            verdict = msprt_step(
                config, control=_WIDE_CONTROL[:n], treatment=_WIDE_CONTROL[:n]
            )
            assert verdict.decision != "reject_h0", (alpha, n)

    def test_cli_warns_on_beta_and_still_decides(self, tmp_path, monkeypatch):
        result = _run_cli(
            tmp_path, monkeypatch, _H0_CONTROL, _H0_CONTROL,
            extra=["--alpha", "0.05", "--beta", "0.95"],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        out = _plain(result.output)
        assert "--beta no longer does anything" in out
        assert "decision reject_h0" not in out

    def test_cli_still_runs_the_documented_pair(self, tmp_path, monkeypatch):
        result = _run_cli(
            tmp_path, monkeypatch, _H0_CONTROL, _H0_CONTROL,
            extra=["--alpha", "0.01", "--beta", "0.10"],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))

    def test_beta_help_says_it_is_retired(self):
        from soup_cli.cli import app

        result = runner.invoke(app, ["ab", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        out = _plain(result.output)
        assert "Retired (#1418)" in out


class TestStandardisedEffectOverflow:
    def test_bound_is_the_largest_squarable_float(self):
        # Imported here, not at module scope, so the rest of the file still
        # collects against a tree without the constant.
        from soup_cli.utils.ab_test import MAX_STANDARDISED_EFFECT

        assert MAX_STANDARDISED_EFFECT == math.sqrt(sys.float_info.max)
        # The premise: ** raises where * returns inf, which is why the bound is
        # checked instead of the result.
        with pytest.raises(OverflowError):
            (MAX_STANDARDISED_EFFECT * 2.0) ** 2

    @pytest.mark.parametrize("rows_per_arm", [2, 40])
    def test_cli_exits_1_naming_effect_size(self, tmp_path, monkeypatch, rows_per_arm):
        """At 2 rows per arm too: the statistic runs before the burn-in check."""
        control = _H0_CONTROL[:rows_per_arm]
        result = _run_cli(
            tmp_path, monkeypatch, control, control,
            extra=["--effect-size", "1e200"],
        )
        assert result.exit_code == 1, (result.output, repr(result.exception))
        out = _plain(result.output)
        assert "--effect-size" in out
        assert "OverflowError" not in out

    def test_near_constant_column_overflows_at_the_default_effect_size(self):
        """It is the ratio, not the flag: values 0 and 1e-160 reach it at 0.1."""
        near_constant = [0.0, 1e-160] * 20
        with pytest.raises(ValueError) as exc:
            msprt_step(
                MsprtConfig(metric="judge_score"),
                control=near_constant,
                treatment=near_constant,
            )
        message = str(exc.value)
        assert "--effect-size" in message
        assert "judge_score" in message

    def test_a_workable_effect_size_still_decides(self):
        verdict = msprt_step(
            MsprtConfig(metric="judge_score", effect_size=0.03),
            control=_H0_CONTROL,
            treatment=_H0_CONTROL,
        )
        assert verdict.decision in {"continue", "accept_h0", "reject_h0"}
        assert math.isfinite(verdict.log_likelihood_ratio)
