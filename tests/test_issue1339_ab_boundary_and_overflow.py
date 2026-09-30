"""#1339 — `soup ab` refuses `alpha + beta >= 1` and an overflowing standardised effect.

Two gaps listed as pre-existing in #1310's description and left out of it.

**The boundaries cross.** ``MsprtConfig`` checked ``alpha`` and ``beta`` in
(0, 1) one at a time. The reject boundary ``log((1 - beta) / alpha)`` stays above
the accept boundary ``log(beta / (1 - alpha))`` only while ``alpha + beta < 1``;
past that there is no ``continue`` band, and since reject is tested first, the
gap between them rejects. ``--alpha 0.6 --beta 0.5`` rejected on two identical
arms, and ``--alpha 0.05 --beta 0.95`` (a power of 0.95 typed as beta) put both
boundaries at 0 and rejected a true H0 in 255 of 1000 single looks.

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
    @pytest.mark.parametrize(
        ("alpha", "beta"),
        [
            (0.6, 0.5),  # reject at -0.182, accept at +0.223: the gap rejects
            (0.05, 0.95),  # a power of 0.95 typed as beta: both boundaries at 0
            (0.5, 0.5),  # exactly 1.0, the degenerate edge
        ],
    )
    def test_config_refuses_a_crossed_pair(self, alpha, beta):
        with pytest.raises(ValueError) as exc:
            MsprtConfig(metric="judge_score", alpha=alpha, beta=beta)
        message = str(exc.value)
        assert "alpha + beta" in message
        # Both values named, so the operator sees which one to move.
        assert str(alpha) in message and str(beta) in message

    @pytest.mark.parametrize(
        ("alpha", "beta"),
        [
            (0.05, 0.20),  # the defaults
            (0.01, 0.10),  # the docs/evaluation.md example
            (0.6, 0.39),  # just below 1.0: still allowed
        ],
    )
    def test_config_accepts_a_usable_pair(self, alpha, beta):
        cfg = MsprtConfig(metric="judge_score", alpha=alpha, beta=beta)
        assert (cfg.alpha, cfg.beta) == (alpha, beta)
        # The band the refusal is about: reject stays strictly above accept.
        upper = math.log((1.0 - cfg.beta) / cfg.alpha)
        lower = math.log(cfg.beta / (1.0 - cfg.alpha))
        assert upper > lower

    def test_identical_arms_no_longer_reject_at_alpha_0_6_beta_0_5(self):
        """The reported symptom, now unreachable: refused before the step runs.

        On main this pair reaches ``msprt_step`` and returns ``reject_h0`` with
        ``direction worse`` on two byte-identical arms.
        """
        with pytest.raises(ValueError):
            msprt_step(
                MsprtConfig(
                    metric="judge_score", alpha=0.6, beta=0.5, effect_size=0.01
                ),
                control=_WIDE_CONTROL,
                treatment=_WIDE_CONTROL,
            )

    def test_cli_exits_2_naming_alpha_and_beta(self, tmp_path, monkeypatch):
        result = _run_cli(
            tmp_path, monkeypatch, _H0_CONTROL, _H0_CONTROL,
            extra=["--alpha", "0.05", "--beta", "0.95"],
        )
        assert result.exit_code == 2, (result.output, repr(result.exception))
        out = _plain(result.output)
        assert "alpha + beta" in out
        assert "0.95" in out

    def test_cli_still_runs_the_documented_pair(self, tmp_path, monkeypatch):
        result = _run_cli(
            tmp_path, monkeypatch, _H0_CONTROL, _H0_CONTROL,
            extra=["--alpha", "0.01", "--beta", "0.10"],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))

    def test_beta_help_states_the_bound(self):
        from soup_cli.cli import app

        result = runner.invoke(app, ["ab", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        out = _plain(result.output)
        assert "alpha + beta" in out


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
