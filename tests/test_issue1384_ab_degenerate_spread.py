"""#1384 — `soup ab` names a column whose spread a float cannot carry.

Two crashes found while reviewing #1378, on ``main`` and on #1378's head alike.

**Too little spread.** A metric column of ``0`` and ``1e-161`` has a positive
pooled variance (``2.5e-323``, a subnormal), but ``var * (1/n_c + 1/n_t)``
underflows to ``0.0`` at 40 rows per arm, and ``z = diff / pooled_se`` raised a
bare ``ZeroDivisionError``. The square root of the smallest positive float is
about ``2.2e-162``, so an underflow shows as a standard error of exactly ``0.0``.

**Too much spread.** A column of ``0`` and ``1e200``: ``(x - mean) ** 2`` raised
``OverflowError`` while the variance was being computed (``**`` raises where
``*`` returns ``inf``). An arm whose values sum past the largest float had an
infinite mean, and ``soup ab`` quietly reported ``continue`` with a ``nan``
log-likelihood ratio.

All three are now refused with a ``ValueError`` naming the column (the
overflowing sum with its own message, since its spread may be 0), which
``commands/ab.py`` turns into exit 1 and a message, not a traceback.
"""

from __future__ import annotations

import json
import math

import pytest
from typer.testing import CliRunner

from soup_cli.utils.ab_test import MsprtConfig, msprt_step
from tests.test_issue1227_ab_two_sided import _plain

runner = CliRunner()

# The issue's inputs, 40 rows per arm, at the issue's --alpha 0.1.
_TINY = [0.0, 1e-161] * 20
_HUGE = [0.0, 1e200] * 20


def _write_rows(path, metric, control, treatment):
    # Not test_issue1227's _write_ab: it rounds treatment values to 6 places,
    # which would turn 1e-161 into 0.
    rows = [{"arm": "control", metric: x} for x in control]
    rows += [{"arm": "treatment", metric: x} for x in treatment]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def _run_cli(tmp_path, monkeypatch, control, treatment, extra=()):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    _write_rows(tmp_path / "ab.jsonl", "latency", control, treatment)
    return runner.invoke(
        app,
        ["ab", "--input", "ab.jsonl", "--metric", "latency", "--alpha", "0.1", *extra],
    )


class TestSpreadBelowAFloat:
    def test_premise_the_standard_error_underflows(self):
        """Without this the tests below would not be exercising the underflow."""
        mean = sum(_TINY) / len(_TINY)
        var = sum((x - mean) * (x - mean) for x in _TINY) / (len(_TINY) - 1)
        assert var > 0.0
        assert var * (1.0 / 40 + 1.0 / 40) == 0.0

    def test_step_refuses_naming_the_column(self):
        with pytest.raises(ValueError) as exc:
            msprt_step(
                MsprtConfig(metric="latency", alpha=0.1),
                control=_TINY,
                treatment=_TINY,
            )
        message = str(exc.value)
        assert "'latency'" in message
        assert "below what the test statistic can represent" in message
        assert "Rescale" in message

    def test_cli_exits_1_with_a_message_not_a_traceback(self, tmp_path, monkeypatch):
        result = _run_cli(tmp_path, monkeypatch, [0.0] * 40, _TINY)
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert isinstance(result.exception, SystemExit), repr(result.exception)
        out = _plain(result.output)
        assert "'latency' column's spread is below" in out
        assert "ZeroDivisionError" not in out
        assert "Traceback" not in out

    def test_the_same_values_still_run_where_the_error_does_not_underflow(self):
        """At 2 rows per arm the standard error is about 5e-162, not 0: the
        refusal is keyed on the underflow, not on how small the values are."""
        verdict = msprt_step(
            MsprtConfig(metric="latency", alpha=0.1, effect_size=1e-162),
            control=[0.0, 1e-161],
            treatment=[0.0, 1e-161],
        )
        assert verdict.decision == "continue"  # burn-in
        assert math.isfinite(verdict.log_likelihood_ratio)


class TestSpreadAboveAFloat:
    @pytest.mark.parametrize(
        ("control", "treatment"),
        [
            (_HUGE, _HUGE),  # the issue's input: (x - mean) ** 2 overflowed
            ([1.0, 2.0] * 20, _HUGE),  # the treatment arm alone
        ],
        ids=["deviation-overflows", "treatment-only"],
    )
    def test_step_refuses_naming_the_column(self, control, treatment):
        with pytest.raises(ValueError) as exc:
            msprt_step(
                MsprtConfig(metric="latency", alpha=0.1),
                control=control,
                treatment=treatment,
            )
        message = str(exc.value)
        assert "'latency'" in message
        assert "above what the test statistic can represent" in message
        assert "Rescale" in message

    def test_cli_exits_1_with_a_message_not_a_traceback(self, tmp_path, monkeypatch):
        result = _run_cli(tmp_path, monkeypatch, _HUGE, _HUGE)
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert isinstance(result.exception, SystemExit), repr(result.exception)
        out = _plain(result.output)
        assert "'latency' column's spread is above" in out
        assert "1e+200" in out
        assert "OverflowError" not in out
        assert "Traceback" not in out

    def test_large_values_whose_variance_fits_still_decide(self):
        """0 and 1e153, 40 rows per arm: a variance near 2.6e305 is a float."""
        values = [0.0, 1e153] * 20
        verdict = msprt_step(
            MsprtConfig(metric="latency", alpha=0.1, effect_size=1e152),
            control=values,
            treatment=values,
        )
        assert verdict.decision in {"continue", "accept_h0"}
        assert math.isfinite(verdict.log_likelihood_ratio)


class TestValuesTooLargeToSum:
    @pytest.mark.parametrize(
        ("control", "treatment"),
        [
            ([1e308] * 40, [1.0, 2.0] * 20),  # one arm's sum overflows: mean inf
            ([1e308] * 40, [1e308] * 40),  # no spread at all, yet the sums overflow
            ([1.0, 2.0] * 20, [1e308] * 40),  # the treatment arm's sum alone
        ],
        ids=["arm-sum-overflows", "constant-column", "treatment-sum-overflows"],
    )
    def test_step_names_the_size_not_the_spread(self, control, treatment):
        with pytest.raises(ValueError) as exc:
            msprt_step(
                MsprtConfig(metric="latency", alpha=0.1),
                control=control,
                treatment=treatment,
            )
        message = str(exc.value)
        assert "'latency'" in message
        assert "sum overflows a float" in message
        assert "spread" not in message
        assert "Rescale" in message
