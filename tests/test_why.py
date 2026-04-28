"""Tests for v0.34.0 Part C — soup why explainer."""

from __future__ import annotations

import math

from soup_cli.utils.why import Finding, diagnose


def _row(step, loss, grad_norm=1.0):
    return {"step": step, "loss": loss, "grad_norm": grad_norm}


class TestDiagnose:
    def test_empty_metrics(self):
        result = diagnose([])
        assert len(result) == 1
        assert result[0].category == "no_metrics"

    def test_short_run(self):
        result = diagnose([_row(i, 2.0 - i * 0.01) for i in range(5)])
        assert any(finding.category == "too_few_steps" for finding in result)

    def test_healthy_run_no_findings(self):
        # Decreasing loss across 100 steps with healthy gradients
        rows = [_row(i, 2.0 * 0.99 ** i, grad_norm=2.0) for i in range(100)]
        result = diagnose(rows)
        assert result == []

    def test_nan_detected(self):
        rows = [_row(0, 2.0), _row(10, float("nan"))]
        result = diagnose(rows)
        assert any(finding.category == "nan_loss" for finding in result)
        assert result[0].severity == "critical"

    def test_inf_detected(self):
        rows = [_row(0, 2.0), _row(10, math.inf)]
        result = diagnose(rows)
        assert any(finding.category == "nan_loss" for finding in result)

    def test_plateau_detected(self):
        rows = [_row(i, 2.0 + 0.0001 * i) for i in range(50)]
        result = diagnose(rows)
        assert any(finding.category == "loss_flat" for finding in result)

    def test_explosion_detected(self):
        rows = [_row(i, 1.0 + i * 0.5) for i in range(20)]
        result = diagnose(rows)
        cats = {finding.category for finding in result}
        assert "loss_diverged" in cats

    def test_high_grad_norm_detected(self):
        rows = [_row(i, 2.0 * 0.99 ** i, grad_norm=100.0) for i in range(30)]
        result = diagnose(rows)
        assert any(finding.category == "grad_norm_high" for finding in result)

    def test_lr_too_low_warning(self):
        rows = [_row(i, 2.0 * 0.99 ** i) for i in range(30)]
        config = {"training": {"lr": 1e-9}}
        result = diagnose(rows, config)
        assert any(finding.category == "lr_too_low" for finding in result)

    def test_lr_too_high_warning(self):
        rows = [_row(i, 2.0 * 0.99 ** i) for i in range(30)]
        config = {"training": {"lr": 0.1}}
        result = diagnose(rows, config)
        assert any(finding.category == "lr_too_high" for finding in result)

    def test_severity_ordering(self):
        # Mix critical (nan) + warning (lr_too_low). Critical first.
        rows = [_row(0, 2.0), _row(1, float("nan"))]
        config = {"training": {"lr": 1e-9}}
        result = diagnose(rows, config)
        assert result[0].severity == "critical"
        # Subsequent items must be warning or info
        for item in result[1:]:
            assert item.severity != "critical"

    def test_too_few_steps_suppressed_when_other_findings_present(self):
        # NaN at step 3 of a 4-step run: NaN should fire but too_few_steps must NOT.
        rows = [_row(0, 2.0), _row(1, 1.5), _row(2, 1.0), _row(3, float("nan"))]
        result = diagnose(rows)
        cats = {finding.category for finding in result}
        assert "nan_loss" in cats
        assert "too_few_steps" not in cats

    def test_plateau_first_zero_skipped(self):
        # If initial loss is 0, plateau detector must short-circuit, not divide by zero.
        rows = [_row(i, 0.0) for i in range(50)]
        # Should not raise
        result = diagnose(rows)
        cats = {finding.category for finding in result}
        assert "loss_flat" not in cats

    def test_finding_is_frozen(self):
        finding = Finding(
            category="x", severity="info", message="m", suggestion="s",
        )
        try:
            finding.category = "y"  # type: ignore[misc]
        except Exception:
            return
        raise AssertionError("Finding should be frozen")
