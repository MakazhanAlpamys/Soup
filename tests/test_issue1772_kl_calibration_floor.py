"""Regression tests for Issue #1772: KL calibration probability floor.

eval.calibrate.kl_divergence previously floored each candidate probability at
1e-12 before taking the log. For finite logits with candidate probabilities
below 1e-12 (logit gaps > ~27.6), this artificially underestimated KL divergence,
causing run_calibration to return MINOR when the actual divergence is above
its documented MAJOR boundary (>= 0.20).

The fix computes KL divergence using numerically stable log-softmax without
an artificial probability floor.
"""

from __future__ import annotations

import math
import sys

import pytest

from soup_cli.eval.calibrate import (
    CalibrationReport,
    _log_softmax,
    _log_softmax_and_probs,
    _softmax,
    classify_kl_delta,
    kl_divergence,
    run_calibration,
)


class TestKlCalibrationFloorRegression:
    """Verify that candidate logit gaps are not capped by an artificial probability floor."""

    def test_reproduction_gap_115_reported_as_major(self):
        """Reproduction case from #1772: gap 115 gives KL ~0.6533, which is MAJOR."""
        baseline = [math.log(0.006 / 0.994), 0.0]
        candidate = [-115.0, 0.0]

        # Analytical reference
        prob = 0.006
        cand_prob = 1.0 / (1.0 + math.exp(115.0))
        expected_kl = (
            prob * math.log(prob / cand_prob)
            + (1.0 - prob) * math.log((1.0 - prob) / (1.0 - cand_prob))
        )

        kl = kl_divergence(baseline, candidate)
        assert kl == pytest.approx(expected_kl, rel=1e-6)
        assert kl == pytest.approx(0.65332206125, rel=1e-6)
        assert classify_kl_delta(kl) == "MAJOR"

        report = run_calibration([baseline], [candidate])
        assert isinstance(report, CalibrationReport)
        assert report.mean_kl == pytest.approx(expected_kl, rel=1e-6)
        # Buggy code returned MINOR with mean_kl = 0.129108187945
        assert report.delta_status == "MAJOR"

    def test_candidate_gap_60_reported_as_major(self):
        """Candidate gap 60 gives KL ~0.3233, which is MAJOR (> 0.20)."""
        baseline = [math.log(0.006 / 0.994), 0.0]
        candidate = [-60.0, 0.0]

        kl = kl_divergence(baseline, candidate)
        assert kl == pytest.approx(0.32332206125, rel=1e-6)

        report = run_calibration([baseline], [candidate])
        assert report.delta_status == "MAJOR"

    def test_candidate_gap_1000_reported_as_major(self):
        """Candidate gap 1000 gives KL ~5.9633, which is MAJOR (> 0.20)."""
        baseline = [math.log(0.006 / 0.994), 0.0]
        candidate = [-1000.0, 0.0]

        kl = kl_divergence(baseline, candidate)
        assert kl == pytest.approx(5.96332206125, rel=1e-6)

        report = run_calibration([baseline], [candidate])
        assert report.delta_status == "MAJOR"

    def test_kl_strictly_monotonic_with_gap(self):
        """Increasing the gap must strictly increase KL, not plateau at 0.1291."""
        baseline = [math.log(0.006 / 0.994), 0.0]
        gaps = [30.0, 60.0, 115.0, 200.0, 500.0, 1000.0]
        kl_values = [kl_divergence(baseline, [-g, 0.0]) for g in gaps]

        for i in range(len(kl_values) - 1):
            assert kl_values[i] < kl_values[i + 1], (
                f"KL did not increase strictly from gap {gaps[i]} ({kl_values[i]}) "
                f"to gap {gaps[i+1]} ({kl_values[i+1]})"
            )


class TestHighPrecisionReferenceSweep:
    """Compare kl_divergence against closed-form 2-class KL over a range of gaps."""

    @pytest.mark.parametrize("gap", [10.0, 20.0, 25.0, 28.0, 40.0, 60.0, 115.0, 500.0])
    def test_two_class_kl_matches_analytical(self, gap: float):
        p0 = 0.006
        p1 = 0.994
        baseline = [math.log(p0 / p1), 0.0]
        candidate = [-gap, 0.0]

        # Closed form:
        # log_q0 = -gap - log(1 + exp(-gap))
        # log_q1 = -log(1 + exp(-gap))
        log_denom = math.log1p(math.exp(-gap)) if gap < 700 else 0.0
        log_q0 = -gap - log_denom
        log_q1 = -log_denom
        expected = p0 * (math.log(p0) - log_q0) + p1 * (math.log(p1) - log_q1)

        computed = kl_divergence(baseline, candidate)
        assert computed == pytest.approx(expected, rel=1e-6)


class TestHealthyControlsPreserved:
    """Ensure standard distributions, shift invariance, and boundaries remain correct."""

    def test_identical_distributions_exact_zero(self):
        kl = kl_divergence([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert kl == pytest.approx(0.0, abs=1e-12)

    def test_global_logit_shift_invariance(self):
        # Adding a constant to all logits does not change the softmax distribution
        kl1 = kl_divergence([1.0, 2.0, 3.0], [101.0, 102.0, 103.0])
        assert kl1 == pytest.approx(0.0, abs=1e-12)

        kl2 = kl_divergence([-50.0, -40.0, -30.0], [50.0, 60.0, 70.0])
        assert kl2 == pytest.approx(0.0, abs=1e-12)

    def test_ordinary_logits_positive(self):
        kl = kl_divergence([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
        assert kl > 0.0
        assert kl == pytest.approx(1.1504207652, rel=1e-6)

    def test_small_divergence_classified_as_ok(self):
        baseline = [1.0, 2.0, 3.0]
        # Tiny perturbation
        candidate = [1.01, 2.0, 2.99]
        report = run_calibration([baseline], [candidate])
        assert report.delta_status == "OK"
        assert report.mean_kl < 0.05

    def test_moderate_divergence_classified_as_minor(self):
        # Construct candidate logits that produce mean KL in [0.05, 0.20)
        baseline = [0.0, 0.0]
        # For symmetric 2-class, p = [0.5, 0.5]
        # Candidate q with gap delta: KL = 0.5 * (log(0.5) - log(q0)) + 0.5 * (log(0.5) - log(q1))
        # With gap ~0.7, KL ~ 0.06 (MINOR)
        candidate = [0.7, 0.0]
        report = run_calibration([baseline], [candidate])
        assert 0.05 <= report.mean_kl < 0.20
        assert report.delta_status == "MINOR"


class TestMultiPromptCalibration:
    """Verify run_calibration across multiple prompt logit rows."""

    def test_batch_prompts_mean_and_per_prompt(self):
        b1 = [math.log(0.006 / 0.994), 0.0]
        q1 = [-115.0, 0.0]  # ~0.6533 (MAJOR)

        b2 = [1.0, 2.0, 3.0]
        q2 = [1.0, 2.0, 3.0]  # 0.0 (OK)

        report = run_calibration([b1, b2], [q1, q2])
        assert report.num_prompts == 2
        assert len(report.per_prompt_kl) == 2
        assert report.per_prompt_kl[0] == pytest.approx(0.65332206125, rel=1e-6)
        assert report.per_prompt_kl[1] == pytest.approx(0.0, abs=1e-12)

        expected_mean = (report.per_prompt_kl[0] + report.per_prompt_kl[1]) / 2.0
        assert report.mean_kl == pytest.approx(expected_mean, rel=1e-9)
        assert report.delta_status == "MAJOR"  # mean is ~0.3266 >= 0.20


class TestNumericalStabilityEdgeCases:
    """Test extreme logits, large positive/negative offsets, and vocabulary scales."""

    def test_large_positive_logits_no_overflow(self):
        base = [500.0, 500.0, 500.0]
        cand = [500.0, 500.0, 500.0]
        assert kl_divergence(base, cand) == pytest.approx(0.0, abs=1e-12)

    def test_large_negative_logits_no_underflow(self):
        base = [-500.0, -500.0, -500.0]
        cand = [-500.0, -500.0, -500.0]
        assert kl_divergence(base, cand) == pytest.approx(0.0, abs=1e-12)

    def test_baseline_probability_underflow_contributes_zero(self):
        # When an element of baseline has logit -1000 while max is 0,
        # its probability is exp(-1000) = 0.0 in float64.
        # Its contribution p * (log_p - log_q) must be cleanly skipped without NaN.
        base = [0.0, -1000.0]
        cand = [0.0, -500.0]
        kl = kl_divergence(base, cand)
        assert math.isfinite(kl)
        assert kl == pytest.approx(0.0, abs=1e-12)

    def test_large_vocabulary_scale(self):
        # 10,000-element vocabulary
        vocab_size = 10_000
        base = [float(i % 50) for i in range(vocab_size)]
        cand = [float(i % 50) for i in range(vocab_size)]
        assert kl_divergence(base, cand) == pytest.approx(0.0, abs=1e-12)

    def test_extreme_dynamic_range_difference_finite_major(self):
        # Gap between logits exceeds float max capacity (e.g. 1e308 vs -1e308),
        # which causes float subtraction overflow.
        # Must produce finite KL divergence saturated to float bounds and classify as MAJOR.
        base = [[0.0, 0.0]]
        cand = [[1e308, -1e308]]
        kl = kl_divergence(base[0], cand[0])
        assert math.isfinite(kl)
        assert kl == 0.5 * sys.float_info.max
        report = run_calibration(base, cand)
        assert report.delta_status == "MAJOR"
        assert report.mean_kl == 0.5 * sys.float_info.max

        # Polarized extreme logits saturate KL to float_info.max
        kl_sat = kl_divergence([1e308, -1e308], [-1e308, 1e308])
        assert kl_sat == sys.float_info.max
        report_sat = run_calibration([[1e308, -1e308]], [[-1e308, 1e308]])
        assert report_sat.delta_status == "MAJOR"
        assert report_sat.mean_kl == sys.float_info.max

    def test_multi_prompt_extreme_dynamic_range_saturation(self):
        # When two or more prompts each saturate at sys.float_info.max,
        # sum(per_prompt) must not overflow to inf causing classify_kl_delta to raise.
        base = [[1e308, -1e308]] * 2
        cand = [[-1e308, 1e308]] * 2
        report = run_calibration(base, cand)
        assert report.num_prompts == 2
        assert report.per_prompt_kl == (sys.float_info.max, sys.float_info.max)
        assert report.mean_kl == sys.float_info.max
        assert report.delta_status == "MAJOR"

        # Three or more saturated prompts also clamp to sys.float_info.max
        base3 = [[1e308, -1e308]] * 3
        cand3 = [[-1e308, 1e308]] * 3
        report3 = run_calibration(base3, cand3)
        assert report3.num_prompts == 3
        assert report3.mean_kl == sys.float_info.max
        assert report3.delta_status == "MAJOR"

        # Mixed saturated and zero prompts evaluate mean without intermediate overflow
        base_mixed = [[1e308, -1e308], [0.0, 0.0]]
        cand_mixed = [[-1e308, 1e308], [0.0, 0.0]]
        report_mixed = run_calibration(base_mixed, cand_mixed)
        assert report_mixed.num_prompts == 2
        assert report_mixed.per_prompt_kl == (sys.float_info.max, 0.0)
        assert report_mixed.mean_kl == 0.5 * sys.float_info.max
        assert report_mixed.delta_status == "MAJOR"

    def test_extreme_dynamic_range_identical_zero(self):
        base = [1e308, -1e308]
        cand = [1e308, -1e308]
        assert kl_divergence(base, cand) == pytest.approx(0.0, abs=1e-12)

    def test_single_element_logits_zero(self):
        assert kl_divergence([5.0], [-10.0]) == 0.0
        assert kl_divergence([0.0], [0.0]) == 0.0

    def test_internal_helpers_basic_properties(self):
        logits = [2.0, 1.0, 0.0]
        p = _softmax(logits)
        log_p = _log_softmax(logits)
        log_p2, p2 = _log_softmax_and_probs(logits)

        assert p == pytest.approx(p2)
        assert log_p == pytest.approx(log_p2)
        assert sum(p) == pytest.approx(1.0)
        for prob, lp in zip(p, log_p):
            assert math.exp(lp) == pytest.approx(prob)


class TestNumpyArrayInteroperability:
    """Kernel operates on numpy arrays as documented."""

    def test_numpy_1d_float64_and_float32(self):
        np = pytest.importorskip("numpy")
        p64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        q64 = np.array([3.0, 2.0, 1.0], dtype=np.float64)
        kl64 = kl_divergence(p64, q64)
        assert kl64 == pytest.approx(1.1504207652, rel=1e-6)

        p32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        q32 = np.array([3.0, 2.0, 1.0], dtype=np.float32)
        kl32 = kl_divergence(p32, q32)
        assert kl32 == pytest.approx(1.1504207652, rel=1e-6)

    def test_numpy_empty_array_rejected(self):
        np = pytest.importorskip("numpy")
        with pytest.raises(ValueError, match="must not be empty"):
            kl_divergence(np.array([]), np.array([]))

    def test_numpy_2d_batch_calibration(self):
        np = pytest.importorskip("numpy")
        base = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        quant = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        report = run_calibration(base, quant)
        assert report.delta_status == "OK"
        assert report.mean_kl == pytest.approx(0.0, abs=1e-12)


class TestValidationGuards:
    """Verify input validation and error cases."""

    def test_empty_logits_rejected(self):
        with pytest.raises(ValueError):
            kl_divergence([], [])

        with pytest.raises(ValueError):
            _softmax([])

        with pytest.raises(ValueError):
            _log_softmax([])

        with pytest.raises(ValueError):
            _log_softmax_and_probs([])

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            kl_divergence([1.0, 2.0], [1.0])

    def test_non_finite_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            kl_divergence([1.0, float("nan")], [1.0, 2.0])

        with pytest.raises(ValueError, match="finite"):
            kl_divergence([1.0, 2.0], [1.0, float("inf")])

        with pytest.raises(ValueError, match="finite"):
            kl_divergence([1.0, 2.0], [1.0, float("-inf")])

    def test_bool_rejected(self):
        with pytest.raises(ValueError):
            kl_divergence([True, False], [1.0, 2.0])  # type: ignore[list-item]
