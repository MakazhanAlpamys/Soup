"""Closes #206 — Variance-based diversity score for K>2 reward models in soup data active-sample.

v0.63.0 ``score_uncertainty`` raised ValueError on len(scores) > 2 and
``_row_uncertainty`` fell back to a broken ``max(scores) - min(scores)``
range (monotone-broken — a third score equal to the existing mean still
spiked the score, so adding redundant evidence inflated uncertainty).

This change generalises ``score_uncertainty`` to up to K=32 RMs via
population-variance with a 4x scaling so the formula stays in [0, 1] for
scores in [0, 1]:

    K=0  -> 0.0                                  (unchanged)
    K=1  -> 1 - |2*s - 1|        (max-entropy)   (unchanged)
    K=2  -> |s1 - s2|            (disagreement)  (unchanged)
    K>=3 -> 4 * pop_variance     (clamped [0,1])  (NEW)
    K>32 -> ValueError (DoS cap, was K>2)

Monotonicity invariant: adding a fresh score equal to the running mean
*decreases* population variance (the new contribution to the sum-of-squares
is zero while the denominator grows), so the uncertainty estimate cannot
spike when the K-th RM just agrees with the consensus.
"""

from __future__ import annotations

import math

import pytest

# ---------------------------------------------------------------------------
# K>=3 variance path
# ---------------------------------------------------------------------------


def test_k_equals_3_returns_finite_unit_value():
    from soup_cli.utils.active_sampler import score_uncertainty

    s = score_uncertainty(scores=[0.1, 0.5, 0.9])
    assert math.isfinite(s)
    assert 0.0 <= s <= 1.0
    # 4 * pop_var([0.1, 0.5, 0.9]) = 4 * (0.16 + 0 + 0.16) / 3 = 0.4266666...
    # Tight tolerance (1e-9) catches K vs K-1 denominator drift.
    assert s == pytest.approx(4.0 * 0.32 / 3.0, abs=1e-9)


def test_k_equals_4_returns_finite_unit_value():
    from soup_cli.utils.active_sampler import score_uncertainty

    s = score_uncertainty(scores=[0.2, 0.4, 0.6, 0.8])
    assert 0.0 <= s <= 1.0
    # mean=0.5, var = ((-.3)^2 + (-.1)^2 + (.1)^2 + (.3)^2)/4 = 0.20/4 = 0.05
    # 4*var = 0.20
    assert s == pytest.approx(0.20, abs=0.001)


def test_k_equals_8_returns_finite_unit_value():
    from soup_cli.utils.active_sampler import score_uncertainty

    scores = [0.0, 0.2, 0.4, 0.5, 0.5, 0.6, 0.8, 1.0]
    s = score_uncertainty(scores=scores)
    assert 0.0 <= s <= 1.0
    assert math.isfinite(s)


def test_max_uncertainty_when_half_zero_half_one():
    from soup_cli.utils.active_sampler import score_uncertainty

    # Half-0 half-1: pop variance = 0.25, 4*var = 1.0 — maximum disagreement
    s = score_uncertainty(scores=[0.0, 0.0, 1.0, 1.0])
    assert s == pytest.approx(1.0, abs=1e-9)


def test_zero_uncertainty_when_all_agree():
    from soup_cli.utils.active_sampler import score_uncertainty

    s = score_uncertainty(scores=[0.5, 0.5, 0.5, 0.5, 0.5])
    assert s == 0.0


# ---------------------------------------------------------------------------
# Monotonicity — adding a score at the mean must decrease/hold variance
# ---------------------------------------------------------------------------


def test_adding_score_at_mean_decreases_uncertainty():
    """Critical correctness invariant — see acceptance criteria in #206."""
    from soup_cli.utils.active_sampler import score_uncertainty

    base = [0.1, 0.5, 0.9]
    mean = sum(base) / len(base)
    u_before = score_uncertainty(scores=base)
    u_after = score_uncertainty(scores=base + [mean])
    assert u_after < u_before, (
        f"adding the mean ({mean}) must reduce variance — "
        f"before={u_before}, after={u_after}"
    )


def test_adding_score_at_mean_holds_when_already_constant():
    from soup_cli.utils.active_sampler import score_uncertainty

    # All-equal -> variance=0 already; adding the same value keeps it at 0.
    # Float reality: (0.4-0.4)^2 / N is ~1e-32 not literally 0.0 because
    # the mean computation introduces ULP noise. The math is correct;
    # the assertion just has to tolerate that noise.
    base = [0.4, 0.4, 0.4]
    u_before = score_uncertainty(scores=base)
    u_after = score_uncertainty(scores=base + [0.4])
    assert u_before == pytest.approx(0.0, abs=1e-12)
    assert u_after == pytest.approx(0.0, abs=1e-12)
    assert u_after <= u_before  # monotonicity invariant still holds


def test_adding_disagreeing_score_increases_uncertainty():
    """Sanity counter-test: an outlier RM should INCREASE uncertainty."""
    from soup_cli.utils.active_sampler import score_uncertainty

    base = [0.5, 0.5, 0.5]
    u_before = score_uncertainty(scores=base)
    u_after = score_uncertainty(scores=base + [0.0])
    assert u_after > u_before


# ---------------------------------------------------------------------------
# K=1 and K=2 formulas preserved
# ---------------------------------------------------------------------------


def test_k_equals_1_max_entropy_formula_preserved():
    from soup_cli.utils.active_sampler import score_uncertainty

    assert score_uncertainty(scores=[0.5]) == 1.0
    assert score_uncertainty(scores=[0.0]) == 0.0
    assert score_uncertainty(scores=[1.0]) == 0.0


def test_k_equals_2_disagreement_formula_preserved():
    from soup_cli.utils.active_sampler import score_uncertainty

    # The issue calls out "variance gives the same answer for K=2 up to
    # scaling" — but the literal pairwise-disagreement |s1 - s2| formula
    # is what existing operators depend on. Don't drift it.
    assert score_uncertainty(scores=[0.1, 0.9]) == pytest.approx(0.8)
    assert score_uncertainty(scores=[0.5, 0.5]) == 0.0


# ---------------------------------------------------------------------------
# Cap at K=32 — DoS defence
# ---------------------------------------------------------------------------


def test_k_equals_32_accepted_at_boundary():
    from soup_cli.utils.active_sampler import score_uncertainty

    s = score_uncertainty(scores=[0.5] * 32)
    assert s == 0.0


def test_k_equals_32_with_disagreement_accepted():
    from soup_cli.utils.active_sampler import score_uncertainty

    scores = [0.0] * 16 + [1.0] * 16
    s = score_uncertainty(scores=scores)
    assert s == pytest.approx(1.0)


def test_k_equals_33_rejected_at_boundary():
    from soup_cli.utils.active_sampler import score_uncertainty

    with pytest.raises(ValueError, match="32"):
        score_uncertainty(scores=[0.5] * 33)


def test_max_rm_scores_constant_is_32():
    """Lock the cap so future drift fails loudly."""
    from soup_cli.utils.active_sampler import _MAX_RM_SCORES

    assert _MAX_RM_SCORES == 32


# ---------------------------------------------------------------------------
# Per-element validation still propagates at K>2
# ---------------------------------------------------------------------------


def test_k3_rejects_bool_entry():
    from soup_cli.utils.active_sampler import score_uncertainty

    with pytest.raises(TypeError, match="bool"):
        score_uncertainty(scores=[0.5, 0.5, True])


def test_k3_rejects_non_finite_entry():
    from soup_cli.utils.active_sampler import score_uncertainty

    with pytest.raises(ValueError, match="finite"):
        score_uncertainty(scores=[0.5, 0.5, float("nan")])
    with pytest.raises(ValueError, match="finite"):
        score_uncertainty(scores=[0.5, 0.5, float("inf")])


def test_k3_rejects_out_of_range_entry():
    from soup_cli.utils.active_sampler import score_uncertainty

    with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
        score_uncertainty(scores=[0.5, 0.5, 1.5])
    with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
        score_uncertainty(scores=[0.5, 0.5, -0.1])


def test_k3_rejects_non_numeric_entry():
    from soup_cli.utils.active_sampler import score_uncertainty

    with pytest.raises(TypeError, match="number"):
        score_uncertainty(scores=[0.5, 0.5, "0.5"])


# ---------------------------------------------------------------------------
# _row_uncertainty K>2 path delegates to variance (no more max-min fallback)
# ---------------------------------------------------------------------------


def test_row_uncertainty_k3_uses_variance_not_max_minus_min():
    """Regression: the old max-min fallback was monotone-broken.

    Two rows with the SAME (min, max) but different middle scores must
    now score differently when the inner scores' spread differs.
    """
    from soup_cli.utils.active_sampler import _row_uncertainty

    # Both rows have min=0.1, max=0.9 -> old max-min = 0.8 for both.
    row_consensus = {"rm_scores": [0.1, 0.5, 0.9]}
    row_polarised = {"rm_scores": [0.1, 0.1, 0.9]}
    u_consensus = _row_uncertainty(row_consensus)
    u_polarised = _row_uncertainty(row_polarised)
    # New behaviour: variance picks up the structural difference.
    assert u_consensus != u_polarised
    # Polarised (more mass at extremes) has higher variance.
    assert u_polarised > u_consensus


def test_row_uncertainty_k4_returns_unit_value():
    from soup_cli.utils.active_sampler import _row_uncertainty

    row = {"rm_scores": [0.2, 0.4, 0.6, 0.8]}
    u = _row_uncertainty(row)
    assert 0.0 <= u <= 1.0
    assert math.isfinite(u)


def test_row_uncertainty_k33_returns_zero_isolated():
    """K>32 in row data: do NOT crash the loop — fall through to 0.0.

    Matches existing isolation policy: bad data on one row never breaks
    the whole batch (see ``_row_uncertainty`` try/except around _validate_score).
    """
    from soup_cli.utils.active_sampler import _row_uncertainty

    row = {"rm_scores": [0.5] * 33}
    assert _row_uncertainty(row) == 0.0


def test_row_uncertainty_k1_and_k2_paths_preserved():
    """K=1 and K=2 rows still route to their existing closed-form formulas."""
    from soup_cli.utils.active_sampler import _row_uncertainty

    # K=1
    assert _row_uncertainty({"rm_scores": [0.5]}) == 1.0
    assert _row_uncertainty({"rm_scores": [0.0]}) == 0.0
    # K=2
    assert _row_uncertainty({"rm_scores": [0.1, 0.9]}) == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# sample_uncertain_rows end-to-end with K=3 rows
# ---------------------------------------------------------------------------


def test_sample_uncertain_rows_triple_rm(tmp_path, monkeypatch):
    """Closes #206 — pick rows where 3 RMs disagree the most."""
    import json

    from soup_cli.utils.active_sampler import sample_uncertain_rows

    monkeypatch.chdir(tmp_path)
    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    rows = [
        {"id": "a", "rm_scores": [0.0, 0.5, 1.0]},   # high variance
        {"id": "b", "rm_scores": [0.5, 0.5, 0.5]},   # zero variance
        {"id": "c", "rm_scores": [0.4, 0.5, 0.6]},   # low variance
    ]
    inp.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    plan = sample_uncertain_rows(str(inp), output_path=str(out), budget=2)
    assert plan.rows_selected == 2
    out_rows = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()]
    ids = [r["id"] for r in out_rows]
    # Highest variance "a" first; "c" beats "b" (zero variance) for the second slot.
    assert ids == ["a", "c"]


# ---------------------------------------------------------------------------
# Source-grep regression guards
# ---------------------------------------------------------------------------


def test_max_minus_min_fallback_removed():
    """The v0.63.0 max-min fallback is a monotone-broken anti-pattern.

    If this guard ever fires, someone re-introduced the broken K>2 path —
    they must use variance via score_uncertainty instead.
    """
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "soup_cli" / "utils" / "active_sampler.py"
    text = src.read_text(encoding="utf-8")
    # The exact broken line; if a comment mentions max/min that's fine.
    assert "return max(scores_list) - min(scores_list)" not in text


def test_score_uncertainty_no_top_level_heavy_imports():
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "soup_cli" / "utils" / "active_sampler.py"
    text = src.read_text(encoding="utf-8")
    for forbidden in ("import torch", "import numpy", "import statistics"):
        assert f"\n{forbidden}" not in text, (
            f"active_sampler.py must stay pure-stdlib (no top-level {forbidden!r})"
        )


# ---------------------------------------------------------------------------
# TDD review follow-ups
# ---------------------------------------------------------------------------


def test_k_equals_31_accepted_below_cap():
    """HIGH: project convention is N-1, N, N+1 boundary coverage.

    A regression that set ``_MAX_RM_SCORES = 30`` would otherwise pass
    every K=32/K=33 test in this file.
    """
    from soup_cli.utils.active_sampler import score_uncertainty

    s = score_uncertainty(scores=[0.5] * 31)
    assert 0.0 <= s <= 1.0
    assert math.isfinite(s)


def test_row_uncertainty_explicit_field_overrides_k3_rm_scores():
    """HIGH: ``_row_uncertainty`` field-priority contract.

    Explicit ``uncertainty`` field MUST win over rm_scores even when
    rm_scores would also produce a valid K>=3 variance score. A regression
    that flipped the priority order could otherwise pass silently because
    rm_scores=[0.5,0.5,0.5] returns 0.0 — same as a missing field.
    """
    from soup_cli.utils.active_sampler import _row_uncertainty

    assert _row_uncertainty(
        {"uncertainty": 0.9, "rm_scores": [0.5, 0.5, 0.5]}
    ) == pytest.approx(0.9)


def test_old_k2_deferred_error_message_removed():
    """HIGH: source-grep regression guard for the broken K>2 ValueError text.

    The v0.63.0 code raised ``ValueError("...K>2 RMs deferred...")``. We've
    lifted that. Without this guard, a future revert could re-introduce the
    deferred-message text and only the K=33 boundary test would catch it
    (matching only the literal "32" — not the deferred phrasing).
    """
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "soup_cli" / "utils" / "active_sampler.py"
    text = src.read_text(encoding="utf-8")
    assert "K>2 RMs deferred" not in text
    assert "deferred to a future release" not in text


def test_empty_scores_returns_zero():
    """MEDIUM: K=0 contract explicit (was implied, never asserted in v0.63.1)."""
    from soup_cli.utils.active_sampler import score_uncertainty

    assert score_uncertainty(scores=[]) == 0.0


def test_row_uncertainty_scalar_rm_score_path_preserved():
    """MEDIUM: scalar ``rm_score`` (Priority-3 path) preserved.

    The K>2 routing rewrite could plausibly have broken the scalar branch
    without any new test catching it.
    """
    from soup_cli.utils.active_sampler import _row_uncertainty

    assert _row_uncertainty({"rm_score": 1.0}) == pytest.approx(0.0)
    assert _row_uncertainty({"rm_score": 0.0}) == pytest.approx(0.0)
    assert _row_uncertainty({"rm_score": 0.5}) == pytest.approx(1.0)


def test_row_uncertainty_k3_with_nan_score_isolates_to_zero():
    """MEDIUM: K>=3 row with NaN entry must isolate to 0.0, not crash batch.

    Matches the K>cap and bool/oversize isolation policy already in
    ``_row_uncertainty`` — bad data on one row never breaks the batch.
    """
    from soup_cli.utils.active_sampler import _row_uncertainty

    assert _row_uncertainty({"rm_scores": [0.5, 0.5, float("nan")]}) == 0.0
    assert _row_uncertainty({"rm_scores": [0.5, 0.5, float("inf")]}) == 0.0


def test_k2_equal_scores_strict_zero():
    """MEDIUM: K=2 with equal scores returns *strict* 0.0.

    The K=2 path uses IEEE 754 subtraction (no accumulated error) so we
    can assert strict equality. Pairs with the K>=3 monotonicity-with-
    constant-input test (which uses ``abs=1e-12`` tolerance) to document
    where strict-zero vs sub-ULP residual applies.
    """
    from soup_cli.utils.active_sampler import score_uncertainty

    assert score_uncertainty(scores=[0.4, 0.4]) == 0.0
    assert score_uncertainty(scores=[0.0, 0.0]) == 0.0
    assert score_uncertainty(scores=[1.0, 1.0]) == 0.0
