"""v0.63.0 review follow-up coverage gaps.

Closes findings from the tdd-guide wave 1 review:
- HIGH: msprt zero-variance path returns "continue"
- HIGH: detect_common_prefix partial-majority binary-search path
- MEDIUM: score_uncertainty exact boundary semantics
- MEDIUM: rolling_kl identical-distribution + disjoint vocabulary
- MEDIUM: validate_budget + validate_threshold exact endpoints
- LOW: _signal_from_thumbs exact numeric boundaries
- LOW: no-heavy-top-level-imports source-grep regression guard
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# HIGH — msprt zero-variance returns "continue"
# ---------------------------------------------------------------------------


def test_msprt_step_zero_variance_returns_continue():
    """When both arms are constant + identical, pooled SE is 0 -> continue.

    Documents the canonical fall-through: without variability the test
    cannot distinguish the hypotheses, so we must defer a verdict until
    more samples arrive.
    """
    from soup_cli.utils.ab_test import MsprtConfig, msprt_step

    cfg = MsprtConfig(metric="latency")
    verdict = msprt_step(cfg, control=[1.0] * 50, treatment=[1.0] * 50)
    assert verdict.decision == "continue"
    assert verdict.log_likelihood_ratio == 0.0


# ---------------------------------------------------------------------------
# HIGH — detect_common_prefix partial-majority binary-search activates
# ---------------------------------------------------------------------------


def test_detect_common_prefix_partial_majority_binary_search_activates():
    """At threshold=0.66, 2/3 rows share `[SYS] ` while the third lacks it.

    Forces the binary-search-over-templates branch (since no 100% prefix
    exists) and asserts the discovered prefix matches the 2/3 cohort.
    """
    from soup_cli.utils.prune_prompt import detect_common_prefix

    rows = [
        "[SYS] be safe.\nUser: a",
        "[SYS] be safe.\nUser: b",
        "Completely different start with no shared chars at all",
    ]
    prefix = detect_common_prefix(rows, min_frequency=0.66)
    assert prefix.startswith("[SYS] ")
    # The shared portion should be at least "[SYS] " (6 chars)
    assert len(prefix) >= 6


# ---------------------------------------------------------------------------
# MEDIUM — score_uncertainty exact boundary float semantics
# ---------------------------------------------------------------------------


def test_score_uncertainty_max_entropy_at_0_5_is_exactly_1_0():
    """score=0.5 -> uncertainty exactly 1.0 (peak entropy)."""
    from soup_cli.utils.active_sampler import score_uncertainty

    assert score_uncertainty(scores=[0.5]) == 1.0


def test_score_uncertainty_at_extremes_is_exactly_0_0():
    """score=0.0 AND score=1.0 -> uncertainty exactly 0.0."""
    from soup_cli.utils.active_sampler import score_uncertainty

    assert score_uncertainty(scores=[0.0]) == 0.0
    assert score_uncertainty(scores=[1.0]) == 0.0


# ---------------------------------------------------------------------------
# MEDIUM — rolling_kl identical + disjoint vocabularies
# ---------------------------------------------------------------------------


def test_rolling_kl_identical_distributions_is_near_zero():
    """Identical p and q -> KL ~= 0 (modulo floating-point noise)."""
    from soup_cli.utils.drift_alarm import rolling_kl

    p = {"alpha": 0.4, "beta": 0.3, "gamma": 0.3}
    q = {"alpha": 0.4, "beta": 0.3, "gamma": 0.3}
    assert rolling_kl(p, q) == pytest.approx(0.0, abs=1e-9)


def test_rolling_kl_disjoint_distributions_is_positive_and_finite():
    """Vocabularies don't overlap — _EPS smoothing must yield finite > 0."""
    from soup_cli.utils.drift_alarm import rolling_kl

    p = {"alpha": 0.5, "beta": 0.5}  # tokens NOT in q
    q = {"gamma": 0.5, "delta": 0.5}
    kl = rolling_kl(p, q)
    assert math.isfinite(kl)
    assert kl > 0.0


# ---------------------------------------------------------------------------
# MEDIUM — validate_budget exact endpoints
# ---------------------------------------------------------------------------


def test_validate_budget_exact_lower_boundary_one_accepted():
    from soup_cli.utils.active_sampler import validate_budget

    assert validate_budget(1) == 1


def test_validate_budget_zero_rejected_with_message():
    from soup_cli.utils.active_sampler import validate_budget

    with pytest.raises(ValueError, match=r">= 1|>=1|at least 1"):
        validate_budget(0)


def test_validate_budget_exact_upper_boundary_100000_accepted():
    from soup_cli.utils.active_sampler import validate_budget

    assert validate_budget(100_000) == 100_000


def test_validate_budget_100001_rejected_with_message():
    from soup_cli.utils.active_sampler import validate_budget

    with pytest.raises(ValueError, match=r"100[_,]?000|100000"):
        validate_budget(100_001)


# ---------------------------------------------------------------------------
# MEDIUM — validate_threshold exact endpoints
# ---------------------------------------------------------------------------


def test_validate_threshold_exact_boundary_100_0_accepted():
    from soup_cli.utils.drift_alarm import validate_threshold

    assert validate_threshold(100.0) == 100.0


def test_validate_threshold_zero_rejected_with_gt_zero_message():
    from soup_cli.utils.drift_alarm import validate_threshold

    with pytest.raises(ValueError, match=r"> 0|>0"):
        validate_threshold(0.0)


def test_validate_threshold_above_100_rejected():
    from soup_cli.utils.drift_alarm import validate_threshold

    with pytest.raises(ValueError, match=r"<= 100|<=100"):
        validate_threshold(100.0001)


# ---------------------------------------------------------------------------
# LOW — _signal_from_thumbs exact numeric boundaries
# ---------------------------------------------------------------------------


def test_signal_from_thumbs_exact_numeric_boundaries():
    """score=1.0 -> thumbs_up; score=0.0 -> thumbs_down; score=0.5 -> none."""
    from soup_cli.utils.ingest_sources import _signal_from_thumbs

    assert _signal_from_thumbs(1.0) == "thumbs_up"
    assert _signal_from_thumbs(0.0) == "thumbs_down"
    assert _signal_from_thumbs(0.5) == "none"
    # >1 also clamps up; <0 also clamps down
    assert _signal_from_thumbs(2.5) == "thumbs_up"
    assert _signal_from_thumbs(-1.0) == "thumbs_down"


# ---------------------------------------------------------------------------
# LOW — source-grep regression guard: no heavy top-level imports
# ---------------------------------------------------------------------------


_V0630_UTIL_MODULES = (
    "soup_cli/utils/ingest_sources.py",
    "soup_cli/utils/prune_prompt.py",
    "soup_cli/utils/active_sampler.py",
    "soup_cli/utils/ab_test.py",
    "soup_cli/utils/drift_alarm.py",
)


@pytest.mark.parametrize("module_path", _V0630_UTIL_MODULES)
def test_v0630_no_heavy_top_level_imports(module_path):
    """The 5 new util modules must not top-level-import torch/transformers/peft.

    Project policy: heavy deps are lazy-imported inside the call sites that
    need them. A regression here would make `soup --help` slow to import.
    """
    repo_root = Path(__file__).resolve().parent.parent
    text = (repo_root / module_path).read_text(encoding="utf-8")
    for needle in ("import torch", "from torch", "import transformers",
                   "from transformers", "import peft", "from peft",
                   "import trl", "from trl"):
        # Check only the top of the file (first 50 lines = imports zone)
        head = "\n".join(text.splitlines()[:50])
        assert needle not in head, (
            f"{module_path} has a top-level {needle!r} (heavy dep — must be lazy)"
        )
