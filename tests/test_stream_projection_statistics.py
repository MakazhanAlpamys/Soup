"""Whole-pair statistics retain the effect threshold and correction family."""
import importlib.util
from pathlib import Path

import pytest


def _helper():
    path = Path(__file__).parents[1] / 'benchmarks/harness/stream_projection_statistics.py'
    spec = importlib.util.spec_from_file_location('projection_statistics', path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


@pytest.mark.parametrize('values,passes', [([3.0]*5, True), ([2.0]*5, True),
                                        ([1.9]*5, False), ([-2.0]*5, False)])
def test_effect_threshold_is_independent_of_positive_interval(values, passes):
    row = _helper().summarize_pairs(values)
    assert row['passes_rule'] is passes
    assert row['ordinary_95'] == [values[0], values[0]]
    assert row['family_adjusted_97_5'] == [values[0], values[0]]


def test_heterogeneous_pairs_stay_in_the_bootstrap():
    row = _helper().summarize_pairs([-30, 3, 3.1, 3.2, 3.3])
    assert row['median_reduction_percent'] == 3.1
    assert row['family_adjusted_97_5'][0] < 0
    assert row['passes_rule'] is False


@pytest.mark.parametrize('values', [[3]*4, [3]*6, [3]*4+[float('nan')], [float('inf')]*5])
def test_missing_or_invalid_pairs_are_not_silently_dropped(values):
    with pytest.raises(ValueError, match='five finite'):
        _helper().summarize_pairs(values)


def test_reallocation_uses_six_balanced_blocks_and_three_contrasts():
    result = _helper().summarize_reallocation_blocks([2.1]*6)
    assert result['passes_rule'] is True
    assert result['n_blocks'] == 6
    assert result['correction_family_contrasts'] == 3
    assert result['adjusted_quantiles'] == [0.05/6, 1-0.05/6]
    assert result['family_adjusted_ci'] == [2.1, 2.1]


def test_reallocation_does_not_accept_a_small_effect_or_drop_a_block():
    helper = _helper()
    assert helper.summarize_reallocation_blocks([1.99]*6)['passes_rule'] is False
    with pytest.raises(ValueError, match='six finite'):
        helper.summarize_reallocation_blocks([3]*5)
