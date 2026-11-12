"""Timing completeness and the gate-before-timing entry-point guard."""
import importlib.util
import sys
from pathlib import Path

import pytest


def _helper(name):
    path = Path(__file__).parents[1] / f'benchmarks/harness/stream_projection_{name}.py'
    spec = importlib.util.spec_from_file_location('projection_'+name, path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


@pytest.mark.parametrize('values', [[1.0]*49, [1.0]*51, [1.0]*49+[0],
                                   [1.0]*49+[float('nan')], [1.0]*49+[float('inf')]])
def test_invalid_or_incomplete_timing_is_refused(values):
    with pytest.raises(RuntimeError, match='50 finite positive'):
        _helper('timing').validate_times(values)


def test_complete_timing_is_accepted():
    _helper('timing').validate_times([0.5]*50)


@pytest.mark.parametrize('extra', [[], ['--reference', 'ref.json', '--profile']])
def test_timing_cannot_bypass_reference_or_enter_with_profiler(monkeypatch, extra):
    monkeypatch.setattr(sys, 'argv', ['gate', '--weights', 'unused', '--shards', 'unused',
        '--gpu', 'GPU-test', '--census', 'unused', '--out', 'unused', '--arm', 'auto', *extra])
    with pytest.raises(ValueError, match='reference and no active profiler'):
        _helper('exactness').main(after_exactness=lambda *args: None)
