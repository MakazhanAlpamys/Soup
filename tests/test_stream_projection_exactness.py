"""Fail-closed comparisons for full-model projection-retention receipts."""
import importlib.util
from pathlib import Path

import pytest


def _helper():
    path = Path(__file__).parents[1] / 'benchmarks/harness/stream_projection_exactness.py'
    spec = importlib.util.spec_from_file_location('projection_exactness', path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def test_tensor_records_bind_values_shape_and_dtype():
    import torch
    record = _helper().tensor_record
    tensor = torch.arange(12, dtype=torch.float32)
    assert record(tensor) == record(tensor.clone())
    assert record(tensor) != record(tensor.reshape(3, 4))
    assert record(tensor) != record(tensor.to(torch.float64))
    changed = tensor.clone()
    changed[-1] += 1
    assert record(tensor) != record(changed)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_tensor_cannot_be_an_exactness_reference(value):
    import torch
    with pytest.raises(RuntimeError, match='nonfinite'):
        _helper().tensor_record(torch.tensor([value]))


def test_missing_gradient_is_refused():
    with pytest.raises(RuntimeError, match='missing'):
        _helper().tensor_record(None)


def test_exactness_requires_both_complete_steps_and_identical_adapter_sets():
    helper = _helper()
    step = {'logits': 'a', 'loss': 'b', 'gradients': {'adapter': 'c'},
            'parameters': {'adapter': 'd'}}
    reference = [step, step]
    helper.check_steps(reference, reference)
    with pytest.raises(RuntimeError, match='two'):
        helper.check_steps(reference, reference[:1])
    corrupt = step | {'gradients': {}}
    with pytest.raises(RuntimeError, match='exact'):
        helper.check_steps(reference, [step, corrupt])
    corrupt = step | {'parameters': {'adapter': 'e'}}
    with pytest.raises(RuntimeError, match='exact'):
        helper.check_steps(reference, [step, corrupt])


def test_matching_but_incomplete_receipts_are_not_exactness_proof():
    with pytest.raises(RuntimeError, match='incomplete'):
        _helper().check_steps([{}, {}], [{}, {}])
