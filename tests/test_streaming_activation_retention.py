"""Activation retention must detect mutations caused by recycled weight buffers."""

import pytest

from benchmarks.harness.streaming_residency_controls import activation_retention as candidate
from tests.test_streaming_residency_candidate import fixture, trajectory


@pytest.mark.parametrize("arm", ["baseline", "resident", "retained", "both"])
@pytest.mark.parametrize("indices", [[2], [2, 3], [0, 1, 4, 5]])
def test_four_arm_exact_trajectory(arm, indices):
    import torch

    from benchmarks.harness.streaming_residency_candidate import install_persistent_layers

    reference, native = fixture()
    model, runtime = fixture()
    if arm in {"resident", "both"}:
        install_persistent_layers(runtime, indices)
    state = None
    if arm in {"retained", "both"}:
        state = candidate.install_retention(model, indices, resident=arm == "both")
    expected = trajectory(reference, native)
    actual = trajectory(model, runtime)
    assert all(torch.equal(left, right) for left, right in zip(expected, actual, strict=True))
    if state:
        assert all(value == 4 for value in state["body_calls"].values())
        assert state["retained_forward_calls"] == len(indices) * 4
        assert bool(state["weight_snapshot_bytes"]) == (arm == "retained")
        assert bool(state["borrowed_weight_bytes"]) == (arm == "both")


def test_unsafe_checkpoint_bypass_mutation_is_killed():
    import torch

    reference, native = fixture()
    model, runtime = fixture()
    model[2].use_checkpoint = False
    expected = trajectory(reference, native)
    try:
        actual = trajectory(model, runtime)
    except RuntimeError as error:
        assert "inplace" in str(error) or "modified" in str(error)
        return
    assert any(not torch.equal(left, right)
               for left, right in zip(expected, actual, strict=True))


def test_refuse_falsely_claimed_resident_storage():
    model, _runtime = fixture()
    with pytest.raises(ValueError, match="ownership"):
        candidate.install_retention(model, [2], resident=True)


def test_missing_weight_snapshot_mutation_is_killed(monkeypatch):
    import torch

    monkeypatch.setattr(candidate, "_snapshot", lambda value: value.detach())
    reference, native = fixture()
    model, runtime = fixture()
    candidate.install_retention(model, [2, 3], resident=False)
    expected = trajectory(reference, native)
    actual = trajectory(model, runtime)
    assert any(not torch.equal(left, right)
               for left, right in zip(expected, actual, strict=True))


def test_missing_backward_prefetch_repair_mutation_is_killed():
    model, runtime = fixture()
    original = model[1]._body
    candidate.install_retention(model, [2], resident=False)
    model[1]._body = original
    with pytest.raises(RuntimeError, match="scheduler bug"):
        trajectory(model, runtime)
