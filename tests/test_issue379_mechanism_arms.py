"""Pin every mechanism-arm verdict in the public harness decision path."""

import sys

from tests.test_issue379_mechanism import (
    _patch_successful_measurement,
    gradients,
    module,
)


def _run_with_exact_arm(monkeypatch, exact_arm):
    _patch_successful_measurement(monkeypatch)

    def run_once(_sources, _state, arm, *, bypass_pool):
        if arm == "clone":
            return gradients(1.0)
        return gradients(1.0 if arm == exact_arm else 2.0)

    monkeypatch.setattr(module, "run_once", run_once)
    monkeypatch.setattr(sys, "argv", ["mechanism.py"])
    return module.main()


def test_control_must_reproduce_a_mismatch(monkeypatch):
    assert _run_with_exact_arm(monkeypatch, "control") == 1


def test_sync_must_not_remove_the_mismatch(monkeypatch):
    assert _run_with_exact_arm(monkeypatch, "sync") == 1


def test_clone_quantstate_must_not_remove_the_mismatch(monkeypatch):
    assert _run_with_exact_arm(monkeypatch, "clone_quantstate") == 1


def test_clone_packed_must_not_remove_the_mismatch(monkeypatch):
    assert _run_with_exact_arm(monkeypatch, "clone_packed") == 1


def test_control_must_remain_wrong_after_repetition(monkeypatch):
    _patch_successful_measurement(monkeypatch)
    calls = 0

    def run_once(_sources, _state, arm, *, bypass_pool):
        nonlocal calls
        if arm == "clone":
            return gradients(1.0)
        if arm == "control":
            calls += 1
            return gradients(1.0 if calls == module.CORRECTNESS_REPEATS else 2.0)
        return gradients(2.0)

    monkeypatch.setattr(module, "run_once", run_once)
    monkeypatch.setattr(sys, "argv", ["mechanism.py"])
    assert module.main() == 1
