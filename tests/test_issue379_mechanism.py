import importlib.util
import sys
from pathlib import Path

import pytest
import torch

MODULE_PATH = Path(__file__).parents[1] / "benchmarks" / "harness" / "mechanism.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("mechanism_harness", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def gradients(*values):
    return {f"0.p{index}": torch.tensor([value]) for index, value in enumerate(values)}


def test_exact_gradient_count_distinguishes_exact_and_mismatch():
    reference = gradients(1.0, 2.0)

    assert module.exact_gradient_count(reference, gradients(1.0, 2.0)) == (2, 2)
    assert module.exact_gradient_count(reference, gradients(1.0, 3.0)) == (1, 2)
    assert module.max_gradient_diff(reference, gradients(1.0, 3.0)) == pytest.approx(1.0)


def test_gradient_comparison_rejects_non_finite_values():
    with pytest.raises(RuntimeError, match="non-finite"):
        module.exact_gradient_count(gradients(float("nan")), gradients(float("nan")))

    with pytest.raises(RuntimeError, match="non-finite"):
        module.max_gradient_diff(gradients(1.0), gradients(float("inf")))


def test_gradient_comparison_rejects_empty_or_different_parameter_sets():
    with pytest.raises(RuntimeError, match="key sets differ"):
        module.exact_gradient_count({"resident.p": torch.ones(1)}, {"streamed.p": torch.ones(1)})


def _patch_successful_measurement(monkeypatch):
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module.torch.cuda, "get_device_name", lambda _index: "test-gpu")
    monkeypatch.setattr(module, "load_sources", lambda: object())
    monkeypatch.setattr(module, "make_lora_parameters", lambda: object())
    monkeypatch.setattr(module, "make_arm_state", lambda *_args: object())

    def run_once(_sources, _state, arm, *, bypass_pool):
        if arm == "clone" and bypass_pool:
            return gradients(1.0)
        return gradients(2.0 if arm != "clone" else 1.0)

    monkeypatch.setattr(module, "run_once", run_once)


def test_main_returns_success_for_a_reproduced_mechanism(monkeypatch):
    _patch_successful_measurement(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["mechanism.py"])

    assert module.main() == 0


def test_main_rejects_a_weakened_no_mismatch_verdict(monkeypatch):
    _patch_successful_measurement(monkeypatch)
    monkeypatch.setattr(module, "run_once", lambda *_args, **_kwargs: gradients(1.0))
    monkeypatch.setattr(sys, "argv", ["mechanism.py"])

    assert module.main() == 1


def test_bypass_pool_negative_control_is_detected(monkeypatch):
    _patch_successful_measurement(monkeypatch)
    monkeypatch.setattr(
        module,
        "run_once",
        lambda _sources, _state, arm, *, bypass_pool: gradients(
            1.0 if bypass_pool or arm == "clone" else 2.0
        ),
    )
    monkeypatch.setattr(sys, "argv", ["mechanism.py", "--bypass-pool"])

    assert module.main() == 1
