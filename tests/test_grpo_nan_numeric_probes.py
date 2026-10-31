"""CPU-only assertions for the GRPO non-finite-gradient numerical probes."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

HARNESS = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "harness"
    / "grpo_nan_numeric_probes.py"
)


def _load_harness():
    name = "_grpo_nan_numeric_probes_test_module"
    spec = importlib.util.spec_from_file_location(name, HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


@pytest.fixture(scope="module")
def probe_run():
    torch = pytest.importorskip("torch")
    cuda_before = torch.cuda.is_initialized()
    results = _load_harness().run_probes()
    cuda_after = torch.cuda.is_initialized()
    return results, cuda_before, cuda_after


@pytest.fixture(scope="module")
def probe_results(probe_run):
    return probe_run[0]


def test_probe_runs_cpu_only_and_reports_its_torch_version(probe_run):
    results, cuda_before, cuda_after = probe_run
    if not cuda_before:
        assert not cuda_after
    assert results["torch_version"]


def test_split_k_bound_is_many_orders_below_fp32_overflow(probe_results):
    result = probe_results["split_k_bound"]
    assert result["k"] == 65_536
    assert result["operand_bound"] == 10.0
    assert result["bound"] == 6_553_600.0
    assert result["orders_below_fp32_max"] > 31.0


def test_dtype_range_probe_distinguishes_bf16_from_fp16(probe_results):
    result = probe_results["dtype_ranges"]
    assert result["fp16_max"] == pytest.approx(65_504.0)
    assert result["bf16_max"] > 1e38
    assert result["ln_bf16_max"] == pytest.approx(88.7189, abs=0.01)


def test_constructed_masked_exponential_has_finite_loss_and_nonfinite_gradient(probe_results):
    result = probe_results["constructed_masked_exponential"]
    assert result["loss_finite"] is True
    assert result["gradient_finite"] is False
    assert result["nonfinite_gradient_elements"] == 1
    assert result["nonfinite_positions"] == [2]


def test_detached_logp_ratio_is_one_and_has_finite_gradient(probe_results):
    result = probe_results["detached_logp_ratio"]
    assert result["ratio_all_one"] is True
    assert result["ratio_finite"] is True
    assert result["gradient_finite"] is True
    assert result["gradient_all_one"] is True


def test_boolean_selection_can_keep_an_existing_infinity_out_of_the_loss(probe_results):
    result = probe_results["boolean_selection"]
    assert result["loss_finite"] is True
    assert result["gradient_finite"] is True
    assert result["inactive_gradient"] == 0.0


def test_boolean_selection_after_overflowing_exp_has_nonfinite_gradient(probe_results):
    result = probe_results["selection_after_overflow"]
    assert result["loss_finite"] is True
    assert result["gradient_finite"] is False


def test_adamw_none_and_zero_gradient_have_different_step_semantics(probe_results):
    result = probe_results["adamw_skip_semantics"]
    assert result["none_grad_weight"] == pytest.approx(2.0)
    assert result["none_grad_state_entries"] == 0
    assert result["zero_grad_weight"] == pytest.approx(1.9)
    assert result["zero_grad_state_entries"] == 1


def test_nanmean_can_hide_one_nonfinite_contributor(probe_results):
    result = probe_results["nanmean_visibility"]
    assert result["input_has_nonfinite"] is True
    assert result["nanmean"] == pytest.approx(0.242)


def test_noncontiguous_slice_has_finite_ordinary_cpu_backward(probe_results):
    result = probe_results["view_backward"]
    assert result["slice_is_contiguous"] is False
    assert result["hidden_gradient_finite"] is True


def test_selftest_exits_one_and_names_the_failing_probe(monkeypatch, capsys):
    pytest.importorskip("torch")
    harness = _load_harness()
    results = harness.run_probes()
    results["view_backward"]["slice_is_contiguous"] = True
    monkeypatch.setattr(harness, "run_probes", lambda: results)

    assert harness.main(["--selftest", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["selftest_passed"] is False
    assert "view_backward" in payload["selftest_error"]


def test_selftest_json_is_machine_readable_and_passes():
    pytest.importorskip("torch")
    completed = subprocess.run(
        [sys.executable, str(HARNESS), "--self-test", "--json"],
        capture_output=True,
        check=False,
        encoding="utf-8",
        timeout=90,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["selftest_passed"] is True
    assert payload["execution_device"] == "cpu"
    assert payload["split_k_bound"]["bound"] == 6_553_600.0


def test_harness_keeps_torch_import_lazy_for_a_standalone_cpu_probe():
    source = HARNESS.read_text(encoding="utf-8")
    assert "\nimport torch\n" not in source
    assert "import torch" in source
