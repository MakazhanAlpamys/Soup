"""CPU-only tests for the three STEP 13 / #379 numerics harnesses.

The measurements themselves need a CUDA card (and are recorded in the harness
JSONs). What can be pinned here, with torch only: the dispatch reader, the
direct fallback counter, the forced-fused patch on a fake module, the fail-closed
gates, and that each harness's ``--self-test`` passes and a no-CUDA host is a
SKIP, not a false pass.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

_HARNESS = Path(__file__).resolve().parents[1] / "benchmarks" / "harness"


def _load(name):
    path = _HARNESS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


numerics = _load("numerics")
numerics2 = _load("numerics2")
forcecheck = _load("forcecheck")

HARNESSES = (numerics, numerics2, forcecheck)


class TestSelectedCustom:
    @pytest.mark.parametrize("harness", HARNESSES)
    def test_above_the_row_cap_bitsandbytes_is_not_asked(self, harness):
        def must_not_run(*args):
            raise AssertionError("bitsandbytes does not ask past its row cap")

        ops = SimpleNamespace(
            _gemm_4bit_use_custom_fn=must_not_run, _gemm_4bit_custom_max_m=1536
        )
        assert harness.selected_custom(ops, 0, "bf16", 2048, 4096, 4096, 64) is False

    @pytest.mark.parametrize("harness", HARNESSES)
    def test_a_misaligned_inner_dimension_falls_back_without_asking(self, harness):
        def must_not_run(*args):
            raise AssertionError("misaligned K never reaches the custom kernel")

        ops = SimpleNamespace(
            _gemm_4bit_use_custom_fn=must_not_run, _gemm_4bit_custom_max_m=1536
        )
        assert harness.selected_custom(ops, 0, "bf16", 8, 4096, 4100, 64) is False

    @pytest.mark.parametrize("harness", HARNESSES)
    def test_a_build_without_the_private_symbol_is_unknown_not_guessed(self, harness):
        assert harness.selected_custom(SimpleNamespace(), 0, "bf16", 8, 4096, 4096, 64) is None
        assert harness.selected_custom(None, 0, "bf16", 8, 4096, 4096, 64) is None


class TestCounterAndForcing:
    @pytest.mark.parametrize("harness", HARNESSES)
    def test_the_counter_observes_the_real_hook_not_a_gate(self, harness):
        ops = SimpleNamespace(
            _gemm_4bit_kernel_impl=lambda: None, _dequant_linear_fallback=lambda: None
        )
        fn = SimpleNamespace(dequantize_4bit=lambda: None)
        with harness.counter(ops, fn) as counts:
            ops._dequant_linear_fallback()
            fn.dequantize_4bit()
        assert counts["fallback"] == 1
        assert counts["dequant"] == 1
        assert counts["kernel"] == 0

    @pytest.mark.parametrize("harness", (numerics2, forcecheck))
    def test_force_fused_is_scoped_and_restores_the_globals(self, harness):
        ops = SimpleNamespace(
            _gemm_4bit_use_custom_fn=lambda *args: False, _gemm_4bit_custom_max_m=1536
        )
        assert harness.selected_custom(ops, 0, "bf16", 2048, 4096, 4096, 64) is False
        with harness.force_fused(ops):
            assert harness.selected_custom(ops, 0, "bf16", 2048, 4096, 4096, 64) is True
        assert harness.selected_custom(ops, 0, "bf16", 2048, 4096, 4096, 64) is False
        assert ops._gemm_4bit_custom_max_m == 1536


class TestStats:
    def test_numerics_stats_counts_bit_differences_in_float32(self):
        import torch

        left = torch.tensor([60000.0], dtype=torch.float16)
        right = torch.tensor([-60000.0], dtype=torch.float16)
        s = numerics.stats(left, right)
        assert s["exact"] is False
        assert s["max_abs"] == 120000.0  # a float16 subtraction would overflow to inf


class TestGates:
    def test_numerics_vacuity_gate(self):
        good = {
            "M": 2048,
            "bnb_selected_custom_kernel": False,
            "resident_fallback_calls": 1,
            "forward": {"exact": True},
            "gradient": {"exact": True},
            "control_detected": True,
        }
        assert numerics.evaluate([good])["exit_code"] == 0
        assert numerics.evaluate([dict(good, bnb_selected_custom_kernel=True)])["exit_code"] == 2
        assert numerics.evaluate([dict(good, control_detected=False)])["exit_code"] == 3
        assert numerics.evaluate([dict(good, gradient={"exact": False})])["exit_code"] == 4

    def test_forcecheck_path_control_gate(self):
        large = {
            "set": "default", "fallback_calls": 1, "kernel_calls": 0, "dequant_wrapper_calls": 0
        }
        small = {
            "set": "default", "fallback_calls": 0, "kernel_calls": 1, "dequant_wrapper_calls": 0
        }
        forced = {
            "set": "forced", "fallback_calls": 0, "kernel_calls": 1, "dequant_wrapper_calls": 0
        }
        assert forcecheck.evaluate([large, small, forced])["exit_code"] == 0
        assert forcecheck.evaluate([small, forced])["exit_code"] == 2
        blind = dict(large, dequant_wrapper_calls=2)
        assert forcecheck.evaluate([blind, small, forced])["exit_code"] == 3
        assert forcecheck.evaluate([large])["exit_code"] == 4
        assert forcecheck.evaluate([large, small, dict(forced, fallback_calls=1)])["exit_code"] == 5


class TestSelfTestAndNoCuda:
    @pytest.mark.parametrize("harness", HARNESSES)
    def test_self_test_passes(self, harness, capsys):
        assert harness.main(["--self-test"]) == 0
        assert "--self-test: OK" in capsys.readouterr().out

    @pytest.mark.parametrize("harness", HARNESSES)
    def test_no_cuda_is_a_skip_not_a_false_pass(self, harness, monkeypatch, capsys):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert harness.main([]) == 0
        assert "CUDA is required" in capsys.readouterr().out
