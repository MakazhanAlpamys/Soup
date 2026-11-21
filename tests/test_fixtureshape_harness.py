"""CPU-only tests for benchmarks/harness/fixtureshape.py (#379).

The CUDA measurement needs a card; these pin the parts that do not: the
dispatch reader, the direct three-hook counter, the per-shape window summary,
each fail-closed gate, the torch-only self-test, and that a no-CUDA default is
a SKIP rather than a false pass.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "harness" / "fixtureshape.py"


def _load_harness():
    name = "_fixtureshape_step13_test_module"
    spec = importlib.util.spec_from_file_location(name, _PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


harness = _load_harness()


class TestDispatchAndCounter:
    def test_above_the_row_cap_bitsandbytes_is_not_asked(self):
        def must_not_run(*args):
            raise AssertionError("bitsandbytes does not ask past its row cap")

        ops = SimpleNamespace(
            _gemm_4bit_use_custom_fn=must_not_run, _gemm_4bit_custom_max_m=1536
        )
        assert harness.selected_custom(ops, 0, "bf16", 2048, 4096, 4096, 64) is False

    def test_a_build_without_the_private_symbol_is_unknown_not_guessed(self):
        assert harness.selected_custom(SimpleNamespace(), 0, "bf16", 8, 4096, 4096, 64) is None
        assert harness.selected_custom(None, 0, "bf16", 8, 4096, 4096, 64) is None

    def test_the_counter_reads_the_real_hooks(self):
        ops = SimpleNamespace(
            _gemm_4bit_kernel_impl=lambda: None, _dequant_linear_fallback=lambda: None
        )
        fn = SimpleNamespace(dequantize_4bit=lambda: None)
        with harness.counter(ops, fn) as counts:
            ops._gemm_4bit_kernel_impl()
            ops._dequant_linear_fallback()
            fn.dequantize_4bit()
        assert counts["kernel"] == 1 and counts["fallback"] == 1 and counts["dequant"] == 1


class TestWindowsAndGates:
    @staticmethod
    def _rows():
        def row(**updates):
            base = {
                "set": "fixture", "N": 64, "K": 64, "M": 8,
                "kernel_calls": 1, "fallback_calls": 0,
                "forward": {"exact": False}, "gradient": {"exact": True},
                "control_detected": True,
            }
            base.update(updates)
            return base

        return (
            row(),
            row(M=64, kernel_calls=0, fallback_calls=1, forward={"exact": True}),
            row(set="control", N=4096, K=4096, kernel_calls=0,
                fallback_calls=1, forward={"exact": True}),
        )

    def test_windows_are_per_shape_and_per_path(self):
        fused, fallback, control = self._rows()
        got = harness.windows([fused, fallback, control], "fixture")
        assert got == {"64x64": {"kernel": [8], "fallback": [64]}}

    def test_every_gate_and_the_passing_case(self):
        fused, fallback, control = self._rows()
        assert harness.evaluate([fused, fallback, control])["exit_code"] == 0
        assert harness.evaluate([fallback, control])["exit_code"] == 2
        assert harness.evaluate([fused, control])["exit_code"] == 3
        bad_control = dict(fused, control_detected=False)
        assert harness.evaluate([bad_control, fallback, control])["exit_code"] == 4
        diff_on_fallback = dict(fallback, forward={"exact": False})
        assert harness.evaluate([fused, diff_on_fallback, control])["exit_code"] == 5
        bad_grad = dict(fused, gradient={"exact": False})
        assert harness.evaluate([bad_grad, fallback, control])["exit_code"] == 6


class TestEntryPoints:
    def test_self_test_passes(self, capsys):
        assert harness.main(["--self-test"]) == 0
        assert "--self-test: OK" in capsys.readouterr().out

    def test_no_cuda_is_a_skip_not_a_false_pass(self, monkeypatch, capsys):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert harness.main([]) == 0
        assert "CUDA is required" in capsys.readouterr().out
