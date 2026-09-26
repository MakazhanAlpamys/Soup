"""Load-bearing tests for the CPU fixture-window mechanism harness (#379)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

HARNESS = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "harness"
    / "fixture_window_cpu.py"
)


def _load_harness():
    name = "_fixture_window_cpu_test_module"
    spec = importlib.util.spec_from_file_location(name, HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _row(
    harness,
    *,
    inference_packed=False,
    training_packed=False,
    inference_diff=0.0,
    training_diff=0.0,
    variant2_abs=100.0,
    inference_rel=None,
    emulated_bf16_rel=0.004,
):
    if inference_rel is None:
        inference_rel = inference_diff / variant2_abs if variant2_abs else float("inf")
    return harness.Row(
        out_features=64,
        in_features=64,
        m=8,
        inference_packed_for_cpu=inference_packed,
        training_packed_for_cpu=training_packed,
        variant2_max_abs=variant2_abs,
        emulated_bf16_vs_variant2_max_abs=emulated_bf16_rel * variant2_abs,
        emulated_bf16_rel=emulated_bf16_rel,
        inference_vs_variant2_max_abs=inference_diff,
        inference_vs_variant2_rel=inference_rel,
        training_vs_variant2_max_abs=training_diff,
        inference_vs_training_max_abs=max(inference_diff, training_diff),
    )


def _result(harness, rows):
    verdict = harness._evaluate_rows(rows)
    return {
        "rows": [row.__dict__ for row in rows],
        "packed_inference_rows": sum(row.inference_packed_for_cpu for row in rows),
        "packed_training_rows": sum(row.training_packed_for_cpu for row in rows),
        "training_control_exact": all(
            row.training_vs_variant2_max_abs == 0.0 for row in rows
        ),
        **verdict,
    }


def test_cpu_recorded_shapes_and_m_grid_are_pinned():
    harness = _load_harness()
    assert harness.FIXTURE_SHAPES == ((64, 64), (256, 64))
    assert harness.M_VALUES == (8, 16, 32, 64, 128, 256)


def test_real_row_uses_shipped_variant2_and_exact_training_control(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("bitsandbytes")

    harness = _load_harness()
    import soup_cli.utils.layer_stream_runtime as runtime

    real_install = runtime.install_dequant_forward
    calls = {"n": 0}

    def counted_install(module):
        calls["n"] += 1
        return real_install(module)

    monkeypatch.setattr(runtime, "install_dequant_forward", counted_install)
    row = harness.measure_row(64, 64, 8, seed=3)

    assert calls["n"] == 1
    assert row.training_packed_for_cpu is False
    assert row.training_vs_variant2_max_abs == 0.0
    assert row.variant2_max_abs > 0.0
    # One bf16 rounding of input, weight and output: a few 1e-3 relative, never 0.
    assert 1e-3 < row.emulated_bf16_rel < harness.PACKED_EFFECT_REL_ENVELOPE
    assert row.emulated_bf16_rel == pytest.approx(
        row.emulated_bf16_vs_variant2_max_abs / row.variant2_max_abs
    )


def test_nf4_fixture_uses_double_quant():
    torch = pytest.importorskip("torch")
    pytest.importorskip("bitsandbytes")

    harness = _load_harness()
    layer = harness._quantized_linear(
        torch.randn(64, 64, dtype=torch.float32),
        compute_dtype=torch.float32,
    )
    assert layer.weight.quant_state.nested is True


def test_measure_row_requires_eval_inference_and_reports_packing(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("bitsandbytes")

    harness = _load_harness()
    import soup_cli.utils.layer_stream_runtime as runtime

    events = []

    class QuantState:
        packing_format_for_cpu = False

    class Weight:
        def __init__(self):
            self.quant_state = QuantState()

    class FakeLayer:
        def __init__(self):
            self.training = True
            self.weight = Weight()

        def train(self):
            self.training = True
            events.append("train")
            return self

        def eval(self):
            self.training = False
            events.append("eval")
            return self

        def __call__(self, x):
            if not self.training and not x.requires_grad:
                self.weight.quant_state.packing_format_for_cpu = True
                return x + 1.0
            return x

    monkeypatch.setattr(harness, "_quantized_linear", lambda *a, **k: FakeLayer())
    monkeypatch.setattr(runtime, "install_dequant_forward", lambda _module: 1)
    monkeypatch.setattr(harness, "_emulated_bf16_reference", lambda _layer, x: x)

    row = harness.measure_row(4, 4, 2, seed=5)

    generator = torch.Generator().manual_seed(harness._row_seed(5, 4, 4, 2))
    torch.randn((4, 4), generator=generator, dtype=torch.float32)
    x = torch.randn((2, 4), generator=generator, dtype=torch.float32)
    expected_max = float(x.abs().max())

    assert "eval" in events
    assert row.inference_packed_for_cpu is True
    assert row.training_packed_for_cpu is False
    assert row.training_vs_variant2_max_abs == 0.0
    assert row.variant2_max_abs == pytest.approx(expected_max)
    assert row.inference_vs_variant2_max_abs == 1.0
    assert row.inference_vs_variant2_rel == pytest.approx(1.0 / expected_max)


def test_run_probe_counts_inference_packed_not_training_packed(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("bitsandbytes")
    harness = _load_harness()

    monkeypatch.setattr(
        harness,
        "measure_row",
        lambda out_features, in_features, m: _row(
            harness,
            inference_packed=True,
            training_packed=False,
            inference_diff=1.0,
        ),
    )
    result = harness.run_probe(
        m_values=(8, 32),
        shapes=((64, 64),),
    )
    assert result["packed_inference_rows"] == 2
    assert result["packed_training_rows"] == 0
    assert result["verdict"] == "mechanism_reproduced"


@pytest.mark.parametrize(
    "case,expected_code,expected_verdict",
    [
        ("absent", 2, "mechanism_absent"),
        ("training_packed", 3, "control_failed"),
        ("training_diff", 3, "control_failed"),
        ("packed_no_diff", 4, "packed_effect_absent"),
        ("packed_too_large", 5, "packed_effect_out_of_range"),
        ("success", 0, "mechanism_reproduced"),
    ],
)
def test_main_pins_every_gate_exit_code(
    monkeypatch,
    capsys,
    case,
    expected_code,
    expected_verdict,
):
    harness = _load_harness()

    if case == "absent":
        row_list = [_row(harness)]
    elif case == "training_packed":
        row_list = [
            _row(
                harness,
                inference_packed=True,
                training_packed=True,
                inference_diff=1.0,
            )
        ]
    elif case == "training_diff":
        row_list = [
            _row(
                harness,
                inference_packed=True,
                inference_diff=1.0,
                training_diff=0.25,
            )
        ]
    elif case == "packed_no_diff":
        row_list = [_row(harness, inference_packed=True)]
    elif case == "packed_too_large":
        row_list = [_row(harness, inference_packed=True, inference_diff=3.0)]
    else:
        row_list = [
            _row(
                harness,
                inference_packed=True,
                inference_diff=1.0,
            )
        ]

    result = _result(harness, row_list)
    monkeypatch.setattr(harness, "run_probe", lambda: result)

    code = harness.main([])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert code == expected_code
    assert payload["exit_code"] == expected_code
    assert payload["verdict"] == expected_verdict
    if expected_code:
        assert captured.err.strip() == payload["reason"]
    else:
        assert captured.err == ""


def test_survey_is_explicit_opt_out_of_missing_capability_only(monkeypatch, capsys):
    harness = _load_harness()
    result = _result(harness, [_row(harness)])
    assert result["exit_code"] == harness.EXIT_MECHANISM_ABSENT
    monkeypatch.setattr(harness, "run_probe", lambda: result)

    assert harness.main(["--survey"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["verdict"] == "mechanism_absent"
    assert captured.err == ""


def test_survey_does_not_waive_a_broken_control(monkeypatch, capsys):
    harness = _load_harness()
    result = _result(harness, [_row(harness, training_diff=0.25)])
    assert result["exit_code"] == harness.EXIT_CONTROL_FAILED
    monkeypatch.setattr(harness, "run_probe", lambda: result)

    assert harness.main(["--survey"]) == harness.EXIT_CONTROL_FAILED
    captured = capsys.readouterr()
    assert json.loads(captured.out)["verdict"] == "control_failed"
    assert captured.err.strip() == result["reason"]


def test_json_schema_carries_verdict_and_attribution():
    harness = _load_harness()
    rows = [
        _row(
            harness,
            inference_packed=True,
            inference_diff=1.0,
        )
    ]
    verdict = harness._evaluate_rows(rows)
    assert verdict == {
        "verdict": "mechanism_reproduced",
        "exit_code": 0,
        "reason": (
            "every packed row differs from Soup variant 2 within the 2% "
            "rounding-scale envelope while the training control is exact."
        ),
        "attribution": (
            "bitsandbytes packed CPU inference path; exact lower-level kernel "
            "reported separately"
        ),
    }


def test_loader_does_not_leak_dynamic_module_registration():
    _load_harness()
    assert "_fixture_window_cpu_test_module" not in sys.modules


def test_script_documents_avx512_and_kernels_attribution_boundary():
    source = HARNESS.read_text(encoding="utf-8")
    assert "AVX512-BF16 is required" in source
    assert "kernels-community/quantization-bitsandbytes" in source
    assert "best-effort lower-level kernel attribution" in source


def test_control_is_checked_on_every_row_before_capability():
    harness = _load_harness()
    rows = [
        _row(harness),
        _row(harness, training_diff=0.125),
    ]
    verdict = harness._evaluate_rows(rows)
    assert verdict["verdict"] == "control_failed"
    assert verdict["exit_code"] == harness.EXIT_CONTROL_FAILED


def test_packed_effect_must_exist_on_every_packed_row():
    harness = _load_harness()
    rows = [
        _row(harness, inference_packed=True, inference_diff=1.0),
        _row(harness, inference_packed=True, inference_diff=0.0),
    ]
    verdict = harness._evaluate_rows(rows)
    assert verdict["verdict"] == "packed_effect_absent"


def test_packed_effect_must_stay_inside_relative_envelope_on_every_row():
    harness = _load_harness()
    rows = [
        _row(harness, inference_packed=True, inference_diff=1.0),
        _row(harness, inference_packed=True, inference_diff=3.0),
    ]
    verdict = harness._evaluate_rows(rows)
    assert verdict["verdict"] == "packed_effect_out_of_range"
    assert verdict["exit_code"] == harness.EXIT_EFFECT_OUT_OF_RANGE


def test_default_run_probe_visits_the_complete_recorded_grid(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("bitsandbytes")
    harness = _load_harness()
    seen = []

    def fake_measure(out_features, in_features, m):
        seen.append((out_features, in_features, m))
        return _row(harness)

    monkeypatch.setattr(harness, "measure_row", fake_measure)
    result = harness.run_probe()
    expected = {
        (out_features, in_features, m)
        for out_features, in_features in harness.FIXTURE_SHAPES
        for m in harness.M_VALUES
    }
    assert set(seen) == expected
    assert len(seen) == len(expected) == 12
    assert len(result["rows"]) == 12


def test_measure_row_refuses_when_variant2_patch_does_not_apply(monkeypatch):
    pytest.importorskip("torch")
    harness = _load_harness()
    import soup_cli.utils.layer_stream_runtime as runtime

    monkeypatch.setattr(harness, "_quantized_linear", lambda *a, **k: object())
    monkeypatch.setattr(runtime, "install_dequant_forward", lambda _module: 0)
    with pytest.raises(RuntimeError, match="patched 0 layers, expected 1"):
        harness.measure_row(64, 64, 8)


def test_harness_source_has_no_model_or_dataset_download_calls():
    source = HARNESS.read_text(encoding="utf-8")
    forbidden = ("from_pretrained(", "load_dataset(", "hf_hub_download(", "requests.")
    assert not any(token in source for token in forbidden)


@pytest.mark.parametrize(
    ("scale", "expected_code"),
    [
        (3.0, 5),
        (1.001, 0),
    ],
)
def test_real_measurement_drives_relative_envelope(
    monkeypatch, scale, expected_code
):
    pytest.importorskip("torch")
    pytest.importorskip("bitsandbytes")
    harness = _load_harness()
    import soup_cli.utils.layer_stream_runtime as runtime

    class QuantState:
        packing_format_for_cpu = False

    class Weight:
        def __init__(self):
            self.quant_state = QuantState()

    class FakeLayer:
        def __init__(self):
            self.training = True
            self.weight = Weight()

        def train(self):
            self.training = True
            return self

        def eval(self):
            self.training = False
            return self

        def __call__(self, x):
            if not self.training and not x.requires_grad:
                self.weight.quant_state.packing_format_for_cpu = True
                return x * scale
            return x

    monkeypatch.setattr(harness, "_quantized_linear", lambda *a, **k: FakeLayer())
    monkeypatch.setattr(runtime, "install_dequant_forward", lambda _module: 1)
    monkeypatch.setattr(harness, "_emulated_bf16_reference", lambda _layer, x: x)

    row = harness.measure_row(4, 4, 2, seed=11)
    verdict = harness._evaluate_rows([row])
    assert verdict["exit_code"] == expected_code
    if expected_code == 0:
        assert row.inference_vs_variant2_rel == pytest.approx(0.001, rel=1e-4)
    else:
        assert row.inference_vs_variant2_rel == pytest.approx(2.0)


def test_envelope_boundary_is_pinned_exactly():
    harness = _load_harness()
    at_limit = _row(
        harness,
        inference_packed=True,
        inference_diff=2.0,
        variant2_abs=100.0,
        inference_rel=harness.PACKED_EFFECT_REL_ENVELOPE,
    )
    over_limit = _row(
        harness,
        inference_packed=True,
        inference_diff=2.0001,
        variant2_abs=100.0,
        inference_rel=harness.PACKED_EFFECT_REL_ENVELOPE + 1e-6,
    )
    assert harness._evaluate_rows([at_limit])["exit_code"] == harness.EXIT_OK
    assert (
        harness._evaluate_rows([over_limit])["exit_code"]
        == harness.EXIT_EFFECT_OUT_OF_RANGE
    )


def test_run_probe_publishes_emulated_bf16_margin(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("bitsandbytes")
    harness = _load_harness()
    rows = iter(
        [
            _row(harness, emulated_bf16_rel=0.003),
            _row(harness, emulated_bf16_rel=0.0048),
        ]
    )
    monkeypatch.setattr(harness, "measure_row", lambda *a, **k: next(rows))

    result = harness.run_probe(m_values=(8, 16), shapes=((64, 64),))
    assert result["emulated_bf16_worst_rel"] == pytest.approx(0.0048)
    assert result["packed_effect_rel_envelope"] == 0.02
    assert result["envelope_over_emulated_worst"] == pytest.approx(
        0.02 / 0.0048
    )
    assert (
        result["packed_effect_rel_envelope"]
        > result["emulated_bf16_worst_rel"]
    )


def test_row_seed_uses_full_tuple_and_breaks_old_collision():
    harness = _load_harness()
    # The old seed + out_features + m formula collides for these two rows.
    assert 64 + 256 == 256 + 64
    left = harness._row_seed(17, 64, 64, 256)
    right = harness._row_seed(17, 256, 64, 64)
    assert left != right


def test_one_packed_training_row_fails_even_next_to_clean_row():
    harness = _load_harness()
    rows = [
        _row(harness, training_packed=True),
        _row(harness, training_packed=False),
    ]
    verdict = harness._evaluate_rows(rows)
    assert verdict["exit_code"] == harness.EXIT_CONTROL_FAILED


def test_cpu_kernel_attribution_states(monkeypatch):
    pytest.importorskip("bitsandbytes")
    import bitsandbytes.backends.cpu.ops as cpu_ops

    harness = _load_harness()
    monkeypatch.setattr(harness, "_cpu_gemv_registered", lambda: False)
    monkeypatch.delattr(cpu_ops, "gemm_4bit_forward_kernel", raising=False)
    assert harness._cpu_packed_kernel_attribution() == "not-registered"

    monkeypatch.setattr(cpu_ops, "gemm_4bit_forward_kernel", None, raising=False)
    assert (
        harness._cpu_packed_kernel_attribution()
        == "native-bitsandbytes-cpu-gemv"
    )

    monkeypatch.setattr(
        cpu_ops,
        "gemm_4bit_forward_kernel",
        lambda *a, **k: None,
        raising=False,
    )
    assert harness._cpu_packed_kernel_attribution() == "kernels-community"


def test_run_probe_reports_kernel_attribution_fields(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("bitsandbytes")
    harness = _load_harness()
    monkeypatch.setattr(
        harness,
        "measure_row",
        lambda *a, **k: _row(harness),
    )
    monkeypatch.setattr(
        harness, "_cpu_packed_kernel_attribution", lambda: "not-registered"
    )
    monkeypatch.setattr(harness, "_cpu_gemv_registered", lambda: False)

    result = harness.run_probe(m_values=(8,), shapes=((64, 64),))
    assert result["cpu_packed_kernel"] == "not-registered"
    assert result["cpu_gemv_4bit_registered"] is False


def test_real_avx512_host_runs_default_gate_when_available(capsys):
    functional = pytest.importorskip("bitsandbytes.functional")
    if not functional.has_avx512bf16():
        pytest.skip("requires an AVX512-BF16 host")

    harness = _load_harness()
    code = harness.main([])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    with capsys.disabled():
        print(captured.out)
    assert payload["has_avx512bf16"] is True
    assert payload["packed_inference_rows"] > 0
    assert code == harness.EXIT_OK
