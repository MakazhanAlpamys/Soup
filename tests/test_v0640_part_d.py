"""v0.64.0 Part D — Hardware-fit calculator tests."""

from __future__ import annotations

import dataclasses
import math

import pytest

# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------


def test_module_imports():
    from soup_cli.utils import hardware_fit

    assert hasattr(hardware_fit, "HardwareFitInput")
    assert hasattr(hardware_fit, "VRAMBreakdown")
    assert hasattr(hardware_fit, "HardwareFitReport")
    assert hasattr(hardware_fit, "estimate_peak_vram_gb")
    assert hasattr(hardware_fit, "decide_hardware_fit")
    assert hasattr(hardware_fit, "validate_seq_len")
    assert hasattr(hardware_fit, "validate_batch_size")
    assert hasattr(hardware_fit, "VRAM_SAFETY_MARGIN")


# ---------------------------------------------------------------------------
# VRAM_SAFETY_MARGIN
# ---------------------------------------------------------------------------


def test_safety_margin_is_10pct():
    from soup_cli.utils.hardware_fit import VRAM_SAFETY_MARGIN

    assert VRAM_SAFETY_MARGIN == pytest.approx(0.10, abs=1e-6)


# ---------------------------------------------------------------------------
# validate_seq_len
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("v", [64, 1024, 8192, 1_048_576])
def test_validate_seq_len_happy(v):
    from soup_cli.utils.hardware_fit import validate_seq_len

    assert validate_seq_len(v) == v


@pytest.mark.parametrize("bad", [True, False, "1024", -1, 0, 63, 1_048_577, 1.5])
def test_validate_seq_len_rejects(bad):
    from soup_cli.utils.hardware_fit import validate_seq_len

    with pytest.raises((TypeError, ValueError)):
        validate_seq_len(bad)


# ---------------------------------------------------------------------------
# validate_batch_size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("v", [1, 4, 64, 1024])
def test_validate_batch_size_happy(v):
    from soup_cli.utils.hardware_fit import validate_batch_size

    assert validate_batch_size(v) == v


@pytest.mark.parametrize("bad", [True, False, "4", -1, 0, 1025, 1.5])
def test_validate_batch_size_rejects(bad):
    from soup_cli.utils.hardware_fit import validate_batch_size

    with pytest.raises((TypeError, ValueError)):
        validate_batch_size(bad)


# ---------------------------------------------------------------------------
# HardwareFitInput
# ---------------------------------------------------------------------------


def test_input_frozen():
    from soup_cli.utils.hardware_fit import HardwareFitInput

    inp = HardwareFitInput(
        params_b=7.0, seq_len=2048, batch_size=4,
        optimizer="adamw_torch", quant="4bit", peft="lora",
        gradient_checkpointing=True,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        inp.batch_size = 1  # type: ignore[misc]


def test_input_rejects_negative_params():
    from soup_cli.utils.hardware_fit import HardwareFitInput

    with pytest.raises(ValueError, match="params"):
        HardwareFitInput(
            params_b=-1.0, seq_len=2048, batch_size=4,
            optimizer="adamw_torch", quant="4bit", peft="lora",
            gradient_checkpointing=True,
        )


def test_input_rejects_invalid_quant():
    from soup_cli.utils.hardware_fit import HardwareFitInput

    with pytest.raises(ValueError, match="quant"):
        HardwareFitInput(
            params_b=7.0, seq_len=2048, batch_size=4,
            optimizer="adamw_torch", quant="bogus", peft="lora",
            gradient_checkpointing=True,
        )


def test_input_rejects_invalid_peft():
    from soup_cli.utils.hardware_fit import HardwareFitInput

    with pytest.raises(ValueError, match="peft"):
        HardwareFitInput(
            params_b=7.0, seq_len=2048, batch_size=4,
            optimizer="adamw_torch", quant="4bit", peft="bogus",
            gradient_checkpointing=True,
        )


def test_input_rejects_bool_params():
    from soup_cli.utils.hardware_fit import HardwareFitInput

    with pytest.raises(TypeError, match="bool"):
        HardwareFitInput(
            params_b=True, seq_len=2048, batch_size=4,  # type: ignore[arg-type]
            optimizer="adamw_torch", quant="4bit", peft="lora",
            gradient_checkpointing=True,
        )


def test_input_rejects_non_bool_gradient_ckpt():
    from soup_cli.utils.hardware_fit import HardwareFitInput

    with pytest.raises(TypeError, match="gradient_checkpointing"):
        HardwareFitInput(
            params_b=7.0, seq_len=2048, batch_size=4,
            optimizer="adamw_torch", quant="4bit", peft="lora",
            gradient_checkpointing=1,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# VRAMBreakdown
# ---------------------------------------------------------------------------


def test_vram_breakdown_frozen():
    from soup_cli.utils.hardware_fit import VRAMBreakdown

    v = VRAMBreakdown(
        weights_gb=2.0,
        optimizer_gb=0.5,
        gradients_gb=0.5,
        activations_gb=1.0,
        overhead_gb=0.5,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.weights_gb = 0.0  # type: ignore[misc]


def test_vram_breakdown_rejects_negative():
    from soup_cli.utils.hardware_fit import VRAMBreakdown

    with pytest.raises(ValueError, match="negative"):
        VRAMBreakdown(
            weights_gb=-1.0,
            optimizer_gb=0.5,
            gradients_gb=0.5,
            activations_gb=1.0,
            overhead_gb=0.5,
        )


def test_vram_breakdown_total():
    from soup_cli.utils.hardware_fit import VRAMBreakdown

    v = VRAMBreakdown(
        weights_gb=2.0,
        optimizer_gb=0.5,
        gradients_gb=0.5,
        activations_gb=1.0,
        overhead_gb=0.5,
    )
    assert v.total_gb == pytest.approx(4.5)


# ---------------------------------------------------------------------------
# estimate_peak_vram_gb
# ---------------------------------------------------------------------------


def test_estimate_peak_vram_returns_breakdown():
    from soup_cli.utils.hardware_fit import HardwareFitInput, VRAMBreakdown, estimate_peak_vram_gb

    inp = HardwareFitInput(
        params_b=7.0, seq_len=2048, batch_size=4,
        optimizer="adamw_torch", quant="4bit", peft="lora",
        gradient_checkpointing=True,
    )
    breakdown = estimate_peak_vram_gb(inp)
    assert isinstance(breakdown, VRAMBreakdown)
    assert breakdown.weights_gb > 0
    assert breakdown.total_gb > 0


def test_estimate_peak_vram_4bit_smaller_than_fp16():
    from soup_cli.utils.hardware_fit import HardwareFitInput, estimate_peak_vram_gb

    base = dict(params_b=7.0, seq_len=1024, batch_size=1,
                optimizer="adamw_torch", peft="lora",
                gradient_checkpointing=True)
    fp16 = estimate_peak_vram_gb(HardwareFitInput(quant="none", **base))
    q4 = estimate_peak_vram_gb(HardwareFitInput(quant="4bit", **base))
    assert q4.weights_gb < fp16.weights_gb


def test_estimate_peak_vram_lora_smaller_than_full():
    from soup_cli.utils.hardware_fit import HardwareFitInput, estimate_peak_vram_gb

    base = dict(params_b=7.0, seq_len=1024, batch_size=1,
                optimizer="adamw_torch", quant="4bit",
                gradient_checkpointing=True)
    full = estimate_peak_vram_gb(HardwareFitInput(peft="full", **base))
    lora = estimate_peak_vram_gb(HardwareFitInput(peft="lora", **base))
    assert lora.optimizer_gb < full.optimizer_gb


def test_estimate_peak_vram_seq_len_scales_activations():
    from soup_cli.utils.hardware_fit import HardwareFitInput, estimate_peak_vram_gb

    base = dict(params_b=1.0, batch_size=1,
                optimizer="adamw_torch", quant="4bit", peft="lora",
                gradient_checkpointing=False)
    short = estimate_peak_vram_gb(HardwareFitInput(seq_len=512, **base))
    long_ = estimate_peak_vram_gb(HardwareFitInput(seq_len=8192, **base))
    assert long_.activations_gb > short.activations_gb


def test_estimate_peak_vram_grad_ckpt_reduces_activations():
    from soup_cli.utils.hardware_fit import HardwareFitInput, estimate_peak_vram_gb

    base = dict(params_b=7.0, seq_len=4096, batch_size=1,
                optimizer="adamw_torch", quant="4bit", peft="lora")
    off = estimate_peak_vram_gb(HardwareFitInput(gradient_checkpointing=False, **base))
    on = estimate_peak_vram_gb(HardwareFitInput(gradient_checkpointing=True, **base))
    assert on.activations_gb < off.activations_gb


def test_estimate_peak_vram_finite():
    from soup_cli.utils.hardware_fit import HardwareFitInput, estimate_peak_vram_gb

    inp = HardwareFitInput(
        params_b=70.0, seq_len=8192, batch_size=8,
        optimizer="adamw_torch", quant="none", peft="full",
        gradient_checkpointing=False,
    )
    bd = estimate_peak_vram_gb(inp)
    assert math.isfinite(bd.total_gb)


def test_estimate_peak_vram_rejects_non_input():
    from soup_cli.utils.hardware_fit import estimate_peak_vram_gb

    with pytest.raises(TypeError):
        estimate_peak_vram_gb("not an input")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# decide_hardware_fit
# ---------------------------------------------------------------------------


def test_decide_hardware_fit_ok(tmp_path):
    from soup_cli.utils.hardware_fit import HardwareFitInput, decide_hardware_fit

    inp = HardwareFitInput(
        params_b=1.0, seq_len=1024, batch_size=1,
        optimizer="adamw_torch", quant="4bit", peft="lora",
        gradient_checkpointing=True,
    )
    # Plenty of headroom
    report = decide_hardware_fit(inp, available_vram_gb=24.0)
    assert report.ok is True
    assert report.peak_vram_gb > 0
    assert report.required_with_margin_gb >= report.peak_vram_gb


def test_decide_hardware_fit_oom():
    from soup_cli.utils.hardware_fit import HardwareFitInput, decide_hardware_fit

    inp = HardwareFitInput(
        params_b=70.0, seq_len=8192, batch_size=8,
        optimizer="adamw_torch", quant="none", peft="full",
        gradient_checkpointing=False,
    )
    report = decide_hardware_fit(inp, available_vram_gb=8.0)
    assert report.ok is False
    assert "exceed" in report.reason.lower() or "oom" in report.reason.lower() or \
        "available" in report.reason.lower()


def test_decide_hardware_fit_rejects_negative_vram():
    from soup_cli.utils.hardware_fit import HardwareFitInput, decide_hardware_fit

    inp = HardwareFitInput(
        params_b=1.0, seq_len=1024, batch_size=1,
        optimizer="adamw_torch", quant="4bit", peft="lora",
        gradient_checkpointing=True,
    )
    with pytest.raises(ValueError, match="vram"):
        decide_hardware_fit(inp, available_vram_gb=-1.0)


def test_decide_hardware_fit_rejects_bool_vram():
    from soup_cli.utils.hardware_fit import HardwareFitInput, decide_hardware_fit

    inp = HardwareFitInput(
        params_b=1.0, seq_len=1024, batch_size=1,
        optimizer="adamw_torch", quant="4bit", peft="lora",
        gradient_checkpointing=True,
    )
    with pytest.raises(TypeError, match="bool"):
        decide_hardware_fit(inp, available_vram_gb=True)  # type: ignore[arg-type]


def test_decide_hardware_fit_rejects_non_finite_vram():
    from soup_cli.utils.hardware_fit import HardwareFitInput, decide_hardware_fit

    inp = HardwareFitInput(
        params_b=1.0, seq_len=1024, batch_size=1,
        optimizer="adamw_torch", quant="4bit", peft="lora",
        gradient_checkpointing=True,
    )
    with pytest.raises(ValueError, match="finite"):
        decide_hardware_fit(inp, available_vram_gb=float("nan"))


def test_hardware_fit_report_frozen():
    from soup_cli.utils.hardware_fit import HardwareFitReport, VRAMBreakdown

    bd = VRAMBreakdown(
        weights_gb=1.0, optimizer_gb=0.5, gradients_gb=0.5,
        activations_gb=1.0, overhead_gb=0.5,
    )
    rep = HardwareFitReport(
        ok=True,
        peak_vram_gb=3.5,
        required_with_margin_gb=3.85,
        available_vram_gb=24.0,
        breakdown=bd,
        reason="ok",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        rep.ok = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Source-wiring regression
# ---------------------------------------------------------------------------


def test_no_heavy_top_level_imports():
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "soup_cli" / "utils" / "hardware_fit.py"
    text = src.read_text(encoding="utf-8")
    import re
    for bad in ["^import torch", "^from torch", "^import transformers", "^from transformers"]:
        assert not re.search(bad, text, re.MULTILINE)
