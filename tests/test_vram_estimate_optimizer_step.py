"""The optimizer step's phase of the peak (#2277).

The forward/backward pass and the optimizer step peak at different times. On a short
full fine-tune with ``adamw_torch`` the step is the larger: the parameters, their
gradients, the optimizer state and the foreach scratch (one fp32 copy of
``exp_avg_sq``) hold 20 bytes a parameter after the activations are freed. The
``optimizer_step`` term is how far the step exceeds the pass, so the total is the
larger of the two phases.
"""

from __future__ import annotations

import dataclasses

import pytest

from soup_cli.utils import vram_estimate as ve
from soup_cli.utils.hardware_fit import (
    HardwareFitInput,
    breakdown_from_estimate,
    estimate_peak_vram_gb,
    unextrapolated_floor,
)
from soup_cli.utils.model_shape import shape_from_config
from soup_cli.utils.profiler import profile_result
from soup_cli.utils.vram_estimate import RunSpec, estimate_vram

#: HuggingFaceTB/SmolLM2-135M-Instruct, the model #2277 measured.
SMOLLM2_135M = {
    "model_type": "llama",
    "hidden_size": 576,
    "intermediate_size": 1536,
    "num_hidden_layers": 30,
    "num_attention_heads": 9,
    "num_key_value_heads": 3,
    "vocab_size": 49152,
    "tie_word_embeddings": True,
}
PARAMS = 134_515_008


@pytest.fixture
def smol():
    return shape_from_config(SMOLLM2_135M, params=PARAMS, params_source="safetensors")


def _full(seq_len: int, optimizer: str = "adamw_torch") -> RunSpec:
    return RunSpec(seq_len=seq_len, batch_size=1, peft="full", optimizer=optimizer)


def _allocated(est) -> int:
    return est.total_bytes - est.terms["overhead"]


def _pass(est) -> int:
    return sum(v for k, v in est.terms.items() if k not in ("optimizer_step", "overhead"))


def test_a_short_adamw_torch_full_fine_tune_is_priced_at_the_step(smol):
    est = estimate_vram(smol, _full(64))
    # weights 4 + gradients 4 + AdamW state 8 + foreach scratch 4 bytes a parameter.
    assert _allocated(est) == 20 * PARAMS
    assert est.terms["optimizer_step"] == 20 * PARAMS - _pass(est) > 0
    assert any("optimizer step" in note for note in est.notes)


def test_the_overhead_slack_covers_the_step(smol):
    est = estimate_vram(smol, _full(64))
    slack = ve.OVERHEAD_SLACK * 20 * PARAMS
    assert est.terms["overhead"] == pytest.approx(ve.OVERHEAD_FIXED_BYTES + slack, abs=1)


@pytest.mark.parametrize("seq_len", [64, 128, 256, 512, 1024])
def test_the_total_is_the_larger_phase(smol, seq_len):
    est = estimate_vram(smol, _full(seq_len))
    assert _allocated(est) == max(_pass(est), 20 * PARAMS)


def test_a_long_run_is_priced_at_the_pass(smol):
    est = estimate_vram(smol, _full(1024))
    assert est.terms["optimizer_step"] == 0
    assert not any("optimizer step" in note for note in est.notes)


@pytest.mark.parametrize("optimizer", ["adamw_torch_fused", "adamw_bnb_8bit", "sgd"])
def test_only_adamw_torch_has_step_scratch(smol, optimizer):
    """The fused kernel takes no foreach scratch; nothing else is measured yet."""
    assert estimate_vram(smol, _full(64, optimizer)).terms["optimizer_step"] == 0


def test_a_lora_run_is_priced_at_the_pass(smol):
    run = RunSpec(seq_len=64, batch_size=1, peft="lora", lora_r=8)
    assert estimate_vram(smol, run).terms["optimizer_step"] == 0


def test_without_the_scratch_the_short_run_is_priced_at_the_pass(smol, monkeypatch):
    """Mutation check: the scratch is what lifts the step above the pass at 64 tokens."""
    monkeypatch.setattr(ve, "OPTIMIZER_STEP_SCRATCH_BYTES_PER_PARAM", {})
    est = estimate_vram(smol, _full(64))
    assert est.terms["optimizer_step"] == 0
    assert _allocated(est) < 20 * PARAMS


def test_the_breakdown_carries_the_term(smol):
    est = estimate_vram(smol, _full(64))
    inp = HardwareFitInput(
        params_b=PARAMS / 1e9,
        seq_len=64,
        batch_size=1,
        optimizer="adamw_torch",
        quant="none",
        peft="full",
        gradient_checkpointing=False,
        shape=smol,
    )
    breakdown = estimate_peak_vram_gb(inp)
    assert breakdown.optimizer_step_gb == pytest.approx(est.gb("optimizer_step"))
    assert breakdown.total_gb == pytest.approx(est.total_gb)


def test_the_floor_does_not_count_the_step(smol):
    """The step's excess sits over activations, so it stays out of the unextrapolated floor."""
    breakdown = breakdown_from_estimate(estimate_vram(smol, _full(64)))
    without = dataclasses.replace(breakdown, optimizer_step_gb=0.0)
    inp = HardwareFitInput(
        params_b=PARAMS / 1e9,
        seq_len=64,
        batch_size=1,
        optimizer="adamw_torch",
        quant="none",
        peft="full",
        gradient_checkpointing=False,
        shape=smol,
    )
    assert unextrapolated_floor(None, inp, breakdown) == unextrapolated_floor(None, inp, without)


def test_profile_reports_the_term(smol):
    breakdown = breakdown_from_estimate(estimate_vram(smol, _full(64)))
    result = profile_result(breakdown, model_name="HuggingFaceTB/SmolLM2-135M-Instruct")
    assert result["optimizer_step_memory_gb"] == round(breakdown.optimizer_step_gb, 2) > 0
