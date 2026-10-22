"""``utils/vram_estimate.py``: the shared peak-VRAM estimate's terms, one at a time (#2049).

The numbers the terms are calibrated against are in
``tests/test_vram_estimate_calibration.py``; these tests pin what each term depends on.
"""

from __future__ import annotations

import dataclasses

import pytest

from soup_cli.utils import vram_estimate as ve
from soup_cli.utils.layer_stream import LOGITS_LOSS_BYTES_PER_ELEMENT
from soup_cli.utils.model_shape import guessed_shape, shape_from_config
from soup_cli.utils.vram_estimate import RunSpec, estimate_vram, fit_verdict
from tests._tiny_checkpoint import TINY_LLAMA

# Qwen2.5-0.5B's own config: a 151,936 vocabulary on a 0.5B model.
QWEN_05B = {
    "model_type": "qwen2",
    "hidden_size": 896,
    "intermediate_size": 4864,
    "num_hidden_layers": 24,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "vocab_size": 151936,
    "tie_word_embeddings": True,
}
QWEN_PARAMS = 494_032_768


@pytest.fixture
def qwen():
    return shape_from_config(QWEN_05B, params=QWEN_PARAMS, params_source="safetensors")


def _run(**kw) -> RunSpec:
    return RunSpec(**{"seq_len": 1024, "batch_size": 1, "lora_r": 8, **kw})


class TestLogits:
    def test_it_grows_with_vocab_seq_and_batch(self, qwen):
        base = estimate_vram(qwen, _run()).terms["logits"]
        assert estimate_vram(qwen, _run(seq_len=2048)).terms["logits"] == pytest.approx(2 * base)
        assert estimate_vram(qwen, _run(batch_size=2)).terms["logits"] == pytest.approx(2 * base)
        small_vocab = dataclasses.replace(qwen, vocab_size=151936 // 4)
        quarter = estimate_vram(small_vocab, _run()).terms["logits"]
        assert quarter == pytest.approx(base / 4, rel=1e-6)

    def test_checkpointing_does_not_take_it_below_the_loss_arithmetic(self, qwen):
        on = estimate_vram(qwen, _run(gradient_checkpointing=True)).terms["logits"]
        elements = 151936 * 1024
        assert on >= elements * LOGITS_LOSS_BYTES_PER_ELEMENT

    def test_it_dominates_a_long_large_vocabulary_run(self, qwen):
        est = estimate_vram(qwen, _run(seq_len=2048, gradient_checkpointing=True))
        assert est.terms["logits"] == max(est.terms.values())

    def test_an_unknown_vocabulary_is_not_priced_and_is_said(self):
        est = estimate_vram(guessed_shape(0.5), _run())
        assert est.terms["logits"] == 0
        assert any("logits term is not counted" in note for note in est.notes)


class TestWeights:
    def test_bf16_is_two_bytes_a_parameter(self, qwen):
        assert estimate_vram(qwen, _run()).terms["weights"] == QWEN_PARAMS * 2

    def test_full_fine_tuning_loads_fp32(self, qwen):
        est = estimate_vram(qwen, _run(peft="full", lora_r=0))
        assert est.terms["weights"] == QWEN_PARAMS * 4
        assert est.terms["adapters"] == 0
        assert est.terms["gradients"] == QWEN_PARAMS * 4

    @pytest.mark.parametrize("quant", ["4bit", "8bit"])
    def test_kbit_loading_keeps_the_embeddings_unquantized(self, qwen, quant):
        est = estimate_vram(qwen, _run(quant=quant, peft="qlora" if quant == "4bit" else "lora"))
        embed = 151936 * 896  # tied: one matrix
        assert est.terms["weights"] > embed * 2
        flat = QWEN_PARAMS * ve.BYTES_PER_PARAM_BY_QUANT[quant]
        # The flat figure under-priced a 0.5B model with this vocabulary.
        assert est.terms["weights"] > flat

    @pytest.mark.parametrize("quant", sorted(ve.BYTES_PER_PARAM_BY_QUANT))
    def test_without_a_vocabulary_the_flat_table_applies(self, quant):
        shape = guessed_shape(2.0)
        est = estimate_vram(shape, _run(quant=quant))
        assert est.terms["weights"] == pytest.approx(
            shape.params * ve.BYTES_PER_PARAM_BY_QUANT[quant], rel=1e-9
        )


class TestAdapters:
    def test_sized_from_the_real_dims(self, qwen):
        # q_proj 896 -> 896, v_proj 896 -> 2 * 64: r * (in + out), 24 layers, fp32.
        expected = 24 * 8 * ((896 + 896) + (896 + 128)) * ve.ADAPTER_BYTES_PER_PARAM
        assert estimate_vram(qwen, _run()).terms["adapters"] == expected

    def test_all_linear_costs_more_than_auto(self, qwen):
        auto = estimate_vram(qwen, _run()).terms["adapters"]
        every = estimate_vram(qwen, _run(target_modules="all-linear")).terms["adapters"]
        assert every > 4 * auto

    @pytest.mark.parametrize(
        ("optimizer", "per_param"), [("adamw_torch", 8), ("adamw_bnb_8bit", 2)]
    )
    def test_optimizer_state_per_trainable_parameter(self, qwen, optimizer, per_param):
        est = estimate_vram(qwen, _run(optimizer=optimizer))
        trainable = est.terms["gradients"] // 4
        assert est.terms["optimizer"] == trainable * per_param


class TestActivations:
    def test_checkpointing_lowers_them(self, qwen):
        off = estimate_vram(qwen, _run()).terms["activations"]
        medium = estimate_vram(qwen, _run(gradient_checkpointing="medium")).terms["activations"]
        on = estimate_vram(qwen, _run(gradient_checkpointing=True)).terms["activations"]
        assert on < medium < off

    @pytest.mark.parametrize("tier", [True, "full", "auto"])
    def test_full_tiers_price_alike(self, qwen, tier):
        assert (
            estimate_vram(qwen, _run(gradient_checkpointing=tier)).terms["activations"]
            == estimate_vram(qwen, _run(gradient_checkpointing=True)).terms["activations"]
        )

    def test_they_grow_with_tokens(self, qwen):
        one = estimate_vram(qwen, _run()).terms["activations"]
        assert estimate_vram(qwen, _run(seq_len=2048)).terms["activations"] > 1.9 * one


class TestDequant:
    def test_only_under_4bit(self, qwen):
        assert estimate_vram(qwen, _run()).terms["dequant"] == 0
        assert estimate_vram(qwen, _run(quant="8bit")).terms["dequant"] == 0
        # The largest projection in bf16: 4864 x 896.
        assert estimate_vram(qwen, _run(quant="4bit", peft="qlora")).terms["dequant"] == (
            4864 * 896 * 2 * ve.DEQUANT_COPIES
        )


def test_overhead_is_on_top_of_everything_allocated(qwen):
    est = estimate_vram(qwen, _run())
    allocated = sum(v for k, v in est.terms.items() if k != "overhead")
    assert est.terms["overhead"] >= ve.OVERHEAD_FIXED_BYTES
    assert est.total_bytes == allocated + est.terms["overhead"]


def test_every_term_is_reported(qwen):
    assert tuple(estimate_vram(qwen, _run()).terms) == ve.TERM_NAMES


class TestFitVerdict:
    @pytest.mark.parametrize(
        ("total", "available", "verdict"),
        [
            (9.0, 10.0, "fits"),
            (10.0 / 1.1, 10.0, "fits"),
            (9.5, 10.0, "tight"),
            (10.0, 10.0, "tight"),
            (10.01, 10.0, "does not fit"),
        ],
    )
    def test_thresholds(self, total, available, verdict):
        assert fit_verdict(total, available) == verdict


class TestRunSpec:
    @pytest.mark.parametrize(
        "bad",
        [
            {"seq_len": 0},
            {"batch_size": -1},
            {"lora_r": -1},
            {"seq_len": True},
            {"quant": "hqq:4bit"},
            {"peft": "prefix"},
        ],
    )
    def test_refuses(self, bad):
        with pytest.raises((TypeError, ValueError)):
            _run(**bad)


def test_the_tiny_fixture_prices_every_term():
    shape = shape_from_config(TINY_LLAMA, params=10_000, params_source="safetensors")
    est = estimate_vram(shape, _run(quant="4bit", peft="qlora"))
    assert all(value > 0 for value in est.terms.values())
