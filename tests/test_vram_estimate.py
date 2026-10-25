"""``utils/vram_estimate.py``: the shared peak-VRAM estimate's terms, one at a time (#2049).

The numbers the terms are calibrated against are in
``tests/test_vram_estimate_calibration.py``; these tests pin what each term depends on.
"""

from __future__ import annotations

import dataclasses
import math
import re

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

    def test_a_model_without_an_lm_head_builds_none(self, qwen):
        with_head = estimate_vram(qwen, _run()).terms
        without = estimate_vram(qwen, _run(lm_head=False)).terms
        assert with_head["logits"] > 0
        assert without["logits"] == 0
        assert {k: v for k, v in without.items() if k not in ("logits", "overhead")} == {
            k: v for k, v in with_head.items() if k not in ("logits", "overhead")
        }

    def test_a_fused_loss_is_still_charged_and_said(self, qwen):
        plain = estimate_vram(qwen, _run())
        fused = estimate_vram(qwen, _run(fused_loss=("use_liger",)))
        assert fused.terms == plain.terms
        assert [note for note in fused.notes if "fused loss" in note] == [
            "training.use_liger is set: if the fused loss applies, the full logits are not "
            f"built, so the logits term ({plain.terms['logits'] / 1e9:.1f} GB) overstates "
            "the peak. The fused path is not measured."
        ]
        assert not any("fused loss" in note for note in plain.notes)

    def test_no_logits_means_no_fused_loss_note(self, qwen):
        est = estimate_vram(qwen, _run(lm_head=False, fused_loss=("use_cut_ce",)))
        assert not any("fused loss" in note for note in est.notes)


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

    def test_the_tiers_exactly(self, qwen):
        # 1,024 tokens on Qwen2.5-0.5B: one layer keeps 51,930,726.4 bytes, a
        # checkpoint boundary 1,024 x 896 x 2 = 1,835,008. Off keeps all 24 layers;
        # full keeps 24 boundaries and one layer being recomputed; medium keeps 12
        # layers, 12 boundaries and the one being recomputed.
        def activations(tier):
            return estimate_vram(qwen, _run(gradient_checkpointing=tier)).terms["activations"]

        assert activations(False) == 1_246_337_434
        assert activations(True) == 95_970_919
        assert activations("medium") == 697_119_540
        # Selective is priced as no checkpointing, an upper bound.
        assert activations("selective") == activations(False)

    def test_an_odd_layer_count_keeps_the_extra_half(self, qwen):
        # 3 layers: medium keeps ceil(3 / 2) = 2 of them, where floor division
        # would keep 1. One layer keeps 51,930,726.4 bytes and a boundary
        # 1,835,008, so 2 kept layers, one boundary and the one being
        # recomputed come to 3 x 51,930,726.4 + 1,835,008 = 157,627,187.2.
        three = dataclasses.replace(qwen, num_layers=3)
        medium = estimate_vram(three, _run(gradient_checkpointing="medium")).terms["activations"]
        assert medium == 157_627_188

    @pytest.mark.parametrize(
        ("tier", "said"),
        [("medium", "Medium checkpointing"), ("selective", "Selective checkpointing")],
    )
    def test_an_unmeasured_tier_is_said(self, qwen, tier, said):
        est = estimate_vram(qwen, _run(gradient_checkpointing=tier))
        assert [note for note in est.notes if "checkpointing" in note][0].startswith(said)
        assert "not measured" in [note for note in est.notes if said in note][0]

    @pytest.mark.parametrize("tier", [False, True, "full"])
    def test_a_measured_tier_gets_no_tier_note(self, qwen, tier):
        est = estimate_vram(qwen, _run(gradient_checkpointing=tier))
        assert not any("checkpointing" in note for note in est.notes)

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
        # One bf16 copy of it: 4864 x 896 x 2 bytes.
        assert estimate_vram(qwen, _run(quant="4bit", peft="qlora")).terms["dequant"] == 8_716_288


def test_overhead_is_on_top_of_everything_allocated(qwen):
    est = estimate_vram(qwen, _run())
    allocated = sum(v for k, v in est.terms.items() if k != "overhead")
    assert est.total_bytes == allocated + est.terms["overhead"]
    # 0.5 GB of CUDA context and workspaces, plus 5% of what is allocated.
    assert est.terms["overhead"] == math.ceil(0.5e9 + 0.05 * allocated)


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
        ("bad", "error", "message"),
        [
            ({"seq_len": 0}, ValueError, "seq_len and batch_size must be positive"),
            ({"batch_size": -1}, ValueError, "seq_len and batch_size must be positive"),
            ({"lora_r": -1}, ValueError, "lora_r >= 0"),
            ({"seq_len": True}, TypeError, "seq_len must be int, got bool"),
            ({"quant": "hqq:4bit"}, ValueError, "unknown quant 'hqq:4bit'"),
            ({"peft": "prefix"}, ValueError, "unknown peft 'prefix'"),
            ({"lm_head": 1}, TypeError, "lm_head must be bool, got int"),
            ({"fused_loss": ("use_flash_attn",)}, ValueError, "unknown fused_loss"),
        ],
    )
    def test_refuses(self, bad, error, message):
        with pytest.raises(error, match=re.escape(message)):
            _run(**bad)


def test_the_tiny_fixture_prices_every_term():
    shape = shape_from_config(TINY_LLAMA, params=10_000, params_source="safetensors")
    est = estimate_vram(shape, _run(quant="4bit", peft="qlora"))
    assert all(value > 0 for value in est.terms.values())
