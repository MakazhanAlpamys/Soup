"""#2609: the VRAM estimate prices the non-quantised weights of a 4/8-bit run as bf16.

``prepare_kbit_model`` (``utils/kbit_prepare.py``) keeps embeddings, norms and an untied
head in bf16 on a card with bf16 hardware, where PEFT's own preparation made them fp32. The
estimate follows: ``KBIT_UNQUANTIZED_BYTES`` is 2 bytes, and ``RunSpec.kbit_fp32`` prices
the fp32 case (a card without bf16, ``grpo_fp16``, and every measurement taken before the
change, ``benchmarks/evidence/vram-2049-resident.json``). The logits term is NOT changed:
the profile of the A/B (``benchmarks/gate-2609-kbit-fp32-ab.md``) kept three fp32
``[8, 512, 152064]`` tensors, 7,128 MiB, in the peak after the cast back.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from soup_cli.utils import vram_estimate as ve
from soup_cli.utils.hardware_fit import (
    HardwareFitInput,
    estimate_peak_vram_gb,
    hardware_fit_input_from_config,
)
from soup_cli.utils.model_shape import shape_from_config
from soup_cli.utils.vram_estimate import RunSpec, estimate_vram, weights_bytes

MIB = 2**20

# Qwen2.5-7B-Instruct's own config: untied head, a 152,064 vocabulary.
QWEN_7B = {
    "model_type": "qwen2",
    "hidden_size": 3584,
    "intermediate_size": 18944,
    "num_hidden_layers": 28,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "vocab_size": 152064,
    "tie_word_embeddings": False,
}
QWEN_7B_PARAMS = 7_615_616_512
QWEN_7B_EMBEDDING_PARAMS = 2 * 152064 * 3584  # embed_tokens and lm_head

# Qwen2.5-0.5B: tied embeddings, so one matrix.
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
QWEN_05B_PARAMS = 494_032_768


@pytest.fixture
def qwen_7b():
    return shape_from_config(QWEN_7B, params=QWEN_7B_PARAMS, params_source="safetensors")


@pytest.fixture
def qwen_05b():
    return shape_from_config(QWEN_05B, params=QWEN_05B_PARAMS, params_source="safetensors")


def _run(**kw) -> RunSpec:
    return RunSpec(**{"seq_len": 512, "batch_size": 8, "lora_r": 16, "quant": "4bit",
                      "peft": "qlora", "gradient_checkpointing": True, **kw})


class TestConstants:
    def test_the_default_is_two_bytes_and_the_fp32_case_is_four(self):
        assert ve.KBIT_UNQUANTIZED_BYTES == 2.0
        assert ve.KBIT_UNQUANTIZED_FP32_BYTES == 4.0

    def test_the_logits_extra_is_unchanged(self):
        # Arm B of the A/B removed 0 MiB of logits: the three fp32 tensors stayed.
        assert ve.KBIT_LOGITS_EXTRA_BYTES == 2


class TestWeights:
    @pytest.mark.parametrize("quant", ["4bit", "8bit"])
    def test_embeddings_norms_and_head_are_bf16_by_default(self, qwen_7b, quant):
        default = weights_bytes(qwen_7b, _run(quant=quant))
        fp32 = weights_bytes(qwen_7b, _run(quant=quant, kbit_fp32=True))
        assert fp32 - default == QWEN_7B_EMBEDDING_PARAMS * 2

    def test_untied_embed_and_head_are_both_counted(self, qwen_7b):
        per = ve.NF4_BYTES_PER_PARAM
        linear = QWEN_7B_PARAMS - QWEN_7B_EMBEDDING_PARAMS
        assert weights_bytes(qwen_7b, _run()) == math.ceil(
            linear * per + QWEN_7B_EMBEDDING_PARAMS * 2.0
        )

    def test_tied_embeddings_are_one_matrix(self, qwen_05b):
        embed = 151936 * 896
        fp32 = weights_bytes(qwen_05b, _run(kbit_fp32=True))
        assert fp32 - weights_bytes(qwen_05b, _run()) == embed * 2

    @pytest.mark.parametrize("quant", ["none", "gptq", "mxfp4", "fp8"])
    def test_the_flag_changes_nothing_off_bitsandbytes(self, qwen_7b, quant):
        run = _run(quant=quant, peft="lora")
        assert weights_bytes(qwen_7b, run) == weights_bytes(
            qwen_7b, dataclasses.replace(run, kbit_fp32=True)
        )

    def test_unsloth_keeps_its_measured_two_bytes_either_way(self, qwen_7b):
        run = _run(backend="unsloth")
        assert weights_bytes(qwen_7b, run) == weights_bytes(
            qwen_7b, dataclasses.replace(run, kbit_fp32=True)
        )
        assert weights_bytes(qwen_7b, run) == weights_bytes(qwen_7b, _run())

    def test_the_flag_must_be_a_bool(self):
        for bad in (1, "yes", None):
            with pytest.raises(TypeError, match="kbit_fp32 must be bool"):
                _run(kbit_fp32=bad)


class TestLogitsAreUnchanged:
    @pytest.mark.parametrize("quant", ["4bit", "8bit"])
    def test_the_flag_does_not_touch_the_logits_term(self, qwen_7b, quant):
        run = _run(quant=quant)
        assert estimate_vram(qwen_7b, run).terms["logits"] == estimate_vram(
            qwen_7b, dataclasses.replace(run, kbit_fp32=True)
        ).terms["logits"]

    def test_the_logits_term_still_carries_the_kbit_extra(self, qwen_7b):
        elements = 152064 * 512 * 8
        kbit = estimate_vram(qwen_7b, _run()).terms["logits"]
        plain = estimate_vram(qwen_7b, _run(quant="none", peft="lora")).terms["logits"]
        assert kbit - plain == ve.KBIT_LOGITS_EXTRA_BYTES * elements


class TestAgainstTheMeasuredRun:
    """Qwen2.5-7B, NF4, LoRA r16, 8 x 512, checkpointing, RTX 4090 (the A/B of #2609).

    ``max_memory_allocated`` over 105 steps: 17,425 MiB with PEFT's fp32 cast (arm A),
    13,520 MiB with embeddings, head and norms cast back to bf16 (arm B).
    """

    TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    MEASURED_FP32_MIB = 17_425
    MEASURED_BF16_MIB = 13_520

    def _allocated_mib(self, shape, **kw) -> float:
        est = estimate_vram(shape, _run(target_modules=self.TARGETS, **kw))
        return (est.total_bytes - est.terms["overhead"]) / MIB

    def test_the_fp32_estimate_is_inside_the_calibrated_four_bit_band(self, qwen_7b):
        # The calibration band of tests/test_vram_estimate_calibration.py: 0.91 to 1.10.
        ratio = self._allocated_mib(qwen_7b, kbit_fp32=True) / self.MEASURED_FP32_MIB
        assert 0.91 <= ratio <= 1.10, ratio

    def test_the_bf16_estimate_is_an_upper_bound_of_the_measured_peak(self, qwen_7b):
        ratio = self._allocated_mib(qwen_7b) / self.MEASURED_BF16_MIB
        # Known over-prediction, on one measured row: the terms fitted on fp32-stream rows
        # (the bf16 copy of the fp32 head autocast made, 1,040 MiB, the fp32 layer inputs,
        # 784 MiB) were not re-fitted. An under-prediction would let a run through that
        # then runs out of memory, so the ceiling is what this test pins.
        assert 1.0 <= ratio <= 1.30, ratio

    def test_the_change_is_smaller_than_the_measured_drop(self, qwen_7b):
        # The estimate moves by the weights alone (2,081 MiB); the measured drop of
        # 3,905 MiB also holds two terms the estimate does not carry separately.
        moved = self._allocated_mib(qwen_7b, kbit_fp32=True) - self._allocated_mib(qwen_7b)
        assert moved == pytest.approx(QWEN_7B_EMBEDDING_PARAMS * 2 / MIB, abs=1.0)
        assert moved < self.MEASURED_FP32_MIB - self.MEASURED_BF16_MIB


class TestHardwareFit:
    def _input(self, qwen_7b, **kw):
        return HardwareFitInput(
            params_b=QWEN_7B_PARAMS / 1e9,
            seq_len=512,
            batch_size=8,
            optimizer="adamw_torch",
            quant="4bit",
            peft="qlora",
            gradient_checkpointing=True,
            shape=qwen_7b,
            lora_r=16,
            **kw,
        )

    def test_the_flag_reaches_the_estimate(self, qwen_7b):
        default = estimate_peak_vram_gb(self._input(qwen_7b))
        fp32 = estimate_peak_vram_gb(self._input(qwen_7b, kbit_fp32=True))
        assert fp32.estimate.terms["weights"] - default.estimate.terms["weights"] == (
            QWEN_7B_EMBEDDING_PARAMS * 2
        )

    def test_the_flag_must_be_a_bool(self, qwen_7b):
        with pytest.raises(TypeError, match="kbit_fp32 must be bool"):
            self._input(qwen_7b, kbit_fp32="yes")

    def _config(self, tmp_path, **training):
        from soup_cli.config.loader import load_config_from_string

        text = (
            "base: Qwen/Qwen2.5-0.5B-Instruct\n"
            "task: sft\n"
            "data:\n  train: ./data.jsonl\n  format: alpaca\n  max_length: 512\n"
            "training:\n  quantization: 4bit\n  batch_size: 2\n"
            + "".join(f"  {key}: {value}\n" for key, value in training.items())
        )
        return load_config_from_string(text)

    def test_the_default_input_prices_bf16(self, tmp_path):
        inp = hardware_fit_input_from_config(self._config(tmp_path), allow_network=False)
        assert inp is not None
        assert inp.kbit_fp32 is False

    def test_a_card_without_bf16_prices_fp32(self, tmp_path):
        inp = hardware_fit_input_from_config(
            self._config(tmp_path), allow_network=False, kbit_fp32=True
        )
        assert inp is not None
        assert inp.kbit_fp32 is True

    def test_grpo_fp16_prices_fp32(self, tmp_path):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(
            "base: Qwen/Qwen2.5-0.5B-Instruct\n"
            "task: grpo\n"
            "data:\n  train: ./data.jsonl\n  format: chatml\n  max_length: 512\n"
            "training:\n  quantization: 4bit\n  batch_size: 2\n  grpo_fp16: true\n"
        )
        inp = hardware_fit_input_from_config(cfg, allow_network=False)
        assert inp is not None
        assert inp.kbit_fp32 is True

    def test_the_train_preflight_input_takes_the_flag(self, tmp_path):
        from soup_cli.commands.train import _build_hardware_fit_input

        cfg = self._config(tmp_path)
        assert _build_hardware_fit_input(cfg, kbit_fp32=True).kbit_fp32 is True
        assert _build_hardware_fit_input(cfg).kbit_fp32 is False

    @pytest.mark.parametrize(
        ("flags", "expected"),
        [((True, False), False), ((False, True), True)],
        ids=["bf16-card", "fp16-only-card"],
    )
    def test_the_preflight_asks_the_card(self, monkeypatch, tmp_path, flags, expected):
        # bf16_fp16_flags("cuda") is (bf16, fp16): fp16 true is a CUDA card without bf16.
        import soup_cli.utils.gpu as gpu
        from soup_cli.commands import train as train_mod

        seen = []
        monkeypatch.setattr(gpu, "bf16_fp16_flags", lambda device, **kw: flags)
        monkeypatch.setattr(
            train_mod, "_build_hardware_fit_input", lambda cfg, **kw: seen.append(kw)
        )
        train_mod._hardware_fit_preflight(
            self._config(tmp_path), {"memory_total_bytes": int(24e9)}, allow_oom_attempt=False
        )
        assert [kw["kbit_fp32"] for kw in seen] == [expected]
