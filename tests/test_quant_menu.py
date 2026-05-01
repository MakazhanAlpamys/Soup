"""Quant menu — GPTQ / AWQ / HQQ / AQLM / EETQ / MXFP4 train-time wiring (v0.38.0).

Tests are organised Part by Part to mirror the implementation plan. Heavy deps
(gptqmodel, autoawq, hqq, aqlm, eetq) are mocked — none are required at import.
"""

from __future__ import annotations

import pytest

from soup_cli.config.loader import load_config_from_string

# ---------------------------------------------------------------------------
# Part A — GPTQ
# ---------------------------------------------------------------------------


class TestGPTQSchema:
    def test_gptq_quantization_value_accepted(self):
        cfg = load_config_from_string(
            """
base: TheBloke/Llama-2-7B-GPTQ
task: sft
data:
  train: data.jsonl
training:
  quantization: gptq
"""
        )
        assert cfg.training.quantization == "gptq"

    def test_gptq_disable_exllama_default_true(self):
        cfg = load_config_from_string(
            """
base: m
task: sft
data:
  train: d.jsonl
training:
  quantization: gptq
"""
        )
        # exllama backend is broken with PEFT — default to triton.
        assert cfg.training.gptq_disable_exllama is True

    def test_gptq_disable_exllama_can_be_overridden(self):
        cfg = load_config_from_string(
            """
base: m
task: sft
data: {train: d.jsonl}
training:
  quantization: gptq
  gptq_disable_exllama: false
"""
        )
        assert cfg.training.gptq_disable_exllama is False

    def test_gptq_quantization_aware_combo_rejected(self):
        # GPTQ + int8 QAT does not compose — torchao.quant_api expects bf16/fp16.
        with pytest.raises(
            ValueError,
            match="gptq.*quantization_aware|quantization_aware.*gptq|incompatible",
        ):
            load_config_from_string(
                """
base: m
task: sft
data: {train: d.jsonl}
training:
  quantization: gptq
  quantization_aware: true
"""
            )


class TestGPTQConfigBuilder:
    def test_build_gptq_config_disables_exllama_when_flag_true(self):
        from soup_cli.utils.quant_menu import build_gptq_config

        cfg = build_gptq_config(disable_exllama=True)
        # Returned object is a dict-shaped config (we intentionally avoid
        # importing transformers' GPTQConfig at module level).
        assert cfg["bits"] in (2, 3, 4, 8)  # default emitted
        assert cfg["use_exllama"] is False

    def test_build_gptq_config_can_enable_exllama(self):
        from soup_cli.utils.quant_menu import build_gptq_config

        cfg = build_gptq_config(disable_exllama=False)
        assert cfg["use_exllama"] is True


class TestGPTQValidator:
    def test_gptq_train_requires_existing_quantize_config(self, tmp_path):
        from soup_cli.utils.quant_menu import validate_gptq_checkpoint

        # No quantize_config.json → loud failure.
        with pytest.raises(FileNotFoundError, match="GPTQ"):
            validate_gptq_checkpoint(str(tmp_path))

    def test_gptq_train_accepts_dir_with_quantize_config(self, tmp_path):
        from soup_cli.utils.quant_menu import validate_gptq_checkpoint

        (tmp_path / "quantize_config.json").write_text("{}", encoding="utf-8")
        # Returns None on success (no exception).
        validate_gptq_checkpoint(str(tmp_path))

    def test_gptq_train_accepts_hf_repo_id(self):
        from soup_cli.utils.quant_menu import validate_gptq_checkpoint

        # HF repo id (not a local path) is allowed — HF will fetch the file.
        validate_gptq_checkpoint("TheBloke/Llama-2-7B-GPTQ")

    def test_gptq_validator_rejects_null_byte(self):
        from soup_cli.utils.quant_menu import validate_gptq_checkpoint

        with pytest.raises(ValueError, match="null"):
            validate_gptq_checkpoint("foo\x00bar")


# ---------------------------------------------------------------------------
# Part B — AWQ
# ---------------------------------------------------------------------------


class TestAWQ:
    def test_awq_quantization_value_accepted(self):
        cfg = load_config_from_string(
            """
base: TheBloke/Llama-2-7B-AWQ
task: sft
data: {train: d.jsonl}
training: {quantization: awq}
"""
        )
        assert cfg.training.quantization == "awq"

    def test_awq_qat_combo_rejected(self):
        with pytest.raises(ValueError, match="awq.*quantization_aware|incompatible"):
            load_config_from_string(
                """
base: m
task: sft
data: {train: d.jsonl}
training: {quantization: awq, quantization_aware: true}
"""
            )

    def test_build_awq_config_returns_dict(self):
        from soup_cli.utils.quant_menu import build_awq_config

        cfg = build_awq_config()
        assert cfg["bits"] == 4
        assert cfg["version"] in ("gemm", "gemv")

    def test_validate_awq_checkpoint_local_missing(self, tmp_path):
        from soup_cli.utils.quant_menu import validate_awq_checkpoint

        with pytest.raises(FileNotFoundError, match="AWQ"):
            validate_awq_checkpoint(str(tmp_path))

    def test_validate_awq_checkpoint_local_present(self, tmp_path):
        from soup_cli.utils.quant_menu import validate_awq_checkpoint

        (tmp_path / "quant_config.json").write_text("{}", encoding="utf-8")
        validate_awq_checkpoint(str(tmp_path))


# ---------------------------------------------------------------------------
# Part C — HQQ 1-8 bit
# ---------------------------------------------------------------------------


class TestHQQ:
    def test_all_hqq_bit_rates_accepted(self):
        for bits in (1, 2, 3, 4, 5, 6, 8):
            cfg = load_config_from_string(
                f"""
base: m
task: sft
data: {{train: d.jsonl}}
training: {{quantization: 'hqq:{bits}bit'}}
"""
            )
            assert cfg.training.quantization == f"hqq:{bits}bit"

    def test_hqq_7bit_rejected(self):
        # 7-bit is intentionally not in the Literal — HQQ does not support it.
        with pytest.raises(ValueError, match="hqq:7bit|Input should"):
            load_config_from_string(
                """
base: m
task: sft
data: {train: d.jsonl}
training: {quantization: 'hqq:7bit'}
"""
            )

    def test_build_hqq_config_extracts_bits(self):
        from soup_cli.utils.quant_menu import build_hqq_config

        cfg = build_hqq_config(quantization="hqq:4bit")
        assert cfg["bits"] == 4
        assert cfg["group_size"] >= 32

    def test_build_hqq_config_rejects_non_hqq_string(self):
        from soup_cli.utils.quant_menu import build_hqq_config

        with pytest.raises(ValueError, match="hqq:"):
            build_hqq_config(quantization="awq")


# ---------------------------------------------------------------------------
# Part D — AQLM 2-bit
# ---------------------------------------------------------------------------


class TestAQLM:
    def test_aqlm_value_accepted(self):
        cfg = load_config_from_string(
            """
base: BlackSamorez/Llama-2-7b-AQLM-2Bit-1x16-hf
task: sft
data: {train: d.jsonl}
training: {quantization: aqlm}
"""
        )
        assert cfg.training.quantization == "aqlm"

    def test_build_aqlm_config_forces_2bit(self):
        from soup_cli.utils.quant_menu import build_aqlm_config

        cfg = build_aqlm_config()
        assert cfg["bits"] == 2  # AQLM is always 2-bit

    def test_aqlm_qat_combo_rejected(self):
        with pytest.raises(ValueError, match="aqlm|incompatible"):
            load_config_from_string(
                """
base: m
task: sft
data: {train: d.jsonl}
training: {quantization: aqlm, quantization_aware: true}
"""
            )


# ---------------------------------------------------------------------------
# Part E — EETQ 8-bit
# ---------------------------------------------------------------------------


class TestEETQ:
    def test_eetq_value_accepted(self):
        cfg = load_config_from_string(
            """
base: m
task: sft
data: {train: d.jsonl}
training: {quantization: eetq}
"""
        )
        assert cfg.training.quantization == "eetq"

    def test_build_eetq_config_forces_8bit(self):
        from soup_cli.utils.quant_menu import build_eetq_config

        cfg = build_eetq_config()
        assert cfg["bits"] == 8


# ---------------------------------------------------------------------------
# Part F — MXFP4 / FP8 dequantize-on-load
# ---------------------------------------------------------------------------


class TestMXFP4:
    def test_mxfp4_value_accepted(self):
        cfg = load_config_from_string(
            """
base: m
task: sft
data: {train: d.jsonl}
training: {quantization: mxfp4}
"""
        )
        assert cfg.training.quantization == "mxfp4"

    def test_build_mxfp4_config_uses_bnb_4bit(self):
        from soup_cli.utils.quant_menu import build_mxfp4_config

        cfg = build_mxfp4_config()
        assert cfg["load_in_4bit"] is True
        assert cfg["bnb_4bit_quant_type"] == "mxfp4"


class TestFP8Dequant:
    def test_fp8_value_accepted(self):
        cfg = load_config_from_string(
            """
base: m
task: sft
data: {train: d.jsonl}
training: {quantization: fp8}
"""
        )
        assert cfg.training.quantization == "fp8"

    def test_build_fp8_dequant_config_sets_dequantize(self):
        from soup_cli.utils.quant_menu import build_fp8_dequant_config

        cfg = build_fp8_dequant_config()
        assert cfg["dequantize"] is True


# ---------------------------------------------------------------------------
# Part G — BNB 4-bit + FSDP quant_storage
# ---------------------------------------------------------------------------


class TestBNBQuantStorage:
    def test_quant_storage_default_none(self):
        cfg = load_config_from_string(
            """
base: m
task: sft
data: {train: d.jsonl}
training: {quantization: 4bit}
"""
        )
        assert cfg.training.bnb_4bit_quant_storage is None

    def test_quant_storage_bf16_accepted_with_4bit(self):
        cfg = load_config_from_string(
            """
base: m
task: sft
data: {train: d.jsonl}
training: {quantization: 4bit, bnb_4bit_quant_storage: bfloat16}
"""
        )
        assert cfg.training.bnb_4bit_quant_storage == "bfloat16"

    def test_quant_storage_with_8bit_rejected(self):
        with pytest.raises(ValueError, match="bnb_4bit_quant_storage"):
            load_config_from_string(
                """
base: m
task: sft
data: {train: d.jsonl}
training: {quantization: 8bit, bnb_4bit_quant_storage: bfloat16}
"""
            )

    def test_quant_storage_with_mxfp4_accepted(self):
        cfg = load_config_from_string(
            """
base: m
task: sft
data: {train: d.jsonl}
training: {quantization: mxfp4, bnb_4bit_quant_storage: bfloat16}
"""
        )
        assert cfg.training.bnb_4bit_quant_storage == "bfloat16"

    def test_quant_storage_invalid_dtype_rejected(self):
        with pytest.raises(ValueError, match="quant_storage|Input should"):
            load_config_from_string(
                """
base: m
task: sft
data: {train: d.jsonl}
training: {quantization: 4bit, bnb_4bit_quant_storage: int4}
"""
            )


# ---------------------------------------------------------------------------
# Part H — Compatibility matrix (quant × multi-GPU)
# ---------------------------------------------------------------------------


class TestCompatMatrix:
    def test_compat_check_hqq_zero3_rejected(self):
        from soup_cli.utils.quant_menu import check_quant_distributed_compat

        problems = check_quant_distributed_compat(
            quantization="hqq:4bit", deepspeed="zero3", fsdp=False
        )
        assert any("hqq" in p.lower() and "zero" in p.lower() for p in problems)

    def test_compat_check_hqq_fsdp_rejected(self):
        from soup_cli.utils.quant_menu import check_quant_distributed_compat

        problems = check_quant_distributed_compat(
            quantization="hqq:4bit", deepspeed=None, fsdp=True
        )
        assert any("hqq" in p.lower() and "fsdp" in p.lower() for p in problems)

    def test_compat_check_eetq_zero3_rejected(self):
        from soup_cli.utils.quant_menu import check_quant_distributed_compat

        problems = check_quant_distributed_compat(
            quantization="eetq", deepspeed="zero3", fsdp=False
        )
        assert any("eetq" in p.lower() for p in problems)

    def test_compat_check_aqlm_zero3_rejected(self):
        from soup_cli.utils.quant_menu import check_quant_distributed_compat

        problems = check_quant_distributed_compat(
            quantization="aqlm", deepspeed="zero3", fsdp=False
        )
        assert any("aqlm" in p.lower() for p in problems)

    def test_compat_check_4bit_fsdp_with_quant_storage_clean(self):
        from soup_cli.utils.quant_menu import check_quant_distributed_compat

        # 4bit + FSDP + bf16 quant_storage = clean (the canonical QLoRA-FSDP combo).
        problems = check_quant_distributed_compat(
            quantization="4bit",
            deepspeed=None,
            fsdp=True,
            bnb_4bit_quant_storage="bfloat16",
        )
        assert problems == []

    def test_compat_check_4bit_fsdp_warns_without_quant_storage(self):
        from soup_cli.utils.quant_menu import check_quant_distributed_compat

        problems = check_quant_distributed_compat(
            quantization="4bit",
            deepspeed=None,
            fsdp=True,
            bnb_4bit_quant_storage=None,
        )
        # Returned as a 'warning' tag, not a hard error.
        warnings = [p for p in problems if "warning" in p.lower()]
        assert any("quant_storage" in w.lower() for w in warnings)

    def test_compat_check_gptq_ddp_clean(self):
        from soup_cli.utils.quant_menu import check_quant_distributed_compat

        problems = check_quant_distributed_compat(
            quantization="gptq", deepspeed=None, fsdp=False
        )
        assert problems == []

    def test_compat_check_normalizes_hyphenated_deepspeed(self):
        from soup_cli.utils.quant_menu import check_quant_distributed_compat

        # Confirm `--deepspeed zero-3` normalises to zero3 — matches the
        # `replace("-", "")` step in `_distributed_strategy`. Prevents a
        # silent-pass cliff where a hyphenated preset bypasses the matrix.
        problems = check_quant_distributed_compat(
            quantization="hqq:4bit", deepspeed="zero-3", fsdp=False
        )
        assert any("hqq" in p.lower() and "zero" in p.lower() for p in problems)

    def test_compat_check_unknown_deepspeed_treated_as_ddp(self):
        from soup_cli.utils.quant_menu import check_quant_distributed_compat

        # Conservative fallback: unrecognised preset → ddp (no false-positive
        # rejection for forward-compat with future DeepSpeed presets).
        problems = check_quant_distributed_compat(
            quantization="hqq:4bit", deepspeed="some-future-preset", fsdp=False
        )
        assert problems == []

    def test_compat_check_invalid_quant_rejected(self):
        from soup_cli.utils.quant_menu import check_quant_distributed_compat

        with pytest.raises(ValueError, match="unknown quantization"):
            check_quant_distributed_compat(
                quantization="bogus", deepspeed=None, fsdp=False
            )

    def test_compat_check_eetq_zero3_message_specific(self):
        from soup_cli.utils.quant_menu import check_quant_distributed_compat

        problems = check_quant_distributed_compat(
            quantization="eetq", deepspeed="zero3", fsdp=False
        )
        # Message must explicitly name BOTH the format AND the strategy.
        assert any(
            "eetq" in p.lower() and ("zero" in p.lower() or "stage" in p.lower())
            for p in problems
        )

    def test_compat_check_eetq_fsdp_rejected(self):
        from soup_cli.utils.quant_menu import check_quant_distributed_compat

        problems = check_quant_distributed_compat(
            quantization="eetq", deepspeed=None, fsdp=True
        )
        assert any("eetq" in p.lower() and "fsdp" in p.lower() for p in problems)

    def test_compat_check_aqlm_fsdp_rejected(self):
        from soup_cli.utils.quant_menu import check_quant_distributed_compat

        problems = check_quant_distributed_compat(
            quantization="aqlm", deepspeed=None, fsdp=True
        )
        assert any("aqlm" in p.lower() and "fsdp" in p.lower() for p in problems)

    def test_compat_check_hqq_zero3_message_specific(self):
        from soup_cli.utils.quant_menu import check_quant_distributed_compat

        problems = check_quant_distributed_compat(
            quantization="hqq:4bit", deepspeed="zero3", fsdp=False
        )
        # Per security-review finding: assert both keywords present.
        assert any(
            "hqq" in p.lower() and ("zero" in p.lower() or "stage" in p.lower())
            for p in problems
        )


# ---------------------------------------------------------------------------
# SoupConfig-level cross-validator tests (Quant Menu task/backend/modality gate)
# ---------------------------------------------------------------------------


class TestQuantMenuSchemaGates:
    def test_quant_menu_with_dpo_task_rejected(self):
        with pytest.raises(ValueError, match="Quant Menu|v0.38.1|task="):
            load_config_from_string(
                """
base: TheBloke/Llama-2-7B-GPTQ
task: dpo
data: {train: d.jsonl}
training: {quantization: gptq}
"""
            )

    def test_quant_menu_with_mlx_backend_rejected(self):
        with pytest.raises(ValueError, match="mlx backend|mlx"):
            load_config_from_string(
                """
base: m
task: sft
backend: mlx
data: {train: d.jsonl}
training: {quantization: hqq:4bit}
"""
            )

    def test_quant_menu_with_vision_modality_rejected(self):
        with pytest.raises(ValueError, match="modality|vision|v0.38.1"):
            load_config_from_string(
                """
base: m
task: sft
modality: vision
data: {train: d.jsonl, format: llava}
training: {quantization: gptq}
"""
            )


# ---------------------------------------------------------------------------
# Builder-side adversarial coverage (per TDD review)
# ---------------------------------------------------------------------------


class TestBuilderAdversarial:
    def test_build_gptq_config_invalid_bits_rejected(self):
        from soup_cli.utils.quant_menu import build_gptq_config

        with pytest.raises(ValueError, match="GPTQ bits"):
            build_gptq_config(bits=5)

    def test_build_awq_config_non_4bit_rejected(self):
        from soup_cli.utils.quant_menu import build_awq_config

        with pytest.raises(ValueError, match="AWQ supports only 4-bit"):
            build_awq_config(bits=8)

    def test_build_awq_config_invalid_version_rejected(self):
        from soup_cli.utils.quant_menu import build_awq_config

        with pytest.raises(ValueError, match="AWQ version"):
            build_awq_config(version="invalid")

    def test_build_hqq_config_bool_group_size_rejected(self):
        from soup_cli.utils.quant_menu import build_hqq_config

        with pytest.raises(ValueError, match="group_size"):
            build_hqq_config(quantization="hqq:4bit", group_size=True)  # type: ignore[arg-type]

    def test_build_hqq_config_undersize_group_rejected(self):
        from soup_cli.utils.quant_menu import build_hqq_config

        with pytest.raises(ValueError, match="group_size"):
            build_hqq_config(quantization="hqq:4bit", group_size=16)

    def test_parse_hqq_bits_oversized_suffix_rejected(self):
        from soup_cli.utils.quant_menu import parse_hqq_bits

        with pytest.raises(ValueError, match="HQQ suffix exceeds"):
            parse_hqq_bits("hqq:1234567890123456bit")

    def test_parse_hqq_bits_non_int_suffix_rejected(self):
        from soup_cli.utils.quant_menu import parse_hqq_bits

        with pytest.raises(ValueError, match="HQQ bits must be int"):
            parse_hqq_bits("hqq:fourbit")

    def test_parse_hqq_bits_missing_bit_suffix_rejected(self):
        from soup_cli.utils.quant_menu import parse_hqq_bits

        with pytest.raises(ValueError, match="must end with 'bit'"):
            parse_hqq_bits("hqq:4")

    def test_validate_awq_null_byte_rejected(self):
        from soup_cli.utils.quant_menu import validate_awq_checkpoint

        with pytest.raises(ValueError, match="null"):
            validate_awq_checkpoint("foo\x00bar")


# ---------------------------------------------------------------------------
# Loader entry point — `build_quantization_config_for_loader`
# ---------------------------------------------------------------------------


class TestLoaderEntryPoint:
    def _tcfg(self, **overrides):
        # Lightweight stand-in mirroring the TrainingConfig surface used by
        # build_quantization_config_for_loader. Keeps the test mock-free for
        # fields that don't need transformers.
        from soup_cli.config.schema import TrainingConfig

        return TrainingConfig(**overrides)

    def test_loader_returns_none_for_quantization_none(self):
        from soup_cli.utils.quant_menu import build_quantization_config_for_loader

        tcfg = self._tcfg(quantization="none")
        assert build_quantization_config_for_loader(tcfg=tcfg, base="m") is None

    def test_loader_unknown_quant_raises(self):
        from soup_cli.utils.quant_menu import build_quantization_config_for_loader

        # Bypass Pydantic — we want to confirm the loader has its own guard.
        class FakeTcfg:
            quantization = "totally-bogus"
            gptq_disable_exllama = True
            bnb_4bit_quant_storage = None

        with pytest.raises(ValueError, match="unknown quantization"):
            build_quantization_config_for_loader(
                tcfg=FakeTcfg(),  # type: ignore[arg-type]
                base="m",
            )
