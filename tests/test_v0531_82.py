"""v0.53.1 #82 — Autopilot pre-quantized base detection.

Tests for ``detect_prequantized_format`` + ``decide_quantization`` accepting an
optional pre-quantized hint so a base like ``TheBloke/Llama-2-7B-Chat-GPTQ`` is
recommended ``gptq`` instead of ``4bit``-on-top-of-already-quantized.
"""

from __future__ import annotations

import json

import pytest

# --- detect_prequantized_format ---------------------------------------------


class TestDetectPrequantizedFormat:
    def test_none_for_clean_name(self):
        from soup_cli.autopilot.decisions import detect_prequantized_format

        assert detect_prequantized_format("meta-llama/Llama-3.1-8B") is None

    def test_gptq_name_match(self):
        from soup_cli.autopilot.decisions import detect_prequantized_format

        assert (
            detect_prequantized_format("TheBloke/Llama-2-7B-Chat-GPTQ") == "gptq"
        )

    def test_gptq_lowercase(self):
        from soup_cli.autopilot.decisions import detect_prequantized_format

        assert detect_prequantized_format("some-org/llama-7b-gptq") == "gptq"

    def test_awq_name_match(self):
        from soup_cli.autopilot.decisions import detect_prequantized_format

        assert detect_prequantized_format("TheBloke/Mistral-7B-AWQ") == "awq"

    def test_hqq_name_match(self):
        from soup_cli.autopilot.decisions import detect_prequantized_format

        result = detect_prequantized_format("mobiuslabsgmbh/Llama-3.1-8B-HQQ-4bit")
        assert result == "hqq:4bit"

    def test_hqq_explicit_bits(self):
        from soup_cli.autopilot.decisions import detect_prequantized_format

        assert (
            detect_prequantized_format("some-org/model-HQQ-2bit") == "hqq:2bit"
        )

    def test_aqlm_name_match(self):
        from soup_cli.autopilot.decisions import detect_prequantized_format

        assert detect_prequantized_format("ISTA-DASLab/Llama-3-8B-AQLM") == "aqlm"

    def test_eetq_name_match(self):
        from soup_cli.autopilot.decisions import detect_prequantized_format

        assert detect_prequantized_format("some-org/model-EETQ") == "eetq"

    def test_fp8_name_match(self):
        from soup_cli.autopilot.decisions import detect_prequantized_format

        assert (
            detect_prequantized_format("neuralmagic/Meta-Llama-3-8B-FP8") == "fp8"
        )

    def test_config_quantization_method(self):
        from soup_cli.autopilot.decisions import detect_prequantized_format

        cfg = {"quantization_config": {"quant_method": "gptq", "bits": 4}}
        assert detect_prequantized_format("clean/name", cfg) == "gptq"

    def test_config_overrides_clean_name(self):
        from soup_cli.autopilot.decisions import detect_prequantized_format

        cfg = {"quantization_config": {"quant_method": "awq"}}
        assert detect_prequantized_format("meta/clean-llama", cfg) == "awq"

    def test_config_hqq_with_bits(self):
        from soup_cli.autopilot.decisions import detect_prequantized_format

        cfg = {"quantization_config": {"quant_method": "hqq", "bits": 2}}
        assert detect_prequantized_format("clean/name", cfg) == "hqq:2bit"

    def test_config_unknown_method(self):
        from soup_cli.autopilot.decisions import detect_prequantized_format

        cfg = {"quantization_config": {"quant_method": "weirdq"}}
        assert detect_prequantized_format("clean/name", cfg) is None

    def test_config_non_dict_quantization_config(self):
        from soup_cli.autopilot.decisions import detect_prequantized_format

        cfg = {"quantization_config": "gptq"}  # malformed
        assert detect_prequantized_format("clean/name", cfg) is None

    def test_config_non_dict_root(self):
        from soup_cli.autopilot.decisions import detect_prequantized_format

        assert detect_prequantized_format("clean/name", "not-a-dict") is None

    def test_empty_name_raises(self):
        from soup_cli.autopilot.decisions import detect_prequantized_format

        with pytest.raises(ValueError):
            detect_prequantized_format("")

    def test_null_byte_name_raises(self):
        from soup_cli.autopilot.decisions import detect_prequantized_format

        with pytest.raises(ValueError):
            detect_prequantized_format("evil\x00name")

    def test_non_string_name_raises(self):
        from soup_cli.autopilot.decisions import detect_prequantized_format

        with pytest.raises(TypeError):
            detect_prequantized_format(123)

    def test_bool_name_raises(self):
        from soup_cli.autopilot.decisions import detect_prequantized_format

        with pytest.raises(TypeError):
            detect_prequantized_format(True)

    def test_word_boundary_no_substring(self):
        from soup_cli.autopilot.decisions import detect_prequantized_format

        # 'agptqa' should NOT match — word boundary
        assert detect_prequantized_format("some-org/agptqa-model") is None

    def test_config_probe_path(self, tmp_path, monkeypatch):
        from soup_cli.autopilot.decisions import detect_prequantized_format_from_path

        monkeypatch.chdir(tmp_path)
        config_dir = tmp_path / "model"
        config_dir.mkdir()
        cfg_file = config_dir / "config.json"
        cfg_file.write_text(json.dumps({
            "quantization_config": {"quant_method": "gptq", "bits": 4},
        }), encoding="utf-8")

        assert detect_prequantized_format_from_path("./model") == "gptq"

    def test_config_probe_path_missing(self, tmp_path, monkeypatch):
        from soup_cli.autopilot.decisions import detect_prequantized_format_from_path

        monkeypatch.chdir(tmp_path)
        assert detect_prequantized_format_from_path("./nope") is None

    def test_config_probe_malformed_json(self, tmp_path, monkeypatch):
        from soup_cli.autopilot.decisions import detect_prequantized_format_from_path

        monkeypatch.chdir(tmp_path)
        config_dir = tmp_path / "model"
        config_dir.mkdir()
        (config_dir / "config.json").write_text("{not json", encoding="utf-8")

        # Should not raise; returns None
        assert detect_prequantized_format_from_path("./model") is None

    def test_config_probe_path_outside_cwd_returns_none(
        self, tmp_path, monkeypatch,
    ):
        """Security review H2 — out-of-cwd model_dir silently falls through."""
        from soup_cli.autopilot.decisions import detect_prequantized_format_from_path

        monkeypatch.chdir(tmp_path)
        outside = tmp_path.parent / "outside_probe"
        outside.mkdir(exist_ok=True)
        (outside / "config.json").write_text(
            '{"quantization_config": {"quant_method": "gptq"}}',
            encoding="utf-8",
        )
        # Out-of-cwd path: soft-probe returns None (no read attempted)
        assert detect_prequantized_format_from_path(str(outside)) is None


# --- decide_quantization with prequantized hint -----------------------------


class TestDecideQuantizationPrequantized:
    def test_prequantized_hint_returned(self):
        from soup_cli.autopilot.decisions import decide_quantization

        # Even with plenty VRAM, prequantized hint takes precedence
        result = decide_quantization(
            model_params_b=7.0, vram_gb=80.0, prequantized="gptq",
        )
        assert result == "gptq"

    def test_prequantized_awq(self):
        from soup_cli.autopilot.decisions import decide_quantization

        assert (
            decide_quantization(
                model_params_b=7.0, vram_gb=24.0, prequantized="awq",
            )
            == "awq"
        )

    def test_prequantized_hqq(self):
        from soup_cli.autopilot.decisions import decide_quantization

        assert (
            decide_quantization(
                model_params_b=7.0, vram_gb=24.0, prequantized="hqq:4bit",
            )
            == "hqq:4bit"
        )

    def test_no_prequantized_falls_through_to_vram_logic(self):
        from soup_cli.autopilot.decisions import decide_quantization

        # Same as legacy behaviour when prequantized=None
        assert (
            decide_quantization(model_params_b=7.0, vram_gb=80.0)
            == "none"
        )

    def test_invalid_prequantized_raises(self):
        from soup_cli.autopilot.decisions import decide_quantization

        with pytest.raises(ValueError):
            decide_quantization(
                model_params_b=7.0, vram_gb=24.0, prequantized="evilq",
            )

    def test_prequantized_bool_rejected(self):
        from soup_cli.autopilot.decisions import decide_quantization

        with pytest.raises(TypeError):
            decide_quantization(
                model_params_b=7.0, vram_gb=24.0, prequantized=True,
            )

    def test_prequantized_null_byte_rejected(self):
        from soup_cli.autopilot.decisions import decide_quantization

        with pytest.raises(ValueError):
            decide_quantization(
                model_params_b=7.0, vram_gb=24.0, prequantized="ev\x00il",
            )

    def test_prequantized_none_legacy(self):
        from soup_cli.autopilot.decisions import decide_quantization

        # Explicit None == no hint == legacy behaviour
        assert (
            decide_quantization(
                model_params_b=15.0, vram_gb=24.0, prequantized=None,
            )
            == "4bit"
        )

    def test_mxfp4_name_match(self):
        """L2: mxfp4 word-boundary regex coverage."""
        from soup_cli.autopilot.decisions import detect_prequantized_format

        assert detect_prequantized_format("some-org/model-MXFP4") == "mxfp4"
        assert detect_prequantized_format("some-org/notmxfp4good") is None

    def test_bnb_4bit_alias_via_config(self):
        """L5: config quant_method=bitsandbytes_4bit aliases to '4bit'."""
        from soup_cli.autopilot.decisions import detect_prequantized_format

        cfg = {"quantization_config": {"quant_method": "bitsandbytes_4bit"}}
        assert detect_prequantized_format("clean/name", cfg) == "4bit"

    def test_bnb_8bit_alias_via_config(self):
        """L5: config quant_method=bnb_8bit aliases to '8bit'."""
        from soup_cli.autopilot.decisions import detect_prequantized_format

        cfg = {"quantization_config": {"quant_method": "bnb_8bit"}}
        assert detect_prequantized_format("clean/name", cfg) == "8bit"

    def test_decide_quantization_accepts_4bit_alias(self):
        """L5: ``prequantized='4bit'`` short-circuits VRAM heuristic."""
        from soup_cli.autopilot.decisions import decide_quantization

        # Even with plenty of VRAM, '4bit' wins
        assert (
            decide_quantization(
                model_params_b=7.0, vram_gb=80.0, prequantized="4bit",
            )
            == "4bit"
        )
        assert (
            decide_quantization(
                model_params_b=7.0, vram_gb=80.0, prequantized="8bit",
            )
            == "8bit"
        )
