"""#1466: `quantization: mxfp4` must load, not die after the model load.

MXFP4 is transformers' own format (`Mxfp4Config`), not a bitsandbytes
`quant_type`: bitsandbytes accepts only `nf4` and `fp4`, so the
`BitsAndBytesConfig(bnb_4bit_quant_type="mxfp4")` Soup used to build was
rejected for every linear layer, and clashed with the `Mxfp4Config` a
pre-quantized checkpoint (the one Autopilot passes through) carries.
transformers also refuses to train an MXFP4 model as loaded and points at
`Mxfp4Config(dequantize=True)`, which is what the loader now returns.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from soup_cli.config.loader import load_config_from_string

_MXFP4_YAML = """
base: {base}
task: sft
data: {{train: d.jsonl}}
training: {{quantization: mxfp4}}
"""


def _tiny_llama(path: Path, *, mxfp4: bool) -> str:
    """A two-layer Llama on disk; `mxfp4` marks its config as the checkpoint
    of a pre-quantized base does."""
    import torch
    import transformers

    config = transformers.LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    transformers.LlamaForCausalLM(config).to(torch.bfloat16).save_pretrained(path)
    if mxfp4:
        config_path = path / "config.json"
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        raw["quantization_config"] = {"quant_method": "mxfp4"}
        config_path.write_text(json.dumps(raw), encoding="utf-8")
    return str(path)


def _loader_config(base: str):
    from soup_cli.utils.quant_menu import build_quantization_config_for_loader

    cfg = load_config_from_string(_MXFP4_YAML.format(base=base))
    return build_quantization_config_for_loader(tcfg=cfg.training, base=cfg.base)


class TestLoaderConfig:
    def test_mxfp4_is_transformers_own_config_dequantized(self, tmp_path):
        from transformers import BitsAndBytesConfig, Mxfp4Config

        quant = _loader_config(_tiny_llama(tmp_path / "base", mxfp4=True))

        assert isinstance(quant, Mxfp4Config)
        assert not isinstance(quant, BitsAndBytesConfig)
        # transformers cannot train an MXFP4 model as loaded.
        assert quant.dequantize is True

    def test_a_base_marked_mxfp4_loads_with_the_loader_config(self, tmp_path):
        """The Autopilot pass-through case: the checkpoint says Mxfp4Config."""
        import torch
        import transformers

        base = _tiny_llama(tmp_path / "base", mxfp4=True)

        model = transformers.AutoModelForCausalLM.from_pretrained(
            base, quantization_config=_loader_config(base), dtype=torch.bfloat16
        )

        assert model.model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16

    def test_autopilot_pass_through_round_trips_to_the_same_config(self, tmp_path):
        from transformers import Mxfp4Config

        from soup_cli.autopilot.decisions import (
            decide_quantization,
            detect_prequantized_format,
        )

        fmt = decide_quantization(
            20.0,
            80.0,
            prequantized=detect_prequantized_format(
                "openai/gpt-oss-20b", {"quantization_config": {"quant_method": "mxfp4"}}
            ),
        )

        assert fmt == "mxfp4"
        base = _tiny_llama(tmp_path / "base", mxfp4=True)
        assert isinstance(_loader_config(base), Mxfp4Config)

    def test_a_hub_id_is_taken_on_faith(self):
        from transformers import Mxfp4Config

        assert isinstance(_loader_config("openai/gpt-oss-20b"), Mxfp4Config)


class TestLocalBaseThatIsNotMxfp4:
    def test_a_plain_local_base_is_refused(self, tmp_path):
        """Dequantizing a base that was never MXFP4 loads it as it is, so the
        run would train unquantized while the config says otherwise."""
        base = _tiny_llama(tmp_path / "plain", mxfp4=False)

        with pytest.raises(ValueError, match="mxfp4") as raised:
            _loader_config(base)

        # The directory name, not the absolute path.
        assert str(tmp_path) not in str(raised.value)

    def test_a_local_base_of_another_format_is_refused(self, tmp_path):
        base = Path(_tiny_llama(tmp_path / "other", mxfp4=False))
        raw = json.loads((base / "config.json").read_text(encoding="utf-8"))
        raw["quantization_config"] = {"quant_method": "gptq"}
        (base / "config.json").write_text(json.dumps(raw), encoding="utf-8")

        with pytest.raises(ValueError, match="mxfp4"):
            _loader_config(str(base))


class TestSchema:
    def test_quant_storage_is_bitsandbytes_only(self):
        """`bnb_4bit_quant_storage` is a BitsAndBytesConfig argument; MXFP4
        has no such field, so it would be a silent no-op."""
        with pytest.raises(ValueError, match="bnb_4bit_quant_storage"):
            load_config_from_string(
                """
base: m
task: sft
data: {train: d.jsonl}
training: {quantization: mxfp4, bnb_4bit_quant_storage: bfloat16}
"""
            )

    def test_the_field_no_longer_calls_mxfp4_a_bnb_type(self):
        from soup_cli.config.schema import TrainingConfig

        description = TrainingConfig.model_fields["quantization"].description or ""
        assert "BNB 4-bit MXFP4" not in description


class TestKbitPrep:
    def test_kbit_prep_would_upcast_a_dequantized_model(self, tmp_path):
        """Why `mxfp4` left the trainers' k-bit tuple (pinned per trainer in
        test_v0405_part_a.py): the dequantized model is plain bf16, and peft
        casts every bf16 parameter of a model that is not k-bit to fp32."""
        import torch
        import transformers
        from peft import prepare_model_for_kbit_training

        base = _tiny_llama(tmp_path / "base", mxfp4=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            base, quantization_config=_loader_config(base), dtype=torch.bfloat16
        )

        prepared = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=False
        )

        weight = prepared.model.layers[0].self_attn.q_proj.weight
        assert weight.dtype == torch.float32
