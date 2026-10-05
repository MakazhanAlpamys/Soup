"""Tests for new PEFT methods — VeRA + OLoRA (Part D of v0.25.0)."""

import pytest

# ---------------------------------------------------------------------------
# LoraConfig schema fields
# ---------------------------------------------------------------------------


class TestLoraConfigFields:
    def test_use_vera_default_false(self):
        from soup_cli.config.schema import LoraConfig

        cfg = LoraConfig()
        assert cfg.use_vera is False

    def test_use_olora_default_false(self):
        from soup_cli.config.schema import LoraConfig

        cfg = LoraConfig()
        assert cfg.use_olora is False

    def test_use_vera_enabled(self):
        from soup_cli.config.schema import LoraConfig

        cfg = LoraConfig(use_vera=True)
        assert cfg.use_vera is True

    def test_use_olora_enabled(self):
        from soup_cli.config.schema import LoraConfig

        cfg = LoraConfig(use_olora=True)
        assert cfg.use_olora is True


# ---------------------------------------------------------------------------
# Mutual exclusion
# ---------------------------------------------------------------------------


class TestPeftMutualExclusion:
    def test_vera_and_olora_rejected(self):
        from pydantic import ValidationError

        from soup_cli.config.schema import LoraConfig

        with pytest.raises(ValidationError):
            LoraConfig(use_vera=True, use_olora=True)

    def test_vera_and_dora_rejected(self):
        from pydantic import ValidationError

        from soup_cli.config.schema import LoraConfig

        with pytest.raises(ValidationError):
            LoraConfig(use_vera=True, use_dora=True)

    def test_olora_and_dora_rejected(self):
        from pydantic import ValidationError

        from soup_cli.config.schema import LoraConfig

        with pytest.raises(ValidationError):
            LoraConfig(use_olora=True, use_dora=True)


# ---------------------------------------------------------------------------
# Peft builder helper
# ---------------------------------------------------------------------------


class TestPeftBuilder:
    def test_standard_lora_returns_lora_config(self):
        from soup_cli.config.schema import LoraConfig as SchemaLoraConfig
        from soup_cli.utils.peft_builder import build_peft_config

        schema_cfg = SchemaLoraConfig()
        result = build_peft_config(
            schema_cfg,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )
        # Expect dict with `peft_cls` key + init kwargs
        assert result["peft_cls"] == "LoraConfig"

    def test_olora_adds_init_weights(self):
        from soup_cli.config.schema import LoraConfig as SchemaLoraConfig
        from soup_cli.utils.peft_builder import build_peft_config

        schema_cfg = SchemaLoraConfig(use_olora=True)
        result = build_peft_config(
            schema_cfg,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )
        assert result["peft_cls"] == "LoraConfig"
        assert result["init_kwargs"].get("init_lora_weights") == "olora"

    def test_vera_returns_vera_config(self):
        from soup_cli.config.schema import LoraConfig as SchemaLoraConfig
        from soup_cli.utils.peft_builder import build_peft_config

        schema_cfg = SchemaLoraConfig(use_vera=True)
        result = build_peft_config(
            schema_cfg,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )
        assert result["peft_cls"] == "VeraConfig"

    def test_dora_preserved(self):
        from soup_cli.config.schema import LoraConfig as SchemaLoraConfig
        from soup_cli.utils.peft_builder import build_peft_config

        schema_cfg = SchemaLoraConfig(use_dora=True)
        result = build_peft_config(
            schema_cfg,
            target_modules=["q_proj"],
            task_type="CAUSAL_LM",
        )
        assert result["peft_cls"] == "LoraConfig"
        assert result["init_kwargs"].get("use_dora") is True

    def test_target_modules_propagated_lora(self):
        from soup_cli.config.schema import LoraConfig as SchemaLoraConfig
        from soup_cli.utils.peft_builder import build_peft_config

        modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        result = build_peft_config(
            SchemaLoraConfig(),
            target_modules=modules,
            task_type="CAUSAL_LM",
        )
        assert result["init_kwargs"]["target_modules"] == modules

    def test_target_modules_propagated_vera(self):
        from soup_cli.config.schema import LoraConfig as SchemaLoraConfig
        from soup_cli.utils.peft_builder import build_peft_config

        modules = ["q_proj", "v_proj"]
        result = build_peft_config(
            SchemaLoraConfig(use_vera=True),
            target_modules=modules,
            task_type="CAUSAL_LM",
        )
        assert result["init_kwargs"]["target_modules"] == modules

    def test_task_type_propagated(self):
        from soup_cli.config.schema import LoraConfig as SchemaLoraConfig
        from soup_cli.utils.peft_builder import build_peft_config

        result = build_peft_config(
            SchemaLoraConfig(),
            target_modules=["q_proj"],
            task_type="SEQ_CLS",
        )
        assert result["init_kwargs"]["task_type"] == "SEQ_CLS"


# ---------------------------------------------------------------------------
# Sweep integration
# ---------------------------------------------------------------------------


class TestPeftSweep:
    def test_sweep_accepts_use_vera(self, tmp_path, monkeypatch):
        from soup_cli.commands.sweep import _parse_sweep_params

        params = _parse_sweep_params(["lora.use_vera=true,false"])
        assert "lora.use_vera" in params

    def test_sweep_accepts_use_olora(self):
        from soup_cli.commands.sweep import _parse_sweep_params

        params = _parse_sweep_params(["lora.use_olora=true,false"])
        assert "lora.use_olora" in params


# ---------------------------------------------------------------------------
# End-to-end config loads
# ---------------------------------------------------------------------------


class TestPeftYamlConfig:
    def test_yaml_with_vera(self):
        from soup_cli.config.loader import load_config_from_string

        yaml_str = """
base: meta-llama/Llama-3.1-8B-Instruct
task: sft
data:
  train: ./data/train.jsonl
  format: auto
training:
  epochs: 1
  lora:
    use_vera: true
output: ./output
"""
        cfg = load_config_from_string(yaml_str)
        assert cfg.training.lora.use_vera is True
        assert cfg.training.lora.use_olora is False

    def test_yaml_with_olora(self):
        from soup_cli.config.loader import load_config_from_string

        yaml_str = """
base: meta-llama/Llama-3.1-8B-Instruct
task: sft
data:
  train: ./data/train.jsonl
  format: auto
training:
  epochs: 1
  lora:
    use_olora: true
output: ./output
"""
        cfg = load_config_from_string(yaml_str)
        assert cfg.training.lora.use_olora is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
