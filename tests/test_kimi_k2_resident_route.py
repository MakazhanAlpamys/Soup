"""The resident route's guarded config loader (#2104): load_resident_model_config.

A local twin of the streamed loader's tests, built on synthetic tmp_path configs only
(no network). The streamed route's wording must never leak into a resident refusal.
"""

import json

import pytest

pytest.importorskip("transformers")

#: A minimal K2-shaped config: every published required key present (tiny dimensions), no
#: auto_map, no quantization_config (both are optional). Spelled out inline on purpose: the
#: shared tiny builder lives in tests/test_kimi_k2_streaming.py and its extraction is Task 4.
_MINIMAL_K2 = {
    "architectures": ["DeepseekV3ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "auto_map": {
        "AutoConfig": "fake_remote_module.FakeConfig",
        "AutoModelForCausalLM": "fake_remote_module.FakeModel",
    },
    "aux_loss_alpha": 0.001,
    "bos_token_id": 0,
    "eos_token_id": 1,
    "first_k_dense_replace": 1,
    "hidden_act": "silu",
    "hidden_size": 64,
    "initializer_range": 0.02,
    "intermediate_size": 128,
    "kv_lora_rank": 16,
    "max_position_embeddings": 2048,
    "model_type": "kimi_k2",
    "moe_intermediate_size": 32,
    "moe_layer_freq": 1,
    "n_group": 1,
    "n_routed_experts": 8,
    "n_shared_experts": 1,
    "norm_topk_prob": True,
    "num_attention_heads": 4,
    "num_experts_per_tok": 2,
    "num_hidden_layers": 3,
    "num_key_value_heads": 4,
    "num_nextn_predict_layers": 0,
    "pretraining_tp": 1,
    "q_lora_rank": 24,
    "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 8,
    "rms_norm_eps": 1e-06,
    "rope_theta": 50000.0,
    "routed_scaling_factor": 2.827,
    "rope_scaling": {
        "beta_fast": 1.0,
        "beta_slow": 1.0,
        "factor": 32.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 64,
        "type": "yarn",
    },
    "scoring_func": "sigmoid",
    "seq_aux": True,
    "tie_word_embeddings": False,
    "topk_group": 1,
    "topk_method": "noaux_tc",
    "torch_dtype": "bfloat16",
    "transformers_version": "4.51.3",
    "use_cache": True,
    "v_head_dim": 16,
    "vocab_size": 128,
}


def _write_config(tmp_path, raw):
    config_dir = tmp_path / "model"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    return str(config_dir)


def test_resident_loads_a_guarded_native_config(tmp_path):
    from transformers import DeepseekV3Config

    from soup_cli.utils.kimi_k2 import load_resident_model_config

    config = load_resident_model_config(_write_config(tmp_path, dict(_MINIMAL_K2)))
    assert isinstance(config, DeepseekV3Config)
    assert config.model_type == "deepseek_v3"
    # The rewrite is in-memory only: auto_map is dropped from the config, not just the file.
    assert "auto_map" not in config.to_diff_dict()
    assert not hasattr(config, "auto_map")


def test_resident_refusal_names_the_field_without_streamed_wording(tmp_path):
    from soup_cli.utils.kimi_k2 import (
        UnsupportedKimiK2Error,
        load_resident_model_config,
    )

    raw = dict(_MINIMAL_K2)
    del raw["routed_scaling_factor"]
    with pytest.raises(UnsupportedKimiK2Error) as caught:
        load_resident_model_config(_write_config(tmp_path, raw))
    message = str(caught.value)
    assert "routed_scaling_factor" in message
    assert "stream" not in message
    assert "layer streaming" not in message
    assert "training.stream_layers" not in message


def test_resident_falls_through_to_autoconfig_for_other_model_types(tmp_path):
    from transformers import AutoConfig, LlamaConfig

    from soup_cli.utils.kimi_k2 import load_resident_model_config

    llama = {
        "model_type": "llama",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_attention_heads": 2,
        "num_hidden_layers": 1,
        "num_key_value_heads": 2,
        "vocab_size": 128,
    }
    config_dir = _write_config(tmp_path, llama)
    loaded = load_resident_model_config(config_dir)
    direct = AutoConfig.from_pretrained(config_dir)
    assert type(loaded) is type(direct) is LlamaConfig
    assert loaded.hidden_size == 32
