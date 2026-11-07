"""Layer streaming admits Kimi K2 through transformers' own ``DeepseekV3Config``.

The gate is ``benchmarks/gate-kimi-k2-admission.md`` (G1-G3, pre-registered). Kimi-K2-Base and
-Instruct declare ``model_type: kimi_k2`` and an ``auto_map`` onto their repository's own code,
which layer streaming never runs (#1933), so ``AutoConfig`` refused them as custom code before
the architecture allowlist was reached. ``utils/kimi_k2.py`` loads the raw ``config.json`` into
the native class (``model_type`` -> ``deepseek_v3``, ``auto_map`` dropped) behind a guard; every
variant the guard refuses is refused by name, naming the field and its value.

Nothing here imports or executes K2's own Python. The router and yarn references in the G1c
section are transcribed by hand from K2's ``modeling_deepseek.py`` read as text.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import shutil
from types import SimpleNamespace

import pytest

#: moonshotai/Kimi-K2-Base ``config.json`` at revision ce72df012259dcc55d945e890f815fe7ef69159c
#: (sha256 8c13ae1049df55f29b3bdcae69a562433f243ff70dac251d819ecad8dbdf7439), verbatim. The
#: Kimi-K2-Instruct file at revision fd1984e2b7a3350dbf7305fe73a4ede25c14de50 is byte-identical.
K2_BASE_CONFIG = {
    "architectures": ["DeepseekV3ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "auto_map": {
        "AutoConfig": "configuration_deepseek.DeepseekV3Config",
        "AutoModel": "modeling_deepseek.DeepseekV3Model",
        "AutoModelForCausalLM": "modeling_deepseek.DeepseekV3ForCausalLM",
    },
    "aux_loss_alpha": 0.001,
    "bos_token_id": 163584,
    "eos_token_id": 163585,
    "first_k_dense_replace": 1,
    "hidden_act": "silu",
    "hidden_size": 7168,
    "initializer_range": 0.02,
    "intermediate_size": 18432,
    "kv_lora_rank": 512,
    "max_position_embeddings": 131072,
    "model_type": "kimi_k2",
    "moe_intermediate_size": 2048,
    "moe_layer_freq": 1,
    "n_group": 1,
    "n_routed_experts": 384,
    "n_shared_experts": 1,
    "norm_topk_prob": True,
    "num_attention_heads": 64,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 61,
    "num_key_value_heads": 64,
    "num_nextn_predict_layers": 0,
    "pretraining_tp": 1,
    "q_lora_rank": 1536,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "quantization_config": {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
    },
    "rms_norm_eps": 1e-06,
    "rope_theta": 50000.0,
    "routed_scaling_factor": 2.827,
    "rope_scaling": {
        "beta_fast": 1.0,
        "beta_slow": 1.0,
        "factor": 32.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
        "type": "yarn",
    },
    "scoring_func": "sigmoid",
    "seq_aux": True,
    "tie_word_embeddings": False,
    "topk_group": 1,
    "topk_method": "noaux_tc",
    "torch_dtype": "bfloat16",
    "transformers_version": "4.48.3",
    "use_cache": True,
    "v_head_dim": 128,
    "vocab_size": 163840,
}

#: The fields moonshotai/Kimi-K2-Thinking's ``config.json`` (revision
#: a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55, sha256
#: 666ee58fc3871ecc369a22bc8c100075c2ad24ccbe8b3e336fa5f726e6111c15) sets differently from Base,
#: trimmed: its serialised generation defaults (``do_sample``, ``top_k``, ...) are left out.
K2_THINKING_CHANGES = {
    "eos_token_id": 163586,
    "ep_size": 1,
    "max_position_embeddings": 262144,
    "pad_token_id": 163839,
    "quantization_config": {
        "config_groups": {
            "group_0": {
                "input_activations": None,
                "output_activations": None,
                "targets": ["Linear"],
                "weights": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": False,
                    "group_size": 32,
                    "num_bits": 4,
                    "observer": "minmax",
                    "observer_kwargs": {},
                    "strategy": "group",
                    "symmetric": True,
                    "type": "int",
                },
            }
        },
        "format": "pack-quantized",
        "ignore": [
            "lm_head",
            "re:.*self_attn.*",
            "re:.*shared_experts.*",
            "re:.*mlp\\.(gate|up|gate_up|down)_proj.*",
        ],
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
    },
    "rms_norm_eps": 1e-05,
    "rope_scaling": {
        "beta_fast": 1.0,
        "beta_slow": 1.0,
        "factor": 64.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
        "type": "yarn",
    },
    "transformers_version": "4.51.3",
}

#: Every field moonshotai/Kimi-K2-Instruct-0905's ``config.json`` (revision
#: ac6c49f04883bd0a0598b790693a72061c676629, sha256
#: 4cd10b0f3cb1c1dfdbfcdd2f303057500f570b24f671ce5eb7650b6a511807cc; the file only, read by the
#: second review) sets differently from Base, untrimmed. It drops no Base key. The decoder is
#: the same; beyond max_position_embeddings, rms_norm_eps and the yarn factor it adds
#: ``ep_size``, ``pad_token_id`` and the serialised ``PretrainedConfig`` defaults.
K2_INSTRUCT_0905_CHANGES = {
    "_attn_implementation_autoset": False, "_name_or_path": "", "add_cross_attention": False,
    "bad_words_ids": None, "begin_suppress_tokens": None, "chunk_size_feed_forward": 0,
    "cross_attention_hidden_size": None, "decoder_start_token_id": None,
    "diversity_penalty": 0.0, "do_sample": False, "early_stopping": False,
    "encoder_no_repeat_ngram_size": 0, "ep_size": 1, "exponential_decay_length_penalty": None,
    "finetuning_task": None, "forced_bos_token_id": None, "forced_eos_token_id": None,
    "id2label": {"0": "LABEL_0", "1": "LABEL_1"}, "is_decoder": False,
    "is_encoder_decoder": False, "label2id": {"LABEL_0": 0, "LABEL_1": 1},
    "length_penalty": 1.0, "max_length": 20, "max_position_embeddings": 262144, "min_length": 0,
    "no_repeat_ngram_size": 0, "num_beam_groups": 1, "num_beams": 1, "num_return_sequences": 1,
    "output_attentions": False, "output_hidden_states": False, "output_scores": False,
    "pad_token_id": 163839, "prefix": None, "problem_type": None, "pruned_heads": {},
    "remove_invalid_values": False, "repetition_penalty": 1.0, "return_dict": True,
    "return_dict_in_generate": False, "rms_norm_eps": 1e-05,
    "rope_scaling": {
        "beta_fast": 1.0, "beta_slow": 1.0, "factor": 64.0, "mscale": 1.0, "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096, "type": "yarn",
    },
    "sep_token_id": None, "suppress_tokens": None, "task_specific_params": None,
    "temperature": 1.0, "tf_legacy_loss": False, "tie_encoder_decoder": False,
    "tokenizer_class": None, "top_k": 50, "top_p": 1.0, "torchscript": False,
    "transformers_version": "4.51.3", "typical_p": 1.0, "use_bfloat16": False,
}


def _base(**changes):
    raw = copy.deepcopy(K2_BASE_CONFIG)
    raw.update(copy.deepcopy(changes))
    return raw


def _without(*keys, raw=None):
    raw = _base() if raw is None else raw
    for key in keys:
        raw.pop(key)
    return raw


def _rope(**changes):
    rope = dict(K2_BASE_CONFIG["rope_scaling"], **changes)
    return {"rope_scaling": rope}


def _rope_without(*keys):
    rope = dict(K2_BASE_CONFIG["rope_scaling"])
    for key in keys:
        rope.pop(key)
    return {"rope_scaling": rope}


def _write_config(directory, raw):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    return str(directory)


def _native_dict(raw):
    """The rewrite done by hand: what the tests compare Soup's against."""
    native = {key: value for key, value in copy.deepcopy(raw).items() if key != "auto_map"}
    native["model_type"] = "deepseek_v3"
    return native


# ==========================================================================
# G1a: the rewritten config carries every K2 field
# ==========================================================================
def test_the_published_base_config_loads_into_deepseek_v3_field_by_field(tmp_path):
    pytest.importorskip("transformers")
    from transformers import DeepseekV3Config

    from soup_cli.utils.kimi_k2 import load_stream_model_config

    config = load_stream_model_config(_write_config(tmp_path / "k2", K2_BASE_CONFIG))
    assert type(config) is DeepseekV3Config
    assert config.model_type == "deepseek_v3"
    assert getattr(config, "auto_map", None) is None
    for key, value in K2_BASE_CONFIG.items():
        if key in ("model_type", "auto_map"):
            continue
        if key == "torch_dtype":
            assert str(config.torch_dtype) == "torch." + value
            continue
        got = config.rope_parameters["rope_theta"] if key == "rope_theta" else getattr(config, key)
        if key == "rope_scaling":
            # transformers' rope standardisation adds two keys to the dict it is handed (a
            # native deepseek_v3 config.json gets the same); every K2 key keeps its value.
            assert {name: got[name] for name in value} == value
            assert {name: got[name] for name in set(got) - set(value)} == {
                "rope_theta": K2_BASE_CONFIG["rope_theta"], "rope_type": "yarn"
            }
            continue
        assert got == value, key
    for key, value in K2_BASE_CONFIG["rope_scaling"].items():
        assert config.rope_parameters[key] == value, key
    assert config.rope_parameters["rope_type"] == "yarn"


def test_autoconfig_alone_still_refuses_the_k2_folder_as_custom_code(tmp_path):
    """Today's failure, the control for the test above: the admission is the rewrite."""
    pytest.importorskip("transformers")
    from transformers import AutoConfig

    folder = _write_config(tmp_path / "k2", K2_BASE_CONFIG)
    with pytest.raises(ValueError, match="custom code"):
        AutoConfig.from_pretrained(folder, trust_remote_code=False)


def test_the_k2_path_builds_what_autoconfig_builds_for_the_same_dict(tmp_path):
    pytest.importorskip("transformers")
    from transformers import AutoConfig

    from soup_cli.utils.kimi_k2 import load_stream_model_config

    k2_folder = _write_config(tmp_path / "k2", K2_BASE_CONFIG)
    native_folder = _write_config(tmp_path / "native", _native_dict(K2_BASE_CONFIG))
    k2 = load_stream_model_config(k2_folder)
    native = AutoConfig.from_pretrained(native_folder)
    assert (k2.name_or_path, native.name_or_path) == (k2_folder, native_folder)
    left, right = k2.to_dict(), native.to_dict()
    left.pop("_name_or_path")
    right.pop("_name_or_path")
    assert left == right


def test_the_loader_never_loads_remote_code(tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    import transformers.dynamic_module_utils as dynamic
    import transformers.models.auto.configuration_auto as configuration_auto

    from soup_cli.utils.kimi_k2 import load_stream_model_config

    def _refuse(*_args, **_kwargs):
        raise AssertionError("remote code was loaded")

    monkeypatch.setattr(dynamic, "get_class_from_dynamic_module", _refuse)
    monkeypatch.setattr(configuration_auto, "get_class_from_dynamic_module", _refuse)
    config = load_stream_model_config(_write_config(tmp_path / "k2", K2_BASE_CONFIG))
    assert config.model_type == "deepseek_v3"


def test_the_rewrite_copies_deeply_and_changes_only_model_type_and_auto_map():
    from soup_cli.utils.kimi_k2 import kimi_k2_native_config_dict

    raw = _base()
    native = kimi_k2_native_config_dict(raw)
    assert raw == K2_BASE_CONFIG
    assert native == _native_dict(K2_BASE_CONFIG)
    # transformers' rope standardisation writes into the rope dict it is handed.
    assert native["rope_scaling"] is not raw["rope_scaling"]
    assert native["quantization_config"] is not raw["quantization_config"]


def test_a_config_that_is_not_k2_goes_through_autoconfig_unchanged(tmp_path):
    pytest.importorskip("transformers")
    from transformers import AutoConfig

    from soup_cli.utils.kimi_k2 import load_stream_model_config

    llama = {
        "model_type": "llama", "hidden_size": 8, "intermediate_size": 16,
        "num_attention_heads": 2, "num_hidden_layers": 1, "vocab_size": 32,
    }
    folder = _write_config(tmp_path / "llama", llama)
    got, want = load_stream_model_config(folder), AutoConfig.from_pretrained(folder)
    assert type(got) is type(want)
    assert got.to_dict() == want.to_dict()


def test_another_custom_code_repo_keeps_transformers_own_error(tmp_path):
    pytest.importorskip("transformers")
    from transformers import AutoConfig

    from soup_cli.utils.kimi_k2 import UnsupportedKimiK2Error, load_stream_model_config

    raw = {"model_type": "acme_lm", "auto_map": {"AutoConfig": "configuration_acme.AcmeConfig"}}
    folder = _write_config(tmp_path / "acme", raw)
    with pytest.raises(ValueError) as soup_error:
        load_stream_model_config(folder)
    with pytest.raises(ValueError) as hf_error:
        AutoConfig.from_pretrained(folder, trust_remote_code=False)
    assert not isinstance(soup_error.value, UnsupportedKimiK2Error)
    assert str(soup_error.value) == str(hf_error.value)


def test_a_folder_without_a_config_keeps_transformers_own_error(tmp_path):
    pytest.importorskip("transformers")
    from transformers import AutoConfig

    from soup_cli.utils.kimi_k2 import UnsupportedKimiK2Error, load_stream_model_config

    folder = tmp_path / "empty"
    folder.mkdir()
    with pytest.raises(Exception) as soup_error:
        load_stream_model_config(str(folder))
    with pytest.raises(Exception) as hf_error:
        AutoConfig.from_pretrained(str(folder), trust_remote_code=False)
    assert not isinstance(soup_error.value, UnsupportedKimiK2Error)
    assert type(soup_error.value) is type(hf_error.value)
    assert str(soup_error.value) == str(hf_error.value)


def test_trust_remote_code_true_goes_to_autoconfig_without_a_rewrite(monkeypatch):
    pytest.importorskip("transformers")
    import transformers

    from soup_cli.utils.kimi_k2 import load_stream_model_config

    seen = []
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained",
        staticmethod(lambda model_id, **kwargs: seen.append((model_id, kwargs)) or "remote"),
    )
    assert load_stream_model_config("some/repo", trust_remote_code=True) == "remote"
    assert seen == [("some/repo", {"trust_remote_code": True})]


def test_revision_kwargs_reach_both_config_reads(monkeypatch):
    pytest.importorskip("transformers")
    import transformers

    from soup_cli.utils.kimi_k2 import load_stream_model_config

    seen = {}

    def _auto(model_id, **kwargs):
        seen["auto"] = (model_id, kwargs)
        raise ValueError("contains custom code")

    def _raw(cls, model_id, **kwargs):
        seen["raw"] = (model_id, kwargs)
        return copy.deepcopy(K2_BASE_CONFIG), {}

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", staticmethod(_auto))
    monkeypatch.setattr(transformers.PretrainedConfig, "get_config_dict", classmethod(_raw))
    config = load_stream_model_config("moonshotai/Kimi-K2-Base", revision="abc123")
    assert config.model_type == "deepseek_v3"
    assert seen["auto"] == (
        "moonshotai/Kimi-K2-Base", {"trust_remote_code": False, "revision": "abc123"}
    )
    assert seen["raw"][0] == "moonshotai/Kimi-K2-Base"
    assert seen["raw"][1]["revision"] == "abc123"


def test_the_kimi_k2_model_type_alone_is_still_refused_by_the_allowlist():
    """A config that did not come through the guarded loader (K2's own remote config class
    keeps ``model_type == 'kimi_k2'``) is still refused, naming the admitted family."""
    from soup_cli.utils.layer_stream import stream_arch_of

    with pytest.raises(ValueError, match="deepseek_v3") as caught:
        stream_arch_of(SimpleNamespace(model_type="kimi_k2"))
    assert "kimi_k2" in str(caught.value)


@pytest.mark.parametrize("text_type", ["kimi_k2", "deepseek_v3"])
def test_kimi_k25_stays_refused_by_name(text_type):
    """K2.5 / K2.6 are a vision-language wrapper (``kimi_k25``): not a K2 the guard admits."""
    from soup_cli.utils.layer_stream import stream_arch_of

    text_config = SimpleNamespace(model_type=text_type)
    config = SimpleNamespace(model_type="kimi_k25", text_config=text_config)
    with pytest.raises(ValueError, match="kimi_k25"):
        stream_arch_of(config)


def test_the_rewritten_k2_is_keyed_like_deepseek_v3_everywhere_streaming_keys_on_arch(tmp_path):
    pytest.importorskip("transformers")
    from soup_cli.utils.fp8_block import FP8_BLOCK_ARCHS, fp8_block_size
    from soup_cli.utils.kimi_k2 import load_stream_model_config
    from soup_cli.utils.layer_shard import (
        PER_EXPERT_LAYOUT_ARCHS,
        layer_buffer_keys,
        source_identity_includes_config,
    )
    from soup_cli.utils.layer_stream import stream_arch_of

    config = load_stream_model_config(_write_config(tmp_path / "k2", K2_BASE_CONFIG))
    arch = stream_arch_of(config)
    assert arch == "deepseek_v3"
    assert arch in PER_EXPERT_LAYOUT_ARCHS and arch in FP8_BLOCK_ARCHS
    assert layer_buffer_keys(arch) == ("mlp.gate.e_score_correction_bias",)
    assert source_identity_includes_config(arch) is True
    assert fp8_block_size(config.quantization_config, arch=arch) == (128, 128)


@pytest.mark.parametrize("q_lora_rank", [1536, None], ids=["q_lora", "q_proj"])
def test_lora_targets_are_the_ones_the_kimi_k2_row_gives(tmp_path, q_lora_rank):
    """#2071's ``kimi_k2`` rows and the ``deepseek_v3`` rows agree, so the rewrite does not
    change what ``target_modules: auto`` adapts."""
    pytest.importorskip("transformers")
    from soup_cli.utils.kimi_k2 import load_stream_model_config
    from soup_cli.utils.peft_wiring import resolve_lora_target_modules

    raw = _base(q_lora_rank=q_lora_rank)
    config = load_stream_model_config(_write_config(tmp_path / "k2", raw))
    as_kimi = SimpleNamespace(model_type="kimi_k2", q_lora_rank=q_lora_rank)
    got = resolve_lora_target_modules(config, "auto")
    assert tuple(got) == tuple(resolve_lora_target_modules(as_kimi, "auto"))
    assert ("q_proj" in got) is (q_lora_rank is None)


# ==========================================================================
# G1b: the guard
# ==========================================================================
_THINKING_QUANT = K2_THINKING_CHANGES["quantization_config"]

#: (id, changes to the Base dict or a callable building the dict, field named, value shown)
_REFUSED = [
    ("architectures", {"architectures": ["DeepseekV3ForSequenceClassification"]},
     "architectures", "['DeepseekV3ForSequenceClassification']"),
    ("architectures-absent", lambda: _without("architectures"), "architectures", "absent"),
    ("topk_method", {"topk_method": "greedy"}, "topk_method", "'greedy'"),
    ("topk_method-absent", lambda: _without("topk_method"), "topk_method", "absent"),
    ("scoring_func", {"scoring_func": "softmax"}, "scoring_func", "'softmax'"),
    ("scoring_func-absent", lambda: _without("scoring_func"), "scoring_func", "absent"),
    ("n_group", {"n_group": 8}, "n_group", "8"),
    ("n_group-absent", lambda: _without("n_group"), "n_group", "absent"),
    ("topk_group", {"topk_group": 4}, "topk_group", "4"),
    ("num_experts_per_tok", {"num_experts_per_tok": 1}, "num_experts_per_tok", "1"),
    ("num_experts_per_tok-float", {"num_experts_per_tok": 8.0}, "num_experts_per_tok", "8.0"),
    ("num_experts_per_tok-bool", {"num_experts_per_tok": True, "norm_topk_prob": False},
     "num_experts_per_tok", "True"),
    ("moe_layer_freq", {"moe_layer_freq": 2}, "moe_layer_freq", "2"),
    ("ep_size", {"ep_size": 8}, "ep_size", "8"),
    ("ep_size-text", {"ep_size": "1"}, "ep_size", "'1'"),
    ("n_routed_experts", {"n_routed_experts": None}, "n_routed_experts", "None"),
    ("n_shared_experts", {"n_shared_experts": None}, "n_shared_experts", "None"),
    ("num_key_value_heads", {"num_key_value_heads": 8}, "num_key_value_heads", "8"),
    ("rope_interleave", {"rope_interleave": False}, "rope_interleave", "False"),
    ("rope-not-a-dict", {"rope_scaling": "yarn"}, "rope_scaling", "'yarn'"),
    ("rope-type", _rope(type="linear"), "rope_scaling.type", "'linear'"),
    ("rope-type-absent", _rope_without("type"), "rope_scaling.type", "absent"),
    ("rope-rope_type", _rope(rope_type="dynamic"), "rope_scaling.rope_type", "'dynamic'"),
    ("rope-truncate", _rope(truncate=False), "rope_scaling['truncate']", "False"),
    ("rope-attention_factor", _rope(attention_factor=1.5),
     "rope_scaling['attention_factor']", "1.5"),
    ("rope-factor-absent", _rope_without("factor"), "rope_scaling.factor", "absent"),
    ("rope-factor-text", _rope(factor="32"), "rope_scaling.factor", "'32'"),
    ("rope-original-absent", _rope_without("original_max_position_embeddings"),
     "rope_scaling.original_max_position_embeddings", "absent"),
    ("rope-beta_fast-zero", _rope(beta_fast=0), "rope_scaling.beta_fast", "0"),
    ("rope-beta_slow-zero", _rope(beta_slow=0.0), "rope_scaling.beta_slow", "0.0"),
    ("rope-mscale-absent", _rope_without("mscale"), "rope_scaling.mscale", "absent"),
    ("rope-mscale-zero", _rope(mscale=0), "rope_scaling.mscale", "0"),
    ("rope-mscale-alone", _rope(mscale=0.707, mscale_all_dim=0), "rope_scaling.mscale", "0.707"),
    ("quant-compressed-tensors", {"quantization_config": _THINKING_QUANT},
     "quantization_config.quant_method", "'compressed-tensors'"),
    ("quant-gptq", {"quantization_config": {"quant_method": "gptq", "bits": 4}},
     "quantization_config.quant_method", "'gptq'"),
    ("quant-not-a-dict", {"quantization_config": "fp8"}, "quantization_config", "'fp8'"),
    ("quant-fp8-per-tensor", {"quantization_config": {"quant_method": "fp8"}},
     "quantization_config", "per-tensor FP8 scales"),
    ("quant-fp8-e5m2", {"quantization_config": dict(K2_BASE_CONFIG["quantization_config"],
                                                    fmt="e5m2")},
     "quantization_config", "e5m2"),
    # Added in the review round (benchmarks/gate-kimi-k2-admission.md, section 7): the guard is an
    # allowlist of the published top-level keys, and num_key_value_heads is required.
    ("num_key_value_heads-absent", lambda: _without("num_key_value_heads"),
     "num_key_value_heads", "absent"),
    ("num_key_value_heads-divides", {"num_key_value_heads": 32}, "num_key_value_heads", "32"),
    ("rope_parameters", {"rope_parameters": {"rope_type": "yarn", "factor": 32.0,
                                             "attention_factor": 1.5, "truncate": False}},
     "rope_parameters", "'attention_factor': 1.5"),
    ("partial_rotary_factor", {"partial_rotary_factor": 0.5}, "partial_rotary_factor", "0.5"),
    ("head_dim", {"head_dim": 128}, "head_dim", "128"),
    ("rope-partial_rotary_factor", _rope(partial_rotary_factor=0.5),
     "rope_scaling['partial_rotary_factor']", "0.5"),
    ("rope-empty", {"rope_scaling": {}}, "rope_scaling", "{}"),
    ("ep_size-1", {"ep_size": 1}, "ep_size", "1"),
    ("rope_interleave-true", {"rope_interleave": True}, "rope_interleave", "True"),
    ("pad_token_id", {"pad_token_id": 163839}, "pad_token_id", "163839"),
    ("sliding_window", {"sliding_window": 4096}, "sliding_window", "4096"),
    # Round-2 review: a null value of a key outside the allowlist is refused like any other.
    ("head_dim-null", {"head_dim": None}, "head_dim", "None"),
    # Second review (LOW 3): a published key may not be left out, or the native class would
    # compute with its own default (rope_theta 10000.0, routed_scaling_factor 2.5, ...). The
    # first two were admitted before; every published key is covered by the test below.
    ("moe_layer_freq-absent", lambda: _without("moe_layer_freq"), "moe_layer_freq", "absent"),
    ("rope_scaling-absent", lambda: _without("rope_scaling"), "rope_scaling", "absent"),
    ("rope_theta-absent", lambda: _without("rope_theta"), "rope_theta", "absent"),
    ("routed_scaling_factor-absent", lambda: _without("routed_scaling_factor"),
     "routed_scaling_factor", "absent"),
    ("rms_norm_eps-absent", lambda: _without("rms_norm_eps"), "rms_norm_eps", "absent"),
    ("norm_topk_prob-absent", lambda: _without("norm_topk_prob"), "norm_topk_prob", "absent"),
    ("torch_dtype-absent", lambda: _without("torch_dtype"), "torch_dtype", "absent"),
]


def _refused_raw(changes):
    return changes() if callable(changes) else _base(**changes)


@pytest.mark.parametrize(
    ("changes", "field", "shown"), [case[1:] for case in _REFUSED], ids=[c[0] for c in _REFUSED]
)
def test_each_guard_failure_is_refused_naming_the_field_and_its_value(changes, field, shown):
    from soup_cli.utils.kimi_k2 import (
        UnsupportedKimiK2Error,
        kimi_k2_guard_failures,
        kimi_k2_native_config_dict,
    )

    raw = _refused_raw(changes)
    failures = kimi_k2_guard_failures(raw)
    assert [failure.field for failure in failures] == [field]
    with pytest.raises(UnsupportedKimiK2Error) as caught:
        kimi_k2_native_config_dict(raw)
    message = str(caught.value)
    assert f"{field}=" in message
    assert shown in message
    assert "kimi_k2" in message
    assert "out of scope for now" in message


#: Published keys a K2 config may leave out: the rewrite drops ``auto_map``, and a copy
#: dequantised to bf16 has no ``quantization_config``.
_OPTIONAL_PUBLISHED = {"auto_map", "quantization_config"}


@pytest.mark.parametrize("key", sorted(set(K2_BASE_CONFIG) - _OPTIONAL_PUBLISHED))
def test_every_published_key_but_auto_map_and_quantization_config_is_required(key):
    """Second review (LOW 3): a K2 file without a published key was admitted when no row read
    that key, and the native class then computed with its own default for it."""
    from soup_cli.utils.kimi_k2 import UnsupportedKimiK2Error, kimi_k2_native_config_dict

    with pytest.raises(UnsupportedKimiK2Error) as caught:
        kimi_k2_native_config_dict(_without(key))
    assert f"{key}=absent" in str(caught.value)


@pytest.mark.parametrize("key", sorted(_OPTIONAL_PUBLISHED))
def test_auto_map_and_quantization_config_may_be_absent(key):
    from soup_cli.utils.kimi_k2 import kimi_k2_guard_failures

    assert kimi_k2_guard_failures(_without(key)) == ()


def test_top_1_without_norm_topk_prob_is_refused_on_both_rows():
    """Absent, ``norm_topk_prob`` reads as renormalising, which K2 skips at top-1; the
    presence row names the missing key as well."""
    from soup_cli.utils.kimi_k2 import kimi_k2_guard_failures

    raw = _without("norm_topk_prob", raw=_base(num_experts_per_tok=1))
    failures = kimi_k2_guard_failures(raw)
    assert [failure.field for failure in failures] == ["num_experts_per_tok", "norm_topk_prob"]
    assert [failure.value for failure in failures] == ["1", "absent"]


def test_a_key_its_own_row_refuses_as_absent_is_named_once():
    from soup_cli.utils.kimi_k2 import kimi_k2_guard_failures

    failures = kimi_k2_guard_failures(_without("architectures", "n_group"))
    assert [failure.field for failure in failures] == ["architectures", "n_group"]


def test_the_refusal_is_a_value_error_raised_by_the_loader_before_any_model(tmp_path):
    pytest.importorskip("transformers")
    from soup_cli.utils.kimi_k2 import UnsupportedKimiK2Error, load_stream_model_config

    folder = _write_config(tmp_path / "k2", _base(n_group=8))
    with pytest.raises(UnsupportedKimiK2Error, match=r"n_group=8") as caught:
        load_stream_model_config(folder)
    assert isinstance(caught.value, ValueError)


def test_thinking_is_refused_and_quantization_config_is_the_first_failing_field(tmp_path):
    pytest.importorskip("transformers")
    from soup_cli.utils.kimi_k2 import (
        UnsupportedKimiK2Error,
        kimi_k2_guard_failures,
        load_stream_model_config,
    )

    raw = _base(**K2_THINKING_CHANGES)
    # The registered rows first; the review round's allowlist then names the two fields of
    # Thinking's file that the published Base / Instruct file does not have.
    assert [failure.field for failure in kimi_k2_guard_failures(raw)] == [
        "quantization_config.quant_method", "ep_size", "pad_token_id"
    ]
    with pytest.raises(UnsupportedKimiK2Error) as caught:
        load_stream_model_config(_write_config(tmp_path / "thinking", raw))
    assert "quantization_config.quant_method='compressed-tensors'" in str(caught.value)


_ADMITTED = [
    ("published", {}),
    ("rope_scaling-null", {"rope_scaling": None}),
    ("rope-rope_type-yarn", _rope(rope_type="yarn")),
    ("rope-mscale-pair-absent", _rope_without("mscale", "mscale_all_dim")),
    ("rope-mscale-1-alone", _rope(mscale=1.0, mscale_all_dim=0)),
    ("rope-mscale-pair-other", _rope(mscale=0.707, mscale_all_dim=0.707)),
    ("rope-betas-absent", _rope_without("beta_fast", "beta_slow")),
    ("rope-factor-int", _rope(factor=32)),
    ("quantization_config-absent", lambda: _without("quantization_config")),
    ("quantization_config-null", {"quantization_config": None}),
    ("fp8-scale_fmt-float", {"quantization_config": dict(K2_BASE_CONFIG["quantization_config"],
                                                         scale_fmt="float")}),
    ("fp8-no-fmt", lambda: _base(quantization_config={
        "activation_scheme": "dynamic", "quant_method": "fp8", "weight_block_size": [128, 128]})),
    ("top-1-without-norm", {"num_experts_per_tok": 1, "norm_topk_prob": False}),
    ("n_shared_experts-0", {"n_shared_experts": 0}),
    ("metadata", {"_commit_hash": "ce72df012259dcc55d945e890f815fe7ef69159c",
                  "_name_or_path": "moonshotai/Kimi-K2-Base", "dtype": "bfloat16"}),
    # transformers 5 spells the published torch_dtype as dtype: either satisfies the key.
    ("torch_dtype-as-dtype", lambda: _without("torch_dtype", raw=_base(dtype="bfloat16"))),
]


def test_thinking_without_its_quantization_is_still_refused_by_the_allowlist():
    """Thinking's file carries fields the published Base / Instruct file does not."""
    from soup_cli.utils.kimi_k2 import kimi_k2_guard_failures

    raw = _without("quantization_config", raw=_base(**K2_THINKING_CHANGES))
    failures = kimi_k2_guard_failures(raw)
    assert [failure.field for failure in failures] == ["ep_size", "pad_token_id"]
    assert all("unexpected field" in failure.reason for failure in failures)
    assert all("training.stream_layers" in failure.reason for failure in failures)


def test_kimi_k2_instruct_0905_is_refused_as_out_of_scope_before_its_field_list(tmp_path):
    """Second review (LOW 4): the 0905 file was refused with a long list of serialised defaults
    and "and N more", never saying that the release is not admitted. Every refusal now says so,
    before the list; the fields it names are exactly the keys the published file lacks."""
    pytest.importorskip("transformers")
    from soup_cli.utils.kimi_k2 import (
        _METADATA_KEYS,
        UnsupportedKimiK2Error,
        kimi_k2_guard_failures,
        load_stream_model_config,
    )

    raw = _base(**K2_INSTRUCT_0905_CHANGES)
    extra = [key for key in raw if key not in K2_BASE_CONFIG and key not in _METADATA_KEYS]
    assert [failure.field for failure in kimi_k2_guard_failures(raw)] == extra
    with pytest.raises(UnsupportedKimiK2Error) as caught:
        load_stream_model_config(_write_config(tmp_path / "0905", raw))
    message = str(caught.value)
    scope = message.index("Other Kimi K2 releases (Kimi-K2-Instruct-0905, Kimi-K2-Thinking)")
    assert "out of scope for now" in message[scope:message.index("This config differs:")]
    assert message.index("This config differs:") > scope
    assert f"and {len(extra) - 10} more" in message


def test_the_allowlist_is_exactly_the_published_keys_plus_named_metadata():
    from soup_cli.utils.kimi_k2 import _METADATA_KEYS, _PUBLISHED_KEYS

    assert _PUBLISHED_KEYS == set(K2_BASE_CONFIG)
    assert _METADATA_KEYS == {"_commit_hash", "_name_or_path", "dtype"}


_BASE_REVISION = "ce72df012259dcc55d945e890f815fe7ef69159c"


@pytest.mark.parametrize("layout", ["hub-snapshot", "plain-folder"])
def test_the_loader_admits_the_metadata_transformers_reader_adds(tmp_path, layout):
    """transformers' reader adds ``_commit_hash`` to the dict on 5.17 (None for a plain folder,
    the commit for a Hub cache's ``snapshots/<commit>/`` folder) and not at all on 5.19 (CI runs
    39626025729, 39758361029), where the config has no ``_commit_hash`` attribute either. So
    presence is asserted for neither: whatever the reader records, the loader admits and keeps."""
    pytest.importorskip("transformers")
    from transformers import PretrainedConfig

    from soup_cli.utils.kimi_k2 import _METADATA_KEYS, load_stream_model_config

    if layout == "hub-snapshot":
        directory = tmp_path / "models--moonshotai--Kimi-K2-Base" / "snapshots" / _BASE_REVISION
    else:
        directory = tmp_path / "k2"
    folder = _write_config(directory, K2_BASE_CONFIG)
    read = PretrainedConfig.get_config_dict(folder)[0]
    assert set(read) - set(K2_BASE_CONFIG) <= _METADATA_KEYS
    config = load_stream_model_config(folder)
    assert config.model_type == "deepseek_v3"
    # Whatever commit the reader recorded survives the rewrite (the snapshot's, on 5.17); on
    # 5.19 neither the dict nor the config carries one.
    assert getattr(config, "_commit_hash", None) == read.get("_commit_hash")


@pytest.mark.parametrize("changes", [case[1] for case in _ADMITTED], ids=[c[0] for c in _ADMITTED])
def test_admitted_variants_pass_the_guard(changes):
    from soup_cli.utils.kimi_k2 import kimi_k2_guard_failures, kimi_k2_native_config_dict

    raw = _refused_raw(changes)
    assert kimi_k2_guard_failures(raw) == ()
    assert kimi_k2_native_config_dict(raw)["model_type"] == "deepseek_v3"


def test_every_failure_is_listed_in_guard_order():
    from soup_cli.utils.kimi_k2 import UnsupportedKimiK2Error, kimi_k2_native_config_dict

    raw = _base(
        quantization_config=_THINKING_QUANT, n_group=8, architectures=["X"], **_rope(truncate=1)
    )
    with pytest.raises(UnsupportedKimiK2Error) as caught:
        kimi_k2_native_config_dict(raw)
    message = str(caught.value)
    positions = [
        message.index(name)
        for name in ("architectures=", "n_group=", "rope_scaling['truncate']=",
                     "quantization_config.quant_method=")
    ]
    assert positions == sorted(positions)


def test_a_refusal_never_quotes_raw_control_bytes_and_stays_short():
    from soup_cli.utils.kimi_k2 import UnsupportedKimiK2Error, kimi_k2_native_config_dict

    hostile_key = "\x1b[2Jkey\u202e"
    hostile = "\x1b[31mred\x1b[0m\u202e" + "x" * 5000
    raw = _base(topk_method=hostile, **_rope(**{"\x1b]0;title\x07": 1}))
    raw[hostile_key] = hostile
    with pytest.raises(UnsupportedKimiK2Error) as caught:
        kimi_k2_native_config_dict(raw)
    message = str(caught.value)
    assert "\x1b" not in message and "\u202e" not in message and "\x07" not in message
    assert len(message) < 2500


def test_non_ascii_key_names_and_values_are_shown_escaped():
    """Round-2 review: a homoglyph key (Cyrillic U+0435 in "head_dim") or value (U+043E in
    "noaux_tc") must not reach the message raw, where it would read as the ASCII name."""
    from soup_cli.utils.kimi_k2 import UnsupportedKimiK2Error, kimi_k2_native_config_dict

    key = "h" + chr(0x435) + "ad_dim"
    value = "n" + chr(0x43E) + "aux_tc"
    raw = _base(topk_method=value)
    raw[key] = 128
    with pytest.raises(UnsupportedKimiK2Error) as caught:
        kimi_k2_native_config_dict(raw)
    message = str(caught.value)
    assert message.isascii()
    assert "u0435" in message and "u043e" in message


def test_a_refusal_stays_bounded_whatever_the_key_names_and_their_number():
    """Round-2 review: a 200,000-character identifier key gave a 400 KB message."""
    from soup_cli.utils.kimi_k2 import (
        UnsupportedKimiK2Error,
        kimi_k2_guard_failures,
        kimi_k2_native_config_dict,
    )

    raw = _base()
    raw["k" * 200_000] = 1
    for idx in range(99):
        raw[f"extra_field_{idx}"] = idx
    assert len(kimi_k2_guard_failures(raw)) == 100
    with pytest.raises(UnsupportedKimiK2Error) as caught:
        kimi_k2_native_config_dict(raw)
    message = str(caught.value)
    assert len(message) < 3000
    assert "and 90 more" in message
    assert "k" * 81 not in message


def test_the_guard_reads_a_mapping_only():
    from soup_cli.utils.kimi_k2 import UnsupportedKimiK2Error, kimi_k2_native_config_dict

    with pytest.raises(UnsupportedKimiK2Error, match="not a JSON object"):
        kimi_k2_native_config_dict(["kimi_k2"])


# ==========================================================================
# G1c: the numeric guard rows, against a hand transcription of K2's code
# ==========================================================================
def _k2_gate(hidden, weight, bias, *, top_k, n_group, topk_group, norm_topk_prob, scaling):
    """K2's ``MoEGate.forward`` for ``noaux_tc`` + ``sigmoid``, transcribed from
    ``modeling_deepseek.py`` (sha256 ce753fc9...) read as text."""
    import torch
    from torch.nn.functional import linear

    bsz, seq_len, h = hidden.shape
    hidden = hidden.view(-1, h)
    logits = linear(hidden.type(torch.float32), weight.type(torch.float32), None)
    scores = logits.sigmoid()
    n_routed = weight.shape[0]
    scores_for_choice = scores.view(bsz * seq_len, -1) + bias.unsqueeze(0)
    group_scores = (
        scores_for_choice.view(bsz * seq_len, n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(bsz * seq_len, n_group, n_routed // n_group)
        .reshape(bsz * seq_len, -1)
    )
    tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
    _, topk_idx = torch.topk(tmp_scores, k=top_k, dim=-1, sorted=False)
    topk_weight = scores.gather(1, topk_idx)
    if top_k > 1 and norm_topk_prob:
        denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
        topk_weight = topk_weight / denominator
    return topk_idx, topk_weight * scaling


def _router_pair(*, n_group=1, topk_group=1, top_k=8, experts=384, bias_shift=0.0):
    """(K2 transcription, native router) on the same random weights."""
    import torch
    from transformers import DeepseekV3Config
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3TopkRouter

    hidden_size = 32
    config = DeepseekV3Config(
        hidden_size=hidden_size, n_routed_experts=experts, num_experts_per_tok=top_k,
        n_group=n_group, topk_group=topk_group, norm_topk_prob=True, routed_scaling_factor=2.827,
    )
    generator = torch.Generator().manual_seed(5)
    weight = torch.randn(experts, hidden_size, generator=generator)
    bias = torch.randn(experts, generator=generator) * 0.1 + bias_shift
    hidden = torch.randn(2, 9, hidden_size, generator=generator)
    router = DeepseekV3TopkRouter(config)
    with torch.no_grad():
        router.weight.copy_(weight)
        router.e_score_correction_bias.copy_(bias)
        _logits, native_weight, native_idx = router(hidden)
    k2_idx, k2_weight = _k2_gate(
        hidden, weight, bias, top_k=top_k, n_group=n_group, topk_group=topk_group,
        norm_topk_prob=True, scaling=2.827,
    )

    def by_expert(idx, weights):
        order = idx.argsort(dim=-1)
        return idx.gather(1, order), weights.gather(1, order)

    return by_expert(k2_idx, k2_weight), by_expert(native_idx, native_weight)


def test_the_native_router_equals_k2s_at_k2s_group_and_top_k_values():
    import torch

    pytest.importorskip("transformers")
    (k2_idx, k2_weight), (native_idx, native_weight) = _router_pair()
    assert torch.equal(k2_idx, native_idx)
    assert torch.equal(k2_weight, native_weight)


def test_the_routers_differ_once_a_group_is_masked_so_n_group_is_guarded():
    """K2 fills masked experts with 0.0: an all-negative biased score loses to them."""
    import torch

    pytest.importorskip("transformers")
    (k2_idx, _), (native_idx, _) = _router_pair(
        n_group=2, topk_group=1, top_k=2, experts=8, bias_shift=-5.0
    )
    assert not torch.equal(k2_idx, native_idx)


def test_the_routers_differ_at_top_1_so_num_experts_per_tok_is_guarded():
    import torch

    pytest.importorskip("transformers")
    (k2_idx, k2_weight), (native_idx, native_weight) = _router_pair(top_k=1, experts=8)
    assert torch.equal(k2_idx, native_idx)
    assert not torch.equal(k2_weight, native_weight)


def _k2_mscale(scale, mscale=1):
    return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0


def _k2_softmax_scale(q_head_dim, rope_scaling):
    """``DeepseekV3Attention.softmax_scale`` in K2's code, in the same operation order."""
    scale = q_head_dim ** (-0.5)
    if rope_scaling is not None:
        mscale_all_dim = rope_scaling.get("mscale_all_dim", 0)
        if mscale_all_dim:
            mscale = _k2_mscale(rope_scaling["factor"], mscale_all_dim)
            scale = scale * mscale * mscale
    return scale


def _k2_rope(dim, base, rope_scaling, seq_len):
    """K2's rotary cos/sin cache (``DeepseekV3RotaryEmbedding`` or its yarn subclass),
    transcribed from ``modeling_deepseek.py``."""
    import torch

    if rope_scaling is None:
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(seq_len, dtype=inv_freq.dtype)
        emb = torch.cat((torch.outer(t, inv_freq),) * 2, dim=-1)
        return emb.cos(), emb.sin()
    factor = rope_scaling["factor"]
    original = rope_scaling.get("original_max_position_embeddings", 4096)
    beta_fast = rope_scaling.get("beta_fast", 32)
    beta_slow = rope_scaling.get("beta_slow", 1)
    mscale = rope_scaling.get("mscale", 1)
    mscale_all_dim = rope_scaling.get("mscale_all_dim", 0)

    def correction_dim(rotations):
        return (dim * math.log(original / (rotations * 2 * math.pi))) / (2 * math.log(base))

    low = max(math.floor(correction_dim(beta_fast)), 0)
    high = min(math.ceil(correction_dim(beta_slow)), dim - 1)
    freq_extra = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    freq_inter = 1.0 / (factor * base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    top = high + 0.001 if low == high else high
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (top - low), 0, 1)
    inv_freq_mask = 1.0 - ramp
    inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
    t = torch.arange(seq_len, dtype=torch.float32)
    emb = torch.cat((torch.outer(t, inv_freq),) * 2, dim=-1)
    attention = float(_k2_mscale(factor, mscale) / _k2_mscale(factor, mscale_all_dim))
    return emb.cos() * attention, emb.sin() * attention


def _k2_reference(raw, seq_len):
    """(cos, sin, softmax scale) K2's code computes for a config dict, by transcription."""
    rope_scaling = raw.get("rope_scaling")
    cos, sin = _k2_rope(raw["qk_rope_head_dim"], raw["rope_theta"], rope_scaling, seq_len)
    q_head_dim = raw["qk_nope_head_dim"] + raw["qk_rope_head_dim"]
    return cos, sin, _k2_softmax_scale(q_head_dim, rope_scaling)


def _native_rope(raw, seq_len):
    """(cos, sin, attention scaling) of the native class for a K2 dict, guard bypassed."""
    import torch
    from transformers import DeepseekV3Config
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
        DeepseekV3Attention,
        DeepseekV3RotaryEmbedding,
    )

    config = DeepseekV3Config.from_dict(_native_dict(raw))
    rotary = DeepseekV3RotaryEmbedding(config)
    positions = torch.arange(seq_len)[None, :]
    cos, sin = rotary(torch.zeros(1, dtype=torch.float32), positions)
    with torch.device("meta"):
        attention = DeepseekV3Attention(config, layer_idx=0)
    return cos[0], sin[0], attention.scaling


#: The investigation's claim was bit-identity at 4096 positions.
_ROPE_SEQ = 4096


@pytest.mark.parametrize(
    "changes",
    [{}, {"rope_scaling": None}, _rope(mscale=1.0, mscale_all_dim=0),
     _rope_without("mscale", "mscale_all_dim"), _rope(mscale=0.707, mscale_all_dim=0.707),
     _rope_without("beta_fast", "beta_slow"), _rope(factor=8.0)],
    ids=["published", "no-scaling", "mscale-1-alone", "no-mscale", "mscale-pair",
         "default-betas", "factor-8"],
)
def test_the_native_rope_equals_k2s_on_admitted_values(changes):
    import torch

    pytest.importorskip("transformers")
    raw = _base(**changes)
    k2_cos, k2_sin, k2_scale = _k2_reference(raw, _ROPE_SEQ)
    cos, sin, scale = _native_rope(raw, _ROPE_SEQ)
    assert torch.equal(cos, k2_cos)
    assert torch.equal(sin, k2_sin)
    assert scale == k2_scale


@pytest.mark.parametrize(
    "changes",
    [_rope_without("original_max_position_embeddings"), _rope_without("mscale"),
     _rope(mscale=0.707, mscale_all_dim=0), _rope(attention_factor=1.5), _rope(beta_fast=0)],
    ids=["no-original", "mscale_all_dim-alone", "mscale-alone", "attention_factor", "beta-0"],
)
def test_the_ropes_differ_on_each_refused_rope_variant(changes):
    import torch

    pytest.importorskip("transformers")
    raw = _base(**changes)
    cos, sin, scale = _native_rope(raw, _ROPE_SEQ)
    try:
        k2_cos, k2_sin, k2_scale = _k2_reference(raw, _ROPE_SEQ)
    except ZeroDivisionError:
        assert changes == _rope(beta_fast=0)
        return  # K2's own code cannot compute it; the native class silently would
    assert not (torch.equal(cos, k2_cos) and torch.equal(sin, k2_sin) and scale == k2_scale)


# ==========================================================================
# G2: streamed vs resident parity on a tiny K2-like checkpoint (CPU fp32)
# ==========================================================================
_TINY = dict(
    vocab_size=128, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
    num_hidden_layers=3, first_k_dense_replace=1, num_attention_heads=4,
    num_key_value_heads=4, n_shared_experts=1, n_routed_experts=8, n_group=1, topk_group=1,
    num_experts_per_tok=2, kv_lora_rank=16, q_lora_rank=24, qk_nope_head_dim=16,
    qk_rope_head_dim=8, v_head_dim=16, max_position_embeddings=2048, bos_token_id=0,
    eos_token_id=1,
)
_TINY_YARN = {
    "beta_fast": 1.0, "beta_slow": 1.0, "factor": 32.0, "mscale": 1.0, "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 64,
}
_TINY_LAYERS = 3
_BLOCK = (16, 32)
#: The real checkpoint carries 61 of these (BF16 [56]); neither K2's code (non-persistent
#: buffer) nor the native class (no per-layer rotary module) reads them.
_STALE_INV_FREQ = "self_attn.rotary_emb.inv_freq"
_STALE_VALUE = 12345.0


def _tiny_k2_raw():
    """The real Base ``config.json`` with tiny dimensions substituted."""
    raw = _without("quantization_config", raw=_base(**_TINY))
    raw["rope_scaling"] = dict(_TINY_YARN, type="yarn")
    return raw


def _native_tiny_config(*, yarn=True):
    """The resident control's config, from keyword arguments: never through Soup's loader."""
    from transformers import DeepseekV3Config

    rope = {"rope_type": "yarn", "rope_theta": 50000.0, **_TINY_YARN}
    if not yarn:
        rope = {"rope_type": "default", "rope_theta": 50000.0}
    return DeepseekV3Config(
        **_TINY, rope_parameters=rope, rms_norm_eps=1e-6, routed_scaling_factor=2.827,
        norm_topk_prob=True, tie_word_embeddings=False, num_mtp_layers=0,
    )


def _fp8_tools():
    from tests.test_fp8_block_streaming import _harness, _reference_dequant

    return _harness(), _reference_dequant


@pytest.fixture(scope="module")
def k2_checkpoints(tmp_path_factory):
    """{"bf16": (K2 folder, resident folder), "fp8": (...)}: the same seeded weights."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from safetensors.torch import load_file, save_file
    from transformers import DeepseekV3ForCausalLM

    root = tmp_path_factory.mktemp("kimi-k2-tiny")
    torch.manual_seed(11)
    model = DeepseekV3ForCausalLM(_native_tiny_config()).to(torch.float32).eval()
    generator = torch.Generator().manual_seed(23)
    with torch.no_grad():
        for layer in model.model.layers[1:]:
            bias = layer.mlp.gate.e_score_correction_bias
            bias.copy_(torch.randn(bias.shape, generator=generator) * 2.0)
        for parameter in model.parameters():
            parameter.data = parameter.data.to(torch.bfloat16)
    native = root / "native-bf16"
    model.save_pretrained(str(native))
    state = load_file(str(native / "model.safetensors"))
    # The real index's per-expert spelling, which the sharder stacks.
    assert "model.layers.1.mlp.experts.0.gate_proj.weight" in state
    stale = {
        f"model.layers.{idx}.{_STALE_INV_FREQ}": torch.full((7,), _STALE_VALUE).to(torch.bfloat16)
        for idx in range(_TINY_LAYERS)
    }
    k2 = root / "k2-bf16"
    k2.mkdir()
    save_file({**state, **stale}, str(k2 / "model.safetensors"), metadata={"format": "pt"})
    (k2 / "config.json").write_text(json.dumps(_tiny_k2_raw()), encoding="utf-8")

    harness, reference_dequant = _fp8_tools()
    fp8 = root / "k2-fp8"
    harness.convert_checkpoint(str(k2), str(fp8), block=_BLOCK, log=lambda _line: None)
    fp8_state = load_file(str(fp8 / "model.safetensors"))
    assert any(value.dtype == torch.float8_e4m3fn for value in fp8_state.values())
    ref_state = {}
    for key, value in fp8_state.items():
        if key.endswith("_scale_inv") or key.endswith(_STALE_INV_FREQ):
            continue
        scale = fp8_state.get(key + "_scale_inv")
        ref_state[key] = value if scale is None else reference_dequant(value, scale, _BLOCK)
    ref = root / "native-fp8-ref"
    ref.mkdir()
    save_file(ref_state, str(ref / "model.safetensors"), metadata={"format": "pt"})
    shutil.copyfile(native / "config.json", ref / "config.json")
    assert not any(name.endswith(".py") for folder in (k2, fp8) for name in os.listdir(folder))
    return {"bf16": (str(k2), str(native)), "fp8": (str(fp8), str(ref))}


def _resident(weights, layout="", config=None):
    import torch
    from transformers import DeepseekV3ForCausalLM

    extra = {} if config is None else {"config": config}
    model, info = DeepseekV3ForCausalLM.from_pretrained(
        weights, dtype=torch.float32, output_loading_info=True, **extra
    )
    assert not (info["missing_keys"] or info["unexpected_keys"] or info["mismatched_keys"]), info
    if layout == "per_expert":
        # The streamed experts follow the eager loop; the resident control must too.
        model.set_experts_implementation("eager")
    return model.eval()


def _lora_config(model_config):
    from peft import LoraConfig, TaskType

    from soup_cli.utils.peft_wiring import resolve_lora_target_modules

    return LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
        target_modules=list(resolve_lora_target_modules(model_config, "auto")),
        task_type=TaskType.CAUSAL_LM,
    )


def _streamed(tmp_path, weights, layout):
    from soup_cli.utils.kimi_k2 import load_stream_model_config
    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream import stream_arch_of
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    config = load_stream_model_config(weights)
    arch = stream_arch_of(config)
    shards = str(tmp_path / "shards")
    index = shard_checkpoint(
        weights, shards, dtype="float32", arch=arch,
        num_hidden_layers=config.num_hidden_layers, expert_layout=layout,
    )
    model, runtime = build_streamed_model(
        model_id=weights, shard_dir=shards, index=index, lora_config=_lora_config(config),
        device="cpu", dtype="float32", buffers=2, pin=False, seed=3,
    )
    return model, runtime, config, shards, index


def _paired(tmp_path, k2_checkpoints, source, layout):
    import torch
    from peft import get_peft_model

    weights, resident_weights = k2_checkpoints[source]
    model, runtime, config, shards, index = _streamed(tmp_path, weights, layout)
    resident = get_peft_model(_resident(resident_weights, layout), _lora_config(config))
    generator = torch.Generator().manual_seed(17)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.02)
    src = {k.replace(".inner.", "."): v for k, v in model.state_dict().items() if "lora_" in k}
    dst = {k: v for k, v in resident.state_dict().items() if "lora_" in k}
    assert src and set(src) == set(dst)
    for key, value in src.items():
        dst[key].copy_(value)
    model.eval()
    resident.eval()
    ids = torch.randint(0, 128, (2, 12), generator=generator)
    return SimpleNamespace(
        model=model, runtime=runtime, resident=resident, ids=ids, shards=shards, index=index
    )


def _refuse_remote_code(monkeypatch):
    import transformers.dynamic_module_utils as dynamic
    import transformers.models.auto.configuration_auto as configuration_auto

    def _refuse(*_args, **_kwargs):
        raise AssertionError("remote code was loaded")

    monkeypatch.setattr(dynamic, "get_class_from_dynamic_module", _refuse)
    monkeypatch.setattr(configuration_auto, "get_class_from_dynamic_module", _refuse)


_G2_CELLS = [("", "bf16"), ("per_expert", "bf16"), ("", "fp8"), ("per_expert", "fp8")]


@pytest.mark.parametrize(
    ("layout", "source"), _G2_CELLS, ids=[f"{lay or 'fused'}-{src}" for lay, src in _G2_CELLS]
)
def test_the_streamed_k2_matches_the_resident_native_model_bit_exactly(
    tmp_path, k2_checkpoints, monkeypatch, layout, source
):
    import torch

    from soup_cli.utils.layer_stream_runtime import canonical_named_parameters

    _refuse_remote_code(monkeypatch)
    pair = _paired(tmp_path, k2_checkpoints, source, layout)
    try:
        with torch.no_grad():
            got = pair.model(input_ids=pair.ids, labels=pair.ids)
            want = pair.resident(input_ids=pair.ids, labels=pair.ids)
        assert torch.equal(got.logits, want.logits), (got.logits - want.logits).abs().max()
        assert torch.equal(got.loss, want.loss)

        pair.model.zero_grad(set_to_none=True)
        pair.resident.zero_grad(set_to_none=True)
        pair.model(input_ids=pair.ids, labels=pair.ids).loss.backward()
        pair.resident(input_ids=pair.ids, labels=pair.ids).loss.backward()

        def grads(net):
            return {
                name: parameter.grad
                for name, parameter in canonical_named_parameters(net)
                if "lora_" in name and parameter.grad is not None
            }

        streamed, resident = grads(pair.model), grads(pair.resident)
        assert streamed and set(streamed) == set(resident)
        assert any(
            ".layers.0." in name and float(gradient.abs().sum()) > 0
            for name, gradient in streamed.items()
        )
        for name, gradient in streamed.items():
            assert torch.equal(gradient, resident[name]), name
    finally:
        pair.runtime.close()


@pytest.mark.parametrize("layout", ["", "per_expert"], ids=["fused", "per_expert"])
def test_the_stale_rotary_inv_freq_is_never_streamed(tmp_path, k2_checkpoints, layout):
    """The sharder writes the stale key into each layer file as stored (the docs say so), and
    the runtime never streams it: no module of either implementation reads it."""
    from safetensors.torch import load_file

    from soup_cli.utils.layer_shard import layer_paths

    pair = _paired(tmp_path, k2_checkpoints, "bf16", layout)
    try:
        for path in layer_paths(pair.shards, pair.index):
            assert _STALE_INV_FREQ in load_file(path)
        assert all(
            not any("rotary_emb" in key for key in keys)
            for keys in pair.runtime.pool.active_keys_by_layer
        )
        assert not any(
            name.endswith("self_attn.rotary_emb") for name, _ in pair.model.named_modules()
        )
    finally:
        pair.runtime.close()


def test_the_streamed_k2_is_built_from_the_native_config(tmp_path, k2_checkpoints):
    import torch.nn as nn
    from transformers import DeepseekV3Config

    for source in ("bf16", "fp8"):
        pair = _paired(tmp_path / source, k2_checkpoints, source, "")
        try:
            base = pair.model.get_base_model()
            assert type(base.config) is DeepseekV3Config
            assert base.config.model_type == "deepseek_v3"
            assert getattr(base.config, "auto_map", None) is None
            assert getattr(base.config, "quantization_config", None) is None
            attention = base.model.layers[0].inner.self_attn
            assert type(attention.o_proj.base_layer) is nn.Linear
        finally:
            pair.runtime.close()


def test_controls_the_yarn_rope_the_router_bias_and_the_fp8_loss_each_move_the_logits(
    k2_checkpoints,
):
    """Without these the parity above could pass on a model insensitive to them."""
    import torch

    _k2, native = k2_checkpoints["bf16"]
    _fp8, fp8_ref = k2_checkpoints["fp8"]
    ids = torch.randint(0, 128, (2, 12), generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        yarn = _resident(native).forward(input_ids=ids).logits
        default = _resident(native, config=_native_tiny_config(yarn=False)).forward(
            input_ids=ids
        ).logits
        unbiased_model = _resident(native)
        for layer in unbiased_model.model.layers[1:]:
            layer.mlp.gate.e_score_correction_bias.zero_()
        unbiased = unbiased_model.forward(input_ids=ids).logits
        dequantised = _resident(fp8_ref).forward(input_ids=ids).logits
    assert not torch.equal(yarn, default)
    assert not torch.equal(yarn, unbiased)
    assert not torch.equal(yarn, dequantised)


# ==========================================================================
# G3: the disk pre-flight over the real Kimi-K2-Base header layout
# ==========================================================================
#: From the real Base ``model.safetensors.index.json`` (sha256 c1f1d16c...ddcbc):
#: its key count and ``metadata.total_size``.
K2_INDEX_TENSORS = 139_644
K2_INDEX_TOTAL_SIZE = 1_029_173_256_720
#: The stale per-layer ``rotary_emb.inv_freq`` shape: the one for which the header map's byte
#: total equals ``total_size`` exactly (benchmarks/gate-kimi-k2-admission.md, G3).
K2_STALE_INV_FREQ_SHAPE = (56,)
_ITEMSIZE = {"F8_E4M3": 1, "BF16": 2, "F32": 4}


def k2_header_map(raw, *, block=(128, 128)):
    """name -> (file, safetensors dtype, shape) of the published K2 checkpoint, from its config.

    One decoder layer per file (``model-<N+1>-of-61.safetensors``), the embedding in the first,
    the final norm and the head in the last, as the real index lays them out. Every linear
    weight is F8_E4M3 with an F32 ``weight_scale_inv``; norms, embeddings, head, router weight
    and the stale ``inv_freq`` are BF16; the router bias is F32.
    """
    hidden = raw["hidden_size"]
    heads = raw["num_attention_heads"]
    q_head = raw["qk_nope_head_dim"] + raw["qk_rope_head_dim"]
    shared = raw["n_shared_experts"] * raw["moe_intermediate_size"]
    moe, dense = raw["moe_intermediate_size"], raw["intermediate_size"]
    n_layers, experts, vocab = raw["num_hidden_layers"], raw["n_routed_experts"], raw["vocab_size"]
    files = [f"model-{idx + 1}-of-{n_layers}.safetensors" for idx in range(n_layers)]
    out = {
        "model.embed_tokens.weight": (files[0], "BF16", (vocab, hidden)),
        "model.norm.weight": (files[-1], "BF16", (hidden,)),
        "lm_head.weight": (files[-1], "BF16", (vocab, hidden)),
    }

    def linear(prefix, rows, cols, file):
        out[prefix + ".weight"] = (file, "F8_E4M3", (rows, cols))
        out[prefix + ".weight_scale_inv"] = (
            file, "F32", (-(-rows // block[0]), -(-cols // block[1]))
        )

    for idx in range(n_layers):
        file, layer = files[idx], f"model.layers.{idx}."
        for name, size in (("input_layernorm", hidden), ("post_attention_layernorm", hidden),
                           ("self_attn.q_a_layernorm", raw["q_lora_rank"]),
                           ("self_attn.kv_a_layernorm", raw["kv_lora_rank"])):
            out[layer + name + ".weight"] = (file, "BF16", (size,))
        out[layer + _STALE_INV_FREQ] = (file, "BF16", K2_STALE_INV_FREQ_SHAPE)
        linear(layer + "self_attn.q_a_proj", raw["q_lora_rank"], hidden, file)
        linear(layer + "self_attn.q_b_proj", heads * q_head, raw["q_lora_rank"], file)
        linear(layer + "self_attn.kv_a_proj_with_mqa",
               raw["kv_lora_rank"] + raw["qk_rope_head_dim"], hidden, file)
        linear(layer + "self_attn.kv_b_proj",
               heads * (raw["qk_nope_head_dim"] + raw["v_head_dim"]), raw["kv_lora_rank"], file)
        linear(layer + "self_attn.o_proj", hidden, heads * raw["v_head_dim"], file)
        if idx < raw["first_k_dense_replace"]:
            linear(layer + "mlp.gate_proj", dense, hidden, file)
            linear(layer + "mlp.up_proj", dense, hidden, file)
            linear(layer + "mlp.down_proj", hidden, dense, file)
            continue
        out[layer + "mlp.gate.weight"] = (file, "BF16", (experts, hidden))
        out[layer + "mlp.gate.e_score_correction_bias"] = (file, "F32", (experts,))
        linear(layer + "mlp.shared_experts.gate_proj", shared, hidden, file)
        linear(layer + "mlp.shared_experts.up_proj", shared, hidden, file)
        linear(layer + "mlp.shared_experts.down_proj", hidden, shared, file)
        for expert in range(experts):
            linear(layer + f"mlp.experts.{expert}.gate_proj", moe, hidden, file)
            linear(layer + f"mlp.experts.{expert}.up_proj", moe, hidden, file)
            linear(layer + f"mlp.experts.{expert}.down_proj", hidden, moe, file)
    return out


def serve_header_map(monkeypatch, header_map, root="k2"):
    """Replace ``read_header`` so it serves ``header_map``; return the source paths."""
    from soup_cli.utils import safetensors_reader
    from soup_cli.utils.safetensors_reader import _DTYPES, TensorRange

    by_file = {}
    for name, (file, dtype, shape) in header_map.items():
        by_file.setdefault(file, []).append((name, dtype, shape))

    def _read_header(path):
        entries, cursor = {}, 8
        for name, dtype, shape in by_file[os.path.basename(path)]:
            size = math.prod(shape) * _ITEMSIZE[dtype]
            entries[name] = TensorRange(
                name=name, dtype=_DTYPES[dtype], shape=tuple(shape), start=cursor,
                end=cursor + size,
            )
            cursor += size
        return entries

    monkeypatch.setattr(safetensors_reader, "read_header", _read_header)
    return [os.path.join(root, file) for file in sorted(by_file)]


def _nf4_bytes(numel):
    """NF4 under double quantisation, written independently of Soup's helper."""
    blocks = -(-numel // 64)
    return -(-numel // 2) + blocks + 4 * -(-blocks // 256) + 4


def k2_cache_oracle(header_map, suffixes, layout):
    """(per-layer payload, payload outside the layers) of a bf16 NF4 K2 cache, by hand."""
    layer_re = re.compile(r"model\.layers\.(\d+)\.(.+)$")
    expert_re = re.compile(r"mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")
    layers, other = {}, 16 * 4 + 256 * 4  # the two shared NF4 code tables
    experts = {}
    for name, (_file, _dtype, shape) in header_map.items():
        if name.endswith("_scale_inv"):
            continue
        numel = math.prod(shape)
        match = layer_re.match(name)
        if match is None:
            other += 2 * numel
            continue
        idx, short = int(match.group(1)), match.group(2)
        expert = expert_re.match(short)
        if expert is not None:
            pair = experts.setdefault((idx, int(expert.group(1))), [0, 0])
            pair[expert.group(2) == "down"] += numel
            continue
        if short in suffixes:
            stored = _nf4_bytes(numel)
        elif short == "mlp.gate.e_score_correction_bias":
            stored = 4 * numel
        else:
            stored = 2 * numel
        layers[idx] = layers.get(idx, 0) + stored
    for (idx, _expert), (gate_up, down) in experts.items():
        if layout == "per_expert":
            layers[idx] += _nf4_bytes(gate_up) + _nf4_bytes(down)
        else:
            layers[idx] += 2 * (gate_up + down)
    return tuple(layers[idx] for idx in range(len(layers))), other


def test_the_k2_header_map_reproduces_the_real_index_totals():
    header_map = k2_header_map(K2_BASE_CONFIG)
    assert len(header_map) == K2_INDEX_TENSORS
    total = sum(math.prod(shape) * _ITEMSIZE[dtype] for _f, dtype, shape in header_map.values())
    assert total == K2_INDEX_TOTAL_SIZE


#: The published config at small sizes, every other field (and so every module the skeleton
#: builds and names) as published. build_meta_skeleton on the full config makes each MoE layer's
#: fused experts with ``torch.empty`` on the CPU before ``init_empty_weights`` moves them to
#: meta: 22.5 GB in one allocation, which Linux refuses on a 16 GB CI runner (run 39626025729,
#: ubuntu 3.11 and 3.12: "DefaultCPUAllocator: can't allocate memory").
_K2_SIZE_TWIN = {
    "hidden_size": 64, "intermediate_size": 128, "moe_intermediate_size": 32,
    "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 4,
    "n_routed_experts": 8, "kv_lora_rank": 16, "q_lora_rank": 24, "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 8, "v_head_dim": 16,
}
#: What the meta probe of the full-size published config reported (gate G3b, 11 suffixes): the
#: five MLA projections, the dense MLP's three and the shared expert's three.
K2_QUANT_SUFFIXES = frozenset({
    "self_attn.q_a_proj.weight", "self_attn.q_b_proj.weight",
    "self_attn.kv_a_proj_with_mqa.weight", "self_attn.kv_b_proj.weight",
    "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
    "mlp.down_proj.weight", "mlp.shared_experts.gate_proj.weight",
    "mlp.shared_experts.up_proj.weight", "mlp.shared_experts.down_proj.weight",
})


@pytest.fixture(scope="module")
def k2_quant_suffixes(tmp_path_factory):
    """(full-size config folder, the meta probe's NF4 suffixes), the probe run on the size twin:
    module names do not depend on sizes, and the twin's set must equal the full probe's."""
    pytest.importorskip("transformers")
    pytest.importorskip("bitsandbytes")
    from soup_cli.utils.layer_stream_runtime import build_meta_skeleton, quantised_layer_suffixes

    twin = _base(**_K2_SIZE_TWIN)
    assert {key for key in K2_BASE_CONFIG if twin[key] != K2_BASE_CONFIG[key]} == set(
        _K2_SIZE_TWIN
    )
    assert twin["num_hidden_layers"] > twin["first_k_dense_replace"]  # dense and MoE layers
    folder = _write_config(tmp_path_factory.mktemp("k2-full"), K2_BASE_CONFIG)
    probe = build_meta_skeleton(
        _write_config(tmp_path_factory.mktemp("k2-twin"), twin), dtype="bfloat16", quant="nf4"
    )
    suffixes = quantised_layer_suffixes(probe)
    del probe
    assert suffixes == K2_QUANT_SUFFIXES
    return folder, suffixes


@pytest.mark.parametrize("layout", ["per_expert", ""], ids=["per_expert", "fused"])
def test_the_disk_preflight_sizes_the_real_k2_cache_exactly(
    monkeypatch, k2_quant_suffixes, layout
):
    from soup_cli.utils.kimi_k2 import load_stream_model_config
    from soup_cli.utils.layer_shard import estimate_shard_cache_bytes
    from soup_cli.utils.layer_stream import stream_arch_of

    folder, suffixes = k2_quant_suffixes
    assert "self_attn.kv_a_proj_with_mqa.weight" in suffixes
    assert "mlp.gate.weight" not in suffixes
    config = load_stream_model_config(folder)
    header_map = k2_header_map(K2_BASE_CONFIG)
    paths = serve_header_map(monkeypatch, header_map)
    estimate = estimate_shard_cache_bytes(
        paths, dtype="bfloat16", arch=stream_arch_of(config), quant="nf4",
        quant_suffixes=suffixes, double_quant=True, expert_layout=layout,
        num_hidden_layers=config.num_hidden_layers,
        quantization_config=config.quantization_config,
    )
    layer_bytes, other_bytes = k2_cache_oracle(header_map, suffixes, layout)
    assert estimate.layer_bytes == layer_bytes
    assert estimate.other_bytes == other_bytes


# ==========================================================================
# the trainer setup
# ==========================================================================
def _deepseek_setup_cfg():
    return SimpleNamespace(
        model_type="deepseek_v3", hidden_size=4, num_hidden_layers=1, vocab_size=8,
        intermediate_size=8,
    )


def test_the_setup_loads_the_model_config_through_the_k2_loader(tmp_path, monkeypatch):
    import soup_cli.utils.kimi_k2 as kimi_k2
    from tests.test_issue602_qwen4_ple_streaming import _drive_qwen4_streaming_setup

    seen = []
    model_cfg = _deepseek_setup_cfg()

    def _loader(base, **kwargs):
        seen.append((base, kwargs))
        return model_cfg

    monkeypatch.setattr(kimi_k2, "load_stream_model_config", _loader)
    _drive_qwen4_streaming_setup(tmp_path, monkeypatch, arch="deepseek_v3", model_cfg=model_cfg)
    assert seen == [("qwen4-test", {"trust_remote_code": False})]


def test_trust_remote_code_reaches_the_tokenizer_but_never_the_streamed_config(
    tmp_path, monkeypatch
):
    """Round-2 review: with --trust-remote-code the config loader must still get False, or a K2
    base would run its repository's configuration code; the tokenizer keeps the user's flag."""
    import soup_cli.utils.kimi_k2 as kimi_k2
    from tests.test_issue602_qwen4_ple_streaming import _drive_qwen4_streaming_setup

    seen = []
    model_cfg = _deepseek_setup_cfg()

    def _loader(base, **kwargs):
        seen.append(kwargs)
        return model_cfg

    calls = {}
    monkeypatch.setattr(kimi_k2, "load_stream_model_config", _loader)
    _drive_qwen4_streaming_setup(
        tmp_path, monkeypatch, arch="deepseek_v3", model_cfg=model_cfg, load_calls=calls,
        trust_remote_code=True,
    )
    assert [kwargs["trust_remote_code"] for kwargs in seen] == [False]
    assert [call["trust_remote_code"] for call in calls["tokenizer"]] == [True]
    # Second review (MEDIUM 2): the tokenizer is handed the native config, so AutoTokenizer
    # does not read the config again with the user's flag.
    assert [call["config"] for call in calls["tokenizer"]] == [model_cfg]


def _sentinel_k2_repository(root, sentinel):
    """A local K2 base: the published config.json (its auto_map onto configuration_deepseek),
    a ``configuration_deepseek.py`` that writes ``sentinel`` when imported, and a tiny native
    fast tokenizer (no tokenizer code in the folder)."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    folder = root / "kimi-k2-local"
    _write_config(folder, K2_BASE_CONFIG)
    (folder / "configuration_deepseek.py").write_text(
        "import pathlib\n"
        f"pathlib.Path({str(sentinel)!r}).write_text('imported', encoding='utf-8')\n"
        "from transformers import PretrainedConfig\n\n\n"
        "class DeepseekV3Config(PretrainedConfig):\n"
        "    model_type = 'kimi_k2'\n",
        encoding="utf-8",
    )
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "hello": 3, "world": 4}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(folder / "tokenizer.json"))
    (folder / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast", "bos_token": "<s>", "eos_token": "</s>",
        "unk_token": "<unk>",
    }), encoding="utf-8")
    return str(folder)


def test_the_streamed_tokenizer_load_never_imports_the_repositorys_config_code(
    tmp_path, monkeypatch
):
    """Second review (MEDIUM 2), with the real AutoTokenizer: the setup's own tokenizer call,
    with --trust-remote-code, must not import the repository's configuration module. Without
    ``config=`` AutoTokenizer calls AutoConfig with the user's flag first, which follows the
    auto_map and runs the module (traced on transformers 5.17; this test fails then)."""
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    import ast
    import sys

    import transformers
    import transformers.dynamic_module_utils as dynamic

    from soup_cli.utils.kimi_k2 import load_stream_model_config
    from soup_cli.utils.run_provenance import base_revision_kwargs

    sentinel = tmp_path / "configuration_deepseek-was-imported"
    folder = _sentinel_k2_repository(tmp_path, sentinel)
    # Should the module be imported after all, its copy lands in tmp_path, not the user cache.
    monkeypatch.setattr(dynamic, "HF_MODULES_CACHE", str(tmp_path / "hf-modules"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    source_path = os.path.join(
        os.path.dirname(__file__), os.pardir, "src", "soup_cli", "trainer", "stream_setup.py"
    )
    with open(source_path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and isinstance(node.func.value, ast.Name) and node.func.value.id == "AutoTokenizer"
             and node.func.attr == "from_pretrained"]
    assert len(calls) == 1
    env = {
        "cfg": SimpleNamespace(base=folder),
        "self": SimpleNamespace(_trust_remote_code=True),
        "AutoTokenizer": transformers.AutoTokenizer,
        "model_config": load_stream_model_config(folder, trust_remote_code=False),
        "base_revision_kwargs": base_revision_kwargs,
    }
    assert not sentinel.exists()  # the config loader ran no repository code
    expression = compile(ast.Expression(body=calls[0]), "<stream tokenizer>", "eval")
    # A load that imported the module may then fail on its stand-in class: the sentinel is
    # checked first, so the failure is this test's own, not the stand-in's AttributeError.
    try:
        tokenizer, load_error = eval(expression, env), None
    except Exception as error:  # noqa: BLE001 -- reported below, after the sentinel check
        tokenizer, load_error = None, error
    assert not sentinel.exists(), "the tokenizer load imported configuration_deepseek.py"
    assert load_error is None, repr(load_error)
    assert tokenizer("hello world")["input_ids"][-2:] == [3, 4]


def test_the_setup_refuses_a_guard_failure_before_resolving_any_weight(tmp_path, monkeypatch):
    import soup_cli.utils.kimi_k2 as kimi_k2
    from tests.test_issue602_qwen4_ple_streaming import _drive_qwen4_streaming_setup

    def _loader(_base, **_kwargs):
        raise kimi_k2.UnsupportedKimiK2Error("Kimi K2 ... n_group=8")

    def _resolve(*_args, **_kwargs):
        raise AssertionError("weights were resolved")

    calls = {}
    monkeypatch.setattr(kimi_k2, "load_stream_model_config", _loader)
    with pytest.raises(kimi_k2.UnsupportedKimiK2Error, match="n_group=8"):
        _drive_qwen4_streaming_setup(
            tmp_path, monkeypatch, _resolve, arch="deepseek_v3", model_cfg=_deepseek_setup_cfg(),
            load_calls=calls,
        )
    # Before the tokenizer too (review round): a refused K2 is not first sent to install the
    # tokenizer's requirements.
    assert "tokenizer" not in calls


def test_the_meta_skeleton_loads_its_config_through_the_k2_loader(monkeypatch):
    import soup_cli.utils.kimi_k2 as kimi_k2
    from soup_cli.utils.layer_stream_runtime import build_meta_skeleton

    seen = []

    def _loader(model_id, **kwargs):
        seen.append((model_id, kwargs))
        raise RuntimeError("stop here")

    monkeypatch.setattr(kimi_k2, "load_stream_model_config", _loader)
    with pytest.raises(RuntimeError, match="stop here"):
        build_meta_skeleton("some/base", dtype="float32")
    assert seen == [("some/base", {"trust_remote_code": False})]


def _drive_failed_tokenizer(monkeypatch, failure, hint):
    """The real setup up to its tokenizer load, which raises ``failure``; returns
    (the raised exception, what was printed)."""
    import sys
    from io import StringIO

    from rich.console import Console

    import soup_cli.trainer.stream_setup as stream_setup
    import soup_cli.utils.kimi_k2 as kimi_k2

    def _tokenizer(*_args, **_kwargs):
        raise failure

    monkeypatch.setitem(
        sys.modules, "transformers",
        SimpleNamespace(
            AutoConfig=SimpleNamespace(from_pretrained=None),
            AutoTokenizer=SimpleNamespace(from_pretrained=_tokenizer),
        ),
    )
    monkeypatch.setitem(
        sys.modules, "peft", SimpleNamespace(TaskType=SimpleNamespace(CAUSAL_LM="CAUSAL_LM"))
    )
    monkeypatch.setattr(kimi_k2, "kimi_k2_tokenizer_hint", hint)
    # The config loads first (review round), so this drive starts past it.
    monkeypatch.setattr(
        kimi_k2, "load_stream_model_config", lambda *_a, **_k: _deepseek_setup_cfg()
    )
    buffer = StringIO()
    monkeypatch.setattr(stream_setup, "console", Console(file=buffer, width=200))

    class _Wrapper(stream_setup.StreamingSetupMixin):
        device = "cpu"
        _trust_remote_code = False

    cfg = SimpleNamespace(base="moonshotai/Kimi-K2-Base", task="sft")
    with pytest.raises(BaseException) as caught:
        _Wrapper()._setup_streaming_transformers(cfg, SimpleNamespace())
    return caught.value, buffer.getvalue()


@pytest.mark.parametrize(
    "failure",
    [ImportError("This modeling file requires the following packages: tiktoken"),
     ValueError("the tokenizer needs custom code")],
    ids=["tiktoken-missing", "custom-code"],
)
def test_a_failed_tokenizer_load_prints_the_k2_hint_and_reraises(monkeypatch, failure):
    """A missing tiktoken surfaces as an ImportError from transformers' remote-code loader."""
    seen = []

    def _hint(base, **kwargs):
        seen.append((base, kwargs))
        return "Kimi K2 tokenizer: pip install tiktoken"

    raised, printed = _drive_failed_tokenizer(monkeypatch, failure, _hint)
    assert raised is failure
    assert seen == [("moonshotai/Kimi-K2-Base", {})]
    assert "pip install tiktoken" in printed


def test_a_hint_that_fails_never_masks_the_tokenizers_own_error(monkeypatch):
    failure = ImportError("tiktoken")

    def _broken_hint(*_args, **_kwargs):
        raise RuntimeError("the hint itself broke")

    raised, printed = _drive_failed_tokenizer(monkeypatch, failure, _broken_hint)
    assert raised is failure
    assert "the hint itself broke" not in printed


# ==========================================================================
# the tokenizer hint
# ==========================================================================
@pytest.mark.parametrize("available", [False, True], ids=["tiktoken-missing", "tiktoken-present"])
def test_the_tokenizer_hint_for_a_k2_base(tmp_path, monkeypatch, available):
    pytest.importorskip("transformers")
    import soup_cli.utils.kimi_k2 as kimi_k2

    monkeypatch.setattr(kimi_k2, "_tiktoken_available", lambda: available)
    hint = kimi_k2.kimi_k2_tokenizer_hint(_write_config(tmp_path / "k2", K2_BASE_CONFIG))
    assert hint is not None
    assert "--trust-remote-code" in hint
    assert "DeepseekV3" in hint
    assert ("pip install tiktoken" in hint) is (not available)


def test_the_tokenizer_hint_stays_quiet_off_k2(tmp_path):
    pytest.importorskip("transformers")
    from soup_cli.utils.kimi_k2 import kimi_k2_tokenizer_hint

    llama = {"model_type": "llama", "hidden_size": 8, "num_attention_heads": 2}
    assert kimi_k2_tokenizer_hint(_write_config(tmp_path / "llama", llama)) is None
    empty = tmp_path / "empty"
    empty.mkdir()
    assert kimi_k2_tokenizer_hint(str(empty)) is None


def test_the_kimi_k2_module_is_import_light():
    """Imported on the trainer path before torch is needed: no torch or transformers at
    module level (the runtime property is pinned for the CLI in test_cli_startup_is_light)."""
    import ast
    import inspect

    import soup_cli.utils.kimi_k2 as kimi_k2

    tree = ast.parse(inspect.getsource(kimi_k2))
    top = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    names = {
        (node.module or "") if isinstance(node, ast.ImportFrom) else alias.name
        for node in top
        for alias in getattr(node, "names", [])
    }
    assert not any(name.split(".")[0] in ("torch", "transformers") for name in names), names
