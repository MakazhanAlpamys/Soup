"""The shared tiny Kimi-K2-shaped checkpoint builder.

Extracted from ``tests/test_kimi_k2_streaming.py`` (the G2 fixture, #2104 task 4) so the
resident-parity recorder (``benchmarks/harness/kimi_k2_resident_parity_record.py``) builds
the exact same seeded checkpoints the streaming suite's parity tests use. The seeds, the
build order and the written bytes are unchanged by the extraction: the streaming suite's
G2 tests still pass bit-exactly against checkpoints this module builds.

The checkpoints hold no ``.py`` file; nothing here imports or executes Kimi K2's own
Python. The FP8 copy is written by ``benchmarks/harness/fp8_block_quantize.py``, loaded by
path through ``importlib`` (the pattern of ``tests/test_fp8_block_streaming.py``), and the
independent block-loop dequantisation cross-check stays in that test file.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import sys

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


def _base(**changes):
    raw = copy.deepcopy(K2_BASE_CONFIG)
    raw.update(copy.deepcopy(changes))
    return raw


def _without(*keys, raw=None):
    raw = _base() if raw is None else raw
    for key in keys:
        raw.pop(key)
    return raw


#: The tiny model's dimensions: the same values tests/test_kimi_k2_streaming.py's G2 gate
#: registers (benchmarks/gate-kimi-k2-admission.md, section 2).
_TINY = dict(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=128,
    moe_intermediate_size=32,
    num_hidden_layers=3,
    first_k_dense_replace=1,
    num_attention_heads=4,
    num_key_value_heads=4,
    n_shared_experts=1,
    n_routed_experts=8,
    n_group=1,
    topk_group=1,
    num_experts_per_tok=2,
    kv_lora_rank=16,
    q_lora_rank=24,
    qk_nope_head_dim=16,
    qk_rope_head_dim=8,
    v_head_dim=16,
    max_position_embeddings=2048,
    bos_token_id=0,
    eos_token_id=1,
)
_TINY_YARN = {
    "beta_fast": 1.0,
    "beta_slow": 1.0,
    "factor": 32.0,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
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
        **_TINY,
        rope_parameters=rope,
        rms_norm_eps=1e-6,
        routed_scaling_factor=2.827,
        norm_topk_prob=True,
        tie_word_embeddings=False,
        num_mtp_layers=0,
    )


def _fp8_harness():
    """``benchmarks/harness/fp8_block_quantize.py``, loaded by path (never via a tests import)."""
    import importlib.util
    from pathlib import Path

    harness = Path(__file__).resolve().parent.parent / "benchmarks" / "harness"
    if str(harness) not in sys.path:
        sys.path.insert(0, str(harness))
    spec = importlib.util.spec_from_file_location(
        "fp8_block_quantize", harness / "fp8_block_quantize.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_tiny_k2_checkpoints(root):
    """``{"bf16": (K2 folder, resident folder), "fp8": (...)}``: the same seeded weights.

    Seeded ``torch.manual_seed(11)`` fp32 ``DeepseekV3ForCausalLM`` on
    :func:`_native_tiny_config`, ``e_score_correction_bias`` re-seeded with
    ``Generator(23)``, everything saved bf16. The K2 folder is the resident folder's
    safetensors plus a stale per-layer BF16 ``rotary_emb.inv_freq`` and the real Base
    ``config.json`` with tiny dimensions substituted (:func:`_tiny_k2_raw`); it holds no
    ``.py`` file. The FP8 copy is written by ``fp8_block_quantize.py`` (block 16 x 32), and
    the fourth folder dequantises that copy through the independent block loop in
    ``tests/test_fp8_block_streaming.py``.
    """
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import DeepseekV3ForCausalLM

    from tests.test_fp8_block_streaming import _reference_dequant

    root = str(root)
    torch.manual_seed(11)
    model = DeepseekV3ForCausalLM(_native_tiny_config()).to(torch.float32).eval()
    generator = torch.Generator().manual_seed(23)
    with torch.no_grad():
        for layer in model.model.layers[1:]:
            bias = layer.mlp.gate.e_score_correction_bias
            bias.copy_(torch.randn(bias.shape, generator=generator) * 2.0)
        for parameter in model.parameters():
            parameter.data = parameter.data.to(torch.bfloat16)
    native = os.path.join(root, "native-bf16")
    model.save_pretrained(native)
    state = load_file(os.path.join(native, "model.safetensors"))
    # The real index's per-expert spelling, which the sharder stacks.
    assert "model.layers.1.mlp.experts.0.gate_proj.weight" in state
    stale = {
        f"model.layers.{idx}.{_STALE_INV_FREQ}": torch.full((7,), _STALE_VALUE).to(torch.bfloat16)
        for idx in range(_TINY_LAYERS)
    }
    k2 = os.path.join(root, "k2-bf16")
    os.makedirs(k2)
    save_file({**state, **stale}, os.path.join(k2, "model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(k2, "config.json"), "w", encoding="utf-8") as handle:
        handle.write(json.dumps(_tiny_k2_raw()))

    harness = _fp8_harness()
    fp8 = os.path.join(root, "k2-fp8")
    harness.convert_checkpoint(k2, fp8, block=_BLOCK, log=lambda _line: None)
    fp8_state = load_file(os.path.join(fp8, "model.safetensors"))
    assert any(value.dtype == torch.float8_e4m3fn for value in fp8_state.values())
    ref_state = {}
    for key, value in fp8_state.items():
        if key.endswith("_scale_inv") or key.endswith(_STALE_INV_FREQ):
            continue
        scale = fp8_state.get(key + "_scale_inv")
        ref_state[key] = value if scale is None else _reference_dequant(value, scale, _BLOCK)
    ref = os.path.join(root, "native-fp8-ref")
    os.makedirs(ref)
    save_file(ref_state, os.path.join(ref, "model.safetensors"), metadata={"format": "pt"})
    shutil.copyfile(os.path.join(native, "config.json"), os.path.join(ref, "config.json"))
    assert not any(name.endswith(".py") for folder in (k2, fp8) for name in os.listdir(folder))
    return {"bf16": (k2, native), "fp8": (fp8, ref)}
