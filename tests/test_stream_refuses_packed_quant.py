"""Layer streaming refuses, by name and before a byte is written, a source it cannot read.

Kimi-K2-Thinking ships its routed experts as compressed-tensors ``pack-quantized`` INT4:
``quantization_config.quant_method: "compressed-tensors"`` and, per linear,
``<prefix>.weight_packed`` (int32 words, eight int4 values each), ``<prefix>.weight_scale``
and ``<prefix>.weight_shape`` (int64). On ``model_type: deepseek_v3`` the sharder
accepted it: ``plan_fp8_block`` refused compressed-tensors only beside FP8 tensors, so it
cast every int32 word and every int64 shape to the stream dtype, wrote the whole cache,
and the streamed build then failed with "shard is missing decoder weights". A source
whose names still match (bitsandbytes LLM.int8(): an int8 ``weight`` plus an ``SCB``)
did not fail at all: the int8 codes were streamed as weights and trained on.

Refused now, in one shared path (``utils/fp8_block.py``, where #2061 put the FP8 one):

- a ``quantization_config`` declaring any ``quant_method`` but ``fp8``
  (:func:`check_source_quantization` from the config alone; :func:`plan_fp8_block`
  again over the headers, which the sharder and the disk pre-flight both run);
- a tensor stored as an integer or packed type, or named like a quantiser's companion
  (:func:`refuse_unsupported_tensor`), in the header plan and under every plain read.

Kept: plain floating point, DeepSeek-V3's block FP8, qwen4_exp's oMLX/oQ affine bundles,
and the integer buffers a supported model really has (qwen4_exp's PLE hashing tables).
"""

from __future__ import annotations

import json
import os
import re
import types

import pytest

_TAIL = "Layer streaming cannot dequantise it"
_POINTER = "Point training.base at a bf16 or FP8-block copy of this model"
_FP8_TAIL = "DeepSeek-V3's block FP8"

#: Kimi-K2-Thinking's own quantization_config (text of its config.json, 2026-10).
K2_THINKING_QUANT = {
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
}
K2_NEEDLE = (
    "quant_method 'compressed-tensors' (format 'pack-quantized', 4-bit int weights in "
    "groups of 32)"
)
DEEPSEEK_FP8 = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [128, 128],
}

HIDDEN, INTER, VOCAB, EXPERTS, LAYERS = 64, 32, 16, 2, 2


# ==========================================================================
# synthetic checkpoints (real safetensors files in tmp_path)
# ==========================================================================
def _dense_state(prefix="model."):
    """A tiny llama-shaped bf16 state; ``prefix`` spells a wrapper (``language_model.``)."""
    import torch

    generator = torch.Generator().manual_seed(5)

    def normal(*shape):
        return torch.randn(*shape, generator=generator).to(torch.bfloat16)

    state = {
        f"{prefix}embed_tokens.weight": normal(VOCAB, HIDDEN),
        f"{prefix}norm.weight": normal(HIDDEN),
        "lm_head.weight": normal(VOCAB, HIDDEN),
    }
    for layer in range(LAYERS):
        base = f"{prefix}layers.{layer}."
        state[base + "input_layernorm.weight"] = normal(HIDDEN)
        state[base + "self_attn.o_proj.weight"] = normal(HIDDEN, HIDDEN)
        state[base + "mlp.down_proj.weight"] = normal(HIDDEN, INTER)
    return state


def _k2_state(prefix="model."):
    """Kimi-K2-Thinking's layout on a tiny DeepSeek-V3: bf16 attention, router and norms,
    float32 router bias and rotary buffer, compressed-tensors INT4 routed experts."""
    import torch

    generator = torch.Generator().manual_seed(7)
    state = {
        f"{prefix}embed_tokens.weight": torch.randn(VOCAB, HIDDEN).to(torch.bfloat16),
        f"{prefix}norm.weight": torch.ones(HIDDEN, dtype=torch.bfloat16),
        "lm_head.weight": torch.randn(VOCAB, HIDDEN).to(torch.bfloat16),
    }
    for layer in range(LAYERS):
        base = f"{prefix}layers.{layer}."
        state[base + "input_layernorm.weight"] = torch.ones(HIDDEN, dtype=torch.bfloat16)
        state[base + "self_attn.o_proj.weight"] = torch.randn(HIDDEN, HIDDEN).to(
            torch.bfloat16
        )
        state[base + "self_attn.rotary_emb.inv_freq"] = torch.ones(4)
        state[base + "mlp.gate.weight"] = torch.randn(EXPERTS, HIDDEN).to(torch.bfloat16)
        state[base + "mlp.gate.e_score_correction_bias"] = torch.zeros(EXPERTS)
        for expert in range(EXPERTS):
            for name, (rows, cols) in (
                ("gate_proj", (INTER, HIDDEN)),
                ("up_proj", (INTER, HIDDEN)),
                ("down_proj", (HIDDEN, INTER)),
            ):
                key = base + f"mlp.experts.{expert}.{name}."
                state[key + "weight_packed"] = torch.randint(
                    -(2**31), 2**31 - 1, (rows, cols // 8), dtype=torch.int32,
                    generator=generator,
                )
                state[key + "weight_scale"] = torch.rand(
                    rows, cols // 32, generator=generator
                ).to(torch.bfloat16)
                state[key + "weight_shape"] = torch.tensor([rows, cols], dtype=torch.int64)
    return state


def _write(directory, state, *, quant=None, model_type="deepseek_v3"):
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=True)
    save_file(state, str(directory / "model.safetensors"))
    config = {"model_type": model_type, "num_hidden_layers": LAYERS}
    if quant is not None:
        config["quantization_config"] = quant
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(directory)


def _shard(weights, out, *, arch="deepseek_v3"):
    from soup_cli.utils.layer_shard import shard_checkpoint

    kwargs = {"num_hidden_layers": LAYERS} if arch == "deepseek_v3" else {}
    return shard_checkpoint(weights, str(out), dtype="bfloat16", arch=arch, **kwargs)


def _estimate(weights, *, arch="deepseek_v3"):
    from soup_cli.utils.fp8_block import read_checkpoint_quantization_config
    from soup_cli.utils.layer_shard import _discover_safetensors, estimate_shard_cache_bytes

    return estimate_shard_cache_bytes(
        _discover_safetensors(weights), dtype="bfloat16", arch=arch,
        num_hidden_layers=LAYERS if arch == "deepseek_v3" else None,
        quantization_config=read_checkpoint_quantization_config(weights),
    )


def _refusals(tmp_path, weights, *, arch="deepseek_v3"):
    """(sharder message, estimator message); asserts the sharder wrote nothing."""
    out = tmp_path / "shards"
    with pytest.raises(ValueError) as sharder:
        _shard(weights, out, arch=arch)
    assert not out.exists() or not os.listdir(out)
    with pytest.raises(ValueError) as estimator:
        _estimate(weights, arch=arch)
    return str(sharder.value), str(estimator.value)


# ==========================================================================
# the reported source: Kimi-K2-Thinking's compressed-tensors INT4 on deepseek_v3
# ==========================================================================
def test_a_k2_thinking_int4_source_is_refused_before_a_byte_is_written(tmp_path):
    weights = _write(tmp_path / "k2", _k2_state(), quant=K2_THINKING_QUANT)
    sharder, estimator = _refusals(tmp_path, weights)
    assert K2_NEEDLE in sharder
    assert _TAIL in sharder and _POINTER in sharder
    assert estimator == sharder


def test_the_k2_refusal_is_the_shared_quantised_source_error_not_an_fp8_one(tmp_path):
    from soup_cli.utils.fp8_block import UnsupportedFP8Error, UnsupportedQuantizedSourceError

    weights = _write(tmp_path / "k2", _k2_state(), quant=K2_THINKING_QUANT)
    with pytest.raises(UnsupportedQuantizedSourceError) as caught:
        _shard(weights, tmp_path / "s")
    assert not isinstance(caught.value, UnsupportedFP8Error)
    assert issubclass(UnsupportedFP8Error, UnsupportedQuantizedSourceError)


@pytest.mark.parametrize(
    "prefix", ["model.", "language_model.model.", "model.language_model."],
    ids=["plain", "language_model", "vlm"],
)
def test_without_its_config_the_packed_tensors_still_refuse_in_one_voice(tmp_path, prefix):
    """No quantization_config at all (stripped, or a hand conversion): the int32 words
    are refused by their own name and type, whatever wrapper prefix spells them."""
    weights = _write(tmp_path / "k2", _k2_state(prefix))
    sharder, estimator = _refusals(tmp_path, weights)
    assert "model.layers.0.mlp.experts.0.down_proj.weight_packed" in sharder
    assert "is stored as 'I32': compressed-tensors pack-quantized" in sharder
    assert _TAIL in sharder and _POINTER in sharder
    assert estimator == sharder


def test_the_setup_refuses_a_k2_base_before_resolving_its_weights(tmp_path, monkeypatch):
    from tests.test_issue602_qwen4_ple_streaming import _drive_qwen4_streaming_setup

    def _must_not_resolve(*_args, **_kwargs):
        raise AssertionError("weights resolved for a base that cannot stream")

    model_cfg = types.SimpleNamespace(
        model_type="deepseek_v3", hidden_size=HIDDEN, num_hidden_layers=LAYERS,
        vocab_size=VOCAB, intermediate_size=INTER, quantization_config=K2_THINKING_QUANT,
    )
    with pytest.raises(ValueError) as caught:
        _drive_qwen4_streaming_setup(
            tmp_path, monkeypatch, resolve_weights=_must_not_resolve, arch="deepseek_v3",
            model_cfg=model_cfg,
        )
    assert K2_NEEDLE in str(caught.value) and _TAIL in str(caught.value)


# ==========================================================================
# the config gate: every declared quantiser but fp8
# ==========================================================================
_CT_GROUP = {"group_0": {"targets": ["Linear"], "weights": {"num_bits": 8, "type": "int"}}}
_CONFIG_SPELLINGS = {
    "k2-thinking": (K2_THINKING_QUANT, K2_NEEDLE, False),
    "ct-int8": (
        {"quant_method": "compressed-tensors", "format": "int-quantized",
         "config_groups": _CT_GROUP},
        "(format 'int-quantized', 8-bit int weights)", False,
    ),
    "ct-nvfp4": (
        {"quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized"},
        "(format 'nvfp4-pack-quantized')", False,
    ),
    "ct-fp8": (
        {"quant_method": "compressed-tensors", "format": "float-quantized"},
        "(format 'float-quantized')", True,
    ),
    "ct-fp8-by-group": (
        {"quant_method": "compressed-tensors",
         "config_groups": {"g": {"weights": {"num_bits": 8, "type": "float"}}}},
        "(8-bit float weights)", True,
    ),
    "ct-bare": ({"quant_method": "compressed-tensors"}, "'compressed-tensors'.", False),
    "gptq": ({"quant_method": "gptq", "bits": 4, "group_size": 128}, "'gptq' (4-bit)", False),
    "awq": ({"quant_method": "awq", "bits": 4, "version": "gemm"}, "'awq' (4-bit)", False),
    "bnb-4bit": (
        {"quant_method": "bitsandbytes", "load_in_4bit": True,
         "bnb_4bit_quant_type": "nf4"},
        "'bitsandbytes' (4-bit)", False,
    ),
    "bnb-8bit": (
        {"quant_method": "bitsandbytes", "load_in_8bit": True}, "'bitsandbytes' (8-bit)",
        False,
    ),
    "bnb-without-method": (
        {"load_in_4bit": True}, "declares load_in_4bit: bitsandbytes pre-quantised 4-bit",
        False,
    ),
    "hqq": ({"quant_method": "hqq", "nbits": 4}, "'hqq'", False),
    "aqlm": ({"quant_method": "aqlm", "nbits_per_codebook": 16}, "'aqlm'", False),
    "eetq": ({"quant_method": "eetq", "weights": "int8"}, "'eetq'", False),
    "torchao": ({"quant_method": "torchao", "quant_type": {}}, "'torchao'", False),
    "quanto": ({"quant_method": "quanto", "weights": "int8"}, "'quanto'", False),
    "mxfp4": ({"quant_method": "mxfp4"}, "'mxfp4'", False),
    "fbgemm-fp8": ({"quant_method": "fbgemm_fp8"}, "'fbgemm_fp8'", True),
    "mxfp8": ({"quant_method": "mxfp8"}, "'mxfp8'", True),
    "unknown": ({"quant_method": "my-own-quantizer"}, "'my-own-quantizer'", False),
    "not-a-string": ({"quant_method": 5}, "quant_method 5.", False),
    "empty-string": ({"quant_method": ""}, "quant_method ''.", False),
    "list": ({"quant_method": ["gptq"]}, "quant_method ['gptq'].", False),
}


@pytest.mark.parametrize("spelling", sorted(_CONFIG_SPELLINGS))
def test_every_declared_quantiser_but_fp8_is_refused_by_name(spelling):
    from soup_cli.utils.fp8_block import (
        UnsupportedFP8Error,
        UnsupportedQuantizedSourceError,
        check_source_quantization,
    )

    config, needle, is_fp8 = _CONFIG_SPELLINGS[spelling]
    with pytest.raises(UnsupportedQuantizedSourceError) as caught:
        check_source_quantization(config, arch="deepseek_v3")
    message = str(caught.value)
    assert needle in message
    assert isinstance(caught.value, UnsupportedFP8Error) is is_fp8
    assert (_FP8_TAIL if is_fp8 else _POINTER) in message
    assert "Dequantise" in message or "dequantise" in message


def test_every_quantiser_transformers_registers_but_fp8_is_refused():
    """The registry of the installed transformers (5.17: 27 names), not a copy of it."""
    pytest.importorskip("transformers")
    from transformers.quantizers.auto import AUTO_QUANTIZATION_CONFIG_MAPPING

    from soup_cli.utils.fp8_block import (
        UnsupportedQuantizedSourceError,
        check_source_quantization,
    )

    methods = sorted(set(AUTO_QUANTIZATION_CONFIG_MAPPING) | {"bitsandbytes"})
    assert {"compressed-tensors", "gptq", "awq", "bitsandbytes_4bit", "hqq"} <= set(methods)
    accepted = []
    for method in methods:
        try:
            check_source_quantization({"quant_method": method}, arch="llama")
        except UnsupportedQuantizedSourceError as exc:
            assert repr(method) in str(exc), method
        else:
            accepted.append(method)
    assert accepted == []
    # 'fp8' is accepted only as DeepSeek-V3's block format, on deepseek_v3:
    assert check_source_quantization(DEEPSEEK_FP8, arch="deepseek_v3") == (128, 128)


def test_a_transformers_quantization_config_object_is_read_too():
    transformers = pytest.importorskip("transformers")
    from soup_cli.utils.fp8_block import (
        UnsupportedQuantizedSourceError,
        check_source_quantization,
    )

    with pytest.raises(UnsupportedQuantizedSourceError, match=r"'bitsandbytes' \(4-bit\)"):
        check_source_quantization(
            transformers.BitsAndBytesConfig(load_in_4bit=True), arch="llama"
        )


@pytest.mark.parametrize(
    "config",
    [
        None,
        {},
        {"bits": 4, "group_size": 32, "mode": "affine"},
        {"group_size": 64, "bits": 4, "model.layers.0.mlp.gate": {"bits": 8, "group_size": 64}},
        {"quant_method": None},
        {"load_in_4bit": False, "load_in_8bit": False},
    ],
    ids=["none", "empty", "oq-affine", "mlx-overrides", "null-method", "bnb-flags-off"],
)
def test_a_config_that_declares_no_quantiser_passes_the_gate(config):
    """The oMLX/oQ affine spelling declares no quant_method: its tensors decide."""
    from soup_cli.utils.fp8_block import check_source_quantization

    assert check_source_quantization(config, arch="qwen4_exp") is None


def _corrupt_checkpoint_with_config(directory, quantization_config):
    """A config naming a refused quantiser beside weights no reader could open."""
    directory.mkdir(parents=True)
    (directory / "model.safetensors").write_bytes(b"\xff" * 64)
    config = {"model_type": "deepseek_v3", "quantization_config": quantization_config}
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(directory)


@pytest.mark.parametrize("spelling", ["k2-thinking", "gptq", "bnb-4bit", "ct-fp8"])
def test_the_config_alone_refuses_before_any_header_in_every_caller(tmp_path, spelling):
    from soup_cli.utils.fp8_block import check_source_quantization
    from soup_cli.utils.layer_shard import estimate_shard_cache_bytes, shard_checkpoint

    config = _CONFIG_SPELLINGS[spelling][0]
    weights = _corrupt_checkpoint_with_config(tmp_path / "w", config)
    with pytest.raises(ValueError) as gate:
        check_source_quantization(config, arch="deepseek_v3")
    with pytest.raises(ValueError) as sharder:
        shard_checkpoint(weights, str(tmp_path / "s"), dtype="bfloat16", arch="deepseek_v3")
    with pytest.raises(ValueError) as estimator:
        estimate_shard_cache_bytes(
            [os.path.join(weights, "model.safetensors")], dtype="bfloat16",
            arch="deepseek_v3", quantization_config=config,
        )
    assert str(sharder.value) == str(estimator.value) == str(gate.value)


_MALFORMED_GROUPS = {
    "list": ["group_0"],
    "string": "group_0",
    "group-not-a-mapping": {"g": ["weights"]},
    "weights-not-a-mapping": {"g": {"weights": "int4"}},
    "bool-bits": {"g": {"weights": {"num_bits": True, "type": "int"}}},
}


@pytest.mark.parametrize("spelling", sorted(_MALFORMED_GROUPS))
def test_a_malformed_compressed_tensors_config_is_still_refused_cleanly(spelling):
    from soup_cli.utils.fp8_block import (
        UnsupportedFP8Error,
        UnsupportedQuantizedSourceError,
        check_source_quantization,
    )

    config = {"quant_method": "compressed-tensors", "config_groups": _MALFORMED_GROUPS[spelling]}
    with pytest.raises(UnsupportedQuantizedSourceError) as caught:
        check_source_quantization(config, arch="deepseek_v3")
    assert not isinstance(caught.value, UnsupportedFP8Error)
    assert "declares quant_method 'compressed-tensors'. " in str(caught.value)


def test_a_bool_group_size_is_not_printed_as_one():
    from soup_cli.utils.fp8_block import UnsupportedQuantizedSourceError, check_source_quantization

    config = {
        "quant_method": "compressed-tensors",
        "config_groups": {"g": {"weights": {"num_bits": 4, "type": "int", "group_size": True}}},
    }
    with pytest.raises(UnsupportedQuantizedSourceError) as caught:
        check_source_quantization(config, arch="deepseek_v3")
    assert "(4-bit int weights). " in str(caught.value)


def test_a_bool_bit_width_is_not_printed_as_one():
    from soup_cli.utils.fp8_block import UnsupportedQuantizedSourceError, check_source_quantization

    with pytest.raises(UnsupportedQuantizedSourceError) as caught:
        check_source_quantization({"quant_method": "gptq", "bits": True}, arch="llama")
    assert "declares quant_method 'gptq'. " in str(caught.value)


def test_the_header_plan_repeats_the_config_gate():
    """plan_fp8_block is the one plan both callers run: it refuses a declared quantiser
    even when no FP8 tensor is present (where it used to return None)."""
    from soup_cli.utils.fp8_block import UnsupportedQuantizedSourceError, plan_fp8_block

    tensors = {"model.layers.0.o_proj.weight": ("BF16", (4, 4))}
    with pytest.raises(UnsupportedQuantizedSourceError, match="'gptq'"):
        plan_fp8_block({"quant_method": "gptq"}, tensors, arch="llama")
    assert plan_fp8_block(None, tensors, arch="llama") is None


# ==========================================================================
# the tensor rule: stored types and quantiser names
# ==========================================================================
_INT_DTYPES = {
    "int8": "I8", "int16": "I16", "int32": "I32", "int64": "I64", "uint8": "U8",
    "bool": "BOOL", "uint16": "U16", "uint32": "U32", "uint64": "U64",
}
_WHERE = {
    "layer": "model.layers.1.mlp.down_proj.weight",
    "extras": "model.norm.weight",
    "head": "lm_head.weight",
}


@pytest.mark.parametrize("where", sorted(_WHERE))
@pytest.mark.parametrize("dtype", sorted(_INT_DTYPES))
def test_an_integer_or_packed_tensor_is_refused_wherever_it_sits(tmp_path, dtype, where):
    """U16/U32/U64 are types the header reader has no spelling for: the pre-flight used
    to call the header unreadable and size the cache at a guess; it now refuses the
    tensor in the sharder's words."""
    import torch

    state = _dense_state()
    key = _WHERE[where]
    state[key] = torch.ones(state[key].shape, dtype=torch.float32).to(getattr(torch, dtype))
    weights = _write(tmp_path / "w", state, model_type="llama")
    sharder, estimator = _refusals(tmp_path, weights, arch="llama")
    assert f"checkpoint tensor {key!r} is stored as {_INT_DTYPES[dtype]!r}" in sharder
    assert _TAIL in sharder and _POINTER in sharder
    assert estimator == sharder


@pytest.mark.parametrize(
    ("suffix", "needle"),
    [
        ("weight_scale", "compressed-tensors ('weight_scale'"),
        ("weight_global_scale", "compressed-tensors NVFP4"),
        ("input_scale", "static activation scales"),
        ("SCB", "bitsandbytes LLM.int8()"),
        ("weight.absmax", "bitsandbytes 4-bit"),
        ("weight.quant_map", "bitsandbytes 4-bit"),
        ("weight.nested_absmax", "bitsandbytes 4-bit"),
        ("qweight_proj.weight", None),
        ("qweight", "GPTQ or AWQ"),
        ("g_idx", "GPTQ"),
        ("W_q", "HQQ"),
        ("codebooks", "AQLM"),
        ("weight_scales", "EETQ"),
    ],
)
def test_a_float_tensor_named_by_a_quantiser_is_refused(tmp_path, suffix, needle):
    """A float companion beside a weight that is still quantised: the scales of an INT4
    or int8 weight streamed as a parameter while the weight itself is not dequantised."""
    import torch

    state = _dense_state()
    key = f"model.layers.0.self_attn.o_proj.{suffix}"
    state[key] = torch.ones(HIDDEN, 1)
    weights = _write(tmp_path / "w", state, model_type="llama")
    if needle is None:  # a name that only CONTAINS one is not refused
        _shard(weights, tmp_path / "s", arch="llama")
        _estimate(weights, arch="llama")
        return
    sharder, estimator = _refusals(tmp_path, weights, arch="llama")
    assert f"checkpoint tensor {key!r} ('F32') belongs to {needle}" in sharder
    assert _TAIL in sharder and _POINTER in sharder
    assert estimator == sharder


_Q = "model.layers.0.self_attn.q_proj."
#: Each quantiser's own spelling of its companion tensors, written out here (not read off
#: QUANTISED_KEY_SUFFIXES), with the format the refusal must name.
_QUANTISER_SPELLINGS = {
    _Q + "weight_packed": "compressed-tensors pack-quantized (int weights packed",
    _Q + "weight_shape": "compressed-tensors pack-quantized ('weight_shape'",
    _Q + "weight_global_scale": "compressed-tensors NVFP4",
    _Q + "weight_zero_point": "compressed-tensors (asymmetric zero points)",
    _Q + "weight_g_idx": "compressed-tensors (activation-order group index)",
    _Q + "weight_scale": "compressed-tensors ('weight_scale'; DeepSeek-V3's block FP8",
    _Q + "input_global_scale": "static activation scales (NVFP4)",
    _Q + "input_scale": "static activation scales",
    _Q + "output_scale": "static activation scales",
    _Q + "qweight": "GPTQ or AWQ (int4 weights packed into int32 'qweight' words)",
    _Q + "qzeros": "GPTQ or AWQ (int32-packed 'qzeros' zero points)",
    _Q + "g_idx": "GPTQ (activation-order 'g_idx')",
    _Q + "SCB": "bitsandbytes LLM.int8()",
    _Q + "weight_format": "bitsandbytes LLM.int8()",
    _Q + "weight.quant_state.bitsandbytes__nf4": "bitsandbytes 4-bit",
    _Q + "weight.quant_state.bitsandbytes__fp4": "bitsandbytes 4-bit",
    _Q + "weight.quant_map": "bitsandbytes 4-bit",
    _Q + "weight.nested_quant_map": "bitsandbytes 4-bit",
    _Q + "weight.absmax": "bitsandbytes 4-bit",
    _Q + "weight.nested_absmax": "bitsandbytes 4-bit",
    _Q + "W_q": "HQQ (packed 'W_q')",
    _Q + "codebooks": "AQLM (additive codebooks)",
    _Q + "codes": "AQLM (int codes)",
    _Q + "weight_scales": "EETQ",
    _Q + "weight._data": "quanto (quantised '_data')",
    _Q + "weight._scale": "quanto (quantised '_scale')",
    _Q + "_weight_qdata": "torchao",
}


@pytest.mark.parametrize("key", sorted(_QUANTISER_SPELLINGS))
def test_each_quantisers_companion_spelling_is_refused_naming_it(key):
    from soup_cli.utils.fp8_block import UnsupportedQuantizedSourceError, refuse_unsupported_tensor

    with pytest.raises(UnsupportedQuantizedSourceError) as caught:
        refuse_unsupported_tensor(key, "BF16")
    assert _QUANTISER_SPELLINGS[key] in str(caught.value)


@pytest.mark.parametrize(
    ("dtype", "needle"),
    [
        ("I32", "int32, the word GPTQ, AWQ and compressed-tensors pack int4 weights into"),
        ("U32", "uint32, the word oMLX/oQ affine (MLX) checkpoints pack"),
        ("U8", "uint8, which bitsandbytes 4-bit packs NF4/FP4 nibbles into"),
        ("I8", "int8, an int8-quantised weight"),
        ("F4", "FP4, packed MXFP4/NVFP4 values"),
        ("I16", "an integer or packed type"),
        ("BOOL", "an integer or packed type"),
    ],
)
def test_a_plain_name_in_a_packed_type_is_named_by_what_the_type_holds(dtype, needle):
    from soup_cli.utils.fp8_block import UnsupportedQuantizedSourceError, refuse_unsupported_tensor

    with pytest.raises(UnsupportedQuantizedSourceError) as caught:
        refuse_unsupported_tensor("model.layers.0.self_attn.q_proj.weight", dtype)
    assert f"is stored as {dtype!r}: {needle}" in str(caught.value)


def test_an_mlx_affine_source_on_another_architecture_is_refused_naming_the_format(tmp_path):
    """oMLX/oQ affine is read on qwen4_exp only; elsewhere its uint32 words would be cast."""
    import torch

    state = _dense_state()
    state["model.layers.0.self_attn.o_proj.weight"] = torch.zeros(
        (HIDDEN, HIDDEN // 8), dtype=torch.uint32
    )
    state["model.layers.0.self_attn.o_proj.scales"] = torch.ones(HIDDEN, 1).to(torch.bfloat16)
    state["model.layers.0.self_attn.o_proj.biases"] = torch.zeros(HIDDEN, 1).to(torch.bfloat16)
    weights = _write(
        tmp_path / "w", state, quant={"bits": 4, "group_size": 32}, model_type="llama"
    )
    sharder, estimator = _refusals(tmp_path, weights, arch="llama")
    assert "is stored as 'U32': uint32, the word oMLX/oQ affine (MLX)" in sharder
    assert "on qwen4_exp only" in sharder
    assert estimator == sharder


def test_a_header_only_type_is_refused_under_its_canonical_name(tmp_path):
    """The pre-flight words a type its reader lacks by the sharder's canonical key, so a
    wrapper prefix does not make the two messages differ."""
    import torch

    state = _dense_state("language_model.model.")
    state["language_model.model.layers.1.mlp.down_proj.weight"] = torch.zeros(
        (HIDDEN, INTER), dtype=torch.uint32
    )
    weights = _write(tmp_path / "w", state, model_type="llama")
    sharder, estimator = _refusals(tmp_path, weights, arch="llama")
    assert "checkpoint tensor 'model.layers.1.mlp.down_proj.weight' is stored as 'U32'" in sharder
    assert estimator == sharder


def test_a_header_dtype_no_quantiser_writes_still_reads_as_an_unreadable_header(tmp_path):
    """Only types safetensors defines become the refusal; a corrupt spelling stays the
    pre-flight's 'header unreadable' fallback, as before."""
    import struct

    from soup_cli.utils.layer_shard import SourceHeaderError, estimate_shard_cache_bytes

    for raw in ("XYZ", "F8_E8M0", "C64"):
        payload = json.dumps(
            {"model.layers.0.w": {"dtype": raw, "shape": [1], "data_offsets": [0, 1]}}
        ).encode()
        path = tmp_path / f"{raw}.safetensors"
        path.write_bytes(struct.pack("<Q", len(payload)) + payload + b"\0")
        with pytest.raises(SourceHeaderError):
            estimate_shard_cache_bytes([str(path)], dtype="bfloat16", arch="llama")


def test_the_disk_preflight_lets_the_refusal_through_instead_of_guessing(tmp_path):
    """``_preflight_cache_estimate`` falls back to the coarse figure on an unreadable
    header only: a source the sharder would refuse is refused before any copy."""
    import torch

    from soup_cli.trainer.stream_setup import _preflight_cache_estimate, _ResolvedWeights
    from soup_cli.utils.fp8_block import UnsupportedQuantizedSourceError

    state = _dense_state()
    state["model.layers.0.self_attn.o_proj.weight"] = torch.zeros(
        (HIDDEN, HIDDEN // 8), dtype=torch.uint32
    )
    weights = _write(tmp_path / "w", state, model_type="llama")
    with pytest.raises(UnsupportedQuantizedSourceError, match="'U32'"):
        _preflight_cache_estimate(_ResolvedWeights(weights), arch="llama", dtype="bfloat16")


# ==========================================================================
# the backstop under every plain read
# ==========================================================================
@pytest.mark.parametrize("dtype", ["I8", "I16", "I32", "I64", "U8", "U16", "U32", "BOOL", "F4"])
def test_the_plain_read_backstop_refuses_an_integer_or_packed_tensor(dtype):
    from soup_cli.utils.fp8_block import UnsupportedFP8Error, UnsupportedQuantizedSourceError
    from soup_cli.utils.layer_shard import _refuse_fp8

    with pytest.raises(UnsupportedQuantizedSourceError) as caught:
        _refuse_fp8("model.layers.0.q_proj.weight", dtype)
    assert not isinstance(caught.value, UnsupportedFP8Error)
    assert repr(dtype) in str(caught.value)


@pytest.mark.parametrize(
    "key",
    [
        "model.layers.0.q_proj.weight_packed",
        "model.layers.0.q_proj.SCB",
        "model.layers.0.q_proj.weight.absmax",
    ],
)
def test_the_plain_read_backstop_refuses_a_quantiser_companion(key):
    from soup_cli.utils.fp8_block import UnsupportedQuantizedSourceError
    from soup_cli.utils.layer_shard import _refuse_fp8

    with pytest.raises(UnsupportedQuantizedSourceError, match="belongs to"):
        _refuse_fp8(key, "BF16")


@pytest.mark.parametrize(
    ("key", "dtype"),
    [
        ("model.layers.0.q_proj.weight", "BF16"),
        ("model.layers.0.q_proj.weight", "F16"),
        ("model.layers.0.q_proj.weight", "F32"),
        ("model.layers.0.q_proj.weight", "F64"),
        ("model.layers.0.ple.ple_embedding.layer_multipliers", "I64"),
        ("language_model.model.layers.3.ple.ple_embedding.ngram_heads_offsets", "I64"),
        ("model.layers.3.ple.ple_embedding.ngram_heads_vocab_sizes", "I32"),
        ("model.visual.embeddings.position_ids", "I64"),
        ("model.layers.0.self_attn.rotary_emb.inv_freq", "F32"),
        ("model.layers.0.mlp.gate.e_score_correction_bias", "F32"),
        ("model.layers.0.q_proj.scales", "BF16"),
        ("model.layers.0.q_proj.biases", "BF16"),
    ],
)
def test_the_plain_read_backstop_passes_floats_and_real_integer_buffers(key, dtype):
    from soup_cli.utils.layer_shard import _refuse_fp8

    _refuse_fp8(key, dtype)


def test_a_plain_read_of_an_integer_tensor_raises_before_the_cast(tmp_path):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    from soup_cli.utils.fp8_block import UnsupportedQuantizedSourceError
    from soup_cli.utils.layer_shard import _read_tensor

    path = str(tmp_path / "w.safetensors")
    save_file({"w": torch.arange(8, dtype=torch.int32).reshape(2, 4)}, path)
    with safe_open(path, framework="pt") as handle:
        with pytest.raises(UnsupportedQuantizedSourceError, match="'I32'"):
            _read_tensor(handle, "w", "bfloat16")


# ==========================================================================
# what must keep working
# ==========================================================================
def test_the_fp8_block_source_still_plans_beside_float_buffers():
    from soup_cli.utils.fp8_block import plan_fp8_block

    weight = "model.layers.0.self_attn.o_proj.weight"
    tensors = {
        weight: ("F8_E4M3", (256, 256)),
        weight + "_scale_inv": ("F32", (2, 2)),
        "model.layers.0.input_layernorm.weight": ("BF16", (256,)),
        "model.layers.0.mlp.gate.e_score_correction_bias": ("F32", (8,)),
        "model.layers.0.self_attn.rotary_emb.inv_freq": ("F32", (32,)),
    }
    plan = plan_fp8_block(DEEPSEEK_FP8, tensors, arch="deepseek_v3")
    assert plan.weights == {weight}


def test_a_qwen4_oq_bundle_with_its_integer_buffers_still_shards(tmp_path):
    """The supported quantised source: uint32 affine weights with '.scales'/'.biases',
    the PLE's int64 hashing tables, and the oQ PLE 'weight_scale' and vision weights the
    sharder drops (so the plan must not see them: a '.weight_scale' would be refused)."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    from soup_cli.utils.layer_shard import layer_shard_path, shard_checkpoint

    weights = tmp_path / "weights"
    weights.mkdir()
    ple = "language_model.model.layers.0.ple.ple_embedding."
    save_file(
        {
            "language_model.model.layers.0.proj.weight": torch.tensor(
                [[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.uint32
            ),
            "language_model.model.layers.0.proj.scales": torch.ones((2, 1)).to(
                torch.bfloat16
            ),
            "language_model.model.layers.0.proj.biases": torch.zeros((2, 1)).to(
                torch.bfloat16
            ),
            ple + "layer_multipliers": torch.tensor([3, 5], dtype=torch.int64),
            ple + "ngram_heads_vocab_sizes": torch.tensor([17, 19], dtype=torch.int64),
            ple + "ngram_heads_offsets": torch.tensor([0, 17], dtype=torch.int64),
            ple + "ngram_embedding.shards.0.weight": torch.tensor(
                [[0, 1, 2, 3]] * 3, dtype=torch.uint32
            ),
            ple + "ngram_embedding.shards.0.scales": torch.ones((3, 1)).to(torch.bfloat16),
            ple + "ngram_embedding.shards.0.biases": torch.zeros((3, 1)).to(torch.bfloat16),
            ple + "ngram_embedding.weight_scale": torch.ones(1),
            "vision_tower.blocks.0.attn.weight_scale": torch.ones(1),
            "vision_tower.embeddings.position_ids": torch.arange(4),
        },
        weights / "model.safetensors",
    )
    (weights / "config.json").write_text(
        json.dumps({"quantization_config": {"bits": 4, "group_size": 32, "mode": "affine"}}),
        encoding="utf-8",
    )
    index = shard_checkpoint(str(weights), str(tmp_path / "s"), dtype="float32", arch="qwen4_exp")
    assert "proj.weight" in index.layer_keys
    assert "ple.ple_embedding.layer_multipliers" in index.layer_keys
    with safe_open(layer_shard_path(str(tmp_path / "s"), 0), framework="pt") as handle:
        assert handle.get_tensor("proj.weight").shape == (2, 32)


@pytest.mark.parametrize(
    "tensors",
    [
        {"model.layers.0.proj.weight": ("U32", (2, 4)),
         "model.layers.0.proj.scales": ("BF16", (2, 1))},
        {"model.layers.0.proj.weight": ("U32", (2, 4)),
         "model.layers.0.proj.biases": ("BF16", (2, 1))},
        {"model.layers.0.proj.packed": ("U32", (2, 4)),
         "model.layers.0.proj.scales": ("BF16", (2, 1)),
         "model.layers.0.proj.biases": ("BF16", (2, 1))},
    ],
    ids=["scales-only", "biases-only", "not-a-weight"],
)
def test_on_qwen4_only_a_complete_affine_weight_is_exempt(tensors):
    """The exemption is the oQ triple ``<m>.weight`` + ``<m>.scales`` + ``<m>.biases``."""
    from soup_cli.utils.fp8_block import UnsupportedQuantizedSourceError, plan_fp8_block

    with pytest.raises(UnsupportedQuantizedSourceError, match="'U32'"):
        plan_fp8_block(None, tensors, arch="qwen4_exp")
    complete = {
        "model.layers.0.proj.weight": ("U32", (2, 4)),
        "model.layers.0.proj.scales": ("BF16", (2, 1)),
        "model.layers.0.proj.biases": ("BF16", (2, 1)),
    }
    assert plan_fp8_block(None, complete, arch="qwen4_exp") is None


def test_the_same_affine_bundle_off_qwen4_is_refused_before_layer_0_is_written(tmp_path):
    """The exemption is qwen4_exp's own reader: elsewhere the uint32 words would be cast.
    In the LAST layer, so a refusal left to the read backstop would come after layer 0's
    file was written; the header plan refuses before any."""
    import torch

    state = _dense_state()
    base = f"model.layers.{LAYERS - 1}.self_attn.o_proj."
    state[base + "weight"] = torch.zeros((HIDDEN, HIDDEN // 8), dtype=torch.uint32)
    state[base + "scales"] = torch.ones(HIDDEN, 1).to(torch.bfloat16)
    state[base + "biases"] = torch.zeros(HIDDEN, 1).to(torch.bfloat16)
    weights = _write(tmp_path / "w", state, model_type="qwen3")
    sharder, estimator = _refusals(tmp_path, weights, arch="qwen3")
    assert f"checkpoint tensor {base + 'weight'!r} is stored as 'U32'" in sharder
    assert estimator == sharder


_TINY_COMMON = dict(
    vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
    num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=64,
    bos_token_id=1, eos_token_id=2, pad_token_id=0,
)
_TINY_LINEAR = dict(
    linear_key_head_dim=4, linear_value_head_dim=4, linear_num_key_heads=2,
    linear_num_value_heads=2, layer_types=["linear_attention", "full_attention"],
)
#: model_type -> (stream family, extra config). Every SUPPORTED_STREAM_ARCHS family, plus
#: the qwen3_5 aliases that stream as qwen3.
_TINY_FAMILIES = {
    "llama": ("llama", {}),
    "qwen2": ("qwen2", {}),
    "qwen3": ("qwen3", {}),
    "mistral": ("mistral", {}),
    "gemma": ("gemma", {}),
    "gemma2": ("gemma2", {}),
    "gemma3_text": ("gemma3_text", {}),
    "phi": ("phi", {}),
    "phi3": ("phi3", {}),
    "deepseek_v3": ("deepseek_v3", dict(
        hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
        first_k_dense_replace=1, num_attention_heads=4, num_key_value_heads=4,
        n_shared_experts=1, n_routed_experts=4, n_group=2, topk_group=1,
        num_experts_per_tok=2, kv_lora_rank=16, q_lora_rank=24, qk_nope_head_dim=16,
        qk_rope_head_dim=8, v_head_dim=16, head_dim=None,
    )),
    "qwen3_5_text": ("qwen3", _TINY_LINEAR),
    "qwen3_5_moe_text": ("qwen3", dict(
        _TINY_LINEAR, num_experts=2, num_experts_per_tok=1, moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
    )),
    "qwen4_exp": ("qwen4_exp", None),
}


def test_every_supported_family_has_a_tiny_checkpoint_below():
    from soup_cli.utils.layer_stream import SUPPORTED_STREAM_ARCHS

    assert set(SUPPORTED_STREAM_ARCHS) <= {family for family, _ in _TINY_FAMILIES.values()}


@pytest.mark.parametrize("model_type", sorted(_TINY_FAMILIES))
def test_every_supported_family_shards_its_own_checkpoint(tmp_path, model_type):
    """The supported families' own save_pretrained output is never refused. Its only
    integer tensors are qwen4_exp's three PLE hashing tables (int64), pinned here."""
    import torch

    transformers = pytest.importorskip("transformers")
    from soup_cli.utils.fp8_block import FLOAT_SOURCE_DTYPES
    from soup_cli.utils.layer_shard import _discover_safetensors, _read_source_tensors

    family, extra = _TINY_FAMILIES[model_type]
    if extra is None:
        from tests.test_issue602_qwen4_ple_streaming import _tiny_qwen4_config

        config = _tiny_qwen4_config()
    else:
        kwargs = {k: v for k, v in dict(_TINY_COMMON, **extra).items() if v is not None}
        config = transformers.AutoConfig.for_model(model_type, **kwargs)
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    weights = str(tmp_path / "model")
    model.save_pretrained(weights, safe_serialization=True)
    del model

    tensors = _read_source_tensors(_discover_safetensors(weights))
    integers = sorted(
        key.rsplit(".", 1)[-1] for key, (dtype, _shape) in tensors.items()
        if dtype not in FLOAT_SOURCE_DTYPES
    )
    expected = (
        ["layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes"]
        if family == "qwen4_exp" else []
    )
    assert integers == expected

    index = _shard(weights, tmp_path / "s", arch=family)
    assert index.n_layers == 2
    if family != "qwen4_exp":  # qwen4_exp has its own estimator
        assert _estimate(weights, arch=family).total_bytes > 0


# ==========================================================================
# hygiene: untrusted names reach a terminal
# ==========================================================================
def test_a_refusal_never_quotes_raw_control_bytes():
    from soup_cli.utils.fp8_block import (
        UnsupportedQuantizedSourceError,
        check_source_quantization,
        refuse_unsupported_tensor,
    )

    hostile = "x\x1b[31m\x07\r"
    with pytest.raises(UnsupportedQuantizedSourceError) as by_config:
        check_source_quantization({"quant_method": hostile}, arch="llama")
    with pytest.raises(UnsupportedQuantizedSourceError) as by_format:
        check_source_quantization(
            {"quant_method": "compressed-tensors", "format": hostile}, arch="llama"
        )
    with pytest.raises(UnsupportedQuantizedSourceError) as by_kind:
        check_source_quantization(
            {"quant_method": "compressed-tensors",
             "config_groups": {"g": {"weights": {"num_bits": 4, "type": hostile}}}},
            arch="llama",
        )
    with pytest.raises(UnsupportedQuantizedSourceError) as by_tensor:
        refuse_unsupported_tensor(f"model.layers.0.{hostile}.weight", hostile)
    for caught in (by_config, by_format, by_kind, by_tensor):
        message = str(caught.value)
        assert not any(ord(char) < 32 for char in message), message


# ==========================================================================
# review round: the refusal comes before anything is created, renamed or reused
# ==========================================================================
def _tree_state(directory):
    """name -> bytes of every file under ``directory`` (a cache is small here)."""
    state = {}
    for base, _dirs, files in os.walk(directory):
        for name in files:
            path = os.path.join(base, name)
            with open(path, "rb") as handle:
                state[os.path.relpath(path, directory)] = handle.read()
    return state


def _int32_llama(directory):
    """A llama source with no quantization_config and one int32 decoder weight."""
    import torch

    state = _dense_state()
    state["model.layers.1.mlp.down_proj.weight"] = torch.zeros((HIDDEN, INTER), dtype=torch.int32)
    return _write(directory, state, model_type="llama")


@pytest.mark.parametrize("source", ["k2-config", "int32-no-config"])
def test_a_refusal_creates_no_cache_folder(tmp_path, source):
    from soup_cli.utils.layer_shard import shard_checkpoint

    if source == "k2-config":
        weights = _write(tmp_path / "w", _k2_state(), quant=K2_THINKING_QUANT)
    else:
        weights = _int32_llama(tmp_path / "w")
    arch = "deepseek_v3" if source == "k2-config" else "llama"
    out = tmp_path / "cache" / "model"
    with pytest.raises(ValueError, match=_TAIL):
        shard_checkpoint(weights, str(out), dtype="bfloat16", arch=arch)
    assert not out.exists()
    assert not (tmp_path / "cache").exists()


def test_a_refusal_creates_no_stripe_folder(tmp_path):
    from soup_cli.utils.layer_shard import shard_checkpoint

    weights = _int32_llama(tmp_path / "w")
    stripe = tmp_path / "second-drive"
    stripe.mkdir()
    with pytest.raises(ValueError, match=_TAIL):
        shard_checkpoint(
            weights, str(tmp_path / "cache" / "model"), dtype="bfloat16", arch="llama",
            stripe_roots=[str(stripe)],
        )
    assert os.listdir(stripe) == []


def test_a_refused_source_leaves_a_valid_prior_cache_untouched(tmp_path):
    """A good cache, then the source replaced by one that is refused: the cache miss used
    to quarantine index.json (renamed to index.json.resharding) before the refusal."""
    import shutil

    from soup_cli.utils.layer_shard import shard_checkpoint

    weights = tmp_path / "w"
    _write(weights, _dense_state(), model_type="llama")
    out = tmp_path / "cache"
    shard_checkpoint(str(weights), str(out), dtype="bfloat16", arch="llama")
    before = _tree_state(out)
    assert "index.json" in before

    shutil.rmtree(weights)
    _int32_llama(weights)
    with pytest.raises(ValueError, match="is stored as 'I32'"):
        shard_checkpoint(str(weights), str(out), dtype="bfloat16", arch="llama")
    assert _tree_state(out) == before
    assert not (out / "index.json.resharding").exists()


def test_a_cache_sharded_before_the_refusal_existed_is_not_reused(tmp_path, monkeypatch):
    """A cache that older code wrote from an int32 source (no quantization_config) is a
    cache HIT for the same source: the refusal must not sit behind the hit's return."""
    import soup_cli.utils.layer_shard as layer_shard
    from soup_cli.utils.fp8_block import _classify_fp8

    weights = _int32_llama(tmp_path / "w")
    out = tmp_path / "cache"
    with monkeypatch.context() as old_code:  # the sharder as it was: FP8 checks only
        old_code.setattr(layer_shard, "plan_fp8_block", _classify_fp8)
        old_code.setattr(layer_shard, "refuse_unsupported_tensor", lambda *_a, **_k: None)
        layer_shard.shard_checkpoint(weights, str(out), dtype="bfloat16", arch="llama")
    before = _tree_state(out)
    with pytest.raises(ValueError, match="is stored as 'I32'"):
        layer_shard.shard_checkpoint(weights, str(out), dtype="bfloat16", arch="llama")
    assert _tree_state(out) == before


def test_the_setup_refuses_a_bad_source_in_the_preflight(tmp_path, monkeypatch):
    """No quantization_config, so the config gate passes; the disk pre-flight reads the
    headers and refuses there: before its cache lookup, before the resolver goes on to
    copy anything, and before the sharder."""
    from tests.test_issue602_qwen4_ple_streaming import _drive_qwen4_streaming_setup

    weights = _write(tmp_path / "k2", _k2_state())
    size = os.path.getsize(os.path.join(weights, "model.safetensors"))
    plan = types.SimpleNamespace(
        weights_dir=weights, source_dir=weights, source_bytes=size, materialized_copy_bytes=0,
        materialize_bytes=0, needs_materialization=False,
        source_files=(("model.safetensors", size, 0),), source_revision=None,
    )

    def _resolve(_base, *, before_materialize=None):
        before_materialize(plan)
        raise AssertionError("the resolver went on past a source that cannot stream")

    model_cfg = types.SimpleNamespace(
        model_type="deepseek_v3", hidden_size=HIDDEN, num_hidden_layers=LAYERS,
        vocab_size=VOCAB, intermediate_size=INTER, quantization_config=None,
    )
    with pytest.raises(ValueError, match="weight_packed' is stored as 'I32'"):
        _drive_qwen4_streaming_setup(
            tmp_path, monkeypatch, resolve_weights=_resolve, arch="deepseek_v3",
            model_cfg=model_cfg,
        )


# ==========================================================================
# review round: the four surviving mutants
# ==========================================================================
@pytest.mark.parametrize(
    "key",
    [
        "model.xposition_ids",
        "model.layers.0.xple.ple_embedding.layer_multipliers",
        "model.layers.0.xple.ple_embedding.ngram_heads_vocab_sizes",
        "model.layers.0.xple.ple_embedding.ngram_heads_offsets",
    ],
)
def test_an_integer_buffer_is_matched_by_its_whole_last_names(key):
    """Each exemption starts at a dot: a name that merely ends in the same letters is
    another tensor, and an int64 one is refused."""
    from soup_cli.utils.fp8_block import UnsupportedQuantizedSourceError, refuse_unsupported_tensor

    with pytest.raises(UnsupportedQuantizedSourceError, match="'I64'"):
        refuse_unsupported_tensor(key, "I64")


@pytest.mark.parametrize("raw", ["F6_E2M3", "F6_E3M2", "F4", "U16", "U32", "U64"])
def test_every_header_only_quantiser_type_is_refused_by_the_preflight(tmp_path, raw):
    """Types the pre-flight's header reader has no spelling for, written by MX/NVFP4 and
    MLX quantisers: refused by name, not sized at a guess."""
    import struct

    from soup_cli.utils.fp8_block import UnsupportedQuantizedSourceError
    from soup_cli.utils.layer_shard import estimate_shard_cache_bytes

    payload = json.dumps(
        {"model.layers.0.mlp.down_proj.weight": {
            "dtype": raw, "shape": [4, 4], "data_offsets": [0, 12]}}
    ).encode()
    path = tmp_path / "model.safetensors"
    path.write_bytes(struct.pack("<Q", len(payload)) + payload + b"\0" * 12)
    with pytest.raises(UnsupportedQuantizedSourceError) as caught:
        estimate_shard_cache_bytes([str(path)], dtype="bfloat16", arch="llama")
    assert f"is stored as {raw!r}" in str(caught.value)


@pytest.mark.parametrize("method", ["FP8", "Fp8", "fp8 ", " fp8"])
def test_quant_method_fp8_is_matched_exactly_as_transformers_matches_it(method):
    """transformers looks the method up as an exact key (``AUTO_QUANTIZATION_CONFIG_MAPPING``
    has 'fp8' only; 'FP8' is an unknown quantisation type to it). So is it here: any other
    spelling is a quantiser layer streaming does not read, refused naming it."""
    from soup_cli.utils.fp8_block import UnsupportedQuantizedSourceError, check_source_quantization

    config = {**DEEPSEEK_FP8, "quant_method": method}
    with pytest.raises(UnsupportedQuantizedSourceError) as caught:
        check_source_quantization(config, arch="deepseek_v3")
    assert f"declares quant_method {method!r}" in str(caught.value)


def test_transformers_matches_quant_method_case_sensitively():
    pytest.importorskip("transformers")
    from transformers.quantizers.auto import AUTO_QUANTIZATION_CONFIG_MAPPING

    assert "fp8" in AUTO_QUANTIZATION_CONFIG_MAPPING
    assert "FP8" not in AUTO_QUANTIZATION_CONFIG_MAPPING
    assert all(name == name.lower() for name in AUTO_QUANTIZATION_CONFIG_MAPPING)


@pytest.mark.parametrize("bits", [4, 6, 16])
def test_a_compressed_tensors_float_group_is_fp8_only_at_eight_bits(bits):
    """NVFP4 declares float weights of 4 bits: that is not an FP8 format."""
    from soup_cli.utils.fp8_block import (
        UnsupportedFP8Error,
        UnsupportedQuantizedSourceError,
        check_source_quantization,
    )

    config = {
        "quant_method": "compressed-tensors",
        "config_groups": {"g": {"weights": {"num_bits": bits, "type": "float"}}},
    }
    with pytest.raises(UnsupportedQuantizedSourceError) as caught:
        check_source_quantization(config, arch="deepseek_v3")
    assert not isinstance(caught.value, UnsupportedFP8Error)
    assert f"({bits}-bit float weights)" in str(caught.value)


# ==========================================================================
# review round: a malformed or nested quantization_config
# ==========================================================================
@pytest.mark.parametrize(
    "value", ["fp8", ["fp8"], 4, True, False, 0.5], ids=["str", "list", "int", "true",
                                                         "false", "float"]
)
def test_a_quantization_config_that_is_not_a_mapping_is_refused_as_malformed(tmp_path, value):
    """It used to be read as 'no quantization_config' and pass the gate."""
    from soup_cli.utils.fp8_block import UnsupportedQuantizedSourceError, check_source_quantization
    from soup_cli.utils.layer_shard import estimate_shard_cache_bytes, shard_checkpoint

    weights = _corrupt_checkpoint_with_config(tmp_path / "w", value)
    with pytest.raises(UnsupportedQuantizedSourceError) as gate:
        check_source_quantization(value, arch="deepseek_v3")
    assert f"quantization_config is {value!r}" in str(gate.value)
    assert "not a mapping" in str(gate.value)
    with pytest.raises(ValueError) as sharder:
        shard_checkpoint(weights, str(tmp_path / "s"), dtype="bfloat16", arch="deepseek_v3")
    with pytest.raises(ValueError) as estimator:
        estimate_shard_cache_bytes(
            [os.path.join(weights, "model.safetensors")], dtype="bfloat16",
            arch="deepseek_v3", quantization_config=value,
        )
    assert str(sharder.value) == str(estimator.value) == str(gate.value)


def test_a_quantization_config_under_text_config_is_read_too(tmp_path):
    from soup_cli.utils.fp8_block import read_checkpoint_quantization_config
    from soup_cli.utils.layer_shard import shard_checkpoint

    weights = tmp_path / "w"
    weights.mkdir()
    (weights / "model.safetensors").write_bytes(b"\xff" * 64)
    (weights / "config.json").write_text(
        json.dumps({"model_type": "deepseek_v3",
                    "text_config": {"quantization_config": K2_THINKING_QUANT}}),
        encoding="utf-8",
    )
    assert read_checkpoint_quantization_config(str(weights)) == K2_THINKING_QUANT
    with pytest.raises(ValueError, match=re.escape(K2_NEEDLE)):
        shard_checkpoint(str(weights), str(tmp_path / "s"), dtype="bfloat16", arch="deepseek_v3")


def test_the_top_level_quantization_config_wins_over_the_nested_one(tmp_path):
    from soup_cli.utils.fp8_block import quantization_config_of

    nested = {"quantization_config": {"quant_method": "gptq"}}
    assert quantization_config_of(
        {"quantization_config": DEEPSEEK_FP8, "text_config": nested}
    ) == DEEPSEEK_FP8
    assert quantization_config_of({"text_config": nested}) == {"quant_method": "gptq"}
    assert quantization_config_of({"text_config": "gptq"}) is None
    assert quantization_config_of({}) is None
    config = types.SimpleNamespace(
        quantization_config=None, text_config=types.SimpleNamespace(quantization_config=None)
    )
    assert quantization_config_of(config) is None
    config.text_config.quantization_config = {"quant_method": "awq"}
    assert quantization_config_of(config) == {"quant_method": "awq"}


def test_the_setup_reads_a_nested_quantization_config(tmp_path, monkeypatch):
    from tests.test_issue602_qwen4_ple_streaming import _drive_qwen4_streaming_setup

    def _must_not_resolve(*_args, **_kwargs):
        raise AssertionError("weights resolved for a base that cannot stream")

    model_cfg = types.SimpleNamespace(
        model_type="deepseek_v3", hidden_size=HIDDEN, num_hidden_layers=LAYERS,
        vocab_size=VOCAB, intermediate_size=INTER, quantization_config=None,
        text_config=None,
    )
    model_cfg.text_config = types.SimpleNamespace(
        model_type="deepseek_v3", hidden_size=HIDDEN, num_hidden_layers=LAYERS,
        vocab_size=VOCAB, intermediate_size=INTER, quantization_config=K2_THINKING_QUANT,
    )
    with pytest.raises(ValueError, match=re.escape(K2_NEEDLE)):
        _drive_qwen4_streaming_setup(
            tmp_path, monkeypatch, resolve_weights=_must_not_resolve, arch="deepseek_v3",
            model_cfg=model_cfg,
        )


# ==========================================================================
# review round: the header reader's error survives pickling and copying
# ==========================================================================
def test_the_unsupported_dtype_error_round_trips_through_pickle_and_copy():
    import copy
    import pickle

    from soup_cli.utils.safetensors_reader import UnsupportedDtypeError

    error = UnsupportedDtypeError("w.safetensors: tensor 'x' ...", tensor="x", dtype="U32")
    for clone in (pickle.loads(pickle.dumps(error)), copy.copy(error), copy.deepcopy(error)):
        assert type(clone) is UnsupportedDtypeError
        assert str(clone) == str(error)
        assert (clone.tensor, clone.dtype) == ("x", "U32")
    bare = UnsupportedDtypeError("message only")
    assert (bare.tensor, bare.dtype) == ("", None)


# ==========================================================================
# pins: the cache format, and a packed tensor inside an FP8 block source
# ==========================================================================
def test_the_shard_format_version_is_not_bumped():
    """A bump invalidates EVERY existing layer cache (the 30 GB demo cache among them), so
    it must be deliberate, never a side effect. This refusal does not need one: the source
    is checked before a cache hit returns, so a stale cache of a bad source is refused."""
    from soup_cli.utils.layer_shard import _SHARD_FORMAT_VERSION

    assert _SHARD_FORMAT_VERSION == 5


def _fp8_block_state():
    """A tiny DeepSeek-V3 FP8 block source: e4m3 linears with float32 block scales."""
    import torch

    state = _dense_state()
    for layer in range(LAYERS):
        key = f"model.layers.{layer}.self_attn.o_proj.weight"
        state[key] = state[key].to(torch.float32).to(torch.float8_e4m3fn)
        state[key + "_scale_inv"] = torch.ones(1, 1)
    return state


@pytest.mark.parametrize(
    ("key", "dtype", "needle"),
    [
        ("model.layers.1.mlp.experts.0.gate_proj.weight_packed", "int32",
         "is stored as 'I32': compressed-tensors pack-quantized"),
        ("model.layers.0.mlp.down_proj.weight", "uint8",
         "is stored as 'U8': uint8, which bitsandbytes 4-bit packs"),
    ],
    ids=["i32-weight-packed", "u8-weight"],
)
def test_a_packed_tensor_beside_an_fp8_block_source_is_refused(tmp_path, key, dtype, needle):
    """An accepted FP8 plan covers its FP8 weights and their scales only: every other tensor
    still meets the tensor rule, in the same words for the sharder and the pre-flight."""
    import torch

    state = _fp8_block_state()
    state[key] = torch.zeros((HIDDEN, INTER // 8), dtype=getattr(torch, dtype))
    weights = _write(tmp_path / "w", state, quant=DEEPSEEK_FP8)
    sharder, estimator = _refusals(tmp_path, weights)
    assert f"checkpoint tensor {key!r} {needle}" in sharder
    assert _TAIL in sharder and _POINTER in sharder
    assert estimator == sharder


def test_the_same_fp8_block_source_without_the_packed_tensor_shards(tmp_path):
    """The control: the FP8 block source alone is planned and sharded, not refused."""
    weights = _write(tmp_path / "w", _fp8_block_state(), quant=DEEPSEEK_FP8)
    assert _shard(weights, tmp_path / "s").n_layers == LAYERS
    assert _estimate(weights).total_bytes > 0
