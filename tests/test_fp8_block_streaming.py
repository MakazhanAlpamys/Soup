"""Layer streaming shards a DeepSeek-V3 FP8 block-quantised checkpoint by dequantising it.

DeepSeek-V3 (and Kimi K2) are published with each decoder linear as ``float8_e4m3fn``
plus a float32 ``<name>.weight_scale_inv`` per 128 x 128 block, and ``config.json`` names
the format. The sharder used to refuse any FP8 tensor; it now dequantises each one at
shard time (``w = w_fp8.float() * scale_inv`` per block, cropped at a ragged edge), casts
to the stream dtype and stores it by the ordinary rules. Every other FP8 spelling stays
refused, naming what is supported.

Gate record: ``benchmarks/gate-fp8-shard.md``. G1 (dequant exactness) and G2 (streamed vs
resident parity, byte-identical caches) are the tests below.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import types

import pytest

#: DeepSeek-V3's own ``quantization_config`` (Kimi K2's is the same).
DEEPSEEK_FP8 = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [128, 128],
}
_SUPPORTED_TAIL = "DeepSeek-V3's block FP8"
_LORA_TARGETS = ["q_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]


def _harness():
    import importlib.util
    import sys
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


def _reference_dequant(weight_fp8, scale_inv, block):
    """Written here, independently of Soup and the harness: one block at a time."""
    import torch

    rows, cols = weight_fp8.shape
    block_rows, block_cols = block
    out = torch.full((rows, cols), float("nan"), dtype=torch.float32)
    for i in range((rows + block_rows - 1) // block_rows):
        for j in range((cols + block_cols - 1) // block_cols):
            rs = slice(i * block_rows, min((i + 1) * block_rows, rows))
            cs = slice(j * block_cols, min((j + 1) * block_cols, cols))
            out[rs, cs] = weight_fp8[rs, cs].to(torch.float32) * scale_inv[i, j]
    assert not torch.isnan(out).any()  # every element was written exactly once
    return out


def _blockwise_weight(shape, block, seed, *, zero_block=True):
    """bf16 normals with a different magnitude per block (and one all-zero block)."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    rows, cols = shape
    weight = torch.randn(rows, cols, generator=generator)
    grid_rows = (rows + block[0] - 1) // block[0]
    grid_cols = (cols + block[1] - 1) // block[1]
    magnitudes = torch.exp(torch.randn(grid_rows, grid_cols, generator=generator) * 2.0)
    expanded = magnitudes.repeat_interleave(block[0], 0)[:rows]
    weight = weight * expanded.repeat_interleave(block[1], 1)[:, :cols]
    if zero_block and grid_rows * grid_cols > 1:
        weight[: block[0], : block[1]] = 0.0
    return weight.to(torch.bfloat16)


# ==========================================================================
# G1 — dequantisation exactness
# ==========================================================================
G1_CASES = [
    ((576, 2048), (128, 128)),
    ((129, 257), (128, 128)),
    ((127, 1), (128, 128)),
    ((1, 1), (128, 128)),
    ((256, 384), (128, 128)),
    ((2048, 1408), (128, 128)),
    ((24, 64), (16, 32)),
    ((100, 70), (16, 32)),
    ((64, 96), (16, 32)),
]
G1_IDS = [f"{s[0]}x{s[1]}-b{b[0]}x{b[1]}" for s, b in G1_CASES]
_TILEABLE = [
    (shape, block) for shape, block in G1_CASES
    if shape[0] % block[0] == 0 and shape[1] % block[1] == 0
]


@pytest.mark.parametrize(("shape", "block"), G1_CASES, ids=G1_IDS)
def test_dequant_equals_the_reference_block_loop(shape, block):
    import torch

    from soup_cli.utils.fp8_block import dequantize_fp8_block

    weight, scale = _harness().quantize_fp8_block(_blockwise_weight(shape, block, 7), block)
    assert weight.dtype == torch.float8_e4m3fn
    assert tuple(scale.shape) == (-(-shape[0] // block[0]), -(-shape[1] // block[1]))
    got = dequantize_fp8_block(weight, scale, block)
    want = _reference_dequant(weight, scale, block)
    assert got.dtype == torch.float32
    assert torch.equal(got, want), (got - want).abs().max().item()
    for dtype in (torch.bfloat16, torch.float16):
        assert torch.equal(got.to(dtype), want.to(dtype))


def test_every_finite_e4m3_code_point_dequantises_exactly():
    """+-448, every subnormal and both zeros, under a different scale per 8x8 block."""
    import torch

    from soup_cli.utils.fp8_block import dequantize_fp8_block

    codes = torch.arange(256, dtype=torch.int32).to(torch.uint8)
    values = codes.view(torch.float8_e4m3fn)
    finite = values[torch.isfinite(values.to(torch.float32))]
    assert finite.numel() == 254  # 0x7F and 0xFF are the e4m3fn NaNs
    weight = torch.cat((finite, finite[:2])).reshape(16, 16)
    scale = torch.rand(2, 2, generator=torch.Generator().manual_seed(3)) * 5.0 + 1e-3
    got = dequantize_fp8_block(weight, scale, (8, 8))
    assert torch.equal(got, _reference_dequant(weight, scale, (8, 8)))
    assert float(got.abs().max()) > 0.0


@pytest.mark.parametrize(
    ("shape", "block"), _TILEABLE, ids=[f"{s[0]}x{s[1]}" for s, _ in _TILEABLE]
)
def test_transformers_dequant_agrees_on_tileable_shapes(shape, block):
    """The second oracle: transformers' own CPU-callable FP8 block dequant."""
    import torch

    finegrained = pytest.importorskip("transformers.integrations.finegrained_fp8")
    from soup_cli.utils.fp8_block import dequantize_fp8_block

    weight, scale = _harness().quantize_fp8_block(_blockwise_weight(shape, block, 11), block)
    theirs = finegrained.Fp8Dequantize(None)._dequantize_one(
        weight, scale, output_dtype=torch.float32
    )
    assert torch.equal(dequantize_fp8_block(weight, scale, block), theirs)


@pytest.mark.parametrize(
    ("shape", "block"), _TILEABLE, ids=[f"{s[0]}x{s[1]}" for s, _ in _TILEABLE]
)
def test_the_harness_quantiser_writes_what_transformers_writes(shape, block):
    """The converter used by G2 and G3 produces transformers' own FP8 bytes and scales."""
    import torch

    transformers = pytest.importorskip("transformers")
    finegrained = pytest.importorskip("transformers.integrations.finegrained_fp8")

    source = _blockwise_weight(shape, block, 13)
    config = transformers.FineGrainedFP8Config(weight_block_size=block)
    quantizer = types.SimpleNamespace(quantization_config=config)
    theirs = finegrained.Fp8Quantize(quantizer).convert({"x.weight": source})
    weight, scale = _harness().quantize_fp8_block(source, block)
    assert torch.equal(weight.view(torch.uint8), theirs["x.weight"].view(torch.uint8))
    assert torch.equal(scale, theirs["x.weight_scale_inv"])


def test_dequant_refuses_a_non_finite_scale():
    import torch

    from soup_cli.utils.fp8_block import UnsupportedFP8Error, dequantize_fp8_block

    weight = torch.zeros(4, 4).to(torch.float8_e4m3fn)
    for bad in (float("inf"), float("nan")):
        scale = torch.ones(1, 1)
        scale[0, 0] = bad
        with pytest.raises(UnsupportedFP8Error, match="non-finite"):
            dequantize_fp8_block(weight, scale, (4, 4), name="w")


@pytest.mark.parametrize(
    ("weight_dtype", "scale_shape", "scale_dtype", "needle"),
    [
        ("float8_e5m2", (1, 1), "float32", "not a 2-D float8_e4m3fn"),
        ("bfloat16", (1, 1), "float32", "not a 2-D float8_e4m3fn"),
        ("float8_e4m3fn", (2, 1), "float32", "a float32 [1, 1] is needed"),
        ("float8_e4m3fn", (1, 1), "bfloat16", "a float32 [1, 1] is needed"),
    ],
    ids=["e5m2", "bf16", "grid", "scale-dtype"],
)
def test_dequant_refuses_a_tensor_outside_the_format(
    weight_dtype, scale_shape, scale_dtype, needle
):
    import torch

    from soup_cli.utils.fp8_block import UnsupportedFP8Error, dequantize_fp8_block

    weight = torch.zeros(4, 4).to(getattr(torch, weight_dtype))
    scale = torch.ones(scale_shape, dtype=getattr(torch, scale_dtype))
    with pytest.raises(UnsupportedFP8Error, match=re.escape(needle)):
        dequantize_fp8_block(weight, scale, (4, 4))


# ==========================================================================
# the config: the published spellings are accepted, every other is refused by name
# ==========================================================================
def test_the_published_config_spellings_are_accepted():
    transformers = pytest.importorskip("transformers")
    from soup_cli.utils.fp8_block import fp8_block_size

    assert fp8_block_size(DEEPSEEK_FP8, arch="deepseek_v3") == (128, 128)
    # transformers' own to_dict: an enum quant_method, a tuple block, scale_fmt "float"
    exported = transformers.FineGrainedFP8Config().to_dict()
    assert fp8_block_size(exported, arch="deepseek_v3") == (128, 128)
    assert fp8_block_size(transformers.FineGrainedFP8Config(), arch="deepseek_v3") == (128, 128)
    minimal = {"quant_method": "fp8", "weight_block_size": [64, 32]}
    assert fp8_block_size(minimal, arch="deepseek_v3") == (64, 32)


@pytest.mark.parametrize(
    "config", [None, {}, {"quant_method": "bitsandbytes", "load_in_4bit": True}, "fp8"],
    ids=["none", "empty", "bnb", "not-a-mapping"],
)
def test_a_config_without_fp8_is_not_an_fp8_source(config):
    from soup_cli.utils.fp8_block import fp8_block_size

    assert fp8_block_size(config, arch="llama") is None


_BAD_CONFIGS = {
    "per-tensor-missing": ({"quant_method": "fp8"}, "per-tensor FP8 scales"),
    "per-tensor-null": (
        {"quant_method": "fp8", "weight_block_size": None},
        "per-tensor FP8 scales",
    ),
    "block-one-int": ({**DEEPSEEK_FP8, "weight_block_size": [128]}, "two positive ints"),
    "block-zero": ({**DEEPSEEK_FP8, "weight_block_size": [0, 128]}, "two positive ints"),
    "block-bool": ({**DEEPSEEK_FP8, "weight_block_size": [True, 128]}, "two positive ints"),
    "block-float": ({**DEEPSEEK_FP8, "weight_block_size": [128.0, 128]}, "two positive ints"),
    "block-string": ({**DEEPSEEK_FP8, "weight_block_size": "128"}, "two positive ints"),
    "e5m2": ({**DEEPSEEK_FP8, "fmt": "e5m2"}, "e5m2 is not"),
    "ue8m0": ({**DEEPSEEK_FP8, "scale_fmt": "ue8m0"}, "ue8m0 power-of-two scales"),
    "static": ({**DEEPSEEK_FP8, "activation_scheme": "static"}, "static activation scales"),
}


@pytest.mark.parametrize("spelling", sorted(_BAD_CONFIGS))
def test_an_unsupported_fp8_config_is_refused_naming_the_format(spelling):
    from soup_cli.utils.fp8_block import UnsupportedFP8Error, fp8_block_size

    config, needle = _BAD_CONFIGS[spelling]
    with pytest.raises(UnsupportedFP8Error) as caught:
        fp8_block_size(config, arch="deepseek_v3")
    assert needle in str(caught.value)
    assert _SUPPORTED_TAIL in str(caught.value)


@pytest.mark.parametrize(
    ("block", "accepted"),
    [
        ([2**20, 2**20], True),
        ([2**20 + 1, 128], False),
        ([128, 2**20 + 1], False),
        ([2**63, 128], False),
        ([2**63, 2**63], False),
        ([2**80, 2**80], False),
    ],
    ids=["at-cap", "rows-over", "cols-over", "rows-2^63", "both-2^63", "both-2^80"],
)
def test_weight_block_size_is_refused_past_its_range(block, accepted):
    from soup_cli.utils.fp8_block import UnsupportedFP8Error, fp8_block_size

    config = {**DEEPSEEK_FP8, "weight_block_size": block}
    if accepted:
        assert fp8_block_size(config, arch="deepseek_v3") == tuple(block)
        return
    with pytest.raises(UnsupportedFP8Error) as caught:
        fp8_block_size(config, arch="deepseek_v3")
    assert "two positive ints of at most 1048576" in str(caught.value)
    assert _SUPPORTED_TAIL in str(caught.value)


def test_an_out_of_range_block_is_refused_alike_by_plan_estimator_sharder_and_setup(
    tmp_path, monkeypatch
):
    """One check: every entry point refuses it in the same words, before a header is read."""
    from soup_cli.utils.fp8_block import UnsupportedFP8Error, fp8_block_size, plan_fp8_block
    from soup_cli.utils.layer_shard import estimate_shard_cache_bytes, shard_checkpoint
    from tests.test_issue602_qwen4_ple_streaming import _drive_qwen4_streaming_setup

    config = {**DEEPSEEK_FP8, "weight_block_size": [2**63, 2**63]}
    with pytest.raises(UnsupportedFP8Error) as direct:
        fp8_block_size(config, arch="deepseek_v3")
    with pytest.raises(UnsupportedFP8Error) as planned:
        plan_fp8_block(config, _header_tensors(), arch="deepseek_v3")
    with pytest.raises(UnsupportedFP8Error) as estimated:
        estimate_shard_cache_bytes(
            [str(tmp_path / "gone.safetensors")], dtype="bfloat16", arch="deepseek_v3",
            quantization_config=config,
        )
    weights = _corrupt_checkpoint_with_config(tmp_path / "w", config)
    with pytest.raises(UnsupportedFP8Error) as sharded:
        shard_checkpoint(weights, str(tmp_path / "s"), dtype="bfloat16", arch="deepseek_v3")

    def _must_not_resolve(*_args, **_kwargs):
        raise AssertionError("weights resolved for a base that cannot stream")

    with pytest.raises(UnsupportedFP8Error) as setup:
        _drive_qwen4_streaming_setup(
            tmp_path, monkeypatch, resolve_weights=_must_not_resolve, arch="deepseek_v3",
            model_cfg=_fp8_setup_cfg(config),
        )
    messages = {str(caught.value) for caught in (direct, planned, estimated, sharded, setup)}
    assert len(messages) == 1


@pytest.mark.parametrize("arch", ["llama", "qwen3", "qwen4_exp", ""])
def test_an_fp8_base_on_another_architecture_is_refused(arch):
    from soup_cli.utils.fp8_block import UnsupportedFP8Error, fp8_block_size

    with pytest.raises(UnsupportedFP8Error, match="admitted: deepseek_v3"):
        fp8_block_size(DEEPSEEK_FP8, arch=arch)


# ==========================================================================
# the tensors: every FP8 tensor needs its scale, every scale its FP8 tensor
# ==========================================================================
def _header_tensors(**overrides):
    """key -> (safetensors dtype, shape) for one FP8 block layer (block 128 x 128)."""
    tensors = {
        "model.embed_tokens.weight": ("BF16", (8, 300)),
        "model.layers.0.input_layernorm.weight": ("BF16", (300,)),
        "model.layers.0.self_attn.o_proj.weight": ("F8_E4M3", (300, 300)),
        "model.layers.0.self_attn.o_proj.weight_scale_inv": ("F32", (3, 3)),
        "model.layers.0.mlp.gate.weight": ("BF16", (4, 300)),
        "model.layers.0.mlp.experts.0.down_proj.weight": ("F8_E4M3", (300, 129)),
        "model.layers.0.mlp.experts.0.down_proj.weight_scale_inv": ("F32", (3, 2)),
    }
    for key, value in overrides.items():
        if value is None:
            tensors.pop(key)
        else:
            tensors[key] = value
    return tensors


_O = "model.layers.0.self_attn.o_proj.weight"
_BAD_TENSORS = {
    "fp8-without-scale": ({_O + "_scale_inv": None}, "has no block scale"),
    "scale-without-tensor": ({_O: None}, "has no tensor"),
    "scale-beside-bf16": ({_O: ("BF16", (300, 300))}, "not an FP8 tensor"),
    "misshapen-scale": ({_O + "_scale_inv": ("F32", (3, 2))}, "needs [3, 3]"),
    "transposed-grid": (
        {"model.layers.0.mlp.experts.0.down_proj.weight_scale_inv": ("F32", (2, 3))},
        "needs [3, 2]",
    ),
    "per-tensor-scale": ({_O + "_scale_inv": ("F32", ())}, "a per-tensor scale is not"),
    "per-tensor-scale-1": ({_O + "_scale_inv": ("F32", (1,))}, "a per-tensor scale is not"),
    "bf16-scale": ({_O + "_scale_inv": ("BF16", (3, 3))}, "is 'BF16', not F32"),
    "fp8-scale": ({_O + "_scale_inv": ("F8_E4M3", (3, 3))}, "is 'F8_E4M3', not F32"),
    "e5m2-weight": ({_O: ("F8_E5M2", (300, 300))}, "is 'F8_E5M2', not F8_E4M3"),
    "fnuz-weight": ({_O: ("F8_E4M3FNUZ", (300, 300))}, "not F8_E4M3"),
    "fp8-vector": (
        {_O: ("F8_E4M3", (300,)), _O + "_scale_inv": ("F32", (3,))},
        "covers 2-D weights only",
    ),
    "fp8-3d": (
        {_O: ("F8_E4M3", (2, 300, 300)), _O + "_scale_inv": ("F32", (2, 3, 3))},
        "covers 2-D weights only",
    ),
}


@pytest.mark.parametrize("spelling", sorted(_BAD_TENSORS))
def test_an_fp8_tensor_layout_outside_the_format_is_refused_by_name(spelling):
    from soup_cli.utils.fp8_block import UnsupportedFP8Error, plan_fp8_block

    overrides, needle = _BAD_TENSORS[spelling]
    with pytest.raises(UnsupportedFP8Error) as caught:
        plan_fp8_block(DEEPSEEK_FP8, _header_tensors(**overrides), arch="deepseek_v3")
    assert needle in str(caught.value)
    assert _SUPPORTED_TAIL in str(caught.value)


@pytest.mark.parametrize(
    ("config", "needle"),
    [
        (None, "declares no FP8 quantization_config"),
        ({"quant_method": "compressed-tensors", "format": "float-quantized"},
         "'compressed-tensors', whose scales are not 'weight_scale_inv' block scales"),
        ({"quant_method": "bitsandbytes"}, "declares quant_method 'bitsandbytes'"),
    ],
    ids=["no-config", "compressed-tensors", "other-method"],
)
def test_fp8_tensors_without_the_fp8_block_config_are_refused(config, needle):
    from soup_cli.utils.fp8_block import UnsupportedFP8Error, plan_fp8_block

    with pytest.raises(UnsupportedFP8Error) as caught:
        plan_fp8_block(config, _header_tensors(), arch="deepseek_v3")
    assert needle in str(caught.value)


def test_an_enum_quant_method_is_named_by_its_value():
    """transformers' config objects carry ``quant_method`` as an enum member."""
    transformers = pytest.importorskip("transformers")
    from soup_cli.utils.fp8_block import UnsupportedFP8Error, plan_fp8_block

    bnb = transformers.BitsAndBytesConfig(load_in_4bit=True)
    with pytest.raises(UnsupportedFP8Error) as caught:
        plan_fp8_block(bnb, _header_tensors(), arch="deepseek_v3")
    assert "declares quant_method 'bitsandbytes'" in str(caught.value)


def test_a_compressed_tensors_checkpoint_with_weight_scale_is_refused():
    """compressed-tensors FP8 spells its scale ``weight_scale``, not ``weight_scale_inv``."""
    from soup_cli.utils.fp8_block import UnsupportedFP8Error, plan_fp8_block

    tensors = _header_tensors(**{_O + "_scale_inv": None})
    tensors[_O.removesuffix(".weight") + ".weight_scale"] = ("F32", (300, 1))
    with pytest.raises(UnsupportedFP8Error, match="whose scales are not 'weight_scale_inv'"):
        plan_fp8_block({"quant_method": "compressed-tensors"}, tensors, arch="deepseek_v3")


def test_a_complete_fp8_block_layout_is_planned():
    from soup_cli.utils.fp8_block import plan_fp8_block

    plan = plan_fp8_block(DEEPSEEK_FP8, _header_tensors(), arch="deepseek_v3")
    assert plan.block == (128, 128)
    assert plan.weights == {_O, "model.layers.0.mlp.experts.0.down_proj.weight"}
    assert plan.scales == {key + "_scale_inv" for key in plan.weights}


def test_no_fp8_tensor_means_no_plan_even_under_an_fp8_config():
    from soup_cli.utils.fp8_block import plan_fp8_block

    plain = {"model.layers.0.self_attn.o_proj.weight": ("BF16", (4, 4))}
    assert plan_fp8_block(DEEPSEEK_FP8, plain, arch="deepseek_v3") is None
    assert plan_fp8_block(None, plain, arch="llama") is None


def test_a_refusal_never_quotes_raw_control_bytes():
    """Header keys are untrusted and the message reaches a terminal."""
    from soup_cli.utils.fp8_block import UnsupportedFP8Error, plan_fp8_block

    tensors = {"model.layers.0.\x1b[31mx.weight": ("F8_E4M3", (4, 4))}
    with pytest.raises(UnsupportedFP8Error) as caught:
        plan_fp8_block(DEEPSEEK_FP8, tensors, arch="deepseek_v3")
    assert "\x1b" not in str(caught.value)


# ==========================================================================
# the sharder on synthetic checkpoints
# ==========================================================================
HIDDEN, INTER, VOCAB, EXPERTS, LAYERS = 6, 5, 8, 2, 2
BLOCK = (4, 4)  # every matrix below is ragged against it


def _synthetic_state(seed=1, *, fp8_embeddings=False):
    """A DeepSeek-V3-shaped bf16 state, the linears as the harness converts them."""
    import torch

    generator = torch.Generator().manual_seed(seed)

    def normal(*shape):
        return torch.randn(*shape, generator=generator).to(torch.bfloat16)

    state = {
        "model.embed_tokens.weight": normal(VOCAB, HIDDEN),
        "model.norm.weight": normal(HIDDEN),
        "lm_head.weight": normal(VOCAB, HIDDEN),
    }
    linears = []
    for layer in range(LAYERS):
        prefix = f"model.layers.{layer}."
        state[prefix + "input_layernorm.weight"] = normal(HIDDEN)
        linears.append(prefix + "self_attn.o_proj.weight")
        state[linears[-1]] = normal(HIDDEN, HIDDEN)
        state[prefix + "mlp.gate.weight"] = normal(EXPERTS, HIDDEN)
        state[prefix + "mlp.gate.e_score_correction_bias"] = torch.randn(
            EXPERTS, generator=generator
        )
        for expert in range(EXPERTS):
            for name, shape in (
                ("gate_proj", (INTER, HIDDEN)),
                ("up_proj", (INTER, HIDDEN)),
                ("down_proj", (HIDDEN, INTER)),
            ):
                linears.append(prefix + f"mlp.experts.{expert}.{name}.weight")
                state[linears[-1]] = normal(*shape)
    if fp8_embeddings:  # transformers' modules_to_convert: FP8 embedding tables
        linears += ["model.embed_tokens.weight", "lm_head.weight"]
    return state, linears


def _to_fp8(state, linears, block=BLOCK):
    quantize = _harness().quantize_fp8_block
    out = dict(state)
    for key in linears:
        out[key], out[key + "_scale_inv"] = quantize(state[key], block)
    return out


def _dequantised(fp8_state, block=BLOCK):
    """The reference checkpoint: every FP8 weight replaced by its fp32 reference dequant."""
    out = {}
    for key, value in fp8_state.items():
        if key.endswith("_scale_inv"):
            continue
        scale = fp8_state.get(key + "_scale_inv")
        out[key] = value if scale is None else _reference_dequant(value, scale, block)
    return out


def _write(directory, state, *, quant=DEEPSEEK_FP8, block=BLOCK, layers=LAYERS):
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=True)
    save_file(state, str(directory / "model.safetensors"))
    config = {"model_type": "deepseek_v3", "num_hidden_layers": layers}
    if quant is not None:
        config["quantization_config"] = {**quant, "weight_block_size": list(block)}
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(directory)


def _synthetic_pair(tmp_path, **kwargs):
    """(FP8 checkpoint dir, reference-dequantised checkpoint dir)."""
    state, linears = _synthetic_state(**kwargs)
    fp8_state = _to_fp8(state, linears)
    fp8 = _write(tmp_path / "fp8", fp8_state)
    ref = _write(tmp_path / "ref", _dequantised(fp8_state), quant=None)
    return fp8, ref


def _cache_files(directory):
    return sorted(
        name for name in os.listdir(directory)
        if name.endswith(".safetensors") and not name.startswith(".soup.")
    )


def _assert_same_cache(left, right, left_index, right_index):
    import torch
    from safetensors.torch import load_file

    assert _cache_files(left) == _cache_files(right)
    for name in _cache_files(left):
        mine, theirs = load_file(os.path.join(left, name)), load_file(os.path.join(right, name))
        assert sorted(mine) == sorted(theirs), name
        assert not any(key.endswith("_scale_inv") for key in mine), name
        for key in mine:
            assert mine[key].dtype == theirs[key].dtype, (name, key)
            assert torch.equal(mine[key], theirs[key]), (name, key)
    ignored = {"source_fingerprint", "source_files"}
    for field_name in left_index.__dataclass_fields__:
        if field_name not in ignored:
            assert getattr(left_index, field_name) == getattr(right_index, field_name), field_name


def _shard(weights, out, **kwargs):
    from soup_cli.utils.layer_shard import shard_checkpoint

    kwargs.setdefault("dtype", "float32")
    kwargs.setdefault("arch", "deepseek_v3")
    kwargs.setdefault("num_hidden_layers", LAYERS)
    return shard_checkpoint(weights, str(out), **kwargs)


@pytest.mark.parametrize("layout", ["", "per_expert"], ids=["fused", "per_expert"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_the_fp8_source_shards_to_the_cache_of_its_dequantised_weights(tmp_path, layout, dtype):
    fp8, ref = _synthetic_pair(tmp_path)
    mine = _shard(fp8, tmp_path / "a", dtype=dtype, expert_layout=layout)
    theirs = _shard(ref, tmp_path / "b", dtype=dtype, expert_layout=layout)
    _assert_same_cache(str(tmp_path / "a"), str(tmp_path / "b"), mine, theirs)
    assert mine.total_params == theirs.total_params


def test_fp8_embeddings_and_head_are_dequantised_into_their_shards(tmp_path):
    """transformers' modules_to_convert can make the embedding tables FP8 too."""
    fp8, ref = _synthetic_pair(tmp_path, fp8_embeddings=True)
    mine = _shard(fp8, tmp_path / "a")
    theirs = _shard(ref, tmp_path / "b")
    assert mine.large_keys == ("lm_head.weight", "model.embed_tokens.weight")
    _assert_same_cache(str(tmp_path / "a"), str(tmp_path / "b"), mine, theirs)


def test_the_router_and_its_bias_keep_their_stored_dtypes(tmp_path):
    import torch
    from safetensors.torch import load_file

    from soup_cli.utils.layer_shard import layer_paths

    fp8, _ref = _synthetic_pair(tmp_path)
    index = _shard(fp8, tmp_path / "s", dtype="bfloat16")
    tensors = load_file(layer_paths(str(tmp_path / "s"), index)[1])
    assert tensors["mlp.gate.e_score_correction_bias"].dtype == torch.float32
    assert tensors["mlp.gate.weight"].dtype == torch.bfloat16
    assert tensors["mlp.experts.gate_up_proj"].dtype == torch.bfloat16
    assert not any(key.endswith("_scale_inv") for key in tensors)


def test_the_plain_read_backstop_still_refuses_an_fp8_tensor(tmp_path):
    """A cast that bypassed the plan would keep the raw e4m3 values: never allowed."""
    from safetensors import safe_open

    from soup_cli.utils.fp8_block import UnsupportedFP8Error
    from soup_cli.utils.layer_shard import _read_tensor

    fp8, _ref = _synthetic_pair(tmp_path)
    with safe_open(os.path.join(fp8, "model.safetensors"), framework="pt") as handle:
        for key in (
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.self_attn.o_proj.weight_scale_inv",
        ):
            with pytest.raises(UnsupportedFP8Error, match="cast without its scales"):
                _read_tensor(handle, key, "bfloat16")


def _mutated_checkpoint(tmp_path, spelling):
    """An FP8 checkpoint in one refused spelling (key, dtype, config or architecture)."""
    import torch

    state, linears = _synthetic_state()
    fp8_state = _to_fp8(state, linears)
    o_proj = "model.layers.0.self_attn.o_proj.weight"
    quant, arch = DEEPSEEK_FP8, "deepseek_v3"
    if spelling == "fp8-without-scale":
        del fp8_state[o_proj + "_scale_inv"]
    elif spelling == "scale-without-tensor":
        fp8_state[o_proj] = state[o_proj]
    elif spelling == "misshapen-scale":
        fp8_state[o_proj + "_scale_inv"] = torch.ones(1, 2)
    elif spelling == "e5m2-weight":
        fp8_state[o_proj] = fp8_state[o_proj].to(torch.float32).to(torch.float8_e5m2)
    elif spelling == "per-tensor-config":
        quant = {"quant_method": "fp8", "weight_block_size": None}
    elif spelling == "compressed-tensors":
        # compressed-tensors' own FP8 spelling: a bare quant_method no longer says FP8,
        # and is refused as a quantised source of unknown format.
        quant = {"quant_method": "compressed-tensors", "format": "float-quantized"}
    elif spelling == "no-config":
        quant = None
    elif spelling == "llama":
        arch = "llama"
    else:  # pragma: no cover — a typo in the parametrize list
        raise AssertionError(spelling)
    if quant is not None and quant.get("weight_block_size", 1) is None:
        weights = _write(tmp_path / "w", fp8_state, quant=None)
        config = json.loads((tmp_path / "w" / "config.json").read_text(encoding="utf-8"))
        config["quantization_config"] = quant
        (tmp_path / "w" / "config.json").write_text(json.dumps(config), encoding="utf-8")
    else:
        weights = _write(tmp_path / "w", fp8_state, quant=quant)
    return weights, arch


_SHARDER_SPELLINGS = [
    "fp8-without-scale", "scale-without-tensor", "misshapen-scale", "e5m2-weight",
    "per-tensor-config", "compressed-tensors", "no-config", "llama",
]


@pytest.mark.parametrize("spelling", _SHARDER_SPELLINGS)
def test_the_sharder_refuses_before_anything_is_written(tmp_path, spelling):
    from soup_cli.utils.fp8_block import UnsupportedFP8Error
    from soup_cli.utils.layer_shard import shard_checkpoint

    weights, arch = _mutated_checkpoint(tmp_path, spelling)
    out = tmp_path / "s"
    with pytest.raises(UnsupportedFP8Error, match="Dequantise"):
        shard_checkpoint(weights, str(out), dtype="bfloat16", arch=arch, num_hidden_layers=2)
    assert not out.exists() or not os.listdir(out)


@pytest.mark.parametrize("spelling", _SHARDER_SPELLINGS)
def test_the_disk_preflight_refuses_in_the_sharders_words(tmp_path, spelling):
    from soup_cli.utils.fp8_block import read_checkpoint_quantization_config
    from soup_cli.utils.layer_shard import (
        _discover_safetensors,
        estimate_shard_cache_bytes,
        shard_checkpoint,
    )

    weights, arch = _mutated_checkpoint(tmp_path, spelling)
    with pytest.raises(ValueError) as sharder:
        shard_checkpoint(
            weights, str(tmp_path / "s"), dtype="bfloat16", arch=arch, num_hidden_layers=2
        )
    with pytest.raises(ValueError) as estimator:
        estimate_shard_cache_bytes(
            _discover_safetensors(weights), dtype="bfloat16", arch=arch, num_hidden_layers=2,
            quantization_config=read_checkpoint_quantization_config(weights),
        )
    assert str(estimator.value) == str(sharder.value)


def _corrupt_checkpoint_with_config(directory, quantization_config):
    """A config that names a refused FP8 spelling beside weights no reader could open."""
    directory.mkdir(parents=True)
    (directory / "model.safetensors").write_bytes(b"\xff" * 64)
    config = {"model_type": "deepseek_v3", "quantization_config": quantization_config}
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(directory)


def test_the_sharder_refuses_an_fp8_config_spelling_before_reading_a_header(tmp_path):
    from soup_cli.utils.fp8_block import UnsupportedFP8Error
    from soup_cli.utils.layer_shard import shard_checkpoint

    weights = _corrupt_checkpoint_with_config(tmp_path / "w", {**DEEPSEEK_FP8, "fmt": "e5m2"})
    with pytest.raises(UnsupportedFP8Error, match="e5m2 is not"):
        shard_checkpoint(weights, str(tmp_path / "s"), dtype="bfloat16", arch="deepseek_v3")


def test_the_estimate_refuses_an_fp8_config_spelling_before_reading_a_header(tmp_path):
    from soup_cli.utils.fp8_block import UnsupportedFP8Error
    from soup_cli.utils.layer_shard import estimate_shard_cache_bytes

    with pytest.raises(UnsupportedFP8Error, match="per-tensor FP8 scales"):
        estimate_shard_cache_bytes(
            [str(tmp_path / "gone.safetensors")], dtype="bfloat16", arch="deepseek_v3",
            quantization_config={"quant_method": "fp8"},
        )


# ==========================================================================
# config.json, read as untrusted input
# ==========================================================================
@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (None, None),
        (b"", None),
        (b"not json", None),
        (b"[1, 2]", None),
        # Returned as found since the review round of the packed-quant refusal: the
        # gate refuses a declaration that is not a mapping as malformed.
        (b'{"quantization_config": "fp8"}', "fp8"),
        (b'{"model_type": "deepseek_v3"}', None),
        (b'{"quantization_config": {"quant_method": "fp8"}}', {"quant_method": "fp8"}),
    ],
    ids=["missing", "empty", "not-json", "not-object", "not-mapping", "absent", "fp8"],
)
def test_the_checkpoint_config_is_read_bounded(tmp_path, content, expected):
    from soup_cli.utils.fp8_block import read_checkpoint_quantization_config

    if content is not None:
        (tmp_path / "config.json").write_bytes(content)
    assert read_checkpoint_quantization_config(str(tmp_path)) == expected


def test_a_config_too_deep_to_parse_is_not_read(tmp_path, monkeypatch):
    """``json`` raises RecursionError past the interpreter's nesting limit, and where that
    limit falls differs between builds (5,000 levels raise on Windows CPython 3.12.10 and
    parse on Linux 3.12.15), so the error is raised here directly rather than by depth."""
    import soup_cli.utils.fp8_block as fp8_block

    (tmp_path / "config.json").write_bytes(b'{"quantization_config": {"quant_method": "fp8"}}')

    def _too_deep(_handle):
        raise RecursionError("maximum recursion depth exceeded while decoding a JSON array")

    monkeypatch.setattr(fp8_block, "json", types.SimpleNamespace(load=_too_deep))
    assert fp8_block.read_checkpoint_quantization_config(str(tmp_path)) is None


def test_an_oversized_config_is_not_read(tmp_path, monkeypatch):
    import soup_cli.utils.fp8_block as fp8_block

    payload = b'{"quantization_config": {"quant_method": "fp8"}}'
    (tmp_path / "config.json").write_bytes(payload)
    monkeypatch.setattr(fp8_block, "_CONFIG_LIMIT", len(payload) - 1)
    assert fp8_block.read_checkpoint_quantization_config(str(tmp_path)) is None
    monkeypatch.setattr(fp8_block, "_CONFIG_LIMIT", len(payload))
    assert fp8_block.read_checkpoint_quantization_config(str(tmp_path)) == {
        "quant_method": "fp8"
    }


@pytest.mark.requires_symlink
def test_a_symlinked_config_is_not_read(tmp_path):
    from soup_cli.utils.fp8_block import read_checkpoint_quantization_config

    real = tmp_path / "elsewhere.json"
    real.write_text(json.dumps({"quantization_config": DEEPSEEK_FP8}), encoding="utf-8")
    os.symlink(str(real), str(tmp_path / "config.json"))
    assert read_checkpoint_quantization_config(str(tmp_path)) is None


# ==========================================================================
# the disk pre-flight: the estimate is the cache the sharder writes
# ==========================================================================
def _estimate_for(weights, **kwargs):
    from soup_cli.utils.fp8_block import read_checkpoint_quantization_config
    from soup_cli.utils.layer_shard import _discover_safetensors, estimate_shard_cache_bytes

    kwargs.setdefault("dtype", "float32")
    kwargs.setdefault("arch", "deepseek_v3")
    kwargs.setdefault("num_hidden_layers", LAYERS)
    return estimate_shard_cache_bytes(
        _discover_safetensors(weights),
        quantization_config=read_checkpoint_quantization_config(weights),
        **kwargs,
    )


def _payload(path):
    import struct

    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
    return os.path.getsize(path) - 8 - length


@pytest.mark.parametrize("layout", ["", "per_expert"], ids=["fused", "per_expert"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_the_estimate_is_exact_for_an_fp8_source(tmp_path, layout, dtype):
    from soup_cli.utils.expert_stream import expert_shard_name
    from soup_cli.utils.layer_shard import layer_paths, shard_cache_disk_bytes

    fp8, ref = _synthetic_pair(tmp_path)
    estimate = _estimate_for(fp8, dtype=dtype, expert_layout=layout)
    # the scales are charged nothing: the same figure as the dequantised checkpoint
    assert estimate == _estimate_for(ref, dtype=dtype, expert_layout=layout)
    out = tmp_path / "s"
    index = _shard(fp8, out, dtype=dtype, expert_layout=layout)
    for idx, path in enumerate(layer_paths(str(out), index)):
        experts = out / expert_shard_name(idx)
        payload = _payload(path) + (_payload(str(experts)) if experts.exists() else 0)
        assert estimate.layer_bytes[idx] == payload, idx
    other = sum(
        _payload(str(out / name)) for name in os.listdir(out)
        if name == "extras.safetensors" or name.startswith("large_")
    )
    assert estimate.other_bytes == other
    assert estimate.total_bytes >= shard_cache_disk_bytes(str(out), index)


def test_the_estimate_without_the_config_refuses_an_fp8_source(tmp_path):
    from soup_cli.utils.fp8_block import UnsupportedFP8Error
    from soup_cli.utils.layer_shard import _discover_safetensors, estimate_shard_cache_bytes

    fp8, _ref = _synthetic_pair(tmp_path)
    with pytest.raises(UnsupportedFP8Error, match="declares no FP8 quantization_config"):
        estimate_shard_cache_bytes(
            _discover_safetensors(fp8), dtype="bfloat16", arch="deepseek_v3",
            num_hidden_layers=LAYERS,
        )


def test_the_header_reader_reads_fp8_dtypes_at_one_byte(tmp_path):
    import torch
    from safetensors.torch import save_file

    from soup_cli.utils.safetensors_reader import read_header

    path = str(tmp_path / "f.safetensors")
    save_file(
        {
            "a": torch.zeros(3, 5).to(torch.float8_e4m3fn),
            "b": torch.zeros(7).to(torch.float8_e5m2),
        },
        path,
    )
    header = read_header(path)
    assert (header["a"].dtype, header["a"].nbytes) == ("float8_e4m3fn", 15)
    assert (header["b"].dtype, header["b"].nbytes) == ("float8_e5m2", 7)


def test_the_preflight_estimate_is_not_approximate_for_an_fp8_source(tmp_path):
    from soup_cli.trainer.stream_setup import _preflight_cache_estimate
    from soup_cli.utils.fp8_block import read_checkpoint_quantization_config

    fp8, _ref = _synthetic_pair(tmp_path)
    plan = types.SimpleNamespace(weights_dir=fp8, needs_materialization=False)
    got = _preflight_cache_estimate(
        plan, dtype="float32", arch="deepseek_v3", quant="none", quant_suffixes=(),
        double_quant=True, expert_layout="", num_hidden_layers=LAYERS,
        quantization_config=read_checkpoint_quantization_config(fp8),
    )
    assert got == _estimate_for(fp8)


# ==========================================================================
# the cache identity: a bf16 cache is never reused for the FP8 source, nor the reverse
# ==========================================================================
def _replace_checkpoint(target, source):
    """Put ``source``'s files at ``target`` keeping the old files' mtimes: the size and the
    config must invalidate the cache on their own."""
    for name in ("model.safetensors", "config.json"):
        old = os.stat(os.path.join(target, name)).st_mtime_ns
        shutil.copyfile(os.path.join(source, name), os.path.join(target, name))
        os.utime(os.path.join(target, name), ns=(old, old))


@pytest.mark.parametrize("direction", ["bf16-then-fp8", "fp8-then-bf16"])
def test_a_cache_of_one_format_is_never_reused_for_the_other(tmp_path, direction):
    import torch
    from safetensors.torch import load_file

    from soup_cli.utils.layer_shard import layer_paths

    state, linears = _synthetic_state()
    fp8_state = _to_fp8(state, linears)
    plain = _write(tmp_path / "plain", state, quant=None)
    fp8 = _write(tmp_path / "fp8", fp8_state)
    first, second = (plain, fp8) if direction == "bf16-then-fp8" else (fp8, plain)
    weights = tmp_path / "w"
    shutil.copytree(first, weights)
    cache = tmp_path / "cache"
    _shard(str(weights), cache)
    _replace_checkpoint(str(weights), second)
    notes = []
    index = _shard(str(weights), cache, notify=notes.append)
    assert any("Re-sharding layer cache" in note for note in notes), notes
    want_state = _dequantised(fp8_state) if second == fp8 else state
    layer = load_file(layer_paths(str(cache), index)[0])
    assert torch.equal(
        layer["self_attn.o_proj.weight"],
        want_state["model.layers.0.self_attn.o_proj.weight"].to(torch.float32),
    )


def test_every_fp8_admitted_architecture_keys_its_cache_on_config_json():
    from soup_cli.utils.fp8_block import FP8_BLOCK_ARCHS
    from soup_cli.utils.layer_shard import source_identity_includes_config

    assert FP8_BLOCK_ARCHS
    assert all(source_identity_includes_config(arch) for arch in FP8_BLOCK_ARCHS)


# ==========================================================================
# the setup: refusals from the config alone, and the pre-flight is fed the config
# ==========================================================================
def _fp8_setup_cfg(quantization_config):
    return types.SimpleNamespace(
        model_type="deepseek_v3", hidden_size=HIDDEN, num_hidden_layers=LAYERS,
        vocab_size=VOCAB, intermediate_size=INTER, quantization_config=quantization_config,
    )


def test_the_setup_refuses_an_unsupported_fp8_base_before_resolving_weights(
    tmp_path, monkeypatch
):
    from tests.test_issue602_qwen4_ple_streaming import _drive_qwen4_streaming_setup

    def _must_not_resolve(*_args, **_kwargs):
        raise AssertionError("weights resolved for a base that cannot stream")

    with pytest.raises(ValueError, match="per-tensor FP8 scales"):
        _drive_qwen4_streaming_setup(
            tmp_path, monkeypatch, resolve_weights=_must_not_resolve, arch="deepseek_v3",
            model_cfg=_fp8_setup_cfg({"quant_method": "fp8"}),
        )


def test_the_setup_refuses_an_fp8_base_on_another_architecture(tmp_path, monkeypatch):
    from tests.test_issue602_qwen4_ple_streaming import _drive_qwen4_streaming_setup

    def _must_not_resolve(*_args, **_kwargs):
        raise AssertionError("weights resolved for a base that cannot stream")

    cfg = _fp8_setup_cfg(DEEPSEEK_FP8)
    cfg.model_type = "llama"
    with pytest.raises(ValueError, match="admitted: deepseek_v3"):
        _drive_qwen4_streaming_setup(
            tmp_path, monkeypatch, resolve_weights=_must_not_resolve, arch="llama",
            model_cfg=cfg,
        )


def test_the_setup_hands_the_fp8_config_to_the_disk_preflight(tmp_path, monkeypatch):
    import soup_cli.trainer.stream_setup as stream_setup
    from tests.test_issue602_qwen4_ple_streaming import _drive_qwen4_streaming_setup

    fp8, _ref = _synthetic_pair(tmp_path)
    size = os.path.getsize(os.path.join(fp8, "model.safetensors"))
    plan = types.SimpleNamespace(
        weights_dir=fp8, source_dir=fp8, source_bytes=size, materialized_copy_bytes=0,
        materialize_bytes=0, needs_materialization=False,
        source_files=(("model.safetensors", size, 0),), source_revision=None,
    )

    def _resolve(_base, *, before_materialize=None):
        before_materialize(plan)
        return fp8

    panels = []
    real = stream_setup._render_stream_disk_preflight
    monkeypatch.setattr(
        stream_setup, "_render_stream_disk_preflight",
        lambda **kwargs: (panels.append(kwargs), real(**kwargs))[1],
    )
    monkeypatch.setattr(stream_setup, "_disk_volume", lambda _path: (1, 10**15))
    with open(os.path.join(fp8, "config.json"), encoding="utf-8") as handle:
        quantization_config = json.load(handle)["quantization_config"]
    _drive_qwen4_streaming_setup(
        tmp_path, monkeypatch, resolve_weights=_resolve, arch="deepseek_v3",
        model_cfg=_fp8_setup_cfg(quantization_config),
    )
    from soup_cli.utils.layer_stream import resolve_stream_dtype

    expected = _estimate_for(fp8, dtype=resolve_stream_dtype("cpu"))
    assert len(panels) == 1
    assert panels[0]["approximate"] is False
    assert panels[0]["shard_bytes"] == expected.total_bytes


# ==========================================================================
# G2 — a tiny DeepseekV3ForCausalLM: FP8-sourced stream vs the dequantised resident model
# ==========================================================================
def _tiny_config(q_lora_rank):
    from transformers import DeepseekV3Config

    return DeepseekV3Config(
        vocab_size=128, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
        num_hidden_layers=3, first_k_dense_replace=1, num_attention_heads=4,
        num_key_value_heads=4, n_shared_experts=1, n_routed_experts=8, n_group=2,
        topk_group=1, num_experts_per_tok=2, kv_lora_rank=16, q_lora_rank=q_lora_rank,
        qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16, max_position_embeddings=128,
    )


def _build_tiny(root, q_lora_rank, block):
    """(bf16 original, FP8 block copy, reference-dequantised copy) of one tiny model."""
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import DeepseekV3ForCausalLM

    torch.manual_seed(11)
    model = DeepseekV3ForCausalLM(_tiny_config(q_lora_rank)).to(torch.float32).eval()
    generator = torch.Generator().manual_seed(23)
    with torch.no_grad():
        for layer in model.model.layers[1:]:
            bias = layer.mlp.gate.e_score_correction_bias
            bias.copy_(torch.randn(bias.shape, generator=generator) * 2.0)
        for parameter in model.parameters():
            parameter.data = parameter.data.to(torch.bfloat16)
    original = root / "bf16"
    model.save_pretrained(str(original))
    fp8 = root / "fp8"
    _harness().convert_checkpoint(str(original), str(fp8), block=block, log=lambda _l: None)
    fp8_state = load_file(str(fp8 / "model.safetensors"))
    assert any(value.dtype == torch.float8_e4m3fn for value in fp8_state.values())
    ref = root / "ref"
    ref.mkdir()
    save_file(_dequantised(fp8_state, block), str(ref / "model.safetensors"))
    shutil.copyfile(original / "config.json", ref / "config.json")
    return str(original), str(fp8), str(ref)


@pytest.fixture(scope="module")
def tiny_models(tmp_path_factory):
    pytest.importorskip("transformers")
    built = {}

    def get(q_lora_rank, block):
        key = (q_lora_rank, tuple(block))
        if key not in built:
            root = tmp_path_factory.mktemp(f"tiny-q{q_lora_rank}-b{block[0]}x{block[1]}")
            built[key] = _build_tiny(root, q_lora_rank, tuple(block))
        return built[key]

    return get


def _tiny_lora():
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
        target_modules=_LORA_TARGETS, task_type=TaskType.CAUSAL_LM,
    )


def _resident(weights, layout):
    import torch
    from transformers import DeepseekV3ForCausalLM

    model = DeepseekV3ForCausalLM.from_pretrained(weights, dtype=torch.float32).eval()
    if layout == "per_expert":
        # The streamed experts follow the eager loop; the resident control must too.
        model.set_experts_implementation("eager")
    return model


def _streamed_vs_resident(tmp_path, fp8, ref, layout):
    import torch
    from peft import get_peft_model

    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    shards = str(tmp_path / "shards")
    index = _shard(fp8, shards, num_hidden_layers=3, expert_layout=layout)
    model, runtime = build_streamed_model(
        model_id=fp8, shard_dir=shards, index=index, lora_config=_tiny_lora(),
        device="cpu", dtype="float32", buffers=2, pin=False, seed=3,
    )
    resident = get_peft_model(_resident(ref, layout), _tiny_lora())
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
    return model, runtime, resident, ids


G2_CELLS = [
    ("", 24, (16, 32)),
    ("", None, (16, 32)),
    ("per_expert", 24, (16, 32)),
    ("per_expert", None, (16, 32)),
    ("", 24, (128, 128)),
]


@pytest.mark.parametrize(
    ("layout", "q_lora_rank", "block"), G2_CELLS,
    ids=[f"{lay or 'fused'}-q{q}-b{b[0]}x{b[1]}" for lay, q, b in G2_CELLS],
)
def test_streamed_from_fp8_matches_the_dequantised_resident_bit_exactly(
    tmp_path, tiny_models, layout, q_lora_rank, block
):
    import torch

    from soup_cli.utils.layer_stream_runtime import canonical_named_parameters

    _original, fp8, ref = tiny_models(q_lora_rank, block)
    model, runtime, resident, ids = _streamed_vs_resident(tmp_path, fp8, ref, layout)
    try:
        with torch.no_grad():
            got = model(input_ids=ids, labels=ids)
            want = resident(input_ids=ids, labels=ids)
        assert torch.equal(got.logits, want.logits), (got.logits - want.logits).abs().max()
        assert torch.equal(got.loss, want.loss)

        model.zero_grad(set_to_none=True)
        resident.zero_grad(set_to_none=True)
        model(input_ids=ids, labels=ids).loss.backward()
        resident(input_ids=ids, labels=ids).loss.backward()

        def grads(net):
            return {
                name: parameter.grad
                for name, parameter in canonical_named_parameters(net)
                if "lora_" in name and parameter.grad is not None
            }

        streamed, resident_grads = grads(model), grads(resident)
        assert streamed and set(streamed) == set(resident_grads)
        assert any(
            ".layers.0." in name and float(gradient.abs().sum()) > 0
            for name, gradient in streamed.items()
        )
        for name, gradient in streamed.items():
            assert torch.equal(gradient, resident_grads[name]), name
    finally:
        runtime.close()


def test_fp8_is_lossy_so_the_parity_pins_the_dequant(tiny_models):
    """The control: the dequantised model is not the bf16 original."""
    import torch

    original, _fp8, ref = tiny_models(24, (16, 32))
    ids = torch.randint(0, 128, (2, 12), generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        dequantised = _resident(ref, "").forward(input_ids=ids).logits
        bf16 = _resident(original, "").forward(input_ids=ids).logits
    assert not torch.equal(dequantised, bf16)


def test_the_streamed_model_is_built_bf16_not_with_transformers_fp8_layers(
    tmp_path, tiny_models
):
    import torch.nn as nn

    _original, fp8, ref = tiny_models(24, (16, 32))
    model, runtime, _resident_model, _ids = _streamed_vs_resident(tmp_path, fp8, ref, "")
    try:
        base = model.get_base_model()
        names = {type(module).__name__ for module in base.modules()}
        assert not any("FP8" in name or "Fp8" in name for name in names), names
        attention = base.model.layers[0].inner.self_attn
        assert type(attention.o_proj.base_layer) is nn.Linear
        assert getattr(base.config, "quantization_config", None) is None
    finally:
        runtime.close()


def test_the_meta_skeleton_never_hands_an_fp8_config_to_from_config(tiny_models, monkeypatch):
    """Built bf16 whatever a transformers version would do with the FP8 config."""
    import transformers

    from soup_cli.utils.layer_stream_runtime import build_meta_skeleton

    _original, fp8, _ref = tiny_models(24, (16, 32))
    seen = []
    real = transformers.AutoModelForCausalLM.from_config

    def _spy(config, **kwargs):
        seen.append(getattr(config, "quantization_config", None))
        return real(config, **kwargs)

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_config", _spy)
    build_meta_skeleton(fp8, dtype="bfloat16")
    assert seen == [None]


def test_the_resident_view_of_the_checkpoint_is_untouched(tiny_models):
    """Only the skeleton's own config object loses the FP8 config: config.json and a
    fresh resident config still carry it."""
    from transformers import AutoConfig

    from soup_cli.utils.layer_stream_runtime import build_meta_skeleton

    _original, fp8, _ref = tiny_models(24, (16, 32))
    path = os.path.join(fp8, "config.json")
    before = open(path, "rb").read()
    build_meta_skeleton(fp8, dtype="bfloat16")
    assert open(path, "rb").read() == before
    fresh = AutoConfig.from_pretrained(fp8).quantization_config
    assert fresh["quant_method"] == "fp8" and fresh["weight_block_size"] == [16, 32]


@pytest.mark.parametrize("layout", ["", "per_expert"], ids=["fused", "per_expert"])
@pytest.mark.parametrize(
    ("quant", "dtype"),
    [("none", "float32"), ("none", "bfloat16"), ("nf4", "bfloat16")],
    ids=["none-fp32", "none-bf16", "nf4-bf16"],
)
def test_the_fp8_cache_is_the_dequantised_cache_byte_for_byte(
    tmp_path, tiny_models, layout, quant, dtype
):
    """Where NF4 is gated on the CPU: the CUDA NF4 path gets exactly these bytes."""
    _original, fp8, ref = tiny_models(24, (16, 32))
    kwargs = dict(dtype=dtype, num_hidden_layers=3, expert_layout=layout, quant=quant)
    if quant == "nf4":
        pytest.importorskip("bitsandbytes")
        from soup_cli.utils.layer_stream_runtime import (
            build_meta_skeleton,
            quantised_layer_suffixes,
        )

        suffixes = quantised_layer_suffixes(build_meta_skeleton(fp8, dtype=dtype, quant="nf4"))
        assert "self_attn.kv_a_proj_with_mqa.weight" in suffixes
        kwargs.update(quant_suffixes=suffixes, quant_device="cpu", double_quant=True)
    mine = _shard(fp8, tmp_path / "a", **kwargs)
    theirs = _shard(ref, tmp_path / "b", **kwargs)
    _assert_same_cache(str(tmp_path / "a"), str(tmp_path / "b"), mine, theirs)
    assert _estimate_for(fp8, **{k: v for k, v in kwargs.items() if k != "quant_device"}) == (
        _estimate_for(ref, **{k: v for k, v in kwargs.items() if k != "quant_device"})
    )


# ==========================================================================
# bounded work: a hostile block size, and a big tensor, cost one band of fp32
# ==========================================================================
class _TensorSizes:
    """Records the element count and dtype of every tensor a torch call returns."""

    def __init__(self):
        from torch.overrides import TorchFunctionMode

        sizes = self.sizes = []

        class _Mode(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                import torch

                out = func(*args, **(kwargs or {}))
                if isinstance(out, torch.Tensor):
                    sizes.append((out.numel(), out.dtype))
                return out

        self.mode = _Mode()

    def __enter__(self):
        self.mode.__enter__()
        return self

    def __exit__(self, *exc):
        self.mode.__exit__(*exc)

    def largest(self, dtype):
        return max((numel for numel, kind in self.sizes if kind == dtype), default=0)


@pytest.mark.parametrize("block", [(2**24, 2**24), (2**24, 4), (4, 2**24)],
                         ids=["both", "rows", "cols"])
def test_a_huge_block_size_stays_cheap_and_exact(block):
    """A ``weight_block_size`` far past the matrix must not expand the scale to its size:
    a 4 x 4 weight with a [1, 1] scale used to allocate a 4 x 2**24 fp32 grid."""
    import torch

    from soup_cli.utils.fp8_block import dequantize_fp8_block

    generator = torch.Generator().manual_seed(2)
    weight = (torch.randn(4, 4, generator=generator) * 30).to(torch.float8_e4m3fn)
    scale = torch.tensor([[0.375]])
    with _TensorSizes() as seen:
        got = dequantize_fp8_block(weight, scale, block, dtype=torch.bfloat16)
    assert torch.equal(got, (weight.to(torch.float32) * 0.375).to(torch.bfloat16))
    assert max(numel for numel, _kind in seen.sizes) <= 16


def test_an_fp8_source_with_a_huge_block_size_shards_exactly(tmp_path):
    import torch
    from safetensors.torch import load_file

    from soup_cli.utils.layer_shard import layer_paths

    state, linears = _synthetic_state()
    huge = (2**20, 2**20)  # the largest weight_block_size accepted
    fp8_state = _to_fp8(state, linears, block=huge)
    assert all(tuple(fp8_state[key + "_scale_inv"].shape) == (1, 1) for key in linears)
    fp8 = _write(tmp_path / "fp8", fp8_state, block=huge)
    index = _shard(fp8, tmp_path / "s")
    tensors = load_file(layer_paths(str(tmp_path / "s"), index)[0])
    key = "model.layers.0.self_attn.o_proj.weight"
    want = fp8_state[key].to(torch.float32) * fp8_state[key + "_scale_inv"][0, 0]
    assert torch.equal(tensors["self_attn.o_proj.weight"], want)


def test_dequant_works_in_bands_into_the_stream_dtype(monkeypatch):
    """The fp32 product is formed one band of rows at a time and cast into a preallocated
    output: the peak is the output plus one band, not 9-11 bytes per element."""
    import torch

    import soup_cli.utils.fp8_block as fp8_block

    block = (128, 128)
    weight, scale = _harness().quantize_fp8_block(_blockwise_weight((300, 200), block, 9), block)
    monkeypatch.setattr(fp8_block, "_DEQUANT_BAND_ELEMENTS", 1000)
    with _TensorSizes() as seen:
        got = fp8_block.dequantize_fp8_block(weight, scale, block, dtype=torch.bfloat16)
    want = _reference_dequant(weight, scale, block).to(torch.bfloat16)
    assert got.dtype == torch.bfloat16
    assert torch.equal(got, want)
    # one band is 1000 // 200 = 5 rows; its scale grid before the crop is 5 x 256
    assert seen.largest(torch.float32) <= 5 * 256
    assert [numel for numel, kind in seen.sizes if kind == torch.bfloat16 and numel == 60000]


@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
def test_the_banded_result_is_the_unbanded_cast(dtype):
    import torch

    from soup_cli.utils.fp8_block import dequantize_fp8_block

    block = (128, 128)
    weight, scale = _harness().quantize_fp8_block(
        _blockwise_weight((2048, 1408), block, 21), block
    )
    target = getattr(torch, dtype)
    want = _reference_dequant(weight, scale, block).to(target)
    assert torch.equal(dequantize_fp8_block(weight, scale, block, dtype=target), want)
    # bands that do not line up with the 128-row blocks (100,000 // 1408 = 71 rows)
    import soup_cli.utils.fp8_block as fp8_block

    original = fp8_block._DEQUANT_BAND_ELEMENTS
    fp8_block._DEQUANT_BAND_ELEMENTS = 100_000
    try:
        assert torch.equal(dequantize_fp8_block(weight, scale, block, dtype=target), want)
    finally:
        fp8_block._DEQUANT_BAND_ELEMENTS = original


# ==========================================================================
# values the format cannot hold are refused, naming the tensor
# ==========================================================================
@pytest.mark.parametrize("code", [0x7F, 0xFF], ids=["nan", "neg-nan"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_a_nan_fp8_byte_is_refused_after_dequant(code, dtype):
    import torch

    from soup_cli.utils.fp8_block import UnsupportedFP8Error, dequantize_fp8_block

    raw = torch.zeros(4, 4, dtype=torch.uint8)
    raw[2, 1] = code
    weight = raw.view(torch.float8_e4m3fn)
    with pytest.raises(UnsupportedFP8Error) as caught:
        dequantize_fp8_block(
            weight, torch.ones(1, 1), (128, 128), name="layers.0.q", dtype=getattr(torch, dtype)
        )
    text = str(caught.value)
    assert "'layers.0.q'" in text and "NaN" in text


def test_a_float16_overflow_is_refused_suggesting_bfloat16():
    import torch

    from soup_cli.utils.fp8_block import UnsupportedFP8Error, dequantize_fp8_block

    weight = torch.full((2, 2), 448.0).to(torch.float8_e4m3fn)
    scale = torch.full((1, 1), 200.0)  # 448 x 200 = 89,600 > 65,504
    with pytest.raises(UnsupportedFP8Error) as caught:
        dequantize_fp8_block(weight, scale, (128, 128), name="w", dtype=torch.float16)
    assert "overflows float16" in str(caught.value) and "bfloat16" in str(caught.value)
    for dtype in (torch.bfloat16, torch.float32):
        out = dequantize_fp8_block(weight, scale, (128, 128), name="w", dtype=dtype)
        assert torch.equal(out, torch.full((2, 2), 89600.0).to(dtype))


def test_a_negative_scale_is_refused():
    import torch

    from soup_cli.utils.fp8_block import UnsupportedFP8Error, dequantize_fp8_block

    weight = torch.ones(4, 4).to(torch.float8_e4m3fn)
    scale = torch.tensor([[0.5, -0.25]])
    with pytest.raises(UnsupportedFP8Error, match="negative"):
        dequantize_fp8_block(weight, scale, (4, 2), name="w")


def test_a_zero_scale_on_an_all_zero_block_is_kept():
    import torch

    from soup_cli.utils.fp8_block import dequantize_fp8_block

    weight = torch.zeros(4, 4).to(torch.float8_e4m3fn)
    weight[:, 2:] = torch.ones(4, 2).to(torch.float8_e4m3fn)
    scale = torch.tensor([[0.0, 0.5], [-0.0, 0.5]])
    got = dequantize_fp8_block(weight, scale, (2, 2), name="w")
    assert torch.equal(got, _reference_dequant(weight, scale, (2, 2)))


def test_the_sharder_refuses_a_nan_weight_naming_it_before_the_index(tmp_path):
    import torch

    from soup_cli.utils.fp8_block import UnsupportedFP8Error

    state, linears = _synthetic_state()
    fp8_state = _to_fp8(state, linears)
    key = "model.layers.1.mlp.experts.0.down_proj.weight"
    raw = fp8_state[key].view(torch.uint8).clone()
    raw[0, 0] = 0x7F
    fp8_state[key] = raw.view(torch.float8_e4m3fn)
    fp8 = _write(tmp_path / "fp8", fp8_state)
    with pytest.raises(UnsupportedFP8Error) as caught:
        _shard(fp8, tmp_path / "s", dtype="bfloat16")
    assert key in str(caught.value)
    assert not (tmp_path / "s" / "index.json").exists()


# ==========================================================================
# a mixed checkpoint: a bf16 linear beside FP8 siblings is stored as it is
# ==========================================================================
def test_a_bf16_linear_beside_fp8_siblings_is_cast_as_stored(tmp_path):
    """Real DeepSeek-V3 checkpoints mix the two: only the FP8 tensors carry a scale."""
    import torch
    from safetensors.torch import load_file

    from soup_cli.utils.layer_shard import layer_paths

    state, linears = _synthetic_state()
    kept = "model.layers.1.self_attn.o_proj.weight"
    fp8_state = _to_fp8(state, [key for key in linears if key != kept])
    assert fp8_state[kept].dtype == torch.bfloat16 and kept + "_scale_inv" not in fp8_state
    fp8 = _write(tmp_path / "fp8", fp8_state)
    ref = _write(tmp_path / "ref", _dequantised(fp8_state), quant=None)
    mine = _shard(fp8, tmp_path / "a", dtype="bfloat16")
    theirs = _shard(ref, tmp_path / "b", dtype="bfloat16")
    _assert_same_cache(str(tmp_path / "a"), str(tmp_path / "b"), mine, theirs)
    layer = load_file(layer_paths(str(tmp_path / "a"), mine)[1])
    assert torch.equal(layer["self_attn.o_proj.weight"], state[kept])


# ==========================================================================
# the backstop and the skeleton, directly
# ==========================================================================
@pytest.mark.parametrize(
    ("key", "dtype"),
    [
        ("model.layers.0.q_proj.weight", "F8_E4M3"),
        ("model.layers.0.q_proj.weight", "F8_E5M2"),
        ("model.layers.0.q_proj.weight", "F8_E4M3FNUZ"),
        ("model.layers.0.q_proj.weight_scale_inv", "F32"),
    ],
    ids=["e4m3", "e5m2", "e4m3fnuz", "stray-scale"],
)
def test_the_plain_read_backstop_refuses_every_fp8_spelling(key, dtype):
    from soup_cli.utils.fp8_block import UnsupportedFP8Error
    from soup_cli.utils.layer_shard import _refuse_fp8

    with pytest.raises(UnsupportedFP8Error, match="cast without its scales"):
        _refuse_fp8(key, dtype)


# U8 used to pass here too: the backstop now refuses every integer or packed type
# (tests/test_stream_refuses_packed_quant.py), so a plain tensor is a floating one.
@pytest.mark.parametrize("dtype", ["BF16", "F16", "F32", "F64"])
def test_the_plain_read_backstop_passes_a_plain_tensor(dtype):
    from soup_cli.utils.layer_shard import _refuse_fp8

    _refuse_fp8("model.layers.0.q_proj.weight", dtype)


def test_the_skeleton_keeps_a_quantization_config_that_is_not_fp8(tiny_models, monkeypatch):
    """Only an FP8 config is dropped. Any other one reaches ``from_config`` and the built
    model exactly as before this change (pinned against the pre-change code: the config's
    dict is handed through untouched)."""
    import transformers

    from soup_cli.utils.layer_stream_runtime import build_meta_skeleton

    original, _fp8, _ref = tiny_models(24, (16, 32))
    weights = os.path.join(os.path.dirname(original), "bnb-config")
    if not os.path.isdir(weights):
        shutil.copytree(original, weights)
        config_path = os.path.join(weights, "config.json")
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
        config["quantization_config"] = {"quant_method": "bitsandbytes", "load_in_4bit": True}
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(config, handle)
    seen = []
    real = transformers.AutoModelForCausalLM.from_config

    def _spy(config, **kwargs):
        seen.append(getattr(config, "quantization_config", None))
        return real(config, **kwargs)

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_config", _spy)
    model = build_meta_skeleton(weights, dtype="bfloat16")
    expected = {"quant_method": "bitsandbytes", "load_in_4bit": True}
    assert seen == [expected]
    assert model.config.quantization_config == expected
