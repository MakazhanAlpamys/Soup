"""The resident route's guarded config loader (#2104): load_resident_model_config.

A local twin of the streamed loader's tests, built on synthetic tmp_path configs only
(no network). The streamed route's wording must never leak into a resident refusal.
"""

import json
import os
import re
import shutil
from pathlib import Path

import pytest

pytest.importorskip("transformers")

#: A minimal K2-shaped config: every published required key present (tiny dimensions), an
#: auto_map (optional; the rewrite drops it), no quantization_config (optional). Spelled out
#: inline so this file does not depend on the shared builder's evolving shape (that builder
#: lives in tests/kimi_k2_tiny.py and carries the parity fixtures' checkpoints).
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


#: Kimi-K2-Thinking's quantization_config spelling (compressed-tensors INT4): the guard's
#: quant_method refusal row is one of the two streamed-wording leak sites (ruling D15).
_THINKING_INT4_QUANT = {
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "quantization_status": "compressed",
}


#: An FP8 quantization_config in a misspelled spelling (per-tensor FP8 without
#: weight_block_size): fp8_block's refusal tail names layer streaming — the third D15
#: leak site, reached through the UnsupportedFP8Error branch.
_MISSPELLED_FP8_QUANT = {"quant_method": "fp8", "activation_scheme": "dynamic"}


def _assert_no_streamed_wording(message):
    lowered = message.lower()
    assert "stream" not in lowered
    assert "layer streaming" not in lowered
    assert "training.stream_layers" not in lowered


def test_resident_refusal_for_an_unexpected_key_has_no_streamed_wording(tmp_path):
    from soup_cli.utils.kimi_k2 import (
        UnsupportedKimiK2Error,
        load_resident_model_config,
    )

    raw = dict(_MINIMAL_K2, ep_size=1)
    with pytest.raises(UnsupportedKimiK2Error) as caught:
        load_resident_model_config(_write_config(tmp_path, raw))
    message = str(caught.value)
    assert "'ep_size'" in message
    _assert_no_streamed_wording(message)


def test_resident_refusal_for_a_thinking_style_quant_config_has_no_streamed_wording(tmp_path):
    from soup_cli.utils.kimi_k2 import (
        UnsupportedKimiK2Error,
        load_resident_model_config,
    )

    raw = dict(_MINIMAL_K2, quantization_config=_THINKING_INT4_QUANT)
    with pytest.raises(UnsupportedKimiK2Error) as caught:
        load_resident_model_config(_write_config(tmp_path, raw))
    message = str(caught.value)
    assert "quantization_config.quant_method" in message
    assert "compressed-tensors" in message
    _assert_no_streamed_wording(message)


def test_resident_refusal_for_a_misspelled_fp8_config_has_no_streamed_wording(tmp_path):
    from soup_cli.utils.kimi_k2 import (
        UnsupportedKimiK2Error,
        load_resident_model_config,
    )

    raw = dict(_MINIMAL_K2, quantization_config=_MISSPELLED_FP8_QUANT)
    with pytest.raises(UnsupportedKimiK2Error) as caught:
        load_resident_model_config(_write_config(tmp_path, raw))
    message = str(caught.value)
    assert "per-tensor FP8" in message
    assert "resident route dequantises" in message
    _assert_no_streamed_wording(message)


def test_streamed_refusal_for_these_cases_keeps_its_streamed_wording():
    # The 186-test streamed suite pins the streamed messages; this pins the three leak-site
    # rows keep their streamed wording when the route is the streamed one.
    from soup_cli.utils.kimi_k2 import (
        UnsupportedKimiK2Error,
        kimi_k2_native_config_dict,
    )

    with pytest.raises(UnsupportedKimiK2Error) as unexpected:
        kimi_k2_native_config_dict(dict(_MINIMAL_K2, ep_size=1))
    assert "training.stream_layers admits kimi_k2" in str(unexpected.value)

    with pytest.raises(UnsupportedKimiK2Error) as quant:
        kimi_k2_native_config_dict(
            dict(_MINIMAL_K2, quantization_config=_THINKING_INT4_QUANT)
        )
    assert "layer streaming reads Kimi K2 unquantised" in str(quant.value)

    with pytest.raises(UnsupportedKimiK2Error) as fp8:
        kimi_k2_native_config_dict(
            dict(_MINIMAL_K2, quantization_config=_MISSPELLED_FP8_QUANT)
        )
    assert "Layer streaming dequantises one FP8 format" in str(fp8.value)


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


# ==========================================================================
# load_resident_model: the native resident model load with FP8 dequant-on-load
# =========================================================================
pytest.importorskip("torch")


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
_BLOCK = (16, 32)
#: The per-expert checkpoint spelling -> which fused native parameter it lands in.
_EXPERT_KEY_RE = re.compile(r"^(.*)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
#: The real checkpoint carries 61 of these (BF16 [56]); neither K2's code (non-persistent
#: buffer) nor the native class reads them, so the resident load must tolerate them.
_STALE_INV_FREQ = "self_attn.rotary_emb.inv_freq"
_STALE_VALUE = 12345.0


def _native_tiny_config():
    """The control's config, from keyword arguments: never through Soup's loader."""
    from transformers import DeepseekV3Config

    rope = {"rope_type": "yarn", "rope_theta": 50000.0, **_TINY_YARN}
    return DeepseekV3Config(
        **_TINY, rope_parameters=rope, rms_norm_eps=1e-6, routed_scaling_factor=2.827,
        norm_topk_prob=True, tie_word_embeddings=False, num_mtp_layers=0,
    )


def _fp8_harness():
    """benchmarks/harness/fp8_block_quantize.py, loaded the way test_fp8_block_streaming does."""
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


@pytest.fixture(scope="module")
def resident_checkpoints(tmp_path_factory):
    """{"k2-bf16", "k2-fp8", "native"}: the same seeded weights, three spellings."""
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file, save_file
    from transformers import DeepseekV3ForCausalLM

    root = tmp_path_factory.mktemp("kimi-k2-resident")
    torch.manual_seed(11)
    model = DeepseekV3ForCausalLM(_native_tiny_config()).to(torch.float32).eval()
    generator = torch.Generator().manual_seed(23)
    with torch.no_grad():
        for layer in model.model.layers[1:]:
            bias = layer.mlp.gate.e_score_correction_bias
            bias.copy_(torch.randn(bias.shape, generator=generator) * 2.0)
        for parameter in model.parameters():
            parameter.data = parameter.data.to(torch.bfloat16)
    native = root / "native"
    model.save_pretrained(str(native))
    state = load_file(str(native / "model.safetensors"))
    stale = {
        f"model.layers.{idx}.{_STALE_INV_FREQ}": torch.full((7,), _STALE_VALUE).to(torch.bfloat16)
        for idx in range(_TINY["num_hidden_layers"])
    }
    k2_bf16 = root / "k2-bf16"
    k2_bf16.mkdir()
    save_file({**state, **stale}, str(k2_bf16 / "model.safetensors"), metadata={"format": "pt"})
    (k2_bf16 / "config.json").write_text(json.dumps(_MINIMAL_K2), encoding="utf-8")

    k2_fp8 = root / "k2-fp8"
    _fp8_harness().convert_checkpoint(
        str(k2_bf16), str(k2_fp8), block=_BLOCK, log=lambda _line: None
    )
    assert not any(name.endswith(".py") for name in os.listdir(k2_fp8))
    return {"k2-bf16": str(k2_bf16), "k2-fp8": str(k2_fp8), "native": str(native)}


def _reference_model(resident_checkpoints):
    import torch
    from transformers import DeepseekV3ForCausalLM

    return DeepseekV3ForCausalLM.from_pretrained(
        resident_checkpoints["native"], dtype=torch.float32
    ).eval()


def _ids(torch):
    generator = torch.Generator().manual_seed(17)
    return torch.randint(0, 128, (2, 12), generator=generator)


def test_resident_model_of_the_bf16_copy_matches_native_from_pretrained(resident_checkpoints):
    import torch
    from transformers import DeepseekV3ForCausalLM

    from soup_cli.utils.kimi_k2 import load_resident_model

    model = load_resident_model(
        resident_checkpoints["k2-bf16"], model_kwargs={"dtype": torch.float32}
    )
    assert type(model) is DeepseekV3ForCausalLM
    reference = _reference_model(resident_checkpoints)
    ids = _ids(torch)
    with torch.no_grad():
        got = model(ids).logits
        want = reference(ids).logits
    assert (got - want).abs().max() < 1e-5  # bf16-storage rounding only


def test_resident_model_of_the_fp8_copy_dequantises_like_the_independent_reference(
    resident_checkpoints,
):
    import torch
    from safetensors.torch import load_file
    from transformers import DeepseekV3ForCausalLM

    from soup_cli.utils.kimi_k2 import load_resident_model

    fp8_dir = resident_checkpoints["k2-fp8"]
    model = load_resident_model(fp8_dir, model_kwargs={"dtype": torch.float32})
    assert type(model) is DeepseekV3ForCausalLM

    # Weight by weight against an independent per-block dequant loop, not our own helper.
    # Per-expert tensors fuse into the native model's stacked experts; compare the fused
    # parameter against the same fusion built from the independent reference dequants.
    fp8_state = load_file(os.path.join(fp8_dir, "model.safetensors"))
    loaded = model.state_dict()
    experts = {}
    for key, value in fp8_state.items():
        if key.endswith("_scale_inv") or key.endswith(_STALE_INV_FREQ):
            continue
        match = _EXPERT_KEY_RE.match(key)
        if match is not None:
            experts.setdefault(match.group(1), {}).setdefault(
                int(match.group(2)), {}
            )[match.group(3)] = key
            continue
        scale = fp8_state.get(key + "_scale_inv")
        want = value if scale is None else _reference_dequant(value, scale, _BLOCK)
        got = loaded[key].to(torch.float32)
        assert torch.allclose(got, want.to(torch.float32), rtol=1e-5, atol=1e-6), key
    for prefix, per_expert in sorted(experts.items()):
        n = len(per_expert)

        def _ref(key):
            return _reference_dequant(
                fp8_state[key], fp8_state[key + "_scale_inv"], _BLOCK
            )

        gate_up = torch.stack(
            [
                torch.cat([_ref(per_expert[e]["gate_proj"]), _ref(per_expert[e]["up_proj"])], dim=0)
                for e in range(n)
            ],
            dim=0,
        )
        down = torch.stack([_ref(per_expert[e]["down_proj"]) for e in range(n)], dim=0)
        got_gate_up = loaded[prefix + ".mlp.experts.gate_up_proj"].to(torch.float32)
        got_down = loaded[prefix + ".mlp.experts.down_proj"].to(torch.float32)
        assert torch.allclose(got_gate_up, gate_up, rtol=1e-5, atol=1e-6), prefix
        assert torch.allclose(got_down, down, rtol=1e-5, atol=1e-6), prefix

    reference = _reference_model(resident_checkpoints)
    ids = _ids(torch)
    with torch.no_grad():
        got = model(ids).logits
        want = reference(ids).logits
    # FP8-block weight quantisation noise through three bf16 layers: measured max delta
    # 0.021 against logits of abs max ~0.58 (K2's sigmoid scoring keeps logits small).
    assert (got - want).abs().max() < 5e-2


def test_resident_model_refuses_an_fp8_weight_missing_its_scale(tmp_path, resident_checkpoints):
    import torch
    from safetensors.torch import load_file, save_file

    from soup_cli.utils.kimi_k2 import load_resident_model

    src = resident_checkpoints["k2-fp8"]
    dst = tmp_path / "fp8-missing-scale"
    dst.mkdir()
    state = load_file(os.path.join(src, "model.safetensors"))
    victim = sorted(key for key in state if key.endswith("_scale_inv"))[0]
    del state[victim]
    save_file(state, str(dst / "model.safetensors"), metadata={"format": "pt"})
    shutil.copyfile(os.path.join(src, "config.json"), dst / "config.json")

    weight_key = victim[: -len("_scale_inv")]
    with pytest.raises(ValueError) as caught:
        load_resident_model(str(dst), model_kwargs={"dtype": torch.float32})
    message = str(caught.value)
    assert weight_key in message
    assert victim in message
    _assert_no_streamed_wording(message)


def test_resident_model_refuses_a_loader_quantization_config(resident_checkpoints):
    import torch

    from soup_cli.utils.kimi_k2 import load_resident_model

    with pytest.raises(ValueError) as caught:
        load_resident_model(
            resident_checkpoints["k2-fp8"],
            model_kwargs={
                "dtype": torch.float32,
                "quantization_config": {"quant_method": "bitsandbytes", "load_in_4bit": True},
            },
        )
    message = str(caught.value)
    assert "quantization_config" in message
    assert "bf16" in message
    assert "dequantised" in message


def test_resident_model_passes_through_for_non_k2_bases(tmp_path, monkeypatch):
    import transformers

    from soup_cli.utils import layer_shard
    from soup_cli.utils.kimi_k2 import load_resident_model

    config_dir = _write_config(
        tmp_path,
        {
            "model_type": "llama",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "num_key_value_heads": 2,
            "vocab_size": 128,
        },
    )
    sentinel = object()
    calls = {}

    def fake_from_pretrained(base, **kwargs):
        calls["base"] = base
        calls["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", fake_from_pretrained
    )

    def _boom(*_args, **_kwargs):
        raise AssertionError("the FP8 dequant machinery was touched")

    monkeypatch.setattr(layer_shard, "_discover_safetensors", _boom)

    model_kwargs = {"dtype": "bfloat16", "device_map": "cpu"}
    result = load_resident_model(config_dir, model_kwargs=dict(model_kwargs))
    assert result is sentinel
    assert calls["base"] == config_dir
    assert calls["kwargs"] == model_kwargs


def test_resident_model_of_the_fp8_copy_never_loads_remote_code(resident_checkpoints, monkeypatch):
    import torch
    import transformers.dynamic_module_utils as dynamic
    import transformers.models.auto.configuration_auto as configuration_auto

    from soup_cli.utils.kimi_k2 import load_resident_model

    def _refuse(*_args, **_kwargs):
        raise AssertionError("remote code was loaded")

    monkeypatch.setattr(dynamic, "get_class_from_dynamic_module", _refuse)
    monkeypatch.setattr(configuration_auto, "get_class_from_dynamic_module", _refuse)

    model = load_resident_model(
        resident_checkpoints["k2-fp8"], model_kwargs={"dtype": torch.float32}
    )
    assert model.config.model_type == "deepseek_v3"


# ==========================================================================
# Fix round (review + cold critic): device maps, config kwarg, buffer dtypes,
# tied weights, expert-shape validation
# ==========================================================================


def _record_model_to(monkeypatch):
    """A spy on DeepseekV3ForCausalLM.to recording the device instead of moving."""
    import torch
    from transformers import DeepseekV3ForCausalLM

    recorded = []

    def fake_to(self, device=None, *args, **kwargs):
        recorded.append(torch.device(device) if device is not None else None)
        return self

    monkeypatch.setattr(DeepseekV3ForCausalLM, "to", fake_to)
    return recorded


def test_resident_model_resolves_device_map_auto(resident_checkpoints, monkeypatch):
    import torch

    from soup_cli.utils.kimi_k2 import load_resident_model

    recorded = _record_model_to(monkeypatch)
    load_resident_model(
        resident_checkpoints["k2-bf16"], model_kwargs={"device_map": "auto"}
    )
    expected = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    assert recorded == [expected]


def _fake_cuda(monkeypatch, *, count):
    """Pin the loader's CUDA probes so both the serving and the refusing branch are
    testable on any host (the weights never move: ``.to`` is spied on separately)."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: count > 0)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    return torch


def test_resident_model_resolves_single_entry_device_map(resident_checkpoints, monkeypatch):
    torch = _fake_cuda(monkeypatch, count=1)

    from soup_cli.utils.kimi_k2 import load_resident_model

    recorded = _record_model_to(monkeypatch)
    load_resident_model(
        resident_checkpoints["k2-bf16"], model_kwargs={"device_map": {"": 0}}
    )
    assert recorded == [torch.device("cuda", 0)]


def test_resident_model_refuses_an_unavailable_cuda_device(resident_checkpoints, monkeypatch):
    _fake_cuda(monkeypatch, count=0)

    from soup_cli.utils.kimi_k2 import load_resident_model

    for device_map in ({"": 0}, "cuda:0"):
        with pytest.raises(ValueError) as caught:
            load_resident_model(
                resident_checkpoints["k2-bf16"], model_kwargs={"device_map": device_map}
            )
        message = str(caught.value)
        assert "cuda:0" in message
        assert "refusing before loading" in message


def test_resident_model_refuses_an_out_of_range_cuda_ordinal(resident_checkpoints, monkeypatch):
    _fake_cuda(monkeypatch, count=2)

    from soup_cli.utils.kimi_k2 import load_resident_model

    with pytest.raises(ValueError) as caught:
        load_resident_model(
            resident_checkpoints["k2-bf16"], model_kwargs={"device_map": {"": 5}}
        )
    message = str(caught.value)
    assert "cuda:5" in message
    assert "serves 2 CUDA device(s)" in message


def test_resident_model_refuses_a_multi_device_map(resident_checkpoints):
    from soup_cli.utils.kimi_k2 import load_resident_model

    with pytest.raises(NotImplementedError) as caught:
        load_resident_model(
            resident_checkpoints["k2-bf16"],
            model_kwargs={"device_map": {"": 0, "model.layers.0": 1}},
        )
    assert "device_map" in str(caught.value)


def test_resident_model_refuses_a_prebuilt_config(resident_checkpoints):
    import torch

    from soup_cli.utils.kimi_k2 import (
        UnsupportedKimiK2Error,
        load_resident_model,
    )

    with pytest.raises(UnsupportedKimiK2Error) as caught:
        load_resident_model(
            resident_checkpoints["k2-bf16"],
            model_kwargs={"dtype": torch.float32, "config": _native_tiny_config()},
        )
    message = str(caught.value)
    assert "config" in message
    assert "rope" in message
    assert "streamed route" in message


def test_resident_model_keeps_float_buffers_at_their_stored_dtype(resident_checkpoints):
    import torch

    from soup_cli.utils.kimi_k2 import load_resident_model

    model = load_resident_model(
        resident_checkpoints["k2-bf16"], model_kwargs={"dtype": torch.bfloat16}
    )
    # e_score_correction_bias is float32 in the checkpoint and in from_pretrained's own
    # load (_keep_in_fp32_modules); only parameters take the requested dtype.
    bias = model.model.layers[1].mlp.gate.e_score_correction_bias
    assert bias.dtype == torch.float32
    assert model.model.embed_tokens.weight.dtype == torch.bfloat16
    ids = _ids(torch)
    with torch.no_grad():
        logits = model(ids).logits
    assert logits.shape == (2, 12, 128)


@pytest.mark.parametrize("head_in_shards", [False, True], ids=["head-dropped", "head-shipped"])
def test_resident_model_loads_a_tied_checkpoint(tmp_path, head_in_shards):
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import DeepseekV3ForCausalLM

    from soup_cli.utils.kimi_k2 import load_resident_model

    torch.manual_seed(11)
    config = _native_tiny_config()
    config.tie_word_embeddings = True
    model = DeepseekV3ForCausalLM(config).to(torch.float32).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.data = parameter.data.to(torch.bfloat16)
    tied = tmp_path / "tied"
    model.save_pretrained(str(tied))
    # A tied checkpoint may omit the head or still ship it: both are the same matrix, and
    # a shipped one must not be treated as a tensor the native model has no parameter for.
    state = load_file(str(tied / "model.safetensors"))
    state.pop("lm_head.weight", None)
    if head_in_shards:
        # A copy, not the same storage: save_file refuses tensors that share memory.
        state["lm_head.weight"] = state["model.embed_tokens.weight"].clone()
    save_file(state, str(tied / "model.safetensors"), metadata={"format": "pt"})
    (tied / "config.json").write_text(
        json.dumps(dict(_MINIMAL_K2, tie_word_embeddings=True)), encoding="utf-8"
    )

    loaded = load_resident_model(str(tied), model_kwargs={"dtype": torch.float32})
    assert loaded.lm_head.weight is loaded.model.embed_tokens.weight
    ids = _ids(torch)
    with torch.no_grad():
        logits = loaded(ids).logits
    assert logits.shape == (2, 12, 128)


def test_resident_model_refuses_mismatched_expert_shapes(tmp_path, resident_checkpoints):
    import torch
    from safetensors.torch import load_file, save_file

    from soup_cli.utils.kimi_k2 import load_resident_model

    src = resident_checkpoints["k2-bf16"]
    dst = tmp_path / "bad-experts"
    dst.mkdir()
    state = load_file(os.path.join(src, "model.safetensors"))
    victim = "model.layers.1.mlp.experts.1.gate_proj.weight"
    state[victim] = torch.cat([state[victim], state[victim][:1]], dim=0)
    save_file(state, str(dst / "model.safetensors"), metadata={"format": "pt"})
    shutil.copyfile(os.path.join(src, "config.json"), dst / "config.json")

    with pytest.raises(ValueError) as caught:
        load_resident_model(str(dst), model_kwargs={"dtype": torch.float32})
    message = str(caught.value)
    assert victim in message
    assert "[33, 64]" in message
    assert "[32, 64]" in message


def test_rope_override_is_refused_on_the_resident_route(resident_checkpoints):
    from soup_cli.utils.kimi_k2 import resident_kimi_k2_rope_refusal

    reason = resident_kimi_k2_rope_refusal("moonshotai/Kimi-K2-Base", "yarn")
    assert reason is not None
    assert "rope_scaling_type 'yarn'" in reason
    assert "cannot be applied here" in reason
    # The remedy must not name the streamed route as one that applies it: K2 refuses the
    # field there too (#1697), so the message points at removing it instead.
    assert "#1697" in reason
    assert "which applies it" not in reason

    # A local K2 checkpoint takes the same route, so it takes the same refusal.
    local = resident_kimi_k2_rope_refusal(resident_checkpoints["k2-bf16"], "yarn")
    assert local is not None and "streamed route" in local

    # Nothing to refuse: no override, another model, or the streamed route owning the run.
    assert resident_kimi_k2_rope_refusal("moonshotai/Kimi-K2-Base", None) is None
    assert resident_kimi_k2_rope_refusal("meta-llama/Llama-3-8B", "yarn") is None
    assert (
        resident_kimi_k2_rope_refusal(
            "moonshotai/Kimi-K2-Base", "yarn", stream_layers=True
        )
        is None
    )


def test_sft_refuses_a_rope_override_before_fetching_a_tokenizer_or_config(monkeypatch):
    import transformers

    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.sft import SFTTrainerWrapper

    def _fetched(*args, **kwargs):
        raise AssertionError("a refused run must not fetch a tokenizer or a config")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", _fetched)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", _fetched)

    cfg = load_config_from_string(
        "base: moonshotai/Kimi-K2-Base\ntask: sft\ndata: {train: data.jsonl}\n"
        "training: {rope_scaling_type: yarn}\n"
    )
    wrapper = SFTTrainerWrapper(cfg, device="cpu")
    with pytest.raises(ValueError) as caught:
        wrapper._setup_transformers(cfg, cfg.training)
    assert "rope_scaling_type 'yarn'" in str(caught.value)


def _record_console_prints(monkeypatch, module):
    """Swap a trainer module's rich console for a recorder; returns the captured lines."""

    class _Recorder:
        def __init__(self):
            self.lines: list[str] = []

        def print(self, *args, **kwargs):
            self.lines.append(" ".join(str(arg) for arg in args))

    recorder = _Recorder()
    monkeypatch.setattr(module, "console", recorder)
    return recorder.lines


def _tokenizer_load_fails(monkeypatch):
    """Make every AutoTokenizer.from_pretrained raise, as a bare custom-code load does."""

    def _fails(*args, **kwargs):
        raise RuntimeError("tokenizer load failed")

    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", _fails)


def test_sft_tokenizer_failure_on_a_kimi_k2_base_prints_the_hint(tmp_path, monkeypatch):
    """The resident sft route's tokenizer failure names the K2 fix (#2104 final critic F3):
    the streamed route has printed this hint since #2107, while a default resident K2 run
    died bare at the tokenizer — no --trust-remote-code / tiktoken pointer at all."""
    import soup_cli.trainer.sft as sft_mod
    import soup_cli.utils.kimi_k2 as kimi_k2
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.sft import SFTTrainerWrapper

    # Without auto_map: wrapper init probes a LOCAL base for auto_map and refuses
    # before the tokenizer site; the hint test needs to reach the tokenizer.
    raw = {k: v for k, v in _MINIMAL_K2.items() if k != "auto_map"}
    base = Path(_write_config(tmp_path, raw)).as_posix()
    monkeypatch.setattr(kimi_k2, "_tiktoken_available", lambda: False)
    _tokenizer_load_fails(monkeypatch)
    printed = _record_console_prints(monkeypatch, sft_mod)

    cfg = load_config_from_string(f"base: {base}\ntask: sft\ndata: {{train: data.jsonl}}\n")
    wrapper = SFTTrainerWrapper(cfg, device="cpu")
    with pytest.raises(RuntimeError, match="tokenizer load failed"):
        wrapper._setup_transformers(cfg, cfg.training)

    output = "\n".join(printed)
    assert "--trust-remote-code" in output
    assert "tiktoken" in output


def test_grpo_tokenizer_failure_on_a_kimi_k2_base_prints_the_hint(tmp_path, monkeypatch):
    """The grpo resident route wires the same hint at its own tokenizer load site."""
    import soup_cli.trainer.grpo as grpo_mod
    import soup_cli.utils.kimi_k2 as kimi_k2
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.grpo import GRPOTrainerWrapper

    # Without auto_map: wrapper init probes a LOCAL base for auto_map and refuses
    # before the tokenizer site; the hint test needs to reach the tokenizer.
    raw = {k: v for k, v in _MINIMAL_K2.items() if k != "auto_map"}
    base = Path(_write_config(tmp_path, raw)).as_posix()
    monkeypatch.setattr(kimi_k2, "_tiktoken_available", lambda: False)
    _tokenizer_load_fails(monkeypatch)
    printed = _record_console_prints(monkeypatch, grpo_mod)

    cfg = load_config_from_string(f"base: {base}\ntask: grpo\ndata: {{train: data.jsonl}}\n")
    wrapper = GRPOTrainerWrapper(cfg, device="cpu")
    with pytest.raises(RuntimeError, match="tokenizer load failed"):
        wrapper._setup_transformers(cfg, cfg.training)

    output = "\n".join(printed)
    assert "--trust-remote-code" in output
    assert "tiktoken" in output


def test_sft_tokenizer_failure_off_kimi_k2_bases_stays_quiet(tmp_path, monkeypatch):
    """The hint is K2-gated: another base's tokenizer failure re-raises with no hint text."""
    import soup_cli.trainer.sft as sft_mod
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.sft import SFTTrainerWrapper

    llama = {
        "model_type": "llama",
        "hidden_size": 8,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "intermediate_size": 16,
        "vocab_size": 32,
        "num_hidden_layers": 1,
    }
    base = Path(_write_config(tmp_path, llama)).as_posix()
    _tokenizer_load_fails(monkeypatch)
    printed = _record_console_prints(monkeypatch, sft_mod)

    cfg = load_config_from_string(f"base: {base}\ntask: sft\ndata: {{train: data.jsonl}}\n")
    wrapper = SFTTrainerWrapper(cfg, device="cpu")
    with pytest.raises(RuntimeError, match="tokenizer load failed"):
        wrapper._setup_transformers(cfg, cfg.training)

    assert all("--trust-remote-code" not in line for line in printed)


# ==========================================================================
# Maintainer follow-up 2026-11-12, item (a): the committed published tensor
# names cover what the resident loader writes (#2104)
# =========================================================================
#
# The maintainer: "Do not fetch model.safetensors.index.json inside a test:
# commit the list of names or a header map." The list lives at
# tests/fixtures/kimi_k2/published_tensor_names.txt: the weight_map keys of
# moonshotai/Kimi-K2-Base's model.safetensors.index.json at the admission
# gate's pinned revision; the file's header comment records the revision, the
# index's own sha256, the fetch date and the name count. These tests are the
# only consumers and read the committed file, never the network. The
# documented mutation path is editing that file: renaming a published name
# must make the coverage check fail.

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "kimi_k2"
_PUBLISHED_NAMES_TXT = _FIXTURES_DIR / "published_tensor_names.txt"
#: Recorded in the fixture's header: the pinned index's name count.
_PUBLISHED_NAME_COUNT = 139_644


def _published_names():
    """The committed published tensor names (header comments stripped)."""
    lines = _PUBLISHED_NAMES_TXT.read_text(encoding="utf-8").splitlines()
    return [line for line in lines if line and not line.startswith("#")]


def test_the_committed_published_names_match_the_pinned_index():
    names = _published_names()
    assert len(names) == _PUBLISHED_NAME_COUNT
    assert names == sorted(set(names))
    # Structural sanity probes. Two correct the first-guess reading (both are recorded in
    # the fixture's header): layer 0 is dense — the first routed-expert layer is layer 1 —
    # and K2-Base publishes lm_head.weight untied, so it IS in the list.
    for present in (
        "model.embed_tokens.weight",
        "model.layers.0.mlp.gate_proj.weight",  # dense layer 0: no experts under it
        "model.layers.1.mlp.experts.0.gate_proj.weight",
        "model.layers.1.mlp.experts.0.gate_proj.weight_scale_inv",
        "model.layers.60.mlp.experts.383.down_proj.weight",
        "model.layers.60.mlp.experts.383.down_proj.weight_scale_inv",
        "model.layers.1.mlp.gate.e_score_correction_bias",
        "model.norm.weight",
        "lm_head.weight",
    ):
        assert present in names, present
    assert "model.layers.0.mlp.experts.0.gate_proj.weight" not in names


def _published_subset_cases():
    """(published name, expected key in the loaded state dict, expected shape).

    Shapes derive from the tiny dims (``_TINY``). Per-expert published spellings map to
    the fused native parameters the loader writes (``...mlp.experts.gate_up_proj`` /
    ``...mlp.experts.down_proj``), and published layer numbers (1..60) map onto the tiny
    model's single MoE layer (1): the checkpoint naming scheme, not the layer count, is
    what this pins.
    """
    t = _TINY
    h = t["hidden_size"]
    i = t["intermediate_size"]
    m = t["moe_intermediate_size"]
    e = t["n_routed_experts"]
    a = t["num_attention_heads"]
    kv = t["kv_lora_rank"]
    q = t["q_lora_rank"]
    head = t["qk_nope_head_dim"] + t["qk_rope_head_dim"]
    v = t["v_head_dim"]
    vocab = t["vocab_size"]
    gate_up = "model.layers.1.mlp.experts.gate_up_proj"
    down = "model.layers.1.mlp.experts.down_proj"
    return [
        ("model.embed_tokens.weight", None, (vocab, h)),
        ("model.layers.0.self_attn.q_a_proj.weight", None, (q, h)),
        ("model.layers.0.self_attn.q_a_layernorm.weight", None, (q,)),
        ("model.layers.0.self_attn.q_b_proj.weight", None, (a * head, q)),
        # K2's published MQA spelling: one kv down-projection with the rope key baked in.
        (
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
            None,
            (kv + t["qk_rope_head_dim"], h),
        ),
        ("model.layers.0.self_attn.kv_a_layernorm.weight", None, (kv,)),
        (
            "model.layers.0.self_attn.kv_b_proj.weight",
            None,
            (a * (t["qk_nope_head_dim"] + v), kv),
        ),
        ("model.layers.0.self_attn.o_proj.weight", None, (h, a * v)),
        ("model.layers.0.input_layernorm.weight", None, (h,)),
        ("model.layers.0.post_attention_layernorm.weight", None, (h,)),
        ("model.layers.0.mlp.gate_proj.weight", None, (i, h)),  # dense layer 0
        ("model.layers.0.mlp.up_proj.weight", None, (i, h)),
        ("model.layers.0.mlp.down_proj.weight", None, (h, i)),
        ("model.layers.1.mlp.gate.weight", None, (e, h)),  # the router
        ("model.layers.1.mlp.shared_experts.gate_proj.weight", None, (m, h)),
        ("model.layers.1.mlp.shared_experts.up_proj.weight", None, (m, h)),
        ("model.layers.1.mlp.shared_experts.down_proj.weight", None, (h, m)),
        # Routed experts: per-expert checkpoint spelling -> the fused stacked parameter;
        # expert 0 and the last (tiny 7, published 383) on both projections.
        ("model.layers.1.mlp.experts.0.gate_proj.weight", gate_up, (e, 2 * m, h)),
        ("model.layers.1.mlp.experts.7.up_proj.weight", gate_up, (e, 2 * m, h)),
        ("model.layers.60.mlp.experts.383.gate_proj.weight", gate_up, (e, 2 * m, h)),
        ("model.layers.1.mlp.experts.0.down_proj.weight", down, (e, h, m)),
        ("model.layers.60.mlp.experts.383.down_proj.weight", down, (e, h, m)),
        ("model.norm.weight", None, (h,)),
        ("lm_head.weight", None, (vocab, h)),
    ]


_PUBLISHED_SUBSET = _published_subset_cases()


def _assert_published_subset_in(names):
    """Every representative published name is in the committed list (fails loudly,
    naming the misses — this is what a mutated file must trip)."""
    present = set(names)
    missing = [name for name, _key, _shape in _PUBLISHED_SUBSET if name not in present]
    assert not missing, f"published names missing from the committed list: {missing}"


@pytest.fixture(scope="module")
def tiny_k2_built(tmp_path_factory):
    """The shared seeded tiny K2 checkpoints: bf16 + FP8 + dequantised materialisation."""
    from tests.kimi_k2_tiny import build_tiny_k2_checkpoints

    return build_tiny_k2_checkpoints(tmp_path_factory.mktemp("kimi-k2-task6"))


@pytest.fixture(scope="module")
def tiny_k2_loaded_state(tiny_k2_built):
    import torch

    from soup_cli.utils.kimi_k2 import load_resident_model

    model = load_resident_model(
        tiny_k2_built["bf16"][0], model_kwargs={"dtype": torch.bfloat16}
    )
    return model.state_dict()


@pytest.mark.parametrize(
    ("published", "fused_key", "shape"),
    _PUBLISHED_SUBSET,
    ids=[published for published, _key, _shape in _PUBLISHED_SUBSET],
)
def test_published_names_cover_the_loaded_state_dict(
    tiny_k2_loaded_state, published, fused_key, shape
):
    """Coverage (maintainer item a): each representative published spelling the loader
    consumes has its counterpart in the loaded state dict — fused for per-expert tensors,
    identical otherwise — at the shape the tiny dims imply."""
    _assert_published_subset_in(_published_names())
    key = fused_key if fused_key is not None else published
    assert key in tiny_k2_loaded_state, key
    assert tuple(tiny_k2_loaded_state[key].shape) == shape, key


def test_published_names_coverage_fails_when_the_file_is_mutated():
    """The mutation-flavoured check: the coverage test above genuinely reads the committed
    file, so the documented mutation path is editing the file — rename one published name
    and the coverage check must fail (a constant baked into the test could not fail this
    way)."""
    names = _published_names()
    victim = "model.layers.1.mlp.experts.0.gate_proj.weight"
    tampered = [
        "model.layers.1.mlp.experts.0.gate_project.weight" if name == victim else name
        for name in names
    ]
    assert victim in names
    with pytest.raises(AssertionError):
        _assert_published_subset_in(tampered)


# ==========================================================================
# Maintainer follow-up 2026-11-12, item (c): the FP8 resident load is bit for
# bit the dequantised materialisation (#2104)
# =========================================================================
#
# Streamed precedent: test_streamed_from_fp8_matches_the_dequantised_resident_bit_exactly
# (tests/test_fp8_block_streaming.py) pins the streamed build's logits/loss/grads against
# a resident model holding the dequantised weights. This cell is the resident-route
# counterpart, to the same scope: load_resident_model on the FP8 checkpoint must
# materialise, BIT FOR BIT, the same state dict a plain native from_pretrained produces on
# the dequantised folder the fixtures' builder writes (built["fp8"] = (FP8 K2 folder,
# dequantised folder)), and the two models must then compute the same logits and loss.
#
# Buffers are included: e_score_correction_bias is float32 in both loads (the loader keeps
# stored buffer dtypes), so the bit-for-bit claim covers the whole state dict. Parameters
# are bfloat16 in both, which is why the comparison also checks dtypes.


def _resident_fp8_vs_reference(fp8_dir, deq_dir):
    import torch
    from transformers import DeepseekV3ForCausalLM

    from soup_cli.utils.kimi_k2 import load_resident_model

    got = load_resident_model(str(fp8_dir), model_kwargs={"dtype": torch.bfloat16})
    want = DeepseekV3ForCausalLM.from_pretrained(str(deq_dir), dtype=torch.bfloat16)
    return got, want


def _assert_state_bit_identical(got, want):
    import torch

    got_state = got.state_dict()
    want_state = want.state_dict()
    assert set(got_state) == set(want_state)
    for name in sorted(got_state):
        left, right = got_state[name], want_state[name]
        assert left.dtype == right.dtype, (name, left.dtype, right.dtype)
        assert torch.equal(left, right), name


def test_resident_fp8_load_is_bit_for_bit_the_dequantised_materialisation(tiny_k2_built):
    """Maintainer item c: the FP8 resident load equals, bit for bit, a resident native
    model holding the same dequantised weights (streamed-side pin:
    test_streamed_from_fp8_matches_the_dequantised_resident_bit_exactly) — weights, then
    the forward pass that consumes them."""
    import torch

    got, want = _resident_fp8_vs_reference(*tiny_k2_built["fp8"])
    _assert_state_bit_identical(got, want)

    ids = _ids(torch)
    for model in (got, want):
        model.eval()
    with torch.no_grad():
        got_out = got(input_ids=ids, labels=ids)
        want_out = want(input_ids=ids, labels=ids)
    assert torch.equal(got_out.logits, want_out.logits)
    assert torch.equal(got_out.loss, want_out.loss)


def test_resident_fp8_load_bit_exactness_catches_a_flipped_scale(tmp_path, tiny_k2_built):
    """The mandatory mutation check for item c: flip one FP8 block scale in the
    checkpoint and the bit-for-bit comparison must fail — i.e. the parity cell above
    would catch this mutation, so it cannot pass on a stale or shortcut dequant."""
    from safetensors.torch import load_file, save_file

    fp8_dir, deq_dir = tiny_k2_built["fp8"]
    mutated = tmp_path / "fp8-flipped-scale"
    mutated.mkdir()
    state = load_file(os.path.join(fp8_dir, "model.safetensors"))
    victim = sorted(key for key in state if key.endswith("_scale_inv"))[0]
    state[victim] = state[victim] * 2.0
    save_file(state, str(mutated / "model.safetensors"), metadata={"format": "pt"})
    shutil.copyfile(os.path.join(fp8_dir, "config.json"), mutated / "config.json")

    got, want = _resident_fp8_vs_reference(str(mutated), deq_dir)
    with pytest.raises(AssertionError):
        _assert_state_bit_identical(got, want)


def test_resident_model_refuses_fp8_bytes_without_an_fp8_config(tmp_path, resident_checkpoints):
    """A tensor stored as FP8 under a config that declares no FP8 format is refused before
    any weight is read: FP8 bytes have no meaning without their block scales, and casting
    them as numbers is the silent-garbage path (#2104 review, C3)."""
    import torch
    from safetensors.torch import load_file, save_file

    from soup_cli.utils.kimi_k2 import load_resident_model

    src = resident_checkpoints["k2-bf16"]
    dst = tmp_path / "stray-fp8"
    dst.mkdir()
    state = load_file(os.path.join(src, "model.safetensors"))
    victim = "model.layers.0.input_layernorm.weight"
    state[victim] = state[victim].to(torch.float8_e5m2)
    save_file(state, str(dst / "model.safetensors"), metadata={"format": "pt"})
    shutil.copyfile(os.path.join(src, "config.json"), dst / "config.json")

    with pytest.raises(ValueError) as caught:
        load_resident_model(str(dst), model_kwargs={"dtype": torch.float32})
    message = str(caught.value)
    assert victim in message
    assert "holds FP8 tensors or block scales" in message


def test_a_dtype_the_route_cannot_read_is_refused_by_name():
    """The dtype table's fallback: a spelling the header reader gains later must be a named
    refusal naming the tensor, never a bare ``KeyError`` (#2104 review, C3)."""
    import torch

    from soup_cli.utils.kimi_k2 import _resident_torch_dtype

    with pytest.raises(ValueError) as caught:
        _resident_torch_dtype("model.norm.weight", "U16", torch)
    message = str(caught.value)
    assert "model.norm.weight" in message
    assert "stored as U16" in message
    assert "cannot read" in message

    with pytest.raises(ValueError) as fp8:
        _resident_torch_dtype("model.layers.1.mlp.experts.0.down_proj.weight", "F8_E5M2", torch)
    assert "no FP8 block plan" in str(fp8.value)


def test_resident_model_refuses_a_tensor_spelled_twice(tmp_path, resident_checkpoints):
    """Two shards declaring the same tensor is refused where the route reads its headers,
    not resolved by shard order (#2104 review, C3)."""
    import torch
    from safetensors.torch import load_file, save_file

    from soup_cli.utils.kimi_k2 import load_resident_model

    src = resident_checkpoints["k2-bf16"]
    dst = tmp_path / "spelled-twice"
    dst.mkdir()
    state = load_file(os.path.join(src, "model.safetensors"))
    first, second = {}, {}
    for index, key in enumerate(sorted(state)):
        (first if index % 2 else second)[key] = state[key]
    shared = "model.norm.weight"
    first[shared] = state[shared]
    second[shared] = state[shared]
    save_file(first, str(dst / "model-00001-of-00002.safetensors"), metadata={"format": "pt"})
    save_file(second, str(dst / "model-00002-of-00002.safetensors"), metadata={"format": "pt"})
    shutil.copyfile(os.path.join(src, "config.json"), dst / "config.json")

    with pytest.raises(ValueError) as caught:
        load_resident_model(str(dst), model_kwargs={"dtype": torch.float32})
    message = str(caught.value)
    assert "spells one streamed tensor twice" in message
    assert shared in message


# ==========================================================================
# Cold-review follow-ups (2026-11-12, adversarial pass): the paths the suite
# never exercised, and the checks it never pinned
# ==========================================================================


def _k2_variant(tmp_path, src, *, drop=(), add=None, name="variant"):
    """A copy of a K2 folder with tensors removed/added, config unchanged."""
    from safetensors.torch import load_file, save_file

    dst = tmp_path / name
    dst.mkdir()
    state = load_file(os.path.join(src, "model.safetensors"))
    for key in drop:
        state.pop(key, None)
    if add:
        state.update(add)
    save_file(state, str(dst / "model.safetensors"), metadata={"format": "pt"})
    shutil.copyfile(os.path.join(src, "config.json"), dst / "config.json")
    return str(dst)


@pytest.mark.parametrize(
    "mutation", ["missing-projection", "missing-expert", "extra-expert"]
)
def test_resident_model_refuses_an_incomplete_expert_set(
    tmp_path, resident_checkpoints, mutation
):
    """The per-expert completeness check is production code: removing it let a partial
    checkpoint reach the fuse (missing projection -> bare KeyError, extra expert -> silently
    dropped). Each of the three shapes must be refused by name."""
    import torch
    from safetensors.torch import load_file

    from soup_cli.utils.kimi_k2 import load_resident_model

    src = resident_checkpoints["k2-bf16"]
    prefix = "model.layers.1.mlp.experts"
    projections = ("gate_proj", "up_proj", "down_proj")
    drop, add = [], None
    if mutation == "missing-projection":
        drop = [f"{prefix}.3.up_proj.weight"]
    elif mutation == "missing-expert":
        drop = [f"{prefix}.5.{projection}.weight" for projection in projections]
    else:
        state = load_file(os.path.join(src, "model.safetensors"))
        add = {
            f"{prefix}.8.{projection}.weight": state[
                f"{prefix}.0.{projection}.weight"
            ].clone()
            for projection in projections
        }

    path = _k2_variant(tmp_path, src, drop=drop, add=add)
    with pytest.raises(ValueError) as caught:
        load_resident_model(path, model_kwargs={"dtype": torch.float32})
    message = str(caught.value)
    assert "not every index 0.." in message
    assert "The checkpoint is damaged or not a complete Kimi K2 copy." in message


def test_resident_model_refuses_a_truncated_up_proj(tmp_path, resident_checkpoints):
    """A shorter up_proj broadcasts into its slice of the fused gate_up tensor and silently
    duplicates rows: the per-expert shapes must be checked against each other, not only
    expert-to-expert."""
    import torch
    from safetensors.torch import load_file

    from soup_cli.utils.kimi_k2 import load_resident_model

    src = resident_checkpoints["k2-bf16"]
    prefix = "model.layers.1.mlp.experts"
    state = load_file(os.path.join(src, "model.safetensors"))
    drop, add = [], {}
    for index in range(_TINY["n_routed_experts"]):
        key = f"{prefix}.{index}.up_proj.weight"
        drop.append(key)
        add[key] = state[key][:1].clone()

    path = _k2_variant(tmp_path, src, drop=drop, add=add)
    with pytest.raises(ValueError) as caught:
        load_resident_model(path, model_kwargs={"dtype": torch.float32})
    message = str(caught.value)
    assert "pairs gate_proj" in message
    assert "up_proj" in message


def test_resident_model_refuses_a_fused_and_per_expert_spelling_together(
    tmp_path, resident_checkpoints
):
    """Both spellings in one layer would fight over the same parameters: refuse instead of
    letting the fuse overwrite whatever the checkpoint also shipped fused."""
    import torch

    from soup_cli.utils.kimi_k2 import load_resident_model

    routed = _TINY["n_routed_experts"]
    moe = _TINY["moe_intermediate_size"]
    hidden = _TINY["hidden_size"]
    add = {
        "model.layers.1.mlp.experts.gate_up_proj": torch.zeros(
            (routed, 2 * moe, hidden), dtype=torch.bfloat16
        ),
        "model.layers.1.mlp.experts.down_proj": torch.zeros(
            (routed, hidden, moe), dtype=torch.bfloat16
        ),
    }
    path = _k2_variant(tmp_path, resident_checkpoints["k2-bf16"], add=add)
    with pytest.raises(ValueError) as caught:
        load_resident_model(path, model_kwargs={"dtype": torch.float32})
    message = str(caught.value)
    assert "carries both per-expert tensors and a fused" in message


def test_resident_model_refuses_a_tied_checkpoint_whose_head_differs(tmp_path):
    """A checkpoint that declares tied embeddings but ships a different head is
    mislabelled; dropping the tensor silently would train on the embedding."""
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import DeepseekV3ForCausalLM

    from soup_cli.utils.kimi_k2 import load_resident_model

    torch.manual_seed(11)
    config = _native_tiny_config()
    config.tie_word_embeddings = True
    model = DeepseekV3ForCausalLM(config).to(torch.float32).eval()
    tied = tmp_path / "tied-mismatch"
    model.save_pretrained(str(tied))
    state = load_file(str(tied / "model.safetensors"))
    state["lm_head.weight"] = torch.zeros_like(state["model.embed_tokens.weight"])
    save_file(state, str(tied / "model.safetensors"), metadata={"format": "pt"})
    (tied / "config.json").write_text(
        json.dumps(dict(_MINIMAL_K2, tie_word_embeddings=True)), encoding="utf-8"
    )

    with pytest.raises(ValueError) as caught:
        load_resident_model(str(tied), model_kwargs={"dtype": torch.float32})
    assert "differs from model.embed_tokens.weight" in str(caught.value)


def test_the_config_loader_never_runs_repository_config_code(monkeypatch, resident_checkpoints):
    """transformers prompts on an ``auto_map`` config unless told otherwise, and a "y" would
    execute the repository's configuration module — bypassing the guard. The loader must
    pass ``trust_remote_code=False`` itself."""
    import transformers
    from transformers import DeepseekV3Config

    from soup_cli.utils import kimi_k2

    seen = {}
    real = transformers.AutoConfig.from_pretrained

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", spy)
    config = kimi_k2.load_resident_model_config(resident_checkpoints["k2-bf16"])
    assert isinstance(config, DeepseekV3Config)
    assert seen["trust_remote_code"] is False

    with pytest.raises(ValueError) as caught:
        kimi_k2.load_resident_model_config(
            resident_checkpoints["k2-bf16"], trust_remote_code=True
        )
    assert "trust_remote_code is not honoured here" in str(caught.value)


def test_resident_model_applies_attn_implementation(resident_checkpoints):
    import torch

    from soup_cli.utils.kimi_k2 import load_resident_model

    model = load_resident_model(
        resident_checkpoints["k2-bf16"],
        model_kwargs={"dtype": torch.float32, "attn_implementation": "eager"},
    )
    assert model.config._attn_implementation == "eager"


def test_resident_model_honours_a_plain_device_string(resident_checkpoints, monkeypatch):
    import torch

    from soup_cli.utils.kimi_k2 import load_resident_model

    recorded = _record_model_to(monkeypatch)
    load_resident_model(
        resident_checkpoints["k2-bf16"],
        model_kwargs={"dtype": torch.float32, "device_map": "cpu"},
    )
    assert recorded == [torch.device("cpu")]


def test_resident_model_forwards_the_revision_on_the_hub_path(
    monkeypatch, resident_checkpoints
):
    import huggingface_hub
    import torch

    from soup_cli.utils import kimi_k2

    seen = {}
    real_config_loader = kimi_k2.load_resident_model_config

    def fake_snapshot(*, repo_id, revision=None, allow_patterns=None):
        seen["snapshot"] = (repo_id, revision, tuple(allow_patterns or ()))
        return resident_checkpoints["k2-bf16"]

    def fake_config_loader(model_id, *, revision=None, **kwargs):
        seen["config"] = (model_id, revision)
        return real_config_loader(resident_checkpoints["k2-bf16"])

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    monkeypatch.setattr(kimi_k2, "load_resident_model_config", fake_config_loader)
    kimi_k2.load_resident_model(
        "moonshotai/Kimi-K2-Base",
        model_kwargs={"dtype": torch.float32, "revision": "0123456789abcdef"},
    )
    assert seen["snapshot"][0] == "moonshotai/Kimi-K2-Base"
    assert seen["snapshot"][1] == "0123456789abcdef"
    assert "*.safetensors" in seen["snapshot"][2]
    assert seen["config"] == ("moonshotai/Kimi-K2-Base", "0123456789abcdef")


def test_resident_model_finds_a_scale_in_another_shard(tmp_path, resident_checkpoints):
    """Weights and their ``_scale_inv`` need not live in the same file; splitting them must
    give the same model, bit for bit."""
    import torch
    from safetensors.torch import load_file, save_file

    from soup_cli.utils.kimi_k2 import load_resident_model

    src = resident_checkpoints["k2-fp8"]
    dst = tmp_path / "split-fp8"
    dst.mkdir()
    state = load_file(os.path.join(src, "model.safetensors"))
    scales = {key: value for key, value in state.items() if key.endswith("_scale_inv")}
    weights = {key: value for key, value in state.items() if not key.endswith("_scale_inv")}
    save_file(weights, str(dst / "model-00001-of-00002.safetensors"), metadata={"format": "pt"})
    save_file(scales, str(dst / "model-00002-of-00002.safetensors"), metadata={"format": "pt"})
    shutil.copyfile(os.path.join(src, "config.json"), dst / "config.json")

    split = load_resident_model(str(dst), model_kwargs={"dtype": torch.float32})
    whole = load_resident_model(src, model_kwargs={"dtype": torch.float32})
    split_state, whole_state = split.state_dict(), whole.state_dict()
    assert set(split_state) == set(whole_state)
    for name in sorted(whole_state):
        assert torch.equal(split_state[name], whole_state[name]), name


def test_the_manifest_pins_the_recording_it_claims():
    """The manifest is evidence only if its pins are asserted, not merely written down."""
    manifest = json.loads(
        (_FIXTURES_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["transformers_pin"] == "4.53.3"
    assert manifest["recording_env"]["transformers"] == "4.53.3"
    assert manifest["recording_env"]["torch"] == "2.15.0+cpu"
    assert manifest["modeling_deepseek.py"]["sha256"] == (
        "ce753fc9076ce13ce9557e680e2f5bd68ae8929a8225772968a9a22a1d17b7d4"
    )
    assert manifest["modeling_deepseek.py"]["revision"] == (
        "ce72df012259dcc55d945e890f815fe7ef69159c"
    )
    assert manifest["configuration_deepseek.py"]["sha256"] == (
        "54f29dd0d56e9a13f639c703e524daa71794e360050012aea3788ad7cfc38b17"
    )
    assert manifest["inputs"]["seed"] == 977
    assert manifest["inputs"]["lengths"] == [8, 16, 33]
    assert len(manifest["inputs"]["ids"]) == 3
    assert manifest["recording_command"].endswith("kimi_k2_resident_parity_record.py")
