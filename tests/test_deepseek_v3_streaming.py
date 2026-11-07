"""Layer streaming for the DeepSeek-V3 architecture (MLA + DeepSeekMoE).

The gate for admitting an architecture is resident-vs-streamed parity on a
real (tiny) model, forward and backward stated separately, the same rule every
other entry of ``SUPPORTED_STREAM_ARCHS`` passed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

_LORA_TARGETS = ["q_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]


# --------------------------------------------------------------------------
# allowlist
# --------------------------------------------------------------------------
def test_deepseek_v3_is_an_admitted_stream_arch():
    from soup_cli.utils.layer_stream import SUPPORTED_STREAM_ARCHS, stream_arch_of

    assert "deepseek_v3" in SUPPORTED_STREAM_ARCHS
    assert stream_arch_of(SimpleNamespace(model_type="deepseek_v3")) == "deepseek_v3"


def test_kimi_k2_is_admitted_through_the_guarded_config_loader(tmp_path):
    """Kimi-K2-Base / -Instruct declare ``model_type: kimi_k2``, which AutoConfig refuses as
    custom code. The streamed model's config loader (``utils/kimi_k2.py``) reads it into the
    native ``DeepseekV3Config`` after a guard, so a K2 that passes streams as this family. The
    gate is ``tests/test_kimi_k2_streaming.py`` (benchmarks/gate-kimi-k2-admission.md)."""
    import json

    pytest.importorskip("transformers")
    from soup_cli.utils.kimi_k2 import load_stream_model_config
    from soup_cli.utils.layer_stream import stream_arch_of
    from tests.test_kimi_k2_streaming import K2_BASE_CONFIG

    (tmp_path / "config.json").write_text(json.dumps(K2_BASE_CONFIG), encoding="utf-8")
    assert stream_arch_of(load_stream_model_config(str(tmp_path))) == "deepseek_v3"


def test_kimi_k2_that_fails_the_guard_is_refused_by_name(tmp_path):
    """Kimi-K2-Thinking (compressed-tensors INT4) among them: the field and value are named."""
    import json

    pytest.importorskip("transformers")
    from soup_cli.utils.kimi_k2 import UnsupportedKimiK2Error, load_stream_model_config
    from tests.test_kimi_k2_streaming import K2_BASE_CONFIG, K2_THINKING_CHANGES

    thinking = dict(K2_BASE_CONFIG, **K2_THINKING_CHANGES)
    (tmp_path / "config.json").write_text(json.dumps(thinking), encoding="utf-8")
    with pytest.raises(UnsupportedKimiK2Error) as caught:
        load_stream_model_config(str(tmp_path))
    assert "quantization_config.quant_method='compressed-tensors'" in str(caught.value)
    assert "kimi_k2" in str(caught.value)


def test_a_kimi_k2_config_that_bypassed_the_loader_is_still_refused_naming_the_allowlist():
    """``kimi_k2`` itself is not an admitted name: a config still carrying it (K2's own remote
    config class keeps it) did not pass the guard, and the message points at the family."""
    from soup_cli.utils.layer_stream import stream_arch_of

    with pytest.raises(ValueError, match="deepseek_v3") as caught:
        stream_arch_of(SimpleNamespace(model_type="kimi_k2"))
    assert "kimi_k2" in str(caught.value)


# --------------------------------------------------------------------------
# sharder: synthetic safetensors, no model
# --------------------------------------------------------------------------
def _write_layer_checkpoint(path, *, n_layers, n_experts, extra_layer_keys=()):
    import torch
    from safetensors.torch import save_file

    hidden, inter = 4, 3
    state = {
        "model.embed_tokens.weight": torch.zeros(8, hidden),
        "model.norm.weight": torch.ones(hidden),
        "lm_head.weight": torch.zeros(8, hidden),
    }
    for layer in range(n_layers):
        prefix = f"model.layers.{layer}."
        state[prefix + "input_layernorm.weight"] = torch.ones(hidden)
        state[prefix + "mlp.gate.weight"] = torch.full((n_experts, hidden), float(layer))
        state[prefix + "mlp.gate.e_score_correction_bias"] = torch.tensor(
            [0.1 + 1e-3 * expert for expert in range(n_experts)], dtype=torch.float32
        )
        for expert in range(n_experts):
            base = 1000.0 * layer + 10.0 * expert
            state[prefix + f"mlp.experts.{expert}.gate_proj.weight"] = torch.full(
                (inter, hidden), base + 1.0
            )
            state[prefix + f"mlp.experts.{expert}.up_proj.weight"] = torch.full(
                (inter, hidden), base + 2.0
            )
            state[prefix + f"mlp.experts.{expert}.down_proj.weight"] = torch.full(
                (hidden, inter), base + 3.0
            )
    for key in extra_layer_keys:
        state[key] = torch.ones(hidden)
    path.mkdir(parents=True, exist_ok=True)
    save_file(state, str(path / "model.safetensors"))
    (path / "config.json").write_text('{"model_type": "deepseek_v3"}', encoding="utf-8")
    return str(path)


def _layer_file(shards, index, layer):
    from soup_cli.utils.layer_shard import layer_paths

    return layer_paths(str(shards), index)[layer]


def test_per_expert_keys_stack_gate_first_in_numeric_expert_order(tmp_path):
    """11 experts: a lexical sort would put expert 10 before expert 2."""
    import torch
    from safetensors.torch import load_file

    from soup_cli.utils.layer_shard import shard_checkpoint

    weights = _write_layer_checkpoint(tmp_path / "w", n_layers=2, n_experts=11)
    index = shard_checkpoint(
        weights, str(tmp_path / "s"), dtype="float32", arch="deepseek_v3", num_hidden_layers=2
    )
    for layer in range(2):
        tensors = load_file(_layer_file(tmp_path / "s", index, layer))
        assert not any(".experts.0." in key for key in tensors)
        gate_up = tensors["mlp.experts.gate_up_proj"]
        down = tensors["mlp.experts.down_proj"]
        assert tuple(gate_up.shape) == (11, 6, 4)
        assert tuple(down.shape) == (11, 4, 3)
        for expert in range(11):
            base = 1000.0 * layer + 10.0 * expert
            assert torch.equal(gate_up[expert, :3], torch.full((3, 4), base + 1.0))
            assert torch.equal(gate_up[expert, 3:], torch.full((3, 4), base + 2.0))
            assert torch.equal(down[expert], torch.full((4, 3), base + 3.0))


def test_mtp_layers_past_num_hidden_layers_are_dropped(tmp_path):
    from soup_cli.utils.layer_shard import shard_checkpoint

    mtp = ("model.layers.2.enorm.weight", "model.layers.2.hnorm.weight")
    weights = _write_layer_checkpoint(
        tmp_path / "w", n_layers=2, n_experts=2, extra_layer_keys=mtp
    )
    index = shard_checkpoint(
        weights, str(tmp_path / "s"), dtype="float32", arch="deepseek_v3", num_hidden_layers=2
    )
    assert index.n_layers == 2


def test_mtp_control_without_the_layer_count_keeps_the_extra_layer(tmp_path):
    """The control for the test above: the extra layer is real input."""
    from soup_cli.utils.layer_shard import shard_checkpoint

    mtp = ("model.layers.2.enorm.weight", "model.layers.2.hnorm.weight")
    weights = _write_layer_checkpoint(
        tmp_path / "w", n_layers=2, n_experts=2, extra_layer_keys=mtp
    )
    index = shard_checkpoint(weights, str(tmp_path / "s"), dtype="float32", arch="deepseek_v3")
    assert index.n_layers == 3


@pytest.mark.parametrize("bad", [True, 0, -1, 2.0, "2", 100000])
def test_num_hidden_layers_is_validated(tmp_path, bad):
    from soup_cli.utils.layer_shard import shard_checkpoint

    weights = _write_layer_checkpoint(tmp_path / "w", n_layers=2, n_experts=2)
    with pytest.raises(ValueError, match="num_hidden_layers must be an int"):
        shard_checkpoint(
            weights, str(tmp_path / "s"), dtype="float32", arch="deepseek_v3",
            num_hidden_layers=bad,
        )


def test_router_correction_bias_stays_fp32_under_a_bf16_stream(tmp_path):
    """transformers keeps ``e_score_correction_bias`` in fp32 on load; a bf16
    shard would route with rounded values the resident model never sees."""
    import torch
    from safetensors.torch import load_file

    from soup_cli.utils.layer_shard import shard_checkpoint

    weights = _write_layer_checkpoint(tmp_path / "w", n_layers=2, n_experts=4)
    index = shard_checkpoint(
        weights, str(tmp_path / "s"), dtype="bfloat16", arch="deepseek_v3", num_hidden_layers=2
    )
    tensors = load_file(_layer_file(tmp_path / "s", index, 1))
    bias = tensors["mlp.gate.e_score_correction_bias"]
    assert bias.dtype == torch.float32
    assert torch.equal(bias, torch.tensor([0.1, 0.101, 0.102, 0.103], dtype=torch.float32))
    assert tensors["mlp.gate.weight"].dtype == torch.bfloat16
    assert tensors["mlp.experts.gate_up_proj"].dtype == torch.bfloat16


@pytest.mark.parametrize("fp8_key", ["weight", "scale"])
def test_fp8_checkpoint_is_refused_before_anything_is_written(tmp_path, fp8_key):
    """DeepSeek-V3 / Kimi K2 publish FP8 weights plus ``weight_scale_inv`` block
    scales. A cast would keep the raw e4m3 values and drop the scales, so an FP8
    tensor or a block scale is accepted only as that block format, declared by
    ``config.json`` (and then dequantised at shard time, see
    ``tests/test_fp8_block_streaming.py``). This checkpoint has no config, so either
    spelling is refused naming the supported format."""
    import torch
    from safetensors.torch import save_file

    from soup_cli.utils.layer_shard import shard_checkpoint

    weights = tmp_path / "w"
    weights.mkdir()
    state = {
        "model.embed_tokens.weight": torch.zeros(8, 4),
        "model.layers.0.input_layernorm.weight": torch.ones(4),
        "model.layers.0.self_attn.o_proj.weight": torch.zeros(4, 4),
    }
    if fp8_key == "weight":
        state["model.layers.0.self_attn.o_proj.weight"] = torch.zeros(4, 4).to(
            torch.float8_e4m3fn
        )
    else:
        state["model.layers.0.self_attn.o_proj.weight_scale_inv"] = torch.ones(1, 1)
    save_file(state, str(weights / "model.safetensors"))
    with pytest.raises(ValueError, match="Dequantise this checkpoint to bf16 first") as caught:
        shard_checkpoint(str(weights), str(tmp_path / "s"), dtype="bfloat16", arch="deepseek_v3")
    assert "declares no FP8 quantization_config" in str(caught.value)
    assert not (tmp_path / "s" / "index.json").exists()


def test_layer_buffers_load_from_the_shard_at_the_stored_dtype(tmp_path):
    """Covers each branch of `_load_layer_buffers` on a hand-built layer: a
    persistent buffer is replaced by the shard's (fp32 over a bf16 skeleton);
    a non-persistent one and one the shard lacks keep their init values; a
    LoRA-style ``.base_layer.`` segment maps to the checkpoint name."""
    import torch
    import torch.nn as nn
    from safetensors.torch import save_file

    from soup_cli.utils.layer_stream_runtime import _load_layer_buffers

    layer = nn.Module()
    layer.gate = nn.Module()
    layer.gate.register_buffer("bias", torch.zeros(3, dtype=torch.bfloat16))
    layer.gate.register_buffer("scratch", torch.zeros(3), persistent=False)
    layer.gate.register_buffer("absent", torch.full((2,), 5.0))
    layer.proj = nn.Module()
    layer.proj.base_layer = nn.Module()
    layer.proj.base_layer.register_buffer("table", torch.zeros(2))
    layer.proj.register_buffer("counts", torch.arange(4))
    shard = {
        "gate.bias": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
        "gate.scratch": torch.ones(3),
        "proj.table": torch.tensor([7.0, 8.0]),
        # An integer buffer the sharder cast to the stream dtype (qwen4_exp has
        # one): not listed, so it must keep the skeleton's own value and dtype.
        "proj.counts": torch.zeros(4),
    }
    path = str(tmp_path / "layer_000.safetensors")
    save_file(shard, path)
    specs = [{key: (tuple(value.shape), "x") for key, value in shard.items()}]
    keys = ("gate.bias", "gate.scratch", "gate.absent", "proj.table")

    assert _load_layer_buffers([layer], [path], specs, keys) == 2
    assert layer.gate.bias.dtype == torch.float32
    assert torch.equal(layer.gate.bias, shard["gate.bias"])
    assert torch.equal(layer.gate.scratch, torch.zeros(3))
    assert torch.equal(layer.gate.absent, torch.full((2,), 5.0))
    assert torch.equal(layer.proj.base_layer.table, shard["proj.table"])
    assert torch.equal(layer.proj.counts, torch.arange(4))


def test_only_deepseek_v3_streams_a_layer_buffer():
    from soup_cli.utils.layer_shard import layer_buffer_keys

    assert layer_buffer_keys("deepseek_v3") == ("mlp.gate.e_score_correction_bias",)
    for arch in ("llama", "qwen4_exp", "qwen3", ""):
        assert layer_buffer_keys(arch) == ()


def test_layer_buffer_shape_mismatch_names_the_buffer(tmp_path):
    import torch
    import torch.nn as nn
    from safetensors.torch import save_file

    from soup_cli.utils.layer_stream_runtime import _load_layer_buffers

    layer = nn.Module()
    layer.register_buffer("bias", torch.zeros(4))
    path = str(tmp_path / "layer_000.safetensors")
    save_file({"bias": torch.zeros(3)}, path)
    with pytest.raises(ValueError, match="buffer 'bias' has shape"):
        _load_layer_buffers([layer], [path], [{"bias": ((3,), "F32")}], ("bias",))


def _deepseek_setup_cfg():
    return SimpleNamespace(
        model_type="deepseek_v3",
        hidden_size=4,
        num_hidden_layers=1,
        vocab_size=8,
        intermediate_size=8,
    )


@pytest.mark.parametrize(
    ("arch", "expected"), [("deepseek_v3", 1), ("llama", None)], ids=["deepseek_v3", "llama"]
)
def test_setup_hands_the_config_layer_count_to_the_sharder_for_deepseek_v3_only(
    tmp_path, monkeypatch, arch, expected
):
    """Without it a real DeepSeek-V3 (61 layers + the MTP layer 61) shards 62
    layers and the install refuses with "model has 61 decoder layers"."""
    from tests.test_issue602_qwen4_ple_streaming import _drive_qwen4_streaming_setup

    calls = []
    model_cfg = _deepseek_setup_cfg()
    model_cfg.model_type = arch
    _drive_qwen4_streaming_setup(
        tmp_path, monkeypatch, arch=arch, model_cfg=model_cfg, shard_calls=calls
    )
    assert len(calls) == 1
    assert calls[0]["num_hidden_layers"] == expected


def test_streamed_model_is_built_from_native_code_while_the_tokenizer_trusts_remote(
    tmp_path, monkeypatch
):
    """Moonlight-16B (deepseek_v3) ships its tokenizer only as remote code, and its
    remote modelling code keeps one module per expert, which the stacked shards do
    not fit. So the tokenizer gets the user's trust_remote_code and the model side
    (config, skeleton, install) never does."""
    from tests.test_issue602_qwen4_ple_streaming import _drive_qwen4_streaming_setup

    calls = {}
    _drive_qwen4_streaming_setup(
        tmp_path, monkeypatch, arch="deepseek_v3", model_cfg=_deepseek_setup_cfg(),
        load_calls=calls, trust_remote_code=True,
    )
    assert [call["trust_remote_code"] for call in calls["tokenizer"]] == [True]
    assert [call["trust_remote_code"] for call in calls["config"]] == [False]
    assert [call["trust_remote_code"] for call in calls["build"]] == [False]


def test_setup_meta_probe_is_built_from_native_code(monkeypatch):
    """The NF4 / moe_lora probe in the setup: the same rule as the model itself."""
    import soup_cli.utils.layer_stream_runtime as runtime
    from soup_cli.trainer.stream_setup import StreamingSetupMixin

    seen = []
    monkeypatch.setattr(runtime, "build_meta_skeleton", lambda *a, **k: seen.append(k) or "probe")
    probe = StreamingSetupMixin._stream_meta_skeleton("base", dtype="bfloat16", quant="nf4")
    assert probe == "probe"
    assert seen == [{"dtype": "bfloat16", "quant": "nf4", "trust_remote_code": False}]


def test_planner_fingerprints_config_json_for_deepseek_v3(tmp_path, monkeypatch):
    """The planner must agree with the sharder (`source_identity_includes_config`),
    or it reports a stale cache as reusable and the disk pre-flight under-counts."""
    import types

    import torch
    from safetensors.torch import save_file

    import soup_cli.utils.layer_shard as layer_shard
    from tests.test_issue602_qwen4_ple_streaming import _drive_qwen4_streaming_setup

    seen = []
    real = layer_shard.checkpoint_source_components

    def _recording(weights_dir, source_files, *, include_config):
        seen.append(include_config)
        return real(weights_dir, source_files, include_config=include_config)

    monkeypatch.setattr(layer_shard, "checkpoint_source_components", _recording)
    weights = tmp_path / "weights"
    weights.mkdir()
    blob = weights / "model.safetensors"
    save_file({"model.layers.0.proj.weight": torch.ones(2, 2)}, str(blob))
    (weights / "config.json").write_text('{"model_type": "deepseek_v3"}', encoding="utf-8")
    stat = blob.stat()
    plan = types.SimpleNamespace(
        weights_dir=str(weights),
        source_bytes=stat.st_size,
        materialized_copy_bytes=0,
        materialize_bytes=0,
        needs_materialization=False,
        source_files=((str(blob), stat.st_size, int(stat.st_mtime)),),
    )

    def _resolve(_base, *, before_materialize=None):
        if before_materialize is not None:
            before_materialize(plan)
        return str(weights)

    _drive_qwen4_streaming_setup(
        tmp_path, monkeypatch, resolve_weights=_resolve, arch="deepseek_v3",
        model_cfg=_deepseek_setup_cfg(),
    )
    assert seen == [True], seen


def test_config_is_part_of_the_cache_identity_for_deepseek_v3():
    from soup_cli.utils.layer_shard import source_identity_includes_config

    assert source_identity_includes_config("deepseek_v3") is True
    assert source_identity_includes_config("qwen4_exp") is True
    assert source_identity_includes_config("llama") is False


# --------------------------------------------------------------------------
# resident vs streamed parity on a real tiny DeepseekV3ForCausalLM (CPU fp32)
# --------------------------------------------------------------------------
def _tiny_deepseek_v3_dir(tmp_path, *, q_lora_rank):
    import torch
    from transformers import DeepseekV3Config, DeepseekV3ForCausalLM

    torch.manual_seed(11)
    config = DeepseekV3Config(
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
        n_group=2,
        topk_group=1,
        num_experts_per_tok=2,
        kv_lora_rank=16,
        q_lora_rank=q_lora_rank,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        max_position_embeddings=128,
    )
    model = DeepseekV3ForCausalLM(config).to(torch.float32).eval()
    generator = torch.Generator().manual_seed(23)
    with torch.no_grad():
        for layer in model.model.layers[config.first_k_dense_replace :]:
            # Large enough to change which experts win: the streamed run must
            # load it from the shard, not keep the skeleton's zeros.
            bias = layer.mlp.gate.e_score_correction_bias
            bias.copy_(torch.randn(bias.shape, generator=generator) * 2.0)
    weights = tmp_path / "model"
    model.save_pretrained(str(weights))
    return str(weights), model


def _tiny_lora():
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
        target_modules=_LORA_TARGETS, task_type=TaskType.CAUSAL_LM,
    )


def _build(tmp_path, *, q_lora_rank):
    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    weights, resident = _tiny_deepseek_v3_dir(tmp_path, q_lora_rank=q_lora_rank)
    shards = str(tmp_path / "shards")
    index = shard_checkpoint(
        weights, shards, dtype="float32", arch="deepseek_v3", num_hidden_layers=3
    )
    model, runtime = build_streamed_model(
        model_id=weights, shard_dir=shards, index=index,
        lora_config=_tiny_lora(), device="cpu", dtype="float32",
        buffers=2, pin=False, seed=3,
    )
    return model, runtime, resident


def _copy_lora(src, dst):
    def norm(key):
        return key.replace(".inner.", ".")

    src_lora = {norm(k): v for k, v in src.state_dict().items() if "lora_" in k}
    dst_lora = {norm(k): v for k, v in dst.state_dict().items() if "lora_" in k}
    assert src_lora and set(src_lora) == set(dst_lora)
    for key, val in src_lora.items():
        dst_lora[key].copy_(val)


def _paired(tmp_path, q_lora_rank):
    import torch
    from peft import get_peft_model

    model, runtime, resident = _build(tmp_path, q_lora_rank=q_lora_rank)
    ref = get_peft_model(resident, _tiny_lora())
    generator = torch.Generator().manual_seed(17)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.02)
    _copy_lora(model, ref)
    model.eval()
    ref.eval()
    ids = torch.randint(0, 128, (2, 12), generator=generator)
    return model, runtime, ref, ids


@pytest.mark.parametrize("q_lora_rank", [24, None], ids=["q_lora", "q_proj"])
def test_streamed_forward_matches_resident_bit_exactly(tmp_path, q_lora_rank):
    import torch

    model, runtime, ref, ids = _paired(tmp_path, q_lora_rank)
    with torch.no_grad():
        got = model(input_ids=ids, labels=ids)
        want = ref(input_ids=ids, labels=ids)
    assert torch.equal(got.logits, want.logits), (got.logits - want.logits).abs().max().item()
    assert torch.equal(got.loss, want.loss)
    runtime.close()


@pytest.mark.parametrize("q_lora_rank", [24, None], ids=["q_lora", "q_proj"])
def test_streamed_backward_matches_resident_lora_gradients_bit_exactly(tmp_path, q_lora_rank):
    import torch

    from soup_cli.utils.layer_stream_runtime import canonical_named_parameters

    model, runtime, ref, ids = _paired(tmp_path, q_lora_rank)
    model.zero_grad(set_to_none=True)
    ref.zero_grad(set_to_none=True)
    model(input_ids=ids, labels=ids).loss.backward()
    ref(input_ids=ids, labels=ids).loss.backward()

    def grads(net):
        return {
            name: parameter.grad
            for name, parameter in canonical_named_parameters(net)
            if "lora_" in name and parameter.grad is not None
        }

    streamed, resident = grads(model), grads(ref)
    assert streamed and set(streamed) == set(resident)
    # Every MoE layer's attention adapters receive gradient through the
    # streamed experts of the layers above them.
    assert any(".layers.0." in name for name in streamed)
    assert any(float(gradient.abs().sum()) > 0.0 for gradient in streamed.values())
    for name, gradient in streamed.items():
        assert torch.equal(gradient, resident[name]), name
    runtime.close()


def test_router_bias_is_loaded_and_the_tiny_model_depends_on_it(tmp_path):
    """Makes the parity tests above sensitive to the bias: a zero bias changes
    the logits of this model, and the streamed layers hold the checkpoint's."""
    import torch

    model, runtime, ref, ids = _paired(tmp_path, 24)
    base = ref.get_base_model()
    streamed_layers = model.get_base_model().model.layers
    for idx in range(1, 3):
        want = base.model.layers[idx].mlp.gate.e_score_correction_bias
        got = streamed_layers[idx].inner.mlp.gate.e_score_correction_bias
        assert float(want.abs().sum()) > 0.0
        assert got.dtype == want.dtype
        assert torch.equal(got, want)
    with torch.no_grad():
        before = ref(input_ids=ids).logits
        for idx in range(1, 3):
            base.model.layers[idx].mlp.gate.e_score_correction_bias.zero_()
        after = ref(input_ids=ids).logits
    assert not torch.equal(before, after)
    runtime.close()
