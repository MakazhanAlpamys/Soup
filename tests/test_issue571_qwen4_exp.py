"""#571 weight-free compatibility gates for Qwen3.8-Flash-Next's text decoder."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_qwen4_exp_text_auto_lora_covers_every_linear_family():
    from soup_cli.utils.peft_wiring import resolve_lora_target_modules

    model = SimpleNamespace(config=SimpleNamespace(model_type="qwen4_exp_text"))

    assert resolve_lora_target_modules(model, "auto") == "all-linear"
    assert resolve_lora_target_modules(model, ["auto"]) == "all-linear"


def test_qwen4_exp_explicit_lora_targets_are_not_overridden():
    from soup_cli.utils.peft_wiring import resolve_lora_target_modules

    model = SimpleNamespace(config=SimpleNamespace(model_type="qwen4_exp_text"))
    explicit = ["in_proj_qkv", "q_proj"]

    assert resolve_lora_target_modules(model, explicit) is explicit


def test_qwen4_exp_outer_config_completes_text_decoder_modules(monkeypatch):
    from transformers import AutoConfig

    from soup_cli.utils.completions import complete_target_modules

    outer = SimpleNamespace(
        model_type="qwen4_exp",
        text_config=SimpleNamespace(model_type="qwen4_exp_text"),
    )
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *_args, **_kwargs: outer)

    targets = complete_target_modules("", base="Qwen/Qwen3.8-Flash-Next")

    assert "q_proj" in targets
    assert "index_qk_proj" in targets
    assert "in_proj_qkv" in targets
    assert "in_proj_z" in targets
    assert "shared_expert_gate" in targets
    assert "input_mix_weight_down" in targets
    assert "key_proj" in targets


def test_qwen4_exp_text_is_detected_as_moe_without_numeric_hints():
    from soup_cli.utils.moe import detect_moe_model

    model = SimpleNamespace(config=SimpleNamespace(model_type="qwen4_exp_text"))

    assert detect_moe_model(model) is True


def test_qwen4_exp_streaming_is_admitted_only_through_its_text_alias():
    from soup_cli.utils.layer_stream import _STREAM_ARCH_ALIASES, SUPPORTED_STREAM_ARCHS

    assert "qwen4_exp" in SUPPORTED_STREAM_ARCHS
    assert "qwen4_exp_text" not in SUPPORTED_STREAM_ARCHS
    assert "qwen4_exp" not in _STREAM_ARCH_ALIASES
    assert _STREAM_ARCH_ALIASES["qwen4_exp_text"] == "qwen4_exp"


def _tiny_qwen4_exp_config():
    try:
        from transformers import Qwen4ExpConfig, Qwen4ExpTextConfig
    except ImportError:
        pytest.skip("installed Transformers release does not include qwen4_exp yet")

    text = Qwen4ExpTextConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_conv_kernel_dim=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts_per_tok=1,
        num_experts=2,
        layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=64,
        hc_count=2,
        hc_lowrank=4,
        ple_layer_ids=[],
        use_cache=False,
        indexer_n_heads=1,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=8,
        indexer_compress_ratio=2,
    )
    return Qwen4ExpConfig(text_config=text.to_dict())


def test_qwen4_exp_outer_config_builds_text_only_causal_lm_and_lora():
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    from soup_cli.utils.peft_wiring import (
        apply_pre_lora_patches,
        resolve_lora_target_modules,
    )

    model = AutoModelForCausalLM.from_config(_tiny_qwen4_exp_config(), dtype=torch.float32)

    assert type(model).__name__ == "Qwen4ExpForCausalLM"
    assert model.config.model_type == "qwen4_exp_text"
    assert not [
        name
        for name, _ in model.named_parameters()
        if "vision" in name or "visual" in name
    ]

    targets = resolve_lora_target_modules(model, "auto")
    torch_int32 = torch.int32
    indexers = [
        module
        for module in model.modules()
        if type(module).__name__ == "Qwen4ExpTextQSAIndexer"
    ]
    class_forward = type(indexers[0]).forward
    apply_pre_lora_patches(model, "Qwen/Qwen3.8-Flash-Next")

    # The compatibility shim must remain instance-local: neither Torch's dtype
    # singleton nor the upstream Transformers class may be changed globally.
    assert torch.int32 is torch_int32
    assert type(indexers[0]).forward is class_forward
    int32_scatter_supported = True
    try:
        torch.zeros((1, 1), dtype=torch.bool).scatter(
            -1, torch.zeros((1, 1), dtype=torch.int32), True
        )
    except RuntimeError as exc:
        assert "Expected dtype int64 for index" in str(exc)
        int32_scatter_supported = False
    if int32_scatter_supported:
        assert indexers[0].forward.__func__ is class_forward
    else:
        assert indexers[0].forward.__func__ is not class_forward
    patched_forward = indexers[0].forward.__func__
    apply_pre_lora_patches(model, "Qwen/Qwen3.8-Flash-Next")
    assert indexers[0].forward.__func__ is patched_forward

    model = get_peft_model(
        model,
        LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=targets,
            task_type=TaskType.CAUSAL_LM,
        ),
    )
    batch = torch.tensor([[1, 2, 3, 4]])
    loss = model(input_ids=batch, labels=batch).loss

    assert torch.isfinite(loss).item()
    loss.backward()
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable
    for family in (
        "linear_attn.in_proj_qkv",
        "self_attn.q_proj",
        "self_attn.indexer.index_qk_proj",
        "mlp.shared_expert.gate_proj",
        "attn_hyper_connection.input_mix_weight_down",
    ):
        assert any(family in name for name in trainable)
    # The four-token probe is too short to activate QSA's compressed-block
    # indexer, but every other selected text-decoder family must participate in
    # backward. The indexer assertion above still proves adapter injection.
    assert all(
        parameter.grad is not None
        for name, parameter in trainable.items()
        if "self_attn.indexer.index_qk_proj" not in name
    )
