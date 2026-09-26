"""#573 — Qwen4-Exp LoRA over routed-expert raw parameters."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError


def _soup_config(*, task="sft", backend="transformers", modality="text", lora=None):
    from soup_cli.config.schema import SoupConfig

    return SoupConfig(
        base="Qwen/Qwen3.8-Flash-Next",
        task=task,
        backend=backend,
        modality=modality,
        data={"train": "train.jsonl"},
        training={
            "quantization": "none",
            "lora": lora or {"dropout": 0, "target_parameters": "auto"},
        },
    )


def test_target_parameters_schema_is_opt_in_and_normalizes_explicit_names():
    from soup_cli.config.schema import LoraConfig

    assert LoraConfig().target_parameters is None
    config = LoraConfig(
        dropout=0,
        target_parameters=[
            " mlp.experts.gate_up_proj ",
            "mlp.experts.down_proj",
            "mlp.experts.gate_up_proj",
        ],
    )

    assert config.target_parameters == [
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
    ]


@pytest.mark.parametrize(
    ("lora", "message"),
    [
        ({"target_parameters": "auto"}, "dropout=0"),
        (
            {"dropout": 0, "target_parameters": "auto", "use_dora": True},
            "use_dora",
        ),
        (
            {"dropout": 0, "target_parameters": "auto", "use_vera": True},
            "use_vera",
        ),
        (
            {
                "dropout": 0,
                "target_parameters": "auto",
                "init_strategy": "pissa",
            },
            "init_strategy='random'",
        ),
        ({"dropout": 0, "target_parameters": ["auto"]}, "scalar 'auto'"),
    ],
)
def test_target_parameters_rejects_unsupported_peft_combinations(lora, message):
    with pytest.raises(ValidationError, match=message):
        _soup_config(lora=lora)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"task": "dpo"}, "task='sft' or task='pretrain'"),
        ({"backend": "mlx"}, "backend='transformers'"),
        ({"modality": "vision"}, "modality='text'"),
    ],
)
def test_target_parameters_is_scoped_to_resident_text_sft_and_pretrain(
    overrides, message
):
    kwargs = {
        "task": "sft",
        "backend": "transformers",
        "modality": "text",
    }
    kwargs.update(overrides)

    with pytest.raises(ValidationError, match=message):
        _soup_config(**kwargs)


def test_qwen4_exp_auto_resolves_both_routed_expert_parameters():
    from soup_cli.utils.peft_wiring import resolve_lora_target_parameters

    outer = SimpleNamespace(
        model_type="qwen4_exp",
        text_config=SimpleNamespace(model_type="qwen4_exp_text"),
    )

    assert resolve_lora_target_parameters(outer, "auto") == [
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
    ]


def test_target_parameters_accepts_continued_pretraining():
    config = _soup_config(task="pretrain")

    assert config.training.lora.target_parameters == "auto"


def test_target_parameter_auto_fails_closed_for_unknown_architecture():
    from soup_cli.utils.peft_wiring import resolve_lora_target_parameters

    with pytest.raises(ValueError, match="has no mapping"):
        resolve_lora_target_parameters(SimpleNamespace(model_type="unknown"), "auto")


def test_shared_sft_pretrain_kwargs_forward_parameter_and_rank_targets():
    from soup_cli.utils.peft_wiring import build_lora_config_kwargs

    lora = _soup_config(
        lora={
            "r": 16,
            "alpha": 32,
            "dropout": 0,
            "target_parameters": "auto",
            "rank_pattern": {"experts.gate_up_proj": 2},
            "alpha_pattern": {"experts.gate_up_proj": 4},
        }
    ).training.lora
    kwargs = build_lora_config_kwargs(
        lora,
        target_modules="all-linear",
        target_parameters=["mlp.experts.gate_up_proj"],
        task_type="CAUSAL_LM",
    )

    assert kwargs["target_modules"] == "all-linear"
    assert kwargs["target_parameters"] == ["mlp.experts.gate_up_proj"]
    assert kwargs["rank_pattern"] == {"experts.gate_up_proj": 2}
    assert kwargs["alpha_pattern"] == {"experts.gate_up_proj": 4}


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


def test_qwen4_exp_expert_adapters_backward_save_and_reload(tmp_path):
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from peft.utils.save_and_load import get_peft_model_state_dict
    from transformers import AutoModelForCausalLM

    from soup_cli.utils.peft_wiring import (
        build_lora_config_kwargs,
        resolve_lora_target_modules,
        resolve_lora_target_parameters,
    )

    config = _tiny_qwen4_exp_config()
    torch.manual_seed(7)
    base = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    soup_lora = _soup_config(
        lora={
            "r": 2,
            "alpha": 4,
            "dropout": 0,
            "target_parameters": "auto",
            "rank_pattern": {"experts.gate_up_proj": 1},
        }
    ).training.lora
    model = get_peft_model(
        base,
        LoraConfig(
            **build_lora_config_kwargs(
                soup_lora,
                target_modules=resolve_lora_target_modules(base, "auto"),
                target_parameters=resolve_lora_target_parameters(base, "auto"),
                task_type=TaskType.CAUSAL_LM,
            )
        ),
    )

    batch = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    loss = model(input_ids=batch, labels=batch).loss
    assert torch.isfinite(loss).item()
    loss.backward()

    expert_adapter_params = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "mlp.experts" in name
    }
    assert expert_adapter_params
    assert any("base_layer.lora_A" in name for name in expert_adapter_params)
    assert any("experts.lora_A" in name for name in expert_adapter_params)
    assert all(parameter.grad is not None for parameter in expert_adapter_params.values())
    assert all(
        torch.isfinite(parameter.grad).all().item()
        for parameter in expert_adapter_params.values()
    )

    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.1,
    )
    optimizer.step()
    before = get_peft_model_state_dict(model)
    model.save_pretrained(tmp_path)

    torch.manual_seed(7)
    fresh = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    loaded = PeftModel.from_pretrained(fresh, tmp_path, is_trainable=True)
    after = get_peft_model_state_dict(loaded)

    assert loaded.peft_config["default"].target_parameters == [
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
    ]
    assert before.keys() == after.keys()
    assert all(torch.equal(before[key].cpu(), after[key].cpu()) for key in before)
