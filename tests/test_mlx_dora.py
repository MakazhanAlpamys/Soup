"""The supported MLX DoRA contract, without requiring Apple hardware."""

import pytest
from pydantic import ValidationError

from soup_cli.config.schema import LoraConfig, SoupConfig
from soup_cli.trainer.mlx_sft import build_mlx_adapter_config


@pytest.mark.parametrize("quantization", ["none", "4bit"])
def test_mlx_text_sft_accepts_dora(quantization):
    cfg = SoupConfig(
        base="local-llama",
        backend="mlx",
        data={"train": "train.jsonl"},
        training={"quantization": quantization, "lora": {"use_dora": True}},
    )
    assert cfg.training.lora.use_dora is True
    assert cfg.training.quantization == quantization


@pytest.mark.parametrize("use_dora", [False, True])
def test_adapter_metadata_preserves_the_method_and_parameters(use_dora):
    cfg = LoraConfig(
        r=4, alpha=12, dropout=0.1, target_modules=["self_attn.v_proj"], use_dora=use_dora
    )
    written = build_mlx_adapter_config(cfg, adapter_path="adapters")
    assert written["fine_tune_type"] == ("dora" if use_dora else "lora")
    assert written["adapter_path"] == "adapters"
    assert written["lora_parameters"] == {
        "rank": 4, "scale": 3.0, "dropout": 0.1, "keys": ["self_attn.v_proj"]
    }


@pytest.mark.parametrize(
    "overrides, training",
    [
        ({"backend": "unsloth"}, {}),
        ({"task": "dpo"}, {}),
        ({"modality": "vision"}, {}),
        ({"modality": "audio"}, {}),
        ({}, {"quantization": "8bit"}),
    ],
)
def test_unimplemented_dora_routes_remain_refused(overrides, training):
    with pytest.raises(ValidationError, match="[Dd][Oo][Rr][Aa]|variant 'dora'"):
        SoupConfig(
            **{"base": "local-llama", "backend": "mlx", **overrides},
            data={"train": "train.jsonl"},
            training={"quantization": "none", **training, "lora": {"use_dora": True}},
        )


@pytest.mark.parametrize("task", ["grpo", "pretrain"])
def test_non_sft_dora_is_refused_by_the_dora_task_gate(task: str) -> None:
    # Pin the DoRA-specific refusal: the general MLX task validator must not
    # hide a missing task gate here by rejecting the config for another reason.
    with pytest.raises(ValidationError, match="MLX DoRA requires task='sft'"):
        SoupConfig(
            base="local-llama", backend="mlx", task=task,
            data={"train": "train.jsonl"},
            training={"quantization": "none", "lora": {"use_dora": True}},
        )


@pytest.mark.parametrize("option", [
    {"use_rslora": True}, {"rank_pattern": {"q_proj": 2}}, {"alpha_pattern": {"q_proj": 4}}
])
def test_mlx_dora_refuses_unimplemented_scaling_options(option):
    with pytest.raises(ValidationError, match="MLX DoRA does not support"):
        SoupConfig(
            base="local-llama", backend="mlx", data={"train": "train.jsonl"},
            training={"lora": {"use_dora": True, **option}},
        )
