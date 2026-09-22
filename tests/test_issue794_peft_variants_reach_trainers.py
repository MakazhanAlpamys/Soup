"""Regression tests for #794: advertised PEFT variants reach live trainer paths."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import ValidationError


def _config(**lora):
    from soup_cli.config.schema import SoupConfig

    return SoupConfig(
        base="tiny-local-llama",
        task="sft",
        data={"train": "train.jsonl"},
        training={
            "quantization": "none",
            "lora": {
                "r": 8,
                "alpha": 16,
                "dropout": 0.0,
                "target_modules": ["q_proj", "v_proj"],
                **lora,
            },
        },
    )


def _model():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            pad_token_id=0,
        )
    ).to(torch.float32)


def _setup_sft(config, model):
    from soup_cli.trainer.sft import SFTTrainerWrapper

    wrapper = SFTTrainerWrapper.__new__(SFTTrainerWrapper)
    wrapper.config = config
    wrapper.device = "cpu"
    wrapper._trust_remote_code = False
    wrapper.model = None
    wrapper.tokenizer = None
    tokenizer = SimpleNamespace(pad_token=None, eos_token="</s>", chat_template=None)
    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model),
    ):
        wrapper._setup_transformers(config, config.training)
    return wrapper.model


def test_pissa_changes_base_weight_through_real_sft_setup() -> None:
    import torch

    model = _model()
    before = model.model.layers[0].self_attn.q_proj.weight.detach().clone()
    wrapped = _setup_sft(_config(init_strategy="pissa"), model)
    q_proj = wrapped.base_model.model.model.layers[0].self_attn.q_proj

    assert wrapped.peft_config["default"].init_lora_weights == "pissa"
    assert not torch.equal(q_proj.base_layer.weight.detach(), before)


@pytest.mark.parametrize(
    "lora",
    [
        pytest.param({"init_strategy": "olora"}, id="init-strategy"),
        pytest.param({"use_olora": True}, id="legacy-flag"),
    ],
)
def test_olora_changes_base_weight_through_real_sft_setup(lora: dict) -> None:
    import torch

    model = _model()
    before = model.model.layers[0].self_attn.q_proj.weight.detach().clone()
    wrapped = _setup_sft(_config(**lora), model)
    q_proj = wrapped.base_model.model.model.layers[0].self_attn.q_proj

    assert wrapped.peft_config["default"].init_lora_weights == "olora"
    assert not torch.equal(q_proj.base_layer.weight.detach(), before)


def test_vera_builds_vera_layers_not_lora_layers_through_real_sft_setup() -> None:
    wrapped = _setup_sft(_config(use_vera=True, r=16), _model())
    plain_lora = _setup_sft(_config(r=16), _model())
    q_proj = wrapped.base_model.model.model.layers[0].self_attn.q_proj

    assert type(wrapped.peft_config["default"]).__name__ == "VeraConfig"
    assert type(q_proj).__module__.startswith("peft.tuners.vera")
    assert hasattr(q_proj, "vera_lambda_b")
    assert not hasattr(q_proj, "lora_A")
    assert wrapped.get_nb_trainable_parameters()[0] == 56
    assert plain_lora.get_nb_trainable_parameters()[0] == 896


def test_loftq_changes_base_weight_through_real_sft_setup() -> None:
    import torch

    model = _model()
    before = model.model.layers[0].self_attn.q_proj.weight.detach().clone()
    real_tensor_to = torch.Tensor.to

    def _keep_peft_loftq_compute_on_cpu(tensor, *args, **kwargs):
        """PEFT 0.20 hard-codes its LoftQ scratch device to CUDA/XPU.

        Remap only that device choice so the real bitsandbytes quantization and
        PEFT LoftQ SVD execute on CPU in the cross-platform test matrix.
        """
        positional = list(args)
        if positional and positional[0] == "cuda":
            positional[0] = "cpu"
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return real_tensor_to(tensor, *positional, **kwargs)

    with patch.object(torch.Tensor, "to", _keep_peft_loftq_compute_on_cpu):
        wrapped = _setup_sft(
            _config(init_strategy="loftq", loftq_iter=3, loftq_bits=8), model
        )
    q_proj = wrapped.base_model.model.model.layers[0].self_attn.q_proj

    config = wrapped.peft_config["default"]
    assert config.init_lora_weights == "loftq"
    assert config.loftq_config == {"loftq_bits": 8, "loftq_iter": 3}
    assert not torch.equal(q_proj.base_layer.weight.detach(), before)


@pytest.mark.parametrize("backend", ["mlx", "unsloth"])
@pytest.mark.parametrize("lora", [{"init_strategy": "pissa"}, {"use_vera": True}])
def test_unwired_backends_refuse_peft_variants(backend: str, lora: dict) -> None:
    from soup_cli.config.schema import SoupConfig

    with pytest.raises(ValidationError, match="requires backend='transformers'"):
        SoupConfig(
            base="tiny-local-llama",
            task="sft",
            backend=backend,
            data={"train": "train.jsonl"},
            training={"quantization": "none", "lora": lora},
        )


def test_loftq_refuses_prequantized_base() -> None:
    from soup_cli.config.schema import SoupConfig

    with pytest.raises(ValidationError, match="LoftQ quantizes the base model"):
        SoupConfig(
            base="tiny-local-llama",
            task="sft",
            data={"train": "train.jsonl"},
            training={"lora": {"init_strategy": "loftq"}},
        )


def test_pissa_refuses_prequantized_base() -> None:
    from soup_cli.config.schema import SoupConfig

    with pytest.raises(ValidationError, match="PiSSA computes an SVD"):
        SoupConfig(
            base="tiny-local-llama",
            task="sft",
            data={"train": "train.jsonl"},
            training={"lora": {"init_strategy": "pissa"}},
        )


@pytest.mark.parametrize(
    "lora",
    [
        {"init_strategy": "pissa"},
        {"init_strategy": "olora"},
        {"init_strategy": "loftq"},
        {"use_vera": True},
    ],
)
def test_moe_lora_routing_refuses_unconsumed_peft_variants(lora: dict) -> None:
    from soup_cli.config.schema import SoupConfig

    with pytest.raises(ValidationError, match="loads existing adapters"):
        SoupConfig(
            base="tiny-local-llama",
            task="moe_lora_routing",
            data={"train": "train.jsonl"},
            training={
                "mole_task_adapters": ["./adapter-a", "./adapter-b"],
                "lora": lora,
            },
        )


def test_distill_explicit_4bit_resolves_before_loftq_validation() -> None:
    from soup_cli.config.deprecation import SoupConfigDeprecationWarning
    from soup_cli.config.schema import SoupConfig

    with pytest.warns(SoupConfigDeprecationWarning, match="task='distill'"):
        config = SoupConfig(
            base="tiny-local-llama",
            task="distill",
            data={"train": "train.jsonl"},
            training={
                "teacher_model": "tiny-local-teacher",
                "quantization": "4bit",
                "lora": {"init_strategy": "loftq"},
            },
        )

    assert config.training.quantization == "none"
    assert config.training.lora.init_strategy == "loftq"
