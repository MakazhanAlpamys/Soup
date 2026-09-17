"""Regression tests for #803: one LoRA-config path across every trainer."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

_WRAPPERS = (
    ("dpo", "DPOTrainerWrapper"),
    ("kto", "KTOTrainerWrapper"),
    ("orpo", "ORPOTrainerWrapper"),
    ("simpo", "SimPOTrainerWrapper"),
    ("ipo", "IPOTrainerWrapper"),
    ("bco", "BCOTrainerWrapper"),
    ("grpo", "GRPOTrainerWrapper"),
    ("reward_model", "RewardModelTrainerWrapper"),
    ("pretrain", "PretrainTrainerWrapper"),
    ("sft", "SFTTrainerWrapper"),
)


def _config(task: str):
    from soup_cli.config.schema import SoupConfig

    training = {
        "quantization": "none",
        "lora": {
            "r": 16,
            "alpha": 32,
            "target_modules": ["q_proj", "v_proj"],
            "rank_pattern": {"q_proj": 4},
            "alpha_pattern": {"q_proj": 8},
        },
    }
    if task == "online_dpo":
        training["reward_model"] = "tiny-local-reward"
    return SoupConfig(
        base="tiny-local-llama",
        task=task,
        data={"train": "train.jsonl"},
        training=training,
    )


def _model(*, sequence_classification: bool = False):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM, LlamaForSequenceClassification

    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        pad_token_id=0,
    )
    cls = LlamaForSequenceClassification if sequence_classification else LlamaForCausalLM
    return cls(config).to(torch.float32)


def _tokenizer():
    return SimpleNamespace(
        pad_token=None,
        eos_token="</s>",
        chat_template=None,
    )


def _bare_wrapper(wrapper_cls, config):
    wrapper = wrapper_cls.__new__(wrapper_cls)
    wrapper.config = config
    wrapper.device = "cpu"
    wrapper._trust_remote_code = False
    wrapper.model = None
    wrapper.tokenizer = None
    return wrapper


@pytest.mark.parametrize(("module_name", "class_name"), _WRAPPERS)
def test_every_transformers_wrapper_builds_configured_per_module_rank_on_cpu(
    module_name: str,
    class_name: str,
) -> None:
    module = importlib.import_module(f"soup_cli.trainer.{module_name}")
    wrapper_cls = getattr(module, class_name)
    config = _config(module_name)
    model = _model(sequence_classification=module_name == "reward_model")
    wrapper = _bare_wrapper(wrapper_cls, config)

    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=_tokenizer()),
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model),
        patch(
            "transformers.AutoModelForSequenceClassification.from_pretrained",
            return_value=model,
        ),
    ):
        wrapper._setup_transformers(config, config.training)

    ranks = {
        name.rsplit(".", maxsplit=1)[-1]: module.lora_A["default"].weight.shape[0]
        for name, module in wrapper.model.named_modules()
        if name.endswith(("q_proj", "v_proj")) and hasattr(module, "lora_A")
    }
    assert ranks == {"q_proj": 4, "v_proj": 16}
    peft_config = wrapper.model.peft_config["default"]
    assert peft_config.rank_pattern == {"q_proj": 4}
    assert peft_config.alpha_pattern == {"q_proj": 8}


def test_online_dpo_preserves_patterns_in_deferred_peft_config() -> None:
    from soup_cli.trainer.online_dpo import OnlineDPOTrainerWrapper

    config = _config("online_dpo")
    wrapper = _bare_wrapper(OnlineDPOTrainerWrapper, config)

    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=_tokenizer()),
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=_model()),
    ):
        wrapper._setup_transformers(config, config.training)

    assert wrapper.tokenizer.chat_template is not None
    assert wrapper.peft_config.rank_pattern == {"q_proj": 4}
    assert wrapper.peft_config.alpha_pattern == {"q_proj": 8}


def test_shared_builder_rejects_a_pattern_blind_config_double() -> None:
    from soup_cli.utils.peft_wiring import build_lora_config_kwargs

    incomplete = SimpleNamespace(
        r=16,
        alpha=32,
        dropout=0.0,
        use_dora=False,
        use_rslora=False,
    )
    with pytest.raises(AttributeError, match="rank_pattern"):
        build_lora_config_kwargs(
            incomplete,
            target_modules=["q_proj", "v_proj"],
            target_parameters=None,
            task_type="CAUSAL_LM",
        )


def test_trainers_cannot_construct_lora_config_outside_shared_builder() -> None:
    from soup_cli import trainer

    trainer_dir = Path(trainer.__file__).parent
    offenders = []
    for path in trainer_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if any(
            isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Name) and node.func.id == "LoraConfig")
                or (isinstance(node.func, ast.Attribute) and node.func.attr == "LoraConfig")
            )
            for node in ast.walk(tree)
        ):
            offenders.append(path.name)
    offenders.sort()

    assert offenders == [], (
        "trainer modules must use soup_cli.utils.peft_wiring.build_lora_config; "
        f"direct LoraConfig construction found in: {offenders}"
    )
