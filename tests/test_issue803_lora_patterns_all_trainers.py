"""Regression tests for #803: one LoRA-config path across every trainer."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def test_dpo_builds_the_configured_per_module_rank_on_cpu() -> None:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer.dpo import DPOTrainerWrapper

    config = SoupConfig(
        base="tiny-local-llama",
        task="dpo",
        data={"train": "train.jsonl"},
        training={
            "quantization": "none",
            "lora": {
                "r": 16,
                "alpha": 32,
                "target_modules": ["q_proj", "v_proj"],
                "rank_pattern": {"q_proj": 4},
                "alpha_pattern": {"q_proj": 8},
            },
        },
    )
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
    ).to(torch.float32)
    tokenizer = SimpleNamespace(pad_token="<pad>", eos_token="</s>")
    wrapper = DPOTrainerWrapper(config, device="cpu")

    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model),
    ):
        wrapper._setup_transformers(config, config.training)

    q_proj = next(
        module
        for name, module in wrapper.model.named_modules()
        if name.endswith("q_proj") and hasattr(module, "lora_A")
    )
    assert q_proj.lora_A["default"].weight.shape[0] == 4
    peft_config = wrapper.model.peft_config["default"]
    assert peft_config.rank_pattern == {"q_proj": 4}
    assert peft_config.alpha_pattern == {"q_proj": 8}


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
