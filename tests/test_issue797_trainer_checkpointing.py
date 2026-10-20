"""#797, item 2: the real TRL/HF trainer must receive the YAML setting.

Local random Llama weights and a real tokenizer keep this offline. Model loads,
PEFT, config construction and trainer construction are real. Tests pin HF's
device choice to CPU; the online judge is a deterministic local evaluator.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from soup_cli.config.loader import load_config_from_string
from tests._tiny_hf_models import tiny_model_dir

if TYPE_CHECKING:
    from soup_cli.eval.judge import JudgeScore

_WRAPPERS = {
    "grpo": "GRPOTrainerWrapper",
    "ipo": "IPOTrainerWrapper",
    "bco": "BCOTrainerWrapper",
    "reward_model": "RewardModelTrainerWrapper",
    "online_dpo": "OnlineDPOTrainerWrapper",
    "pretrain": "PretrainTrainerWrapper",
    "embedding": "EmbeddingTrainerWrapper",
}


@pytest.fixture(autouse=True)
def train_stack(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in ("torch", "transformers", "peft", "trl", "datasets"):
        pytest.importorskip(name, reason=f"{name} requires the train extra")
    import torch
    from transformers import TrainingArguments

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    # ACCELERATE_USE_CPU alone still lets HF choose MPS on a Mac. Set the public
    # flag before real HF initialization, without changing the checkpointing value.
    real_post_init = TrainingArguments.__post_init__

    def cpu_post_init(args: TrainingArguments) -> None:
        args.use_cpu = True
        real_post_init(args)

    monkeypatch.setattr(TrainingArguments, "__post_init__", cpu_post_init)
    yield
    torch.set_num_threads(threads)


class _LocalJudge:
    def compare_pair(self, prompt: str, resp_a: str, resp_b: str) -> int:
        return 0

    def evaluate(
        self, prompt: str, response: str, category: str = "default"
    ) -> JudgeScore:
        from soup_cli.eval.judge import JudgeScore

        return JudgeScore(prompt=prompt, response=response, weighted_score=1.0)


def _build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task: str, checkpointing: bool | str
) -> tuple[Any, Any]:
    import yaml

    monkeypatch.chdir(tmp_path)
    training = {
        "quantization": "none",
        "gradient_checkpointing": checkpointing,
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "epochs": 1,
        "warmup_ratio": 0.0,
        "save_steps": 1000,
        "seed": 7,
        "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
    }
    if task == "grpo":
        training.update(num_generations=2, reward_fn="format")
    elif task == "online_dpo":
        training.update(online_dpo_judge="ollama://local-test", online_dpo_max_new_tokens=8)
        import soup_cli.trainer.online_dpo as online_dpo

        monkeypatch.setattr(online_dpo, "_ONLINE_DPO_JUDGE_OVERRIDE", _LocalJudge())
    data = {"train": "unused.jsonl", "max_length": 64}
    if task not in {"pretrain", "embedding"}:
        data["chat_template"] = "chatml"
    cfg = load_config_from_string(yaml.safe_dump({
        "base": tiny_model_dir("llama"),
        "task": task,
        "backend": "transformers",
        "data": data,
        "training": training,
        "output": str(tmp_path / "out"),
    }))
    if task in {"grpo", "online_dpo"}:
        rows = [{"prompt": "hello world"}] * 4
    elif task == "pretrain":
        rows = [{"text": "hello world hi there friend"}] * 4
    elif task == "embedding":
        rows = [{"anchor": "hello world", "positive": "hi there friend"}] * 4
    else:
        rows = [{"prompt": "hello", "chosen": " world", "rejected": " friend"}] * 4
    cls = getattr(importlib.import_module(f"soup_cli.trainer.{task}"), _WRAPPERS[task])
    wrapper = cls(cfg, device="cpu")
    wrapper.setup({"train": rows})
    trainer = getattr(wrapper.trainer, "_trainer", wrapper.trainer)
    assert trainer.args.device.type == "cpu"
    assert next(trainer.model.parameters()).device.type == "cpu"
    return wrapper, trainer


@pytest.mark.parametrize("task", _WRAPPERS)
@pytest.mark.parametrize("checkpointing", [False, True], ids=["disabled", "enabled"])
def test_yaml_checkpointing_reaches_real_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task: str, checkpointing: bool
) -> None:
    _, trainer = _build(tmp_path, monkeypatch, task, checkpointing)
    assert trainer.args.gradient_checkpointing is checkpointing, task


@pytest.mark.parametrize("task", _WRAPPERS)
@pytest.mark.parametrize("checkpointing", [False, True], ids=["disabled", "enabled"])
def test_checkpointing_state_after_cpu_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task: str, checkpointing: bool
) -> None:
    _, trainer = _build(tmp_path, monkeypatch, task, checkpointing)
    trainer.args.max_steps = 1
    trainer.args.save_strategy = "no"
    trainer.args.disable_tqdm = True
    trainer.train()
    assert trainer.state.global_step == 1
    assert trainer.model.is_gradient_checkpointing is checkpointing, task


@pytest.mark.parametrize("task", _WRAPPERS)
@pytest.mark.parametrize("tier", ["full", "medium", "selective", "auto"])
def test_enabled_tiers_keep_non_sft_full_checkpointing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task: str, tier: str
) -> None:
    # These wrappers have no selective hooks: retain the documented full fallback.
    _, trainer = _build(tmp_path, monkeypatch, task, tier)
    assert trainer.args.gradient_checkpointing is True, (task, tier)
