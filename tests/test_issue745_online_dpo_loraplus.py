"""Issue #745: ``task: online_dpo`` wires LoRA+.

Online DPO is the one wrapper that never calls ``get_peft_model``: it builds a
``LoraConfig`` and hands it to TRL as ``peft_config=``, and ``OnlineDPOTrainer``
applies the adapter inside its own ``__init__``. The wiring-coverage scan only
looked for ``get_peft_model(``, so it could not see this trainer, and
``loraplus_lr_ratio`` parsed and was silently ignored.

``OnlineDPOTrainer`` is a ``transformers.Trainer`` subclass that builds its
optimizer lazily at ``train()`` and forwards ``optimizers=`` to
``Trainer.__init__``, so it takes the same post-construction
``attach_loraplus_optimizer`` as the other lazy trainers: by the time the
wrapper's ``setup`` returns, ``trainer.model`` is the PEFT model and the attached
optimizer is the one ``create_optimizer_and_scheduler`` builds the scheduler
around.

The first test runs in the default suite against a stand-in trainer that applies
``peft_config`` the way TRL does. The ``smoke`` test builds the real TRL trainer
on a tiny Hub model and checks the scheduler binding end to end.
"""

from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")
pytest.importorskip("trl")

LR = 1e-4
RATIO = 16.0
B_GROUP_LR = round(LR * RATIO, 12)

_DATASET = {
    "train": [
        {"messages": [{"role": "user", "content": "hi there friend"}]},
        {"messages": [{"role": "user", "content": "hello"}]},
    ]
}


def _config(ratio):
    from soup_cli.config.loader import load_config_from_string

    ratio_line = f"  loraplus_lr_ratio: {ratio}\n" if ratio is not None else ""
    return load_config_from_string(
        "base: hf-internal-testing/tiny-random-gpt2\n"
        "task: online_dpo\n"
        "data:\n  train: x.jsonl\n  max_length: 64\n"
        "training:\n"
        "  online_dpo_judge: \"ollama://m\"\n"
        "  epochs: 1\n"
        "  batch_size: 2\n"
        "  online_dpo_max_new_tokens: 6\n"
        f"  lr: {LR}\n"
        f"{ratio_line}"
        "output: ./out\n"
    )


def _group_lrs(optimizer):
    return {round(group["lr"], 12) for group in optimizer.param_groups}


def _setup_against_a_stand_in_trainer(ratio):
    """Run the real wrapper ``setup`` with a trainer that does what TRL does with
    ``peft_config``: wrap the model in ``__init__`` and leave the optimizer to be
    built later."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM

    import soup_cli.trainer._trl_compat as trl_compat
    from soup_cli.trainer.online_dpo import OnlineDPOTrainerWrapper

    real_resolve = trl_compat.resolve_trl_symbol

    class StandInOnlineDPOTrainer:
        def __init__(self, model, args, peft_config=None, **kwargs):
            self.model = get_peft_model(model, peft_config) if peft_config else model
            self.args = args
            self.optimizer = None
            self.callback_handler = MagicMock()

        def add_callback(self, callback):
            pass

    def resolve(name, *modules):
        if name == "OnlineDPOTrainer":
            return StandInOnlineDPOTrainer
        return real_resolve(name, *modules)

    def setup_transformers(self, cfg, tcfg):
        model_cfg = AutoConfig.for_model(
            "llama", hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, vocab_size=128,
        )
        self.model = AutoModelForCausalLM.from_config(model_cfg)
        self.tokenizer = MagicMock()
        self.tokenizer.pad_token = "pad"
        self.peft_config = LoraConfig(
            r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )

    wrapper = OnlineDPOTrainerWrapper(_config(ratio), device="cpu")
    with mock_patch.object(trl_compat, "resolve_trl_symbol", side_effect=resolve), \
         mock_patch.object(
             OnlineDPOTrainerWrapper, "_setup_transformers", setup_transformers
         ), \
         mock_patch.object(
             OnlineDPOTrainerWrapper, "_build_judge_or_reward",
             return_value={"reward_funcs": [lambda **kw: [0.0]]},
         ):
        wrapper.setup(_DATASET)
    return wrapper


def test_online_dpo_attaches_the_loraplus_optimizer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    wrapper = _setup_against_a_stand_in_trainer(RATIO)

    optimizer = wrapper.trainer.optimizer
    assert optimizer is not None, "loraplus_lr_ratio was ignored on online_dpo"
    assert B_GROUP_LR in _group_lrs(optimizer)
    assert LR in _group_lrs(optimizer)


def test_online_dpo_leaves_trl_its_own_optimizer_without_a_ratio(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    wrapper = _setup_against_a_stand_in_trainer(None)

    assert wrapper.trainer.optimizer is None


@pytest.mark.smoke
def test_real_online_dpo_trainer_schedules_the_loraplus_optimizer(tmp_path, monkeypatch):
    """On the real TRL trainer: the attached LoRA+ optimizer survives
    ``create_optimizer_and_scheduler`` (the call ``train()`` makes), and the
    scheduler is built around it, so warmup and decay reach the B group."""
    import soup_cli.trainer.online_dpo as od
    from soup_cli.eval.judge import JudgeScore
    from soup_cli.trainer.online_dpo import OnlineDPOTrainerWrapper

    class _Judge:
        def compare_pair(self, prompt, resp_a, resp_b):
            return 0

        def evaluate(self, prompt, response, category="default"):
            return JudgeScore(prompt=prompt, response=response, weighted_score=1.0)

    monkeypatch.chdir(tmp_path)
    od._ONLINE_DPO_JUDGE_OVERRIDE = _Judge()
    try:
        wrapper = OnlineDPOTrainerWrapper(_config(RATIO), device="cpu")
        wrapper.setup(_DATASET)
    finally:
        od._ONLINE_DPO_JUDGE_OVERRIDE = None

    trainer = wrapper.trainer
    attached = trainer.optimizer
    assert attached is not None, "loraplus_lr_ratio was ignored on online_dpo"
    assert B_GROUP_LR in _group_lrs(attached)

    trainer.create_optimizer_and_scheduler(num_training_steps=10)

    assert trainer.optimizer is attached
    assert trainer.lr_scheduler.optimizer is attached
    assert B_GROUP_LR in {round(lr, 12) for lr in trainer.lr_scheduler.base_lrs}
