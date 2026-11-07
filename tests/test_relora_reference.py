"""#1835: a ReLoRA restart moves the frozen reference of a loss that has one.

trl builds the reference of DPO, IPO, KTO, BCO and GRPO on the policy's own
base weights: a frozen ``ref`` adapter, or the model with its adapter
disabled. A ReLoRA restart merges the trained adapter into those weights, so
from the first restart on the "frozen" reference is the policy as of that
step, and nothing says so. ``relora_steps`` is refused for those tasks.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from soup_cli.config.schema import SoupConfig

TRAIN = "tests/fixtures/sample_train.jsonl"


def _cfg(task, data=None, **training):
    return SoupConfig(
        base="test/model",
        task=task,
        data={"train": TRAIN, **(data or {})},
        training={"batch_size": 2, "quantization": "none", **training},
    )


def _refusal(task, data=None, **training):
    with pytest.raises(ValidationError) as excinfo:
        _cfg(task, data, relora_steps=100, **training)
    return excinfo.value.errors()[0]["msg"]


_DPO_DATA = {"format": "dpo"}

#: (task, data, training, how the message names it)
WITH_A_REFERENCE = [
    ("dpo", _DPO_DATA, {}, "task='dpo'"),
    ("ipo", _DPO_DATA, {}, "task='ipo'"),
    ("bco", _DPO_DATA, {}, "task='bco'"),
    ("kto", {"format": "kto"}, {}, "task='kto'"),
    ("grpo", None, {}, "task='grpo', grpo_beta="),
    (
        "preference",
        _DPO_DATA,
        {"preference_loss": "dpo"},
        "task='preference', preference_loss='dpo'",
    ),
    (
        "preference",
        _DPO_DATA,
        {"preference_loss": "ipo"},
        "task='preference', preference_loss='ipo'",
    ),
    (
        "preference",
        _DPO_DATA,
        {"preference_loss": "bco"},
        "task='preference', preference_loss='bco'",
    ),
    (
        "preference",
        _DPO_DATA,
        {"preference_loss_weights": {"dpo": 0.7, "simpo": 0.3}},
        "task='preference', preference_loss_weights with dpo",
    ),
    (
        "preference",
        _DPO_DATA,
        {"preference_loss_weights": {"simpo": 0.5, "ipo": 0.3, "dpo": 0.2}},
        "task='preference', preference_loss_weights with dpo, ipo",
    ),
    (
        "preference",
        _DPO_DATA,
        # Loads, and is refused only when it runs (bco beside a paired loss).
        {"preference_loss_weights": {"bco": 0.6, "simpo": 0.4}},
        "task='preference', preference_loss_weights with bco",
    ),
]

WITHOUT_ONE = [
    ("sft", None, {}),
    ("orpo", _DPO_DATA, {}),
    ("simpo", _DPO_DATA, {}),
    ("grpo", None, {"grpo_beta": 0}),
    ("preference", _DPO_DATA, {"preference_loss": "simpo"}),
    ("preference", _DPO_DATA, {"preference_loss": "orpo"}),
    ("preference", _DPO_DATA, {"preference_loss_weights": {"simpo": 0.7, "orpo": 0.3}}),
]


class TestTheRefusal:
    @pytest.mark.parametrize(("task", "data", "training", "named"), WITH_A_REFERENCE)
    def test_a_task_with_a_reference_refuses_relora_with_the_reason(
        self, task, data, training, named
    ):
        message = _refusal(task, data, **training)

        assert f"training.relora_steps cannot be used here ({named}" in message
        assert "frozen reference" in message and "merges the trained adapter" in message
        if "preference_loss_weights" in training:
            # A blend has a second way out: drop the terms that need a reference.
            terms = named.rsplit(" with ", 1)[1]
            assert message.rstrip().endswith(
                f"Remove training.relora_steps, or remove {terms} from "
                "training.preference_loss_weights."
            )
        else:
            assert message.rstrip().endswith("Remove training.relora_steps.")

    @pytest.mark.parametrize(("task", "data", "training", "named"), WITH_A_REFERENCE)
    def test_the_same_config_loads_without_relora(self, task, data, training, named):
        """The premise of each row above: it is ``relora_steps`` that is refused."""
        assert _cfg(task, data, **training).training.relora_steps is None

    @pytest.mark.parametrize(("task", "data", "training"), WITHOUT_ONE)
    def test_a_task_without_a_reference_keeps_relora(self, task, data, training):
        assert _cfg(task, data, relora_steps=100, **training).training.relora_steps == 100


class TestThePremise:
    """What the refusal prevents, on a real run. The config is built without
    ``relora_steps`` and given it afterwards, past the validator."""

    @pytest.mark.parametrize("task", ["dpo", "ipo"])
    def test_a_restart_moves_trls_frozen_reference(self, tmp_path, monkeypatch, task):
        """trl's own ``compute_ref_log_probs`` on one fixed batch: unchanged
        until the restart at step 2, and another model's numbers from then on.
        If this fails, trl or PEFT keeps the reference apart from the base the
        restart merges into, and the refusal can go."""
        torch = pytest.importorskip("torch")
        from transformers import TrainerCallback

        from soup_cli.config.loader import load_config_from_string
        from soup_cli.trainer.dpo import DPOTrainerWrapper
        from soup_cli.trainer.ipo import IPOTrainerWrapper
        from tests.test_issue1425_dpo_ipo_blend import ROWS, _tiny_model_dir

        monkeypatch.chdir(tmp_path)
        base = _tiny_model_dir(tmp_path)
        cfg = load_config_from_string(
            f"base: {base.as_posix()}\ntask: {task}\n"
            "data: {train: ./unused.jsonl, format: dpo, max_length: 64}\n"
            "training:\n  epochs: 3\n  batch_size: 2\n  lr: 0.05\n  quantization: none\n"
            "  lora: {r: 4, alpha: 8, dropout: 0.0}\noutput: ./out\n"
        )
        cfg = cfg.model_copy(
            update={"training": cfg.training.model_copy(update={"relora_steps": 2})}
        )
        wrapper = (DPOTrainerWrapper if task == "dpo" else IPOTrainerWrapper)(cfg, device="cpu")
        wrapper.setup({"train": ROWS})
        trainer = wrapper.trainer
        batch = trainer._prepare_inputs(next(iter(trainer.get_train_dataloader())))
        seen = {}

        def read(step):
            training = trainer.model.training
            with torch.no_grad():
                seen[step] = torch.cat(list(trainer.compute_ref_log_probs(batch))).clone()
            trainer.model.train(training)

        class _Probe(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                read(0)

            def on_step_end(self, args, state, control, **kwargs):
                read(state.global_step)

        trainer.add_callback(_Probe())

        wrapper.train()

        assert sorted(seen) == [0, 1, 2, 3]
        assert torch.equal(seen[1], seen[0])
        assert float((seen[2] - seen[0]).abs().max()) > 0.1
