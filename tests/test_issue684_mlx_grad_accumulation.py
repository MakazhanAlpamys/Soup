"""#684: MLX SFT ignores gradient_accumulation_steps.

MLXSFTTrainerWrapper.train() built mlx-lm's TrainingArgs without a
grad_accumulation_steps kwarg, so mlx-lm's own dataclass default of 1
always applied regardless of the validated schema field
training.gradient_accumulation_steps (default 4, ge=1). The written
adapter_config.json also hardcoded grad_accumulation_steps: 1 rather than
reading back the effective value, so the drift could not even be detected
from the output afterwards.

Caveat that still matters: MLX only runs on Apple Silicon. Both tests here
execute the real train() method body against minimal fake mlx / mlx_lm
modules registered in sys.modules (the same harness
test_issue634_mlx_resume.py uses) - real control flow, real call
ordering, not source-text inspection - but the fakes stand in for real
mlx-lm behaviour, not a substitute for an end-to-end run on real Metal.
"""

from __future__ import annotations

import json
import sys
import types


def _install_fake_mlx(monkeypatch):
    """Register minimal fake ``mlx`` / ``mlx_lm`` modules in ``sys.modules``
    so ``MLXSFTTrainerWrapper.train()`` runs for real, without Apple
    Silicon or the real packages installed. Same harness as
    test_issue634_mlx_resume.py's helper of the same name.
    """
    mlx = types.ModuleType("mlx")
    mlx_core = types.ModuleType("mlx.core")
    mlx_optimizers = types.ModuleType("mlx.optimizers")
    mlx_optimizers.AdamW = lambda **kwargs: object()

    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm_tuner = types.ModuleType("mlx_lm.tuner")

    mlx_lm_tuner_callbacks = types.ModuleType("mlx_lm.tuner.callbacks")

    class _FakeTrainingCallback:
        pass

    mlx_lm_tuner_callbacks.TrainingCallback = _FakeTrainingCallback

    mlx_lm_tuner_datasets = types.ModuleType("mlx_lm.tuner.datasets")
    mlx_lm_tuner_datasets.CacheDataset = lambda rows: rows
    mlx_lm_tuner_datasets.create_dataset = lambda rows, tokenizer, args: rows

    mlx_lm_tuner_trainer = types.ModuleType("mlx_lm.tuner.trainer")

    class _FakeTrainingArgs:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    mlx_lm_tuner_trainer.TrainingArgs = _FakeTrainingArgs

    captured_calls = []

    def _fake_train(**kwargs):
        captured_calls.append(kwargs)
        callback = kwargs.get("training_callback")
        if callback is not None:
            callback.on_train_loss_report({"train_loss": 0.5})

    mlx_lm_tuner_trainer.train = _fake_train

    mlx_lm_tuner_utils = types.ModuleType("mlx_lm.tuner.utils")
    mlx_lm_tuner_utils.linear_to_lora_layers = lambda model, num_layers, config: None

    fake_modules = {
        "mlx": mlx,
        "mlx.core": mlx_core,
        "mlx.optimizers": mlx_optimizers,
        "mlx_lm": mlx_lm,
        "mlx_lm.tuner": mlx_lm_tuner,
        "mlx_lm.tuner.callbacks": mlx_lm_tuner_callbacks,
        "mlx_lm.tuner.datasets": mlx_lm_tuner_datasets,
        "mlx_lm.tuner.trainer": mlx_lm_tuner_trainer,
        "mlx_lm.tuner.utils": mlx_lm_tuner_utils,
    }
    for name, module in fake_modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    return captured_calls


class _FakeMlxModel:
    def __init__(self):
        self.layers = []

    def freeze(self):
        pass

    def load_weights(self, path, strict=True):
        pass


def _mlx_wrapper(tmp_path, **training):
    from soup_cli.config.schema import DataConfig, SoupConfig, TrainingConfig
    from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper

    cfg = SoupConfig(
        base="mlx-community/Llama-3.1-8B-Instruct-4bit",
        task="sft",
        backend="mlx",
        data=DataConfig(train="./data/train.jsonl", format="chatml"),
        training=TrainingConfig(**training),
        output=str(tmp_path),
    )
    wrapper = MLXSFTTrainerWrapper(cfg)
    wrapper.model = _FakeMlxModel()
    wrapper.tokenizer = object()
    wrapper._dataset = {"train": [{"text": "hi"}], "val": []}
    return wrapper


class TestTrainingArgsCarriesTheConfiguredGradAccumulationSteps:
    def test_configured_value_reaches_training_args(self, tmp_path, monkeypatch):
        captured_calls = _install_fake_mlx(monkeypatch)
        wrapper = _mlx_wrapper(
            tmp_path, epochs=1, lr=1e-4, batch_size=1, gradient_accumulation_steps=6
        )

        wrapper.train()

        assert len(captured_calls) == 1
        args = captured_calls[0]["args"]
        assert args.grad_accumulation_steps == 6, (
            "TrainingArgs must carry the schema's gradient_accumulation_steps, "
            "not mlx-lm's own dataclass default of 1"
        )

    def test_a_different_configured_value_is_not_a_coincidence(self, tmp_path, monkeypatch):
        """Same assertion at a second value, so this cannot pass by
        happening to match one hardcoded number."""
        captured_calls = _install_fake_mlx(monkeypatch)
        wrapper = _mlx_wrapper(
            tmp_path, epochs=1, lr=1e-4, batch_size=1, gradient_accumulation_steps=2
        )

        wrapper.train()

        args = captured_calls[0]["args"]
        assert args.grad_accumulation_steps == 2


class TestAdapterConfigJsonRecordsTheEffectiveValue:
    def test_written_metadata_matches_the_configured_value(self, tmp_path, monkeypatch):
        _install_fake_mlx(monkeypatch)
        wrapper = _mlx_wrapper(
            tmp_path, epochs=1, lr=1e-4, batch_size=1, gradient_accumulation_steps=8
        )

        wrapper.train()

        adapter_config = json.loads((tmp_path / "adapter_config.json").read_text())
        assert adapter_config["grad_accumulation_steps"] == 8, (
            "adapter_config.json must record the value training actually used, "
            "not a hardcoded 1 that can never reveal the drift after the fact"
        )
