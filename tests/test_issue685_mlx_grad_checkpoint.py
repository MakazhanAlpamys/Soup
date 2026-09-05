"""#685: MLX SFT ignores gradient_checkpointing.

MLXSFTTrainerWrapper.train() builds mlx-lm's TrainingArgs without a
grad_checkpoint kwarg, so mlx-lm's own dataclass default of False always
applies regardless of the validated schema field
training.gradient_checkpointing (bool or tier literal, config/schema.py).
The written adapter_config.json also hardcoded grad_checkpoint: false
rather than reading back the effective value, so the drift could not even
be detected from the output afterwards.

Sibling of test_issue684_mlx_grad_accumulation.py (#684), same file, same
harness: real train() control flow against minimal fake mlx / mlx_lm
modules registered in sys.modules, not source-text inspection.

Tier-string resolution: gradient_checkpointing accepts a plain bool OR one
of "selective"/"medium"/"full"/"auto" (a granularity mlx-lm has no concept
of: TrainingArgs.grad_checkpoint is a single bool). This mirrors the
repo's own established convention for the same field on every other
backend: commands/train.py's hardware-fit predictor
(`gc = bool(getattr(tcfg, "gradient_checkpointing", False))`) and
layer_stream.should_enable_hf_gradient_checkpointing
(`bool(gradient_checkpointing) and not stream_layers`, used by the
DPO/KTO/ORPO/SimPO wrappers) both resolve the same tiered field with a
plain bool() coercion rather than branching per tier. Any non-empty tier
string is truthy, so "selective"/"medium"/"full"/"auto" all resolve to
True (checkpointing enabled) and False/absent resolves to False, exactly
as those two call sites already treat it.

Caveat that still matters: MLX only runs on Apple Silicon. These tests
exercise the real function body via the fake-module harness, not a
substitute for an end-to-end run on real Metal hardware, and this PR does
not attempt to distinguish mlx-lm's checkpointing granularity (there is
none to distinguish; it wraps only the first decoder layer type,
uniformly).
"""

from __future__ import annotations

import json
import sys
import types

import pytest


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
        self.layers = [object()]

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


@pytest.mark.parametrize(
    "requested,expected",
    [
        (True, True),
        (False, False),
        ("selective", True),
        ("medium", True),
        ("full", True),
        ("auto", True),
    ],
)
def test_grad_checkpoint_reaches_training_args(
    tmp_path, monkeypatch, requested, expected
):
    """The value mlx-lm's TrainingArgs actually receives, for every value the
    schema accepts. Before the fix this kwarg was never passed at all, so
    mlx-lm's own dataclass default (False) applied every time, and a test
    that only inspects adapter_config.json cannot see that, since the
    written metadata comes from the same local variable regardless of
    whether it is threaded into the TrainingArgs(...) call. Deleting
    grad_checkpoint=grad_checkpoint from that call must fail here even
    though it changes nothing about the JSON write."""
    captured_calls = _install_fake_mlx(monkeypatch)
    wrapper = _mlx_wrapper(
        tmp_path, epochs=1, lr=1e-4, batch_size=1, gradient_checkpointing=requested
    )

    wrapper.train()

    args = captured_calls[0]["args"]
    assert args.grad_checkpoint is expected, (
        f"gradient_checkpointing={requested!r} must resolve to "
        f"grad_checkpoint={expected} on the TrainingArgs mlx-lm receives"
    )


@pytest.mark.parametrize(
    "requested,expected",
    [
        (True, True),
        (False, False),
        ("selective", True),
        ("medium", True),
        ("full", True),
        ("auto", True),
    ],
)
def test_grad_checkpoint_is_recorded_in_adapter_config(
    tmp_path, monkeypatch, requested, expected
):
    """The value recorded in adapter_config.json, for every value the schema
    accepts. Kept separate from the TrainingArgs assertion above so the two
    halves have distinct kill-sets: reverting only the metadata write while
    keeping the TrainingArgs kwarg fails only this test, and deleting the
    TrainingArgs kwarg while keeping the metadata write fails only the
    other one."""
    _install_fake_mlx(monkeypatch)
    wrapper = _mlx_wrapper(
        tmp_path, epochs=1, lr=1e-4, batch_size=1, gradient_checkpointing=requested
    )

    wrapper.train()

    written = json.loads((tmp_path / "adapter_config.json").read_text())
    assert written["grad_checkpoint"] is expected, (
        f"gradient_checkpointing={requested!r} must resolve to "
        f"grad_checkpoint={expected} in the written adapter_config.json"
    )


def test_default_is_unchanged(tmp_path, monkeypatch):
    """No explicit setting keeps the schema default (False) end to end,
    on both the TrainingArgs kwarg and the written metadata."""
    captured_calls = _install_fake_mlx(monkeypatch)
    wrapper = _mlx_wrapper(tmp_path, epochs=1, lr=1e-4, batch_size=1)

    wrapper.train()

    assert captured_calls[0]["args"].grad_checkpoint is False
    written = json.loads((tmp_path / "adapter_config.json").read_text())
    assert written["grad_checkpoint"] is False


class TestTierFlatteningIsAnnounced:
    """The bool()-flattening from a tier string to a single on/off switch is
    silent otherwise: a user asking for 'selective' gets 'full' with no
    indication anything was collapsed. Same warning mechanism and same
    _warnings_for-style harness as test_issue353_mlx_seed_warning.py."""

    @staticmethod
    def _warnings_for(monkeypatch, **training):
        from io import StringIO

        from rich.console import Console

        from soup_cli.config.schema import DataConfig, SoupConfig, TrainingConfig
        from soup_cli.trainer import mlx_sft

        buffer = StringIO()
        monkeypatch.setattr(mlx_sft, "console", Console(file=buffer, width=200))
        cfg = SoupConfig(
            base="mlx-community/Llama-3.1-8B-Instruct-4bit",
            task="sft",
            backend="mlx",
            data=DataConfig(train="./data/train.jsonl", format="chatml"),
            training=TrainingConfig(**training),
            output="./out",
        )
        mlx_sft.MLXSFTTrainerWrapper(cfg)._check_unsupported()
        return buffer.getvalue()

    def test_a_tier_string_is_named(self, monkeypatch):
        out = self._warnings_for(
            monkeypatch, epochs=1, lr=1e-4, batch_size=1, gradient_checkpointing="selective"
        )
        assert "gradient_checkpointing" in out
        assert "selective" in out

    def test_a_plain_bool_says_nothing(self, monkeypatch):
        """CONTROL. A bool is not a tier; flattening a bool to itself has
        nothing to announce."""
        out = self._warnings_for(
            monkeypatch, epochs=1, lr=1e-4, batch_size=1, gradient_checkpointing=True
        )
        assert "gradient_checkpointing" not in out
