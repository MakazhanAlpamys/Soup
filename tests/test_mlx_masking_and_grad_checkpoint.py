"""MLX SFT silently ignored two schema fields it has an mlx-lm home for.

``data.train_on_responses_only`` defaults to True and every transformers
trainer honours it, but the MLX path never set it: mlx-lm's
``create_dataset`` reads ``mask_prompt`` off the args object with a
``getattr`` default of False, and ``TrainingArgs`` declares no such field,
so the prompt was always trained on. On a chat dataset that is invisible;
on one with a long shared system prompt — a schema, a retrieved document —
the prompt tokens are almost all of the loss.

``training.gradient_checkpointing`` had the same shape: mlx-lm's
``TrainingArgs.grad_checkpoint`` exists, and this trainer passed a
hardcoded ``False``, so the one knob that makes a long-prompt dataset fit
in Metal memory did nothing here. ``adapter_config.json`` wrote both
values as literal ``False`` too, so neither drift was visible afterwards.

Same caveat as test_issue684_mlx_grad_accumulation, whose fake-mlx harness
this reuses: real control flow through ``train()``, but fakes standing in
for real mlx-lm behaviour rather than an end-to-end Metal run.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

from test_issue684_mlx_grad_accumulation import (  # noqa: E402
    _FakeMlxModel,
    _install_fake_mlx,
)


def _wrapper(tmp_path, *, data=None, **training):
    from soup_cli.config.schema import DataConfig, SoupConfig, TrainingConfig
    from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper

    cfg = SoupConfig(
        base="mlx-community/Llama-3.1-8B-Instruct-4bit",
        task="sft",
        backend="mlx",
        data=DataConfig(train="./data/train.jsonl", format="chatml", **(data or {})),
        training=TrainingConfig(epochs=1, lr=1e-4, batch_size=1, **training),
        output=str(tmp_path),
    )
    wrapper = MLXSFTTrainerWrapper(cfg)
    wrapper.model = _FakeMlxModel()
    wrapper.tokenizer = object()
    wrapper._dataset = {"train": [{"text": "hi"}] * 4, "val": []}
    return wrapper


def _dataset_args_seen(monkeypatch):
    """Capture the args object mlx-lm's own ``create_dataset`` would read
    ``mask_prompt`` from — the only place the value has any effect."""
    _install_fake_mlx(monkeypatch)
    seen = []
    datasets = sys.modules["mlx_lm.tuner.datasets"]
    monkeypatch.setattr(
        datasets, "create_dataset", lambda rows, tokenizer, args: seen.append(args) or rows
    )
    return seen


class TestPromptMaskingFollowsTrainOnResponsesOnly:
    def test_the_schema_default_masks_the_prompt(self, tmp_path, monkeypatch):
        seen = _dataset_args_seen(monkeypatch)

        _wrapper(tmp_path).train()

        assert getattr(seen[0], "mask_prompt", False) is True, (
            "data.train_on_responses_only defaults to True, so create_dataset "
            "must see mask_prompt=True — its getattr default of False means an "
            "unset attribute trains on the prompt with no warning"
        )

    def test_opting_out_trains_on_the_prompt(self, tmp_path, monkeypatch):
        seen = _dataset_args_seen(monkeypatch)

        _wrapper(tmp_path, data={"train_on_responses_only": False}).train()

        assert getattr(seen[0], "mask_prompt", None) is False

    def test_adapter_config_records_the_effective_masking(self, tmp_path, monkeypatch):
        _install_fake_mlx(monkeypatch)

        _wrapper(tmp_path).train()

        adapter_config = json.loads((tmp_path / "adapter_config.json").read_text())
        assert adapter_config["mask_prompt"] is True, (
            "adapter_config.json must record what training actually did, not a "
            "hardcoded False that hides the drift after the run"
        )


class TestGradientCheckpointingReachesMlxLm:
    def test_enabled_by_bool(self, tmp_path, monkeypatch):
        captured_calls = _install_fake_mlx(monkeypatch)

        _wrapper(tmp_path, gradient_checkpointing=True).train()

        assert captured_calls[0]["args"].grad_checkpoint is True

    def test_enabled_by_tier_string(self, tmp_path, monkeypatch):
        # mlx-lm checkpoints all blocks or none, so every tier means "on"
        # here — but a tier must not read as False the way a hardcoded
        # literal did.
        captured_calls = _install_fake_mlx(monkeypatch)

        _wrapper(tmp_path, gradient_checkpointing="selective").train()

        assert captured_calls[0]["args"].grad_checkpoint is True

    def test_the_schema_default_leaves_it_off(self, tmp_path, monkeypatch):
        captured_calls = _install_fake_mlx(monkeypatch)

        _wrapper(tmp_path).train()

        assert captured_calls[0]["args"].grad_checkpoint is False

    def test_adapter_config_records_the_effective_value(self, tmp_path, monkeypatch):
        _install_fake_mlx(monkeypatch)

        _wrapper(tmp_path, gradient_checkpointing="full").train()

        adapter_config = json.loads((tmp_path / "adapter_config.json").read_text())
        assert adapter_config["grad_checkpoint"] is True
