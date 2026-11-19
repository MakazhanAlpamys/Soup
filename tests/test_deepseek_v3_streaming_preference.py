"""Preference losses (dpo, orpo, simpo, kto) through a streamed DeepSeek-V3 model.

``tests/test_deepseek_v3_streaming.py`` gates the architecture on the language-model
loss only. A preference step adds a reference forward with the adapter disabled
before the policy forward and backward, through the same streamed layers (MLA
attention, shared and routed experts, the router bias streamed at fp32). Same tiny
checkpoint, real ``setup()`` path, as ``_build_streamed_wrapper`` does for Llama.
"""

from __future__ import annotations

import math

import pytest

from tests.conftest import accelerator_device
from tests.test_deepseek_v3_streaming import _LORA_TARGETS, _tiny_deepseek_v3_dir
from tests.test_v07204 import (
    _MIN_BATCH,
    _TASK_ROWS,
    _batch_on,
    _loss_of,
    _match_streamed_dtype,
    _randomise_lora_b,
    _stream_cfg,
    _sync_adapters,
    _wrapper_for,
    _write_tiny_tokenizer,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

_Q_LORA_RANKS = pytest.mark.parametrize("q_lora_rank", [24, None], ids=["q_lora", "q_proj"])


def _build_streamed_deepseek_wrapper(tmp_path, monkeypatch, task, q_lora_rank, device=None):
    """The tiny DeepSeek-V3 checkpoint through the task wrapper's real ``setup()``."""
    weights, resident = _tiny_deepseek_v3_dir(tmp_path, q_lora_rank=q_lora_rank)
    _write_tiny_tokenizer(weights)
    monkeypatch.setenv("SOUP_LAYER_STREAM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.chdir(tmp_path)
    cfg = _stream_cfg(
        weights,
        tmp_path / "out",
        task=task,
        batch_size=_MIN_BATCH.get(task, 1),
        lora={"r": 4, "alpha": 8, "target_modules": list(_LORA_TARGETS)},
    )
    wrapper = _wrapper_for(task)(cfg, device=device or accelerator_device())
    wrapper.setup({"train": _TASK_ROWS[task](8)})
    return wrapper, resident


def _resident_peft_like(wrapper, resident):
    """The resident model with the streamed wrapper's adapters, weights included."""
    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
        target_modules=list(_LORA_TARGETS), task_type=TaskType.CAUSAL_LM,
    )
    resident_peft = get_peft_model(resident, lora_config)
    # TRL 0.29 snapshots a non-zero initial policy as a frozen ``ref`` adapter.
    if "ref" in wrapper.model.peft_config:
        resident_peft.add_adapter("ref", lora_config)
    copied = _sync_adapters(resident_peft, wrapper.model)
    assert copied > 0, "vacuous: no adapter tensors copied"
    return _match_streamed_dtype(resident_peft, wrapper.model)


def _loss_and_lora_grads(trainer, model, batch):
    from soup_cli.utils.layer_stream_runtime import canonical_named_parameters

    model.train()
    model.zero_grad(set_to_none=True)
    loss = _loss_of(trainer, model, batch)
    loss.mean().backward()
    grads = {
        name: param.grad.detach().clone()
        for name, param in canonical_named_parameters(model)
        if param.grad is not None and "lora_" in name
    }
    return loss.detach().clone(), grads


@_Q_LORA_RANKS
def test_streamed_dpo_loss_and_lora_gradients_match_resident_bit_exactly(
    tmp_path, monkeypatch, q_lora_rank
):
    """CPU float32: the streamed dpo loss and every LoRA gradient equal a resident run's."""
    import torch

    wrapper, resident = _build_streamed_deepseek_wrapper(
        tmp_path, monkeypatch, "dpo", q_lora_rank, device="cpu"
    )
    try:
        _randomise_lora_b(wrapper.model)
        resident_peft = _resident_peft_like(wrapper, resident)
        batch = _batch_on(wrapper.model, next(iter(wrapper.trainer.get_train_dataloader())))
        streamed_loss, streamed_grads = _loss_and_lora_grads(wrapper.trainer, wrapper.model, batch)
        resident_loss, resident_grads = _loss_and_lora_grads(wrapper.trainer, resident_peft, batch)
    finally:
        wrapper._close_stream_runtime()

    assert torch.equal(streamed_loss, resident_loss), (
        f"streamed dpo loss differs from resident by "
        f"{(streamed_loss - resident_loss).abs().max().item()}"
    )
    assert streamed_grads and set(streamed_grads) == set(resident_grads)
    # The first (dense) layer's adapters get gradient through the streamed experts above.
    assert any(".layers.0." in name for name in streamed_grads)
    assert any(float(grad.abs().sum()) > 0.0 for grad in streamed_grads.values())
    for name, grad in streamed_grads.items():
        assert torch.equal(grad, resident_grads[name]), f"gradient differs at {name}"


@_Q_LORA_RANKS
@pytest.mark.parametrize("task", ["orpo", "simpo", "kto"])
def test_a_streamed_preference_train_step_completes_with_a_finite_loss(
    tmp_path, monkeypatch, task, q_lora_rank
):
    """No ``device=``: these losses move the batch to ``accelerator.device`` inside TRL."""
    wrapper, _ = _build_streamed_deepseek_wrapper(tmp_path, monkeypatch, task, q_lora_rank)
    wrapper.trainer.args.max_steps = 1
    try:
        wrapper.trainer.train()
        loads = wrapper._stream_runtime.pool.loads
    finally:
        wrapper._close_stream_runtime()
    losses = [entry["loss"] for entry in wrapper.trainer.state.log_history if "loss" in entry]
    assert losses and math.isfinite(losses[0]), losses
    assert loads > 0, "no layer was streamed"
