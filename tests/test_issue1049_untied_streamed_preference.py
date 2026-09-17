"""#1049 — an untied streamed checkpoint must take a preference step.

`embed_tokens` and an untied `lm_head` share ONE device slot, and
`StreamedLargeLayer.forward` handed `functional_call` a view of it. A projection
saves its weight to build `grad wrt x`, so the reference forward of a preference
loss refilled those bytes in place and the policy backward died with
``one of the variables needed for gradient computation has been modified by an
inplace operation``.

The streaming preference tests all used TIED fixtures, and a tied checkpoint
streams one key, so nothing here was covered. Llama-3-8B-class checkpoints are
untied, which is the common real-world shape.
"""

from __future__ import annotations

import pytest
from test_v07204 import (
    _ALL_PREFERENCE,
    _batch_on,
    _build_streamed_wrapper,
    _loss_of,
    _match_streamed_dtype,
    _randomise_lora_b,
    _sync_adapters,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.mark.parametrize("task", [*_ALL_PREFERENCE, "sft"])
def test_an_untied_streamed_train_step_completes(tmp_path, monkeypatch, task):
    """dpo and kto raise on main; orpo, simpo and sft pass there and must keep passing."""
    wrapper, _, _ = _build_streamed_wrapper(
        tmp_path, monkeypatch, task=task, device="cpu", tie=False
    )
    wrapper.trainer.args.max_steps = 1
    try:
        wrapper.trainer.train()
    finally:
        wrapper._close_stream_runtime()


@pytest.mark.parametrize("task", _ALL_PREFERENCE)
def test_an_untied_streamed_loss_and_gradients_match_a_resident_run(tmp_path, monkeypatch, task):
    """The private copy must not change what is computed: loss and every adapter
    gradient are bit-identical to a resident run of the same loss."""
    import torch
    from peft import LoraConfig, TaskType, get_peft_model

    wrapper, resident, _ = _build_streamed_wrapper(
        tmp_path, monkeypatch, task=task, device="cpu", tie=False
    )
    _randomise_lora_b(wrapper.model)

    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    resident_peft = get_peft_model(resident, lora_config)
    if "ref" in wrapper.model.peft_config:
        resident_peft.add_adapter("ref", lora_config)
    copied = _sync_adapters(resident_peft, wrapper.model)
    assert copied > 0, "vacuous: no adapter tensors copied"

    _match_streamed_dtype(resident_peft, wrapper.model)
    batch = _batch_on(wrapper.model, next(iter(wrapper.trainer.get_train_dataloader())))

    def loss_and_grads(model):
        model.train()
        model.zero_grad(set_to_none=True)
        loss = _loss_of(wrapper.trainer, model, batch)
        loss.mean().backward()
        grads = {
            name: param.grad.detach().clone()
            for name, param in model.named_parameters()
            if param.grad is not None and "lora_" in name
        }
        return loss.detach().clone(), grads

    try:
        streamed_loss, streamed_grads = loss_and_grads(wrapper.model)
        resident_loss, resident_grads = loss_and_grads(resident_peft)
    finally:
        wrapper._close_stream_runtime()

    assert torch.equal(streamed_loss, resident_loss), (
        f"{task}: untied streamed loss differs from resident by "
        f"{(streamed_loss - resident_loss).abs().max().item()}"
    )
    assert streamed_grads and set(streamed_grads) == set(resident_grads)
    for name, grad in streamed_grads.items():
        assert torch.equal(grad, resident_grads[name]), f"{task}: gradient differs at {name}"


def test_the_private_copy_is_taken_only_when_it_is_needed():
    """The copy costs one head-sized buffer, so it is taken only where the slot can
    be refilled while autograd holds the bytes: the projection of an untied
    checkpoint, under a live graph. Not for a tied slot, not under ``no_grad``, and
    not for the embedding, whose backward reads the indices rather than the weight.
    Pinned directly, because the memory cost is the trade."""
    import torch
    import torch.nn as nn

    from soup_cli.utils.layer_stream_runtime import _streamed_large_layer_class

    class _Pool:
        """One slot, exactly like LargeLayerBufferPool: `wait` returns a view of it."""

        def __init__(self, keys):
            self.specs = dict.fromkeys(keys, ((4, 3), "float32"))
            self.buffer = torch.arange(12.0).view(4, 3)

        def wait(self, key):
            return self.buffer

    def recorder(base):
        class _Recorder(base):
            """Records the weight `functional_call` substituted in."""

            def __init__(self):
                if base is nn.Embedding:
                    super().__init__(4, 3)
                else:
                    super().__init__(3, 4, bias=False)
                self.weight.requires_grad_(False)
                self.seen = []

            def forward(self, x):
                self.seen.append(self.weight)
                return super().forward(x)

        return _Recorder()

    cls = _streamed_large_layer_class()

    def weight_seen_by(pool, base, grad):
        inner = recorder(base)
        layer = cls(inner, "lm_head", pool)
        x = torch.zeros(1, 3) if base is nn.Linear else torch.zeros(1, dtype=torch.long)
        with torch.set_grad_enabled(grad):
            layer(x)
        return inner.seen[0]

    untied, tied = _Pool(["embed_tokens", "lm_head"]), _Pool(["embed_tokens"])
    projection = weight_seen_by(untied, nn.Linear, grad=True)
    projection_nograd = weight_seen_by(untied, nn.Linear, grad=False)
    projection_tied = weight_seen_by(tied, nn.Linear, grad=True)
    embedding = weight_seen_by(untied, nn.Embedding, grad=True)

    assert projection.data_ptr() != untied.buffer.data_ptr(), (
        "an untied projection under a live graph must hand autograd a private copy"
    )
    assert torch.equal(projection, untied.buffer), "the copy must be the same bytes"
    assert projection_tied.data_ptr() == tied.buffer.data_ptr(), (
        "a tied checkpoint streams one key, so its slot is never refilled: no copy"
    )
    assert projection_nograd.data_ptr() == untied.buffer.data_ptr(), (
        "no backward can follow under no_grad, so no copy"
    )
    assert embedding.data_ptr() == untied.buffer.data_ptr(), (
        "embedding_backward reads the indices, not the weight values: no copy"
    )
