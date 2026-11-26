"""The backward of the L2L step runs outside ``torch.autocast``, as in accumulation.

HF's loop calls ``backward`` after the forward's autocast block has exited. The L2L step is
called inside that block (it runs forward, recompute and backward itself), so its backward
calls must switch autocast off, or the gradient kernels run under another dtype policy than
accumulation's and the gradients differ in bf16 (the losses then disagree with accumulation).

The instrument is a hook on each LoRA parameter, which runs inside the backward and reads
the autocast state of the thread that computes the gradient.
"""

from __future__ import annotations

import pytest

from soup_cli.utils.l2l_step import L2LStep, capture_layer_call
from tests.test_l2l_probe_harness import (  # noqa: F401
    SEQ,
    _load,
    _micro_batches,
    streamed,
)


@pytest.fixture()
def sched():
    return _load("l2l_schedule")


def _shipped_store(k, n_layers=3, hidden=64):
    import torch

    from soup_cli.utils.l2l_activations import ActivationStore

    return ActivationStore(
        n_layers=n_layers,
        k=k,
        chunk_shape=(1, SEQ, hidden),
        dtype=torch.float32,
        device="cpu",
        pin=False,
    )


def _watch_autocast(model):
    import torch

    seen = []
    handles = [
        param.register_hook(lambda grad: seen.append(torch.is_autocast_enabled("cpu")))
        for name, param in model.named_parameters()
        if "lora_" in name and param.requires_grad
    ]
    return seen, handles


def test_the_hook_sees_autocast_when_a_backward_runs_inside_it(sched, streamed):  # noqa: F811
    import torch

    model, _runtime = streamed
    sched.make_non_vacuous(model)
    seen, handles = _watch_autocast(model)
    ids = _micro_batches(1)[0]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        model(input_ids=ids, labels=ids).loss.backward()
    for handle in handles:
        handle.remove()
    assert seen and all(seen)


def test_accumulation_backward_outside_autocast_is_seen_as_off(sched, streamed):  # noqa: F811
    import torch

    model, _runtime = streamed
    sched.make_non_vacuous(model)
    seen, handles = _watch_autocast(model)
    ids = _micro_batches(1)[0]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    for handle in handles:
        handle.remove()
    assert seen and not any(seen)


@pytest.mark.parametrize("k", [1, 2])
def test_l2l_backward_runs_with_autocast_off_inside_the_trainers_autocast(
    sched, streamed, k  # noqa: F811
):
    import torch

    model, runtime = streamed
    sched.make_non_vacuous(model)
    mbs = _micro_batches(k)
    step = L2LStep(model, runtime, capture_layer_call(model, mbs[0]))
    sched.zero_grads(model)
    seen, handles = _watch_autocast(model)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        step.run(mbs, _shipped_store(k))
    for handle in handles:
        handle.remove()
    assert seen and not any(seen)


def test_l2l_forward_and_recompute_still_run_under_autocast(sched, streamed):  # noqa: F811
    """Only the backward leaves autocast; the layer calls keep the trainer's mixed precision."""
    import torch

    model, runtime = streamed
    sched.make_non_vacuous(model)
    mbs = _micro_batches(1)
    step = L2LStep(model, runtime, capture_layer_call(model, mbs[0]))
    states = []
    layer = step.layers[0]
    real = layer._body

    def spy(*args, **kwargs):
        states.append((torch.is_autocast_enabled("cpu"), torch.is_grad_enabled()))
        return real(*args, **kwargs)

    layer._body = spy
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            step.run(mbs, _shipped_store(1))
    finally:
        layer._body = real
    assert len(states) == 2  # forward, then the recompute with grad
    assert all(on for on, _grad in states)
