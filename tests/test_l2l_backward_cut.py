"""The L2L reverse walk skips only decoder layers below every trainable weight."""

from contextlib import closing

import pytest

from tests.test_l2l_probe_harness import _load, _micro_batches, streamed  # noqa: F401
from tests.test_l2l_step import _store


def _grads(model):
    return {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }


@pytest.mark.parametrize("layers", [[0, 2], [1, 2], [2], []])
@pytest.mark.parametrize(
    ("train_embedding", "split"), [(False, ()), (True, ()), (False, ("cpu",))]
)
def test_cut_step_matches_accumulation_and_skips_prefix(
    streamed,  # noqa: F811
    monkeypatch,
    layers,
    train_embedding,
    split,
):
    import torch

    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call, model_parts
    from soup_cli.utils.layer_select import adapter_params_by_layer, apply_layer_mask

    model, runtime = streamed
    harness = _load("l2l_schedule")
    harness.make_non_vacuous(model)
    _, owner = model_parts(model)
    if train_embedding:

        class AdaptedEmbedding(torch.nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base
                self.offset = torch.nn.Parameter(torch.zeros(64))

            def forward(self, ids):
                return self.base(ids) + self.offset

        monkeypatch.setattr(owner, "embed_tokens", AdaptedEmbedding(owner.embed_tokens))

    batches = _micro_batches(2)
    step = L2LStep(model, runtime, capture_layer_call(model, batches[0]), split_devices=split)
    with closing(_store(2)) as full_store:
        step.run(batches, full_store)
    full_loads = step.counters.layer_loads

    by_layer = adapter_params_by_layer(model)
    apply_layer_mask(by_layer, layers)
    model.zero_grad(set_to_none=True)
    if not layers and not train_embedding:
        # An ordinary accumulation loss has no graph when the whole model is frozen.
        ga_losses, ga_grads = None, {}
    else:
        ga_losses = harness.ga_step(model, batches)
        ga_grads = _grads(model)
    model.zero_grad(set_to_none=True)
    capture_layer_call(model, batches[0])

    with closing(_store(2)) as store:
        puts, gets = [], []
        put, get = store.put, store.get

        def track_put(layer, mb, tensor):
            puts.append((layer, mb))
            put(layer, mb, tensor)

        def track_get(layer, mb):
            gets.append((layer, mb))
            return get(layer, mb)

        monkeypatch.setattr(store, "put", track_put)
        monkeypatch.setattr(store, "get", track_get)
        losses = step.run(batches, store)

    floor = 0 if train_embedding else min(layers, default=3)
    assert puts == [(idx, mb) for idx in range(floor, 3) for mb in range(2)]
    assert gets == [(idx, mb) for idx in range(2, floor - 1, -1) for mb in range(2)]
    got = _grads(model)
    assert set(ga_grads) == set(got)
    for name, grad in got.items():
        torch.testing.assert_close(grad, ga_grads[name], rtol=1e-5, atol=1e-7)
    assert len(losses) == 2
    if ga_losses is not None:
        assert harness.losses_equal(ga_losses, losses)
    if floor >= 2:
        assert step.counters.layer_loads < full_loads
    assert runtime.prefetcher.backward_floor == 0
