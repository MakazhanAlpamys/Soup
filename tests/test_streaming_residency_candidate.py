"""Persistent slots must preserve checkpointed gradients under pool recycling."""

from types import SimpleNamespace

import pytest


def fixture(device="cpu"):
    import torch

    from soup_cli.utils.layer_stream_runtime import (
        LayerBufferPool,
        StreamedDecoderLayer,
        StreamPrefetcher,
    )

    class Source:
        nbytes = 6 * 4 * 4 * 4

        def __init__(self):
            self.calls = []
            self.weights = [torch.eye(4) * (0.5 + idx / 16) for idx in range(6)]
            if device == "cuda":
                self.weights = [value.pin_memory() for value in self.weights]

        def get(self, idx, name):
            assert name == "base"
            self.calls.append(idx)
            return self.weights[idx]

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Parameter(torch.empty(4, 4, device="meta"), requires_grad=False)
            self.adapter = torch.nn.Parameter(torch.full((4, 4), 0.03125, device=device))

        def forward(self, hidden):
            return (hidden @ self.base.T + hidden @ self.adapter.T).tanh()

    source = Source()
    pool = LayerBufferPool({"base": ((4, 4), "float32")}, device=device)
    stream = torch.cuda.Stream() if device == "cuda" else None
    prefetcher = StreamPrefetcher(pool, source, 6, stream=stream)
    model = torch.nn.Sequential(*[
        StreamedDecoderLayer(Inner(), idx, pool, prefetcher, {"base": "base"})
        for idx in range(6)
    ])
    runtime = SimpleNamespace(pool=pool, source=source, prefetcher=prefetcher,
                              n_layers=6, tier="ram")
    return model, runtime


def trajectory(model, runtime):
    import torch

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
    outputs = []
    for _step in range(4):
        runtime.prefetcher.prime()
        device = runtime.pool.buffers[0]["base"].device
        hidden = (torch.arange(8, device=device).reshape(2, 4).float() / 16).requires_grad_(True)
        result = model(hidden)
        loss = result.square().mean()
        loss.backward()
        outputs.extend([result.detach().clone(), loss.detach().clone(), hidden.grad.clone()])
        outputs.extend(p.grad.clone() for p in model.parameters() if p.requires_grad)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    outputs.extend(p.detach().clone() for p in model.parameters() if p.requires_grad)
    return outputs


@pytest.mark.parametrize("indices", [[], [2], [2, 3], [0, 1, 4, 5]])
def test_exact_checkpointed_training_and_no_resident_reloads(indices):
    import torch

    from benchmarks.harness.streaming_residency_candidate import install_persistent_layers

    reference, native = fixture()
    candidate, runtime = fixture()
    state = install_persistent_layers(runtime, indices)
    preloads = list(runtime.source.calls)
    expected = trajectory(reference, native)
    actual = trajectory(candidate, runtime)
    assert len(expected) == len(actual)
    assert all(torch.equal(first, second) for first, second in zip(expected, actual))
    assert preloads == indices
    assert not set(indices).intersection(runtime.source.calls[len(preloads):])
    assert state["resident_bytes"] == len(indices) * 64
    assert runtime.pool.n == 2
    assert len(runtime.pool.buffers) == 2 + len(indices)
    assert len(runtime.pool.generation) == len(runtime.pool.buffers)
    assert state["dynamic_loads"] > 0
    if indices:
        assert runtime.pool.loads < native.pool.loads
    for idx in indices:
        slot = runtime.pool.slot_for(idx)
        assert runtime.pool.owner[slot] == idx
        assert runtime.pool.generation[slot] == 1
        assert torch.equal(runtime.pool.buffers[slot]["base"], runtime.source.weights[idx])


@pytest.mark.parametrize("indices", [[True], [-1], [6], [2, 2], [3, 2], list(range(6))])
def test_invalid_indices_are_refused_before_allocation(indices):
    from benchmarks.harness.streaming_residency_candidate import install_persistent_layers

    _model, runtime = fixture()
    with pytest.raises(ValueError):
        install_persistent_layers(runtime, indices)
    assert len(runtime.pool.buffers) == 2


def test_persistent_owner_corruption_fails_loudly_and_double_install_refused():
    from benchmarks.harness.streaming_residency_candidate import install_persistent_layers

    _model, runtime = fixture()
    install_persistent_layers(runtime, [2])
    with pytest.raises(ValueError, match="already installed"):
        install_persistent_layers(runtime, [2])
    runtime.pool.owner[runtime.pool.slot_for(2)] = 3
    with pytest.raises(RuntimeError, match="owner changed"):
        runtime.pool.load_async(2, runtime.source)


@pytest.mark.gpu
@pytest.mark.parametrize("indices", [[2, 3], [0, 1, 4, 5]])
def test_cuda_async_recycling_preserves_complete_training(indices):
    import torch

    from benchmarks.harness.streaming_residency_candidate import install_persistent_layers

    reference, native = fixture("cuda")
    candidate, runtime = fixture("cuda")
    install_persistent_layers(runtime, indices)
    expected = trajectory(reference, native)
    actual = trajectory(candidate, runtime)
    torch.cuda.synchronize()
    assert all(torch.equal(first, second) for first, second in zip(expected, actual))
    assert not set(indices).intersection(runtime.source.calls[len(indices):])
