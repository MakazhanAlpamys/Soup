"""Private slots preserve training and fail atomically before publication."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize("indices", [[], [2], [2, 3], [0, 1, 4, 5], list(range(6))])
def test_complete_updates_are_exact_and_private_slots_are_never_reloaded(indices):
    import torch

    from tests.stream_residency_fixture import fixture, trajectory

    reference, ordinary = fixture()
    candidate, runtime = fixture()
    pool = runtime.pool
    state = pool.reserve_layers(indices, runtime.source, runtime.prefetcher.stream)
    preloads = list(runtime.source.calls)
    actual = trajectory(candidate, runtime)
    expected = trajectory(reference, ordinary)
    assert len(actual) == len(expected)
    assert all(torch.equal(first, second) for first, second in zip(actual, expected))
    assert preloads == indices
    assert not set(indices).intersection(runtime.source.calls[len(preloads) :])
    assert pool.n == 2
    assert len(pool.buffers) == 2 + len(indices)
    assert state["bytes"] == pool.resident_bytes == 64 * len(indices)
    for idx in indices:
        assert pool.generation[pool.slot_for(idx)] == 1
    if indices:
        assert pool.loads < ordinary.pool.loads


@pytest.mark.parametrize("indices", [[True], [-1], [2, 2], [3, 2]])
def test_invalid_selection_is_refused_before_any_allocation(indices, monkeypatch):
    import torch

    from tests.stream_residency_fixture import fixture

    _model, runtime = fixture()

    def refuse(*args, **kwargs):
        pytest.fail("invalid selection allocated storage")

    monkeypatch.setattr(torch, "empty_like", refuse)
    with pytest.raises(ValueError):
        runtime.pool.reserve_layers(indices, runtime.source)


@pytest.mark.parametrize("corrupt", ["owner", "generation"])
def test_corrupt_private_slot_fails_in_load_and_wait(corrupt):
    from tests.stream_residency_fixture import fixture

    _model, runtime = fixture()
    pool = runtime.pool
    pool.reserve_layers([2], runtime.source)
    slot = pool.slot_for(2)
    if corrupt == "owner":
        pool.owner[slot] = 3
    else:
        pool.generation[slot] += 1
    for action in (lambda: pool.load_async(2, runtime.source), lambda: pool.wait(2)):
        with pytest.raises(RuntimeError, match="owner|holds layer|generation"):
            action()


def test_resident_load_is_a_real_no_op():
    from tests.stream_residency_fixture import fixture

    _model, runtime = fixture()
    pool = runtime.pool
    pool.reserve_layers([2], runtime.source)
    calls = list(runtime.source.calls)
    loads = pool.loads
    generations = list(pool.generation)
    slot = pool.slot_for(2)
    for _ in range(3):
        assert pool.load_async(2, runtime.source) == slot
    assert runtime.source.calls == calls
    assert pool.loads == loads
    assert pool.generation == generations


def test_repeated_reservation_requires_clear_and_clear_releases_wrapper_views():
    import weakref

    from soup_cli.utils.layer_stream_runtime import clear_stream_substitution_cache
    from tests.stream_residency_fixture import fixture, trajectory

    model, runtime = fixture()
    pool = runtime.pool
    initial_bytes = pool.nbytes
    pool.reserve_layers([2, 3], runtime.source)
    trajectory(model, runtime)
    held = weakref.ref(pool.buffers[pool.slot_for(2)]["base"])
    assert model[2]._cached_substitution_buffers is not None
    model[2]._slot_ptrs = {2: frozenset({123})}
    with pytest.raises(ValueError, match="already reserved"):
        pool.reserve_layers([4], runtime.source)
    clear_stream_substitution_cache(model)
    assert model[2]._slot_ptrs == {}
    pool.clear_resident_layers()
    pool.clear_resident_layers()
    assert held() is None
    assert pool.nbytes == initial_bytes
    assert pool.resident_slots == {}
    assert len(pool.buffers) == len(pool.owner) == len(pool.generation) == pool.n
    pool.reserve_layers([0, 5], runtime.source)
    trajectory(model, runtime)


def test_second_allocation_failure_leaves_pool_and_scheduler_usable(monkeypatch):
    import weakref

    import torch

    from tests.stream_residency_fixture import fixture, trajectory

    model, runtime = fixture()
    pool = runtime.pool
    original_buffers = tuple(pool.buffers)
    original_bytes = pool.nbytes
    original_empty = torch.empty_like
    calls = 0
    allocations = []

    def allocate(value):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise torch.OutOfMemoryError("injected allocation failure")
        allocated = original_empty(value)
        allocations.append(weakref.ref(allocated))
        return allocated

    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty_like", allocate)
        with pytest.raises(torch.OutOfMemoryError, match="injected") as failure:
            pool.reserve_layers([2, 3], runtime.source)
    assert failure.traceback is not None
    assert allocations and all(ref() is None for ref in allocations)
    assert all(first is second for first, second in zip(pool.buffers, original_buffers))
    assert len(pool.buffers) == 2
    assert pool.nbytes == original_bytes
    assert pool.owner == [None, None]
    assert pool.loads == 0
    assert pool.resident_slots == {}
    assert not hasattr(pool, "generation")
    assert runtime.source.calls == []
    trajectory(model, runtime)


def test_preload_failure_releases_source_and_rolls_back_every_slot(monkeypatch):
    from tests.stream_residency_fixture import fixture, trajectory

    model, runtime = fixture()
    pool = runtime.pool
    get = runtime.source.get
    released = []

    def load(idx, name):
        if idx == 3:
            raise OSError("injected source failure")
        return get(idx, name)

    runtime.source.release = lambda idx, event: released.append((idx, event))
    with monkeypatch.context() as patch:
        patch.setattr(runtime.source, "get", load)
        with pytest.raises(OSError, match="injected"):
            pool.reserve_layers([2, 3], runtime.source)
    assert released == [(2, None), (3, None)]
    assert len(pool.buffers) == len(pool.owner) == 2
    assert pool.owner == [None, None]
    assert pool.loads == 0
    assert pool.resident_slots == {}
    trajectory(model, runtime)


def test_active_keys_keep_mixed_dtypes_and_zero_dim_sidecars():
    import torch

    from soup_cli.utils.layer_stream_runtime import LayerBufferPool

    spec = {
        "packed": ((4, 1), "uint8"),
        "absmax": ((2,), "uint8"),
        "offset": ((), "float32"),
        "dense": ((3, 3), "float32"),
    }
    keys = [("dense",), ("packed", "absmax", "offset")]
    pool = LayerBufferPool(spec, active_keys_by_layer=keys, device="cpu")
    source = type("Source", (), {})()
    source.store = [
        {"dense": torch.eye(3)},
        {
            "packed": torch.arange(4, dtype=torch.uint8).reshape(4, 1),
            "absmax": torch.tensor([13, 17], dtype=torch.uint8),
            "offset": torch.tensor(2.5),
        },
    ]
    source.get = lambda idx, name: source.store[idx][name]
    assert pool.resident_slot_bytes(0) == 36
    assert pool.resident_slot_bytes(1) == 10
    assert pool.reserve_layers([1], source)["bytes"] == 10
    held = pool.wait(1)
    assert set(held) == set(keys[1])
    for name in held:
        assert held[name].dtype == source.store[1][name].dtype
        assert held[name].shape == source.store[1][name].shape
        assert torch.equal(held[name], source.store[1][name])
    with pytest.raises(ValueError, match="outside"):
        pool.clear_resident_layers()
        pool.reserve_layers([2], source)


def test_native_nf4_residency_preserves_real_lora_updates_and_sidecars(tmp_path):
    import torch

    pytest.importorskip("bitsandbytes")
    pytest.importorskip("peft")
    from tests.test_v07202 import _nf4_stream, _randomise_lora_b

    model, runtime, _weights, _index, _shards = _nf4_stream(tmp_path, n_layers=4)
    _randomise_lora_b(model)
    generator = torch.Generator().manual_seed(17)
    batches = [torch.randint(0, 64, (1, 8), generator=generator) for _ in range(3)]
    parameters = [param for param in model.parameters() if param.requires_grad]
    initial = [param.detach().clone() for param in parameters]

    def run():
        optimizer = torch.optim.AdamW(parameters, lr=0.001, foreach=False)
        capture = []
        for ids in batches:
            model.zero_grad(set_to_none=True)
            out = model(input_ids=ids, labels=ids)
            out.loss.backward()
            capture.extend([out.logits.detach().clone(), out.loss.detach().clone()])
            capture.extend(param.grad.detach().clone() for param in parameters)
            optimizer.step()
        capture.extend(param.detach().clone() for param in parameters)
        return capture

    expected = run()
    with torch.no_grad():
        for param, value in zip(parameters, initial):
            param.copy_(value)
    pool = runtime.pool
    state = pool.reserve_layers([0, 3], runtime.source, runtime.prefetcher.stream)
    assert state["bytes"] == sum(pool.resident_slot_bytes(idx) for idx in [0, 3])
    assert any(value.dtype == torch.uint8 for value in pool.wait(0).values())
    assert any(value.ndim == 0 for value in pool.wait(0).values())
    actual = run()
    assert len(actual) == len(expected)
    assert all(torch.equal(first, second) for first, second in zip(actual, expected))
    assert any(bool(param.grad.abs().sum() > 0) for param in parameters)
    runtime.close()


@pytest.mark.gpu
@pytest.mark.parametrize("indices", [[2, 3], [0, 1, 4, 5], list(range(6))])
def test_cuda_async_native_residency_preserves_complete_updates(indices):
    import torch

    from tests.stream_residency_fixture import fixture, trajectory

    reference, ordinary = fixture("cuda")
    candidate, runtime = fixture("cuda")
    runtime.pool.reserve_layers(indices, runtime.source, runtime.prefetcher.stream)
    expected = trajectory(reference, ordinary)
    actual = trajectory(candidate, runtime)
    torch.cuda.synchronize()
    assert all(torch.equal(first, second) for first, second in zip(actual, expected))
    assert not set(indices).intersection(runtime.source.calls[len(indices) :])
