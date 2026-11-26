"""GPU-resident layers must not occupy the RAM hotset or trigger SSD reads."""

from collections import Counter

import pytest

from tests.test_stream_ram_cache import source


def test_promoted_cache_hit_does_not_borrow_its_stale_staging_slot(tmp_path):
    import threading

    import torch

    from soup_cli.utils.async_disk_source import AsyncDiskSource
    from soup_cli.utils.safetensors_reader import identity_of

    # A completed setup probe can leave layer 0 in an idle rotating slot when
    # the independent RAM hotset subsequently promotes that same layer.
    ours = object.__new__(AsyncDiskSource)
    ours._ready = threading.Condition()
    ours._error, ours._closed, ours.pinned = None, False, True
    cached = torch.tensor([10.0])
    ours._cache = {0: {"weight": cached}}
    ours._slots = [{"weight": torch.tensor([10.0])}, {"weight": torch.tensor([20.0])}]
    ours._slot_of = {0: 0, 1: 1}
    ours._live, ours._drain = [False, False], [None, None]
    ours._resident = frozenset()
    ours._cache_borrowed, ours._cache_events = {0}, {}
    ours._cache_stats = {"tensor_hits": 0}
    path = tmp_path / "cached-layer"
    path.write_bytes(b"unchanged cached shard")
    ours._paths = {0: path}
    ours._open_buffered = lambda filename: open(filename, "rb")
    with path.open("rb") as handle:
        ours._identities = {0: identity_of(handle)}
    ours._check_the_reader_is_making_progress = lambda _idx: None
    ours._note_direction = lambda _idx: None
    ours._plan_queue = lambda _idx: []

    class PendingCopy:
        def query(self):
            return False

    event = PendingCopy()
    assert ours.get(0, "weight") is cached
    ours.release(0, event)
    # The cached copy drains from its private arena, not the old rotating slot.
    ours._slots[0]["weight"].fill_(-1)
    assert torch.equal(cached, torch.tensor([10.0]))
    assert torch.equal(ours.get(1, "weight"), torch.tensor([20.0]))
    # Cache hits must still reject a genuinely outstanding *cold* staging loan.
    with pytest.raises(RuntimeError, match="on loan"):
        ours.get(0, "weight")
    ours.release(1)
    assert ours._cache_events[0] is event


@pytest.mark.parametrize("depth", [1, 2, 4])
@pytest.mark.parametrize("ram_bytes", [0, 16384])
def test_source_skips_gpu_layers_and_preserves_rotating_pool_bytes(
    tmp_path, monkeypatch, depth, ram_bytes,
):
    import torch

    from soup_cli.utils.async_disk_source import AsyncDiskSource
    from soup_cli.utils.layer_stream_runtime import DiskSource, LayerBufferPool
    from tests.test_issue971_async_disk_source import _raw_bytes

    reads = Counter()
    original = AsyncDiskSource._range_works

    def record(self, idx, region):
        reads[idx] += 1
        return original(self, idx, region)

    monkeypatch.setattr(AsyncDiskSource, "_range_works", record)
    folder, specs, ours = source(tmp_path, monkeypatch, depth=depth, budget=ram_bytes)
    shipped = DiskSource(folder, 8, specs)
    pool = LayerBufferPool(specs[0], n_buffers=2, device="cpu")
    try:
        pool.reserve_layers([3], ours)
        ours.set_resident_layers([3])
        stats = ours.ram_cache_stats()
        assert stats["resident_indices"] == [3]
        assert 3 not in stats["indices"]
        if ram_bytes:
            assert len(stats["indices"]) == 2
            assert stats["allocated_bytes"] <= ram_bytes
        walk = list(range(8)) + list(reversed(range(8)))
        for _ in range(2):
            for idx in walk:
                pool.load_async(idx, ours)
                pool.wait(idx)
        before = reads.copy()
        for _ in range(3):
            for idx in walk:
                pool.load_async(idx, ours)
                for name, actual in pool.wait(idx).items():
                    assert torch.equal(_raw_bytes(actual), _raw_bytes(shipped.get(idx, name)))
        for idx in stats["indices"] + [3]:
            assert reads[idx] == before[idx], "hot layers must not be read speculatively"
        with pytest.raises(RuntimeError, match="resident"):
            ours.get(3, next(iter(specs[3])))
        ours.set_resident_layers([])
        pool.clear_resident_layers()
        for idx in walk:
            pool.load_async(idx, ours)
            pool.wait(idx)
    finally:
        ours.close()
        shipped.close()


def test_ram_cache_can_be_enabled_after_residency_without_duplicate_allocation(
    tmp_path, monkeypatch,
):
    from soup_cli.utils import async_disk_source

    _folder, _specs, ours = source(tmp_path, monkeypatch, budget=0)
    original = ours._allocate_staging
    allocations = []

    def allocate(sizes):
        assert not ours._cache_arenas
        allocations.append(list(sizes))
        return original(sizes)

    monkeypatch.setattr(ours, "_allocate_staging", allocate)
    try:
        ours.set_resident_layers([3])
        ours.configure_ram_cache(16384)
        assert ours.ram_cache_stats()["indices"] == [2, 4]
        assert len(allocations) == 1
        ours.set_resident_layers([2, 3])
        assert not ({2, 3} & set(ours.ram_cache_stats()["indices"]))
        assert len(allocations) == 2
        assert async_disk_source.AsyncDiskSource is type(ours)
    finally:
        ours.close()


def test_cache_replanning_drains_events_before_releasing_memory(tmp_path, monkeypatch):
    _folder, specs, ours = source(tmp_path, monkeypatch)
    idx = ours.ram_cache_stats()["indices"][0]
    called = []

    class Event:
        def synchronize(self):
            assert ours._cache_arenas
            called.append(True)

    try:
        ours.get(idx, next(iter(specs[idx])))
        with pytest.raises(RuntimeError, match="loan|borrow"):
            ours.set_resident_layers([idx])
        ours.release(idx, Event())
        ours.set_resident_layers([idx])
        assert called == [True]
        assert idx not in ours.ram_cache_stats()["indices"]
    finally:
        ours.close()


@pytest.mark.parametrize("indices", [[True], [-1], [8], [1, 1], [3, 2], [1.5]])
def test_invalid_resident_indices_do_not_change_live_cache(tmp_path, monkeypatch, indices):
    _folder, _specs, ours = source(tmp_path, monkeypatch)
    try:
        before = ours.ram_cache_stats()
        with pytest.raises(ValueError, match="indices"):
            ours.set_resident_layers(indices)
        assert ours.ram_cache_stats() == before
    finally:
        ours.close()


@pytest.mark.parametrize("refuse", [False, True])
def test_disk_planner_publishes_exclusions_and_restores_them_on_fallback(refuse):
    from tests.test_stream_residency_planner import configure, fixture

    model, runtime, memory, _calls, good = fixture()
    runtime.tier = "disk"
    exclusions = []

    class Source:
        def set_resident_layers(self, indices):
            exclusions.append(tuple(indices))

    runtime.source = Source()

    def measure(*args, **kwargs):
        assert (
            not runtime.pool.resident_slots or exclusions[-1] == tuple(runtime.pool.resident_slots)
        )
        peak = good(*args, **kwargs)
        if refuse and runtime.pool.resident_slots:
            from dataclasses import replace

            peak = replace(peak, reserved_bytes=9999)
        return peak

    plan = configure(model, runtime, memory, measure)
    assert exclusions[0] == (1, 2, 3)
    if refuse:
        assert not plan.enabled
        assert exclusions[-1] == ()
        assert runtime.pool.resident_slots == {}
    else:
        assert plan.indices == (1, 2, 3)


@pytest.mark.gpu
def test_pinned_joint_cache_uses_real_cuda_pool_and_retains_exact_bytes(tmp_path, monkeypatch):
    import torch

    from soup_cli.utils.async_disk_source import AsyncDiskSource
    from soup_cli.utils.layer_stream_runtime import DiskSource, LayerBufferPool, RamSource
    from tests.test_issue971_async_disk_source import _raw_bytes

    folder, specs, seed_source = source(tmp_path, monkeypatch, budget=0)
    seed_source.close()
    ours = AsyncDiskSource(folder, 8, specs, pin=True, ram_cache_bytes=0)
    reference = DiskSource(folder, 8, specs)
    pool = LayerBufferPool(specs[0], device="cuda")
    stream = torch.cuda.Stream()
    retained = []
    try:
        pool.reserve_layers([3], ours, stream)
        ours.set_resident_layers([3])
        ours.configure_ram_cache(16384)
        assert set(ours.ram_cache_stats()["indices"]).isdisjoint(pool.resident_slots)
        for idx in (list(range(8)) + list(reversed(range(8)))) * 3:
            pool.load_async(idx, ours, stream)
            retained.append((idx, {key: value.clone() for key, value in pool.wait(idx).items()}))
        ours.close()
        torch.cuda.synchronize()
        for idx, values in retained:
            for name, actual in values.items():
                assert torch.equal(_raw_bytes(actual.cpu()), _raw_bytes(reference.get(idx, name)))
        assert RamSource.layer_specs_from_shards(folder, 8) == specs
    finally:
        ours.close()
        reference.close()


def _promotion_fixture(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import save_file

    from soup_cli.utils.layer_shard import layer_shard_path
    from soup_cli.utils.layer_stream_runtime import RamSource

    folder, _specs, seed_source = source(tmp_path, monkeypatch, budget=0)
    seed_source.close()
    # A distinct singleton group stays in ordinary staging, like the real
    # untied vocabulary weight; it must not disappear into the decoder hotset.
    save_file({"lm_head.weight": torch.rand(64, 16)}, layer_shard_path(folder, 7))
    return folder, RamSource.layer_specs_from_shards(folder, 8)


def _promotion_budget(ours):
    from soup_cli.utils.layer_stream_runtime import plan_pinned_arenas
    from soup_cli.utils.safetensors_reader import SECTOR_BYTES

    return plan_pinned_arenas(
        [ours._plans[idx].span for idx in (0, 5, 6)], align=SECTOR_BYTES
    ).pinned_bytes


def test_promotion_fixture_admits_a_nonempty_native_pinned_hotset(tmp_path, monkeypatch):
    from soup_cli.utils.async_disk_source import AsyncDiskSource
    from soup_cli.utils.stream_ram_cache import plan_hotset

    folder, specs = _promotion_fixture(tmp_path, monkeypatch)
    ours = AsyncDiskSource(folder, 8, specs, pin=False, ram_cache_bytes=0)
    try:
        plan = plan_hotset(
            [item.span for item in ours._plans], ours._group_of, ours.read_ahead,
            _promotion_budget(ours), pinned=True, excluded=[1, 2, 3, 4],
        )
        assert plan["indices"] == [0, 5, 6]
        assert plan["allocated_bytes"] == _promotion_budget(ours) > 0
        assert ours._group_of[7] != ours._group_of[0]
    finally:
        ours.close()


@pytest.mark.gpu
def test_promoted_probe_slot_does_not_block_large_layer_dma(tmp_path, monkeypatch):
    import torch

    from soup_cli.utils.async_disk_source import AsyncDiskSource
    from soup_cli.utils.layer_stream_runtime import (
        DiskSource,
        LargeLayerBufferPool,
        LayerBufferPool,
    )
    from tests.test_issue971_async_disk_source import _raw_bytes

    folder, specs = _promotion_fixture(tmp_path, monkeypatch)
    ours = AsyncDiskSource(folder, 8, specs, pin=True, ram_cache_bytes=0)
    # Demand reads still use the real reader. Suppress speculation only to keep
    # the completed setup probe's stale layer-0 mapping deterministic.
    monkeypatch.setattr(ours, "_plan_queue", lambda _idx: [])
    reference = DiskSource(folder, 8, specs)
    pool = LayerBufferPool(specs[0], device="cuda")
    stream = torch.cuda.Stream()
    try:
        pool.reserve_layers([1, 2, 3, 4], ours, stream)
        ours.set_resident_layers([1, 2, 3, 4])
        pool.load_async(0, ours, stream)
        pool.wait(0)
        ours.configure_ram_cache(_promotion_budget(ours))
        assert ours.ram_cache_stats()["indices"] == [0, 5, 6]
        assert 0 in ours._slot_of, "the regression requires the probe's idle staging mapping"
        pool.load_async(0, ours, stream)
        cached = {name: value.clone() for name, value in pool.wait(0).items()}

        name = next(iter(specs[7]))
        large = LargeLayerBufferPool({name: specs[7][name]}, {name: 7}, device="cuda")
        large.load_async(name, ours, stream)
        cold = large.wait(name).clone()
        ours.close()
        torch.cuda.synchronize()
        for key, value in cached.items():
            assert torch.equal(_raw_bytes(value.cpu()), _raw_bytes(reference.get(0, key)))
        assert torch.equal(_raw_bytes(cold.cpu()), _raw_bytes(reference.get(7, name)))
    finally:
        ours.close()
        reference.close()


@pytest.mark.parametrize("fail", [False, True])
def test_real_trainer_allocates_ram_only_after_gpu_planning(tmp_path, monkeypatch, fail):
    from types import SimpleNamespace

    import soup_cli.utils.stream_residency as residency
    from tests.test_issue623_stream_pin_allocation import _drive_wiring

    seen = []

    def configure(_model, runtime, *_args, **_kwargs):
        seen.append(runtime)
        runtime.source.configure_ram_cache.assert_not_called()
        if fail:
            runtime.source.configure_ram_cache.side_effect = OSError("cache setup sentinel")
        return SimpleNamespace(indices=(1,))

    monkeypatch.setattr(residency, "configure_stream_residency", configure)

    def run():
        return _drive_wiring(
            tmp_path, monkeypatch, stream_pin_yaml="", stream_source="disk",
            extra_training_yaml="  stream_ram_cache_mb: 1024\n  stream_resident_layers: auto\n",
        )

    if fail:
        with pytest.raises(OSError, match="sentinel"):
            run()
        seen[0].close.assert_called_once()
    else:
        kwargs, _text = run()
        assert "ram_cache_bytes" not in kwargs
        seen[0].source.configure_ram_cache.assert_called_once_with(1024 * 2**20)
