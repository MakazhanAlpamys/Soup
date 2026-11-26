"""The SSD hotset must preserve bytes, reader liveness, and DMA ownership."""

import os
from collections import Counter

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from soup_cli.utils.async_disk_source import AsyncDiskSource  # noqa: E402
from soup_cli.utils.layer_stream_runtime import DiskSource, RamSource  # noqa: E402
from tests.test_issue971_async_disk_source import _raw_bytes, _shards  # noqa: E402


def source(tmp_path, monkeypatch, *, depth=2, budget=16384):
    from soup_cli.utils import stream_ram_cache

    monkeypatch.setattr(
        stream_ram_cache,
        "available_memory",
        lambda: {
            "available_bytes": 8 * 2**30,
            "constraints": [],
            "host_available_bytes": 8 * 2**30,
        },
    )
    folder = _shards(tmp_path, n_layers=8)
    specs = RamSource.layer_specs_from_shards(folder, 8)
    ours = AsyncDiskSource(folder, 8, specs, pin=False, read_ahead=depth, ram_cache_bytes=budget)
    return folder, specs, ours


@pytest.mark.parametrize("depth", [1, 2, 4])
def test_cached_mixed_dtypes_scalars_match_disk_in_both_directions(tmp_path, monkeypatch, depth):
    folder, specs, ours = source(tmp_path, monkeypatch, depth=depth)
    shipped = DiskSource(folder, 8, specs)
    try:
        assert len(ours.ram_cache_stats()["indices"]) == 2
        for idx in list(range(8)) + list(reversed(range(8))) + list(range(8)):
            for name in specs[idx]:
                actual, expected = ours.get(idx, name), shipped.get(idx, name)
                assert (actual.dtype, actual.shape) == (expected.dtype, expected.shape)
                assert torch.equal(_raw_bytes(actual), _raw_bytes(expected))
            ours.release(idx)
        stats = ours.ram_cache_stats()
        assert stats["allocated_bytes"] <= 16384
        assert stats["payload_bytes"] > 0 and stats["tensor_hits"] > 0
    finally:
        ours.close()
        shipped.close()


def test_cached_layers_are_never_requeued_for_disk_reads(tmp_path, monkeypatch):
    counts = Counter()
    original = AsyncDiskSource._range_works

    def observed(self, idx, region):
        counts[idx] += 1
        return original(self, idx, region)

    monkeypatch.setattr(AsyncDiskSource, "_range_works", observed)
    _folder, specs, ours = source(tmp_path, monkeypatch)
    cached = ours.ram_cache_stats()["indices"]
    try:
        for _ in range(3):
            for idx in list(range(8)) + list(reversed(range(8))):
                for name in specs[idx]:
                    ours.get(idx, name)
                ours.release(idx)
        assert {idx: counts[idx] for idx in cached} == {idx: 1 for idx in cached}
        assert counts[0] > 1
    finally:
        ours.close()


def test_default_source_does_not_probe_or_allocate_cache(tmp_path, monkeypatch):
    from soup_cli.utils import stream_ram_cache

    def fail():
        pytest.fail("cache disabled must not probe memory")

    monkeypatch.setattr(stream_ram_cache, "available_memory", fail)
    folder = _shards(tmp_path)
    ours = AsyncDiskSource(folder, 4, RamSource.layer_specs_from_shards(folder, 4), pin=False)
    try:
        assert ours.ram_cache_stats()["allocated_bytes"] == 0
    finally:
        ours.close()


@pytest.mark.parametrize("budget", [True, False, -1, 1.5, "1", None])
def test_noninteger_or_negative_budget_refuses_before_io(tmp_path, budget):
    with pytest.raises(ValueError, match="ram_cache_bytes"):
        AsyncDiskSource(str(tmp_path), 8, {}, pin=False, ram_cache_bytes=budget)


def test_budget_smaller_than_one_real_allocation_uses_no_hotset(tmp_path, monkeypatch):
    _folder, _specs, ours = source(tmp_path, monkeypatch, budget=4095)
    try:
        assert ours.ram_cache_stats()["indices"] == []
        assert ours.ram_cache_stats()["allocated_bytes"] == 0
    finally:
        ours.close()


def test_cached_hit_still_refuses_changed_shard(tmp_path, monkeypatch):
    _folder, specs, ours = source(tmp_path, monkeypatch)
    idx = ours.ram_cache_stats()["indices"][0]
    try:
        path = ours._paths[idx]
        before = os.stat(path)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1000000000))
        with pytest.raises(RuntimeError, match="changed"):
            ours.get(idx, next(iter(specs[idx])))
    finally:
        ours.close()


def test_cached_hit_does_not_hide_reader_failure_or_closed_state(tmp_path, monkeypatch):
    _folder, specs, ours = source(tmp_path, monkeypatch)
    idx = ours.ram_cache_stats()["indices"][0]
    name = next(iter(specs[idx]))
    ours._fail(ValueError("reader failure sentinel"))
    with pytest.raises(ValueError, match="sentinel"):
        ours.get(idx, name)
    ours.close()
    ours.close()
    with pytest.raises((ValueError, RuntimeError), match="sentinel|closed"):
        ours.get(idx, name)


def test_cache_close_waits_for_copy_before_freeing_arenas(tmp_path, monkeypatch):
    _folder, specs, ours = source(tmp_path, monkeypatch)
    idx = ours.ram_cache_stats()["indices"][0]
    called = []

    class Event:
        def synchronize(self):
            assert ours._cache_arenas
            called.append(True)

        def query(self):
            return False

    ours.get(idx, next(iter(specs[idx])))
    ours.release(idx, Event())
    ours.close()
    assert called == [True]
    assert not ours._cache_arenas


def test_failed_preload_closes_reader_and_range_workers(tmp_path, monkeypatch):
    from soup_cli.utils import stream_ram_cache

    monkeypatch.setattr(
        stream_ram_cache,
        "available_memory",
        lambda: {
            "available_bytes": 8 * 2**30,
            "constraints": [],
            "host_available_bytes": 8 * 2**30,
        },
    )
    folder = _shards(tmp_path, n_layers=8)
    specs = RamSource.layer_specs_from_shards(folder, 8)
    original = AsyncDiskSource._range_works
    seen = []

    def broken(self, idx, region):
        seen.append(self)
        if idx == 3:
            raise OSError("preload read sentinel")
        return original(self, idx, region)

    monkeypatch.setattr(AsyncDiskSource, "_range_works", broken)
    with pytest.raises(OSError, match="sentinel"):
        AsyncDiskSource(folder, 8, specs, pin=False, ram_cache_bytes=16384)
    assert seen and seen[-1]._closed
    assert not seen[-1]._thread.is_alive()
    assert all(pool._closed for pool in seen[-1]._pools.values())


@pytest.mark.gpu
@pytest.mark.parametrize("depth", [1, 2])
def test_pinned_cache_through_real_cuda_rotating_pool(tmp_path, monkeypatch, depth):
    from soup_cli.utils import stream_ram_cache
    from soup_cli.utils.layer_stream_runtime import LayerBufferPool

    monkeypatch.setattr(
        stream_ram_cache,
        "available_memory",
        lambda: {
            "available_bytes": 8 * 2**30,
            "constraints": [],
            "host_available_bytes": 8 * 2**30,
        },
    )
    folder = _shards(tmp_path, n_layers=8)
    specs = RamSource.layer_specs_from_shards(folder, 8)
    ours = AsyncDiskSource(folder, 8, specs, read_ahead=depth, pin=True, ram_cache_bytes=16384)
    shipped = DiskSource(folder, 8, specs)
    pool = LayerBufferPool(specs[0], n_buffers=2, device="cuda")
    stream = torch.cuda.Stream()
    retained = []
    try:
        for idx in (list(range(8)) + list(reversed(range(8)))) * 3:
            pool.load_async(idx, ours, stream)
            actual = pool.wait(idx)
            retained.append((idx, {name: value.clone() for name, value in actual.items()}))
        # Close while the last copy/consumer can still be pending: cache memory
        # must survive its source release events. No per-layer global synchronize.
        ours.close()
        torch.cuda.synchronize()
        for idx, values in retained:
            for name, actual in values.items():
                expected = shipped.get(idx, name)
                assert torch.equal(_raw_bytes(actual.cpu()), _raw_bytes(expected))
    finally:
        ours.close()
        shipped.close()
