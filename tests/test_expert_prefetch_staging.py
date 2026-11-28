"""Bounded expert reads through the Python pool and its runtime installer."""

import threading
from types import SimpleNamespace

import pytest

from soup_cli.utils.layer_stream_runtime import _install_expert_streaming


@pytest.mark.parametrize(
    ("options", "layout", "message"),
    [
        (
            {"expert_pool": True, "expert_prefetch_staging": 2},
            "per_expert",
            "expert_prefetch_staging=2 needs expert_prefetch=True",
        ),
        (
            {"expert_prefetch": True, "expert_prefetch_staging": 2},
            "per_expert",
            "expert_prefetch_staging=2 needs expert_pool=True",
        ),
        (
            {"expert_pool": True, "expert_prefetch": True, "expert_prefetch_staging": 2},
            "fused",
            "expert_prefetch_staging=2 needs a cache sharded with expert_layout='per_expert'",
        ),
        (
            {
                "expert_pool": True,
                "expert_prefetch": True,
                "expert_prefetch_staging": 2,
                "stream_experts": object(),
            },
            "per_expert",
            "expert_prefetch_staging=2 cannot be combined with training.stream_experts",
        ),
    ],
)
def test_staging_requires_the_python_single_expert_pool(options, layout, message):
    with pytest.raises(ValueError) as error:
        _install_expert_streaming(
            [], "unused", SimpleNamespace(expert_layout=layout), device="cpu", codes=None,
            **options,
        )
    assert str(error.value) == message


@pytest.fixture
def staged_pool(tmp_path):
    from tests.test_expert_buffer_pool import _experts, _store

    experts = _experts()
    reader, pool = _store(tmp_path, experts, prefetch=True, prefetch_staging=2)
    try:
        yield reader, pool, experts
    finally:
        pool.close()
        reader.close()


@pytest.mark.parametrize("bad", [False, True, 0, 3, "2", 2.0, None])
def test_prefetch_staging_requires_an_integer_one_or_two(tmp_path, bad):
    from soup_cli.utils.expert_stream import ExpertBufferPool
    from tests.test_expert_buffer_pool import _experts, _store

    reader, _, _ = _store(tmp_path, _experts(), with_pool=False)
    try:
        with pytest.raises(ValueError) as error:
            ExpertBufferPool(
                reader, device="cpu", dtype="float32", weight_budget_bytes=1,
                prefetch=True, prefetch_staging=bad,
            )
        assert str(error.value) == "prefetch_staging must be 1 or 2"
    finally:
        reader.close()


@pytest.mark.parametrize("bad", [False, True, 0, 3, "2", 1.0, 2.0, None])
def test_installer_rejects_invalid_staging_before_opening_shards(bad):
    with pytest.raises(ValueError) as error:
        _install_expert_streaming(
            [], "unused", SimpleNamespace(expert_layout="per_expert"), device="cpu", codes=None,
            expert_pool=True, expert_prefetch=True, expert_prefetch_staging=bad,
        )
    assert str(error.value) == "expert_prefetch_staging must be 1 or 2"


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (
            {"prefetch": False},
            "prefetch_staging=2 needs prefetch=True",
        ),
        (
            {"prefetch": True, "group_width": 2},
            "prefetch_staging=2 cannot be combined with group_width other than 1",
        ),
        (
            {"prefetch": True, "packed_only": True},
            "prefetch_staging=2 cannot be combined with packed_only=True",
        ),
    ],
)
def test_pool_refuses_inapplicable_staging_before_allocation(tmp_path, options, message):
    from soup_cli.utils.expert_stream import ExpertBufferPool
    from tests.test_expert_buffer_pool import _experts, _store

    reader, _, _ = _store(tmp_path, _experts(), with_pool=False)
    try:
        with pytest.raises(ValueError) as error:
            ExpertBufferPool(
                reader, device="cpu", dtype="float32", weight_budget_bytes=1,
                prefetch_staging=2, **options,
            )
        assert str(error.value) == message
    finally:
        reader.close()


def test_two_slots_keep_both_hints_on_a_miss_and_deduplicate(staged_pool):
    from soup_cli.utils.safetensors_reader import SECTOR_BYTES

    reader, pool, experts = staged_pool
    pool.prefetch(reader, 1)
    pool.prefetch(reader, 2)
    pool.prefetch(reader, 1)
    slot_bytes = reader.max_span + SECTOR_BYTES
    assert pool.stats()["prefetch"]["staging"] == 2
    assert pool.stats()["prefetch"]["host_staging_bytes"] == 2 * slot_bytes
    assert pool.plan.host_staging_bytes == 3 * slot_bytes
    for expert in (0, 1, 2):
        with pool.lease(reader, expert) as pair:
            assert all(actual.equal(expected) for actual, expected in zip(pair, experts[expert]))
    assert pool.prefetch_hits == 2
    assert pool.prefetch_discards == 0
    assert reader.fetches == pool.fetches == 3
    assert reader.bytes_read == pool.logical_bytes_read


def test_third_hint_reuses_oldest_slot_without_altering_the_active_lease(staged_pool):
    reader, pool, experts = staged_pool
    pool.prefetch(reader, 1)
    pool.prefetch(reader, 2)
    with pool.lease(reader, 1) as pair:
        pool.prefetch(reader, 3)
        pool.prefetch(reader, 4)
        assert all(actual.equal(expected) for actual, expected in zip(pair, experts[1]))
    assert pool.prefetch_discards == 1
    for expert in (3, 4, 2):
        with pool.lease(reader, expert) as pair:
            assert all(actual.equal(expected) for actual, expected in zip(pair, experts[expert]))
    assert pool.prefetch_hits == 3
    assert pool.fetches == 4
    assert reader.fetches == 5


def test_close_drains_both_slots_on_one_worker(staged_pool, monkeypatch):
    reader, pool, _ = staged_pool
    original = reader.read_host
    calls = []
    started = threading.Event()
    release = threading.Event()

    def read(expert, staging):
        if expert:
            calls.append((expert, threading.current_thread(), staging.data_ptr()))
        if expert == 1:
            started.set()
            assert release.wait(timeout=5)
        return original(expert, staging)

    monkeypatch.setattr(reader, "read_host", read)
    try:
        with pool.lease(reader, 0):
            pool.prefetch(reader, 1)
            assert started.wait(timeout=5), "prefetch did not start during the current lease"
            pool.prefetch(reader, 2)
            assert [expert for expert, _, _ in calls] == [1]
    finally:
        release.set()
    pool.close()
    pool.prefetch(reader, 3)
    assert [expert for expert, _, _ in calls] == [1, 2]
    assert reader.fetches == 3
    assert calls[0][1] is calls[1][1]
    assert calls[0][1] is not threading.current_thread()
    assert not calls[0][1].is_alive()
    assert calls[0][2] != calls[1][2]
    assert pool.stats()["prefetch"]["host_staging_bytes"] == 0


def test_failed_hint_is_reported_on_its_lease_and_the_pool_recovers(staged_pool, monkeypatch):
    reader, pool, experts = staged_pool
    original = reader.read_host

    def read(expert, staging):
        if expert == 1 and threading.current_thread() is not threading.main_thread():
            raise OSError("expert read failed")
        return original(expert, staging)

    monkeypatch.setattr(reader, "read_host", read)
    pool.prefetch(reader, 1)
    pool.prefetch(reader, 2)
    with pool.lease(reader, 0):
        pass
    with pytest.raises(OSError) as error, pool.lease(reader, 1):
        pass
    assert str(error.value) == "expert read failed"
    with pool.lease(reader, 1) as pair:
        assert all(actual.equal(expected) for actual, expected in zip(pair, experts[1]))
    assert pool.prefetch_discards == 1


@pytest.mark.gpu
@pytest.mark.parametrize("async_lease", [False, True])
def test_cuda_two_slots_reuse_packed_staging_safely(tmp_path, async_lease, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("bitsandbytes")
    from soup_cli.utils.safetensors_reader import SECTOR_BYTES
    from tests.test_expert_buffer_pool import _experts, _store

    experts = _experts()
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    reader, reference = _store(reference_dir, experts, nf4=True, device="cuda")
    try:
        expected = []
        for expert in range(4):
            with reference.lease(reader, expert) as pair:
                expected.append(tuple(weight.clone() for weight in pair))
    finally:
        reference.close()
        reader.close()

    reader, pool = _store(
        tmp_path, experts, nf4=True, device="cuda", transfer="packed", prefetch=True,
        prefetch_staging=2,
    )
    pool.async_lease = async_lease
    try:
        slot_bytes = reader.max_span + SECTOR_BYTES
        assert pool._pinned.is_pinned()
        assert pool.plan.host_pinned_bytes == pool._pinned.numel()
        assert pool.stats()["prefetch"]["host_staging_bytes"] == 2 * slot_bytes
        assert pool.plan.host_staging_bytes == 3 * slot_bytes
        original = reader.read_host
        worker_buffers = []

        def read(expert, staging):
            if threading.current_thread() is not threading.main_thread():
                worker_buffers.append((staging.data_ptr(), staging.is_pinned()))
            return original(expert, staging)

        monkeypatch.setattr(reader, "read_host", read)
        got = []
        for expert in range(4):
            with pool.lease(reader, expert) as pair:
                got.append(tuple(weight.clone() for weight in pair))
                for hint in range(expert + 1, min(4, expert + 3)):
                    pool.prefetch(reader, hint)
            assert (pool._h2d_event is not None) == async_lease
        torch.cuda.synchronize(pool.device)
        assert len(worker_buffers) == pool.prefetch_hits == 3
        assert len(set(worker_buffers)) == 2 and not any(pinned for _, pinned in worker_buffers)
        assert pool.h2d_copies == pool.fetches == 4
        assert reader.bytes_read == pool.logical_bytes_read
        for actual, wanted in zip(got, expected):
            assert all(a.equal(b) for a, b in zip(actual, wanted))
    finally:
        pool.close()
        reader.close()
