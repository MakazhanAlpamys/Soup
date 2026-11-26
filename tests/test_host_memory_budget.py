"""Host-memory budget: atomic leases and the weights source that draws on one."""

import pytest

from soup_cli.utils.host_memory_budget import HostMemoryBudget


def test_reservations_are_exclusive_and_released():
    budget = HostMemoryBudget(100, reserved_bytes=20)
    first = budget.reserve("weights", 50)
    with pytest.raises(MemoryError):
        budget.reserve("activations", 31)
    with pytest.raises(ValueError):
        budget.reserve("weights", 1)
    assert budget.used_bytes == 70
    first.resize(30)
    second = budget.reserve("activations", 50)
    assert budget.used_bytes == 100
    first.close()
    first.close()
    assert budget.used_bytes == 70
    second.close()
    assert budget.used_bytes == 20
    with pytest.raises(RuntimeError):
        first.resize(1)


def test_failed_growth_does_not_change_lease():
    budget = HostMemoryBudget(100)
    lease = budget.reserve("weights", 50)
    with pytest.raises(MemoryError):
        lease.resize(101)
    assert budget.used_bytes == 50
    lease.close()


@pytest.mark.parametrize("bad", [True, -1, 1.5])
def test_invalid_byte_counts_fail(bad):
    with pytest.raises(ValueError):
        HostMemoryBudget(bad)


def test_two_concurrent_consumers_cannot_overbook():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    budget = HostMemoryBudget(100)
    ready = Barrier(2)

    def claim(name):
        ready.wait()
        try:
            return budget.reserve(name, 70)
        except MemoryError:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        leases = list(executor.map(claim, ["weights", "activations"]))
    assert sum(lease is not None for lease in leases) == 1
    assert budget.used_bytes == 70
    for lease in leases:
        if lease:
            lease.close()
    assert budget.used_bytes == 0


def test_weights_source_leases_its_share_and_releases_it_on_close(tmp_path):
    from soup_cli.utils.partial_ram_source import PartialRamBudget, PartialRamSource
    from tests.test_stream_partial_ram import shards

    paths, specs = shards(tmp_path)
    budget = HostMemoryBudget(1_000_000, reserved_bytes=200)
    source = PartialRamSource(
        str(tmp_path),
        5,
        specs,
        pin=False,
        shard_paths=paths,
        budget=PartialRamBudget(1_000_000, 200, 136),
        host_budget=budget,
    )
    try:
        assert 200 < budget.used_bytes < 1_000_000
        assert budget.used_bytes == 200 + source.host_allocated_bytes
    finally:
        source.close()
    assert budget.used_bytes == 200


def test_weights_source_refuses_a_budget_that_does_not_match_its_envelope(tmp_path):
    from soup_cli.utils.partial_ram_source import PartialRamBudget, PartialRamSource
    from tests.test_stream_partial_ram import shards

    paths, specs = shards(tmp_path)
    with pytest.raises(ValueError, match="shared host budget must match"):
        PartialRamSource(
            str(tmp_path),
            5,
            specs,
            pin=False,
            shard_paths=paths,
            budget=PartialRamBudget(1_000_000, 200, 136),
            host_budget=HostMemoryBudget(900_000, reserved_bytes=200),
        )


def test_a_failed_second_reservation_leaves_the_first_charged():
    budget = HostMemoryBudget(100)
    lease = budget.reserve("stream.weights", 90)
    with pytest.raises(MemoryError):
        budget.reserve("second.consumer", 20)
    assert budget.used_bytes == 90
    lease.close()
    assert budget.used_bytes == 0
