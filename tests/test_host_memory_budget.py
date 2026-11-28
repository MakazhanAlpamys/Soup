"""HostMemoryBudget: atomic leases that two consumers share without overbooking."""

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
