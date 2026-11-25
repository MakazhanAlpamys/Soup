"""cgroup_memory_bounds: a container must not plan a host-sized resident store.

The bounds are combined with psutil the way ``layer_stream.total_ram_bytes`` and
``free_ram_bytes`` will combine them once that wiring lands; ``_total`` / ``_free`` below do the
same min() so the tests pin the numbers a planner would see.
"""
from types import SimpleNamespace

import psutil
import pytest

from soup_cli.utils.container_memory import cgroup_memory_bounds

GIB = 1024**3


def _total():
    limit, _ = cgroup_memory_bounds()
    host = int(psutil.virtual_memory().total)
    return host if limit is None else min(host, limit)


def _free():
    _, available = cgroup_memory_bounds()
    host = int(psutil.virtual_memory().available)
    return host if available is None else min(host, available)


@pytest.fixture
def container(monkeypatch):
    from pathlib import Path

    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(
        total=128 * GIB, available=100 * GIB))
    files = {
        "/proc/self/cgroup": "0::/\n",
        "/sys/fs/cgroup/memory.max": str(32 * GIB),
        "/sys/fs/cgroup/memory.current": str(8 * GIB),
        "/sys/fs/cgroup/memory.stat": f"inactive_file {3 * GIB}\n",
    }

    def read(path, *args, **kwargs):
        normalized = path.as_posix()
        for suffix, content in files.items():
            if normalized.endswith(suffix):
                return content
        raise FileNotFoundError(path)

    monkeypatch.setattr(Path, "read_text", read)
    return files


def test_effective_total_is_the_real_container_limit(container):
    assert _total() == 32 * GIB


def test_available_reclaims_only_inactive_file_cache(container):
    assert _free() == 27 * GIB



def test_host_available_remains_the_upper_bound(container, monkeypatch):
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(
        total=128 * GIB, available=5 * GIB))
    assert _free() == 5 * GIB


def test_unlimited_container_preserves_host_measurement(container):
    container["/sys/fs/cgroup/memory.max"] = "max"
    assert _total() == 128 * GIB
    assert _free() == 100 * GIB


def test_absent_controller_preserves_host_measurement(container):
    container.clear()
    assert _total() == 128 * GIB
    assert _free() == 100 * GIB


def test_visible_parent_limit_also_constrains_the_child(container):
    container["/proc/self/cgroup"] = "0::/worker/job\n"
    for group, limit, usage, inactive in (
        ("worker", 24, 10, 3), ("worker/job", 32, 8, 3),
    ):
        prefix = "/sys/fs/cgroup/" + group
        container[prefix + "/memory.max"] = str(limit * GIB)
        container[prefix + "/memory.current"] = str(usage * GIB)
        container[prefix + "/memory.stat"] = f"inactive_file {inactive * GIB}\n"
    assert _total() == 24 * GIB
    assert _free() == 17 * GIB


def test_v1_accounts_for_descendant_file_cache(container):
    container.clear()
    prefix = "/sys/fs/cgroup/memory"
    container.update({
        "/proc/self/cgroup": "7:cpu:/ignored\n9:memory:/\n",
        prefix + "/memory.limit_in_bytes": str(32 * GIB),
        prefix + "/memory.usage_in_bytes": str(8 * GIB),
        prefix + "/memory.stat": f"inactive_file 0\ntotal_inactive_file {3 * GIB}\n",
    })
    assert _total() == 32 * GIB
    assert _free() == 27 * GIB


def test_unlimited_v1_does_not_require_usage(container):
    container.clear()
    container["/sys/fs/cgroup/memory/memory.limit_in_bytes"] = str(2**63 - 4096)
    assert _total() == 128 * GIB
    assert _free() == 100 * GIB


def test_known_limit_without_usage_has_no_known_free_headroom(container):
    del container["/sys/fs/cgroup/memory.current"]
    assert _total() == 32 * GIB
    assert _free() == 0


@pytest.mark.parametrize("stat", ["", "broken", "inactive_file -100\n", "active_file 9000\n"])
def test_unknown_or_active_cache_is_not_reclaimed(container, stat):
    container["/sys/fs/cgroup/memory.stat"] = stat
    assert _free() == 24 * GIB


def test_bad_usage_cannot_produce_negative_or_excess_headroom(container):
    container["/sys/fs/cgroup/memory.current"] = str(40 * GIB)
    container["/sys/fs/cgroup/memory.stat"] = ""
    assert _free() == 0
    container["/sys/fs/cgroup/memory.stat"] = f"inactive_file {100 * GIB}\n"
    assert _free() == 32 * GIB


def test_membership_cannot_escape_the_controller_root(container):
    container["/proc/self/cgroup"] = "0::/../../outside\n"
    container["/outside/memory.max"] = str(GIB)
    assert _total() == 32 * GIB


@pytest.mark.parametrize("protection", ["memory.low", "memory.min"])
def test_protected_allocation_requires_explicit_provider_opt_in(container, protection):
    container["/sys/fs/cgroup/memory.max"] = str(56 * GIB)
    container[f"/sys/fs/cgroup/{protection}"] = str(32 * GIB)
    assert _total() == 56 * GIB
    assert _free() == 51 * GIB
    assert cgroup_memory_bounds(use_protection_budget=True) == (32 * GIB, 27 * GIB)


def test_reservation_is_still_bounded_by_a_smaller_hard_limit(container):
    container["/sys/fs/cgroup/memory.max"] = str(24 * GIB)
    container["/sys/fs/cgroup/memory.low"] = str(32 * GIB)
    assert cgroup_memory_bounds(use_protection_budget=True) == (24 * GIB, 19 * GIB)


def test_zero_reservation_adds_no_artificial_constraint(container):
    container["/sys/fs/cgroup/memory.low"] = "0"
    assert _total() == 32 * GIB
    assert _free() == 27 * GIB


def test_v1_soft_allocation_is_a_conservative_budget(container):
    container.clear()
    prefix = "/sys/fs/cgroup/memory"
    container[prefix + "/memory.limit_in_bytes"] = str(56 * GIB)
    container[prefix + "/memory.soft_limit_in_bytes"] = str(32 * GIB)
    container[prefix + "/memory.usage_in_bytes"] = str(8 * GIB)
    assert cgroup_memory_bounds(use_protection_budget=True) == (32 * GIB, 24 * GIB)


@pytest.mark.parametrize("protection", ["memory.low", "memory.min"])
@pytest.mark.parametrize("group", ["", "/worker"])
def test_reclaim_protection_does_not_change_capacity(container, protection, group):
    container["/proc/self/cgroup"] = "0::/worker/job\n"
    container["/sys/fs/cgroup/memory.current"] = str(GIB // 2)
    container["/sys/fs/cgroup/memory.stat"] = ""

    for protected_bytes in (0, GIB):
        container[f"/sys/fs/cgroup{group}/{protection}"] = str(protected_bytes)
        assert _total() == 32 * GIB
        assert _free() == 32 * GIB - GIB // 2


@pytest.mark.parametrize("hard", ["max", "malformed"])
def test_protection_without_a_hard_limit_preserves_host_capacity(container, hard):
    container["/sys/fs/cgroup/memory.max"] = hard
    container["/sys/fs/cgroup/memory.low"] = str(GIB)
    container["/sys/fs/cgroup/memory.min"] = str(2 * GIB)
    assert _total() == 128 * GIB
    assert _free() == 100 * GIB


def test_v1_soft_limit_does_not_reduce_default_capacity(container):
    container.clear()
    prefix = "/sys/fs/cgroup/memory"
    container[prefix + "/memory.limit_in_bytes"] = str(32 * GIB)
    container[prefix + "/memory.soft_limit_in_bytes"] = str(GIB)
    container[prefix + "/memory.usage_in_bytes"] = str(GIB // 2)
    assert _total() == 32 * GIB
    assert _free() == 32 * GIB - GIB // 2


def test_clean_active_checkpoint_cache_is_reclaimable(container):
    container["/sys/fs/cgroup/memory.current"] = str(28 * GIB)
    container["/sys/fs/cgroup/memory.stat"] = (
        f"file {23 * GIB}\ninactive_file {4 * GIB}\nactive_file {19 * GIB}\n"
        "shmem 0\nfile_dirty 0\nfile_writeback 0\n")
    assert _free() == 27 * GIB



def test_dirty_shared_and_writeback_pages_are_not_free_cache(container):
    container["/sys/fs/cgroup/memory.current"] = str(28 * GIB)
    container["/sys/fs/cgroup/memory.stat"] = (
        f"file {23 * GIB}\ninactive_file {4 * GIB}\nactive_file {19 * GIB}\n"
        f"shmem {10 * GIB}\nfile_dirty {5 * GIB}\nfile_writeback {3 * GIB}\n")
    assert _free() == 9 * GIB
