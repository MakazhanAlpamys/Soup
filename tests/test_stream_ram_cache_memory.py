"""Memory budgets use actual allocations and the current cgroup hierarchy."""

from types import SimpleNamespace

import pytest

from soup_cli.utils import stream_ram_cache
from soup_cli.utils.stream_ram_cache import available_memory, cache_budget, plan_hotset

GIB = 2**30


def fixture(tmp_path, version=2, member="/tenant/worker"):
    mount = tmp_path / "cg"
    own = mount / "worker"
    own.mkdir(parents=True)
    cgroup = tmp_path / "cgroup"
    cgroup.write_text(("0::" if version == 2 else "2:memory:") + member + "\n")
    mountinfo = tmp_path / "mountinfo"
    kind, options = ("cgroup2", "rw") if version == 2 else ("cgroup", "rw,memory")
    mountinfo.write_text(f"1 0 0:1 /tenant {mount.as_posix()} rw - {kind} cgroup {options}\n")
    return mount, own, {"cgroup_file": cgroup, "mountinfo_file": mountinfo}


def write(folder, **values):
    for key, value in values.items():
        (folder / key).write_text(str(value))


def test_v2_high_and_ancestor_limit_are_respected(tmp_path):
    root, own, files = fixture(tmp_path)
    write(
        own,
        **{
            "memory.max": 20 * GIB,
            "memory.high": 16 * GIB,
            "memory.low": GIB,
            "memory.current": 18 * GIB,
            "memory.stat": f"file {15 * GIB}\nshmem {2 * GIB}\n"
            f"file_dirty {GIB}\nfile_writeback {GIB}\n"
            f"unevictable {GIB}\n",
        },
    )
    write(root, **{"memory.max": 40 * GIB, "memory.current": 35 * GIB, "memory.stat": "file 0\n"})
    report = available_memory(host_available=30 * GIB, **files)
    assert report["available_bytes"] == 5 * GIB
    assert len(report["constraints"]) == 3


def test_v2_low_is_a_protection_not_an_upper_limit(tmp_path):
    _root, own, files = fixture(tmp_path)
    write(
        own,
        **{
            "memory.max": 20 * GIB,
            "memory.low": GIB,
            "memory.current": 4 * GIB,
            "memory.stat": "file 0\n",
        },
    )
    assert available_memory(host_available=30 * GIB, **files)["available_bytes"] == 16 * GIB


def test_v1_clean_active_file_cache_is_reclaimable_but_dirty_and_shmem_are_not(tmp_path):
    _root, own, files = fixture(tmp_path, version=1)
    write(
        own,
        **{
            "memory.limit_in_bytes": 42 * GIB,
            "memory.soft_limit_in_bytes": 30 * GIB,
            "memory.usage_in_bytes": 40 * GIB,
            "memory.stat": f"total_cache {32 * GIB}\ntotal_shmem {GIB}\n"
            f"total_dirty {GIB}\ntotal_writeback {GIB}\n"
            f"total_unevictable {GIB}\ntotal_inactive_file {2 * GIB}\n",
        },
    )
    report = available_memory(host_available=50 * GIB, **files)
    assert report["available_bytes"] == 18 * GIB
    assert report["constraints"][0]["reclaimable_file_bytes"] == 28 * GIB


@pytest.mark.parametrize("spelling", ["max", str(2**63 - 4096)])
def test_unlimited_limit_does_not_create_a_fake_cap(tmp_path, spelling):
    _root, own, files = fixture(tmp_path)
    write(own, **{"memory.max": spelling, "memory.current": GIB, "memory.stat": "file 0\n"})
    assert available_memory(host_available=9 * GIB, **files)["available_bytes"] == 9 * GIB


def test_failed_probe_never_admits_unchecked_cache_memory(tmp_path):
    _root, own, files = fixture(tmp_path)
    write(own, **{"memory.max": "not-an-integer", "memory.current": 1})
    assert available_memory(host_available=9 * GIB, **files)["available_bytes"] == 0


def test_mount_root_cannot_map_an_unrelated_member(tmp_path):
    _root, _own, files = fixture(tmp_path, member="/other/worker")
    assert available_memory(host_available=9 * GIB, **files)["available_bytes"] == 0


def test_cgroup_nested_mapping_and_host_available_outer_bound(tmp_path):
    root, _own, files = fixture(tmp_path, member="/tenant/worker/leaf")
    leaf = root / "worker/leaf"
    leaf.mkdir()
    write(leaf, **{"memory.max": 20 * GIB, "memory.current": GIB, "memory.stat": "file 0\n"})
    assert available_memory(host_available=3 * GIB, **files)["available_bytes"] == 3 * GIB


def test_budget_retains_thirty_percent_and_runtime_reserve():
    assert cache_budget(20 * GIB, 10 * GIB) == 7 * GIB - 512 * 2**20
    assert cache_budget(1, 10 * GIB) == 1
    assert cache_budget(GIB, 128 * 2**20) == 0


def test_pageable_budget_includes_alignment_slack_and_centers_hotset():
    plan = plan_hotset([4096] * 8, [0] * 8, 2, 16384, pinned=False)
    assert plan["indices"] == [3, 4]
    assert plan["allocated_bytes"] == 16384


def test_singleton_groups_already_staged_are_not_duplicated():
    plan = plan_hotset([4096] * 10, [0] * 8 + [1, 2], 2, 10 * GIB, pinned=False)
    assert plan["indices"] == list(range(8))


def test_pinned_budget_charges_arena_rounding_instead_of_payload_only():
    plan = plan_hotset([700 * 2**20] * 20, [0] * 20, 2, 2 * GIB, pinned=True)
    assert plan["indices"] == [9, 10]
    assert plan["allocated_bytes"] == 2 * GIB


def test_pinned_hotset_charges_sector_alignment_before_rounding():
    plan = plan_hotset([4096] * 4, [0] * 4, 1, 8192, pinned=True)
    assert plan == {"indices": [1], "allocated_bytes": 8192}


def test_unknown_host_memory_refuses_cache(tmp_path):
    missing = tmp_path / "missing"
    assert (
        available_memory(host_available=0, cgroup_file=missing, mountinfo_file=missing)[
            "available_bytes"
        ]
        == 0
    )


@pytest.mark.parametrize(("platform", "expected"), [("darwin", 10 * GIB), ("linux", 0)])
def test_missing_proc_cgroup_respects_platform(tmp_path, monkeypatch, platform, expected):
    monkeypatch.setattr(stream_ram_cache, "sys", SimpleNamespace(platform=platform))
    report = available_memory(
        host_available=10 * GIB,
        cgroup_file=tmp_path / "no-cgroup",
        mountinfo_file=tmp_path / "no-mountinfo",
    )
    assert report["available_bytes"] == expected
    assert ("error" in report) == (platform == "linux")
