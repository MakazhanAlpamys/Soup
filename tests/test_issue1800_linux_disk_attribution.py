"""Linux disk classification follows the cache volume, not unrelated hardware."""

import os
import platform
from pathlib import Path
from types import SimpleNamespace

import pytest

from soup_cli.utils import layer_stream as ls


@pytest.fixture
def sysfs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Real directories with emulated sysfs links (also usable on Windows)."""
    devices = tmp_path / "sys" / "devices"
    dev_block = tmp_path / "sys" / "dev" / "block"
    dev_block.mkdir(parents=True)
    links = {}
    realpath = os.path.realpath
    real_stat = os.stat
    cache = str(tmp_path / "cache")
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(ls, "_LINUX_SYS_DEV_BLOCK", str(dev_block), raising=False)
    monkeypatch.setattr(os, "major", lambda dev: dev >> 8, raising=False)
    monkeypatch.setattr(os, "minor", lambda dev: dev & 255, raising=False)

    def stat(path, *args, **kwargs):
        if str(path) == cache:
            return SimpleNamespace(st_dev=2049)
        return real_stat(path, *args, **kwargs)

    def resolve(path, *args, **kwargs):
        key = os.path.normpath(str(path))
        return links[key] if key in links else realpath(path, *args, **kwargs)

    def device(name, rotational=None, *, parent=None, slaves=()):
        node = (parent or devices) / name
        node.mkdir(parents=True, exist_ok=True)
        if parent is not None:
            (node / "partition").write_text("1", encoding="utf-8")
        if rotational is not None:
            (node / "queue").mkdir(exist_ok=True)
            (node / "queue/rotational").write_text(str(rotational), encoding="utf-8")
        if slaves:
            (node / "slaves").mkdir(exist_ok=True)
            for member in slaves:
                link = node / "slaves" / member.name
                link.mkdir()
                links[str(link)] = str(member)
        return node

    monkeypatch.setattr(os, "stat", stat)
    monkeypatch.setattr(os.path, "realpath", resolve)
    measured_paths = []
    monkeypatch.setattr(
        ls, "_measure_seq_read_bytes_per_s",
        lambda path: measured_paths.append(path) or 150_000_000,
    )
    return SimpleNamespace(
        device=device, links=links, volume=os.path.normpath(f"{dev_block}/8:1"),
        cache=cache, measured_paths=measured_paths,
    )


def test_rotational_cache_does_not_borrow_unrelated_nvme(sysfs, monkeypatch) -> None:
    sysfs.device("nvme0n1", 0)
    sysfs.links[sysfs.volume] = str(sysfs.device("sda", 1))
    real_listdir = os.listdir
    monkeypatch.setattr(
        os, "listdir",
        lambda path: ["nvme0n1", "sda"] if str(path) == "/sys/block" else real_listdir(path),
    )
    verdict = ls._probe_disk_kind(sysfs.cache)
    assert verdict == ls.DiskClassification("hdd", 150_000_000)
    assert sysfs.measured_paths == [sysfs.cache]
    with pytest.raises(ValueError, match="not NVMe"):
        ls.choose_tier(1000, 100, verdict)


@pytest.mark.parametrize("name,rotational,expected", [("nvme0n1", 0, "nvme"), ("sda", 0, "ssd")])
def test_attributed_solid_state_device_never_measures(sysfs, name, rotational, expected) -> None:
    sysfs.links[sysfs.volume] = str(sysfs.device(name, rotational))
    assert ls._probe_disk_kind(sysfs.cache) == ls.DiskClassification(expected)
    assert sysfs.measured_paths == []


@pytest.mark.parametrize("name,expected", [("nvme0n1", "nvme"), ("sda", "ssd")])
def test_partition_resolves_to_its_parent_disk(sysfs, name, expected) -> None:
    disk = sysfs.device(name, 0)
    sysfs.links[sysfs.volume] = str(sysfs.device(name + "p1", parent=disk))
    assert ls._probe_disk_kind(sysfs.cache).kind == expected
    assert sysfs.measured_paths == []


@pytest.mark.parametrize("stack", ["dm-0", "md0"])
def test_stacked_volume_classifies_its_members(sysfs, stack) -> None:
    disk = sysfs.device("nvme0n1", 0)
    part = sysfs.device("nvme0n1p1", parent=disk)
    sysfs.links[sysfs.volume] = str(sysfs.device(stack, 0, slaves=[part]))
    assert ls._probe_disk_kind(sysfs.cache).kind == "nvme"


def test_mixed_stack_uses_worst_member(sysfs) -> None:
    nvme = sysfs.device("nvme0n1", 0)
    sata = sysfs.device("sda", 0)
    sysfs.links[sysfs.volume] = str(sysfs.device("dm-0", 0, slaves=[nvme, sata]))
    assert ls._probe_disk_kind(sysfs.cache).kind == "ssd"


def test_mixed_rotational_stack_uses_volume_measurement(sysfs) -> None:
    nvme = sysfs.device("nvme0n1", 0)
    hdd = sysfs.device("sda", 1)
    sysfs.links[sysfs.volume] = str(sysfs.device("md0", 0, slaves=[nvme, hdd]))
    assert ls._probe_disk_kind(sysfs.cache) == ls.DiskClassification("hdd", 150_000_000)
    assert sysfs.measured_paths == [sysfs.cache]


@pytest.mark.parametrize("measured,expected", [(1_500_000_000, "nvme"), (None, "hdd")])
def test_unattributed_volume_falls_back_to_its_own_read(
    sysfs, monkeypatch, measured, expected,
) -> None:
    calls = []
    monkeypatch.setattr(ls, "_measure_seq_read_bytes_per_s", lambda p: calls.append(p) or measured)
    assert ls._probe_disk_kind(sysfs.cache) == ls.DiskClassification(expected, measured)
    assert calls == [sysfs.cache]


def test_fast_virtio_keeps_issue365_measured_fallback(sysfs, monkeypatch) -> None:
    sysfs.links[sysfs.volume] = str(sysfs.device("vda", 1))
    monkeypatch.setattr(ls, "_measure_seq_read_bytes_per_s", lambda path: 1_500_000_000)
    assert ls._probe_disk_kind(sysfs.cache) == ls.DiskClassification("nvme", 1_500_000_000)


def test_unreadable_path_still_attempts_its_own_measurement(sysfs, monkeypatch) -> None:
    stat = os.stat

    def unreadable(path, *args, **kwargs):
        if str(path) == sysfs.cache:
            raise OSError("volume inaccessible")
        return stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", unreadable)
    assert ls._probe_disk_kind(sysfs.cache).kind == "hdd"
    assert sysfs.measured_paths == [sysfs.cache]


def test_cyclic_sysfs_members_do_not_recurse_forever(sysfs) -> None:
    device = sysfs.device("dm-0", 0)
    link = device / "slaves" / "cycle"
    link.mkdir(parents=True)
    sysfs.links[str(link)] = str(device)
    sysfs.links[sysfs.volume] = str(device)
    assert ls._probe_disk_kind(sysfs.cache).kind == "hdd"
    assert sysfs.measured_paths == [sysfs.cache]


def test_dangling_sysfs_entry_is_unknown_even_if_its_name_looks_like_nvme(sysfs) -> None:
    gone = Path(sysfs.volume).parent.parent / "devices" / "gone" / "nvme0n1"
    sysfs.links[sysfs.volume] = str(gone)
    assert ls._probe_disk_kind(sysfs.cache) == ls.DiskClassification("hdd", 150_000_000)
    assert sysfs.measured_paths == [sysfs.cache]


def test_stack_with_an_unattributable_member_is_measured_not_trusted(sysfs) -> None:
    ssd = sysfs.device("sda", 0)
    odd = sysfs.device("odd0")
    sysfs.links[sysfs.volume] = str(sysfs.device("dm-0", 0, slaves=[ssd, odd]))
    assert ls._probe_disk_kind(sysfs.cache) == ls.DiskClassification("hdd", 150_000_000)
    assert sysfs.measured_paths == [sysfs.cache]


@pytest.mark.parametrize("content", [b"\xff\xfe", b"2", b""])
def test_garbage_rotational_value_is_measured_not_guessed(sysfs, content) -> None:
    disk = sysfs.device("sdx")
    (disk / "queue").mkdir()
    (disk / "queue" / "rotational").write_bytes(content)
    sysfs.links[sysfs.volume] = str(disk)
    assert ls._probe_disk_kind(sysfs.cache) == ls.DiskClassification("hdd", 150_000_000)
    assert sysfs.measured_paths == [sysfs.cache]


def test_a_very_deep_stack_is_bounded_and_measured(sysfs) -> None:
    node = sysfs.device("n0", 0)
    for index in range(1, 40):
        node = sysfs.device(f"n{index}", 0, slaves=[node])
    sysfs.links[sysfs.volume] = str(node)
    assert ls._probe_disk_kind(sysfs.cache) == ls.DiskClassification("hdd", 150_000_000)
    assert sysfs.measured_paths == [sysfs.cache]
