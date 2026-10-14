"""Tests for the arm validity watch (benchmarks/harness/arm_validity.py, F1 tickets 02-07).

The module is benchmark harness code, not shipped. Every sensor reading below is
synthetic: the fake sensors return made-up numbers, and no test writes into
benchmarks/results/. The Windows and macOS paths run here with mocks only; no
test calls a Windows API or runs on a Mac.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[1] / "benchmarks" / "harness"


@pytest.fixture(scope="module")
def validity():
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    name = "_arm_validity_test_module"
    spec = importlib.util.spec_from_file_location(name, HARNESS / "arm_validity.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _readings(*values):
    """A synthetic sensor that returns ``values`` in order, then repeats the last one."""
    queue = list(values)

    def sensor():
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return sensor


ON_MAINS = {"battery": True, "online": {"ACAD": True, "ucsi-source-psy-USBC000:001": False}}
OFF_MAINS = {"battery": True, "online": {"ACAD": False, "ucsi-source-psy-USBC000:001": False}}
NO_BATTERY = {"battery": False, "online": {"ucsi-source-psy-USBC000:001": False}}


GB = 1_000_000_000


def _processes(own, others=None, uninspectable=0):
    """A synthetic process-reads snapshot: ``others`` maps pid to (name, read bytes)."""
    return {
        "own": own,
        "others": {
            pid: {"name": name, "read_bytes": read} for pid, (name, read) in (others or {}).items()
        },
        "uninspectable": uninspectable,
    }


QUIET_PROCESSES = _processes(own=10 * GB, others={4242: ("firefox", 300_000)}, uninspectable=292)


def _disk(*values):
    """A synthetic disk counter for nvme0n1 that returns ``values`` like ``_readings``."""
    read = _readings(*values)

    def sensor(device):
        assert device == "nvme0n1"
        return read()

    return sensor


def _foreign_sensors(kwargs):
    kwargs.setdefault("model_path", "/synthetic/models/qwen")
    kwargs.setdefault("disk_device", lambda path: "nvme0n1")
    kwargs.setdefault("disk_reads", _disk(50 * GB, 50 * GB))
    kwargs.setdefault("other_disk_reads", _other_disks(100 * GB, 100 * GB))
    kwargs.setdefault("process_reads", _readings(QUIET_PROCESSES))


MIB = 1024 * 1024

COMMIT_BEFORE = {"used_bytes": 9_200 * MIB, "limit_bytes": 24_000 * MIB}
COMMIT_AFTER = {"used_bytes": 21_900 * MIB, "limit_bytes": 24_000 * MIB}

PEERS = [{"pid": 3141, "type": "C", "name": "/usr/bin/python3", "used_mib": 412}]


def _box_sensors(kwargs):
    kwargs.setdefault("ram_available", _readings(7_800 * MIB, 6_100 * MIB))
    kwargs.setdefault("commit_charge", _readings(COMMIT_BEFORE, COMMIT_AFTER))
    kwargs.setdefault("gpu_memory_used", _readings(12, 2_900))
    kwargs.setdefault("gpu_processes", _readings([], PEERS))
    kwargs.setdefault("gpu_clock", _readings({"sm_clock_mhz": 1_500, "reasons": []}))


def _watch(validity, **kwargs):
    kwargs.setdefault("platform", "linux")
    kwargs.setdefault("sleep_offset", _readings(25_588.0, 25_588.0))
    kwargs.setdefault("suspend_count", _readings(7, 7))
    kwargs.setdefault("power_supplies", _readings(ON_MAINS))
    _foreign_sensors(kwargs)
    _box_sensors(kwargs)
    with validity.ArmWatch(arm="A", round=2, run=5, label="synthetic", **kwargs) as watch:
        pass
    return watch.record


def test_clean_arm_is_ok_with_identity_and_suspend_check(validity):
    record = _watch(validity)

    assert record["arm"] == "A"
    assert record["round"] == 2
    assert record["run"] == 5
    assert record["label"] == "synthetic"
    suspend = record["checks"]["suspend"]
    assert suspend["outcome"] == "ok"
    assert suspend["reason"]
    assert suspend["evidence"]["sleep_offset_start_s"] == 25_588.0
    assert suspend["evidence"]["sleep_offset_end_s"] == 25_588.0
    assert record["outcome"] == "ok"
    json.dumps(record)


def test_boot_time_offset_growth_over_one_second_voids_the_arm(validity):
    record = _watch(validity, sleep_offset=_readings(25_588.0, 25_650.5))

    suspend = record["checks"]["suspend"]
    assert suspend["outcome"] == "void"
    assert "62.5" in suspend["reason"]
    assert suspend["evidence"]["sleep_offset_start_s"] == 25_588.0
    assert suspend["evidence"]["sleep_offset_end_s"] == 25_650.5
    assert record["outcome"] == "void"


def test_boot_time_offset_jitter_under_one_second_stays_ok(validity):
    record = _watch(validity, sleep_offset=_readings(25_588.0, 25_588.9))

    assert record["checks"]["suspend"]["outcome"] == "ok"


def test_suspend_counter_increase_voids_the_arm(validity):
    record = _watch(validity, suspend_count=_readings(7, 8))

    suspend = record["checks"]["suspend"]
    assert suspend["outcome"] == "void"
    assert "suspend count" in suspend["reason"]
    assert suspend["evidence"]["suspend_count_start"] == 7
    assert suspend["evidence"]["suspend_count_end"] == 8
    assert record["outcome"] == "void"


def _raises():
    raise PermissionError("synthetic: permission denied")


@pytest.mark.parametrize(
    ("sensors", "reason_part"),
    [
        ({"sleep_offset": _raises}, "PermissionError"),
        ({"sleep_offset": _readings(None)}, "returned None"),
        ({"suspend_count": _raises}, "PermissionError"),
        ({"suspend_count": _readings(7, None)}, "returned None"),
    ],
    ids=["offset-raises", "offset-none", "count-raises", "count-none-at-end"],
)
def test_silent_sensor_makes_suspend_unknown_with_reason(validity, sensors, reason_part):
    record = _watch(validity, **sensors)

    suspend = record["checks"]["suspend"]
    assert suspend["outcome"] == "unknown"
    assert reason_part in suspend["reason"]
    assert record["outcome"] == "unknown"


def test_sleep_seen_by_one_sensor_is_void_even_when_the_other_is_silent(validity):
    record = _watch(validity, sleep_offset=_readings(25_588.0, 25_650.5), suspend_count=_raises)

    assert record["checks"]["suspend"]["outcome"] == "void"
    assert record["outcome"] == "void"


def test_unknown_check_that_is_not_required_leaves_the_arm_ok(validity):
    record = _watch(validity, suspend_count=_raises, required=())

    assert record["checks"]["suspend"]["outcome"] == "unknown"
    assert record["required"] == []
    assert record["outcome"] == "ok"


def test_void_check_voids_the_arm_even_when_not_required(validity):
    record = _watch(validity, suspend_count=_readings(7, 8), required=())

    assert record["outcome"] == "void"


def test_every_check_is_required_by_default(validity):
    record = _watch(validity)

    assert record["required"] == ["suspend", "power", "foreign_reads"]


def test_unknown_required_check_name_is_rejected(validity):
    with pytest.raises(ValueError, match="powr"):
        validity.ArmWatch(arm="A", round=0, run=0, required=("suspend", "powr"))


def test_wall_clock_jump_alone_does_not_void_on_linux(validity, monkeypatch):
    wall = _readings(1_000.0, 4_600.0)
    monkeypatch.setattr(validity.time, "time", wall)

    record = _watch(validity)

    assert record["checks"]["suspend"]["outcome"] == "ok"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux sensors only")
def test_real_linux_sensors_return_values(validity):
    if not Path(validity.SUSPEND_SUCCESS_PATH).exists():
        pytest.skip(f"this kernel has no {validity.SUSPEND_SUCCESS_PATH}")
    with validity.ArmWatch(arm="A", round=0, run=0, label="real sensors, test only") as watch:
        pass

    evidence = watch.record["checks"]["suspend"]["evidence"]
    assert isinstance(evidence["sleep_offset_start_s"], float)
    assert isinstance(evidence["sleep_offset_end_s"], float)
    assert isinstance(evidence["suspend_count_start"], int)
    assert isinstance(evidence["suspend_count_end"], int)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux sensors only")
def test_real_linux_power_sensor_returns_values(validity):
    if not Path(validity.POWER_SUPPLY_DIR).exists():
        pytest.skip(f"this kernel has no {validity.POWER_SUPPLY_DIR}")
    with validity.ArmWatch(arm="A", round=0, run=0, label="real sensors, test only") as watch:
        pass

    power = watch.record["checks"]["power"]["evidence"]
    assert power["samples"] >= 2
    assert power["unknown_samples"] == 0
    reading = power["last_reading"]
    assert isinstance(reading["battery"], bool)
    assert all(isinstance(online, bool) for online in reading["online"].values())


class _CountingSensor:
    """A synthetic sensor like ``_readings`` that also counts its calls."""

    def __init__(self, *values):
        self.calls = 0
        self._next = _readings(*values)

    def __call__(self):
        self.calls += 1
        return self._next()


def _watch_while_sampling(validity, sensor, samples, sampled="power_supplies", **kwargs):
    """Run an arm with ``sensor`` as the ``sampled`` sensor on a 10 ms interval
    until it has been called ``samples`` times."""
    interval = {"power_supplies": "power_interval_s", "gpu_clock": "gpu_clock_interval_s"}
    kwargs[sampled] = sensor
    kwargs[interval[sampled]] = 0.01
    kwargs.setdefault("platform", "linux")
    kwargs.setdefault("sleep_offset", _readings(25_588.0))
    kwargs.setdefault("suspend_count", _readings(7))
    kwargs.setdefault("power_supplies", _readings(ON_MAINS))
    _foreign_sensors(kwargs)
    _box_sensors(kwargs)
    with validity.ArmWatch(arm="A", round=2, run=5, label="synthetic", **kwargs) as watch:
        deadline = time.monotonic() + 5.0
        while sensor.calls < samples and time.monotonic() < deadline:
            time.sleep(0.005)
    assert sensor.calls >= samples, f"the {sampled} sampler never ran"
    return watch.record


def test_one_sample_off_mains_voids_the_arm(validity):
    sensor = _CountingSensor(ON_MAINS, ON_MAINS, OFF_MAINS, ON_MAINS)
    record = _watch_while_sampling(validity, sensor, samples=4)

    power = record["checks"]["power"]
    assert power["outcome"] == "void"
    assert "off mains" in power["reason"]
    assert power["evidence"]["samples"] == sensor.calls
    off = power["evidence"]["off_mains_samples"]
    assert len(off) == 1
    assert off[0]["online"] == OFF_MAINS["online"]
    assert record["outcome"] == "void"


def test_all_samples_on_mains_keeps_power_ok(validity):
    sensor = _CountingSensor(ON_MAINS)
    record = _watch_while_sampling(validity, sensor, samples=3)

    power = record["checks"]["power"]
    assert power["outcome"] == "ok"
    assert power["reason"]
    assert power["evidence"]["samples"] == sensor.calls
    assert power["evidence"]["off_mains_samples"] == []
    assert record["outcome"] == "ok"
    json.dumps(record)


def test_no_battery_keeps_power_ok_with_that_reason(validity):
    record = _watch(validity, power_supplies=_readings(NO_BATTERY))

    power = record["checks"]["power"]
    assert power["outcome"] == "ok"
    assert "no battery" in power["reason"]
    assert power["evidence"]["off_mains_samples"] == []
    assert record["outcome"] == "ok"


@pytest.mark.parametrize(
    ("sensor", "reason_part"),
    [
        (_raises, "PermissionError"),
        (_readings(None), "returned None"),
        (_readings(ON_MAINS, None), "returned None"),
    ],
    ids=["raises", "none", "none-in-one-sample"],
)
def test_silent_power_sensor_makes_power_unknown_with_reason(validity, sensor, reason_part):
    record = _watch(validity, power_supplies=sensor)

    power = record["checks"]["power"]
    assert power["outcome"] == "unknown"
    assert reason_part in power["reason"]
    assert record["outcome"] == "unknown"


def test_off_mains_sample_voids_even_when_another_sample_is_silent(validity):
    record = _watch(validity, power_supplies=_readings(OFF_MAINS, None))

    assert record["checks"]["power"]["outcome"] == "void"
    assert record["outcome"] == "void"


def test_two_gb_of_foreign_reads_voids_the_arm(validity):
    record = _watch(
        validity,
        disk_reads=_disk(50 * GB, 53 * GB),
        process_reads=_readings(_processes(own=10 * GB), _processes(own=11 * GB)),
    )

    foreign = record["checks"]["foreign_reads"]
    assert foreign["outcome"] == "void"
    assert foreign["evidence"]["foreign_read_bytes"] == 2 * GB
    assert foreign["evidence"]["device"] == "nvme0n1"
    assert foreign["evidence"]["model_path"] == "/synthetic/models/qwen"
    assert "nvme0n1" in foreign["reason"]
    assert foreign["unattributed_bytes"] == 2 * GB
    assert foreign["readers"] == []
    assert record["outcome"] == "void"
    json.dumps(record)


def test_reads_only_by_the_benchmark_and_its_children_stay_ok(validity):
    record = _watch(
        validity,
        disk_reads=_disk(50 * GB, 58 * GB),
        process_reads=_readings(
            _processes(own=10 * GB, others={4242: ("firefox", 300_000)}),
            _processes(own=18 * GB, others={4242: ("firefox", 300_000)}),
        ),
    )

    foreign = record["checks"]["foreign_reads"]
    assert foreign["outcome"] == "ok"
    assert foreign["evidence"]["own_read_bytes"] == 8 * GB
    assert foreign["evidence"]["foreign_read_bytes"] == 0
    assert foreign["readers"] == []
    assert foreign["unattributed_bytes"] == 0
    assert record["outcome"] == "ok"


def test_same_user_reader_is_named_and_the_rest_is_unattributed(validity):
    record = _watch(
        validity,
        disk_reads=_disk(50 * GB, 54 * GB),
        process_reads=_readings(
            _processes(own=10 * GB, others={4242: ("firefox", 300_000)}, uninspectable=292),
            _processes(
                own=11 * GB,
                others={4242: ("firefox", 300_000), 5150: ("rg", 2 * GB)},
                uninspectable=293,
            ),
        ),
    )

    foreign = record["checks"]["foreign_reads"]
    assert foreign["outcome"] == "void"
    assert foreign["evidence"]["foreign_read_bytes"] == 3 * GB
    assert foreign["readers"] == [{"pid": 5150, "name": "rg", "read_bytes": 2 * GB}]
    assert foreign["unattributed_bytes"] == 1 * GB
    assert foreign["evidence"]["uninspectable_processes"] == 293


def _other_disks(*values):
    """A synthetic counter for every disk but the watched one, like ``_disk``."""
    read = _readings(*values)

    def sensor(device):
        assert device == "nvme0n1"
        return read()

    return sensor


def test_own_reads_on_another_disk_do_not_hide_foreign_reads_on_the_watched_disk(validity):
    # 2 GB of own reads may all be on sda, and the 1.5 GB on nvme0n1 may all be foreign.
    record = _watch(
        validity,
        disk_reads=_disk(50 * GB, 51_500_000_000),
        other_disk_reads=_other_disks(100 * GB, 102 * GB),
        process_reads=_readings(_processes(own=10 * GB), _processes(own=12 * GB)),
    )

    foreign = record["checks"]["foreign_reads"]
    assert foreign["outcome"] == "unknown"
    assert "nvme0n1" in foreign["reason"]
    assert "own reads" in foreign["reason"]
    assert foreign["evidence"]["foreign_read_bytes"] is None
    assert foreign["evidence"]["foreign_read_bytes_min"] == 0
    assert foreign["evidence"]["foreign_read_bytes_max"] == 1_500_000_000
    assert foreign["evidence"]["other_disk_read_bytes_start"] == 100 * GB
    assert foreign["evidence"]["other_disk_read_bytes_end"] == 102 * GB
    assert record["outcome"] == "unknown"
    json.dumps(record)


def test_foreign_reads_over_the_limit_void_even_when_own_reads_hit_another_disk(validity):
    # Own reads are 2 GB at most, so at least 2 GB of the 4 GB on nvme0n1 are foreign.
    record = _watch(
        validity,
        disk_reads=_disk(50 * GB, 54 * GB),
        other_disk_reads=_other_disks(100 * GB, 102 * GB),
        process_reads=_readings(_processes(own=10 * GB), _processes(own=12 * GB)),
    )

    foreign = record["checks"]["foreign_reads"]
    assert foreign["outcome"] == "void"
    assert foreign["evidence"]["foreign_read_bytes_min"] == 2 * GB
    assert foreign["evidence"]["foreign_read_bytes_max"] == 4 * GB
    assert record["outcome"] == "void"


def test_small_watched_disk_reads_stay_ok_when_own_reads_hit_another_disk(validity):
    record = _watch(
        validity,
        disk_reads=_disk(50 * GB, 50_200_000_000),
        other_disk_reads=_other_disks(100 * GB, 102 * GB),
        process_reads=_readings(_processes(own=10 * GB), _processes(own=12 * GB)),
    )

    foreign = record["checks"]["foreign_reads"]
    assert foreign["outcome"] == "ok"
    assert foreign["evidence"]["foreign_read_bytes_max"] == 200_000_000
    assert record["outcome"] == "ok"


def test_own_reads_with_quiet_other_disks_are_all_on_the_watched_disk(validity):
    record = _watch(
        validity,
        disk_reads=_disk(50 * GB, 58 * GB),
        other_disk_reads=_other_disks(100 * GB, 100 * GB),
        process_reads=_readings(_processes(own=10 * GB), _processes(own=18 * GB)),
    )

    foreign = record["checks"]["foreign_reads"]
    assert foreign["outcome"] == "ok"
    assert foreign["evidence"]["foreign_read_bytes"] == 0
    assert foreign["evidence"]["foreign_read_bytes_min"] == 0
    assert foreign["evidence"]["foreign_read_bytes_max"] == 0


def test_silent_other_disks_sensor_makes_foreign_reads_unknown_with_reason(validity):
    record = _watch(validity, other_disk_reads=_other_disks(100 * GB, None))

    foreign = record["checks"]["foreign_reads"]
    assert foreign["outcome"] == "unknown"
    assert "returned None" in foreign["reason"]
    assert foreign["unattributed_bytes"] is None


@pytest.mark.parametrize(
    ("sensors", "reason_part"),
    [
        ({"disk_reads": lambda device: _raises()}, "PermissionError"),
        ({"disk_reads": _disk(50 * GB, None)}, "returned None"),
        ({"process_reads": _raises}, "PermissionError"),
        ({"process_reads": _readings(None)}, "returned None"),
        ({"disk_device": lambda path: _raises()}, "PermissionError"),
        ({"disk_device": lambda path: None}, "maps to no block device"),
        ({"model_path": None}, "no model path"),
    ],
    ids=[
        "disk-raises", "disk-none-at-end", "processes-raise", "processes-none",
        "device-raises", "no-device", "no-model-path",
    ],
)
def test_silent_disk_sensor_makes_foreign_reads_unknown_with_reason(
    validity, sensors, reason_part
):
    record = _watch(validity, **sensors)

    foreign = record["checks"]["foreign_reads"]
    assert foreign["outcome"] == "unknown"
    assert reason_part in foreign["reason"]
    assert foreign["unattributed_bytes"] is None
    assert record["outcome"] == "unknown"
    json.dumps(record)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux sensors only")
def test_real_linux_disk_sensors_return_values(validity):
    model_path = str(HARNESS)
    try:
        device = validity.disk_device(model_path)
    except ValueError as exc:
        pytest.skip(str(exc))
    if device is None:
        pytest.skip(f"{model_path} is on a file system with no block device")
    with validity.ArmWatch(
        arm="A", round=0, run=0, label="real sensors, test only", model_path=model_path
    ) as watch:
        pass

    foreign = watch.record["checks"]["foreign_reads"]
    assert foreign["outcome"] != "unknown", foreign["reason"]
    evidence = foreign["evidence"]
    assert isinstance(evidence["device"], str)
    assert isinstance(evidence["disk_read_bytes_start"], int)
    assert evidence["disk_read_bytes_end"] >= evidence["disk_read_bytes_start"]
    assert isinstance(evidence["other_disk_read_bytes_start"], int)
    assert isinstance(evidence["own_read_bytes"], int)
    assert isinstance(evidence["uninspectable_processes"], int)
    assert isinstance(foreign["unattributed_bytes"], int)
    json.dumps(watch.record)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux sensors only")
def test_unreaped_child_makes_foreign_reads_unknown_with_reason(validity):
    child = None
    with validity.ArmWatch(
        arm="A", round=0, run=0, label="real sensors, test only",
        model_path="/synthetic/models/qwen", disk_device=lambda path: "nvme0n1",
        disk_reads=_disk(50 * GB, 50 * GB),
    ) as watch:
        child = subprocess.Popen(["true"])
        deadline = time.monotonic() + 5.0
        status = Path(f"/proc/{child.pid}/status")
        while "State:\tZ" not in status.read_text() and time.monotonic() < deadline:
            time.sleep(0.01)
    child.wait()

    foreign = watch.record["checks"]["foreign_reads"]
    assert foreign["outcome"] == "unknown"
    assert str(child.pid) in foreign["reason"]
    assert "reap" in foreign["reason"]


def test_box_stamps_before_and_after_hold_the_machine_state(validity):
    record = _watch(validity, power_supplies=_readings(ON_MAINS, OFF_MAINS))

    before = record["box_stamp_before"]
    after = record["box_stamp_after"]
    assert isinstance(before["unix_s"], float)
    assert after["unix_s"] >= before["unix_s"]
    assert before["ram_available_bytes"] == 7_800 * MIB
    assert after["ram_available_bytes"] == 6_100 * MIB
    assert before["power_source"] == "mains"
    assert after["power_source"] == "battery"
    assert before["gpu_processes"] == []
    assert after["gpu_processes"] == PEERS
    assert before["gpu_memory_used_mib"] == 12
    assert after["gpu_memory_used_mib"] == 2_900
    assert before["commit_charge"] == COMMIT_BEFORE
    assert after["commit_charge"] == COMMIT_AFTER
    assert before["unknown"] == {}
    json.dumps(record)


def test_failing_commit_sensor_is_unknown_in_the_box_stamp_and_leaves_the_arm_ok(validity):
    record = _watch(validity, commit_charge=_raises)

    for stamp in (record["box_stamp_before"], record["box_stamp_after"]):
        assert stamp["commit_charge"] is None
        assert "PermissionError" in stamp["unknown"]["commit_charge"]
        assert stamp["ram_available_bytes"] is not None
    assert record["outcome"] == "ok"


def test_linux_commit_charge_is_committed_as_over_commit_limit(validity, monkeypatch, tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       32000000 kB\nMemAvailable:   20000000 kB\n"
        "CommitLimit:    24576000 kB\nCommitted_AS:    9420800 kB\n"
    )
    monkeypatch.setattr(validity, "MEMINFO_PATH", str(meminfo))

    assert validity.commit_charge() == {
        "used_bytes": 9_420_800 * 1024, "limit_bytes": 24_576_000 * 1024
    }


def test_linux_commit_charge_without_the_meminfo_lines_raises(validity, monkeypatch, tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       32000000 kB\nMemAvailable:   20000000 kB\n")
    monkeypatch.setattr(validity, "MEMINFO_PATH", str(meminfo))

    with pytest.raises(ValueError, match="Committed_AS"):
        validity.commit_charge()


def test_failing_ram_sensor_is_unknown_in_the_box_stamp_and_leaves_the_arm_ok(validity):
    record = _watch(validity, ram_available=_raises)

    for stamp in (record["box_stamp_before"], record["box_stamp_after"]):
        assert stamp["ram_available_bytes"] is None
        assert "PermissionError" in stamp["unknown"]["ram_available_bytes"]
        assert stamp["gpu_memory_used_mib"] is not None
    assert record["outcome"] == "ok"


def _clock(mhz, *reasons):
    return {"sm_clock_mhz": mhz, "reasons": list(reasons)}


def test_clock_trace_is_stored_as_min_median_max(validity):
    sensor = _CountingSensor(_clock(1_800), _clock(1_200), _clock(1_500))
    record = _watch_while_sampling(validity, sensor, samples=3, sampled="gpu_clock")

    clock = record["gpu_clock"]
    assert clock["outcome"] == "ok"
    assert clock["reason"]
    assert clock["samples"] == sensor.calls
    assert clock["unknown_samples"] == 0
    assert clock["sm_clock_mhz"] == {"min": 1_200, "median": 1_500, "max": 1_800}
    assert clock["reasons_seen"] == {}
    assert record["outcome"] == "ok"
    json.dumps(record)


def test_throttle_reason_is_recorded_and_leaves_the_arm_ok(validity):
    sensor = _CountingSensor(
        _clock(1_800), _clock(900, "sw_thermal_slowdown", "sw_power_cap"), _clock(1_700)
    )
    record = _watch_while_sampling(validity, sensor, samples=3, sampled="gpu_clock")

    clock = record["gpu_clock"]
    assert clock["outcome"] == "ok"
    assert clock["reasons_seen"] == {"sw_thermal_slowdown": 1, "sw_power_cap": 1}
    assert "sw_thermal_slowdown" in clock["reason"]
    assert clock["sm_clock_mhz"]["min"] == 900
    assert record["outcome"] == "ok"


def _real_gpu_sensors(validity):
    return {
        "gpu_memory_used": validity.gpu_memory_used,
        "gpu_processes": validity.gpu_processes,
        "gpu_clock": validity.gpu_clock,
    }


def test_missing_nvidia_smi_gives_unknown_gpu_fields_with_the_same_shape(
    validity, monkeypatch, tmp_path
):
    normal = _watch(validity)
    monkeypatch.setenv("PATH", str(tmp_path))
    record = _watch(validity, **_real_gpu_sensors(validity))

    clock = record["gpu_clock"]
    assert clock["outcome"] == "unknown"
    assert "nvidia-smi" in clock["reason"]
    assert clock["sm_clock_mhz"] == {"min": None, "median": None, "max": None}
    assert clock["reasons_seen"] == {}
    assert clock.keys() == normal["gpu_clock"].keys()
    for side in ("box_stamp_before", "box_stamp_after"):
        stamp = record[side]
        assert stamp.keys() == normal[side].keys()
        assert stamp["gpu_processes"] is None
        assert stamp["gpu_memory_used_mib"] is None
        assert "nvidia-smi" in stamp["unknown"]["gpu_processes"]
        assert "nvidia-smi" in stamp["unknown"]["gpu_memory_used_mib"]
    assert record.keys() == normal.keys()
    assert record["outcome"] == "ok"
    json.dumps(record)


def _fake_nvidia_smi(directory, script):
    """A synthetic nvidia-smi on PATH: a shell script whose case branches answer each query."""
    tool = directory / "nvidia-smi"
    tool.write_text(f'#!/bin/sh\ncase "$*" in\n{script}\nesac\n')
    tool.chmod(0o755)


@pytest.mark.skipif(sys.platform == "win32", reason="the fake nvidia-smi is a shell script")
def test_nvidia_smi_output_is_parsed_into_the_record(validity, monkeypatch, tmp_path):
    _fake_nvidia_smi(tmp_path, """\
  *clocks.sm*) echo "1500, 0x0000000000000024" ;;
  *memory.used*) echo "2900" ;;
  *PIDS*) cat <<'END' ;;
==============NVSMI LOG==============

Attached GPUs                             : 1
GPU 00000000:01:00.0
    Processes
        GPU instance ID                   : N/A
        Compute instance ID               : N/A
        Process ID                        : 5122
            Type                          : G
            Name                          : /usr/bin/gnome-shell
            Used GPU Memory               : 1 MiB
        GPU instance ID                   : N/A
        Compute instance ID               : N/A
        Process ID                        : 3141
            Type                          : C
            Name                          : C:\\tools\\run: a, b.exe
            Used GPU Memory               : Not available
END""")
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    record = _watch(validity, **_real_gpu_sensors(validity))

    clock = record["gpu_clock"]
    assert clock["outcome"] == "ok"
    assert clock["sm_clock_mhz"] == {"min": 1_500, "median": 1_500, "max": 1_500}
    assert clock["reasons_seen"] == {
        "sw_power_cap": clock["samples"], "sw_thermal_slowdown": clock["samples"]
    }
    stamp = record["box_stamp_after"]
    assert stamp["gpu_memory_used_mib"] == 2_900
    assert stamp["gpu_processes"] == [
        {"pid": 5122, "type": "G", "name": "/usr/bin/gnome-shell", "used_mib": 1},
        {"pid": 3141, "type": "C", "name": "C:\\tools\\run: a, b.exe", "used_mib": None},
    ]
    assert stamp["unknown"] == {}


@pytest.mark.skipif(sys.platform == "win32", reason="the fake nvidia-smi is a shell script")
def test_failing_nvidia_smi_gives_unknown_with_its_message(validity, monkeypatch, tmp_path):
    _fake_nvidia_smi(tmp_path, '  *) echo "No devices were found" >&2; exit 6 ;;')
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    record = _watch(validity, **_real_gpu_sensors(validity))

    clock = record["gpu_clock"]
    assert clock["outcome"] == "unknown"
    assert "No devices were found" in clock["reason"]
    assert "No devices were found" in record["box_stamp_before"]["unknown"]["gpu_processes"]
    assert record["outcome"] == "ok"


def _logging_nvidia_smi(directory):
    """A synthetic nvidia-smi that appends its arguments to ``calls.log`` and answers any GPU."""
    tool = directory / "nvidia-smi"
    log = directory / "calls.log"
    tool.write_text(f"""#!/bin/sh
echo "$*" >> {log}
case "$*" in
  *clocks.sm*) echo "1500, 0x0000000000000000" ;;
  *memory.used*) echo "2900" ;;
  *PIDS*) echo "Attached GPUs : 1" ;;
esac
""")
    tool.chmod(0o755)
    return log


def _gpu_ids_asked(log):
    return {
        argument.removeprefix("--id=")
        for line in log.read_text().splitlines()
        for argument in line.split()
        if argument.startswith("--id=")
    }


@pytest.mark.skipif(sys.platform == "win32", reason="the fake nvidia-smi is a shell script")
@pytest.mark.parametrize(
    ("visible", "gpu_id"),
    [
        ("1", "1"),
        ("2,0", "2"),
        (" 1 , 0", "1"),
        ("GPU-5c1f2d3e-aaaa-bbbb-cccc-0123456789ab,1", "GPU-5c1f2d3e-aaaa-bbbb-cccc-0123456789ab"),
        ("MIG-5c1f2d3e-aaaa-bbbb-cccc-0123456789ab", "MIG-5c1f2d3e-aaaa-bbbb-cccc-0123456789ab"),
    ],
    ids=["index", "first-of-two", "spaces", "uuid", "mig"],
)
def test_gpu_sensors_read_the_gpu_the_workload_sees_as_cuda_0(
    validity, monkeypatch, tmp_path, visible, gpu_id
):
    log = _logging_nvidia_smi(tmp_path)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    record = _watch(validity, **_real_gpu_sensors(validity))

    assert _gpu_ids_asked(log) == {gpu_id}
    assert record["gpu"] == {"id": gpu_id, "source": "CUDA_VISIBLE_DEVICES"}
    assert record["gpu_clock"]["outcome"] == "ok"
    assert record["box_stamp_after"]["unknown"] == {}
    json.dumps(record)


@pytest.mark.skipif(sys.platform == "win32", reason="the fake nvidia-smi is a shell script")
def test_gpu_sensors_read_gpu_0_when_cuda_visible_devices_is_unset(
    validity, monkeypatch, tmp_path
):
    log = _logging_nvidia_smi(tmp_path)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    record = _watch(validity, **_real_gpu_sensors(validity))

    assert _gpu_ids_asked(log) == {"0"}
    assert record["gpu"] == {"id": "0", "source": "default"}


@pytest.mark.skipif(sys.platform == "win32", reason="the fake nvidia-smi is a shell script")
@pytest.mark.parametrize("visible", ["", "-1", "NoDevFiles", ",1"], ids=repr)
def test_workload_with_no_visible_gpu_gives_unknown_gpu_fields_without_asking_nvidia_smi(
    validity, monkeypatch, tmp_path, visible
):
    log = _logging_nvidia_smi(tmp_path)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    record = _watch(validity, **_real_gpu_sensors(validity))

    assert not log.exists()
    assert record["gpu"] == {"id": None, "source": "CUDA_VISIBLE_DEVICES"}
    assert record["gpu_clock"]["outcome"] == "unknown"
    assert "CUDA_VISIBLE_DEVICES" in record["gpu_clock"]["reason"]
    assert "CUDA_VISIBLE_DEVICES" in record["box_stamp_before"]["unknown"]["gpu_processes"]
    assert record["outcome"] == "ok"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux sensors only")
def test_real_box_stamp_and_gpu_clock_sensors_return_values(validity):
    if shutil.which("nvidia-smi") is None:
        pytest.skip("nvidia-smi is not on PATH")
    with validity.ArmWatch(arm="A", round=0, run=0, label="real sensors, test only") as watch:
        pass

    for side in ("box_stamp_before", "box_stamp_after"):
        stamp = watch.record[side]
        assert stamp["unknown"] == {}, stamp["unknown"]
        assert stamp["ram_available_bytes"] > 0
        assert stamp["commit_charge"]["used_bytes"] > 0
        assert stamp["commit_charge"]["limit_bytes"] > 0
        assert stamp["power_source"] in ("mains", "battery")
        assert isinstance(stamp["gpu_processes"], list)
        assert stamp["gpu_memory_used_mib"] >= 0
    clock = watch.record["gpu_clock"]
    assert clock["outcome"] == "ok", clock["reason"]
    assert clock["sm_clock_mhz"]["min"] > 0
    json.dumps(watch.record)


def test_every_check_is_unknown_on_macos_and_the_arm_still_runs(validity):
    with validity.ArmWatch(
        arm="A", round=0, run=0, label="synthetic", platform="darwin",
        gpu_memory_used=_readings(12), gpu_processes=_readings([]),
        gpu_clock=_readings({"sm_clock_mhz": 1_500, "reasons": []}),
    ) as watch:
        pass
    record = watch.record

    for name in ("suspend", "power", "foreign_reads"):
        check = record["checks"][name]
        assert check["outcome"] == "unknown"
        assert check["reason"] == "not implemented on macOS"
    assert record["checks"]["foreign_reads"]["readers"] == []
    assert record["checks"]["foreign_reads"]["unattributed_bytes"] is None
    stamp = record["box_stamp_before"]
    assert stamp["unknown"]["ram_available_bytes"] == "not implemented on macOS"
    assert stamp["unknown"]["commit_charge"] == "not implemented on macOS"
    assert stamp["unknown"]["power_source"] == "not implemented on macOS"
    assert record["outcome"] == "unknown"
    json.dumps(record)


class _FakeKernel32:
    """A synthetic kernel32.dll. ``GetSystemPowerStatus`` answers with the next
    (ACLineStatus, BatteryFlag) pair like ``_readings``, or fails on None."""

    def __init__(
        self, *power, avail_phys=6_100 * MIB, total_page_file=24_000 * MIB,
        avail_page_file=2_100 * MIB,
    ):
        self._power = _readings(*power)
        self._avail_phys = avail_phys
        self._total_page_file = total_page_file
        self._avail_page_file = avail_page_file

    def GetSystemPowerStatus(self, status_ref):  # noqa: N802 - the Windows API name
        answer = self._power()
        if answer is None:
            return 0
        status_ref._obj.ACLineStatus, status_ref._obj.BatteryFlag = answer
        return 1

    def GlobalMemoryStatusEx(self, status_ref):  # noqa: N802 - the Windows API name
        status_ref._obj.ullAvailPhys = self._avail_phys
        status_ref._obj.ullTotalPageFile = self._total_page_file
        status_ref._obj.ullAvailPageFile = self._avail_page_file
        return 1


#: GetSystemPowerStatus answers: (ACLineStatus, BatteryFlag).
AC_ONLINE_CHARGING = (1, 8)
AC_OFFLINE_HIGH = (0, 1)
AC_ONLINE_NO_BATTERY = (1, 128)


def _fake_psutil(*snapshots, children=()):
    """A synthetic psutil module. Each ``process_iter`` call lists the next snapshot
    like ``_readings``: a dict of pid to (name, read bytes), where None read bytes
    means access denied. ``children`` are the pids of this process's descendants."""
    module = types.ModuleType("psutil")
    module.Error = type("Error", (Exception,), {})
    module.AccessDenied = type("AccessDenied", (module.Error,), {})
    module.NoSuchProcess = type("NoSuchProcess", (module.Error,), {})
    snapshot = _readings(*snapshots)

    class Process:
        def __init__(self, pid=None, name=None, read=None):
            self.pid = os.getpid() if pid is None else pid
            self.info = {"pid": self.pid, "name": name}
            self._read = read

        def io_counters(self):
            if self._read is None:
                raise module.AccessDenied(self.pid)
            return types.SimpleNamespace(read_bytes=self._read)

        def children(self, recursive=False):
            assert recursive
            return [Process(pid) for pid in children]

    def process_iter(attrs):
        return [Process(pid, name, read) for pid, (name, read) in snapshot().items()]

    module.Process = Process
    module.process_iter = process_iter
    return module


ME = os.getpid()
QUIET_WINDOWS = {ME: ("python.exe", 10 * GB), 4242: ("MsMpEng.exe", 300_000), 4: ("System", None)}


def _windows_watch(validity, monkeypatch, kernel32=None, psutil=None, **kwargs):
    kernel32 = kernel32 or _FakeKernel32(AC_ONLINE_CHARGING)
    monkeypatch.setattr(validity, "_kernel32", lambda: kernel32)
    monkeypatch.setitem(sys.modules, "psutil", psutil or _fake_psutil(QUIET_WINDOWS))
    kwargs.setdefault("wall_clock", _readings(1_000.0, 1_002.0))
    _box_sensors(kwargs)
    with validity.ArmWatch(
        arm="A", round=2, run=5, label="synthetic", platform="win32", **kwargs
    ) as watch:
        pass
    return watch.record


@pytest.mark.parametrize(
    "end, outcome", [(1_002.0, "ok"), (1_030.0, "ok"), (1_030.5, "void"), (4_600.0, "void")]
)
def test_wall_clock_gap_over_30_seconds_voids_on_windows(validity, monkeypatch, end, outcome):
    record = _windows_watch(validity, monkeypatch, wall_clock=_readings(1_000.0, end))

    suspend = record["checks"]["suspend"]
    assert suspend["outcome"] == outcome
    assert suspend["evidence"]["max_gap_s"] == end - 1_000.0
    assert f"{end - 1_000.0:.1f} s" in suspend["reason"]
    assert record["outcome"] == outcome


def test_silent_wall_clock_makes_suspend_unknown_on_windows(validity, monkeypatch):
    def broken():
        raise OSError("clock unavailable")

    record = _windows_watch(validity, monkeypatch, wall_clock=broken)

    suspend = record["checks"]["suspend"]
    assert suspend["outcome"] == "unknown"
    assert "clock unavailable" in suspend["reason"]


def test_one_power_status_off_mains_voids_on_windows(validity, monkeypatch):
    kernel32 = _FakeKernel32(AC_ONLINE_CHARGING, AC_OFFLINE_HIGH)
    record = _windows_watch(validity, monkeypatch, kernel32=kernel32)

    power = record["checks"]["power"]
    assert power["outcome"] == "void"
    assert power["reason"].startswith("1 of 2 power samples off mains")
    assert record["box_stamp_after"]["power_source"] == "battery"
    assert record["outcome"] == "void"


@pytest.mark.parametrize("status", [AC_ONLINE_NO_BATTERY, (255, 128)])
def test_no_system_battery_keeps_power_ok_on_windows(validity, monkeypatch, status):
    record = _windows_watch(validity, monkeypatch, kernel32=_FakeKernel32(status))

    power = record["checks"]["power"]
    assert power["outcome"] == "ok"
    assert "no battery" in power["reason"]


@pytest.mark.parametrize(
    "status, reason_part",
    [((255, 1), "AC line status 255"), (None, "GetSystemPowerStatus failed")],
)
def test_unreadable_power_status_makes_power_unknown_on_windows(
    validity, monkeypatch, status, reason_part
):
    record = _windows_watch(validity, monkeypatch, kernel32=_FakeKernel32(status))

    power = record["checks"]["power"]
    assert power["outcome"] == "unknown"
    assert reason_part in power["reason"]


def test_one_process_reading_two_gb_voids_on_windows(validity, monkeypatch):
    after = {
        ME: ("python.exe", 12 * GB),
        4242: ("MsMpEng.exe", 300_000),
        5150: ("SearchIndexer.exe", 2 * GB),
        4: ("System", None),
    }
    record = _windows_watch(validity, monkeypatch, psutil=_fake_psutil(QUIET_WINDOWS, after))

    foreign = record["checks"]["foreign_reads"]
    assert foreign["outcome"] == "void"
    assert "SearchIndexer.exe" in foreign["reason"]
    assert foreign["readers"] == [{"pid": 5150, "name": "SearchIndexer.exe", "read_bytes": 2 * GB}]
    assert foreign["unattributed_bytes"] is None
    assert foreign["evidence"]["own_read_bytes"] == 2 * GB
    assert foreign["evidence"]["foreign_read_bytes"] == 2 * GB
    assert foreign["evidence"]["uninspectable_processes"] == 1
    assert record["outcome"] == "void"


def test_reads_split_under_the_limit_per_process_stay_ok_on_windows(validity, monkeypatch):
    after = {
        **QUIET_WINDOWS,
        5150: ("SearchIndexer.exe", 600_000_000),
        5151: ("OneDrive.exe", 600_000_000),
    }
    record = _windows_watch(validity, monkeypatch, psutil=_fake_psutil(QUIET_WINDOWS, after))

    foreign = record["checks"]["foreign_reads"]
    assert foreign["outcome"] == "ok"
    assert foreign["evidence"]["foreign_read_bytes"] == 1_200_000_000
    assert [reader["pid"] for reader in foreign["readers"]] == [5150, 5151]


def test_child_process_reads_are_not_foreign_on_windows(validity, monkeypatch):
    after = {**QUIET_WINDOWS, 6000: ("python.exe", 5 * GB)}
    psutil = _fake_psutil(QUIET_WINDOWS, after, children=(6000,))
    record = _windows_watch(validity, monkeypatch, psutil=psutil)

    foreign = record["checks"]["foreign_reads"]
    assert foreign["outcome"] == "ok"
    assert foreign["readers"] == []
    assert foreign["evidence"]["own_read_bytes"] == 5 * GB


def test_without_psutil_foreign_reads_is_unknown_on_windows(validity, monkeypatch):
    kernel32 = _FakeKernel32(AC_ONLINE_CHARGING)
    monkeypatch.setattr(validity, "_kernel32", lambda: kernel32)
    monkeypatch.setitem(sys.modules, "psutil", None)  # makes ``import psutil`` fail
    kwargs = {"wall_clock": _readings(1_000.0, 1_002.0)}
    _box_sensors(kwargs)
    with validity.ArmWatch(arm="A", round=0, run=0, platform="win32", **kwargs) as watch:
        pass
    record = watch.record

    foreign = record["checks"]["foreign_reads"]
    assert foreign["outcome"] == "unknown"
    assert foreign["reason"] == "psutil not installed"
    assert foreign["readers"] == []
    assert record["checks"]["suspend"]["outcome"] == "ok"
    assert record["checks"]["power"]["outcome"] == "ok"
    assert record["outcome"] == "unknown"
    json.dumps(record)


def test_windows_defaults_fill_the_box_stamp(validity, monkeypatch):
    record = _windows_watch(
        validity, monkeypatch, ram_available=None, commit_charge=None  # the Windows defaults
    )

    stamp = record["box_stamp_before"]
    assert stamp["ram_available_bytes"] == 6_100 * MIB
    assert stamp["commit_charge"] == {
        "used_bytes": 21_900 * MIB, "limit_bytes": 24_000 * MIB
    }
    assert stamp["power_source"] == "mains"
    assert stamp["unknown"] == {}
    assert record["outcome"] == "ok"
    json.dumps(record)


# Repeats rule and block order (ticket 07). Run values are synthetic seconds per step.


def test_block_order_is_a_b_b_a_for_each_block(validity):
    assert validity.block_order(3) == [("A", "B", "B", "A")] * 3


def _blocks(validity, *runs, void=()):
    """Blocks from run values in A, B, B, A order; blocks at indexes in ``void`` hold a void run."""
    return [
        [validity.Run(value, "void" if index in void and position == 1 else "ok")
         for position, value in enumerate(block)]
        for index, block in enumerate(runs)
    ]


#: The 10 A runs are 130, 70, 130, 70 and six 100s; so are the 10 B runs. Each
#: arm's variance is 3600 / 9 = 400, so the pooled sigma is 20. The mean of A is
#: 100, so delta is 10 and N = ceil(7.85 * 2^2) = 32.
PILOT = (
    (130, 70, 100, 100),
    (70, 130, 100, 100),
    (100, 100, 70, 130),
    (100, 100, 130, 70),
    (100, 100, 100, 100),
)

CALM = (100, 100, 100, 100)

#: A pair of blocks whose A runs are 140, 140, 60, 60 and whose B runs are 60,
#: 60, 140, 140: the mean of A stays 100 and each arm's squared deviations add 6400.
NOISY_PAIR = ((140, 60, 60, 140), (60, 140, 140, 60))

#: The 27 blocks that take the pilot up to N = 32: one calm block and 13 noisy pairs.
RUN_UP = (CALM, *(NOISY_PAIR * 13))


def test_blocks_needed_follows_the_formula_on_known_inputs(validity):
    assert validity.blocks_needed(_blocks(validity, *PILOT)) == 32


def test_repeats_gives_sigma_the_mean_of_a_and_n(validity):
    assert validity.repeats(_blocks(validity, *PILOT)) == validity.Repeats(20.0, 100.0, 32)


def test_blocks_needed_is_never_below_two(validity):
    # The A runs are all 100 and the B runs are 101, 101, 99, 99 (variance 4 / 3),
    # so the pooled sigma is 0.82, delta is 10 and the formula gives
    # ceil(7.85 * 0.0067) = 1 block.
    blocks = _blocks(validity, (100, 101, 101, 100), (100, 99, 99, 100))

    assert validity.blocks_needed(blocks) == 2


def test_blocks_to_add_runs_the_pilot_then_up_to_n(validity):
    assert validity.blocks_to_add([]) == 5
    assert validity.blocks_to_add(_blocks(validity, *PILOT[:3])) == 2
    # The pilot counts toward N = 32, so 27 blocks follow it.
    assert validity.blocks_to_add(_blocks(validity, *PILOT)) == 27
    # N is not recomputed before it is reached, however noisy the new blocks are.
    assert validity.blocks_to_add(_blocks(validity, *PILOT, *RUN_UP[:10])) == 17


def test_recomputed_n_that_is_larger_gives_one_top_up(validity):
    # At 32 blocks each arm's squared deviations add to 3600 + 13 * 6400 = 86800
    # over 64 runs, so sigma^2 = 86800 / 63 = 1377.8 and N = ceil(108.2) = 109.
    blocks = _blocks(validity, *PILOT, *RUN_UP)
    assert validity.blocks_to_add(blocks) == 77

    # Top-up blocks far noisier still would raise N again if it were recomputed.
    # N stays at 109, so the rule never asks for a second top-up.
    noisier = ((200, 0, 0, 200), (0, 200, 200, 0)) * 20
    assert validity.blocks_to_add(blocks + _blocks(validity, *noisier)) == 37
    assert validity.blocks_to_add(blocks + _blocks(validity, *noisier, *noisier)) == 0


def test_recomputed_n_that_is_smaller_ends_the_comparison(validity):
    # At 32 blocks the squared deviations are the pilot's 3600 over 64 runs, so
    # sigma^2 = 57.1 and N = ceil(4.5) = 5, below the 32 blocks already run.
    calm = (CALM,) * 27

    assert validity.blocks_to_add(_blocks(validity, *PILOT, *calm)) == 0


def test_block_with_a_void_run_is_replaced_and_does_not_count(validity):
    # The void block's difference of 500 would raise N far above 32 if it counted.
    blocks = _blocks(validity, *PILOT[:2], (100, 600, 600, 100), *PILOT[2:4], void={2})

    assert validity.blocks_to_add(blocks) == 1  # 4 valid blocks of the 5-block pilot
    blocks += _blocks(validity, PILOT[4])
    assert validity.blocks_needed(blocks) == 32
    assert validity.blocks_to_add(blocks) == 27


@pytest.mark.parametrize(
    "runs, message",
    [
        (((100, 110, 110),), "4 runs"),
        (((100, 110, 110, 100),), "at least 2 blocks"),
        (((0, 10, 10, 0), (0, -10, -10, 0)), "mean of A"),
    ],
)
def test_blocks_needed_rejects_input_it_cannot_size(validity, runs, message):
    with pytest.raises(ValueError, match=message):
        validity.blocks_needed(_blocks(validity, *runs))
