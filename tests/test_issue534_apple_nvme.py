"""Regression coverage for issue #534: Apple Fabric hides internal NVMe."""

import json
import platform
import plistlib
import subprocess


def _completed(args, stdout, returncode=0):
    return subprocess.CompletedProcess(args=args, returncode=returncode, stdout=stdout, stderr=b"")


def test_apple_fabric_apfs_store_is_matched_to_nvme_inventory(monkeypatch):
    import soup_cli.utils.layer_stream as layer_stream

    disk_info = {
        "BusProtocol": "Apple Fabric",
        "SolidState": True,
        "APFSPhysicalStores": [{"APFSPhysicalStore": "disk0s2"}],
    }
    nvme_profile = {
        "SPNVMeDataType": [
            {
                "_name": "Apple SSD Controller",
                "_items": [
                    {
                        "bsd_name": "disk0",
                        "device_model": "APPLE SSD",
                        "volumes": [{"bsd_name": "disk0s2"}],
                    }
                ],
            }
        ]
    }
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        if args[0] == "/usr/sbin/diskutil":
            return _completed(args, plistlib.dumps(disk_info))
        return _completed(args, json.dumps(nvme_profile).encode())

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        layer_stream,
        "_resolve_tool",
        lambda name, *_fallbacks: f"/usr/sbin/{name}",
    )
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = layer_stream._probe_disk_kind("/private/var/model-shards")

    assert result.kind == "nvme"
    diskutil_args = calls[0][0]
    assert diskutil_args[:-1] == [
        "/usr/sbin/diskutil",
        "info",
        "-plist",
    ]
    assert diskutil_args[-1].replace("\\", "/").endswith("/private/var/model-shards")
    assert calls[1][0] == [
        "/usr/sbin/system_profiler",
        "-json",
        "-detailLevel",
        "mini",
        "SPNVMeDataType",
    ]
    assert all(call[1]["shell"] is False for call in calls)


def test_ordinary_sata_ssd_stays_ssd_without_nvme_inventory_probe(monkeypatch):
    import soup_cli.utils.layer_stream as layer_stream

    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return _completed(
            args,
            plistlib.dumps({"BusProtocol": "SATA", "SolidState": True}),
        )

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        layer_stream,
        "_resolve_tool",
        lambda name, *_fallbacks: f"/usr/sbin/{name}",
    )
    monkeypatch.setattr(subprocess, "run", fake_run)

    assert layer_stream._probe_disk_kind("/Volumes/SATA").kind == "ssd"
    assert len(calls) == 1


def test_directory_inside_apfs_volume_is_resolved_through_df(monkeypatch):
    import soup_cli.utils.layer_stream as layer_stream

    disk_info = {
        "BusProtocol": "Apple Fabric",
        "SolidState": True,
        "APFSPhysicalStores": [{"APFSPhysicalStore": "disk0s2"}],
    }
    nvme_profile = {"SPNVMeDataType": [{"_items": [{"bsd_name": "disk0"}]}]}
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        if args[0].endswith("diskutil") and args[-1] != "/dev/disk3s5":
            return _completed(args, plistlib.dumps({"Error": "not a mount point"}), 1)
        if args[0].endswith("df"):
            return _completed(
                args,
                b"Filesystem 512-blocks Used Available Capacity Mounted on\n"
                b"/dev/disk3s5 100 50 50 50% /System/Volumes/Data\n",
            )
        if args[0].endswith("diskutil"):
            return _completed(args, plistlib.dumps(disk_info))
        return _completed(args, json.dumps(nvme_profile).encode())

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        layer_stream,
        "_resolve_tool",
        lambda name, *_fallbacks: f"/usr/sbin/{name}",
    )
    monkeypatch.setattr(subprocess, "run", fake_run)

    assert layer_stream._probe_disk_kind("relative/shards").kind == "nvme"
    assert calls[1][0][1:] == ["-P", calls[1][0][-1]]
    assert calls[2][0][-1] == "/dev/disk3s5"


def test_unmatched_apple_fabric_store_is_not_promoted_to_nvme(monkeypatch):
    import soup_cli.utils.layer_stream as layer_stream

    disk_info = {
        "BusProtocol": "Apple Fabric",
        "SolidState": True,
        "APFSPhysicalStores": [{"APFSPhysicalStore": "disk9s2"}],
    }
    nvme_profile = {
        "SPNVMeDataType": [
            {"_items": [{"bsd_name": "disk0", "volumes": [{"bsd_name": "disk0s2"}]}]}
        ]
    }

    def fake_run(args, **_kwargs):
        if args[0].endswith("diskutil"):
            return _completed(args, plistlib.dumps(disk_info))
        return _completed(args, json.dumps(nvme_profile).encode())

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        layer_stream,
        "_resolve_tool",
        lambda name, *_fallbacks: f"/usr/sbin/{name}",
    )
    monkeypatch.setattr(subprocess, "run", fake_run)

    assert layer_stream._probe_disk_kind("/Volumes/UnknownAPFS").kind == "ssd"


def test_unknown_darwin_media_remains_conservative(monkeypatch):
    import soup_cli.utils.layer_stream as layer_stream

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        layer_stream,
        "_resolve_tool",
        lambda name, *_fallbacks: "/usr/sbin/diskutil" if name == "diskutil" else None,
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda args, **_kwargs: _completed(
            args,
            plistlib.dumps({"BusProtocol": "Apple Fabric", "SolidState": False}),
        ),
    )

    assert layer_stream._probe_disk_kind("/Volumes/Unknown").kind == "unknown"


def test_malformed_diskutil_plist_fails_closed(monkeypatch):
    import soup_cli.utils.layer_stream as layer_stream

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        layer_stream,
        "_resolve_tool",
        lambda name, *_fallbacks: "/usr/sbin/diskutil" if name == "diskutil" else None,
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda args, **_kwargs: _completed(args, b"not a property list"),
    )

    assert layer_stream._probe_disk_kind("/Volumes/Broken").kind == "unknown"
