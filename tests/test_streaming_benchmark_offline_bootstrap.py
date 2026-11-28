"""Reject corrupted or mismatched bootstrap bytes before any installer executes."""

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.harness.streaming_controls.offline_bootstrap import verify_bundle


def fixture(tmp_path: Path) -> tuple[Path, dict]:
    wheel = tmp_path / "uv-0.13.5-py3-none-manylinux_2_17_x86_64.whl"
    wheel.write_bytes(b"test wheel bytes")
    manifest = {
        "filename": wheel.name,
        "bytes": wheel.stat().st_size,
        "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
    }
    return wheel, manifest


def test_readback_and_tamper(tmp_path):
    wheel, manifest = fixture(tmp_path)
    assert verify_bundle(tmp_path, manifest) == wheel
    wheel.write_bytes(b"changed")
    with pytest.raises(ValueError, match="^Wheel size mismatch$"):
        verify_bundle(tmp_path, manifest)


@pytest.mark.parametrize("name", ["../outside.whl", "/tmp/outside.whl", "a/b.whl"])
def test_path_escape(tmp_path, name):
    _, manifest = fixture(tmp_path)
    manifest["filename"] = name
    with pytest.raises(ValueError, match="^Unsafe wheel filename$"):
        verify_bundle(tmp_path, manifest)


@pytest.mark.requires_symlink
def test_symlink_rejected(tmp_path):
    wheel, manifest = fixture(tmp_path)
    target = tmp_path / "target"
    wheel.rename(target)
    wheel.symlink_to(target)
    with pytest.raises(ValueError, match="^Expected contained regular wheel$"):
        verify_bundle(tmp_path, manifest)


def test_pinned_manifest_format():
    root = Path(__file__).parents[1]
    manifest = json.loads(
        (root / "benchmarks/harness/streaming_controls/bootstrap_uv.json").read_text()
    )
    assert manifest["version"] == "0.13.5"
    assert manifest["sha256"] == "fa8d26e08e34c29c0c4e820cf3181757086d547240b4f4d31ff1d8bd810ca946"


def test_inventory_preserves_locked_versions_and_hashes():
    import re

    root = Path(__file__).parents[1] / "benchmarks/harness/streaming_controls"
    manifest = json.loads((root / "offline_artifacts.json").read_text())
    expected = {}
    for name, digest in manifest["lock_sha256"].items():
        text = (root / name).read_text()
        assert hashlib.sha256(text.encode()).hexdigest() == digest
        current = None
        for line in text.splitlines():
            match = re.match(r"^([\w.-]+)==([^\s;]+)", line)
            if match:
                current = match.groups()
                expected.setdefault(current, set())
            if current:
                expected[current].update(re.findall(r"--hash=sha256:([0-9a-f]{64})", line))
    rows = [row for row in manifest["files"] if row["name"] != "uv"]
    assert len(rows) == len(expected)
    assert {(r["name"], r["version"]) for r in rows} == set(expected)
    for row in rows:
        assert row["sha256"] in expected[(row["name"], row["version"])]
        assert row["url"].startswith("https://files.pythonhosted.org/")
