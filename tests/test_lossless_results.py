"""Artifact round trips and failure injection, independent of model arithmetic."""

import importlib.util
import io
import json
import os
import struct
import tarfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "lossless_results", Path(__file__).parents[1] / "benchmarks/harness/lossless_results.py",
)
CODEC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CODEC)


def rewrite(archive, change_manifest=None, change_object=False):
    with tarfile.open(archive, "r:") as source:
        contents = {entry.name: source.extractfile(entry).read() for entry in source}
    if change_manifest is not None:
        manifest = json.loads(contents["original-files.json"])
        change_manifest(manifest)
        contents["original-files.json"] = json.dumps(manifest).encode()
    if change_object:
        key = next(name for name in contents if name.startswith("objects/"))
        data = bytearray(contents[key])
        data[len(data) // 2] ^= 1
        contents[key] = bytes(data)
    with tarfile.open(archive, "w") as target:
        for name, data in contents.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            target.addfile(entry, io.BytesIO(data))


def fixture_bundle(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    random = os.urandom(2301)
    floats = b"".join(struct.pack("<f", value) for value in [0, -0.0, 1, -2.5, float("nan")])
    (root / "random").write_bytes(random)
    (root / "empty").write_bytes(b"")
    (root / "trace.safetensors").write_bytes(floats * 2048)
    archive = tmp_path / "bundle.tar"
    index = CODEC.pack(root, ["random", "empty", "trace.safetensors"], archive)
    return root, archive, index


def test_binary_roundtrip_including_nan_payloads_and_empty(tmp_path):
    root, archive, index = fixture_bundle(tmp_path)
    proof = CODEC.unpack(archive, tmp_path / "retained")
    assert proof["all_original_files_verified"] == 3
    assert proof["archive_sha256"] == index["archive_sha256"]
    for path in root.iterdir():
        assert (tmp_path / "retained" / path.name).read_bytes() == path.read_bytes()


def test_dedup_and_xor_restore_changed_reference_and_chunk_boundaries(tmp_path):
    root = tmp_path / "source"
    size = CODEC.CHUNK_BYTES + 128
    original = os.urandom(size)
    altered = bytearray(original)
    altered[100] ^= 1
    altered[CODEC.CHUNK_BYTES + 4] ^= 64
    for name, data in [("a", original), ("b", original), ("c", altered)]:
        folder = root / name
        folder.mkdir(parents=True)
        (folder / "trace.safetensors").write_bytes(data)
    archive = tmp_path / "bundle.tar"
    index = CODEC.pack(root, [f"{name}/trace.safetensors" for name in "abc"], archive)
    assert index["unique_chunks"] == 4
    assert index["archive_bytes"] < size * 1.05
    proof = CODEC.unpack(archive, tmp_path / "retained")
    assert proof["all_original_files_verified"] == 3
    assert proof["duplicate_file_bytes_hardlinked"] == size
    assert (tmp_path / "retained/c/trace.safetensors").read_bytes() == altered
    with tarfile.open(archive) as source:
        manifest = json.load(source.extractfile("original-files.json"))
    assert sum("reference" in row for row in manifest["objects"].values()) == 2


def test_byteplane_calibration_preserves_fp32_bf16_bits(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    random = os.urandom(256 * 1024)
    payload = bytearray(len(random) * 2)
    payload[2::4] = random[::2]
    payload[3::4] = random[1::2]
    (root / "trace").write_bytes(payload)
    index = CODEC.pack(root, ["trace"], tmp_path / "bundle.tar")
    assert index["archive_bytes"] < len(payload) * .6
    CODEC.unpack(tmp_path / "bundle.tar", tmp_path / "retained")
    assert (tmp_path / "retained/trace").read_bytes() == payload


def test_reject_corrupt_encoded_payload(tmp_path):
    _, archive, _ = fixture_bundle(tmp_path)
    rewrite(archive, change_object=True)
    with pytest.raises(ValueError, match="encoded chunk hash"):
        CODEC.unpack(archive, tmp_path / "retained")


def test_reject_incorrect_original_file_hash(tmp_path):
    _, archive, _ = fixture_bundle(tmp_path)
    rewrite(archive, lambda value: value["files"]["random"].update(sha256="0" * 64))
    with pytest.raises(ValueError, match="file hash"):
        CODEC.unpack(archive, tmp_path / "retained")


def test_reject_incorrect_reconstructed_chunk_hash(tmp_path):
    _, archive, _ = fixture_bundle(tmp_path)

    # Pick the compressible floating-point object, not the random raw object.
    def changed_plane(value):
        for row in value["objects"].values():
            if row["codec"] == "zlib":
                row["width"] = 1
                return
        raise AssertionError("fixture has no compressed object")

    rewrite(archive, changed_plane)
    with pytest.raises(ValueError, match="reconstructed chunk hash"):
        CODEC.unpack(archive, tmp_path / "retained")


@pytest.mark.parametrize("name", ["../escape", "/escape", "C:/escape", "bad\\name", "."])
def test_reject_traversal_on_pack(tmp_path, name):
    with pytest.raises(ValueError, match="path"):
        CODEC.pack(tmp_path, [name], tmp_path / "bundle.tar")


def test_reject_traversal_on_decode(tmp_path):
    _, archive, _ = fixture_bundle(tmp_path)

    def mutate(value):
        value["files"]["../escape"] = value["files"].pop("random")

    rewrite(archive, mutate)
    with pytest.raises(ValueError, match="path"):
        CODEC.unpack(archive, tmp_path / "retained")
    assert not (tmp_path / "escape").exists()


def test_reject_reserved_manifest_name(tmp_path):
    with pytest.raises(ValueError, match="reserved"):
        CODEC.pack(tmp_path, ["original-files.json"], tmp_path / "bundle.tar")
