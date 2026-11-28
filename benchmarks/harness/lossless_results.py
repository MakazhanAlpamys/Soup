"""Lossless, bounded-memory artifact archives; no tensor arithmetic or filtering.

All original bytes survive. Content-addressed chunks deduplicate repeated files;
byte-plane shuffling exposes zero float mantissa bytes; XOR against the first
same-name chunk compresses related captures. Every encoded chunk, reconstructed
chunk and reconstructed file has an independent SHA-256 check.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import io
import json
import os
import tarfile
import tempfile
import zlib
from pathlib import Path, PurePosixPath
from typing import Any

CHUNK_BYTES = 4 * 1024 * 1024


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    hashed = hashlib.sha256()
    with path.open("rb") as source:
        for data in iter(lambda: source.read(CHUNK_BYTES), b""):
            hashed.update(data)
    return hashed.hexdigest()


def relative(name: str) -> str:
    path = PurePosixPath(name)
    if (not name or name == "." or "\\" in name or ":" in name
            or path.is_absolute() or ".." in path.parts or str(path) != name):
        raise ValueError("invalid artifact path")
    return name


def within(root: Path, name: str) -> Path:
    target = root / relative(name)
    base, resolved = os.path.realpath(root), os.path.realpath(target)
    if os.path.commonpath([base, resolved]) != base:
        raise ValueError("artifact escapes root")
    return target


def plane(data: bytes, width: int, reverse: bool = False) -> bytes:
    if width == 1:
        return data
    if width != 4 or len(data) % width:
        raise ValueError("invalid byte-plane size")
    if not reverse:
        return b"".join(data[index::width] for index in range(width))
    output = bytearray(len(data))
    size = len(data) // width
    for index in range(width):
        output[index::width] = data[index * size:(index + 1) * size]
    return bytes(output)


def xor(left: bytes, right: bytes) -> bytes:
    if len(left) != len(right):
        raise ValueError("XOR reference length mismatch")
    return (int.from_bytes(left, "little") ^ int.from_bytes(right, "little")).to_bytes(
        len(left), "little",
    )


def pack(root: Path, names: list[str], archive: Path) -> dict[str, Any]:
    """Pack static/snapshotted input files into a standalone uncompressed tar.

    Payload objects are already compressed. Each archive is self-contained;
    no preceding export, cached model or external base tensor is needed.
    """
    root, archive = Path(root), Path(archive)
    files, objects, anchors = {}, {}, {}
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lossless-results-", dir=archive.parent) as tmp:
        temporary = Path(tmp)
        for name in sorted(set(names)):
            if name == "original-files.json":
                raise ValueError("reserved manifest name")
            path = within(root, name)
            if path.is_symlink() or not path.is_file():
                raise ValueError("artifact must be a regular file")
            hashed, size, refs = hashlib.sha256(), 0, []
            with path.open("rb") as source:
                for index, data in enumerate(iter(lambda: source.read(CHUNK_BYTES), b"")):
                    hashed.update(data)
                    size += len(data)
                    identity = sha(data)
                    refs.append(identity)
                    key = (path.name, index, len(data))
                    if identity not in objects:
                        width = 4 if len(data) % 4 == 0 else 1
                        encoded = zlib.compress(plane(data, width), level=3)
                        record = {"bytes": len(data), "codec": "zlib", "width": width}
                        if key in anchors:
                            first_id, first_path, offset = anchors[key]
                            # Reference original input only while encoding. Decoder
                            # uses the already archived content-addressed object.
                            with first_path.open("rb") as first:
                                first.seek(offset)
                                base = first.read(len(data))
                            if sha(base) != first_id:
                                raise ValueError("input changed while encoding")
                            delta = zlib.compress(plane(xor(data, base), width), level=3)
                            if len(delta) < len(encoded):
                                encoded = delta
                                record["reference"] = first_id
                        if len(encoded) >= len(data):
                            encoded = data
                            record = {"bytes": len(data), "codec": "raw", "width": 1}
                        record.update(encoded_bytes=len(encoded), encoded_sha256=sha(encoded))
                        objects[identity] = record
                        (temporary / identity).write_bytes(encoded)
                    anchors.setdefault(key, (identity, path, index * CHUNK_BYTES))
            files[name] = {"bytes": size, "sha256": hashed.hexdigest(), "chunks": refs}
        manifest = {"format": "soup-lossless-results-v1", "chunk_bytes": CHUNK_BYTES,
                    "files": files, "objects": objects}
        manifest_bytes = (json.dumps(manifest, sort_keys=True, separators=(",", ":"))
                          + "\n").encode()
        partial = archive.with_name(archive.name + ".incoming")
        with tarfile.open(partial, "w") as output:
            member = tarfile.TarInfo("original-files.json")
            member.size = len(manifest_bytes)
            output.addfile(member, io.BytesIO(manifest_bytes))
            for identity, record in sorted(objects.items()):
                member = tarfile.TarInfo("objects/" + identity)
                member.size = record["encoded_bytes"]
                with (temporary / identity).open("rb") as source:
                    output.addfile(member, source)
        os.replace(partial, archive)
    return {"archive_bytes": archive.stat().st_size, "archive_sha256": file_sha(archive),
            "original_bytes": sum(row["bytes"] for row in files.values()),
            "original_file_count": len(files), "unique_chunks": len(objects),
            "format": manifest["format"], "manifest_sha256": sha(manifest_bytes)}


def unpack(archive: Path, destination: Path) -> dict[str, Any]:
    """Reconstruct and independently hash every original file before success."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:") as source:
        members = source.getmembers()
        if any(not member.isfile() for member in members):
            raise ValueError("archive contains a non-file entry")
        if len({member.name for member in members}) != len(members):
            raise ValueError("duplicate archive entry")
        raw_manifest = source.extractfile("original-files.json").read()
        manifest = json.loads(raw_manifest)
        if (manifest["format"] != "soup-lossless-results-v1"
                or manifest["chunk_bytes"] != CHUNK_BYTES):
            raise ValueError("unsupported lossless format")
        records = manifest["objects"]
        expected = {"original-files.json"} | {"objects/" + name for name in records}
        if {member.name for member in members} != expected:
            raise ValueError("archive object set mismatch")
        visiting = set()

        @functools.lru_cache(maxsize=8)
        def chunk(identity: str) -> bytes:
            if (len(identity) != 64 or any(char not in "0123456789abcdef" for char in identity)
                    or identity in visiting):
                raise ValueError("invalid or cyclic chunk reference")
            visiting.add(identity)
            try:
                row = records[identity]
                if not 0 < row["bytes"] <= CHUNK_BYTES:
                    raise ValueError("invalid chunk size")
                encoded = source.extractfile("objects/" + identity).read()
                if len(encoded) != row["encoded_bytes"] or sha(encoded) != row["encoded_sha256"]:
                    raise ValueError("encoded chunk hash mismatch")
                if row["codec"] == "raw":
                    data = encoded
                elif row["codec"] == "zlib":
                    decompressor = zlib.decompressobj()
                    decoded = decompressor.decompress(encoded, row["bytes"] + 1)
                    if (not decompressor.eof or decompressor.unused_data
                            or len(decoded) != row["bytes"]):
                        raise ValueError("invalid compressed chunk")
                    data = plane(decoded, row["width"], reverse=True)
                    if "reference" in row:
                        data = xor(data, chunk(row["reference"]))
                else:
                    raise ValueError("unknown chunk codec")
                if len(data) != row["bytes"] or sha(data) != identity:
                    raise ValueError("reconstructed chunk hash mismatch")
                return data
            finally:
                visiting.discard(identity)

        verified, hardlinked = 0, 0
        known_files = {}
        for name, row in manifest["files"].items():
            target = within(destination, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".lossless-incoming")
            within(destination, str(partial.relative_to(destination)).replace("\\", "/"))
            if target.is_symlink() or partial.is_symlink():
                raise ValueError("symlink destination forbidden")
            key = (row["bytes"], row["sha256"])
            original = known_files.get(key)
            if original is not None:
                if original.stat().st_size != row["bytes"] or file_sha(original) != row["sha256"]:
                    raise ValueError("verified duplicate changed")
                try:
                    if partial.exists():
                        partial.unlink()
                    os.link(original, partial)
                except OSError:
                    # Filesystems without hard links retain the ordinary exact path.
                    pass
                else:
                    os.replace(partial, target)
                    verified += 1
                    hardlinked += row["bytes"]
                    continue
            hashed, size = hashlib.sha256(), 0
            with partial.open("wb") as output:
                for identity in row["chunks"]:
                    data = chunk(identity)
                    output.write(data)
                    hashed.update(data)
                    size += len(data)
            if size != row["bytes"] or hashed.hexdigest() != row["sha256"]:
                raise ValueError("reconstructed file hash mismatch")
            os.replace(partial, target)
            known_files.setdefault(key, target)
            verified += 1
        within(destination, "original-files.json").write_bytes(raw_manifest)
    return {"all_original_files_verified": verified,
            "reconstructed_bytes": sum(row["bytes"] for row in manifest["files"].values()),
            "duplicate_file_bytes_hardlinked": hardlinked,
            "manifest_sha256": sha(raw_manifest), "archive_sha256": file_sha(Path(archive))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["pack", "unpack"])
    parser.add_argument("archive", type=Path)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--selection", type=Path, help="JSON list of relative file paths")
    args = parser.parse_args()
    if args.mode == "pack":
        if args.selection is None:
            parser.error("pack requires --selection")
        result = pack(args.folder, json.loads(args.selection.read_text()), args.archive)
    else:
        result = unpack(args.archive, args.folder)
    from rich.console import Console

    Console().print(json.dumps(result))


if __name__ == "__main__":
    main()
