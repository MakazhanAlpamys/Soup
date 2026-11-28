"""Readback verification for a separately staged, pinned Linux bootstrap wheel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


def verify_bundle(root: Path, manifest: dict) -> Path:
    name = manifest["filename"]
    if (
        not isinstance(name, str)
        or not re.fullmatch(r"[A-Za-z0-9_.-]+\.whl", name)
        or name in {".", ".."}
    ):
        raise ValueError("Unsafe wheel filename")
    root = Path(os.path.realpath(root))
    wheel = root / name
    if (
        wheel.is_symlink()
        or not wheel.is_file()
        or os.path.commonpath([str(root), os.path.realpath(wheel)]) != str(root)
    ):
        raise ValueError("Expected contained regular wheel")
    if type(manifest["bytes"]) is not int or wheel.stat().st_size != manifest["bytes"]:
        raise ValueError("Wheel size mismatch")
    digest = hashlib.sha256()
    with wheel.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != manifest["sha256"]:
        raise ValueError("Wheel SHA mismatch")
    return wheel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name("bootstrap_uv.json").read_text())
    wheel = verify_bundle(args.directory, manifest)
    from rich.console import Console

    Console().print(f"PASS_BOOTSTRAP_SHA: {wheel.name}; Linux execution not tested")


if __name__ == "__main__":
    main()
