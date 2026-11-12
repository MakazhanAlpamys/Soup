"""Resolve the hash-bound external #2162 evidence without implicit downloads."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def destination(root: Path, name: str) -> Path:
    if not name or "\\" in name or ":" in name or name.startswith("/"):
        raise ValueError("invalid evidence path")
    base = os.path.normcase(os.path.realpath(root))
    target = root / name
    actual = os.path.normcase(os.path.realpath(target))
    if os.path.commonpath([base, actual]) != base:
        raise ValueError("evidence path escapes root")
    return target


def inspect(root: Path, bulk: Path | None, key: str) -> tuple[dict, dict, list[str]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected = manifest[key]
    external = manifest["bulk"]["files_sha256"]
    assert external and all(expected[name] == value for name, value in external.items())
    paths, missing = {}, []
    for name, digest in expected.items():
        destination(root, name)  # Validate even when an external file is absent.
        if name in external:
            target = destination(bulk, root.name + "/" + name) if bulk is not None else None
        else:
            target = destination(root, name)
        if target is None or not target.is_file():
            if name not in external:
                raise FileNotFoundError(root / name)
            missing.append(root.name + "/" + name)
            continue
        raw = target.read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(raw).hexdigest() == digest, f"evidence hash mismatch: {target}"
        paths[name] = target
    return manifest, paths, sorted(missing)


@contextmanager
def materialize(root: Path, paths: dict) -> Iterator[Path]:
    """Build a temporary complete tree for the original full audit bodies."""
    with tempfile.TemporaryDirectory(prefix="soup2162-evidence-") as folder:
        target = Path(folder) / root.name
        target.mkdir()
        shutil.copyfile(root / "manifest.json", target / "manifest.json")
        for name, source in paths.items():
            output = destination(target, name)
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, output)
        yield target


def partial_report(title: str, result: dict, rows: list[tuple[str, dict]]) -> str:
    lines = [
        f"# {title} — numbers-only", "",
        "Numerical inputs in Git were hash-checked and recomputed. "
        "Full gate verdict: unavailable until the external bulk evidence is supplied.", "",
        "| Comparison | Median reduction | Ordinary 95% | Family-adjusted interval |",
        "|---|---:|---|---|",
    ]
    for name, value in rows:
        ordinary, adjusted = value["ordinary_95"], value["family_adjusted_ci"]
        lines.append(
            f"| {name} | {value['median_reduction_percent']:.6f}% | "
            f"[{ordinary[0]:.6f}%, {ordinary[1]:.6f}%] | "
            f"[{adjusted[0]:.6f}%, {adjusted[1]:.6f}%] |"
        )
    lines += ["", "Missing bulk files:", ""]
    lines += [f"- `{name}`" for name in result["missing_bulk_files"]]
    lines += ["", "Checks requiring those files:", ""]
    lines += [f"- {name}" for name in result["unavailable_checks"]]
    return "\n".join(lines) + "\n"
