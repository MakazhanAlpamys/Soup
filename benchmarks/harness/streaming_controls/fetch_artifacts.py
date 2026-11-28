"""Stage hash-pinned artifacts on a host for offline installation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path
from urllib.parse import urlsplit


def inventory(manifest: dict) -> list[dict]:
    """Validate the complete inventory before any request or directory creation."""
    rows = [{**row, "path": "wheels/" + row["filename"]} for row in manifest["files"]]
    runtime = manifest["python"]
    rows.append(
        {
            "path": runtime["name"],
            "url": runtime["browser_download_url"],
            "bytes": runtime["size"],
            "sha256": runtime["digest"].removeprefix("sha256:"),
        }
    )
    seen = set()
    for index, row in enumerate(rows):
        name = row["path"].removeprefix("wheels/")
        if not re.fullmatch(r"[A-Za-z0-9_.+-]+", name) or name in {".", ".."}:
            raise ValueError("Unsafe artifact filename")
        if row["path"] in seen:
            raise ValueError("Duplicate artifact")
        seen.add(row["path"])
        if type(row["bytes"]) is not int or row["bytes"] <= 0:
            raise ValueError("Invalid artifact length")
        if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError("Invalid artifact SHA")
        url = urlsplit(row["url"])
        valid_source = (
            url.hostname == "files.pythonhosted.org"
            if index < len(rows) - 1
            else url.hostname == "github.com"
            and url.path.startswith("/astral-sh/python-build-standalone/releases/download/")
        )
        if (
            url.scheme != "https"
            or not valid_source
            or url.username
            or url.password
            or url.port not in (None, 443)
            or url.query
            or url.fragment
        ):
            raise ValueError("Unexpected artifact source")
    if sum(row["bytes"] for row in rows) > 3_400_000_000:
        raise ValueError("Artifact traffic bound exceeded")
    return rows


def fetch(manifest_path: Path, root: Path, receipt_path: Path, deadline: float) -> dict:
    """One request per artifact, no retry/resume/fallback; fail before installation."""
    if not math.isfinite(deadline) or not 0 < deadline - time.time() <= 1200:
        raise ValueError("Require an unexpired deadline within 1200 seconds")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    rows = inventory(manifest)
    for name, expected in manifest["lock_sha256"].items():
        if (
            Path(name).name != name
            or hashlib.sha256((manifest_path.parent / name).read_bytes()).hexdigest() != expected
        ):
            raise ValueError("Lock SHA mismatch")
    root = root.absolute()
    if os.path.realpath(root) != str(root):
        raise ValueError("Symlink destination refused")
    if root.exists() or receipt_path.exists() or receipt_path.is_symlink():
        raise FileExistsError("Fresh destination and receipt required; no retry")
    root.mkdir(parents=False)
    (root / "wheels").mkdir()
    state = {
        "status": "FETCHING",
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "deadline_unix_s": deadline,
        "expected_bytes": sum(r["bytes"] for r in rows),
        "verified_bytes": 0,
        "files": [],
        "performance_verdict": "NO_VERDICT",
    }

    def save() -> None:
        temporary = receipt_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
        temporary.replace(receipt_path)

    try:
        save()
        for row in rows:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError("Artifact deadline expired")
            target = root / row["path"]
            partial = target.with_name(target.name + ".partial")
            state["current_file"] = row["path"]
            save()
            result = subprocess.run(
                [
                    "curl",
                    "--disable",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--location",
                    "--proto",
                    "=https",
                    "--proto-redir",
                    "=https",
                    "--retry",
                    "0",
                    "--connect-timeout",
                    str(min(15, remaining)),
                    "--max-time",
                    str(remaining),
                    "--max-filesize",
                    str(row["bytes"]),
                    "--output",
                    str(partial),
                    "--write-out",
                    "%{http_code}",
                    row["url"],
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=remaining,
                check=False,
            )
            if result.returncode != 0 or result.stdout.strip() != "200":
                raise RuntimeError(
                    f"Artifact transfer failed: {row['path']}; "
                    f"exit={result.returncode}; http={result.stdout.strip()}"
                )
            if partial.is_symlink() or partial.stat().st_size != row["bytes"]:
                raise ValueError("Artifact length mismatch")
            digest = hashlib.sha256()
            with partial.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 << 20), b""):
                    if time.time() >= deadline:
                        raise TimeoutError("Artifact readback deadline expired")
                    digest.update(chunk)
            if digest.hexdigest() != row["sha256"]:
                raise ValueError("Artifact SHA mismatch")
            partial.rename(target)
            state["files"].append({k: row[k] for k in ("path", "bytes", "sha256")})
            state["verified_bytes"] += row["bytes"]
            save()
        state["status"] = "PASS_ALL_ARTIFACTS_SHA_READBACK"
    except BaseException as error:
        state.update(status="BLOCKED", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        state["finished_unix_s"] = time.time()
        save()
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--deadline-unix-s", type=float, required=True)
    args = parser.parse_args()
    fetch(args.manifest, args.directory, args.receipt, args.deadline_unix_s)


if __name__ == "__main__":
    main()
