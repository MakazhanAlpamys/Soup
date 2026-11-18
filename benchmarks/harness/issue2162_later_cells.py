"""Verify the archived 5090 and second-3090 #2162 measurement cells.

By default it recomputes the published block statistics from Git snapshots.
With --bundle it runs the preserved CPU auditors in a temporary tree. It never
starts a model, downloads an archive, or changes the original campaign directories.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
MANIFEST = RESULTS / "issue2162-later-cells-v1" / "manifest.json"
MAX_EXPANDED = 80 * 1024 * 1024


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def destination(root: Path, name: str) -> Path:
    if not name or "\\" in name or ":" in name or name.startswith("/"):
        raise ValueError(f"unsafe archive path: {name!r}")
    base = os.path.normcase(os.path.realpath(root))
    target = root / name
    resolved = os.path.normcase(os.path.realpath(target))
    if os.path.commonpath([base, resolved]) != base:
        raise ValueError(f"archive path escapes extraction root: {name!r}")
    return target


def entries(archive: ZipFile, root: Path) -> list:
    seen: set[str] = set()
    total = 0
    files = []
    for info in archive.infolist():
        name = info.filename
        destination(root, name)
        if name in seen:
            raise ValueError(f"duplicate archive member: {name}")
        seen.add(name)
        if stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK:
            raise ValueError(f"symbolic link in archive: {name}")
        if not info.is_dir():
            total += info.file_size
            if total > MAX_EXPANDED:
                raise ValueError("archive expanded size exceeds 80 MiB")
            files.append(info)
    return files


def extract_bytes(raw: bytes, root: Path) -> None:
    with ZipFile(io.BytesIO(raw)) as archive:
        for info in entries(archive, root):
            target = destination(root, info.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def snapshot_checks(manifest: dict) -> None:
    for name, expected in manifest["snapshots"].items():
        path = destination(RESULTS, name)
        raw = path.read_bytes().replace(b"\r\n", b"\n")
        if len(raw) != expected["bytes"] or hashlib.sha256(raw).hexdigest() != expected["sha256"]:
            raise ValueError(f"Git snapshot differs from manifest: {name}")


def published_medians(root: Path) -> dict[str, float]:
    pattern = re.compile(
        r"^\| ((?:block|confirm)[1-6]-(?:native|qkvo|mixed)) \| ([0-9]+\.[0-9]{9}) \|"
    )
    found = {}
    for line in (root / "results.md").read_text(encoding="utf-8").splitlines():
        if match := pattern.match(line):
            found[match.group(1)] = float(match.group(2))
    return found


def check_comparison(value: dict, reductions: list[float]) -> dict:
    if len(reductions) != 6 or len(value["reductions_percent"]) != 6:
        raise ValueError("changed six-block/confirmation roster")
    if not np.allclose(reductions, value["reductions_percent"], rtol=0, atol=1e-6):
        raise ValueError("published paired reductions differ from medians")
    recorded = np.array(value["reductions_percent"], dtype=float)
    if not np.isfinite(recorded).all() or value["resamples"] != 10000:
        raise ValueError("invalid published bootstrap inputs")
    median = float(np.median(recorded))
    draws = np.random.default_rng(value["seed"]).choice(recorded, size=(10000, 6), replace=True)
    boot = np.median(draws, axis=1)
    quantiles = np.quantile(boot, [0.025, 0.975, *value["adjusted_quantiles"]])
    expected = [*value["ordinary_95"], *value["family_adjusted_ci"]]
    if abs(median - value["median_reduction_percent"]) > 1e-9:
        raise ValueError("published median differs from paired reductions")
    if not np.allclose(quantiles, expected, rtol=0, atol=1e-9):
        raise ValueError("published intervals differ from bootstrap")
    positive = value["ordinary_95"][0] > 0 and value["family_adjusted_ci"][0] > 0
    if value["positive_lower_bounds"] != positive:
        raise ValueError("published lower-bound flag differs")
    if value["passes_two_percent"] != (median >= 2 and positive):
        raise ValueError("published two-percent flag differs")
    return {
        "median_reduction_percent": median,
        "ordinary_95": list(quantiles[:2]),
        "family_adjusted_ci": list(quantiles[2:]),
    }


def numbers_check(root: Path) -> dict:
    stage = json.loads((root / "stage-a-decision.json").read_text(encoding="utf-8"))
    audit = json.loads((root / "independent-audit.json").read_text(encoding="utf-8"))
    medians = published_medians(root)
    blocks = stage["blocks"]
    if set(blocks) != {str(n) for n in range(1, 7)}:
        raise ValueError("published block roster differs")
    for number, block in blocks.items():
        for arm in ("native", "qkvo", "mixed"):
            tag = f"block{number}-{arm}"
            if tag not in medians or abs(medians[tag] - block[arm]) > 1e-8:
                raise ValueError(f"report median differs from decision block: {tag}")
    pairs = {
        "mixed_vs_native": ("native", "mixed"),
        "mixed_vs_qkvo": ("qkvo", "mixed"),
        "qkvo_vs_native": ("native", "qkvo"),
    }
    comparisons = {}
    for name, (baseline, candidate) in pairs.items():
        reductions = [
            100 * (blocks[str(n)][baseline] - blocks[str(n)][candidate]) / blocks[str(n)][baseline]
            for n in range(1, 7)
        ]
        value = stage["comparisons"][name]
        if value["correction_family"] != audit["correction_family"] or value["seed"] != 42:
            raise ValueError(f"published family or seed differs: {name}")
        comparisons[name] = check_comparison(value, reductions)
    if stage["comparisons"]["mixed_vs_native"] != audit["primary"]:
        raise ValueError("decision and independent-audit primary differ")
    if stage["eligible"] != audit["stage_a_passes"]:
        raise ValueError("decision and independent-audit Stage A flags differ")
    confirmation = None
    if (root / "confirmation-decision.json").exists():
        decision = json.loads((root / "confirmation-decision.json").read_text(encoding="utf-8"))
        reductions = []
        for number in range(1, 7):
            native, mixed = medians[f"confirm{number}-native"], medians[f"confirm{number}-mixed"]
            reductions.append(100 * (native - mixed) / native)
        confirmation = check_comparison(decision["comparison"], reductions)
        if decision["comparison"] != audit["confirmation"]:
            raise ValueError("decision and independent-audit confirmation differ")
        if decision["passes"] != audit["confirmation_passes"]:
            raise ValueError("confirmation flags differ")
    elif audit["confirmation"] is not None or audit["confirmation_passes"] is not None:
        raise ValueError("missing published confirmation decision")
    return {
        "comparisons": comparisons,
        "confirmation": confirmation,
        "stage_a_numerical_flag": stage["comparisons"]["mixed_vs_native"]["passes_two_percent"],
    }


def bundle_checks(archive: ZipFile, manifest: dict) -> None:
    files = entries(archive, RESULTS)
    names = {info.filename for info in files}
    inventory = json.loads(archive.read("inventory.json"))
    if names != set(inventory) | {"inventory.json"}:
        raise ValueError("bundle inventory does not cover its members exactly")
    for name, expected in inventory.items():
        raw = archive.read(name)  # ZipFile checks the CRC.
        if len(raw) != expected["bytes"] or hashlib.sha256(raw).hexdigest() != expected["sha256"]:
            raise ValueError(f"bundle member differs from inventory: {name}")
    for name, expected in manifest["nested_archives"].items():
        if inventory.get(name) != expected:
            raise ValueError(f"nested archive differs from manifest: {name}")


def copy_member(archive: ZipFile, name: str, root: Path, relative: str) -> None:
    output = destination(root, relative)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(archive.read(name))


def prepare_5090(archive: ZipFile, root: Path) -> tuple[Path, Path]:
    run = root / "5090-performance-v1"
    run.mkdir(parents=True)
    extract_bytes(archive.read("rtx5090/evidence.zip"), run)
    extract_bytes(archive.read("rtx5090/source.zip"), run / "source")
    for name in archive.namelist():
        if name.startswith("rtx5090/run-extra/"):
            copy_member(archive, name, run, name.removeprefix("rtx5090/run-extra/"))
        elif name.startswith("rtx5090/recovery-attempts/"):
            copy_member(archive, name, run, name.removeprefix("rtx5090/"))
        elif name.startswith("rtx5090/parent/"):
            copy_member(archive, name, root, name.removeprefix("rtx5090/parent/"))
    for path in run.glob("*.py"):
        shutil.copyfile(path, root / path.name)
    return root / "audit-5090-desktop-resume.py", run


def prepare_3090(archive: ZipFile, root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True)
    extract_bytes(archive.read("rtx3090/evidence.zip"), root)
    extract_bytes(archive.read("rtx3090/source.zip"), root / "source")
    for name in archive.namelist():
        if name.startswith("rtx3090/scripts/"):
            copy_member(archive, name, root, name.removeprefix("rtx3090/scripts/"))
    return root / "audit.py", root


def run_audit(script: Path, run: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(script), str(run)],
        cwd=script.parent,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"{script.name} failed ({result.returncode}): {result.stderr[-4000:]}")


def compare_outputs(run: Path, snapshot: Path) -> dict:
    actual = json.loads((run / "independent-audit.json").read_text(encoding="utf-8"))
    expected = json.loads((snapshot / "independent-audit.json").read_text(encoding="utf-8"))
    actual.pop("completed_at", None)
    expected.pop("completed_at", None)
    if actual != expected:
        raise ValueError(f"CPU audit differs from preserved result: {snapshot.name}")
    actual_report = (run / "results.md").read_bytes().replace(b"\r\n", b"\n")
    expected_report = (snapshot / "results.md").read_bytes().replace(b"\r\n", b"\n")
    if actual_report != expected_report:
        raise ValueError(f"report differs from preserved result: {snapshot.name}")
    if not actual.get("verified"):
        raise ValueError(f"CPU audit is not verified: {snapshot.name}")
    return {
        "audit_verified": True,
        "timing_processes": actual.get("timing_processes"),
        "stage_a_passes": actual.get("stage_a_passes"),
        "confirmation_passes": actual.get("confirmation_passes"),
    }


def verify(bundle: Path | None) -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    snapshot_checks(manifest)
    numbers = {
        "rtx5090": numbers_check(RESULTS / "issue2162-later-5090-v1"),
        "rtx3090": numbers_check(RESULTS / "issue2162-later-3090-v1"),
    }
    if bundle is None:
        return {
            "mode": "numbers-only",
            "verified": False,
            "numbers_verified": True,
            "numbers": numbers,
            "snapshots_verified": len(manifest["snapshots"]),
            "missing": [manifest["bundle"]["filename"]],
            "unavailable_checks": [
                "raw timing, exactness and telemetry verification",
                "source and recovery-attempt verification",
                "independent CPU auditor replay",
            ],
        }
    expected = manifest["bundle"]
    if bundle.stat().st_size != expected["bytes"] or digest(bundle) != expected["sha256"]:
        raise ValueError("bundle differs from the manifest SHA-256 or size")
    with ZipFile(bundle) as archive:
        bundle_checks(archive, manifest)
        with tempfile.TemporaryDirectory(prefix="soup2162-later-cells-") as temp:
            scratch = Path(temp)
            script5090, run5090 = prepare_5090(archive, scratch / "rtx5090")
            script3090, run3090 = prepare_3090(archive, scratch / "rtx3090")
            run_audit(script5090, run5090)
            run_audit(script3090, run3090)
            return {
                "mode": "full-cpu-audit",
                "verified": True,
                "snapshots_verified": len(manifest["snapshots"]),
                "rtx5090": compare_outputs(run5090, RESULTS / "issue2162-later-5090-v1"),
                "rtx3090": compare_outputs(run3090, RESULTS / "issue2162-later-3090-v1"),
            }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, help="previously downloaded versioned archive")
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    args = parser.parse_args()
    result = verify(args.bundle)
    report = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(report, end="")
    else:
        args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
