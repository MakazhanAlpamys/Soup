"""Recompute training parity and I/O validity from the archived benchmark receipts.

The committed ZIP fixtures contain the members consumed here and a catalog of
their original hashes. Replay verifies the catalog and every member before
using the receipts. Archived programs and setup logs are outside the replay input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile
from zipfile import Path as ZipPath

from benchmarks.harness.arm_validity import CHECKS, Run, arm_outcome, blocks_to_add, repeats
from benchmarks.harness.streaming_controls.foreign_read_control import assess as assess_foreign
from benchmarks.harness.streaming_controls.io_calibration import assess as assess_calibration
from benchmarks.harness.streaming_controls.isolated_compare import compare_results

# Digests of the replay catalogs derived from the two original archive manifests.
CATALOG_SHA256 = frozenset(
    {
        "10b506f64b8837cb455911d7f24f32b4f6e457bc4f90a709a5011b0e032a43d0",
        "e900b9e94a8efd0b83f63c62a5710720b3fa03dbfac0fa912f4a421b09ad11d6",
    }
)
TOKENS = 255
GIB = 1 << 30


def _sha(path: Path | ZipPath) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path | ZipPath) -> Any:
    return json.loads(path.read_bytes())


class Checks:
    def __init__(self) -> None:
        self.results: dict[str, bool] = {}

    def __call__(self, name: str, condition: Any) -> None:
        self.results[name] = bool(condition)

    @property
    def failed(self) -> list[str]:
        return [name for name, ok in self.results.items() if not ok]


def verify_packet(archive: ZipFile, check: Checks) -> dict[str, Any]:
    try:
        catalog = archive.getinfo("manifest.json")
        if catalog.file_size > 65536:
            raise ValueError("replay catalog is too large")
        raw = archive.read(catalog)
    except (BadZipFile, KeyError, ValueError):
        check("catalog_sha256", False)
        return {}
    trusted = hashlib.sha256(raw).hexdigest() in CATALOG_SHA256
    check("catalog_sha256", trusted)
    if not trusted:
        return {}
    meta = json.loads(raw)
    members = meta["members"]
    check("member_set", sorted(archive.namelist()) == sorted(["manifest.json", *members]))
    for name, entry in members.items():
        try:
            info = archive.getinfo(name)
            valid = (
                info.file_size == entry["bytes"]
                and hashlib.sha256(archive.read(info)).hexdigest() == entry["sha256"]
            )
        except (BadZipFile, KeyError):
            valid = False
        check(f"member:{name}", valid)
    return meta


def replay(packet: Path) -> dict[str, Any]:
    check = Checks()
    with ZipFile(packet) as archive:
        meta = verify_packet(archive, check)
        if check.failed:
            return {
                "status": "REPLAY_CHECKS_FAILED",
                "performance_verdict": "NO_VERDICT_INVALID_EVIDENCE",
                "failed_checks": check.failed,
            }
        return _replay(ZipPath(archive), meta, check)


def _replay(e: ZipPath, meta: dict[str, Any], check: Checks) -> dict[str, Any]:
    expected_model = _read(Path(__file__).with_name("representative_model.json"))
    phases = ["fetch", "setup", "controls", "prepare", "measurement"]
    for name in phases:
        row = _read(e / f"{name}_supervised" / "result.json")
        if name == "measurement":
            # A stopped series records its own failure; it is reported, not passed.
            phase_ok = row["group_cleanup_confirmed"] is True
        else:
            phase_ok = row["status"] == "PASS" and row["returncode"] == 0
            phase_ok = phase_ok and row["group_cleanup_confirmed"] is True
        check(f"{name}_supervised", phase_ok)
    cases = ET.fromstring((e / "cuda_tests.xml").read_bytes()).findall(".//testcase")
    clean = [
        c for c in cases if not any(c.find(k) is not None for k in ("failure", "error", "skipped"))
    ]
    check("cuda_controls_all_passed", cases and len(clean) == len(cases))
    fetch = _read(e / "artifact_fetch.json")
    check(
        "artifacts_readback",
        fetch["status"] == "PASS_ALL_ARTIFACTS_SHA_READBACK"
        and fetch["verified_bytes"] == fetch["expected_bytes"],
    )
    model = _read(e / "model_download.json")
    check(
        "model_sha_sizes",
        model["status"] == "PASS_DOWNLOAD_SHA256" and model["files"] == expected_model["files"],
    )

    m = e / "comparison"
    result = _read(m / "result.json")
    order = result["order"]
    planned_blocks = result.get("planned_abba_blocks", len(order) // 4)
    check("registered_order", order == list("ABBA") * planned_blocks)
    check("cleanup_reserve", result["outer_deadline_unix_s"] - result["work_deadline_unix_s"] == 40)
    prepared = e / "prepared.json"
    check(
        "prepared_identity",
        _sha(prepared) == result["prepared_sha256"]
        and _read(prepared)["identity"] == expected_model,
    )

    calibration = _read(m / "io_calibration/receipt.json")
    check(
        "io_calibration",
        assess_calibration(calibration["samples"])["status"] == "PASS_COUNTER_CALIBRATION"
        and calibration["boot_id"] == meta["boot_id"]
        and calibration["device"] == "nvme0n1",
    )
    for i, s in enumerate(calibration["samples"]):
        check(
            f"io_calibration_raw_{i}",
            s["raw_device_end"] - s["raw_device_start"] == s["device_bytes"]
            and s["raw_process_end"] - s["raw_process_start"] == s["process_bytes"],
        )
    foreign = _read(m / "foreign_read_control/receipt.json")
    validity_field = meta["archive_validity_field"]
    foreign_validity = foreign[validity_field]
    check(
        "foreign_reader_control",
        assess_foreign(foreign_validity, foreign["completed_bytes"], foreign["hashes_match"])[
            "status"
        ]
        == "PASS_FOREIGN_READER_DETECTED"
        and foreign_validity["observer_pid"] != foreign_validity["reader_pid"],
    )

    arm_dirs = sorted(
        (path for path in m.iterdir() if path.is_dir() and path.name.startswith("arm_")),
        key=lambda path: path.name,
    )
    completed = result["arms"]
    items, rows, validity_failures, partial = [], [], [], []
    for directory in arm_dirs:
        index = int(directory.name.split("_")[1])
        arm = directory.name.split("_")[2]
        item = _read(directory / "result.json")
        if index >= len(completed):
            partial.append(
                {
                    "index": index,
                    "arm": arm,
                    "status": item.get("status"),
                    "records": len(item.get("records", [])),
                }
            )
            continue
        check(f"arm{index}_aggregate_copy", item == completed[index])
        check(f"arm{index}_registered_position", order[index] == arm)
        sup = _read(m / f"supervisor_{index:02d}" / "result.json")
        check(
            f"arm{index}_supervision",
            sup["status"] == "PASS"
            and sup["returncode"] == 0
            and sup["group_cleanup_confirmed"] is True,
        )
        r = item["records"][0]
        check(
            f"arm{index}_workload",
            item["workload_sha256"] == result["prepared_sha256"]
            and item["effective_tokens_per_step"] == TOKENS
            and item["weight_sha256"] == expected_model,
        )
        validity = r[validity_field]
        clock = validity["gpu_clock"]
        outcome = arm_outcome(validity["checks"], CHECKS)
        if outcome != "ok":
            validity_failures.append(
                {
                    "index": index,
                    "arm": arm,
                    "outcome": outcome,
                    "foreign_read_bytes": validity["checks"]["foreign_reads"]["evidence"][
                        "foreign_read_bytes"
                    ],
                }
            )
        check(
            f"arm{index}_clock",
            clock["outcome"] == "ok" and clock["samples"] >= 2 and clock["unknown_samples"] == 0,
        )
        # Frozen rule (benchmarks/harness/l2l_box.confirm_gpu_owner): an exclusive local CUDA
        # initialization on the sole GPU. NVML may report a host PID, not the container PID.
        owner = item["gpu_ownership"]
        check(
            f"arm{index}_ownership",
            owner["before"] == []
            and len(owner["after"]) == 1
            and owner["context_active"] is True
            and owner["single_device"] is True
            and owner["method"] == "exclusive-local-CUDA-initialization",
        )
        stats = r["initial_stats"]
        check(
            f"arm{index}_residency",
            (r["resident_indices"] == [])
            if arm == "A"
            else (0 < len(r["resident_indices"]) < 32 and stats["resident_bytes"] <= 4 * GIB),
        )
        check(f"arm{index}_direct_io", r["direct_io"] is True)
        if arm == "B":
            check(f"arm{index}_accounted_peak", stats["accounted_peak_bytes"] <= 16 * GIB)
            check(f"arm{index}_cache_served", r["final_stats"]["resident_tensor_bytes_served"] > 0)
        steps = [s for s in r["updates"] if not s["warmup"]]
        check(f"arm{index}_steps", len(steps) == 4 and len(r["updates"]) == 6)
        items.append(item)
        rows.append(
            {
                "index": index,
                "arm": arm,
                "validity": validity["outcome"],
                "median_window_seconds": statistics.median(
                    s["instrumented_window_seconds"] for s in steps
                ),
                "mean_process_read_bytes": statistics.mean(
                    s["physical_io"]["process_bytes"] for s in steps
                ),
                "mean_device_read_bytes": statistics.mean(
                    s["physical_io"]["device_bytes"] for s in steps
                ),
                "resident_layers": len(r["resident_indices"]),
                "accounted_peak_bytes": stats["accounted_peak_bytes"],
                "peak_cuda_allocated_bytes": r["peak_cuda_allocated_bytes"],
            }
        )

    numerical = compare_results(items, order[: len(items)])
    check("numerical_exact_completed_arms", numerical["status"] == "NUMERICAL_PASS")
    complete_series = len(items) == len(order)
    if complete_series:
        blocks = [
            [Run(rows[i + j]["median_window_seconds"], rows[i + j]["validity"]) for j in range(4)]
            for i in range(0, len(rows), 4)
        ]
        try:
            rep = repeats(blocks)
            repetition = {
                "sigma": rep.sigma,
                "mean_a": rep.mean_a,
                "blocks_needed": rep.blocks_needed,
                "blocks_to_add": blocks_to_add(blocks),
            }
        except ValueError as error:
            repetition = {"error": str(error)}
    else:
        repetition = {"error": "series incomplete; repetitions rule not evaluated"}

    summary = {}
    for arm in ("A", "B"):
        group = [row for row in rows if row["arm"] == arm and row["validity"] == "ok"]
        summary[arm] = {
            "valid_arms": len(group),
            "mean_median_window_seconds": statistics.mean(
                r["median_window_seconds"] for r in group
            ),
            "mean_process_read_bytes": statistics.mean(r["mean_process_read_bytes"] for r in group),
            "mean_device_read_bytes": statistics.mean(r["mean_device_read_bytes"] for r in group),
        }
        summary[arm]["tokens_per_second"] = TOKENS / summary[arm]["mean_median_window_seconds"]
    read_reduction = 100 * (
        1 - summary["B"]["mean_process_read_bytes"] / summary["A"]["mean_process_read_bytes"]
    )

    if validity_failures:
        status, verdict = "BLOCKED_FOREIGN_READS", "NO_VERDICT_INVALID_EVIDENCE"
    elif repetition.get("blocks_to_add", 1) != 0:
        status, verdict = "NUMERICAL_PASS_GATES_PASS", "NO_VERDICT_INSUFFICIENT_REPETITIONS"
    else:
        status, verdict = "NUMERICAL_PASS_GATES_PASS", "GATES_PASS_PENDING_REVIEW"
    if check.failed:
        status = "REPLAY_CHECKS_FAILED"
    return {
        "status": status,
        "performance_verdict": verdict,
        "failed_checks": check.failed,
        "checks_run": len(check.results),
        "archive_sha256": meta["archive_sha256"],
        "source_manifest_sha256": meta["source_manifest_sha256"],
        "recorded_gate_sha256": meta["gate_source_sha256"],
        "replay_members": len(meta["members"]),
        "original_members": meta["original_member_count"],
        "cuda_controls_passed": len(clean),
        "planned_arms": len(order),
        "completed_arms": len(items),
        "partial_arms": partial,
        "validity_failures": validity_failures,
        "numerical_completed_arms": numerical,
        "repetitions": repetition,
        "summary_valid_arms": summary,
        "descriptive_process_read_reduction_pct": read_reduction,
        "per_arm": rows,
        "scope": ("Descriptive timing only for dense Llama 3.1 8B BF16 streamed training."),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("packet", type=Path)
    parser.add_argument("--summary", action="store_true", help="omit per-arm rows")
    args = parser.parse_args()
    report = replay(args.packet)
    if args.summary:
        report.pop("per_arm", None)
    from rich.console import Console

    Console().print_json(data=report, highlight=False)
    return 0 if not report["failed_checks"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
