"""Compare ordered local Soup benchmark reports without running a model.

Pass report paths in execution order, for example A1 B1 B2 A2. Classification
uses the recorded variant, not filename substrings. The bootstrap independently
resamples whole measured requests within each arm. Its interval is conditional
on the observed runs and does not cover unobserved thermal/scheduling drift.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import statistics
from pathlib import Path
from typing import Any


def workspace_path(path: str | Path) -> Path:
    root = os.path.realpath(Path(__file__).absolute().parent.parent)
    target = os.path.realpath(path)
    if os.path.normcase(os.path.commonpath((root, target))) != os.path.normcase(root):
        raise ValueError(f"Paths must stay inside the experimental workspace: {path}")
    return Path(target)


def percentile(sorted_values: list[float], fraction: float) -> float:
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def throughput(samples: list[dict[str, Any]]) -> float:
    return sum(row["tokens"] for row in samples) / sum(row["seconds"] for row in samples)


def arm_summary(reports: list[dict[str, Any]]) -> dict[str, Any]:
    samples = [sample for report in reports for sample in report["samples"]]
    peak_stages = [
        stage
        for report in reports
        for stage in [report["load"], *report.get("warmups", []), *report["samples"]]
    ]
    elapsed = sum(row["seconds"] for row in samples)
    tokens = sum(row["tokens"] for row in samples)
    result = {
        "runs": len(reports),
        "requests": len(samples),
        "generated_tokens": tokens,
        "request_seconds_total": elapsed,
        "aggregate_tokens_per_second": tokens / elapsed,
        "median_request_tokens_per_second": statistics.median(
            row["tokens"] / row["seconds"] for row in samples
        ),
        "latency_per_token_ms": elapsed * 1000 / tokens,
        "median_request_seconds": statistics.median(row["seconds"] for row in samples),
        "request_tokens_per_second_min": min(row["tokens"] / row["seconds"] for row in samples),
        "request_tokens_per_second_max": max(row["tokens"] / row["seconds"] for row in samples),
        "cpu_utilization_percent": sum(row["cpu_seconds"] for row in samples) / elapsed * 100,
        "peak_allocated_bytes_including_load_and_warmup": max(
            row["peak_allocated_bytes"] for row in peak_stages
        ),
        "peak_reserved_bytes_including_load_and_warmup": max(
            row["peak_reserved_bytes"] for row in peak_stages
        ),
        "peak_allocated_bytes_measured_only": max(row["peak_allocated_bytes"] for row in samples),
        "peak_reserved_bytes_measured_only": max(row["peak_reserved_bytes"] for row in samples),
        "phase_measurements": sum(len(report.get("phases", [])) for report in reports),
    }
    return result


def check_conditions(reference: dict[str, Any], report: dict[str, Any]) -> dict[str, bool | None]:
    reference_affinity = reference["environment"].get("cpu_affinity", {}).get("effective")
    affinity = report["environment"].get("cpu_affinity", {}).get("effective")
    return {
        "completed": report["status"] == "complete",
        "same_workload": report["workload"] == reference["workload"],
        "same_token_ids": all(
            report["correctness"][key] == reference["correctness"][key]
            for key in ("input_ids", "output_ids", "attention_mask")
        ),
        "all_requests_identical": report["correctness"]["all_requests_identical"],
        "same_versions": report["environment"]["versions"] == reference["environment"]["versions"],
        "same_config_fingerprints": report["environment"]["config_fingerprints"]
        == reference["environment"]["config_fingerprints"],
        "same_gpu": report["environment"]["gpu"] == reference["environment"]["gpu"],
        "same_dtype": report["model"]["dtype"] == reference["model"]["dtype"],
        "same_device_map": report["model"]["hf_device_map"] == reference["model"]["hf_device_map"],
        "same_telemetry_setting": report["protocol"]["telemetry"]
        == reference["protocol"]["telemetry"],
        "same_effective_cpu_affinity": affinity == reference_affinity
        if reference_affinity is not None and affinity is not None
        else None,
    }


def compare(
    paths: list[Path],
    *,
    baseline_variant: str,
    optimized_variant: str,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    baseline = [report for report in reports if report["variant"] == baseline_variant]
    optimized = [report for report in reports if report["variant"] == optimized_variant]
    if not baseline or not optimized or len(baseline) + len(optimized) != len(reports):
        raise ValueError(
            "Each report must belong to exactly one requested arm, and both arms are required"
        )
    reference = baseline[0]
    ordered = []
    for path, report in zip(paths, reports, strict=True):
        checks = check_conditions(reference, report)
        if any(value is False for value in checks.values()):
            failed = [key for key, value in checks.items() if value is False]
            raise ValueError(f"Invalid comparison for {path.name}: {', '.join(failed)}")
        ordered.append(
            {
                "path": str(path),
                "variant": report["variant"],
                "started_at_utc": report["started_at_utc"],
                "source_checkout": report["environment"].get("source_checkout"),
                "git_sha": report["environment"]["git_sha"],
                "git_status": report["environment"]["git_status"],
                "checks": checks,
                "aggregate_tokens_per_second": throughput(report["samples"]),
                "request_count": len(report["samples"]),
                "raw_request_tokens_per_second": [
                    sample["tokens"] / sample["seconds"] for sample in report["samples"]
                ],
            }
        )
    base_summary = arm_summary(baseline)
    opt_summary = arm_summary(optimized)
    base_samples = [sample for report in baseline for sample in report["samples"]]
    opt_samples = [sample for report in optimized for sample in report["samples"]]
    rng = random.Random(seed)
    bootstrap = []
    for _ in range(bootstrap_iterations):
        resampled_base = rng.choices(base_samples, k=len(base_samples))
        resampled_opt = rng.choices(opt_samples, k=len(opt_samples))
        bootstrap.append((throughput(resampled_opt) / throughput(resampled_base) - 1) * 100)
    bootstrap.sort()
    improvement = (
        opt_summary["aggregate_tokens_per_second"] / base_summary["aggregate_tokens_per_second"] - 1
    ) * 100
    base_run_tps = [throughput(report["samples"]) for report in baseline]
    opt_run_tps = [throughput(report["samples"]) for report in optimized]
    return {
        "schema_version": 1,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "ordered_runs": ordered,
        "baseline": base_summary,
        "optimized": opt_summary,
        "improvement_percent": improvement,
        "bootstrap_improvement_percent_95_interval": {
            "low": percentile(bootstrap, 0.025),
            "high": percentile(bootstrap, 0.975),
            "iterations": bootstrap_iterations,
            "seed": seed,
            "method": "Percentile bootstrap; independent whole-request resampling within each arm",
            "limitation": "Conditional on observed runs. Requests within a run may be correlated; "
            "this interval does not include unobserved device-state/run-order drift.",
        },
        "run_drift": {
            "baseline_per_run_tokens_per_second": base_run_tps,
            "optimized_per_run_tokens_per_second": opt_run_tps,
            "baseline_max_over_min": max(base_run_tps) / min(base_run_tps),
            "optimized_max_over_min": max(opt_run_tps) / min(opt_run_tps),
            "pairing": "No paired statistical claim. Input execution order retained explicitly.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", help="JSON reports in execution order")
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-variant", default="baseline")
    parser.add_argument("--optimized-variant", default="production_cuda_graphs")
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    if args.bootstrap_iterations < 100:
        parser.error("Use at least 100 bootstrap iterations")
    if args.baseline_variant == args.optimized_variant:
        parser.error("Baseline and optimized variant names must differ")
    output = workspace_path(args.output)
    result = compare(
        [workspace_path(path) for path in args.reports],
        baseline_variant=args.baseline_variant,
        optimized_variant=args.optimized_variant,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    from rich.console import Console

    console = Console()
    interval = result["bootstrap_improvement_percent_95_interval"]
    console.print(f"Baseline: {result['baseline']['aggregate_tokens_per_second']:.3f} tok/s")
    console.print(f"Optimized: {result['optimized']['aggregate_tokens_per_second']:.3f} tok/s")
    console.print(
        f"Improvement: {result['improvement_percent']:+.2f}% "
        f"(request bootstrap 95% interval {interval['low']:+.2f}% to {interval['high']:+.2f}%)"
    )
    console.print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
