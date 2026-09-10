"""Benchmark for dataset validation and statistics (validate_and_stats).

Measures execution time of validate_and_stats on representative repository datasets
comparing the previous implementation with the optimized implementation.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from soup_cli.data.formats import FORMAT_SIGNATURES
from soup_cli.data.loader import load_raw_data
from soup_cli.data.validator import validate_and_stats


def previous_validate_and_stats(data: list[dict], expected_format: str | None = None) -> dict:
    """Original implementation of validate_and_stats before optimization."""
    if not data:
        return {
            "total": 0,
            "columns": [],
            "avg_length": 0,
            "min_length": 0,
            "max_length": 0,
            "empty_fields": 0,
            "duplicates": 0,
            "issues": ["Dataset is empty"],
            "valid_rows": 0,
        }

    columns = list(data[0].keys())

    # Compute text lengths (join all string values)
    lengths = []
    empty_count = 0
    for row in data:
        text = " ".join(str(v) for v in row.values() if v)
        lengths.append(len(text))
        for v in row.values():
            if v is None:
                empty_count += 1

    # Detect duplicates by stringifying rows
    row_strs = [str(sorted(row.items())) for row in data]
    dup_count = len(row_strs) - len(set(row_strs))

    # Validate format
    issues = []
    valid_rows = len(data)
    if expected_format and expected_format in FORMAT_SIGNATURES:
        required = FORMAT_SIGNATURES[expected_format]
        invalid = 0
        for row in data:
            if not required.issubset(row.keys()):
                invalid += 1
        valid_rows = len(data) - invalid
        if invalid > 0:
            issues.append(
                f"{invalid} rows missing required keys for '{expected_format}' format: {required}"
            )

    if dup_count > 0:
        issues.append(f"{dup_count} duplicate rows found")
    if empty_count > 0:
        issues.append(f"{empty_count} empty fields found")

    # Check for very short samples
    short = sum(1 for length in lengths if length < 10)
    if short > 0:
        issues.append(f"{short} samples are very short (<10 chars)")

    return {
        "total": len(data),
        "columns": columns,
        "avg_length": round(sum(lengths) / len(lengths)),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "empty_fields": empty_count,
        "duplicates": dup_count,
        "issues": issues,
        "valid_rows": valid_rows,
    }


def load_benchmark_dataset() -> list[dict]:
    """Load representative data from repository examples and fixtures."""
    repo_root = Path(__file__).resolve().parent.parent
    search_dirs = [
        repo_root / "examples" / "data",
        repo_root / "src" / "soup_cli" / "data" / "_fixtures",
    ]
    raw_rows: list[dict] = []
    for sdir in search_dirs:
        for jsonl_file in sdir.rglob("*.jsonl"):
            try:
                raw_rows.extend(load_raw_data(jsonl_file))
            except Exception:
                pass

    if not raw_rows:
        raise RuntimeError("No benchmark data files found in repository")

    # Replicate to 20,000 rows (representative of mid-sized fine-tuning datasets)
    target_size = 20000
    dataset: list[dict] = []
    while len(dataset) < target_size:
        for row in raw_rows:
            dataset.append(dict(row))
            if len(dataset) >= target_size:
                break
    return dataset


def benchmark_fn(fn: Callable[[list[dict], str | None], dict], data: list[dict], runs: int = 15) -> tuple[float, dict]:
    """Benchmark a function with warmups and return (median_time, result)."""
    for _ in range(3):
        fn(data, "alpaca")

    times: list[float] = []
    result = {}
    for _ in range(runs):
        start = time.perf_counter()
        result = fn(data, "alpaca")
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    times.sort()
    median_time = times[len(times) // 2]
    return median_time, result


def run_benchmark() -> None:
    data = load_benchmark_dataset()
    print(f"Loaded benchmark dataset: {len(data)} rows from repository fixtures")

    prev_time, prev_result = benchmark_fn(previous_validate_and_stats, data)
    print(f"Previous Implementation (median of 15 runs): {prev_time:.4f}s")

    curr_time, curr_result = benchmark_fn(validate_and_stats, data)
    print(f"Current Implementation  (median of 15 runs): {curr_time:.4f}s")

    assert prev_result == curr_result, f"Results mismatch!\nPrev: {prev_result}\nCurr: {curr_result}"
    print("Correctness check: PASS (identical outputs)")

    if prev_time > 0:
        reduction = (prev_time - curr_time) / prev_time * 100.0
        speedup = prev_time / curr_time if curr_time > 0 else float("inf")
        print(f"\nExecution Time: {prev_time:.4f}s -> {curr_time:.4f}s")
        print(f"Reduction: {reduction:.1f}%")
        print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    run_benchmark()
