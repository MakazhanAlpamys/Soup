"""Apply the pre-declared --cuda-graphs gate to the harness and CLI outputs.

    python cuda_graph_decode_verdict.py harness COMPARISON REPORT... --output VERDICT
    python cuda_graph_decode_verdict.py cli CLI_DIR --output VERDICT

The thresholds are copied from the design of 2026-09-24 and were fixed before any
run; editing them after a measurement defeats the gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MIN_THROUGHPUT_RATIO = 1.20  # (b) slowest graph run over the fastest baseline run
MAX_EXTRA_RESERVED_BYTES = 256 * 1024 * 1024  # (c)
_STRICT_CHECKS = ("completed", "same_workload", "same_token_ids", "all_requests_identical")
_CLI_ARMS = ("a1", "b1", "b2", "a2")
_RESERVED = "peak_reserved_bytes_including_load_and_warmup"


def _load(path: str | Path) -> dict:
    # utf-8-sig: Windows PowerShell 5.1 writes a BOM with -Encoding utf8.
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def harness_verdict(comparison: dict, reports: list[dict]) -> dict:
    runs = comparison["ordered_runs"]
    strict = all(run["checks"].get(name) is True for run in runs for name in _STRICT_CHECKS)
    matched = all(value is not False for run in runs for value in run["checks"].values())
    base = [report for report in reports if report["variant"] == "baseline"]
    graph = [report for report in reports if report["variant"] == "production_cuda_graphs"]
    if not base or not graph:
        raise ValueError("need at least one baseline and one production_cuda_graphs report")
    base_rates = [report["summary"]["aggregate_tokens_per_second"] for report in base]
    graph_rates = [report["summary"]["aggregate_tokens_per_second"] for report in graph]
    ratio = min(graph_rates) / max(base_rates)
    extra = max(r["summary"][_RESERVED] for r in graph) - max(r["summary"][_RESERVED] for r in base)
    return {
        "a_identical_ids_and_matched_conditions": strict and matched,
        "b_slowest_graph_over_fastest_baseline": ratio,
        "b_pass": ratio >= MIN_THROUGHPUT_RATIO,
        "c_extra_reserved_bytes": extra,
        "c_pass": extra <= MAX_EXTRA_RESERVED_BYTES,
        "baseline_rates": base_rates,
        "graph_rates": graph_rates,
    }


def cli_verdict(cli_dir: Path) -> dict:
    wall = _load(cli_dir / "cli-wall-seconds.json")
    rows = {}
    for arm in _CLI_ARMS:
        lines = (cli_dir / f"out-{arm}.jsonl").read_text(encoding="utf-8").splitlines()
        rows[arm] = [json.loads(line)["response"] for line in lines if line.strip()]
    identical = bool(rows["a1"]) and all(rows[arm] == rows["a1"] for arm in _CLI_ARMS)
    graph = [wall["b1"], wall["b2"]]
    base = [wall["a1"], wall["a2"]]
    return {
        "identical_responses": identical,
        "rows": len(rows["a1"]),
        "wall_seconds": wall,
        "d_pass": identical and max(graph) < min(base),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    modes = parser.add_subparsers(dest="mode", required=True)
    harness = modes.add_parser("harness")
    harness.add_argument("comparison")
    harness.add_argument("reports", nargs="+")
    harness.add_argument("--output", required=True)
    cli = modes.add_parser("cli")
    cli.add_argument("cli_dir")
    cli.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.mode == "harness":
        result = harness_verdict(_load(args.comparison), [_load(path) for path in args.reports])
        passed = (
            result["a_identical_ids_and_matched_conditions"] and result["b_pass"]
            and result["c_pass"]
        )
    else:
        result = cli_verdict(Path(args.cli_dir))
        passed = result["d_pass"]
    result["pass"] = passed
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
