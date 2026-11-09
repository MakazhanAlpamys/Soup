#!/usr/bin/env python3
"""What each part of the series costs: run count, estimated GPU minutes, ``--max-hours``.

    python dry_run_parts.py                         # all three parts and the total
    python dry_run_parts.py --part part2 --rate 2.23
    python dry_run_parts.py --results <dir>         # measured wall times replace the estimates
    python dry_run_parts.py --part part1 --max-hours-only    # just the number (make_payload.sh)

The estimator is the one the budget rule uses (``orchestrate.estimate_plan``: the STEP 3 / STEP 4
timings and guesses per arm, plus a start-up per run; with ``--results`` the median measured wall
time of an arm and shape replaces its estimate, as it does while a series runs). The suggested
``--max-hours`` of a part = set-up (35 min) + all of its runs, conditional ones included + a 15
minute margin, rounded up to the next 0.25 h. Nothing is run and nothing is written.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gates  # noqa: E402
import orchestrate  # noqa: E402

Table = dict[tuple[str, str], float]


@dataclass
class PartReport:
    part: str
    phases: list[str]
    runs: int
    warmups: int
    conditional_runs: int
    gpu_minutes: float
    gpu_minutes_without_conditional: float
    suggested_max_hours: float
    groups: dict[str, float]
    rate_usd_per_hour: float | None = None
    cost_at_max_hours_usd: float | None = None
    expected_cost_usd: float | None = None


def part_report(part: str, table: Table | None, rate: float | None) -> PartReport:
    plan = orchestrate.build_plan(None, None, part)
    estimate = orchestrate.estimate_plan(plan, table)
    hours = orchestrate.suggested_max_hours(estimate["minutes"])
    report = PartReport(
        part=part,
        phases=list(orchestrate.PART_PHASES[part]),
        runs=estimate["runs"],
        warmups=estimate["warmups"],
        conditional_runs=estimate["conditional_runs"],
        gpu_minutes=round(estimate["minutes"], 1),
        gpu_minutes_without_conditional=round(estimate["minutes_unconditional"], 1),
        suggested_max_hours=hours,
        groups={key: round(value, 1) for key, value in estimate["by_group"].items()},
    )
    if rate is not None:
        expected_hours = (orchestrate.SETUP_MIN_PER_PART + estimate["minutes_unconditional"]) / 60
        report.rate_usd_per_hour = rate
        report.cost_at_max_hours_usd = round(hours * rate, 2)
        report.expected_cost_usd = round(expected_hours * rate, 2)
    return report


def render(report: PartReport) -> str:
    runs_note = f"{report.runs} runs"
    if report.warmups:
        runs_note += f" (incl. {report.warmups} warm-ups)"
    lines = [
        f"{report.part}: {', '.join(report.phases)}",
        f"  {runs_note}, estimated GPU time {report.gpu_minutes:.0f} min "
        f"({report.gpu_minutes / 60:.2f} h)",
    ]
    if report.conditional_runs:
        lines[-1] += (
            f"; {report.gpu_minutes_without_conditional:.0f} min without the "
            f"{report.conditional_runs} conditional runs"
        )
    lines.append(
        f"  suggested --max-hours {report.suggested_max_hours:.2f} (setup "
        f"{orchestrate.SETUP_MIN_PER_PART:.0f} min + runs + margin "
        f"{orchestrate.MARGIN_MIN_PER_PART:.0f} min, rounded up to 0.25 h)"
    )
    if report.rate_usd_per_hour is not None:
        lines.append(
            f"  at ${report.rate_usd_per_hour:.2f}/h: up to ${report.cost_at_max_hours_usd:.2f} "
            f"at --max-hours; about ${report.expected_cost_usd:.2f} for setup + the runs that "
            "always happen, with no margin"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--part", choices=orchestrate.PARTS, default=None, help="one part only")
    env_rate = os.environ.get("RENTAL_RATE_USD_PER_HOUR")
    parser.add_argument(
        "--rate",
        type=float,
        default=float(env_rate) if env_rate else None,
        help="USD per hour (default: RENTAL_RATE_USD_PER_HOUR); adds the cost lines",
    )
    parser.add_argument(
        "--results",
        default=None,
        help="a results folder: measured wall times replace the built-in estimates",
    )
    parser.add_argument("--json", action="store_true", help="print the numbers as json")
    parser.add_argument(
        "--max-hours-only", action="store_true", help="with --part: print only --max-hours"
    )
    args = parser.parse_args(argv)
    if args.max_hours_only and not args.part:
        parser.error("--max-hours-only needs --part")

    table = None
    if args.results:
        table = orchestrate.measured_wall_minutes(gates.load_runs(args.results))
    parts = [args.part] if args.part else list(orchestrate.PARTS)
    reports = [part_report(part, table, args.rate) for part in parts]
    if args.max_hours_only:
        print(f"{reports[0].suggested_max_hours:.2f}")
        return 0
    if args.json:
        print(json.dumps([asdict(report) for report in reports], indent=2))
        return 0

    for report in reports:
        print(render(report))
    if len(reports) > 1:
        full = orchestrate.estimate_plan(orchestrate.build_plan(), table)
        total_hours = sum(report.suggested_max_hours for report in reports)
        line = (
            f"all three parts: {sum(report.runs for report in reports)} runs, "
            f"{sum(report.gpu_minutes for report in reports):.0f} min of GPU time "
            f"(the full plan: {full['runs']} runs, {full['minutes']:.0f} min); "
            f"{total_hours:.2f} h of rental at the suggested --max-hours"
        )
        if args.rate is not None:
            line += f", up to ${total_hours * args.rate:.2f} at ${args.rate:.2f}/h"
        print(line)
        print("note: three rentals pay three set-ups; one single series pays one")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
