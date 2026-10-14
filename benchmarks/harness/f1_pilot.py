#!/usr/bin/env python3
"""Pilot for the F1 repeats rule: this machine's run-to-run spread (ADR 0003).

Runs ``soup bench train`` in A-B-B-A blocks with the same config for A and B,
so any difference between the arms is noise. Each run is a fresh process,
watched by ``arm_validity.ArmWatch``, and its number is the report's
``timing.median_seconds``, the median of its timed steps. The block order and
the block count come from ``arm_validity``: a block with a void run is kept on
record and replaced. When 5 blocks hold no void run, ``arm_validity.repeats``
gives sigma, the mean of A and N, the blocks a 10% effect needs on this machine.

Usage, on mains, from the directory that holds the config's data files::

    python benchmarks/harness/f1_pilot.py --config bench.yaml \\
        --output benchmarks/results/f1-pilot/pilot.json

The JSON report holds the label (default "RTX 3050, real"), the settings, sigma,
the mean of A, N and every run with its full arm record. The per-run
``soup bench train`` reports go into a ``runs`` directory next to it.

Benchmark harness code, not shipped. Needs a Soup install with a GPU torch.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ContextManager, Dict, List, Optional, Protocol, Sequence, TypedDict

from arm_validity import (
    PILOT_BLOCKS,
    ArmWatch,
    Repeats,
    Run,
    block_order,
    blocks_to_add,
    repeats,
    valid_blocks,
)
from rich.console import Console
from rich.table import Table

#: The label of the report this script saves on the RTX 3050 laptop.
DEFAULT_LABEL = "RTX 3050, real"

#: Blocks the pilot runs before it gives up on void runs: the pilot twice over.
DEFAULT_MAX_BLOCKS = 2 * PILOT_BLOCKS

console = Console()


class Watched(Protocol):
    """An arm watch once its arm has run: ``record`` holds the arm record."""

    record: Optional[Dict[str, object]]


#: Runs one arm in a fresh process and returns the median of its timed steps.
Runner = Callable[[str, int, int], float]

#: Makes the watch for one run from its arm, round and run number.
WatchFactory = Callable[..., ContextManager[Watched]]


class PilotRun(TypedDict):
    """One run of the pilot: its number, its arm record and that record's outcome."""

    arm: str
    round: int
    run: int
    value: float
    outcome: str
    record: Dict[str, object]


@dataclass
class Pilot:
    """Every block run, void ones included, and the repeats rule read from them."""

    blocks: List[List[PilotRun]]
    repeats: Repeats


def _as_runs(blocks: Sequence[Sequence[PilotRun]]) -> List[List[Run]]:
    return [[Run(run["value"], run["outcome"]) for run in block] for block in blocks]


def _run_block(
    arms: Sequence[str], first_round: int, first_run: int, runner: Runner, watch: WatchFactory
) -> List[PilotRun]:
    block: List[PilotRun] = []
    for position, arm in enumerate(arms):
        round, run = first_round + position // 2, first_run + position
        with watch(arm=arm, round=round, run=run) as watched:
            value = runner(arm, round, run)
        record = watched.record
        assert record is not None
        block.append(
            {
                "arm": arm,
                "round": round,
                "run": run,
                "value": value,
                "outcome": str(record["outcome"]),
                "record": record,
            }
        )
    return block


def run_pilot(runner: Runner, watch: WatchFactory, max_blocks: int = DEFAULT_MAX_BLOCKS) -> Pilot:
    """Run blocks until ``PILOT_BLOCKS`` of them hold no void run, then apply the repeats rule.

    A block is two rounds: A, B and then B, A. Raises RuntimeError once
    ``max_blocks`` blocks have run without enough valid ones.
    """
    blocks: List[List[PilotRun]] = []
    while len(valid := valid_blocks(_as_runs(blocks))) < PILOT_BLOCKS:
        if len(blocks) == max_blocks:
            raise RuntimeError(
                f"{len(blocks)} blocks run, {len(valid)} without a void run; "
                f"the pilot needs {PILOT_BLOCKS}"
            )
        # blocks_to_add counts the void blocks to replace; run the first of them.
        arms = block_order(blocks_to_add(_as_runs(blocks)))[0]
        first_run = len(blocks) * len(arms) + 1
        blocks.append(_run_block(arms, 2 * len(blocks) + 1, first_run, runner, watch))
    return Pilot(blocks, repeats(_as_runs(blocks)))


def save_report(pilot: Pilot, path: Path, *, label: str, settings: Dict[str, object]) -> None:
    """Write the pilot as JSON: label, settings, sigma, the mean of A, N and every run."""
    report = {
        "label": label,
        "settings": settings,
        "sigma": pilot.repeats.sigma,
        "mean_a": pilot.repeats.mean_a,
        "blocks_needed": pilot.repeats.blocks_needed,
        "blocks": pilot.blocks,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


class BenchTrain:
    """A ``Runner`` that starts ``soup bench train`` in a fresh process for each run."""

    def __init__(self, config: Path, steps: int, warmup: int, reports_dir: Path) -> None:
        self.config = config.resolve()
        self.steps = steps
        self.warmup = warmup
        self.reports_dir = reports_dir.resolve()

    def __call__(self, arm: str, round: int, run: int) -> float:
        report = self.reports_dir / f"run{run:03d}-{arm}.json"
        command = [
            sys.executable, "-m", "soup_cli.cli", "bench", "train",
            "--config", str(self.config),
            "--steps", str(self.steps),
            "--warmup", str(self.warmup),
            "--output", str(report),
        ]  # fmt: skip
        console.print(f"[dim]run {run} (arm {arm}, round {round})[/]")
        subprocess.run(command, cwd=self.config.parent, check=True, stdout=subprocess.DEVNULL)
        timing = json.loads(report.read_text(encoding="utf-8"))["timing"]
        return float(timing["median_seconds"])


def model_path(config: Path) -> str:
    """The model files the runs read: the config's ``base`` if it is a local path,
    else the Hugging Face cache that holds the downloaded model."""
    from huggingface_hub.constants import HF_HUB_CACHE

    from soup_cli.config.loader import load_config

    base = os.path.join(config.resolve().parent, load_config(config).base)
    return base if os.path.exists(base) else HF_HUB_CACHE


def print_pilot(pilot: Pilot, label: str) -> None:
    """Print every run's value and outcome, then sigma, the mean of A and N."""
    table = Table(title=f"F1 pilot ({label})")
    for column in ("block", "A", "B", "B", "A", "outcomes"):
        table.add_column(column, justify="right" if column != "outcomes" else "left")
    for index, block in enumerate(pilot.blocks, start=1):
        values = [f"{run['value']:.4f}" for run in block]
        outcomes = ", ".join(str(run["outcome"]) for run in block)
        table.add_row(str(index), *values, outcomes)
    console.print(table)
    console.print(f"sigma (pooled within-arm): {pilot.repeats.sigma:.5f} s")
    console.print(f"mean of A: {pilot.repeats.mean_a:.5f} s")
    console.print(f"N for a 10% effect: [bold]{pilot.repeats.blocks_needed}[/] blocks")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, required=True, help="soup config for A and B")
    parser.add_argument("--steps", type=int, default=12, help="optimizer steps per run")
    parser.add_argument("--warmup", type=int, default=4, help="leading steps left out")
    parser.add_argument("--output", type=Path, required=True, help="where to write the JSON")
    parser.add_argument("--label", default=DEFAULT_LABEL, help="label of the saved report")
    parser.add_argument(
        "--model-path",
        default=None,
        help="path to the model files (default: the config's local base, else the HF cache)",
    )
    parser.add_argument("--max-blocks", type=int, default=DEFAULT_MAX_BLOCKS)
    args = parser.parse_args(argv)

    models = args.model_path or model_path(args.config)
    reports_dir = args.output.parent / "runs"
    reports_dir.mkdir(parents=True, exist_ok=True)
    runner = BenchTrain(args.config, args.steps, args.warmup, reports_dir)

    def watch(*, arm: str, round: int, run: int) -> ArmWatch:
        return ArmWatch(arm=arm, round=round, run=run, label=args.label, model_path=models)

    pilot = run_pilot(runner, watch, max_blocks=args.max_blocks)
    settings: Dict[str, object] = {
        "config": str(args.config),
        "steps": args.steps,
        "warmup": args.warmup,
        "model_path": models,
        "run_value": "timing.median_seconds of soup bench train",
    }
    save_report(pilot, args.output, label=args.label, settings=settings)
    print_pilot(pilot, args.label)
    console.print(f"[dim]Report:[/] {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
