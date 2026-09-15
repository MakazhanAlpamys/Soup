"""soup rewind — name the dataset rows behind a training loss spike."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

console = Console()

_SEVERITY_COLOR = {"critical": "red", "warning": "yellow"}

_FINGERPRINT_MISMATCH = (
    "[yellow]Preview withheld: the dataset on disk differs from the one this "
    "run trained on (fingerprint mismatch).[/]"
)


def _fmt_loss(value: float) -> str:
    """A loss for the table: four decimals, or the bare word for a non-finite."""
    if value != value:  # NaN
        return "nan"
    if value in (float("inf"), float("-inf")):
        return "inf" if value > 0 else "-inf"
    return f"{value:.4f}"


def _fmt_ratio(value: float) -> str:
    if value != value or value in (float("inf"), float("-inf")):
        return "inf"
    return f"{value:.1f}"


def rewind(
    run_id: Optional[str] = typer.Argument(
        None,
        help="Run ID (or prefix). Defaults to the most recent run.",
    ),
    step: Optional[int] = typer.Option(
        None,
        "--step",
        help="Detail this step instead of the worst spike.",
    ),
    top: int = typer.Option(
        10,
        "--top",
        help="How many rows to list for the detailed step.",
    ),
    json_path: Optional[Path] = typer.Option(
        None,
        "--json",
        help="Write the spikes and the ranked rows to this path as JSON.",
    ),
    no_preview: bool = typer.Option(
        False,
        "--no-preview",
        help="Skip the row previews (does not load the dataset).",
    ),
):
    """Name the dataset rows behind a loss spike, from the training flight recorder.

    Reads ``<output>/rewind.jsonl`` — written during SFT training when
    ``training.rewind_log`` is on — finds the steps whose loss jumped, and
    ranks the rows in the worst one by how much of that step's loss they
    carried.
    """
    from soup_cli.experiment.tracker import ExperimentTracker
    from soup_cli.monitoring.rewind_log import RewindLog, RewindLogError, read_rewind_log
    from soup_cli.utils import rewind as rewind_utils

    tracker = ExperimentTracker()
    if run_id is None:
        runs = tracker.list_runs(limit=1)
        if not runs:
            console.print("[yellow]No runs found yet. Train a model first.[/]")
            raise typer.Exit(1)
        target = runs[0]
    else:
        target = tracker.get_run(run_id)
        if target is None:
            console.print(f"[red]Run not found:[/] {escape(run_id)}")
            raise typer.Exit(1)

    output_dir = target.get("output_dir")
    if not output_dir:
        console.print("[red]Run has no output_dir recorded[/]")
        raise typer.Exit(1)

    path = Path(output_dir) / RewindLog.FILENAME
    if not path.exists():
        console.print(
            f"[red]No rewind log at[/] {escape(str(path))}\n"
            "Three things cause this: the run set [bold]training.rewind_log: false[/], "
            "it was not an SFT run (the recorder is SFT-only), or it predates the "
            "flight recorder."
        )
        raise typer.Exit(1)

    try:
        run = read_rewind_log(path)
    except RewindLogError as exc:
        console.print(f"[red]Cannot read rewind log:[/] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    recorded = run.steps()
    console.print(
        Panel(
            f"Run: [bold]{escape(str(target['run_id']))}[/]\n"
            f"Backend: {escape(str(run.header.get('backend') or '-'))} | "
            f"Task: {escape(str(run.header.get('task') or '-'))} | "
            f"Steps recorded: {len(recorded)}",
            title="soup rewind",
        )
    )

    spikes = rewind_utils.find_spikes(run)

    if step is None:
        if not spikes:
            console.print("[green]No loss spikes recorded.[/]")
            return
        table = Table(title="Loss spikes")
        table.add_column("Severity", style="bold", no_wrap=True)
        table.add_column("Step", justify="right", no_wrap=True)
        table.add_column("Loss", justify="right", no_wrap=True)
        table.add_column("Baseline", justify="right", no_wrap=True)
        table.add_column("×", justify="right", no_wrap=True)
        for spike in spikes:
            color = _SEVERITY_COLOR.get(spike.severity, "white")
            table.add_row(
                f"[{color}]{spike.severity}[/]",
                str(spike.step),
                _fmt_loss(spike.loss),
                _fmt_loss(spike.baseline),
                _fmt_ratio(spike.ratio),
            )
        console.print(table)
        detail_step = spikes[0].step
    else:
        detail_step = step
        if not run.batches_for(detail_step):
            console.print(
                f"[red]No records for step {detail_step}[/] in {escape(str(path))}"
            )
            raise typer.Exit(1)

    rows = rewind_utils.rank_rows(run, detail_step)[:top]
    previews = None
    if not no_preview and rows:
        previews = _row_previews(target, run, rows, rewind_utils)

    table = Table(title=f"Step {detail_step} rows")
    table.add_column("Row", justify="right", no_wrap=True)
    table.add_column("Loss", justify="right", no_wrap=True)
    table.add_column("Tokens", justify="right", no_wrap=True)
    table.add_column("Share", justify="right", no_wrap=True)
    if previews is not None:
        table.add_column("Preview")
    for ranked in rows:
        cells = [
            str(ranked.row),
            _fmt_loss(ranked.loss),
            f"{ranked.tokens:,}",
            f"{ranked.share:.1%}",
        ]
        if previews is not None:
            cells.append(escape(previews.get(ranked.row, "")))
        table.add_row(*cells)
    console.print(table)

    if rows:
        worst = rows[0]
        console.print(
            f"row {worst.row}: {worst.share:.0%} of step {detail_step}'s loss, "
            f"{worst.tokens:,} tokens"
        )
        if worst.share >= 0.5:
            console.print("Inspect or drop this row, then retrain.")

    if json_path is not None:
        payload = {
            "run_id": target["run_id"],
            "step": detail_step,
            "spikes": [asdict(s) for s in spikes],
            "rows": [asdict(r) for r in rows],
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[dim]Wrote {escape(str(json_path))}[/]")


def _row_previews(target, run, rows, rewind_utils) -> dict[int, str] | None:
    """Preview text for `rows`, or None when there is none to show.

    Every way this can come up short is a note rather than a failure: the
    ranking is the answer, the previews only save a trip to the dataset.
    """
    raw_config = target.get("config_json")
    config = None
    if raw_config:
        try:
            config = json.loads(raw_config)
        except (json.JSONDecodeError, TypeError):
            config = None
    if not isinstance(config, dict):
        console.print("[dim]No usable config recorded for this run — previews skipped.[/]")
        return None

    try:
        previews = rewind_utils.load_row_previews(
            config,
            [r.row for r in rows],
            expected_fingerprint=run.header["dataset_fingerprint"],
        )
    except (ValueError, TypeError, KeyError, OSError) as exc:
        console.print(f"[dim]Could not rebuild the dataset for previews: {escape(str(exc))}[/]")
        return None

    if previews is None:
        console.print(_FINGERPRINT_MISMATCH)
    return previews
