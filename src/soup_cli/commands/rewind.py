"""soup rewind — name the dataset rows behind a training loss spike."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from soup_cli.utils.terminal import strip_control

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


def _json_record(record: dict) -> dict:
    """One payload record with every non-finite float replaced by ``null``.

    ``json.dumps`` writes NaN and Infinity as bare tokens that strict parsers
    reject, and a run whose loss went non-finite is exactly the run ``soup
    rewind`` exists for. The record keeps the information that the value was
    non-finite with ``non_finite: true``, so a ``null`` here is not confused
    with a field that was never measured (#1818).
    """
    out: dict = {}
    non_finite = False
    for key, value in record.items():
        if isinstance(value, float) and not math.isfinite(value):
            out[key] = None
            non_finite = True
        else:
            out[key] = value
    if non_finite:
        out["non_finite"] = True
    return out


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
        min=1,
        help="How many rows to list for the detailed step (at least 1).",
    ),
    json_path: Optional[str] = typer.Option(
        None,
        "--json",
        help="Write the spikes and the ranked rows to this path as JSON "
        "(must stay under the current directory).",
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
    from soup_cli.experiment.tracker import ExperimentTracker, run_lookup_failure
    from soup_cli.monitoring.rewind_log import RewindLog, RewindLogError, read_rewind_log
    from soup_cli.utils import rewind as rewind_utils
    from soup_cli.utils.paths import check_report_output_path

    # The destination is validated first, before the tracker is opened or the log is
    # read, so a refused path fails at once instead of after the analysis has run.
    if json_path is not None:
        try:
            check_report_output_path(json_path, "--json")
        except (TypeError, ValueError) as exc:
            console.print(Text.assemble(("Invalid output:", "red"), f" {strip_control(exc)}"))
            raise typer.Exit(2) from exc

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
            console.print(Text(run_lookup_failure(tracker, run_id, markup=False), style="red"))
            raise typer.Exit(1)

    candidates = _log_candidates(target, RewindLog.FILENAME)
    if not candidates:
        console.print("[red]Run has no output_dir recorded[/]")
        raise typer.Exit(1)

    path = next((c for c in candidates if c.exists()), candidates[0])
    if not path.exists():
        console.print(
            Text.assemble(
                ("No rewind log at", "red"),
                f" {strip_control(path)}\nThree things cause this: the run set ",
                ("training.rewind_log: false", "bold"),
                ", it was not an SFT run (the recorder is SFT-only), or it predates the "
                "flight recorder.",
            )
        )
        raise typer.Exit(1)

    try:
        run = read_rewind_log(path)
    except RewindLogError as exc:
        console.print(Text.assemble(("Cannot read rewind log:", "red"), f" {strip_control(exc)}"))
        raise typer.Exit(1) from exc

    recorded = run.steps()
    console.print(
        Panel(
            Text.assemble(
                "Run: ",
                (strip_control(target["run_id"]), "bold"),
                f"\nBackend: {strip_control(run.header.get('backend') or '-')} | "
                f"Task: {strip_control(run.header.get('task') or '-')} | "
                f"Steps recorded: {len(recorded)}",
            ),
            title="soup rewind",
        )
    )

    if not run.batches:
        # A header with no batches means the recorder was switched off after it
        # opened the file (packing / padding-free, a resumed run, or a failure
        # it reported at the time). Running the spike detector over nothing and
        # printing "No loss spikes recorded" in green reads as "your run was
        # clean" -- a confident answer from an empty file is worse than none.
        console.print(
            Text.assemble(
                ("No micro-batches were recorded", "yellow"),
                f" in {strip_control(path)} — the file holds only its header.\n"
                "The recorder was switched off during the run; look for a ",
                ("Rewind log off", "bold"),
                " or ",
                ("Rewind recorder disabled", "bold"),
                " line in its output. This says nothing about whether the run was clean.",
            )
        )
        raise typer.Exit(1)

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
                Text.assemble(
                    (f"No records for step {detail_step}", "red"),
                    f" in {strip_control(path)}",
                )
            )
            raise typer.Exit(1)

    ranked_rows = rewind_utils.rank_rows(run, detail_step)
    non_finite = sum(1 for ranked in ranked_rows if not math.isfinite(ranked.loss))
    rows = ranked_rows[:top]
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
        cells: list[str | Text] = [
            str(ranked.row),
            _fmt_loss(ranked.loss),
            f"{ranked.tokens:,}",
            f"{ranked.share:.1%}",
        ]
        if previews is not None:
            cells.append(Text(strip_control(previews.get(ranked.row, ""))))
        table.add_row(*cells)
    console.print(table)

    if rows:
        worst = rows[0]
        if not math.isfinite(worst.loss):
            # rank_rows lists non-finite rows first. A non-finite loss carries no
            # share, so the "N% of the step's loss" headline would name a finite
            # row and send the reader to the wrong one.
            console.print(
                f"row {worst.row}: loss is non-finite ({_fmt_loss(worst.loss)}) in step "
                f"{detail_step}, {worst.tokens:,} tokens"
            )
            if non_finite > 1:
                console.print(
                    f"{non_finite} rows in this step are non-finite; the table lists "
                    "them first."
                )
            console.print("Inspect or drop this row, then retrain.")
        else:
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
            "spikes": [_json_record(asdict(s)) for s in spikes],
            "rows": [_json_record(asdict(r)) for r in rows],
        }
        from soup_cli.utils.paths import atomic_write_text

        try:
            atomic_write_text(
                json.dumps(payload, indent=2, allow_nan=False),
                json_path,
                prefix=".rewind.",
                suffix=".json.tmp",
                field="--json",
            )
        except (OSError, ValueError) as exc:
            console.print(
                Text.assemble(("Cannot write the JSON:", "red"), f" {strip_control(exc)}")
            )
            raise typer.Exit(1) from exc
        console.print(Text(f"Wrote {strip_control(json_path)}", style="dim"))


def _log_candidates(target: dict, filename: str) -> list[Path]:
    """Where this run's rewind log can be, most specific first.

    The tracker records ``output_dir`` only when a run FINISHES, and a crashed run
    is exactly the one ``soup rewind`` exists for. So the recorded directory comes
    first, then the run's own config: ``output/experiment_name`` (the transformers
    layout) and ``output`` (MLX). A relative ``output`` resolves against the current
    directory, as it did for ``soup train``. The recorded directory is absolute for a
    run finished since #2725; an older row may hold it relative, and it resolves
    against the current directory the same way.
    """
    dirs: list[Path] = []
    if target.get("output_dir"):
        dirs.append(Path(target["output_dir"]))
    try:
        config = json.loads(target.get("config_json") or "{}")
    except (json.JSONDecodeError, TypeError):
        config = {}
    output = config.get("output") if isinstance(config, dict) else None
    if isinstance(output, str) and output:
        experiment = config.get("experiment_name")
        if isinstance(experiment, str) and experiment:
            dirs.append(Path(output) / experiment)
        dirs.append(Path(output))
    seen: list[Path] = []
    for d in dirs:
        candidate = d / filename
        if candidate not in seen:
            seen.append(candidate)
    return seen


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
        console.print(
            Text(f"Could not rebuild the dataset for previews: {strip_control(exc)}", style="dim")
        )
        return None

    if previews is None:
        console.print(_FINGERPRINT_MISMATCH)
    return previews
