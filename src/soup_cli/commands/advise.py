"""`soup advise` — the pre-flight decision (v0.54.0).

Top-level usage:

  soup advise data.jsonl --goal "make our chatbot better"

Subcommands:

  soup advise explain   — print the rubric / evidence trail of the last verdict
  soup advise compare   — show prior verdicts from advise_history.jsonl

The default invocation (with a positional data path) is the headline UX.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from soup_cli.utils.advise import (
    CHOICES,
    ROIEstimate,
    Verdict,
    build_verdict,
    classify_task,
    compute_dataset_profile,
    format_verdict_rubric,
    load_advise_dataset,
    synth_probe_baselines,
    synth_probe_lora_delta,
)
from soup_cli.utils.advise_history import (
    current_project_name,
    history_path,
    load_history,
    record_verdict,
    summarise_history,
)

console = Console()

app = typer.Typer(
    name="advise",
    no_args_is_help=True,
    help=(
        "Pre-flight decision: should you fine-tune, RAG, or stick with "
        "prompt engineering? Run this BEFORE you spend 8 hours on a GPU."
    ),
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _last_verdict_path() -> str:
    """Path to the per-user last-verdict scratch file used by `explain`.

    Lives under ``~/.soup/`` so concurrent `soup advise` invocations from
    different shells do NOT clobber each other's scratch via a shared
    tempdir filename (review-fix policy mirroring ``history_path()``).
    """
    return os.path.join(os.path.expanduser("~"), ".soup", "advise_last.json")


def _write_last_verdict(verdict: Verdict) -> None:
    """Persist the most recent verdict so `soup advise explain` can recall it.

    Atomic via ``tempfile.mkstemp`` + ``os.replace`` and rejects symlinks
    at the target — matches v0.33.0 #22 / v0.43.0 Part C / v0.44.0 Part B
    TOCTOU policy (code-review HIGH fix).
    """
    payload = {
        "choice": verdict.choice,
        "confidence": verdict.confidence,
        "reason": verdict.reason,
        "reverse_when": verdict.reverse_when,
        "task_category": verdict.task_category,
        "roi": {
            "prompt_eng_delta": verdict.estimated_roi.prompt_eng_delta,
            "rag_delta": verdict.estimated_roi.rag_delta,
            "sft_delta": verdict.estimated_roi.sft_delta,
            "sft_wall_clock_secs": verdict.estimated_roi.sft_wall_clock_secs,
            "sft_cost_usd": verdict.estimated_roi.sft_cost_usd,
        },
    }
    path = _last_verdict_path()
    # Symlink rejection on the RAW target path BEFORE any write (TOCTOU).
    try:
        if stat.S_ISLNK(os.lstat(path).st_mode):
            return
    except FileNotFoundError:
        pass
    parent = os.path.dirname(path)
    try:
        if parent:
            os.makedirs(parent, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=parent or None, prefix=".advise_last_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        os.replace(tmp, path)
    except OSError:
        # Non-fatal — explain just won't have anything to recall.
        pass


def _read_last_verdict() -> Optional[Verdict]:
    path = _last_verdict_path()
    # Symlink rejection on the raw path BEFORE open (security-review MEDIUM
    # fix — matches v0.53.7 #106 TOCTOU policy). A planted symlink to e.g.
    # /etc/passwd would otherwise leak its contents through json.load below.
    try:
        if stat.S_ISLNK(os.lstat(path).st_mode):
            return None
    except FileNotFoundError:
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    roi_data = payload.get("roi") or {}
    try:
        roi = ROIEstimate(
            prompt_eng_delta=roi_data.get("prompt_eng_delta"),
            rag_delta=roi_data.get("rag_delta"),
            sft_delta=roi_data.get("sft_delta"),
            sft_wall_clock_secs=roi_data.get("sft_wall_clock_secs"),
            sft_cost_usd=roi_data.get("sft_cost_usd"),
        )
        return Verdict(
            choice=str(payload.get("choice")),
            confidence=float(payload.get("confidence", 0.0)),
            reason=str(payload.get("reason", "")),
            reverse_when=str(payload.get("reverse_when", "")),
            task_category=str(payload.get("task_category", "")),
            estimated_roi=roi,
        )
    except (TypeError, ValueError):
        return None


def _render_verdict_panel(verdict: Verdict, *, profile_row_count: int) -> Panel:
    body = (
        f"[bold cyan]Choice:[/]      [bold]{escape(verdict.choice)}[/]\n"
        f"[bold cyan]Confidence:[/]  [bold]{verdict.confidence:.2f}[/]\n"
        f"[bold cyan]Task:[/]        {escape(verdict.task_category)}\n"
        f"[bold cyan]Rows:[/]        {profile_row_count}\n\n"
        f"[bold]Why:[/]\n  {escape(verdict.reason)}\n\n"
        f"[bold]Flip when:[/]\n  {escape(verdict.reverse_when)}"
    )
    return Panel(body, title="[bold green]soup advise — verdict[/]")


def _render_roi_table(roi: ROIEstimate) -> Table:
    table = Table(title="ROI deltas (probe)")
    table.add_column("Path", style="bold cyan")
    table.add_column("Delta", justify="right")
    table.add_column("Notes", style="dim")
    rows = [
        ("prompt_eng", roi.prompt_eng_delta, "Zero-shot + few-shot baseline"),
        ("rag", roi.rag_delta, "Retrieval-augmented baseline"),
        ("sft", roi.sft_delta, "100-step LoRA probe"),
    ]
    for label, val, note in rows:
        if val is None:
            table.add_row(label, "—", escape(note))
        else:
            sign = "+" if val >= 0 else ""
            table.add_row(label, f"{sign}{val:.3f}", escape(note))
    if roi.sft_wall_clock_secs is not None:
        table.add_row(
            "sft ETA", f"{roi.sft_wall_clock_secs:.0f}s",
            "Estimated full-run wall clock",
        )
    return table


# ---------------------------------------------------------------------------
# Default (callback): soup advise <data> --goal <s>
# ---------------------------------------------------------------------------

@app.command(name="run")
def advise_run(
    data: str = typer.Argument(
        ...,
        help="Path to JSONL dataset to advise on (must stay under cwd).",
    ),
    goal: Optional[str] = typer.Option(
        None,
        "--goal",
        "-g",
        help=(
            "One-line goal statement: e.g. \"make our chatbot more concise\". "
            "Sharpens task classification."
        ),
    ),
    probe: bool = typer.Option(
        False,
        "--probe",
        help=(
            "Also run a 10-minute ROI probe (zero/few-shot + RAG baseline + "
            "100-step LoRA). Heuristic stubs in v0.54.0; live in v0.54.1."
        ),
    ),
    record: bool = typer.Option(
        False,
        "--record",
        help=(
            "Append this verdict to advise_history.jsonl with accepted=True. "
            "Use `--no-record` to skip. Stored under ~/.soup/."
        ),
    ),
    notes: str = typer.Option(
        "",
        "--notes",
        help="Optional notes attached to the recorded verdict (<=4096 chars).",
    ),
) -> None:
    """Render a Verdict for the supplied dataset.

    Default subcommand: `soup advise <data>` is rewritten to
    `soup advise run <data>` by the top-level CLI dispatcher.
    """
    try:
        rows = load_advise_dataset(data)
    except (ValueError, TypeError, FileNotFoundError) as exc:
        console.print(f"[red]Dataset error:[/] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    try:
        task_category = classify_task(rows, goal=goal)
        profile = compute_dataset_profile(rows)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]Analysis failed:[/] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    roi = ROIEstimate()
    if probe:
        try:
            baselines = synth_probe_baselines(rows)
            sft_delta, wall_clock = synth_probe_lora_delta(rows)
        except (TypeError, ValueError) as exc:
            console.print(f"[red]Probe failed:[/] {escape(str(exc))}")
            raise typer.Exit(1) from exc
        # Pick the best of zero-shot vs few-shot as the "prompt_eng" baseline.
        prompt_eng = max(baselines["zero_shot"], baselines["few_shot"])
        roi = ROIEstimate(
            prompt_eng_delta=prompt_eng,
            rag_delta=baselines["rag"],
            sft_delta=sft_delta,
            sft_wall_clock_secs=wall_clock,
        )

    # v0.71.5 #163 — bias the rubric by this project's past accepted-verdict
    # outcomes. Best-effort: a missing / unreadable history must never block
    # a verdict, so any failure falls back to the un-biased rubric.
    history = None
    project = None
    try:
        history = load_history(limit=20)
        project = current_project_name()
    except (TypeError, ValueError, OSError):
        history = None
        project = None

    try:
        verdict = build_verdict(
            profile,
            task_category,
            goal=goal,
            roi=roi,
            history=history,
            project=project,
        )
    except (TypeError, ValueError) as exc:
        console.print(f"[red]Verdict build failed:[/] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    console.print(_render_verdict_panel(verdict, profile_row_count=profile.row_count))
    if probe:
        console.print(_render_roi_table(roi))

    _write_last_verdict(verdict)

    if record:
        try:
            record_verdict(verdict, accepted=True, notes=notes)
            console.print(
                f"[dim]Recorded to {escape(history_path())}[/]"
            )
        except (TypeError, ValueError, OSError) as exc:
            console.print(
                f"[yellow]History record failed:[/] {escape(str(exc))}"
            )

    console.print(
        "[dim]Next: `soup advise explain` for the rubric, "
        "or `soup advise compare` for past verdicts.[/]"
    )


# ---------------------------------------------------------------------------
# `soup advise explain`
# ---------------------------------------------------------------------------

@app.command()
def explain() -> None:
    """Print the rubric, weights, and evidence trail of the last verdict."""
    verdict = _read_last_verdict()
    if verdict is None:
        console.print(
            "[yellow]No prior verdict cached.[/] Run `soup advise <data>` "
            "first."
        )
        raise typer.Exit(1)
    if verdict.choice not in CHOICES:
        console.print(
            "[red]Cached verdict is malformed[/] — re-run `soup advise`."
        )
        raise typer.Exit(1)
    console.print(escape(format_verdict_rubric(verdict)))


# ---------------------------------------------------------------------------
# `soup advise compare`
# ---------------------------------------------------------------------------

@app.command()
def compare(
    limit: int = typer.Option(
        20,
        "--limit",
        min=1,
        max=1000,
        help="How many recent verdicts to show.",
    ),
) -> None:
    """Show prior verdicts from advise_history.jsonl."""
    try:
        entries = load_history(limit=limit)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]History error:[/] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    if not entries:
        console.print(
            "[yellow]No history yet.[/] Run `soup advise <data> --record` "
            "to start tracking decisions."
        )
        return

    summary = summarise_history(entries)
    summary_line = "  ".join(
        f"[cyan]{escape(choice)}[/]: [bold]{count}[/]"
        for choice, count in summary.items()
    )
    console.print(f"[bold]Recent verdicts[/]    {summary_line}\n")

    table = Table(title="Advise history (newest first)")
    table.add_column("When", style="dim")
    table.add_column("Project", style="cyan")
    table.add_column("Choice", style="bold magenta")
    table.add_column("Task", style="green")
    table.add_column("Conf", justify="right")
    table.add_column("Accepted", justify="center")
    table.add_column("Outcome", justify="right")
    for entry in entries:
        when_short = entry.timestamp[:16] if entry.timestamp else "—"
        outcome_render = (
            f"{entry.outcome:+.2f}" if entry.outcome is not None else "—"
        )
        table.add_row(
            escape(when_short),
            escape(entry.project),
            escape(entry.choice),
            escape(entry.task_category),
            f"{entry.confidence:.2f}",
            "[green]y[/]" if entry.accepted else "[red]n[/]",
            outcome_render,
        )
    console.print(table)
