"""soup loop — CLI-first data flywheel (v0.58.0 capstone).

Subcommands:

    soup loop init <served-model> --eval <suite> --baseline <ref>
    soup loop status
    soup loop watch [--detach] [--max-iterations N]
    soup loop pause
    soup loop resume
    soup loop canary <new-adapter> --traffic 5% [--autoroll-on-regress]
    soup loop replay <iteration-id>

State lives in ``.soup/loop.yaml``; per-iteration artifacts under
``.soup-loops/<iteration-id>/iteration.json``. Both paths are cwd-
contained + symlink-rejected (TOCTOU defence — matches v0.33.0 / v0.43.0
/ v0.55.0 / v0.56.0 / v0.57.0 policy).
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from soup_cli.utils.canary_router import CanaryPolicy
from soup_cli.utils.loop_budget import parse_budget_string
from soup_cli.utils.loop_daemon import WatchConfig, watch
from soup_cli.utils.loop_iteration import list_iterations, read_iteration
from soup_cli.utils.loop_state import (
    LoopState,
    init_state,
    read_state,
    write_state,
)

app = typer.Typer(
    name="loop",
    help="Data flywheel: traces -> pairs -> train -> gate -> deploy.",
    no_args_is_help=True,
)
console = Console()


def _safe_read() -> LoopState:
    try:
        return read_state()
    except FileNotFoundError as exc:
        console.print(
            f"[red]No loop state found.[/] Run [bold]soup loop init[/] first."
            f"\n  detail: {escape(str(exc))}"
        )
        raise typer.Exit(code=2)
    except (ValueError, TypeError) as exc:
        console.print(f"[red]loop state invalid:[/] {escape(str(exc))}")
        raise typer.Exit(code=2)


@app.command("init")
def init_cmd(
    served_model: str = typer.Argument(..., help="Served model id (e.g. registry://abc12)."),
    eval_suite: str = typer.Option(..., "--eval", help="Eval suite path or registry ref."),
    baseline: str = typer.Option(..., "--baseline", help="Baseline registry id or file."),
    monthly_budget: Optional[str] = typer.Option(
        None, "--monthly-budget", help="Monthly USD cap (e.g. 50usd, 100)."
    ),
    max_runs_per_day: Optional[int] = typer.Option(
        None, "--max-runs-per-day", help="Cap on iteration starts per UTC day."
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite existing loop.yaml."),
) -> None:
    """Create the .soup/loop.yaml control file (one-time setup)."""
    budget_usd: Optional[float] = None
    if monthly_budget is not None:
        try:
            budget_usd = parse_budget_string(monthly_budget)
        except (TypeError, ValueError) as exc:
            console.print(f"[red]invalid --monthly-budget:[/] {escape(str(exc))}")
            raise typer.Exit(code=2)
    if max_runs_per_day is not None and (
        isinstance(max_runs_per_day, bool) or max_runs_per_day < 1
    ):
        console.print("[red]--max-runs-per-day must be a positive int[/]")
        raise typer.Exit(code=2)
    try:
        state, path = init_state(
            served_model=served_model,
            eval_suite=eval_suite,
            baseline=baseline,
            monthly_budget_usd=budget_usd,
            max_runs_per_day=max_runs_per_day,
            force=force,
        )
    except (FileExistsError, FileNotFoundError, TypeError, ValueError) as exc:
        console.print(f"[red]init failed:[/] {escape(str(exc))}")
        raise typer.Exit(code=2)
    console.print(
        Panel.fit(
            f"loop state created at [bold]{escape(os.path.relpath(path))}[/]\n"
            f"served_model: [bold]{escape(state.served_model)}[/]\n"
            f"eval_suite:   [bold]{escape(state.eval_suite)}[/]\n"
            f"baseline:     [bold]{escape(state.baseline)}[/]",
            title="soup loop init",
        )
    )


@app.command("status")
def status_cmd() -> None:
    """Show counters and current status."""
    state = _safe_read()
    table = Table(title="soup loop status", show_header=False)
    table.add_column("field", style="bold")
    table.add_column("value")
    table.add_row("status", f"[bold]{escape(state.status)}[/]")
    table.add_row("served_model", escape(state.served_model))
    table.add_row("eval_suite", escape(state.eval_suite))
    table.add_row("baseline", escape(state.baseline))
    table.add_row("traces_collected", str(state.traces_collected))
    table.add_row("pairs_distilled", str(state.pairs_distilled))
    table.add_row("runs_gated", str(state.runs_gated))
    table.add_row("adapters_shipped", str(state.adapters_shipped))
    table.add_row("iteration_count", str(state.iteration_count))
    if state.canary_active:
        table.add_row(
            "canary",
            f"{escape(state.canary_active)} @ "
            f"{state.canary_traffic_pct or 0:.1f}%",
        )
    if state.monthly_budget_usd is not None:
        table.add_row(
            "budget",
            f"${state.spent_this_month_usd:.2f} / ${state.monthly_budget_usd:.2f}",
        )
    if state.max_runs_per_day is not None:
        table.add_row(
            "runs_today",
            f"{state.runs_today} / {state.max_runs_per_day}",
        )
    console.print(table)


@app.command("pause")
def pause_cmd() -> None:
    """Pause the watch daemon at the next iteration boundary."""
    state = _safe_read()
    if state.status == "stopped":
        console.print("[yellow]loop is already stopped[/]")
        raise typer.Exit(code=0)
    state = state.with_status("paused")
    write_state(state)
    console.print("[green]loop paused[/]")


@app.command("resume")
def resume_cmd() -> None:
    """Resume a paused loop (next watch iteration picks up automatically)."""
    state = _safe_read()
    if state.status != "paused":
        console.print(f"[yellow]loop is {state.status}, not paused[/]")
        raise typer.Exit(code=0)
    state = state.with_status("running")
    write_state(state)
    console.print("[green]loop resumed[/]")


@app.command("watch")
def watch_cmd(
    foreground: bool = typer.Option(
        False,
        "--foreground",
        help="Run in foreground (default).",
    ),
    detach: bool = typer.Option(
        False,
        "--detach",
        help="Spawn a background subprocess running --foreground.",
    ),
    max_iterations: Optional[int] = typer.Option(
        None,
        "--max-iterations",
        help="Stop after N iterations (test/demo use).",
    ),
    poll_interval: float = typer.Option(
        60.0, "--poll-interval", help="Seconds between iterations [1, 3600]."
    ),
) -> None:
    """Run the harvest → train → gate → deploy daemon."""
    if detach and foreground:
        console.print("[red]--detach and --foreground are mutually exclusive[/]")
        raise typer.Exit(code=2)
    _ = _safe_read()  # ensure state exists before forking
    if detach:
        argv = [
            sys.executable,
            "-m",
            "soup_cli.cli",
            "loop",
            "watch",
            "--foreground",
            "--poll-interval",
            str(poll_interval),
        ]
        if max_iterations is not None:
            argv.extend(["--max-iterations", str(max_iterations)])
        proc = subprocess.Popen(  # noqa: S603 — argv is internal, no shell
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
        console.print(f"[green]watch detached[/] pid={proc.pid}")
        return
    try:
        cfg = WatchConfig(
            poll_interval_sec=float(poll_interval),
            max_iterations=max_iterations,
        )
    except (TypeError, ValueError) as exc:
        console.print(f"[red]invalid watch config:[/] {escape(str(exc))}")
        raise typer.Exit(code=2)
    final_state, ran = watch(cfg)
    console.print(
        f"[green]watch exited[/] iterations={ran} status={escape(final_state.status)}"
    )


@app.command("canary")
def canary_cmd(
    new_adapter: str = typer.Argument(..., help="Adapter id/path to canary."),
    traffic: str = typer.Option(
        "5%", "--traffic", help='Traffic share, e.g. "5%" or "5".'
    ),
    autoroll_on_regress: bool = typer.Option(
        True,
        "--autoroll-on-regress/--no-autoroll-on-regress",
        help="Roll back automatically on MAJOR verdict.",
    ),
) -> None:
    """Promote ``new_adapter`` as the canary with a traffic split."""
    pct = _parse_traffic(traffic)
    state = _safe_read()
    # CanaryPolicy validates name shape + cross-fields; reuse here so the
    # state file can never disagree with the live router schema.
    try:
        policy = CanaryPolicy(
            stable=state.served_model,
            canary=new_adapter,
            traffic_pct=pct,
            sticky_on_rollback=autoroll_on_regress,
        )
    except (TypeError, ValueError) as exc:
        console.print(f"[red]invalid canary policy:[/] {escape(str(exc))}")
        raise typer.Exit(code=2)
    # ``write_state`` refreshes updated_at on every persist, so we
    # explicitly route through replace() rather than mutate in place.
    new_state = replace(
        state,
        canary_active=policy.canary,
        canary_traffic_pct=policy.traffic_pct,
        canary_autoroll_on_regress=autoroll_on_regress,
    )
    write_state(new_state)
    # Reload so the in-memory value reflects the persisted updated_at.
    new_state = read_state()
    console.print(
        f"[green]canary set:[/] {escape(policy.canary or '')} @ "
        f"{policy.traffic_pct:.1f}% (autoroll={autoroll_on_regress})"
    )


@app.command("replay")
def replay_cmd(
    iteration_id: Optional[str] = typer.Argument(
        None, help="Iteration id (omit to list all)."
    ),
) -> None:
    """Replay a recorded loop iteration manifest."""
    if iteration_id is None:
        ids = list_iterations()
        if not ids:
            console.print("[yellow]no iterations recorded yet[/]")
            return
        console.print("\n".join(escape(i) for i in ids))
        return
    try:
        record = read_iteration(iteration_id)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        console.print(f"[red]replay failed:[/] {escape(str(exc))}")
        raise typer.Exit(code=2)
    table = Table(title=f"replay {escape(record.iteration_id)}", show_header=False)
    table.add_column("field", style="bold")
    table.add_column("value")
    for k, v in record.to_dict().items():
        table.add_row(escape(str(k)), escape(str(v)))
    console.print(table)


def _parse_traffic(raw: str) -> float:
    """Parse ``"5%"`` / ``"5"`` / ``"  5.5 %"`` into a percent float."""
    if not isinstance(raw, str):
        console.print(f"[red]--traffic must be a string, got {type(raw).__name__}[/]")
        raise typer.Exit(code=2)
    txt = raw.strip()
    if txt.endswith("%"):
        txt = txt[:-1].strip()
    try:
        pct = float(txt)
    except ValueError:
        console.print(f"[red]invalid --traffic:[/] {escape(raw)}")
        raise typer.Exit(code=2)
    if not (0.0 <= pct <= 100.0):
        console.print("[red]--traffic must be in [0, 100][/]")
        raise typer.Exit(code=2)
    return pct
