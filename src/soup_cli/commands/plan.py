"""`soup plan` — Terraform-shape pre-flight summary (v0.64.0 Part B).

Renders the cost / ETA / SHA / peak-VRAM summary for a planned training
run and writes ``soup.tfstate`` for ``soup apply`` to consult.
"""

from __future__ import annotations

import datetime as _dt

import typer
import yaml
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from soup_cli.utils.paths import is_under_cwd
from soup_cli.utils.terraform_plan import (
    DEFAULT_STATE_FILE,
    TrainingState,
    build_plan,
    write_state,
)

console = Console()


def _load_yaml_config(path: str) -> dict:
    import os
    import stat as _stat

    if not isinstance(path, str):
        raise TypeError(f"config path must be str, got {type(path).__name__}")
    if "\x00" in path:
        raise ValueError("config path must not contain null bytes")
    if not is_under_cwd(path):
        raise ValueError(f"config {path!r} is outside cwd")
    # TOCTOU defence: reject symlink at the YAML path BEFORE open so a
    # pre-placed `soup.yaml -> /etc/shadow` cannot redirect the read.
    if os.path.lexists(path):
        st = os.lstat(path)
        if _stat.S_ISLNK(st.st_mode):
            raise ValueError("config path must not be a symlink")
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError("config must parse to a dict")
    return data


def plan_cmd(
    config: str = typer.Option(
        "soup.yaml", "--config", "-c", help="Path to soup.yaml."
    ),
    state_file: str = typer.Option(
        DEFAULT_STATE_FILE,
        "--state",
        help="Path to write the state file (default: ./soup.tfstate).",
    ),
) -> None:
    """Render a pre-flight training plan + write ``soup.tfstate``."""
    try:
        cfg = _load_yaml_config(config)
    except FileNotFoundError:
        console.print(f"[red]Config not found: {escape(config)}[/]")
        raise typer.Exit(1) from None
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    try:
        plan = build_plan(cfg)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    state = TrainingState(
        plan=plan,
        applied=False,
        applied_at=None,
        run_id=None,
    )
    try:
        write_state(state, state_file)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    table = Table(title="Training plan")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("base", escape(plan.base))
    table.add_row("task", escape(plan.task))
    table.add_row("config_sha", escape(plan.config_sha[:16] + "..."))
    table.add_row("dataset_sha", escape(plan.dataset_sha[:16] + "..."))
    table.add_row("estimated_cost", f"${plan.estimated_cost_usd:.4f}")
    table.add_row("estimated_minutes", f"{plan.estimated_minutes:.1f}")
    table.add_row("peak_vram_gb", f"{plan.peak_vram_gb:.1f}")
    table.add_row("spot_price_usd_per_hour", f"${plan.spot_price_usd_per_hour:.2f}")
    console.print(table)
    console.print(
        Panel(
            f"[green]Plan written to {escape(state_file)}.[/]\n"
            "Review the numbers, then run `soup apply` to execute.",
            title="plan",
            border_style="green",
        )
    )


def apply_cmd(
    config: str = typer.Option(
        "soup.yaml", "--config", "-c", help="Path to soup.yaml."
    ),
    state_file: str = typer.Option(
        DEFAULT_STATE_FILE,
        "--state",
        help="Path to the state file written by `soup plan`.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Verify drift and exit without invoking training.",
    ),
) -> None:
    """Execute the planned training run, refusing on drift (v0.64.0)."""
    from soup_cli.utils.terraform_plan import build_plan, detect_drift, read_state

    try:
        cfg = _load_yaml_config(config)
    except FileNotFoundError:
        console.print(f"[red]Config not found: {escape(config)}[/]")
        raise typer.Exit(1) from None
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    try:
        state = read_state(state_file)
    except FileNotFoundError:
        console.print(
            f"[red]No state file at {escape(state_file)}; run `soup plan` first.[/]"
        )
        raise typer.Exit(1) from None
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    try:
        plan_now = build_plan(cfg)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    drift = detect_drift(state, plan_now)
    if drift.has_drift:
        fields = ", ".join(drift.changed_fields)
        console.print(
            Panel(
                f"[red]Drift detected.[/] Plan no longer matches state: {escape(fields)}\n"
                f"Run `soup plan` again to refresh, then re-apply.",
                title="apply",
                border_style="red",
            )
        )
        raise typer.Exit(3)

    if dry_run:
        console.print(
            Panel(
                f"[green]Dry-run.[/] No drift detected; would proceed.\n"
                f"Expected cost: ${plan_now.estimated_cost_usd:.4f} | "
                f"ETA: {plan_now.estimated_minutes:.1f} min",
                title="apply",
                border_style="green",
            )
        )
        return

    # Mark state applied; the actual `soup train` invocation is left to
    # the operator. Future v0.64.1 may inline a subprocess call.
    new_state = TrainingState(
        plan=state.plan,
        applied=True,
        applied_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
        run_id=None,
    )
    try:
        write_state(new_state, state_file)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    console.print(
        Panel(
            f"[green]Plan locked in {escape(state_file)}.[/]\n"
            f"Now run: [bold]soup train --config {escape(config)}[/]\n"
            "(The live `apply -> train` subprocess handoff lands in v0.64.1.)",
            title="apply",
            border_style="green",
        )
    )


__all__ = ["apply_cmd", "plan_cmd"]
