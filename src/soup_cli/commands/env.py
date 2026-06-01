"""`soup env` — hermetic env lockfile + ABI status (v0.64.0 Part C).

Sub-commands:
- ``soup env lock`` — snapshot the current env into ``soup-env.lock``.
- ``soup env status`` — print currently-locked env summary.
- ``soup env check`` — compare current env against ``soup-env.lock`` and
  report any ABI-sensitive drift (exit 3 on drift).
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from soup_cli.utils.env_lock import (
    DEFAULT_LOCK_FILE,
    check_abi_compat,
    read_lock,
    render_install_plan,
    snapshot_env,
    write_lock,
    write_requirements_txt,
)
from soup_cli.utils.paths import is_under_cwd

console = Console()

env_app = typer.Typer(
    name="env",
    help="Hermetic env lockfile + ABI drift detection (v0.64.0).",
    no_args_is_help=True,
)


@env_app.command("lock")
def env_lock_cmd(
    output: str = typer.Option(
        DEFAULT_LOCK_FILE,
        "--output",
        "-o",
        help="Path to write the lock file (default: ./soup-env.lock).",
    ),
) -> None:
    """Snapshot the current env into a lock file."""
    if "\x00" in output:
        console.print("[red]output path must not contain null bytes[/]")
        raise typer.Exit(2)
    if not is_under_cwd(output):
        console.print(f"[red]output {escape(output)!r} is outside cwd[/]")
        raise typer.Exit(2)

    try:
        lock = snapshot_env()
        write_lock(lock, output)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    console.print(
        Panel(
            f"[green]Locked {len(lock.entries)} packages to "
            f"{escape(output)}[/]\n"
            f"Python: {escape(lock.python_version)} | "
            f"Platform: {escape(lock.platform)} | "
            f"CUDA: {escape(lock.cuda_version or 'none')}",
            title="env lock",
            border_style="green",
        )
    )


@env_app.command("status")
def env_status_cmd(
    lock_path: str = typer.Option(
        DEFAULT_LOCK_FILE,
        "--lock",
        help="Path to the lock file (default: ./soup-env.lock).",
    ),
) -> None:
    """Print currently-locked env summary."""
    try:
        lock = read_lock(lock_path)
    except FileNotFoundError:
        console.print(
            f"[yellow]No lock file at {escape(lock_path)}; "
            "run `soup env lock` first.[/]"
        )
        raise typer.Exit(1) from None
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    header = Table(title="env status (locked)")
    header.add_column("Field")
    header.add_column("Value")
    header.add_row("soup_version", escape(lock.soup_version))
    header.add_row("python_version", escape(lock.python_version))
    header.add_row("platform", escape(lock.platform))
    header.add_row("cuda_version", escape(lock.cuda_version or "none"))
    header.add_row("created_at", escape(lock.created_at))
    header.add_row("entries", str(len(lock.entries)))
    console.print(header)

    if lock.entries:
        body = Table(title="packages")
        body.add_column("Name")
        body.add_column("Version")
        body.add_column("Source")
        for e in lock.entries:
            body.add_row(escape(e.name), escape(e.version), escape(e.source))
        console.print(body)


@env_app.command("check")
def env_check_cmd(
    lock_path: str = typer.Option(
        DEFAULT_LOCK_FILE,
        "--lock",
        help="Path to the lock file to compare against.",
    ),
) -> None:
    """Compare the current env against the lock and report drift."""
    try:
        locked = read_lock(lock_path)
    except FileNotFoundError:
        console.print(
            f"[red]No lock file at {escape(lock_path)}; "
            "run `soup env lock` first.[/]"
        )
        raise typer.Exit(1) from None
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    current = snapshot_env()
    report = check_abi_compat(locked, current)
    if report.ok:
        console.print(
            Panel(
                "[green]ABI-clean.[/] No drift detected.",
                title="env check",
                border_style="green",
            )
        )
        return

    body = "\n".join(f"- {escape(c)}" for c in report.changes)
    console.print(
        Panel(
            f"[red]{report.drift_count} ABI-sensitive drift(s):[/]\n{body}",
            title="env check",
            border_style="red",
        )
    )
    raise typer.Exit(3)


@env_app.command("fix")
def env_fix_cmd(
    lock_path: str = typer.Option(
        DEFAULT_LOCK_FILE,
        "--lock",
        help="Path to the lock file to render an install plan from.",
    ),
    fmt: str = typer.Option(
        "uv-pip",
        "--format",
        help="Install-plan format: uv-pip (copy/paste uv commands) | requirements.",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Optionally also write a requirements.txt to this path (under cwd).",
    ),
) -> None:
    """Render a reproducible install plan from ``soup-env.lock``.

    Print-only by design — recreating a venv is environment-dependent, so
    v0.71.1 emits the install commands for manual copy/paste (or scripting)
    rather than shelling out to a package manager (v0.71.1 #209).
    """
    try:
        lock = read_lock(lock_path)
    except FileNotFoundError:
        console.print(
            f"[red]No lock file at {escape(lock_path)}; "
            "run `soup env lock` first.[/]"
        )
        raise typer.Exit(1) from None
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    try:
        plan = render_install_plan(lock, fmt=fmt)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    console.print(
        Panel(
            escape(plan.rstrip("\n")),
            title=f"env fix — install plan ({escape(fmt)})",
            border_style="green",
        )
    )

    if output is not None:
        if "\x00" in output:
            console.print("[red]output path must not contain null bytes[/]")
            raise typer.Exit(2)
        if not is_under_cwd(output):
            console.print(f"[red]output {escape(output)!r} is outside cwd[/]")
            raise typer.Exit(2)
        try:
            write_requirements_txt(lock, output)
        except (TypeError, ValueError) as exc:
            console.print(f"[red]{escape(str(exc))}[/]")
            raise typer.Exit(2) from exc
        console.print(f"[green]Wrote requirements to {escape(output)}[/]")


__all__ = ["env_app"]
