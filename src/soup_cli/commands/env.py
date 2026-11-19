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
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from soup_cli.utils.env_lock import (
    DEFAULT_LOCK_FILE,
    check_abi_compat,
    current_declared_bounds_check,
    read_lock,
    render_install_plan,
    snapshot_env,
    write_lock,
    write_requirements_txt,
)
from soup_cli.utils.paths import is_under_cwd
from soup_cli.utils.terminal import strip_control, terminal_text

console = Console()

# Both an ABI drift against the lock and an installed package violating Soup's
# own declared dependency bound exit with this code (#368).
DRIFT_EXIT_CODE = 3

env_app = typer.Typer(
    name="env",
    help="Hermetic env lockfile + ABI drift detection (v0.64.0).",
    no_args_is_help=True,
)


# Paths, lock-file fields, declared bounds and error text come from outside the
# program, so every line that carries them is printed as ``Text``: shown as
# typed, never read as markup or ``:name:`` emoji, control characters removed
# (#2489).


def _refuse(message: object) -> None:
    console.print(terminal_text(message, "red"))


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
        _refuse(f"output {output!r} is outside cwd")
        raise typer.Exit(2)

    try:
        lock = snapshot_env()
        write_lock(lock, output)
    except (TypeError, ValueError) as exc:
        _refuse(exc)
        raise typer.Exit(2) from exc

    console.print(
        Panel(
            Text.assemble(
                (f"Locked {len(lock.entries)} packages to {strip_control(output)}", "green"),
                f"\nPython: {strip_control(lock.python_version)} | "
                f"Platform: {strip_control(lock.platform)} | "
                f"CUDA: {strip_control(lock.cuda_version or 'none')}",
            ),
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
            Text.assemble(
                "No lock file at ",
                terminal_text(lock_path),
                "; run `soup env lock` first.",
                style="yellow",
            )
        )
        raise typer.Exit(1) from None
    except (TypeError, ValueError) as exc:
        _refuse(exc)
        raise typer.Exit(2) from exc

    header = Table(title="env status (locked)")
    header.add_column("Field")
    header.add_column("Value")
    header.add_row("soup_version", terminal_text(lock.soup_version))
    header.add_row("python_version", terminal_text(lock.python_version))
    header.add_row("platform", terminal_text(lock.platform))
    header.add_row("cuda_version", terminal_text(lock.cuda_version or "none"))
    header.add_row("created_at", terminal_text(lock.created_at))
    header.add_row("entries", str(len(lock.entries)))
    console.print(header)

    if lock.entries:
        body = Table(title="packages")
        body.add_column("Name")
        body.add_column("Version")
        body.add_column("Source")
        for e in lock.entries:
            body.add_row(terminal_text(e.name), terminal_text(e.version), terminal_text(e.source))
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
    # Audit installed packages against Soup's OWN declared bounds first — this
    # is independent of any lock file, so it catches the `pip install vllm`
    # transformers/torch downgrade even for a user who never ran `soup env lock`
    # (#368). The bound is read from package metadata, not a hardcoded copy.
    bounds = current_declared_bounds_check()

    # #368 review finding 5 — run the lock diagnostic REGARDLESS of the bounds
    # outcome, so a bounds violation no longer hides "no lock file" / ABI drift.
    # `lock_exit` records the lock half's exit code (None = clean); the bounds
    # violation takes precedence at the end but is printed after the lock line.
    lock_exit = None
    try:
        locked = read_lock(lock_path)
    except FileNotFoundError:
        console.print(
            Text.assemble(
                "No lock file at ",
                terminal_text(lock_path),
                "; run `soup env lock` first.",
                style="red",
            )
        )
        lock_exit = 1
    except (TypeError, ValueError) as exc:
        _refuse(exc)
        lock_exit = 2
    else:
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
        else:
            body = Text.assemble(
                (f"{report.drift_count} ABI-sensitive drift(s):", "red"),
                *(f"\n- {strip_control(c)}" for c in report.changes),
            )
            console.print(
                Panel(
                    body,
                    title="env check",
                    border_style="red",
                )
            )
            lock_exit = DRIFT_EXIT_CODE

    if not bounds.ok:
        report_text = Text.assemble(
            (
                f"{bounds.violation_count} package(s) violate this "
                "distribution's own declared bounds:",
                "red",
            ),
            *(
                f"\n- {strip_control(v.name)} {strip_control(v.installed)} "
                f"violates declared bound {strip_control(v.specifier)}"
                for v in bounds.violations
            ),
            "\n\n",
            (
                "A later install (e.g. `pip install vllm`) likely moved "
                "these packages outside the range soup-cli declares. Reinstall "
                "soup-cli, or pin the listed package(s) back into range.",
                "yellow",
            ),
        )
        console.print(
            Panel(
                report_text,
                title="env check",
                border_style="red",
            )
        )
        raise typer.Exit(DRIFT_EXIT_CODE)

    if lock_exit is not None:
        raise typer.Exit(lock_exit)


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
            Text.assemble(
                "No lock file at ",
                terminal_text(lock_path),
                "; run `soup env lock` first.",
                style="red",
            )
        )
        raise typer.Exit(1) from None
    except (TypeError, ValueError) as exc:
        _refuse(exc)
        raise typer.Exit(2) from exc

    try:
        plan = render_install_plan(lock, fmt=fmt)
    except (TypeError, ValueError) as exc:
        _refuse(exc)
        raise typer.Exit(2) from exc

    console.print(
        Panel(
            terminal_text(plan.rstrip("\n")),
            title=terminal_text(f"env fix — install plan ({fmt})"),
            border_style="green",
        )
    )

    if output is not None:
        if "\x00" in output:
            console.print("[red]output path must not contain null bytes[/]")
            raise typer.Exit(2)
        if not is_under_cwd(output):
            _refuse(f"output {output!r} is outside cwd")
            raise typer.Exit(2)
        try:
            write_requirements_txt(lock, output)
        except (TypeError, ValueError) as exc:
            _refuse(exc)
            raise typer.Exit(2) from exc
        console.print(Text.assemble("Wrote requirements to ", terminal_text(output), style="green"))


__all__ = ["env_app"]
