"""v0.45.0 Part A — `soup plugins` CLI."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from soup_cli import plugins as plugins_pkg

app = typer.Typer(
    name="plugins",
    help="List, enable, disable Soup plugins.",
    rich_markup_mode="rich",
    no_args_is_help=False,
    invoke_without_command=True,
)
console = Console()


@app.callback()
def _default(ctx: typer.Context) -> None:
    """When invoked with no subcommand, list registered plugins."""
    if ctx.invoked_subcommand is None:
        _show_table()


@app.command("list")
def list_cmd() -> None:
    """List all registered plugins."""
    _show_table()


@app.command("install")
def install_cmd(name: str = typer.Argument(..., help="Plugin name")) -> None:
    """Install advisory — actual installation lives in v0.45.1."""
    safe = escape(name)
    console.print(
        f"[yellow]Plugin install for [bold]{safe}[/] is advisory in v0.45.0; "
        "live install lands in v0.45.1.[/]"
    )
    console.print(
        "Drop your plugin module under [bold]soup_cli/plugins/[/] and call "
        "[bold]register_plugin(...)[/] at import time."
    )


@app.command("enable")
def enable_cmd(name: str = typer.Argument(..., help="Plugin name")) -> None:
    safe = escape(name)
    try:
        changed = plugins_pkg.enable_plugin(name)
    except KeyError:
        console.print(f"[red]Unknown plugin: {safe}[/]")
        raise typer.Exit(code=1)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(code=2)
    state = "enabled" if changed else "already enabled"
    console.print(f"[green]Plugin {safe} {state}.[/]")


@app.command("disable")
def disable_cmd(name: str = typer.Argument(..., help="Plugin name")) -> None:
    safe = escape(name)
    try:
        changed = plugins_pkg.disable_plugin(name)
    except KeyError:
        console.print(f"[red]Unknown plugin: {safe}[/]")
        raise typer.Exit(code=1)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(code=2)
    state = "disabled" if changed else "already disabled"
    console.print(f"[yellow]Plugin {safe} {state}.[/]")


def _show_table() -> None:
    plugins_view = plugins_pkg.list_plugins()
    if not plugins_view:
        console.print("[dim]No plugins registered.[/]")
        return
    table = Table(title="Soup plugins")
    table.add_column("name")
    table.add_column("version")
    table.add_column("state")
    table.add_column("hooks")
    table.add_column("description")
    for name in sorted(plugins_view):
        spec = plugins_view[name]
        hooks = sorted(plugins_pkg.discover_hooks(spec.plugin).keys())
        state = "[green]enabled[/]" if spec.enabled else "[yellow]disabled[/]"
        table.add_row(
            escape(spec.name),
            escape(spec.version),
            state,
            ", ".join(hooks) if hooks else "[dim]none[/]",
            escape(spec.description),
        )
    console.print(table)
