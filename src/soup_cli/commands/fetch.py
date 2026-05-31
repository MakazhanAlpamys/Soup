"""v0.44.0 Part D — `soup fetch <namespace> <name>` example/config fetcher."""

from __future__ import annotations

import os
import shutil
import stat

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from soup_cli.utils.fetch_examples import (
    fetch_examples_dir,
    get_entry,
    list_entries,
)
from soup_cli.utils.paths import is_under_cwd

console = Console()


def fetch(
    namespace: str = typer.Argument(
        ...,
        help="One of: examples, configs, deepspeed_configs.",
    ),
    name: str = typer.Argument(
        None,
        help="Catalog entry name (omit to list).",
    ),
    output: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Destination path (default: ./<filename> in cwd).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite if --output already exists.",
    ),
) -> None:
    """Fetch a ready-to-edit example config from the bundled catalog."""
    if name is None:
        try:
            entries = list_entries(namespace)
        except ValueError as exc:
            console.print(f"[red]{escape(str(exc))}[/]")
            raise typer.Exit(code=2) from exc
        if not entries:
            console.print(f"[yellow]No entries in namespace {namespace!r}.[/]")
            return
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Name")
        table.add_column("Filename")
        table.add_column("Description")
        for entry in entries.values():
            table.add_row(
                escape(entry.name),
                escape(entry.filename),
                escape(entry.description),
            )
        console.print(table)
        return
    entry = get_entry(name)
    if entry is None or entry.namespace != namespace:
        console.print(
            f"[red]Unknown {escape(namespace)} entry: {escape(str(name))}.[/]"
        )
        raise typer.Exit(code=2)
    bundled_root = os.path.realpath(fetch_examples_dir())
    src = os.path.realpath(os.path.join(bundled_root, entry.filename))
    # Defence-in-depth: confirm the catalog entry stays inside the bundled dir.
    try:
        common = os.path.commonpath([src, bundled_root])
    except ValueError:
        common = ""
    if common != bundled_root or not os.path.isfile(src):
        console.print(
            f"[red]Bundled file is missing or escaped its root: "
            f"{escape(entry.filename)}.[/]"
        )
        raise typer.Exit(code=1)
    target_path = output or entry.filename
    if not is_under_cwd(target_path):
        console.print(
            f"[red]--output must stay under cwd: "
            f"{escape(os.path.basename(target_path))}.[/]"
        )
        raise typer.Exit(code=2)
    real_target = os.path.realpath(target_path)
    # Symlink-at-target rejection (TOCTOU defence) — matches v0.33.0 #22 /
    # v0.40.2 #51 / v0.43.0 Part C policy. Apply BEFORE the existence check
    # so a symlink-with-no-real-file can never be silently overwritten.
    try:
        link_stat = os.lstat(real_target)
    except FileNotFoundError:
        link_stat = None
    except OSError as exc:
        console.print(
            f"[red]Cannot stat target {escape(os.path.basename(real_target))}: "
            f"{escape(type(exc).__name__)}[/]"
        )
        raise typer.Exit(code=1) from exc
    if link_stat is not None and stat.S_ISLNK(link_stat.st_mode):
        console.print(
            f"[red]Refusing to overwrite symlink at "
            f"{escape(os.path.basename(real_target))}[/]"
        )
        raise typer.Exit(code=1)
    if link_stat is not None and not force:
        console.print(
            f"[red]{escape(os.path.basename(real_target))} already exists. "
            "Use --force to overwrite.[/]"
        )
        raise typer.Exit(code=1)
    parent = os.path.dirname(real_target)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    shutil.copyfile(src, real_target)
    console.print(
        f"[green]Wrote[/] {escape(real_target)}\n"
        f"[dim]{escape(entry.description)}[/]"
    )
