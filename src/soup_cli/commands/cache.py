"""soup cache — list and prune stale layer-streaming cache slots (#2196)."""

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from soup_cli.utils.terminal import for_terminal

console = Console()

app = typer.Typer(no_args_is_help=True)

_MODEL = typer.Option(..., "--model", help="Base model id whose cache slots to inspect.")
_CACHE_DIR = typer.Option(None, "--cache-dir", help="Cache root (default: the layer-stream root).")


def _current(model: str, cache_dir: Optional[str]) -> str:
    from soup_cli.utils.layer_shard import shard_dir_path

    return shard_dir_path(model, cache_dir)


def _notice() -> None:
    console.print(
        "[dim]The commit a training run pins is not known here, so this command treats the "
        "bare-slug slot as current; a committed slot is protected only by the rules shown. "
        "Worst case is a re-shard.[/]"
    )


@app.command("list")
def list_slots(model: str = _MODEL, cache_dir: Optional[str] = _CACHE_DIR) -> None:
    """List every cache slot of a base, with sizes and paths."""
    from soup_cli.utils.layer_cache_slots import list_base_slots

    slots = list_base_slots(model, cache_dir, _current(model, cache_dir))
    if not slots:
        console.print("No cache slots for this base.")
        return
    table = Table(title="Layer-stream cache slots")
    for col in ("Slot", "GB", "Status", "Path"):
        table.add_column(col)
    for slot in slots:
        status = f"kept: {slot.reason}" if slot.referenced else "stale"
        table.add_row(
            for_terminal(slot.name), f"{slot.bytes / 1e9:.2f}",
            for_terminal(status), for_terminal(slot.path),
        )
        for stripe in slot.stripes:
            table.add_row(
                "  stripe", f"{stripe.bytes / 1e9:.2f}", "delete manually",
                for_terminal(stripe.path),
            )
    console.print(table)
    _notice()


@app.command("prune")
def prune(
    model: str = _MODEL,
    cache_dir: Optional[str] = _CACHE_DIR,
    yes: bool = typer.Option(False, "--yes", help="Actually remove. Without it: dry run."),
) -> None:
    """Remove unreferenced slots of a base. Dry run unless --yes."""
    from soup_cli.utils.layer_cache_slots import prune_slots

    results = prune_slots(model, cache_dir, _current(model, cache_dir), confirm=yes)
    if not results:
        console.print("Nothing to prune.")
        return
    verb = "Removed" if yes else "Would remove"
    failed = 0
    for slot, error in results:
        if error is not None:
            failed += 1
            console.print(f"[red]Failed[/] {for_terminal(slot.path)}: {for_terminal(error)}")
        else:
            console.print(f"{verb} {for_terminal(slot.path)} ({slot.bytes / 1e9:.2f} GB)")
        for stripe in slot.stripes:
            console.print(
                f"  stripe folder kept, delete manually: {for_terminal(stripe.path)} "
                f"({stripe.bytes / 1e9:.2f} GB)"
            )
    if not yes:
        console.print("Dry run. Re-run with --yes to remove.")
        _notice()
    if failed:
        raise typer.Exit(1)
