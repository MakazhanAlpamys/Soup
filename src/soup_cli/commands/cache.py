"""soup prune-layer-cache — remove stale layer-streaming cache slots (#2196)."""

from typing import Optional

import typer
from rich.console import Console

from soup_cli.utils.terminal import for_terminal

console = Console()

_MODEL = typer.Option(..., "--model", help="Base model id whose cache slots to inspect.")
_CACHE_DIR = typer.Option(None, "--cache-dir", help="Cache root (default: the layer-stream root).")


def _current(model: str, cache_dir: Optional[str]) -> str:
    from soup_cli.utils.layer_shard import shard_dir_path

    return shard_dir_path(model, cache_dir)


def _notice() -> None:
    console.print(
        "[dim]Kept: the newest complete slot, any slot being written or resharded, any slot "
        "without a readable index.json. The commit a run pins is not known here.[/]"
    )


def prune_layer_cache(
    model: str = _MODEL,
    cache_dir: Optional[str] = _CACHE_DIR,
    yes: bool = typer.Option(False, "--yes", help="Actually remove. Without it: dry run."),
) -> None:
    """Remove stale layer-cache slots of a base. Dry run (the listing) unless --yes."""
    from soup_cli.utils.layer_cache_slots import prune_slots

    results = prune_slots(model, cache_dir, _current(model, cache_dir), confirm=yes)
    if not results:
        console.print("Nothing to prune.")
        _notice()
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
