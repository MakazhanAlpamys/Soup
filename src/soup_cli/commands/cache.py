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
        " Use --keep <commit> for each pinned revision to protect explicitly."
    )


def prune_layer_cache(
    model: str = _MODEL,
    cache_dir: Optional[str] = _CACHE_DIR,
    yes: bool = typer.Option(False, "--yes", help="Actually remove. Without it: dry run."),
    keep: Optional[list[str]] = typer.Option(
        None, "--keep", help="Protect a lowercase 40-hex commit. Repeat up to 64 times."
    ),
) -> None:
    """Remove stale layer-cache slots of a base. Dry run (the listing) unless --yes."""
    from soup_cli.utils.layer_cache_slots import prune_slots
    from soup_cli.utils.run_provenance import validate_identity

    try:
        if len(keep or []) > 64:
            raise ValueError("at most 64 values are allowed")
        for revision in keep or []:
            validate_identity(revision, "hub_commit")
    except ValueError as exc:
        console.print(f"Invalid --keep: {for_terminal(str(exc))}", emoji=False)
        raise typer.Exit(1) from exc
    revisions = set(keep or [])
    try:
        results = prune_slots(
            model, cache_dir, _current(model, cache_dir), confirm=yes,
            **({"keep_revisions": revisions} if revisions else {}),
        )
    except ValueError as exc:
        console.print(for_terminal(str(exc)), emoji=False)
        raise typer.Exit(1) from exc
    if not results:
        console.print("Nothing to prune.")
        _notice()
        return
    verb = "Removed" if yes else "Would remove"
    failed = 0
    for slot, error in results:
        if error is not None:
            failed += 1
            console.print(
                f"[red]Failed[/] {for_terminal(slot.path)}: {for_terminal(error)}",
                soft_wrap=True,
                emoji=False,
            )
        else:
            console.print(
                f"{verb} {for_terminal(slot.path)} ({slot.bytes / 1e9:.2f} GB)",
                soft_wrap=True,
                emoji=False,
            )
        for stripe in slot.stripes:
            console.print(
                f"  stripe folder kept, delete manually: {for_terminal(stripe.path)} "
                f"({stripe.bytes / 1e9:.2f} GB)",
                soft_wrap=True,
                emoji=False,
            )
    if not yes:
        console.print("Dry run. Re-run with --yes to remove.")
        _notice()
    if failed:
        raise typer.Exit(1)
