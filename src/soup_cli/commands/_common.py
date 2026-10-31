"""Small shared command-line guards."""

from collections.abc import Iterable

import typer
from rich.console import Console

from soup_cli.utils.paths import ensure_distinct_paths
from soup_cli.utils.terminal import for_terminal


def refuse_output_over_input(
    inputs: Iterable[tuple[str, str | None]],
    outputs: Iterable[tuple[str, str | None]],
    *,
    console: Console,
    exit_code: int = 1,
) -> None:
    """Refuse a colliding output before reading or writing, preserving the caller's exit code.

    The path utility owns identity checks (real paths, Windows spellings and
    links). Optional outputs are omitted; every refusal uses terminal-safe text.
    """
    try:
        ensure_distinct_paths(
            [(label, path) for label, path in inputs if path],
            [(label, path) for label, path in outputs if path],
        )
    except ValueError as exc:
        console.print(f"[red]{for_terminal(exc)}[/]")
        raise typer.Exit(exit_code) from exc
