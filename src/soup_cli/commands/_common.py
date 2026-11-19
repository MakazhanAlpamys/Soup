"""Small shared command-line guards."""

import math
from collections.abc import Iterable
from typing import Optional

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
        console.print(f"[red]{for_terminal(exc)}[/]", emoji=False)
        raise typer.Exit(exit_code) from exc


def _describe_range(kind: object) -> str:
    """A Click range as Click words it, e.g. ``0.0<=x<=1.0``."""
    low = getattr(kind, "min", None)
    high = getattr(kind, "max", None)
    text = "x"
    if low is not None:
        text = f"{low}{'<' if getattr(kind, 'min_open', False) else '<='}{text}"
    if high is not None:
        text = f"{text}{'<' if getattr(kind, 'max_open', False) else '<='}{high}"
    return text


def refuse_nan(param: typer.CallbackParam, value: Optional[float]) -> Optional[float]:
    """Refuse NaN on a ranged float option, as Click refuses ``inf`` (#2906).

    ``typer.Option(min=..., max=...)`` on a float becomes a Click ``FloatRange``,
    which checks ``value < min`` and ``value > max``; both are false for NaN, and
    ``float()`` reads ``nan`` in any case and with a sign. Pass it as the option's
    ``callback``: it runs after Click's conversion, on the number, before the
    command reads anything, and raises Click's out-of-range usage error, so each
    command keeps its own usage exit code. ``None`` (an unset ``Optional``)
    passes through.
    """
    if value is None or not math.isnan(value):
        return value
    raise typer.BadParameter(f"nan is not in the range {_describe_range(param.type)}.")
