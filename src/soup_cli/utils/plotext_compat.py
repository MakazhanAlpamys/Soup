"""Compatibility helpers for plotext's module and figure APIs."""

from __future__ import annotations

import os
import re
from typing import Any


def _v6_figure(plotext: Any) -> Any | None:
    """Return plotext 6's figure object, or ``None`` for the 5.x API."""
    figure = getattr(plotext, "figure", None)
    return figure if figure is not None and not callable(figure) else None


def _print_chart(plotext: Any, raw: str, console: Any = None) -> None:
    """Output a rendered chart via the Rich console, stripping escapes when uncolored."""
    from rich.console import Console
    from rich.text import Text

    cons = console if console is not None else Console()
    if "NO_COLOR" in os.environ or cons.no_color or not cons.is_terminal:
        clean = getattr(plotext, "uncolorize", None)
        if callable(clean):
            text_str = clean(raw)
        else:
            text_str = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", raw)
        cons.print(Text(text_str))
    else:
        cons.print(Text.from_ansi(raw))


def render_histogram(
    plotext: Any,
    values: list[int],
    *,
    bins: int,
    title: str,
    xlabel: str,
    ylabel: str,
    theme: str | None = None,
    console: Any = None,
) -> None:
    """Render a histogram with either plotext 5.x or 6.x."""
    figure = _v6_figure(plotext)
    if figure is not None:
        figure.clear()
        figure.draw(figure.hist(values, bins=bins))
        figure.title(title)
        figure.label(xlabel, axis=0)
        figure.label(ylabel, axis=1)
        if theme is not None:
            figure.theme(theme)
        raw = figure.build()
    else:
        plotext.clf()
        plotext.hist(values, bins=bins)
        plotext.title(title)
        plotext.xlabel(xlabel)
        plotext.ylabel(ylabel)
        if theme is not None:
            plotext.theme(theme)
        raw = plotext.build()

    _print_chart(plotext, raw, console=console)


def render_line(
    plotext: Any,
    x_values: list[int],
    y_values: list[float],
    *,
    label: str,
    title: str,
    xlabel: str,
    ylabel: str,
    theme: str | None = None,
    console: Any = None,
) -> None:
    """Render a labelled line with either plotext 5.x or 6.x."""
    figure = _v6_figure(plotext)
    if figure is not None:
        figure.clear()
        signal = figure.signal(x_values, y_values).lines().label(label)
        figure.draw(signal)
        figure.title(title)
        figure.label(xlabel, axis=0)
        figure.label(ylabel, axis=1)
        if theme is not None:
            figure.theme(theme)
        raw = figure.build()
    else:
        plotext.clf()
        plotext.plot(x_values, y_values, label=label)
        plotext.title(title)
        plotext.xlabel(xlabel)
        plotext.ylabel(ylabel)
        if theme is not None:
            plotext.theme(theme)
        raw = plotext.build()

    _print_chart(plotext, raw, console=console)
