"""Compatibility helpers for plotext's module and figure APIs."""

from __future__ import annotations

from typing import Any

from rich.text import Text


def _v6_figure(plotext: Any) -> Any | None:
    """Return plotext 6's figure object, or ``None`` for the 5.x API."""
    figure = getattr(plotext, "figure", None)
    return figure if figure is not None and not callable(figure) else None


def render_histogram(
    plotext: Any,
    values: list[int],
    *,
    console: Any,
    bins: int,
    title: str,
    xlabel: str,
    ylabel: str,
    theme: str | None = None,
) -> None:
    """Render a histogram through Rich with either plotext 5.x or 6.x."""
    figure = _v6_figure(plotext)
    if figure is not None:
        figure.clear()
        figure.draw(figure.hist(values, bins=bins))
        figure.title(title)
        figure.label(xlabel, axis=0)
        figure.label(ylabel, axis=1)
        if theme is not None:
            figure.theme(theme)
        console.print(Text.from_ansi(str(figure.build())))
        return

    plotext.clf()
    plotext.hist(values, bins=bins)
    plotext.title(title)
    plotext.xlabel(xlabel)
    plotext.ylabel(ylabel)
    if theme is not None:
        plotext.theme(theme)
    console.print(Text.from_ansi(str(plotext.build())))


def render_line(
    plotext: Any,
    x_values: list[int],
    y_values: list[float],
    *,
    console: Any,
    label: str,
    title: str,
    xlabel: str,
    ylabel: str,
    theme: str | None = None,
) -> None:
    """Render a labelled line through Rich with either plotext 5.x or 6.x."""
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
        console.print(Text.from_ansi(str(figure.build())))
        return

    plotext.clf()
    plotext.plot(x_values, y_values, label=label)
    plotext.title(title)
    plotext.xlabel(xlabel)
    plotext.ylabel(ylabel)
    if theme is not None:
        plotext.theme(theme)
    console.print(Text.from_ansi(str(plotext.build())))
