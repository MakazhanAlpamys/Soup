"""Compatibility helpers for plotext's module and figure APIs."""

from __future__ import annotations

from typing import Any


def _v6_figure(plotext: Any) -> Any | None:
    """Return plotext 6's figure object, or ``None`` for the 5.x API."""
    figure = getattr(plotext, "figure", None)
    return figure if figure is not None and not callable(figure) else None


def render_histogram(
    plotext: Any,
    values: list[int],
    *,
    bins: int,
    title: str,
    xlabel: str,
    ylabel: str,
    theme: str,
) -> None:
    """Render a histogram with either plotext 5.x or 6.x."""
    figure = _v6_figure(plotext)
    if figure is not None:
        figure.clear()
        figure.draw(figure.hist(values, bins=bins))
        figure.title(title)
        figure.label(xlabel, axis=0)
        figure.label(ylabel, axis=1)
        figure.theme(theme)
        figure.show()
        return

    plotext.clf()
    plotext.hist(values, bins=bins)
    plotext.title(title)
    plotext.xlabel(xlabel)
    plotext.ylabel(ylabel)
    plotext.theme(theme)
    plotext.show()


def render_line(
    plotext: Any,
    x_values: list[int],
    y_values: list[float],
    *,
    label: str,
    title: str,
    xlabel: str,
    ylabel: str,
    theme: str,
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
        figure.theme(theme)
        figure.show()
        return

    plotext.clf()
    plotext.plot(x_values, y_values, label=label)
    plotext.title(title)
    plotext.xlabel(xlabel)
    plotext.ylabel(ylabel)
    plotext.theme(theme)
    plotext.show()
