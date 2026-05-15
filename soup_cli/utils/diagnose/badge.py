"""SVG badge renderer for `soup diagnose --badge` (v0.56.0).

Renders a 6-cell mini scorecard + overall verdict pill. Pure-string SVG
(no external deps) — embeddable in model cards and Twitter previews.
All user-controlled text is HTML-escaped to prevent SVG injection.
"""

from __future__ import annotations

from html import escape

from soup_cli.utils.diagnose.report import FAILURE_MODES, FailureReport

_VERDICT_COLOUR = {
    "OK": "#3fb950",       # green
    "MINOR": "#d29922",    # amber
    "MAJOR": "#f85149",    # red
}


def _safe(text: str, *, max_len: int = 64) -> str:
    """Escape + truncate user-controlled text for safe SVG embedding."""
    if not isinstance(text, str):
        text = str(text)
    if "\x00" in text:
        text = text.replace("\x00", "")
    if len(text) > max_len:
        text = text[: max_len - 1] + "…"
    return escape(text, quote=True)


def render_badge_svg(report: FailureReport) -> str:
    """Render a compact diagnose badge as a self-contained SVG string."""
    if not isinstance(report, FailureReport):
        raise TypeError("report must be FailureReport")
    overall = report.overall
    overall_colour = _VERDICT_COLOUR.get(overall, "#8b949e")
    cells = []
    for index, mode in enumerate(FAILURE_MODES):
        score = report.scores.get(mode)
        colour = _VERDICT_COLOUR.get(score.verdict if score else "OK", "#8b949e")
        x = 8 + index * 60
        label = _safe(mode.replace("_", " "), max_len=14)
        # Defence-in-depth — even the formatted float passes through
        # `_safe` so a crafted FailureScore subclass cannot inject SVG.
        score_text = _safe(f"{score.score:.2f}" if score is not None else "—", max_len=8)
        cells.append(
            f'<rect x="{x}" y="40" width="56" height="36" rx="6" fill="{colour}" />'
            f'<text x="{x + 28}" y="60" font-size="11" fill="#ffffff" '
            f'text-anchor="middle" font-family="monospace">{label}</text>'
            f'<text x="{x + 28}" y="73" font-size="10" fill="#ffffff" '
            f'text-anchor="middle" font-family="monospace">{score_text}</text>'
        )
    name = _safe(report.adapter or report.run_id, max_len=48)
    width = 8 + len(FAILURE_MODES) * 60 + 8
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="92" '
        f'viewBox="0 0 {width} 92">'
        '<rect width="100%" height="100%" rx="8" fill="#0d1117" />'
        f'<text x="12" y="22" font-size="13" font-family="monospace" fill="#c9d1d9">'
        f'soup diagnose: {name}</text>'
        f'<rect x="{width - 80}" y="8" width="72" height="20" rx="10" '
        f'fill="{overall_colour}" />'
        f'<text x="{width - 44}" y="22" font-size="12" font-family="monospace" '
        f'fill="#ffffff" text-anchor="middle">{escape(overall, quote=True)}</text>'
        + "".join(cells)
        + "</svg>"
    )
