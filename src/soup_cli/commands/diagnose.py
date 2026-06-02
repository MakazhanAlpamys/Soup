"""soup diagnose — post-training model report card (v0.56.0).

Top-level CLI command (NOT a sub-group) — operators type:

    soup diagnose <run-id>
    soup diagnose <run-id> --output diagnose.json
    soup diagnose <run-id> --badge diagnose.svg
    soup diagnose <run-id> --attach-to-registry <id>

The live probe runners (forgetting / refusal / format / mode_collapse /
memorization / contamination) require a base + adapter model pair; this
release computes neutral OK scores when no probe evidence is supplied,
matching the v0.27.0 / v0.50.0 stub-then-live cadence. Operators with a
SoupConfig + an evidence dict can call ``soup_cli.utils.diagnose.diagnose``
to produce a real report card.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from soup_cli import __version__
from soup_cli.utils.diagnose import FailureReport, compose_report
from soup_cli.utils.diagnose.badge import render_badge_svg
from soup_cli.utils.diagnose.report import FAILURE_MODES, FailureScore, classify_score
from soup_cli.utils.diagnose.runner import (
    build_report,
    neutral_score,
    write_report,
)
from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

console = Console()

# 16 MiB cap on evidence JSON files (security review HIGH — prevents
# multi-GB / `/dev/zero` symlink-pointed OOM at json.load time).
_MAX_EVIDENCE_BYTES = 16 * 1024 * 1024


def _verdict_style(verdict: str) -> str:
    return {"OK": "green", "MINOR": "yellow", "MAJOR": "red"}.get(
        verdict, "white"
    )


def _render_report(report: FailureReport) -> None:
    table = Table(title=f"soup diagnose: {escape(report.adapter or report.run_id)}")
    table.add_column("Mode", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Verdict")
    table.add_column("Evidence", overflow="fold")
    for mode in FAILURE_MODES:
        score = report.scores.get(mode)
        if score is None:
            continue
        table.add_row(
            escape(mode),
            f"{score.score:.3f}",
            f"[{_verdict_style(score.verdict)}]{score.verdict}[/]",
            escape(score.evidence),
        )
    console.print(table)
    console.print(
        Panel.fit(
            f"[bold {_verdict_style(report.overall)}]{report.overall}[/] — "
            f"run [bold]{escape(report.run_id)}[/]",
            title="overall",
        )
    )


def _load_evidence(path: str) -> dict:
    enforce_under_cwd_and_no_symlink(path, "evidence path")
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        raise typer.BadParameter(f"evidence path unreadable: {exc}") from exc
    if size > _MAX_EVIDENCE_BYTES:
        raise typer.BadParameter(
            f"evidence file exceeds {_MAX_EVIDENCE_BYTES} bytes"
        )
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise typer.BadParameter("evidence file must contain a JSON object")
    return payload


def _scores_from_evidence(payload: dict) -> dict:
    """Translate a user-supplied evidence dict into FailureScore entries."""
    raw_scores = payload.get("scores", {})
    if not isinstance(raw_scores, dict):
        raise typer.BadParameter("evidence.scores must be an object")
    out = {}
    for mode in FAILURE_MODES:
        entry = raw_scores.get(mode)
        if entry is None:
            continue
        if not isinstance(entry, dict):
            raise typer.BadParameter(f"scores.{mode} must be an object")
        score = entry.get("score", 1.0)
        evidence = entry.get("evidence", "supplied by --evidence")
        verdict = entry.get("verdict") or classify_score(score)
        out[mode] = FailureScore(
            mode=mode,
            score=float(score),
            verdict=verdict,
            evidence=str(evidence),
        )
    return out


def _write_badge(badge_path: str, svg: str) -> None:
    """Atomic SVG write with TOCTOU symlink rejection.

    Mirrors the v0.33.0 #22 / v0.43.0 / v0.46.0 / v0.47.0 project policy:
    `os.lstat + S_ISLNK` on the RAW path BEFORE `realpath`, then
    `tempfile.mkstemp + os.replace` for the atomic swap.
    """
    enforce_under_cwd_and_no_symlink(badge_path, "badge path")
    if os.path.lexists(badge_path):
        st = os.lstat(badge_path)
        if stat.S_ISLNK(st.st_mode):
            raise ValueError("badge path must not be a symlink")
    parent = os.path.dirname(os.path.realpath(badge_path)) or "."
    if not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".badge-", suffix=".svg.tmp", dir=parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(svg)
        os.replace(tmp_path, badge_path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise


def _attach_to_registry(report: FailureReport, registry_id: str, output: str) -> None:
    try:
        from soup_cli.registry.attach import attach_artifact
    except Exception as exc:  # noqa: BLE001 — registry is optional
        console.print(
            f"[yellow]Warning:[/] could not import registry attach helper: "
            f"{escape(type(exc).__name__)}"
        )
        return
    try:
        attach_artifact(registry_id, "diagnose_report", output)
        console.print(
            f"[green]Attached[/] diagnose_report to registry entry "
            f"[bold]{escape(registry_id)}[/]"
        )
    except Exception as exc:  # noqa: BLE001
        console.print(
            f"[yellow]Warning:[/] could not attach to registry: "
            f"{escape(type(exc).__name__)}: {escape(str(exc))}"
        )


def diagnose(
    run_id: str = typer.Argument(..., help="Registry run id (or any opaque tag)."),
    base: str = typer.Option("", "--base", help="Base model name (informational)."),
    adapter: str = typer.Option("", "--adapter", help="Adapter name (informational)."),
    evidence_path: Optional[str] = typer.Option(
        None,
        "--evidence",
        help="JSON file with pre-computed probe scores (see README).",
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Write the report JSON to this path."
    ),
    badge: Optional[str] = typer.Option(
        None, "--badge", help="Write an SVG badge to this path."
    ),
    attach_to_registry: Optional[str] = typer.Option(
        None, "--attach-to-registry", help="Attach the report to a registry entry id."
    ),
) -> None:
    """Compute a 6-mode FailureReport for a completed run."""
    if not isinstance(run_id, str) or not run_id.strip():
        raise typer.BadParameter("run_id must be a non-empty string")
    if "\x00" in run_id or len(run_id) > 512:
        raise typer.BadParameter("run_id has a null byte or is too int")

    scores = {}
    extras = {}
    if evidence_path:
        try:
            payload = _load_evidence(evidence_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            console.print(
                f"[red]Error:[/] cannot read --evidence: "
                f"{escape(type(exc).__name__)}"
            )
            raise typer.Exit(code=1) from exc
        scores = _scores_from_evidence(payload)
        # Sanitise extras — null-byte rejection + 256-char cap on both
        # key and value (security review MEDIUM — prevents Rich markup
        # injection through user-supplied evidence metadata).
        for key, value in (payload.get("extras") or {}).items():
            key_s = str(key)
            value_s = str(value)
            if "\x00" in key_s or "\x00" in value_s:
                console.print(
                    "[red]Error:[/] extras key/value must not contain null bytes"
                )
                raise typer.Exit(code=1)
            extras[key_s[:256]] = value_s[:256]

    # Fill missing modes with neutral OK + advisory.
    for mode in FAILURE_MODES:
        scores.setdefault(mode, neutral_score(mode, "no evidence"))

    report = build_report(
        run_id=run_id,
        base=base,
        adapter=adapter,
        scores=scores,
        soup_version=__version__,
        extras=extras,
    )
    _render_report(report)

    if output:
        try:
            write_report(report, output)
            console.print(f"[green]Wrote[/] {escape(output)}")
        except (OSError, ValueError) as exc:
            console.print(
                f"[red]Error:[/] cannot write --output: "
                f"{escape(type(exc).__name__)}: {escape(str(exc))}"
            )
            raise typer.Exit(code=1) from exc

    if badge:
        try:
            svg = render_badge_svg(report)
            _write_badge(badge, svg)
            console.print(f"[green]Badge written[/] to {escape(badge)}")
        except (OSError, ValueError, TypeError) as exc:
            console.print(
                f"[red]Error:[/] cannot write --badge: "
                f"{escape(type(exc).__name__)}: {escape(str(exc))}"
            )
            raise typer.Exit(code=1) from exc

    if attach_to_registry and output:
        _attach_to_registry(report, attach_to_registry, output)
    elif attach_to_registry and not output:
        console.print(
            "[yellow]Warning:[/] --attach-to-registry needs --output (skipped)."
        )

    if report.overall == "MAJOR":
        raise typer.Exit(code=2)


# Compose a tiny helper so the CLI module is callable from tests.
__all__ = ["diagnose", "compose_report", "build_report", "render_badge_svg"]
