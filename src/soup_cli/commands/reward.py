"""soup reward — synthesize a deterministic reward verifier (v0.71.40).

    soup reward synth <references.jsonl> -o reward.py
        [--kind auto|numeric|json_schema|regex|tool_call]

Infers a deterministic verifier from a dataset of reference (gold) outputs, emits a
readable / committable ``.py`` that rides ``load_reward_fn``'s existing ``.py`` path,
and REFUSES to emit a degenerate verifier via a mandatory calibration report.

Exit codes mirror ``soup ship`` / ``soup shrink``: 0 = emitted, 2 = refused (the
verifier could not discriminate references from perturbed negatives), 1 = usage /
runtime error.
"""

from __future__ import annotations

import json
import math
import os
from typing import NoReturn, Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from soup_cli.utils import reward_synth as rs
from soup_cli.utils.paths import atomic_write_text, enforce_under_cwd_and_no_symlink

app = typer.Typer(help="Synthesize a deterministic reward verifier from reference outputs.")
console = Console()

_MAX_INPUT_BYTES = 64 * 1024 * 1024
_MAX_ROWS = 1_000_000
_ALLOWED_KINDS = ("auto",) + rs.KINDS


@app.callback()
def _reward() -> None:
    """Reward-verifier tooling. Forces ``synth`` to be a named subcommand (a
    single-command Typer app otherwise collapses and eats the subcommand token)."""


def _fail(message: str, code: int = 1) -> NoReturn:
    """Print a red error and exit (raises internally — a forgotten ``raise`` at a
    call site can never silently become a no-op; mirrors ``commands/ship.py``)."""
    console.print(f"[red]{escape(message)}[/]")
    raise typer.Exit(code)


def _read_jsonl(path: str, label: str) -> list[dict]:
    """Read a JSONL file: cwd-contained, O_NOFOLLOW, size- and row-capped."""
    enforce_under_cwd_and_no_symlink(path, label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{label} unreadable: {exc}") from exc
    rows: list[dict] = []
    with os.fdopen(fd, "r", encoding="utf-8") as handle:
        if os.fstat(handle.fileno()).st_size > _MAX_INPUT_BYTES:
            raise ValueError(f"{label} exceeds {_MAX_INPUT_BYTES} bytes")
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
            if len(rows) >= _MAX_ROWS:
                break
    return rows


def _spec_summary(kind: str, spec: object) -> str:
    """One-line human summary of the induced spec (for --plan-only / the panel)."""
    if kind == "numeric" and isinstance(spec, rs.NumericSpec):
        return f"numeric (float={spec.is_float}, tolerance={spec.tolerance})"
    if kind == "json_schema" and isinstance(spec, dict):
        props = ", ".join(sorted(spec.get("properties", {}))) or "(none)"
        return f"json_schema (type={spec.get('type')}, keys={props})"
    if kind == "tool_call" and isinstance(spec, rs.ToolCallSpec):
        return f"tool_call (names={list(spec.names)}, arg_keys={list(spec.arg_keys)})"
    if kind == "regex":
        return f"regex (pattern={spec})"
    return kind


def _render_report_panel(report: rs.CalibrationReport, kind: str, out_path: str) -> Panel:
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(justify="right", style="dim")
    table.add_column()
    table.add_row("verifier kind", escape(kind))
    table.add_row("references (accepted)", f"{report.positives} ({report.pos_accept:.0%})")
    table.add_row("negatives (accepted)", f"{report.negatives} ({report.neg_accept:.0%})")
    table.add_row("discrimination", f"{report.discrimination:.2f}")
    table.add_row("precision", f"{report.precision:.2f}")
    table.add_row("emitted", escape(out_path))
    return Panel(table, title="[bold green]reward verifier synthesized[/]",
                 border_style="green")


@app.command()
def synth(
    references: str = typer.Argument(
        ..., help="JSONL of reference (gold) outputs — a gold field or chat messages."
    ),
    output: Optional[str] = typer.Option(
        None, "-o", "--output", help="Where to write the verifier .py."
    ),
    kind: str = typer.Option(
        "auto", "--kind",
        help="Verifier family: auto | numeric | json_schema | regex | tool_call.",
    ),
    field: str = typer.Option(
        "answer", "--field", help="Gold-output field (default: answer)."
    ),
    tolerance: Optional[float] = typer.Option(
        None, "--tolerance", help="Numeric match tolerance (numeric kind only)."
    ),
    min_discrimination: float = typer.Option(
        rs.DEFAULT_MIN_DISCRIMINATION, "--min-discrimination",
        help="Refuse to emit unless (accept_rate refs - accept_rate negatives) >= this.",
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing output."),
    plan_only: bool = typer.Option(
        False, "--plan-only", help="Detect + report the induced spec; write nothing."
    ),
    output_report: Optional[str] = typer.Option(
        None, "--output-report", help="Also write the calibration report as JSON."
    ),
) -> None:
    """Synthesize a deterministic reward verifier from reference outputs."""
    if kind not in _ALLOWED_KINDS:
        _fail(f"unknown --kind {kind!r}; options: {', '.join(_ALLOWED_KINDS)}")
    if not 0.0 <= min_discrimination <= 1.0:
        _fail("--min-discrimination must be in [0.0, 1.0]")
    if tolerance is not None and (not math.isfinite(tolerance) or tolerance < 0):
        _fail("--tolerance must be a finite, non-negative number")

    # Read references.
    try:
        rows = _read_jsonl(references, "references path")
    except (ValueError, OSError) as exc:
        _fail(str(exc))
    if not rows:
        _fail("references file has no JSON-object rows")

    # Detect + induce + render (no file written yet).
    try:
        result = rs.synthesize(rows, field=field, kind=kind, tolerance=tolerance,
                               rel_hint=(output or "reward.py"))
    except (ValueError, TypeError) as exc:
        _fail(str(exc))

    if plan_only:
        console.print(Panel(
            escape(_spec_summary(result.kind, result.spec)),
            title=f"[bold]plan: {escape(result.kind)} verifier[/]", border_style="cyan"))
        raise typer.Exit(0)

    # Output guards.
    if not output:
        _fail("must pass -o/--output to emit a verifier (or use --plan-only)")
    if not output.endswith(".py"):
        _fail(f"output must be a .py file, got {output!r}")
    try:
        enforce_under_cwd_and_no_symlink(output, "output path")
    except (ValueError, OSError) as exc:
        _fail(str(exc))
    if os.path.exists(output) and not force:
        _fail(f"{output!r} already exists — pass --force to overwrite")
    # Validate the optional report path UP FRONT so a bad report path can never
    # cause us to destroy an already-written, already-accepted verifier later.
    if output_report:
        try:
            enforce_under_cwd_and_no_symlink(output_report, "report path")
        except (ValueError, OSError) as exc:
            _fail(str(exc))

    # Write, then LOAD it back through the real reward-loader path (round-trip
    # validation) and calibrate the loaded callable.
    atomic_write_text(result.source, output)
    try:
        from soup_cli.trainer.rewards import load_reward_fn
        reward_fn = load_reward_fn(output)
        golds = rs.extract_golds(rows, field=field)
        negatives = rs.perturb_negatives(golds, result.kind)
        report = rs.calibrate(reward_fn, golds, negatives, kind=result.kind,
                              min_discrimination=min_discrimination)
    except Exception as exc:  # noqa: BLE001 — clean up the partial artifact
        _cleanup(output)
        _fail(f"calibration failed: {exc}")

    # Write the diagnostic report (path already validated up front) for BOTH the
    # accepted and refused cases. Best-effort: a write failure here warns but must
    # not delete an otherwise-valid verifier.
    if output_report:
        try:
            from dataclasses import asdict
            atomic_write_text(json.dumps(asdict(report), indent=2), output_report)
        except OSError as exc:
            console.print(f"[yellow]Warning: could not write report: {escape(str(exc))}[/]")

    if report.refused:
        _cleanup(output)
        console.print(Panel(
            escape(report.reason),
            title="[bold red]verifier refused (not emitted)[/]", border_style="red"))
        raise typer.Exit(2)

    console.print(_render_report_panel(report, result.kind, output))
    raise typer.Exit(0)


def _cleanup(path: str) -> None:
    """Remove a just-written artifact; best-effort (never masks the real error)."""
    try:
        os.remove(path)
    except OSError:
        pass
